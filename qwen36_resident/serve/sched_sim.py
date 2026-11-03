# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop simulation of the multi profile's prefill / decode scheduling, on CPU.

The real TTScheduler of the vllm-tt-plugin on PYTHONPATH (TT chunk core, decode interleave, KV cache manager at the
served pool) runs on a virtual clock. Each step's duration comes from a cost model of what the rows adapter does for
it (vllm_rows.ResidentRowsAdapter: PP ticks per span, the drain, handoff + first-token launch per prompt; decode =
one replay per launch with live rows). The client is ttis' vllm bench serve: `conc` users, fixed ISL / OSL (max_tokens,
no EOS), a new request as soon as one finishes. Policies are the plugin's knobs (additional_config tt), so a policy
simulated here is the one served.

The cost model is calibrated on the served c=1 TTFTs and the rows grid (unthrottled chips); it ranks policies, the
device runs (profiles/resident/sched/) give the numbers.

usage: PYTHONPATH=<plugin src>:<tt-metal> python -m qwen36_resident.serve.sched_sim [--isl 2048 --osl 128 ...]
"""
import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

# served c=1 TTFT (ms) at prompt length T, ttis multi_20261030-1332 (includes HTTP + first token)
TTFT_C1 = {128: 135, 1024: 307, 2048: 444, 4096: 801, 8192: 1211, 16384: 2631, 32768: 6570, 65536: 17019, 131072: 50446}
CLIENT_MS = 15  # HTTP + tokenize share of a c=1 TTFT
PER_PROMPT_MS = 60  # reset + handoff + first-token launch + logits read of one prompt (inside TTFT_C1)
FIRST_MS = 20  # of which the first-token launch + sync + logits read (one per launch when batched)
# what-ifs (CLI): the step's first-token launches batched per launch (vllm_rows), and a step's whole prompts streamed
# back to back through the PP pipeline (fill / drain once per step: a prefill-side API that does not exist yet)
WHATIF = dict(batched_first=False, pp_pipelined=False)
# rows-mode decode step per replayed launch (ms) at 1/2/4/8 active rows, short context (rows_grid_c6436.json)
LAUNCH_MS = {1: 14.46, 2: 15.12, 4: 16.64, 8: 20.14}
RELOAD_MS = 3.0  # a decode step after a prefill rewrites token states and page tables on the host
ROWS = 8
# PPPrefill.TICK_MS (pick_chunk's estimate)
TICK_MS = {128: (22.4, 1.6), 256: (32.5, 2.9), 512: (45.5, 3.2), 1024: (81.0, 3.0)}
PP_CHUNKS = (128, 256, 512, 1024)
UNIT = 1024


def pick_chunk(T):
    def est(C):
        base, slope = TICK_MS[C]
        return ((T + C - 1) // C + 3) * (base + slope * T / 2048)

    return min(PP_CHUNKS, key=est)


def interp(table, x):
    xs = sorted(table)
    if x <= xs[0]:
        return table[xs[0]] * x / xs[0] if x < xs[0] else table[xs[0]]
    for a, b in zip(xs, xs[1:]):
        if x <= b:
            return table[a] + (table[b] - table[a]) * (x - a) / (b - a)
    a, b = xs[-2], xs[-1]
    return table[b] + (table[b] - table[a]) * (x - b) / (b - a)


def prefill_ms(total):
    """device time of one whole prompt of `total` tokens (PP ticks + per-prompt work)"""
    return max(interp(TTFT_C1, total) - CLIENT_MS, PER_PROMPT_MS + 1)


def tick_ms(total):
    """one PP tick of a prompt of `total` tokens at its chunk: the prompt's PP time spread over its ticks"""
    C = pick_chunk(total - 1)
    ticks = math.ceil((total - 1) / C) + 3
    return (prefill_ms(total) - PER_PROMPT_MS) / ticks, C


def launch_ms(rows):
    return interp(LAUNCH_MS, rows) if rows else 0.0


class Model:
    """the rows adapter's step cost and its slot map (lowest launch first)"""

    def __init__(self, n_slots):
        self.free = list(range(n_slots))
        self.slot = {}
        self.last_prefill = False

    def step_ms(self, kind, spans, sched):
        if kind == "decode":
            per_launch = {}
            for rid in spans:
                k = self.slot[rid] // ROWS
                per_launch[k] = per_launch.get(k, 0) + 1
            ms = sum(launch_ms(n) for n in per_launch.values())
            if self.last_prefill:
                ms += RELOAD_MS
            self.last_prefill = False
            return ms
        self.last_prefill = True
        ms = 0.0
        wholes, first_launches = 0, set()
        for rid, (start, end) in spans.items():
            req = sched.requests[rid]
            total = req.num_tokens  # the sequence the chunk core passes as prefill_total_lens
            tk, C = tick_ms(total)
            final = end >= total
            n_pp = (end - 1 if final else end) - start  # the PP takes all tokens but the last
            ticks = math.ceil(max(n_pp, 0) / C)
            if rid not in self.slot:
                self.slot[rid] = self.free.pop(0)
            if final:
                ticks = math.ceil((total - 1) / C) + 3 - start // C
                if start == 0:
                    wholes += 1
                    if WHATIF["pp_pipelined"] and wholes > 1:
                        ticks -= 3  # streamed behind the previous prompt: no own fill / drain
                ms += PER_PROMPT_MS - FIRST_MS
                if WHATIF["batched_first"]:
                    first_launches.add(self.slot[rid] // ROWS)
                else:
                    ms += FIRST_MS
            ms += ticks * tk
        return ms + FIRST_MS * len(first_launches)

    def release(self, rid):
        s = self.slot.pop(rid, None)
        if s is not None:
            self.free.append(s)
            self.free.sort()


def make_scheduler(max_num_seqs, max_model_len, pool_blocks, tt):
    from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    import vllm_tt_plugin
    from vllm_tt_plugin.scheduler import TTScheduler

    from qwen36_resident.vllm_rows import resident_capabilities

    class ResidentModel:
        model_capabilities = resident_capabilities(chunk_core=True)

    model_cfg = Path(vllm_tt_plugin.__file__).resolve().parents[2] / "tests" / "model_configs" / "qwen2"
    mc = ModelConfig(model=str(model_cfg), dtype="float16", seed=42, skip_tokenizer_init=True)
    mc.max_model_len = max_model_len
    sc = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=2048,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        async_scheduling=False,
        is_encoder_decoder=False,
    )
    cc = CacheConfig(block_size=64, gpu_memory_utilization=0.9, cache_dtype="auto", enable_prefix_caching=False)
    with (
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch("vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups", return_value=None),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs", return_value=["TTQwen2ForCausalLM"]
        ),
        patch("vllm.model_executor.model_loader.utils.get_model_architecture", return_value=(ResidentModel, None)),
    ):
        config = VllmConfig(
            scheduler_config=sc,
            model_config=mc,
            cache_config=cc,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            additional_config={"tt": dict(tt)},
        )
    cc.num_gpu_blocks = pool_blocks
    kvc = KVCacheConfig(
        num_blocks=pool_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size=64, num_kv_heads=1, head_size=1, dtype=torch.float32)
            )
        ],
    )
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kvc,
        block_size=64,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(config),
    )


def simulate(
    isl,
    osl,
    conc,
    n,
    tt,
    max_num_seqs=16,
    max_model_len=262144,
    pool_blocks=24064,
    client_ms=5.0,
    rate=None,
    seed=0,
    jitter_ms=0.0,
):
    """ttis' closed loop (rate None: a new request as soon as one finishes), or vllm bench --request-rate: Poisson
    arrivals at `rate` req/s, still at most `conc` in flight. isl / osl: a length, or a (lo, hi) range drawn uniformly
    per request. jitter_ms: each request reaches the engine up to this much later (uniform), e.g. a burst's
    tokenization in the API server."""
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    rng = np.random.default_rng(seed)
    draw = lambda v: int(rng.integers(v[0], v[1] + 1)) if isinstance(v, tuple) else int(v)
    spec = [(draw(isl), draw(osl)) for _ in range(n)]
    planned = np.cumsum(rng.exponential(1000.0 / rate, n)) if rate else None
    sched = make_scheduler(max_num_seqs, max_model_len, pool_blocks, tt)
    model = Model(-(-max_num_seqs // ROWS) * ROWS)
    init_none_hash(sha256)
    now = 0.0
    arrivals = []  # (time, rid)
    sent, first, toks, done, lens = {}, {}, {}, {}, {}
    issued = 0
    in_flight = 0

    def issue(t):
        nonlocal issued, in_flight
        rid = f"r{issued}"
        t = max(t, planned[issued]) if planned is not None else t
        lens[rid] = spec[issued]
        issued += 1
        in_flight += 1
        arrivals.append((t + client_ms + float(rng.uniform(0, jitter_ms)), rid))
        sent[rid] = t

    for _ in range(min(conc, n)):
        issue(0.0)
    steps = []
    while len(done) < n:
        arrivals.sort()
        while arrivals and arrivals[0][0] <= now + 1e-9:
            _, rid = arrivals.pop(0)
            p = SamplingParams(max_tokens=lens[rid][1], ignore_eos=True)
            p.update_from_generation_config({}, eos_token_id=2)
            sched.add_request(
                Request(
                    request_id=rid,
                    prompt_token_ids=[7] * lens[rid][0],
                    sampling_params=p,
                    pooling_params=None,
                    block_hasher=get_request_block_hasher(64, sha256),
                )
            )
        if not sched.has_requests():
            now = arrivals[0][0]
            continue
        starts = {r: q.num_computed_tokens for r, q in sched.requests.items()}
        out = sched.schedule()
        spans = {r: (starts.get(r, 0), starts.get(r, 0) + k) for r, k in out.num_scheduled_tokens.items()}
        if not spans:
            # nothing schedulable now: wait for the next arrival
            now = arrivals[0][0] if arrivals else now + 1.0
            sched.update_from_output(
                out,
                ModelRunnerOutput(
                    req_ids=[],
                    req_id_to_index={},
                    sampled_token_ids=[],
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                ),
            )
            continue
        kind = "prefill" if any(s < sched.requests[r].num_prompt_tokens for r, (s, _) in spans.items()) else "decode"
        ms = model.step_ms(kind, spans, sched)
        now += ms
        steps.append((kind, ms, len(spans)))
        tokens = {r: ([] if sched.requests[r].is_prefill_chunk else [5]) for r in spans}
        ids = list(tokens)
        mro = ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[tokens[r] for r in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        for eco in sched.update_from_output(out, mro).values():
            for o in eco.outputs:
                if o.new_token_ids:
                    toks.setdefault(o.request_id, []).extend([now] * len(o.new_token_ids))
                    first.setdefault(o.request_id, now)
                if o.finish_reason is not None and o.request_id not in done:
                    done[o.request_id] = now
                    model.release(o.request_id)
                    in_flight -= 1
                    if issued < n:
                        issue(now)
    dur = max(done.values())
    ttft = np.array([first[r] - sent[r] for r in done]) / 1000
    tpot = np.array([(toks[r][-1] - toks[r][0]) / max(len(toks[r]) - 1, 1) for r in done])
    itl = np.concatenate([np.diff(toks[r]) for r in done])
    pf = sum(ms for k, ms, _ in steps if k == "prefill")
    return dict(
        isl=isl,
        osl=osl,
        conc=conc,
        n=n,
        rate=rate,
        tt=tt,
        ttft_mean_s=float(ttft.mean()),
        ttft_med_s=float(np.median(ttft)),
        ttft_p99_s=float(np.percentile(ttft, 99)),
        tpot_ms=float(tpot.mean()),
        itl_p50_ms=float(np.percentile(itl, 50)),
        itl_p99_ms=float(np.percentile(itl, 99)),
        itl_max_ms=float(itl.max()),
        tok_s=float(sum(o for _, o in spec) / (dur / 1000)),
        prefill_frac=pf / dur,
        decode_steps=sum(k == "decode" for k, _, _ in steps),
        mean_decode_rows=float(np.mean([r for k, _, r in steps if k == "decode"])),
    )


POLICIES = {
    "base": {"prefill_chunk_tokens": 1024},
    "il1": {"prefill_chunk_tokens": 1024, "decode_interleave_prefill_steps": 1},
    "il1d2": {"prefill_chunk_tokens": 1024, "decode_interleave_prefill_steps": 1, "decode_interleave_decode_steps": 2},
    "il1d4": {"prefill_chunk_tokens": 1024, "decode_interleave_prefill_steps": 1, "decode_interleave_decode_steps": 4},
    "noil": {"prefill_chunk_tokens": 1024, "decode_interleave_enabled": False},
    # gangs: cap while requests decode / while nothing does (0 = the step's token budget)
    "g2k": {"prefill_chunk_tokens": 1024, "prefill_gang_tokens": 2048},
    "g8k": {"prefill_chunk_tokens": 1024, "prefill_gang_tokens": 8192},
    "g32k": {"prefill_chunk_tokens": 1024, "prefill_gang_tokens": 32768},
    "g32k_i64k": {"prefill_chunk_tokens": 1024, "prefill_gang_tokens": 32768, "prefill_gang_idle_tokens": 65536},
    "g32k_i32k": {"prefill_chunk_tokens": 1024, "prefill_gang_tokens": 32768, "prefill_gang_idle_tokens": 32768},
    "g16k_i16k": {"prefill_chunk_tokens": 1024, "prefill_gang_tokens": 16384, "prefill_gang_idle_tokens": 16384},
}
SHAPES = [(128, 128), (2048, 128), (8192, 128), (10000, 128), (8192, 1024), (10000, 1024)]


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--shapes", default=",".join(f"{i}/{o}" for i, o in SHAPES))
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--waves", type=int, default=4, help="prompts = waves x conc (ttis: 4 at <= 4K/128, 2 above)")
    ap.add_argument("--rate", type=float, default=None, help="Poisson arrivals (req/s) instead of the closed loop")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--batched-first", action="store_true", help="what-if: first-token launches batched per launch")
    ap.add_argument(
        "--pp-pipelined", action="store_true", help="what-if: a step's whole prompts share one PP fill/drain"
    )
    ap.add_argument("--per-prompt-ms", type=float, default=60.0)
    ap.add_argument("--first-ms", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import logging

    globals().update(PER_PROMPT_MS=a.per_prompt_ms, FIRST_MS=a.first_ms)
    WHATIF.update(batched_first=a.batched_first, pp_pipelined=a.pp_pipelined)
    logging.disable(logging.INFO)
    rows = []
    rng_of = lambda v: tuple(int(x) for x in v.split("-")) if "-" in v else int(v)
    for shape in a.shapes.split(","):
        isl, osl = (rng_of(x) for x in shape.split("/"))
        for name in a.policies.split(","):
            runs = [
                simulate(
                    isl, osl, a.conc, a.waves * a.conc, POLICIES[name], rate=a.rate, seed=sd, jitter_ms=a.jitter_ms
                )
                for sd in range(a.seeds)
            ]
            r = {
                k: (float(np.mean([x[k] for x in runs])) if isinstance(runs[0][k], float) else runs[0][k])
                for k in runs[0]
            }
            r["policy"] = name
            rows.append(r)
            print(
                f"{shape:>12} {name:>9}: TPOT {r['tpot_ms']:6.1f} ms  ITL p50 {r['itl_p50_ms']:6.1f} "
                f"p99 {r['itl_p99_ms']:6.1f} max {r['itl_max_ms']:6.0f}  TTFT mean {r['ttft_mean_s']:5.2f} s "
                f"med {r['ttft_med_s']:5.2f} p99 {r['ttft_p99_s']:5.2f}  {r['tok_s']:6.1f} tok/s  "
                f"prefill {r['prefill_frac']:.2f}  "
                f"rows/decode {r['mean_decode_rows']:.1f}",
                flush=True,
            )
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

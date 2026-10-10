# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Laguna-S device performance without the vLLM server: time to first token and decode speed, next to the speed of
light (SoL) and the 50% targets that demo/roofline.py computes.

perf_demo.py measures through vLLM (HTTP, chat template, scheduler, streaming), which the SoL does not model. This
script drives the model directly in one process and times only device work plus the token readback:

    TTFT, batch 1      prefill of exactly L random tokens + LM head on the last row + on-device greedy sampling +
                       reading the token back. "eager" dispatches the ops from the host (what serving does above the
                       traced 128/256 buckets), median of 3; "traced" replays a captured trace of the same ops (device
                       time + readback only), median of 5, for L <= TT_LAGUNA_PIPE_CHUNK (the single-shot path)
    TTFT, batch 32     32 prompts of L tokens prefilled the way the server does it: packed n at a time when n * L fits
                       the 8192-row prefill (L <= 4096; one packed forward per group), else one prompt at a time. A
                       user's TTFT is the time from the start until its group's tokens are read back (prompts all
                       arrive at t = 0). "last" (all 32 users have a token) compares with roofline.py's batch-32 SoL,
                       which prefills all 32 prompts in one pass
    decode tok/s       1000 / (time per step of the captured decode trace at a context of L tokens), averaged over
                       --decode-steps steps; batch 32 decodes 32 users with distinct tokens
    --modes dflash     batch 1 with DFlash speculative decoding, through the server's own model class (the one vLLM
                       loads: prefill, then one committed token per decode call, DFlash rounds inside) without vLLM.
                       TTFT = its prefill call; decode tok/s = tokens / time over --dflash-tokens decode calls. KV is
                       a pool just large enough for the request (block table as wide as the pool), uniform layout.
                       Prompts are real text (a summarize request over tech_reports/*.md), not random tokens (the server's hybrid KV needs vLLM's per-group block tables; at <= 8K
                       context the sliding layers read their 512-token window either way)

    python models/demos/laguna/demo/perf_direct.py                         # batch 1 and 32, 128 .. 8K tokens
    python models/demos/laguna/demo/perf_direct.py --batch 1 --input-lens 128,1024
    python models/demos/laguna/demo/perf_direct.py --modes dflash
"""

from __future__ import annotations

import os

for _key, _value in {
    "TT_LAGUNA_MODEL": "poolside/Laguna-S-2.1",
    "LAGUNA_PROFILE": "p150x4",
    "TT_VISIBLE_DEVICES": "0,1,2,3",
    "LAGUNA_FABRIC_CONFIG": "FABRIC_1D_RING",
    "TT_LAGUNA_CCL_TOPOLOGY": "ring",
    "TT_LAGUNA_CCL_NUM_LINKS": "2",
    "TT_LAGUNA_DECODE_SDPA_PC": "1",
    "TT_LAGUNA_MOE_TOKEN_DISPATCH": "1",
    "TT_LAGUNA_MOE_PREFILL_TILE_SPARSE": "0",
    "TT_LAGUNA_STREAMING_PREFILL": "1",
    "TT_LAGUNA_PIPE_CHUNK": "2048",
    "TT_LAGUNA_PREFILL_FAST": "1",
    "TT_LAGUNA_PREFILL_FAST_CHUNK": "8192",
    "TT_LAGUNA_PREFILL_SDPA_CHUNK": "8192",
    "TT_LAGUNA_DECODE_K": "64",
    "TT_LAGUNA_DECODE_EXP": "0",
    "TT_LAGUNA_DECODE_MAXCORES": "16",
}.items():
    os.environ.setdefault(_key, _value)  # serve_vllm.sh's p150x4 settings, before any Laguna import reads them
if "dflash" in " ".join(__import__("sys").argv):
    # serve_vllm.sh's DFlash settings (read when the serving class is imported)
    for _key, _value in {"TT_LAGUNA_DFLASH": "1", "LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES": "1", "TT_LAGUNA_HYBRID_KV": "0",
                         "TT_LAGUNA_PREFIX_CACHE": "0"}.items():  # fmt: skip
        os.environ.setdefault(_key, _value)

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

import ttnn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from models.demos.laguna.demo import roofline  # noqa: E402
from models.demos.laguna.tests.laguna_test_utils import close_mesh, open_mesh, resolve_profile  # noqa: E402
from models.demos.laguna.tt.generator import LagunaGenerator  # noqa: E402

DEFAULT_INPUT_LENS = [128, 1024, 2048, 4096, 8192]
PACK_ROWS = 8192  # largest packed prefill (rows) the server uses
PACK_MAX_LEN = 4096  # longest prompt the server packs


def rep(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


def random_ids(n, seed):
    return torch.randint(1000, 100000, (n,), generator=torch.Generator().manual_seed(seed))


def release(mesh, tid):
    if tid is not None:
        ttnn.release_trace(mesh, tid)


def time_decode(gen, mesh, batch, context, steps):
    """ms per step of the captured decode trace, users at position ``context`` (distinct tokens per user)."""
    host = gen._host
    toks = random_ids(batch, 7).to(torch.int32).reshape(1, 1, 1, batch)
    gen._host_rank4_tok = lambda _t: host(toks, ttnn.uint32)
    gen._host_pos = lambda p: host(torch.full((batch,), int(p), dtype=torch.int32), ttnn.int32)
    gen._host_ridx = lambda p: host(torch.full((1, batch), int(p), dtype=torch.int32), ttnn.uint32)
    st = gen._decode_trace_state(batch, gen._page_table, context, 0)
    for _ in range(4):
        ttnn.execute_trace(mesh, st["tid"], cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    start = time.perf_counter()
    for _ in range(steps):
        ttnn.execute_trace(mesh, st["tid"], cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - start) / steps * 1e3
    release(mesh, st["tid"])
    gen._trace.pop(batch, None)
    return ms


def ttft_batch1(gen, mesh, length):
    """(eager, traced or None) median seconds of a prefill of ``length`` tokens + last-row LM head + greedy sample +
    token readback."""
    m, H = gen.model, gen.hidden
    tokbuf = rep(mesh, random_ids(length, length).to(torch.int32).reshape(1, length), ttnn.uint32)
    tok = rep(mesh, torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)

    def body():
        x = m.embed_prefill(tokbuf)
        h = m.prefill_layers(x, gen._kv_cache, gen._page_table, user_id=0, start_pos=0)
        last = ttnn.reshape(ttnn.slice(h, [0, length - 1, 0], [1, length, H]), (1, 1, 1, H))
        gen._greedy_sample(m.lm_head_shards_decode(last), 1, tok)

    body()  # compile
    gen._read_token(tok, 1)
    eager = []
    for _ in range(3):
        start = time.perf_counter()
        body()
        gen._read_token(tok, 1)
        eager.append(time.perf_counter() - start)
    if length > int(os.environ["TT_LAGUNA_PIPE_CHUNK"]):
        return statistics.median(eager), None
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    traced = []
    for _ in range(5):
        start = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        gen._read_token(tok, 1)
        traced.append(time.perf_counter() - start)
    release(mesh, tid)
    return statistics.median(eager), statistics.median(traced)


def ttft_batch32(gen, mesh, length, batch):
    """Per-user TTFT (seconds) for ``batch`` prompts of ``length`` tokens arriving together, prefilled in the server's
    groups (packed when n * length fits PACK_ROWS); one untimed pass first builds the programs."""
    m, H = gen.model, gen.hidden
    bpu = gen._kv_cache[0]["blocks_per_user"]
    table = torch.arange(batch * bpu, dtype=torch.int32).reshape(batch, bpu)
    n = 1
    if length <= PACK_MAX_LEN:
        while n * 2 <= batch and n * 2 * length <= PACK_ROWS:
            n *= 2
    groups = [list(range(i, min(i + n, batch))) for i in range(0, batch, n)]
    tok32 = rep(mesh, torch.zeros([1, 1, 1, 32], dtype=torch.int32), ttnn.uint32)
    onehot = torch.zeros([1, 1, 32, n * length])  # row j picks user j's last token (the server's selection matmul)
    for j in range(n):
        onehot[0, 0, j, j * length + length - 1] = 1.0
    sel = rep(mesh, onehot, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    fills = {}

    def run_group(users):
        ids = torch.cat([random_ids(length, 1000 + u) for u in users]).to(torch.int32).reshape(1, -1)
        x = m.embed_prefill(rep(mesh, ids, ttnn.uint32))
        if len(users) > 1:
            key = tuple(users)
            if key not in fills:
                fills[key] = rep(mesh, table[users], ttnn.int32)
            h = m.prefill_layers_packed(x, gen._kv_cache, fills[key], length, len(users))
        else:
            h = m.prefill_layers(x, gen._kv_cache, gen._page_table, user_id=users[0], start_pos=0)
        last = ttnn.matmul(sel, ttnn.reshape(h, (1, 1, len(users) * length, H)))
        gen._greedy_sample(m.lm_head_shards_decode(last), 32, tok32)
        return gen._read_token(tok32, len(users))

    run_group(groups[0])  # compile
    ttnn.synchronize_device(mesh)
    done = []
    start = time.perf_counter()
    for users in groups:
        run_group(users)
        done += [time.perf_counter() - start] * len(users)
    return done, n


DFLASH_MAX_MODEL_LEN = 1048512  # serve_vllm.sh config with TT_LAGUNA_DFLASH=1


def text_prompt(tokenizer, length):
    """A real request of exactly ``length`` tokens: the chat template around "Summarize this document." and the
    start of the repository's tech_reports/*.md. DFlash's speed depends on how predictable the answer is; random
    tokens make Laguna repeat one token, which the draft predicts perfectly."""
    docs = sorted((REPO_ROOT / "tech_reports").rglob("*.md"))
    text = "\n\n".join(d.read_text(errors="ignore") for d in docs)
    doc = tokenizer(text, add_special_tokens=False)["input_ids"]

    def build(n):
        msg = [{"role": "user", "content": "Summarize this document.\n\n" + tokenizer.decode(doc[:n])}]
        rendered = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        return tokenizer(rendered, add_special_tokens=False)["input_ids"]

    n = max(1, length - (len(build(0)) - 0))
    for _ in range(8):  # re-tokenizing the decoded prefix can shift by a few tokens
        ids = build(n)
        if len(ids) == length:
            break
        n += length - len(ids)
    ids = build(n)[:length]
    return torch.tensor(ids, dtype=torch.int64)
KV_BLOCK = 64


def run_dflash(lens, tokens_out):
    """Batch-1 DFlash through the serving model class, no vLLM. Returns {L: row}."""
    from transformers import AutoConfig
    from vllm_tt_plugin.model_input import TTSamplingParams

    from models.demos.laguna.tt import generator_vllm as gv

    # every KV block of the request is allocated up front (as with the server's 16-token DFlash look-ahead), so
    # every round is a full 16-row round
    gv.LagunaForCausalLM._dflash_lookahead_tokens = staticmethod(lambda: 16)
    mesh = open_mesh(ttnn, resolve_profile("p150x4", trace_region_size=128_000_000))
    rows = {}
    model = None
    try:
        hf = AutoConfig.from_pretrained(os.environ["TT_LAGUNA_MODEL"], trust_remote_code=True)
        model = gv.LagunaForCausalLM.initialize_vllm_model(hf, mesh, 1, max_seq_len=DFLASH_MAX_MODEL_LEN)
        pool = -(-(max(lens) + tokens_out + 64) // KV_BLOCK) + 1
        kv = model.allocate_kv_cache((pool, 2, KV_BLOCK, 128), torch.bfloat16, len(model.model.layers))
        width = pool  # block-table width (the paged KV update needs it <= the cache's block count)
        pt = torch.zeros((1, width), dtype=torch.int32)
        pt[0, :pool] = torch.arange(pool, dtype=torch.int32)
        warm = dict(kv_cache=kv, can_sample_on_device=True)
        dec = dict(kv_cache=kv, max_batch_size=1, num_blocks=width, can_sample_on_device=True)
        model.warmup_model_prefill(enable_trace=False, **warm)
        model.warmup_model_decode(enable_trace=False, **dec)
        if hasattr(model, "already_warmed_up_prefill"):
            model.already_warmed_up_prefill = False
        model.warmup_model_prefill(enable_trace=True, **warm)
        model.warmup_model_decode(enable_trace=True, **dec)
        sp = TTSamplingParams(temperature=[0.0], top_k=[1], top_p=[1.0], seed=[0])

        def first_token(out):
            out = out[0] if isinstance(out, tuple) else out
            return int(torch.as_tensor(out).reshape(-1)[0])

        for length in lens:
            ids = text_prompt(model.tokenizer, length).reshape(1, length)
            start = time.perf_counter()
            tok = first_token(model.prefill_forward(tokens=ids, page_table=pt, kv_cache=kv, enable_trace=True,
                                                    prompt_lens=[length], start_pos=[0], sampling_params=sp))  # fmt: skip
            ttft = time.perf_counter() - start
            rounds_before = len(model._dflash_controller.rounds)
            pos = length
            generated = [tok]
            start = time.perf_counter()
            for i in range(tokens_out):
                out = model.decode_forward(tokens=[[tok]], start_pos=[pos], page_table=pt, kv_cache=kv,
                                           enable_trace=True, read_from_device=True, sampling_params=sp,
                                           reset_batch=(i == 0))  # fmt: skip
                tok, pos = first_token(out), pos + 1
                generated.append(tok)
            seconds = time.perf_counter() - start
            rounds = model._dflash_controller.rounds[rounds_before:]
            full = [r for r in rounds if not r.target_only]
            row = {"ttft_s": ttft, "decode_tok_s": tokens_out / seconds, "rounds": len(rounds),
                   "accepted_mean": statistics.mean(r.accepted_drafts for r in full) if full else 0.0,
                   "draft_ms_mean": statistics.mean(r.draft_ms for r in full) if full else 0.0,
                   "verify_ms_mean": statistics.mean(r.verify_ms for r in full) if full else 0.0,
                   "tokens": generated}  # fmt: skip
            rows[length] = row
            print(f"[perf direct] dflash input {length:>6,}: TTFT {ttft * 1e3:7.1f} ms | decode "
                  f"{row['decode_tok_s']:6.1f} tok/s/user | {len(rounds)} rounds, {row['accepted_mean']:.2f} drafts "
                  f"accepted per round, draft {row['draft_ms_mean']:.1f} ms + verify {row['verify_ms_mean']:.1f} ms",
                  flush=True)  # fmt: skip
    finally:
        if model is not None and hasattr(model, "close_dflash"):
            model.close_dflash()
        close_mesh(ttnn, mesh)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-lens", default=None, help="comma-separated (default 128,1024,2048,4096,8192)")
    parser.add_argument("--batch", default="1,32", help="comma-separated batch sizes: 1 and/or 32 (default both)")
    parser.add_argument("--decode-steps", type=int, default=64, help="decode trace replays timed per point")
    parser.add_argument("--output", default=None, help="JSON results path")
    parser.add_argument("--modes", default="normal", help="normal (default) or dflash (batch 1, DFlash decoding)")
    parser.add_argument("--dflash-tokens", type=int, default=256, help="tokens decoded per input length with DFlash")
    args = parser.parse_args()
    if args.modes == "dflash":
        lens = [int(v) for v in args.input_lens.split(",")] if args.input_lens else DEFAULT_INPUT_LENS
        rows = run_dflash(lens, args.dflash_tokens)
        print("\nBatch 1, DFlash speculative decoding (no vLLM):\n")
        print("| Input tokens | Decode tok/s/user | TTFT | Drafts accepted per round | Draft ms + verify ms per round |")
        print("|---:|---:|---:|---:|---:|")
        for length, r in rows.items():
            print(f"| {length:,} | {r['decode_tok_s']:.1f} | {r['ttft_s'] * 1e3:.0f} ms | {r['accepted_mean']:.2f} | "
                  f"{r['draft_ms_mean']:.1f} + {r['verify_ms_mean']:.1f} |")  # fmt: skip
        if args.output:
            Path(args.output).write_text(json.dumps({"dflash": {str(k): v for k, v in rows.items()}}, indent=1))
        return 0
    lens = [int(v) for v in args.input_lens.split(",")] if args.input_lens else DEFAULT_INPUT_LENS
    batches = [int(v) for v in args.batch.split(",")]
    steps = args.decode_steps

    sol_m = roofline.model_numbers(roofline.checkpoint_dir(os.environ["TT_LAGUNA_MODEL"]))
    mesh = open_mesh(ttnn, resolve_profile("p150x4", trace_region_size=500_000_000))
    gen = None
    results = {}
    try:
        gen = LagunaGenerator.from_pretrained(mesh, max_seq_len=-(-(max(lens) + steps + 256) // 128) * 128)
        for batch in batches:
            rows = {}
            gen._ensure_cache(batch, max(lens) + steps + 64)
            for length in lens:
                sol_step = roofline.decode_step(sol_m, batch, length)["step_ms"]
                sol_ttft = roofline.prefill(sol_m, length, batch)["ttft_s"]
                if batch == 1:
                    eager, traced = ttft_batch1(gen, mesh, length)
                    row = {"ttft_s": eager, "ttft_s_traced": traced}
                    note = f"TTFT {eager * 1e3:8.1f} ms eager" + (f", {traced * 1e3:.1f} ms traced" if traced else "")
                else:
                    per_user, n = ttft_batch32(gen, mesh, length, batch)
                    row = {"ttft_s_mean": statistics.mean(per_user), "ttft_s_first": per_user[0],
                           "ttft_s_last": per_user[-1], "packed_per_group": n}  # fmt: skip
                    note = (f"TTFT mean {row['ttft_s_mean']:6.2f} s (first {per_user[0]:.2f}, last {per_user[-1]:.2f}, "
                            f"{n} per prefill)")  # fmt: skip
                ms = time_decode(gen, mesh, batch, length, steps)
                row.update({"decode_ms_per_step": ms, "decode_tok_s_user": 1e3 / ms, "decode_tok_s_total": batch * 1e3 / ms,
                            "sol_decode_tok_s_user": 1e3 / sol_step, "sol_ttft_s": sol_ttft})  # fmt: skip
                rows[length] = row
                print(f"[perf direct] batch {batch:2d} input {length:>6,}: {note} | decode {1e3 / ms:6.1f} tok/s/user "
                      f"({ms:.2f} ms/step) | SoL decode {1e3 / sol_step:.0f}, TTFT {sol_ttft * 1e3:,.0f} ms", flush=True)
            results[batch] = rows
    finally:
        if gen is not None:
            gen.teardown()
        close_mesh(ttnn, mesh)

    for batch, rows in results.items():
        print(f"\nBatch {batch} (device only, no vLLM). Targets = 50% of SoL: half the decode speed, twice the TTFT.\n")
        if batch == 1:
            print("| Input tokens | Decode tok/s/user | Decode SoL | Decode target | TTFT eager | TTFT traced | TTFT SoL | "
                  "TTFT target |")
            print("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for length, r in rows.items():
                traced = f"{r['ttft_s_traced'] * 1e3:.1f} ms" if r["ttft_s_traced"] else "-"
                print(f"| {length:,} | {r['decode_tok_s_user']:.1f} | {r['sol_decode_tok_s_user']:.0f} | "
                      f"{r['sol_decode_tok_s_user'] / 2:.0f} | {r['ttft_s'] * 1e3:.1f} ms | {traced} | "
                      f"{r['sol_ttft_s'] * 1e3:.0f} ms | {2 * r['sol_ttft_s'] * 1e3:.0f} ms |")  # fmt: skip
        else:
            print("| Input tokens | Decode tok/s/user | Decode SoL | Decode target | Decode tok/s total | TTFT last | "
                  "TTFT mean | TTFT SoL | TTFT target | Users per prefill |")
            print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for length, r in rows.items():
                print(f"| {length:,} | {r['decode_tok_s_user']:.1f} | {r['sol_decode_tok_s_user']:.0f} | "
                      f"{r['sol_decode_tok_s_user'] / 2:.0f} | {r['decode_tok_s_total']:.0f} | {r['ttft_s_last'] * 1e3:,.0f} ms | "
                      f"{r['ttft_s_mean'] * 1e3:,.0f} ms | {r['sol_ttft_s'] * 1e3:,.0f} ms | {2 * r['sol_ttft_s'] * 1e3:,.0f} ms | "
                      f"{r['packed_per_group']} |")  # fmt: skip
    if args.output:
        Path(args.output).write_text(json.dumps({str(b): r for b, r in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

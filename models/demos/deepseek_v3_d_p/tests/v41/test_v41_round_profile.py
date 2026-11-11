# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The served speculative round as the generator runs it, for timing and for the device profiler:
``initialize_vllm_model`` and the two-phase warm-up as ``test_v41_spec_generator.py`` (every
served capture), then ``V41_RP_LIVE`` requests (1: the fused one-slot round, the ``vd:c1`` variant; 32: the
uniform
multi-slot step ``v:32`` and its standalone draft call) prefilled one after another from prompts of
``V41_RP_PROMPT`` tokens of real text and decoded together through the plugin's ragged contract (``[rows, 1 +
K]``
tokens a call, the drafts ``take_drafts`` gave fed back: a call is a round).

``V41_RP_DISTINCT=1`` (default): every request's prompt is the book rotated by its own offset, so the requests
route to different experts as served requests do (32 identical prompts understate the routed
expert FFN); ``0``: one prompt for all. ``V41_RP_WARM`` rounds run first (the joins land in the first), then
``V41_RP_ROUNDS`` measured rounds. ``V41_RP_PROFILE=1`` (under ``TT_METAL_DEVICE_PROFILER=1`` with the C++
post-process and mid-run dumps): the device profiler is read after the warm-up, after the
prefills and after every round, so a round's markers never overflow the buffer and each measured round is one
replay session of its capture in ``cpp_device_perf_report.csv``. ``V41_RP_DRAIN_SKIP=warm-up,..`` leaves the
named
reads out (their phases overflow the buffers and are dropped). ``V41_RP_TEMP`` (default 0: greedy) samples
every
row at that temperature (the device program is the same). ``V41_RP_OUT`` takes the JSON: per
round the wall, the verify step's and the draft call's logged items and every row's emitted count and tokens
(two sessions of one prompt set and seed are compared token for token).

``V41_RP_OPMAP=<path>``: during the warm-up (where every served capture is made) a ``sys.setprofile`` hook
records,
for every call of a function of this model's ``tt/`` sources, the device operation ids it issued (``[start,
end)``
of ``ttnn._ttnn.get_device_operation_id``, the call depth, ``file:function`` and the ``layer`` / ``source``
argument
if it has one), written to the path as JSON. A replayed op keeps its captured id (the report's ``GLOBAL CALL
COUNT``
is ``id * 1024 + device``), so every row of a traced session is attributed to the model function that issued
it by
the innermost span holding its id, with no anchors on the op order. The hook is removed before the
prefills."""

import json
import os
import sys
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _dram_bytes, parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_generator_vllm import _page_table, _trace_bytes
from models.demos.deepseek_v3_d_p.tt.v41 import generator_vllm
from models.demos.deepseek_v3_d_p.tt.v41.generator_vllm import DeepseekV41ForCausalLM
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint


class _Params:  # the plugin's TTSamplingParams as lists per row
    def __init__(self, n: int, temperature: float):
        greedy = temperature <= 0
        self.temperature = [0.0 if greedy else temperature] * n
        self.top_k = [1 if greedy else 0] * n
        self.top_p = [0.0 if greedy else 1.0] * n
        self.repetition_penalty = 1.0


class _OpMap:
    """The ``sys.setprofile`` hook of ``V41_RP_OPMAP``: spans of device operation ids by the function that
    issued
    them (the functions under ``root``, this model's ``tt/`` sources; spans that issued no op are dropped)."""

    HINTS = ("layer", "layer_idx", "source")

    def __init__(self, root: str):
        self.root = root
        self.stack: list[tuple] = []
        self.spans: list[tuple] = []
        self.op_id = ttnn._ttnn.get_device_operation_id

    def __call__(self, frame, event, arg):
        if event == "call":
            code = frame.f_code
            if code.co_filename.startswith(self.root):
                hint = None
                for name in self.HINTS:
                    value = frame.f_locals.get(name)
                    if isinstance(value, int) and not isinstance(value, bool):
                        hint = value
                        break
                name = f"{os.path.basename(code.co_filename)}:{code.co_name}"
                self.stack.append((id(frame), name, hint, self.op_id()))
        elif event == "return" and self.stack and self.stack[-1][0] == id(frame):
            _, name, hint, start = self.stack.pop()
            end = self.op_id()
            if end > start:
                self.spans.append((start, end, len(self.stack), name, hint))

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {"root": self.root, "fields": ["start", "end", "depth", "function", "hint"], "spans": self.spans}, f
            )


def _book() -> torch.Tensor:
    """Every token of the tests' book (``oracle.text_tokens`` refuses a length past it and names what it has)."""
    try:
        return orc.text_tokens(10**8)[0]
    except ValueError as e:
        return orc.text_tokens(int(str(e).split()[0]))[0]


@pytest.mark.timeout(int(os.environ.get("V41_RP_TIMEOUT", 4 * 3600)))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    galaxy_meshes(trace_region_size=int(os.environ.get("V41_RP_TRACE_REGION", 3489660928))),
    indirect=True,
)
def test_v41_round_profile(mesh_device, device_params, monkeypatch):
    k = int(os.environ.get("TT_V41_SPEC", "0"))
    assert k > 0, "run with TT_V41_SPEC=K"
    layers = parse_layers(os.environ.get("V41_RP_LAYERS", "all"))
    prompt_len = int(os.environ.get("V41_RP_PROMPT", 32768))
    live = int(os.environ.get("V41_RP_LIVE", 1))
    batch = int(os.environ.get("V41_RP_BATCH", 32))
    distinct = os.environ.get("V41_RP_DISTINCT", "1") == "1"
    warm = int(os.environ.get("V41_RP_WARM", 8))
    rounds = int(os.environ.get("V41_RP_ROUNDS", 16))
    profile = os.environ.get("V41_RP_PROFILE", "0") == "1"
    # the reads to leave out (prefixes of "warm-up", "prefills", "warm round N", "round N"): with the raw
    # device log
    # written, every read dumps what the buffers hold, and an unread phase only overflows them (its markers
    # dropped)
    drain_skip = [w for w in os.environ.get("V41_RP_DRAIN_SKIP", "").split(",") if w]
    temperature = float(os.environ.get("V41_RP_TEMP", 0))
    max_seq_len = int(os.environ.get("V41_RP_MAX_SEQ", 1053696))
    assert 1 <= live <= batch, (live, batch)
    monkeypatch.setenv("TT_V41_LAYERS", ",".join(str(l) for l in layers))
    if resolve_checkpoint() is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    result = {
        "k": k,
        "layers": len(layers),
        "prompt": prompt_len,
        "live": live,
        "batch": batch,
        "distinct": distinct,
        "warm": warm,
        "rounds_requested": rounds,
        "profile": profile,
        "temperature": temperature,
        "max_model_len": max_seq_len,
        "flags": {kk: v for kk, v in os.environ.items() if kk.startswith(("V41_", "TT_V41_", "TT_METAL_"))},
    }
    drains = []

    def drain(what: str) -> None:
        if profile and not any(what.startswith(skip) for skip in drain_skip):
            t0 = time.time()
            ttnn.ReadDeviceProfiler(mesh_device)
            drains.append({"after": what, "s": round(time.time() - t0, 1)})
            logger.info(f"round profile: profiler read after {what} in {drains[-1]['s']} s")

    opmap_path = os.environ.get("V41_RP_OPMAP")
    opmap = _OpMap(str(Path(generator_vllm.__file__).resolve().parents[1]) + os.sep) if opmap_path else None
    try:
        result["device_ids"] = [int(d) for d in mesh_device.get_device_ids()]  # mesh row-major
    except Exception as e:
        logger.warning(f"device ids unavailable: {e}")
    t0 = time.time()
    model = DeepseekV41ForCausalLM.initialize_vllm_model(
        None, mesh_device, max_batch_size=batch, max_seq_len=max_seq_len
    )
    result["build_s"] = round(time.time() - t0, 1)
    chunk = result["chunk"] = int(model.model.chunk)  # the prefill chunk the generator was built on (TT_V41_CHUNK)
    result["dram_after_build_bytes"] = _dram_bytes(mesh_device)
    kv = model.allocate_kv_cache((16, 1, 64, 512), torch.bfloat16, len(layers))
    t0 = time.time()
    if opmap is not None:
        sys.setprofile(opmap)
    try:
        for enable_trace in (False, True):
            model.warmup_model_prefill(enable_trace=enable_trace, kv_cache=kv, can_sample_on_device=False)
            model.warmup_model_decode(
                enable_trace=enable_trace, kv_cache=kv, max_batch_size=batch, num_blocks=16, can_sample_on_device=False
            )
    finally:
        if opmap is not None:
            sys.setprofile(None)
            opmap.dump(opmap_path)
            result["opmap_spans"] = len(opmap.spans)
    result["warmup_s"] = round(time.time() - t0, 1)
    result["dram_after_warmup_bytes"] = _dram_bytes(mesh_device)
    result["trace_after_warmup_bytes"] = _trace_bytes(mesh_device)
    result["captures"] = sorted(model.decoder.captures)
    logger.info(
        f"round profile: built in {result['build_s']} s, warmed up in {result['warmup_s']} s; "
        f"captures {result['captures']}"
    )
    drain("warm-up")

    book = _book()
    have = int(book.numel())
    reps = -(-prompt_len // have)
    prompts = []
    for i in range(live):
        offset = (i * (have // live)) if distinct else 0
        prompts.append(torch.roll(book, -offset).repeat(reps)[:prompt_len])
    result["book_tokens"] = have
    keys = list(range(1, live + 1))  # the requests' first block ids

    def prefill(key: int, prompt: torch.Tensor) -> int:
        n = int(prompt.numel())
        out = None
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            out = model.prefill_forward(
                tokens=prompt.view(1, -1).to(torch.int32),
                prompt_lens=torch.tensor([end]),
                page_table=_page_table([key]),
                start_pos=torch.tensor([start], dtype=torch.int32),
                kv_cache=kv,
                enable_trace=True,
                prompt_total_lens=torch.tensor([n]),
            )
        assert out.shape == (1, 1, C.VOCAB_SIZE)
        return int(out[0, 0].argmax())

    t0 = time.time()
    last = []
    prefill_s = []
    for key, prompt in zip(keys, prompts):
        t1 = time.time()
        last.append(prefill(key, prompt))
        prefill_s.append(round(time.time() - t1, 2))
    result["prefill_s"] = prefill_s
    result["prefill_total_s"] = round(time.time() - t0, 1)
    result["first_tokens"] = list(last)
    logger.info(f"round profile: {live} prefills of {prompt_len} tokens in {result['prefill_total_s']} s")
    drain("prefills")

    drafts = [[] for _ in keys]
    emitted_n = [0] * live
    table = _page_table(keys)

    def one_round(tag: str) -> dict:
        tokens = torch.tensor(
            [[last[i]] + drafts[i] + [-1] * (k - len(drafts[i])) for i in range(live)], dtype=torch.int32
        )
        positions = torch.tensor([prompt_len + emitted_n[i] for i in range(live)], dtype=torch.int32)
        t0 = time.time()
        out = model.decode_forward(
            tokens=tokens,
            start_pos=positions,
            page_table=table,
            kv_cache=kv,
            enable_trace=True,
            read_from_device=True,
            sampling_params=_Params(live, temperature),
        )
        wall = time.time() - t0
        given = [len(d) for d in drafts]
        proposed = model.take_drafts()
        emitted, rows_out = [], []
        for i in range(live):
            row = [int(t) for t in out[i].tolist() if t >= 0]
            assert row, f"row {i}: a round emits at least one token"
            emitted.append(len(row))
            rows_out.append(row)
            emitted_n[i] += len(row)
            last[i] = row[-1]
            drafts[i] = [int(t) for t in proposed.get(i, [])][:k]
        entry = {
            "tag": tag,
            "wall_s": round(wall, 4),
            "given": given,
            "emitted": emitted,
            "tokens": rows_out,
            "verify": dict(model.decoder.last),
            "draft": dict(model.drafter.last) if live > 1 else None,
        }
        logger.info(f"round profile: round {json.dumps(entry, default=str)}")
        return entry

    log = []
    for r in range(warm):
        log.append(one_round(f"warm{r}"))
        drain(f"warm round {r}")
    for r in range(rounds):
        log.append(one_round(f"round{r}"))
        drain(f"round {r}")
    measured = [e for e in log if e["tag"].startswith("round")]
    steady = [e for e in measured if e["verify"].get("mode") == "replay"]

    def med(values: list) -> float | None:
        values = sorted(v for v in values if v is not None)
        return round(values[len(values) // 2], 4) if values else None

    result["rounds"] = log
    result["drains"] = drains
    result["summary"] = {
        "measured": len(measured),
        "steady": len(steady),
        "variants": sorted({str(e["verify"].get("variant")) for e in steady}),
        "wall_s_median": med([e["wall_s"] for e in steady]),
        "verify_device_s_median": med([e["verify"].get("device_s") for e in steady]),
        "verify_inputs_s_median": med([e["verify"].get("inputs_s") for e in steady]),
        "draft_device_s_median": med([(e["draft"] or {}).get("device_s") for e in steady]),
        "tokens_per_round_per_user": (
            round(sum(sum(e["emitted"]) for e in steady) / (len(steady) * live), 3) if steady else None
        ),
    }
    result["dram_after_bytes"] = _dram_bytes(mesh_device)
    logger.info("V41_ROUND_PROFILE_RESULT " + json.dumps({kk: v for kk, v in result.items() if kk != "rounds"}))
    out_path = os.environ.get("V41_RP_OUT")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1, default=str)
    for trace in model.prefill_traces.values():
        trace.release()
    model.decoder.release()
    assert steady, "no measured round replayed its capture"

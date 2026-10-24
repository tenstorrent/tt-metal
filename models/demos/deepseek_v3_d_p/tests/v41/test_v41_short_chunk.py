# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The short-chunk prefill geometry (``TT_V41_SHORT_CHUNK``) against the full-chunk path,
through ``DeepseekV41ForCausalLM`` as the generator test: a prompt of ``V41_SC_PROMPT`` tokens (1,028: the
served
isl 1,024 plus the template's 4) that fits the short chunk is prefilled by ``prefill_forward`` (the short
state's
capture), eagerly through the full-chunk model into a full-chunk state, and by a capture on that state; the
three
are compared on the logits and on every cache row the prompt wrote (KV, index-K, window carries, compressor
carries), and the first prefill's device time is recorded for each. Then ``V41_SC_STEPS`` decode steps follow
the
short-state join and the full-state join and their tokens are compared. A prompt of ``V41_SC_LONG`` tokens
(2,052) that does not fit the short chunk must take the full-chunk path. The warm-up runs in the served order
and,
after its decode steps, main L1's resident blocks must still be one contiguous region at the top (the decode
L1 allocation clash). ``V41_SC_MAX_SEQ`` (16,384) is the context: the served 1,053,696 makes
the test the server's warm-up (the trace region then defaults to the catalog's 2 GiB,
``V41_SC_TRACE_REGION``).
Results: ``V41_SC_RESULT {json}``, ``V41_SC_OUT``."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_decode import TRACE_REGION
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _dram_bytes, parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_generator_vllm import _page_table
from models.demos.deepseek_v3_d_p.tt.v41.cache import WINDOW_SLOT, V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.generator_vllm import DeepseekV41ForCausalLM
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat


def _rows(mesh_device, t: ttnn.Tensor, first: int, count: int) -> torch.Tensor:
    """Rows ``[first, first + count)`` of the replicated state tensor ``t`` on the first and the last chip."""
    if t.dtype == MlaKvCacheFormat.SCALED_FP8.storage_dtype:
        every = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        picked = [every[0], every[-1]]
    else:
        chips = ttnn.get_device_tensors(t)
        picked = [ttnn.to_torch(chips[0])[0], ttnn.to_torch(chips[-1])[0]]
    return torch.stack([x[..., first : first + count, :] for x in picked])


def _snapshot(mesh_device, logits: torch.Tensor, state: V41PrefillState, total: int) -> dict:
    """What a ``total``-token prompt wrote, in geometry-independent coordinates: per source the window slot
    and
    the prompt's window scratch rows, the compressed rows and the index-K rows it completed; every layer's
    window
    carry; the compressor carries."""
    g, cfg = state.geometry, state.config
    out = {"logits": logits}
    for source, t in state.kv.items():
        ratio = cfg.compress_ratio(source)
        out[f"kv[{source}] window"] = _rows(mesh_device, t, 0, WINDOW_SLOT + total)
        out[f"kv[{source}] compressed"] = _rows(mesh_device, t, g.kv_row_of_compressed(0), total // ratio)
    if state.swa_scratch is not None:
        out["swa_scratch"] = _rows(mesh_device, state.swa_scratch, 0, WINDOW_SLOT + total)
    for source, t in state.index_k.items():
        out[f"index_k[{source}]"] = _rows(mesh_device, t, 0, total // cfg.compress_ratio(source))
    for layer, t in state.window_carry.items():
        out[f"window_carry[{layer}]"] = _rows(mesh_device, t, 0, t.shape[-2])
    for source, carry in state.compressor_carry.items():
        for i, t in enumerate(carry or ()):
            out[f"compressor_carry[{source}][{i}]"] = _rows(mesh_device, t, 0, t.shape[-2])
    return out


def _differences(name: str, expected: dict, actual: dict) -> list[str]:
    out = []
    for k, want in expected.items():
        got = actual.get(k)
        if got is None:
            out.append(f"{name}: {k} missing")
            continue
        if want.shape != got.shape:
            out.append(f"{name}: {k} shape {tuple(want.shape)} vs {tuple(got.shape)}")
            continue
        if torch.equal(want, got):
            continue
        bad = want != got
        count = int(bad.sum())
        worst = float((want.float() - got.float()).abs().max())
        if want.dim() == 1:  # the logits
            out.append(
                f"{name}: {k} ({count} of {want.numel()} values, max |diff| {worst:.4g}, argmax "
                f"{int(want.argmax())} vs {int(got.argmax())}, max |want| {float(want.abs().max()):.4g})"
            )
            continue
        rows = bad.reshape(-1, want.shape[-2], want.shape[-1]).any(-1).any(0).nonzero().flatten()
        out.append(
            f"{name}: {k} ({count} of {want.numel()} values, rows {int(rows.min())}..{int(rows.max())} of "
            f"{want.shape[-2]}, max |diff| {worst:.4g})"
        )
    return out


def _trace_region() -> int:
    """The decode test's region, or the served 2 GiB when ``V41_SC_MAX_SEQ`` is past the first score bucket
    (the
    generic captures of every bucket up to it: 33 at the 1M catalog)."""
    return TRACE_REGION if int(os.environ.get("V41_SC_MAX_SEQ", 16384)) <= 16384 else 2 * 1024**3


def _l1_resident(mesh_device) -> dict:
    """Main L1's allocated blocks (allocator-relative addresses): their count, bytes, lowest address and end,
    and
    whether they form one contiguous region at the top of L1 (``contiguous``: no free gap between them)."""
    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    blocks = [(int(r["address"]), int(r["size"])) for r in v.block_table if r.get("allocated") == "yes"]
    if not blocks:
        return {"blocks": 0, "bytes": 0, "lowest": None, "end": None, "contiguous": True}
    lowest, end = min(a for a, _ in blocks), max(a + n for a, n in blocks)
    total = sum(n for _, n in blocks)
    return {"blocks": len(blocks), "bytes": total, "lowest": lowest, "end": end, "contiguous": lowest + total == end}


@pytest.mark.timeout(int(os.environ.get("V41_SC_TIMEOUT", 2 * 3600)))
@pytest.mark.parametrize(
    "mesh_device, device_params",
    galaxy_meshes(trace_region_size=int(os.environ.get("V41_SC_TRACE_REGION", _trace_region()))),
    indirect=True,
)
def test_v41_short_chunk(mesh_device, device_params, monkeypatch):
    layers = parse_layers(os.environ.get("V41_SC_LAYERS", "all"))
    prompt = int(os.environ.get("V41_SC_PROMPT", 1028))
    long = int(os.environ.get("V41_SC_LONG", 2052))
    steps = int(os.environ.get("V41_SC_STEPS", 8))
    batch = int(os.environ.get("V41_SC_BATCH", 8))
    chunk = int(os.environ.get("TT_V41_CHUNK", 5120))
    short = os.environ.get("TT_V41_SHORT_CHUNK", "2048")
    repeat = int(os.environ.get("V41_SC_REPEAT", 3))
    monkeypatch.setenv("TT_V41_CHUNK", str(chunk))
    monkeypatch.setenv("TT_V41_SHORT_CHUNK", short)
    monkeypatch.setenv("TT_V41_LAYERS", ",".join(str(l) for l in layers))
    monkeypatch.setenv("TT_V41_TRACE_PREFILL_LENGTHS", f"{prompt},{long}")
    monkeypatch.setenv("TT_V41_TRACE_PREFILL_SLACK", "0")
    monkeypatch.setenv("TT_V41_PREFILL_GENERIC", os.environ.get("TT_V41_PREFILL_GENERIC", "1"))
    monkeypatch.setenv("TT_V41_PREFILL_SLOTS", os.environ.get("TT_V41_PREFILL_SLOTS", "1"))
    if resolve_checkpoint() is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    # the full-chunk state's context: at least the first score-width bucket, so both geometries score at 8K
    max_seq_len = int(os.environ.get("V41_SC_MAX_SEQ", 16384))
    result = {
        "layers": layers,
        "prompt": prompt,
        "long": long,
        "steps": steps,
        "chunk": chunk,
        "short": short,
        "max_seq": max_seq_len,
    }

    t0 = time.time()
    dram0 = _dram_bytes(mesh_device)
    model = DeepseekV41ForCausalLM.initialize_vllm_model(
        None, mesh_device, max_batch_size=batch, max_seq_len=max_seq_len
    )
    result["build_s"] = round(time.time() - t0, 1)
    result["short_chunks"] = [c for c, _, _ in model.short]
    result["short_max_seq"] = [v.max_seq_len for _, v, _ in model.short]
    result["dram_before_build"] = dram0
    result["dram_after_build"] = _dram_bytes(mesh_device)
    kv = model.allocate_kv_cache((16, 1, 64, 512), torch.bfloat16, len(layers))
    t0 = time.time()
    model.warmup_model_prefill(enable_trace=False, kv_cache=kv, can_sample_on_device=False)
    # the served order: the warm-up's program-cache entries (the collectives' global
    # semaphores) are resident 64-byte L1 blocks, stacked from the top of L1 while nothing else holds it. An
    # entry
    # first built inside a decode step (the legacy ttnn.reduce_scatter of the decode tail) lands under the
    # step's
    # transients instead, far below the stack, and the routed expert FFN's static CBs then clash with it at
    # the next
    # block ("L1 buffer allocated at 911104"): after the decode warm-up the resident blocks must still be one
    # contiguous region at the top
    result["l1_before_decode"] = _l1_resident(mesh_device)
    model.warmup_model_decode(
        enable_trace=False, kv_cache=kv, max_batch_size=batch, num_blocks=16, can_sample_on_device=False
    )
    result["l1_after_decode"] = _l1_resident(mesh_device)
    logger.info(f"L1 resident before decode {result['l1_before_decode']} after {result['l1_after_decode']}")
    assert result["l1_after_decode"]["contiguous"], (
        f"the decode warm-up left an L1 block under the step's transients: {result['l1_after_decode']} "
        f"(before the decode: {result['l1_before_decode']})"
    )
    model.warmup_model_prefill(enable_trace=True, kv_cache=kv, can_sample_on_device=False)
    model.warmup_model_decode(
        enable_trace=True, kv_cache=kv, max_batch_size=batch, num_blocks=16, can_sample_on_device=False
    )
    result["warmup_s"] = round(time.time() - t0, 1)
    result["dram_after_warmup"] = _dram_bytes(mesh_device)
    result["prefill_traces"] = sorted(
        f"chunk {model.models[k[0]].chunk} length {k[1]} segments {t.num_segments}"
        for k, t in model.prefill_traces.items()
    )
    assert model.short, "no short chunk geometry was built"
    short_chunk, short_model, short_state = model.short[0]
    assert model._short_state(prompt) is short_state, f"the {prompt}-token prompt has no short capture"
    assert model._short_state(long) is None, f"the {long}-token prompt fits the short chunk"

    text = orc.text_tokens(2 * long + steps + 2)[0]
    book = text[: prompt + steps + 1]
    other = text[long : 2 * long]

    def prefill(block: int, tokens: torch.Tensor) -> tuple[torch.Tensor, float, dict]:
        t0 = time.time()
        out = model.prefill_forward(
            tokens=tokens[None].to(torch.int32),
            prompt_lens=torch.tensor([int(tokens.numel())]),
            page_table=_page_table([block]),
            start_pos=torch.tensor([0], dtype=torch.int32),
            kv_cache=kv,
            enable_trace=True,
        )
        wall = time.time() - t0
        state = model.pending[block][0]
        trace = model.prefill_traces.get((id(state), int(tokens.numel())))
        last = dict(trace.last) if trace is not None and getattr(trace, "last", None) else {}
        return out[0, 0].clone(), wall, {"state_chunk": model.models[id(state)].chunk, "wall_s": round(wall, 3), **last}

    sample_on_device = DeepseekV41ForCausalLM.SAMPLE_ON_DEVICE

    class _Params:
        def __init__(self, n: int):
            self.temperature, self.top_k, self.top_p = [0.0] * n, [1] * n, [0.0] * n
            self.repetition_penalty = 1.0

    def decode(rows: list[tuple[int, int, int]]) -> list[int]:
        out = model.decode_forward(
            tokens=torch.tensor([[t] for _, t, _ in rows], dtype=torch.int32),
            start_pos=torch.tensor([p for _, _, p in rows], dtype=torch.int32),
            page_table=_page_table([b for b, _, _ in rows]),
            kv_cache=kv,
            enable_trace=True,
            read_from_device=True,
            sampling_params=_Params(len(rows)) if sample_on_device else None,
        )
        if sample_on_device:
            return [int(x) for x in out]
        return [int(out[i, 0].argmax()) for i in range(len(rows))]

    # --- A: the short state's capture, replayed ``repeat`` times (the first prefill_forward's phases) ---
    runs = []
    for k in range(repeat):
        logits_a, wall, phases = prefill(100 + k, book[:prompt])
        assert phases["state_chunk"] == short_chunk, phases
        runs.append(phases)
        if k == 0:
            snap_a = _snapshot(mesh_device, logits_a, short_state, prompt)
        else:
            assert torch.equal(logits_a, snap_a["logits"]), "the short replays differ run to run"
    result["short_replays"] = runs
    logger.info(f"short chunk {short_chunk}: replays {runs}")
    # the pending requests' joins (into free slots) before the states are reused below
    tokens_a = decode([(100, int(book[prompt]), prompt)])
    for k in range(1, repeat):
        decode([(100 + k, int(book[prompt]), prompt)])
    # --- B: the full-chunk model, eagerly, into a full-chunk state ---
    full_state = model.prefill_states[0]
    t0 = time.time()
    logits_b = model.model.prefill_into(book[:prompt], full_state)[-1].clone()
    result["full_eager_wall_s"] = round(time.time() - t0, 3)
    snap_b = _snapshot(mesh_device, logits_b, full_state, prompt)
    diffs = _differences("short replay vs full eager", snap_b, snap_a)
    result["short_vs_full_eager"] = diffs
    for d in diffs:
        logger.warning(d)
    # --- C: the full-chunk state's capture of the same length (what the served path replayed before) ---
    t0 = time.time()
    trace_c = model._capture_prefill(full_state, prompt)
    result["full_capture_s"] = round(time.time() - t0, 1)
    runs = []
    for k in range(repeat):
        logits_c, _ = trace_c.run(book[:prompt])
        logits_c = logits_c[-1].clone()
        runs.append({k: v for k, v in trace_c.last.items() if isinstance(v, float)})
    result["full_replays"] = runs
    logger.info(f"full chunk {chunk}: replays {runs}")
    snap_c = _snapshot(mesh_device, logits_c, full_state, prompt)
    diffs = _differences("full replay vs full eager", snap_b, snap_c)
    result["full_replay_vs_full_eager"] = diffs
    for d in diffs:
        logger.warning(d)
    result["short_vs_full_replay"] = _differences("short replay vs full replay", snap_c, snap_a)
    a, b = snap_a["logits"].double(), snap_c["logits"].double()
    top = lambda t, k: set(t.topk(k).indices.tolist())
    result["logits_vs_full"] = {
        "pcc": round(float(torch.corrcoef(torch.stack([a, b]))[0, 1]), 6),
        "max_abs_diff": round(float((a - b).abs().max()), 4),
        "max_abs": round(float(b.abs().max()), 4),
        "argmax_equal": int(a.argmax()) == int(b.argmax()),
        "top5_overlap": len(top(a, 5) & top(b, 5)),
        "top20_overlap": len(top(a, 20) & top(b, 20)),
    }
    logger.info(f"short vs full logits: {result['logits_vs_full']}")
    # the full state's prefill joins a slot through the generator's bookkeeping, then the decode tokens
    key_b = 200
    model.pending[key_b] = (full_state, book[:prompt], {})
    tokens_b = decode([(key_b, int(book[prompt]), prompt)])
    preds_a, preds_b = [tokens_a[0]], [tokens_b[0]]
    for s in range(1, steps):
        p = prompt + s
        fed = int(book[p])
        preds_a.append(decode([(100, fed, p)])[0])
        preds_b.append(decode([(key_b, fed, p)])[0])
    result["decode_tokens_short"] = preds_a
    result["decode_tokens_full"] = preds_b
    result["decode_tokens_equal"] = preds_a == preds_b
    result["book_top1_short"] = round(sum(t == int(book[prompt + s + 1]) for s, t in enumerate(preds_a)) / steps, 4)
    result["book_top1_full"] = round(sum(t == int(book[prompt + s + 1]) for s, t in enumerate(preds_b)) / steps, 4)
    # --- D: a prompt that does not fit the short chunk takes the full-chunk path ---
    logits_d, wall, phases = prefill(300, other[:long])
    assert phases["state_chunk"] == chunk, phases
    result["long_prefill"] = phases
    decode([(300, int(logits_d.argmax()), long)])
    # --- E: the short prompt again after the longer one (the states are reused) ---
    logits_e, wall, phases = prefill(301, book[:prompt])
    assert phases["state_chunk"] == short_chunk, phases
    result["short_after_long"] = phases
    result["short_after_long_equal"] = bool(torch.equal(logits_e, snap_a["logits"]))
    result["dram_after_requests"] = _dram_bytes(mesh_device)
    logger.info("V41_SC_RESULT " + json.dumps(result))
    out = os.environ.get("V41_SC_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
    assert not result["short_vs_full_replay"] or os.environ.get("V41_SC_STRICT", "0") != "1"
    for trace in model.prefill_traces.values():
        trace.release()
    model.decoder.release()

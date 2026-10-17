# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""``V41GenericPrefillTrace`` (position-generic chunk captures, one per score-width bucket) against the untraced
prefill on real layers at a long prompt.

A prompt of ``V41_GENERIC_SEQ`` tokens (default 81,907: 16 chunks of 5,120, an odd final chunk so the compressor
carry runs) through ``V41_GENERIC_LAYERS`` (default the sharing schedule 0,2,3,20,21,24: a ratio-0 layer, a ratio-2
source and consumer, the candidate source, a consumer and a candidate index source) must give, bit-identically to
``TtV41Transformer.prefill`` (a fresh classic state): the last position's logits and every cache row the prompt
wrote (KV, index-K, window carries, compressor carry), (a) through a generic state untraced and (b) replayed from
the bucket captures, for the captured prompt, another prompt and the first again. Reports the eager and traced
times per chunk, the capture count, segments and trace-region bytes, and the chunks replayed.

Longer prompts exercise more of the selection's width-dependent paths with the production candidate count
(2,048 blocks): candidates above 16,384 rows, the two-level selection above 65,536 (13+ chunks), the candidate
index sources' sparse scoring above 327,680 (65+ chunks). ``V41_GENERIC_OUT`` writes the report as JSON.
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _dram_bytes, parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.cache import WINDOW_SLOT
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer, V41GenericPrefillTrace
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

TRACE_REGION = 2 * 1024 * 1024 * 1024  # the served configuration's
DEFAULT_LAYERS = "0,2,3,20,21,24"
CHUNK = 5120


def _trace_bytes(mesh_device) -> int | None:
    try:
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.TRACE)
        return int(view.total_bytes_allocated_per_bank * view.num_banks)
    except Exception as e:
        logger.warning(f"trace statistics unavailable: {e}")
        return None


def _prompts(total: int, tokenizer) -> list[torch.Tensor]:
    """Two different prompts [total] of real text (the book, repeated when it runs out)."""
    import bz2

    book = bz2.open(orc._HERE.parents[3] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2", "rt").read()
    text = book[book.index("It was the best of times") :]
    text = (text * (24 * total // len(text) + 1))[: 24 * total]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= 2 * total, f"{len(ids)} tokens of text, {2 * total} needed"
    ids = torch.tensor(ids, dtype=torch.int64)
    return [ids[:total], ids[total : 2 * total]]


def _snapshot(mesh_device, logits, state, total: int) -> dict:
    """The logits and the rows the prompt wrote: per source its KV rows (window slot, scratch and the compressed
    rows), its index-K rows, every layer's window carry, the compressor carries."""

    def down(t, rows=None):
        # the state tensors are replicated: the first and the last chip's copies, the rows the prompt wrote
        chips = ttnn.get_device_tensors(t)
        picked = [ttnn.to_torch(chips[0]), ttnn.to_torch(chips[-1])]
        return torch.stack([x if rows is None else x[:, :, :rows] for x in picked])

    g, out = state.geometry, {"logits": logits}
    # the window scratch rows past the final chunk's valid length are dead (never read; their values are whatever
    # the padded rows computed over stale inputs), so they are left out of the KV comparison
    dead = slice(WINDOW_SLOT + (total - (total - 1) // g.chunk * g.chunk), g.window_rows)
    for source, t in state.kv.items():
        rows = down(t, g.kv_row_of_compressed(total // state.config.compress_ratio(source)))
        rows[:, :, :, dead] = 0
        out[f"kv[{source}]"] = rows
    if state.swa_scratch is not None:
        rows = down(state.swa_scratch)
        rows[:, :, :, dead] = 0
        out["swa_scratch"] = rows
    for source, t in state.index_k.items():
        out[f"index_k[{source}]"] = down(t, total // state.config.compress_ratio(source))
    for layer, t in state.window_carry.items():
        out[f"window_carry[{layer}]"] = down(t)
    for source, carry in state.compressor_carry.items():
        for i, t in enumerate(carry or ()):
            out[f"compressor_carry[{source}][{i}]"] = down(t)
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
        rows = bad.reshape(-1, want.shape[-2], want.shape[-1]).any(-1).any(0).nonzero().flatten()
        worst = float((want.float() - got.float()).abs().max())
        out.append(
            f"{name}: {k} ({count} values, rows {int(rows.min())}..{int(rows.max())} of {want.shape[-2]}, "
            f"max |diff| {worst:.4g})"
        )
    return out


@pytest.mark.timeout(int(os.environ.get("V41_GENERIC_TIMEOUT", 4 * 3600)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_v41_prefill_generic(mesh_device, device_params, monkeypatch):
    layers = parse_layers(os.environ.get("V41_GENERIC_LAYERS", DEFAULT_LAYERS))
    exact_width = os.environ.get("V41_GENERIC_EXACT_WIDTH", "0") == "1"
    if exact_width:
        # the generic graph at the classic score width: proves the position inputs, masks and scatter writes
        # bit-exact against the classic path (the bucket rounding only changes the selection's tie-breaking)
        from models.demos.deepseek_v3_d_p.tt.v41.cache import V41ChunkTables

        monkeypatch.setattr(
            V41ChunkTables, "bucket_width", lambda self, ratio, start, length: -(-(start + length) // ratio // 32) * 32
        )
    total = int(os.environ.get("V41_GENERIC_SEQ", 16 * CHUNK - 13))
    max_seq = int(os.environ.get("V41_GENERIC_MAX_SEQ", -(-total // CHUNK) * CHUNK))
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(orc.HF_SNAPSHOT)
    spec = orc.real_spec(tuple(layers), max_seq, checkpoint=ckpt.root)  # identity for the weight cache only
    root = weight_cache_dir(spec, mesh_device.shape)
    prompts = _prompts(total, tokenizer)
    t0 = time.time()
    engram, engram_hash = build_engram_modules(mesh_device, C, ckpt, layers, tokenizer)
    top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
    model = TtV41Transformer(
        mesh_device,
        C,
        layers,
        lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer),
        top["embed.weight"],
        top["norm.weight"],
        top["head.weight"],
        max_seq_len=max_seq,
        chunk=CHUNK,
        engram=engram,
        engram_hash=engram_hash,
        weight_cache_path=root,
    )
    report = {
        "exact_width": exact_width,
        "layers": layers,
        "tokens": total,
        "max_seq": max_seq,
        "chunk": CHUNK,
        "chunks": -(-total // CHUNK),
        "mesh": list(mesh_device.shape),
        "build_s": round(time.time() - t0, 1),
        "dram_after_build_bytes": _dram_bytes(mesh_device),
    }
    logger.info(f"generic prefill: built {len(layers)} layers in {report['build_s']} s")

    def timed(fn):
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        return out, time.time() - t0

    # the classic reference: a fresh state per prompt (the compile run first)
    model.prefill(prompts[0], 1)
    classic, classic_s = [], []
    for p in prompts:
        (logits, state), s = timed(lambda: model.prefill(p, 1))
        classic.append(_snapshot(mesh_device, logits, state, total))
        classic_s.append(round(s, 3))
        del state
    logger.info(f"generic prefill: classic eager prefills {classic_s} s")
    mismatches = []

    # (a) the generic state, untraced
    gstate = model.new_state(generic=True)
    report["buckets"] = gstate.tables.buckets
    generic = []
    for p in prompts:
        (logits, _), eager_s = timed(lambda: (model.prefill_into(p, gstate), None))
        generic.append(_snapshot(mesh_device, logits, gstate, total))
    classic_vs_generic = [_differences(f"generic eager prompt {i}", classic[i], generic[i]) for i in range(2)]
    pcc = [
        float(torch.corrcoef(torch.stack([classic[i]["logits"].flatten(), generic[i]["logits"].flatten()]))[0, 1])
        for i in range(2)
    ]
    same_argmax = [bool(classic[i]["logits"].argmax() == generic[i]["logits"].argmax()) for i in range(2)]
    report |= {"classic_vs_generic": classic_vs_generic, "logits_pcc": pcc, "same_argmax": same_argmax}
    if exact_width:
        mismatches += classic_vs_generic[0] + classic_vs_generic[1]
    # the reference the replays must match bit-identically is the generic state's own untraced result (the same
    # graph); against the classic path the bucket widths only move the selection's tie-breaks (reported above)
    reference = classic if exact_width else generic
    logger.info(
        f"generic prefill: generic eager prefill {eager_s:.3f} s; classic vs generic {classic_vs_generic} pcc {pcc}"
    )
    report["dram_before_capture_bytes"] = _dram_bytes(mesh_device)

    # (b) the bucket captures and their replays
    t0 = time.time()
    trace = V41GenericPrefillTrace(model, gstate, {})
    capture_s = time.time() - t0
    report |= {
        "capture_s": round(capture_s, 1),
        "captured_at": trace.captured_at,
        "captures": len(trace.controllers),
        "trace_segments": trace.num_segments,
        "trace_bytes": _trace_bytes(mesh_device),
        "dram_after_capture_bytes": _dram_bytes(mesh_device),
    }
    logger.info(f"generic prefill: {report['captures']} captures in {capture_s:.1f} s: {report}")
    traced_s, phases = [], []
    for i in (0, 1, 0):
        (logits, _), s = timed(lambda: trace.run(prompts[i]))
        traced_s.append(round(s, 3))
        phases.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in trace.last.items()})
        mismatches += _differences(f"traced prompt {i}", reference[i], _snapshot(mesh_device, logits, gstate, total))
        logger.info(f"generic prefill: traced run of prompt {i} {s:.3f} s {trace.last}")
    trace.release()
    chunks = report["chunks"]
    report |= {
        "classic_eager_s": classic_s,
        "generic_eager_s": round(eager_s, 3),
        "traced_s": traced_s,
        "traced_phases": phases,
        "classic_eager_s_per_chunk": round(min(classic_s) / chunks, 3),
        "traced_s_per_chunk": round(min(traced_s) / chunks, 3),
        "bit_identical": not mismatches,
        "mismatches": mismatches,
        "compared": sorted(classic[0]),
    }
    logger.info(f"V41_GENERIC_RESULT {json.dumps(report)}")
    out = os.environ.get("V41_GENERIC_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
    assert not mismatches, "\n".join(mismatches)

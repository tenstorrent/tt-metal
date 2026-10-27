# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The packed prefill chunk (``TT_V41_PACK``): several whole prompts in one
generic chunk, each isolated from the others.

Builds ``V41_PACK_LAYERS`` (default the sharing schedule 0,1,2,3,20,21,24: two ratio-0 layers, an Engram
layer, a
ratio-2 KV + index source and its consumer, the candidate source and a ratio-1 consumer of each kind) with the
checkpoint's weights on a generic state of ``V41_PACK_MAX_SEQ`` tokens (10,240: the 8K bucket rounded to the
chunk)
and ``K`` distinct prompts of ``V41_PACK_PROMPT`` tokens (1,028) for each ``K`` of ``V41_PACK_KS`` (``2``;
``2,4``).

(d) flag off: a prompt prefilled on a state built without ``TT_V41_PACK`` equals the same prompt on a state
built
    with it, bit for bit (logits, every cache row it wrote, carries, selections), and ``prefill_packed``
    refuses to
    run without the flag;
(a) each segment's logits of the packed chunk equal the unpacked single-prompt prefill of the same prompt on
the same
    geometry (the 5,120-row chunk eagerly);
(b) each segment's cache rows (window scratch, compressed KV, index-K and its tiled twin, in the segment's own
    coordinates), window carries and every index source's top-k selection equal the unpacked run's, and no
    selected
    row lies outside the segment's ``[offset / r, (i + 1) // r)``; mutating the earlier neighbour leaves a
    segment's
    rows and logits unchanged;
(c) negative controls: the packed chunk with the single-prompt window table (``leak-window``) and with the
index
    mask's lower bound dropped (``leak-index``) must each change the later segment's outputs (the test is
    sensitive
    to a cross-segment attention), and must not change the first segment's.

``V41_PACK_OFFSETS=640,1024,2`` also packs prompt B behind a first segment of each of those lengths (a rank
boundary, a tile boundary, a two-row offset) and compares B with its single run and across the offsets: what
in a
segment's values depends on where it sits. ``V41_PACK_PROBE=1`` records each block's attention-norm,
attention,
FFN-norm and FFN outputs for the segment's first and last rows, so a difference is attributed to a stage.

Every comparison is bit-for-bit (``torch.equal``). The test asserts the isolation contract: the selection
ranges,
(d), segment 0 against its single run (the same rows), every segment under its earlier neighbour's mutation,
the
pack's determinism and the negative controls. A later segment against its single run (the cross-layout
comparison: the same prompt at other rows) is reported (``V41_PACK_RESULT`` JSON, ``V41_PACK_OUT``: counts,
max
|diff|, rows and PCC per key, the first differing block stage) and asserted only under ``V41_PACK_STRICT=1``:
Tests showed it differs by the TP all-reduce's row-dependent summation order and the top-k's tie
order, as the short chunk differs from the full one. ``V41_PACK_REPEAT`` (3) eager runs time the single-prompt
chunk and each pack with a device
sync; ``V41_PACK_TRACE=1`` also captures each (``SubDeviceTraceController``, as ``V41GenericPrefillTrace``)
and
replays it ``V41_PACK_REPLAYS`` (5) times for the device time.
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
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.cache import PACK, WINDOW_SLOT, V41PrefillState, pack_segments
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules
from models.demos.deepseek_v3_d_p.tt.v41.indexer import SENTINEL
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer, _Request
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

DEFAULT_LAYERS = "0,1,2,3,20,21,24"
TRACE_REGION = 256 * 1024 * 1024


def _down(mesh_device, t) -> torch.Tensor:
    """Every chip's copy of ``t`` as one host tensor [chips, ...] (row-major (sp, tp) chip order)."""
    if t.dtype == MlaKvCacheFormat.SCALED_FP8.storage_dtype:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return torch.cat([ttnn.to_torch(c) for c in ttnn.get_device_tensors(t)], dim=0)


def _rows(state: V41PrefillState, every: torch.Tensor, t, first: int, count: int) -> torch.Tensor:
    """Rows ``[first, first + count)`` of the state tensor ``t`` (``every``: its ``_down``) on the first and
    the last
    chip, SCALED_FP8 rows decoded to their fp32 values (a byte difference is judged on the values)."""
    every = every.reshape(every.shape[0], -1, every.shape[-1])  # [chips, rows, W]
    assert every.shape[1] == t.shape[-2], (tuple(every.shape), tuple(t.shape))
    picked = [every[0, first : first + count], every[-1, first : first + count]]
    if t.dtype == MlaKvCacheFormat.SCALED_FP8.storage_dtype:
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import reconstruct_scaled_fp8_kv_cache

        picked = [reconstruct_scaled_fp8_kv_cache(x.contiguous(), state.kv_geometry).float() for x in picked]
    return torch.stack(picked)


class _Segment:
    """One prompt's rows of a chunk: ``offset`` its first chunk row, ``n`` its length; the single-prompt
    reference
    is the segment at offset 0."""

    def __init__(self, offset: int, n: int):
        self.offset, self.n = offset, n


def _snapshot(mesh_device, state: V41PrefillState, segs: list, logits: list, carries: list, selections, blocks):
    """What each prompt of ``segs`` wrote into ``state``, in segment coordinates (row 0 the prompt's first
    row): per
    source its window scratch rows, compressed rows, index-K rows (and the tiled twin's); every layer's window
    carry
    (``carries[g]``: the layer -> tensor dict of segment g); the compressor carries' presence; the index
    sources'
    selections (in score order, and sorted: the selected set) and the block outputs of the segment's rows
    (``selections[g]``, ``blocks[g]``: already per segment).
    Each state tensor is read back once for every segment."""
    g_, cfg = state.geometry, state.config
    out = [{"logits": logits[g]} for g in range(len(segs))]
    for source, t in state.kv.items():
        r, every = cfg.compress_ratio(source), _down(mesh_device, t)
        for g, seg in enumerate(segs):
            out[g][f"kv[{source}] window"] = _rows(state, every, t, WINDOW_SLOT + seg.offset, seg.n)
            out[g][f"kv[{source}] compressed"] = _rows(
                state, every, t, g_.kv_row_of_compressed(seg.offset // r), seg.n // r
            )
    if state.swa_scratch is not None:
        t = state.swa_scratch
        every = _down(mesh_device, t)
        for g, seg in enumerate(segs):
            out[g]["swa_scratch window"] = _rows(state, every, t, WINDOW_SLOT + seg.offset, seg.n)
    for name, caches in (("index_k", state.index_k), ("index_k_tiled", state.index_k_tiled)):
        for source, t in caches.items():
            r, every = cfg.compress_ratio(source), _down(mesh_device, t)
            for g, seg in enumerate(segs):
                out[g][f"{name}[{source}]"] = _rows(state, every, t, seg.offset // r, seg.n // r)
    for g, seg in enumerate(segs):
        for layer, t in carries[g].items():
            out[g][f"window_carry[{layer}]"] = _rows(state, _down(mesh_device, t), t, 0, WINDOW_SLOT)
        for source, carry in state.compressor_carry.items():
            out[g][f"compressor_carry[{source}]"] = torch.tensor([0 if carry is None else 1])
        for layer, ids in selections[g].items():
            out[g][f"topk[{layer}]"] = ids
            out[g][f"topk_sorted[{layer}]"] = ids.sort(dim=-1).values  # the set: the sentinel sorts last
        for layer, x in blocks[g].items():
            out[g][f"x[{layer}]"] = x
    return out


def _u32(t: torch.Tensor) -> torch.Tensor:
    """A uint32 device tensor's host values as int64 (``to_torch`` hands uint32 back as int32: the sentinel
    reads
    -1 there)."""
    t = t.to(torch.int64)
    return torch.where(t < 0, t + (1 << 32), t)


def _chunk_rows(t: torch.Tensor, chunk: int, off: int, n: int) -> torch.Tensor:
    """Rows ``[off, off + n)`` of the chunk from a per-chip query-sharded tensor ``t`` [chips, 1, rows, W]
    (chip c
    holds chunk rows ``[c * rows, (c + 1) * rows)``) -> [n, W]."""
    flat = t.reshape(-1, t.shape[-1])
    assert flat.shape[0] == chunk, (tuple(t.shape), chunk)
    return flat[off : off + n]


def _sp_rows(t: torch.Tensor, chunk: int, sp: int, tp: int, off: int, n: int) -> torch.Tensor:
    """Rows ``[off, off + n)`` of the chunk from an SP-sharded, TP-split tensor ``t`` [chips, 1, chunk/sp,
    W/tp]
    (SP rank a holds chunk rows ``[a * chunk/sp, ...)``, its TP chips the width slices) -> [n, tp, W/tp]."""
    per = chunk // sp
    t = t.reshape(sp, tp, per, t.shape[-1]).permute(0, 2, 1, 3).reshape(chunk, tp, t.shape[-1])
    return t[off : off + n]


def _selection_check(ids: torch.Tensor, ratio: int, off: int, n: int) -> dict:
    """Every selected row of the segment's queries lies in ``[off / r, (i + 1) // r)`` or is the sentinel."""
    base = off // ratio
    i = torch.arange(off, off + n).view(-1, 1)
    valid = ids != SENTINEL
    low = valid & (ids < base)
    high = valid & (ids >= (i + 1) // ratio)
    return {
        "rows": n,
        "selected": int(valid.sum()),
        "below_segment": int(low.sum()),
        "beyond_visible": int(high.sum()),
    }


def _compare(name: str, want: dict, got: dict) -> tuple[list[str], dict]:
    """Bit-for-bit comparison of two snapshots: the differing keys described, and per key a record."""
    lines, record = [], {}
    for k, a in want.items():
        b = got.get(k)
        if b is None:
            lines.append(f"{name}: {k} missing")
            record[k] = "missing"
            continue
        if a.shape != b.shape:
            lines.append(f"{name}: {k} shape {tuple(a.shape)} vs {tuple(b.shape)}")
            record[k] = f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
            continue
        if torch.equal(a, b):
            record[k] = "equal"
            continue
        af, bf = a.float(), b.float()
        bad = a != b
        count = int(bad.sum())
        worst = float((af - bf).abs().max())
        entry = {
            "differ": count,
            "of": a.numel(),
            "max_abs_diff": round(worst, 6),
            "max_abs": round(float(af.abs().max()), 4),
        }
        if a.dim() >= 2:  # rows: the second-to-last axis, over every leading axis
            rows = bad.reshape(-1, a.shape[-2], a.shape[-1]).any(-1).any(0).nonzero().flatten()
            entry["rows"] = {
                "differ": int(rows.numel()),
                "of": a.shape[-2],
                "first": int(rows.min()),
                "last": int(rows.max()),
            }
        if a.dim() == 1 and a.numel() > 1024:  # the logits
            x, y = af.double(), bf.double()
            entry["pcc"] = round(float(torch.corrcoef(torch.stack([x, y]))[0, 1]), 6)
            entry["argmax_equal"] = int(x.argmax()) == int(y.argmax())
        record[k] = entry
        lines.append(f"{name}: {k} {entry}")
    return lines, record


def _capture(model: TtV41Transformer, forward):
    controller = SubDeviceTraceController(model.mesh_device)
    moes = [block.ffn for block in model.blocks]
    for moe in moes:
        moe.set_trace_controller(controller)
    ttnn.synchronize_device(model.mesh_device)
    controller.begin_capture()
    try:
        out = forward()
    finally:
        controller.end_capture()
        for moe in moes:
            moe.set_trace_controller(None)
    return controller, out


def _replays(model: TtV41Transformer, controller, n: int) -> list[float]:
    times = []
    for _ in range(n):
        ttnn.synchronize_device(model.mesh_device)
        t0 = time.perf_counter()
        controller.replay(blocking=False)
        ttnn.synchronize_device(model.mesh_device)
        times.append(round(time.perf_counter() - t0, 4))
    return times


@pytest.mark.timeout(int(os.environ.get("V41_PACK_TIMEOUT", 2 * 3600)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_v41_pack(mesh_device, device_params, expect_error):
    layers = parse_layers(os.environ.get("V41_PACK_LAYERS", DEFAULT_LAYERS))
    chunk = int(os.environ.get("TT_V41_CHUNK", 5120))
    max_seq = int(os.environ.get("V41_PACK_MAX_SEQ", 10240))
    prompt = int(os.environ.get("V41_PACK_PROMPT", 1028))
    ks = [int(k) for k in os.environ.get("V41_PACK_KS", "2").split(",")]
    strict = os.environ.get("V41_PACK_STRICT", "0") == "1"
    repeat = int(os.environ.get("V41_PACK_REPEAT", 3))
    trace = os.environ.get("V41_PACK_TRACE", "0") == "1"
    replays = int(os.environ.get("V41_PACK_REPLAYS", 5))
    assert os.environ.get("TT_V41_PREFILL_GENERIC", "0") == "1", "the packed chunk runs on generic states"
    os.environ[PACK] = "1"
    kmax = max(ks)
    assert kmax * prompt <= chunk, (kmax, prompt, chunk)
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    from transformers import AutoTokenizer

    sp, tp = tuple(mesh_device.shape)
    tokenizer = AutoTokenizer.from_pretrained(orc.HF_SNAPSHOT)
    spec = orc.real_spec(tuple(layers), max_seq, checkpoint=ckpt.root)
    root = weight_cache_dir(spec, mesh_device.shape)
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
        chunk=chunk,
        engram=engram,
        engram_hash=engram_hash,
        weight_cache_path=root,
    )
    result = {
        "layers": layers,
        "chunk": chunk,
        "max_seq": max_seq,
        "prompt": prompt,
        "ks": ks,
        "kv_format": model.kv_format.value if hasattr(model.kv_format, "value") else str(model.kv_format),
        "build_s": round(time.time() - t0, 1),
        "checks": {},
        "timing": {},
    }
    logger.info(f"pack: built {len(layers)} layers in {result['build_s']} s")
    # distinct texts: prompt g is the book's g-th window; the mutation prompt is the window after the last
    text = orc.text_tokens((kmax + 1) * prompt + 1)[0]
    prompts = [text[g * prompt : (g + 1) * prompt].clone() for g in range(kmax + 1)]
    index_sources = [l for l in layers if l in C.INDEX_SOURCE_LAYERS]
    buffers: dict = {}
    failures: list[str] = []  # the isolation contract (always asserted) and, under V41_PACK_STRICT=1, the rest
    reported: list[str] = []  # the cross-layout differences (a segment against its single run at row 0)

    block_rows = int(os.environ.get("V41_PACK_BLOCK_ROWS", 32 if len(layers) <= 8 else 0))
    result["block_rows"] = block_rows
    probe = int(os.environ.get("V41_PACK_PROBE", "0"))
    probed: dict = {}  # (layer, stage) -> the whole SP-sharded output of the stage, the latest chunk's

    def probing(layer: int, stage: str, f, named: bool = False):
        """``f`` wrapped to record its output (SP-sharded tensors only) under ``stage``, numbered per call
        within
        the chunk (``named``: by the call's second argument, ``_dense``'s weight name)."""

        def wrapped(*args, **kwargs):
            out = f(*args, **kwargs)
            if named:
                name = f"{stage}:{args[1]}"
            else:
                name = f"{stage}#{sum(1 for k in probed if k[0] == layer and k[1].startswith(stage + '#'))}"
            if hasattr(out, "shape") and len(out.shape) == 4 and out.shape[1] == 1 and out.shape[2] == chunk // sp:
                probed[(layer, name)] = _down(mesh_device, out)
            return out

        return wrapped

    if probe:  # record each block's sub-module outputs (the attention's input norm, the attention, the FFN's)
        for layer, block in zip(layers, model.blocks):
            for stage, module in (
                ("attn_norm", block.attn_norm),
                ("attn", block.attn),
                ("ffn_norm", block.ffn_norm),
                ("ffn", block.ffn),
            ):
                module.forward = probing(layer, stage, module.forward)
            if probe >= 2:  # and inside the attention: the projections, the TP all-reduces, the RoPE
                block.attn._dense = probing(layer, "dense", block.attn._dense, named=True)
                block.attn.ccl.tp_all_reduce = probing(layer, "allreduce", block.attn.ccl.tp_all_reduce)
                block.attn._rope = probing(layer, "rope", block.attn._rope)

    def recorder(segments: list, selections: list, blocks: list):
        """``on_block`` recording, per segment, every index source's selection (in the segment's coordinates)
        and
        the first and last ``block_rows`` rows of each block's output."""

        def on_block(layer, x, start, length):
            if layer in index_sources:
                ids = _u32(_down(mesh_device, state_of[0].selection["topk"]))
                for g, (off, n) in enumerate(segments):
                    rows = _chunk_rows(ids, chunk, off, n)
                    base = off // C.compress_ratio(layer)
                    selections[g][layer] = torch.where(rows == SENTINEL, rows, rows - base)
            if block_rows:
                for (l, stage), every in list(probed.items()):  # the stages first: the first divergence in order
                    if l != layer:
                        continue
                    for g, (off, n) in enumerate(segments):
                        rows = _sp_rows(every, chunk, sp, tp, off, n)
                        blocks[g][f"{layer}] {stage}"] = torch.cat([rows[:block_rows], rows[-block_rows:]])
                    del probed[(l, stage)]
                every = _down(mesh_device, x)
                for g, (off, n) in enumerate(segments):
                    rows = _sp_rows(every, chunk, sp, tp, off, n)
                    blocks[g][layer] = torch.cat([rows[:block_rows], rows[-block_rows:]])

        return on_block

    state_of = [None]

    def single(state: V41PrefillState, tokens: torch.Tensor) -> tuple[dict, dict]:
        """The unpacked prefill of one prompt at row 0 (eager, generic): its snapshot and the selection
        checks."""
        state_of[0] = state
        n = int(tokens.numel())
        selections, blocks = [{}], [{}]
        request = _Request(tokens, n - 1, None, None)
        model.reset_state(state)
        scored = model._run(
            request, state, buffers, lambda c, forward: forward(), recorder([(0, n)], selections, blocks)
        )
        logits = model._logits(scored)[-1].clone()
        checks = {l: _selection_check(ids, C.compress_ratio(l), 0, n) for l, ids in selections[0].items()}
        snap = _snapshot(mesh_device, state, [_Segment(0, n)], [logits], [state.window_carry], selections, blocks)
        return snap[0], checks

    def packed(state: V41PrefillState, parts: list[torch.Tensor], step=None, timings=None):
        """The packed prefill of ``parts``: per segment its snapshot (in segment coordinates) and selection
        checks."""
        state_of[0] = state
        segments = list(pack_segments([int(t.numel()) for t in parts]))
        selections, blocks = [{} for _ in parts], [{} for _ in parts]
        logits, layout = model.prefill_packed(
            parts, state, buffers, on_block=recorder(segments, selections, blocks), timings=timings, step=step
        )
        assert list(layout) == segments, (layout, segments)
        checks = []
        for g, (off, n) in enumerate(segments):
            chk = {}
            for l, ids in selections[g].items():
                base = off // C.compress_ratio(l)
                chk[l] = _selection_check(torch.where(ids == SENTINEL, ids, ids + base), C.compress_ratio(l), off, n)
            checks.append(chk)
        segs = [_Segment(off, n) for off, n in segments]
        carries = [state.segment_carry[g] for g in range(len(parts))]
        snaps = _snapshot(
            mesh_device, state, segs, [logits[g].clone() for g in range(len(parts))], carries, selections, blocks
        )
        return snaps, checks, tuple(segments)

    def judge(name: str, want: dict, got: dict, must_differ: list | None = None, exact: bool = True):
        """Compare two snapshots and record the result. ``exact``: a difference fails the test (the isolation
        contract: the flag off, segment 0, a segment under its neighbour's mutation); else (a segment against
        its
        single-prompt run at another row: the cross-layout comparison) it is reported, and fails only under
        ``V41_PACK_STRICT=1``. ``must_differ``: a negative control, which fails when none of the keys
        differ."""
        lines, record = _compare(name, want, got)
        result["checks"][name] = record
        for line in lines:
            logger.warning(line)
        first = [k for k in record if k.startswith("x[") and record[k] != "equal"]
        if first:
            result["checks"][name]["first_divergence"] = first[0]
        if must_differ is not None:
            differ = [k for k in must_differ if record.get(k) != "equal"]
            result["checks"][name]["must_differ"] = {k: record.get(k) for k in must_differ}
            if not differ:
                failures.append(f"{name}: expected a difference in {must_differ}, found none")
        elif lines and (exact or strict):
            failures.append(f"{name}: {len(lines)} keys differ: " + "; ".join(lines[:3]))
        elif lines:
            reported.append(f"{name}: {len(lines)} keys differ: " + "; ".join(lines[:3]))
        return lines

    # --- (d) the flag off: a state built without TT_V41_PACK, the same prompt bit for bit
    # --------------------------
    os.environ.pop(PACK, None)
    state_off = model.new_state(generic=True)
    assert not state_off.tables._pack and not state_off.tables._base, "a state built without the flag"
    snap_off, _ = single(state_off, prompts[0])
    with expect_error(AssertionError, f"{PACK}=1 is required"):
        model.prefill_packed(prompts[:2], state_off, buffers)
    os.environ[PACK] = "1"
    state = model.new_state(generic=True)
    assert state.tables._pack, "a state built with the flag"
    refs, ref_checks = {}, {}
    for g in range(kmax + 1):
        refs[g], ref_checks[g] = single(state, prompts[g])
        for l, chk in ref_checks[g].items():
            assert chk["below_segment"] == 0 and chk["beyond_visible"] == 0, (g, l, chk)
    result["checks"]["reference_selection"] = {str(l): ref_checks[0][l] for l in ref_checks[0]}
    judge("(d) flag off vs on, single prompt", snap_off, refs[0])
    del state_off

    # --- (a), (b): the packed chunk against the unpacked prefill of each prompt
    # ----------------------------------
    for k in ks:
        parts = prompts[:k]
        timings = {}
        snaps, checks, segments = packed(state, parts, timings=timings)
        result["timing"][f"pack{k}_first_eager"] = {kk: round(v, 4) for kk, v in timings.items()}
        result["checks"][f"pack{k} layout"] = list(segments)
        for g in range(k):
            result["checks"][f"pack{k} seg{g} selection"] = {str(l): c for l, c in checks[g].items()}
            for l, chk in checks[g].items():
                if chk["below_segment"] or chk["beyond_visible"]:
                    failures.append(f"pack{k} seg{g} layer {l}: cross-segment selection {chk}")
            judge(f"(a,b) pack{k} seg{g} vs single", refs[g], snaps[g], exact=g == 0)
        assert state.start == sum(n for _, n in segments)
        # the earlier neighbour mutated: the later segments must not move
        if k >= 2:
            mutated = [prompts[kmax]] + parts[1:]
            snaps_m, checks_m, _ = packed(state, mutated)
            for g in range(1, k):
                judge(f"(b) pack{k} seg{g} with seg0 mutated vs pack{k}", snaps[g], snaps_m[g])
            judge(f"(b) pack{k} seg0 mutated vs single", refs[kmax], snaps_m[0])  # row 0: exact
            # --- (c) negative controls: a cross-segment window, then a cross-segment index mask
            # ---------------
            tables = state.tables
            keys = [kk for kk in refs[1] if not kk.startswith("x[") and not kk.startswith("compressor")]
            tables._pack_windows[tuple(segments)] = tables._window_table(0)
            snaps_w, checks_w, _ = packed(state, parts)
            del tables._pack_windows[tuple(segments)]
            result["checks"][f"(c) pack{k} leak-window seg1 selection"] = {str(l): c for l, c in checks_w[1].items()}
            judge(f"(c) pack{k} leak-window seg1 vs single", refs[1], snaps_w[1], must_differ=keys)
            judge(f"(c) pack{k} leak-window seg0 vs single", refs[0], snaps_w[0])
            pairs = tables.prepare_pack_host(segments)
            bases = {id(b) for b in tables._base.values()}
            leaky = [
                (
                    (
                        ttnn.from_torch(
                            torch.zeros(sp, tp, chunk // (sp * tp), 1),
                            dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1)),
                        )
                        if id(b) in bases
                        else t
                    ),
                    b,
                )
                for t, b in pairs
            ]
            tables._pack_host[tuple(segments)] = leaky
            snaps_i, checks_i, _ = packed(state, parts)
            del tables._pack_host[tuple(segments)]
            result["checks"][f"(c) pack{k} leak-index seg1 selection"] = {str(l): c for l, c in checks_i[1].items()}
            crossed = sum(c["below_segment"] for c in checks_i[1].values())
            result["checks"][f"(c) pack{k} leak-index seg1 rows selected below the segment"] = crossed
            if not crossed and index_sources:
                failures.append(f"(c) pack{k} leak-index: no selection below the segment")
            judge(f"(c) pack{k} leak-index seg1 vs single", refs[1], snaps_i[1], must_differ=keys)
            judge(f"(c) pack{k} leak-index seg0 vs single", refs[0], snaps_i[0])
            # the isolation restored: the pack again equals the references
            snaps_r, _, _ = packed(state, parts)
            for g in range(k):
                judge(f"(a,b) pack{k} seg{g} vs single, after the controls", refs[g], snaps_r[g], exact=g == 0)
                judge(f"(b) pack{k} seg{g} after the controls vs before", snaps[g], snaps_r[g])  # deterministic

    # --- attribution: prompt B at other offsets (a shorter first segment), against its single run and each
    # other ---
    offsets = [int(v) for v in os.environ.get("V41_PACK_OFFSETS", "").split(",") if v]
    if offsets:
        at = {}
        for off in offsets:
            snaps_o, checks_o, layout = packed(state, [prompts[0][:off], prompts[1]])
            at[off] = snaps_o[1]
            result["checks"][f"offset {off} layout"] = list(layout)
            for l, chk in checks_o[1].items():
                if chk["below_segment"] or chk["beyond_visible"]:
                    failures.append(f"offset {off} seg1 layer {l}: cross-segment selection {chk}")
            judge(f"(b) B at offset {off} vs single", refs[1], snaps_o[1], exact=False)
        for i, a in enumerate(offsets):
            for b in offsets[i + 1 :]:
                judge(f"(b) B at offset {a} vs at offset {b}", at[a], at[b], exact=False)

    # --- timing: the single-prompt chunk and each pack, eager with a device sync; then captured and replayed
    # --------
    def eager_single(tokens):
        request = _Request(tokens, int(tokens.numel()) - 1, None, None)
        model.reset_state(state)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        model._run(request, state, buffers, lambda c, forward: forward())
        ttnn.synchronize_device(mesh_device)
        return time.perf_counter() - t0

    def eager_pack(parts):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        model.prefill_packed(parts, state, buffers)
        ttnn.synchronize_device(mesh_device)
        return time.perf_counter() - t0

    cases = {"single": lambda: eager_single(prompts[0])}
    for k in ks:
        cases[f"pack{k}"] = lambda k=k: eager_pack(prompts[:k])
    for name, run in cases.items():
        times = [round(run(), 4) for _ in range(repeat)]
        result["timing"][f"{name}_eager_s"] = times
        logger.info(f"pack timing: {name} eager {times}")
    if trace:
        controllers = []

        def captured_single(tokens):
            n = int(tokens.numel())
            request = _Request(tokens, n - 1, None, None)
            model.reset_state(state)
            host = next(model._host_chunks(request, 0, state.tables))
            token_ids, engram_inputs = model._stage(host, buffers)
            state.tables.prepare_chunk(0, n, host.tables)
            controller, _ = _capture(
                model, lambda: model._chunk_forward(state, host, token_ids, engram_inputs, None, None, n - 1, None)
            )
            return controller

        def captured_pack(parts):
            box = []

            def step(forward):
                controller, out = _capture(model, forward)
                box.append(controller)
                return out

            model.prefill_packed(parts, state, buffers, step=step)
            return box[0]

        captures = {"single": lambda: captured_single(prompts[0])}
        for k in ks:
            captures[f"pack{k}"] = lambda k=k: captured_pack(prompts[:k])
        for name, make in captures.items():
            t0 = time.perf_counter()
            controller = make()
            ttnn.synchronize_device(mesh_device)
            controllers.append(controller)
            capture_s = time.perf_counter() - t0
            times = _replays(model, controller, replays)
            result["timing"][f"{name}_replay_s"] = times
            result["timing"][f"{name}_capture"] = {"s": round(capture_s, 2), "segments": controller.num_segments}
            logger.info(f"pack timing: {name} replay {times} (capture {capture_s:.1f} s)")
        for controller in controllers:
            controller.release()

    result["failures"], result["reported"] = failures, reported
    for line in reported:
        logger.warning("cross-layout: " + line)
    for line in failures:
        logger.error(line)
    logger.info("V41_PACK_RESULT " + json.dumps(result))
    out = os.environ.get("V41_PACK_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=1)
    assert not failures, "\n".join(failures)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-pipeline multichip decoder stack for zai-org/GLM-5.3-Flash on homelab-1's 8 Blackhole chips (Stage 4).

Mesh plan (doc/multichip_decoder/README.md, $GLM53_WS/doc/ttnn/04-multichip-decoder/mesh_plan.json):

* Hardware: 8 Blackhole chips (4 p300a) cabled as a 2x4 Ethernet mesh, 2 links per edge, no wrap-around. The 45 decoder
  layers (~177 GiB of weights + state) do not fit fewer than 7 chips; the strategy is a byte-weighted LAYER PIPELINE
  over eight 1x1 submeshes of the 2x4 parent mesh (mesh/glm8_2x4_mesh_graph_descriptor.textproto). Plan of record =
  partition.json `pipeline8_ring_wrap` (coordinator decision 2026-10-18 ~05:40Z): 9 segments on the Hamiltonian cycle
  0-2-5-6-7-4-3-1-(0); chip 0 runs the head (embedding + L0 + L1.attn) and, after the closing on-board hop 1 -> 0, the
  tail (L41.ffn .. L44 + final norm / lm_head / sampler), so token feedback never leaves chip 0. Cut points may fall
  between the attention and FFN halves of a layer (balance within 10 % at B = 1 / 8 / 32); the tensor handed over is
  the post-attention streams, same contract as at a layer boundary. The whole-layer 1x8 line 0-2-5-6-7-4-3-1 is the
  fallback plan (`pipeline8_ring`). Weights, KDA recurrent state, MLA latent pages and page tables of a layer live only
  on the chip(s) running it, unsharded; every tensor keeps the single-chip optimized decoder's layout, dtype and program
  configs (no per-device shape change, no padding). TP/EP rejected for this week (mesh_plan.json).
* Per-layer decoders are the single-chip OPTIMIZED decoder (`optimized_decoder.OptimizedDecoder`, Stage 3; the Stage 2
  `FusedDecoder` works through the same contract with eager prefill): residual streams `[1, 1, rows, 16384]` fp32 (4
  mHC streams), decode rows = B user slots (`cur_pos[u] == -1` = inactive), prefill rows = one physical chunk of one
  user. Whole-layer units run the decoder's own `decode_forward` / Stage 3 traced prefill body; half-layer units run
  `forward_unit` (the same sublayer objects and op sequence, split at the post-attention streams).
* Stacked-decoder I/O contract: the stack's input and output are the same `[1, 1, rows, 16384]` fp32 TILE DRAM tensor
  (input on the first stage's chip, output on the last stage's chip = chip 0 in the wrap plan); the host touches them
  only at the stack boundary (Stage 6: embedding / final norm + lm_head + sampler on chip 0).
* Hand-off: `ttnn.experimental.send_direct_async` / `recv_direct_async` over a MeshSocket pair per hop, one connection
  per forwarding link of the edge (`ttnn.get_forwarding_link_indices`: 2 here), fabric payload 15232 B (Stage 04a:
  43-50 us per 2 MiB decode hop, bit-exact). Every hop is checked at construction: distinct chips, row / column
  neighbours of the opened mesh, a forwarding route. Decode: inside each chip's trace (recv at the head, send at the
  tail); a step = `execute_trace` on every stage in pipeline order, non-blocking; the device-side socket handshake
  orders the chips. Prefill: eager socket transfer of each chunk between stage programs. `host` = D2H/H2D fallback.
* Decode: one trace per (stage, micro-batch group): [recv] -> units -> [send | copy into the persistent out_buf] ->
  optional `ttnn.plus_one(cur_pos, skip_negative_entries=True)` on the stage's own position copy. `cur_pos` and the page
  table are persistent per-stage copies (one page table per stage shared by its MLA layers), written by the host only
  at request boundaries.
* Prefill (optimized decoder): traced per bucket as Stage 3 does (whole layers: Stage 3's traced body; half layers: the
  same body split at the post-attention streams; KDA through the scratch recurrent state, MLA through the chunk / user
  tables), on per-chip SHARED ping-pong buffers (unit i of a stage reads bufs[p][i % 2], writes bufs[p][(i + 1) % 2]):
  a stage's units chain on device; the persistent prefill I/O is 2 x sum(buckets) x 64 KiB per chip (508 MiB). Model
  chunk plan (all layers): `plan_traced_chunks(len, 64, buckets)` = full power-of-two chunks + one tail < 64 tokens
  padded to 64 (MLA) / round_up(n, 32) (KDA); padded rows are zeroed at every unit boundary of the tail chunk (what the
  single-chip host loop feeds each layer). Any logical length >= 1 is accepted.

Setup order (compile-all-then-capture, $tt-enable-tracing): construction (weights; split units loaded first and their
unused half released before the chip's other layers load, so the load peak stays below the final footprint) ->
`allocate` (caches, page tables, positions, hand-off and prefill buffers) -> `prepare_prefill` (eager pass of every
bucket through the whole pipeline: compiles every prefill, hand-off, mask and re-bucketing program) ->
`warm_and_capture` (eager decode step per group, state restore, decode trace per (stage, group), then the prefill
traces). After that, prefill and decode allocate only transient tensors; an uncovered prefill shape raises.
Teardown: `release` (traces) -> `close_links` (socket pairs) -> close the mesh (`close_pipeline_mesh` does all three).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import torch

import ttnn

HC_WIDTH = 4 * 4096
DRAM = ttnn.DRAM_MEMORY_CONFIG
PARTS = ("full", "attn", "ffn")
MODEL_ALIGN = 64  # lcm of the per-kind physical alignments {kda 32, mla 64}: the model-level chunk granularity
# The 1x8 line embedded in the 2x4 mesh (row 0 = 0,2,5,6; row 1 = 1,3,4,7; on-board columns 0-1, 2-3, 5-4, 6-7):
LINE_ORDER = (0, 2, 5, 6, 7, 4, 3, 1)


# ====================================================================================================== plan objects
@dataclass(frozen=True)
class Unit:
    layer: int
    part: str = "full"  # "full" | "attn" | "ffn"

    def __post_init__(self):
        if self.part not in PARTS:
            raise ValueError(f"unit part {self.part!r} not in {PARTS}")

    def __str__(self):
        return f"L{self.layer}" + ("" if self.part == "full" else f".{self.part}")


@dataclass
class StagePlan:
    """One pipeline stage = one chip. `coord` is the stage's coordinate in the parent mesh."""

    coord: tuple[int, int]
    units: list[Unit]
    chip: int | None = None  # physical chip index (informational; TT_VISIBLE_DEVICES numbering)
    owns_embedding: bool = False
    owns_head: bool = False

    @property
    def layers(self) -> list[int]:
        out = []
        for u in self.units:
            if u.layer not in out:
                out.append(u.layer)
        return out

    @property
    def whole_layers(self) -> bool:
        return all(u.part == "full" for u in self.units)


def parse_units(spec) -> list[Unit]:
    """[0, 1, "2.attn"] / ["L2.ffn", "L3"] / ["L6.attn", ..] -> merged units (an attn immediately followed by the ffn
    of the same layer becomes one full unit)."""
    raw = []
    for s in spec:
        if isinstance(s, Unit):
            raw.append(s)
            continue
        m = re.fullmatch(r"L?(\d+)(?:\.(attn|ffn))?", str(s))
        if not m:
            continue  # embed / norm+lm_head entries of partition.json are not decoder units
        raw.append(Unit(int(m.group(1)), m.group(2) or "full"))
    out = []
    for u in raw:
        if out and u.part == "ffn" and out[-1].part == "attn" and out[-1].layer == u.layer:
            out[-1] = Unit(u.layer, "full")
        else:
            out.append(u)
    return out


def unit_halves(p: StagePlan) -> dict:
    """layer -> None (whole layer on this stage) or "attn" / "ffn" (the half this stage runs)."""
    out = {}
    for l in p.layers:
        parts = {u.part for u in p.units if u.layer == l}
        out[l] = None if "full" in parts or parts == {"attn", "ffn"} else next(iter(parts))
    return out


def load_jobs(stages: list[StagePlan], halves: list[dict]) -> dict:
    """Per chip (coord) the ordered loads [(stage index, layer, half)] of every stage on that chip, stages in pipeline
    order and, within a stage, split units first (their unused half is released before the stage's whole layers load:
    C-7) -- exactly the serial path's per-chip order, so a parallel load makes the same per-chip allocation sequence
    (same addresses: Stage 12 DRAM images are valid for both paths)."""
    jobs = {}
    for si, p in enumerate(stages):
        own = sorted(((si, l, halves[si][l]) for l in p.layers), key=lambda j: j[2] is None)  # stable: layer order
        jobs.setdefault(tuple(p.coord), []).extend(own)
    return jobs


def validate_plan(stages: list[StagePlan]):
    """Units must form one contiguous sequence of layer halves across the stages."""
    seq = []
    for s in stages:
        for u in s.units:
            seq += [(u.layer, "attn"), (u.layer, "ffn")] if u.part == "full" else [(u.layer, u.part)]
    first = seq[0][0]
    expect = [(l, p) for l in range(first, first + len(seq) // 2) for p in ("attn", "ffn")]
    if seq != expect:
        raise ValueError(f"pipeline units are not a contiguous run of layers: {seq}")


def plan_from_partition(path: str, coords: dict[int, tuple[int, int]], key: str | None = None) -> list[StagePlan]:
    """partition.json -> one StagePlan per pipeline SEGMENT (the file's recommended plan by default; Stage 4 uses the
    whole-layer line `pipeline8_ring`). `coords` maps chip -> parent-mesh coordinate (`chip_coords` of the opened
    mesh: the runtime picks the embedding)."""
    d = json.load(open(path))
    key = key or d.get("recommended", "pipeline8_ring")
    stages = []
    for seg in d[key]["stages"]:
        stages.append(
            StagePlan(
                coord=coords[seg["chip"]],
                units=parse_units(seg["units"]),
                chip=seg["chip"],
                owns_embedding="embed" in seg["units"],
                owns_head="norm+lm_head" in seg["units"],
            )
        )
    validate_plan(stages)
    return stages


def even_plan(layers: list[int], chips: list[int], coords: dict[int, tuple[int, int]]) -> list[StagePlan]:
    """Contiguous whole-layer split of `layers` over `chips` in that order (sub-mesh development plans)."""
    n, k = len(layers), len(chips)
    out, i = [], 0
    for j, c in enumerate(chips):
        m = n // k + (1 if j < n % k else 0)
        out.append(StagePlan(coord=coords[c], units=[Unit(l) for l in layers[i : i + m]], chip=c))
        i += m
    out = [s for s in out if s.units]
    validate_plan(out)
    return out


def chip_coords(mesh, visible: list[int] | None = None) -> dict[int, tuple[int, int]]:
    """Physical chip -> parent-mesh coordinate. `visible` = TT_VISIBLE_DEVICES order (process device id i is chip
    visible[i]); default: the env variable."""
    if visible is None:
        env = os.environ.get("TT_VISIBLE_DEVICES", "")
        visible = [int(x) for x in env.split(",") if x.strip()] or list(range(mesh.get_num_devices()))
    ids = list(mesh.get_device_ids())
    rows, cols = mesh.shape[0], mesh.shape[1]
    out = {}
    for i, dev in enumerate(ids):
        out[visible[dev]] = (i // cols, i % cols)
    assert len(out) == rows * cols
    return out


# ================================================================================================ unit forward paths
def _fused_layer_parts(layer):
    """The half-layer path reuses the fused decoder layer's own sublayer objects."""
    need = ("attn_hc", "ffn_hc", "attn_hc_sm", "ffn_hc_sm", "attn_hc_comp", "ffn_hc_comp", "_norm", "_ffn", "fcfg")
    missing = [n for n in need if not hasattr(layer, n)]
    if missing:
        raise NotImplementedError(f"half-layer units need FusedDecoderLayer sublayers; missing {missing}")


def forward_unit(dec, x, attention, *, decode: bool, part: str):
    """One unit of `dec` on streams x [1,1,T,16384] fp32 -> [1,1,T,16384] fp32 DRAM. part="full" is exactly
    `dec.layer.forward`; "attn" returns the post-attention streams s1 (flat layout), "ffn" continues from them. attn
    followed by ffn executes the layer's op sequence on the same values plus one data-movement pair at the split (fused
    mHC: to_flat / to_stacked; stream-major: from / to stream-major); the optimized decoder's decode path with
    `mhc_l1` keeps its L1 copies on both sides of the split (OptimizedDecoderLayer.forward)."""
    layer = dec.layer
    if part == "full":
        return layer.forward(x, attention, decode=decode)
    _fused_layer_parts(layer)
    from models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder import (
        from_stream_major,
        to_flat,
        to_stacked,
        to_stream_major,
    )

    ocfg = getattr(layer, "ocfg", None)
    if decode and ocfg is not None and getattr(ocfg, "mhc_l1", False):
        out = _forward_unit_l1(layer, x, attention, part, to_flat, to_stacked, from_stream_major)
        if out is not None:
            return out
    fcfg = layer.fcfg
    mode = fcfg.mhc_decode if decode else fcfg.mhc_prefill
    if mode == "auto":
        mode = "fused" if x.shape[-2] <= fcfg.mhc_auto_fused_max_rows else "stream_major"
    n, d = layer.cfg.hc_mult, layer.cfg.hidden_size
    t = x.shape[-2]
    if (
        mode == "composite"
        and hasattr(layer.attn_hc_comp, "pre_streams")
        and getattr(ocfg, "mhc_prefill_reuse_streams", False)
    ):
        # OptimizedDecoderLayer._prefill_shared_streams split at the attention / FFN boundary: the 4 stream slices of
        # the half's input are shared by its pre and post; the cut carries concat(s1), whose slices are s1 exactly
        hc = layer.attn_hc_comp if part == "attn" else layer.ffn_hc_comp
        h, post, comb, xs = hc.pre_streams(x)
        if part == "attn":
            y = attention(layer._norm(h, layer.input_norm))
        else:
            y = layer._ffn(h, decode)
        ttnn.deallocate(h)
        s = hc.post_streams(y, xs, post, comb)
        for u in xs:
            ttnn.deallocate(u)
        out = ttnn.concat(s, dim=-1)
        for u in s:
            ttnn.deallocate(u)
        return out
    if mode == "composite":
        if part == "attn":
            h, post, comb = layer.attn_hc_comp.pre(x)
            a = attention(layer._norm(h, layer.input_norm))
            return layer.attn_hc_comp.post(a, x, post, comb)
        h, post, comb = layer.ffn_hc_comp.pre(x)
        m = layer._ffn(h, decode)
        return layer.ffn_hc_comp.post(m, x, post, comb)
    if mode == "stream_major":
        x_sm = to_stream_major(x, n)
        if part == "attn":
            h, post, comb = layer.attn_hc_sm.pre(x_sm)
            a = attention(layer._norm(h, layer.input_norm))
            s1 = layer.attn_hc_sm.post(a, x_sm, post, comb)
        else:
            h, post, comb = layer.ffn_hc_sm.pre(x_sm)
            m = layer._ffn(h, decode)
            s1 = layer.ffn_hc_sm.post(m, x_sm, post, comb)
        ttnn.deallocate(x_sm)
        out = from_stream_major(s1)
        ttnn.deallocate(s1)
        return out
    assert mode == "fused", mode
    if part == "attn":
        x_stk = to_stacked(x, n)
        h, post, comb = layer.attn_hc.pre(x, x_stk)
        a = attention(layer._norm(ttnn.reshape(h, (1, 1, t, d)), layer.input_norm))
        s1 = layer.attn_hc.post(a, x_stk, post, comb)
        ttnn.deallocate(x_stk)
        return to_flat(s1)
    s1 = to_stacked(x, n)  # x = to_flat(s1) of the attn half
    s1_flat = None if layer.ffn_hc.proj_mode(t) == "stacked" else x
    h, post, comb = layer.ffn_hc.pre(s1_flat, s1)
    m = layer._ffn(ttnn.reshape(h, (1, 1, t, d)), decode)
    s2 = layer.ffn_hc.post(m, s1, post, comb)
    ttnn.deallocate(s1)
    return to_flat(s2)


def _forward_unit_l1(layer, streams, attention, part, to_flat, to_stacked, from_stream_major):
    """Half of OptimizedDecoderLayer.forward's decode path with mhc_l1 (mHC on an L1 copy of the streams, output to
    DRAM). Returns None for the composite mHC mode (the generic path handles it)."""
    L1, DRAM_ = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    fcfg, ocfg = layer.fcfg, layer.ocfg
    t = streams.shape[-2]
    mode = fcfg.mhc_decode
    if mode == "auto":
        mode = "fused" if t <= fcfg.mhc_auto_fused_max_rows else "stream_major"
    if mode == "composite":
        return None
    n, d = layer.cfg.hc_mult, layer.cfg.hidden_size
    x = ttnn.to_memory_config(streams, L1)
    if mode == "stream_major":
        sm_mem = dict(memory_config=L1) if getattr(ocfg, "mhc_sm_l1", False) else {}
        x_sm = ttnn.concat([ttnn.slice(x, (0, 0, 0, i * d), (1, 1, t, (i + 1) * d)) for i in range(n)], dim=1, **sm_mem)
        ttnn.deallocate(x)
        if part == "attn":
            h, post, comb = layer.attn_hc_sm.pre(x_sm)
            a = attention(layer._norm(h, layer.input_norm))
            s1 = layer.attn_hc_sm.post(a, x_sm, post, comb)
            ttnn.deallocate(x_sm)
            return ttnn.to_memory_config(from_stream_major(s1), DRAM_)
        h, post, comb = layer.ffn_hc_sm.pre(x_sm)
        m = layer._ffn(h, True)
        s2 = layer.ffn_hc_sm.post(m, x_sm, post, comb)
        ttnn.deallocate(x_sm)
        return ttnn.to_memory_config(from_stream_major(s2), DRAM_)
    if part == "attn":
        x_stk = to_stacked(x, n)
        h, post, comb = layer.attn_hc.pre(x, x_stk)
        ttnn.deallocate(x)
        a = attention(layer._norm(ttnn.reshape(h, (1, 1, t, d)), layer.input_norm))
        s1 = layer.attn_hc.post(a, x_stk, post, comb)
        ttnn.deallocate(x_stk)
        return ttnn.to_memory_config(to_flat(s1), DRAM_)
    s1 = to_stacked(x, n)  # x = L1 copy of to_flat(s1) of the attn half
    s1_flat = None if layer.ffn_hc.proj_mode(t) == "stacked" else x
    h, post, comb = layer.ffn_hc.pre(s1_flat, s1)
    ttnn.deallocate(x)
    m = layer._ffn(ttnn.reshape(h, (1, 1, t, d)), True)
    s2 = layer.ffn_hc.post(m, s1, post, comb)
    ttnn.deallocate(s1)
    return ttnn.to_memory_config(to_flat(s2), DRAM_)


def _device_tensors(obj, seen=None, depth=0):
    """Every ttnn.Tensor reachable from a decoder sub-module (attributes, lists, tuples, dicts)."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 6:
        return
    seen.add(id(obj))
    if isinstance(obj, ttnn.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _device_tensors(v, seen, depth + 1)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _device_tensors(v, seen, depth + 1)
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("models.autoports"):
        for v in vars(obj).values():
            yield from _device_tensors(v, seen, depth + 1)


def drop_unused_half(dec, keep: str) -> int:
    """Half-layer partitions only: release the other half's sublayer weights. Returns the bytes released."""
    sub = dec.layer.mlp if keep == "attn" else dec.layer.attn
    before = dram_allocated(dec.device)
    for t in _device_tensors(sub):
        if t.storage_type() == ttnn.StorageType.DEVICE and t.is_allocated():
            t.deallocate(True)
    setattr(dec.layer, "mlp" if keep == "attn" else "attn", None)  # any accidental use fails loudly
    return before - dram_allocated(dec.device)


def dram_allocated(mesh) -> int:
    """Bytes allocated in DRAM on one chip (allocator view; all banks)."""
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return int(v.total_bytes_allocated_per_bank) * int(v.num_banks)


def dram_total(mesh) -> int:
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return int(v.total_bytes_per_bank) * int(v.num_banks)


def decode_attention(dec, cache, page_table, cur_pos):
    """The attention callable `FusedDecoder.decode_forward` binds (half-layer units)."""
    if dec.is_kda:
        masks = dec._active_masks(cur_pos) if hasattr(dec, "_active_masks") else dec._active_mask(cur_pos)
        return lambda h: dec.layer.attn.decode(h, cache.kda, masks)
    return lambda h: dec.layer.attn.decode(h, cache.latent, page_table, cur_pos)


def prefill_attention(dec, cache, user_slot):
    """The attention callable `FunctionalDecoder.prefill_chunk` binds (half-layer units)."""
    if dec.is_kda:
        return lambda h: dec.layer.attn.prefill(h, cache.kda, user_slot, dec._valid_end, dec._valid_end_f)
    return lambda h: dec.layer.attn.prefill(h, cache.latent, cache.chunk_table, cache.user_table, dec._chunk_start)


def rezero_rows(x, n_valid: int, phys: int):
    """[1,1,P0,W] -> [1,1,phys,W] with rows >= n_valid exactly 0.0 (eager prefill path)."""
    if x.shape[-2] == phys == n_valid:
        return x
    w = x.shape[-1]
    xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    xr = ttnn.slice(xr, (0, 0, 0, 0), (1, 1, n_valid, w))
    if phys > n_valid:
        z = ttnn.zeros(
            (1, 1, phys - n_valid, w),
            dtype=xr.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=x.device(),
            memory_config=DRAM,
        )
        xr = ttnn.concat([xr, z], dim=2)
    return ttnn.to_layout(xr, ttnn.TILE_LAYOUT)


def model_chunk_plan(seq_len: int, buckets: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Model-level traced chunk plan shared by every layer: [(offset, valid tokens, physical rows)] = full power-of-two
    buckets (>= 64) then one tail of < 64 valid tokens padded to 64 (Stage 3 `plan_traced_chunks` at the model
    alignment). Each layer kind then runs the chunk at its own bucket (`layer_bucket`)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import plan_traced_chunks

    return plan_traced_chunks(seq_len, MODEL_ALIGN, tuple(b for b in buckets if b % MODEL_ALIGN == 0))


# per-chunk fixed cost of a pipeline prefill chunk in "rows" of the linear cost: the 45-layer stack measured 4413 ms
# for one 2048-row chunk and 677 ms for one 128-row chunk (E4 perf.jsonl, run 20261018T214623Z) -> 1.946 ms per row +
# 428 ms per chunk (every chunk walks all units and hops), i.e. one chunk costs as much as ~220 extra padded rows
CHUNK_OVERHEAD_ROWS = 220
# KDA recurrence chunk (rows): a prefill chunk must never hold a FULLY padded one (Stage 1 padding contract). Measured on
# the 8-chip stack (E4 acceptance_v2, post_prefill_state_* rows): a 65-token prompt in a 128-row chunk and a 130-token
# prompt in a 256-row chunk left a wrong KDA state (PCC ~0 vs an independently loaded instance) while every valid output
# row stayed bit-exact; padding < 32 rows (100 -> 128, 2047 -> 2048) is bit-exact including the state
KDA_RECURRENCE_ROWS = 32


def prefill_chunk_plan(seq_len: int, buckets: tuple[int, ...], policy: str = "plan") -> list[tuple[int, int, int]]:
    """Chunk plan of a prompt [(offset, valid tokens, physical rows)].
    "plan" (DEFAULT, the device-accepted path): `model_chunk_plan` = full power-of-two buckets + one tail < 64 tokens
    (KDA layers run the tail at round_up(n, 32) rows, see layer_bucket), so no chunk holds a fully padded 32-row KDA
    recurrence chunk (Stage 1 padding contract: prepare_chunk_recurrence leaves such chunks' terms unspecified and the
    scan still adds them to the state — a wrong KDA state handed to decode, measured on the 8-chip stack, E4 §3c).
    "padded" / "auto" (opt-in, NOT device-validated with the guard below): full largest-bucket chunks, then the
    remainder as ONE chunk padded to the smallest bucket >= it, only when that pads fewer than KDA_RECURRENCE_ROWS rows
    ("auto": and only when cheaper under CHUNK_OVERHEAD_ROWS); otherwise the plan. Stage 3's F1 ("one padded 2048 chunk
    for 1999 tokens") pads 49 rows and is never produced. Padded rows are zero and masked at every unit boundary; KDA
    treats them as identity steps only while no 32-row recurrence chunk is entirely padded; MLA pages past the prompt
    are overwritten by decode before they are attended (causal)."""
    plan = model_chunk_plan(seq_len, buckets)
    if policy == "plan" or len(plan) <= 1:
        return plan
    if policy not in ("padded", "auto"):
        raise ValueError(f"chunk policy {policy!r}")
    mb = tuple(sorted(b for b in buckets if b % MODEL_ALIGN == 0))
    top = mb[-1]
    out, off = [], 0
    while seq_len - off > top:
        out.append((off, top, top))
        off += top
    r = seq_len - off
    p = min(b for b in mb if b >= r)
    if p > MODEL_ALIGN and p - r >= KDA_RECURRENCE_ROWS:
        return plan  # would hold a fully padded KDA recurrence chunk (the <= 64-row tail is safe: layer_bucket)
    out.append((off, r, p))
    if policy == "padded":
        return out
    cost = lambda pl: sum(phys + CHUNK_OVERHEAD_ROWS for _, _, phys in pl)
    return out if cost(out) < cost(plan) else plan


def layer_bucket(dec, phys: int, n_valid: int) -> int:
    """Physical rows a layer runs a model chunk at: the chunk's rows, except the tail of a KDA layer (32-row recurrence
    chunks: round_up(n, 32), so it never holds a fully padded 32-row chunk)."""
    if phys == MODEL_ALIGN and n_valid < MODEL_ALIGN:
        return -(-n_valid // dec.prefill_align) * dec.prefill_align
    return phys


# ========================================================================================================= hand-off
class EthLink:
    """One pipeline hop over Ethernet fabric: a MeshSocket pair from stage k's submesh to stage k+1's submesh."""

    def __init__(
        self, src, dst, *, mode: str = "direct", connections: int = 2, fifo_bytes: int = 65536, core_row: int = 0
    ):
        if mode not in ("direct", "fifo"):
            raise ValueError(mode)
        self.mode = mode
        c0 = ttnn.MeshCoordinate(0, 0)
        conns = [
            ttnn.SocketConnection(
                ttnn.MeshCoreCoord(c0, ttnn.CoreCoord(i, core_row)),
                ttnn.MeshCoreCoord(c0, ttnn.CoreCoord(i, core_row + 1)),
            )
            for i in range(connections)
        ]
        size = 128 if mode == "direct" else fifo_bytes  # direct: the FIFO only carries the address handshake
        cfg = ttnn.SocketConfig(conns, ttnn.SocketMemoryConfig(ttnn.BufferType.L1, size))
        self.send_socket, self.recv_socket = ttnn.create_socket_pair(src, dst, cfg)

    def send(self, t):
        (ttnn.experimental.send_direct_async if self.mode == "direct" else ttnn.experimental.send_async)(
            t, self.send_socket
        )

    def recv(self, buf):
        (ttnn.experimental.recv_direct_async if self.mode == "direct" else ttnn.experimental.recv_async)(
            buf, self.recv_socket
        )


def host_handoff(src_tensor, dst_tensor, host_buf=None, *, cq_id=None):
    """Device -> host -> device (CQ read into a reusable host tensor, CQ write into the persistent device buffer).
    Not traceable: runs between stage replays (fallback, plan C)."""
    if host_buf is None:
        host_buf = ttnn.allocate_tensor_on_host(src_tensor.spec, src_tensor.device())
    ttnn.copy_device_to_host_tensor(src_tensor, host_buf, blocking=True, cq_id=cq_id)
    ttnn.copy_host_to_device_tensor(host_buf, dst_tensor, cq_id=cq_id)
    return host_buf


def _zeros(mesh, rows, dtype=ttnn.float32, width=HC_WIDTH, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(torch.zeros(1, 1, rows, width), dtype=dtype, layout=layout, device=mesh, memory_config=DRAM)


# ===================================================================================================== stage runtime
@dataclass
class _Group:
    """Per micro-batch group state of one stage: caches, per-chip step inputs, hand-off buffers, traces."""

    caches: dict = field(default_factory=dict)  # layer -> DecoderCache
    page_table: object = None  # [slots, blocks_per_seq] int32 device tensor shared by the chip's MLA layers
    cur_pos: object = None  # [B] int32 device tensor (persistent)
    streams_in: object = None  # [1,1,B,16384] fp32 persistent input (first stage: host-written; else hand-off target)
    out_buf: object = None  # persistent copy of the stage output (last stage: the step output; host hand-off: D2H src)
    probes: list = field(default_factory=list)  # optional persistent per-unit output copies (inspection / tests)
    trace_id: object = None
    host_buf: object = None
    # traced prefill: layer -> {bucket: PrefillTrace} on the stage's shared ping-pong buffers
    prefill_traces: dict = field(default_factory=dict)


class PipelineStage:
    def __init__(self, plan: StagePlan, submesh, decoders: dict):
        self.plan = plan
        self.mesh = submesh
        self.decoders = decoders  # layer -> per-layer decoder on this submesh
        parts = {}
        for u in plan.units:
            parts.setdefault(u.layer, set()).add(u.part)
        # layers whose attention runs on this chip own the per-layer state (KDA state / MLA pages)
        self.state_layers = [l for l in plan.layers if parts[l] != {"ffn"}]
        self.mla_layers = [l for l in self.state_layers if not decoders[l].is_kda]
        self.groups: list[_Group] = []
        self.link_in: EthLink | None = None
        self.link_out: EthLink | None = None
        self._prefill_buf = {}  # eager path: per-shape receive buffers
        self.pbufs: dict = {}  # traced prefill: bucket -> [A, B] ping-pong buffers (shared by every layer and group)
        self.pmask: dict = {}  # traced prefill: tail bucket -> [1,1,p,1] fp32 valid-row mask
        self.plus_one = False

    # ------------------------------------------------------------------------------------------------- decode
    def decode_units(self, g: _Group, x):
        """Device-only decode of this stage's units for group g, starting from x (a persistent buffer). Whole layers run
        the decoder's own public `decode_forward` (the single-chip program sequence). Every intermediate output is
        freed once consumed, so no trace-owned tensor stays live after a capture."""
        for i, u in enumerate(self.plan.units):
            dec = self.decoders[u.layer]
            if u.part == "full":
                y = dec.decode_forward(x, cache=g.caches[u.layer], page_table=g.page_table, cur_pos=g.cur_pos)
            else:
                attn = None if u.part == "ffn" else decode_attention(dec, g.caches[u.layer], g.page_table, g.cur_pos)
                y = forward_unit(dec, x, attn, decode=True, part=u.part)
            if g.probes:
                ttnn.copy(y, g.probes[i])
            if x is not g.streams_in:
                ttnn.deallocate(x)
            x = y
        return x

    def snapshot(self, g: _Group):
        return {l: self.decoders[l].snapshot_state(g.caches[l]) for l in self.state_layers}

    def restore(self, g: _Group, snap):
        for l, s in snap.items():
            self.decoders[l].restore_state(g.caches[l], s)


class MultichipDecoder:
    """Pipeline of per-chip decoder stages (see module docstring)."""

    def __init__(
        self,
        parent_mesh,
        stages: list[StagePlan],
        *,
        loader,
        handoff: str = "eth_direct",
        eth_connections: int = 2,
        eth_fifo_bytes: int = 65536,
        submeshes: dict | None = None,
        link_core_row: int = 0,
        drop_unused: bool = True,
        traced_prefill: bool | None = None,
        parallel_load: bool = False,
    ):
        """`loader(layer, submesh, half)` builds the per-layer single-chip decoder (OptimizedDecoder.from_weight_cache);
        parallel_load: one host thread per chip (1x1 submesh) loads that chip's layers (split units of all its stages
        first, Stage 04a review C-7), the chips concurrently (each submesh has its own command queue, allocator and API
        lock; the GIL is released inside ttnn.load_tensor / to_device / copy_host_to_device_tensor). Off = the serial,
        pipeline-order load of the accepted path.
        half = None (whole layer) or "attn" / "ffn" (half-layer partitions). `submeshes` (coord -> 1x1 submesh) shares
        the per-chip submeshes with other users in the process (one submesh = one allocator per chip and process).
        traced_prefill: None = automatic (every stage holds whole layers of a decoder with Stage 3's traced prefill)."""
        if handoff not in ("eth_direct", "eth_fifo", "host"):
            raise ValueError(handoff)
        validate_plan(stages)
        self.parent = parent_mesh
        self.handoff = handoff
        self.submeshes = submeshes if submeshes is not None else {}
        self.stages: list[PipelineStage] = []
        self.released = {}
        self.hops = []  # per hop: coords, forwarding links, socket connections
        if handoff.startswith("eth"):
            self.hops = self._validate_hops(stages, eth_connections)  # before any weight load or socket
        for p in stages:  # submeshes in the main thread, before any load
            if tuple(p.coord) not in self.submeshes:
                self.submeshes[tuple(p.coord)] = parent_mesh.create_submesh(
                    ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(*p.coord)
                )
        halves = [unit_halves(p) for p in stages]
        if parallel_load:
            decs = self._load_parallel(stages, halves, loader, drop_unused)
        else:
            decs = {}
            for si, p in enumerate(stages):
                sub = self.submeshes[tuple(p.coord)]
                # split units first: their unused half is released before the chip's other layers load, so the load
                # peak never exceeds the final footprint (Stage 04a review C-7)
                for l in sorted(p.layers, key=lambda l: halves[si][l] is None):
                    decs[(si, l)] = loader(l, sub, halves[si][l])
                    if halves[si][l] is not None and drop_unused:
                        self.released[(tuple(p.coord), l)] = drop_unused_half(decs[(si, l)], halves[si][l])
        for si, p in enumerate(stages):
            self.stages.append(PipelineStage(p, self.submeshes[tuple(p.coord)], {l: decs[(si, l)] for l in p.layers}))
        supported = all(hasattr(d, "_traced_body") for s in self.stages for d in s.decoders.values())
        self.traced_prefill = supported if traced_prefill is None else traced_prefill
        if self.traced_prefill and not supported:
            raise ValueError("traced prefill needs a decoder with Stage 3 traced bodies (OptimizedDecoder)")
        self.links = []
        if handoff.startswith("eth"):
            try:
                for k, hop in enumerate(self.hops):
                    link = EthLink(
                        self.stages[k].mesh,
                        self.stages[k + 1].mesh,
                        mode=handoff[4:],
                        connections=hop["connections"],
                        fifo_bytes=eth_fifo_bytes,
                        core_row=(link_core_row + 2 * k) % 8,
                    )
                    self.links.append(link)
                    self.stages[k].link_out = link
                    self.stages[k + 1].link_in = link
            except BaseException:
                # the caller never receives this object: delete the socket pairs made so far before the error reaches
                # close_pipeline_mesh (4a review C-3 on the constructor path)
                self.close_links()
                raise
        # traced-prefill ping-pong buffers / tail masks: one set per chip, shared by every stage on it (the ring
        # wrap-around's head and tail stages on chip 0 never run concurrently: one command queue per chip)
        self.pbufs: dict = {}
        self.pmask: dict = {}
        self.captured = False
        self.max_batch = None
        self.max_seq_len = None
        self.groups = 0
        self.prefill_captured = False

    def _load_parallel(self, stages, halves, loader, drop_unused):
        """{(stage index, layer): decoder}: one thread per chip, each loading load_jobs(...)[chip] in order."""
        from concurrent.futures import ThreadPoolExecutor

        jobs = load_jobs(stages, halves)

        def load_chip(coord):
            sub, out = self.submeshes[coord], {}
            for si, l, half in jobs[coord]:
                out[(si, l)] = loader(l, sub, half)
                if half is not None and drop_unused:
                    self.released[(coord, l)] = drop_unused_half(out[(si, l)], half)
            return out

        decs = {}
        with ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="chip-load") as ex:
            futures = {coord: ex.submit(load_chip, coord) for coord in jobs}
            errors = []
            for coord, f in futures.items():  # wait for every chip, then re-raise the first failure
                try:
                    decs.update(f.result())
                except BaseException as e:  # noqa: BLE001
                    errors.append((coord, e))
        if errors:
            raise RuntimeError(f"parallel load failed on chip(s) {[c for c, _ in errors]}") from errors[0][1]
        return decs

    def _validate_hops(self, stages, eth_connections):
        """Every hop must join two distinct chips that are row / column neighbours of the opened parent mesh and have a
        fabric route (Stage 04a review C-4: no routing TT_FATAL may fire later on an 8-chip fabric mesh). Socket
        connections = min(requested, forwarding links of the edge) (C-9)."""
        hops = []
        for k in range(len(stages) - 1):
            a, b = tuple(stages[k].coord), tuple(stages[k + 1].coord)
            if a == b:
                raise ValueError(f"hop {k}: stages {k} and {k + 1} are on the same chip {a}")
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                raise ValueError(f"hop {k}: {a} -> {b} is not a mesh edge (row / column neighbours only)")
            links = None
            try:
                src = self.parent.get_fabric_node_id(ttnn.MeshCoordinate(*a))
                dst = self.parent.get_fabric_node_id(ttnn.MeshCoordinate(*b))
                links = tuple(ttnn.get_forwarding_link_indices(src, dst))
            except Exception as e:  # noqa: BLE001  (fabric disabled: no route query possible)
                raise ValueError(
                    f"hop {k}: {a} -> {b}: no fabric route query ({e!r}); open the mesh with fabric"
                ) from e
            if not links:
                raise ValueError(f"hop {k}: {a} -> {b}: no forwarding link")
            hops.append(
                dict(
                    src=a,
                    dst=b,
                    src_chip=stages[k].chip,
                    dst_chip=stages[k + 1].chip,
                    links=list(links),
                    connections=min(eth_connections, len(links)),
                )
            )
        return hops

    def close_links(self):
        """Release every socket pair (call before closing the mesh, on every path; Stage 04a review C-3)."""
        for st in self.stages:
            st.link_in = st.link_out = None
        for link in self.links:
            link.send_socket = link.recv_socket = None
        self.links = []
        import gc

        gc.collect()

    # ----------------------------------------------------------------------------------------------- queries
    @property
    def layers(self) -> list[int]:
        """Layers of the stack in order (a split layer once)."""
        out = []
        for s in self.stages:
            for l in s.plan.layers:
                if l not in out:
                    out.append(l)
        return out

    @property
    def split_layers(self) -> list[int]:
        return [u.layer for s in self.stages for u in s.plan.units if u.part == "attn"]

    def decoder(self, layer: int):
        for s in self.stages:
            if layer in s.decoders:
                return s.decoders[layer]
        raise KeyError(layer)

    def stage_of(self, layer: int) -> PipelineStage:
        for s in self.stages:
            if layer in s.state_layers:
                return s
        raise KeyError(layer)

    def unit_buckets(self, l: int) -> tuple[int, ...]:
        """Traced-prefill buckets of layer l (OptimizedDecoder.trace_buckets rule from its max_seq_len)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import PREFILL_CHUNK, bucket_for
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import traced_buckets

        dec = self.decoder(l)
        msl = self.max_seq_len[l] if isinstance(self.max_seq_len, dict) else self.max_seq_len
        reach = bucket_for(min(msl, PREFILL_CHUNK), dec.prefill_align)
        return tuple(p for p in traced_buckets(dec.prefill_align) if p <= reach) or (dec.prefill_align,)

    def buckets(self) -> tuple[int, ...]:
        """Union of the traced-prefill bucket sets of the stack's layers (KDA 32.., MLA 64..)."""
        out = set()
        for l in self.layers:
            out |= set(self.unit_buckets(l))
        return tuple(sorted(out))

    # ----------------------------------------------------------------------------------------------- state
    def allocate(
        self,
        *,
        max_batch: int,
        max_seq_len: int | dict,
        microbatches: int = 1,
        probe_units: bool = False,
        plus_one: bool = False,
    ):
        """Caches + persistent per-chip step inputs, hand-off and prefill buffers for `microbatches` groups of
        `max_batch` slots each (KDA decode processes every slot of its state: rows == max_batch per group). Must
        precede the first trace capture on every chip. probe_units: one persistent output copy per unit (tests).
        plus_one: each chip's decode trace advances its own cur_pos copy (ttnn.plus_one, inactive -1 entries kept)."""
        self.max_batch, self.groups, self.max_seq_len = max_batch, microbatches, max_seq_len
        for st in self.stages:
            st.groups = []
            st.plus_one = plus_one
            for _ in range(microbatches):
                g = _Group()
                for l in st.state_layers:
                    dec = st.decoders[l]
                    msl = max_seq_len[l] if isinstance(max_seq_len, dict) else max_seq_len
                    g.caches[l] = dec.allocate_cache(max_batch=max_batch, max_seq_len=msl)
                if st.mla_layers:
                    l0 = st.mla_layers[0]
                    g.page_table = st.decoders[l0].make_page_table(g.caches[l0])
                    for l in st.mla_layers[1:]:
                        if g.caches[l].blocks_per_seq != g.caches[l0].blocks_per_seq:
                            raise ValueError("MLA layers of one chip need one page-table geometry")
                        g.caches[l].page_table_host = g.caches[l0].page_table_host
                g.cur_pos = ttnn.from_torch(
                    torch.full((max_batch,), -1, dtype=torch.int32), dtype=ttnn.int32, device=st.mesh
                )
                g.streams_in = _zeros(st.mesh, max_batch)
                if st is self.stages[-1] or self.handoff == "host":
                    g.out_buf = _zeros(st.mesh, max_batch)
                if probe_units:
                    g.probes = [_zeros(st.mesh, max_batch) for _ in st.plan.units]
                st.groups.append(g)
        if self.traced_prefill:
            for st in self.stages:
                key = tuple(st.plan.coord)
                if key not in self.pbufs:
                    self.pbufs[key] = {p: [_zeros(st.mesh, p), _zeros(st.mesh, p)] for p in self.buckets()}
                    # fp32 masks: the kept rows keep their exact fp32 values
                    # one per bucket: a padded chunk (chunk policy "padded" / "auto") masks at its own bucket
                    self.pmask[key] = {p: _zeros(st.mesh, p, width=1) for p in sorted(set(self.buckets()) | {32})}
                st.pbufs, st.pmask = self.pbufs[key], self.pmask[key]

    def blocks_per_seq(self) -> int | None:
        for st in self.stages:
            for l in st.mla_layers:
                return st.groups[0].caches[l].blocks_per_seq
        return None

    def set_page_table(self, table: torch.Tensor, *, group: int = 0):
        """Request-boundary helper: one [slots, blocks_per_seq] table shared by every MLA layer of the stack, written in
        place into every chip's persistent copy (a captured decode trace sees the new mapping) and into the host copy
        prefill chunk tables are built from. Entries must lie in [0, num_blocks) and live slots must not share a
        block (checked: rows are distinct sets)."""
        bps = self.blocks_per_seq()
        if bps is None:
            return
        t = table.to(torch.int32).reshape(self.max_batch, bps)
        nblocks = self.max_batch * bps
        if int(t.min()) < 0 or int(t.max()) >= nblocks:
            raise ValueError(f"page-table entries outside [0, {nblocks})")
        if len(set(t.reshape(-1).tolist())) != t.numel():
            raise ValueError("page-table rows share physical blocks")
        host = ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for st in self.stages:
            if not st.mla_layers:
                continue
            g = st.groups[group]
            for l in st.mla_layers:
                g.caches[l].page_table_host = t
            ttnn.copy_host_to_device_tensor(host, g.page_table)

    def set_decode_inputs(self, positions, *, group: int = 0):
        """Host boundary: cur_pos (absolute position per slot, -1 = inactive) into every chip's persistent copy."""
        host = ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), dtype=ttnn.int32)
        for st in self.stages:
            for l in st.state_layers:
                st.decoders[l].check_decode_positions(positions, st.groups[group].caches[l])
            ttnn.copy_host_to_device_tensor(host, st.groups[group].cur_pos)

    def read_positions(self, *, group: int = 0) -> list[list[int]]:
        """Every chip's cur_pos copy (tests of the in-trace plus_one)."""
        return [ttnn.to_torch(st.groups[group].cur_pos).reshape(-1).tolist() for st in self.stages]

    def write_streams(self, streams: torch.Tensor, *, group: int = 0):
        """Host boundary: first stage's decode input [B, 16384] (Stage 6: the embedding writes it on chip 0)."""
        x = streams.reshape(1, 1, self.max_batch, HC_WIDTH).float().contiguous()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), self.stages[0].groups[group].streams_in
        )

    def output(self, *, group: int = 0):
        """The stack's decode output buffer [1, 1, B, 16384] fp32 on the last chip (persistent)."""
        return self.stages[-1].groups[group].out_buf

    def reset_slot(self, slot: int, *, group: int = 0):
        for st in self.stages:
            for l in st.state_layers:
                st.decoders[l].reset_slot(st.groups[group].caches[l], slot)

    # ---------------------------------------------------------------------------------------------- decode
    def _stage_decode(self, k: int, g_idx: int):
        """One stage's decode program sequence (recorded as that chip's trace for this group)."""
        st = self.stages[k]
        g = st.groups[g_idx]
        if st.link_in is not None:
            st.link_in.recv(g.streams_in)
        out = st.decode_units(g, g.streams_in)
        if st.link_out is not None:
            st.link_out.send(out)
        if g.out_buf is not None:
            ttnn.copy(out, g.out_buf)  # persistent stage output (step output / host hand-off source)
        if out is not g.streams_in:
            ttnn.deallocate(out)
        if st.plus_one:
            ttnn.plus_one(g.cur_pos, skip_negative_entries=True)

    def decode_eager(self, group: int = 0):
        """Untraced decode step (warm-up / debug): every stage's ops in pipeline order."""
        for k, st in enumerate(self.stages):
            self._stage_decode(k, group)
            if self.handoff == "host" and k + 1 < len(self.stages):
                g = st.groups[group]
                g.host_buf = host_handoff(g.out_buf, self.stages[k + 1].groups[group].streams_in, g.host_buf)
        return self.stages[-1].groups[group].out_buf

    def _snapshot_positions(self, gi):
        return [ttnn.clone(st.groups[gi].cur_pos) for st in self.stages]

    def _restore_positions(self, gi, snaps):
        for st, s in zip(self.stages, snaps):
            ttnn.copy(s, st.groups[gi].cur_pos)
            ttnn.deallocate(s)

    def warm_and_capture(self, *, prefill: bool = True):
        """Warm every decode program of every group (one eager step on the captured tensors), restore the state and
        positions the warm-up advanced, capture one decode trace per (chip, group), then (traced prefill) one prefill
        trace per (layer, bucket). All persistent buffers exist already; prepare_prefill must have run."""
        for gi in range(self.groups):
            snaps = [st.snapshot(st.groups[gi]) for st in self.stages]
            pos = self._snapshot_positions(gi)
            self.decode_eager(gi)
            for st in self.stages:
                ttnn.synchronize_device(st.mesh)
            for st, s in zip(self.stages, snaps):
                st.restore(st.groups[gi], s)
            self._restore_positions(gi, pos)
            for st in self.stages:
                ttnn.synchronize_device(st.mesh)
        for gi in range(self.groups):
            for k, st in enumerate(self.stages):
                g = st.groups[gi]
                g.trace_id = ttnn.begin_trace_capture(st.mesh, cq_id=0)
                try:
                    self._stage_decode(k, gi)
                finally:  # never leave the CQ in capture mode (a later close / read would hang)
                    ttnn.end_trace_capture(st.mesh, g.trace_id, cq_id=0)
        if self.traced_prefill and prefill:
            self._capture_prefill()
        self.captured = True

    def decode_step(self, group: int = 0, *, blocking: bool = True):
        """One traced decode step of `group`: replay every chip's trace in pipeline order. eth: all replays enqueued
        non-blocking (device-side socket handshakes order them); host: replay k, wait, D2H/H2D into k+1's input."""
        for k, st in enumerate(self.stages):
            g = st.groups[group]
            ttnn.execute_trace(st.mesh, g.trace_id, cq_id=0, blocking=False)
            if self.handoff == "host" and k + 1 < len(self.stages):
                g.host_buf = host_handoff(g.out_buf, self.stages[k + 1].groups[group].streams_in, g.host_buf)
        if blocking:
            self.synchronize()
        return self.stages[-1].groups[group].out_buf

    def decode_steps(self, groups: list[int], steps: int, *, blocking: bool = True):
        """Pipelined dispatch of `steps` traced decode steps of every group in `groups` (micro-batches in flight: chip
        k runs one group while chip k+1 runs the previous one). Work items (step n, group g) are issued in order; per
        command queue: the body stages (all but the last) of the first W items, then per item its last stage followed by
        the body of item + W. W = min(len(groups), body stages - 1) when the last stage shares a chip with the first
        (ring wrap-around): every hop is a socket with one message in flight, so at most body - 1 groups can sit between
        chip 0's head and tail; one more head ahead of the oldest tail closes a cyclic wait (each chip parked in a send;
        observed on the 4-chip ring, work_log 2026-10-18 05:48Z). W <= len(groups) also keeps the body of (n + 1, g) behind
        the tail of (n, g) on chip 0 (token feedback order). Later steps reuse the stages' persistent inputs (positions
        advance in-trace with plus_one; Stage 6 feeds tokens on device)."""
        if self.handoff == "host":
            for _ in range(steps):
                for g in groups:
                    self.decode_step(g, blocking=False)
            if blocking:
                self.synchronize()
            return
        body, tail = self.stages[:-1], self.stages[-1]
        shared = tuple(tail.plan.coord) == tuple(body[0].plan.coord)
        w = min(len(groups), len(body) - 1) if shared else len(groups)
        w = max(w, 1)
        items = [g for _ in range(steps) for g in groups]

        def run_body(g):
            for st in body:
                ttnn.execute_trace(st.mesh, st.groups[g].trace_id, cq_id=0, blocking=False)

        for g in items[:w]:
            run_body(g)
        for i, g in enumerate(items):
            ttnn.execute_trace(tail.mesh, tail.groups[g].trace_id, cq_id=0, blocking=False)
            if i + w < len(items):
                run_body(items[i + w])
        if blocking:
            self.synchronize()

    def max_inflight(self, groups: int) -> int:
        body, tail = self.stages[:-1], self.stages[-1]
        shared = tuple(tail.plan.coord) == tuple(body[0].plan.coord)
        return max(1, min(groups, len(body) - 1) if shared else groups)

    def synchronize(self):
        for st in self.stages:
            ttnn.synchronize_device(st.mesh)

    def release(self):
        """Release every trace (decode and prefill). Buffers stay allocated (a new capture may follow)."""
        for st in self.stages:
            for g in st.groups:
                if g.trace_id is not None:
                    ttnn.release_trace(st.mesh, g.trace_id)
                    g.trace_id = None
                for traces in g.prefill_traces.values():
                    for t in traces.values():
                        if t.trace_id is not None:
                            ttnn.release_trace(st.mesh, t.trace_id)
                            t.trace_id = None
        self.prefill_captured = False

    def free(self):
        """Release traces and every buffer allocate() made (caches, inputs, prefill buffers)."""
        self.release()
        for st in self.stages:
            for g in st.groups:
                for t in (g.cur_pos, g.streams_in, g.out_buf, g.page_table, *g.probes):
                    if t is not None and t.is_allocated():
                        ttnn.deallocate(t)
                for c in g.caches.values():
                    for t in _device_tensors(c):
                        if t.storage_type() == ttnn.StorageType.DEVICE and t.is_allocated():
                            t.deallocate(True)
            st.groups = []
            st.pbufs, st.pmask = {}, {}
            for t in st._prefill_buf.values():
                ttnn.deallocate(t)
            st._prefill_buf = {}
        for d in list(self.pbufs.values()) + list(self.pmask.values()):
            for v in d.values():
                for t in v if isinstance(v, list) else [v]:
                    ttnn.deallocate(t)
        self.pbufs, self.pmask = {}, {}
        self.groups = 0
        self.captured = False

    # --------------------------------------------------------------------------------------- traced prefill
    def _unit_traces(self, st: PipelineStage, g: _Group, i: int, u: Unit):
        """{bucket: PrefillTrace} of unit i of stage st: x = bufs[p][i % 2], y = bufs[p][(i + 1) % 2]."""
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import PrefillTrace

        key = (u.layer, u.part)
        if key not in g.prefill_traces:
            g.prefill_traces[key] = {
                p: PrefillTrace(bucket=p, x=st.pbufs[p][i % 2], y=st.pbufs[p][(i + 1) % 2], host_io=False)
                for p in self.unit_buckets(u.layer)
            }
        return g.prefill_traces[key]

    @staticmethod
    def _unit_body(dec, u: Unit, t, cache):
        """Captured work of one bucket of a unit: whole layer = Stage 3's traced body (layer over the persistent input,
        output copied into the persistent output); half layer = the same body split at the post-attention streams
        (attention slot-independent through the scratch KDA state / the MLA chunk tables)."""
        if u.part == "full":
            dec._traced_body(t, cache)
            return
        moe = dec.layer.mlp if u.part == "ffn" and dec.spec.mlp == "moe" else None
        if moe is not None:
            moe.traced_prefill = True  # the MoE takes its traced-prefill path, as in Stage 3's body
        try:
            attn = prefill_attention(dec, cache, None) if u.part == "attn" else None
            y = forward_unit(dec, t.x, attn, decode=False, part=u.part)
        finally:
            if moe is not None:
                moe.traced_prefill = False
        ttnn.copy(y, t.y)
        ttnn.deallocate(y)

    def _capture_prefill(self):
        """One trace per (group, unit, bucket) over the shared buffers (Stage 3 capture_prefill_traces body)."""
        for st in self.stages:
            for g in st.groups:
                for i, u in enumerate(st.plan.units):
                    dec = st.decoders[u.layer]
                    cache = g.caches.get(u.layer)
                    traces = self._unit_traces(st, g, i, u)
                    if u.part == "full":
                        dec.capture_prefill_traces(cache, traces)
                        continue
                    for t in traces.values():
                        if cache is not None:
                            dec._prime_chunk(cache, t.bucket)
                        tid = ttnn.begin_trace_capture(st.mesh, cq_id=0)
                        try:
                            self._unit_body(dec, u, t, cache)
                        except BaseException:
                            ttnn.end_trace_capture(st.mesh, tid, cq_id=0)  # never leave the queue capturing
                            ttnn.release_trace(st.mesh, tid)
                            raise
                        ttnn.end_trace_capture(st.mesh, tid, cq_id=0)
                        t.trace_id = tid
                    if cache is not None:
                        dec.reset_slot(cache, 0)
            ttnn.synchronize_device(st.mesh)
        self.prefill_captured = True

    def _run_unit(self, st, g, dec, u: Unit, t, *, user_slot, chunk_start, n_valid, eager):
        """Stage 3 run_prefill_trace (non-blocking replay) for whole layers; for half layers the same steps around the
        unit's trace (only the attention half owns the layer's state: chunk inputs + KDA scratch state). Before the
        capture (`eager`), the same work runs eagerly."""
        cache = g.caches.get(u.layer)
        if u.part == "full" and not eager:
            dec.run_prefill_trace(t, cache, user_slot=user_slot, chunk_start=chunk_start, n_valid=n_valid)
            return
        if u.part != "ffn":
            dec.begin_prefill_chunk(cache, user_slot=user_slot, chunk_start=chunk_start, n_valid=n_valid)
            if dec.is_kda:
                dec._scratch_state_in(cache, user_slot)
        if eager:
            self._unit_body(dec, u, t, cache)
        else:
            if t.trace_id is None:
                raise RuntimeError(f"{u} bucket {t.bucket}: no captured prefill trace")
            ttnn.execute_trace(st.mesh, t.trace_id, cq_id=0, blocking=False)
        if u.part != "ffn" and dec.is_kda:
            ttnn.fill_cache(cache.kda.S, cache.kda.prefill_S, user_slot)

    def _mask_rows(self, st, buf, p, n_valid):
        """Rows >= n_valid of buf [1,1,p,W] -> exact 0.0 (padded chunks only), in place (mask written per chunk)."""
        if n_valid >= p:
            return
        masked = ttnn.where(st.pmask[p], buf, 0.0)
        ttnn.copy(masked, buf)
        ttnn.deallocate(masked)

    def _rebucket(self, src, dst):
        """Tail transition between a 64-row and a 32-row bucket buffer (rows past 32 are zero / dropped)."""
        ps, pd = src.shape[-2], dst.shape[-2]
        if pd < ps:
            y = ttnn.slice(src, (0, 0, 0, 0), (1, 1, pd, HC_WIDTH), memory_config=DRAM)
        else:
            y = ttnn.pad(src, ((0, 0), (0, 0), (0, pd - ps), (0, 0)), 0.0, memory_config=DRAM)
        ttnn.copy(y, dst)
        ttnn.deallocate(y)

    def _stage_prefill_chunk(self, k, g_idx, *, phys, n_valid, chunk_start, user_slot, eager, keep=None):
        """Stage k's units over one model chunk whose input is in bufs[phys][0] of the stage's chip. Returns the output
        buffer (bufs[phys][U % 2], U = number of units); `keep` (dict) collects every unit's valid output rows on the
        host (tests): key "L<l>" for layer outputs, "L<l>.attn" for an attention half."""
        st = self.stages[k]
        g = st.groups[g_idx]
        tail = n_valid < phys
        if phys not in st.pbufs:
            raise ValueError(f"prefill chunk of {phys} rows: no prepared bucket (prepared: {sorted(st.pbufs)})")
        if tail:  # valid-row masks of the buckets this chunk's units run at (one small write each)
            used = {layer_bucket(st.decoders[u.layer], phys, n_valid) for u in st.plan.units}
            for p, m in st.pmask.items():
                if p not in used:
                    continue
                v = torch.zeros(1, 1, p, 1)
                v[0, 0, : min(n_valid, p)] = 1.0
                ttnn.copy_host_to_device_tensor(ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), m)
        cur_p = phys
        for i, u in enumerate(st.plan.units):
            dec = st.decoders[u.layer]
            p = layer_bucket(dec, phys, n_valid)
            if p != cur_p:  # input of unit i lives in bufs[cur_p][i % 2]; unit i reads bufs[p][i % 2]
                self._rebucket(st.pbufs[cur_p][i % 2], st.pbufs[p][i % 2])
                cur_p = p
            t = self._unit_traces(st, g, i, u)[p]
            self._run_unit(st, g, dec, u, t, user_slot=user_slot, chunk_start=chunk_start, n_valid=n_valid, eager=eager)
            if tail:
                self._mask_rows(st, t.y, p, n_valid)
            if keep is not None:
                key = f"L{u.layer}" + (".attn" if u.part == "attn" else "")
                keep.setdefault(key, []).append(ttnn.to_torch(t.y).float().reshape(-1, HC_WIDTH)[:n_valid])
        last = len(st.plan.units) % 2
        if cur_p != phys:
            self._rebucket(st.pbufs[cur_p][last], st.pbufs[phys][last])
        return st.pbufs[phys][last]

    def prefill_host_chunks(self, streams, *, chunk_policy: str = "plan") -> list:
        """Host ttnn tensors (TILE fp32, zero padded) of every model chunk of a prompt, for prefill_forward(host_chunks=)
        (the host tilization of a 2048-row chunk costs ~0.1 s; Stage 6 produces the chunk on chip 0 instead)."""
        rows = streams.reshape(-1, HC_WIDTH).float()
        out = []
        for off, n, phys in prefill_chunk_plan(rows.shape[0], self.buckets(), chunk_policy):
            chunk = torch.zeros(1, 1, phys, HC_WIDTH)
            chunk[0, 0, :n] = rows[off : off + n]
            out.append(ttnn.from_torch(chunk, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT))
        return out

    def _prefill_traced(
        self,
        streams,
        *,
        user_slot,
        start_pos,
        group,
        keep_unit_outputs,
        eager,
        return_output,
        host_chunks=None,
        chunk_policy="plan",
    ):
        rows = streams.reshape(-1, HC_WIDTH).float()
        seq_len = rows.shape[0]
        for st in self.stages:
            for l in st.state_layers:
                c = st.groups[group].caches[l]
                if start_pos + seq_len > c.max_seq_len:
                    raise ValueError(f"prefill end {start_pos + seq_len} > allocated max_seq_len {c.max_seq_len}")
        if start_pos % MODEL_ALIGN:
            raise ValueError(f"prefill continuation start {start_pos} must be a multiple of {MODEL_ALIGN} (MLA page)")
        keep = {} if keep_unit_outputs else None
        outs = []
        first = self.stages[0]
        for ci, (off, n, phys) in enumerate(prefill_chunk_plan(seq_len, self.buckets(), chunk_policy)):
            if phys not in first.pbufs:
                raise ValueError(f"prefill chunk of {phys} rows: no prepared bucket")
            if host_chunks is not None:
                host = host_chunks[ci]
            else:
                chunk = torch.zeros(1, 1, phys, HC_WIDTH)
                chunk[0, 0, :n] = rows[off : off + n]
                host = ttnn.from_torch(chunk, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, first.pbufs[phys][0])
            for k, st in enumerate(self.stages):
                if k > 0:
                    src = self.stages[k - 1]
                    src_buf = src.pbufs[phys][len(src.plan.units) % 2]
                    if self.handoff == "host":
                        host_handoff(src_buf, st.pbufs[phys][0])
                    else:
                        src.link_out.send(src_buf)
                        st.link_in.recv(st.pbufs[phys][0])
                y = self._stage_prefill_chunk(
                    k,
                    group,
                    phys=phys,
                    n_valid=n,
                    chunk_start=start_pos + off,
                    user_slot=user_slot,
                    eager=eager,
                    keep=keep,
                )
            if return_output:
                outs.append(ttnn.to_torch(y).float().reshape(-1, HC_WIDTH)[:n])
        if not return_output:
            return None
        out = torch.cat(outs).reshape(1, seq_len, HC_WIDTH)
        if keep_unit_outputs:
            return out, {k: torch.cat(v).reshape(1, seq_len, HC_WIDTH) for k, v in keep.items()}
        return out

    # --------------------------------------------------------------------------------------------- prefill
    def prepare_prefill(self, lengths=None, *, user_slot: int = 0, group: int = 0):
        """Before the first decode capture. Traced prefill: one eager pass of every model bucket (and both tail
        shapes, KDA 32 / 64 rows) through the whole pipeline on the shared buffers (compiles every traced body, the
        hand-off of every bucket shape, the tail masks and re-bucketing), plus the slot-indexed KDA scratch-state reads
        of every slot (Stage 3 prepare_prefill_traces). Eager prefill: one zero chunk of every logical length in
        `lengths`. Overwrites `user_slot`'s state, then resets it."""
        if not self.traced_prefill:
            for n in lengths or (2048,):
                self.reset_slot(user_slot, group=group)
                self._prefill_eager(torch.zeros(1, n, HC_WIDTH), user_slot=user_slot, group=group)
            self.reset_slot(user_slot, group=group)
            self.synchronize()
            return
        lens = sorted({b for b in self.buckets() if b % MODEL_ALIGN == 0} | {20, 40})  # 20: KDA tail 32, 40: 64
        for g_idx in range(self.groups):
            for n in lens:
                self._prefill_traced(
                    torch.zeros(1, n, HC_WIDTH),
                    user_slot=user_slot,
                    start_pos=0,
                    group=g_idx,
                    keep_unit_outputs=False,
                    eager=True,
                    return_output=True,
                )
            for st in self.stages:
                g = st.groups[g_idx]
                for l in st.state_layers:
                    dec, cache = st.decoders[l], g.caches[l]
                    if dec.is_kda:
                        for slot in range(cache.max_batch):
                            dec._scratch_state_in(cache, slot)
                        ttnn.fill_cache(cache.kda.S, cache.kda.prefill_S, 0)
            self.reset_slot(user_slot, group=g_idx)
        # the row-mask programs of every bucket (a padded chunk masks at its own bucket; the eager passes above only
        # masked the 32 / 64-row tails): compiled here, before any capture
        done = set()
        for st in self.stages:
            if tuple(st.plan.coord) in done:
                continue
            done.add(tuple(st.plan.coord))
            for p in st.pmask:
                if p in st.pbufs:
                    self._mask_rows(st, st.pbufs[p][0], p, p - 1)
        self.synchronize()

    def prefill_forward(
        self,
        streams: torch.Tensor,
        *,
        user_slot: int = 0,
        start_pos: int = 0,
        group: int = 0,
        keep_unit_outputs: bool = False,
        return_output: bool = True,
        host_chunks: list | None = None,
        chunk_policy: str = "plan",
    ):
        """Host orchestration of a prompt through the pipeline: streams [1, seq_len, 16384] (or [1, seq_len, 4, 4096])
        -> torch [1, seq_len, 16384] fp32 output of the last layer (valid rows only). Any seq_len >= 1; continuation
        (start_pos > 0) at a multiple of 64. Traced (after warm_and_capture) when the stack supports it, else eager.
        chunk_policy: prefill_chunk_plan policy: "plan" (default, device-accepted) | "auto" | "padded" (opt-in, guarded
        to < 32 padded rows, not device-validated); host_chunks must use the same.
        keep_unit_outputs: also returns {"L<l>": torch rows} of every layer (tests; one host read per layer and
        chunk)."""
        if self.traced_prefill:
            if not self.prefill_captured:
                raise RuntimeError("traced prefill: call prepare_prefill and warm_and_capture first")
            return self._prefill_traced(
                streams,
                user_slot=user_slot,
                start_pos=start_pos,
                group=group,
                keep_unit_outputs=keep_unit_outputs,
                eager=False,
                return_output=return_output,
                host_chunks=host_chunks,
                chunk_policy=chunk_policy,
            )
        return self._prefill_eager(
            streams, user_slot=user_slot, start_pos=start_pos, group=group, keep_unit_outputs=keep_unit_outputs
        )

    def _prefill_eager(
        self, streams, *, user_slot: int = 0, start_pos: int = 0, group: int = 0, keep_unit_outputs: bool = False
    ):
        """Stage 04a eager chunked prefill (decoders without traced bodies, half-layer units)."""
        from models.autoports.zai_org_glm_5_3_flash.tt.functional_decoder import PREFILL_CHUNK, bucket_for

        rows = streams.reshape(-1, HC_WIDTH).float()
        seq_len = rows.shape[0]
        outs, unit_rows = [], {}
        for off in range(0, seq_len, PREFILL_CHUNK):
            n = min(PREFILL_CHUNK, seq_len - off)
            first = self.stages[0].decoders[self.stages[0].plan.units[0].layer]
            p0 = bucket_for(n, first.prefill_align)
            chunk = torch.zeros(1, 1, p0, HC_WIDTH)
            chunk[0, 0, :n] = rows[off : off + n]
            x = ttnn.from_torch(
                chunk, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.stages[0].mesh, memory_config=DRAM
            )
            for st in self.stages:
                for l in st.state_layers:
                    st.decoders[l].begin_prefill_chunk(
                        st.groups[group].caches[l], user_slot=user_slot, chunk_start=start_pos + off, n_valid=n
                    )
            for k, st in enumerate(self.stages):
                g = st.groups[group]
                if k > 0:
                    x = self._prefill_handoff(k, x)
                for u in st.plan.units:
                    dec = st.decoders[u.layer]
                    if u.part in ("full", "attn"):  # layer start: the layer's own bucket, padded rows zeroed
                        y = rezero_rows(x, n, bucket_for(n, dec.prefill_align))
                        if y is not x:
                            ttnn.deallocate(x)
                        x = y
                    attn = None if u.part == "ffn" else prefill_attention(dec, g.caches[u.layer], user_slot)
                    y = forward_unit(dec, x, attn, decode=False, part=u.part)
                    ttnn.deallocate(x)
                    x = y
                    if keep_unit_outputs and u.part != "attn":
                        unit_rows.setdefault(f"L{u.layer}", []).append(
                            ttnn.to_torch(x).float().reshape(-1, HC_WIDTH)[:n]
                        )
            outs.append(ttnn.to_torch(x).float().reshape(-1, HC_WIDTH)[:n])
            ttnn.deallocate(x)
        out = torch.cat(outs).reshape(1, seq_len, HC_WIDTH)
        if keep_unit_outputs:
            return out, {k: torch.cat(v).reshape(1, seq_len, HC_WIDTH) for k, v in unit_rows.items()}
        return out

    def _prefill_handoff(self, k: int, x):
        """Eager chunk hand-off stage k-1 -> k: socket transfer into a per-shape receive buffer, or the host path."""
        st = self.stages[k]
        key = tuple(x.shape)
        buf = st._prefill_buf.get(key)
        if buf is None:
            if self.captured:  # Stage 04a review C-5: no persistent allocation once traces are live
                raise ValueError(f"eager prefill hand-off of shape {key}: not prepared before the capture")
            buf = st._prefill_buf[key] = ttnn.allocate_tensor_on_device(x.spec, st.mesh)
        if self.handoff == "host":
            host_handoff(x, buf)
        else:
            self.stages[k - 1].link_out.send(x)
            st.link_in.recv(buf)
        ttnn.deallocate(x)
        return ttnn.clone(buf, memory_config=DRAM)

    # ------------------------------------------------------------------------------------------- footprint
    def dram_report(self) -> dict:
        """Allocator view per chip: {chip: {"allocated": bytes, "total": bytes}}."""
        out = {}
        for st in self.stages:
            c = st.plan.chip if st.plan.chip is not None else str(st.plan.coord)
            out[c] = {"allocated": dram_allocated(st.mesh), "total": dram_total(st.mesh)}
        return out


# Blackhole fabric packet payload: default 4352 B (one fp32 tile page per packet); 14 x 1088 = 15232 B is the arch
# maximum and cut the 2 MiB decode hop from 136 to 55 us with 2 socket connections (Stage 04a handoff_bench.jsonl).
FABRIC_PAYLOAD_BYTES = 15232


def open_pipeline_mesh(
    shape=(2, 4),
    *,
    trace_region_size: int = 600_000_000,
    fabric: bool = True,
    mgd: str | None = None,
    fabric_config=None,
    payload_bytes: int | None = FABRIC_PAYLOAD_BYTES,
):
    """Parent mesh for a pipeline over `shape` chips (TT_VISIBLE_DEVICES selects the chips). Fabric (FABRIC_1D) is set
    BEFORE the mesh opens (socket hops need it; every hop of the plan must be a mesh edge). `mgd`: mesh graph
    descriptor path (required for CUSTOM clusters: 1, 4, 6 or 8 P300 chips; mesh/glm8_2x4_* for all 8 chips)."""
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    if fabric:
        cfg = fabric_config or ttnn.FabricConfig.FABRIC_1D
        if payload_bytes:
            rc = ttnn.FabricRouterConfig()
            rc.max_packet_payload_size_bytes = payload_bytes
            ttnn.set_fabric_config(cfg, router_config=rc)
        else:
            ttnn.set_fabric_config(cfg)
    return ttnn.open_mesh_device(ttnn.MeshShape(*shape), trace_region_size=trace_region_size)


def close_pipeline_mesh(mesh, model: MultichipDecoder | None = None, *, fabric: bool = True):
    """Release traces, then quiesce the parent (it quiesces every submesh and clears their command-queue in-use flags)
    and close."""
    if model is not None:
        try:
            model.release()
            model.synchronize()
        finally:
            model.close_links()
    mesh.quiesce_devices()
    ttnn.close_mesh_device(mesh)
    if fabric:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

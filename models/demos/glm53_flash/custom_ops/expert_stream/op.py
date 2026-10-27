# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash routed experts as DRAM-bank-streaming generic ops (one chip's slice of the hybrid placement).

After ``models/demos/gpt_oss/tt/fused_decode/stream.py`` (ExpertGateUpStream / ExpertDownStream), adapted to the
chip's slice of the hybrid placement (E_local experts of the column, I_local-wide intermediate slice of the row:
256 on the 8x4, 1024 on a (2,4) block, 2048 on one chip; K = H = 4096), bfp8 weights, no biases, the router's dense
routing matrix and the clamped SwiGLU (``silu(min(g, 10)) * clamp(u, -10, 10)``).

Weights are stored width-sharded over the DRAM banks (``stream_memory_config``), bank-major and column-major, so a
worker core next to a bank streams whole weight columns (K tiles each) as contiguous byte ranges in large NOC
packets, expert-major so an expert's columns for a core are contiguous:

* gate|up: ``[E, K, 2 I]`` -> ``ppb = I / 32 / banks`` column pairs per bank: bank ``b`` holds the pairs
  ``(gate b ppb + q, up b ppb + q)``, ``q < ppb``, of every expert (``2 ppb`` columns of ``K/32`` tiles per expert,
  ``expert_stride`` bytes).
* down: ``[E, I, H]`` -> bank ``b`` holds output tile columns ``[b C, (b+1) C)`` (``C = H / 32 / banks = 16``) of
  every expert, ``I / 32`` K tiles per column: expert e's columns ``[c0, c0 + cols)`` of a core are contiguous.

Block ops (every kernel reads the routing matrix ``[1, 1, 32, E]`` BF16 TILE itself and scans the active experts, so
the calls are trace-safe with data-dependent routing and need no host-side counts); one 32-token block per call:

* ``ExpertGateUpStream(x, routing, w13, act)``: ``G`` groups of 8 cores (one core per bank per group); group ``g``
  streams the active experts ``g, g + G, ...``; per expert and column pair: gate and up columns against the resident
  ``[32, K]`` activation (HiFi2, fp32 acc), ``clamped_silu_glu`` in DST, the routing weights as ``D_e @ h``
  (``D_e = diag(w[:, e])``, exact), written to ``act [1, E, 32, I]`` (tile ``(e, b ppb + q)``).
* ``ExpertDownStream(act, routing, w2, out)``: ``R`` readers per bank, ``cols = C / R`` output columns each
  (``<= 8``: the fp32 DST capacity, ``cols * I/32 <= 128`` tiles per expert block); for every active expert the core
  multiplies ``act[e]`` (``I/32`` tiles) against its columns' segments, accumulating the expert sum in DST, and
  writes its ``[32, 32]`` output tiles of ``out [1, 1, 32, H]`` once (zeros when no expert is active).

Dispatch ops (phase 2, ``T`` gathered tokens > 32, every active expert's weights read once): the per-expert token
counts ``counts = sum_t (routing != 0)`` (``[1, 1, 1, E]`` BF16, two ttnn ops) give every kernel the same pass
enumeration (expert ascending, 32 routed rows per pass) and the same LPT expert -> core-group assignment:

* ``ExpertGateUpDispatch(x_rm, routing, counts, w13, act, table)``: per pass the writer gathers the pass's rows of the
  ROW_MAJOR ``x [1, 1, T, K]`` (one 8 KB read per row) into a row-major block that the compute tilizes, then the
  gate|up / SwiGLU / ``D @ h`` as above into ``act [1, P, 32, I]`` (``P = T/4 + E`` passes at most); bank 0 writes
  the pass table ``[1, 1, P, 64]`` uint16 (``[n_rows, e, t_0 .. t_31]``). With one pair per bank an expert's columns
  stay resident over its passes; with more pairs they are re-streamed per pass.
* ``ExpertDownDispatch(act, table, counts, w2, out)``: 8 readers per bank (2 output columns each, 64 cores); phase 1
  computes ``h_k = act[k] @ W2[e][:, cols]`` per pass into a resident BF16 buffer, phase 2 gathers every token's
  contributions (``<= 8`` per token) into slot tiles and sums them in fp32 DST (identity-tile matmuls, HiFi4) into
  ``out [1, 1, T, H]``.

Exactness: the same products (BF16 x BFP8, HiFi2) and fp32 accumulation as the ``sparse_matmul`` block, the expert
sum in fp32 (the slab path sums BF16 tiles), the routing weight as one BF16 x BF16 product per element (bit-exact
with the multiply it replaces). ``golden`` is the torch reference on the bfp8-rounded weights.

E1 variants of the block ops (decode roofline action E1, 2026-10-26; every one behind a constructor flag defaulting to
today's program, selected in ``tt/moe.py`` by ``GLM53_STREAM_FOLD`` / ``GLM53_STREAM_SCAN`` / ``GLM53_STREAM_FUSED``):

* ``routing_fold``: the kernels take the router's FULL ``[1, 1, 32, E_total]`` routing matrix plus a per-chip 32 B
  uint16 page holding the chip's column base (``fold_base_tensor``) and scan the window of tiles covering their E
  experts themselves (``kernels/routing_scan.hpp`` ``fold_*``), which removes the per-column selector matmul of
  ``tt/moe.py`` (a 0/1 identity block at HiFi4: reading the columns is bit-identical to multiplying by it).
* ``scan_words``: the word-wise active-expert scan (``scan_active_words``: 8 word loads per zero face row instead of
  16 element loads with index arithmetic) in all four block kernels -- the per-launch prelude before any weight read.
* ``fused`` (``ExpertStreamFused``): gate|up and down in ONE generic op, the down cores placed first next to their
  banks and the gate|up cores around them; the gate|up writers count their expert's tile arrivals into 4-bit slots of
  per-group semaphores on every down core (``fused_semaphores``), the down writer polls the slot of the expert it
  loads next: the down stream of expert e overlaps the gate|up stream of the later experts and one launch replaces
  two (the down op's prelude -- routing read, scan, mailbox -- no longer waits for the gate|up op to end).

Regions-sourced weights (2026-10-27, ``regions_src=True`` on every op, ``RegionsSource`` in place of ``w13`` / ``w2``;
``regions.py``, ``doc/mesh/regions_decode.md``): the ops read the regions prefill path's per-expert DRAM-interleaved
tensors (``gate[e]`` / ``up[e]`` [K, I], ``down[e]`` [I, H]) instead of the stacked bank-sharded copy, which is then
not needed at all (9.41 GiB per chip at the 8x4 shapes). A DRAM-interleaved tile tensor already places tile column
``n`` in bank ``n % banks`` (when ``I/32`` is a multiple of the bank count) at the K stride ``ppb`` tiles, so at the
8x4 (``ppb = 1``) the gate|up stream is byte-for-byte the stacked one from a different base address per expert
(common runtime args); the down block comes k-major (one packet of ``cols`` tiles per K tile) and the down compute
indexes it so. The ``REGIONS_SRC`` kernel variants are preprocessor-gated: the default programs are unchanged, and
the results are bit-identical to the stacked ops (the same products in the same order; the act / out tiles land at
the same global columns through a different (bank, column) map).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn
from models.demos.glm53_flash.custom_ops.expert_stream.plan import fused_core_plan, fused_semaphores
from models.demos.glm53_flash.custom_ops.expert_stream.regions import MAX_RUNTIME_ARGS, plan_regions_stream

KERNEL_DIR = Path(
    os.environ.get("GLM53_STREAM_KERNEL_DIR", "") or Path(__file__).parent / "kernels"
)  # A/B of kernel sets
TILE = 32
BF16_TILE_BYTES = 2048
BFP8_TILE_BYTES = 1088
BFP4_TILE_BYTES = 576
WEIGHT_TILE_BYTES = {ttnn.bfloat8_b: BFP8_TILE_BYTES, ttnn.bfloat4_b: BFP4_TILE_BYTES}
MAX_PACKET_BYTES = 16384  # Blackhole NOC max burst
SWIGLU_LIMIT = 10.0  # the fused clamped_silu_glu SFPU op's compile-time limit (GLM-5.3-Flash swiglu_limit)
MAX_PICKS = 8  # routed experts per token (top-8): bounds the passes and the gather slots of the dispatch ops
TABLE_WIDTH = 64  # uint16 entries per pass-table row: [n_rows, e, t_0 .. t_31, pad]


def _packet_bytes(tiles: int, tile_bytes: int = BFP8_TILE_BYTES, packet_tiles: int | None = None) -> int:
    """The largest NOC packet (a divisor of `tiles` weight tiles) that fits the max burst, or ``packet_tiles``
    tiles (a divisor of ``tiles`` within the burst) when given (the ``GLM53_STREAM_PACKET_TILES`` knob)."""
    if packet_tiles is not None:
        assert tiles % packet_tiles == 0 and packet_tiles * tile_bytes <= MAX_PACKET_BYTES, (tiles, packet_tiles)
        return packet_tiles * tile_bytes
    return max(d for d in range(1, tiles + 1) if tiles % d == 0 and d * tile_bytes <= MAX_PACKET_BYTES) * tile_bytes


# ---------------------------------------------------------------------------------------------------- layouts
def _bank_columns(cols: torch.Tensor, banks: int) -> torch.Tensor:
    """[E, C * banks, k, 32, 32] column tiles -> [rows, banks * 32]: bank b's shard = its C columns, expert-major,
    as one column of tiles."""
    C = cols.shape[1] // banks
    return torch.cat([cols[:, b * C : (b + 1) * C].reshape(-1, TILE) for b in range(banks)], dim=-1)


def gate_up_stream_layout(w13: torch.Tensor, banks: int = 8) -> torch.Tensor:
    """[E, K, 2 I] (``[gate | up]`` columns) -> [rows, banks * 32]: bank b holds the column pairs
    (gate b ppb + q, up b ppb + q), q < ppb = I / 32 / banks, of every expert: expert -> pair -> (gate, up) -> K tile.
    """
    E, K, W = w13.shape
    half = W // TILE // 2
    assert half % banks == 0, f"{half} column pairs do not split over the {banks} banks (I must be a multiple of 256)"
    ppb = half // banks
    cols = w13.reshape(E, K // TILE, TILE, W // TILE, TILE).permute(0, 3, 1, 2, 4)  # [E, n, k, 32, 32]
    order = [c for b in range(banks) for q in range(ppb) for c in (b * ppb + q, half + b * ppb + q)]
    return _bank_columns(cols[:, order], banks)


def down_stream_layout(w2: torch.Tensor, banks: int = 8) -> torch.Tensor:
    """[E, I, H] -> [rows, banks * 32]: bank b holds output tile columns [b C, (b+1) C), expert -> column -> K tile."""
    E, I, H = w2.shape
    assert (H // TILE) % banks == 0, (H, banks)
    cols = w2.reshape(E, I // TILE, TILE, H // TILE, TILE).permute(0, 3, 1, 2, 4)
    return _bank_columns(cols, banks)


def gate_up_rows(E: int, K: int, I: int, banks: int = 8) -> int:
    return E * 2 * (I // TILE // banks) * (K // TILE) * TILE


def down_rows(E: int, I: int, H: int, banks: int = 8) -> int:
    return E * (H // TILE // banks) * (I // TILE) * TILE


def stream_memory_config(rows: int, banks: int = 8) -> ttnn.MemoryConfig:
    """DRAM width-sharded over the banks: bank b holds the [rows, 32] shard b."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (rows, TILE),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def upload_stream_weights(device, layout: torch.Tensor, dtype=ttnn.bfloat8_b, mesh_mapper=None) -> ttnn.Tensor:
    """One chip: [rows, banks * 32] -> the width-sharded DRAM tensor [1, 1, rows, banks * 32]. On a mesh: the
    chips' layouts stacked [D, rows, banks * 32] with ``mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0)`` (each
    chip its own [1, 1, rows, banks * 32] shard)."""
    rows = layout.shape[-2]
    lead = layout.shape[0] if layout.dim() == 3 else 1
    return ttnn.from_torch(
        layout.reshape(lead, 1, rows, -1).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=stream_memory_config(rows),
        mesh_mapper=mesh_mapper,
    )


class RegionsSource:
    """The regions prefill path's per-expert weight lists as the stream / dispatch ops' weight source
    (``regions_src=True``; ``tt/moe.py`` ``_build_dispatch_path`` builds the lists, ``regions.py`` the layout
    math): ``gate[e]`` / ``up[e]`` ``[K, I]`` and ``down[e]`` ``[I, H]`` TILE tensors, DRAM interleaved, one
    buffer each. The kernels take every expert's buffer address as common runtime args (``gate_up_addrs``: gate
    then up, 2 E words; ``down_addrs``: E words; the per-kernel limit is ``MAX_RUNTIME_ARGS``) and walk the
    interleaved pages bank-locally themselves; ``plan`` (``plan_regions_stream``) says whether the shapes allow it
    (``bank_local``) and sizes the packets. ``tensors`` is the io list handed to the generic op (the buffers stay
    in the program's dependency set, ahead of today's io order). On a mesh the lists are mesh tensors with one
    address per expert on every chip (the slices are mesh-uniform allocations); on a program-cache hit the common
    runtime args are rewritten in place (tt_metal ``apply_descriptor_runtime_args``), so every layer's list applies."""

    def __init__(self, gate, up, down, banks: int, w13_dtype=ttnn.bfloat8_b, w2_dtype=ttnn.bfloat8_b):
        assert len(gate) == len(up) == len(down) > 0, (len(gate), len(up), len(down))
        self.E = len(gate)
        K, I = int(gate[0].shape[-2]), int(gate[0].shape[-1])
        I2, H = int(down[0].shape[-2]), int(down[0].shape[-1])
        assert I2 == I, f"down [{I2}, {H}] does not match gate [{K}, {I}]"
        for name, lst, shape, dtype in (
            ("gate", gate, (K, I), w13_dtype),
            ("up", up, (K, I), w13_dtype),
            ("down", down, (I, H), w2_dtype),
        ):
            for e, t in enumerate(lst):
                assert tuple(t.shape)[-2:] == shape, (name, e, tuple(t.shape), shape)
                assert t.layout == ttnn.TILE_LAYOUT, (name, e, t.layout)
                mc = t.memory_config()
                assert mc.buffer_type == ttnn.BufferType.DRAM and not t.is_sharded(), (name, e, mc)
                assert t.dtype == dtype, (name, e, t.dtype, dtype)
        self.gate, self.up, self.down = list(gate), list(up), list(down)
        self.K, self.I, self.H = K, I, H
        self.banks = banks
        self.w13_dtype, self.w2_dtype = w13_dtype, w2_dtype
        self.plan = plan_regions_stream(K, I, H, banks=banks, w13=w13_dtype, w2=w2_dtype)
        if not self.plan.bank_local:
            raise ValueError(f"regions-sourced stream at K={K} I={I} H={H} on {banks} banks: {self.plan.reason}")
        self.gate_up_addrs = [int(t.buffer_address()) for t in self.gate] + [int(t.buffer_address()) for t in self.up]
        self.down_addrs = [int(t.buffer_address()) for t in self.down]
        assert len(self.gate_up_addrs) <= MAX_RUNTIME_ARGS, f"{self.E} experts: too many common runtime args"

    def tensors(self) -> list:
        return self.gate + self.up + self.down

    def check(self, E: int, K: int, I: int, H: int | None = None):
        assert (self.E, self.K, self.I) == (E, K, I), ((self.E, self.K, self.I), (E, K, I))
        assert H is None or self.H == H, (self.H, H)


# ---------------------------------------------------------------------------------------------------- golden
def bfp8_round(w: torch.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()


def bfp4_round(w: torch.Tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()


def expert_stream_golden(
    x: torch.Tensor, routing: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, limit=SWIGLU_LIMIT
):
    """x [M, K] bf16, routing [M, E] (bf16 weights, 0 = not routed), w13 [E, K, 2I], w2 [E, I, H] (bfp8-rounded
    floats) -> (act [E, M, I], y [M, H]) in fp32: act[e] = w[:, e, None] * silu(min(x W_g, limit)) * clamp(x W_u),
    y = sum_e act[e] @ W2[e] (inactive experts' act rows are zero)."""
    M, K = x.shape
    E, _, W = w13.shape
    I = W // 2
    xf = x.float()
    act = torch.zeros(E, M, I)
    y = torch.zeros(M, w2.shape[-1])
    rw = routing.float()
    for e in range(E):
        if rw[:, e].abs().sum() == 0:
            continue
        g = (xf @ w13[e, :, :I]).clamp(max=limit)
        u = (xf @ w13[e, :, I:]).clamp(-limit, limit)
        h = torch.nn.functional.silu(g) * u
        act[e] = rw[:, e : e + 1] * h.bfloat16().float()  # the device packs h to bf16 before the weight product
        y += act[e].bfloat16().float() @ w2[e]
    return act, y


def dispatch_passes(routing: torch.Tensor) -> list[tuple[int, list[int]]]:
    """The dispatch ops' pass enumeration of a [T, E] routing matrix: (expert, token rows) per pass, experts
    ascending, 32 rows per pass."""
    passes = []
    for e in range(routing.shape[1]):
        rows = torch.nonzero(routing[:, e] != 0).flatten().tolist()
        for p in range(0, len(rows), TILE):
            passes.append((e, rows[p : p + TILE]))
    return passes


def expert_dispatch_golden(
    x: torch.Tensor, routing: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, limit=SWIGLU_LIMIT
):
    """As ``expert_stream_golden`` for T tokens plus the per-pass activations the dispatch gate|up op writes:
    (act_pass [P, 32, I] (rows past a pass's count zero), y [T, H], passes)."""
    act, y = expert_stream_golden(x, routing, w13, w2, limit)
    passes = dispatch_passes(routing)
    act_pass = torch.zeros(len(passes), TILE, act.shape[-1])
    for k, (e, rows) in enumerate(passes):
        act_pass[k, : len(rows)] = act[e, rows]
    return act_pass, y, passes


def routing_counts(routing: ttnn.Tensor) -> ttnn.Tensor:
    """[1, 1, T, E] BF16 routing matrix -> [1, 1, 1, E] BF16 per-expert token counts (exact up to T = 256)."""
    return ttnn.sum(ttnn.ne(routing, 0.0), dim=2, keepdim=True)


IDX_PAD = 4096  # index padding value (an expert id no chip owns; exact in BF16)


def ep_routing_matrix(idx: torch.Tensor, wts: torch.Tensor, base: int, E_local: int) -> torch.Tensor:
    """Host: the top-k lists idx [T, K] (global expert ids) / wts [T, K] -> this chip's [T, E_local] routing
    matrix (zero where the expert is not the chip's)."""
    T = idx.shape[0]
    r = torch.zeros(T, E_local)
    for t in range(T):
        for k in range(idx.shape[1]):
            e = int(idx[t, k]) - base
            if 0 <= e < E_local:
                r[t, e] = float(wts[t, k])
    return r.bfloat16()


def expert_dispatch_golden_ep(x, idx, wts, base, E_local, w13, w2, limit=SWIGLU_LIMIT):
    """As ``expert_dispatch_golden`` with the routing given as top-k lists (EP mode)."""
    return expert_dispatch_golden(x, ep_routing_matrix(idx, wts, base, E_local), w13, w2, limit)


def ep_index_tiles(idx: torch.Tensor, wts: torch.Tensor, device, K: int = MAX_PICKS):
    """Host -> device: idx [T, K] (ints), wts [T, K] -> the [1, 1, T, 32] BF16 TILE index and weight tensors the EP
    ops read (columns >= K padded with IDX_PAD / 0)."""
    T = idx.shape[0]
    it = torch.full((1, 1, T, TILE), float(IDX_PAD))
    it[0, 0, :, :K] = idx.float()
    wt = torch.zeros(1, 1, T, TILE)
    wt[0, 0, :, :K] = wts.float()
    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return mk(it), mk(wt)


def rm_page_bytes(t: ttnn.Tensor) -> int:
    """The aligned page size of a ROW_MAJOR interleaved L1 tensor (one page per row; 16-byte L1 alignment). The EP
    row-major routing path keeps the dispatched rows and the metadata in L1 (the all_to_all outputs' memory config),
    so a DRAM tensor (64-byte pages) is rejected rather than mis-addressed."""
    assert t.layout == ttnn.ROW_MAJOR_LAYOUT, t.layout
    assert t.memory_config().buffer_type == ttnn.BufferType.L1, "EP row-major routing tensors must live in L1"
    return -(-t.shape[-1] * 2 // 16) * 16


def ep_rm_inputs(x: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor, device):
    """Host -> device, the mesh path's per-chip inputs as all_to_all_dispatch delivers them: the token rows with the
    routing weights as a K-wide BF16 tail ([1, 1, T, H + K] ROW_MAJOR L1) and the ids [1, T, 1, K] UINT16 ROW_MAJOR
    L1 (``ExpertDispatch(ep=True, ep_rm=True)``)."""
    T, K = idx.shape
    xw = torch.cat([x.float(), wts.float()], dim=-1).bfloat16().reshape(1, 1, T, -1)
    xw_dev = ttnn.from_torch(
        xw, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    idx_dev = ttnn.from_torch(
        idx.to(torch.int32).reshape(1, T, 1, K),
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return xw_dev, idx_dev


def ep_base_tensor(base: int, device, mesh_mapper=None):
    """The per-chip expert base id as a [1, 1, 1, 16] UINT16 ROW_MAJOR page (on a mesh: [D, 1, 1, 16] sharded)."""
    t = torch.zeros(1, 1, 1, 16, dtype=torch.int32) if isinstance(base, int) else base
    if isinstance(base, int):
        t[0, 0, 0, 0] = base
    return ttnn.from_torch(t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=mesh_mapper)


def fold_base_host(cols: int, E_local: int) -> torch.Tensor:
    """Host: the per-column expert base of the selector fold as a ``[1, cols, 1, 16]`` int32 tensor (column c's
    page holds ``c * E_local`` at element 0; sharded over the mesh columns by the caller: ``GLMMesh.upload(...,
    cols=1, layout=ROW_MAJOR)``)."""
    t = torch.zeros(1, cols, 1, 16, dtype=torch.int32)
    for c in range(cols):
        t[0, c, 0, 0] = c * E_local
    return t


def fold_base_tensor(col_base: int, device):
    """One chip: the ``[1, 1, 1, 16]`` uint16 ROW_MAJOR base page of the selector fold (``routing_fold``)."""
    return ep_base_tensor(col_base, device)


def fold_window(col_base: int, E_local: int) -> tuple[int, int, int]:
    """(first tile, column offset inside it, tiles read) of the fold window: the kernels' ``fold_first_tile`` /
    ``fold_col_off`` / ``fold_tiles`` (host mirror for the tests and the planner)."""
    return col_base >> 5, col_base & 31, ((col_base & 31) + E_local + 31) >> 5


def fold_r_tiles(E_local: int) -> int:
    """The worst-case fold window in tiles (the kernels' compile-time ``R_TILES`` and the routing CB's pages)."""
    return (E_local + 62) // 32


def mesh_rows(device) -> int:
    """Rows of the mesh a device handle spans (1 on a single chip). The EP row-major dispatch delivers
    ``rows * T_users`` rows per chip in mesh-row order, which the reduce-scatter combine's output chunks split back."""
    shape = getattr(device, "shape", None)
    return int(list(shape)[0]) if shape is not None else 1


def ep_rs_chunk(device, t_valid: int | None) -> int | None:
    """Tokens per output chunk of the EP row-major down op under ``GLM53_EP_RS_REMAP=1`` (default 0: ``None``).
    The reduce-scatter EP combine (``tt/moe.py`` ``TtEPExperts.combine_rows``, ``GLM53_EP_COMBINE=rs``) hands mesh
    row r chunk r of ``out`` on the tile-row dim, which needs row r's tokens in a 32-row-aligned chunk. At
    ``T_users = t_valid / rows`` tokens per mesh row (``t_valid = rows * T_users``: the all_to_all_dispatch delivers
    every row's tokens in row order) with ``T_users % 32 != 0`` (4 users/row = 32 users on the 8x4) the plain
    ``[1, 1, t_valid, H]`` layout (dispatched row i at tile i // 32, row i % 32) interleaves the rows' tokens inside
    one tile; with the remap the down writer lays ``out`` out as ``[1, 1, rows * 32, H]``: dispatched row i at chunk
    i // T_users (= its mesh row), row i % T_users, rows T_users .. 31 of every chunk exact zeros
    (``kernels/writer_down_dispatch.cpp``). Returns ``T_users`` then, else ``None``. The op / a2a combines read
    ``out`` row for row and must not run with the remap (their shape checks fail loudly: ``Td != rows * T``)."""
    if t_valid is None or os.environ.get("GLM53_EP_RS_REMAP", "0") != "1":
        return None
    rows = mesh_rows(device)
    if t_valid % rows:
        return None
    t_users = t_valid // rows
    return t_users if t_users % TILE else None


# ---------------------------------------------------------------------------------------------------- placement
class _BankStreamOp:
    """Worker placement shared by the streaming ops (after gpt_oss fused_decode/stream.py): ``readers`` cores per
    DRAM bank, from the bank's optimal NOC0 worker core outwards (same row first, then the rows above / below), each
    with a NOC virtual channel distinct from earlier cores on its row; coalesced into core ranges."""

    def __init__(self, device, readers: int, exclude: set[tuple[int, int]] | None = None):
        """``exclude``: logical (x, y) cores another kernel set of the same program occupies (the fused op places
        its down cores first, then the gate|up cores around them); ``None`` = today's placement exactly."""
        self.device = device
        self.banks = device.dram_grid_size().x
        self.readers = readers
        bank_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        assert len(bank_workers) == self.banks, (len(bank_workers), self.banks)
        grid = device.compute_with_storage_grid_size()
        exclude = set(exclude or ())
        used = {(c.x, c.y) for c in bank_workers} | exclude
        placed = []
        for bank, core in enumerate(bank_workers):
            cores = [] if (core.x, core.y) in exclude else [core]
            # candidates by distance: same row first, then neighbouring rows
            cands = sorted(
                ((abs(x - core.x) + 4 * abs(y - core.y), x, y) for y in range(grid.y) for x in range(grid.x)),
            )
            for _, x, y in cands:
                if len(cores) >= readers:
                    break
                if (x, y) not in used:
                    cores.append(ttnn.CoreCoord(x, y))
                    used.add((x, y))
            assert len(cores) == readers, f"no room for {readers} readers for bank {bank}"
            placed += [(c, bank, r) for r, c in enumerate(cores)]
        vcs = []
        for i, (core, bank, r) in enumerate(placed):
            vc = (bank + r) & 0x3
            for j in range(i):
                if placed[j][0].y == core.y and vcs[j] == vc:
                    vc = (vc + 1) & 0x3
            vcs.append(vc)
        self.assignments = [(c, b, r, v) for (c, b, r), v in zip(placed, vcs)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.assignments[0][0], self.assignments[0][0])])
        for c, *_ in self.assignments[1:]:
            self.cores = self.cores.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
        self.core_set = {(c.x, c.y) for c, *_ in self.assignments}  # logical coords (the fused op's exclusion set)

    def noc_xy(self) -> list[int]:
        """The cores' NOC coordinates packed ``x << 16 | y`` (``worker_core_from_logical_core``: the translated
        coordinates a kernel hands ``get_noc_addr``), in assignment order."""
        out = []
        for c, *_ in self.assignments:
            n = self.device.worker_core_from_logical_core(c)
            out.append((int(n.x) << 16) | int(n.y))
        return out

    def _cb(self, index: int, dtype, page: int, pages: int) -> ttnn.CBDescriptor:
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )

    def _kernels(
        self,
        name: str,
        reader_ct,
        reader_rt,
        writer_ct,
        writer_rt,
        compute_ct,
        fidelity=ttnn.MathFidelity.HiFi2,
        defines=(),
        reader_common=None,
        compute_defines=(),
    ) -> list:
        """The three kernel descriptors of one streaming program on this placement's cores. ``defines``: the E1
        preprocessor variants of the data-movement kernels (``("ROUTING_FOLD", "1")`` ...); empty = today's kernels
        (a kernel descriptor without defines is today's descriptor). ``reader_common``: the reader's common runtime
        args (the regions-sourced per-expert addresses); ``compute_defines``: the compute kernel's variants
        (``REGIONS_SRC`` on the down computes); both default to today's descriptors."""
        dm = ttnn.DataMovementConfigDescriptor
        extra = {"defines": [tuple(d) for d in defines]} if defines else {}
        reader_extra = dict(extra)
        if reader_common:
            reader_extra["common_runtime_args"] = [int(a) for a in reader_common]
        compute_extra = {"defines": [tuple(d) for d in compute_defines]} if compute_defines else {}
        return [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"reader_{name}.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                **reader_extra,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"writer_{name}.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                **extra,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"compute_{name}.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False
                ),
                **compute_extra,
            ),
        ]

    def _program(
        self,
        name: str,
        cbs,
        reader_ct,
        reader_rt,
        writer_ct,
        writer_rt,
        compute_ct,
        fidelity=ttnn.MathFidelity.HiFi2,
        defines=(),
        reader_common=None,
        compute_defines=(),
    ) -> ttnn.ProgramDescriptor:
        kernels = self._kernels(
            name,
            reader_ct,
            reader_rt,
            writer_ct,
            writer_rt,
            compute_ct,
            fidelity,
            defines,
            reader_common=reader_common,
            compute_defines=compute_defines,
        )
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


REGIONS_DEFINE = ("REGIONS_SRC", "1")


def _regions_source(w, regions_src: bool, E: int, K: int, I: int, H: int | None = None):
    """The ``RegionsSource`` behind a weight argument of a ``regions_src`` op (None on a stacked op); the flag and
    the argument must agree (a stacked tensor on a regions-sourced program would be walked as interleaved pages)."""
    is_src = isinstance(w, RegionsSource)
    assert is_src == bool(regions_src), (
        "regions_src=True takes a RegionsSource in place of the stacked weights"
        if regions_src
        else "a RegionsSource needs regions_src=True"
    )
    if is_src:
        w.check(E, K, I, H)
        return w
    return None


def _down_readers(C: int, seg_tiles: int) -> int:
    """Readers per bank for the down ops: the fewest that keep cols <= 8 (fp32 DST) and an expert's block of
    cols * seg_tiles tiles <= 128 (three 139 KB slots): 2 at I = 256, 4 at 1024, 8 at 2048."""
    for r in (1, 2, 4, 8, 16):
        if C % r == 0 and C // r <= 8 and (C // r) * seg_tiles <= 128:
            return r
    raise ValueError(f"no reader split for {C} columns x {seg_tiles} K tiles")


# ---------------------------------------------------------------------------------------------------- block ops
def _e1_defines(routing_fold: bool, scan_words: bool, fused: str | None = None) -> list[tuple[str, str]]:
    """The preprocessor defines of the E1 kernel variants (empty = today's kernels)."""
    d = []
    if routing_fold:
        d.append(("ROUTING_FOLD", "1"))
    if scan_words:
        d.append(("SCAN_WORDS", "1"))
    if fused:
        d.append((fused, "1"))
    return d


class ExpertGateUpStream(_BankStreamOp):
    """(x, routing, w13, act) -> act. x [1, 1, 32, K] BF16 TILE (M <= 32 token rows, zero-padded); routing
    [1, 1, 32, E] BF16 TILE; w13: ``upload_stream_weights(gate_up_stream_layout(...))``; act [1, E, 32, I] BF16
    TILE (written at the active experts' tiles). ``groups`` = cores per bank = how many experts stream in
    parallel; the activation (K / 32 tiles, 256 KB at K = 4096) stays resident on every core.
    ``routing_fold``: routing = the router's full [1, 1, 32, E_total] matrix and ``__call__`` takes the chip's
    ``base`` page (``fold_base_tensor``); ``scan_words``: the word-wise scan; both default off (today's program).
    ``regions_src``: ``w13`` is a ``RegionsSource`` (the regions lists; the REGIONS_SRC kernels, the per-expert
    addresses as common runtime args, single-tile packets at ppb > 1)."""

    def __init__(
        self,
        device,
        E: int,
        K: int,
        I: int,
        groups: int = 12,
        w_dtype=ttnn.bfloat8_b,
        routing_fold: bool = False,
        scan_words: bool = False,
        exclude: set[tuple[int, int]] | None = None,
        regions_src: bool = False,
    ):
        super().__init__(device, groups, exclude=exclude)
        self.E, self.K, self.I = E, K, I
        self.kt = K // TILE
        self.i_tiles = I // TILE
        assert (
            self.i_tiles % self.banks == 0
        ), f"I / 32 = {self.i_tiles} tile columns do not split over {self.banks} banks"
        self.ppb = self.i_tiles // self.banks  # column pairs per bank
        self.groups = groups
        self.w_dtype, self.w_tile = w_dtype, WEIGHT_TILE_BYTES[w_dtype]  # bfp8 or bfp4 gate|up weights
        self.col_bytes = self.kt * self.w_tile
        self.expert_stride = 2 * self.ppb * self.col_bytes
        self.regions_src = bool(regions_src)
        if self.regions_src:
            # the regions lists' K stride is ppb tiles: whole-column packets at ppb = 1 (today's size), one tile else
            self.regions_plan = plan_regions_stream(K, I, None, banks=self.banks, w13=w_dtype)
            self.page_bytes = self.regions_plan.gate_packet_bytes
        else:
            self.page_bytes = _packet_bytes(self.kt, self.w_tile)
        self.routing_fold, self.scan_words = bool(routing_fold), bool(scan_words)

    def descriptors(self, x, routing, w13, act, base=None, cb=(0, 1, 2, 3, 4, 5, 16), fused_rt=None):
        """(cbs, kernels) of the gate|up program. ``cb``: the CB indices (x, w, r0, r1, d, h, out); ``fused_rt``:
        the FUSED_SIGNAL runtime tail of the writer (``ExpertStreamFused``). ``w13``: the stacked tensor, or the
        ``RegionsSource`` under ``regions_src``."""
        cb_x, cb_w, cb_r0, cb_r1, cb_d, cb_h, cb_out = cb
        assert (base is not None) == self.routing_fold, "routing_fold takes the chip's base page"
        src = _regions_source(w13, self.regions_src, self.E, self.K, self.I)
        w_addr = 0 if src is not None else w13.buffer_address()
        e_tiles = -(-self.E // TILE)
        r_pages = fold_r_tiles(self.E) + 1 if self.routing_fold else e_tiles  # + the base page
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, BF16_TILE_BYTES, self.kt),
            self._cb(cb_w, self.w_dtype, self.w_tile, 3 * self.kt),
            self._cb(cb_r0, ttnn.bfloat16, BF16_TILE_BYTES, r_pages),
            self._cb(cb_r1, ttnn.bfloat16, BF16_TILE_BYTES, r_pages),
            self._cb(cb_d, ttnn.bfloat16, BF16_TILE_BYTES, 2),
            self._cb(cb_h, ttnn.bfloat16, BF16_TILE_BYTES, 2),
            self._cb(cb_out, ttnn.bfloat16, BF16_TILE_BYTES, 2),
        ]
        base_rt = [base.buffer_address()] if self.routing_fold else []
        tail = list(fused_rt or [])
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [w_addr, routing.buffer_address(), bank, vc, r] + base_rt
            writer_rt[core.x][core.y] = (
                [x.buffer_address(), routing.buffer_address(), act.buffer_address(), bank, r] + base_rt + tail
            )
        reader_ct = [
            cb_w,
            cb_r0,
            self.kt,
            self.E,
            TILE,
            self.groups,
            self.expert_stride,
            self.w_tile,
            self.page_bytes,
            self.ppb,
        ]
        reader_ct += ttnn.TensorAccessorArgs(routing).get_compile_time_args()
        writer_ct = [cb_x, cb_r1, cb_d, cb_out, self.kt, self.E, TILE, self.groups, self.i_tiles, self.ppb]
        for t in (x, routing, act):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        if self.routing_fold:  # the base page accessor, after today's blocks (the kernels index it last)
            reader_ct += ttnn.TensorAccessorArgs(base).get_compile_time_args()
            writer_ct += ttnn.TensorAccessorArgs(base).get_compile_time_args()
        compute_ct = [cb_x, cb_w, cb_d, cb_h, cb_out, self.kt, self.ppb]
        defines = _e1_defines(self.routing_fold, self.scan_words, "FUSED_SIGNAL" if fused_rt is not None else None)
        if src is not None:
            defines.append(REGIONS_DEFINE)
        kernels = self._kernels(
            "gate_up",
            reader_ct,
            reader_rt,
            writer_ct,
            writer_rt,
            compute_ct,
            defines=defines,
            reader_common=src.gate_up_addrs if src is not None else None,
        )
        return cbs, kernels

    def io(self, x, routing, w13, act, base=None) -> list:
        """The generic op's io list in today's order (the weights first: the stacked tensor, or the regions lists)."""
        weights = w13.gate + w13.up if isinstance(w13, RegionsSource) else [w13]
        return weights + [x, routing, act] + ([base] if base is not None else [])

    def __call__(self, x, routing, w13, act, base=None):
        cbs, kernels = self.descriptors(x, routing, w13, act, base=base)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op(self.io(x, routing, w13, act, base), program)
        return act


class ExpertDownStream(_BankStreamOp):
    """(act, routing, w2, out) -> out. act [1, E, 32, I] BF16 TILE (the gate|up stream's output), routing as above,
    w2: ``upload_stream_weights(down_stream_layout(...))``, out [1, 1, 32, H] BF16 TILE DRAM (the expert sum).
    ``readers`` cores per bank split the bank's C = H / 32 / banks output columns (cols = C / readers <= 8;
    None = the fewest readers whose expert block fits, ``_down_readers``). ``routing_fold`` / ``scan_words`` as the
    gate|up op. ``regions_src``: ``w2`` is a ``RegionsSource`` (one packet of ``cols`` tiles per K tile, k-major)."""

    def __init__(
        self,
        device,
        E: int,
        I: int,
        H: int,
        readers: int | None = None,
        routing_fold: bool = False,
        scan_words: bool = False,
        exclude: set[tuple[int, int]] | None = None,
        regions_src: bool = False,
    ):
        banks = device.dram_grid_size().x
        self.seg_tiles = I // TILE
        self.C = H // TILE // banks
        if readers is None:
            readers = _down_readers(self.C, self.seg_tiles)
        super().__init__(device, readers, exclude=exclude)
        self.E, self.I, self.H = E, I, H
        assert self.C % readers == 0, f"{self.C} columns per bank do not split over {readers} readers"
        self.cols = self.C // readers
        assert self.cols <= 8, f"{self.cols} output columns per core exceed the 8 fp32 DST tiles"
        self.seg_bytes = self.seg_tiles * BFP8_TILE_BYTES
        self.block_tiles = self.cols * self.seg_tiles
        self.regions_src = bool(regions_src)
        if self.regions_src:
            self.regions_plan = plan_regions_stream(H, I, H, banks=self.banks, down_readers=readers)
            assert self.regions_plan.cols == self.cols, (self.regions_plan.cols, self.cols)
            self.page_bytes = self.regions_plan.down_packet_bytes  # one K tile of the core's columns
        else:
            self.page_bytes = _packet_bytes(self.seg_tiles)
        self.bank_expert_stride = self.C * self.seg_bytes
        self.routing_fold, self.scan_words = bool(routing_fold), bool(scan_words)

    def descriptors(self, act, routing, w2, out, base=None, cb=(0, 1, 2, 3, 16), fused_rt=None):
        """(cbs, kernels) of the down program. ``cb``: (a, w, r0, r1, out); ``fused_rt``: the FUSED_WAIT runtime
        tail of the writer (``ExpertStreamFused``). ``w2``: the stacked tensor, or the ``RegionsSource``."""
        cb_a, cb_w, cb_r0, cb_r1, cb_out = cb
        assert (base is not None) == self.routing_fold, "routing_fold takes the chip's base page"
        src = _regions_source(w2, self.regions_src, self.E, self.H, self.I, self.H)
        w_addr = 0 if src is not None else w2.buffer_address()
        e_tiles = -(-self.E // TILE)
        r_pages = fold_r_tiles(self.E) + 1 if self.routing_fold else e_tiles
        cbs = [
            self._cb(cb_a, ttnn.bfloat16, BF16_TILE_BYTES, 2 * self.seg_tiles),
            self._cb(cb_w, ttnn.bfloat8_b, BFP8_TILE_BYTES, 3 * self.block_tiles),
            self._cb(cb_r0, ttnn.bfloat16, BF16_TILE_BYTES, r_pages),
            self._cb(cb_r1, ttnn.bfloat16, BF16_TILE_BYTES, r_pages),
            self._cb(cb_out, ttnn.bfloat16, BF16_TILE_BYTES, self.cols),
        ]
        base_rt = [base.buffer_address()] if self.routing_fold else []
        tail = list(fused_rt or [])
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [
                w_addr,
                routing.buffer_address(),
                bank,
                vc,
                r * self.cols,
            ] + base_rt
            writer_rt[core.x][core.y] = (
                [
                    routing.buffer_address(),
                    act.buffer_address(),
                    out.buffer_address(),
                    bank * self.C + r * self.cols,
                ]
                + base_rt
                + tail
            )
        reader_ct = [
            cb_w,
            cb_r0,
            self.seg_tiles,
            self.E,
            TILE,
            self.cols,
            self.bank_expert_stride,
            BFP8_TILE_BYTES,
            self.page_bytes,
        ]
        reader_ct += ttnn.TensorAccessorArgs(routing).get_compile_time_args()
        writer_ct = [cb_r1, cb_a, cb_out, self.seg_tiles, self.E, TILE, self.cols]
        for t in (routing, act, out):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        if self.routing_fold:
            reader_ct += ttnn.TensorAccessorArgs(base).get_compile_time_args()
            writer_ct += ttnn.TensorAccessorArgs(base).get_compile_time_args()
        if src is not None:  # the output column map's constants, after every accessor (writer_down.cpp CT_TAIL)
            writer_ct += [self.H // TILE, self.banks]
        compute_ct = [cb_a, cb_w, cb_out, self.seg_tiles, self.cols]
        defines = _e1_defines(self.routing_fold, self.scan_words, "FUSED_WAIT" if fused_rt is not None else None)
        if src is not None:
            defines.append(REGIONS_DEFINE)
        kernels = self._kernels(
            "down",
            reader_ct,
            reader_rt,
            writer_ct,
            writer_rt,
            compute_ct,
            defines=defines,
            reader_common=src.down_addrs if src is not None else None,
            compute_defines=[REGIONS_DEFINE] if src is not None else (),
        )
        return cbs, kernels

    def io(self, act, routing, w2, out, base=None) -> list:
        """The generic op's io list in today's order (the weights first)."""
        weights = list(w2.down) if isinstance(w2, RegionsSource) else [w2]
        return weights + [act, routing, out] + ([base] if base is not None else [])

    def __call__(self, act, routing, w2, out, base=None):
        cbs, kernels = self.descriptors(act, routing, w2, out, base=base)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op(self.io(act, routing, w2, out, base), program)
        return out


class ExpertStreamFused:
    """The chip's routed-expert block as ONE generic op (E1 (b)): the gate|up kernels on ``groups`` cores per bank
    and the down kernels on ``down_readers`` cores per bank (placed first, nearest their banks; disjoint sets),
    synchronised per expert through the arrival-counter semaphores (``plan.fused_semaphores``: group g's k-th expert
    = 4-bit slot k % 7 of semaphore ``g * sems_per_group + k // 7``, counted to ``banks`` by the gate|up writers,
    polled by the down writer: kernels/writer_gate_up.cpp FUSED_SIGNAL, kernels/writer_down.cpp FUSED_WAIT). The
    gate|up and down core counts come from ``plan.fused_core_plan`` unless given (12 + 2 per bank on the 13x10
    grid, 6 + 2 on 8x8). Same numerics as ``ExpertStream`` (the same compute kernels, the same per-expert math)."""

    SEM_BASE = 0

    def __init__(
        self,
        device,
        E: int,
        K: int,
        I: int,
        H: int,
        groups: int | None = None,
        down_readers: int | None = None,
        w13_dtype=ttnn.bfloat8_b,
        routing_fold: bool = False,
        scan_words: bool = False,
        regions_src: bool = False,
    ):
        grid = device.compute_with_storage_grid_size()
        banks = device.dram_grid_size().x
        plan = fused_core_plan((grid.x, grid.y), E, I, H, banks=banks, groups=groups, down_readers=down_readers)
        self.groups, self.down_readers = plan["groups"], plan["down_readers"]
        self.sems_per_group, self.n_sems = fused_semaphores(self.groups, E)
        self.down = ExpertDownStream(
            device,
            E,
            I,
            H,
            self.down_readers,
            routing_fold=routing_fold,
            scan_words=scan_words,
            regions_src=regions_src,
        )
        self.gate_up = ExpertGateUpStream(
            device,
            E,
            K,
            I,
            self.groups,
            w_dtype=w13_dtype,
            routing_fold=routing_fold,
            scan_words=scan_words,
            exclude=self.down.core_set,
            regions_src=regions_src,
        )
        assert not (self.gate_up.core_set & self.down.core_set), "fused placement: the two kernel sets overlap"
        self.E, self.K, self.I, self.H = E, K, I, H
        self.banks = banks
        self.regions_src = bool(regions_src)
        self.down_xy = self.down.noc_xy()

    alloc = None  # set below (shared with ExpertStream)

    def forward(self, x, routing, w13, w2, act, out, base=None):
        gu_cbs, gu_kernels = self.gate_up.descriptors(
            x,
            routing,
            w13,
            act,
            base=base,
            cb=(0, 1, 2, 3, 4, 5, 16),
            fused_rt=[len(self.down_xy), self.SEM_BASE, self.sems_per_group] + self.down_xy,
        )
        dn_cbs, dn_kernels = self.down.descriptors(
            act,
            routing,
            w2,
            out,
            base=base,
            cb=(8, 9, 10, 11, 17),
            fused_rt=[self.groups, self.SEM_BASE, self.sems_per_group, self.banks],
        )
        sems = [
            ttnn.SemaphoreDescriptor(self.SEM_BASE + i, ttnn.CoreType.WORKER, self.down.cores, 0)
            for i in range(self.n_sems)
        ]
        program = ttnn.ProgramDescriptor(kernels=gu_kernels + dn_kernels, semaphores=sems, cbs=gu_cbs + dn_cbs)
        if isinstance(w13, RegionsSource):
            weights = w13.tensors()  # w2 is the same source
        else:
            weights = [w13, w2]
        ttnn.generic_op(weights + [x, routing, act, out] + ([base] if base is not None else []), program)
        return out

    __call__ = forward
    golden = staticmethod(expert_stream_golden)


class ExpertStream:
    """The chip's routed-expert block as the two streaming ops: ``forward(x, routing, act, out)``. ``fused`` runs
    ``ExpertStreamFused`` (one program) instead; ``routing_fold`` / ``scan_words`` select the E1 kernel variants
    (``forward`` then takes the chip's ``base`` page); ``regions_src``: ``forward`` takes the ``RegionsSource`` as
    both ``w13`` and ``w2`` (the regions lists serve the block; no stacked copy)."""

    def __init__(
        self,
        device,
        E: int,
        K: int,
        I: int,
        H: int,
        groups: int = 12,
        down_readers: int | None = None,
        w13_dtype=ttnn.bfloat8_b,
        routing_fold: bool = False,
        scan_words: bool = False,
        fused: bool = False,
        fused_groups: int | None = None,
        fused_down_readers: int | None = None,
        regions_src: bool = False,
    ):
        self.fused = bool(fused)
        self.routing_fold = bool(routing_fold)
        self.regions_src = bool(regions_src)
        if self.fused:
            self.fused_op = ExpertStreamFused(
                device,
                E,
                K,
                I,
                H,
                groups=fused_groups,
                down_readers=fused_down_readers,
                w13_dtype=w13_dtype,
                routing_fold=routing_fold,
                scan_words=scan_words,
                regions_src=regions_src,
            )
            self.gate_up, self.down = self.fused_op.gate_up, self.fused_op.down
        else:
            self.fused_op = None
            self.gate_up = ExpertGateUpStream(
                device,
                E,
                K,
                I,
                groups,
                w_dtype=w13_dtype,
                routing_fold=routing_fold,
                scan_words=scan_words,
                regions_src=regions_src,
            )
            self.down = ExpertDownStream(
                device,
                E,
                I,
                H,
                down_readers,
                routing_fold=routing_fold,
                scan_words=scan_words,
                regions_src=regions_src,
            )
        self.E, self.K, self.I, self.H = E, K, I, H

    def alloc(self, device, act_l1: bool = False):
        """The act and out buffers for one call (allocate inside a trace or keep persistent). act_l1 keeps the
        E x 32 x I activations in L1 interleaved (1.2 MB at the 8x4 shapes, ~2 % off the down op)."""
        act = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.E, TILE, self.I]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG if act_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, self.H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        return act, out

    def forward(self, x, routing, w13, w2, act, out, base=None):
        if self.fused_op is not None:
            return self.fused_op.forward(x, routing, w13, w2, act, out, base=base)
        self.gate_up(x, routing, w13, act, base=base)
        self.down(act, routing, w2, out, base=base)
        return out

    __call__ = forward
    golden = staticmethod(expert_stream_golden)  # (x [M,K], routing [M,E], w13 [E,K,2I], w2 [E,I,H]) -> (act, y)


ExpertStreamFused.alloc = ExpertStream.alloc


# ---------------------------------------------------------------------------------------------------- dispatch ops
def max_passes(T: int, E: int) -> int:
    """Upper bound of the pass count: sum_e ceil(n_e / 32) with sum_e n_e <= MAX_PICKS * T."""
    return MAX_PICKS * T // TILE + E


class ExpertGateUpDispatch(_BankStreamOp):
    """(x_rm, routing, counts, w13, act, table) -> act, table. x_rm [1, 1, T, K] BF16 ROW_MAJOR; routing
    [1, 1, T, E] BF16 TILE; counts = ``routing_counts(routing)``; w13 as the block op; act [1, P, 32, I] BF16 TILE
    (the per-pass activations, P = ``max_passes``); table [1, 1, P, 64] UINT16 ROW_MAJOR (the pass rows)."""

    def __init__(
        self,
        device,
        E: int,
        K: int,
        I: int,
        T: int,
        groups: int = 12,
        ep: bool = False,
        picks: int = MAX_PICKS,
        w_dtype=ttnn.bfloat8_b,
        units: bool = False,
        ep_rm: bool = False,
        unit_pairs: int = 1,
        prefetch_pairs: int = 1,
        packet_tiles: int | None = None,
        gather_rotate: bool = False,
        regions_src: bool = False,
    ):
        super().__init__(device, groups)
        assert T % TILE == 0 and T <= 256, T  # counts are exact BF16 integers up to 256
        assert groups <= 32, groups
        self.E, self.K, self.I, self.T = E, K, I, T
        # regions_src: the regions lists' per-expert gate / up tensors as the weight source (REGIONS_SRC kernels,
        # common runtime args); the pair-unit kernels keep the stacked layout only
        self.regions_src = bool(regions_src)
        assert not (self.regions_src and units), "the units kernels take the stacked layout only"
        # decode ledger item (d) knobs (custom_ops/expert_stream/plan.py; the defaults are today's program):
        # unit_pairs = pairs per work unit of the units kernels (ppb / unit_pairs cores per bank per expert),
        # prefetch_pairs = pairs in flight ahead of the compute (the weight ring has prefetch_pairs + 1 slots),
        # packet_tiles = weight tiles per NOC packet (None = the largest divisor of K/32 within 16 KB),
        # gather_rotate = rotate every core's row-gather / activation-read start (H1, the L1 convoy)
        self.unit_pairs, self.prefetch_pairs, self.gather_rotate = (
            int(unit_pairs),
            int(prefetch_pairs),
            bool(gather_rotate),
        )
        assert self.prefetch_pairs >= 1, prefetch_pairs
        self.ep, self.picks = ep, picks  # EP: routing = top-k index / weight tiles + the chip's expert base
        # ep_rm (the mesh path): the ids as the dispatch metadata rows [t_valid, K] UINT16 and the weights as the
        # K-wide BF16 tail of every token row ([x | w] ROW_MAJOR, both in L1); t_valid <= T rows are scanned
        self.ep_rm = ep_rm
        assert not (ep_rm and not ep), "ep_rm is an EP mode"
        assert not (ep_rm and units), "the units kernels take the tile-layout routing only"
        # units: the (expert, column pair) work units (`*_gate_up_dispatch_units.cpp`) instead of whole experts per
        # core: 11-12 % faster with many active full-width experts (36 per chip), 2x SLOWER with few (the 8x4's 9)
        self.units = units
        self.kt = K // TILE
        self.i_tiles = I // TILE
        assert (
            self.i_tiles % self.banks == 0
        ), f"I / 32 = {self.i_tiles} tile columns do not split over {self.banks} banks"
        self.ppb = self.i_tiles // self.banks
        self.groups = groups
        self.w_dtype, self.w_tile = w_dtype, WEIGHT_TILE_BYTES[w_dtype]  # bfp8 or bfp4 gate|up weights
        self.col_bytes = self.kt * self.w_tile
        self.expert_stride = 2 * self.ppb * self.col_bytes
        if self.regions_src:
            self.regions_plan = plan_regions_stream(
                K, I, None, banks=self.banks, w13=w_dtype, packet_tiles=packet_tiles
            )
            assert self.regions_plan.bank_local, self.regions_plan.reason
            self.page_bytes = self.regions_plan.gate_packet_bytes
        else:
            self.page_bytes = _packet_bytes(self.kt, self.w_tile, packet_tiles)
        assert (
            self.ppb % self.unit_pairs == 0
        ), f"{self.unit_pairs} pairs per unit do not divide {self.ppb} pairs per bank"
        self.row_bytes = K * 2
        self.P = max_passes(T, E)

    def __call__(self, x_rm, routing, counts, w13, act, table, base=None, t_valid=None):
        """Non-EP: routing [1,1,T,E] BF16 TILE, counts [1,1,1,E]. EP: routing = the weight tiles [1,1,T,32],
        counts = the index tiles [1,1,T,32] (``ep_index_tiles``), base = ``ep_base_tensor``. EP row-major: x_rm =
        the [x | w] rows (L1), counts = the id rows [1, t_valid, 1, K] UINT16 (L1), routing unused (pass x_rm),
        t_valid = the valid row count (<= T)."""
        cb_xrm, cb_x, cb_w, cb_r, cb_c0, cb_c1, cb_d, cb_h, cb_t, cb_out = 0, 1, 2, 3, 4, 5, 6, 7, 8, 16
        src = _regions_source(w13, self.regions_src, self.E, self.K, self.I)
        w_addr = 0 if src is not None else w13.buffer_address()
        e_tiles = -(-self.E // TILE)
        rt = self.T // TILE
        c_tiles = rt + 1 if self.ep else e_tiles  # EP: the index tiles + the expert base page
        r_tiles = rt if self.ep else rt * e_tiles  # EP: the weight tiles
        cbs = [
            self._cb(cb_xrm, ttnn.bfloat16, BF16_TILE_BYTES, self.kt),  # 32 row-major rows of K: kt tile-sized pages
            self._cb(cb_x, ttnn.bfloat16, BF16_TILE_BYTES, self.kt),
            self._cb(cb_w, self.w_dtype, self.w_tile, 2 * self.kt * (self.prefetch_pairs + 1)),  # (gate, up) pair slots
            self._cb(cb_r, ttnn.bfloat16, BF16_TILE_BYTES, r_tiles),
            self._cb(cb_c0, ttnn.bfloat16, BF16_TILE_BYTES, c_tiles),
            self._cb(cb_c1, ttnn.bfloat16, BF16_TILE_BYTES, c_tiles),
            self._cb(cb_d, ttnn.bfloat16, BF16_TILE_BYTES, 2),
            self._cb(cb_h, ttnn.bfloat16, BF16_TILE_BYTES, 2),
            self._cb(cb_t, ttnn.uint16, TABLE_WIDTH * 2, 2),
            self._cb(cb_out, ttnn.bfloat16, BF16_TILE_BYTES, 2),
        ]
        assert (base is not None) == self.ep, "EP mode takes the expert base tensor"
        extra_rt = [base.buffer_address()] if self.ep else []
        writer_extra = list(extra_rt)
        if self.ep_rm:
            assert t_valid is not None and 0 < t_valid <= self.T, (t_valid, self.T)
            assert tuple(counts.shape)[1] == t_valid and tuple(counts.shape)[-1] == self.picks, counts.shape
            assert x_rm.shape[-1] == self.K + self.picks, x_rm.shape
            extra_rt += [t_valid, rm_page_bytes(counts)]
            writer_extra += [t_valid, rm_page_bytes(counts), rm_page_bytes(x_rm)]
        ep_mode = 2 if self.ep_rm else int(self.ep)
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [w_addr, counts.buffer_address(), bank, vc, r] + extra_rt
            writer_rt[core.x][core.y] = [
                x_rm.buffer_address(),
                routing.buffer_address(),
                counts.buffer_address(),
                act.buffer_address(),
                table.buffer_address(),
                bank,
                r,
            ] + writer_extra
        reader_ct = [
            cb_w,
            cb_c0,
            self.kt,
            self.E,
            self.groups,
            self.expert_stride,
            self.w_tile,
            self.page_bytes,
            self.ppb,
            ep_mode,
            self.T,
            self.picks,
            self.prefetch_pairs + 1,  # 12: weight ring slots
            self.unit_pairs,  # 13: pairs per unit (units kernels)
        ]
        reader_ct += ttnn.TensorAccessorArgs(counts).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(base if self.ep else counts).get_compile_time_args()  # unused if not EP
        writer_ct = [
            cb_xrm,
            cb_r,
            cb_c1,
            cb_d,
            cb_t,
            cb_out,
            self.kt,
            self.E,
            self.T,
            self.groups,
            self.i_tiles,
            self.row_bytes,
            TABLE_WIDTH * 2,
            self.ppb,
            ep_mode,
            self.picks,
            int(self.gather_rotate),  # 16
            self.unit_pairs,  # 17 (units kernels)
        ]
        for t in (x_rm, routing, counts, act, table, base if self.ep else counts):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_xrm, cb_x, cb_w, cb_d, cb_h, cb_out, self.kt, self.ppb if not self.units else self.unit_pairs]
        program = self._program(
            "gate_up_dispatch_units" if self.units else "gate_up_dispatch",
            cbs,
            reader_ct,
            reader_rt,
            writer_ct,
            writer_rt,
            compute_ct,
            defines=[REGIONS_DEFINE] if src is not None else (),
            reader_common=src.gate_up_addrs if src is not None else None,
        )
        weights = src.gate + src.up if src is not None else [w13]
        ttnn.generic_op(weights + [x_rm, routing, counts, act, table] + ([base] if self.ep else []), program)
        return act


class ExpertDownDispatch(_BankStreamOp):
    """(act, table, counts, w2, out) -> out. act / table from the gate|up dispatch, counts as above, w2 as the block
    op, out [1, 1, T, H] BF16 TILE. ``readers`` cores per bank (default 8: 2 output columns each; the resident
    per-pass h buffer is P x cols tiles). EP row-major: ``out_chunk`` (``ep_rs_chunk``) lays ``out`` out in 32-row
    chunks of ``out_chunk`` tokens ([1, 1, ceil(t_valid / out_chunk) * 32, H]) for the reduce-scatter combine."""

    def __init__(
        self,
        device,
        E: int,
        I: int,
        H: int,
        T: int,
        readers: int = 8,
        ep: bool = False,
        picks: int = MAX_PICKS,
        ep_rm: bool = False,
        gather_rotate: bool = False,
        regions_src: bool = False,
    ):
        super().__init__(device, readers)
        assert T % TILE == 0 and T <= 256, T
        self.E, self.I, self.H, self.T = E, I, H, T
        self.ep, self.picks, self.ep_rm = ep, picks, ep_rm
        self.gather_rotate = bool(gather_rotate)  # rotate the per-pass activation reads per core (H1)
        self.regions_src = bool(regions_src)  # the regions lists' per-expert down tensors (REGIONS_SRC kernels)
        self.seg_tiles = I // TILE
        self.C = H // TILE // self.banks
        assert self.C % readers == 0, (self.C, readers)
        self.cols = self.C // readers
        assert self.cols <= 8, self.cols
        self.block_tiles = self.cols * self.seg_tiles
        assert self.block_tiles <= 128, f"{self.block_tiles} tiles per expert block: use more readers"
        self.seg_bytes = self.seg_tiles * BFP8_TILE_BYTES
        if self.regions_src:
            self.regions_plan = plan_regions_stream(H, I, H, banks=self.banks, down_readers=readers)
            assert self.regions_plan.bank_local and self.regions_plan.cols == self.cols, self.regions_plan
            self.page_bytes = self.regions_plan.down_packet_bytes  # one K tile of the core's columns
        else:
            self.page_bytes = _packet_bytes(self.seg_tiles)
        self.bank_expert_stride = self.C * self.seg_bytes
        self.P = max_passes(T, E)
        self.J = MAX_PICKS
        # L1 budget: the resident h buffer (P x cols tiles) grows with T and E; the activation / weight slots give way
        self.a_slots, self.w_slots = 6, 3
        fixed = (self.P * self.cols + 2 * self.J * self.cols + 2 * self.cols + 1) * BF16_TILE_BYTES + self.P * 128
        fixed += (2 * (T // TILE + 1)) * BF16_TILE_BYTES
        while (
            fixed + self.a_slots * self.seg_tiles * BF16_TILE_BYTES + self.w_slots * self.block_tiles * BFP8_TILE_BYTES
            > 1_150_000
        ):
            if self.a_slots > 2:
                self.a_slots -= 1
            elif self.w_slots > 2:
                self.w_slots -= 1
            else:
                raise ValueError(f"down dispatch does not fit L1 at T = {T}, E = {E}, I = {I}")

    def __call__(self, act, table, counts, w2, out, base=None, t_valid=None, out_chunk=None):
        cb_a, cb_w, cb_c0, cb_c1, cb_t, cb_id, cb_h, cb_g, cb_m, cb_out = 0, 1, 2, 3, 4, 5, 6, 7, 8, 16
        src = _regions_source(w2, self.regions_src, self.E, self.H, self.I, self.H)
        w_addr = 0 if src is not None else w2.buffer_address()
        e_tiles = -(-self.E // TILE)
        c_tiles = self.T // TILE + 1 if self.ep else e_tiles
        assert (base is not None) == self.ep, "EP mode takes the expert base tensor"
        extra_rt = [base.buffer_address()] if self.ep else []
        if self.ep_rm:
            assert t_valid is not None and 0 < t_valid <= self.T, (t_valid, self.T)
            extra_rt += [t_valid, rm_page_bytes(counts)]
        ep_mode = 2 if self.ep_rm else int(self.ep)
        # output row placement (writer phase 2): the plain [T, H] layout (dispatched row i at tile i // 32, row
        # i % 32: OUT_TPR = 32, T / 32 tile rows), or the reduce-scatter combine's chunks of ``out_chunk`` tokens
        # (row i at tile i // out_chunk, row i % out_chunk; ``ep_rs_chunk`` from the env unless given). Compile-time:
        # a (T, out_chunk) pair is one program; the generic op's cache keys the descriptor.
        if self.ep_rm and out_chunk is None:
            out_chunk = ep_rs_chunk(self.device, t_valid)
        if out_chunk is None:
            out_tpr, rt_out = TILE, self.T // TILE
        else:
            assert self.ep_rm and 1 <= out_chunk <= TILE, (out_chunk, self.ep_rm)
            out_tpr, rt_out = out_chunk, -(-t_valid // out_chunk)
            assert out.shape[-2] == rt_out * TILE, (tuple(out.shape), rt_out, out_chunk, t_valid)
        cbs = [
            self._cb(cb_a, ttnn.bfloat16, BF16_TILE_BYTES, self.a_slots * self.seg_tiles),
            self._cb(cb_w, ttnn.bfloat8_b, BFP8_TILE_BYTES, self.w_slots * self.block_tiles),
            self._cb(cb_c0, ttnn.bfloat16, BF16_TILE_BYTES, c_tiles),
            self._cb(cb_c1, ttnn.bfloat16, BF16_TILE_BYTES, c_tiles),
            self._cb(cb_t, ttnn.uint16, TABLE_WIDTH * 2, self.P),
            self._cb(cb_id, ttnn.bfloat16, BF16_TILE_BYTES, 1),
            self._cb(cb_h, ttnn.bfloat16, BF16_TILE_BYTES, self.P * self.cols),
            self._cb(cb_g, ttnn.bfloat16, BF16_TILE_BYTES, 2 * self.J * self.cols),
            self._cb(cb_m, ttnn.uint32, 512, 1),  # [n_active, passes per active expert]
            self._cb(cb_out, ttnn.bfloat16, BF16_TILE_BYTES, 2 * self.cols),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [
                w_addr,
                counts.buffer_address(),
                bank,
                vc,
                r * self.cols,
            ] + extra_rt
            writer_rt[core.x][core.y] = [
                counts.buffer_address(),
                table.buffer_address(),
                act.buffer_address(),
                out.buffer_address(),
                bank * self.C + r * self.cols,
            ] + extra_rt
        reader_ct = [
            cb_w,
            cb_c0,
            self.seg_tiles,
            self.E,
            self.cols,
            self.bank_expert_stride,
            BFP8_TILE_BYTES,
            self.page_bytes,
            ep_mode,
            self.T,
            self.picks,
            self.w_slots,
        ]
        reader_ct += ttnn.TensorAccessorArgs(counts).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(base if self.ep else counts).get_compile_time_args()  # unused if not EP
        writer_ct = [
            cb_c1,
            cb_t,
            cb_a,
            cb_id,
            cb_h,
            cb_g,
            cb_out,
            self.seg_tiles,
            self.E,
            self.T,
            self.cols,
            self.J,
            self.P,
            self.H // TILE,
            TABLE_WIDTH * 2,
            cb_m,
            ep_mode,
            self.picks,
            out_tpr,  # 18: tokens per output tile row
            rt_out,  # 19: output tile rows (the compute's phase-2 count below must agree)
            int(self.gather_rotate),  # 20
        ]
        for t in (counts, table, act, out, base if self.ep else counts):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        if src is not None:  # the output column map's bank count, after every accessor (writer_down_dispatch.cpp)
            writer_ct += [self.banks]
        compute_ct = [cb_a, cb_w, cb_id, cb_h, cb_g, cb_out, self.seg_tiles, self.cols, self.J, rt_out, cb_m]
        program = self._program(
            "down_dispatch",
            cbs,
            reader_ct,
            reader_rt,
            writer_ct,
            writer_rt,
            compute_ct,
            fidelity=ttnn.MathFidelity.HiFi4,
            defines=[REGIONS_DEFINE] if src is not None else (),
            reader_common=src.down_addrs if src is not None else None,
            compute_defines=[REGIONS_DEFINE] if src is not None else (),
        )
        weights = list(src.down) if src is not None else [w2]
        ttnn.generic_op(weights + [act, table, counts, out] + ([base] if self.ep else []), program)
        return out


class ExpertDispatch:
    """The chip's routed experts for T gathered tokens as the two dispatch ops (weights read once):
    ``forward(x_rm, routing, counts, w13, w2, act, table, out)``."""

    def __init__(
        self,
        device,
        E: int,
        K: int,
        I: int,
        H: int,
        T: int,
        groups: int = 12,
        down_readers: int = 8,
        ep: bool = False,
        w13_dtype=ttnn.bfloat8_b,
        units: bool = False,
        ep_rm: bool = False,
        unit_pairs: int = 1,
        prefetch_pairs: int = 1,
        packet_tiles: int | None = None,
        gather_rotate: bool = False,
        regions_src: bool = False,
    ):
        self.gate_up = ExpertGateUpDispatch(
            device,
            E,
            K,
            I,
            T,
            groups,
            ep=ep,
            w_dtype=w13_dtype,
            units=units,
            ep_rm=ep_rm,
            unit_pairs=unit_pairs,
            prefetch_pairs=prefetch_pairs,
            packet_tiles=packet_tiles,
            gather_rotate=gather_rotate,
            regions_src=regions_src,
        )
        self.down = ExpertDownDispatch(
            device,
            E,
            I,
            H,
            T,
            down_readers,
            ep=ep,
            ep_rm=ep_rm,
            gather_rotate=gather_rotate,
            regions_src=regions_src,
        )
        self.E, self.K, self.I, self.H, self.T, self.ep, self.ep_rm = E, K, I, H, T, ep, ep_rm
        self.regions_src = bool(regions_src)
        self.P = self.gate_up.P

    def alloc(self, device, act_l1: bool = True, t_valid: int | None = None, out_chunk: int | None = None):
        """act / table / out for one call; ``t_valid`` (EP row-major) gives ``out`` its logical row count (the
        tile padding to T stays physical, so ``to_layout(ROW_MAJOR)`` yields exactly the dispatched rows). With the
        reduce-scatter combine's chunked layout (``out_chunk`` tokens per 32-row chunk, ``ep_rs_chunk`` from the env
        unless given) ``out`` is [1, 1, ceil(t_valid / out_chunk) * 32, H]: every row is logical (the reduce-scatter
        reads the pad rows too)."""
        if self.ep_rm and out_chunk is None:
            out_chunk = ep_rs_chunk(device, t_valid)
        out_rows = -(-t_valid // out_chunk) * TILE if out_chunk is not None else (t_valid or self.T)
        act = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.P, TILE, self.I]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.L1_MEMORY_CONFIG if act_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        table = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.P, TABLE_WIDTH]), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, out_rows, self.H]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        return act, table, out

    @staticmethod
    def prepare(x, routing):
        """The dispatch inputs from the block inputs: x [1, 1, T, K] BF16 TILE -> ROW_MAJOR (L1), the counts."""
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        return x_rm, routing_counts(routing)

    def forward(self, x_rm, routing, counts, w13, w2, act, table, out, base=None, t_valid=None, out_chunk=None):
        """Non-EP: (x_rm, routing, counts). EP: routing = weight tiles, counts = index tiles, base = the chip's
        expert base page (``ep_index_tiles`` / ``ep_base_tensor``). EP row-major: x_rm = the [x | w] rows, counts =
        the id rows, t_valid = the dispatched row count (``ep_rm_inputs``), out_chunk = the chunked output layout
        ``out`` was allocated with (``alloc``; the env default when ``None``)."""
        self.gate_up(x_rm, routing, counts, w13, act, table, base=base, t_valid=t_valid)
        self.down(act, table, counts, w2, out, base=base, t_valid=t_valid, out_chunk=out_chunk)
        return out

    __call__ = forward
    golden = staticmethod(expert_dispatch_golden)
    golden_ep = staticmethod(expert_dispatch_golden_ep)

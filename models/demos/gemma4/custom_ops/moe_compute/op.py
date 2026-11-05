# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``MoeComputeCustom``: the decode MoE kernel (``ttnn.experimental.moe_compute``) as a gemma4-local
``ttnn.generic_op`` with per-expert resident weights (README.md).

The program is the shared op's ComputeOnly program (tilize cores -> matmul ring -> dm1 staging into the combine
cores' L1, no combine kernels): the same core placement (``select_cores`` ports ``select_moe_compute_cores`` and is
checked against the op's public placement helpers), CBs, semaphores, compile-time and runtime args, with these
differences:

* ``c_3`` holds one expert's whole W0/W1 slice of the ring core and ``c_10`` its W2 slice; dm0 streams each once per
  call with ``prefetch_depth`` 16 KB block reads in flight; the compute kernel addresses them by tile index;
* ``layer_id`` is a runtime arg (one program for all layers);
* ``staging_layout``: ``"fused"`` = the shared op's double-buffered halves (its compute_only output layout),
  ``"standalone"`` = one block per expert, the layout ``ttnn.experimental.selective_reduce_combine`` reads;
* ``skip_zero_scores``: zero-score routes (frame padding) are dropped by the tilize scan.

Contract (per device; on a mesh every coordinate runs the program with its own ``linearized_mesh_coord``):

| tensor | shape / dtype / layout | role |
|---|---|---|
| ``tilize_input`` | ``[1, T, H]`` (or ``[T, H]``) bf16 ROW_MAJOR, DRAM or L1 interleaved | the dispatch output: token rows by global token id |
| ``indices`` / ``scores`` | ``[1, T, K]`` uint16 / bf16 ROW_MAJOR, L1 HEIGHT_SHARDED on the tilize drain core | the routing of every token (all K slots) |
| ``mapping`` | ``[D, E_global]`` uint16 ROW_MAJOR | global expert -> linearized mesh coordinate, one row per source device |
| ``w0_w1`` / ``w2`` | the shared op's packed bf4 HEIGHT_SHARDED DRAM tensors (``prepare_*_for_moe_compute`` + ``get_weight_mem_configs``) | ``[L, E, ...]`` layered expert weights |

Returns the shared op's five ComputeOnly outputs: ``(per_expert_total_tokens, expert_activation, e_t, tilize_output,
staging)`` where ``staging`` is ``tilize_output`` (the whole-grid ``[num_cores, 64, H]`` bf16 ROW_MAJOR L1 shard that
the tilize cores, the ring cores and the combine cores share; the staged rows live in the combine cores' shards).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from ttnn.experimental import moe_compute_utils as mcu
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from models.demos.gemma4.custom_ops.grid import BLACKHOLE_L1_PER_CORE

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
TOKENS_PER_CHUNK = 32  # the shared op's chunk (TOKEN_SIZE)
DOUBLE_BUFFER = 2  # the shared tilize-output buffer: two 32-row halves per core
BF4_TILE_BYTES = 576
BF16_TILE_BYTES = 2048
INVALID = 0  # semaphore initial value (hostdevcommon common_values.hpp)
PREFETCH_DEPTH = (
    8  # 16 KB weight-block reads in flight per ring core (4..12 measured alike: the weights left the critical path)
)
TXNS_PER_READ = 1  # one NoC read per weight block (the Blackhole burst); the kernel's 2-transaction form is unused

STAGING_LAYOUTS = {"fused": 0, "standalone": 1}


def _ceil_div(a, b):
    return (a + b - 1) // b


def _align(x, a):
    return _ceil_div(x, a) * a


def _activation_value(activation) -> int:
    if isinstance(activation, int):
        return activation
    return int(activation.value)


# ---------------------------------------------------------------------------------------------------- placement
def _in_box(box, c):
    return box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3]


def _bbox(cores):
    xs = [c[0] for c in cores]
    ys = [c[1] for c in cores]
    return (min(xs), min(ys), max(xs), max(ys))


def _boxes_intersect(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _box_size(b):
    return (b[2] - b[0] + 1) * (b[3] - b[1] + 1)


def _tilize_num_cores(hidden_tiles):
    n = min(4, hidden_tiles)
    while n > 1 and hidden_tiles % n != 0:
        n -= 1
    return max(1, n)


def _find_combine_strip(avoid, grid, strip_h, max_y_incl):
    width = 2
    if width > grid[0] or strip_h == 0:
        return None
    y_limit = min(max_y_incl + 1, grid[1])
    for sx in range(grid[0] - width, -1, -1):
        for sy in range(0, y_limit - strip_h + 1):
            if all((sx + dx, sy + dy) not in avoid for dy in range(strip_h) for dx in range(width)):
                return (sx, sy, sx + width - 1, sy + strip_h - 1)
    return None


def _find_dense_rect(avoid, grid, n, max_y_incl):
    if n == 0:
        return None
    y_limit = min(max_y_incl + 1, grid[1])
    facts = sorted([(w, n // w) for w in range(1, n + 1) if n % w == 0], key=lambda wh: (-wh[0], wh[1]))
    for w, h in facts:
        if w > grid[0] or h == 0 or h > y_limit:
            continue
        for sx in range(grid[0] - w, -1, -1):
            for sy in range(0, y_limit - h + 1):
                if all((sx + dx, sy + dy) not in avoid for dy in range(h) for dx in range(w)):
                    return (sx, sy, sx + w - 1, sy + h - 1)
    return None


def _rect_cores_row_major(box):
    return [(x, y) for y in range(box[1], box[3] + 1) for x in range(box[0], box[2] + 1)]


def _pick_row_major_avoiding(avoid, grid, n, max_y_incl):
    out = []
    y_limit = min(max_y_incl + 1, grid[1])
    for y in range(y_limit):
        for x in range(grid[0]):
            if len(out) >= n:
                return out
            if (x, y) not in avoid:
                out.append((x, y))
    return out


def _find_tilize_2x2(avoid, grid):
    if grid[0] < 2 or grid[1] < 2:
        return None
    sy = grid[1] - 2
    for sx in range(grid[0] - 2, -1, -1):
        if all((sx + dx, sy + dy) not in avoid for dy in range(2) for dx in range(2)):
            return (sx, sy)
    return None


def _pick_tilize_upper_rows(avoid, grid, n, min_y):
    out = []
    for y in range(grid[1] - 1, min_y - 1, -1):
        for x in range(grid[0] - 1, -1, -1):
            if len(out) >= n:
                return out
            if (x, y) not in avoid:
                out.append((x, y))
    return out


def _place_combine_and_tilize(grid, matmul_box, mux, num_combine, target_tilize, hidden_tiles):
    base_avoid = set(mux) | set(_rect_cores_row_major(matmul_box))
    combine_max_y = grid[1] - 3 if grid[1] >= 3 else grid[1] - 1
    tilize_min_y = grid[1] - 2 if grid[1] >= 2 else grid[1] - 1
    strip_h = _ceil_div(num_combine, 2)
    strip = _find_combine_strip(base_avoid, grid, strip_h, combine_max_y)
    if strip is not None:
        combine = _rect_cores_row_major(strip)[:num_combine]
    else:
        rect = _find_dense_rect(base_avoid, grid, num_combine, combine_max_y)
        combine = (
            _rect_cores_row_major(rect)[:num_combine]
            if rect is not None
            else _pick_row_major_avoiding(base_avoid, grid, num_combine, combine_max_y)
        )
    if len(combine) != num_combine:
        return None
    combine_box = _bbox(combine)
    if _box_size(combine_box) != num_combine or _boxes_intersect(combine_box, matmul_box):
        return None
    tilize_avoid = base_avoid | set(_rect_cores_row_major(combine_box))
    for n in range(target_tilize, 0, -1):
        if hidden_tiles % n != 0:
            continue
        blk = _find_tilize_2x2(tilize_avoid, grid)
        if blk is not None:
            sx, sy = blk
            tilize = [(sx + 1, sy + 1), (sx + 1, sy), (sx, sy + 1), (sx, sy)][:n]
        else:
            tilize = _pick_tilize_upper_rows(tilize_avoid, grid, n, tilize_min_y)
        if len(tilize) != n:
            continue
        tilize_box = _bbox(tilize)
        if _boxes_intersect(tilize_box, combine_box) or _boxes_intersect(tilize_box, matmul_box):
            continue
        return combine, tilize, combine_box, tilize_box
    return None


def select_cores(mesh_device, hs, ws, hidden_size, mux_core_range_set):
    """Port of ``select_moe_compute_cores`` (moe_core_placement.cpp): matmul ring = the DRAM-bank-adjacent workers,
    combine = a dense rectangle in the eastern columns below the tilize rows, tilize = the top-right 2x2 block in the
    legacy order (index 0 = drain). Cross-checked against the op's public placement helpers."""
    g = mesh_device.compute_with_storage_grid_size()
    grid = (g.x, g.y)
    hidden_tiles = hidden_size // TILE
    ring = [
        (c.x, c.y)
        for c in ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, ttnn.NOC.RISCV_0_default)
    ]
    ring_size = len(ring)
    mux = set()
    if mux_core_range_set is not None and not mux_core_range_set.empty():
        mux = {(c.x, c.y) for c in ttnn.corerange_to_cores(mux_core_range_set)}
    num_combine = hs * ws
    target_tilize = _tilize_num_cores(hidden_tiles)
    placed = None
    matmul = None
    if not any(c in mux for c in ring):
        placed = _place_combine_and_tilize(grid, _bbox(ring), mux, num_combine, target_tilize, hidden_tiles)
        if placed is not None:
            matmul = ring
    if placed is None:
        x_limit = grid[0] - 2 if grid[0] > 2 else grid[0]
        y_limit = grid[1] - 2 if grid[1] > 2 else grid[1]
        for y_start in range(grid[1]):
            cand = []
            for y in range(y_start, y_limit):
                if (0, y) in mux:
                    continue
                for x in range(x_limit):
                    if len(cand) >= ring_size:
                        break
                    if (x, y) not in mux:
                        cand.append((x, y))
                if len(cand) >= ring_size:
                    break
            if len(cand) != ring_size:
                break
            placed = _place_combine_and_tilize(grid, _bbox(cand), mux, num_combine, target_tilize, hidden_tiles)
            if placed is not None:
                matmul = cand
                break
        assert (
            placed is not None
        ), "no disjoint matmul / combine / tilize placement (select_moe_compute_cores would fail too)"
    combine, tilize, combine_box, tilize_box = placed
    combine = sorted(combine)  # x-major, the order dm1's OUTPUT_SHARD_CORE_MAP and the combine op index by
    # the op's own helpers must agree with the port (they take no None: omit the kwarg without a mux range)
    pk = {} if mux_core_range_set is None else {"mux_core_range_set": mux_core_range_set}
    drain = ttnn.experimental.get_moe_tilize_drain_core(mesh_device, hs, ws, hidden_size, **pk)
    assert (drain.x, drain.y) == tilize[0], f"tilize drain core {tilize[0]} != op helper {(drain.x, drain.y)}"
    op_combine = [(c.x, c.y) for c in ttnn.experimental.get_moe_combine_cores(mesh_device, hs, ws, hidden_size, **pk)]
    assert op_combine == combine, f"combine cores {combine} != op helper {op_combine}"
    bb = ttnn.experimental.get_moe_worker_mcast_bounding_box(mesh_device, hs, ws, hidden_size, **pk)
    all_box = _bbox(matmul + combine + tilize)
    assert (bb.start.x, bb.start.y, bb.end.x, bb.end.y) == all_box, f"worker bbox {all_box} != op helper {bb}"
    return {
        "grid": grid,
        "matmul": matmul,
        "combine": combine,
        "tilize": tilize,
        "matmul_box": _bbox(matmul),
        "combine_box": combine_box,
        "tilize_box": tilize_box,
        "all_box": all_box,
    }


# ---------------------------------------------------------------------------------------------------- geometry
def a2a_exchange_tiles(Nt, n_cores):
    max_cols = _ceil_div(Nt, n_cores)
    if not mcu._w0_w1_compact_for_shape(Nt, n_cores):
        return mcu._even_stride_at_least_a2a_width(max_cols)
    return max(2, max_cols)


def w0_w1_region_tiles(Ht, Nt, n_cores, has_bias, tiles_per_txn):
    """Tiles of the largest ring core's routed W0/W1 slice (the resident region every core reserves)."""
    lay = mcu._w0_w1_compact_layout(Ht + (1 if has_bias else 0), Nt, n_cores, n_cores, tiles_per_txn)
    return max(lay["core_blocks"]) * 2 * tiles_per_txn


def w2_region_tiles(Ht, Nt, n_cores, has_bias, tiles_per_txn):
    return mcu._w2_core_blocks_per_expert(Ht, Nt + (1 if has_bias else 0), n_cores, tiles_per_txn) * 2 * tiles_per_txn


# ---------------------------------------------------------------------------------------------------- the op
class MoeComputeCustom:
    @staticmethod
    def staging_capacity_reason(hidden_size: int, experts_per_device: int, tokens: int, output_height_shard_dim: int):
        """None when the standalone staging layout (one block of ``tokens / hs`` segment rows per expert) fits the
        combine cores' ``[64, H]`` bf16 shard, else why not: ``E x (T / hs) x (H / ws) x 2 B <= 64 x H x 2 B``,
        i.e. ``E x T <= 64 x hs x ws`` (the Galaxy's 4 experts x 256 frame rows fit; a (2,4) block's 16 experts x
        128 rows do not)."""
        hs = int(output_height_shard_dim)
        ws = mcu.auto_output_width_shard_dim(int(hidden_size))
        if int(tokens) % hs:
            return f"tokens {tokens} must split over the {hs} height shards"
        seg_bytes = (int(hidden_size) // TILE // ws) * TILE * 2  # the plan's token_segment_bytes
        block = int(tokens) // hs * seg_bytes
        shard = DOUBLE_BUFFER * TOKENS_PER_CHUNK * int(hidden_size) * 2
        if int(experts_per_device) * block > shard:
            return (
                f"{experts_per_device} experts x {block} B of staging ({tokens} tokens / {hs} shards x {seg_bytes} B "
                f"segments) exceed the {shard} B combine shard"
            )
        return None

    @classmethod
    def unsupported_reason(
        cls, mesh_device, *, hidden_size: int, experts_per_device: int, tokens: int, output_height_shard_dim: int = 4
    ):
        """None when the op can serve ``tokens`` frame rows of ``experts_per_device`` experts on ``mesh_device``,
        else why not. The port is Blackhole-only (one 16 KB NoC read per weight block on the DRAM ring) and its
        standalone staging has a fixed capacity; the wrapper (``tt/experts/moe_decode.py``) keeps the shared op
        with one warning instead of asserting at plan time."""
        if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
            return cls.BLACKHOLE_ONLY
        return cls.staging_capacity_reason(hidden_size, experts_per_device, tokens, output_height_shard_dim)

    BLACKHOLE_ONLY = "Blackhole-only: the moe_compute port assumes the Blackhole DRAM ring and NoC burst"

    def __init__(
        self,
        mesh_device,
        *,
        hidden_size: int,
        intermediate_size: int,
        experts_per_device: int,
        select_experts_k: int,
        output_height_shard_dim: int = 4,
        mux_core_range_set=None,
        activation_type=MoEActivationFunction.GELU,
        activation_limit: float | None = None,
        cluster_axis: int | None = None,
        has_bias: bool = False,
        num_shared_experts_per_device: int = 0,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en: bool = False,
        staging_layout: str = "standalone",
        skip_zero_scores: bool | None = None,
    ):
        assert mesh_device.arch() == ttnn.Arch.BLACKHOLE, self.BLACKHOLE_ONLY
        assert not has_bias, "has_bias is not ported (the bias kernel paths are kept but untested)"
        assert num_shared_experts_per_device == 0, "shared experts are not ported"
        assert staging_layout in STAGING_LAYOUTS, staging_layout
        self.device = mesh_device
        self.hidden = hidden_size
        self.inter = intermediate_size
        self.Ht = hidden_size // TILE
        self.Nt = intermediate_size // TILE
        self.E = experts_per_device
        self.K = select_experts_k
        self.hs = output_height_shard_dim
        self.ws = mcu.auto_output_width_shard_dim(hidden_size)
        self.mux = mux_core_range_set
        self.activation = _activation_value(activation_type)
        self.activation_limit = activation_limit
        self.cluster_axis = cluster_axis
        self.math_fidelity = math_fidelity
        self.fp32_dest_acc_en = fp32_dest_acc_en
        self.staging_layout = STAGING_LAYOUTS[staging_layout]
        self.prefetch_depth = PREFETCH_DEPTH
        self.txns_per_read = TXNS_PER_READ
        # skip_zero_scores is on by default: the decode frame's padding rows carry score 0 on ids 0..K-1 and would
        # otherwise land as whole extra chunks on device 0's experts (321 -> 75 us per call on the Galaxy frame);
        # GEMMA4_MOE_COMPUTE_SKIP_ZERO_SCORES=0 keeps them for a bit-identity check against the shared op.
        self.skip_zero_scores = (
            os.environ.get("GEMMA4_MOE_COMPUTE_SKIP_ZERO_SCORES", "1") == "1"
            if skip_zero_scores is None
            else skip_zero_scores
        )
        assert 1 <= self.prefetch_depth <= 15 and self.txns_per_read in (1, 2)
        assert hidden_size % TILE == 0 and intermediate_size % TILE == 0
        assert self.Ht % self.ws == 0
        self.cores = select_cores(mesh_device, self.hs, self.ws, hidden_size, mux_core_range_set)
        self.ring_size = len(self.cores["matmul"])
        assert self.Nt >= self.ring_size, "at least one intermediate tile per ring core"
        self.tiles_per_txn = mcu._tiles_per_txn(self.Ht, self.Nt, has_bias, self.ring_size)
        self.w01_tiles = w0_w1_region_tiles(self.Ht, self.Nt, self.ring_size, has_bias, self.tiles_per_txn)
        self.w2_tiles = w2_region_tiles(self.Ht, self.Nt, self.ring_size, has_bias, self.tiles_per_txn)
        self.l1_alignment = ttnn.device.get_l1_alignment()
        self._plans = {}

    # ------------------------------------------------------------------------------------------------ outputs
    def allocate_outputs(self, tokens: int):
        """The shared op's five output tensors (allocated per call, like the op does)."""
        dev = self.device
        g = self.cores["grid"]
        num_cores = g[0] * g[1]
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g[0] - 1, g[1] - 1))])
        row_elems = _align(self.E * 4, self.l1_alignment) // 4
        counts = ttnn.allocate_tensor_on_device(
            ttnn.Shape([num_cores, row_elems]),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            dev,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [1, row_elems], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        act_row_bytes = _align((2 * self.E + 1) * 4, self.l1_alignment)
        activation = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, tokens * act_row_bytes // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.L1_MEMORY_CONFIG
        )
        e_t_row_elems = (tokens + 1) * _align(4, self.l1_alignment) // 4
        e_t = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.E, e_t_row_elems]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.L1_MEMORY_CONFIG
        )
        shared = ttnn.allocate_tensor_on_device(
            ttnn.Shape([num_cores * DOUBLE_BUFFER * TOKENS_PER_CHUNK, self.hidden]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            dev,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [DOUBLE_BUFFER * TOKENS_PER_CHUNK, self.hidden], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        return counts, activation, e_t, shared

    # ------------------------------------------------------------------------------------------------ call
    def __call__(self, tilize_input, indices, scores, mapping, w0_w1, w2, layer_id: int = 0):
        assert tilize_input.layout == ttnn.ROW_MAJOR_LAYOUT and tilize_input.dtype == ttnn.bfloat16
        assert indices.dtype == ttnn.uint16 and indices.layout == ttnn.ROW_MAJOR_LAYOUT
        assert scores.dtype == ttnn.bfloat16 and scores.layout == ttnn.ROW_MAJOR_LAYOUT
        assert mapping.dtype == ttnn.uint16 and len(mapping.shape) == 2
        assert w0_w1.dtype == ttnn.bfloat4_b and w2.dtype == ttnn.bfloat4_b
        assert indices.is_sharded() and scores.is_sharded(), "indices / scores are L1 HEIGHT_SHARDED on the drain core"
        # the tilize CBs c_4 / c_6 are bound to the routing shards (cb_descriptor_from_sharded_tensor): the drain core
        # reads them in place, the other tilize cores over the NoC from the drain core's address, so the shards must
        # sit on that one core
        drain = ttnn.CoreCoord(*self.cores["tilize"][0])
        for name, t in (("indices", indices), ("scores", scores)):
            shard_cores = ttnn.corerange_to_cores(t.memory_config().shard_spec.grid)
            assert shard_cores == [drain], f"{name} must be sharded on the tilize drain core {drain}, got {shard_cores}"
        tokens = int(math.prod(tilize_input.shape) // self.hidden)
        assert int(indices.shape[-1]) == self.K
        counts, activation, e_t, shared = self.allocate_outputs(tokens)
        key = (
            tokens,
            int(mapping.buffer_num_pages()),
            int(mapping.shape[1]),
            tuple(w0_w1.shape),
            tuple(w2.shape),
            str(tilize_input.memory_config()),
            str(mapping.memory_config()),
            str(indices.memory_config()),  # the routing shards' page sizes and placement are in the plan
            str(scores.memory_config()),
            int(w0_w1.buffer_num_pages()),
            int(w2.buffer_num_pages()),
        )
        self.last_tokens = tokens
        plan = self._plans.get(key)
        if plan is None:
            plan = self._build_plan(
                tilize_input, indices, scores, mapping, w0_w1, w2, counts, activation, e_t, shared, tokens
            )
            self._plans[key] = plan
        program = self._program(
            plan, tilize_input, indices, scores, mapping, w0_w1, w2, counts, activation, e_t, shared, layer_id
        )
        io = [tilize_input, indices, scores, mapping, w0_w1, w2, counts, activation, e_t, shared]
        ttnn.generic_op(io, program)
        return counts, activation, e_t, shared, shared

    # ------------------------------------------------------------------------------------------------ plan
    def _build_plan(self, tilize_input, indices, scores, mapping, w0_w1, w2, counts, activation, e_t, shared, tokens):
        """Everything of the program that does not depend on buffer addresses: the shared factory's constants."""
        dev = self.device
        C = self.cores
        hidden = self.hidden
        E = self.E
        l1a = self.l1_alignment
        Ht, Nt = self.Ht, self.Nt
        n_ring = self.ring_size
        experts_global = int(mapping.shape[1])
        mapping_pages = int(mapping.buffer_num_pages())

        tilize = C["tilize"]
        matmul = C["matmul"]
        combine = C["combine"]
        n_tilize = len(tilize)
        phys = lambda c: dev.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        tilize_phys = [phys(c) for c in tilize]
        matmul_phys = [phys(c) for c in matmul]
        combine_phys = [phys(c) for c in combine]
        drain_phys = tilize_phys[0]

        # page sizes (the factory reads them from the buffers)
        in_page = int(tilize_input.buffer_page_size())
        in_aligned_page = int(tilize_input.buffer_aligned_page_size())
        idx_page = int(indices.buffer_page_size())
        idx_aligned = int(indices.buffer_aligned_page_size())
        sc_aligned = int(scores.buffer_aligned_page_size())
        map_page = int(mapping.buffer_page_size())
        map_aligned = int(mapping.buffer_aligned_page_size())
        counts_page = int(counts.buffer_page_size())
        act_page = int(activation.buffer_page_size())
        e_t_page = int(e_t.buffer_page_size())
        in_pages = int(tilize_input.buffer_num_pages())
        idx_pages = int(indices.buffer_num_pages())
        sc_pages = int(scores.buffer_num_pages())
        tile_bf16 = BF16_TILE_BYTES
        shared_shard_bytes = DOUBLE_BUFFER * TOKENS_PER_CHUNK * hidden * 2
        shared_cb_num_pages = shared_shard_bytes // tile_bf16
        assert shared_cb_num_pages * tile_bf16 == shared_shard_bytes

        # tilize work split (split_work_to_cores over the tilize core set): one unit per core at H = 2816
        subtoken_bytes = _align(_ceil_div(in_aligned_page, n_tilize), l1a)
        units = _ceil_div(in_aligned_page, subtoken_bytes)
        assert (
            units >= n_tilize and units % n_tilize == 0
        ), f"tilize work split {units} units over {n_tilize} cores needs the two-group split, not ported"
        units_per_core = units // n_tilize
        max_subtoken = units_per_core * subtoken_bytes
        tile_width_bytes = TILE * 2
        max_tiles_per_local_chunk = max_subtoken // tile_width_bytes
        primary_group = n_tilize // 2
        secondary_group = n_tilize - primary_group
        subtoken_sizes = []
        off = 0
        for i in range(n_tilize):
            sz = units_per_core * subtoken_bytes
            if off + sz > in_aligned_page:
                sz = in_aligned_page - off
            subtoken_sizes.append(sz)
            off += sz
        primary_bytes = sum(subtoken_sizes[:primary_group])
        secondary_bytes = sum(subtoken_sizes[primary_group:])
        tokens_per_tilize_core = tokens // n_tilize

        # bounding boxes (physical)
        def pbox(box):
            s = phys((box[0], box[1]))
            e = phys((box[2], box[3]))
            return s.x, s.y, e.x, e.y

        tilize_box = C["tilize_box"]
        matmul_box = C["matmul_box"]
        all_box = C["all_box"]
        tilize_pb, matmul_pb, all_pb = pbox(tilize_box), pbox(matmul_box), pbox(all_box)
        senders = [tilize[0]] + ([tilize[primary_group]] if n_tilize > 1 else [])
        inside = [_in_box(matmul_box, s) for s in senders]
        assert all(inside) or not any(inside), "tilize mcast senders straddle the matmul bbox"
        matmul_mcast_num_dests = _box_size(matmul_box) - 1 if inside[0] else _box_size(matmul_box)

        # mesh
        mesh_rows, mesh_cols = dev.shape
        num_devices = mesh_rows * mesh_cols
        cluster_axis_ct = self.cluster_axis if self.cluster_axis is not None else 1

        # matmul geometry (the shared factory's TT_FATALs on the packed weights)
        tiles_per_txn = self.tiles_per_txn
        tiles_per_block = 2 * tiles_per_txn
        lay = mcu._w0_w1_compact_layout(Ht, Nt, n_ring, n_ring, tiles_per_txn)
        w01_layers = int(w0_w1.shape[1])
        w01_expected_pages = n_ring * w01_layers * E * lay["bank_blocks_per_expert"] * tiles_per_block
        assert (
            int(w0_w1.buffer_num_pages()) == w01_expected_pages
        ), f"w0_w1 has {int(w0_w1.buffer_num_pages())} tile pages, the compact layout needs {w01_expected_pages}"
        w2_layers = int(w2.shape[1])
        w2_core_blocks = mcu._w2_core_blocks_per_expert(Ht, Nt, n_ring, tiles_per_txn)
        w2_expected_pages = n_ring * w2_layers * E * w2_core_blocks * tiles_per_block
        assert (
            int(w2.buffer_num_pages()) == w2_expected_pages
        ), f"w2 has {int(w2.buffer_num_pages())} tile pages, the layout needs {w2_expected_pages}"
        w2_pages_per_ring_core_total = w2_expected_pages // n_ring
        assert w01_layers == w2_layers
        a2a_cb_pages = a2a_exchange_tiles(Nt, n_ring)
        combine_shard_width_tiles = Ht // self.ws
        token_segment_bytes = combine_shard_width_tiles * TILE * 2
        # fused layout: compute_fused_source_buffer_layout(shard {64, H}, segment H/ws, 2 buffers)
        segments_per_shard_row = hidden // (hidden // self.ws)
        rows_per_buffer = (DOUBLE_BUFFER * TOKENS_PER_CHUNK // 2) * segments_per_shard_row
        # standalone layout: one block of T/hs segment rows per expert (the capacity unsupported_reason states)
        reason = self.staging_capacity_reason(hidden, E, tokens, self.hs)
        assert reason is None, reason
        staging_expert_block_bytes = (tokens // self.hs) * token_segment_bytes
        assert E * staging_expert_block_bytes <= shared_shard_bytes
        noc_max_burst = 16384
        if self.txns_per_read == 1:
            assert tiles_per_block * BF4_TILE_BYTES <= noc_max_burst

        # shard_to_bank: shard i of the HEIGHT_SHARDED weights sits on bank grid[i].x (CoreCoord(bank, 0))
        bank_cores = ttnn.corerange_to_cores(w0_w1.memory_config().shard_spec.grid)
        assert len(bank_cores) == n_ring, f"w0_w1 is sharded over {len(bank_cores)} banks, ring has {n_ring} cores"
        assert all(c.y == 0 for c in bank_cores)
        shard_to_bank = [c.x for c in bank_cores]
        assert ttnn.corerange_to_cores(w2.memory_config().shard_spec.grid) == bank_cores

        # ring order: physical (desc y, desc x); neighbour = next ring position
        order = sorted(range(n_ring), key=lambda b: (-matmul_phys[b].y, -matmul_phys[b].x))
        ring_pos_of = {b: p for p, b in enumerate(order)}
        next_bank_of = {order[p]: order[(p + 1) % n_ring] for p in range(n_ring)}
        vchannels = []
        for b, core in enumerate(matmul):
            vc = b & 3
            for j in range(b):
                if matmul[j][1] == core[1]:
                    if vc == vchannels[j]:
                        vc = (vc + 1) & 3
                    break
            vchannels.append(vc)

        # L1 fit of the ring cores' CBs (the shared tensor is allocated, the CBs grow from the allocator base)
        ring_cb_bytes = (
            (self.w01_tiles + self.w2_tiles) * BF4_TILE_BYTES + a2a_cb_pages * n_ring * tile_bf16 + 2 * l1a + 2 * 16
        )
        l1_base = ttnn._ttnn.device.GetAllocatorBaseAddress(dev, ttnn.BufferType.L1)
        assert (
            l1_base + ring_cb_bytes + shared_shard_bytes <= BLACKHOLE_L1_PER_CORE
        ), f"ring core L1: base {l1_base} + CBs {ring_cb_bytes} + shared shard {shared_shard_bytes} exceeds L1"

        # semaphores, in the factory's order (ids as add_semaphore_descriptor assigns them)
        tilize_crs = _crs(tilize)
        matmul_crs = _crs(matmul)
        tilize_matmul_crs = _crs(tilize + matmul)
        all_worker_crs = _crs(tilize + matmul + combine)
        sem_descs = []

        def add_sem(cores_set, crs):
            used = {sid for sid, cs in sem_descs if cs & cores_set}
            sid = next(i for i in range(16) if i not in used)
            sem_descs.append((sid, set(cores_set)))
            return sid, ttnn.SemaphoreDescriptor(id=sid, core_ranges=crs, initial_value=INVALID)

        sem_objs = []

        def sem(cores_list, crs):
            sid, d = add_sem(set(cores_list), crs)
            sem_objs.append(d)
            return sid

        partial_metadata_ready = sem(tilize, tilize_crs)
        tilize_chunk_ready = sem(tilize, tilize_crs)
        previous_chunk_sent = sem(tilize, tilize_crs)
        initial_gather = sem(tilize, tilize_crs)
        ring_sem = sem(matmul, matmul_crs)
        metadata_ready = sem(tilize + matmul, tilize_matmul_crs)
        matmul_chunk_available = sem(tilize + matmul, tilize_matmul_crs)
        matmul_chunk_ready = sem(tilize + matmul, tilize_matmul_crs)
        tilize_combine_sync = sem(matmul, matmul_crs)  # ComputeOnly: unused, allocated on the matmul set
        matmul_combine_sync = sem(matmul, matmul_crs)

        tilize_named = [
            ("tilize_input_cb_id", 8),
            ("tilize_output_cb_id", 0),
            ("total_chunks_cb_id", 3),
            ("indices_tensor_cb_id", 4),
            ("scores_tensor_cb_id", 6),
            ("mapping_tensor_cb_id", 5),
            ("e_t_cb_id", 7),
            ("expert_activation_cb_id", 9),
            ("per_expert_total_tokens_cb_id", 2),
            ("brisc_e_t_cb_id", 10),
            ("brisc_expert_counts_cb_id", 11),
            ("brisc_expert_activation_cb_id", 12),
            ("brisc_activated_count_cb_id", 13),
            ("remote_counts_cb_id", 14),
            ("l1_alignment", l1a),
            ("dram_alignment", ttnn.device.get_dram_alignment()),
            ("e_t_entry_size", l1a),
            ("input_pages", in_pages),
            ("indices_pages", idx_pages),
            ("mapping_pages", mapping_pages),
            ("scores_pages", sc_pages),
            ("shared_cb_num_pages", shared_cb_num_pages),
            ("input_page_size", in_page),
            ("indices_page_size", idx_page),
            ("mapping_page_size", map_page),
            ("per_expert_total_tokens_output_page_size", counts_page),
            ("expert_activation_output_page_size", act_page),
            ("e_t_output_page_size", e_t_page),
            ("tilize_output_page_size", tile_bf16),
            ("aligned_input_page_size", in_aligned_page),
            ("aligned_indices_page_size", idx_aligned),
            ("aligned_mapping_page_size", map_aligned),
            ("aligned_scores_page_size", sc_aligned),
            ("tokens", tokens),
            ("hidden_size", hidden),
            ("remote_counts_entry_size", _align((E + 1) * 4, l1a)),
            ("experts", experts_global),
            ("experts_per_device", E),
            ("selected_experts_k", self.K),
            ("tokens_per_chunk", TOKENS_PER_CHUNK),
            ("num_devices", num_devices),
            ("mesh_rows", mesh_rows),
            ("mesh_cols", mesh_cols),
            ("linearized_mesh_coord", 0),  # replaced per coordinate
            ("cluster_axis", cluster_axis_ct),
            ("drain_core_noc_x", drain_phys.x),
            ("drain_core_noc_y", drain_phys.y),
            ("primary_mcast_gather_group_num_cores", primary_group),
            ("secondary_mcast_gather_group_num_cores", secondary_group),
            ("num_tilize_cores", n_tilize),
            ("tilize_mcast_start_x", tilize_pb[0]),
            ("tilize_mcast_start_y", tilize_pb[1]),
            ("tilize_mcast_end_x", tilize_pb[2]),
            ("tilize_mcast_end_y", tilize_pb[3]),
            ("tilize_bounding_box_num_cores", _box_size(tilize_box)),
            ("num_matmul_cores", n_ring),
            ("matmul_mcast_start_x", matmul_pb[0]),
            ("matmul_mcast_start_y", matmul_pb[1]),
            ("matmul_mcast_end_x", matmul_pb[2]),
            ("matmul_mcast_end_y", matmul_pb[3]),
            ("matmul_bounding_box_num_cores", matmul_mcast_num_dests),
            ("all_worker_cores_mcast_start_x", all_pb[0]),
            ("all_worker_cores_mcast_start_y", all_pb[1]),
            ("all_worker_cores_mcast_end_x", all_pb[2]),
            ("all_worker_cores_mcast_end_y", all_pb[3]),
            ("all_worker_cores_bounding_box_num_cores", _box_size(all_box)),
            ("partial_metadata_ready_semaphore_id", partial_metadata_ready),
            ("metadata_ready_semaphore_id", metadata_ready),
            ("matmul_chunk_available_semaphore_id", matmul_chunk_available),
            ("tilize_chunk_ready_semaphore_id", tilize_chunk_ready),
            ("matmul_chunk_ready_semaphore_id", matmul_chunk_ready),
            ("previous_chunk_sent_semaphore_id", previous_chunk_sent),
            ("initial_gather_semaphore_id", initial_gather),
            ("combine_sync_semaphore_id", tilize_combine_sync),
            ("combine_sync_noc_x", combine_phys[0].x),
            ("combine_sync_noc_y", combine_phys[0].y),
            ("compute_only", 1),
            ("local_output", 0),
            ("skip_zero_scores", 1 if self.skip_zero_scores else 0),
        ]
        tilize_compute_named = [
            ("tilize_input_cb_id", 8),
            ("tilize_output_cb_id", 0),
            ("total_chunks_cb_id", 3),
            ("tokens_per_chunk", TOKENS_PER_CHUNK),
            ("max_tiles_per_local_chunk", max_tiles_per_local_chunk),
            ("shared_cb_num_pages", shared_cb_num_pages),
        ]
        noc_max_burst_bytes = 16384
        matmul_named = [
            ("num_experts", E),
            ("num_shared_experts", 0),
            ("shared_expert_tp_factor", 1),
            ("layer_id", 0),  # runtime arg in this port; kept for the shared kernels' [[maybe_unused]] reads
            ("has_bias", 0),
            ("num_cores", n_ring),
            ("num_banks", n_ring),
            ("w2_pages_per_ring_core_total", w2_pages_per_ring_core_total),
            ("activation_function", self.activation),
            ("activation_limit_bits", _f32_bits(self.activation_limit or 0.0)),
            ("metadata_ready_semaphore_id", metadata_ready),
            ("matmul_chunk_ready_semaphore_id", matmul_chunk_ready),
            ("matmul_chunk_available_semaphore_id", matmul_chunk_available),
            ("per_expert_total_tokens_cb_id", 2),
            ("tokens_per_chunk", TOKENS_PER_CHUNK),
            ("tilize_drain_core_noc_x", drain_phys.x),
            ("tilize_drain_core_noc_y", drain_phys.y),
            ("combine_shard_width_tiles", combine_shard_width_tiles),
            ("tile_height", TILE),
            ("tile_width", TILE),
            ("tile_width_size_bytes", TILE * 2),
            ("token_expert_row_offset", rows_per_buffer),
            ("height_shard_dim", self.hs),
            ("width_shard_dim", self.ws),
            ("hidden_tiles", Ht),
            ("tiles_per_txn", tiles_per_txn),
            ("intermediate_tiles", Nt),
            ("noc_max_burst_bytes", noc_max_burst_bytes),
            ("matmul_combine_sync_semaphore_id", matmul_combine_sync),
            ("compute_only", 1),
            ("local_output", 0),
            ("total_tokens", tokens),
            ("e_t_entry_size", l1a),
            ("local_output_map_cb_id", 9),
            # this port
            ("prefetch_depth", self.prefetch_depth),
            ("txns_per_read", self.txns_per_read),
            ("w01_region_tiles", self.w01_tiles),
            ("w2_cb_id", 10),
            ("staging_layout", self.staging_layout),
            ("staging_expert_block_bytes", staging_expert_block_bytes),
        ]
        shard_map = "{" + "".join(f"{p.x}, {p.y}, " for p in combine_phys) + "}"
        tilize_ct = []
        for t in (tilize_input, indices, scores, mapping, counts, activation, e_t):
            tilize_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        matmul_ct = []
        for t in (w0_w1, w2, shared):
            matmul_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        # per-core tilize scalar args (after the 7 buffer addresses)
        tilize_scalars = []
        goff = 0
        for i in range(n_tilize):
            gather = tilize_phys[0] if i < primary_group else tilize_phys[primary_group]
            goff_group = goff if i < primary_group else goff - primary_bytes
            scal = [
                1 if i == 0 else 0,  # is_drain
                1 if i == primary_group else 0,  # is_secondary_mcaster
                gather.x,
                gather.y,
                goff,
                goff_group,
                primary_bytes if i < primary_group else secondary_bytes,
                subtoken_sizes[i],
                i * tokens_per_tilize_core,
                tokens if i == n_tilize - 1 else (i + 1) * tokens_per_tilize_core,
                i,
            ]
            for p in tilize_phys:
                scal += [p.x, p.y]
            tilize_scalars.append((scal, subtoken_sizes[i] // tile_width_bytes))
            goff += subtoken_sizes[i]

        matmul_scalars = []
        for b, core in enumerate(matmul):
            nb = matmul_phys[next_bank_of[b]]
            matmul_scalars.append(([b, vchannels[b]], [ring_sem, ring_pos_of[b], nb.x, nb.y] + shard_to_bank))

        return {
            "tokens": tokens,
            "tilize_named": tilize_named,
            "tilize_compute_named": tilize_compute_named,
            "matmul_named": matmul_named,
            "tilize_ct": tilize_ct,
            "matmul_ct": matmul_ct,
            "tilize_scalars": tilize_scalars,
            "matmul_scalars": matmul_scalars,
            "shard_map_define": shard_map,
            "sem_objs": sem_objs,
            "tilize_crs": tilize_crs,
            "matmul_crs": matmul_crs,
            "all_worker_crs": all_worker_crs,
            "shared_cb_num_pages": shared_cb_num_pages,
            "idx_aligned": idx_aligned,
            "sc_aligned": sc_aligned,
            "map_aligned": map_aligned,
            "mapping_pages": mapping_pages,
            "e_t_page": e_t_page,
            "max_subtoken": max_subtoken,
            "a2a_cb_pages": a2a_cb_pages,
            "n_ring": n_ring,
            "n_tilize": n_tilize,
            "mesh_rows": mesh_rows,
            "mesh_cols": mesh_cols,
        }

    # ------------------------------------------------------------------------------------------------ program
    def _program(
        self, plan, tilize_input, indices, scores, mapping, w0_w1, w2, counts, activation, e_t, shared, layer_id
    ):
        E = self.E
        l1a = self.l1_alignment
        tokens = plan["tokens"]
        tilize_crs, matmul_crs, all_crs = plan["tilize_crs"], plan["matmul_crs"], plan["all_worker_crs"]
        C = self.cores
        tilize, matmul = C["tilize"], C["matmul"]

        cbs = []
        # c_0 / c_1: the whole-grid shared tensor (tilize output, chunk input, staging), tile pages
        shared_total = plan["shared_cb_num_pages"] * BF16_TILE_BYTES
        for idx in (0, 1):
            d = ttnn.CBDescriptor(
                shared_total,
                shared.memory_config().shard_spec.grid,
                [ttnn.CBFormatDescriptor(idx, ttnn.bfloat16, BF16_TILE_BYTES)],
            )
            d.set_buffer_from_cb(ttnn.cb_descriptor_from_sharded_tensor(idx, shared))
            cbs.append(d)
        # c_2: per-expert token counts (every worker core)
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(2, counts, 0, int(counts.buffer_page_size()), all_crs))
        # tilize CBs
        u32 = ttnn.uint32
        small_page = max(4, l1a, 16)
        cbs.append(_cb(7, tilize_crs, plan["e_t_page"], E, u32))
        cbs.append(
            ttnn.cb_descriptor_from_sharded_tensor(
                4, indices, 0, plan["idx_aligned"] * int(indices.buffer_num_pages()), tilize_crs
            )
        )
        cbs.append(
            ttnn.cb_descriptor_from_sharded_tensor(
                6, scores, 0, plan["sc_aligned"] * int(scores.buffer_num_pages()), tilize_crs
            )
        )
        cbs.append(_cb(5, tilize_crs, plan["map_aligned"], plan["mapping_pages"], ttnn.uint16))
        cbs.append(_cb(8, tilize_crs, plan["max_subtoken"], TOKENS_PER_CHUNK, ttnn.bfloat16))
        cbs.append(_cb(9, tilize_crs, _align((2 * E + 1) * 4, l1a), tokens, u32))
        cbs.append(_cb(10, tilize_crs, _ceil_div(tokens, 2) * l1a * E, 1, u32))
        cbs.append(_cb(11, tilize_crs, 4 * E, 1, u32))
        cbs.append(_cb(12, tilize_crs, _align((2 * E + 1) * 4, l1a) * _ceil_div(tokens, 2), 1, u32))
        cbs.append(_cb(13, tilize_crs, 4, 1, u32))
        cbs.append(_cb(14, tilize_crs, _align((E + 1) * 4, l1a), plan["n_tilize"] - 1, u32))
        cbs.append(_cb(3, tilize_crs, small_page, 1, u32))
        # matmul CBs: resident W0/W1 and W2 regions, ready flags, a2a buffer, metadata
        cbs.append(_cb(3, matmul_crs, BF4_TILE_BYTES, self.w01_tiles, ttnn.bfloat4_b))
        cbs.append(_cb(10, matmul_crs, BF4_TILE_BYTES, self.w2_tiles, ttnn.bfloat4_b))
        cbs.append(_cb(4, matmul_crs, small_page, 1, ttnn.float32))
        cbs.append(_cb(5, matmul_crs, small_page, 1, ttnn.float32))
        cbs.append(_cb(6, matmul_crs, BF16_TILE_BYTES, plan["a2a_cb_pages"] * plan["n_ring"], ttnn.bfloat16))
        cbs.append(_cb(7, matmul_crs, small_page, 2, u32))

        tilize_addrs = [
            int(t.buffer_address()) for t in (tilize_input, indices, scores, mapping, counts, activation, e_t)
        ]
        tilize_rt = ttnn.RuntimeArgs()
        tilize_compute_rt = ttnn.RuntimeArgs()
        for (x, y), (scal, tiles_local) in zip(tilize, plan["tilize_scalars"]):
            tilize_rt[x][y] = tilize_addrs + scal
            tilize_compute_rt[x][y] = [tiles_local]
        matmul_rt = ttnn.RuntimeArgs()
        for (x, y), (head, tail) in zip(matmul, plan["matmul_scalars"]):
            matmul_rt[x][y] = (
                head
                + [int(w0_w1.buffer_address()), int(w2.buffer_address()), int(shared.buffer_address())]
                + tail
                + [int(layer_id)]
            )

        dm_cfg = lambda proc, noc, dyn: ttnn.DataMovementConfigDescriptor(
            processor=proc, noc=noc, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC if dyn else ttnn.NOC_MODE.DM_DEDICATED_NOC
        )
        O2 = ttnn._ttnn.operations.experimental.fabric_mux.KernelBuildOptLevel.O2

        def make_program(linearized_coord):
            named = [(k, linearized_coord if k == "linearized_mesh_coord" else v) for k, v in plan["tilize_named"]]
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "tilize_reader_moe_compute.cpp"),
                    core_ranges=tilize_crs,
                    compile_time_args=plan["tilize_ct"],
                    named_compile_time_args=named,
                    runtime_args=tilize_rt,
                    opt_level=O2,
                    config=dm_cfg(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1, True),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "tilize_writer_moe_compute.cpp"),
                    core_ranges=tilize_crs,
                    compile_time_args=plan["tilize_ct"],
                    named_compile_time_args=named,
                    runtime_args=tilize_rt,
                    opt_level=O2,
                    config=dm_cfg(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1, True),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "tilize_compute_moe_compute.cpp"),
                    core_ranges=tilize_crs,
                    named_compile_time_args=plan["tilize_compute_named"],
                    runtime_args=tilize_compute_rt,
                    config=ttnn.ComputeConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "dm0_moe_compute.cpp"),
                    core_ranges=matmul_crs,
                    compile_time_args=plan["matmul_ct"],
                    named_compile_time_args=plan["matmul_named"],
                    runtime_args=matmul_rt,
                    config=dm_cfg(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0, False),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "dm1_moe_compute.cpp"),
                    core_ranges=matmul_crs,
                    compile_time_args=plan["matmul_ct"],
                    named_compile_time_args=plan["matmul_named"],
                    defines=[("OUTPUT_SHARD_CORE_MAP", plan["shard_map_define"])],
                    runtime_args=matmul_rt,
                    config=dm_cfg(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1, False),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "compute_moe_compute.cpp"),
                    core_ranges=matmul_crs,
                    compile_time_args=plan["matmul_ct"],
                    named_compile_time_args=plan["matmul_named"],
                    runtime_args=matmul_rt,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=self.math_fidelity,
                        math_approx_mode=True,
                        fp32_dest_acc_en=self.fp32_dest_acc_en,
                        dst_full_sync_en=False,
                        bfp8_pack_precise=False,
                    ),
                ),
            ]
            return ttnn.ProgramDescriptor(kernels=kernels, semaphores=plan["sem_objs"], cbs=cbs)

        rows, cols = plan["mesh_rows"], plan["mesh_cols"]
        if rows * cols == 1:
            return make_program(0)
        mesh_programs = {}
        for r in range(rows):
            for c in range(cols):
                coord = ttnn.MeshCoordinate(r, c)
                mesh_programs[ttnn.MeshCoordinateRange(coord, coord)] = make_program(r * cols + c)
        return ttnn.MeshProgramDescriptor(mesh_programs)

    # ------------------------------------------------------------------------------------------------ golden
    @staticmethod
    def golden(
        tokens_torch,
        indices,
        scores,
        local_expert_ids,
        w0,
        w1,
        w2,
        activation=MoEActivationFunction.GELU,
        skip_zero_scores=False,
    ):
        """torch reference of one device's expert outputs. ``tokens_torch`` [T, H] (bf16 or float), ``indices`` /
        ``scores`` [T, K], ``local_expert_ids`` the global ids of this device's experts (slot order), ``w0`` / ``w1``
        [E, H, N], ``w2`` [E, N, H] in float (bf4-rounded). Returns ``(rows, lists)``: ``rows[e]`` the ``[n_e, H]``
        float outputs in the device's token order (ascending token id) and ``lists[e]`` the ``[(token, k)]`` list."""
        x = tokens_torch.float()
        T, K = indices.shape
        lists = [[] for _ in local_expert_ids]
        slot = {gid: e for e, gid in enumerate(local_expert_ids)}
        for t in range(T):
            for k in range(K):
                gid = int(indices[t, k])
                if gid not in slot:
                    continue
                if skip_zero_scores and float(scores[t, k]) == 0.0:
                    continue
                lists[slot[gid]].append((t, k))
        rows = []
        for e, lst in enumerate(lists):
            if not lst:
                rows.append(torch.zeros(0, x.shape[1]))
                continue
            xe = x[[t for t, _ in lst]]
            g = xe @ w0[e].float()
            u = xe @ w1[e].float()
            if _activation_value(activation) == int(MoEActivationFunction.GELU.value):
                h = F.gelu(g, approximate="tanh") * u
            elif _activation_value(activation) == int(MoEActivationFunction.SILU.value):
                h = F.silu(g) * u
            else:
                raise NotImplementedError("golden covers GELU and SILU")
            h = h.bfloat16().float()  # the device packs the intermediate to bf16 before the W2 matmul
            rows.append(h @ w2[e].float())
        return rows, lists

    # ------------------------------------------------------------------------------------------------ host views
    def staging_rows(self, shared_torch, counts_torch, tokens, layout=None):
        """The staged rows of every expert from the host copy of the shared tensor: ``shared_torch``
        ``[num_cores * 64, H]`` (``ttnn.to_torch`` of output 3), ``counts_torch`` the per-expert counts of any
        core, ``tokens`` the call's T. Returns ``rows[e]`` ``[n_e, H]`` bf16 in the expert's token order (dm1 splits
        an expert's tokens over the hs height-shard cores in order, hs cores x ws width slices)."""
        layout = self.staging_layout if layout is None else STAGING_LAYOUTS[layout]
        g = self.cores["grid"]
        grid_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g[0] - 1, g[1] - 1))])
        core_index = {(c.x, c.y): i for i, c in enumerate(ttnn.corerange_to_cores(grid_crs, row_wise=True))}
        shard_rows = DOUBLE_BUFFER * TOKENS_PER_CHUNK
        seg = self.hidden // self.ws
        segs_per_core = shard_rows * self.ws  # segment rows (of seg elements) per core shard
        shards = shared_torch.reshape(-1, shard_rows, self.hidden)
        rows_per_buffer = (shard_rows // 2) * self.ws
        out = []
        for e in range(self.E):
            n = int(counts_torch[e])
            base = (e & 1) * rows_per_buffer if layout == 0 else e * (tokens // self.hs)
            chunk, rem = n // self.hs, n % self.hs
            full = torch.zeros(n, self.hidden, dtype=torch.bfloat16)
            r = 0
            for h in range(self.hs):
                nh = chunk + (1 if h < rem else 0)
                for w in range(self.ws):
                    ci = core_index[self.cores["combine"][h * self.ws + w]]
                    segs = shards[ci].reshape(segs_per_core, seg)
                    full[r : r + nh, w * seg : (w + 1) * seg] = segs[base : base + nh]
                r += nh
            out.append(full)
        return out

    def combined_from_staging(self, shared_torch, counts_torch, e_t_torch, tokens, layout=None):
        """``[K, T, H]`` bf16 like the shared op's combine output: every staged row placed at (k slot, token) using
        the e_t entries (word 0 token id, word 1 k slot); rows with no route stay zero."""
        rows = self.staging_rows(shared_torch, counts_torch, tokens, layout)
        out = torch.zeros(self.K, tokens, self.hidden, dtype=torch.bfloat16)
        entry_words = _align(4, self.l1_alignment) // 4
        e_t_torch = e_t_torch.reshape(self.E, -1)
        for e in range(self.E):
            n = int(counts_torch[e])
            for i in range(n):
                t = int(e_t_torch[e, i * entry_words])
                k = int(e_t_torch[e, i * entry_words + 1])
                out[k, t] = rows[e][i]
        return out


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _cb(index, crs, page_size, num_pages, dtype):
    return ttnn.CBDescriptor(page_size * num_pages, crs, [ttnn.CBFormatDescriptor(index, dtype, page_size)])


def _f32_bits(v: float) -> int:
    import struct

    return struct.unpack("<I", struct.pack("<f", float(v)))[0]

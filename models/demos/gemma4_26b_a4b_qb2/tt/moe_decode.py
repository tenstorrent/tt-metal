# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 routed-expert MoE decode (WS4 prototype; kernels in ttnn/cpp/ttnn/operations/experimental/moe_decode).

Two programs per call, both reading only the k routed experts' weights:

* gate|up: one core per (routing slot s, column pair j): reads x row 0, the slot's expert id and routing weight from
  the packed routing row (``RouterTopk.op_packed``: ``[1, 1, 32, 2k]`` bf16, row 0 = weights then ids), streams that
  expert's gate column j and up column j (2 * kt tiles, contiguous), and writes ``h = gelu(x Wg_j) * (x Wu_j) * w_s``
  as one 1x32 tile of the h buffer.
* down: one core per output column: concatenates the k experts' h rows (k * kts tiles) and their down tiles of the
  column, so one custom_mm over K = k * kts gives ``sum_s w_s * h_s Wd_s``; writes row 0 of the output tile.

Weights are DRAM width-sharded over the banks (layouts below) so cores read large contiguous chunks and every bank
serves the same bytes for any routing. Per chip, tensor-parallel experts: ``I_chip`` intermediate
columns (a multiple of 32) of every expert.
"""

from pathlib import Path

import torch

import ttnn

TILE = 32
TINY_BYTES = 2 * TILE
KERNELS = Path(__file__).resolve().parents[4] / "ttnn/cpp/ttnn/operations/experimental/moe_decode/device/kernels"
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}


def gate_up_layout(gate, up, banks):
    """``gate`` / ``up`` ``[E, H, I]`` -> ``[rows, banks * 32]``: column block b (DRAM bank b) holds K slice b
    (``kt / banks`` tiles) of every (expert, pair) as ``[gate slice | up slice]`` tiles."""
    E, H, inter = gate.shape
    kt, pairs = H // TILE, inter // TILE
    assert kt % banks == 0
    ks = kt // banks
    tiles = [w.reshape(E, banks, ks, TILE, pairs, TILE).permute(1, 0, 4, 2, 3, 5) for w in (gate, up)]
    # [banks, E, pairs, gate/up, ks, 32, 32]
    blocks = torch.stack(tiles, dim=3).reshape(banks, E * pairs * 2 * ks * TILE, TILE)
    return torch.cat(list(blocks), dim=1)


def down_layout(down, banks):
    """``down`` ``[E, I, H]`` -> ``[rows, banks * 32]``: column block b holds output columns
    ``[b * cpb, (b + 1) * cpb)`` of every expert as ``[expert][column][k]`` tiles."""
    E, inter, H = down.shape
    kts, cols = inter // TILE, H // TILE
    assert cols % banks == 0
    cpb = cols // banks
    tiles = down.reshape(E, kts, TILE, banks, cpb, TILE).permute(3, 0, 4, 1, 2, 5)  # [banks, E, cpb, kts, 32, 32]
    return torch.cat(list(tiles.reshape(banks, E * cpb * kts * TILE, TILE)), dim=1)


def upload_per_chip(mesh, per_chip, dtype):
    """One ``[rows, banks * 32]`` layout per chip -> a mesh tensor ``[1, 1, rows, banks * 32]``, DRAM width-sharded
    with one ``[rows, 32]`` shard per bank."""
    banks = mesh.dram_grid_size().x
    rows = per_chip[0].shape[0]
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (rows, TILE),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    return ttnn.from_torch(
        torch.stack(per_chip)[:, None],
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=memory,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def tp_expert_layouts(state, *, tp, banks, inter_pad):
    """Checkpoint experts (``experts.gate_up_proj [E, 2I, H]``, ``experts.down_proj [E, H, I]``) -> per chip, the
    tensor-parallel slice of the intermediate dimension (I padded to ``inter_pad``, split ``tp`` ways) in the
    bank-sharded gate|up and down layouts."""
    gate, up = state["experts.gate_up_proj"].chunk(2, dim=-2)
    pad = inter_pad - gate.shape[-2]
    gate, up = (torch.nn.functional.pad(t.transpose(-2, -1), (0, pad)) for t in (gate, up))  # [E, H, Ip]
    down = torch.nn.functional.pad(state["experts.down_proj"].transpose(-2, -1), (0, 0, 0, pad))  # [E, Ip, H]
    gates, ups, downs = gate.chunk(tp, dim=-1), up.chunk(tp, dim=-1), down.chunk(tp, dim=-2)
    gu = [gate_up_layout(g.contiguous(), u.contiguous(), banks) for g, u in zip(gates, ups)]
    dn = [down_layout(d.contiguous(), banks) for d in downs]
    return gu, dn


class MoeDecodeB1:
    def __init__(self, mesh, *, hidden, inter_chip, k, weight_dtype=ttnn.bfloat8_b, down_dtype=None, grid=None):
        assert hidden % TILE == 0 and inter_chip % TILE == 0 and (4 * k) % 32 == 0
        self.mesh, self.hidden, self.inter = mesh, hidden, inter_chip
        self.kt, self.pairs, self.kts, self.cols, self.k = (
            hidden // TILE,
            inter_chip // TILE,
            inter_chip // TILE,
            hidden // TILE,
            k,
        )
        self.dtype, self.tile_bytes = weight_dtype, TILE_BYTES[weight_dtype]
        self.down_dtype = down_dtype or weight_dtype
        self.down_tile_bytes = TILE_BYTES[self.down_dtype]
        g = mesh.compute_with_storage_grid_size()
        self.gx, self.gy = grid or (int(g.x), int(g.y))
        assert self.k * self.pairs <= self.gx * self.gy and self.cols <= self.gx * self.gy
        self.route_page = 2 * k * 2
        self.banks = mesh.dram_grid_size().x
        self.bank_adjacent_down = True

    @staticmethod
    def _coalesce(cores):
        """Core list -> CoreRangeSet merged into rectangles: dispatch writes kernel binaries per range, and per-core
        ranges made the multi-core programs slow to launch under trace."""
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
        for c in cores[1:]:
            core_set = core_set.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
        return core_set

    def _cores(self, n):
        cores = [ttnn.CoreCoord(i % self.gx, i // self.gx) for i in range(n)]
        return cores, self._coalesce(cores)

    @staticmethod
    def _cb(index, dtype, page, pages, cores, tiny=False):
        kwargs = dict(buffer_index=index, data_format=dtype, page_size=page)
        if tiny:
            kwargs["tile"] = ttnn.TileDescriptor(ttnn.Tile([1, TILE]))
        return ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=cores, format_descriptors=[ttnn.CBFormatDescriptor(**kwargs)]
        )

    @staticmethod
    def _kernels(name, cores, reader, writer, compute, compute_rt=None):
        import os

        defines = [("SKIP_WEIGHTS", "1")] if os.environ.get("MOE_DECODE_SKIP_WEIGHTS") == "1" else []
        dm = ttnn.DataMovementConfigDescriptor
        return [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / f"{name}_reader.cpp"),
                core_ranges=cores,
                compile_time_args=reader[0],
                runtime_args=reader[1],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / f"{name}_writer.cpp"),
                core_ranges=cores,
                compile_time_args=writer[0],
                runtime_args=writer[1],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / f"{name}_compute.cpp"),
                core_ranges=cores,
                compile_time_args=compute,
                runtime_args=compute_rt if compute_rt is not None else [],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]

    def gate_up(self, x, route, w_gu):
        device = x.device()
        h = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.k * self.pairs, TILE]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        cores, core_set = self._cores(self.k * self.pairs)
        cb_x, cb_w, cb_route, cb_scr, cb_sync, cb_out = 0, 1, 2, 3, 4, 16
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, TINY_BYTES, self.kt, core_set, tiny=True),
            self._cb(cb_w, self.dtype, self.tile_bytes, 2 * self.kt, core_set),
            self._cb(cb_route, ttnn.bfloat16, self.route_page, 1, core_set),
            self._cb(cb_scr, ttnn.bfloat16, self.route_page, 1, core_set),
            self._cb(cb_sync, ttnn.bfloat16, 32, 1, core_set),
            self._cb(cb_out, ttnn.bfloat16, TINY_BYTES, 1, core_set, tiny=True),
        ]
        reader_ct = [cb_w, cb_route, self.kt, self.pairs, self.k, self.tile_bytes, self.route_page, self.banks, cb_sync]
        reader_ct += ttnn.TensorAccessorArgs(route).get_compile_time_args()
        writer_ct = [cb_x, cb_out, cb_scr, self.kt, self.pairs, self.k, self.route_page, cb_w, cb_sync]
        writer_ct += [self.tile_bytes, self.banks]
        for t in (x, h, route):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, core in enumerate(cores):
            slot, pair = divmod(i, self.pairs)
            reader_rt[core.x][core.y] = [w_gu.buffer_address(), route.buffer_address(), slot, pair]
            writer_rt[core.x][core.y] = [
                x.buffer_address(),
                h.buffer_address(),
                route.buffer_address(),
                slot,
                pair,
                w_gu.buffer_address(),
            ]
        compute_rt = ttnn.RuntimeArgs()
        for i, core in enumerate(cores):
            compute_rt[core.x][core.y] = [i % self.banks]  # the reader's first bank: (slot * pairs + pair) % banks
        kernels = self._kernels(
            "gate_up",
            core_set,
            (reader_ct, reader_rt),
            (writer_ct, writer_rt),
            [cb_x, cb_w, cb_out, self.kt, self.banks],
            compute_rt,
        )
        ttnn.generic_op([w_gu, x, route, h], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return h

    def down(self, h, route, w_down, out=None):
        device = h.device()
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 1, self.hidden]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
            )
        if self.bank_adjacent_down:
            # column b * cpb + r on the r-th reader of bank b, on the bank's row (gpt-oss row_placement)
            from models.demos.gpt_oss.tt.fused_decode.stream import row_placement

            per_bank = self.cols // self.banks
            placed = row_placement(self.mesh, per_bank)
            cores = [placed[b][r] for b in range(self.banks) for r in range(per_bank)]
            core_set = self._coalesce(cores)
        else:
            cores, core_set = self._cores(self.cols)
        kt = self.k * self.kts
        cb_in0, cb_w, cb_route, cb_scr, cb_sync, cb_out = 0, 1, 2, 3, 4, 16
        cbs = [
            self._cb(cb_in0, ttnn.bfloat16, TINY_BYTES, kt, core_set, tiny=True),
            self._cb(cb_w, self.down_dtype, self.down_tile_bytes, kt, core_set),
            self._cb(cb_route, ttnn.bfloat16, self.route_page, 1, core_set),
            self._cb(cb_scr, ttnn.bfloat16, self.route_page, 1, core_set),
            self._cb(cb_sync, ttnn.bfloat16, 32, 1, core_set),
            self._cb(cb_out, ttnn.bfloat16, TINY_BYTES, 1, core_set, tiny=True),
        ]
        reader_ct = [
            cb_in0,
            cb_w,
            cb_route,
            self.kts,
            self.k,
            self.cols,
            self.down_tile_bytes,
            self.route_page,
            self.banks,
        ]
        reader_ct += [cb_sync]
        for t in (route, h):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [cb_out, cb_w, cb_sync, cb_scr, self.kts, self.k, self.cols, self.down_tile_bytes, self.route_page]
        writer_ct += [self.banks]
        for t in (out, route):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for col, core in enumerate(cores):
            reader_rt[core.x][core.y] = [w_down.buffer_address(), route.buffer_address(), h.buffer_address(), col]
            writer_rt[core.x][core.y] = [out.buffer_address(), col, w_down.buffer_address(), route.buffer_address()]
        kernels = self._kernels(
            "down", core_set, (reader_ct, reader_rt), (writer_ct, writer_rt), [cb_in0, cb_w, cb_out, kt]
        )
        ttnn.generic_op([w_down, route, h, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out

    def __call__(self, x, route, w_gu, w_down):
        return self.down(self.gate_up(x, route, w_gu), route, w_down)


class MoeDecodeB1Fused(MoeDecodeB1):
    """gate|up and down as ONE program on the whole compute grid (kernels fused_*.cpp): the down cores prefetch
    their weights while the gate|up units stream theirs; each gate|up unit keeps its h tile in cb_hout and increments
    the down cores' semaphore, and the down cores pull the tiles. Measured equal to the two-program form (DRAM-bound
    at ~330 GB/s with cross-bank gate|up reads); multicasting the h tiles from 48 sources serialized (93 us)."""

    def __call__(self, x, route, w_gu, w_down, out=None):
        from models.demos.gpt_oss.tt.fused_decode.stream import row_placement

        device = x.device()
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 1, self.hidden]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
            )
        gx, gy = self.gx, self.gy
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        per_bank = self.cols // self.banks
        placed = row_placement(self.mesh, per_bank)
        down = [placed[b][r] for b in range(self.banks) for r in range(per_bank)]  # down[col]
        down_set = {(c.x, c.y) for c in down}
        all_cores = [ttnn.CoreCoord(x, y) for y in range(gy) for x in range(gx)]
        free = [c for c in all_cores if (c.x, c.y) not in down_set]
        gu_cores = (free + down)[: self.k * self.pairs]  # free cores first, then share down cores
        gu_of = {(c.x, c.y): i for i, c in enumerate(gu_cores)}
        col_of = {(c.x, c.y): i for i, c in enumerate(down)}

        kdown = self.k * self.kts
        cb_x, cb_w, cb_hout, cb_h, cb_dw, cb_out, cb_route, cb_scr = 0, 1, 2, 3, 4, 16, 5, 6
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, TINY_BYTES, self.kt, grid, tiny=True),
            self._cb(cb_w, self.dtype, self.tile_bytes, 2 * self.kt, grid),
            self._cb(cb_hout, ttnn.bfloat16, TINY_BYTES, 1, grid, tiny=True),
            self._cb(cb_h, ttnn.bfloat16, TINY_BYTES, kdown, grid, tiny=True),
            self._cb(cb_dw, self.down_dtype, self.down_tile_bytes, kdown, grid),
            self._cb(cb_out, ttnn.bfloat16, TINY_BYTES, 1, grid, tiny=True),
            self._cb(cb_route, ttnn.bfloat16, self.route_page, 1, grid),
            self._cb(cb_scr, ttnn.bfloat16, self.route_page, 1, grid),
        ]
        h_sem = 0
        sems = [ttnn.SemaphoreDescriptor(h_sem, ttnn.CoreType.WORKER, grid, 0)]
        noc = [device.worker_core_from_logical_core(c) for c in down]
        mcast = [0, 0, 0, 0]  # unused (the h tiles are pulled by the down cores)
        reader_ct = [cb_w, cb_dw, cb_h, cb_route, self.kt, self.pairs, self.k, self.kts, self.cols, self.tile_bytes]
        reader_ct += [self.route_page, self.banks, h_sem, cb_hout] + ttnn.TensorAccessorArgs(
            route
        ).get_compile_time_args()
        writer_ct = [cb_x, cb_hout, cb_h, cb_out, cb_scr, self.kt, self.pairs, self.k, self.route_page, len(down)]
        writer_ct += [h_sem, gx * gy]
        for t in (x, out, route):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_x, cb_w, cb_hout, cb_h, cb_dw, cb_out, self.kt, self.banks, kdown]
        down_noc = [v for n in noc for v in (n.x, n.y)]
        gu_noc = [v for c in gu_cores for n in (device.worker_core_from_logical_core(c),) for v in (n.x, n.y)]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c in all_cores:
            key = (c.x, c.y)
            unit = gu_of.get(key)
            is_gu = int(unit is not None)
            slot, pair = divmod(unit, self.pairs) if is_gu else (0, 0)
            col = col_of.get(key)
            has_down = int(col is not None)
            col = col or 0
            reader_rt[c.x][c.y] = [
                w_gu.buffer_address(),
                w_down.buffer_address(),
                route.buffer_address(),
                is_gu,
                slot,
                pair,
                has_down,
                col,
            ] + gu_noc
            writer_rt[c.x][c.y] = (
                [x.buffer_address(), out.buffer_address(), route.buffer_address(), is_gu, slot, pair, has_down, col]
                + mcast
                + down_noc
            )
            compute_rt[c.x][c.y] = [is_gu, (slot * self.pairs + pair) % self.banks, has_down]
        import os

        defines = [("FUSED_NO_SYNC", "1")] if os.environ.get("MOE_FUSED_NO_SYNC") == "1" else []
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "fused_reader.cpp"),
                core_ranges=grid,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "fused_writer.cpp"),
                core_ranges=grid,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "fused_compute.cpp"),
                core_ranges=grid,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        ttnn.generic_op([w_gu, w_down, x, route, out], program)
        return out

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MLP_GU_FUSED split-SwiGLU: hidden = silu(gate) * up from ONE fused [gate|up] decode matmul output.

The fused decode MLP (tt/mlp.py, QWEN36_MLP_GU_FUSED=1d|dsmc) reads the prefill AGMM weight's column order, so its
matmul output is tile-pair interleaved per tile row: [g0 u0 g1 u1 ... g(NP-1) u(NP-1)] (prepare_for_fused_swiglu with
gate first). This one generic_op turns that [.., M, 2*NP*32] bf16 tensor into the [.., M, NP*32] hidden the down
projection reads, in place of the r5 tail (SiLU fused into the gate matmul, a separate up matmul, then ttnn.mul).

Program (kernels/swiglu_pair_{reader,compute,writer}.cpp), on up to ``max_cores`` cores, contiguous output tiles per
core: the reader fetches each output tile's gate and up tiles through a TensorAccessor (any input layout: the 1D
matmul's L1-interleaved output or the DRAM-sharded pipeline's width-sharded one), compute unpacks both to DST and runs
silu_tile + mul_binary_tile (fp32 with fp32_dest_acc_en, one rounding to bf16 at pack), the writer stores the tile in
the output tensor (L1 interleaved by default, the 1D down-projection's in0 layout).

Numerics against r5: the gate's SiLU now runs on the bf16-rounded matmul output (r5 ran it on the fp32 accumulator
inside the gate matmul) and the product is fp32 (r5: a bf16 FPU multiply). Not bit-identical; the error is one bf16
rounding either way. The matmul accumulation itself is unchanged when the fused 1D matmul keeps r5's in0_block_w.
"""

from pathlib import Path

import ttnn

_KERNELS = Path(__file__).resolve().parent / "kernels"
READER_KERNEL = str(_KERNELS / "swiglu_pair_reader.cpp")
COMPUTE_KERNEL = str(_KERNELS / "swiglu_pair_compute.cpp")
WRITER_KERNEL = str(_KERNELS / "swiglu_pair_writer.cpp")

_CB_IN, _CB_OUT = 0, 16
_TILE = 32
_MAX_BATCH = 8  # pairs per reader push: the input CB holds 2 * batch tiles


def split_work(total, cores):
    """[(start, count)] for ``cores`` cores over ``total`` output tiles: contiguous ranges, counts differ by <= 1."""
    cores = max(1, min(int(cores), int(total)))
    base, rem = divmod(int(total), cores)
    out, start = [], 0
    for i in range(cores):
        n = base + (1 if i < rem else 0)
        out.append((start, n))
        start += n
    return out


def pair_tile_ids(o, n_pairs, pair=True):
    """Host mirror of the reader's index math: (gate tile id, up tile id) of output tile ``o`` (row-major)."""
    row, p = divmod(o, n_pairs)
    base = row * 2 * n_pairs
    g = base + 2 * p if pair else base + p
    return g, (g + 1 if pair else g + n_pairs)


class SwiGLUSplit:
    """silu(gate) * up of a fused [gate|up] TILE tensor, one generic_op. No device state: every call allocates its own
    output. ``pair``: tile-pair interleaved input (the model's layout); False reads [gate half | up half]."""

    def __init__(self, mesh_device, pair=True, max_cores=None):
        self.mesh = mesh_device
        self.pair = bool(pair)
        grid = mesh_device.compute_with_storage_grid_size()
        self.grid = ttnn.CoreCoord(grid.x, grid.y)
        cap = grid.x * grid.y
        self.max_cores = cap if not max_cores else max(1, min(int(max_cores), cap))
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
        )
        self._plans = {}

    def supports(self, gu):
        """None if ``gu`` can be split, else the reason."""
        shape = tuple(int(s) for s in gu.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 1:
            return f"shape {shape} is not [1,1,M,W]"
        if gu.layout != ttnn.TILE_LAYOUT:
            return "not TILE layout"
        if gu.dtype != ttnn.bfloat16:
            return f"dtype {gu.dtype} (need bfloat16)"
        if shape[3] % (2 * _TILE) != 0:
            return f"width {shape[3]} is not a whole number of gate/up tile pairs"
        return None

    def _plan(self, mt, n_pairs):
        key = (mt, n_pairs)
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        work = split_work(mt * n_pairs, self.max_cores)
        n = len(work)
        crs = ttnn.num_cores_to_corerangeset(n, self.grid, True)  # row-wise: core i = (i % grid.x, i // grid.x)
        coords = [(i % self.grid.x, i // self.grid.x) for i in range(n)]
        batch = min(_MAX_BATCH, max(c for _, c in work))
        plan = dict(work=work, crs=crs, coords=coords, batch=batch)
        self._plans[key] = plan
        return plan

    def __call__(self, gu, memory_config=ttnn.L1_MEMORY_CONFIG):
        why = self.supports(gu)
        assert why is None, f"SwiGLUSplit: {why}"
        rows, width = int(gu.shape[2]), int(gu.shape[3])
        mt = int(gu.padded_shape[2]) // _TILE
        n_pairs = width // (2 * _TILE)
        p = self._plan(mt, n_pairs)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, rows, n_pairs * _TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, memory_config
        )
        tile_bytes = 2 * _TILE * _TILE  # bf16
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * p["batch"] * tile_bytes,
                core_ranges=p["crs"],
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=_CB_IN, data_format=ttnn.bfloat16, page_size=tile_bytes)
                ],
            ),
            ttnn.CBDescriptor(
                total_size=2 * tile_bytes,
                core_ranges=p["crs"],
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=_CB_OUT, data_format=ttnn.bfloat16, page_size=tile_bytes)
                ],
            ),
        ]
        rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (x, y), (start, count) in zip(p["coords"], p["work"]):
            rd[x][y] = [start, count]
            cp[x][y] = [count]
            wr[x][y] = [start, count]
        reader = ttnn.KernelDescriptor(
            kernel_source=READER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=p["crs"],
            compile_time_args=[_CB_IN, int(self.pair), n_pairs, p["batch"]]
            + list(ttnn.TensorAccessorArgs(gu).get_compile_time_args()),
            common_runtime_args=[gu.buffer_address()],
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=COMPUTE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=p["crs"],
            compile_time_args=[_CB_IN, _CB_OUT],
            runtime_args=cp,
            config=self.compute_config,
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=WRITER_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=p["crs"],
            compile_time_args=[_CB_OUT] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            common_runtime_args=[out.buffer_address()],
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        )
        desc = ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=cbs)
        return ttnn.generic_op([gu, out], desc)

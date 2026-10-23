# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gr_write``: the gated-residual write as one program.  ``residual [1, 4, rows, 640] + block_output [1, 1, rows,
640] * injection [1, 1, rows, 4]`` (bf16 TILE) in place of the chain ``reshape / permute -> multiply -> add`` of
``Qwen38TTNNGatedResidual.write`` / ``write_rows`` (3 programs per write, 96 writes per decode step).

The compute kernel issues the chain's two binary_ng programs on the same operands in the same order
(``kernels/compute.cpp``): ``ttnn.multiply`` = SFPU (``fast_and_approximate_mode`` defaults to False for multiply):
the coefficient column broadcast (the chain's SCALAR / COL broadcast, exact), ``mul_binary_tile`` (fp32 product,
software RNE to bf16, 0 * x = 0), pack bf16; ``ttnn.add`` = FPU (the binding's default is True for add):
``add_tiles(residual, update)`` in the 16-bit dest, pack.  Tolerance class BITWISE.  Rows 1..32 = one tile row (decode,
lanes, the MTP verify body, the 32-row chunk); the 128-row chunk keeps the chain.  One output tile per core (80 cores,
one core rectangle) unless ``QWEN38_FUSED_GR_WRITE_CORES`` groups the tiles column-major onto fewer cores.
"""

from __future__ import annotations

import functools
import os

import ttnn

from .. import program as fp
from ..registry import BITWISE, FusedKernel, GateSpec, register

NAME = "gr_write"
BRANCHES = 4
HIDDEN = 2560
LOCAL_HIDDEN = HIDDEN // 4
HIDDEN_TILES = LOCAL_HIDDEN // fp.TILE  # 20
UNITS = BRANCHES * HIDDEN_TILES  # 80 output tiles
CORES_ENV = "QWEN38_FUSED_GR_WRITE_CORES"
KERNELS = {name: fp.kernel_source(NAME, f"{name}.cpp") for name in ("reader", "compute", "writer")}
CBS = (("cb_block", 0), ("cb_res", 1), ("cb_coef", 2), ("cb_upd", 3), ("cb_out", 16))
CB_PAGES = 2
CB_INDEX = dict(CBS)
NAMED = dict(CBS, hidden_tiles=HIDDEN_TILES, branches=BRANCHES)
READER_ARGS = ("block_addr", "residual_addr", "injection_addr", "first_unit", "units")
WRITER_ARGS = ("out_addr", "first_unit", "units")
COMPUTE_ARGS = ("units",)


def _expect(tensor, shape: tuple[int, ...], label: str) -> None:
    got = tuple(int(v) for v in tensor.shape)
    if got != shape or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(
            f"gr_write {label} must be bf16 TILE {list(shape)}, got {tensor.dtype} {tensor.layout} {list(got)}"
        )


def rows_of(block_output, residual, injection) -> int:
    """The valid rows (1..32, one tile row) of the three operands, whose shapes must agree."""

    rows = int(residual.shape[2])
    if not 1 <= rows <= fp.TILE or int(residual.padded_shape[2]) != fp.TILE:
        raise ValueError(
            f"gr_write takes one tile row (1..{fp.TILE} rows), got {rows} padded {int(residual.padded_shape[2])}"
        )
    _expect(residual, (1, BRANCHES, rows, LOCAL_HIDDEN), "residual")
    _expect(block_output, (1, 1, rows, LOCAL_HIDDEN), "block output")
    _expect(injection, (1, 1, rows, BRANCHES), "injection")
    return rows


def cores_of(mesh, environ=None) -> int:
    grid = mesh.compute_with_storage_grid_size()
    value = (os.environ if environ is None else environ).get(CORES_ENV)
    return min(UNITS, grid.x * grid.y) if not value else int(value)


def gr_write_program(block_output, residual, injection, output, *, cores: int) -> "ttnn.ProgramDescriptor":
    work = fp.split_work(UNITS, residual.device(), cores=cores)
    grid = fp.core_rectangle(work, residual.device())
    cbs = [fp.cb_descriptor(index, ttnn.bfloat16, fp.TILE_BYTES[ttnn.bfloat16], CB_PAGES, grid) for _name, index in CBS]
    reader = fp.reader_kernel(
        KERNELS["reader"],
        grid,
        fp.accessor_args(block_output) + fp.accessor_args(residual) + fp.accessor_args(injection),
        [
            (
                w.core,
                [
                    block_output.buffer_address(),
                    residual.buffer_address(),
                    injection.buffer_address(),
                    w.start,
                    w.count,
                ],
            )
            for w in work
        ],
        named=NAMED,
    )
    writer = fp.writer_kernel(
        KERNELS["writer"],
        grid,
        fp.accessor_args(output),
        [(w.core, [output.buffer_address(), w.start, w.count]) for w in work],
        named=NAMED,
    )
    compute = fp.compute_kernel(
        KERNELS["compute"], grid, [], [(w.core, [w.count]) for w in work], named=NAMED, fp32_dest=False
    )
    return fp.program_descriptor([reader, writer, compute], cbs=cbs)


def gr_write(block_output, residual, injection, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """The new residual ``[1, 4, rows, 640]`` bf16 TILE."""

    rows = rows_of(block_output, residual, injection)
    mesh = residual.device()
    output = fp.allocate((1, BRANCHES, rows, LOCAL_HIDDEN), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, memory_config)
    cores = cores_of(mesh)
    # the block output once per branch tile, the residual and the coefficient tile once per core, the residual out;
    # the multiply and the add per element of the four branches
    meta = fp.program_meta(
        NAME,
        "write",
        rows,
        reads=(residual,),
        writes=(output,),
        dram_bytes=BRANCHES * fp.tensor_bytes(block_output) + cores * fp.tensor_bytes(injection),
        flops=2 * rows * BRANCHES * LOCAL_HIDDEN,
        cores=cores,
    )
    fp.run_program(
        [block_output, residual, injection, output],
        gr_write_program(block_output, residual, injection, output, cores=cores),
        meta=meta,
    )
    output.update_tensor_topology(residual.tensor_topology())  # generic_op leaves the allocation's placement
    return output


def gr_write_composed(block_output, residual, injection, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """The chain op for op: ``write`` (reshape, one scalar per branch tile) at rows = 1, ``write_rows`` (permute, one
    column per branch tile) at rows > 1."""

    rows = rows_of(block_output, residual, injection)
    if rows == 1:
        coefficient = ttnn.reshape(injection, (1, BRANCHES, 1, 1))
    else:
        coefficient = ttnn.permute(injection, (0, 3, 2, 1), memory_config=memory_config)
    update = ttnn.multiply(block_output, coefficient, memory_config=memory_config)
    ttnn.deallocate(coefficient)
    output = ttnn.add(residual, update, memory_config=memory_config)
    ttnn.deallocate(update)
    return output


def write_fused(module, block_output, state):
    """``Qwen38TTNNGatedResidual.write`` on the fused program (the module's checks around ``gr_write``)."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    module._validate_block(block_output)
    module._validate_residual(state.residual)
    module.mesh_contract.validate_tensor(state.injection, placement=TensorPlacement.REPLICATED)
    output = gr_write(block_output, state.residual, state.injection)
    module._validate_residual(output)
    return output


def write_rows_fused(module, block_rows, state):
    """``Qwen38TTNNGatedResidual.write_rows`` on the fused program for one tile row; more rows keep the chain."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    if int(state.residual.shape[2]) > fp.TILE:
        return type(module).write_rows(module, block_rows, state)
    module.mesh_contract.validate_tensor(state.injection, placement=TensorPlacement.REPLICATED)
    output = gr_write(block_rows, state.residual, state.injection)
    module.mesh_contract.validate_tensor(output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    return output


@functools.lru_cache(maxsize=None)
def _oracle(layer: int):
    """The CPU GR oracle of layer ``layer``'s attention block from the checkpoint at ``QWEN38_CHECKPOINT``."""

    root = os.environ.get("QWEN38_CHECKPOINT")
    if not root:
        raise RuntimeError("the gr_write gate computes the injection with the CPU oracle: set QWEN38_CHECKPOINT")
    from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
    from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
    from models.demos.blackhole.qwen38_flash_next.tt.gr import Qwen38GatedResidual, Qwen38GatedResidualWeights

    checkpoint = Qwen38Checkpoint(root)
    placement = Qwen38Placement(checkpoint.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
    weights = Qwen38GatedResidualWeights.from_checkpoint(checkpoint, placement, layer_index=layer, block="attn")
    return Qwen38GatedResidual(weights)


def _gate_inputs(mesh, capture, positions, layer):
    """Device slice 0 of layer ``layer``'s attention write: the captured residual entering the layer, the captured
    attention output, and the injection the CPU oracle derives from that residual."""

    import torch

    index = [capture["positions"].index(p) for p in positions]
    rows = len(positions)
    residual = capture["residual"][index, layer - 1].to(torch.bfloat16).reshape(rows, HIDDEN * BRANCHES)
    block = capture["attn_out"][index, layer].to(torch.bfloat16).reshape(rows, HIDDEN)
    _block_input, state = _oracle(layer).read(residual)

    def upload(host):
        return ttnn.from_torch(
            host.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    branches = residual.reshape(rows, BRANCHES, HIDDEN)[:, :, :LOCAL_HIDDEN].permute(1, 0, 2)
    return {
        "block_output": upload(block[:, :LOCAL_HIDDEN].reshape(1, 1, rows, LOCAL_HIDDEN)),
        "residual": upload(branches.reshape(1, BRANCHES, rows, LOCAL_HIDDEN)),
        "injection": upload(state.injection.to(torch.bfloat16).reshape(1, 1, rows, BRANCHES)),
    }


def _gate_output(result):
    """``[rows, 4 * 640]`` (branch-major) host bf16."""

    return ttnn.to_torch(result).permute(0, 2, 1, 3).reshape(-1, BRANCHES * LOCAL_HIDDEN)


register(
    FusedKernel(
        name=NAME,
        replaces="reshape or permute, multiply, add (the 3-program GR write; 96 writes per step)",
        tolerance=BITWISE,
        fused=gr_write,
        composed=gr_write_composed,
        gate=GateSpec(inputs=_gate_inputs, output=_gate_output, reference=None, layers=tuple(range(1, 48))),
    )
)

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``shared_expert``: the MoE shared expert's nine programs per layer as three.

The composed chain (``Qwen38TTNNMoE._shared_partial``): the gate, up and scalar-gate DRAM-sharded linears on the
five-core hidden shard, ``silu``, ``mul``, the down linear, the scalar's drain to DRAM, ``sigmoid`` and the partial
``x sigmoid`` multiply.  Fused: ONE DRAM-sharded linear over the concatenated ``[gate | up | scalar | 0]`` weight
(every output column keeps its K sequence and K-block reload points, so each column is the separate linear's column
bit for bit), one eltwise program on the five storage cores (``kernels/compute_shared_eltwise.cpp``: silu, the
product into the down linear's input shard, and the sigmoid plus its column broadcast on one core) and the down
linear.  The ``x sigmoid`` multiply and the add of the routed partial run in ``moe_post``, so the fused form returns
the ungated down-linear shard and the broadcast sigmoid tile.  Tolerance class BITWISE.  Rows 1..32.
"""

from __future__ import annotations

import torch

import ttnn

from .. import program as fp
from ..registry import BITWISE, FusedKernel, GateSpec, register
from ...decode_matmul import dram_sharded_matmul_configs, dram_sharded_weight_memory_config

NAME = "shared_expert"
HIDDEN = 2560
INTERMEDIATE = 640
DEVICES = 4
LOCAL_INTERMEDIATE = INTERMEDIATE // DEVICES  # 160: five tiles, one per storage core
GATE_TILES = LOCAL_INTERMEDIATE // fp.TILE
SCALAR_COLUMN = 2 * LOCAL_INTERMEDIATE  # column 320 of the concatenated output
CAT_WIDTH = SCALAR_COLUMN + fp.TILE  # 352: gate | up | scalar padded to a tile
STORAGE_CORES = 5
KERNELS = {name: fp.kernel_source(NAME, f"{name}_shared_eltwise.cpp") for name in ("reader", "compute", "writer")}
BF16 = ttnn.bfloat16
CBS = (
    ("cb_gate", 0, BF16, 1),
    ("cb_up", 1, BF16, 1),
    ("cb_silu", 2, BF16, 1),
    ("cb_scalar", 3, BF16, 1),
    ("cb_sig", 4, BF16, 1),
    ("cb_inter", 16, BF16, 1),
    ("cb_sig_bcast", 17, BF16, 1),
)
CB_INDEX = {name: index for name, index, _dtype, _pages in CBS}
READER_ARGS = ("gate_up_scalar_ws", "tile", "has_scalar")
WRITER_ARGS = ("intermediate", "sigmoid", "tile", "has_scalar")


def concat_shared_weights(
    gate_t: torch.Tensor, up_t: torch.Tensor, scalar_t: torch.Tensor, devices: int = DEVICES
) -> torch.Tensor:
    """``[K, devices * 352]``: per device ``d`` the block ``[gate_t[:, 160d:160d+160] | up_t[:, ...] | scalar_t | 0 x 31]``
    (``gate_t`` / ``up_t`` the ``[K, 640]`` transposed linears, ``scalar_t`` the ``[K, 1]`` transposed scalar gate), so a
    width shard over ``devices`` hands each device its own concatenated weight."""

    k = gate_t.shape[0]
    local = gate_t.shape[1] // devices
    if gate_t.shape != up_t.shape or scalar_t.shape != (k, 1) or local * devices != gate_t.shape[1]:
        raise ValueError(
            f"shared weights must be [K, {devices} x local] x2 and [K, 1], got {tuple(gate_t.shape)} {tuple(scalar_t.shape)}"
        )
    pad = torch.zeros(k, fp.TILE - 1, dtype=gate_t.dtype)
    blocks = [
        torch.cat([gate_t[:, d * local : (d + 1) * local], up_t[:, d * local : (d + 1) * local], scalar_t, pad], dim=1)
        for d in range(devices)
    ]
    return torch.cat(blocks, dim=1).contiguous()


def shared_eltwise_program(gate_up_scalar_ws, intermediate, sigmoid) -> "ttnn.ProgramDescriptor":
    cores_list = [ttnn.CoreCoord(c, 0) for c in range(STORAGE_CORES)]  # the intermediate shard's storage cores
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(cores_list[0], cores_list[-1])])
    named = [(name, index) for name, index, _dtype, _pages in CBS] + [("gate_tiles", GATE_TILES)]
    cbs = [fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, cores) for _name, index, dtype, pages in CBS]
    reader = fp.reader_kernel(
        KERNELS["reader"],
        cores,
        fp.accessor_args(gate_up_scalar_ws),
        [(core, [gate_up_scalar_ws.buffer_address(), c, int(c == 0)]) for c, core in enumerate(cores_list)],
        named=named,
    )
    writer = fp.writer_kernel(
        KERNELS["writer"],
        cores,
        [*fp.accessor_args(intermediate), *fp.accessor_args(sigmoid)],
        [
            (core, [intermediate.buffer_address(), sigmoid.buffer_address(), c, int(c == 0)])
            for c, core in enumerate(cores_list)
        ],
        named=named,
    )
    compute = fp.compute_kernel(
        KERNELS["compute"],
        cores,
        [],
        [(core, [int(c == 0)]) for c, core in enumerate(cores_list)],
        named=named,
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=False,
    )
    return fp.program_descriptor([reader, writer, compute], cbs=cbs)


def shared_expert(
    hidden,
    gate_up_scalar,
    down,
    *,
    gate_up_scalar_program_config,
    down_program_config,
    intermediate_memory_config,
    compute_kernel_config,
    **_composed_only,
):
    """``(partial, sigmoid)``: the down linear's ungated ``[1, 1, rows, 2560]`` bf16 L1 width shard and the
    ``[1, 1, 32, 32]`` bf16 TILE whose row ``r`` holds ``sigmoid(scalar_r)`` in every column (DRAM)."""

    rows = fp.rows_of(hidden)
    mesh = hidden.device()
    gate_up_scalar_ws = ttnn.linear(
        hidden,
        gate_up_scalar,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=gate_up_scalar_program_config,
        compute_kernel_config=compute_kernel_config,
    )
    if tuple(int(v) for v in gate_up_scalar_ws.shape) != (1, 1, rows, CAT_WIDTH):
        raise RuntimeError(
            f"concatenated shared linear produced {list(gate_up_scalar_ws.shape)}, expected [1, 1, {rows}, {CAT_WIDTH}]"
        )
    intermediate = fp.allocate(
        (1, 1, rows, LOCAL_INTERMEDIATE), BF16, ttnn.TILE_LAYOUT, mesh, intermediate_memory_config
    )
    sigmoid = fp.allocate((1, 1, fp.TILE, fp.TILE), BF16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
    # the concatenated linear's L1 shard in, the intermediate shard (L1) and the sigmoid tile out; silu and product
    # per element of the rows x 160 intermediate, the sigmoid on the scalar column and its broadcast
    meta = fp.program_meta(
        NAME,
        "eltwise",
        rows,
        reads=(gate_up_scalar_ws,),
        writes=(intermediate, sigmoid),
        flops=rows * LOCAL_INTERMEDIATE * 3 + rows * fp.TILE,
        cores=STORAGE_CORES,
        outputs=((intermediate, None), (sigmoid, None)),
    )
    fp.run_program(
        [gate_up_scalar_ws, intermediate, sigmoid],
        shared_eltwise_program(gate_up_scalar_ws, intermediate, sigmoid),
        meta=meta,
    )
    ttnn.deallocate(gate_up_scalar_ws)
    partial = ttnn.linear(
        intermediate,
        down,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=down_program_config,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(intermediate)
    return partial, sigmoid


def shared_expert_composed(
    hidden,
    gate,
    up,
    scalar,
    down,
    *,
    gate_up_program_config,
    scalar_program_config,
    down_program_config,
    intermediate_memory_config,
    compute_kernel_config,
    **_fused_only,
):
    """The composed chain for one row tile, op for op as ``Qwen38TTNNMoE._shared_partial`` issues it: the gated
    ``[1, 1, rows, 2560]`` bf16 partial in DRAM."""

    l1_ws = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    dram = ttnn.DRAM_MEMORY_CONFIG
    gate_ws = ttnn.linear(
        hidden,
        gate,
        memory_config=l1_ws,
        program_config=gate_up_program_config,
        compute_kernel_config=compute_kernel_config,
    )
    up_ws = ttnn.linear(
        hidden,
        up,
        memory_config=l1_ws,
        program_config=gate_up_program_config,
        compute_kernel_config=compute_kernel_config,
    )
    gate_activated = ttnn.silu(gate_ws, memory_config=intermediate_memory_config)
    intermediate = ttnn.mul(gate_activated, up_ws, memory_config=intermediate_memory_config)
    ttnn.deallocate(gate_ws)
    ttnn.deallocate(gate_activated)
    ttnn.deallocate(up_ws)
    partial = ttnn.linear(
        intermediate,
        down,
        memory_config=l1_ws,
        program_config=down_program_config,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn.deallocate(intermediate)
    scalar_ws = ttnn.linear(
        hidden,
        scalar,
        memory_config=l1_ws,
        program_config=scalar_program_config,
        compute_kernel_config=compute_kernel_config,
    )
    scalar_dram = ttnn.to_memory_config(scalar_ws, dram)
    ttnn.deallocate(scalar_ws)
    scalar_gate = ttnn.sigmoid(scalar_dram, memory_config=dram)
    gated = ttnn.mul(partial, scalar_gate, memory_config=dram)
    ttnn.deallocate(partial)
    ttnn.deallocate(scalar_dram)
    ttnn.deallocate(scalar_gate)
    return gated


def gated_partial_rows(result) -> torch.Tensor:
    """``[rows, 2560]`` fp32 of the gated shared partial; the fused pair is gated here with the chain's own multiply
    (its fused form runs inside ``moe_post``)."""

    if isinstance(result, tuple):
        partial, sigmoid = result
        rows = fp.rows_of(partial)
        gate = ttnn.slice(sigmoid, (0, 0, 0, 0), (1, 1, rows, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result = ttnn.mul(partial, gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
    return ttnn.to_torch(result).float().reshape(-1, HIDDEN)


def gate_configs(mesh) -> dict:
    """The five-core decode configs both forms take (the model's ``Qwen38TTNNMoE.__init__`` values)."""

    hidden_act, gate_up = dram_sharded_matmul_configs(mesh, HIDDEN, LOCAL_INTERMEDIATE, num_cores=STORAGE_CORES)
    intermediate, down = dram_sharded_matmul_configs(mesh, LOCAL_INTERMEDIATE, HIDDEN, num_cores=STORAGE_CORES)
    _, scalar = dram_sharded_matmul_configs(mesh, HIDDEN, 1, num_cores=STORAGE_CORES)
    _, cat = dram_sharded_matmul_configs(mesh, HIDDEN, CAT_WIDTH, num_cores=STORAGE_CORES)
    return {
        "hidden_act_memory_config": hidden_act,
        "gate_up_program_config": gate_up,
        "scalar_program_config": scalar,
        "gate_up_scalar_program_config": cat,
        "down_program_config": down,
        "intermediate_memory_config": intermediate,
        "compute_kernel_config": ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    }


def upload_weights(mesh, gate_t, up_t, scalar_t, down_t, *, device: int = 0) -> dict:
    """Device ``device``'s shared weights on one chip: the separate ``gate`` / ``up`` / ``scalar`` / ``down`` and the
    concatenated ``gate_up_scalar``, DRAM width-sharded as the model uploads them (``gate_t`` / ``up_t`` ``[K, 640]``,
    ``scalar_t`` ``[K, 1]``, ``down_t`` ``[640, N]``: the transposed checkpoint linears)."""

    def upload(value, k, n):
        return ttnn.from_torch(
            value.to(torch.bfloat16).reshape(1, 1, k, n).contiguous(),
            dtype=BF16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=dram_sharded_weight_memory_config(mesh, k, n),
        )

    lo, hi = device * LOCAL_INTERMEDIATE, (device + 1) * LOCAL_INTERMEDIATE
    cat = concat_shared_weights(gate_t, up_t, scalar_t)[:, device * CAT_WIDTH : (device + 1) * CAT_WIDTH]
    return {
        "gate": upload(gate_t[:, lo:hi], HIDDEN, LOCAL_INTERMEDIATE),
        "up": upload(up_t[:, lo:hi], HIDDEN, LOCAL_INTERMEDIATE),
        "scalar": upload(scalar_t, HIDDEN, 1),
        "down": upload(down_t[lo:hi], LOCAL_INTERMEDIATE, HIDDEN),
        "gate_up_scalar": upload(cat, HIDDEN, CAT_WIDTH),
    }


def _gate_inputs(mesh, capture, positions, layer):
    """The captured MoE input rows in the five-core hidden shard with deterministic bf16 weights (seeded by the layer;
    the class is bitwise between the two forms, so the values are free)."""

    index = [capture["positions"].index(p) for p in positions]
    rows = len(positions)
    hidden = capture["mlp_in"][index, layer].to(torch.bfloat16).reshape(1, 1, rows, HIDDEN)
    generator = torch.Generator().manual_seed(20260914 + layer)
    gate_t = torch.randn(HIDDEN, INTERMEDIATE, generator=generator) * 0.02
    up_t = torch.randn(HIDDEN, INTERMEDIATE, generator=generator) * 0.02
    scalar_t = torch.randn(HIDDEN, 1, generator=generator) * 0.02
    down_t = torch.randn(INTERMEDIATE, HIDDEN, generator=generator) * 0.02
    configs = gate_configs(mesh)
    hidden_dram = ttnn.from_torch(
        hidden, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    hidden_ws = ttnn.to_memory_config(hidden_dram, configs.pop("hidden_act_memory_config"))
    ttnn.deallocate(hidden_dram)
    return {"hidden": hidden_ws, **upload_weights(mesh, gate_t, up_t, scalar_t, down_t), **configs}


register(
    FusedKernel(
        name=NAME,
        replaces="shared gate/up/scalar linears x3 -> 1, silu, mul, scalar drain, sigmoid, partial x sigmoid (9 programs per layer -> 3)",
        tolerance=BITWISE,
        fused=shared_expert,
        composed=shared_expert_composed,
        gate=GateSpec(inputs=_gate_inputs, output=gated_partial_rows, reference=None, layers=tuple(range(48))),
    )
)

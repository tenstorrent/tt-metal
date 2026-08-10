# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Focused generic-op grid comparison for the K=7168, N=2048 MoE SwiGLU shape.

The public operation is a standard C++ device operation.  This benchmark deliberately uses the
retained Python ProgramDescriptor and ``ttnn.generic_op`` so grid/dispatch experiments do not need
a C++ rebuild.  Both comparison runs therefore execute the same kernels and host geometry code.

Examples (Blackhole p150, Tracy-enabled build)::

    MOE_GRID=11x8 MOE_DISPATCH_AXIS=col \
      scripts/run_safe_pytest.sh --profile --no-precompile <this file>

    MOE_GRID=12x8 MOE_DISPATCH_AXIS=row \
      scripts/run_safe_pytest.sh --profile --no-precompile <this file>

    MOE_GRID=12x8 MOE_DISPATCH_AXIS=row MOE_LOGICAL_TRANSPOSE=1 \
      scripts/run_safe_pytest.sh --profile --no-precompile <this file>

The defaults are BF16 row-major activations, BFP4 tiled ND-sharded weights, BFP8 tiled output,
K=7168, N=2048, capacity 5120, and three measured repetitions for each M.  ``MOE_GRID_COUNTS`` and
``MOE_GRID_REPS`` can narrow the sweep.  The manifest maps profiler rows to their device-resident M;
``parse_perf_matrix.py`` refuses to report if that mapping and the CSV length disagree.
"""

import json
import os

import pytest
import torch

import ttnn
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import (
    nd_shard_n_tiles,
    weight_memory_configs,
)
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import (
    Blocking,
    create_program_descriptor,
    make_mailbox,
)


TILE = 32
EMB = int(os.environ.get("MOE_GRID_EMB", 7168))
HIDDEN = int(os.environ.get("MOE_GRID_HIDDEN", 2048))
CAPACITY = int(os.environ.get("MOE_GRID_CAPACITY", 5120))
COUNTS = [int(value) for value in os.environ.get("MOE_GRID_COUNTS", "0,64,128,256,512,1024,2048,4096,5120").split(",")]
REPS = int(os.environ.get("MOE_GRID_REPS", 3))
WARMUP = max(1, int(os.environ.get("MOE_GRID_WARMUP", 1)))
READ_PROFILER_EVERY = int(os.environ.get("MOE_GRID_PROFILER_READ_EVERY", 3))
MANIFEST = os.environ.get("MOE_GRID_MANIFEST", "/tmp/moe_grid_perf_manifest.json")

GRID = tuple(int(value) for value in os.environ.get("MOE_GRID", "11x8").lower().split("x"))
if len(GRID) != 2:
    raise ValueError(f"MOE_GRID must be columnsxrows, got {os.environ.get('MOE_GRID')!r}")
TRANSPOSE_GRID_AXES = os.environ.get("MOE_LOGICAL_TRANSPOSE", "0") not in ("0", "false", "False")

DISPATCH_AXIS = os.environ.get("MOE_DISPATCH_AXIS", "col").strip().lower()
if DISPATCH_AXIS == "row":
    DEVICE_PARAMS = {
        "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "fabric_tensix_config": ttnn.FabricTensixConfig.MUX,
    }
elif DISPATCH_AXIS == "col":
    DEVICE_PARAMS = {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}
else:
    raise ValueError(f"MOE_DISPATCH_AXIS must be 'col' or 'row', got {DISPATCH_AXIS!r}")

NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS = 256, 8
LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 3, 137
BFP4_TILE = ttnn.tile_size(ttnn.bfloat4_b)


def _to_device(tensor, dtype, layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def _count_tensor(count, device):
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    return _to_device(counts, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)


def _idx_tensor(device):
    idx = torch.tensor(
        [(11 + 37 * local) % NUM_GLOBAL_EXPERTS for local in range(NUM_LOCAL_EXPERTS)],
        dtype=torch.int32,
    )
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    return _to_device(idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)


def _compute_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
        bfp8_pack_precise=True,
    )


def _read_bytes(count):
    weights = 3 * (EMB * HIDDEN // 1024) * BFP4_TILE
    return weights + count * EMB * 2


def _weight_memory_configs(device):
    logical_columns, logical_rows = (GRID[1], GRID[0]) if TRANSPOSE_GRID_AXES else GRID
    blocking = Blocking(logical_columns, logical_rows, EMB, HIDDEN, m_t_max=1)
    dram = device.dram_grid_size()
    bank_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])

    def memory_config(n_tiles):
        return ttnn.MemoryConfig(
            ttnn.BufferType.DRAM,
            ttnn.NdShardSpec(
                shard_shape=ttnn.Shape([TILE, n_tiles * TILE]),
                grid=bank_grid,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    return memory_config(blocking.hn_pad), memory_config(blocking.wd_ec_max)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_generic_grid_perf(mesh_device):
    available = mesh_device.compute_with_storage_grid_size()
    assert GRID[0] <= int(available.x) and GRID[1] <= int(available.y), (
        f"requested {GRID[0]}x{GRID[1]} workers with {DISPATCH_AXIS.upper()} dispatch, "
        f"but the device exposes {available.x}x{available.y}"
    )
    assert CAPACITY % TILE == 0
    assert all(0 <= count <= CAPACITY for count in COUNTS)

    torch.manual_seed(42)
    x = torch.randn((1, 1, CAPACITY, EMB), dtype=torch.bfloat16)
    tt_x = _to_device(x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh_device)
    del x

    gate_up_mc, down_mc = (
        _weight_memory_configs(mesh_device)
        if TRANSPOSE_GRID_AXES
        else weight_memory_configs(mesh_device, EMB, HIDDEN, core_grid=GRID)
    )
    tt_weights = [
        _to_device(torch.randn(shape, dtype=torch.bfloat16), ttnn.bfloat4_b, ttnn.TILE_LAYOUT, mesh_device, mc)
        for shape, mc in zip(
            ((EMB, HIDDEN), (EMB, HIDDEN), (HIDDEN, EMB)),
            (gate_up_mc, gate_up_mc, down_mc),
        )
    ]
    shard_widths = [nd_shard_n_tiles(weight) for weight in tt_weights]
    assert all(width > 0 for width in shard_widths), f"generic reader does not see ND shards: {shard_widths}"

    tt_counts = {count: _count_tensor(count, mesh_device) for count in COUNTS}
    tt_idx = _idx_tensor(mesh_device)
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, CAPACITY, EMB]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    mailbox = make_mailbox(mesh_device, int(available.x) * int(available.y))
    config = _compute_config()
    manifest = []

    def dispatch(count, rep, warmup):
        descriptor = create_program_descriptor(
            tt_x,
            *tt_weights,
            tt_counts[count],
            tt_idx,
            tt_output,
            mailbox,
            local_expert_id=LOCAL_EXPERT_ID,
            input_m_tiles=CAPACITY // TILE,
            compute_kernel_config=config,
            core_grid=GRID,
            transpose_grid_axes=TRANSPOSE_GRID_AXES,
        )
        ttnn.generic_op(
            [tt_x, *tt_weights, tt_counts[count], tt_idx, mailbox, tt_output],
            descriptor,
        )
        if TRANSPOSE_GRID_AXES:
            grid_label = f"{GRID[0]}x{GRID[1]}@{GRID[1]}H-by-{GRID[0]}K"
        else:
            grid_label = f"{GRID[0]}x{GRID[1]}"
        manifest.append(
            {
                "op": "moe_fused_swiglu",
                "implementation": "generic_op",
                "dispatch_axis": DISPATCH_AXIS,
                "logical_transpose": TRANSPOSE_GRID_AXES,
                "format": "bf16_rm",
                "wplace": "nd_shard",
                "weight_dtype": "bfp4",
                "w_tile": BFP4_TILE,
                "grid": grid_label,
                "emb": EMB,
                "hidden": HIDDEN,
                "capacity": CAPACITY,
                "count": count,
                "rep": rep,
                "warmup": warmup,
                "read_bytes": _read_bytes(count),
            }
        )

    warmup_count = COUNTS[len(COUNTS) // 2]
    for warmup in range(WARMUP):
        dispatch(warmup_count, warmup, True)
    ttnn.ReadDeviceProfiler(mesh_device)

    since_read = 0
    for count in COUNTS:
        for rep in range(REPS):
            dispatch(count, rep, False)
            since_read += 1
            if since_read >= READ_PROFILER_EVERY:
                ttnn.ReadDeviceProfiler(mesh_device)
                since_read = 0
        print(
            f"[grid_perf] generic {GRID[0]}x{GRID[1]}/{DISPATCH_AXIS} "
            f"logical={'8H-by-12K' if TRANSPOSE_GRID_AXES else 'physical'} "
            f"K={EMB} N={HIDDEN} "
            f"M={count} read_MB={_read_bytes(count) / 1e6:.3f}",
            flush=True,
        )
    ttnn.ReadDeviceProfiler(mesh_device)

    with open(MANIFEST, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    print(
        f"[grid_perf] manifest={MANIFEST} dispatches={len(manifest)} "
        f"worker_grid={GRID[0]}x{GRID[1]} available_grid={available.x}x{available.y} "
        f"logical_transpose={TRANSPOSE_GRID_AXES} "
        f"shard_widths={shard_widths}",
        flush=True,
    )

    for tensor in (tt_x, *tt_weights, *tt_counts.values(), tt_idx, tt_output, mailbox):
        ttnn.deallocate(tensor)

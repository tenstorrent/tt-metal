# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dev probes for the KDA SP state-sharding design. Not a CI test.

* ``test_kda_slab_dram_per_chip``: per-chip DRAM bytes of the K3 KDA slabs (``KdaStates``) and of one
  native scratch carry on Galaxy SP8xTP4, measured as allocator deltas, plus the DRAM left free per chip.
* ``test_convolution_tail_reshape_is_view``: the row-major ``[1, SEG, 192] <-> [3, SEG/3, 192]`` reshape keeps
  the buffer (a metadata view, no device program).
* ``test_sp_gather_partition_trace``: ``ttnn.all_gather(output_tensor=...)`` over SP writes the persistent buffer
  and ``ttnn.mesh_partition`` keeps each SP row's head slice, both under trace capture and replay with new input,
  for the recurrent ``[1, 3, 128, 128]`` FP32 shard and the convolution ``[3, 6, 192]`` BF16 shard (branch-major
  view, gathered on the head-half dim). Run each through ``scripts/run_safe_pytest.sh``.
"""

from __future__ import annotations

import json

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.utils import kda_device_params
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaStates, allocate_native_state, deallocate_state
from models.demos.deepseek_v3_d_p.tt.kimi_k3.kda_geometry import kimi_k3_kda_geometry

_MESH = (8, 4)
_SP_AXIS, _TP_AXIS = 0, 1
_SP, _TP = _MESH
_HEADS_PER_TP = KimiK3Config.KDA_NUM_HEADS // _TP  # 24
_H_STORE = _HEADS_PER_TP // _SP  # 3
_D = KimiK3Config.KDA_HEAD_DIM  # 128
_CONV_ROW = 192  # one 384-byte BF16 page per (branch, head, half)
_GALAXY = pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            _MESH,
            kda_device_params(_MESH, trace_region_size=8 * 1024 * 1024),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=_MESH, topology="mesh-8x4"),
            id="SP8xTP4",
        )
    ],
    indirect=True,
)


def _dram(mesh_device) -> dict[str, int]:
    """Per-chip DRAM (all banks of one chip; every chip allocates the same addresses)."""
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return {
        "banks": view.num_banks,
        "total": view.total_bytes_per_bank * view.num_banks,
        "allocated": view.total_bytes_allocated_per_bank * view.num_banks,
        "free": view.total_bytes_free_per_bank * view.num_banks,
        "largest_free_per_bank": view.largest_contiguous_bytes_free_per_bank,
    }


@_GALAXY
@pytest.mark.parametrize(
    "num_layers, num_slots",
    [(18, 1), (18, 8), (69, 1)],
    ids=["stage18L-1slot", "stage18L-8slots", "model69L-1slot"],
)
def test_kda_slab_dram_per_chip(mesh_device, device_params, num_layers, num_slots) -> None:
    geometry = kimi_k3_kda_geometry(tuple(mesh_device.shape), _SP_AXIS, _TP_AXIS)
    before = _dram(mesh_device)
    scratch = allocate_native_state(mesh_device, geometry)
    after_scratch = _dram(mesh_device)
    states = KdaStates.allocate(
        mesh_device, geometry, sp_axis=_SP_AXIS, layer_ids=tuple(range(num_layers)), num_slots=num_slots
    )
    after_slabs = _dram(mesh_device)
    try:
        # Each chip stores its _H_STORE heads per layer and slot, plus one stored layer of import staging.
        recurrent_logical = num_slots * num_layers * _H_STORE * _D * _D * 4
        convolution_logical = num_slots * num_layers * 3 * _H_STORE * 2 * _CONV_ROW * 2
        staging_logical = _H_STORE * _D * _D * 4 + 3 * _H_STORE * 2 * _CONV_ROW * 2
        slab_bytes = after_slabs["allocated"] - after_scratch["allocated"]
        result = {
            "num_layers": num_layers,
            "num_slots": num_slots,
            "local_heads": geometry.local_heads,
            "recurrent_slab_shape": list(states.recurrent.shape),
            "convolution_slab_shape": list(states.convolution.shape),
            "slab_bytes_per_chip": slab_bytes,
            "slab_logical_bytes_per_chip": recurrent_logical + convolution_logical,
            "staging_logical_bytes_per_chip": staging_logical,
            "slab_bytes_per_slot_per_chip": (slab_bytes - staging_logical) / num_slots,
            "replicated_slab_logical_bytes_per_chip": (recurrent_logical + convolution_logical) * _SP,
            "scratch_bytes_per_chip": after_scratch["allocated"] - before["allocated"],
            "dram_before": before,
            "dram_after": after_slabs,
        }
        print("KDA_SLAB_DRAM=" + json.dumps(result, sort_keys=True))
        # Allocation granularity aside, the slabs and staging are their logical bytes. The ND recurrent slab rounds its
        # 16 KiB shards up to whole rows over the banks (69 layers x 3 heads x 4 bands = 828 shards over 8 banks: 64 KiB
        # of padding); the three other buffers round to 2 KiB per bank.
        slack = before["banks"] * _D * 32 * 4 + 3 * before["banks"] * 2 * 1024
        assert abs(slab_bytes - (recurrent_logical + convolution_logical + staging_logical)) <= slack
        assert (
            abs(
                result["scratch_bytes_per_chip"] - (_HEADS_PER_TP * _D * _D * 4 + 3 * _HEADS_PER_TP * 2 * _CONV_ROW * 2)
            )
            <= 2 * before["banks"] * 2 * 1024
        )
    finally:
        states.deallocate()
        deallocate_state(scratch)


@_GALAXY
def test_convolution_tail_reshape_is_view(mesh_device, device_params) -> None:
    native = ttnn.from_torch(
        torch.randn(1, 3 * _HEADS_PER_TP * 2, _CONV_ROW, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    branch_major = ttnn.reshape(native, (3, _HEADS_PER_TP * 2, _CONV_ROW))
    back = ttnn.reshape(branch_major, (1, 3 * _HEADS_PER_TP * 2, _CONV_ROW))
    addresses = [t.buffer_address() for t in (native, branch_major, back)]
    print("KDA_CONV_RESHAPE_ADDRESSES=" + json.dumps(addresses))
    assert len(set(addresses)) == 1, f"reshape allocated a new buffer: {addresses}"
    assert torch.equal(
        ttnn.to_torch(ttnn.get_device_tensors(branch_major)[0]).reshape(-1),
        ttnn.to_torch(ttnn.get_device_tensors(native)[0]).reshape(-1),
    )


def _shard_to_mesh(mesh_device, full: torch.Tensor, sp_dim: int, dtype, layout) -> ttnn.Tensor:
    """``full`` is ``[TP, ...]``: TP column c takes ``full[c:c+1]``, SP row r its ``r``-th slice of ``sp_dim``."""
    return ttnn.from_torch(
        full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=_MESH, dims=(sp_dim, 0)),
    )


def _per_device(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]


@_GALAXY
def test_sp_gather_partition_trace(mesh_device, device_params) -> None:
    # Recurrent: [TP, 24, D, D] FP32 TILE, shard [1, 3, D, D]; convolution: [TP, 3, 48, 192] BF16 ROW_MAJOR (the
    # branch-major view of the native [1, 144, 192] tail), shard [1, 3, 6, 192].
    kinds = {
        "recurrent": ((_TP, _HEADS_PER_TP, _D, _D), torch.float32, ttnn.float32, ttnn.TILE_LAYOUT),
        "convolution": ((_TP, 3, _HEADS_PER_TP * 2, _CONV_ROW), torch.bfloat16, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    }
    gather_dim = {"recurrent": 1, "convolution": 2}
    inputs, outputs, golden = {}, {}, {}
    for name, (shape, torch_dtype, dtype, layout) in kinds.items():
        golden[name] = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
        inputs[name] = _shard_to_mesh(mesh_device, golden[name], gather_dim[name], dtype, layout)
        outputs[name] = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, *shape[1:]]), dtype, layout, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )

    def step() -> dict[str, ttnn.Tensor]:
        parts = {}
        for name in kinds:
            gathered = ttnn.all_gather(
                inputs[name], gather_dim[name], cluster_axis=_SP_AXIS, output_tensor=outputs[name]
            )
            assert gathered.buffer_address() == outputs[name].buffer_address()
            parts[name] = ttnn.mesh_partition(outputs[name], gather_dim[name], cluster_axis=_SP_AXIS)
        return parts

    def check(parts: dict[str, ttnn.Tensor], what: str) -> None:
        for name in kinds:
            full = golden[name]
            gathered = _per_device(outputs[name])
            partitioned = _per_device(parts[name])
            shards = _per_device(inputs[name])
            for index, (out, part, shard) in enumerate(zip(gathered, partitioned, shards)):
                row, column = divmod(index, _TP)
                assert torch.equal(
                    out.to(full.dtype), full[column : column + 1]
                ), f"{what} {name} gather ({row},{column})"
                assert torch.equal(part, shard), f"{what} {name} partition ({row},{column})"

    warm = step()
    ttnn.synchronize_device(mesh_device)
    check(warm, "eager")
    for part in warm.values():
        ttnn.deallocate(part)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = step()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for replay in range(2):
            for name, (shape, torch_dtype, dtype, layout) in kinds.items():
                golden[name] = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
                host = ttnn.from_torch(
                    golden[name],
                    dtype=dtype,
                    layout=layout,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=_MESH, dims=(gather_dim[name], 0)),
                )
                ttnn.copy_host_to_device_tensor(host, inputs[name])
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            check(traced, f"replay {replay}")
        print("KDA_SP_GATHER_PARTITION_TRACE=ok")
    finally:
        ttnn.release_trace(mesh_device, trace_id)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Extended coverage for point_to_point — the three gaps the acceptance + golden
suites leave open. Deliberately small (7 cases): everything else is already covered.

1. **Packet framing regime B** (`page_segments > 1`, one shard page split across
   several fabric packets). Every shape in the acceptance suite and in
   `eval/golden_tests/point_to_point/feature_spec.py` has a page size <= the fabric
   payload capacity (the widest is a float32 tile at 4096 B), so *both* the sender's
   and the receiver's segmentation branch were entirely unexecuted code. These cases
   use wide ROW_MAJOR rows to force `page_segments == 2`, once with an exact split
   and once with a short trailing segment (the off-by-one-prone path). The test
   asserts up front that the shapes really do land in regime B, so it cannot
   silently degrade into more regime-A coverage if the fabric payload size changes.

2. **L1-interleaved memory config.** The op's contract admits "interleaved, in DRAM
   or L1", `validate()` only rejects *sharded* input, and every other suite uses
   DRAM. This exercises the L1 buffer path through the same TensorAccessor CT args,
   including a row-major page that is not DRAM-alignment-sized.

3. **The caller-supplied `intermediate_tensor` path.** The public signature accepts
   one and `validate()` gates its spec, but no other test passes one.

Verification topology: `bh_quietbox_1x4_hw` — a `(1, 4)` Blackhole mesh with
`FabricConfig.FABRIC_1D`. Opening any other shape hangs fabric init, so MESH_SHAPE
is pinned. Run under the multi-device runner::

    scripts/run_multidevice_sim_pytest.py --op point_to_point --runtime hardware -- \
        tests/ttnn/unit_tests/operations/point_to_point/test_point_to_point_extended.py -v
"""

from math import prod

import pytest
import torch

import ttnn

from ttnn.operations.point_to_point import point_to_point
from ttnn.operations.point_to_point.point_to_point import _packet_dims, _resolve_intermediate_spec

MESH_SHAPE = (1, 4)
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}

SENDER = (0, 0)
RECEIVER = (0, 1)


def _linear_index(coord, mesh_shape):
    return coord[0] * tuple(mesh_shape)[1] + coord[1]


def _shard(mesh_device, shard_shape, dtype, layout, memory_config, seed=7):
    """Mesh-sharded tensor whose per-device shard is exactly `shard_shape`."""
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])
    torch.manual_seed(seed)
    full = torch.randn(full_shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        full = full.to(torch.bfloat16)
    tensor = ttnn.from_torch(
        full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return tensor


def _shards(tensor):
    return [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(tensor)]


def _assert_exact_transfer(mesh_device, input_tensor, output_tensor, label):
    """Receiver shard == sender shard bit-for-bit; every other shard untouched."""
    send_idx = _linear_index(SENDER, mesh_device.shape)
    recv_idx = _linear_index(RECEIVER, mesh_device.shape)
    in_shards = _shards(input_tensor)
    out_shards = _shards(output_tensor)
    expected = list(in_shards)
    expected[recv_idx] = in_shards[send_idx]
    for i, (got, want) in enumerate(zip(out_shards, expected)):
        max_abs = (got - want).abs().max().item()
        assert max_abs == 0.0, f"{label}: device {i} shard differs by {max_abs} (expected a bit-exact copy)"


# --------------------------------------------------------------------------------------
# 1. Packet framing regime B — one page split across several fabric packets
# --------------------------------------------------------------------------------------
# Wide ROW_MAJOR rows: the page size is `last_dim * element_size`, so these exceed the
# fabric payload capacity and must be segmented. bfloat16 @ 4096 elements -> 8192 B
# (an exact 2-way split); float32 @ 2048 elements -> 8192 B against a non-power-of-two
# payload -> a short trailing segment.
SEGMENTED = [
    ((1, 1, 8, 4096), ttnn.bfloat16),
    ((1, 1, 8, 2048), ttnn.float32),
]


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("shard_shape, dtype", SEGMENTED, ids=["bf16_exact_split", "fp32_short_tail"])
def test_segmented_page_framing(mesh_device, shard_shape, dtype):
    """`page_segments > 1`: the sender splits each page, the receiver reassembles it."""
    input_tensor = _shard(mesh_device, shard_shape, dtype, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)

    # Guard the *reason* this case exists: if framing ever stops segmenting these
    # shapes, the test must fail rather than quietly become more regime-A coverage.
    dims = _packet_dims(input_tensor)
    assert dims.page_segments > 1, (
        f"{shard_shape} {dtype} no longer segments (page_segments={dims.page_segments}); "
        "pick a wider row so regime B stays covered"
    )

    output_tensor = point_to_point(input_tensor, ttnn.MeshCoordinate(*SENDER), ttnn.MeshCoordinate(*RECEIVER))
    ttnn.synchronize_device(mesh_device)
    _assert_exact_transfer(mesh_device, input_tensor, output_tensor, f"regime B {shard_shape} {dtype}")


# --------------------------------------------------------------------------------------
# 2. L1-interleaved memory config
# --------------------------------------------------------------------------------------
L1_CASES = [
    ((1, 1, 64, 128), ttnn.TILE_LAYOUT),
    ((1, 1, 32, 48), ttnn.ROW_MAJOR_LAYOUT),  # 96 B row: not a multiple of the 64 B DRAM alignment
]


@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("shard_shape, layout", L1_CASES, ids=["tile", "row_major_96B"])
def test_l1_interleaved_input(mesh_device, shard_shape, layout):
    """An interleaved-L1 shard transfers exactly, and the output keeps its memory config."""
    input_tensor = _shard(mesh_device, shard_shape, ttnn.bfloat16, layout, ttnn.L1_MEMORY_CONFIG)
    assert not input_tensor.is_sharded()

    output_tensor = point_to_point(input_tensor, ttnn.MeshCoordinate(*SENDER), ttnn.MeshCoordinate(*RECEIVER))
    ttnn.synchronize_device(mesh_device)

    assert output_tensor.memory_config() == input_tensor.memory_config()
    _assert_exact_transfer(mesh_device, input_tensor, output_tensor, f"L1 {shard_shape} {layout}")


# --------------------------------------------------------------------------------------
# 3. Caller-supplied intermediate_tensor
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_intermediate_tensor_path(mesh_device):
    """A caller-supplied staging tensor is used; a mismatched one is rejected."""
    input_tensor = _shard(mesh_device, (1, 1, 64, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)

    shape, dtype, layout, mem_cfg = _resolve_intermediate_spec(input_tensor)
    intermediate = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, mesh_device, mem_cfg)

    output_tensor = point_to_point(
        input_tensor,
        ttnn.MeshCoordinate(*SENDER),
        ttnn.MeshCoordinate(*RECEIVER),
        intermediate_tensor=intermediate,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_exact_transfer(mesh_device, input_tensor, output_tensor, "supplied intermediate")

    # One row too few -> spec mismatch -> ValueError (structural misuse, not a
    # registry-model support refusal).
    wrong = ttnn.allocate_tensor_on_device(ttnn.Shape([shape[0] + 1, shape[1]]), dtype, layout, mesh_device, mem_cfg)
    with pytest.raises(ValueError):
        point_to_point(
            input_tensor,
            ttnn.MeshCoordinate(*SENDER),
            ttnn.MeshCoordinate(*RECEIVER),
            intermediate_tensor=wrong,
        )

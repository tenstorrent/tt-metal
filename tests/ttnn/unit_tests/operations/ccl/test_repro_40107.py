import pytest
import torch
from loguru import logger

import ttnn

FABRIC_1D = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}]


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_all_gather_1d_literal_issue(mesh_device):
    input_1d = ttnn.from_torch(
        torch.randn([256], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn_all_gather_0 = ttnn.all_gather(
        input_tensor=input_1d,
        dim=-2,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=None,
    )
    logger.info(f"output shape={ttnn_all_gather_0.shape}")


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_all_reduce_1d_literal_issue(mesh_device):
    input_1d = ttnn.from_torch(
        torch.randn([256], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn_all_gather_0 = ttnn.all_reduce(
        input_tensor=input_1d,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=None,
    )

    logger.info(f"output shape={ttnn_all_gather_0.shape}")


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2), (1, 4)], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
# 16 -> gather dim 上有 tile padding（走 composite）; 33 -> row-major page 不对齐（走 composite）; 64 -> native path
@pytest.mark.parametrize("N", [16, 33, 64])
@pytest.mark.parametrize("dim", [0, -1])
def test_all_gather_1d(mesh_device, layout, N, dim):
    num_devices = mesh_device.get_num_devices()
    torch_input = torch.randn([N], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    output = ttnn.all_gather(tt_input, dim=dim, cluster_axis=1)
    logger.info(f"N={N} layout={layout} dim={dim} in={tt_input.shape} out={output.shape}")

    assert list(output.shape) == [N * num_devices], f"bad output shape {output.shape}"
    expected = torch.cat([torch_input] * num_devices, dim=0)
    got = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[: N * num_devices]
    assert torch.allclose(got.float(), expected.float()), f"MISMATCH\nexp={expected}\ngot={got}"


# persistent output tensor (python kwarg is `output_tensor`). Only the native path uses it --
# the composite path ignores it -- so these all go native after the tile-padding fix.
# shape=[N] is 1D (padded rank 2 != logical rank 1); [1, 1, 32, 256] is the rank-4 control.
@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("shape", [[32], [64], [256], [1, 1, 32, 256]], ids=["1d_32", "1d_64", "1d_256", "rank4"])
def test_all_gather_persistent_output(mesh_device, shape):
    num_devices = mesh_device.get_num_devices()
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    def to_dev(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    tt_input = to_dev(torch_input)
    # gather on dim 0 -> output logical shape is input with dim 0 scaled by num_devices
    out_shape = list(shape)
    out_shape[0] *= num_devices
    persistent_out = to_dev(torch.zeros(out_shape, dtype=torch.bfloat16))
    logger.info(
        f"in logical={tt_input.shape} padded={tt_input.padded_shape} | "
        f"persistent out logical={persistent_out.shape} padded={persistent_out.padded_shape}"
    )

    output = ttnn.all_gather(tt_input, dim=0, cluster_axis=1, output_tensor=persistent_out)

    expected = torch.cat([torch_input] * num_devices, dim=0)

    def first_device_copy(t):
        # the gathered result is replicated across devices; concat, then take device 0's copy
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[: out_shape[0]]

    # Assert on the persistent buffer itself, not only on the return value: the composite path
    # ignores output_tensor and returns a fresh tensor, so checking the return value alone would
    # let a silently-unused buffer pass. persistent_out starts zeroed, so that shows up here.
    assert list(persistent_out.shape) == out_shape, f"persistent buffer shape changed to {persistent_out.shape}"
    got_buffer = first_device_copy(persistent_out)
    assert torch.allclose(
        got_buffer.float(), expected.float()
    ), f"persistent buffer not written, or written with wrong data\nexp={expected}\ngot={got_buffer}"

    # The return value must be the same data as the persistent buffer.
    assert list(output.shape) == out_shape, f"bad output shape {output.shape}, expected {out_shape}"
    got_return = first_device_copy(output)
    assert torch.allclose(
        got_return.float(), got_buffer.float()
    ), f"return value differs from the persistent buffer\nbuffer={got_buffer}\nreturned={got_return}"


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("shape,dim", [([256], 1), ([1, 1, 32, 256], 4), ([1, 1, 32, 256], -5)])
def test_bad_dim_message(mesh_device, shape, dim):
    tt_input = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.all_gather(tt_input, dim=dim, cluster_axis=1)

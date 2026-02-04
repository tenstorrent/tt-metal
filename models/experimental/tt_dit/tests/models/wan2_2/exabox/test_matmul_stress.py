import pytest
import torch
import ttnn
from loguru import logger
from .....utils.tensor import bf16_tensor
from .....layers.linear import Linear, ColParallelLinear
from .....parallel.manager import CCLManager


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
    ],
    indirect=True,
    ids=["fabric_1d_ring"],
)
@pytest.mark.parametrize("mesh_device", [(4, 32)], indirect=True)
def test_matmul_stress(
    mesh_device,
):
    print(mesh_device.shape)

    torch_model = torch.nn.Linear(4096, 4096, bias=False)
    torch_model.eval()

    tt_model = Linear(4096, 4096, bias=False, mesh_device=mesh_device)
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn((1, 4096, 4096), dtype=torch.bfloat16)

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device)

    for i in range(1_000_000_000):
        ttnn.synchronize_device(mesh_device)
        logger.info(f"Iteration {i}")

        for j in range(1000):
            tt_output = tt_model(tt_input_tensor)


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
    ],
    indirect=True,
    ids=["fabric_1d_ring"],
)
@pytest.mark.parametrize("mesh_device", [(4, 32)], indirect=True)
def test_matmul_ccl_stress(
    mesh_device,
    reset_seeds,
):
    """
    Test CCLs on both axes and matmul
    weight: TP on axis 1, FSDP on axis 0
    activation: TP on axis 1, replicated on axis 0

    AG activation on axis 1, AG weight on axis 0, do matmul, check output every 1000 iterations
    """
    print(mesh_device.shape)

    M = 4096
    K = 4096
    N = 4096 * 32

    torch_model = torch.nn.Linear(K, N, bias=False)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)

    tt_model = ColParallelLinear(
        K, N, bias=False, mesh_device=mesh_device, mesh_axis=1, ccl_manager=ccl_manager, fsdp_mesh_axis=0
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn((1, 1, M, K))

    tt_input_tensor = bf16_tensor(torch_input_tensor, device=mesh_device, mesh_axis=1, shard_dim=3)

    tt_input_gathered = ccl_manager.all_gather_persistent_buffer(tt_input_tensor, dim=3, mesh_axis=1)
    tt_output = tt_model(tt_input_gathered)

    logger.info(f"Warmup done")

    # gt = torch_model(torch_input_tensor)

    # tt_output_back = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=mesh_device.shape))
    # for k in range(tt_output_back.shape[0]):
    #     assert_quality(gt, tt_output_back[k], pcc=0.999_500)

    for i in range(1_000_000_000):
        for j in range(1):
            # tt_input_gathered = ccl_manager.all_gather_persistent_buffer(tt_input_tensor, dim=3, mesh_axis=1)
            tt_output = tt_model(tt_input_gathered)

        ttnn.synchronize_device(mesh_device)
        logger.info(f"Iteration {i}")
        # # concat replicated axis 0 to dim 0, TP axis 1 to dim 3
        # tt_output_back = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=mesh_device.shape))
        # for k in range(tt_output_back.shape[0]):
        #     assert_quality(gt, tt_output_back[k], pcc=0.999_500)

import pytest
import torch
import ttnn
from loguru import logger
from .....utils.tensor import bf16_tensor
from .....layers.linear import Linear


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
    ],
    indirect=True,
    ids=["fabric_1d_ring"],
)
@pytest.mark.parametrize("mesh_device", [(4, 32)], indirect=True)
def test_open_4x32_mesh(
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

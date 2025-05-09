import ttnn
import torch
import pytest
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.oft.reference.oftnet import OftNet
from models.experimental.oft.tt.ttnn_resnet import BasicBlock


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [
        (torch.rand((1, 64, 93, 306))),
        # (torch.rand((1, 256, 159, 159))),
    ],
    ids=["image"],
)
@pytest.mark.parametrize(
    "input_params",
    [
        [[1, 93, 306, 64], [1, 93, 306, 64]],
        # [[1, 93, 306, 64], [1, 47, 153, 128]],
        # [[1, 159, 159, 256], [1, 159, 159, 256]],
    ],
)
@pytest.mark.parametrize(
    "path",
    [
        "frontend.layer1.0",
        # "frontend.layer2.0",
        # "topdown.0",
    ],
)
def test_basic_block(device, input_tensor, input_params, path):
    disable_persistent_kernel_cache()

    torch.manual_seed(42)

    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4.0,
    )

    state_dict = model.state_dict()

    model.load_state_dict(state_dict)

    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with torch.inference_mode():
        tt_module = BasicBlock(
            device,
            state_dict,
            path,
            input_params,
            inplanes=64,
            planes=64,
            stride=1,
        )
        ttnn_output = tt_module(device, ttnn_input)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.reshape(input_params[1])
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    torch_module = model.get_submodule(path)
    with torch.inference_mode():
        torch_output = torch_module(input_tensor)

    print(f"torch:{torch_output}")
    print(f"ttnn :{ttnn_output}")
    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")

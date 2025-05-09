import ttnn
import torch
import torch.nn.functional as F
import pytest
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.oft.reference.oftnet import OftNet
from models.experimental.oft.tt.ttnn_resnet import BasicBlock, ResNetFeatures
from models.experimental.oft.tt.ttnn_oft import OFT
from models.experimental.oft.tt.ttnn_oftnet import OftNet as tt_OftNet
from models.experimental.oft.tt.model_preprocessing import (
    create_OFT_model_parameters_resnet,
    create_OFT_model_parameters_oft,
    create_OFT_model_parameters,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
# @pytest.mark.parametrize(
#     "input_tensor",
#     [
#         # (torch.rand((1, 64, 93, 306))), #layer1.0 #layer2.0
#         # (torch.rand((1, 128, 47, 153))), #layer3.0
#         # (torch.rand((1, 256, 24, 77))),
#         # (torch.rand((1, 256, 24, 77))) #layer 4.0
#         (torch.rand((1, 512, 12, 39)))
#         # (torch.rand((1, 256, 159, 159))), #top down layer
#     ],
#     ids=["image"],
# )
# @pytest.mark.parametrize(
#     "input_params",
#     [
#         # [[1, 93, 306, 64], [1, 93, 306, 64]], #layer1.0
#         # [[1, 93, 306, 64], [1, 47, 153, 128]], #layer2.0
#         # [[1, 47, 153, 128], [1, 24, 77, 256]], #layer3.0
#         # [[1, 24, 77, 256], [1, 24, 77, 256]]
#         [[1, 24, 77, 256], [1, 12, 39, 512]] #layer 4.0
#         # [[1, 12, 39, 512], [1, 12, 39, 512]]
#         # [[1, 159, 159, 256], [1, 159, 159, 256]], #topdown.0 shard is none
#     ],
# )
@pytest.mark.parametrize(
    "path",
    [
        # "frontend.layer1.0",
        # "frontend.layer2.0",
        # "frontend.layer3.1",
        # "frontend.layer4.0",
        "topdown.0",
    ],
)
def test_basic_block(device, path):
    disable_persistent_kernel_cache()

    torch.manual_seed(42)

    input_tensor = torch.rand((1, 256, 159, 159))
    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4.0,
    )

    state_dict = model.state_dict()

    model.load_state_dict(state_dict)
    torch_module = model.get_submodule(path)

    parameters = create_OFT_model_parameters_resnet(torch_module, input_tensor, device=device)

    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    with torch.inference_mode():
        tt_module = BasicBlock(
            device,
            parameters,
            parameters.conv_args,
            inplanes=256,
            planes=256,
            stride=1,
        )
        ttnn_output = tt_module(device, ttnn_input)
        ttnn_output = ttnn.to_torch(ttnn_output)
        # ttnn_output = ttnn_output.reshape(input_params[1])
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    with torch.inference_mode():
        torch_output = torch_module(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
# @pytest.mark.parametrize(
#     "input_tensor",
#     [(torch.rand((1, 3, 370, 1224)))],
#     ids=["image"],
# )
def test_resnet_features(device):
    disable_persistent_kernel_cache()

    torch.manual_seed(42)

    input_tensor = torch.rand((1, 3, 370, 1224))

    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4.0,
    )

    state_dict = model.state_dict()

    model.load_state_dict(state_dict)
    torch_module = model.get_submodule("frontend")
    parameters = create_OFT_model_parameters_resnet(torch_module, input_tensor, device=device)

    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with torch.inference_mode():
        torch_output = torch_module(input_tensor)[2]

    with torch.inference_mode():
        tt_module = ResNetFeatures(
            device,
            parameters,
            parameters.conv_args,
            BasicBlock,
            [2, 2, 2, 2],
        )
        ttnn_output = tt_module(device, ttnn_input)[2]
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "cell_size",
    [1],
)
@pytest.mark.parametrize(
    "grid_height",
    [4],
)
def test_oft_module(device, cell_size, grid_height, reset_seeds):
    disable_persistent_kernel_cache()

    features = torch.rand((1, 256, 47, 153))
    calib = torch.rand((1, 3, 4))
    grid = torch.rand((1, 160, 160, 3))

    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=1.0,
        grid_height=4.0,
    )
    model = model.eval()
    state_dict = model.state_dict()
    torch_module = model.get_submodule("oft8")
    parameters = create_OFT_model_parameters_oft(torch_module, (features, calib, grid), device=device)

    y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
    y_corners = ttnn.from_torch(
        y_corners, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    ttnn_input = ttnn.from_torch(
        features, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn_calib = ttnn.from_torch(
        calib, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn_grid = ttnn.from_torch(
        grid, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    torch_module = model.get_submodule("oft8")
    with torch.inference_mode():
        torch_output = torch_module(features, calib, grid)

    with torch.inference_mode():
        tt_module = OFT(
            device,
            parameters,
            y_corners,
            1 / 8,
        )
        ttnn_output = tt_module(device, ttnn_input, ttnn_calib, ttnn_grid)
        ttnn_output = ttnn.to_torch(ttnn_output).float()
        print(f"torch : {torch_output}")
        print(f"size:{torch_output.shape}")
        print(f"tt : {ttnn_output}")
        print(f"size:{ttnn_output.shape}")
        # ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "cell_size",
    [1],
)
@pytest.mark.parametrize(
    "grid_height",
    [4],
)
def test_oftnet(device, grid_height, cell_size):
    disable_persistent_kernel_cache()

    torch.manual_seed(42)

    input_tensor = torch.rand((1, 3, 370, 1224))
    calib = torch.rand((1, 3, 4))
    grid = torch.rand((1, 160, 160, 3))

    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=1.0,
        grid_height=4.0,
    )

    # state_dict = model.state_dict()

    # model.load_state_dict(state_dict)
    torch_module = model
    parameters = create_OFT_model_parameters(torch_module, (input_tensor, calib, grid), device=device)

    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    mean = ttnn.from_torch(
        mean, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    std = ttnn.from_torch(
        std, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    y_corners = torch.arange(0, grid_height, cell_size) - grid_height / 2.0
    y_corners = F.pad(y_corners.view(-1, 1, 1, 1), [1, 1])
    y_corners = ttnn.from_torch(
        y_corners, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_calib = ttnn.from_torch(
        calib, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn_grid = ttnn.from_torch(
        grid, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    with torch.inference_mode():
        tt_module = tt_OftNet(
            device,
            parameters,
            parameters.conv_args,
            BasicBlock,
            [2, 2, 2, 2],
            mean,
            std,
            y_corners,
        )
        ttnn_output = tt_module(device, ttnn_input, ttnn_calib, ttnn_grid)[0]
        ttnn_output = ttnn.to_torch(ttnn_output)
        # ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    with torch.inference_mode():
        torch_output = torch_module(input_tensor, calib, grid)[0]

    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")

import torch
import pytest
from models.experimental.functional_Ultralane_detection_V2.reference.tu_simple_model import Tu_Simple
from models.experimental.functional_Ultralane_detection_V2.Ultra_Fast_Lane_Detection_v2_forked.model.model_culane import (
    parsingNet,
)
from models.experimental.functional_Ultralane_detection_V2.Ultra_Fast_Lane_Detection_v2_forked.configs import (
    tusimple_res34 as cfg,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch_size,input_channels,height,width",
    [
        (1, 3, 320, 800),
    ],
)
def test_tu_simple_res34_inference(batch_size, input_channels, height, width):
    reference_model = Tu_Simple(input_height=height, input_width=width).to(torch.bfloat16)
    repo_model = parsingNet(
        pretrained=True,
        backbone=cfg.backbone,
        num_grid_row=cfg.num_cell_row,
        num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col,
        num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes,
        num_lane_on_col=cfg.num_lanes,
        use_aux=cfg.use_aux,
        input_height=cfg.train_height,
        input_width=cfg.train_width,
        fc_norm=cfg.fc_norm,
    ).to(torch.bfloat16)
    state_dict = torch.load("models/experimental/functional_Ultralane_detection_V2/reference/tusimple_res34.pth")
    reference_model.load_state_dict(state_dict["model"])
    repo_model.load_state_dict(state_dict["model"])
    input_tensor = torch.randn([batch_size, input_channels, height, width], dtype=torch.bfloat16)
    output_repo = repo_model(input_tensor)
    output_reference = reference_model(input_tensor)
    for key in output_reference:
        t1, t2 = output_repo[key], output_reference[key]
        print("they are:", torch.allclose(t1, t2))
        assert_with_pcc(t1, t2, 1.0)

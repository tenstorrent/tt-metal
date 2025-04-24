import ttnn
import torch
import pytest
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import run_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight
from models.experimental.functional_pointpillars.reference.mvx_faster_rcnn import MVXFasterRCNN
from models.experimental.functional_pointpillars.tt.ttnn_hard_vfe import TtHardVFE
from models.experimental.functional_pointpillars.reference.hard_vfe import HardVFE


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, HardVFE):
            parameters["vfe_layers"] = {}
            for index, child in enumerate(model.vfe_layers):
                parameters["vfe_layers"][index] = {}
                # As we are using torch batch_norm1d the norm weights are torch
                parameters["vfe_layers"][index]["norm"] = {}
                parameters["vfe_layers"][index]["norm"] = child.norm
                # parameters["vfe_layers"][index]["norm"]["bias"] = child.norm.weight

                parameters["vfe_layers"][index]["linear"] = {}
                parameters["vfe_layers"][index]["linear"]["weight"] = preprocess_linear_weight(
                    child.linear.weight, dtype=ttnn.bfloat16
                )
                parameters["vfe_layers"][index]["linear"]["weight"] = ttnn.to_device(
                    parameters["vfe_layers"][index]["linear"]["weight"], device=device
                )
                parameters["vfe_layers"][index]["linear"]["bias"] = None

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@run_for_wormhole_b0()
def test_ttnn_hard_vfe(device, use_pretrained_weight, reset_seeds):
    reference_model = MVXFasterRCNN(
        pts_voxel_encoder=True,
        pts_middle_encoder=True,
        pts_backbone=True,
        pts_neck=True,
        pts_bbox_head=True,
        train_cfg=None,
    )
    if use_pretrained_weight == True:
        state_dict = torch.load(
            "/home/ubuntu/punith/tt-metal/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
        )["state_dict"]
        reference_model.load_state_dict(state_dict)
    reference_model.eval()
    reference_model = reference_model.pts_voxel_encoder

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    features = torch.load("models/experimental/functional_pointpillars/features.pt")
    num_points = torch.load("models/experimental/functional_pointpillars/num_points.pt")
    coors = torch.load("models/experimental/functional_pointpillars/coors.pt")
    img_feats = None
    img_metas = None  # It's not none, using none as we are not using this variable inside the hardvfe

    ttnn_features = ttnn.from_torch(features, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_num_points = ttnn.from_torch(num_points, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
    ttnn_coors = ttnn.from_torch(coors, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

    reference_output = reference_model(
        features=features, num_points=num_points, coors=coors, img_feats=img_feats, img_metas=img_metas
    )

    ttnn_model = TtHardVFE(
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=[0.25, 0.25, 8],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        norm_cfg={"type": "BN1d", "eps": 0.001, "momentum": 0.01},
        parameters=parameters,
        device=device,
    )

    ttnn_output = ttnn_model(
        features=ttnn_features, num_points=ttnn_num_points, coors=ttnn_coors, img_feats=img_feats, img_metas=img_metas
    )

    passing, pcc = assert_with_pcc(reference_output, ttnn.to_torch(ttnn_output), 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")

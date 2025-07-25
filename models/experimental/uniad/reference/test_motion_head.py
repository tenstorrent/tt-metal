import torch
import pytest
import sys

import sys, torch
from models.experimental.uniad.reference import uniad_utils
from models.experimental.uniad.reference.motion_head import MotionHead
from collections import OrderedDict

# Alias the old path to the new module where LiDARInstance3DBoxes is now located
sys.modules["mmdet3d"] = sys.modules["models"]
sys.modules["mmdet3d.core.bbox.structures.lidar_box3d"] = uniad_utils
# sys.modules["mmdet3d.core.bbox.structures.lidar_box3"] = uniad_utils


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_uniad_motion_head(
    device,
    reset_seeds,
):
    outs_seg = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head/outs_seg.pt", map_location="cpu"
    )
    outs_track = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head/outs_track.pt", map_location="cpu"
    )
    bev_embed = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head/bev_embed.pt", map_location="cpu"
    )

    # pkl=pickle.load(open("/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head/motion_anchor_infos_mode6.pkl", 'rb'))
    # print(pkl)

    weights_path = "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/uniad_base_e2e.pth"
    reference_model = MotionHead(
        args=(),
        predict_steps=12,
        transformerlayers={
            "type": "MotionTransformerDecoder",
            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "embed_dims": 256,
            "num_layers": 3,
            "transformerlayers": {
                "type": "MotionTransformerAttentionLayer",
                "batch_first": True,
                "attn_cfgs": [
                    {
                        "type": "MotionDeformableAttention",
                        "num_steps": 12,
                        "embed_dims": 256,
                        "num_levels": 1,
                        "num_heads": 8,
                        "num_points": 4,
                        "sample_index": -1,
                    }
                ],
                "feedforward_channels": 512,
                "ffn_dropout": 0.1,
                "operation_order": ("cross_attn", "norm", "ffn", "norm"),
            },
        },
        bbox_coder=None,
        num_cls_fcs=3,
        bev_h=50,
        bev_w=50,
        embed_dims=256,
        num_anchor=6,
        det_layer_num=6,
        group_id_list=[[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        use_nonlinear_optimizer=True,
        anchor_info_path="models/experimental/uniad/reference/motion_head/motion_anchor_infos_mode6.pkl",
        loss_traj={
            "type": "TrajLoss",
            "use_variance": True,
            "cls_loss_weight": 0.5,
            "nll_loss_weight": 0.5,
            "loss_weight_minade": 0.0,
            "loss_weight_minfde": 0.25,
        },
        num_classes=10,
        vehicle_id_list=[0, 1, 2, 3, 4, 6, 7],
        **{"num_query": 300, "predict_modes": 6},
    )
    weights = torch.load(weights_path, map_location=torch.device("cpu"))

    prefix = "motion_head"
    filtered = OrderedDict(
        (
            (k[len(prefix) + 1 :], v)  # Remove the prefix from the key
            for k, v in weights["state_dict"].items()
            if k.startswith(prefix)
        )
    )
    # print("filtered",filtered)
    # state_dict=weights["state_dict"]["pts_bbox_head"]#["transformer"]["decoder"]
    reference_model.load_state_dict(filtered)
    reference_model.eval()

    print("reference model", reference_model)

    reference_output = reference_model.forward_test(bev_embed=bev_embed, outs_track=outs_track, outs_seg=outs_seg)

    torch_output_traj_results = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head/traj_results.pt",
        map_location="cpu",
    )
    torch_output_outs_motion = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head/outs_motion.pt",
        map_location="cpu",
    )

    print("reference_output", reference_output[1])
    print("torch_output_traj_results", torch_output_outs_motion)


def vtest_uniad_motion_he():
    outs_seg = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head/traj_results.pt",
        map_location="cpu",
    )
    outs_track = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head/outs_motion.pt",
        map_location="cpu",
    )
    bev_embed = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head/bev_embed.pt", map_location="cpu"
    )

    outs_seg_1 = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head_1/outs_seg.pt", map_location="cpu"
    )
    outs_track_1 = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head_1/outs_track.pt",
        map_location="cpu",
    )
    bev_embed_1 = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/uniad/reference/motion_head_1/bev_embed.pt",
        map_location="cpu",
    )

    # import torch

    def compare_dicts(d1, d2):
        mismatches = {}
        for key in d1:
            if key not in d2:
                mismatches[key] = "Missing in dict2"
            else:
                v1 = d1[key]
                v2 = d2[key]
                print("v1", v1)
                print("v2", v2)
                if isinstance(v1, dict):
                    mismatches[key] = compare_dicts(v1, v2)
                elif torch.is_tensor(v1) and torch.is_tensor(v2):
                    if not torch.equal(v1, v2):
                        mismatches[key] = "Tensor mismatch"
                elif v1 != v2:
                    mismatches[key] = f"Mismatch: {v1} != {v2}"
        for key in d2:
            if key not in d1:
                mismatches[key] = "Missing in dict1"
        return mismatches

    print("outs_seg", outs_seg)
    print("outs_track", outs_track)
    # print(compare_dicts(outs_seg_1,outs_seg))

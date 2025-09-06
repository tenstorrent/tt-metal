# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import os
import torch
from collections import OrderedDict


def load_torch_model(torch_model, layer="", model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "models/experimental/uniad/uniad_base_e2e.pth"
        if not os.path.exists(weights_path):
            os.system("bash models/experimental/uniad/weights_download.sh")

    weights = torch.load(weights_path, map_location=torch.device("cpu"))
    state_dict = weights.get("state_dict", weights)
    new_bev_h = 50
    new_bev_w = 50
    new_bev_size = new_bev_h * new_bev_w
    if layer == "" or layer == "seg_head" or layer == "pts_bbox_head":
        # 1. Slice row_embed and col_embed from [200, 128] → [50, 128]
        for key in [
            "pts_bbox_head.positional_encoding.row_embed.weight",
            "pts_bbox_head.positional_encoding.col_embed.weight",
        ]:
            if key in state_dict:
                print(f"Slicing {key} from {state_dict[key].shape} to {(new_bev_h, state_dict[key].shape[1])}")
                state_dict[key] = state_dict[key][:new_bev_h, :]

        # 2. Slice bev_embedding from [40000, 256] → [2500, 256]
        for key in ["pts_bbox_head.bev_embedding.weight", "seg_head.bev_embedding.weight"]:
            if key in state_dict:
                print(f"Slicing {key} from {state_dict[key].shape} to {(new_bev_size, state_dict[key].shape[1])}")
                state_dict[key] = state_dict[key][:new_bev_size, :]

        if "criterion.code_weights" in state_dict:
            del state_dict["criterion.code_weights"]
        new_state_dict = state_dict

    if (
        layer == "pts_bbox_head.transformer.decoder"
        or layer == "pts_bbox_head"
        or layer == "motion_head.motionformer.map_interaction_layers.0"
        or layer == "motion_head"
        or layer == "pts_bbox_head.transformer"
        or layer == "planning_head"
        or layer == "img_backbone"
    ):
        new_state_dict = OrderedDict(
            (
                (k[len(layer) + 1 :], v)  # Remove the prefix from the key
                for k, v in state_dict.items()
                if k.startswith(layer)
            )
        )
    else:
        state_dict = {k: v for k, v in state_dict.items() if (k.startswith(layer))}
        new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    return torch_model

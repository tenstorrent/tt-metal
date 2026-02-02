# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import os
import torch


def download_bevformerv2_weights():
    import urllib.request

    file_id = "1hC49RBbDW_qZJNHAfAjsmIezTtPKRevc"
    weights_path = "/tmp/bevformerv2_weights.pth"

    if not os.path.exists(weights_path):
        try:
            import gdown

            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, weights_path, quiet=False)
        except ImportError:
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            urllib.request.urlretrieve(direct_url, weights_path)

    return weights_path


def load_torch_model(torch_model, layer="", model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        weights_path = "/tmp/bevformerv2_weights.pth"
        if not os.path.exists(weights_path):
            weights_path = download_bevformerv2_weights()
    else:
        weights_path = (
            model_location_generator("vision-models/bevformer_v2", model_subdir="", download_if_ci_v2=True)
            / "bevformer_v2_weights.pth"
        )

    torch_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(torch_dict, dict) and "state_dict" in torch_dict:
        torch_dict = torch_dict["state_dict"]

    if layer == "":
        new_state_dict = {}
        for k, v in torch_dict.items():
            if k == "pts_bbox_head.bev_embedding.weight" and hasattr(torch_model, "pts_bbox_head"):
                expected_size = torch_model.pts_bbox_head.bev_h * torch_model.pts_bbox_head.bev_w
                if v.shape[0] != expected_size:
                    new_state_dict[k] = v[:expected_size]
                else:
                    new_state_dict[k] = v
            else:
                new_state_dict[k] = v
    else:
        state_dict = {k: v for k, v in torch_dict.items() if (k.startswith(layer))}
        new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict, strict=False)
    torch_model.eval()

    return torch_model

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import torch
import timm


def load_torch_model(reference_model, target_prefix="", model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True).eval()
        return model
    else:
        model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=False).eval()
        weights = (
            model_location_generator("vision-models/vovnet", model_subdir="", download_if_ci_v2=True)
            / "ese_vovnet19b_dw_ra_in1k.pth"
        )
        state_dict = torch.load(weights)

        model.load_state_dict(state_dict)
        return model.eval()


def get_mesh_mappers(device):
    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.shard_tensor_to_mesh_mapper(device, dim=0)
        weights_mesh_mapper = None
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer

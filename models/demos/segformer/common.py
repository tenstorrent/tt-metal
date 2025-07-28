# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from transformers import AutoConfig, SegformerForImageClassification, SegformerForSemanticSegmentation


def load_config(config_name="configs/segformer_semantic_config.json"):
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    config = AutoConfig.from_pretrained(config_path)
    return config


def load_torch_model(reference_model, target_prefix, module="semantic_sub", model_location_generator=None):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        if module == "image_classification":
            torch_model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
        elif module == "semantic_sub":
            torch_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        state_dict = torch_model.state_dict()
    else:
        weights = (
            model_location_generator("vision-models/segformer", model_subdir="", download_if_ci_v2=True)
            / "segformer_b0_ade_512_512.pth"
        )
        # TODO CIv2 weight load for Image Classification. Issue - https://github.com/tenstorrent/tt-metal/issues/25888
        state_dict = torch.load(weights)

    new_state_dict = {}
    if target_prefix == "":
        module_state_dict = {k: v for k, v in state_dict.items()}
    else:
        module_state_dict = {k: v for k, v in state_dict.items() if (target_prefix in k)}
    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in module_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()
    return reference_model

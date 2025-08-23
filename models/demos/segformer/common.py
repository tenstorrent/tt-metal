# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import zipfile
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
        if module == "image_classification":
            weights = (
                model_location_generator(
                    "vision-models/segformer-classification", model_subdir="", download_if_ci_v2=True
                )
                / "pytorch_model.tensorbin"
            )
        elif module == "semantic_sub":
            weights = (
                model_location_generator(
                    "vision-models/segformer-segmentation", model_subdir="", download_if_ci_v2=True
                )
                / "segformer_b0_ade_512_512.pth"
            )
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


def download_and_unzip_dataset(model_location_generator, dataset_path, dataset_name):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        if not os.path.exists(f"models/demos/segformer/demo/{dataset_name}"):
            os.system("bash models/demos/segformer/demo/data_download.sh")
        return f"models/demos/segformer/demo/{dataset_name}"
    else:
        zip_path = (
            model_location_generator(f"vision-models/{dataset_path}", model_subdir="", download_if_ci_v2=True)
            / f"{dataset_name}.zip"
        )
        extract_dir = zip_path.parent / f"{dataset_name}"

        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        file_path = str(extract_dir / dataset_name)
        return file_path

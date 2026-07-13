# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import re
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


def _segformer_ckpt_key_to_reference(k):
    """Map a transformers-5.x Segformer checkpoint key to this demo's reference-model key name.

    transformers 5.x refactored the HF Segformer state_dict: it renamed the encoder containers
    (``encoder.{patch_embeddings,block,layer_norm}.{i}`` -> ``stages.{i}.{patch_embeddings,blocks,
    layer_norm}``), the attention/mlp submodules (``self.{query,key,value}``/``output.dense`` ->
    ``{q,k,v,o}_proj``; ``sr``/``self.layer_norm`` -> ``sequence_reduction.*``; ``layer_norm_1/2``
    -> ``layernorm_before/after``; ``mlp.dense1/2`` -> ``mlp.fc1/2``) and the decode head
    (``decode_head.linear_c`` -> ``decode_head.linear_projections``). The CI .pth carries the new
    (5.x) names while the reference model uses the original names, so remap new -> old by name
    (parameter shapes are identical). Order: most-specific submodule renames first.
    """
    k = k.replace("attention.sequence_reduction.sequence_reduction", "attention.self.sr")
    k = k.replace("attention.sequence_reduction.layer_norm", "attention.self.layer_norm")
    k = k.replace("attention.q_proj", "attention.self.query")
    k = k.replace("attention.k_proj", "attention.self.key")
    k = k.replace("attention.v_proj", "attention.self.value")
    k = k.replace("attention.o_proj", "attention.output.dense")
    k = k.replace("layernorm_before", "layer_norm_1")
    k = k.replace("layernorm_after", "layer_norm_2")
    k = k.replace("mlp.fc1", "mlp.dense1")
    k = k.replace("mlp.fc2", "mlp.dense2")
    k = re.sub(r"segformer\.stages\.(\d+)\.patch_embeddings\.", r"segformer.encoder.patch_embeddings.\1.", k)
    k = re.sub(r"segformer\.stages\.(\d+)\.blocks\.(\d+)\.", r"segformer.encoder.block.\1.\2.", k)
    k = re.sub(r"segformer\.stages\.(\d+)\.layer_norm\.", r"segformer.encoder.layer_norm.\1.", k)
    k = re.sub(r"decode_head\.linear_projections\.(\d+)\.", r"decode_head.linear_c.\1.", k)
    return k


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
                / "pytorch_model.bin"
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
    # Map by name (order-independent). transformers 5.x renamed the HF Segformer parameters, so
    # the checkpoint keys no longer match the reference model's. Try, in order: (1) direct name
    # match (older checkpoints), (2) remap 5.x names -> reference names by key (shapes identical),
    # (3) positional zip as a last resort (legacy behavior).
    if set(keys).issubset(module_state_dict.keys()):
        new_state_dict = {name: module_state_dict[name] for name in keys}
    else:
        remapped = {_segformer_ckpt_key_to_reference(k): v for k, v in module_state_dict.items()}
        if set(keys).issubset(remapped.keys()):
            new_state_dict = {name: remapped[name] for name in keys}
        else:
            values = [parameter for name, parameter in module_state_dict.items()]
            for i in range(len(keys)):
                new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()
    return reference_model


def download_and_unzip_dataset(model_location_generator, dataset_path, dataset_name):
    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        if not os.path.exists(f"models/demos/vision/segmentation/segformer/demo/{dataset_name}"):
            os.system("bash models/demos/vision/segmentation/segformer/demo/data_download.sh")
        return f"models/demos/vision/segmentation/segformer/demo/{dataset_name}"
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

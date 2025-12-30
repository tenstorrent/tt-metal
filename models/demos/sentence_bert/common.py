# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
import transformers
from loguru import logger
from transformers import AutoConfig


def load_config(config_name="configs/config.json"):
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    config = AutoConfig.from_pretrained(config_path)
    return config


def load_torch_model(reference_model, target_prefix="", model_location_generator=None):
    logger.info(f"load_torch_model called with model_location_generator={model_location_generator is not None}")
    logger.info(f"TT_GH_CI_INFRA in env: {'TT_GH_CI_INFRA' in os.environ}")
    logger.info(f"CI env var: {os.getenv('CI')}")
    logger.info(f"HF_HUB_CACHE env var: {os.getenv('HF_HUB_CACHE')}")

    if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
        logger.info("Using HuggingFace AutoModel.from_pretrained path")
        # Note: Not using local_files_only to let HuggingFace handle cache/download automatically
        # HF_HUB_CACHE is set to /mnt/MLPerf/huggingface/hub in CI
        torch_model = transformers.AutoModel.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
        logger.info("Successfully loaded model from HuggingFace")
        state_dict = torch_model.state_dict()
    else:
        logger.info("Using custom .pth file path (CIv2)")
        weights_path = model_location_generator("vision-models/sentence_bert", model_subdir="", download_if_ci_v2=True)
        logger.info(f"Model location generator returned: {weights_path}")
        weights = weights_path / "bert_base_turkish_cased_mean_nli_stsb_tr.pth"
        logger.info(f"Full weights path: {weights}")
        logger.info(f"Weights file exists: {weights.exists() if hasattr(weights, 'exists') else 'N/A'}")
        state_dict = torch.load(weights)
        logger.info("Successfully loaded weights from .pth file")

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

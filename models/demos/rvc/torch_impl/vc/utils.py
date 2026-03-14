# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json

from safetensors.torch import load_file

from models.demos.rvc.torch_impl.vc.hubert import HubertModel, HubertPretrainingConfig, HubertPretrainingTask


def load_model_ensemble_and_task(
    model_path,
    model_cfg_path,
):
    task = HubertPretrainingTask(HubertPretrainingConfig())

    with open(model_cfg_path) as f:
        cfg = json.load(f)
    model = HubertModel.build_model(cfg["model"], task)
    hubert_state_safetensors = load_file(model_path)

    model.load_state_dict(hubert_state_safetensors, strict=True)
    return model


def load_hubert(config, hubert_path: str, hubert_cfg_path: str):
    hubert_model = load_model_ensemble_and_task(
        hubert_path,
        hubert_cfg_path,
    )
    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.float()
    return hubert_model.eval()

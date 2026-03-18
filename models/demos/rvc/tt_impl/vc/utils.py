# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json

from safetensors.torch import load_file

from models.demos.rvc.torch_impl.vc.hubert import HubertPretrainingConfig, HubertPretrainingTask
from models.demos.rvc.tt_impl.vc.hubert import HubertModel


def load_model_ensemble_and_task(
    model_path,
    model_cfg_path,
    tt_device,
):
    task = HubertPretrainingTask(HubertPretrainingConfig())

    with open(model_cfg_path) as f:
        cfg = json.load(f)
    model = HubertModel.build_model(cfg["model"], task, device=tt_device)
    hubert_state_safetensors = load_file(model_path)

    model.load_state_dict(hubert_state_safetensors)
    return model


def load_hubert(config, hubert_path: str, hubert_cfg_path: str, tt_device):
    hubert_model = load_model_ensemble_and_task(
        hubert_path,
        hubert_cfg_path,
        tt_device,
    )
    return hubert_model.eval()

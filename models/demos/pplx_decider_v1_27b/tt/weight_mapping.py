# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Dict, Sequence

import torch
from safetensors import safe_open

from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

PPLX_DECIDER_HF_MODEL = "perplexity-ai/pplx-decider-v1-27b"
LANGUAGE_MODEL_PREFIX = "language_model."
VISUAL_PREFIX = "visual."


def resolve_checkpoint(hf_model=None) -> Path:
    hf_model = hf_model or os.environ.get("HF_MODEL", PPLX_DECIDER_HF_MODEL)
    if os.path.isfile(os.path.join(hf_model, "config.json")):
        return Path(hf_model)
    from huggingface_hub import snapshot_download

    offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
    return Path(snapshot_download(hf_model, local_files_only=offline))


def load_pplx_state_dict(ckpt_dir, token_ids: Sequence[int], vocab_size: int, dim: int) -> Dict[str, torch.Tensor]:
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    file_to_keys: Dict[str, list] = {}
    for key, filename in weight_map.items():
        if key.startswith(VISUAL_PREFIX):
            continue
        if not key.startswith(LANGUAGE_MODEL_PREFIX):
            raise ValueError(f"Unexpected key in pplx-decider checkpoint: {key}")
        file_to_keys.setdefault(filename, []).append(key)

    # Only the text tower runs, so visual.* is skipped. The "model." prefix is added back because the qwen36
    # remap strips "model.language_model.".
    state_dict: Dict[str, torch.Tensor] = {}
    for filename, keys in file_to_keys.items():
        with safe_open(str(ckpt_dir / filename), framework="pt") as sf:
            for key in keys:
                state_dict["model." + key] = sf.get_tensor(key)

    with safe_open(str(ckpt_dir / "readout.safetensors"), framework="pt") as sf:
        readout = sf.get_tensor("weight")
    if tuple(readout.shape) != (len(token_ids), dim):
        raise ValueError(f"readout shape {tuple(readout.shape)} != {(len(token_ids), dim)}")
    if min(token_ids) < 0 or max(token_ids) >= vocab_size:
        raise ValueError(f"token_ids must be in [0, {vocab_size})")

    # The qwen36 LM head runs after the final norm, like the readout. With the readout rows placed at token_ids
    # and zeros elsewhere, its logits at token_ids are the readout logits and every other logit is exactly 0.
    lm_head = torch.zeros(vocab_size, dim, dtype=readout.dtype)
    lm_head[torch.tensor(token_ids, dtype=torch.long)] = readout
    state_dict["lm_head.weight"] = lm_head

    state_dict = remap_qwen36_state_dict(state_dict)
    if tuple(state_dict["output.weight"].shape) != (vocab_size, dim):
        raise ValueError(f"output.weight shape {tuple(state_dict['output.weight'].shape)} != {(vocab_size, dim)}")
    return state_dict

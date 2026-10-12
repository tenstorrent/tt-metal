# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision part of the DeepSeek-V4.1-Flash config (no ttnn: also imported by the vLLM frontend process, which builds the multimodal processor)."""

import json
import os


class VisionArgs:
    """Vision part of the checkpoint ``config.json`` (names of ``inference/model.py`` ``ModelArgs``)."""

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = checkpoint_config()
        v = cfg["vision_config"]
        self.dim = cfg.get("hidden_size", 5120)
        self.n_layers = v["num_hidden_layers"]
        self.vdim = v["hidden_size"]
        self.n_heads = v["num_attention_heads"]
        self.inter = v["intermediate_size"]
        self.patch = v["patch_size"]
        self.theta = float(v.get("rope_theta", 10000.0))
        self.ratio = v["downsample_ratio"]
        self.max_tokens = v["max_image_tokens"]
        self.min_pixels = v["min_pixels"]
        self.max_wh_ratio = v.get("max_wh_ratio")
        self.image_token_id = cfg["image_token_id"]
        self.eps = 1e-6
        self.head_dim = self.vdim // self.n_heads


def checkpoint_config():
    """The checkpoint's ``config.json`` (DSV41_CKPT, else MODEL_WEIGHTS_DIR, else the default checkpoint directory)."""
    from models.demos.blackhole.deepseek_v41_flash.tt.model_args import CKPT

    for d in (os.environ.get("DSV41_CKPT"), os.environ.get("MODEL_WEIGHTS_DIR"), CKPT):
        if d and os.path.exists(os.path.join(d, "config.json")):
            return json.load(open(os.path.join(d, "config.json")))
    raise FileNotFoundError("no checkpoint config.json (set DSV41_CKPT or MODEL_WEIGHTS_DIR)")


def vision_args_of(hf_config):
    """``VisionArgs`` from the serving config (vllm_config/config.json carries ``vision_config`` / ``image_token_id``), else the checkpoint's own config."""
    d = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    return VisionArgs(d if "vision_config" in d else checkpoint_config())

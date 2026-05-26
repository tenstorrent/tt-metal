#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dump state-dict keys for the same Llama checkpoint in three formats:

    1. hf              -- HuggingFace ``transformers`` (raw safetensors keys)
    2. tt-transformers -- Meta naming used by ``models/tt_transformers``
                          (run through ``convert_hf_to_meta``)
    3. ttml            -- ``Llama/blocks/{i}/...`` naming used by tt-train,
                          read directly from ``LlamaCompositeKV.parameters()``

The ``hf`` and ``tt-transformers`` formats are derived on CPU. The ``ttml``
format requires building the actual model, which opens the mesh device
(closed before the script exits). Edit ``MODEL_ID`` to switch checkpoints.
"""

from __future__ import annotations

import os

os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

import sys
from pathlib import Path
from typing import Dict, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml"
TTML_MODEL_CONFIG_REL = "tt-train/configs/model_configs/llama3_2_1B.yaml"


# ---------------------------------------------------------------------------
# 1) HF format
# ---------------------------------------------------------------------------


def hf_keys(model_id: str) -> Dict[str, Tuple[int, ...]]:
    """Return raw HF state-dict keys -> shape, without ever materialising fp32."""
    import torch
    from transformers import AutoModelForCausalLM

    print(f"[hf] loading {model_id} (CPU, meta-tensor scan)")
    # Use ``torch_dtype=torch.float16`` to keep memory low; we only read keys/shapes.
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    sd = model.state_dict()
    out = {k: tuple(v.shape) for k, v in sd.items()}
    del model, sd
    return out


# ---------------------------------------------------------------------------
# 2) tt-transformers (Meta) format
# ---------------------------------------------------------------------------


def tt_transformers_keys(hf_state_shapes: Dict[str, Tuple[int, ...]], model_id: str) -> Dict[str, Tuple[int, ...]]:
    """Run the HF state dict through ``convert_hf_to_meta`` and return renamed keys.

    No mesh device required. ``convert_hf_to_meta`` only does key renaming +
    Q/K row permutation (which preserves shape).
    """
    import torch
    from transformers import AutoConfig

    from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, standardize_hf_keys

    cfg = AutoConfig.from_pretrained(model_id)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads

    # Build a "shape only" state dict of zero tensors so the Q/K permutation
    # path can run without holding real weights in memory.
    fake_sd = {k: torch.zeros(s, dtype=torch.float16) for k, s in hf_state_shapes.items()}
    fake_sd = standardize_hf_keys(fake_sd)
    meta_sd = convert_hf_to_meta(fake_sd, head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
    return {k: tuple(v.shape) for k, v in meta_sd.items()}


# ---------------------------------------------------------------------------
# 3) ttml format
# ---------------------------------------------------------------------------


def ttml_keys(model_id: str) -> Dict[str, Tuple[int, ...]]:
    """Build the actual ttml ``LlamaCompositeKV`` model and return its parameters.

    Opens the AutoContext mesh device, instantiates the model (no weight
    load), enumerates ``model.parameters()`` to get the live names and
    shapes, then closes the device. The values are 4D shapes
    (``(1, 1, rows, cols)`` for matrices, ``(1, 1, 1, n)`` for vectors)
    matching ttml's on-device tensor layout.
    """
    import ttnn

    import ttml
    from ttml.common.config import DeviceConfig, get_model_config, load_config
    from ttml.models import RunnerType, WeightTyingType
    from ttml.models.llama import LlamaConfig, LlamaRopeScalingConfig
    from transformers import AutoTokenizer

    from utils.llama_overrides import LlamaCompositeKV

    raw = load_config(os.path.join(REPO_ROOT, TTML_DEVICE_CONFIG_REL))
    device_config = DeviceConfig(raw)
    tf_config = get_model_config(os.path.join(REPO_ROOT, TTML_MODEL_CONFIG_REL))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tf_config.vocab_size = len(tokenizer)

    rope_scaling = LlamaRopeScalingConfig(
        scaling_factor=getattr(tf_config, "scaling_factor", 0.0) or 0.0,
        high_freq_factor=getattr(tf_config, "high_freq_factor", 4.0) or 4.0,
        low_freq_factor=getattr(tf_config, "low_freq_factor", 1.0) or 1.0,
        original_context_length=getattr(tf_config, "original_context_length", 0) or 0,
    )
    runner_type = RunnerType.from_string(str(tf_config.runner_type))
    weight_tying = WeightTyingType.Disabled
    if tf_config.weight_tying:
        weight_tying = WeightTyingType.from_string(str(tf_config.weight_tying))

    llama_cfg = LlamaConfig(
        hidden_size=tf_config.embedding_dim,
        intermediate_size=tf_config.intermediate_dim,
        num_hidden_layers=tf_config.num_blocks,
        num_attention_heads=tf_config.num_heads,
        num_key_value_heads=tf_config.num_groups,
        vocab_size=len(tokenizer),
        max_position_embeddings=tf_config.max_sequence_length,
        rope_theta=tf_config.theta or 10000.0,
        attention_dropout=tf_config.dropout_prob,
        mlp_dropout=tf_config.dropout_prob,
        runner_type=runner_type,
        weight_tying=weight_tying,
        rope_scaling=rope_scaling,
    )

    # set_fabric_config must run before any device is opened.
    print("[ttml] set_fabric_config(FABRIC_2D)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    print(f"[ttml] opening mesh device {tuple(device_config.mesh_shape)} and building LlamaCompositeKV")
    if device_config.total_devices() > 1:
        ttml.core.distributed.enable_fabric(device_config.total_devices())
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)

    try:
        model = LlamaCompositeKV(llama_cfg)
        params = {name: tuple(param.shape()) for name, param in model.parameters().items()}
        del model
    finally:
        autograd_ctx.close_device()

    return {name: params[name] for name in sorted(params)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_keys(title: str, items) -> None:
    print()
    print("=" * 78)
    print(f" {title}  (count={len(items)})")
    print("=" * 78)
    if isinstance(items, dict):
        width = max(len(k) for k in items) if items else 0
        for k, shape in items.items():
            print(f"  {k.ljust(width)}  shape={shape}")
    else:
        for k in items:
            print(f"  {k}")


def main() -> None:
    print(f"Model: {MODEL_ID}")

    hf = hf_keys(MODEL_ID)
    print_keys("HF (transformers / safetensors)", hf)

    tt = tt_transformers_keys(hf, MODEL_ID)
    print_keys("tt-transformers (Meta naming, after convert_hf_to_meta)", tt)

    ttml = ttml_keys(MODEL_ID)
    print_keys("ttml (Llama/blocks/{i}/... — live LlamaCompositeKV.parameters())", ttml)


if __name__ == "__main__":
    main()

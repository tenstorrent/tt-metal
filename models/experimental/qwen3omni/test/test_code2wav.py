# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import pytest
import ttnn
from safetensors.torch import safe_open
from transformers import AutoConfig

from models.common.utility_functions import comp_pcc
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.utils.device_management import set_device

# TT implementation
from models.experimental.qwen3omni.tt.code2wav_attn import (
    TTNNQwen3OmniMoeCode2WavAttention,
)

# HF reference
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavAttention,
)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
CHECKPOINT_PATH = "models/experimental/qwen3omni/checkpoints/model-00015-of-00015.safetensors"


# ------------------------------------------------------------
# LOAD STATE DICT (single shard only, no index)
# ------------------------------------------------------------
def load_state_dict():
    """Load full state dict from a single safetensors shard."""
    if not os.path.isfile(CHECKPOINT_PATH):
        return None
    state_dict = {}
    with safe_open(CHECKPOINT_PATH, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def load_weights_into_torch(attn, state_dict, layer_idx):
    prefix = f"code2wav.pre_transformer.layers.{layer_idx}.self_attn"

    attn.q_proj.weight.data = state_dict[f"{prefix}.q_proj.weight"]
    attn.k_proj.weight.data = state_dict[f"{prefix}.k_proj.weight"]
    attn.v_proj.weight.data = state_dict[f"{prefix}.v_proj.weight"]
    attn.o_proj.weight.data = state_dict[f"{prefix}.o_proj.weight"]

    if f"{prefix}.q_proj.bias" in state_dict:
        attn.q_proj.bias.data = state_dict[f"{prefix}.q_proj.bias"]
        attn.k_proj.bias.data = state_dict[f"{prefix}.k_proj.bias"]
        attn.v_proj.bias.data = state_dict[f"{prefix}.v_proj.bias"]
        attn.o_proj.bias.data = state_dict[f"{prefix}.o_proj.bias"]

    return attn


def _has_code2wav_attn_weights(state_dict, layer_idx=0):
    prefix = f"code2wav.pre_transformer.layers.{layer_idx}.self_attn"
    required = [
        f"{prefix}.q_proj.weight",
        f"{prefix}.k_proj.weight",
        f"{prefix}.v_proj.weight",
        f"{prefix}.o_proj.weight",
    ]
    return state_dict is not None and all(k in state_dict for k in required)


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
@pytest.mark.parametrize(
    "batch, seq_len",
    [
        (1, 64),
        (1, 128),
    ],
)
def test_code2wav_attention(device, batch, seq_len):
    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------
    full_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config = full_config.code2wav_config

    if getattr(config, "_attn_implementation", None) is None:
        config._attn_implementation = "eager"

    # --------------------------------------------------------
    # LOAD STATE & WEIGHTS (single shard; skip if code2wav not in this shard)
    # --------------------------------------------------------
    state_dict = load_state_dict()
    layer_idx = 0
    if not _has_code2wav_attn_weights(state_dict, layer_idx):
        pytest.skip(
            f"code2wav.pre_transformer.layers.{layer_idx}.self_attn not in {CHECKPOINT_PATH}; "
            "use a shard that contains code2wav weights."
        )

    # --------------------------------------------------------
    # TORCH MODEL
    # --------------------------------------------------------
    torch_attn = Qwen3OmniMoeCode2WavAttention(config, layer_idx=layer_idx)
    torch_attn = load_weights_into_torch(torch_attn, state_dict, layer_idx)
    torch_attn.eval()

    # --------------------------------------------------------
    # TT MODEL
    # --------------------------------------------------------
    tt_attn = TTNNQwen3OmniMoeCode2WavAttention.from_torch(torch_attn)

    # 🔥 IMPORTANT: ensure exact HF behavior
    tt_attn.use_windowed_attention = False

    set_device(tt_attn, device)
    tt_attn.preprocess_weights()
    tt_attn.move_weights_to_device()

    # --------------------------------------------------------
    # INPUT (bfloat16 to match checkpoint / model weights)
    # --------------------------------------------------------
    dtype = torch.bfloat16
    hidden_states = torch.randn(batch, seq_len, config.hidden_size, dtype=dtype)

    head_dim = config.hidden_size // config.num_attention_heads

    cos = torch.cos(torch.randn(batch, seq_len, head_dim, dtype=dtype))
    sin = torch.sin(torch.randn(batch, seq_len, head_dim, dtype=dtype))

    position_embeddings = (cos, sin)

    attention_mask = None

    # --------------------------------------------------------
    # TORCH FORWARD
    # --------------------------------------------------------
    with torch.no_grad():
        torch_out, _ = torch_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

    # --------------------------------------------------------
    # TT INPUT
    # --------------------------------------------------------
    tt_input = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    tt_mask = ttnn.from_torch(
        attention_mask,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # --------------------------------------------------------
    # TT FORWARD
    # --------------------------------------------------------
    tt_out, _ = tt_attn(
        tt_input,
        position_embeddings=(tt_cos, tt_sin),
        attention_mask=tt_mask,
    )

    tt_out_torch = tt_out.to_torch if isinstance(tt_out, TorchTTNNTensor) else ttnn.to_torch(tt_out)

    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------
    passing, pcc = comp_pcc(torch_out, tt_out_torch)

    print(f"PCC: {pcc}")

    assert passing, f"PCC too low: {pcc}"

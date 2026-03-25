# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from transformers import Qwen3OmniMoeForConditionalGeneration

from models.common.utility_functions import comp_pcc
from models.experimental.tt_symbiote.qwen3omni.tt.audio_attention import TTNNQwenAudioAttention
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.utils.device_management import set_device
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioAttention


MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize(
    "real_weights",
    [
        True,  # Load audio attention from pretrained Qwen3-Omni MoE
        # False,  # Use random weights
    ],
)
def test_qwen_audio_attention(device, seq_len, real_weights):
    # ---------------------------
    # Load PyTorch module (Qwen3-Omni MoE audio attention)
    # ---------------------------
    if real_weights:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        torch_attn = model.thinker.audio_tower.layers[0].self_attn
        torch_attn = torch_attn.to(dtype=torch.bfloat16)
        config = model.config.thinker_config.audio_config
    else:
        from transformers import AutoConfig

        full_config = AutoConfig.from_pretrained(MODEL_NAME)
        config = full_config.thinker_config.audio_config
        torch_attn = Qwen3OmniMoeAudioAttention(config).to(dtype=torch.bfloat16)

    torch_attn.eval()
    torch.set_grad_enabled(False)

    # ---------------------------
    # Convert to TTNN module
    # ---------------------------
    ttnn_attn = TTNNQwenAudioAttention.from_torch(torch_attn)
    set_device(ttnn_attn, device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    # ---------------------------
    # Create input
    # ---------------------------
    embed_dim = config.d_model
    torch_input = torch.randn(seq_len, embed_dim, dtype=torch.bfloat16)

    # cu_seqlens: one sequence of length seq_len -> [0, seq_len]
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.long)

    # PyTorch forward (Qwen3 returns a single tensor)
    with torch.no_grad():
        torch_out = torch_attn(torch_input, cu_seqlens=cu_seqlens)

    # ---------------------------
    # TTNN forward
    # ---------------------------
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_out = ttnn_attn(tt_input)
    # Symbiote may return TorchTTNNTensor; use .to_torch property, not ttnn.to_torch()
    tt_out = tt_out.to_torch if isinstance(tt_out, TorchTTNNTensor) else ttnn.to_torch(tt_out)

    # ---------------------------
    # Compare outputs
    # ---------------------------

    passing, pcc = comp_pcc(torch_out, tt_out)

    print("PCC:", pcc)

    assert passing


import torch
import pytest
import ttnn
from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration

from models.common.utility_functions import comp_pcc
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.utils.device_management import set_device

# TT implementation
from models.experimental.tt_symbiote.qwen3omni.tt.code2wav_attn import (
    TTNNQwen3OmniMoeCode2WavAttention,
)

# HF reference
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavAttention,
)

MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


# ------------------------------------------------------------
# STATE_DICT KEY MAPPING (code2wav attention only)
# ------------------------------------------------------------
# Verified against Qwen/Qwen3-Omni-30B-A3B-Instruct (from_pretrained state_dict).
#
# State dict key (replace {i} with layer index, e.g. 0)  →  torch_attn attribute
# -----------------------------------------------------------------------------------
# code2wav.pre_transformer.layers.{i}.self_attn.q_proj.weight  →  torch_attn.q_proj.weight
# code2wav.pre_transformer.layers.{i}.self_attn.k_proj.weight  →  torch_attn.k_proj.weight
# code2wav.pre_transformer.layers.{i}.self_attn.v_proj.weight  →  torch_attn.v_proj.weight
# code2wav.pre_transformer.layers.{i}.self_attn.o_proj.weight  →  torch_attn.o_proj.weight
#
# Optional (only if config.attention_bias is True; Qwen3-Omni-30B has no bias):
#   code2wav.pre_transformer.layers.{i}.self_attn.q_proj.bias  →  torch_attn.q_proj.bias
#   code2wav.pre_transformer.layers.{i}.self_attn.k_proj.bias  →  torch_attn.k_proj.bias
#   code2wav.pre_transformer.layers.{i}.self_attn.v_proj.bias  →  torch_attn.v_proj.bias
#   code2wav.pre_transformer.layers.{i}.self_attn.o_proj.bias  →  torch_attn.o_proj.bias
#
# Not part of self_attn (parent layer): code2wav.pre_transformer.layers.{i}.self_attn_layer_scale.scale
# q_norm / k_norm are nn.Identity() in this attention (no params).
# -----------------------------------------------------------------------------------


def load_code2wav_attn_from_state_dict(torch_attn, state_dict, layer_idx):
    """Load code2wav attention weights from a state_dict (full model or single shard)."""
    prefix = f"code2wav.pre_transformer.layers.{layer_idx}.self_attn"
    torch_attn.q_proj.weight.data = state_dict[f"{prefix}.q_proj.weight"]
    torch_attn.k_proj.weight.data = state_dict[f"{prefix}.k_proj.weight"]
    torch_attn.v_proj.weight.data = state_dict[f"{prefix}.v_proj.weight"]
    torch_attn.o_proj.weight.data = state_dict[f"{prefix}.o_proj.weight"]
    if f"{prefix}.q_proj.bias" in state_dict:
        torch_attn.q_proj.bias.data = state_dict[f"{prefix}.q_proj.bias"]
        torch_attn.k_proj.bias.data = state_dict[f"{prefix}.k_proj.bias"]
        torch_attn.v_proj.bias.data = state_dict[f"{prefix}.v_proj.bias"]
        torch_attn.o_proj.bias.data = state_dict[f"{prefix}.o_proj.bias"]
    return torch_attn


def has_code2wav_attn_in_state_dict(state_dict, layer_idx=0):
    """True if state_dict contains the code2wav attention weights for the given layer."""
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
@pytest.mark.parametrize(
    "real_weights",
    [
        True,  # Load code2wav attention from pretrained Qwen3-Omni MoE
        # False,  # Use random weights
    ],
)
def test_code2wav_attention(device, batch, seq_len, real_weights):
    # --------------------------------------------------------
    # Load PyTorch module (code2wav attention, layer 0)
    # --------------------------------------------------------
    layer_idx = 0
    if real_weights:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        torch_attn = model.code2wav.pre_transformer.layers[layer_idx].self_attn
        torch_attn = torch_attn.to(dtype=torch.bfloat16)
        config = model.config.code2wav_config
    else:
        full_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
        config = full_config.code2wav_config
        torch_attn = Qwen3OmniMoeCode2WavAttention(config, layer_idx=layer_idx).to(dtype=torch.bfloat16)

    if getattr(config, "_attn_implementation", None) is None:
        config._attn_implementation = "eager"

    torch_attn.eval()
    torch.set_grad_enabled(False)

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

    tt_mask = None
    if attention_mask is not None:
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

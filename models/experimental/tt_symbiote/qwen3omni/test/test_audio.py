# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from safetensors.torch import safe_open
from transformers import AutoConfig

from models.common.utility_functions import comp_pcc
from models.experimental.qwen3omni.tt.audio_attention import TTNNQwenAudioAttention
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.utils.device_management import set_device
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioAttention


MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
CHECKPOINT_PATH = "/home/ubuntu/tt-metal/models/experimental/qwen3omni/checkpoints/model-00001-of-00015.safetensors"


def load_audio_attention_weights(layer=0):
    prefix = f"thinker.audio_tower.layers.{layer}.self_attn"
    weights = {}
    with safe_open(CHECKPOINT_PATH, framework="pt", device="cpu") as f:
        weights["q_w"] = f.get_tensor(f"{prefix}.q_proj.weight")
        weights["q_b"] = f.get_tensor(f"{prefix}.q_proj.bias")

        weights["k_w"] = f.get_tensor(f"{prefix}.k_proj.weight")
        weights["k_b"] = f.get_tensor(f"{prefix}.k_proj.bias")

        weights["v_w"] = f.get_tensor(f"{prefix}.v_proj.weight")
        weights["v_b"] = f.get_tensor(f"{prefix}.v_proj.bias")

        weights["o_w"] = f.get_tensor(f"{prefix}.out_proj.weight")
        weights["o_b"] = f.get_tensor(f"{prefix}.out_proj.bias")

    return weights


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
@pytest.mark.parametrize("seq_len", [128])
def test_qwen_audio_attention(device, seq_len):
    # ---------------------------
    # Load weights (Qwen3-Omni checkpoint)
    # ---------------------------
    weights = load_audio_attention_weights(layer=0)

    # ---------------------------
    # Create PyTorch module (Qwen3-Omni audio attention)
    # ---------------------------
    full_config = AutoConfig.from_pretrained(MODEL_NAME)
    config = full_config.thinker_config.audio_config

    torch_attn = Qwen3OmniMoeAudioAttention(config)
    torch_attn.q_proj.weight.data = weights["q_w"]
    torch_attn.q_proj.bias.data = weights["q_b"]
    torch_attn.k_proj.weight.data = weights["k_w"]
    torch_attn.k_proj.bias.data = weights["k_b"]
    torch_attn.v_proj.weight.data = weights["v_w"]
    torch_attn.v_proj.bias.data = weights["v_b"]
    torch_attn.out_proj.weight.data = weights["o_w"]
    torch_attn.out_proj.bias.data = weights["o_b"]
    torch_attn.eval()

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

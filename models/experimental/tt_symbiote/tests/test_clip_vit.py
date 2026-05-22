# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test for CLIP ViT model with TTNN backend."""

import pytest
import torch
from torch import nn
from transformers import CLIPModel
from transformers.activations import QuickGELUActivation
from transformers.models.clip.modeling_clip import CLIPAttention

from models.experimental.tt_symbiote.modules.attention import (
    PytorchFusedQKVSelfAttention,
    SelfAttentionConfig,
    TTNNFusedQKVSelfAttention,
    TTNNSDPAAttention,
    TTNNSelfAttention,
)
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.activation import TTNNGelu
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


class TTNNCLIPAttention(TTNNSelfAttention):
    """TTNN-accelerated CLIP Attention -- adapts CLIPAttention's
    q_proj/k_proj/v_proj naming to the fused QKV pattern."""

    @classmethod
    def from_torch(cls, clip_attn: CLIPAttention):
        config = SelfAttentionConfig(
            hidden_size=clip_attn.embed_dim,
            num_attention_heads=clip_attn.num_heads,
        )
        new_attn = cls(attention_config=config)
        new_attn._fallback_torch_layer = clip_attn
        new_attn.query_key_value = TTNNFusedQKVSelfAttention.from_torch(
            PytorchFusedQKVSelfAttention(
                clip_attn.q_proj,
                clip_attn.k_proj,
                clip_attn.v_proj,
                clip_attn.num_heads,
                clip_attn.embed_dim,
            ),
        )
        new_attn.out_proj = TTNNLinear.from_torch(clip_attn.out_proj)
        new_attn.sdpa = TTNNSDPAAttention()
        for child in [new_attn.query_key_value, new_attn.out_proj, new_attn.sdpa]:
            child._bypass_tensor_wrapping = False
            child._bypass_explicitly_set = True
        return new_attn

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        """CLIP-compatible forward that delegates to TTNNSelfAttention."""
        result = super().forward(hidden_states, head_mask=None, output_attentions=False)
        context_layer = result[0]
        attn_output = self.out_proj(context_layer)
        return (attn_output, None)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_clip_vit(device):
    """Test CLIP ViT model with TTNN acceleration."""

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(torch.bfloat16)
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.LayerNorm: TTNNLayerNorm,
        CLIPAttention: TTNNCLIPAttention,
        QuickGELUActivation: TTNNGelu,
    }
    register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)
    model.eval()
    torch.set_grad_enabled(False)
    DispatchManager.clear_timings()
    result = model(
        input_ids=torch.randint(0, 49408, (1, 77)),
        pixel_values=torch.randn(1, 3, 224, 224, dtype=torch.bfloat16),
    )
    print(f"Image embeds shape: {result.image_embeds.shape}")
    print(f"Text embeds shape: {result.text_embeds.shape}")
    DispatchManager.save_stats_to_file("clip_vit_timing_stats.csv")

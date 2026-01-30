# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for DeepSeek-OCR model with TTNN backend."""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import pytest
import ttnn
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.activation import TTNNSilu, TTNNGelu
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm, TTNNRMSNorm
from models.experimental.tt_symbiote.modules.attention import LlamaAttention
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.conv import TTNNConv2dNHWC
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from tqdm import tqdm
from torch.nn import functional as F


class DeepseekOCRAttention(LlamaAttention):
    @classmethod
    def from_torch(cls, llama_attn: "LlamaAttention"):
        new_attn = cls()
        new_attn._fallback_torch_layer = llama_attn
        new_attn.num_key_value_groups = getattr(llama_attn, "num_key_value_groups", 1)
        # Fuse Q/K/V for self-attention (zero-pad K bias)
        new_attn.qkv_same_shape = (
            llama_attn.q_proj.weight.shape == llama_attn.k_proj.weight.shape
            and llama_attn.q_proj.weight.shape == llama_attn.v_proj.weight.shape
        )
        if new_attn.qkv_same_shape:
            new_attn.init_fused_parameters(llama_attn.config.num_attention_heads, llama_attn.config.hidden_size)
        else:
            new_attn.init_parameters()
        new_attn.scaling = llama_attn.head_dim**-0.5
        return new_attn

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,  # will become mandatory in v4.46
        **kwargs,
    ):
        if attention_mask is not None:
            print(
                "Warning: attention_mask is not None, but TTNN LlamaAttention does not support it yet."
            )  # --- IGNORE ---
        past_key_values = kwargs.get("past_key_value", past_key_values) if past_key_values is None else past_key_values
        if self.qkv_same_shape:
            query_states, key_states, value_states = self.qkv_proj(hidden_states)
        else:
            input_shape = list(hidden_states.shape)[:-1]
            hidden_shape = (*input_shape, -1, self.torch_layer.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            print("Warning: position_embeddings is None, computing from position_ids.")  # --- IGNORE ---
            cos, sin = self.torch_layer.rotary_emb(value_states.to_torch, TorchTTNNTensor(position_ids).to_torch)
        else:
            cos, sin = position_embeddings

        query_states, key_states = self.rope(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.torch_layer.layer_idx, cache_kwargs
            )

        original_q_len = query_states.shape[2]
        kv_len = key_states.shape[2]

        if self.torch_layer.is_causal and original_q_len < kv_len:
            # Pad query: [B, H, q_len, D] -> [B, H, kv_len, D]
            pad_len = kv_len - original_q_len
            # Create zero padding on device
            pad_shape = (query_states.shape[0], query_states.shape[1], pad_len, query_states.shape[3])
            zero_pad = ttnn.zeros(
                pad_shape,
                device=hidden_states.device(),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=hidden_states.dtype,
            )
            query_states = ttnn.concat([zero_pad, query_states.to_ttnn], dim=2)

        attn_out = self.sdpa(
            self,
            query_states,
            key_states,
            value_states,
            None,
            dropout=0.0,
            scaling=self.scaling,
            is_causal=self.torch_layer.is_causal,
            transpose_output=False,
        )
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out.to_ttnn)
        attn_out = ttnn.squeeze(attn_out, 1)
        # Slice output if query was padded
        if self.torch_layer.is_causal and original_q_len < kv_len:
            # Slice: [B, kv_len, D] -> [B, q_len, D]
            attn_out = attn_out[:, -original_q_len:, :]

        return self.o_proj(attn_out), None, past_key_values


def get_abs_pos_sam(abs_pos, tgt_size):
    dtype = abs_pos.dtype

    src_size = abs_pos.size(1)

    if src_size != tgt_size:
        old_pos_embed = abs_pos.permute(0, 3, 1, 2)
        old_pos_embed = old_pos_embed.to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        return new_pos_embed
    else:
        return abs_pos


class LayerNorm2d(nn.Module):
    def __init__(self, old_layer) -> None:
        super().__init__()
        self.weight = old_layer.weight
        self.bias = old_layer.bias
        self.eps = old_layer.eps

    @classmethod
    def from_torch(cls, old_layer):
        return cls(old_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(3, keepdim=True)
        s = (x - u).pow(2).mean(3, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


class ImageEncoderViT(nn.Module):
    def __init__(self, old_layer) -> None:
        super().__init__()
        self.img_size = old_layer.img_size

        self.patch_embed = old_layer.patch_embed

        self.pos_embed = old_layer.pos_embed

        self.blocks = old_layer.blocks

        self.neck = nn.Sequential(
            *[l if isinstance(l, nn.Conv2d) else LayerNorm2d(l) for l in old_layer.neck.children()]
        )

        self.net_2 = old_layer.net_2
        self.net_3 = old_layer.net_3

    @classmethod
    def from_torch(cls, old_layer):
        return cls(old_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # BCHW -> BHWC
        if self.pos_embed is not None:
            # x = x + self.pos_embed
            x = x + get_abs_pos_sam(self.pos_embed, x.size(1))

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x)
        x2 = self.net_2(x)
        x3 = self.net_3(x2)
        return x3.permute(0, 3, 1, 2)  # BHWC -> BCHW


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_deepseek_ocr(device):
    """Test DeepSeek-OCR model with TTNN acceleration."""

    model_name = "deepseek-ai/DeepSeek-OCR"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    nn_to_nn = {
        model.model.sam_model.__class__: ImageEncoderViT,
        model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,
    }
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.SiLU: TTNNSilu,
        nn.GELU: TTNNGelu,
        nn.LayerNorm: TTNNLayerNorm,
        nn.Conv2d: TTNNConv2dNHWC,
        model.model.layers[0].self_attn.__class__: DeepseekOCRAttention,
    }

    # prompt = "<image>\nFree OCR. "
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    image_file = "test.jpg"
    output_path = "output_deepseek_ocr/"

    # infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

    # Tiny: base_size = 512, image_size = 512, crop_mode = False
    # Small: base_size = 640, image_size = 640, crop_mode = False
    # Base: base_size = 1024, image_size = 1024, crop_mode = False
    # Large: base_size = 1280, image_size = 1280, crop_mode = False

    # Gundam: base_size = 1024, image_size = 640, crop_mode = True
    modules1 = register_module_replacement_dict(model, nn_to_nn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)
    for k, v in tqdm({**modules1, **modules2}.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=output_path,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=True,
        eval_mode=True,
    )
    DispatchManager.save_stats_to_file("deepseek_ocr_timing_stats.csv")
    print(res)

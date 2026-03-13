# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for DeepSeek-OCR model with TTNN backend.

Includes Saurav's TTNN modules: TTNNSAMAttention, TTNNDeepseekV2MoE,
TTNNDeepseekOCRMoEGate, TTNNClipVisionEmbeddings, TTNNNoTPAttention,
and TTNNNoTPFeedForward alongside the original leaf-op replacements
and ImageEncoderViT wrapper.
"""

import os
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import pytest
from models.experimental.tt_symbiote.modules.activation import TTNNSilu, TTNNGelu
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm, TTNNRMSNorm
from models.experimental.tt_symbiote.modules.attention import LlamaAttention, TTNNSAMAttention
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.conv import TTNNConv2dNHWC
from models.experimental.tt_symbiote.modules.moe import TTNNDeepseekV2MoE
from models.experimental.tt_symbiote.tests.deepseek_ocr_vision_model.ttnn_symbiote_vit_model import (
    TTNNClipVisionEmbeddings,
    TTNNNoTPAttention,
    TTNNNoTPFeedForward,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from tqdm import tqdm
from torch.nn import functional as F


def get_abs_pos_sam(abs_pos, tgt_size):
    dtype = abs_pos.dtype
    src_size = abs_pos.size(1)
    if src_size != tgt_size:
        old_pos_embed = abs_pos.permute(0, 3, 1, 2).to(torch.float32)
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).to(dtype)
        return new_pos_embed.permute(0, 2, 3, 1)
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


class TTNNTransformerBlock(nn.Module):
    """Wrapper for NoTPTransformerBlock that routes submodule calls through __call__
    instead of .forward(), enabling TTNNModule dispatch and timing."""

    def __init__(self, old_layer) -> None:
        super().__init__()
        self.self_attn = old_layer.self_attn
        self.mlp = old_layer.mlp
        self.layer_norm1 = old_layer.layer_norm1
        self.layer_norm2 = old_layer.layer_norm2

    @classmethod
    def from_torch(cls, old_layer):
        return cls(old_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.self_attn(self.layer_norm1(x))
        h = x + residual
        out = h + self.mlp(self.layer_norm2(h))
        return out


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
        x = self.patch_embed(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.pos_embed is not None:
            x = x + get_abs_pos_sam(self.pos_embed, x.size(1))
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x)
        x2 = self.net_2(x)
        x3 = self.net_3(x2)
        return x3.permute(0, 3, 1, 2)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_deepseek_ocr(device):
    """Full DeepSeek-OCR end-to-end test with TTNN acceleration and timing CSV output."""

    model_name = "deepseek-ai/DeepSeek-OCR"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )

    # Module-level replacements (nn_to_nn): custom wrappers that rewrite forward()
    vit_block_class = model.model.vision_model.transformer.layers[0].__class__
    nn_to_nn = {
        model.model.sam_model.__class__: ImageEncoderViT,
        model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,
        vit_block_class: TTNNTransformerBlock,
    }

    # Leaf-op and module replacements (nn_to_ttnn): TTNN-backed modules
    sam_attn_class = model.model.sam_model.blocks[0].attn.__class__
    moe_class = model.model.layers[1].mlp.__class__
    vit_embeddings_class = model.model.vision_model.embeddings.__class__
    vit_attn_class = model.model.vision_model.transformer.layers[0].self_attn.__class__
    vit_mlp_class = model.model.vision_model.transformer.layers[0].mlp.__class__
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.SiLU: TTNNSilu,
        nn.GELU: TTNNGelu,
        nn.LayerNorm: TTNNLayerNorm,
        nn.Conv2d: TTNNConv2dNHWC,
        model.model.layers[0].self_attn.__class__: LlamaAttention,
        sam_attn_class: TTNNSAMAttention,
        moe_class: TTNNDeepseekV2MoE,
        vit_embeddings_class: TTNNClipVisionEmbeddings,
        vit_attn_class: TTNNNoTPAttention,
        vit_mlp_class: TTNNNoTPFeedForward,
    }

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    test_dir = os.path.dirname(os.path.abspath(__file__))
    image_file = os.path.join(test_dir, "deepseek_ocr_vision_model", "test.png")
    output_path = "output_deepseek_ocr/"

    modules1 = register_module_replacement_dict(model, nn_to_nn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)
    for k, v in tqdm({**modules1, **modules2}.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    model.eval()
    torch.set_grad_enabled(False)
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

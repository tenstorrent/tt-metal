# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for DeepSeek-OCR model with TTNN backend.

Uses LlamaAttention, TTNNRMSNorm, TTNNSAMAttention, TTNNImageEncoderViT,
TTNNVitModel, TTNNDeepseekV2MoE, and leaf-op replacements.
All major modules run on device.
"""

import os
import shutil
from datetime import datetime

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import pytest
from models.experimental.tt_symbiote.modules.activation import TTNNSilu, TTNNGelu
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm, TTNNRMSNorm
from models.experimental.tt_symbiote.modules.attention import LlamaAttention, TTNNSAMAttention
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.conv import TTNNConv2dNHWC, TTNNImageEncoderViT, TTNNVitModel
from models.experimental.tt_symbiote.modules.moe import TTNNDeepseekV2MoE
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from tqdm import tqdm

# --- HuggingFace model compatibility patches ---
# The DeepSeek-OCR HuggingFace model hardcodes .cuda() calls in its infer()
# method, which fails on Tenstorrent hardware (CPU-only PyTorch, no CUDA).
# tt_symbiote handles device placement via set_device(), so .cuda() is a no-op.
torch.Tensor.cuda = lambda self, *args, **kwargs: self

# The model's prepare_inputs_for_generation uses DynamicCache.seen_tokens,
# which was removed in transformers >=4.57. Restore it as a property.
from transformers import DynamicCache

if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: None

if not hasattr(DynamicCache, "get_usable_length"):

    def _get_usable_length(self, new_seq_length, layer_idx=0):
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    DynamicCache.get_usable_length = _get_usable_length


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
        model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,
    }

    sam_attn_class = model.model.sam_model.blocks[0].attn.__class__
    moe_class = model.model.layers[1].mlp.__class__
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.SiLU: TTNNSilu,
        nn.GELU: TTNNGelu,
        nn.LayerNorm: TTNNLayerNorm,
        nn.Conv2d: TTNNConv2dNHWC,
        model.model.layers[0].self_attn.__class__: LlamaAttention,
        sam_attn_class: TTNNSAMAttention,
        moe_class: TTNNDeepseekV2MoE,
        model.model.vision_model.__class__: TTNNVitModel,
        model.model.sam_model.__class__: TTNNImageEncoderViT,
    }

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    image_file = "test.png"
    output_path = os.path.join(os.path.dirname(__file__), "output_deepseek_ocr")

    modules1 = register_module_replacement_dict(model, nn_to_nn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)
    for k, v in tqdm({**modules1, **modules2}.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    model.eval()
    torch.set_grad_enabled(False)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_path, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    DispatchManager.save_stats_to_file(os.path.join(run_dir, f"timing_stats_{timestamp}.csv"))

    with open(os.path.join(run_dir, "ocr_output.md"), "w") as f:
        f.write(res)

    if os.path.exists(image_file):
        shutil.copy2(image_file, os.path.join(run_dir, os.path.basename(image_file)))

    print(f"\nResults saved to {run_dir}/")
    print(res)

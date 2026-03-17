# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for DeepSeek-OCR model with TTNN backend.

Uses monolithic TTNNVitModel for the full ViT pipeline, TTNNSAMAttention,
TTNNDeepseekV2MoE, LlamaAttention, and leaf-op replacements alongside
the ImageEncoderViT wrapper.
"""

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
from models.experimental.tt_symbiote.modules.conv import TTNNVitModel
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.conv import TTNNImageEncoderViT
from tqdm import tqdm


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
    nn_to_nn = {
        model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,
    }

    # Leaf-op and module replacements (nn_to_ttnn): TTNN-backed modules
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

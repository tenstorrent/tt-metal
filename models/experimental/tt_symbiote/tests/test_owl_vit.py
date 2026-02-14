# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Owl-ViT model with TTNN backend."""

import requests
import torch
from PIL import Image
from torch import nn
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.models.vit.modeling_vit import ViTSelfAttention

from models.experimental.tt_symbiote.modules.attention import TTNNViTSelfAttention
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


def test_owl_vit(device):
    """Test Owl-ViT model with TTNN acceleration."""

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = [["a photo of a cat", "a photo of a dog"]]
    processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    inputs = processor(text=texts, images=image, return_tensors="pt")
    input_tensors = [inputs["pixel_values"], inputs["attention_mask"], inputs["input_ids"]]
    model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        ViTSelfAttention: TTNNViTSelfAttention,
        nn.LayerNorm: TTNNLayerNorm,
    }
    register_module_replacement_dict(model, nn_to_ttnn, model_config={"program_config_ffn": {}})
    set_device(model, device)
    result = model(**inputs)
    print(result)

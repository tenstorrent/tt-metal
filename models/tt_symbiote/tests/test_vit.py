"""Test for ViT model with TTNN backend."""

import torch
from torch import nn
from transformers import AutoModelForImageClassification
from transformers.models.vit.modeling_vit import ViTSelfAttention

from models.tt_symbiote.modules.attention import TTNNViTSelfAttention
from models.tt_symbiote.modules.linear import TTNNLinear
from models.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.tt_symbiote.utils.device_management import set_device
from models.tt_symbiote.utils.module_replacement import register_module_replacement_dict


def test_vit(device):
    """Test ViT model with TTNN acceleration."""
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.LayerNorm: TTNNLayerNorm,
        ViTSelfAttention: TTNNViTSelfAttention,
    }
    register_module_replacement_dict(model, nn_to_ttnn, model_config={"program_config_ffn": {}})
    set_device(model, device)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    result = model(torch.randn(1, 3, 224, 224))
    print(result.logits)

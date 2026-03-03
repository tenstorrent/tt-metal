# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn
from transformers import AutoModel
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.tt_symbiote.modules.activation import TTNNSilu, TTNNGelu
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.conv import TTNNConv2dNHWC
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.core.run_config import DispatchManager


from tqdm import tqdm
import torch.profiler


class VitModel(nn.Module):
    def __init__(self, old_layer) -> None:
        super().__init__()
        self.embeddings = old_layer.embeddings
        self.transformer = old_layer.transformer
        self.pre_layrnorm = old_layer.pre_layrnorm

    @classmethod
    def from_torch(cls, old_layer):
        return cls(old_layer)

    def forward(self, x: torch.Tensor, patch_embeds) -> torch.Tensor:
        x = self.embeddings(x, patch_embeds)
        hidden_states = self.pre_layrnorm(x)
        output = self.transformer(hidden_states)

        return output


class OpenclipWrapper(nn.Module):
    def __init__(self, Openclip):
        super().__init__()
        self.Openclip = Openclip

    def forward(self, x, y):
        return self.Openclip(x, y)


@pytest.fixture(scope="module")
def ocr_model():
    """Load OCR model (HuggingFace)"""
    model = AutoModel.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().to(torch.bfloat16)
    return model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_tt_vit_pcc(device, ocr_model):
    """Run torch VIT and TT VIT with same input; assert PCC >= PCC_THRESHOLD."""
    vision_model = ocr_model.model.vision_model
    model = OpenclipWrapper(vision_model)
    torch.manual_seed(42)
    ref_input_patches = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/vision_model_input_patches.pt"
    )
    ref_input_features = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/vision_model_input_local_features.pt",
        weights_only=False,
    )
    ref_out = vision_model(ref_input_patches, ref_input_features)

    nn_to_nn = {
        model.Openclip.__class__: VitModel,
    }

    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.SiLU: TTNNSilu,
        nn.GELU: TTNNGelu,
        nn.LayerNorm: TTNNLayerNorm,
        nn.Conv2d: TTNNConv2dNHWC,
    }

    modules1 = register_module_replacement_dict(model, nn_to_nn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)

    for k, v in tqdm({**modules1, **modules2}.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    model.eval()

    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    tt_out = model(ref_input_patches, ref_input_features)

    print("#" * 20)
    print(model)
    print("#" * 20)

    passed, message = check_with_pcc(ref_out.float(), tt_out.float(), pcc=0.99)
    logger.info(f"TT VIT PCC : {message}")
    assert passed, f"TT VIT PCC check failed: {message}"

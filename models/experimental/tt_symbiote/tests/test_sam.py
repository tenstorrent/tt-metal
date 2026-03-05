# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SAM image encoder: full TTNN vs torch PCC test (PCC >= 0.99)."""
import pytest
import torch
import ttnn
from torch import nn
from transformers import AutoModel
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.tt_symbiote.modules.attention import TTNNSAMAttention
from models.experimental.tt_symbiote.modules.conv import TTNNSAMBlock, TTNNImageEncoderViT
from models.common.auto_compose import to_torch_auto_compose
from models.experimental.tt_symbiote.utils.device_management import set_device


@pytest.fixture(scope="module")
def ocr_model():
    """Load OCR model (HuggingFace); SAM is ocr_model.model.sam_model."""
    model = AutoModel.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().to(torch.bfloat16)
    return model


class SAMWrapper(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.sam_model = sam_model

    def forward(self, x):
        return self.sam_model(x)


@pytest.mark.parametrize("image_size", [640])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_tt_sam_attention_pcc(device, ocr_model, image_size):
    """Run torch SAM attention and TTNN SAM attention with same input; assert PCC >= 0.99."""
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    attn = ocr_model.model.sam_model.blocks[0].attn
    B, H, W, C = 2, 14, 14, 768
    x = torch.randn((B, H, W, C), dtype=torch.bfloat16)

    ref_out = attn(x)

    ttnn_attn = TTNNSAMAttention.from_torch(attn)
    set_device(ttnn_attn, device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    tt_out = ttnn_attn(x)
    tt_out_torch = (tt_out.to_torch if hasattr(tt_out, "to_torch") else ttnn.to_torch(tt_out)).float()
    ref_out_float = ref_out.float()

    passed, message = check_with_pcc(ref_out_float, tt_out_torch, pcc=0.99)
    logger.info(f"TT SAM attention PCC (image_size={image_size}): {message}")
    assert passed, f"TT SAM attention PCC check failed: {message}"


@pytest.mark.parametrize("image_size", [640])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_tt_sam_block_pcc(device, ocr_model, image_size):
    """Run torch SAM block and TTNN SAM block with same input; assert PCC >= 0.99."""
    torch.manual_seed(42)
    torch.set_grad_enabled(False)

    block = ocr_model.model.sam_model.blocks[0]
    B, H, W, C = 6, 40, 40, 768
    x = torch.randn((B, H, W, C), dtype=torch.bfloat16)

    ref_out = block(x)

    ttnn_block = TTNNSAMBlock.from_torch(block, window_size=0)
    set_device(ttnn_block, device)
    ttnn_block.preprocess_weights()
    ttnn_block.move_weights_to_device()

    tt_out = ttnn_block(x)
    if hasattr(tt_out, "to_torch") and not callable(getattr(tt_out, "to_torch")):
        tt_out_torch = tt_out.to_torch
    else:
        tt_out_torch = to_torch_auto_compose(tt_out, device=ttnn_block.device)
    tt_out_torch = (
        tt_out_torch.float() if isinstance(tt_out_torch, torch.Tensor) else torch.as_tensor(tt_out_torch).float()
    )
    ref_out_float = ref_out.float()

    passed, message = check_with_pcc(ref_out_float, tt_out_torch, pcc=0.99)
    logger.info(f"TT SAM block PCC (image_size={image_size}): {message}")
    assert passed, f"TT SAM block PCC check failed: {message}"


@pytest.mark.parametrize("image_size", [640])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_tt_sam_module_pcc(device, ocr_model, image_size):
    """Full SAM image encoder: torch vs TTNN (TTNNImageEncoderViT). Asserts PCC >= 0.99."""
    sam_model = ocr_model.model.sam_model
    model = SAMWrapper(sam_model)
    torch.manual_seed(42)
    x = torch.load("sam_input.pt")
    x_nchw = x.permute(0, 3, 1, 2) if x.dim() == 4 and x.shape[-1] == 3 else x

    ref_out = sam_model(x_nchw)
    tt_encoder = TTNNImageEncoderViT.from_torch(sam_model)
    object.__setattr__(model, "sam_model", tt_encoder)
    set_device(model, device)
    tt_encoder.preprocess_weights()
    tt_encoder.move_weights_to_device()
    model.eval()
    torch.set_grad_enabled(False)

    tt_out = tt_encoder.forward(x_nchw)
    if hasattr(tt_out, "to_torch") and not callable(getattr(tt_out, "to_torch")):
        tt_out = tt_out.to_torch
    else:
        tt_out = to_torch_auto_compose(tt_out, device=tt_encoder.device)
    tt_out = tt_out.float() if isinstance(tt_out, torch.Tensor) else torch.as_tensor(tt_out).float()

    passed, message = check_with_pcc(ref_out.float(), tt_out, pcc=0.99)
    logger.info(f"TT SAM full encoder PCC (image_size={image_size}): {message}")
    assert passed, f"TT SAM PCC check failed: {message}"

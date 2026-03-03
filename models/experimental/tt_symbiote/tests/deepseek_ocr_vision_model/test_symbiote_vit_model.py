# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoModel
import pytest
import torch

from tqdm import tqdm

from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.tests.deepseek_ocr_vision_model.ttnn_symbiote_vit_model import (
    TTNNClipVisionEmbeddings,
    TTNNNoTPAttention,
    TTNNNoTPFeedForward,
    TTNNNoTPTransformerBlock,
    TTNNNoTPTransformer,
    TTNNVitModel,
)

from tests.ttnn.utils_for_testing import check_with_pcc

from loguru import logger


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
def test_tt_ClipVisionEmbeddings_pcc(device, ocr_model):
    """Run torch ClipVisionEmbeddings and TT ClipVisionEmbeddings with same input; assert PCC >= PCC_THRESHOLD."""
    vision_model = ocr_model.model.vision_model

    torch.manual_seed(42)
    ref_input_patches = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_patches.pt"
    )
    ref_input_features = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_local_features.pt",
        weights_only=False,
    )

    with torch.no_grad():
        ref_out = vision_model(ref_input_patches, ref_input_features)

    ref_input_features = ref_input_features.flatten(2).transpose(1, 2)

    nn_to_ttnn = {
        vision_model.embeddings.__class__: TTNNClipVisionEmbeddings,
    }

    modules = register_module_replacement_dict(vision_model, nn_to_ttnn, model_config=None)
    set_device(vision_model, device)

    for k, v in tqdm({**modules}.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    vision_model.eval()

    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    tt_out = vision_model(ref_input_patches, ref_input_features)
    DispatchManager.save_stats_to_file(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/deepseek_ocr_module_timing_stats.csv"
    )

    passed, message = check_with_pcc(ref_out.float(), tt_out.float(), pcc=0.99)
    logger.info(f"TT VIT PCC : {message}")
    assert passed, f"TT VIT PCC check failed: {message}"


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
def test_tt_NoTPAttention_pcc(device, ocr_model):
    """Run torch NoTPAttention and TT NoTPAttention with same input; assert PCC >= PCC_THRESHOLD."""

    vision_model = ocr_model.model.vision_model

    torch.manual_seed(42)
    ref_input_patches = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_patches.pt"
    )
    ref_input_features = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_local_features.pt",
        weights_only=False,
    )

    with torch.no_grad():
        ref_out = vision_model(ref_input_patches, ref_input_features)

    nn_to_ttnn = {
        vision_model.transformer.layers[0].self_attn.__class__: TTNNNoTPAttention,
    }

    modules = register_module_replacement_dict(vision_model, nn_to_ttnn, model_config=None)
    set_device(vision_model, device)

    for k, v in tqdm({**modules}.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    vision_model.eval()

    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    tt_out = vision_model(ref_input_patches, ref_input_features)
    DispatchManager.save_stats_to_file(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/deepseek_ocr_module_timing_stats.csv"
    )

    passed, message = check_with_pcc(ref_out.float(), tt_out.float(), pcc=0.99)
    logger.info(f"TT VIT PCC : {message}")
    assert passed, f"TT VIT PCC check failed: {message}"


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
def test_tt_NoTPFeedForward_pcc(device, ocr_model):
    """Run torch NoTPFeedForward and TT NoTPFeedForward with same input; assert PCC >= PCC_THRESHOLD."""

    vision_model = ocr_model.model.vision_model

    torch.manual_seed(42)
    ref_input_patches = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_patches.pt"
    )
    ref_input_features = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_local_features.pt",
        weights_only=False,
    )

    with torch.no_grad():
        ref_out = vision_model(ref_input_patches, ref_input_features)

    nn_to_ttnn = {
        vision_model.transformer.layers[0].mlp.__class__: TTNNNoTPFeedForward,
    }

    modules = register_module_replacement_dict(vision_model, nn_to_ttnn, model_config=None)
    set_device(vision_model, device)

    for k, v in tqdm({**modules}.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    vision_model.eval()

    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    tt_out = vision_model(ref_input_patches, ref_input_features)
    DispatchManager.save_stats_to_file(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/deepseek_ocr_module_timing_stats.csv"
    )

    passed, message = check_with_pcc(ref_out.float(), tt_out.float(), pcc=0.99)
    logger.info(f"TT VIT PCC : {message}")
    assert passed, f"TT VIT PCC check failed: {message}"


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
def test_tt_NoTPTransformerBlock_pcc(device, ocr_model):
    """Run torch NoTPTransformerBlock and TT NoTPTransformerBlock with same input; assert PCC >= PCC_THRESHOLD."""

    vision_model = ocr_model.model.vision_model

    torch.manual_seed(42)
    ref_input_patches = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_patches.pt"
    )
    ref_input_features = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_local_features.pt",
        weights_only=False,
    )

    with torch.no_grad():
        ref_out = vision_model(ref_input_patches, ref_input_features)

    nn_to_ttnn = {
        vision_model.transformer.layers[0].__class__: TTNNNoTPTransformerBlock,
    }

    modules = register_module_replacement_dict(vision_model, nn_to_ttnn, model_config=None)
    set_device(vision_model, device)

    for k, v in tqdm({**modules}.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    vision_model.eval()

    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    tt_out = vision_model(ref_input_patches, ref_input_features)
    DispatchManager.save_stats_to_file(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/deepseek_ocr_module_timing_stats.csv"
    )

    passed, message = check_with_pcc(ref_out.float(), tt_out.float(), pcc=0.99)
    logger.info(f"TT VIT PCC : {message}")
    assert passed, f"TT VIT PCC check failed: {message}"


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
def test_tt_NoTPTransformer_pcc(device, ocr_model):
    """Run torch NoTPTransformer and TT NoTPTransformer with same input; assert PCC >= PCC_THRESHOLD."""

    vision_model = ocr_model.model.vision_model

    torch.manual_seed(42)
    ref_input_patches = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_patches.pt"
    )
    ref_input_features = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_local_features.pt",
        weights_only=False,
    )

    with torch.no_grad():
        ref_out = vision_model(ref_input_patches, ref_input_features)

    nn_to_ttnn = {
        vision_model.transformer.__class__: TTNNNoTPTransformer,
    }

    modules = register_module_replacement_dict(vision_model, nn_to_ttnn, model_config=None)
    set_device(vision_model, device)

    for k, v in tqdm({**modules}.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    vision_model.eval()

    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    tt_out = vision_model(ref_input_patches, ref_input_features)
    DispatchManager.save_stats_to_file(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/deepseek_ocr_module_timing_stats.csv"
    )

    passed, message = check_with_pcc(ref_out.float(), tt_out.float(), pcc=0.99)
    logger.info(f"TT VIT PCC : {message}")
    assert passed, f"TT VIT PCC check failed: {message}"


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
def test_tt_vitmodel_pcc(device, ocr_model):
    """Run torch NoTPTransformer and TT NoTPTransformer with same input; assert PCC >= PCC_THRESHOLD."""

    vision_model = ocr_model.model.vision_model

    torch.manual_seed(42)
    ref_input_patches = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_patches.pt"
    )
    ref_input_features = torch.load(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/vision_model_input_local_features.pt",
        weights_only=False,
    )

    with torch.no_grad():
        ref_out = vision_model(ref_input_patches, ref_input_features)

    ref_input_features = ref_input_features.flatten(2).transpose(1, 2)

    vision_model = TTNNVitModel.from_torch(vision_model)
    set_device(vision_model, device)

    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    tt_out = vision_model(ref_input_patches, ref_input_features)
    DispatchManager.save_stats_to_file(
        "models/experimental/tt_symbiote/tests/deepseek_ocr_vision_model/extras/deepseek_ocr_module_timing_stats.csv"
    )

    passed, message = check_with_pcc(ref_out.float(), tt_out.float(), pcc=0.99)
    logger.info(f"TT VIT PCC : {message}")
    assert passed, f"TT VIT PCC check failed: {message}"

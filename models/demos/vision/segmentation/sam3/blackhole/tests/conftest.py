# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys
import os
import unittest.mock as mock

import pytest
import torch

# Add sam3 package path (from tenstorrent-venv or SAM3_VENV_PATH env var)
_venv_path = os.environ.get(
    "SAM3_VENV_PATH",
    os.path.join(os.path.expanduser("~"), ".tenstorrent-venv/lib/python3.12/site-packages"),
)
if os.path.isdir(_venv_path) and _venv_path not in sys.path:
    sys.path.insert(0, _venv_path)

# BPE vocab file path
BPE_PATH = os.environ.get(
    "SAM3_BPE_PATH",
    os.path.join(
        os.environ.get("TT_METAL_HOME", os.path.expanduser("~/tt-metal")),
        "python_env/lib/python3.12/site-packages/open_clip/bpe_simple_vocab_16e6.txt.gz",
    ),
)


def _redirect_cuda_to_cpu(fn):
    """Wrap a torch function to redirect device='cuda' to device='cpu'."""
    def wrapper(*args, **kwargs):
        if "device" in kwargs:
            dev = kwargs["device"]
            if dev is not None and "cuda" in str(dev):
                kwargs["device"] = "cpu"
        return fn(*args, **kwargs)
    return wrapper


def _build_sam3_model_cpu():
    """Build SAM3 model on CPU, patching all CUDA tensor allocations."""
    _originals = {
        n: getattr(torch, n)
        for n in [
            "zeros", "ones", "arange", "empty", "full",
            "randn", "rand", "tensor", "linspace", "logspace", "eye",
        ]
    }

    patches = [mock.patch("torch.cuda.is_available", return_value=False)]
    for name, fn in _originals.items():
        patches.append(mock.patch(f"torch.{name}", _redirect_cuda_to_cpu(fn)))

    for p in patches:
        p.start()

    try:
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            bpe_path=BPE_PATH,
            device="cpu",
            eval_mode=True,
            load_from_HF=False,
            checkpoint_path=None,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
    finally:
        for p in patches:
            p.stop()

    return model


@pytest.fixture(scope="session")
def sam3_reference_model():
    """Load the SAM3 PyTorch reference model (CPU, eval mode, random weights)."""
    return _build_sam3_model_cpu()


@pytest.fixture(scope="session")
def sam3_vit_backbone(sam3_reference_model):
    """Extract just the ViT backbone from the SAM3 model."""
    from sam3.model.vitdet import ViT
    for name, module in sam3_reference_model.named_modules():
        if isinstance(module, ViT):
            return module
    raise RuntimeError("Could not find ViT backbone in SAM3 model")


@pytest.fixture(scope="session")
def sam3_neck(sam3_reference_model):
    """Extract the FPN neck from the SAM3 model."""
    from sam3.model.necks import Sam3DualViTDetNeck
    for name, module in sam3_reference_model.named_modules():
        if isinstance(module, Sam3DualViTDetNeck):
            return module
    raise RuntimeError("Could not find neck in SAM3 model")


@pytest.fixture(scope="session")
def sam3_text_encoder(sam3_reference_model):
    """Extract the text encoder from the SAM3 model."""
    from sam3.model.text_encoder_ve import VETextEncoder
    for name, module in sam3_reference_model.named_modules():
        if isinstance(module, VETextEncoder):
            return module
    raise RuntimeError("Could not find text encoder in SAM3 model")


@pytest.fixture(scope="session")
def sam3_transformer(sam3_reference_model):
    """Extract the transformer encoder+decoder from the SAM3 model."""
    return sam3_reference_model.transformer

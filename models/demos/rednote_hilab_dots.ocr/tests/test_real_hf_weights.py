# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Real-HF-weights PCC tests for rednote-hilab/dots.ocr TTNN blocks.

Each test loads the ACTUAL HuggingFace checkpoint weight(s) for a block via
``tt/weight_loader.py``, runs the TTNN block on the open p150 (blackhole)
device, and compares against the HF PyTorch reference computed with the SAME
real weight. Gate is PCC > 0.99 (real weights have wider dynamic range than
the seed-0 synthetic goldens used during bring-up, so some drift is expected).

The model dir name contains a dot, so blocks/loader are imported by file path
via importlib.
"""
import importlib.util
import os

import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

_TT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tt"))
_REF_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "reference"))

CHECKPOINT_PATH = os.environ.get(
    "DOTS_OCR_CHECKPOINT",
    "/local/ttuser/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots/"
    "c0111ce6bc07803dbc267932ffef0ae3a51dc951",
)

EMBED_DIM = 1536
VISION_RMS_NORM_EPS = 1e-5
# A real RMSNorm gamma from the vision tower (every vision RMSNorm shares
# shape/eps; block-0 norm1 is representative).
VISION_RMSNORM_HF_KEY = "vision_tower.blocks.0.norm1.weight"


def _load_by_path(name, filename, directory):
    spec = importlib.util.spec_from_file_location(name, os.path.join(directory, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tt_rmsnorm = _load_by_path("dots_tt_vision_rmsnorm_rw", "vision_rmsnorm.py", _TT_DIR)
_loader = _load_by_path("dots_weight_loader", "weight_loader.py", _TT_DIR)
_functional = _load_by_path("dots_reference_functional_rw", "functional.py", _REF_DIR)

TtVisionRMSNorm = _tt_rmsnorm.TtVisionRMSNorm
load_vision_rmsnorm_weight = _loader.load_vision_rmsnorm_weight
vision_rmsnorm_forward = _functional.vision_rmsnorm_forward


def _run_vision_rmsnorm_pcc(device):
    """Load the real vision RMSNorm gamma, run TTNN, compare to HF reference.

    Returns (pcc, params_loaded).
    """
    weight = load_vision_rmsnorm_weight(CHECKPOINT_PATH, VISION_RMSNORM_HF_KEY).to(torch.float32)
    params_loaded = int(weight.numel())
    assert weight.shape == (EMBED_DIM,), f"unexpected real weight shape {tuple(weight.shape)}"

    torch.manual_seed(0)
    n_tokens = 256
    torch_input = torch.randn(n_tokens, EMBED_DIM, dtype=torch.float32)

    # HF reference (mirrors modeling_dots_vision.RMSNorm exactly) with the REAL gamma.
    ref_output = vision_rmsnorm_forward(torch_input, weight, eps=VISION_RMS_NORM_EPS)

    tt_norm = TtVisionRMSNorm(device=device, dim=EMBED_DIM, weight=weight, eps=VISION_RMS_NORM_EPS)
    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tt_norm(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(vision_rmsnorm, real weights): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    print(f"real-weights vision_rmsnorm PCC = {pcc} | params_loaded = {params_loaded}")
    assert passing, f"vision_rmsnorm real-weights PCC below 0.99: {pcc_message}"
    return pcc, params_loaded


def test_real_hf_weights_vision_rmsnorm(device):
    pcc, params_loaded = _run_vision_rmsnorm_pcc(device)
    assert params_loaded > 0


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_vision_rmsnorm_pcc(dev)
    finally:
        ttnn.close_device(dev)

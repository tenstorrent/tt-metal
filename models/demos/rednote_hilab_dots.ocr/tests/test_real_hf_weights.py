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
_tt_vision_attention = _load_by_path("dots_tt_vision_attention_rw", "vision_attention.py", _TT_DIR)
_tt_vision_mlp = _load_by_path("dots_tt_vision_mlp_rw", "vision_mlp.py", _TT_DIR)
_loader = _load_by_path("dots_weight_loader", "weight_loader.py", _TT_DIR)
_functional = _load_by_path("dots_reference_functional_rw", "functional.py", _REF_DIR)

TtVisionRMSNorm = _tt_rmsnorm.TtVisionRMSNorm
TtVisionAttention = _tt_vision_attention.TtVisionAttention
TtVisionMLP = _tt_vision_mlp.TtVisionMLP
load_vision_rmsnorm_weight = _loader.load_vision_rmsnorm_weight
load_vision_attention_weights = _loader.load_vision_attention_weights
load_vision_mlp_weights = _loader.load_vision_mlp_weights
vision_rmsnorm_forward = _functional.vision_rmsnorm_forward
vision_attention_forward = _functional.vision_attention_forward
vision_mlp_forward = _functional.vision_mlp_forward

# Vision attention config (modeling_dots_vision): 12 heads, head_dim 128,
# fused QKV no-bias, 2D RoPE theta 1e4, block-diagonal bidirectional attention.
VISION_NUM_HEADS = 12
VISION_HEAD_DIM = 128


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


# Reuse the seed-0 golden's static rope freqs + cu_seqlens so the real-weights
# run uses the exact same precomputed RoPE/block-diagonal tables the synthetic
# bring-up validated; only the QKV/proj weights swap to the real checkpoint.
VISION_ATTENTION_GOLDEN_PATH = os.path.join(_REF_DIR, "golden", "vision_attention.pt")


def _run_vision_attention_pcc(device):
    """Load the real vision attention QKV+proj, run TTNN, compare to HF reference.

    Returns (pcc, params_loaded).
    """
    state_dict = load_vision_attention_weights(CHECKPOINT_PATH, block_idx=0)
    qkv_weight = state_dict["qkv.weight"].to(torch.float32)
    proj_weight = state_dict["proj.weight"].to(torch.float32)
    params_loaded = int(qkv_weight.numel() + proj_weight.numel())
    assert qkv_weight.shape == (3 * EMBED_DIM, EMBED_DIM), tuple(qkv_weight.shape)
    assert proj_weight.shape == (EMBED_DIM, EMBED_DIM), tuple(proj_weight.shape)

    # Static rope freqs + cu_seqlens come from the bring-up golden (same image
    # grid). Only the weights are swapped to the real checkpoint.
    golden = torch.load(VISION_ATTENTION_GOLDEN_PATH, map_location="cpu")
    cu_seqlens = golden["cu_seqlens"]
    rotary_pos_emb = golden["rotary_pos_emb"].to(torch.float32)  # [seq, head_dim//2]
    seq_length = int(rotary_pos_emb.shape[0])

    torch.manual_seed(0)
    torch_input = torch.randn(seq_length, EMBED_DIM, dtype=torch.float32)

    # HF reference (eager) with the REAL weights.
    ref_output = vision_attention_forward(
        torch_input,
        {"qkv.weight": qkv_weight, "proj.weight": proj_weight},
        cu_seqlens,
        rotary_pos_emb,
        num_heads=VISION_NUM_HEADS,
        bias=False,
    )

    tt_attn = TtVisionAttention(
        device=device,
        qkv_weight=qkv_weight,
        proj_weight=proj_weight,
        rotary_pos_emb=rotary_pos_emb,
        cu_seqlens=cu_seqlens,
        seq_length=seq_length,
        num_heads=VISION_NUM_HEADS,
        head_dim=VISION_HEAD_DIM,
    )
    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tt_attn(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(vision_attention, real weights): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    print(f"real-weights vision_attention PCC = {pcc} | params_loaded = {params_loaded}")
    assert passing, f"vision_attention real-weights PCC below 0.99: {pcc_message}"
    return pcc, params_loaded


def test_real_hf_weights_vision_attention(device):
    pcc, params_loaded = _run_vision_attention_pcc(device)
    assert params_loaded > 0


# Vision MLP (DotsSwiGLUFFN): embed_dim 1536, intermediate_size 4224, no bias.
VISION_INTERMEDIATE = 4224


def _run_vision_mlp_pcc(device):
    """Load the real vision MLP fc1/fc2/fc3, run TTNN, compare to HF reference.

    Returns (pcc, params_loaded).
    """
    state_dict = load_vision_mlp_weights(CHECKPOINT_PATH, block_idx=0)
    fc1_weight = state_dict["fc1.weight"].to(torch.float32)  # gate [inter, dim]
    fc2_weight = state_dict["fc2.weight"].to(torch.float32)  # down [dim, inter]
    fc3_weight = state_dict["fc3.weight"].to(torch.float32)  # up   [inter, dim]
    params_loaded = int(fc1_weight.numel() + fc2_weight.numel() + fc3_weight.numel())
    assert fc1_weight.shape == (VISION_INTERMEDIATE, EMBED_DIM), tuple(fc1_weight.shape)
    assert fc3_weight.shape == (VISION_INTERMEDIATE, EMBED_DIM), tuple(fc3_weight.shape)
    assert fc2_weight.shape == (EMBED_DIM, VISION_INTERMEDIATE), tuple(fc2_weight.shape)

    torch.manual_seed(0)
    n_tokens = 256
    torch_input = torch.randn(n_tokens, EMBED_DIM, dtype=torch.float32)

    # HF reference (mirrors DotsSwiGLUFFN exactly) with the REAL weights.
    ref_output = vision_mlp_forward(
        torch_input,
        {"fc1.weight": fc1_weight, "fc2.weight": fc2_weight, "fc3.weight": fc3_weight},
        bias=False,
    )

    tt_mlp = TtVisionMLP(
        device=device,
        fc1_weight=fc1_weight,
        fc3_weight=fc3_weight,
        fc2_weight=fc2_weight,
    )
    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tt_mlp(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(vision_mlp, real weights): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    print(f"real-weights vision_mlp PCC = {pcc} | params_loaded = {params_loaded}")
    assert passing, f"vision_mlp real-weights PCC below 0.99: {pcc_message}"
    return pcc, params_loaded


def test_real_hf_weights_vision_mlp(device):
    pcc, params_loaded = _run_vision_mlp_pcc(device)
    assert params_loaded > 0


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_vision_rmsnorm_pcc(dev)
        _run_vision_attention_pcc(dev)
        _run_vision_mlp_pcc(dev)
    finally:
        ttnn.close_device(dev)

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
_tt_vision_block = _load_by_path("dots_tt_vision_block_rw", "vision_block.py", _TT_DIR)
_tt_vision_patch_merger = _load_by_path("dots_tt_vision_patch_merger_rw", "vision_patch_merger.py", _TT_DIR)
_tt_vision_tower = _load_by_path("dots_tt_vision_tower_rw", "vision_tower.py", _TT_DIR)
_tt_embedding = _load_by_path("dots_tt_embedding_rw", "embedding.py", _TT_DIR)
_tt_rmsnorm_lm = _load_by_path("dots_tt_rmsnorm_rw", "rmsnorm.py", _TT_DIR)
_tt_rope = _load_by_path("dots_tt_rope_rw", "rope.py", _TT_DIR)
_tt_attention = _load_by_path("dots_tt_attention_rw", "attention.py", _TT_DIR)
_tt_mlp = _load_by_path("dots_tt_mlp_rw", "mlp.py", _TT_DIR)
_loader = _load_by_path("dots_weight_loader", "weight_loader.py", _TT_DIR)
_functional = _load_by_path("dots_reference_functional_rw", "functional.py", _REF_DIR)

TtVisionRMSNorm = _tt_rmsnorm.TtVisionRMSNorm
TtRMSNorm = _tt_rmsnorm_lm.TtRMSNorm
TtVisionAttention = _tt_vision_attention.TtVisionAttention
TtVisionMLP = _tt_vision_mlp.TtVisionMLP
TtVisionBlock = _tt_vision_block.TtVisionBlock
TtVisionPatchMerger = _tt_vision_patch_merger.TtVisionPatchMerger
TtVisionTower = _tt_vision_tower.TtVisionTower
TtEmbedding = _tt_embedding.TtEmbedding
TtRoPE = _tt_rope.TtRoPE
TtAttention = _tt_attention.TtAttention
TtMLP = _tt_mlp.TtMLP
load_lm_rope_config = _loader.load_lm_rope_config
load_lm_attention_weights = _loader.load_lm_attention_weights
load_lm_mlp_weights = _loader.load_lm_mlp_weights
load_vision_rmsnorm_weight = _loader.load_vision_rmsnorm_weight
load_embedding_weight = _loader.load_embedding_weight
load_vision_attention_weights = _loader.load_vision_attention_weights
load_vision_mlp_weights = _loader.load_vision_mlp_weights
load_vision_block_weights = _loader.load_vision_block_weights
load_vision_patch_merger_weights = _loader.load_vision_patch_merger_weights
load_vision_tower_weights = _loader.load_vision_tower_weights
load_lm_rmsnorm_weight = _loader.load_lm_rmsnorm_weight
vision_rmsnorm_forward = _functional.vision_rmsnorm_forward
rmsnorm_forward = _functional.rmsnorm_forward
vision_attention_forward = _functional.vision_attention_forward
vision_mlp_forward = _functional.vision_mlp_forward
vision_block_forward = _functional.vision_block_forward
vision_patch_merger_forward = _functional.vision_patch_merger_forward
vision_tower_forward = _functional.vision_tower_forward
embedding_forward = _functional.embedding_forward
attention_forward = _functional.attention_forward
mlp_forward = _functional.mlp_forward
rope_forward = _functional.rope_forward

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


# Vision block (DotsVisionBlock): pre-norm residual composite of the three
# leaves. Reuse the bring-up golden's static rope freqs + cu_seqlens (same image
# grid); only the per-block weights swap to the real block-0 checkpoint.
VISION_BLOCK_GOLDEN_PATH = os.path.join(_REF_DIR, "golden", "vision_block.pt")


def _run_vision_block_pcc(device):
    """Load real block-0 weights into TtVisionBlock, run TTNN, compare to HF ref.

    Composes the per-leaf real-weight loaders via load_vision_block_weights and
    validates the full pre-norm residual layer end to end.

    Returns (pcc, params_loaded).
    """
    state_dict = load_vision_block_weights(CHECKPOINT_PATH, block_idx=0)
    state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}
    params_loaded = int(sum(v.numel() for v in state_dict.values()))

    assert state_dict["norm1.weight"].shape == (EMBED_DIM,), tuple(state_dict["norm1.weight"].shape)
    assert state_dict["norm2.weight"].shape == (EMBED_DIM,), tuple(state_dict["norm2.weight"].shape)
    assert state_dict["attn.qkv.weight"].shape == (3 * EMBED_DIM, EMBED_DIM), tuple(state_dict["attn.qkv.weight"].shape)
    assert state_dict["attn.proj.weight"].shape == (EMBED_DIM, EMBED_DIM), tuple(state_dict["attn.proj.weight"].shape)
    assert state_dict["mlp.fc1.weight"].shape == (VISION_INTERMEDIATE, EMBED_DIM), tuple(
        state_dict["mlp.fc1.weight"].shape
    )
    assert state_dict["mlp.fc3.weight"].shape == (VISION_INTERMEDIATE, EMBED_DIM), tuple(
        state_dict["mlp.fc3.weight"].shape
    )
    assert state_dict["mlp.fc2.weight"].shape == (EMBED_DIM, VISION_INTERMEDIATE), tuple(
        state_dict["mlp.fc2.weight"].shape
    )

    # Static rope freqs + cu_seqlens from the bring-up golden (same image grid).
    golden = torch.load(VISION_BLOCK_GOLDEN_PATH, map_location="cpu")
    cu_seqlens = golden["cu_seqlens"]
    rotary_pos_emb = golden["rotary_pos_emb"].to(torch.float32)  # [seq, head_dim//2]
    seq_length = int(rotary_pos_emb.shape[0])

    torch.manual_seed(0)
    torch_input = torch.randn(seq_length, EMBED_DIM, dtype=torch.float32)

    # HF reference (eager pre-norm residual block) with the REAL block-0 weights.
    ref_output = vision_block_forward(
        torch_input,
        state_dict,
        cu_seqlens,
        rotary_pos_emb,
        num_heads=VISION_NUM_HEADS,
        eps=VISION_RMS_NORM_EPS,
        bias=False,
    )

    tt_block = TtVisionBlock(
        device=device,
        norm1_weight=state_dict["norm1.weight"],
        qkv_weight=state_dict["attn.qkv.weight"],
        proj_weight=state_dict["attn.proj.weight"],
        norm2_weight=state_dict["norm2.weight"],
        fc1_weight=state_dict["mlp.fc1.weight"],
        fc3_weight=state_dict["mlp.fc3.weight"],
        fc2_weight=state_dict["mlp.fc2.weight"],
        rotary_pos_emb=rotary_pos_emb,
        cu_seqlens=cu_seqlens,
        seq_length=seq_length,
        num_heads=VISION_NUM_HEADS,
        head_dim=VISION_HEAD_DIM,
        eps=VISION_RMS_NORM_EPS,
    )
    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tt_block(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(vision_block, real weights): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    print(f"real-weights vision_block PCC = {pcc} | params_loaded = {params_loaded}")
    assert passing, f"vision_block real-weights PCC below 0.99: {pcc_message}"
    return pcc, params_loaded


def test_real_hf_weights_vision_block(device):
    pcc, params_loaded = _run_vision_block_pcc(device)
    assert params_loaded > 0


# Vision PatchMerger (DotsPatchMerger, pre_norm='layernorm'): LayerNorm(eps 1e-6,
# with bias) -> view(-1, context_dim*merge**2) -> Linear -> GELU -> Linear (both
# biased). context_dim 1536, merge 2 -> hidden 6144, out_dim 1536.
PATCH_MERGER_CONTEXT_DIM = 1536
PATCH_MERGER_SPATIAL_MERGE = 2
PATCH_MERGER_HIDDEN = PATCH_MERGER_CONTEXT_DIM * (PATCH_MERGER_SPATIAL_MERGE**2)  # 6144
PATCH_MERGER_LN_EPS = 1e-6


def _run_vision_patch_merger_pcc(device):
    """Load the real PatchMerger weights, run TTNN, compare to HF reference.

    The merger has LayerNorm gamma+beta and biased MLP Linears (unlike the
    rest of the RMSNorm/unbiased vision tower). Returns (pcc, params_loaded).
    """
    state_dict = load_vision_patch_merger_weights(CHECKPOINT_PATH)
    state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}
    params_loaded = int(sum(v.numel() for v in state_dict.values()))

    assert state_dict["ln_q.weight"].shape == (PATCH_MERGER_CONTEXT_DIM,), tuple(state_dict["ln_q.weight"].shape)
    assert state_dict["ln_q.bias"].shape == (PATCH_MERGER_CONTEXT_DIM,), tuple(state_dict["ln_q.bias"].shape)
    assert state_dict["mlp.0.weight"].shape == (PATCH_MERGER_HIDDEN, PATCH_MERGER_HIDDEN), tuple(
        state_dict["mlp.0.weight"].shape
    )
    assert state_dict["mlp.0.bias"].shape == (PATCH_MERGER_HIDDEN,), tuple(state_dict["mlp.0.bias"].shape)
    assert state_dict["mlp.2.weight"].shape == (PATCH_MERGER_CONTEXT_DIM, PATCH_MERGER_HIDDEN), tuple(
        state_dict["mlp.2.weight"].shape
    )
    assert state_dict["mlp.2.bias"].shape == (PATCH_MERGER_CONTEXT_DIM,), tuple(state_dict["mlp.2.bias"].shape)

    torch.manual_seed(0)
    # num_patches must be a multiple of merge**2 (=4); 256 -> 64 merged tokens.
    n_patches = 256
    torch_input = torch.randn(n_patches, PATCH_MERGER_CONTEXT_DIM, dtype=torch.float32)

    # HF reference (mirrors DotsPatchMerger exactly) with the REAL weights.
    ref_output = vision_patch_merger_forward(
        torch_input,
        state_dict,
        context_dim=PATCH_MERGER_CONTEXT_DIM,
        spatial_merge_size=PATCH_MERGER_SPATIAL_MERGE,
        ln_eps=PATCH_MERGER_LN_EPS,
    )

    tt_merger = TtVisionPatchMerger(
        device=device,
        ln_weight=state_dict["ln_q.weight"],
        ln_bias=state_dict["ln_q.bias"],
        fc1_weight=state_dict["mlp.0.weight"],
        fc1_bias=state_dict["mlp.0.bias"],
        fc2_weight=state_dict["mlp.2.weight"],
        fc2_bias=state_dict["mlp.2.bias"],
        context_dim=PATCH_MERGER_CONTEXT_DIM,
        spatial_merge_size=PATCH_MERGER_SPATIAL_MERGE,
        ln_eps=PATCH_MERGER_LN_EPS,
    )
    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tt_merger(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(vision_patch_merger, real weights): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    print(f"real-weights vision_patch_merger PCC = {pcc} | params_loaded = {params_loaded}")
    assert passing, f"vision_patch_merger real-weights PCC below 0.99: {pcc_message}"
    return pcc, params_loaded


def test_real_hf_weights_vision_patch_merger(device):
    pcc, params_loaded = _run_vision_patch_merger_pcc(device)
    assert params_loaded > 0


# Vision tower (DotsVisionTransformer): patch_embed -> N x vision_block ->
# post_trunk RMSNorm -> patch_merger. Validated at the SAME reduced layer count
# the seed-0 bring-up golden used (num_layers=2 vs production 42) so the composed
# real-weight state_dict and the eager reference agree. The seed-0 golden supplies
# the same input pixel_values + grid_thw the synthetic run used; only the weights
# swap to the real checkpoint (composed via load_vision_tower_weights).
VISION_TOWER_GOLDEN_PATH = os.path.join(_REF_DIR, "golden", "vision_tower.pt")


def _run_vision_tower_pcc(device):
    """Load the full real vision-tower weights, run TtVisionTower on device, and
    compare against the HF PyTorch reference computed with the SAME real weights
    at the reduced golden layer count.

    Returns (pcc, params_loaded).
    """
    golden = torch.load(VISION_TOWER_GOLDEN_PATH, map_location="cpu", weights_only=False)
    pixel_values = golden["input"].to(torch.float32)  # [num_patches, ch*tp*ps*ps]
    grid_thw = torch.as_tensor(golden["grid_thw"])  # [num_images, 3]
    cfg = golden["config"]

    num_layers = int(cfg["num_layers"])
    embed_dim = int(cfg["embed_dim"])
    num_heads = int(cfg["num_heads"])
    num_channels = int(cfg["num_channels"])
    temporal_patch_size = int(cfg["temporal_patch_size"])
    patch_size = int(cfg["patch_size"])
    spatial_merge_size = int(cfg["spatial_merge_size"])
    rms_norm_eps = float(cfg["rms_norm_eps"])
    ln_eps = float(cfg["ln_eps"])
    post_norm = bool(cfg["post_norm"])
    hidden_size = int(cfg["hidden_size"])

    # Compose the REAL checkpoint weights for the full tower at the reduced depth.
    state_dict = load_vision_tower_weights(CHECKPOINT_PATH, num_layers=num_layers)
    state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}
    params_loaded = int(sum(v.numel() for v in state_dict.values()))

    # Shape sanity on the composed real state_dict.
    assert state_dict["patch_embed.proj.weight"].shape == (embed_dim, num_channels, patch_size, patch_size)
    assert state_dict["patch_embed.norm.weight"].shape == (embed_dim,)
    assert state_dict["post_trunk_norm.weight"].shape == (embed_dim,)
    for i in range(num_layers):
        assert state_dict[f"blocks.{i}.attn.qkv.weight"].shape == (3 * embed_dim, embed_dim)
    assert state_dict["merger.mlp.0.weight"].shape == (
        embed_dim * spatial_merge_size**2,
        embed_dim * spatial_merge_size**2,
    )

    # HF PyTorch reference (fp32) with the REAL weights at the reduced depth.
    ref_output = vision_tower_forward(
        pixel_values,
        grid_thw,
        state_dict,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_channels=num_channels,
        temporal_patch_size=temporal_patch_size,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        rms_norm_eps=rms_norm_eps,
        ln_eps=ln_eps,
        post_norm=post_norm,
        bias=False,
        hidden_size=hidden_size,
    )

    tt_tower = TtVisionTower(
        device=device,
        state_dict=state_dict,
        grid_thw=grid_thw,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_channels=num_channels,
        temporal_patch_size=temporal_patch_size,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        rms_norm_eps=rms_norm_eps,
        ln_eps=ln_eps,
        post_norm=post_norm,
    )

    # Host-resident patch_embed (documented boundary) -> device input tokens.
    hidden_states = tt_tower.patch_embed(pixel_values)  # [num_patches, embed_dim]
    tt_input = ttnn.from_torch(
        hidden_states,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tt_tower(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(vision_tower, real weights): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    print(f"real-weights vision_tower PCC = {pcc} | params_loaded = {params_loaded} | num_layers = {num_layers}")
    assert passing, f"vision_tower real-weights PCC below 0.99: {pcc_message}"
    return pcc, params_loaded


def test_real_hf_weights_vision_tower(device):
    pcc, params_loaded = _run_vision_tower_pcc(device)
    assert params_loaded > 0


# LM token embedding (Qwen2 nn.Embedding): the real input lookup table
# model.embed_tokens.weight [vocab_size 151936, hidden_size 1536]. Untied from
# lm_head (config.tie_word_embeddings = false). ttnn.embedding is an exact row
# gather, so real weights should still give PCC ~= 1.0 (bf16 row cast is the
# only error source).
VOCAB_SIZE = 151936
EMBEDDING_HF_KEY = "model.embed_tokens.weight"


def _run_embedding_pcc(device):
    """Load the real LM token-embedding table, run TtEmbedding on device, and
    compare against the HF PyTorch reference (F.embedding) with the SAME real
    weight.

    Returns (pcc, params_loaded).
    """
    weight = load_embedding_weight(CHECKPOINT_PATH, EMBEDDING_HF_KEY).to(torch.float32)
    params_loaded = int(weight.numel())
    assert weight.shape == (VOCAB_SIZE, EMBED_DIM), tuple(weight.shape)

    torch.manual_seed(0)
    batch, seq_len = 1, 128
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len), dtype=torch.int64)

    # HF reference (F.embedding gather) with the REAL embedding table.
    ref_output = embedding_forward(input_ids, weight)  # [1, 128, 1536]

    tt_emb = TtEmbedding(device=device, weight=weight)
    # ttnn.embedding consumes row-major uint32 indices.
    tt_input = ttnn.from_torch(
        input_ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tt_emb(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(embedding, real weights): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    print(f"real-weights embedding PCC = {pcc} | params_loaded = {params_loaded}")
    assert passing, f"embedding real-weights PCC below 0.99: {pcc_message}"
    return pcc, params_loaded


def test_real_hf_weights_embedding(device):
    pcc, params_loaded = _run_embedding_pcc(device)
    assert params_loaded > 0


# LM RMSNorm (Qwen2RMSNorm): eps 1e-6 (distinct from vision 1e-5), hidden 1536.
# A real LM RMSNorm gamma -- every decoder layer's input_layernorm /
# post_attention_layernorm and the final model.norm share shape/eps; layer-0
# input_layernorm is representative.
LM_RMS_NORM_EPS = 1e-6
LM_RMSNORM_HF_KEY = "model.layers.0.input_layernorm.weight"


def _run_lm_rmsnorm_pcc(device):
    """Load the real LM RMSNorm gamma, run TtRMSNorm on device, compare to the
    HF Qwen2RMSNorm reference (eps 1e-6) computed with the SAME real weight.

    Returns (pcc, params_loaded).
    """
    weight = load_lm_rmsnorm_weight(CHECKPOINT_PATH, LM_RMSNORM_HF_KEY).to(torch.float32)
    params_loaded = int(weight.numel())
    assert weight.shape == (EMBED_DIM,), f"unexpected real LM rmsnorm weight shape {tuple(weight.shape)}"

    torch.manual_seed(0)
    n_tokens = 256
    torch_input = torch.randn(n_tokens, EMBED_DIM, dtype=torch.float32)

    # HF reference (mirrors Qwen2RMSNorm exactly, eps 1e-6) with the REAL gamma.
    ref_output = rmsnorm_forward(torch_input, weight, eps=LM_RMS_NORM_EPS)

    tt_norm = TtRMSNorm(device=device, dim=EMBED_DIM, weight=weight, eps=LM_RMS_NORM_EPS)
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
    print(f"comp_pcc(rmsnorm, real weights): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    print(f"real-weights rmsnorm PCC = {pcc} | params_loaded = {params_loaded}")
    assert passing, f"rmsnorm real-weights PCC below 0.99: {pcc_message}"
    return pcc, params_loaded


def test_real_hf_weights_rmsnorm(device):
    pcc, params_loaded = _run_lm_rmsnorm_pcc(device)
    assert params_loaded > 0


# LM RoPE (Qwen2 rotary embedding). PARAMETER-FREE: rope has NO learnable
# checkpoint weights -- its inv_freq is a config-derived buffer (computed from
# rope_theta and the per-head dim), not stored in the safetensors. So there are
# zero checkpoint params to load. The "real-weights validation" for rope is
# therefore: re-confirm the cos/sin tables generated on-device match the real HF
# Qwen2RotaryEmbedding instantiated at the PRODUCTION config read from the real
# checkpoint's config.json (rope_theta 1e6, head_dim 128, default rope ->
# attention_scaling 1.0). params_loaded counts the config-derived inv_freq table
# (head_dim/2 = 64) that the module materializes and validates -- a real
# on-device tensor, even though it is config-derived not checkpoint-loaded.
def _run_rope_pcc(device):
    """Validate TtRoPE cos/sin against the real HF Qwen2RotaryEmbedding built
    from the production config read out of the real checkpoint's config.json.

    rope is parameter-free, so this loads the rope CONFIG (head_dim, rope_theta,
    attention_scaling) from the real HF checkpoint rather than a weight tensor,
    builds the HF Qwen2RotaryEmbedding at that config, and compares its cos/sin
    to the on-device TtRoPE tables. Returns (pcc, params_loaded) where
    params_loaded is the materialized inv_freq table count (head_dim/2).
    """
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    rope_cfg = load_lm_rope_config(CHECKPOINT_PATH)
    head_dim = rope_cfg["head_dim"]
    rope_theta = rope_cfg["rope_theta"]
    attention_scaling = rope_cfg["attention_scaling"]
    assert head_dim == 128, f"unexpected real rope head_dim {head_dim}"
    assert rope_theta == 1000000.0, f"unexpected real rope_theta {rope_theta}"
    assert attention_scaling == 1.0, f"unexpected real attention_scaling {attention_scaling}"

    # HF reference: Qwen2RotaryEmbedding derives head_dim from
    # hidden_size // num_attention_heads (1536 // 12 = 128) and reads rope_theta
    # straight from the config, so a Qwen2Config carrying the real hyper-params
    # reproduces the production rotary module exactly.
    hf_cfg = Qwen2Config(
        hidden_size=1536,
        num_attention_heads=12,
        num_key_value_heads=2,
        rope_theta=rope_theta,
        max_position_embeddings=131072,
    )
    hf_rope = Qwen2RotaryEmbedding(config=hf_cfg)
    # inv_freq is the config-derived buffer the module materializes; head_dim/2.
    params_loaded = int(hf_rope.inv_freq.numel())
    assert params_loaded == head_dim // 2, (params_loaded, head_dim)

    seq_len = 128
    position_ids = torch.arange(seq_len, dtype=torch.int64).reshape(1, seq_len)
    # HF rope: forward(x, position_ids) -> (cos, sin), x only supplies dtype/device.
    dummy_x = torch.zeros(1, head_dim, dtype=torch.float32)
    ref_cos, ref_sin = hf_rope(dummy_x, position_ids)
    ref_cos = ref_cos.to(torch.float32)  # [1, seq, head_dim]
    ref_sin = ref_sin.to(torch.float32)

    tt_rope = TtRoPE(
        device=device,
        head_dim=head_dim,
        rope_theta=rope_theta,
        attention_scaling=attention_scaling,
    )
    pos = position_ids.to(torch.float32).reshape(1, 1, seq_len, 1)
    tt_pos = ttnn.from_torch(
        pos,
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_cos, tt_sin = tt_rope(tt_pos)
    got_cos = ttnn.to_torch(tt_cos).to(torch.float32).reshape(ref_cos.shape)
    got_sin = ttnn.to_torch(tt_sin).to(torch.float32).reshape(ref_sin.shape)

    passing_cos, msg_cos = comp_pcc(ref_cos, got_cos, 0.99)
    passing_sin, msg_sin = comp_pcc(ref_sin, got_sin, 0.99)
    print(comp_allclose(ref_cos, got_cos))
    print(comp_allclose(ref_sin, got_sin))
    print(f"comp_pcc(rope/cos, real config): passing={passing_cos}, message={msg_cos}")
    print(f"comp_pcc(rope/sin, real config): passing={passing_sin}, message={msg_sin}")
    pcc_cos = float(str(msg_cos).split("PCC:")[-1].strip()) if "PCC:" in str(msg_cos) else float(msg_cos)
    pcc_sin = float(str(msg_sin).split("PCC:")[-1].strip()) if "PCC:" in str(msg_sin) else float(msg_sin)
    pcc = min(pcc_cos, pcc_sin)
    print(f"real-config rope PCC (min cos/sin) = {pcc} | params_loaded (inv_freq) = {params_loaded}")
    assert passing_cos, f"rope cos real-config PCC below 0.99: {msg_cos}"
    assert passing_sin, f"rope sin real-config PCC below 0.99: {msg_sin}"
    return pcc, params_loaded


def test_real_hf_weights_rope(device):
    pcc, params_loaded = _run_rope_pcc(device)
    assert params_loaded > 0


# LM self-attention (Qwen2Attention): GQA 12 query / 2 KV heads, head_dim 128,
# hidden 1536. DISTINCT from the bias-free vision attention: q/k/v_proj carry
# BIAS (Qwen2 QKV bias), o_proj has no bias. 1D RoPE theta 1e6, causal. Real
# layer-0 weights swap in; cos/sin + causal mask are rebuilt at the production
# config exactly as the seed-0 golden / HF Qwen2RotaryEmbedding do.
LM_NUM_HEADS = 12
LM_NUM_KV_HEADS = 2
LM_HEAD_DIM = 128
LM_ROPE_THETA = 1000000.0


def _run_lm_attention_pcc(device):
    """Load real layer-0 Qwen2 self-attention weights+QKV biases into TtAttention,
    run on device, and compare against the HF eager reference (attention_forward)
    computed with the SAME real weights (causal, 1D RoPE theta 1e6, GQA 12/2).

    Returns (pcc, params_loaded).
    """
    state_dict = load_lm_attention_weights(CHECKPOINT_PATH, layer_idx=0)
    state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}
    params_loaded = int(sum(v.numel() for v in state_dict.values()))

    hidden = EMBED_DIM
    kv_dim = LM_NUM_KV_HEADS * LM_HEAD_DIM  # 256
    assert state_dict["q_proj.weight"].shape == (hidden, hidden), tuple(state_dict["q_proj.weight"].shape)
    assert state_dict["q_proj.bias"].shape == (hidden,), tuple(state_dict["q_proj.bias"].shape)
    assert state_dict["k_proj.weight"].shape == (kv_dim, hidden), tuple(state_dict["k_proj.weight"].shape)
    assert state_dict["k_proj.bias"].shape == (kv_dim,), tuple(state_dict["k_proj.bias"].shape)
    assert state_dict["v_proj.weight"].shape == (kv_dim, hidden), tuple(state_dict["v_proj.weight"].shape)
    assert state_dict["v_proj.bias"].shape == (kv_dim,), tuple(state_dict["v_proj.bias"].shape)
    assert state_dict["o_proj.weight"].shape == (hidden, hidden), tuple(state_dict["o_proj.weight"].shape)

    torch.manual_seed(0)
    batch, seq_len = 1, 128
    torch_input = torch.randn(batch, seq_len, hidden, dtype=torch.float32)

    # Production-config 1D RoPE cos/sin (theta 1e6, head_dim 128) + causal mask,
    # rebuilt exactly as HF Qwen2RotaryEmbedding / the eager reference do.
    position_ids = torch.arange(seq_len, dtype=torch.int64).reshape(batch, seq_len)
    cos, sin = rope_forward(position_ids, head_dim=LM_HEAD_DIM, rope_theta=LM_ROPE_THETA)  # [1, seq, hd]
    causal = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    causal = torch.triu(causal, diagonal=1)
    attention_mask = causal[None, None, :, :]  # [1, 1, seq, seq]

    # HF reference (eager Qwen2Attention) with the REAL layer-0 weights.
    ref_output = attention_forward(
        torch_input,
        state_dict,
        (cos, sin),
        attention_mask=attention_mask,
        num_heads=LM_NUM_HEADS,
        num_kv_heads=LM_NUM_KV_HEADS,
        head_dim=LM_HEAD_DIM,
        bias=True,
    )

    tt_attn = TtAttention(
        device=device,
        q_weight=state_dict["q_proj.weight"],
        k_weight=state_dict["k_proj.weight"],
        v_weight=state_dict["v_proj.weight"],
        q_bias=state_dict["q_proj.bias"],
        k_bias=state_dict["k_proj.bias"],
        v_bias=state_dict["v_proj.bias"],
        o_weight=state_dict["o_proj.weight"],
        cos=cos.reshape(seq_len, LM_HEAD_DIM),
        sin=sin.reshape(seq_len, LM_HEAD_DIM),
        attention_mask=attention_mask.reshape(seq_len, seq_len),
        seq_len=seq_len,
        num_heads=LM_NUM_HEADS,
        num_kv_heads=LM_NUM_KV_HEADS,
        head_dim=LM_HEAD_DIM,
    )
    tt_input = ttnn.from_torch(
        torch_input.reshape(seq_len, hidden),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tt_attn(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).reshape(ref_output.shape)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, 0.99)
    print(comp_allclose(ref_output, tt_output_torch))
    print(f"comp_pcc(attention, real weights): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    print(f"real-weights attention PCC = {pcc} | params_loaded = {params_loaded}")
    assert passing, f"attention real-weights PCC below 0.99: {pcc_message}"
    return pcc, params_loaded


def test_real_hf_weights_attention(device):
    pcc, params_loaded = _run_lm_attention_pcc(device)
    assert params_loaded > 0


# LM MLP (Qwen2MLP): SwiGLU down_proj(silu(gate_proj(x)) * up_proj(x)), no bias.
# hidden_size 1536, intermediate_size 8960.
LM_HIDDEN = 1536
LM_INTERMEDIATE = 8960


def _run_lm_mlp_pcc(device):
    """Load the real LM MLP gate/up/down, run TTNN TtMLP, compare to HF reference.

    Returns (pcc, params_loaded).
    """
    state_dict = load_lm_mlp_weights(CHECKPOINT_PATH, layer_idx=0)
    gate_weight = state_dict["gate_proj.weight"].to(torch.float32)  # [inter, dim]
    up_weight = state_dict["up_proj.weight"].to(torch.float32)  # [inter, dim]
    down_weight = state_dict["down_proj.weight"].to(torch.float32)  # [dim, inter]
    params_loaded = int(gate_weight.numel() + up_weight.numel() + down_weight.numel())
    assert gate_weight.shape == (LM_INTERMEDIATE, LM_HIDDEN), tuple(gate_weight.shape)
    assert up_weight.shape == (LM_INTERMEDIATE, LM_HIDDEN), tuple(up_weight.shape)
    assert down_weight.shape == (LM_HIDDEN, LM_INTERMEDIATE), tuple(down_weight.shape)

    torch.manual_seed(0)
    n_tokens = 128
    torch_input = torch.randn(n_tokens, LM_HIDDEN, dtype=torch.float32)

    # HF reference (mirrors Qwen2MLP exactly) with the REAL weights.
    ref_output = mlp_forward(
        torch_input,
        {"gate_proj.weight": gate_weight, "up_proj.weight": up_weight, "down_proj.weight": down_weight},
    )

    tt_mlp = TtMLP(
        device=device,
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
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
    print(f"comp_pcc(lm_mlp, real weights): passing={passing}, message={pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    print(f"real-weights lm_mlp PCC = {pcc} | params_loaded = {params_loaded}")
    assert passing, f"lm_mlp real-weights PCC below 0.99: {pcc_message}"
    return pcc, params_loaded


def test_real_hf_weights_mlp(device):
    pcc, params_loaded = _run_lm_mlp_pcc(device)
    assert params_loaded > 0


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_vision_rmsnorm_pcc(dev)
        _run_vision_attention_pcc(dev)
        _run_vision_mlp_pcc(dev)
        _run_vision_block_pcc(dev)
        _run_vision_patch_merger_pcc(dev)
        _run_vision_tower_pcc(dev)
        _run_embedding_pcc(dev)
        _run_lm_rmsnorm_pcc(dev)
        _run_rope_pcc(dev)
        _run_lm_attention_pcc(dev)
        _run_lm_mlp_pcc(dev)
    finally:
        ttnn.close_device(dev)

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.ltx.rope_ltx import LTXRopeType, apply_rotary_emb, precompute_freqs_cis
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard


def make_indices_grid(F: int, H: int, W: int) -> torch.Tensor:
    """
    Create a 3D position indices grid (seq_len, 3) for temporal + spatial dims.
    Each row is (t_idx, h_idx, w_idx) for one token in the flattened sequence.
    """
    t_ids = torch.arange(F)
    h_ids = torch.arange(H)
    w_ids = torch.arange(W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    indices_grid = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float()
    return indices_grid.unsqueeze(0)  # (1, seq_len, 3)


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(1, 1), 0, 1],
        [(1, 2), 0, 1],
        [(2, 1), 0, 1],
        [(2, 2), 0, 1],
        [(2, 4), 0, 1],
        [(2, 4), 1, 0],
    ],
    ids=[
        "1x1sp0tp1",
        "1x2sp0tp1",
        "2x1sp0tp1",
        "2x2sp0tp1",
        "2x4sp0tp1",
        "2x4sp1tp0",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "F, H, W",
    [
        (5, 16, 28),  # Small: 2240 tokens
        (5, 30, 52),  # 480p-ish: 7800 tokens
        (5, 34, 64),  # 2K spatial grid (1088x2048 latent H×W): 10880 tokens
    ],
    ids=["small", "480p", "2k"],
)
def test_ltx_rope_interleaved(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int, F: int, H: int, W: int):
    """
    Test LTX-2 interleaved RoPE: compute cos/sin on CPU, apply via ttnn.experimental.rotary_embedding_llama,
    compare against PyTorch reference.
    """
    dim = 4096
    num_heads = 32
    head_dim = dim // num_heads  # 128
    theta = 10000.0
    max_pos = [20, 2048, 2048]
    B = 1

    # Build position indices grid
    indices_grid = make_indices_grid(F, H, W)  # (1, seq_len, 3)
    seq_len = F * H * W
    logger.info(f"indices_grid shape: {indices_grid.shape}, seq_len: {seq_len}")

    # Precompute cos/sin on CPU
    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid,
        dim=dim,
        out_dtype=torch.float32,
        theta=theta,
        max_pos=max_pos,
        num_attention_heads=num_heads,
        rope_type=LTXRopeType.INTERLEAVED,
    )
    logger.info(f"cos_freq shape: {cos_freq.shape}, sin_freq shape: {sin_freq.shape}")

    # Create random input: (B, seq_len, num_heads, head_dim)
    torch.manual_seed(42)
    input_tensor = torch.randn(B, seq_len, num_heads, head_dim, dtype=torch.float32)

    # PyTorch reference: apply RoPE
    # cos/sin from interleaved mode have shape (B, seq_len, dim) — need to match input shape
    # Reshape to (B, seq_len, num_heads, head_dim) for broadcasting
    cos_for_apply = cos_freq.reshape(B, seq_len, num_heads, head_dim)
    sin_for_apply = sin_freq.reshape(B, seq_len, num_heads, head_dim)
    output_ref = apply_rotary_emb(input_tensor, (cos_for_apply, sin_for_apply), LTXRopeType.INTERLEAVED)
    logger.info(f"Reference output shape: {output_ref.shape}")

    # Prepare ttnn tensors
    # ttnn.experimental.rotary_embedding_llama expects (B, num_heads, seq_len, head_dim)
    input_bhnd = input_tensor.permute(0, 2, 1, 3)  # (B, H, N, D)
    cos_bhnd = cos_for_apply.permute(0, 2, 1, 3)  # (B, H, N, D)
    sin_bhnd = sin_for_apply.permute(0, 2, 1, 3)  # (B, H, N, D)

    tt_input = bf16_tensor_2dshard(input_bhnd, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    # cos/sin sharded same as input: SP axis on sequence, TP axis on heads
    tt_cos = bf16_tensor_2dshard(cos_bhnd, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_bhnd, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    # Apply RoPE on device
    tt_output = ttnn.experimental.rotary_embedding_llama(tt_input, tt_cos, tt_sin, tt_trans_mat)

    # Gather output back to host
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    tt_output_torch = tt_output_torch.permute(0, 2, 1, 3)  # Back to (B, N, H, D)

    # Compare
    assert_quality(output_ref, tt_output_torch, pcc=0.99)
    logger.info("PASSED: LTX interleaved RoPE matches PyTorch reference")


# ============================================================================================
# RoPE cos/sin format regression guards.
#
# Walks the LTX-2 cos/sin chain step by step and asserts each step has the format it's
# *supposed* to have, per the documented contract:
#
#     Step 1. position created with .float()             → fp32
#     Step 2. position enters precompute_freqs_cis       → fp32
#     Step 3. frequency table built                      → fp32
#     Step 4. angles = position × freq                   → fp32
#     Step 5. cos/sin of angles                          → fp32
#     Step 6. precompute_freqs_cis returns               → fp32
#
# Bug history: a `.bfloat16()` cast on positions in pipeline_ltx.py once produced grainy/
# robotic audio in LTX-2 generation. These tests guard against the same bug class returning.
# ============================================================================================


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True, ids=["2x4"])
def test_pipeline_rope_positions_stay_fp32(mesh_device: ttnn.MeshDevice):
    """Assert the LTX-2 RoPE cos/sin chain stays fp32 end-to-end in the pipeline.

    (See the "Step 1-6" contract documented in the banner comment above.)

    Stands up a minimal LTXPipeline (bypassing weight loading) and calls all three RoPE
    prep methods (_prepare_rope, _prepare_audio_rope, _prepare_av_cross_pe). The four
    cos/sin helpers (generate_freq_grid, generate_freqs, split_freqs_cis,
    precompute_freqs_cis) are patched to assert fp32 at every step the pipeline triggers.

    Catches the original bug class: a `.bfloat16()` cast in pipeline_ltx.py that silently
    destroys positions' precision in the RoPE chain. Also catches anyone refactoring the
    helpers to use lower precision internally.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from models.tt_dit.models.transformers.ltx import rope_ltx
    from models.tt_dit.pipelines.ltx import pipeline_ltx as pipeline_mod
    from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline

    # Bypass __init__ to avoid loading model weights — we only need the attributes the three
    # RoPE prep methods actually read.
    pipe = LTXPipeline.__new__(LTXPipeline)
    pipe.mesh_device = mesh_device
    pipe.inner_dim = 4096
    pipe.positional_embedding_theta = 10000.0
    pipe.positional_embedding_max_pos = [20, 2048, 2048]
    pipe.num_attention_heads = 32
    pipe.parallel_config = SimpleNamespace(
        sequence_parallel=SimpleNamespace(mesh_axis=0, factor=tuple(mesh_device.shape)[0]),
        tensor_parallel=SimpleNamespace(mesh_axis=1, factor=tuple(mesh_device.shape)[1]),
    )

    counts = {"freq_grid": 0, "freqs": 0, "split_cis": 0, "precompute": 0}

    orig_grid = rope_ltx.generate_freq_grid
    orig_freqs = rope_ltx.generate_freqs
    orig_split = rope_ltx.split_freqs_cis
    orig_precompute = rope_ltx.precompute_freqs_cis

    def checked_grid(*args, **kwargs):
        # Step 3: frequency table built → fp32
        result = orig_grid(*args, **kwargs)
        counts["freq_grid"] += 1
        assert (
            result.dtype == torch.float32
        ), f"[Step 3 FAIL] generate_freq_grid returned {result.dtype}, expected fp32."
        return result

    def checked_freqs(indices, indices_grid, *args, **kwargs):
        # Step 4: angles = position × freq → fp32
        assert (
            indices.dtype == torch.float32
        ), f"[Step 4 FAIL] generate_freqs got freq table as {indices.dtype}, expected fp32."
        assert (
            indices_grid.dtype == torch.float32
        ), f"[Step 4 FAIL] generate_freqs got positions as {indices_grid.dtype}, expected fp32."
        result = orig_freqs(indices, indices_grid, *args, **kwargs)
        counts["freqs"] += 1
        assert (
            result.dtype == torch.float32
        ), f"[Step 4 FAIL] generate_freqs returned angles as {result.dtype}, expected fp32."
        return result

    def checked_split(freqs, *args, **kwargs):
        # Step 5: cos/sin of angles → fp32
        assert (
            freqs.dtype == torch.float32
        ), f"[Step 5 FAIL] split_freqs_cis got angles as {freqs.dtype}, expected fp32."
        cos, sin = orig_split(freqs, *args, **kwargs)
        counts["split_cis"] += 1
        assert cos.dtype == torch.float32, f"[Step 5 FAIL] split_freqs_cis cos is {cos.dtype}, expected fp32."
        assert sin.dtype == torch.float32, f"[Step 5 FAIL] split_freqs_cis sin is {sin.dtype}, expected fp32."
        return cos, sin

    def checked_precompute(indices_grid, *args, **kwargs):
        # Steps 1+2: pipeline created positions with .float() and passed them in as fp32
        assert indices_grid.dtype == torch.float32, (
            f"[Step 1-2 FAIL] Pipeline passed {indices_grid.dtype} positions to "
            f"precompute_freqs_cis. Check for a .bfloat16() call on positions in pipeline_ltx.py."
        )
        cos, sin = orig_precompute(indices_grid, *args, **kwargs)
        counts["precompute"] += 1
        # Step 6: precompute_freqs_cis returns → fp32
        assert cos.dtype == torch.float32, (
            f"[Step 6 FAIL] precompute_freqs_cis returned cos as {cos.dtype}, expected fp32. "
            f"Check the out_dtype argument at the call site."
        )
        assert (
            sin.dtype == torch.float32
        ), f"[Step 6 FAIL] precompute_freqs_cis returned sin as {sin.dtype}, expected fp32."
        return cos, sin

    # NOTE: the pipeline binds ``precompute_freqs_cis`` directly (``from ... import
    # precompute_freqs_cis``), so it must be patched on the pipeline module, not on
    # ``rope_ltx`` — otherwise the pipeline's calls are never intercepted and the
    # count assertion below trips. The other three helpers are still resolved via
    # the ``rope_ltx`` module attribute inside the real ``precompute_freqs_cis``.
    with patch.object(rope_ltx, "generate_freq_grid", checked_grid), patch.object(
        rope_ltx, "generate_freqs", checked_freqs
    ), patch.object(rope_ltx, "split_freqs_cis", checked_split), patch.object(
        pipeline_mod, "precompute_freqs_cis", checked_precompute
    ):
        pipe._prepare_rope(5, 16, 28)
        pipe._prepare_audio_rope(64, 32)
        pipe._prepare_av_cross_pe(5, 16, 28, 64, 32)

    # Sanity: confirm the patches actually fired so we know we exercised the pipeline RoPE paths
    assert counts["precompute"] >= 4, (
        f"Expected ≥4 calls to precompute_freqs_cis (1 video self + 1 audio self + 2 cross), "
        f"got {counts['precompute']}. Test may not be exercising all RoPE paths in the pipeline."
    )

    logger.info(
        f"PASS: steps 1-6 are fp32 throughout the actual pipeline "
        f"({counts['precompute']} precompute, {counts['freqs']} freqs, "
        f"{counts['split_cis']} split_cis, {counts['freq_grid']} freq_grid calls)"
    )

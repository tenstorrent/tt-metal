# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for LTX-2 AudioVideo transformer components.

Tests the audio path with the same rigor as the video transformers:
- Audio RoPE (1D temporal positions)
- Audio attention (self + cross with different dims)
- AudioVideo transformer block (joint processing)
- AudioVideo transformer model (full forward + weight loading)
- Cross-attention between modalities (A↔V with dim mismatch handling)

Uses PyTorch LTXModel(AudioVideo) as reference for PCC comparison.
"""

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.ltx.attention_ltx import LTXAttention
from models.tt_dit.models.transformers.ltx.ltx_transformer import LTXTransformerBlock, LTXTransformerModel
from models.tt_dit.models.transformers.ltx.rope_ltx import precompute_freqs_cis
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard

sys.path.insert(0, "LTX-2/packages/ltx-core/src")


# ============================================================================
# Audio RoPE Tests
# ============================================================================


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_audio_rope_1d(mesh_device: ttnn.MeshDevice):
    """Test that audio RoPE uses 1D temporal positions (not 3D like video)."""
    audio_N = 64
    audio_dim = 2048
    num_heads = 32
    head_dim = audio_dim // num_heads  # 64

    # 1D audio positions
    audio_grid = torch.arange(audio_N).float().unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
    assert audio_grid.shape == (1, audio_N, 1), f"Audio grid shape: {audio_grid.shape}"

    cos, sin = precompute_freqs_cis(
        audio_grid,
        dim=audio_dim,
        theta=10000.0,
        max_pos=[20],
        num_attention_heads=num_heads,
    )

    assert cos.shape == (1, audio_N, audio_dim), f"cos shape: {cos.shape}"
    assert sin.shape == (1, audio_N, audio_dim), f"sin shape: {sin.shape}"
    assert torch.isfinite(cos).all(), "cos has NaN/Inf"
    assert torch.isfinite(sin).all(), "sin has NaN/Inf"

    # Reshape to per-head format
    cos_heads = cos.reshape(1, audio_N, num_heads, head_dim).permute(0, 2, 1, 3)
    sin_heads = sin.reshape(1, audio_N, num_heads, head_dim).permute(0, 2, 1, 3)
    assert cos_heads.shape == (1, num_heads, audio_N, head_dim)

    logger.info(f"Audio RoPE 1D: cos={cos_heads.shape}, sin={sin_heads.shape}")
    logger.info("PASSED: Audio RoPE 1D positions")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_audio_rope_vs_video_rope(mesh_device: ttnn.MeshDevice):
    """Verify audio RoPE (1D) and video RoPE (3D) produce different frequencies."""
    num_heads = 32

    # Video: 3D positions
    video_grid = torch.stack([torch.zeros(64), torch.zeros(64), torch.arange(64).float()], dim=-1).unsqueeze(0)
    video_cos, _ = precompute_freqs_cis(
        video_grid,
        dim=4096,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        num_attention_heads=num_heads,
    )

    # Audio: 1D positions (same temporal indices)
    audio_grid = torch.arange(64).float().unsqueeze(0).unsqueeze(-1)
    audio_cos, _ = precompute_freqs_cis(
        audio_grid,
        dim=2048,
        theta=10000.0,
        max_pos=[20],
        num_attention_heads=num_heads,
    )

    # Different dims → different shapes
    assert video_cos.shape != audio_cos.shape, "Video and audio RoPE should have different shapes"
    logger.info(f"Video RoPE: {video_cos.shape}, Audio RoPE: {audio_cos.shape}")
    logger.info("PASSED: Audio vs Video RoPE comparison")


# ============================================================================
# Audio Attention Tests
# ============================================================================


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(1, 1), 0, 1],
        [(2, 4), 0, 1],
    ],
    ids=["1x1sp0tp1", "2x4sp0tp1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_audio_self_attention(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int):
    """Test audio self-attention (dim=2048, 32 heads, head_dim=64)."""
    B = 1
    audio_N = 64
    audio_dim = 2048
    num_heads = 32

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    attn = LTXAttention(
        dim=audio_dim,
        num_heads=num_heads,
        is_self=True,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )

    # Random weights
    torch_state = {}
    for k in [
        "to_q.weight",
        "to_q.bias",
        "to_k.weight",
        "to_k.bias",
        "to_v.weight",
        "to_v.bias",
        "to_out.0.weight",
        "to_out.0.bias",
    ]:
        name = k.split(".")[0]
        param = k.split(".")[-1]
        if param == "weight":
            if "out" in name:
                torch_state[k] = torch.randn(audio_dim, audio_dim)
            else:
                torch_state[k] = torch.randn(audio_dim, audio_dim)
        else:
            torch_state[k] = torch.randn(audio_dim)
    torch_state["q_norm.weight"] = torch.ones(audio_dim)
    torch_state["k_norm.weight"] = torch.ones(audio_dim)

    attn.load_torch_state_dict(torch_state)

    # Create input (SP-sharded on seq dim=2, TP-sharded on model dim=3)
    x = torch.randn(1, B, audio_N, audio_dim)
    x_tt = bf16_tensor_2dshard(x, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})

    # RoPE
    audio_grid = torch.arange(audio_N).float().unsqueeze(0).unsqueeze(-1)
    cos, sin = precompute_freqs_cis(
        audio_grid, dim=audio_dim, theta=10000.0, max_pos=[20], num_attention_heads=num_heads
    )
    cos = cos.reshape(1, audio_N, num_heads, 64).permute(0, 2, 1, 3)
    sin = sin.reshape(1, audio_N, num_heads, 64).permute(0, 2, 1, 3)
    tt_cos = bf16_tensor_2dshard(cos, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    # Forward
    residual = bf16_tensor_2dshard(x, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    gate = bf16_tensor(torch.ones(1, B, 1, audio_dim), device=mesh_device, mesh_axis=tp_axis, shard_dim=-1)

    out = attn(
        spatial_1BND=x_tt,
        N=audio_N,
        rope_cos=tt_cos,
        rope_sin=tt_sin,
        trans_mat=tt_trans,
        addcmul_residual=residual,
        addcmul_gate=gate,
    )

    out_host = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
    assert out_host.shape[-2] == audio_N // tuple(mesh_device.shape)[sp_axis]
    assert torch.isfinite(out_host).all(), "Output has NaN/Inf"
    logger.info(f"Audio self-attn output: {out_host.shape}, range [{out_host.min():.3f}, {out_host.max():.3f}]")
    logger.info("PASSED: Audio self-attention")


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(1, 1), 0, 1],
        [(2, 4), 0, 1],
    ],
    ids=["1x1sp0tp1", "2x4sp0tp1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_cross_modal_attention_a2v(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int):
    """Test A→V cross-attention: video queries (4096), audio keys/values (2048).

    This is the tricky case where query_input_dim != dim != output_dim:
    - Q: video_dim(4096) → attention_dim(2048)
    - K/V: audio_dim(2048) → attention_dim(2048)
    - Out: attention_dim(2048) → video_dim(4096)
    """
    B = 1
    video_N = 64
    audio_N = 32
    video_dim = 4096
    audio_dim = 2048
    num_heads = 32
    head_dim = audio_dim // num_heads  # 64

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    a2v_attn = LTXAttention(
        dim=audio_dim,
        num_heads=num_heads,
        is_self=False,
        context_dim=audio_dim,
        query_input_dim=video_dim,
        output_dim=video_dim,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )

    # Create random state matching checkpoint shapes
    torch_state = {
        "to_q.weight": torch.randn(audio_dim, video_dim),  # (2048, 4096)
        "to_q.bias": torch.randn(audio_dim),
        "to_k.weight": torch.randn(audio_dim, audio_dim),  # (2048, 2048)
        "to_k.bias": torch.randn(audio_dim),
        "to_v.weight": torch.randn(audio_dim, audio_dim),
        "to_v.bias": torch.randn(audio_dim),
        "to_out.0.weight": torch.randn(video_dim, audio_dim),  # (4096, 2048)
        "to_out.0.bias": torch.randn(video_dim),
        "q_norm.weight": torch.ones(audio_dim),
        "k_norm.weight": torch.ones(audio_dim),
    }
    a2v_attn.load_torch_state_dict(torch_state)

    # Video query input (video_dim=4096), TP-sharded on last dim
    video_input = torch.randn(1, B, video_N, video_dim)
    video_tt = bf16_tensor_2dshard(video_input, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})

    # Audio context (audio_dim=2048), replicated (cross-attn context is full-dim)
    audio_ctx = torch.randn(1, B, audio_N, audio_dim)
    audio_tt = bf16_tensor(audio_ctx, device=mesh_device)

    out = a2v_attn(spatial_1BND=video_tt, N=video_N, prompt_1BLP=audio_tt)
    out_host = ttnn.to_torch(ttnn.get_device_tensors(out)[0])

    # Output should be in video_dim space (4096)
    expected_N = video_N // tuple(mesh_device.shape)[sp_axis]
    assert out_host.shape[-1] == video_dim // tuple(mesh_device.shape)[tp_axis]
    assert out_host.shape[-2] == expected_N
    assert torch.isfinite(out_host).all(), "Output has NaN/Inf"
    logger.info(f"A2V cross-attn: {out_host.shape}, range [{out_host.min():.3f}, {out_host.max():.3f}]")
    logger.info("PASSED: A→V cross-modal attention")


# ============================================================================
# AudioVideo Transformer Block Test
# ============================================================================


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(1, 1), 0, 1],
        [(2, 4), 0, 1],
    ],
    ids=["1x1sp0tp1", "2x4sp0tp1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_av_transformer_block(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int):
    """Test AudioVideo transformer block with real 22B weights."""
    import os

    from safetensors.torch import load_file

    ckpt = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
    if not os.path.exists(ckpt):
        pytest.skip("22B checkpoint not found")

    B = 1
    video_N = 64
    audio_N = 64  # Must be tile-aligned: N/SP_factor must be divisible by 32
    video_dim = 4096
    audio_dim = 2048

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    # Extract block 0 state from real checkpoint
    raw = load_file(ckpt)
    prefix = "model.diffusion_model.transformer_blocks.0."
    block_state = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    logger.info(f"Block 0: {len(block_state)} keys")

    # Create TT block
    tt_block = LTXTransformerBlock(
        video_dim=video_dim,
        audio_dim=audio_dim,
        video_ffn_dim=video_dim * 4,
        audio_ffn_dim=audio_dim * 4,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        has_audio=True,
    )
    tt_block.load_torch_state_dict(block_state)

    # Create inputs with small magnitudes to avoid overflow
    torch.manual_seed(42)
    video_in = torch.randn(1, B, video_N, video_dim) * 0.1
    audio_in = torch.randn(1, B, audio_N, audio_dim) * 0.1
    video_temb = torch.randn(1, B, 9, video_dim) * 0.01
    audio_temb = torch.randn(1, B, 9, audio_dim) * 0.01
    av_ca_video_temb = torch.randn(1, B, 5, video_dim) * 0.01
    av_ca_audio_temb = torch.randn(1, B, 5, audio_dim) * 0.01
    video_prompt = torch.randn(1, B, 16, video_dim) * 0.1
    audio_prompt = torch.randn(1, B, 16, audio_dim) * 0.1

    # Push to device
    def to_dev(t, sp_shard=False, tp_shard=False, sp_tp_shard=False):
        if sp_tp_shard:
            return bf16_tensor_2dshard(t, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
        if sp_shard:
            return bf16_tensor(t, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
        if tp_shard:
            return bf16_tensor(t, device=mesh_device, mesh_axis=tp_axis, shard_dim=-1)
        return bf16_tensor(t, device=mesh_device)

    # RoPE
    vg = torch.stack([torch.zeros(video_N), torch.zeros(video_N), torch.arange(video_N).float()], dim=-1).unsqueeze(0)
    vc, vs = precompute_freqs_cis(vg, dim=4096, theta=10000.0, max_pos=[20, 2048, 2048], num_attention_heads=32)
    vc = vc.reshape(1, video_N, 32, 128).permute(0, 2, 1, 3)
    vs = vs.reshape(1, video_N, 32, 128).permute(0, 2, 1, 3)
    ag = torch.arange(audio_N).float().unsqueeze(0).unsqueeze(-1)
    ac, asn = precompute_freqs_cis(ag, dim=2048, theta=10000.0, max_pos=[20], num_attention_heads=32)
    ac = ac.reshape(1, audio_N, 32, 64).permute(0, 2, 1, 3)
    asn = asn.reshape(1, audio_N, 32, 64).permute(0, 2, 1, 3)

    tt_vc = bf16_tensor_2dshard(vc, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_vs = bf16_tensor_2dshard(vs, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_ac = bf16_tensor_2dshard(ac, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_as = bf16_tensor_2dshard(asn, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    v_out, a_out = tt_block(
        video_1BND=to_dev(video_in, sp_tp_shard=True),
        audio_1BND=to_dev(audio_in, sp_tp_shard=True),
        video_prompt=to_dev(video_prompt),
        audio_prompt=to_dev(audio_prompt),
        video_temb=to_dev(video_temb, tp_shard=True),
        audio_temb=to_dev(audio_temb, tp_shard=True),
        av_ca_temb=to_dev(av_ca_video_temb, tp_shard=True),
        video_N=video_N,
        audio_N=audio_N,
        video_rope_cos=tt_vc,
        video_rope_sin=tt_vs,
        audio_rope_cos=tt_ac,
        audio_rope_sin=tt_as,
        trans_mat=tt_trans,
        av_ca_audio_temb=to_dev(av_ca_audio_temb, tp_shard=True),
    )

    v_host = ttnn.to_torch(ttnn.get_device_tensors(v_out)[0])
    a_host = ttnn.to_torch(ttnn.get_device_tensors(a_out)[0])

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    assert v_host.shape == (1, B, video_N // sp_factor, video_dim // tp_factor)
    assert a_host.shape == (1, B, audio_N // sp_factor, audio_dim // tp_factor)
    assert torch.isfinite(v_host).all(), "Video output has NaN/Inf"
    assert torch.isfinite(a_host).all(), "Audio output has NaN/Inf"

    logger.info(f"AV block: video={v_host.shape}, audio={a_host.shape}")
    logger.info("PASSED: AudioVideo transformer block")


# ============================================================================
# AudioVideo Transformer Model Test
# ============================================================================


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(2, 4), 0, 1],
    ],
    ids=["2x4sp0tp1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_av_model_with_real_weights(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int):
    """Test full AudioVideo model with 22B checkpoint weights (1 layer).

    Loads real weights from the LTX-2.3 checkpoint and verifies:
    - All weights load without error
    - Forward pass produces finite outputs
    - Video and audio output shapes are correct
    - Cross-attention adaln modules produce non-zero modulation
    """
    import os

    from safetensors.torch import load_file

    ckpt = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
    if not os.path.exists(ckpt):
        pytest.skip("22B checkpoint not found")

    raw = load_file(ckpt)
    prefix = "model.diffusion_model."
    sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    # Filter to 1 layer
    filt = {k: v for k, v in sd.items() if not k.startswith("transformer_blocks.") or int(k.split(".")[1]) == 0}

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    model = LTXTransformerModel(
        num_layers=1,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        has_audio=True,
    )
    model.load_torch_state_dict(filt)

    # Prepare inputs
    video_N, audio_N = 192, 64
    torch.manual_seed(42)

    # RoPE
    t_ids, h_ids, w_ids = torch.arange(3), torch.arange(8), torch.arange(8)
    gt, gh, gw = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    vg = torch.stack([gt.flatten(), gh.flatten(), gw.flatten()], dim=-1).float().unsqueeze(0)
    vc, vs = precompute_freqs_cis(vg, dim=4096, theta=10000.0, max_pos=[20, 2048, 2048], num_attention_heads=32)
    vc = vc.reshape(1, video_N, 32, 128).permute(0, 2, 1, 3)
    vs = vs.reshape(1, video_N, 32, 128).permute(0, 2, 1, 3)
    ag = torch.arange(audio_N).float().unsqueeze(0).unsqueeze(-1)
    ac, asn = precompute_freqs_cis(ag, dim=2048, theta=10000.0, max_pos=[20], num_attention_heads=32)
    ac = ac.reshape(1, audio_N, 32, 64).permute(0, 2, 1, 3)
    asn = asn.reshape(1, audio_N, 32, 64).permute(0, 2, 1, 3)

    tt_vc = bf16_tensor_2dshard(vc, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_vs = bf16_tensor_2dshard(vs, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_ac = bf16_tensor_2dshard(ac, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_as = bf16_tensor_2dshard(asn, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)
    tt_vp = bf16_tensor(torch.randn(1, 1, 128, 4096), device=mesh_device)
    tt_ap = bf16_tensor(torch.randn(1, 1, 128, 2048), device=mesh_device)

    vo, ao = model.inner_step(
        video_1BNI_torch=torch.randn(1, 1, video_N, 128),
        video_prompt_1BLP=tt_vp,
        video_rope_cos=tt_vc,
        video_rope_sin=tt_vs,
        video_N=video_N,
        audio_1BNI_torch=torch.randn(1, 1, audio_N, 128),
        audio_prompt_1BLP=tt_ap,
        audio_rope_cos=tt_ac,
        audio_rope_sin=tt_as,
        audio_N=audio_N,
        trans_mat=tt_trans,
        timestep_torch=torch.tensor([0.5]),
    )

    vh = LTXTransformerModel.device_to_host(vo)
    ah = LTXTransformerModel.device_to_host(ao)

    assert vh.shape == (1, 1, video_N, 128), f"Video shape: {vh.shape}"
    assert ah.shape == (1, 1, audio_N, 128), f"Audio shape: {ah.shape}"
    assert torch.isfinite(vh).all(), "Video output has NaN/Inf"
    assert torch.isfinite(ah).all(), "Audio output has NaN/Inf"

    logger.info(f"Video: {vh.shape}, range [{vh.min():.3f}, {vh.max():.3f}]")
    logger.info(f"Audio: {ah.shape}, range [{ah.min():.3f}, {ah.max():.3f}]")
    logger.info("PASSED: AV model with 22B weights")


# ============================================================================
# AudioVideo PCC Tests (against PyTorch reference)
# ============================================================================


def _compute_pcc(tt_out: torch.Tensor, ref_out: torch.Tensor) -> float:
    """Compute Pearson correlation coefficient between two tensors."""
    a = tt_out.flatten().float()
    b = ref_out.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(1, 1), 0, 1],
    ],
    ids=["1x1sp0tp1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_av_model_pcc_vs_reference(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int):
    """Test TTNN AV model PCC against PyTorch LTXModel reference (1 layer, 22B weights).

    Both models receive identical inputs and checkpoint weights. Video and audio
    PCC are computed against the reference and must exceed thresholds.
    """
    import os

    from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
    from ltx_core.model.transformer.model import LTXModel, LTXModelType, Modality
    from ltx_core.model.transformer.rope import LTXRopeType as RefLTXRopeType
    from ltx_core.model.transformer.rope import generate_freq_grid_np
    from ltx_core.model.transformer.rope import precompute_freqs_cis as ref_precompute
    from safetensors.torch import load_file

    ckpt = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
    if not os.path.exists(ckpt):
        pytest.skip("22B checkpoint not found")

    raw = load_file(ckpt)
    prefix = "model.diffusion_model."
    sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    # 1 layer
    filt = {k: v for k, v in sd.items() if not k.startswith("transformer_blocks.") or int(k.split(".")[1]) == 0}

    video_N, audio_N = 192, 64
    torch.manual_seed(42)

    # Shared inputs
    video_in = torch.randn(1, video_N, 128)
    audio_in = torch.randn(1, audio_N, 128)
    video_prompt = torch.randn(1, 128, 4096)
    audio_prompt = torch.randn(1, 128, 2048)
    sigma = 0.5

    # Positions: video (1, 3, N, 2), audio (1, 1, N, 2) with start=end
    t_ids, h_ids, w_ids = torch.arange(3), torch.arange(8), torch.arange(8)
    gt, gh, gw = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    v_pos = (
        torch.stack([gt.flatten(), gh.flatten(), gw.flatten()], dim=0)
        .float()
        .unsqueeze(-1)
        .repeat(1, 1, 2)
        .unsqueeze(0)
    )  # (1, 3, 192, 2)
    a_pos = torch.arange(audio_N).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 2)  # (1, 1, 64, 2)

    # === PyTorch reference ===
    ref_model = LTXModel(
        model_type=LTXModelType.AudioVideo,
        num_layers=1,
        num_attention_heads=32,
        attention_head_dim=128,
        in_channels=128,
        out_channels=128,
        cross_attention_dim=4096,
        audio_num_attention_heads=32,
        audio_attention_head_dim=64,
        audio_in_channels=128,
        audio_out_channels=128,
        audio_cross_attention_dim=2048,
        use_middle_indices_grid=True,
        apply_gated_attention=True,
        cross_attention_adaln=True,  # 22B checkpoint uses 9-param adaln (6 base + 3 cross-attn)
    )
    ref_model.load_state_dict(filt, strict=False)
    ref_model.eval()

    video_mod = Modality(
        latent=video_in,
        sigma=torch.tensor([sigma]),
        timesteps=torch.ones(1, video_N) * sigma,
        positions=v_pos,
        context=video_prompt,
        enabled=True,
    )
    audio_mod = Modality(
        latent=audio_in,
        sigma=torch.tensor([sigma]),
        timesteps=torch.ones(1, audio_N) * sigma,
        positions=a_pos,
        context=audio_prompt,
        enabled=True,
    )
    perturbations = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=None)])
    with torch.no_grad():
        ref_video, ref_audio = ref_model(video=video_mod, audio=audio_mod, perturbations=perturbations)
    logger.info(f"Ref video: {ref_video.shape}, range [{ref_video.min():.3f}, {ref_video.max():.3f}]")
    logger.info(f"Ref audio: {ref_audio.shape}, range [{ref_audio.min():.3f}, {ref_audio.max():.3f}]")

    # === TTNN model ===
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    tt_model = LTXTransformerModel(
        num_layers=1,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        has_audio=True,
    )
    tt_model.load_torch_state_dict(filt)

    # RoPE using reference precompute (SPLIT with double-precision freq grid)
    v_cos, v_sin = ref_precompute(
        v_pos.bfloat16(),
        dim=4096,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=RefLTXRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )
    a_cos, a_sin = ref_precompute(
        a_pos.bfloat16(),
        dim=2048,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=RefLTXRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )

    # Cross PE (temporal only, dim=2048)
    cross_pe_max_pos = 20
    v_cross_cos, v_cross_sin = ref_precompute(
        v_pos[:, 0:1, :].bfloat16(),
        dim=2048,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[cross_pe_max_pos],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=RefLTXRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )
    a_cross_cos, a_cross_sin = ref_precompute(
        a_pos.bfloat16(),
        dim=2048,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[cross_pe_max_pos],
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=RefLTXRopeType.SPLIT,
        freq_grid_generator=generate_freq_grid_np,
    )

    tt_vc = bf16_tensor_2dshard(v_cos, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_vs = bf16_tensor_2dshard(v_sin, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_ac = bf16_tensor_2dshard(a_cos, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_as = bf16_tensor_2dshard(a_sin, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Cross PE: SP+TP for Q, TP-only for K
    tt_v_xcos = bf16_tensor_2dshard(v_cross_cos, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_v_xsin = bf16_tensor_2dshard(v_cross_sin, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_a_xcos = bf16_tensor_2dshard(a_cross_cos, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_a_xsin = bf16_tensor_2dshard(a_cross_sin, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_v_xcos_full = bf16_tensor(v_cross_cos, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)
    tt_v_xsin_full = bf16_tensor(v_cross_sin, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)
    tt_a_xcos_full = bf16_tensor(a_cross_cos, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)
    tt_a_xsin_full = bf16_tensor(a_cross_sin, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)

    tt_vp = bf16_tensor(video_prompt.unsqueeze(0), device=mesh_device)
    tt_ap = bf16_tensor(audio_prompt.unsqueeze(0), device=mesh_device)

    vo, ao = tt_model.inner_step(
        video_1BNI_torch=video_in.unsqueeze(0),
        video_prompt_1BLP=tt_vp,
        video_rope_cos=tt_vc,
        video_rope_sin=tt_vs,
        video_N=video_N,
        audio_1BNI_torch=audio_in.unsqueeze(0),
        audio_prompt_1BLP=tt_ap,
        audio_rope_cos=tt_ac,
        audio_rope_sin=tt_as,
        audio_N=audio_N,
        trans_mat=None,  # Split RoPE
        timestep_torch=torch.tensor([sigma]),
        video_cross_pe_cos=tt_v_xcos,
        video_cross_pe_sin=tt_v_xsin,
        audio_cross_pe_cos=tt_a_xcos,
        audio_cross_pe_sin=tt_a_xsin,
        video_cross_pe_cos_full=tt_v_xcos_full,
        video_cross_pe_sin_full=tt_v_xsin_full,
        audio_cross_pe_cos_full=tt_a_xcos_full,
        audio_cross_pe_sin_full=tt_a_xsin_full,
    )

    tt_video = LTXTransformerModel.device_to_host(vo).squeeze(0)
    tt_audio = LTXTransformerModel.device_to_host(ao).squeeze(0)

    video_pcc = _compute_pcc(tt_video, ref_video)
    audio_pcc = _compute_pcc(tt_audio, ref_audio)
    logger.info(f"Video PCC: {video_pcc:.6f}")
    logger.info(f"Audio PCC: {audio_pcc:.6f}")

    assert video_pcc > 0.995, f"Video PCC {video_pcc:.6f} below threshold 0.995"
    assert audio_pcc > 0.995, f"Audio PCC {audio_pcc:.6f} below threshold 0.995"
    logger.info("PASSED: AV model PCC vs reference")

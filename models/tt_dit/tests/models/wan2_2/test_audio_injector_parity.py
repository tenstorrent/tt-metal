# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parity test: AudioInjector block-diagonal mask vs per-frame rearrange.

Our :meth:`WanS2VTransformer3DModel.after_transformer_block` flattens
per-frame audio K/V to a single sequence ``[1, B, T*(N_audio+1), dim]`` and
applies a block-diagonal cross-attention mask so each Q token attends only
to its own frame's audio K/V tokens. The reference does this differently —
``rearrange("b (t n) c -> (b t) n c")`` Q + audio + per-frame cross-attn.

Mathematically they should be equivalent (softmax with -inf mask zeros out
the off-frame contributions). This test verifies that on real hardware.

Test bar: PCC ≥ 0.99.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from loguru import logger

import ttnn

from ....layers.normalization import DistributedRMSNorm
from ....models.transformers.wan2_2.attention_wan import WanAttention
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, from_torch, local_device_to_torch
from ....utils.test import line_params

# Reduced config that exercises the block-diagonal mask machinery.
DIM = 256
NUM_HEADS = 4
HEAD_DIM = DIM // NUM_HEADS  # 64
T_VIDEO = 8
N_PER_FRAME = 32  # spatial tokens per frame (after patch)
N_AUDIO_PER_FRAME = 5  # num_audio_token + 1 (padding token)
B = 1


def _host_per_frame_cross_attn(
    q_BTNI: torch.Tensor,
    kv_BTAI: torch.Tensor,
    *,
    q_w: torch.Tensor,
    q_b: torch.Tensor,
    k_w: torch.Tensor,
    k_b: torch.Tensor,
    v_w: torch.Tensor,
    v_b: torch.Tensor,
    o_w: torch.Tensor,
    o_b: torch.Tensor,
    norm_q_w: torch.Tensor,
    norm_k_w: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference per-frame cross-attention.

    Mirrors the reference's ``rearrange("b (t n) c -> (b t) n c")`` flow
    with a standard QKV → SDPA → output projection. Reads `q` from
    `q_BTNI` (per-frame spatial), `kv` from `kv_BTAI` (per-frame audio).
    """
    B, T, N, _ = q_BTNI.shape
    _, _, A, _ = kv_BTAI.shape
    out_BTNI = torch.zeros_like(q_BTNI)
    for t in range(T):
        q_frame = q_BTNI[:, t]  # [B, N, dim]
        kv_frame = kv_BTAI[:, t]  # [B, A, dim]
        q = q_frame @ q_w.T + q_b
        k = kv_frame @ k_w.T + k_b
        v = kv_frame @ v_w.T + v_b
        # RMSNorm over last dim, per-token.
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(q.dtype) * norm_q_w
        k = k * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + eps).to(k.dtype) * norm_k_w
        # Split heads: [B, N, num_heads, head_dim] → [B, num_heads, N, head_dim]
        qh = q.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        kh = k.view(B, A, num_heads, head_dim).permute(0, 2, 1, 3)
        vh = v.view(B, A, num_heads, head_dim).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(qh, kh, vh)
        attn = attn.permute(0, 2, 1, 3).reshape(B, N, num_heads * head_dim)
        out_BTNI[:, t] = attn @ o_w.T + o_b
    return out_BTNI


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (2, 4),
            (2, 4),
            1,
            0,
            2,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="bh_2x4sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_audio_injector_block_diagonal_vs_per_frame(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """Block-diagonal mask flat cross-attn ≅ per-frame cross-attn (reference).

    Skips the ref/motion const tokens — this test focuses purely on the
    noisy-portion cross-attn equivalence (which is the math the production
    `after_transformer_block` relies on, with the const region zeroed out
    by `_cached_mask_noisy` after the fact).
    """
    torch.manual_seed(0)

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    # ---- Build cross-attention module. ----
    attn = WanAttention(
        dim=DIM,
        num_heads=NUM_HEADS,
        qk_norm=True,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        is_self=False,
    )
    # Synthesize a torch-convention state dict (random weights), load into the
    # TT module, and reuse the same host tensors as the reference reproduction.
    # WanAttention.is_self=False expects separate to_q / to_k / to_v / to_out
    # entries plus norm_q / norm_k; ``_prepare_torch_state`` interleaves and fuses.
    q_w = torch.randn(DIM, DIM, dtype=torch.float32)
    q_b = torch.randn(DIM, dtype=torch.float32)
    k_w = torch.randn(DIM, DIM, dtype=torch.float32)
    k_b = torch.randn(DIM, dtype=torch.float32)
    v_w = torch.randn(DIM, DIM, dtype=torch.float32)
    v_b = torch.randn(DIM, dtype=torch.float32)
    o_w = torch.randn(DIM, DIM, dtype=torch.float32)
    o_b = torch.randn(DIM, dtype=torch.float32)
    norm_q_w = torch.randn(DIM, dtype=torch.float32)
    norm_k_w = torch.randn(DIM, dtype=torch.float32)
    synth_state = {
        "to_q.weight": q_w,
        "to_q.bias": q_b,
        "to_k.weight": k_w,
        "to_k.bias": k_b,
        "to_v.weight": v_w,
        "to_v.bias": v_b,
        "to_out.0.weight": o_w,
        "to_out.0.bias": o_b,
        "norm_q.weight": norm_q_w,
        "norm_k.weight": norm_k_w,
    }
    incompat = attn.load_torch_state_dict(synth_state, strict=False)
    logger.info(
        f"WanAttention load: missing={len(incompat.missing_keys)} " f"unexpected={len(incompat.unexpected_keys)}"
    )

    # ---- Build inputs. ----
    # Spatial Q: per-frame [B, T, N, dim]. We pass it as a flat sequence to the TT module.
    q_BTNI_torch = torch.randn(B, T_VIDEO, N_PER_FRAME, DIM, dtype=torch.float32)
    # Audio K/V: per-frame [B, T, A, dim].
    kv_BTAI_torch = torch.randn(B, T_VIDEO, N_AUDIO_PER_FRAME, DIM, dtype=torch.float32)

    # ---- Reference per-frame path. ----
    with torch.no_grad():
        ref_out_BTNI = _host_per_frame_cross_attn(
            q_BTNI_torch,
            kv_BTAI_torch,
            q_w=q_w,
            q_b=q_b,
            k_w=k_w,
            k_b=k_b,
            v_w=v_w,
            v_b=v_b,
            o_w=o_w,
            o_b=o_b,
            norm_q_w=norm_q_w,
            norm_k_w=norm_k_w,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
        )
    ref_flat = ref_out_BTNI.reshape(B, T_VIDEO * N_PER_FRAME, DIM)  # [B, N_total, dim]
    logger.info(f"Reference per-frame output: shape={tuple(ref_flat.shape)}")

    # ---- Pure-pytorch block-diagonal mask reference. ----
    # Mirrors /home/kevinmi/wan2_2_ref/wan/modules/model.py:158 WanCrossAttention,
    # extended to take an explicit additive mask. Uses the SAME synthetic
    # weights as the host per-frame reference and the TT path. Comparing this
    # to the host per-frame reference isolates whether the per-frame ≡
    # block-diagonal-mask math is correct; comparing it to TT isolates whether
    # the TT implementation matches its pytorch equivalent.
    N_noisy = T_VIDEO * N_PER_FRAME
    q_flat_torch = q_BTNI_torch.reshape(B, N_noisy, DIM).unsqueeze(0)
    kv_flat_torch = kv_BTAI_torch.reshape(B, T_VIDEO * N_AUDIO_PER_FRAME, DIM).unsqueeze(0)
    # NOTE: use -1e9 instead of float("-inf"). bf16 can represent -inf, but the
    # SDPA exp-approx + softmax path may handle it differently than a large
    # finite negative. The pytorch reference uses the same value so this is a
    # like-for-like comparison.
    MASK_NEG = -1e9
    mask_torch = torch.full((1, 1, N_noisy, T_VIDEO * N_AUDIO_PER_FRAME), MASK_NEG, dtype=torch.float32)
    for t in range(T_VIDEO):
        mask_torch[
            ..., t * N_PER_FRAME : (t + 1) * N_PER_FRAME, t * N_AUDIO_PER_FRAME : (t + 1) * N_AUDIO_PER_FRAME
        ] = 0.0

    def _pyt_block_mask_cross_attn() -> torch.Tensor:
        """Reference WanCrossAttention math + block-diagonal mask.

        Single-device pytorch reproduction. Returns [B, N_noisy, dim].
        """
        eps = 1e-5  # WanAttention default eps; see attention_wan.py:39
        q = q_flat_torch.squeeze(0) @ q_w.T + q_b  # [B, N_noisy, dim]
        k = kv_flat_torch.squeeze(0) @ k_w.T + k_b  # [B, T*A, dim]
        v = kv_flat_torch.squeeze(0) @ v_w.T + v_b
        # Reference WanRMSNorm: over the FULL last dim (i.e. concat of all heads).
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(q.dtype) * norm_q_w
        k = k * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + eps).to(k.dtype) * norm_k_w
        # Split heads: [B, S, num_heads, head_dim] → [B, num_heads, S, head_dim].
        qh = q.view(B, N_noisy, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        kh = k.view(B, T_VIDEO * N_AUDIO_PER_FRAME, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        vh = v.view(B, T_VIDEO * N_AUDIO_PER_FRAME, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        # Block-diagonal mask: [1, 1, N_noisy, T*A]; broadcasts over batch + heads.
        attn = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask_torch.squeeze(0))
        attn = attn.permute(0, 2, 1, 3).reshape(B, N_noisy, NUM_HEADS * HEAD_DIM)
        return attn @ o_w.T + o_b

    with torch.no_grad():
        pyt_flat = _pyt_block_mask_cross_attn().float()
    logger.info(f"Pytorch block-mask output: shape={tuple(pyt_flat.shape)}")

    # ---- TT block-diagonal path. ----

    # Upload tensors with the right sharding.
    sp_factor = parallel_config.sequence_parallel.factor
    # Q is SP-sharded on N (dim 2) and TP-sharded on D (dim 3) — matches the
    # spatial_1BND contract in WanAttention.forward (docstring lines 309-313).
    q_dev = from_torch(
        q_flat_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=[None, None, sp_axis, tp_axis],
    )
    # K/V is replicated.
    kv_dev = bf16_tensor(kv_flat_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    # Mask is SP-sharded on Sq. Critical: pad_value=MASK_NEG so the TILE-padded
    # Sk columns are masked out (default pad fill 0 would leave them unmasked
    # and SDPA would attend to zero-filled K positions, diluting the output).
    mask_dev = from_torch(
        mask_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=[None, None, sp_axis, None],
        pad_value=MASK_NEG,
    )

    # Run our cross-attention.
    out_dev = attn.forward(
        spatial_1BND=q_dev,
        N=N_noisy,
        prompt_1BLP=kv_dev,
        cross_attn_mask=mask_dev,
    )
    # Gather across SP (N) and TP (D).
    out_gather = ccl_manager.all_gather_persistent_buffer(out_dev, dim=2, mesh_axis=sp_axis)
    out_gather = ccl_manager.all_gather_persistent_buffer(out_gather, dim=3, mesh_axis=tp_axis)
    out_flat = local_device_to_torch(out_gather).squeeze(0).float()  # [B, N_noisy, dim]
    logger.info(f"TT block-diagonal output: shape={tuple(out_flat.shape)}")

    # ---- 3-way compare. ----
    # (a) pyt block-mask vs host per-frame: tests whether the per-frame ≡
    #     block-diagonal-mask equivalence holds in pure pytorch math.
    # (b) TT vs pyt block-mask: tests whether the TT implementation matches
    #     its pure-pytorch equivalent.
    # If (a) passes and (b) fails, the TT impl is the bug.
    # If (a) fails, the host per-frame reference has a math bug.
    logger.info("--- (a) pytorch block-mask vs host per-frame reference ---")
    try:
        assert_quality(pyt_flat, ref_flat.float(), pcc=0.99)
        logger.info("PASS: per-frame ≡ block-diagonal-mask equivalence holds in pure pytorch")
        a_passed = True
    except Exception as exc:
        logger.warning(f"FAIL: per-frame ≢ block-diagonal-mask in pure pytorch — host reference bug. {exc}")
        a_passed = False

    logger.info("--- (b) TT vs pytorch block-mask ---")
    try:
        assert_quality(out_flat, pyt_flat, pcc=0.99)
        logger.info("PASS: TT block-mask matches pytorch block-mask")
        b_passed = True
    except Exception as exc:
        logger.warning(f"FAIL: TT block-mask differs from pytorch block-mask — TT impl bug. {exc}")
        b_passed = False

    # The original assertion: TT vs host per-frame. Kept so the test reports
    # the headline PCC, but the localization signal comes from (a) and (b).
    assert a_passed and b_passed, (
        f"3-way mismatch — host_per_frame≡pyt_block_mask: {a_passed}, " f"TT≡pyt_block_mask: {b_passed}"
    )
    assert_quality(out_flat, ref_flat.float(), pcc=0.99)


# ----------------------------------------------------------------------
# Focused localization test: is DistributedRMSNorm with num_heads_per_device
# the source of the audio_injector PCC gap?
#
# WanAttention.forward calls norm_q / norm_k with num_heads_per_device=2 on TP=2.
# If this op does NOT produce the same numerical result as a single-device,
# full-embedding-dim RMSNorm, that explains the divergence.
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (2, 4),
            (2, 4),
            1,
            0,
            2,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="bh_2x4sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_distributed_rms_norm_full_dim_vs_per_head(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """Verifies DistributedRMSNorm(num_heads_per_device=k) ≡ full-embedding-dim RMSNorm.

    Hypothesis: the audio_injector PCC gap (82.76% TT vs pure-pytorch) comes
    from norm_q/norm_k. The reference (wan/modules/model.py:69 WanRMSNorm)
    normalizes over the FULL last dim. If our TT op instead normalizes per-head,
    that's the bug.
    """
    torch.manual_seed(0)

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    # Match the audio_injector test's config.
    dim = DIM  # 256
    num_heads = NUM_HEADS  # 4
    head_dim = HEAD_DIM  # 64
    tp_factor = parallel_config.tensor_parallel.factor  # 2
    n_local_heads = num_heads // tp_factor  # 2
    sp_factor = parallel_config.sequence_parallel.factor  # 4

    # ---- Build the norm module ----
    norm = DistributedRMSNorm(
        embedding_dim=dim,
        norm_eps=1e-5,
        norm_elementwise_affine=True,
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
    )
    weight_torch = torch.randn(dim, dtype=torch.float32)
    norm.load_torch_state_dict({"weight": weight_torch}, strict=False)

    # ---- Input ----
    # Same shape as in the audio_injector test: [1, B, N_noisy, dim], SP-sharded
    # on N, TP-sharded on dim.
    N_noisy = T_VIDEO * N_PER_FRAME  # 256
    x_torch = torch.randn(1, B, N_noisy, dim, dtype=torch.float32)
    x_dev = from_torch(
        x_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=[None, None, sp_axis, tp_axis],
    )

    # ---- TT norm ----
    y_dev = norm(x_dev, num_heads_per_device=n_local_heads)
    # Per-device output layout is [B, n_local_heads, N_local, head_dim] — the op
    # fuses head-split into the norm. Heads axis is dim 1 (TP-fractured); N is
    # dim 2 (SP-fractured); head_dim is dim 3 (replicated across both axes).
    logger.info(f"TT norm output shape (per-device, pre-gather): {tuple(y_dev.shape)}")
    # Gather SP on dim 2 (N), then TP on dim 1 (heads).
    y_gather_sp = ccl_manager.all_gather_persistent_buffer(y_dev, dim=2, mesh_axis=sp_axis)
    logger.info(f"After SP gather (dim=2), shape: {tuple(y_gather_sp.shape)}")
    y_full = ccl_manager.all_gather_persistent_buffer(y_gather_sp, dim=1, mesh_axis=tp_axis)
    logger.info(f"After TP gather (dim=1), shape: {tuple(y_full.shape)}")
    y_torch_full = local_device_to_torch(y_full).float()  # expected [B, H, N, E]
    # Reshape to [B, N, H*E=dim] to match host RMSNorm output.
    y_BHNE = y_torch_full  # [B, H, N, E]
    y_BNHE = y_BHNE.permute(0, 2, 1, 3)  # [B, N, H, E]
    y_BND_tt = y_BNHE.reshape(B, N_noisy, dim)
    logger.info(f"TT norm gathered (host-equivalent): shape={tuple(y_BND_tt.shape)}")

    # ---- Host full-dim RMSNorm reference ----
    # Note: TT output is upcast-fused inside the op but we use the same eps.
    eps = 1e-5
    x_BND = x_torch.squeeze(0)  # [B, N, dim]
    y_host_BND = (x_BND.float() * torch.rsqrt(x_BND.float().pow(2).mean(-1, keepdim=True) + eps) * weight_torch).float()
    logger.info(f"Host full-dim RMSNorm: shape={tuple(y_host_BND.shape)}")

    # ---- Host PER-HEAD RMSNorm reference (the alternative hypothesis) ----
    x_BNHE = x_BND.view(B, N_noisy, num_heads, head_dim)
    y_per_head_BNHE = x_BNHE * torch.rsqrt(x_BNHE.pow(2).mean(-1, keepdim=True) + eps)
    # Apply weight (per-head reshape of full weight).
    w_HE = weight_torch.view(num_heads, head_dim)
    y_per_head_BNHE = y_per_head_BNHE * w_HE
    y_per_head_BND = y_per_head_BNHE.reshape(B, N_noisy, dim)

    # ---- Compare ----
    logger.info("--- TT vs host FULL-DIM RMSNorm ---")
    try:
        assert_quality(y_BND_tt, y_host_BND.float(), pcc=0.99)
        logger.info("PASS: TT matches full-dim RMSNorm")
        tt_is_full_dim = True
    except Exception as exc:
        logger.warning(f"FAIL: TT differs from full-dim RMSNorm: {exc}")
        tt_is_full_dim = False

    logger.info("--- TT vs host PER-HEAD RMSNorm ---")
    try:
        assert_quality(y_BND_tt, y_per_head_BND.float(), pcc=0.99)
        logger.info("PASS: TT matches per-head RMSNorm")
        tt_is_per_head = True
    except Exception as exc:
        logger.warning(f"FAIL: TT differs from per-head RMSNorm: {exc}")
        tt_is_per_head = False

    if tt_is_full_dim:
        logger.info("DistributedRMSNorm is full-dim — NOT the audio_injector bug source.")
    elif tt_is_per_head:
        logger.warning(
            "DistributedRMSNorm computes PER-HEAD RMSNorm — this IS the audio_injector "
            "bug source (reference WanRMSNorm normalizes over full embedding dim)."
        )
    else:
        logger.warning("TT matches neither full-dim NOR per-head — something else is going on.")

    # Fail the test only if TT matches neither — that's a strict regression.
    # If it matches per-head, we want the test to flag this clearly without
    # crashing pytest, so the result is visible.
    assert tt_is_full_dim or tt_is_per_head, "TT norm matches neither full-dim nor per-head RMSNorm"


# ----------------------------------------------------------------------
# Stage-by-stage instrumented audio_injector test.
#
# Manually walks through WanAttention.forward's cross-attn flow, capturing
# the gathered host equivalent at each stage and comparing to the pytorch
# reference. The first stage that drops below PCC 0.99 localizes the bug.
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (2, 4),
            (2, 4),
            1,
            0,
            2,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="bh_2x4sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_audio_injector_staged(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """Step-through TT cross-attn flow with per-stage compare to pytorch.

    Stages: A. post-to_q, B. post-to_kv (k, v), C. post-norm_q, D. post-norm_k,
    E. post-SDPA (after concat_heads + TP-gather), F. post-to_out (final).

    First stage that drops below PCC 0.99 is the bug site.
    """
    import torch.nn.functional as F  # noqa: N812

    torch.manual_seed(0)

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    tp_factor = parallel_config.tensor_parallel.factor
    sp_factor = parallel_config.sequence_parallel.factor

    attn = WanAttention(
        dim=DIM,
        num_heads=NUM_HEADS,
        qk_norm=True,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        is_self=False,
    )
    # Same synth state dict as the headline test.
    q_w = torch.randn(DIM, DIM, dtype=torch.float32)
    q_b = torch.randn(DIM, dtype=torch.float32)
    k_w = torch.randn(DIM, DIM, dtype=torch.float32)
    k_b = torch.randn(DIM, dtype=torch.float32)
    v_w = torch.randn(DIM, DIM, dtype=torch.float32)
    v_b = torch.randn(DIM, dtype=torch.float32)
    o_w = torch.randn(DIM, DIM, dtype=torch.float32)
    o_b = torch.randn(DIM, dtype=torch.float32)
    norm_q_w = torch.randn(DIM, dtype=torch.float32)
    norm_k_w = torch.randn(DIM, dtype=torch.float32)
    attn.load_torch_state_dict(
        {
            "to_q.weight": q_w,
            "to_q.bias": q_b,
            "to_k.weight": k_w,
            "to_k.bias": k_b,
            "to_v.weight": v_w,
            "to_v.bias": v_b,
            "to_out.0.weight": o_w,
            "to_out.0.bias": o_b,
            "norm_q.weight": norm_q_w,
            "norm_k.weight": norm_k_w,
        },
        strict=False,
    )

    # Inputs.
    N_noisy = T_VIDEO * N_PER_FRAME
    L = T_VIDEO * N_AUDIO_PER_FRAME
    q_BTNI_torch = torch.randn(B, T_VIDEO, N_PER_FRAME, DIM, dtype=torch.float32)
    kv_BTAI_torch = torch.randn(B, T_VIDEO, N_AUDIO_PER_FRAME, DIM, dtype=torch.float32)
    q_flat_torch = q_BTNI_torch.reshape(B, N_noisy, DIM).unsqueeze(0)  # [1, B, N_noisy, dim]
    kv_flat_torch = kv_BTAI_torch.reshape(B, L, DIM).unsqueeze(0)  # [1, B, L, dim]
    MASK_NEG = -1e9
    mask_torch = torch.full((1, 1, N_noisy, L), MASK_NEG, dtype=torch.float32)
    for t in range(T_VIDEO):
        mask_torch[
            ..., t * N_PER_FRAME : (t + 1) * N_PER_FRAME, t * N_AUDIO_PER_FRAME : (t + 1) * N_AUDIO_PER_FRAME
        ] = 0.0

    # Pytorch reference: compute each stage on host with the same weights.
    pyt_q = q_flat_torch.squeeze(0) @ q_w.T + q_b  # [B, N_noisy, dim]
    pyt_k = kv_flat_torch.squeeze(0) @ k_w.T + k_b
    pyt_v = kv_flat_torch.squeeze(0) @ v_w.T + v_b
    eps = 1e-5
    pyt_q_normed = pyt_q * torch.rsqrt(pyt_q.float().pow(2).mean(-1, keepdim=True) + eps).to(pyt_q.dtype) * norm_q_w
    pyt_k_normed = pyt_k * torch.rsqrt(pyt_k.float().pow(2).mean(-1, keepdim=True) + eps).to(pyt_k.dtype) * norm_k_w
    pyt_qh = pyt_q_normed.view(B, N_noisy, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    pyt_kh = pyt_k_normed.view(B, L, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    pyt_vh = pyt_v.view(B, L, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    pyt_attn_BHNE = F.scaled_dot_product_attention(pyt_qh, pyt_kh, pyt_vh, attn_mask=mask_torch.squeeze(0))
    pyt_attn_BND = pyt_attn_BHNE.permute(0, 2, 1, 3).reshape(B, N_noisy, DIM)
    pyt_out = pyt_attn_BND @ o_w.T + o_b  # [B, N_noisy, dim]

    # ---- Upload inputs in the production sharding contract. ----
    q_dev = from_torch(
        q_flat_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=[None, None, sp_axis, tp_axis],
    )
    kv_dev = bf16_tensor(kv_flat_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    # Critical: TILE layout pads Sk=40 → 64. Default pad fill is 0, which
    # leaves the padded mask cols UNMASKED. SDPA softmax then distributes
    # attention over 24 zero-padded K positions, diluting the result. Use
    # pad_value=MASK_NEG so padded cols are masked out.
    mask_dev = from_torch(
        mask_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=[None, None, sp_axis, None],
        pad_value=MASK_NEG,
    )

    # Manually replay WanAttention.forward's cross-attn path with captures.
    # Step 0: TP all-gather spatial (because Linear topology + TP>1 → use_nonfused_agmm).
    spatial = ccl_manager.all_gather_persistent_buffer(q_dev, dim=3, mesh_axis=tp_axis)
    # After gather: per device [1, B, N_sp, dim] (full dim, SP-frac on N).

    # ---- Stage A: to_q ----
    q_1BNF = attn.to_q(spatial, compute_kernel_config=attn.mm_compute_kernel_config, parallel_config=None)
    # Output: TP-frac on D → per device [1, B, N_sp, D/tp=128]. Gather SP then TP.
    q_full_dev = ccl_manager.all_gather_persistent_buffer(q_1BNF, dim=2, mesh_axis=sp_axis)
    q_full_dev = ccl_manager.all_gather_persistent_buffer(q_full_dev, dim=3, mesh_axis=tp_axis)
    tt_q_BND = local_device_to_torch(q_full_dev).squeeze(0).float()
    logger.info("--- A. post-to_q ---")
    try:
        assert_quality(tt_q_BND, pyt_q.float(), pcc=0.99)
        logger.info("PASS")
    except Exception as exc:
        logger.warning(f"FAIL: {exc}")

    # ---- Stage B: to_kv ----
    k_1BNF, v_1BNF = attn.to_kv(kv_dev, compute_kernel_config=attn.mm_compute_kernel_config)
    # Each is [1, B, L, D/tp=128] TP-frac on dim 3.
    k_full = ccl_manager.all_gather_persistent_buffer(k_1BNF, dim=3, mesh_axis=tp_axis)
    v_full = ccl_manager.all_gather_persistent_buffer(v_1BNF, dim=3, mesh_axis=tp_axis)
    tt_k_BLD = local_device_to_torch(k_full).squeeze(0).float()
    tt_v_BLD = local_device_to_torch(v_full).squeeze(0).float()
    logger.info("--- B. post-to_kv (k) ---")
    try:
        assert_quality(tt_k_BLD, pyt_k.float(), pcc=0.99)
        logger.info("PASS")
    except Exception as exc:
        logger.warning(f"FAIL: {exc}")
    logger.info("--- B'. post-to_kv (v) ---")
    try:
        assert_quality(tt_v_BLD, pyt_v.float(), pcc=0.99)
        logger.info("PASS")
    except Exception as exc:
        logger.warning(f"FAIL: {exc}")

    # ---- Stage C: norm_q (with head split) ----
    q_BHNE = attn.norm_q(q_1BNF, num_heads_per_device=attn.n_local_heads)
    # Output layout: [B, n_local_heads, N_sp, head_dim]. Gather SP (dim 2) then TP-heads (dim 1).
    q_BHNE_sp = ccl_manager.all_gather_persistent_buffer(q_BHNE, dim=2, mesh_axis=sp_axis)
    q_BHNE_full = ccl_manager.all_gather_persistent_buffer(q_BHNE_sp, dim=1, mesh_axis=tp_axis)
    tt_q_BHNE = local_device_to_torch(q_BHNE_full).float()  # [B, H, N, E]
    logger.info("--- C. post-norm_q (head layout) ---")
    try:
        assert_quality(tt_q_BHNE, pyt_qh.float(), pcc=0.99)
        logger.info("PASS")
    except Exception as exc:
        logger.warning(f"FAIL: {exc}")

    # ---- Stage D: norm_k ----
    k_BHNE = attn.norm_k(k_1BNF, num_heads_per_device=attn.n_local_heads)
    k_BHNE_full = ccl_manager.all_gather_persistent_buffer(k_BHNE, dim=1, mesh_axis=tp_axis)
    tt_k_BHNE = local_device_to_torch(k_BHNE_full).float()  # [B, H, L, E]
    logger.info("--- D. post-norm_k (head layout) ---")
    try:
        assert_quality(tt_k_BHNE, pyt_kh.float(), pcc=0.99)
        logger.info("PASS")
    except Exception as exc:
        logger.warning(f"FAIL: {exc}")

    # ---- Stage E: SDPA ----
    # v needs head-split too (no norm on v): use create_heads.
    def _create_heads(inp):
        out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            inp, num_heads=attn.n_local_heads, num_kv_heads=0, transpose_k_heads=False
        )
        return out

    v_BHNE_dev = _create_heads(v_1BNF)
    spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
        q_BHNE,
        k_BHNE,
        v_BHNE_dev,
        is_causal=False,
        attn_mask=mask_dev,
        program_config=attn.sdpa_program_config,
        compute_kernel_config=attn.sdpa_compute_kernel_config,
    )
    sd_sp = ccl_manager.all_gather_persistent_buffer(spatial_BHNE, dim=2, mesh_axis=sp_axis)
    sd_full = ccl_manager.all_gather_persistent_buffer(sd_sp, dim=1, mesh_axis=tp_axis)
    tt_attn_BHNE = local_device_to_torch(sd_full).float()
    logger.info("--- E. post-SDPA (BHNE) ---")
    try:
        assert_quality(tt_attn_BHNE, pyt_attn_BHNE.float(), pcc=0.99)
        logger.info("PASS")
    except Exception as exc:
        logger.warning(f"FAIL: {exc}")

    # ---- Stage F: concat heads + to_out ----
    spatial_BND_dev = ttnn.transformer.concatenate_heads(spatial_BHNE)
    spatial_BND_dev = ttnn.unsqueeze(spatial_BND_dev, 0)
    # All-gather TP on dim 3 before to_out (matches use_nonfused_agmm path).
    spatial_BND_dev = ccl_manager.all_gather_persistent_buffer(spatial_BND_dev, dim=3, mesh_axis=tp_axis)
    out_dev = attn.to_out(spatial_BND_dev, compute_kernel_config=attn.mm_compute_kernel_config, parallel_config=None)
    out_sp = ccl_manager.all_gather_persistent_buffer(out_dev, dim=2, mesh_axis=sp_axis)
    out_full = ccl_manager.all_gather_persistent_buffer(out_sp, dim=3, mesh_axis=tp_axis)
    tt_out_BND = local_device_to_torch(out_full).squeeze(0).float()
    logger.info("--- F. post-to_out (final) ---")
    try:
        assert_quality(tt_out_BND, pyt_out.float(), pcc=0.99)
        logger.info("PASS")
    except Exception as exc:
        logger.warning(f"FAIL: {exc}")

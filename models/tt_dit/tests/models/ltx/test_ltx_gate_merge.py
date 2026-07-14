# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Equivalence guard for the gate merge: folding the per-head gate into the Q/QKV projection must
leave Q, K, V and the gate itself bit-for-bit what the standalone-gate path produces.

The fold is a silent corruptor — a mis-ordered column block yields a well-shaped tensor of the wrong
data — and no other test can see it: the block harness disables its PCC path whenever audio (hence
the gate) is on. So this compares the merged attention against the unmerged one directly, from the
same weights, rather than against a reference derived from the fold itself."""

import os

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import models.tt_dit.models.transformers.ltx.attention_ltx as attention_ltx
import ttnn
from models.tt_dit.models.transformers.ltx.attention_ltx import LTXAttention
from models.tt_dit.models.transformers.ltx.transformer_ltx import (
    LTXTransformerBlock,
    build_audio_masks,
    build_video_pad_mask,
)
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

# The block harness (shapes, rope, masks, the 22B checkpoint) already exists; reuse it rather than
# re-deriving production's exact latent grids and padding rules here.
from models.tt_dit.tests.models.ltx.test_transformer_ltx import (  # noqa: E402
    AUDIO_CTX_DIM,
    AUDIO_DIM,
    CTX_DIM,
    DIM,
    EPS,
    INPUT_SEED,
    NUM_HEADS,
    PROMPT_LEN,
    _audio_cross_pe_freqs,
    _audio_rope_freqs,
    _audio_seq_lens,
    _make_ccl_manager,
    _make_parallel_config,
    _pad_seq_dim,
    _resolve_checkpoint_22b,
    _sp_pad_len,
    _tt_rope,
    _tt_rope_full,
    _video_cross_pe_freqs,
    _video_rope_freqs,
)
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch, to_torch
from models.tt_dit.utils.test import ring_params

# (id, is_self, dim, num_heads, query_input_dim, M) — every gated projection the AV model runs, at
# both the stage-1 (M=1216) and stage-2 (M=4864) per-device video sequence lengths.
CASES = [
    ("video_self_s1", True, 4096, 32, 4096, 1216),
    ("video_self_s2", True, 4096, 32, 4096, 4864),
    ("video_cross_s1", False, 4096, 32, 4096, 1216),
    ("video_cross_s2", False, 4096, 32, 4096, 4864),
    ("a2v_q_s1", False, 2048, 32, 4096, 1216),
    ("a2v_q_s2", False, 2048, 32, 4096, 4864),
    ("audio_self", True, 2048, 32, 2048, 32),
    ("audio_cross", False, 2048, 32, 2048, 32),
]


def _build(merge, **kwargs):
    """LTXAttention with the gate merge forced on or off (it is otherwise env-scoped)."""
    original = attention_ltx.gate_merge_enabled
    attention_ltx.gate_merge_enabled = lambda: merge
    try:
        return LTXAttention(**kwargs)
    finally:
        attention_ltx.gate_merge_enabled = original


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="ring_bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("case", "is_self", "dim", "num_heads", "query_input_dim", "M"), CASES, ids=[c[0] for c in CASES]
)
def test_gate_merge_matches_standalone_gate(
    mesh_device, sp_axis, tp_axis, num_links, topology, case, is_self, dim, num_heads, query_input_dim, M
):
    torch.manual_seed(0)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    kwargs = dict(
        dim=dim,
        num_heads=num_heads,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_self=is_self,
        query_input_dim=None if is_self else query_input_dim,
        context_dim=None if is_self else dim,
        apply_gated_attention=True,
    )
    merged = _build(True, **kwargs)
    plain = _build(False, **kwargs)
    assert merged.merge_gate and not plain.merge_gate

    state = {
        "q_norm.weight": torch.randn(dim),
        "k_norm.weight": torch.randn(dim),
        "to_q.weight": torch.randn(dim, query_input_dim) * 0.05,
        "to_q.bias": torch.randn(dim) * 0.05,
        "to_k.weight": torch.randn(dim, dim) * 0.05,
        "to_k.bias": torch.randn(dim) * 0.05,
        "to_v.weight": torch.randn(dim, dim) * 0.05,
        "to_v.bias": torch.randn(dim) * 0.05,
        "to_out.0.weight": torch.randn(dim, dim) * 0.05,
        "to_out.0.bias": torch.randn(dim) * 0.05,
        "to_gate_logits.weight": torch.randn(num_heads, query_input_dim) * 0.05,
        "to_gate_logits.bias": torch.randn(num_heads) * 0.05,
    }
    merged.load_torch_state_dict({k: v.clone() for k, v in state.items()})
    plain.load_torch_state_dict({k: v.clone() for k, v in state.items()})

    x = torch.randn(1, 1, M, query_input_dim)
    x_tt = from_torch(
        x, device=mesh_device, layout=ttnn.Layout.TILE, dtype=ttnn.bfloat16, mesh_axes=[None, None, None, tp_axis]
    )

    def project(attn):
        proj = attn.to_qkv if is_self else attn.to_q
        out = proj(x_tt, compute_kernel_config=attn.mm_compute_kernel_config, parallel_config=parallel_config)
        if attn.merge_gate:
            *qkv, gate_logits = out if isinstance(out, list) else [out]
            return qkv, attn._gate_from_logits(gate_logits)
        qkv = out if isinstance(out, list) else [out]
        return qkv, attn._compute_gate(x_tt, parallel_config)

    merged_qkv, merged_gate = project(merged)
    plain_qkv, plain_gate = project(plain)

    assert len(merged_qkv) == len(plain_qkv) == merged.base_chunks
    for m, p in zip(merged_qkv, plain_qkv):
        assert_quality(
            to_torch(p, mesh_axes=[None, None, None, tp_axis]),
            to_torch(m, mesh_axes=[None, None, None, tp_axis]),
            pcc=0.9999,
            relative_rmse=0.01,
        )

    # The merged gate lands in 1BND (concatenate_heads) layout — one column per head CHANNEL — while
    # the standalone gate is (B, H, N, 1) and broadcasts over those channels. Expanding the latter is
    # exactly what the merged fold precomputes, so they must agree value for value.
    plain_1bnd = to_torch(plain_gate, mesh_axes=[None, tp_axis, None, None])  # (B, num_heads, M, 1)
    plain_1bnd = plain_1bnd.permute(3, 0, 2, 1).repeat_interleave(merged.head_dim, dim=-1)  # (1, B, M, dim)
    assert_quality(
        plain_1bnd,
        to_torch(merged_gate, mesh_axes=[None, None, None, tp_axis]),
        pcc=0.9999,
        relative_rmse=0.01,
    )


# (id, is_self, dim, num_heads, N) — the projection test above proves Q/K/V/gate come out of the
# merged weight correctly; this covers what it cannot: the whole forward, i.e. that the gate chunk
# still means the same thing once SDPA has run and `concatenate_heads` has laid the heads back out.
FORWARD_CASES = [
    ("video_self", True, 4096, 32, 9728),
    ("video_cross", False, 4096, 32, 9728),
    ("audio_self", True, 2048, 32, 256),
    ("audio_cross", False, 2048, 32, 256),
]


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="ring_bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("case", "is_self", "dim", "num_heads", "N"), FORWARD_CASES, ids=[c[0] for c in FORWARD_CASES])
def test_gate_merge_forward_matches_standalone_gate(
    mesh_device, sp_axis, tp_axis, num_links, topology, case, is_self, dim, num_heads, N
):
    """Full LTXAttention.forward(), merged gate vs standalone gate, from identical weights."""
    torch.manual_seed(0)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    kwargs = dict(
        dim=dim,
        num_heads=num_heads,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_self=is_self,
        context_dim=None if is_self else dim,
        apply_gated_attention=True,
    )
    merged = _build(True, **kwargs)
    plain = _build(False, **kwargs)
    assert merged.merge_gate and not plain.merge_gate

    state = {
        "q_norm.weight": torch.randn(dim),
        "k_norm.weight": torch.randn(dim),
        "to_q.weight": torch.randn(dim, dim) * 0.05,
        "to_q.bias": torch.randn(dim) * 0.05,
        "to_k.weight": torch.randn(dim, dim) * 0.05,
        "to_k.bias": torch.randn(dim) * 0.05,
        "to_v.weight": torch.randn(dim, dim) * 0.05,
        "to_v.bias": torch.randn(dim) * 0.05,
        "to_out.0.weight": torch.randn(dim, dim) * 0.05,
        "to_out.0.bias": torch.randn(dim) * 0.05,
        "to_gate_logits.weight": torch.randn(num_heads, dim) * 0.05,
        "to_gate_logits.bias": torch.randn(num_heads) * 0.05,
    }
    merged.load_torch_state_dict({k: v.clone() for k, v in state.items()})
    plain.load_torch_state_dict({k: v.clone() for k, v in state.items()})

    # spatial is SP- and TP-fractured, exactly as the block feeds it.
    spatial = torch.randn(1, 1, N, dim)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})

    fwd = dict(N=N)
    if not is_self:
        # Text cross-attn: replicated prompt over SP, TP-fractured on the feature dim (kv_replicated).
        prompt = torch.randn(1, 1, 32, dim)
        fwd["prompt_1BLP"] = from_torch(
            prompt,
            device=mesh_device,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.bfloat16,
            mesh_axes=[None, None, None, tp_axis],
        )
        fwd["kv_replicated"] = True

    out_m = merged(spatial_1BND=tt_spatial, **fwd)
    out_p = plain(spatial_1BND=tt_spatial, **fwd)

    concat = [None, None]
    concat[sp_axis] = 2
    concat[tp_axis] = 3
    to_t = lambda t: ttnn.to_torch(  # noqa: E731
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat, mesh_shape=tuple(mesh_device.shape))
    )
    assert_quality(to_t(out_p), to_t(out_m), pcc=0.999, relative_rmse=0.02)


# ---------------------------------------------------------------------------
# Block-level guard
# ---------------------------------------------------------------------------
# The tests above prove ONE LTXAttention is equivalent merged vs standalone. That is not the
# claim the pipeline needs. A gated block runs SIX attentions, and two of them — audio_to_video
# (output_dim != dim) and video_to_audio (whose K/V stay SP-sharded, so it takes the ring-cross
# SDPA branch) — have a forward no attention test covers, only a projection test. The block also
# adds what an isolated attention cannot: rope, the audio/video padding masks, and the fused
# to_out addcmul that folds the residual into the projection epilogue.
#
# The pipeline decodes to noise while the tests above pass, so the bug lives in exactly that gap.
# This closes it: one real block, real block-0 weights, merged vs standalone, same inputs.
#
# `merge_sel` selects WHICH attentions merge, so the same test bisects: the block builds its
# attentions in a fixed order, and gate_merge_enabled() is consulted once per attention.
_ATTN_ORDER = ("attn1", "attn2", "audio_attn1", "audio_attn2", "a2v", "v2a")


def _build_block(merge: set[str], **kwargs):
    """LTXTransformerBlock with the gate merge forced on for the named attentions only."""
    pending = iter(_ATTN_ORDER)
    original = attention_ltx.gate_merge_enabled
    attention_ltx.gate_merge_enabled = lambda: next(pending) in merge
    try:
        block = LTXTransformerBlock(**kwargs)
    finally:
        attention_ltx.gate_merge_enabled = original
    assert next(pending, None) is None, "block built fewer attentions than _ATTN_ORDER"
    return block


def _block0_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Block 0's weights, read straight out of the checkpoint (per-tensor, not the whole 45GB file)."""
    prefix = "model.diffusion_model.transformer_blocks.0."
    with safe_open(checkpoint_path, framework="pt") as f:
        return {k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="ring_bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("F", "H", "W"), [pytest.param(19, 17, 30, id="stage_1"), pytest.param(19, 34, 60, id="stage_2")]
)
@pytest.mark.parametrize(
    "merge_sel",
    [pytest.param(set(_ATTN_ORDER), id="all"), *[pytest.param({n}, id=n) for n in _ATTN_ORDER]],
)
def test_gate_merge_block_matches_standalone_gate(
    mesh_device, sp_axis, tp_axis, num_links, topology, F, H, W, merge_sel, reset_seeds
):
    """A whole gated LTXTransformerBlock, merged gate vs standalone gate, from identical weights."""
    checkpoint = _resolve_checkpoint_22b("fast")
    if not os.path.exists(checkpoint):
        pytest.skip(f"22B checkpoint not found at {checkpoint}")

    sp_factor = tuple(mesh_device.shape)[sp_axis]
    video_N_real = F * H * W
    video_N = _sp_pad_len(video_N_real, sp_factor)
    audio_N, audio_N_real = _audio_seq_lens(F, sp_factor)

    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    block_kwargs = dict(
        video_dim=DIM,
        video_ffn_dim=DIM * 4,
        video_num_heads=NUM_HEADS,
        video_cross_attention_dim=CTX_DIM,
        audio_dim=AUDIO_DIM,
        audio_ffn_dim=AUDIO_DIM * 4,
        audio_num_heads=NUM_HEADS,
        audio_cross_attention_dim=AUDIO_CTX_DIM,
        eps=EPS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=False,  # production bh_4x8sp1tp0_ring
        has_audio=True,
        apply_gated_attention=True,
        cross_attention_adaln=True,
    )
    merged = _build_block(set(merge_sel), **block_kwargs)
    plain = _build_block(set(), **block_kwargs)
    assert [a.merge_gate for a in (merged.attn1, merged.attn2)] != [False, False] or merge_sel.isdisjoint(
        {"attn1", "attn2"}
    )
    assert not any(
        a.merge_gate
        for a in (
            plain.attn1,
            plain.attn2,
            plain.audio_attn1,
            plain.audio_attn2,
            plain.audio_to_video_attn,
            plain.video_to_audio_attn,
        )
    )

    state = _block0_state_dict(checkpoint)
    merged.load_torch_state_dict({k: v.clone() for k, v in state.items()}, strict=False)
    plain.load_torch_state_dict({k: v.clone() for k, v in state.items()}, strict=False)

    # Inputs: identical for both blocks, built exactly as test_ltx_transformer_block's AV path.
    torch.manual_seed(INPUT_SEED)
    x = torch.randn(1, video_N_real, DIM, dtype=torch.float32)
    context = torch.randn(1, PROMPT_LEN, CTX_DIM, dtype=torch.float32)
    temb = torch.randn(1, 1, 9 * DIM, dtype=torch.float32)
    prompt_temb = torch.randn(1, 1, 2 * DIM, dtype=torch.float32)

    a_x = torch.zeros(1, audio_N, AUDIO_DIM, dtype=torch.float32)
    a_x[:, :audio_N_real, :] = torch.randn(1, audio_N_real, AUDIO_DIM, dtype=torch.float32)
    a_ctx = torch.randn(1, PROMPT_LEN, AUDIO_CTX_DIM, dtype=torch.float32)
    a_temb = torch.randn(1, 1, 9 * AUDIO_DIM, dtype=torch.float32)
    a_prompt_temb = torch.randn(1, 1, 2 * AUDIO_DIM, dtype=torch.float32)
    av_ca_v = torch.randn(1, 1, 5 * DIM, dtype=torch.float32)
    av_ca_a = torch.randn(1, 1, 5 * AUDIO_DIM, dtype=torch.float32)

    a_cos, a_sin = _tt_rope(_audio_rope_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
    vx_cos, vx_sin = _tt_rope(
        _video_cross_pe_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, pad_to=video_N
    )
    ax_cos, ax_sin = _tt_rope(_audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
    ax_cos_full, ax_sin_full = _tt_rope_full(_audio_cross_pe_freqs, audio_N, mesh_device=mesh_device, tp_axis=tp_axis)
    a_attn_mask, a_pad_sp, a_pad_full = build_audio_masks(
        audio_N, audio_N_real, mesh_device=mesh_device, sp_axis=sp_axis
    )
    v_pad_sp = build_video_pad_mask(video_N, video_N_real, mesh_device=mesh_device, sp_axis=sp_axis)

    def forward_kwargs():
        # Rebuilt per block: the block consumes its activation inputs, and both must see the same values.
        return dict(
            video_1BND=bf16_tensor_2dshard(
                _pad_seq_dim(x, video_N, dim=1).unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
            ),
            video_prompt=bf16_tensor(context.unsqueeze(0), device=mesh_device),
            video_temb=bf16_tensor(
                temb.reshape(9, DIM).unsqueeze(1).unsqueeze(1), device=mesh_device, mesh_axis=tp_axis, shard_dim=3
            ),
            video_N=video_N_real,
            video_rope_cos=v_cos,
            video_rope_sin=v_sin,
            trans_mat=tt_trans_mat,
            video_prompt_temb=bf16_tensor(prompt_temb.reshape(2, DIM).unsqueeze(1).unsqueeze(1), device=mesh_device),
            audio_1BND=bf16_tensor_2dshard(
                a_x.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
            ),
            audio_prompt=bf16_tensor(a_ctx.unsqueeze(0), device=mesh_device),
            audio_temb=bf16_tensor(
                a_temb.reshape(9, AUDIO_DIM).unsqueeze(1).unsqueeze(1),
                device=mesh_device,
                mesh_axis=tp_axis,
                shard_dim=3,
            ),
            audio_prompt_temb=bf16_tensor(
                a_prompt_temb.reshape(2, AUDIO_DIM).unsqueeze(1).unsqueeze(1), device=mesh_device
            ),
            av_ca_temb=bf16_tensor(
                av_ca_v.reshape(5, DIM).unsqueeze(1).unsqueeze(1), device=mesh_device, mesh_axis=tp_axis, shard_dim=3
            ),
            av_ca_audio_temb=bf16_tensor(
                av_ca_a.reshape(5, AUDIO_DIM).unsqueeze(1).unsqueeze(1),
                device=mesh_device,
                mesh_axis=tp_axis,
                shard_dim=3,
            ),
            audio_N=audio_N,
            audio_rope_cos=a_cos,
            audio_rope_sin=a_sin,
            video_cross_pe_cos=vx_cos,
            video_cross_pe_sin=vx_sin,
            audio_cross_pe_cos=ax_cos,
            audio_cross_pe_sin=ax_sin,
            audio_cross_pe_cos_full=ax_cos_full,
            audio_cross_pe_sin_full=ax_sin_full,
            audio_attn_mask=a_attn_mask,
            audio_padding_mask=a_pad_sp,
            audio_padding_mask_full=a_pad_full,
            video_padding_mask=v_pad_sp,
        )

    v_cos, v_sin = _tt_rope(
        _video_rope_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, pad_to=video_N
    )
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    concat = [None, None]
    concat[sp_axis] = 2
    concat[tp_axis] = 3
    to_t = lambda t: ttnn.to_torch(  # noqa: E731
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat, mesh_shape=tuple(mesh_device.shape))
    ).squeeze(0)

    def run(block, tag):
        """Forward, sync, and gather to host before anything else touches the device."""
        v, a = block(**forward_kwargs())
        ttnn.synchronize_device(mesh_device)
        v_t = to_t(v)[:, :video_N_real, :]
        a_t = to_t(a)[:, :audio_N_real, :]
        for name, t in ((f"{tag} video", v_t), (f"{tag} audio", a_t)):
            fin = torch.isfinite(t)
            logger.info(
                f"{name}: finite {fin.sum().item()}/{t.numel()}"
                + (f" range=[{t[fin].min().item():.3g}, {t[fin].max().item():.3g}]" if fin.any() else " ALL non-finite")
            )
        return v_t, a_t

    # The standalone block runs twice, before and after the merged one. That is the harness's own
    # control: it pins down that a block's output does not depend on what ran before it on this mesh,
    # so a difference in the middle run is the merge and not the position.
    v_p1, a_p1 = run(plain, "plain#1")
    v_m, a_m = run(merged, "merged")
    v_p2, a_p2 = run(plain, "plain#2")

    assert torch.isfinite(v_p1).all() and torch.isfinite(a_p1).all(), "standalone output not finite — harness bug"
    assert_quality(v_p1, v_p2, pcc=0.9999, relative_rmse=0.01)  # control: position does not matter
    assert_quality(a_p1, a_p2, pcc=0.9999, relative_rmse=0.01)

    assert_quality(v_p1, v_m, pcc=0.99, relative_rmse=0.05)
    assert_quality(a_p1, a_m, pcc=0.99, relative_rmse=0.05)

    # assert_quality cannot answer the question this test exists to answer. PCC is scale-invariant,
    # so a merged block that is uniformly 1.005x the standalone one scores a perfect PCC; RMSE/sigma
    # is printed to one decimal place and asserted at 5%. Both would call that "identical", and the
    # pipeline would then compound it over 48 blocks x 11 steps. So report the error in terms that
    # can actually see it, with the run-to-run control (plain#1 vs plain#2, which must be exactly 0)
    # as the floor. merge_sel attributes any error to the individual attention that causes it.
    sel = ",".join(sorted(merge_sel))
    logger.info(f"===== BLOCK ERROR  merge_sel={{{sel}}}  stage={F}x{H}x{W} =====")
    for tag, p1, p2, m in (("video", v_p1, v_p2, v_m), ("audio", a_p1, a_p2, a_m)):
        logger.info("  " + _fmt(f"{tag} control plain#1 vs plain#2", _err_stats(p1, p2)))
        logger.info("  " + _fmt(f"{tag} MERGED vs standalone", _err_stats(p1, m)))


# ---------------------------------------------------------------------------
# Full-precision gate-logit diagnostic
# ---------------------------------------------------------------------------
# Every test above gates on assert_quality, which reports PCC and RMSE/sigma. Neither can see the
# error that matters here:
#
#   * PCC IS SCALE- AND SHIFT-INVARIANT. A gate that is systematically 1.005x the standalone gate
#     scores PCC = 1.0000 exactly, forever. The block output is then 1.005x too — and the pipeline
#     applies the gate SIX times per block, 48 blocks, 11 steps. A per-application gain of 1+e
#     compounds; PCC never sees it.
#   * RMSE/sigma is logged at ONE decimal place as a percent, so anything under 0.05% prints "0.0 %",
#     and it is asserted at 5% (block) / 1% (projection) — bounds a systematic gain sails through.
#
# So "the merged block matches to 4 decimals" is not evidence of equivalence. This measures the gate
# LOGITS themselves (pre-sigmoid, the merge's actual output) against the standalone gate's logits,
# on REAL block-0 weights, and reports max-abs / max-rel / RMS-rel error AND the least-squares GAIN
# (<m,p>/<p,p>; 1.0 means no systematic scaling). It also bisects math_approx_mode in the same pass:
# the standalone gate rides ColParallelLinear.compute_config (approx=False) while the merged gate
# rides the projection's mm_compute_kernel_config (approx=True), and that is the only config that
# differs between the arms.
#
# Run: pytest test_ltx_gate_merge.py -k gate_logit_precision -s

# (checkpoint name, is_self, dim, query_input_dim, context_dim, output_dim, M) — the six real gated
# attentions of block 0, at their production per-device sequence lengths (stage 2).
_REAL_ATTNS = [
    ("attn1", True, 4096, None, None, None, 4864),
    ("attn2", False, 4096, None, 4096, None, 4864),
    ("audio_attn1", True, 2048, None, None, None, 32),
    ("audio_attn2", False, 2048, None, 2048, None, 32),
    ("audio_to_video_attn", False, 2048, 4096, 2048, 4096, 4864),
    ("video_to_audio_attn", False, 2048, None, 4096, None, 32),
]


def _err_stats(p: torch.Tensor, m: torch.Tensor) -> dict:
    """Error of m (merged) against p (standalone), in the terms PCC cannot express."""
    p = p.detach().to(torch.float64).flatten()
    m = m.detach().to(torch.float64).flatten()
    d = m - p
    denom = p.abs().clamp_min(1e-9)
    return {
        "max_abs": d.abs().max().item(),
        "max_rel": (d.abs() / denom).max().item(),
        "rms_rel": (d.norm() / p.norm().clamp_min(1e-12)).item(),
        # Least-squares slope of m on p. 1.0 == no systematic gain. This is the number PCC hides.
        "gain": ((m * p).sum() / (p * p).sum().clamp_min(1e-12)).item(),
        "bias": (m.mean() - p.mean()).item(),
        "p_absmax": p.abs().max().item(),
    }


def _fmt(tag: str, s: dict) -> str:
    return (
        f"{tag:34s} max_abs={s['max_abs']:.3e}  max_rel={s['max_rel']:.3e}  "
        f"rms_rel={s['rms_rel']:.3e}  GAIN={s['gain']:.9f}  bias={s['bias']:+.3e}"
    )


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="ring_bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
def test_gate_logit_precision(mesh_device, sp_axis, tp_axis, num_links, topology):
    """Merged gate logits vs standalone gate logits, real weights, at full precision."""
    checkpoint = _resolve_checkpoint_22b("fast")
    if not os.path.exists(checkpoint):
        pytest.skip(f"22B checkpoint not found at {checkpoint}")

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    block0 = _block0_state_dict(checkpoint)

    failures = []
    for name, is_self, dim, q_in, ctx, out_dim, M in _REAL_ATTNS:
        torch.manual_seed(0)
        sub = {k[len(name) + 1 :]: v for k, v in block0.items() if k.startswith(name + ".")}
        assert "to_gate_logits.weight" in sub, f"{name} has no gate in the checkpoint"

        kwargs = dict(
            dim=dim,
            num_heads=NUM_HEADS,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_self=is_self,
            context_dim=ctx,
            query_input_dim=q_in,
            output_dim=out_dim,
            apply_gated_attention=True,
        )
        merged = _build(True, **kwargs)
        plain = _build(False, **kwargs)
        merged.load_torch_state_dict({k: v.clone() for k, v in sub.items()}, strict=False)
        plain.load_torch_state_dict({k: v.clone() for k, v in sub.items()}, strict=False)

        in_dim = q_in or dim
        x = torch.randn(1, 1, M, in_dim)
        x_tt = from_torch(
            x, device=mesh_device, layout=ttnn.Layout.TILE, dtype=ttnn.bfloat16, mesh_axes=[None, None, None, tp_axis]
        )

        proj_m = merged.to_qkv if is_self else merged.to_q
        head_dim = dim // NUM_HEADS

        # Merged gate, as production runs it: the projection's config (math_approx_mode=True).
        out_approx = proj_m(
            x_tt, compute_kernel_config=merged.mm_compute_kernel_config, parallel_config=parallel_config
        )
        # Merged gate, forced onto the STANDALONE gate's config (math_approx_mode=False). Same weights,
        # same blocking, same op — this isolates math_approx_mode and nothing else.
        out_noapprox = proj_m(x_tt, compute_kernel_config=proj_m.compute_config, parallel_config=parallel_config)
        # Standalone gate: its own ColParallelLinear (math_approx_mode=False), raw logits, no sigmoid.
        p_logits = plain.to_gate_logits(x_tt, parallel_config=parallel_config)

        gather_w = lambda t: to_torch(t, mesh_axes=[None, None, None, tp_axis])  # noqa: E731
        p_l = gather_w(p_logits)  # (1, 1, M, num_heads)
        m_l_approx = gather_w(out_approx[-1])  # (1, 1, M, dim) — gate repeated over head_dim
        m_l_noapprox = gather_w(out_noapprox[-1])

        # The merged chunk must repeat each head's logit across that head's head_dim channels.
        spread = (
            (
                m_l_approx.reshape(*m_l_approx.shape[:-1], NUM_HEADS, head_dim)
                - m_l_approx[..., ::head_dim].unsqueeze(-1)
            )
            .abs()
            .max()
            .item()
        )
        m_ph_approx = m_l_approx[..., ::head_dim]  # (1, 1, M, num_heads)
        m_ph_noapprox = m_l_noapprox[..., ::head_dim]

        s_approx = _err_stats(p_l, m_ph_approx)
        s_noapprox = _err_stats(p_l, m_ph_noapprox)
        s_approx_vs_noapprox = _err_stats(m_ph_noapprox, m_ph_approx)

        logger.info(f"===== {name}  (dim={dim}, in={in_dim}, M={M}, head_dim={head_dim}) =====")
        logger.info(f"  gate chunk intra-head spread (must be 0): {spread:.3e}")
        logger.info("  " + _fmt("LOGITS merged(approx=T) vs standalone", s_approx))
        logger.info("  " + _fmt("LOGITS merged(approx=F) vs standalone", s_noapprox))
        logger.info("  " + _fmt("LOGITS approx=T vs approx=F", s_approx_vs_noapprox))

        # 2*sigmoid epilogue, the value that actually multiplies the attention output.
        gate_p = ttnn.multiply(ttnn.sigmoid(p_logits), 2.0)
        gate_m = merged._gate_from_logits(out_approx[-1])
        gp = gather_w(gate_p)
        gm = gather_w(gate_m)[..., ::head_dim]
        s_gate = _err_stats(gp, gm)
        logger.info("  " + _fmt("GATE 2*sigmoid merged vs standalone", s_gate))

        # Q/K/V must be untouched by the merge (modulo the to_qkv reblocking).
        for i, qkv_name in enumerate(("q", "k", "v")[: merged.base_chunks]):
            pq = plain.to_qkv if is_self else plain.to_q
            p_out = pq(x_tt, compute_kernel_config=plain.mm_compute_kernel_config, parallel_config=parallel_config)
            p_chunk = (p_out if isinstance(p_out, list) else [p_out])[i]
            s_qkv = _err_stats(gather_w(p_chunk), gather_w(out_approx[i]))
            logger.info("  " + _fmt(f"{qkv_name.upper()} merged vs standalone", s_qkv))

        if spread > 0:
            failures.append(f"{name}: gate chunk is not a clean per-head repeat (spread={spread:.3e})")

    logger.info("\n".join(["", "=" * 90, "GATE-LOGIT PRECISION SUMMARY complete", "=" * 90]))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="ring_bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
def test_merge_is_not_the_reblock(mesh_device, sp_axis, tp_axis, num_links, topology):
    """Is the merged Q/K/V what a RE-BLOCKED standalone to_qkv produces, or something else?

    The merge moves attn1's stage-2 to_qkv from (4864,4096,3072)->(10,4,12) to
    (4864,4096,4096)->(5,8,16). That was assumed to be a pure re-blocking — same math, different
    fp reduction order — and TT_DIT_MM_REBLOCK was built to price exactly that: it forces the
    STANDALONE to_qkv onto (5,8,16) without changing a single weight. End to end that control costs
    frame-PCC 0.9978, which was then adopted as the precision noise floor.

    But the merged pipeline scores 0.868 against the baseline AND 0.868 against that very control.
    The control lands next to the baseline; the merge does not land next to the control. So if the
    merge were only a re-blocking, the re-blocked standalone arm would reproduce the merged Q/K/V
    exactly. This measures whether it does:

        P = standalone, stock blocking      (10,4,12)
        R = standalone, forced (5,8,16)     — the control: same weights, merged arm's blocking
        M = merged                          (5,8,16) natively, N=4096, chunks=4

    R vs M is the whole question. Bit-identical => the merge IS the re-blocking and the e2e
    difference lives somewhere else entirely. Different => the merged matmul is NOT just re-blocked,
    the "noise floor" was calibrated against a perturbation the merge does not make, and the number
    to explain is how P->M differs from P->R.
    """
    checkpoint = _resolve_checkpoint_22b("fast")
    if not os.path.exists(checkpoint):
        pytest.skip(f"22B checkpoint not found at {checkpoint}")

    M, dim = 4864, 4096  # attn1, stage 2, per-device video sequence
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    block0 = _block0_state_dict(checkpoint)
    sub = {k[len("attn1.") :]: v for k, v in block0.items() if k.startswith("attn1.")}

    kwargs = dict(
        dim=dim,
        num_heads=NUM_HEADS,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_self=True,
        apply_gated_attention=True,
    )
    torch.manual_seed(0)
    merged = _build(True, **kwargs)
    plain = _build(False, **kwargs)
    merged.load_torch_state_dict({k: v.clone() for k, v in sub.items()}, strict=False)
    plain.load_torch_state_dict({k: v.clone() for k, v in sub.items()}, strict=False)

    x = torch.randn(1, 1, M, dim)
    x_tt = from_torch(
        x, device=mesh_device, layout=ttnn.Layout.TILE, dtype=ttnn.bfloat16, mesh_axes=[None, None, None, tp_axis]
    )
    gather_w = lambda t: to_torch(t, mesh_axes=[None, None, None, tp_axis])  # noqa: E731

    def project(attn):
        out = attn.to_qkv(x_tt, compute_kernel_config=attn.mm_compute_kernel_config, parallel_config=parallel_config)
        return [gather_w(t) for t in out[:3]]  # q, k, v — drop the merged gate chunk

    # The merged arm's blocking, forced onto the standalone weights. Same math, same weights.
    REBLOCK = "4864,4096,3072=5,8,16,1,4"

    P = project(plain)
    os.environ["TT_DIT_MM_REBLOCK"] = REBLOCK
    try:
        R = project(plain)
    finally:
        os.environ.pop("TT_DIT_MM_REBLOCK", None)
    Mg = project(merged)

    logger.info("===== attn1 to_qkv: merge vs re-block (stage 2, real block-0 weights) =====")
    for i, nm in enumerate("QKV"):
        logger.info("  " + _fmt(f"{nm}  P->R  (reblock control)", _err_stats(P[i], R[i])))
        logger.info("  " + _fmt(f"{nm}  P->M  (the merge)", _err_stats(P[i], Mg[i])))
        logger.info("  " + _fmt(f"{nm}  R->M  (is the merge the reblock?)", _err_stats(R[i], Mg[i])))

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recurrent / chunked DeltaNet forward (the `forward` dispatch).

Behavior-preserving extraction of the original `Qwen36GatedDeltaNet.forward` body.
Operates on the gdn instance: reads weights from `gdn.weights`, config dims from
`gdn.cfg`, mirrored scalar attrs + runtime state from `gdn`. Every ttnn op,
memory_config, and the `gated_deltanet_forward_ttnn` kwargs are verbatim.
"""
import models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet as _gdn_mod
import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import gated_deltanet_forward_ttnn


class _DecodeOProjTtnn:
    """ttnn shim for the decode path only: injects a swept program config into the o_proj
    ttnn.linear inside gated_deltanet_forward_ttnn (that module is outside the editable tree)."""

    def __init__(self, o_w, gdn=None):
        self._o_w = o_w
        self._gdn = gdn

    def __getattr__(self, name):
        return getattr(ttnn, name)

    def deallocate(self, t, *args, **kwargs):
        from models.demos.blackhole.bonsai2.tt.ternary.gdn import MegaSplit

        if isinstance(t, MegaSplit):
            return None
        return ttnn.deallocate(t, *args, **kwargs)

    def linear(self, a, b, *args, **kwargs):
        ternary = getattr(self._gdn, "ternary", None)
        if ternary is not None:
            # One-token decode in Bonsai's own ternary weights (models/demos/blackhole/bonsai2/tt/ternary/gdn.py).
            w = self._gdn.weights
            if b is w.mega_fused_weight:
                from models.demos.blackhole.bonsai2.tt.ternary.gdn import MegaSplit

                parts = ternary.in_proj(a, kwargs.get("compute_kernel_config"))
                return MegaSplit(parts, [w.mega_qkv_dim, w.mega_a_dim, w.mega_b_dim, w.mega_g_dim])
            if b is self._o_w:
                return ternary.out_proj(a)
        if b is self._o_w and kwargs.get("program_config") is None:
            shp = list(a.shape)
            m = 1
            for d in shp[:-1]:
                m *= d
            if m <= 32:
                kwargs["program_config"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(13, 10),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    out_block_h=1,
                    out_block_w=2,
                    per_core_M=1,
                    per_core_N=2,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
        elif kwargs.get("program_config") is None and list(b.shape)[-2:] == [5120, 16480]:
            # decode in_proj (qkvz+ab) 32x5120x16480: explicit 1D config with fuse_batch=True (swept on BH)
            shp = list(a.shape)
            m = 1
            for d in shp[:-1]:
                m *= d
            if m <= 32:
                kwargs["program_config"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(13, 10),
                    in0_block_w=2,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    out_block_h=1,
                    out_block_w=4,
                    per_core_M=1,
                    per_core_N=4,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
        return ttnn.linear(a, b, *args, **kwargs)


class _PrefillInProjTtnn:
    """ttnn shim for the prefill (mode='chunk') path only: the 2048-token in_proj
    (mega fused qkvz+ab, 2048x5120x16480) gets the auto 2D 13x10 geometry with
    in0_block_w=4 instead of the auto in0_block_w=1 (160 K-steps), as mlp.py gate/up."""

    def __getattr__(self, name):
        return getattr(ttnn, name)

    def linear(self, a, b, *args, **kwargs):
        if kwargs.get("program_config") is None and list(b.shape)[-2:] == [5120, 16480]:
            shp = list(a.shape)
            m = 1
            for d in shp[:-1]:
                m *= d
            if m == 2048:
                kwargs["program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(13, 10),
                    in0_block_w=8,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    out_block_h=7,
                    out_block_w=20,
                    per_core_M=7,
                    per_core_N=40,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                )
                kwargs["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=True,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                    dst_full_sync_en=False,
                    throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
                )
        elif kwargs.get("program_config") is None and list(b.shape)[-2:] == [6144, 5120]:
            shp = list(a.shape)
            m = 1
            for d in shp[:-1]:
                m *= d
            if m == 2048:
                kwargs["program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(13, 10),
                    in0_block_w=12,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    out_block_h=7,
                    out_block_w=13,
                    per_core_M=7,
                    per_core_N=13,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                )
                if kwargs.get("memory_config") is None:
                    # o_proj output block-sharded in L1 on 8x8 (sweep: 842.6 vs 951.1 us/call).
                    kwargs["memory_config"] = ttnn.create_sharded_memory_config(
                        shape=[1, 2048, 5120],
                        core_grid=ttnn.CoreGrid(x=8, y=8),
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    )
        return ttnn.linear(a, b, *args, **kwargs)


def recurrent_forward(gdn, x, mode="recurrent", chunk_size=None, valid_len=None):
    if mode == "chunk":
        _orig_ttnn = _gdn_mod.ttnn
        _gdn_mod.ttnn = _PrefillInProjTtnn()
        try:
            return _recurrent_forward_impl(gdn, x, mode=mode, chunk_size=chunk_size, valid_len=valid_len)
        finally:
            _gdn_mod.ttnn = _orig_ttnn
    if mode == "recurrent" and getattr(getattr(gdn, "ternary", None), "step", None) is not None:
        # Bonsai: the whole one-token layer in its own ops (models/demos/blackhole/bonsai2/tt/ternary/gdn.py).
        return gdn.ternary.decode(x, gdn, gdn.compute_kernel_config_decode)
    if mode == "recurrent":
        _orig_ttnn = _gdn_mod.ttnn
        _gdn_mod.ttnn = _DecodeOProjTtnn(gdn.weights.o_proj_weight, gdn)
        try:
            return _recurrent_forward_impl(gdn, x, mode=mode, chunk_size=chunk_size, valid_len=valid_len)
        finally:
            _gdn_mod.ttnn = _orig_ttnn
    return _recurrent_forward_impl(gdn, x, mode=mode, chunk_size=chunk_size, valid_len=valid_len)


def _recurrent_forward_impl(gdn, x, mode="recurrent", chunk_size=None, valid_len=None):
    """Non-kernel GDN forward. mode='chunk' (prefill, may delegate to the prefill kernel) or
    'recurrent' (single-token decode). Reads weights/state/dims off the gdn instance; updates
    gdn's recurrent + conv state in place.

    valid_len: for fixed-bucket masked prefill — x is right-padded to T but only the first
    valid_len positions are real (see gated_deltanet_forward_ttnn). None = no padding."""
    w = gdn.weights
    if chunk_size is None:
        chunk_size = gdn.long_prefill_chunk_size if mode == "chunk" else 64

    # Chunk-parallel prefill via the C++ gated_delta_attn_seq kernel (float32, chunk_size=128).
    seq_masks = w.chunk_seq_masks_long

    output, new_state, new_conv_q, new_conv_k, new_conv_v, new_fused_conv = gated_deltanet_forward_ttnn(
        hidden_states=x,
        q_proj_weight=w.q_proj_weight,
        k_proj_weight=w.k_proj_weight,
        v_proj_weight=w.v_proj_weight,
        a_proj_weight=w.a_proj_weight,
        b_proj_weight=w.b_proj_weight,
        o_proj_weight=w.o_proj_weight,
        q_conv_weight=w.q_conv_weight,
        k_conv_weight=w.k_conv_weight,
        v_conv_weight=w.v_conv_weight,
        q_conv_bias=w.q_conv_bias,
        k_conv_bias=w.k_conv_bias,
        v_conv_bias=w.v_conv_bias,
        A_log=w.A_log,
        dt_bias=w.dt_bias,
        o_norm_weight=w.o_norm_weight,
        g_proj_weight=w.g_proj_weight,
        num_heads=gdn.num_heads,
        num_v_heads=gdn.num_v_heads,
        head_k_dim=gdn.head_k_dim,
        head_v_dim=gdn.head_v_dim,
        conv_kernel_size=gdn.conv_kernel_size,
        use_gate=True,
        norm_eps=gdn.norm_eps,
        device=gdn.device,
        recurrent_state=gdn.recurrent_state,
        mode=mode,
        chunk_size=chunk_size,
        q_weight_taps=w.q_weight_taps,
        k_weight_taps=w.k_weight_taps,
        v_weight_taps=w.v_weight_taps,
        q_bias_dev=w.q_bias_dev,
        k_bias_dev=w.k_bias_dev,
        v_bias_dev=w.v_bias_dev,
        qkv_proj_weight=w.qkv_proj_weight,
        q_dim=gdn.cfg.q_dim,
        k_dim=gdn.cfg.k_dim,
        compute_kernel_config=gdn.compute_kernel_config_decode if mode == "recurrent" else gdn.compute_kernel_config,
        A_neg_precomputed=w.A_neg,
        fused_conv_weight_taps=w.fused_conv_weight_taps,
        fused_conv_bias_dev=w.fused_conv_bias_dev,
        fused_conv_state=gdn.fused_conv_state,
        ab_proj_weight=w.ab_proj_weight,
        mega_fused_weight=w.mega_fused_weight,
        mega_qkv_dim=w.mega_qkv_dim,
        mega_a_dim=w.mega_a_dim,
        mega_b_dim=w.mega_b_dim,
        mega_g_dim=w.mega_g_dim,
        use_inplace_state=True,
        chunk_seq_masks=seq_masks,
        valid_len=valid_len,
    )

    if new_state is not gdn.recurrent_state:
        if list(new_state.shape) != list(gdn.recurrent_state.shape):
            new_state = ttnn.reshape(new_state, list(gdn.recurrent_state.shape))
        ttnn.copy(new_state, gdn.recurrent_state)
        ttnn.deallocate(new_state)
    if new_fused_conv is not gdn.fused_conv_state:
        if new_fused_conv.layout != ttnn.TILE_LAYOUT:
            new_fused_conv = ttnn.to_layout(new_fused_conv, ttnn.TILE_LAYOUT)
        ttnn.copy(new_fused_conv, gdn.fused_conv_state)
        ttnn.deallocate(new_fused_conv)
    return output

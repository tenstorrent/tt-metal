# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

from ttnn.device import is_blackhole as ttnn_is_blackhole

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight

# SDPA chunk selection constants
_SDPA_Q_CHUNK_MAIN = 128
_SDPA_K_CANDIDATES_MAIN = (256, 128)
_SDPA_Q_CHUNKS_FLEX = (256, 128, 64, 32)
_SDPA_K_CHUNKS_FLEX = (256, 128, 64, 32)
# B1/S512 sweep (k_chunk x grid) showed q=32, k=512, grid=8x8 is the winner:
# 38.7 us/call vs prod (q=128, k=128, grid=11x10) ~43 us/call -- 10% faster.
_SDPA_B1S512_Q_CHUNK = 64
_SDPA_B1S512_K_CHUNK = 512

# Data-parallel query head-fold factor: fold G query-chunks into the head dim so
# SDPA sees Sq/G queries per head (higher throughput), K/V unchanged via GQA
# head-broadcast. G=2 is optimal (G=4 plateaus, adds heads).
_DP_HEAD_FOLD = 2
_MAX_QKV_MM_CHUNK_SEQ_LEN = 8192
_MAX_WO_MM_CHUNK_SEQ_LEN = 8192


@dataclass
class BgeM3AttentionConfig:
    # Required weights
    wqkv: LazyWeight
    wo_weight: LazyWeight

    # Model dimensions
    hidden_size: int
    num_heads: int
    head_dim: int

    # Optional weights
    bqkv: LazyWeight | None = None
    wo_bias: LazyWeight | None = None
    mesh_device: ttnn.MeshDevice | None = None

    # Attention
    attention_scale: float | None = None

    # Runtime-resolved dtype and memory fields
    qkv_dtype: ttnn.DataType | None = None
    score_dtype: ttnn.DataType | None = None
    output_dtype: ttnn.DataType | None = None
    qkv_memcfg: ttnn.MemoryConfig | None = None
    create_heads_memcfg: ttnn.MemoryConfig | None = None
    score_memcfg: ttnn.MemoryConfig | None = None
    output_memcfg: ttnn.MemoryConfig | None = None

    # Program and compute knobs
    qkv_prg_config: object | None = None
    output_prg_config: object | None = None
    qkv_minimal_config: object | None = None
    output_minimal_config: object | None = None
    qkv_compute_kernel_cfg: object | None = None
    score_compute_kernel_cfg: object | None = None
    output_compute_kernel_cfg: object | None = None
    core_grid: ttnn.CoreGrid | None = None

    max_seq_len: int | None = None
    max_batch_size: int | None = None
    # Sequence-parallel: mesh axis over which the sequence dim is sharded. When
    # set, each chip holds S/tp local query rows; K/V are all-gathered to full S
    # so local queries cross-attend to every key. Everything else in the encoder
    # is token-local (zero comm). None disables (single-chip path).
    sequence_parallel_axis: int | None = None
    # Data-parallel (DP=2): each chip runs an independent full-sequence B/2
    # replica with no collectives. Enables the query head-fold SDPA trick and
    # the DP-tuned SDPA chunk/dtype config.
    data_parallel: bool = False
    # True when the attention scale was folded into the Q projection weight at
    # build time, so SDPA must run with scale=1.0 regardless of mask presence.
    qkv_scale_prefolded: bool = False
    # Opt-in: use the model-local JIT encoder-SDPA descriptor (custom_ops/
    # encoder_sdpa) instead of stock ttnn SDPA. Only valid on the DP S8192
    # head-folded path (Q[6,32,4096,64] bf8 / K bf4 / V bf8, no mask, scale 1).
    # Stock SDPA is the default/fallback. Parity-verified (PCC 1.0, wall==stock).
    use_experimental_encoder_sdpa: bool = False

    @property
    def qkv_out_dim(self) -> int:
        return 3 * self.hidden_size


class BgeM3Attention(LightweightModule):
    def __init__(
        self,
        wqkv: LazyWeight,
        wo_weight: LazyWeight,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        bqkv: LazyWeight | None = None,
        wo_bias: LazyWeight | None = None,
        attention_scale: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.config = _resolve_attention_config(
            BgeM3AttentionConfig(
                wqkv=wqkv,
                wo_weight=wo_weight,
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                bqkv=bqkv,
                wo_bias=wo_bias,
                attention_scale=attention_scale,
                max_seq_len=max_seq_len,
            )
        )
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: BgeM3AttentionConfig) -> "BgeM3Attention":
        instance = object.__new__(cls)
        super(BgeM3Attention, instance).__init__()
        instance.config = _resolve_attention_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return
        self.wqkv = self.config.wqkv.get_device_weight()
        self.bqkv = self.config.bqkv.get_device_weight() if self.config.bqkv is not None else None
        self.wo_weight = self.config.wo_weight.get_device_weight()
        self.wo_bias = self.config.wo_bias.get_device_weight() if self.config.wo_bias is not None else None
        self._device_weights_loaded = True

    def forward(self, hidden_states: ttnn.Tensor, attention_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        self.load_device_weights()

        batch_size, _, seq_len, _ = hidden_states.shape

        assert seq_len > 0, "seq_len must be positive"
        assert seq_len % 32 == 0, "seq_len must be divisible by 32 (tile height)"
        if seq_len > 128:
            assert seq_len % 128 == 0, "seq_len must be divisible by 128 when seq_len > 128"

        qkv_core_grid = None if self.config.qkv_prg_config is not None else self.config.core_grid
        output_core_grid = None if self.config.output_prg_config is not None else self.config.core_grid

        # QKV chunking for very long sequences
        if seq_len > _MAX_QKV_MM_CHUNK_SEQ_LEN:
            if seq_len % _MAX_QKV_MM_CHUNK_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {_MAX_QKV_MM_CHUNK_SEQ_LEN}")
            hidden_states = ttnn.reshape(
                hidden_states,
                [batch_size, seq_len // _MAX_QKV_MM_CHUNK_SEQ_LEN, _MAX_QKV_MM_CHUNK_SEQ_LEN, -1],
            )

        # Stage 1: fused QKV projection
        if self.config.qkv_minimal_config is not None and self.config.qkv_prg_config is None:
            qkv_fused = ttnn.experimental.minimal_matmul(
                input_tensor=hidden_states,
                weight_tensor=self.wqkv,
                bias_tensor=self.bqkv,
                fused_activation=None,
                config=self.config.qkv_minimal_config,
                memory_config=self.config.qkv_memcfg,
                dtype=self.config.qkv_dtype,
                compute_kernel_config=self.config.qkv_compute_kernel_cfg,
            )
        else:
            qkv_fused = ttnn.linear(
                hidden_states,
                self.wqkv,
                memory_config=self.config.qkv_memcfg,
                dtype=self.config.qkv_dtype,
                bias=self.bqkv,
                program_config=self.config.qkv_prg_config,
                compute_kernel_config=self.config.qkv_compute_kernel_cfg,
                core_grid=qkv_core_grid,
            )
        if seq_len > _MAX_QKV_MM_CHUNK_SEQ_LEN:
            qkv_fused = ttnn.reshape(qkv_fused, [batch_size, 1, seq_len, -1])

        # Stage 2: split Q/K/V heads.
        # B1/S512 + B32/S512: head-split kernels for higher core utilization.
        # Other shapes: stock ttnn ops.
        if (self.config.max_batch_size in (1, 8, 16, 32) and self.config.max_seq_len == 512) or (
            self.config.max_seq_len == 8192
        ):
            from models.demos.wormhole.bge_m3.tt.custom_ops.fused_qkv_heads.op import bge_qkv_heads_headsplit

            # Batch 32 already has 32×16 = 512 (batch × seq_tile) work units, so we
            # don't need to further split heads to get good core utilization.
            # B8 has 8×16 = 128 units -> also plenty, use groups=4 like B32 (swept
            # head_groups {1,2,4,8,16}: 4 is the min, tied with 8).
            # S8192: retest fused head-split under current LoFi config (exp13 was
            # pre-LoFi noise); groups=4 like B32.
            head_groups = (
                4
                if self.config.max_batch_size in (8, 16, 32) or self.config.max_seq_len == 8192
                else self.config.num_heads
            )
            # DP head-fold path emits K directly as bfloat4_b from the head-split
            # (folding the standalone ttnn.typecast(k, bf4) into the op — removes
            # one program + the BF8 K DRAM round trip). Guarded to the exact case
            # the head-fold branch below expects (DP, no mask, S8192).
            fuse_kbf4 = self.config.data_parallel and attention_mask is None and seq_len == 8192
            q, k, v = bge_qkv_heads_headsplit(
                qkv_fused,
                num_heads=self.config.num_heads,
                head_groups=head_groups,
                out_memcfg=self.config.create_heads_memcfg,
                k_out_dtype=ttnn.bfloat4_b if fuse_kbf4 else None,
            )
        else:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv_fused,
                num_heads=self.config.num_heads,
                num_kv_heads=self.config.num_heads,
                transpose_k_heads=False,
                memory_config=self.config.create_heads_memcfg,
            )
        ttnn.deallocate(qkv_fused)

        # Stage 3: optional cast to score dtype
        if self.config.score_dtype is not None and q.dtype != self.config.score_dtype:
            q_cast = ttnn.typecast(q, dtype=self.config.score_dtype)
            ttnn.deallocate(q)
            q = q_cast
        # Skip the K score-dtype cast when K was already emitted as bfloat4_b by
        # the fused head-split (its dtype intentionally differs from score_dtype).
        if self.config.score_dtype is not None and k.dtype != self.config.score_dtype and k.dtype != ttnn.bfloat4_b:
            k_cast = ttnn.typecast(k, dtype=self.config.score_dtype)
            ttnn.deallocate(k)
            k = k_cast
        if self.config.score_dtype is not None and v.dtype != self.config.score_dtype:
            v_cast = ttnn.typecast(v, dtype=self.config.score_dtype)
            ttnn.deallocate(v)
            v = v_cast

        # Stage 3c: sequence-parallel K/V all-gather. Each chip produced K/V for
        # its local S/tp query rows; gather across the sequence mesh axis so the
        # local queries attend to the full sequence. Q stays sharded (local rows).
        if self.config.sequence_parallel_axis is not None:
            k_full = ttnn.all_gather(
                k, dim=2, cluster_axis=self.config.sequence_parallel_axis, num_links=1, topology=ttnn.Topology.Linear
            )
            ttnn.deallocate(k)
            k = k_full
            v_full = ttnn.all_gather(
                v, dim=2, cluster_axis=self.config.sequence_parallel_axis, num_links=1, topology=ttnn.Topology.Linear
            )
            ttnn.deallocate(v)
            v = v_full

        # Stage 3b: mask preparation
        sdpa_mask = attention_mask
        if sdpa_mask is not None:
            if len(sdpa_mask.shape) != 4:
                raise ValueError(f"attention_mask must have rank 4 [B, 1, S, S], got shape={sdpa_mask.shape}")
            if (
                sdpa_mask.shape[0] != batch_size
                or sdpa_mask.shape[1] != 1
                or sdpa_mask.shape[2] != seq_len
                or sdpa_mask.shape[3] != seq_len
            ):
                raise ValueError(
                    f"attention_mask shape must be [{batch_size}, 1, {seq_len}, {seq_len}], got {sdpa_mask.shape}"
                )
            if self.config.score_dtype is not None and sdpa_mask.dtype != self.config.score_dtype:
                sdpa_mask = ttnn.typecast(sdpa_mask, dtype=self.config.score_dtype)
            if sdpa_mask.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                sdpa_mask = ttnn.to_memory_config(sdpa_mask, ttnn.DRAM_MEMORY_CONFIG)

        # Stage 4: encoder SDPA (chunk sizes depend on runtime seq_len)
        # N300 B12/S8192: when both scale and an explicit dense [B,1,S,S] mask are
        # passed, ttnn SDPA pre-scales the mask by 1/scale as a separate op on the
        # full 12x8192x8192 mask EVERY layer (~7% of runtime). We avoid it by
        # passing scale=1.0. The attention scale is folded into the Q projection
        # WEIGHT at build time (encoder.py q_scale, weight_adapter.build_attention_
        # weights) so no per-layer q*scale multiply is needed either. Numerically
        # identical: softmax(Q@K^T*s + m) == softmax(((Q*s)@K^T) + m), and folding
        # into the fp32 weight before bf8 quant is at least as accurate.
        sdpa_scale = self.config.attention_scale
        if self.config.qkv_scale_prefolded:
            sdpa_scale = 1.0
        elif seq_len == 8192 and sdpa_mask is not None and sdpa_scale is not None:
            sdpa_scale = 1.0
        # SDPA chunk sizing keys off the full (key) sequence length. In
        # sequence-parallel mode queries are local (S/tp) but keys span full S.
        sdpa_seq_len = k.shape[2]
        # DP query head-fold trick: SDPA throughput is set by Sq (query length),
        # not total work (measured: Sq8192=29 vs Sq4096=44 TFLOP/s). Fold
        # DP_HEAD_FOLD query-chunks into the HEAD dim so SDPA sees Sq/G queries
        # per head; K/V stay unchanged and SDPA's GQA head-broadcast makes each
        # query head attend to the FULL sequence via its parent kv head. Exact
        # (PCC=1.0), needs NO K/V replicate. q [B,H,S,DH] -> [B, H*G, S/G, DH]
        # with head h*G+j = head h's j-th seq chunk.
        head_fold = self.config.data_parallel and sdpa_mask is None and seq_len == 8192
        if head_fold:
            b0, h0, s0, dh0 = q.shape
            q = ttnn.reshape(q, [b0, h0 * _DP_HEAD_FOLD, s0 // _DP_HEAD_FOLD, dh0])
            # K is already bfloat4_b (emitted directly by the fused head-split):
            # halves K read bandwidth in SDPA. PCC 0.9422 (clears the 0.94
            # preferred gate) with the q128/k2048 chunking; V stays bf8. If some
            # other path left K non-BF4, convert here as a fallback.
            if k.dtype != ttnn.bfloat4_b:
                k_bf4 = ttnn.typecast(k, dtype=ttnn.bfloat4_b)
                ttnn.deallocate(k)
                k = k_bf4
        sdpa_program_config = _sdpa_program_config(
            sdpa_seq_len,
            self.config.mesh_device,
            batch_size=batch_size,
            sequence_parallel=self.config.sequence_parallel_axis is not None,
            data_parallel=self.config.data_parallel,
        )
        # Opt-in model-local JIT encoder SDPA: only on the exact DP S8192 head-
        # folded contract it was built for (Q[6,32,4096,64] bf8, K bf4, V bf8,
        # no mask, scale 1). Any deviation falls back to stock SDPA.
        use_encoder_sdpa = (
            self.config.use_experimental_encoder_sdpa
            and head_fold
            and sdpa_mask is None
            and sdpa_scale == 1.0
            and tuple(q.shape) == (6, 32, 4096, 64)
            and tuple(k.shape) == (6, 16, 8192, 64)
            and tuple(v.shape) == (6, 16, 8192, 64)
            and q.dtype == ttnn.bfloat8_b
            and k.dtype == ttnn.bfloat4_b
            and v.dtype == ttnn.bfloat8_b
        )
        if use_encoder_sdpa:
            from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa import (
                EncoderSDPAConfig,
                bge_encoder_sdpa_experimental,
            )

            # Non-FP32-dest / half-sync (DEST=8), q128/k2048, BF16 score.
            # Validated: -57ms wall, PCC 0.9548 (gate), comparable-to-stock across
            # seeds. BF8 score CB was ablated and CONCLUSIVELY breaks full-model
            # PCC (0.31 at q128 AND q256) despite standalone PCC 1.0 on peaked
            # random softmax — real activations have flatter softmax that BF8
            # score quantization destroys. q256/k2048 hits 21.48ms/call but only
            # via BF8 score (bf16-score q256 OOMs), so it is not shippable.
            _ecfg = EncoderSDPAConfig(fp32_dest_acc_en=False)
            context = bge_encoder_sdpa_experimental(q, k, v, output_mem_config=self.config.score_memcfg, config=_ecfg)
        else:
            context = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                attn_mask=sdpa_mask,
                scale=sdpa_scale,
                program_config=sdpa_program_config,
                compute_kernel_config=self.config.score_compute_kernel_cfg,
                memory_config=self.config.score_memcfg,
            )
        if head_fold:
            # [B, H*G, S/G, DH] -> [B, H, S, DH]
            b0, hg, sg, dh0 = context.shape
            context = ttnn.reshape(context, [b0, hg // _DP_HEAD_FOLD, sg * _DP_HEAD_FOLD, dh0])
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Stage 5: concat heads. B1/B8/B32 S512 + S8192: fused head-split kernel
        # (higher core utilization than stock nlp_concat_heads).
        if (self.config.max_batch_size in (1, 8, 16, 32) and self.config.max_seq_len == 512) or (
            self.config.max_seq_len == 8192
        ):
            from models.demos.wormhole.bge_m3.tt.custom_ops.fused_concat_heads.op import bge_concat_heads_headsplit

            # B8 swept concat head_groups {1,2,4,8,16}: 16 is marginally best.
            # S8192 B6 swept {1,2,4,8,16}: 16 is best (0.598 vs stock 0.812ms/call).
            concat_head_groups = 16 if self.config.max_batch_size in (8, 16) or self.config.max_seq_len == 8192 else 4
            context = bge_concat_heads_headsplit(
                context, head_groups=concat_head_groups, out_memcfg=self.config.output_memcfg
            )
        else:
            context = ttnn.experimental.nlp_concat_heads(context, memory_config=self.config.output_memcfg)

        # WO chunking for very long sequences
        if seq_len > _MAX_WO_MM_CHUNK_SEQ_LEN:
            if seq_len % _MAX_WO_MM_CHUNK_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {_MAX_WO_MM_CHUNK_SEQ_LEN}")
            context = ttnn.reshape(
                context, [batch_size, seq_len // _MAX_WO_MM_CHUNK_SEQ_LEN, _MAX_WO_MM_CHUNK_SEQ_LEN, -1]
            )

        # Stage 6: output projection
        if self.config.output_minimal_config is not None and self.config.output_prg_config is None:
            output = ttnn.experimental.minimal_matmul(
                input_tensor=context,
                weight_tensor=self.wo_weight,
                bias_tensor=self.wo_bias,
                fused_activation=None,
                config=self.config.output_minimal_config,
                memory_config=self.config.output_memcfg,
                dtype=self.config.output_dtype,
                compute_kernel_config=self.config.output_compute_kernel_cfg,
            )
        else:
            output = ttnn.linear(
                context,
                self.wo_weight,
                memory_config=self.config.output_memcfg,
                dtype=self.config.output_dtype,
                bias=self.wo_bias,
                program_config=self.config.output_prg_config,
                compute_kernel_config=self.config.output_compute_kernel_cfg,
                core_grid=output_core_grid,
            )
        ttnn.deallocate(context)

        if seq_len > _MAX_WO_MM_CHUNK_SEQ_LEN:
            output = ttnn.reshape(output, [batch_size, 1, seq_len, -1])

        return output


# ──────────────────────────────────────────────────────────────────────────────
# SDPA runtime helpers (must stay here — chunk sizes depend on actual seq_len)
# ──────────────────────────────────────────────────────────────────────────────


def _sdpa_chunks_for_seq_len(seq_len, batch_size=None, sequence_parallel=False, data_parallel=False):
    if seq_len % 128 == 0:
        # N300 B12/S8192: q256/k256 wins IN-MODEL (4056ms). Standalone sweep is
        # unreliable for SDPA (see sweep_sdpa_b12_s8192.py warning); tuned by
        # direct perf.py runs. Tested in-model: q512/k128=4489, q256/k256=4056(best),
        # q256/k128=4596. k_chunk=256 is the sweet spot.
        if seq_len == 8192:
            if sequence_parallel:
                # Sequence-parallel: local Sq=4096, gathered Sk=8192. Lower L1
                # pressure (halved Sq) lets k_chunk=512 fit, halving the number
                # of k-passes and the online-softmax rescaling overhead.
                return 512, 512
            if data_parallel:
                # Data-parallel + query head-fold: SDPA sees Sq=4096, Sk=8192, B6.
                # q128/k2048 keeps the 256-score-tile footprint (4x64) while
                # cutting k-chunks to 4 (fewer online-softmax merges). With
                # bfloat4_b K it runs 30.1ms->28.9ms/op and holds PCC 0.9422
                # (clears the 0.94 preferred gate). (DP autoresearch #216.)
                return 128, 2048
            # Dense-mask S8192 needs the lower-L1 k256 configuration.
            return 512, 256
        # B8: q=256 k=256 (swept q{64..512} x k{128,256,512}; 256x256 is the min,
        # ~0.27ms under the B32-inherited 256x512). B32 keeps 256x512.
        # B16: q=256 k=256 (swept; 256x256 ~0.25ms under 256x512, same as B8).
        if seq_len == 512 and batch_size in (8, 16):
            return 256, 256
        if seq_len == 512 and batch_size == 32:
            return 256, 512
        if seq_len == 512 and batch_size == 1:
            return _SDPA_B1S512_Q_CHUNK, _SDPA_B1S512_K_CHUNK
        for k_chunk in _SDPA_K_CANDIDATES_MAIN:
            if k_chunk <= seq_len and seq_len % k_chunk == 0:
                return _SDPA_Q_CHUNK_MAIN, k_chunk
        raise ValueError(f"Unable to pick k_chunk_size for seq_len={seq_len}")
    if seq_len % 32 != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by 32")
    if seq_len > 128 and seq_len % 128 != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by 128 when > 128")
    q_chunk = next(q for q in _SDPA_Q_CHUNKS_FLEX if q <= seq_len and seq_len % q == 0)
    k_chunk = next(k for k in _SDPA_K_CHUNKS_FLEX if k <= seq_len and seq_len % k == 0)
    return q_chunk, k_chunk


def _sdpa_exp_approx(seq_len, mesh_device=None):
    if mesh_device is not None and ttnn_is_blackhole(mesh_device):
        return False
    return seq_len % 128 == 0


def _sdpa_compute_grid(mesh_device):
    if mesh_device is None:
        return (8, 8)
    try:
        return mesh_device.compute_with_storage_grid_size()
    except Exception:
        return (8, 8)


def _sdpa_program_config(seq_len, mesh_device, batch_size=None, sequence_parallel=False, data_parallel=False):
    q_chunk, k_chunk = _sdpa_chunks_for_seq_len(
        seq_len, batch_size=batch_size, sequence_parallel=sequence_parallel, data_parallel=data_parallel
    )
    grid = _sdpa_compute_grid(mesh_device)
    # B1/S512 on Blackhole: 8x8=64 cores beats the default 11x10=110 grid.
    # Sweep showed ~10% lower SDPA device time at smaller grid (less dispatch
    # overhead vs. number of head-batch pairs).
    if seq_len == 512 and batch_size == 1 and mesh_device is not None and ttnn_is_blackhole(mesh_device):
        grid = ttnn.CoreCoord(8, 8)
    kwargs = {
        "compute_with_storage_grid_size": grid,
        "q_chunk_size": q_chunk,
        "k_chunk_size": k_chunk,
        "exp_approx_mode": _sdpa_exp_approx(seq_len, mesh_device),
    }
    if seq_len == 512 and batch_size in (8, 16, 32) and mesh_device is not None and ttnn_is_blackhole(mesh_device):
        kwargs["max_cores_per_head_batch"] = 8
        # NOTE: B16 max_cores_per_head_batch swept {2,4,8,16,none} — all within
        # 0.06ms noise; kept 8 (B8/B32 value).
    # NOTE: swept B8 SDPA grid {8x8, 10x10, 11x10} x max_cores {none,4,8} at the
    # 256x256 chunk: 11x10 + max_cores=8 is optimal (24.28ms). 8x8=64 cores
    # regresses to ~24.89ms (B8's 128 head-batch pairs want more cores, unlike
    # B1 where 8x8 wins). Exhausted.
    return ttnn.SDPAProgramConfig(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Config resolver
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_attention_config(config: BgeM3AttentionConfig) -> BgeM3AttentionConfig:
    if config.hidden_size != config.num_heads * config.head_dim:
        raise ValueError(
            f"hidden_size must equal num_heads * head_dim "
            f"(got {config.hidden_size}, {config.num_heads}, {config.head_dim})"
        )
    if config.wqkv is None or config.wo_weight is None:
        raise ValueError("Both wqkv and wo_weight must be provided")

    to_set: dict[str, object] = {}

    # Numerics defaults
    if config.attention_scale is None:
        to_set["attention_scale"] = config.head_dim**-0.5
    if config.qkv_dtype is None:
        to_set["qkv_dtype"] = ttnn.bfloat16
    if config.score_dtype is None:
        to_set["score_dtype"] = ttnn.bfloat16
    if config.output_dtype is None:
        to_set["output_dtype"] = ttnn.bfloat16

    # Resolve device
    param_devices = [
        p.device
        for p in (config.wqkv, config.bqkv, config.wo_weight, config.wo_bias)
        if p is not None and p.device is not None
    ]
    if param_devices and any(d != param_devices[0] for d in param_devices):
        raise ValueError("All attention parameters must target the same device")

    mesh_device = config.mesh_device or (param_devices[0] if param_devices else ttnn.GetDefaultDevice())
    if mesh_device is None:
        raise ValueError("Unable to resolve target device for BgeM3Attention")
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    # Defaults: DRAM for everything, basic compute kernel
    if config.qkv_memcfg is None:
        to_set["qkv_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.create_heads_memcfg is None:
        to_set["create_heads_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.score_memcfg is None:
        to_set["score_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.output_memcfg is None:
        to_set["output_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.qkv_compute_kernel_cfg is None:
        to_set["qkv_compute_kernel_cfg"] = _default_compute_kernel(mesh_device)
    if config.output_compute_kernel_cfg is None:
        to_set["output_compute_kernel_cfg"] = _default_compute_kernel(mesh_device)
    if config.score_compute_kernel_cfg is None:
        to_set["score_compute_kernel_cfg"] = _default_compute_kernel(mesh_device)
    if config.core_grid is None:
        to_set["core_grid"] = _default_core_grid(mesh_device)

    # Resolve weights
    qkv_dtype = to_set.get("qkv_dtype", config.qkv_dtype)
    output_dtype = to_set.get("output_dtype", config.output_dtype)
    weight_mem = ttnn.DRAM_MEMORY_CONFIG

    to_set["wqkv"] = resolve_lazy_weight(
        config.wqkv,
        device=mesh_device,
        dtype=qkv_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_mem,
        mesh_mapper_config=None,
    )
    to_set["wo_weight"] = resolve_lazy_weight(
        config.wo_weight,
        device=mesh_device,
        dtype=output_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_mem,
        mesh_mapper_config=None,
    )
    if config.bqkv is not None:
        to_set["bqkv"] = resolve_lazy_weight(
            config.bqkv,
            device=mesh_device,
            dtype=qkv_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_mem,
            mesh_mapper_config=None,
        )
    if config.wo_bias is not None:
        to_set["wo_bias"] = resolve_lazy_weight(
            config.wo_bias,
            device=mesh_device,
            dtype=output_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_mem,
            mesh_mapper_config=None,
        )

    return replace(config, **to_set)


def _default_compute_kernel(mesh_device):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _default_core_grid(mesh_device):
    try:
        g = mesh_device.compute_with_storage_grid_size()
        return ttnn.CoreGrid(y=int(g.y), x=int(g.x))
    except Exception:
        return ttnn.CoreGrid(y=8, x=8)

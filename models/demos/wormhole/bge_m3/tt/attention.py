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
# B1/S512 SDPA chunking.
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
    # Mesh axis that shards the sequence dim. Each chip keeps S/tp query rows
    # and all-gathers K/V to the full S. None selects the single-chip path.
    sequence_parallel_axis: int | None = None
    # DP=2: every chip runs an independent B/2 full-sequence replica, no collectives.
    data_parallel: bool = False
    # The build folds the attention scale into the Q weight, so SDPA uses scale=1.0.
    qkv_scale_prefolded: bool = False
    # Use the model-local encoder-SDPA descriptor instead of the stock ttnn SDPA.
    use_experimental_encoder_sdpa: bool = False
    # Fuse the QKV matmul, the head scatter, and the K/V BF4 conversion.
    use_qkv_scatter_matmul: bool = False
    # Route masked requests to the high-precision kernels instead of BF4.
    mask_hifi: bool = False

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

        if seq_len > _MAX_QKV_MM_CHUNK_SEQ_LEN:
            if seq_len % _MAX_QKV_MM_CHUNK_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {_MAX_QKV_MM_CHUNK_SEQ_LEN}")
            hidden_states = ttnn.reshape(
                hidden_states,
                [batch_size, seq_len // _MAX_QKV_MM_CHUNK_SEQ_LEN, _MAX_QKV_MM_CHUNK_SEQ_LEN, -1],
            )

        # Stage 1: fused QKV projection
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

        # Stage 2: split Q/K/V heads (fused head-split kernel on the S512 shapes).
        if self.config.max_batch_size in (1, 8, 16, 32) and self.config.max_seq_len == 512:
            from models.demos.wormhole.bge_m3.tt.custom_ops.fused_qkv_heads.op import bge_qkv_heads_headsplit

            head_groups = 4 if self.config.max_batch_size in (8, 16, 32) else self.config.num_heads
            q, k, v = bge_qkv_heads_headsplit(
                qkv_fused,
                num_heads=self.config.num_heads,
                head_groups=head_groups,
                out_memcfg=self.config.create_heads_memcfg,
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
        if self.config.score_dtype is not None and k.dtype != self.config.score_dtype:
            k_cast = ttnn.typecast(k, dtype=self.config.score_dtype)
            ttnn.deallocate(k)
            k = k_cast
        if self.config.score_dtype is not None and v.dtype != self.config.score_dtype:
            v_cast = ttnn.typecast(v, dtype=self.config.score_dtype)
            ttnn.deallocate(v)
            v = v_cast

        # Stage 3b: mask preparation (dense [B, 1, S, S])
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

        # Stage 4: SDPA (chunk sizes depend on runtime seq_len)
        sdpa_program_config = _sdpa_program_config(seq_len, self.config.mesh_device, batch_size=batch_size)
        context = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=sdpa_mask,
            scale=self.config.attention_scale,
            program_config=sdpa_program_config,
            compute_kernel_config=self.config.score_compute_kernel_cfg,
            memory_config=self.config.score_memcfg,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Stage 5: concat heads
        if self.config.max_batch_size in (1, 8, 16, 32) and self.config.max_seq_len == 512:
            from models.demos.wormhole.bge_m3.tt.custom_ops.fused_concat_heads.op import bge_concat_heads_headsplit

            concat_head_groups = 16 if self.config.max_batch_size in (8, 16) else 4
            context = bge_concat_heads_headsplit(
                context, head_groups=concat_head_groups, out_memcfg=self.config.output_memcfg
            )
        else:
            context = ttnn.experimental.nlp_concat_heads(context, memory_config=self.config.output_memcfg)

        if seq_len > _MAX_WO_MM_CHUNK_SEQ_LEN:
            if seq_len % _MAX_WO_MM_CHUNK_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {_MAX_WO_MM_CHUNK_SEQ_LEN}")
            context = ttnn.reshape(
                context, [batch_size, seq_len // _MAX_WO_MM_CHUNK_SEQ_LEN, _MAX_WO_MM_CHUNK_SEQ_LEN, -1]
            )

        # Stage 6: output projection
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


class BgeM3AttentionJit(BgeM3Attention):
    """Attention for the B12/S8192 data-parallel serving shape.

    ModelArgs sets use_jit only for that shape, so this path assumes a local
    batch of 6, sequence length 8192, 16 heads, and head dim 64. It runs the
    QKV scatter, the model-local encoder SDPA with the query head-fold, and
    BF4 K/V. It accepts no mask, or compact [B, 1] valid lengths.
    """

    def forward(self, hidden_states: ttnn.Tensor, attention_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        self.load_device_weights()

        batch_size, _, seq_len, _ = hidden_states.shape
        # The caller supplies compact [B, 1] valid lengths, or no mask at all.
        # A dense mask never reaches this path.
        valid_lengths = attention_mask
        # mask_hifi routes masked requests to the high-precision kernels.
        use_bf4_serving = valid_lengths is None or not self.config.mask_hifi

        q, k, v = self._project_qkv(hidden_states, use_bf4_serving)
        context = self._attend(q, k, v, valid_lengths, use_bf4_serving)
        return self._project_out(context)

    def _project_qkv(self, hidden_states: ttnn.Tensor, use_bf4_serving: bool):
        """Return Q, K, V heads. The scatter fuses the projection, the head
        split, and the BF4 conversion into one program."""
        if use_bf4_serving and self.config.use_qkv_scatter_matmul:
            from models.demos.wormhole.bge_m3.tt.custom_ops.qkv_scatter_matmul import bge_qkv_scatter_matmul

            # Q is BF4 to halve the scatter write. SDPA consumes BF4 directly.
            return bge_qkv_scatter_matmul(
                hidden_states,
                self.wqkv,
                bias_tensor=self.bqkv,
                memory_config=self.config.create_heads_memcfg,
                dtype=ttnn.bfloat4_b,
            )

        from models.demos.wormhole.bge_m3.tt.custom_ops.fused_qkv_heads.op import bge_qkv_heads_headsplit

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
        bf4_kv = ttnn.bfloat4_b if use_bf4_serving else None
        q, k, v = bge_qkv_heads_headsplit(
            qkv_fused,
            num_heads=self.config.num_heads,
            head_groups=4,
            out_memcfg=self.config.create_heads_memcfg,
            k_out_dtype=bf4_kv,
            v_out_dtype=bf4_kv,
        )
        ttnn.deallocate(qkv_fused)
        return q, k, v

    def _attend(self, q, k, v, valid_lengths, use_bf4_serving: bool) -> ttnn.Tensor:
        """Run the encoder SDPA and return context as [B, 1, S, H*D]."""
        from models.demos.wormhole.bge_m3.tt.custom_ops.encoder_sdpa import (
            EncoderSDPAConfig,
            bge_encoder_sdpa_experimental,
        )

        # Cast Q and K only when a previous stage left them off the BF4 contract.
        if not use_bf4_serving and q.dtype != self.config.score_dtype:
            q = _retype(q, self.config.score_dtype)
        if use_bf4_serving and k.dtype != ttnn.bfloat4_b:
            k = _retype(k, ttnn.bfloat4_b)

        # Query head-fold: SDPA throughput follows the query length, so fold
        # _DP_HEAD_FOLD query chunks into the head dim. GQA head-broadcast keeps
        # every query head attending to the full sequence, so this is exact.
        b, h, s, dh = q.shape
        q = ttnn.reshape(q, [b, h * _DP_HEAD_FOLD, s // _DP_HEAD_FOLD, dh])

        if use_bf4_serving:
            config = EncoderSDPAConfig(
                batch=int(q.shape[0]),
                q_chunk_size=256,
                k_chunk_size=2048,
                q_buffer_depth=2,
                k_buffer_depth=1,
                v_buffer_depth=1,
                fp32_dest_acc_en=False,
                direct_concat_heads=True,
                reuse_prev_max_for_exp=True,
                use_runtime_lengths=valid_lengths is not None,
            )
        else:
            config = EncoderSDPAConfig(
                batch=int(q.shape[0]),
                q_chunk_size=128,
                k_chunk_size=512,
                q_buffer_depth=2,
                k_buffer_depth=2,
                v_buffer_depth=2,
                fp32_dest_acc_en=True,
                use_runtime_lengths=valid_lengths is not None,
            )

        context = bge_encoder_sdpa_experimental(
            q,
            k,
            v,
            valid_lengths=valid_lengths,
            output_mem_config=self.config.score_memcfg,
            config=config,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        if not config.direct_concat_heads:
            # Undo the head-fold, then concat the heads.
            b, hg, sg, dh = context.shape
            context = ttnn.reshape(context, [b, hg // _DP_HEAD_FOLD, sg * _DP_HEAD_FOLD, dh])
            from models.demos.wormhole.bge_m3.tt.custom_ops.fused_concat_heads.op import bge_concat_heads_headsplit

            context = bge_concat_heads_headsplit(context, head_groups=16, out_memcfg=self.config.output_memcfg)
        return context

    def _project_out(self, context: ttnn.Tensor) -> ttnn.Tensor:
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
        ttnn.deallocate(context)
        return output


def _retype(tensor: ttnn.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor:
    converted = ttnn.typecast(tensor, dtype=dtype)
    ttnn.deallocate(tensor)
    return converted


# ──────────────────────────────────────────────────────────────────────────────
# SDPA runtime helpers (must stay here — chunk sizes depend on actual seq_len)
# ──────────────────────────────────────────────────────────────────────────────


def _sdpa_chunks_for_seq_len(seq_len, batch_size=None, sequence_parallel=False, data_parallel=False):
    if seq_len % 128 == 0:
        if seq_len == 8192:
            if sequence_parallel:
                # Local Sq=4096, gathered Sk=8192; k_chunk=512 fits the lower L1.
                return 512, 512
            if data_parallel:
                # Query head-fold: Sq=4096, Sk=8192, B6. q128/k2048 uses 4 k-chunks.
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
    # B1/S512 on Blackhole: an 8x8 grid beats the default 11x10.
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

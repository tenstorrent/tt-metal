# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
TT self-attention for Devstral-2 123B (Hugging Face ``Ministral3Attention``).

Per ``reference/model_structure.txt``, each decoder layer has::

    self_attn: Ministral3Attention(
        q_proj: Linear(12288 -> 12288, bias=False)
        k_proj: Linear(12288 -> 1024, bias=False)
        v_proj: Linear(12288 -> 1024, bias=False)
        o_proj: Linear(12288 -> 12288, bias=False)
    )

That is the same *pattern* as Devstral-Small-2 / Ministral3 (GQA, separate Q/K/V/O linears). HF
``forward`` applies RoPE to Q/K, then multiplies Q by ``get_llama_4_attn_scale`` from
``rope_parameters`` (``llama_4_scaling_beta``, ``original_max_position_embeddings``), then
standard grouped attention and ``o_proj`` — see
``transformers.models.ministral3.modeling_ministral3.Ministral3Attention``.

Llama-4-style **query scaling after RoPE** is implemented here on top of
:class:`~models.tt_transformers.tt.attention.Attention` (same device-side path as the small
Ministral3 stack: post-RoPE Q scaling, ``position_ids`` on prefill). This module keeps the **runtime
class name** ``Attention`` so meta / ``layers.{i}.attention.*`` weight keys resolve like production
Llama-family attention.

On **multi-device** meshes (non-Galaxy) with ~12k hidden, width-sharded DRAM ``TilizeDeviceOperation``
for ``wqkv`` / ``wo`` can exceed static L1 circular-buffer limits on **Blackhole** and **Wormhole T3K**.
:class:`TtDevstral2LargeAttention` forces **interleaved DRAM** for those uploads (same mitigation as
:class:`~models.experimental.devstral2_large.tt.tt_ministralmlp.TtDevstral2LargeMLP` weight tilize).

Decode QKV defaults to DRAM-sharded matmuls, which require width-sharded weights; that conflicts
with interleaved uploads and with ``interleaved_to_sharded`` on full weights (huge L1 CBs). When the
same mitigation applies, decode reuses the **short-sequence prefill** multicast QKV program and
**DRAM** QKV output mem configs. **WO decode** for that setup is handled by a dedicated
:meth:`TtDevstral2LargeAttention.forward_decode` (not the generic ``Attention.forward_decode`` tail):
concat output is moved to **L1 interleaved**, a single multicast ``ttnn.linear`` with
:func:`_decode_wo_interleaved_multicast_prog_cfg` and interleaved I/O, then **width-sharded** for
``tt_all_reduce`` — no DRAM-sharded WO matmul and no ``tt_all_gather`` + width-sharded activation
path that conflicts with interleaved ``wo``.

Prefill ``wo`` (:meth:`~models.tt_transformers.tt.attention.Attention.forward_prefill` ``ttnn.linear``)
uses ``ModelArgs.get_attn_wo_program_config``. For ``MatmulMultiCoreReuseMultiCastProgramConfig``, the
TTNN constructor defaults **omitted** ``out_block_h`` / ``out_block_w`` to ``per_core_M`` /
``per_core_N`` (see ``matmul_nanobind.cpp``). For wide 12k ``wo`` that makes ``out_block_w`` ≈ 48
tiles and overflows L1 for circular buffers on wide multi-device paths — unrelated to weight
**sharding** (DRAM width shard is already correct; the issue is per-op **output block** sizing). This
class wraps ``get_attn_wo_program_config`` and supplies minimal ``out_block_*`` and ``out_subblock_w``
for **prefill** when
:func:`~models.experimental.devstral2_large.tt.device_dram_mitigation.devstral2_large_multi_device_dram_mitigation`
is active (conceptually similar to :class:`TtDevstral2LargeRMSNorm` using a tight multicore norm config).

Activation row count for ``program_config`` still comes from shared ``Attention.forward_prefill``.
Decode on multi-device meshes with DRAM tilize mitigation uses :meth:`forward_decode` below instead
of the generic ``Attention.forward_decode`` gather + DRAM-sharded WO tail.
"""

from __future__ import annotations

import math

import ttnn

from models.experimental.devstral2_large.tt.model_utils import (
    devstral2_large_multi_device_dram_mitigation,
)
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode


def _tighten_wo_prefill_prog_cfg(cfg):
    """Force minimal output block sizes for WO prefill matmul on wide multi-device paths.

    If ``out_block_h`` / ``out_block_w`` are not passed to ``MatmulMultiCoreReuseMultiCastProgramConfig``,
    TTNN sets them to ``per_core_M`` / ``per_core_N``. For ~12k-wide WO, ``out_block_w`` then matches
    ``per_core_N`` (~48 tiles) and static CBs overlap reserved L1 — **not** fixed by sequence chunking
    alone. Supply explicit 1×1 output blocks and minimal subblock width.
    """
    if cfg is None:
        return cfg
    mc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    if not isinstance(cfg, mc):
        return cfg
    return mc(
        compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
        in0_block_w=cfg.in0_block_w,
        out_subblock_h=cfg.out_subblock_h,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=cfg.per_core_M,
        per_core_N=cfg.per_core_N,
        transpose_mcast=cfg.transpose_mcast,
        fused_activation=cfg.fused_activation,
        fuse_batch=cfg.fuse_batch,
    )


def _decode_wo_interleaved_multicast_prog_cfg(args):
    """Multicast decode WO matmul: interleaved ``wo`` is incompatible with ``dram_matmul_config``.

    ``nlp_concat_heads_decode`` lays out width-sharded activations with
    ``num_cores_to_corerangeset(n_local_heads, device.compute_with_storage_grid_size(), …)``
    (``nlp_concat_heads_decode_device_operation.cpp``), so the WO input shard grid can span **many
    cores in X**.

    ``ModelArgs.matmul_config`` forwards ``grid_size`` to C++ as
    ``CoreCoord(x=grid_size[0], y=grid_size[1])`` — **not** ``(rows, cols)`` like
    ``find_prefill_grid`` returns. Passing ``(rows, cols)`` therefore shrinks **X** (the width
    core axis) and trips ``check_tensor_in_grid`` even when ``rows``/``cols`` values are individually
    sensible.

    We pass ``grid_size=(cols, rows)`` so **X** / **Y** match ``CoreCoord(x=cols, y=rows)`` in
    ``matmul_config`` (not ``(rows, cols)`` from ``find_prefill_grid``). ``per_core_N`` uses
    ``grid_size[0]`` (= ``cols``); ``k`` divisible by ``32 * grid_size[1]`` uses ``grid_size[1]`` (=
    ``rows``), so only **rows** must divide ``k_WO / 32`` (tile units). Width ``cols`` is chosen to cover
    the concat footprint: ``min(n_local_heads, grid_w)`` cores in the first row when
    ``n_local_heads <= grid_w``, else full row width ``grid_w`` (see
    ``num_cores_to_corerangeset`` with ``row_wise`` in ``work_split.cpp``).
    """
    k_dim = (args.n_heads * args.head_dim) // args.num_devices
    n_dim = args.dim
    m = args.tile_padded_batch_rows
    mg = args.max_grid_size
    gw, gh = int(mg.x), int(mg.y)
    k_tiles = k_dim // ttnn.TILE_SIZE
    n_pack = int(args.n_local_heads)

    num_x = min(n_pack, gw)
    num_y = (n_pack + num_x - 1) // num_x
    req_cols = num_x
    req_rows = min(max(num_y, 1), gh)

    cols = min(max(req_cols, 1), gw)

    row_candidates = [r for r in range(req_rows, gh + 1) if k_tiles % r == 0]
    if not row_candidates:
        row_candidates = [r for r in range(1, gh + 1) if k_tiles % r == 0]
    rows = min(row_candidates) if row_candidates else 1

    # (x, y) for compute_with_storage_grid_size — see docstring; NOT find_prefill (rows, cols) order.
    grid_size = (cols, rows)

    cfg = args.matmul_config(
        m=m,
        k=k_dim,
        n=n_dim,
        grid_size=grid_size,
        fuse_batch=True,
    )
    return _tighten_wo_prefill_prog_cfg(cfg)


def _widen_qkv_prefill_in0_block_w(cfg, in0_block_w: int = 2):
    """Increase in0_block_w for QKV prefill matmul on Devstral-2-123B.

    The shared config defaults to in0_block_w=1 for seq_len<=128. For K=12288 that means
    384 inner-loop iterations per core; doubling to 2 halves the loop overhead. Only touches
    MatmulMultiCoreReuseMultiCastProgramConfig (seq_len<=128 path); MinimalMatmulConfig
    (seq_len>128) is returned unchanged.
    """
    mc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    if not isinstance(cfg, mc):
        return cfg
    return mc(
        compute_with_storage_grid_size=cfg.compute_with_storage_grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=cfg.out_subblock_h,
        out_subblock_w=cfg.out_subblock_w,
        per_core_M=cfg.per_core_M,
        per_core_N=cfg.per_core_N,
        transpose_mcast=cfg.transpose_mcast,
        fused_activation=cfg.fused_activation,
        fuse_batch=cfg.fuse_batch,
    )


def _multi_device_dram_weight_tilize(mesh_device, configuration) -> bool:
    return configuration is not None and devstral2_large_multi_device_dram_mitigation(mesh_device, configuration)


class TtDevstral2LargeAttention(Attention):
    """
    Ministral3-style attention for Devstral-2 123B on Tenstorrent.

    Pass ``llama_4_scaling_beta`` and ``original_max_position_embeddings`` from
    ``Ministral3Config.rope_parameters`` (same kwargs as the small Ministral3 ``TtMinistralAttention``).
    """

    def __init__(
        self,
        *args,
        llama_4_scaling_beta: float | None = None,
        original_max_position_embeddings: int | None = None,
        **kwargs,
    ):
        mesh_device = args[0] if len(args) > 0 else kwargs.get("mesh_device")
        configuration = kwargs.get("configuration")
        orig_create_dram_sharded = None
        patch_dram_tilize = _multi_device_dram_weight_tilize(mesh_device, configuration)
        if patch_dram_tilize:
            orig_create_dram_sharded = configuration.create_dram_sharded_mem_config

            def _interleaved_dram_for_tilize(_m, _n):
                return ttnn.DRAM_MEMORY_CONFIG

            configuration.create_dram_sharded_mem_config = _interleaved_dram_for_tilize  # type: ignore[method-assign]
        try:
            self.llama_4_scaling_beta = llama_4_scaling_beta
            self.original_max_position_embeddings = original_max_position_embeddings
            super().__init__(*args, **kwargs)
            self._use_devstral_custom_forward_decode = bool(patch_dram_tilize) and bool(
                devstral2_large_multi_device_dram_mitigation(mesh_device, getattr(self, "args", None))
            )
            _decode_rope = self.rotary_embedding_decode
            _prefill_rope = self.rotary_embedding_prefill

            def rotary_embedding_decode_wrapped(q, k, rot_mats, current_pos):
                q, k = _decode_rope(q, k, rot_mats, current_pos)
                return self._apply_llama4_query_scale_decode(q, current_pos), k

            def rotary_embedding_prefill_wrapped(q, k, rot_mats):
                q, k = _prefill_rope(q, k, rot_mats)
                return self._apply_llama4_query_scale_prefill(q), k

            self.rotary_embedding_decode = rotary_embedding_decode_wrapped
            self.rotary_embedding_prefill = rotary_embedding_prefill_wrapped
        finally:
            if patch_dram_tilize and orig_create_dram_sharded is not None:
                configuration.create_dram_sharded_mem_config = orig_create_dram_sharded  # type: ignore[method-assign]

        # Without this patch, prefill WO keeps TTNN defaults: out_block_h=per_core_M (often 1),
        # out_block_w=per_core_N (~48) → L1 CB clash on wide multi-device (BH or WH T3K).
        if devstral2_large_multi_device_dram_mitigation(mesh_device, getattr(self, "args", None)):
            _base_get_attn_wo = self.args.get_attn_wo_program_config

            def _get_attn_wo_wide_prefill(mode, seq_len=1, prefetcher=None):
                cfg = _base_get_attn_wo(mode, seq_len, prefetcher)
                if mode != Mode.PREFILL:
                    return cfg
                return _tighten_wo_prefill_prog_cfg(cfg)

            self.args.get_attn_wo_program_config = _get_attn_wo_wide_prefill  # type: ignore[method-assign]

        # Widen in0_block_w for QKV prefill: shared config defaults to 1 (K=12288 → 384 loop
        # iterations). in0_block_w=2 halves that, improving compute pipeline utilization.
        # Bind the real ModelArgs method before reassignment; use a distinct closure cell name so
        # we never recurse if this block is edited alongside other get_* patches.
        _base_get_attn_qkv = self.args.get_attn_qkv_program_config

        def _get_attn_qkv_wide_prefill(mode, seq_len=1, prefetcher=None):
            if patch_dram_tilize and mode == Mode.DECODE and prefetcher is None:
                cfg = _base_get_attn_qkv(Mode.PREFILL, self.args.tile_padded_batch_rows, prefetcher)
                return _widen_qkv_prefill_in0_block_w(cfg)
            cfg = _base_get_attn_qkv(mode, seq_len, prefetcher)
            if mode != Mode.PREFILL:
                return cfg
            return _widen_qkv_prefill_in0_block_w(cfg)

        self.args.get_attn_qkv_program_config = _get_attn_qkv_wide_prefill  # type: ignore[method-assign]

        if patch_dram_tilize:
            _base_get_attn_qkv_mm = self.args.get_attn_qkv_mm_mem_config

            def _get_attn_qkv_mm_interleaved_decode(mode, prefetcher=None):
                if mode == Mode.DECODE and prefetcher is None:
                    return ttnn.DRAM_MEMORY_CONFIG
                return _base_get_attn_qkv_mm(mode, prefetcher)

            self.args.get_attn_qkv_mm_mem_config = _get_attn_qkv_mm_interleaved_decode  # type: ignore[method-assign]

    def _llama4_scaling_enabled(self) -> bool:
        return self.llama_4_scaling_beta is not None and self.original_max_position_embeddings is not None

    def _llama4_scale_factor_from_positions_ttnn(self, pos_tt: ttnn.Tensor) -> ttnn.Tensor:
        """
        ``scaling = 1 + beta * log(1 + floor(pos / original_max_position_embeddings))`` (float32),
        same as HF ``get_llama_4_attn_scale``, shape matches ``pos_tt``.
        """
        orig = float(self.original_max_position_embeddings)
        beta = float(self.llama_4_scaling_beta)
        pos_f = ttnn.typecast(pos_tt, ttnn.float32)
        ratio = ttnn.divide(pos_f, orig)
        floored = ttnn.floor(ratio)
        log_term = ttnn.log1p(floored)
        scaled = ttnn.mul(log_term, beta)
        ones = ttnn.ones_like(pos_f)
        return ttnn.add(ones, scaled)

    def _reshape_decode_positions(self, current_pos: ttnn.Tensor, batch_dim: int) -> ttnn.Tensor:
        """Return positions with shape ``[1, batch_dim]`` for per-row scale (matches ``q`` batch axis)."""
        sh = tuple(current_pos.shape)
        numel = math.prod(sh) if sh else 0
        if numel == batch_dim:
            return ttnn.reshape(current_pos, (1, batch_dim))
        if numel > batch_dim:
            flat = ttnn.reshape(current_pos, (1, 1, 1, numel))
            sliced = flat[:, :, :, :batch_dim]
            return ttnn.reshape(sliced, (1, batch_dim))
        raise ValueError(f"Ministral Llama-4 scale: current_pos has {numel} elements but q_heads batch is {batch_dim}")

    def _apply_llama4_query_scale_decode(self, q_heads, current_pos):
        if not self._llama4_scaling_enabled():
            return q_heads
        b = int(q_heads.shape[1])
        pos_row = self._reshape_decode_positions(current_pos, b)
        scale_f = self._llama4_scale_factor_from_positions_ttnn(pos_row)
        scale_4d = ttnn.reshape(scale_f, (1, b, 1, 1))
        scale_tt = ttnn.typecast(scale_4d, ttnn.bfloat16)

        q_bf16 = q_heads if q_heads.dtype == ttnn.bfloat16 else ttnn.typecast(q_heads, dtype=ttnn.bfloat16)
        out = ttnn.mul(q_bf16, scale_tt, dtype=ttnn.bfloat16)
        if q_heads.dtype != ttnn.bfloat16:
            out = ttnn.typecast(out, dtype=q_heads.dtype)
        ttnn.deallocate(scale_tt)
        if q_bf16 is not q_heads:
            ttnn.deallocate(q_bf16)
        return out

    def _apply_llama4_query_scale_prefill(self, q_heads):
        if not self._llama4_scaling_enabled():
            return q_heads
        pos_tt = getattr(self, "_ministral_prefill_position_ids_tt", None)
        if pos_tt is None:
            return q_heads

        sh = tuple(q_heads.shape)
        if len(sh) != 4:
            return q_heads
        batch_dim, seq_dim = sh[0], sh[2]
        shp = tuple(pos_tt.shape)
        if len(shp) == 2 and shp[0] == batch_dim and shp[1] == seq_dim:
            scale_f = self._llama4_scale_factor_from_positions_ttnn(pos_tt)
        elif len(shp) == 1 and shp[0] == seq_dim and batch_dim == 1:
            scale_f = self._llama4_scale_factor_from_positions_ttnn(ttnn.reshape(pos_tt, (1, seq_dim)))
        else:
            return q_heads

        scale_4d = ttnn.reshape(scale_f, (batch_dim, 1, seq_dim, 1))
        scale_tt = ttnn.typecast(scale_4d, ttnn.bfloat16)

        q_bf16 = q_heads if q_heads.dtype == ttnn.bfloat16 else ttnn.typecast(q_heads, dtype=ttnn.bfloat16)
        out = ttnn.mul(q_bf16, scale_tt, dtype=ttnn.bfloat16)
        if q_heads.dtype != ttnn.bfloat16:
            out = ttnn.typecast(out, dtype=q_heads.dtype)
        ttnn.deallocate(scale_tt)
        if q_bf16 is not q_heads:
            ttnn.deallocate(q_bf16)
        return out

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        page_table=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        """Decode path: TG, prefetcher, and fused AG+MM defer to :meth:`Attention.forward_decode`."""
        if (
            self.prefetcher is not None
            or self.use_fused_all_gather_matmul
            or self.TG
            or not getattr(self, "_use_devstral_custom_forward_decode", False)
        ):
            return super().forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)
        return self._forward_decode_devstral_optimized(x, current_pos, rot_mats, page_table, kv_cache)

    def _forward_decode_devstral_optimized(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats,
        page_table,
        kv_cache,
    ) -> ttnn.Tensor:
        """Same pipeline as ``Attention.forward_decode`` through head concat; tailored WO + reduce tail.

        QKV still uses patched ``get_attn_*`` (prefill-style multicast + DRAM MM) from ``__init__``.
        After ``nlp_concat_heads_decode`` we avoid ``tt_all_gather`` + DRAM-sharded WO: L1 interleaved
        activations, one multicast ``ttnn.linear`` with interleaved ``wo``, then width-shard for
        ``tt_all_reduce`` (matches the generic path's layout for the reduce).
        """
        xqkv_fused_sharded = ttnn.linear(
            x,
            self.wqkv,
            memory_config=self.args.get_attn_qkv_mm_mem_config(Mode.DECODE, self.prefetcher),
            program_config=self.args.get_attn_qkv_program_config(Mode.DECODE, 1, self.prefetcher),
            compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg,
            dtype=self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16,
            global_cb=self.prefetcher.global_cb if self.prefetcher is not None else None,
            sub_device_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
        )
        if self.wqkv_bias_decode:
            num_tiles = int(math.ceil(xqkv_fused_sharded.shape[-2] / self.tile_size))
            xqkv_fused_sharded = xqkv_fused_sharded + self.wqkv_bias_decode[num_tiles - 1]

        ttnn.deallocate(x)
        qkv_all_reduce_mem_cfg = self.args.get_attn_qkv_all_reduce_output_mem_config(
            Mode.DECODE, list(self.mesh_device.shape)[1], self.prefetcher
        )
        xqkv_fused = tt_all_reduce(
            xqkv_fused_sharded,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            memory_config=qkv_all_reduce_mem_cfg
            if qkv_all_reduce_mem_cfg is not None
            else xqkv_fused_sharded.memory_config(),
            sharded=True,
            dtype=self.ccl_dtype,
            topology=self.ccl_topology,
            subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
        )
        if self.TG:
            xqkv_fused = ttnn.matmul(
                self.slice_mat,
                xqkv_fused,
                dtype=ttnn.bfloat16,
                memory_config=self.args.get_attn_create_head_input_mem_config(Mode.DECODE),
            )
        else:
            if self.prefetcher is None:
                # QKV matmul may be DRAM interleaved (Devstral DRAM-tilize mitigation); only sharded tensors
                # may use sharded_to_interleaved — otherwise the op can leave an unallocated shell tensor.
                reduced = xqkv_fused
                if reduced.is_sharded():
                    xqkv_fused = ttnn.sharded_to_interleaved(reduced, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
                    ttnn.deallocate(xqkv_fused_sharded)
                else:
                    xqkv_fused = ttnn.to_memory_config(reduced, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
                    ttnn.deallocate(xqkv_fused_sharded)
            else:
                xqkv_fused = xqkv_fused_sharded
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(
            xqkv_fused, (1, 1, self.batch_size_per_device_group, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3])
        )

        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=self.args.get_attn_create_head_output_mem_config(Mode.DECODE, self.prefetcher),
        )
        norm_config = self.args.get_norm_config("attn", Mode.DECODE, None)
        q_heads_pre_rot_1BQD = self.q_norm(q_heads_pre_rot_1BQD, mode=Mode.DECODE, norm_config=norm_config)
        k_heads_pre_rot_1BKD = self.k_norm(k_heads_pre_rot_1BKD, mode=Mode.DECODE, norm_config=norm_config)
        ttnn.deallocate(xqkv_fused)

        q_heads_1BQD, k_heads_1BKD = self.rotary_embedding_decode(
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos
        )

        ttnn.deallocate(q_heads_pre_rot_1BQD)
        ttnn.deallocate(k_heads_pre_rot_1BKD)
        if kv_cache:
            keys = kv_cache[0]
            values = kv_cache[1]
        else:
            keys = self.layer_past[0]
            values = self.layer_past[1]

        if self.use_qk_fused:
            ttnn.experimental.paged_fused_update_cache(
                keys, k_heads_1BKD, values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
        else:
            ttnn.experimental.paged_update_cache(
                keys, k_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)
        sdpa_decode_prog_cfg = self.args.get_attn_sdpa_decode_program_config(self.prefetcher)
        if page_table is not None:
            attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                page_table_tensor=page_table,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                sliding_window_size=self.sliding_window,
                program_config=sdpa_decode_prog_cfg,
                compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                sliding_window_size=self.sliding_window,
                program_config=sdpa_decode_prog_cfg,
                compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        ttnn.deallocate(q_heads_1BQD)
        attn_output_11BH = ttnn.to_memory_config(
            attn_output_1G4D,
            memory_config=self.args.get_attn_sdpa_output_mem_config(
                Mode.DECODE, self.batch_size_per_device_group, self.prefetcher
            ),
        )

        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_11BH,
            num_heads=self.n_local_heads,
            sub_core_grids=self.prefetcher.all_worker_cores_range_set if self.prefetcher is not None else None,
        )
        ttnn.deallocate(attn_output_11BH)
        ttnn.deallocate(attn_output_1G4D)

        attn_l1 = ttnn.to_memory_config(attn_output_cat, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_output_cat)

        wo_pc = _decode_wo_interleaved_multicast_prog_cfg(self.args)
        # Interleaved L1: do not ``to_memory_config`` into ``get_residual_mem_config(DECODE)`` here —
        # that spec shards **per-device** hidden width; WO output is full ``dim`` and triggers
        # ``shard_grid_fit_error`` (e.g. 128 width shards vs 32 cores).
        dense_out = ttnn.linear(
            attn_l1,
            self.wo,
            program_config=wo_pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
        )
        ttnn.deallocate(attn_l1)

        dense_out_reduced = tt_all_reduce(
            dense_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=0 if (self.TG and self.hidden_size < 8192) else 3,
            topology=self.ccl_topology,
            memory_config=self.args.get_attn_all_reduce_output_mem_config(
                Mode.DECODE, self.hidden_size, list(self.mesh_device.shape)[0], self.prefetcher
            ),
            sharded=False,
            dtype=self.ccl_dtype,
            use_composite=True if self.hidden_size == 8192 else False,
            subdevice_id=self.prefetcher.worker_sub_device_id if self.prefetcher is not None else None,
        )

        if not self.TG:
            # Same reason as above: avoid residual width-shard on full-width tensor; decoder will
            # ``to_memory_config`` via ``tt_ministral3_decode_mem_config`` when needed.
            dense_out_reduced = ttnn.to_memory_config(dense_out_reduced, ttnn.L1_MEMORY_CONFIG)

        return dense_out_reduced

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        position_ids: ttnn.Tensor | None = None,
    ):
        """
        Optional ``position_ids``: device ``ttnn.Tensor`` of integer positions, shape ``[batch, seq]``
        (or ``[seq]`` when ``batch == 1``), dtype typically ``uint32`` / ``int32``. Required for
        Llama-4 Q scaling on prefill when scaling is enabled.
        """
        self._ministral_prefill_position_ids_tt = position_ids
        try:
            return super().forward_prefill(
                x_11SH,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
            )
        finally:
            self._ministral_prefill_position_ids_tt = None

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        position_ids=None,
    ):
        if mode == Mode.PREFILL:
            return self.forward_prefill(
                x,
                rot_mats,
                user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
                position_ids=position_ids,
            )
        return super().forward(
            x,
            current_pos,
            rot_mats,
            user_id=user_id,
            mode=mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )


TtDevstral2LargeAttention.__name__ = "Attention"
TtDevstral2LargeAttention.__qualname__ = "Attention"

TtMinistralAttention = TtDevstral2LargeAttention

__all__ = [
    "TtDevstral2LargeAttention",
    "TtMinistralAttention",
]

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GQA Attention for GLM-4.7-REAP on TT hardware.

Standard Grouped Query Attention (96 Q heads, 8 KV heads, head_dim=128)
with QK norm, partial RoPE (partial_rotary_factor=0.5), and QKV bias.

Reference: tt_transformers/tt/attention.py (Llama/Qwen GQA pattern).
NOT derived from glm4_moe_lite (which uses MLA).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm

from models.demos.glm4_moe.tt.config import Glm4MoeHParams
from models.demos.glm4_moe.tt.layer_weights import DecoderLayerTTWeights


import os
import logging

_REDUCE_IMPL = os.environ.get("GLM4_MOE_REDUCE_IMPL", "host").strip().lower()
_REDUCE_LOG_ONCE = set()

logger = logging.getLogger(__name__)


def _simple_all_reduce_host(tensor, mesh_device, cluster_axis, memory_config=None):
    """Host-side all-reduce fallback for TG 2D mesh.

    Reads device tensors to CPU, sums, and replicates back. Correct but slow.
    """
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    orig_layout = tensor.layout
    orig_dtype = tensor.dtype

    mesh_rows, mesh_cols = list(mesh_device.shape)
    dev_tensors = ttnn.get_device_tensors(tensor)
    num_devs = len(dev_tensors)

    if cluster_axis == 0:
        col0_indices = list(range(0, num_devs, mesh_cols))
        host_sum = ttnn.to_torch(dev_tensors[col0_indices[0]].cpu())
        for idx in col0_indices[1:]:
            host_sum = host_sum + ttnn.to_torch(dev_tensors[idx].cpu())
    elif cluster_axis == 1:
        row0_indices = list(range(0, mesh_cols))
        host_sum = ttnn.to_torch(dev_tensors[row0_indices[0]].cpu())
        for idx in row0_indices[1:]:
            host_sum = host_sum + ttnn.to_torch(dev_tensors[idx].cpu())
    else:
        raise ValueError(f"cluster_axis must be 0 or 1, got {cluster_axis}")

    ttnn.deallocate(tensor, force=False)
    result = ttnn.from_torch(
        host_sum,
        device=mesh_device,
        dtype=orig_dtype,
        layout=orig_layout,
        memory_config=mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return result


def _simple_all_reduce(tensor, mesh_device, cluster_axis, memory_config=None, ccl=None):
    """All-reduce with configurable implementation via GLM4_MOE_REDUCE_IMPL env var.

    Implementations:
      "host"     — Host-side CPU fallback (default, proven correct)
      "native"   — ttnn.all_reduce(cluster_axis=...) on-device
      "rs_ag"    — ttnn.reduce_scatter + ttnn.all_gather 2-step decomposition
    """
    if mesh_device.__class__.__name__ != "MeshDevice":
        return tensor
    mesh_shape = list(mesh_device.shape)
    if mesh_shape == [1, 1]:
        return tensor

    # Save input logical shape — CCL ops (reduce_scatter/all_gather) can lose it,
    # returning physical-padded dims instead of the true logical shape.
    input_logical_shape = [int(d) for d in tensor.shape]

    impl = _REDUCE_IMPL
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG

    # Log once per (impl, axis) combination
    log_key = (impl, cluster_axis)
    if log_key not in _REDUCE_LOG_ONCE:
        _REDUCE_LOG_ONCE.add(log_key)
        logger.info("_simple_all_reduce: impl=%s, cluster_axis=%d, mesh=%s", impl, cluster_axis, mesh_shape)

    result = None

    if impl == "native":
        # On-device all_reduce with explicit Linear topology.
        result = ttnn.all_reduce(
            tensor,
            cluster_axis=cluster_axis,
            memory_config=mc,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(tensor, force=False)

    elif impl == "native_auto":
        # On-device all_reduce with auto topology (let system choose).
        result = ttnn.all_reduce(
            tensor,
            cluster_axis=cluster_axis,
            memory_config=mc,
        )
        ttnn.deallocate(tensor, force=False)

    elif impl == "full_ar":
        # Single all_reduce across ALL devices (no cluster_axis).
        # Works on TG 2D mesh — internally iterates all dims.
        result = ttnn.all_reduce(
            tensor,
            memory_config=mc,
        )
        ttnn.deallocate(tensor, force=False)

    elif impl == "rs_ag":
        # 2-step: reduce_scatter then all_gather along same axis.
        # reduce_scatter sums and scatters along dim=3 (hidden), then all_gather restores.
        scattered = ttnn.reduce_scatter(
            tensor,
            dim=3,
            cluster_axis=cluster_axis,
            memory_config=mc,
        )
        ttnn.deallocate(tensor, force=False)
        result = ttnn.all_gather(
            scattered,
            dim=3,
            cluster_axis=cluster_axis,
            memory_config=mc,
        )
        ttnn.deallocate(scattered, force=False)

    elif impl == "rs_ag_async":
        # Async 2-step with CCL semaphore management (trace-compatible).
        # Uses ttnn.experimental.reduce_scatter_minimal_async + all_gather_async.
        if ccl is None:
            # Fallback to sync rs_ag if CCL not initialized.
            scattered = ttnn.reduce_scatter(tensor, dim=3, cluster_axis=cluster_axis, memory_config=mc)
            ttnn.deallocate(tensor, force=False)
            result = ttnn.all_gather(scattered, dim=3, cluster_axis=cluster_axis, memory_config=mc)
            ttnn.deallocate(scattered, force=False)
        else:
            rs_params = ccl.get_ccl_params_for_reduce_scatter(axis=cluster_axis)
            scattered = ttnn.experimental.reduce_scatter_minimal_async(
                tensor,
                dim=3,
                cluster_axis=cluster_axis,
                memory_config=mc,
                topology=ttnn.Topology.Linear,
                **rs_params,
            )
            ttnn.deallocate(tensor, force=False)

            ag_params = ccl.get_ccl_params_for_all_gather(axis=cluster_axis)
            result = ttnn.experimental.all_gather_async(
                scattered,
                dim=3,
                cluster_axis=cluster_axis,
                memory_config=mc,
                topology=ttnn.Topology.Linear,
                **ag_params,
            )
            ttnn.deallocate(scattered, force=False)

    else:
        # Default: host-side fallback (proven correct)
        result = _simple_all_reduce_host(tensor, mesh_device, cluster_axis, memory_config)

    # CCL ops (reduce_scatter, all_gather, all_reduce) can lose the logical shape,
    # returning physical-padded dims (e.g., TILE_LAYOUT pads batch to 32) instead of
    # the true logical shape. Restore it so downstream ops (ttnn.add, etc.) don't fail
    # with "Invalid subtile broadcast type" due to mismatched batch dims.
    # PERF: Use int comparison to avoid false positives from ttnn Shape objects.
    # For bs=1, CCL ops don't change the shape — skip reshape entirely.
    out_shape = [int(d) for d in result.shape]
    if out_shape != input_logical_shape:
        out_vol = 1
        in_vol = 1
        for d in out_shape:
            out_vol *= d
        for d in input_logical_shape:
            in_vol *= d
        if out_vol == in_vol:
            # Same volume — safe to reshape (just metadata change).
            result = ttnn.reshape(result, input_logical_shape, out_shape)
        else:
            # CCL padded a dimension (e.g., batch 1→32 in TILE_LAYOUT).
            # Slice back to the original logical shape.
            result = ttnn.slice(
                result,
                starts=[0] * len(input_logical_shape),
                ends=input_logical_shape,
            )

    return result


def _simple_all_gather(tensor, mesh_device, cluster_axis, dim, memory_config=None):
    """Simple all-gather using ttnn.all_gather (no semaphore management).

    For initial bringup — can be replaced with optimized CCL wrappers later.
    """
    if mesh_device.__class__.__name__ != "MeshDevice":
        return tensor
    mesh_shape = list(mesh_device.shape)
    if mesh_shape == [1, 1]:
        return tensor
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    return ttnn.all_gather(
        tensor,
        dim=dim,
        num_links=1,
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Linear,
        memory_config=mc,
    )


class Glm4MoeAttention(LightweightModule):
    """GQA attention for GLM-4.7-REAP-218B on Galaxy Wormhole (TG mesh, 32 chips).

    Architecture:
    - 96 Q heads, 8 KV heads (GQA ratio 12:1), head_dim=128
    - TP=8 along columns -> 12 Q heads + 1 KV head per device
    - Fused QKV weight [5120, 1792] per device
    - QKV bias (attention_bias=True)
    - QK norm (RMSNorm per head, dim=128)
    - Partial RoPE: partial_rotary_factor=0.5 -> rotary_dim=64, pass_dim=64
    - Paged KV cache: separate K and V, dtype BF8
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        layer_weights: DecoderLayerTTWeights,
        hparams: Glm4MoeHParams,
        configuration: Any,
        paged_attention_config=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.hparams = hparams

        # Core dimensions
        self.n_heads = hparams.num_attention_heads  # 96
        self.n_kv_heads = hparams.num_key_value_heads  # 8
        self.head_dim = hparams.head_dim  # 128
        self.hidden_size = hparams.hidden_size  # 5120

        # Device topology
        self.num_devices = configuration["num_devices"]  # 32 for Galaxy TG
        self.TG = self.num_devices == 32
        self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices  # 8
        self.num_device_groups = self.num_devices // self.num_devices_per_group  # 4

        # Per-device head counts (TP=8)
        self.n_local_heads = self.n_heads // self.num_devices_per_group  # 12
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group  # 1

        # Batch sizing
        self.max_batch_size = configuration["max_batch_size"]
        self.batch_size_per_device_group = (
            max(self.max_batch_size // self.num_device_groups, 1) if self.TG else self.max_batch_size
        )

        # Partial RoPE
        self.partial_rotary_factor = hparams.partial_rotary_factor  # 0.5
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)  # 64
        self.pass_dim = self.head_dim - self.rotary_dim  # 64

        # Scale for SDPA
        self.scale = self.head_dim**-0.5

        # Paged attention config
        self.paged_attention_config = paged_attention_config

        # Sequence length limits
        self.MAX_QKV_MM_SEQ_LEN = configuration.get("MAX_QKV_MM_SEQ_LEN", 4096)

        # CCL dtype (bfloat16 for TG linear topology)
        self.ccl_dtype = configuration.get("ccl_dtype", ttnn.bfloat16)

        # Compute kernel config — HiFi2 for attention precision through 92 layers
        import os
        _attn_fidelity_raw = os.environ.get("GLM4_MOE_ATTN_FIDELITY", "hifi2").strip().lower()
        _fidelity_map = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2,
                         "hifi3": ttnn.MathFidelity.HiFi3, "hifi4": ttnn.MathFidelity.HiFi4}
        _attn_fidelity = _fidelity_map.get(_attn_fidelity_raw, ttnn.MathFidelity.HiFi2)
        _attn_approx = os.environ.get("GLM4_MOE_ATTN_APPROX", "0").strip() != "0"
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=_attn_fidelity,
            math_approx_mode=_attn_approx,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Weights from layer_weights (already fused and sharded)
        self.wqkv = layer_weights.w_qkv  # [1, 1, 5120, 1792] per device (TP-sharded)
        self.wqkv_bias = layer_weights.w_qkv_bias  # [1, 1, 1, 1792] per device (TP-sharded)
        self.wo = layer_weights.w_o  # [1, 1, 1536, 5120] per device (row-parallel)

        # QK norm (RMSNorm, per-head dim=128, replicated)
        self.q_norm = layer_weights.q_norm
        self.k_norm = layer_weights.k_norm

        # Per-batch-bucket HEIGHT_SHARDED memory configs for decode mode.
        # "auto" L1_HEIGHT_SHARDED_MEMORY_CONFIG doesn't work with to_memory_config —
        # must use explicit shard specs + interleaved_to_sharded.
        # Configs are built lazily per batch bucket and cached in dicts.
        self._grid_size = mesh_device.compute_with_storage_grid_size()
        self._shard_cfg_cache: dict[int, dict] = {}  # batch -> {q, k, rope, trans}

        # TG user selection matrices (for slicing batch from all-reduce output)
        if self.TG:
            B = self.max_batch_size  # logical total batch (e.g. 4, 8, 32)
            Bg = self.batch_size_per_device_group  # per DP group (B // 4)
            Ng = self.num_device_groups  # 4 for Galaxy TG
            # Physical batch dim in TILE_LAYOUT is always padded to multiples of 32.
            # slice_mat width must match physical_batch from WIDTH_SHARDED QKV output.
            B_phys = ((B + 31) // 32) * 32  # tile-padded batch (always 32 for B<=32)

            # slice_mat: [1, num_devices, Bg, B_phys] — each device selects its DP group's batch entries
            weight = torch.zeros(1, self.num_devices, Bg, B_phys)
            for i in range(self.num_devices):
                col = i % Ng
                weight[:, i, :, col * Bg : (col + 1) * Bg] = torch.eye(Bg)

            self.slice_mat = ttnn.from_torch(
                weight,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            )
            # user_selection_matrix: [B_phys, B_phys] — reorders batch after all-gather across DP groups
            user_selection_matrix = torch.eye(Bg, Bg)
            user_selection_matrix = torch.nn.functional.pad(user_selection_matrix, (0, B_phys - Bg), "constant", 0)
            user_selection_matrix = [user_selection_matrix] * Ng
            user_selection_matrix = torch.block_diag(*user_selection_matrix)
            self.user_selection_matrix = ttnn.from_torch(
                user_selection_matrix,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

    def _get_shard_cfgs(self, batch: int) -> dict:
        """Return per-batch shard configs (lazily created and cached)."""
        cached = self._shard_cfg_cache.get(batch)
        if cached is not None:
            return cached
        b = max(batch, 1)
        user_grid = ttnn.num_cores_to_corerangeset(b, self._grid_size, row_wise=True)
        cfgs = {
            "q": ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.head_dim),
                core_grid=user_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
            "k": ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.head_dim),
                core_grid=user_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
        }
        self._shard_cfg_cache[batch] = cfgs
        return cfgs

    def _apply_partial_rope_decode(self, x, cos, sin, trans_mat):
        """Apply partial rotary embedding (decode mode, NeoX-style).

        Only the first rotary_dim (64) dims get rotary encoding;
        the remaining pass_dim (64) dims are passed through unchanged.

        Uses NeoX-style rotation: rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2]).
        GLM-4.7 uses NeoX-style RoPE (confirmed from HuggingFace transformers glm4_moe).
        """
        # x: [1, batch, n_heads, head_dim] = [1, B, H, 128]  (DRAM interleaved after QK norm)
        batch = int(x.shape[1])
        n_heads = int(x.shape[2])

        # Slice rotary and pass-through portions
        x_rot = ttnn.slice(x, [0, 0, 0, 0], [1, batch, n_heads, self.rotary_dim])
        x_pass = ttnn.slice(x, [0, 0, 0, self.rotary_dim], [1, batch, n_heads, self.head_dim])

        # NeoX-style rotate_half: [-x2, x1] where x1 = first half, x2 = second half
        half = self.rotary_dim // 2
        x1 = ttnn.slice(x_rot, [0, 0, 0, 0], [1, batch, n_heads, half])
        x2 = ttnn.slice(x_rot, [0, 0, 0, half], [1, batch, n_heads, self.rotary_dim])
        rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # output = x_rot * cos + rotate_half(x_rot) * sin
        # cos/sin: [1, batch, 1, rotary_dim] — broadcasts over n_heads dim
        x_rot = ttnn.add(
            ttnn.multiply(x_rot, cos),
            ttnn.multiply(rotated, sin),
        )
        ttnn.deallocate(rotated)

        # Concat back: [rotary_dim | pass_dim] = full head_dim
        x = ttnn.concat([x_rot, x_pass], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(x_rot)
        ttnn.deallocate(x_pass)
        return x

    def _apply_partial_rope_prefill(self, x, cos, sin, trans_mat):
        """Apply partial rotary embedding (prefill mode, NeoX-style).

        x: [1, n_heads, seq_len, head_dim]
        cos/sin: [1, 1, padded_len, rotary_dim] — sliced to seq_len, broadcasts over n_heads.
        """
        seq_len = x.shape[2]
        n_heads = x.shape[1]

        x_rot = ttnn.slice(x, [0, 0, 0, 0], [1, n_heads, seq_len, self.rotary_dim])
        x_pass = ttnn.slice(x, [0, 0, 0, self.rotary_dim], [1, n_heads, seq_len, self.head_dim])

        if x_rot.dtype != ttnn.bfloat16:
            x_rot = ttnn.typecast(x_rot, dtype=ttnn.bfloat16)

        # Slice cos/sin to match seq_len (may be padded to max_seq_len)
        cos_sl = ttnn.slice(cos, [0, 0, 0, 0], [1, 1, seq_len, self.rotary_dim])
        sin_sl = ttnn.slice(sin, [0, 0, 0, 0], [1, 1, seq_len, self.rotary_dim])

        # NeoX-style rotate_half: [-x2, x1] where x1 = first half, x2 = second half
        half = self.rotary_dim // 2
        x1 = ttnn.slice(x_rot, [0, 0, 0, 0], [1, n_heads, seq_len, half])
        x2 = ttnn.slice(x_rot, [0, 0, 0, half], [1, n_heads, seq_len, self.rotary_dim])
        rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # output = x_rot * cos + rotate_half(x_rot) * sin
        x_rot = ttnn.add(
            ttnn.multiply(x_rot, cos_sl),
            ttnn.multiply(rotated, sin_sl),
        )
        ttnn.deallocate(rotated)

        x = ttnn.concat([x_rot, x_pass], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(x_rot)
        ttnn.deallocate(x_pass)
        return x

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        page_table=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        """Decode forward: single token per sequence.

        Args:
            x: [seq_len=1, 1, batch, hidden_size=5120]
            current_pos: [batch_size] current token positions
            rot_mats: (cos, sin) rotation matrices for RoPE
            page_table: page table tensor for paged KV cache
            kv_cache: [keys, values] external KV cache tensors

        Returns:
            Output tensor [1, 1, batch, hidden_size=5120]
        """

        # 1. QKV linear
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=self.ccl_dtype if self.TG else ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Add QKV bias
        xqkv = xqkv + self.wqkv_bias

        ttnn.deallocate(x)

        # Column-parallel QKV: each device has its own output slice, no all-reduce needed.
        xqkv = ttnn.to_memory_config(xqkv, ttnn.DRAM_MEMORY_CONFIG)

        # True logical batch from input
        active_batch = int(x.shape[-2])
        dp_size = self.num_device_groups  # 4 for TG
        tg_batch_sliced = self.TG and active_batch > 1 and active_batch % dp_size == 0

        # TG: slice batch to this DP group's entries via slice_mat matmul.
        # slice_mat expects xqkv dim[-2] = B_phys (tile-padded batch = 32).
        # With bs<32, logical dim[-2]<32 but TILE_LAYOUT pads to 32 physically.
        # Reshape to expose physical batch as logical so matmul inner dims match.
        if tg_batch_sliced:
            # Multi-user: slice batch to this DP group entries via matmul.
            xqkv_shape = [int(d) for d in xqkv.shape]
            B_phys = ((self.max_batch_size + 31) // 32) * 32
            if xqkv_shape[-2] != B_phys:
                xqkv = ttnn.reshape(xqkv, (1, 1, B_phys, xqkv_shape[-1]))
            xqkv = ttnn.matmul(
                self.slice_mat,
                xqkv,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            logical_batch_after_slice = active_batch // dp_size
        else:
            # Single-user or unsupported batch size: skip slice_mat.
            xqkv = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
            logical_batch_after_slice = active_batch

        # 3. Reshape xqkv so nlp_create_qkv_heads_decode sees the true logical batch,
        #    not the tile-padded physical batch (32).  Without this, the op outputs
        #    q/k/v with batch=32 which breaks paged_update_cache (page_table has batch=1).
        #    Mirrors tt_transformers/tt/attention.py lines 497-501.
        _fqkv_shape = xqkv.shape
        xqkv = ttnn.reshape(
            xqkv,
            (1, 1, logical_batch_after_slice, int(_fqkv_shape[3])),
            (1, 1, 32, int(_fqkv_shape[3])),
        )

        # Split into Q, K, V heads (output must be HEIGHT_SHARDED per ttnn API)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=self.n_local_heads,  # 12
            num_kv_heads=self.n_local_kv_heads,  # 1
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )

        ttnn.deallocate(xqkv)

        # 4. QK Norm (RMSNorm per head, dim=128)
        # RMSNorm requires interleaved input (HEIGHT_SHARDED not supported by layernorm op)
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        q = self.q_norm(q, mode="decode")
        k = self.k_norm(k, mode="decode")

        # 5. Partial RoPE (rotary_dim=64 of head_dim=128)
        q = self._apply_partial_rope_decode(q, rot_mats[0], rot_mats[1], rot_mats[2])
        k = self._apply_partial_rope_decode(k, rot_mats[0], rot_mats[1], rot_mats[2])

        # Convert back to HEIGHT_SHARDED for paged_update_cache and SDPA
        # (partial RoPE returns DRAM interleaved after concat)
        _shard_cfgs = self._get_shard_cfgs(logical_batch_after_slice)
        q = ttnn.interleaved_to_sharded(q, _shard_cfgs["q"])
        k = ttnn.interleaved_to_sharded(k, _shard_cfgs["k"])

        # 6. KV cache update
        keys = kv_cache[0]
        values = kv_cache[1]

        ttnn.experimental.paged_update_cache(
            keys, k, update_idxs_tensor=current_pos, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            values, v, update_idxs_tensor=current_pos, page_table=page_table
        )
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # 7. SDPA (paged)
        attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            keys,
            values,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)

        # 8. Concat heads (requires HEIGHT_SHARDED input)
        attn_output = ttnn.interleaved_to_sharded(attn_output, _shard_cfgs["q"])
        attn_output = ttnn.experimental.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.n_local_heads,  # 12
        )
        # -> [1, 1, batch, 1536] (12 heads * 128 dim)

        # 9. TG: apply user_selection_matrix to reorder batch entries after concat_heads.
        # Only needed for multi-user: reorders batch entries from DP groups.
        if tg_batch_sliced:
            attn_output = ttnn.to_memory_config(attn_output, ttnn.L1_MEMORY_CONFIG)
            attn_shape = [int(d) for d in attn_output.shape]
            B_phys = ((self.max_batch_size + 31) // 32) * 32
            if attn_shape[-2] != B_phys:
                attn_output = ttnn.reshape(attn_output, (1, 1, B_phys, attn_shape[-1]))
            attn_output = ttnn.matmul(
                self.user_selection_matrix,
                attn_output,
                core_grid=ttnn.CoreGrid(y=4, x=8),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            attn_output = ttnn.reshape(attn_output, (1, 1, active_batch, attn_shape[-1]))

        # 10. Output projection (BF16 output for precision through 92 layers)
        dense_out = ttnn.matmul(
            attn_output,
            self.wo,
            core_grid=ttnn.CoreGrid(y=4, x=8) if self.TG else None,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn_output)

        # 11. All-reduce on cluster_axis=0 (TP reduction for row-parallel O projection)
        dense_out = _simple_all_reduce(
            dense_out,
            self.mesh_device,
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ccl=self.tt_ccl,
        )

        return dense_out

    def forward_prefill(
        self,
        x: ttnn.Tensor,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        """Prefill forward: process entire prompt sequence.

        Args:
            x: [1, 1, seq_len, hidden_size=5120]
            rot_mats: (cos, sin) rotation matrices for RoPE
            user_id: batch index for KV cache fill
            page_table: page table tensor for paged KV cache
            chunk_page_table: page table for chunked prefill
            chunk_start_idx: starting index for chunked prefill
            kv_cache: [keys, values] external KV cache tensors

        Returns:
            Output tensor [1, 1, seq_len, hidden_size=5120]
        """
        seq_len = x.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        # 1. QKV linear (reshape for long sequences)
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            if seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {self.MAX_QKV_MM_SEQ_LEN}")
            x = ttnn.reshape(x, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        xqkv = ttnn.linear(
            x,
            self.wqkv,
            dtype=self.ccl_dtype if self.TG else ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Add QKV bias
        xqkv = xqkv + self.wqkv_bias

        # Column-parallel QKV: each device has its own output slice, no all-reduce needed.

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv = ttnn.reshape(xqkv, [1, 1, seq_len, -1])

        ttnn.deallocate(x)

        # 3. Split into Q, K, V heads (prefill)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.n_local_heads,  # 12
            num_kv_heads=self.n_local_kv_heads,  # 1
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # q: [1, 12, seq_len, 128], k: [1, 1, seq_len, 128], v: [1, 1, seq_len, 128]

        ttnn.deallocate(xqkv)

        # 4. QK Norm
        q = self.q_norm(q, mode="prefill")
        k = self.k_norm(k, mode="prefill")

        # 5. Partial RoPE (prefill mode)
        q = self._apply_partial_rope_prefill(q, rot_mats[0], rot_mats[1], rot_mats[2])
        k = self._apply_partial_rope_prefill(k, rot_mats[0], rot_mats[1], rot_mats[2])

        # 6. Fill KV cache
        keys = kv_cache[0]
        values = kv_cache[1]

        k_8b = ttnn.typecast(k, dtype=keys.dtype)
        ttnn.deallocate(k)
        v_8b = ttnn.typecast(v, dtype=values.dtype)
        ttnn.deallocate(v)

        # For TG with multi-user batch split across DP groups, select the correct
        # column's tensors for KV cache fill. But for batch=1 (tg_batch_sliced=False),
        # ALL 32 devices participate in decode and need the cache filled.
        # K,V are identical across DP columns (same TP position data), so filling
        # all devices directly is correct and avoids device-mapping issues.
        tg_batch_sliced = self.TG and self.max_batch_size > self.batch_size_per_device_group
        if self.TG and tg_batch_sliced:
            k_fill = self._prefill_prepare_tensor_for_kv_cache(k_8b, user_id)
            v_fill = self._prefill_prepare_tensor_for_kv_cache(v_8b, user_id)
        else:
            k_fill = k_8b
            v_fill = v_8b

        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        if fill_page_table is not None:
            block_size = keys.shape[2]
            page_len = fill_page_table.shape[1] * block_size
            k_fill_sliced = k_fill[:, :, :page_len, :] if page_len < k_fill.shape[2] else k_fill
            v_fill_sliced = v_fill[:, :, :page_len, :] if page_len < v_fill.shape[2] else v_fill
            ttnn.experimental.paged_fill_cache(keys, k_fill_sliced, fill_page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values, v_fill_sliced, fill_page_table, batch_idx=user_id)
        else:
            ttnn.fill_cache(keys, k_fill, user_id % self.batch_size_per_device_group)
            ttnn.fill_cache(values, v_fill, user_id % self.batch_size_per_device_group)

        # 7. SDPA — match Q dtype to KV cache dtype (avoid BF8 precision loss for BF16 KV)
        q_8b = ttnn.typecast(q, dtype=keys.dtype)
        ttnn.deallocate(q)

        if chunk_start_idx is not None:
            attn_output = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q_8b,
                input_tensor_k=keys,
                input_tensor_v=values,
                page_table_tensor=fill_page_table,
                chunk_start_idx=chunk_start_idx,
            )
        else:
            attn_output = ttnn.transformer.scaled_dot_product_attention(
                q_8b,
                k_8b,
                v_8b,
                is_causal=True,
                scale=self.scale,
            )

        ttnn.deallocate(q_8b)
        ttnn.deallocate(k_8b)
        ttnn.deallocate(v_8b)

        # 8. Reshape and concat heads
        attn_output = ttnn.reshape(attn_output, [1, self.n_local_heads, -1, self.head_dim])
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # -> [1, 1, seq_len, 1536]

        # Reshape for long sequences
        if seq_len > 1024:
            attn_output = ttnn.reshape(attn_output, [1, seq_len // 1024, 1024, -1])

        # 9. Output projection (BF16 output for precision through 92 layers)
        output = ttnn.linear(
            attn_output,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        if seq_len > 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output)

        # 10. All-reduce on cluster_axis=0 (TP reduction for row-parallel O projection)
        # NOTE: Prefill runs outside trace — do NOT pass ccl to avoid async CCL ops
        # conflicting with the existing captured trace's semaphore state.
        output = _simple_all_reduce(
            output,
            self.mesh_device,
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return output

    def forward(
        self,
        x,
        current_pos=None,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        if mode == "prefill":
            return self.forward_prefill(
                x,
                rot_mats,
                user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
            )
        else:
            return self.forward_decode(
                x,
                current_pos,
                rot_mats,
                page_table=page_table,
                kv_cache=kv_cache,
            )

    def _prefill_prepare_tensor_for_kv_cache(self, key_or_value_layer, user_id):
        """For TG: select tensors from the correct column chips for KV cache fill."""
        tensor_copy = ttnn.clone(key_or_value_layer)
        tensors = ttnn.get_device_tensors(tensor_copy)
        single_column_tensors = tensors[user_id // self.batch_size_per_device_group :: 4]
        multi_device_tensor = ttnn.combine_device_tensors(tensors=single_column_tensors)
        return multi_device_tensor

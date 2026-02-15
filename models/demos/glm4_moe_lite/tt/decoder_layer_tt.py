# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import time
from typing import Any

import torch

import ttnn

from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams


def _profile_add(profile: dict[str, float] | None, key: str, elapsed_s: float) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0)) + float(elapsed_s)


def prepare_decode_rope_inputs_for_rotary_llama_decode_mode_tt(
    *,
    device: Any,
    cos_batch: ttnn.Tensor,  # [1,1,B,rope_dim] TILE
    sin_batch: ttnn.Tensor,  # [1,1,B,rope_dim] TILE
    trans_matrix: ttnn.Tensor,  # [1,1,32,32] TILE
    batch: int,
    rope_dim: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.MemoryConfig]:
    """Prepare HEIGHT_SHARDED cos/sin/trans inputs for decode-mode rotary_embedding_llama."""
    batch = int(batch)
    rope_dim = int(rope_dim)
    if batch <= 0:
        raise ValueError("batch must be > 0")
    if rope_dim <= 0:
        raise ValueError("rope_dim must be > 0")

    grid_size = device.compute_with_storage_grid_size()
    user_grid = ttnn.num_cores_to_corerangeset(int(batch), grid_size, row_wise=True)

    rope_sharded_cfg = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, rope_dim),
        core_grid=user_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Match the DeepSeek decode pattern:
    # - Input cos/sin from gather/embedding is [1, 1, B, rope_dim]
    # - Decode RoPE kernel expects batch in dim=1: [1, B, 1[32], rope_dim]
    #
    # IMPORTANT: Use interleaved_to_sharded (not to_memory_config) to avoid
    # sharding alignment failures for small batches.
    cos_decode = ttnn.transpose(cos_batch, 1, 2)  # [1, B, 1[32], rope_dim]
    sin_decode = ttnn.transpose(sin_batch, 1, 2)  # [1, B, 1[32], rope_dim]
    cos_decode = ttnn.interleaved_to_sharded(cos_decode, rope_sharded_cfg)
    sin_decode = ttnn.interleaved_to_sharded(sin_decode, rope_sharded_cfg)

    trans_mat_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        core_grid=user_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    trans_decode = ttnn.repeat(trans_matrix, ttnn.Shape((1, 1, batch, 1)))
    trans_decode = ttnn.interleaved_to_sharded(trans_decode, trans_mat_mem_config)

    return cos_decode, sin_decode, trans_decode, rope_sharded_cfg


def _mesh_shape(device: Any) -> tuple[int, int]:
    if device.__class__.__name__ != "MeshDevice":
        return (1, 1)
    return (int(device.shape[0]), int(device.shape[1]))


def _tp_cluster_axis(device: Any) -> int | None:
    """Return the mesh axis used for TP-style sharding (preferred: cols)."""
    if device.__class__.__name__ != "MeshDevice":
        return None
    mesh_rows, mesh_cols = _mesh_shape(device)
    if mesh_cols > 1:
        return 1
    if mesh_rows > 1:
        return 0
    return None


def _moe_sparse_tokens_multiple(*, device: Any, moe_runtime: Any) -> int:
    """Return the minimum per-device token multiple required by sparse MoE.

    Sparse experts require global token count (`tokens_per_device * num_dispatch_devices`)
    divisible by `sparsity_block_size`. This computes the minimal per-device multiple.
    """
    block = max(1, int(getattr(moe_runtime, "sparsity_block_size", 32)))
    dispatch_axis = int(getattr(moe_runtime, "dispatch_cluster_axis", 0))
    mesh_rows, mesh_cols = _mesh_shape(device)
    dispatch_devices = int((mesh_rows, mesh_cols)[dispatch_axis])
    dispatch_devices = max(1, dispatch_devices)
    return max(1, block // math.gcd(block, dispatch_devices))


def _parse_math_fidelity(value: str, *, default: ttnn.MathFidelity) -> ttnn.MathFidelity:
    raw = value.strip().lower()
    if not raw:
        return default
    table = {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
        "hifi3": ttnn.MathFidelity.HiFi3,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }
    return table.get(raw, default)


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


@torch.no_grad()
def prepare_decode_rope_and_positions_tt(
    *,
    device: Any,
    rope: dict[str, ttnn.Tensor],
    positions: torch.Tensor,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Prepare shared decode inputs used across all layers for a decode step.

    Returns:
    - tt_positions: TT int32 [B]
    - cos_batch: TT bf16 [1, 1, B, rope_dim]
    - sin_batch: TT bf16 [1, 1, B, rope_dim]
    """
    if positions.dtype not in (torch.int32, torch.int64):
        positions = positions.to(torch.int32)
    if positions.ndim != 1:
        raise ValueError(f"expected positions shape [B], got {tuple(positions.shape)}")

    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

    tt_positions = ttnn.from_torch(
        positions.to(torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # Fetch per-position RoPE cos/sin rows.
    #
    # Important for multi-device correctness:
    # - `ttnn.gather` on MeshDevice has been observed to be fragile for small decode
    #   batches, and it also builds a large repeated index tensor on host.
    # - `ttnn.embedding` is the DeepSeek implementation pattern and yields the exact
    #   [1, batch, rope_dim] rows we need without host-side index expansion.
    positions_clamped = positions.to(torch.int32).clamp_min(0)
    rope_dim = int(rope["cos_matrix"].shape[3])

    batch = int(positions_clamped.shape[0])
    padded_batch = ((batch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    if padded_batch != batch:
        positions_padded = torch.nn.functional.pad(
            positions_clamped.view(1, batch),
            (0, padded_batch - batch),
            "constant",
            0,
        )
    else:
        positions_padded = positions_clamped.view(1, batch)

    rot_idxs = ttnn.from_torch(
        positions_padded.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # embedding([1, B], [1,1,S,D]) -> [1, B, D]
    cos_rows = ttnn.embedding(rot_idxs, rope["cos_matrix"], layout=ttnn.TILE_LAYOUT)
    sin_rows = ttnn.embedding(rot_idxs, rope["sin_matrix"], layout=ttnn.TILE_LAYOUT)

    # `ttnn.unsqueeze_to_4D` is a reshape which can behave like a view. Some
    # TTNN view-like ops do not refcount underlying buffers, so aggressively
    # deallocating the source tensor can lead to use-after-free and corrupt
    # RoPE inputs (observed as nondeterministic / garbled text output).
    # Materialize (clone) the final cos/sin batches before freeing intermediates.
    cos_batch_view = ttnn.unsqueeze_to_4D(cos_rows)  # [1,1,B_pad,D]
    sin_batch_view = ttnn.unsqueeze_to_4D(sin_rows)  # [1,1,B_pad,D]

    if padded_batch != batch:
        cos_batch_view = ttnn.slice(cos_batch_view, [0, 0, 0, 0], [1, 1, batch, rope_dim])
        sin_batch_view = ttnn.slice(sin_batch_view, [0, 0, 0, 0], [1, 1, batch, rope_dim])

    cos_batch = ttnn.clone(cos_batch_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sin_batch = ttnn.clone(sin_batch_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Best-effort cleanup of intermediate buffers (do not deallocate the cloned outputs).
    for t in (rot_idxs, cos_rows, sin_rows, cos_batch_view, sin_batch_view):
        try:
            ttnn.deallocate(t, force=False)
        except Exception:
            pass

    return tt_positions, cos_batch, sin_batch


def _shard_kvpe_update_tensor(
    *,
    device: Any,
    kvpe_new: ttnn.Tensor,
    batch: int,
    kvpe_dim: int,
    skip_defensive_clones: bool = False,
) -> ttnn.Tensor:
    """Transform KVPE update tensor into the sharded layout required by paged_update_cache."""
    # `paged_update_cache` expects the update tensor to be sharded.
    #
    # Correctness: `ttnn.pad` and some view-like ops can alias the input buffer without
    # increasing refcounts. Make the padded/permuted update tensor own its buffer to
    # avoid intermittent use-after-free corruption in decode.
    if skip_defensive_clones:
        # Skip clones: views flow directly through pad -> permute -> shard.
        # kvpe_new must stay alive until paged_update_cache consumes the sharded result.
        kvpe_padded = ttnn.pad(kvpe_new, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)  # [1,32,B,kvpe_dim]
        kvpe_perm = ttnn.permute(kvpe_padded, (0, 2, 1, 3))  # [1,B,32,kvpe_dim]
    else:
        kvpe_padded_view = ttnn.pad(kvpe_new, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)  # [1,32,B,kvpe_dim]
        kvpe_padded = ttnn.clone(kvpe_padded_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # NOTE: kvpe_padded_view may alias kvpe_new; do not deallocate it separately.
        kvpe_perm_view = ttnn.permute(kvpe_padded, (0, 2, 1, 3))  # [1,B,32,kvpe_dim]
        kvpe_perm = ttnn.clone(kvpe_perm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # NOTE: kvpe_perm_view may alias kvpe_padded; do not deallocate it separately.
        ttnn.deallocate(kvpe_padded, force=False)

    # Shard across the (B*32) height dimension so each user gets one 32xkvpe_dim shard.
    grid_size = device.compute_with_storage_grid_size()
    user_grid = ttnn.num_cores_to_corerangeset(int(batch), grid_size, row_wise=True)
    kvpe_sharded_cfg = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, int(kvpe_dim)),
        core_grid=user_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    if skip_defensive_clones:
        kvpe_sharded = ttnn.to_memory_config(kvpe_perm, kvpe_sharded_cfg)
    else:
        kvpe_sharded_view = ttnn.to_memory_config(kvpe_perm, kvpe_sharded_cfg)
        kvpe_sharded = ttnn.clone(kvpe_sharded_view, memory_config=kvpe_sharded_cfg)
        # NOTE: kvpe_sharded_view may alias kvpe_perm; do not deallocate it separately.
        ttnn.deallocate(kvpe_perm, force=False)
    return kvpe_sharded


@torch.no_grad()
def run_decoder_layer_decode_one_step_update_cache_tt(
    *,
    device: Any,
    x_embed_tok: ttnn.Tensor,
    tt_positions: ttnn.Tensor,
    page_table_tt: ttnn.Tensor,
    kvpe_cache: ttnn.Tensor,
    cos_batch: ttnn.Tensor,
    sin_batch: ttnn.Tensor,
    trans_matrix: ttnn.Tensor,
    cos_decode: ttnn.Tensor | None = None,
    sin_decode: ttnn.Tensor | None = None,
    trans_decode: ttnn.Tensor | None = None,
    rope_sharded_cfg: ttnn.MemoryConfig | None = None,
    w: Any,
    hparams: Glm4MoeLiteHParams,
    moe_runtime: Any | None = None,
    profile: dict[str, float] | None = None,
    use_decode_rope: bool = False,
) -> ttnn.Tensor:
    """Run one decode step for a single decoder layer and update its KVPE cache.

    Inputs:
    - x_embed_tok: [1, 1, B, hidden] tiled, batch is in dim=2 (seq axis)
    - tt_positions: TT int32 [B] row-major, absolute positions for each batch slot
    - page_table_tt: [B, max_num_blocks_per_req] int32 row-major on device
    - kvpe_cache: [num_blocks, 1, block_size, kvpe_dim] on device
    - cos_batch/sin_batch/trans_matrix: RoPE tensors for the per-user positions
    - w: weights object with RMSNorms and projection weights (duck-typed)

    Returns:
    - x_out: [1, 1, B, hidden] tiled
    """
    # Decode batch size lives in the "sequence" axis (dim=2) for TT decode
    # convention: [1, 1, B, hidden].
    batch = int(x_embed_tok.shape[2])
    if batch <= 0:
        raise ValueError("batch must be > 0")

    if os.environ.get("GLM4_MOE_LITE_LAYER_IDENTITY", "").strip() == "1":
        # Debug-only: bypass the entire decoder layer (including KV cache update)
        # to isolate nondeterminism outside the attention/MLP stack.
        return ttnn.clone(x_embed_tok, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t_layer0 = time.perf_counter() if profile is not None else 0.0

    # Paged ops expect row-major positions.
    assert tt_positions.layout == ttnn.ROW_MAJOR_LAYOUT, "tt_positions must be ROW_MAJOR_LAYOUT"
    kvpe_dim = int(hparams.kv_lora_rank + hparams.qk_rope_head_dim)
    rope_dim = int(hparams.qk_rope_head_dim)
    if rope_dim <= 0:
        raise ValueError("rope_dim must be > 0")
    if use_decode_rope and rope_dim % ttnn.TILE_SIZE != 0:
        raise ValueError(f"decode RoPE requires rope_dim divisible by {ttnn.TILE_SIZE}, got rope_dim={rope_dim}")

    owns_decode_rope_inputs = False
    if use_decode_rope:
        any_provided = cos_decode is not None or sin_decode is not None or trans_decode is not None
        if any_provided:
            if cos_decode is None or sin_decode is None or trans_decode is None:
                raise ValueError("cos_decode, sin_decode, and trans_decode must be provided together")
            if rope_sharded_cfg is None:
                rope_sharded_cfg = cos_decode.memory_config()
        else:
            cos_decode, sin_decode, trans_decode, rope_sharded_cfg = (
                prepare_decode_rope_inputs_for_rotary_llama_decode_mode_tt(
                    device=device,
                    cos_batch=cos_batch,
                    sin_batch=sin_batch,
                    trans_matrix=trans_matrix,
                    batch=batch,
                    rope_dim=rope_dim,
                )
            )
            owns_decode_rope_inputs = True

        def _rope_decode(t: ttnn.Tensor, *, heads: int) -> ttnn.Tensor:
            # Input t expected in [1, heads, B, rope_dim]. RoPE decode kernel
            # parallelizes over batch (dim=1) and expects HEIGHT_SHARDED inputs
            # with head tiles padded to 32.
            nonlocal rope_sharded_cfg
            assert rope_sharded_cfg is not None
            if int(t.shape[-1]) != rope_dim:
                raise ValueError(f"rope tensor dim mismatch: expected {rope_dim}, got {int(t.shape[-1])}")

            # Move batch into dim=1 for decode mode: [1, B, heads, rope_dim].
            t = ttnn.permute(t, (0, 2, 1, 3))

            # Pad heads up to TILE_SIZE so shard shape height is 32.
            heads = int(heads)
            if heads <= 0:
                raise ValueError("heads must be > 0")
            if heads > ttnn.TILE_SIZE:
                raise ValueError(f"decode RoPE only supports heads <= {ttnn.TILE_SIZE}, got heads={heads}")
            pad_h = ttnn.TILE_SIZE - heads
            if pad_h:
                t = ttnn.pad(t, [(0, 0), (0, 0), (0, pad_h), (0, 0)], 0)

            t = ttnn.to_memory_config(t, rope_sharded_cfg)
            t = ttnn.experimental.rotary_embedding_llama(
                t,
                cos_decode,
                sin_decode,
                trans_decode,
                is_decode_mode=True,
            )

            # Bring output back to interleaved for downstream ops.
            t = ttnn.to_memory_config(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            if pad_h:
                t = ttnn.slice(t, [0, 0, 0, 0], [1, batch, heads, rope_dim])
            t = ttnn.permute(t, (0, 2, 1, 3))  # [1, heads, B, rope_dim]
            return t

    # Optional precision knob for MLP/router matmuls during bring-up. This is
    # intentionally coarse-grained: we prefer correctness over speed here.
    mlp_compute_kernel_config = None
    if os.environ.get("GLM4_MOE_LITE_MOE_FP32_ACC", "").strip() == "1":
        mlp_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    else:
        # LoFi + packer_l1_acc for speed (matches DeepSeek V3 pattern).
        # Fidelity can be overridden via GLM4_MOE_LITE_MLP_FIDELITY env var.
        mlp_fidelity = _parse_math_fidelity(
            os.environ.get("GLM4_MOE_LITE_MLP_FIDELITY", ""),
            default=ttnn.MathFidelity.LoFi,
        )
        mlp_approx = os.environ.get("GLM4_MOE_LITE_MLP_APPROX", "1").strip() != "0"
        mlp_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=mlp_fidelity,
            math_approx_mode=mlp_approx,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    # Use L1 for decode intermediate activations when enabled (reduces DRAM round-trips).
    decode_act_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("GLM4_MOE_LITE_DECODE_L1_ACT", "").strip() == "1" else None

    # ---- Explicit 1D matmul program config (zero-overhead decode optimization) ----
    # When enabled, passes MatmulMultiCoreReuseMultiCast1DProgramConfig to ttnn.linear
    # calls with M <= 1 tile (decode path). This tells the hardware to distribute the
    # output N dimension across more cores without requiring any weight resharding.
    explicit_prog_cfg = _env_bool("GLM4_MOE_LITE_EXPLICIT_PROG_CFG")
    skip_defensive_clones = _env_bool("GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES")
    concat_heads = _env_bool("GLM4_MOE_LITE_CONCAT_HEADS")
    attn_dp = _env_bool("GLM4_MOE_LITE_ATTN_DP")
    fuse_mlp_moe_reduce = _env_bool("GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE")

    def _compute_1d_prog_cfg(b_weight: ttnn.Tensor, m_total: int) -> Any:
        """Compute 1D multicast program config for decode matmuls."""
        K = int(b_weight.shape[-2])
        N = int(b_weight.shape[-1])
        m_tiles = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
        k_tiles = (K + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        n_tiles = (N + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        grid = device.compute_with_storage_grid_size()
        grid_x, grid_y = int(grid.x), int(grid.y)
        num_cores = grid_x * grid_y

        if n_tiles < num_cores:
            # Reduce grid to avoid idle cores (following reference pattern).
            grid_y = max(1, n_tiles // grid_x)
            num_cores = grid_x * grid_y

        per_core_N = max(1, math.ceil(n_tiles / num_cores))

        # in0_block_w must divide k_tiles exactly.  Find the largest valid
        # divisor of k_tiles up to 4 (e.g. kv_lora_rank=576 → k_tiles=18,
        # 18%4≠0 so we fall back to 2 since 18%2==0).
        in0_bw = 1
        for candidate in (4, 3, 2):
            if k_tiles % candidate == 0:
                in0_bw = candidate
                break

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_bw,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=m_tiles,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def _mlp_linear(a: ttnn.Tensor, b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig | None = None) -> ttnn.Tensor:
        kwargs: dict[str, object] = {}
        mc = memory_config if memory_config is not None else decode_act_mc
        if mc is not None:
            kwargs["memory_config"] = mc
        if mlp_compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = mlp_compute_kernel_config
        if explicit_prog_cfg:
            # Only apply 1D prog cfg when M fits in a single tile (decode case)
            # AND weight tensor has batch==1.  Skip for 4D matmuls (e.g. kv_b1/kv_b2
            # with [1,H,B,K] inputs and per-head weights with H>1 batch dim).
            m_total = 1
            for i in range(len(a.shape) - 1):
                m_total *= int(a.shape[i])
            b_batch = 1
            for i in range(len(b.shape) - 2):
                b_batch *= int(b.shape[i])
            if m_total <= ttnn.TILE_SIZE and b_batch == 1:
                kwargs["program_config"] = _compute_1d_prog_cfg(b, m_total)
        return ttnn.linear(a, b, **kwargs)

    tp_axis = _tp_cluster_axis(device)
    tp_enabled = tp_axis is not None and os.environ.get("GLM4_MOE_LITE_TP", "").strip() == "1"
    mesh_rows, mesh_cols = _mesh_shape(device)
    tp_size = int((mesh_rows, mesh_cols)[tp_axis]) if tp_axis is not None else 1

    def _tp_row_parallel_linear_from_replicated(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        """Row-parallel matmul helper for TP-sharded weights.

        `layer_weights.py` shards attention weights along the input dim (dim=2 of [1,1,in,out]).
        To make shapes match, partition the activation's last dim across the same TP axis.
        The per-device outputs are partial dot products and must be summed across devices.
        """
        a_tp = ttnn.mesh_partition(a, dim=3, cluster_axis=tp_axis)
        out = _mlp_linear(a_tp, b)
        ttnn.deallocate(a_tp, force=False)
        out_reduced = ttnn.all_reduce(
            out,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=tp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(out, force=False)
        return out_reduced

    # ---- DRAM-sharded matmul for attention projections (decode-only perf optimization) ----
    # When enabled, attention weights are stored in DRAM WIDTH_SHARDED format (across all
    # DRAM banks), and matmuls use a DRAM-sharded program config that reads weights with
    # full DRAM bandwidth. This is the key optimization from DeepSeek V3 for M=1 decode.
    dram_sharded_enabled = _env_bool("GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS")
    dram_sharded_attn = dram_sharded_enabled and _env_bool("GLM4_MOE_LITE_DRAM_SHARDED_ATTN")
    # MLP DRAM sharding is ON by default when main flag is set; opt out with MLP=0.
    dram_sharded_mlp = dram_sharded_enabled and os.environ.get("GLM4_MOE_LITE_DRAM_SHARDED_MLP", "1").strip() != "0"
    # Standalone sharded MLP flag (no dependency on DRAM_SHARDED_WEIGHTS master switch).
    sharded_mlp = _env_bool("GLM4_MOE_LITE_SHARDED_MLP")
    dram_sharded_mlp = dram_sharded_mlp or sharded_mlp

    if dram_sharded_enabled or sharded_mlp:
        from models.demos.deepseek_v3.utils.config_helpers import (
            get_activation_sharding_core_counts_for_dram_matmul,
            get_dram_sharded_matmul_config,
        )

        _ds_grid = device.compute_with_storage_grid_size()
        _ds_max_cores = _ds_grid.x * _ds_grid.y
        _DS_BATCH = 32  # padded batch size for decode (TILE_SIZE)

        _ds_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        def _ds_act_mc(width: int) -> ttnn.MemoryConfig:
            """Create WIDTH_SHARDED L1 activation config for a given width dimension."""
            cores = max(get_activation_sharding_core_counts_for_dram_matmul(width, _ds_max_cores))
            return ttnn.create_sharded_memory_config_(
                shape=(_DS_BATCH, width // cores),
                core_grid=ttnn.num_cores_to_corerangeset(
                    cores,
                    ttnn.CoreCoord(_ds_grid.x, _ds_grid.y),
                    row_wise=True,
                ),
                strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                tile_layout=True,
                use_height_and_width_as_shard_shape=True,
            )

        def _dram_sharded_linear(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
            """DRAM-sharded matmul for decode. Weight b must be in DRAM WIDTH_SHARDED format."""
            K = int(b.shape[2])
            N = int(b.shape[3])
            input_cores = max(get_activation_sharding_core_counts_for_dram_matmul(K, _ds_max_cores))
            output_cores = max(get_activation_sharding_core_counts_for_dram_matmul(N, _ds_max_cores))

            a_sharded = ttnn.to_memory_config(a, _ds_act_mc(K))

            prog_cfg = get_dram_sharded_matmul_config(
                m=_DS_BATCH, k=K, n=N,
                input_num_shards=input_cores,
                output_num_shards=output_cores,
            )

            result = ttnn.linear(
                a_sharded, b,
                program_config=prog_cfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=_ds_ckc,
            )
            ttnn.deallocate(a_sharded, force=False)
            result_dram = ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(result, force=False)
            return result_dram

        def _dram_sharded_mlp(
            x: ttnn.Tensor,
            w_gate: ttnn.Tensor,
            w_up: ttnn.Tensor,
            w_down: ttnn.Tensor,
        ) -> ttnn.Tensor:
            """Fused gate→silu→up→mul→down MLP entirely in L1 WIDTH_SHARDED.

            Follows DeepSeek V3 decode MLP pattern: reshard input once, keep all
            intermediates in L1 WIDTH_SHARDED, only move final output to DRAM.
            This eliminates 4 DRAM round-trips compared to calling _dram_sharded_linear
            three times independently.
            """
            K_gate = int(w_gate.shape[2])
            N_gate = int(w_gate.shape[3])
            K_down = int(w_down.shape[2])
            N_down = int(w_down.shape[3])

            input_cores = max(get_activation_sharding_core_counts_for_dram_matmul(K_gate, _ds_max_cores))
            inner_cores = max(get_activation_sharding_core_counts_for_dram_matmul(N_gate, _ds_max_cores))
            output_cores = max(get_activation_sharding_core_counts_for_dram_matmul(N_down, _ds_max_cores))

            # 1. Reshard input to L1 WIDTH_SHARDED once (shared between gate and up).
            x_sharded = ttnn.to_memory_config(x, _ds_act_mc(K_gate))

            gate_cfg = get_dram_sharded_matmul_config(
                m=_DS_BATCH, k=K_gate, n=N_gate,
                input_num_shards=input_cores, output_num_shards=inner_cores,
            )

            # 2. Gate projection → stays in L1 WIDTH_SHARDED.
            gate = ttnn.linear(
                x_sharded, w_gate,
                program_config=gate_cfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=_ds_ckc,
            )

            # 3. Up projection → stays in L1 WIDTH_SHARDED (reuses x_sharded!).
            up = ttnn.linear(
                x_sharded, w_up,
                program_config=gate_cfg,  # same shape as gate
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=_ds_ckc,
            )
            ttnn.deallocate(x_sharded, force=False)

            # 4. fused silu(gate) * up → stays in L1 WIDTH_SHARDED.
            x_ff = ttnn.mul(gate, up, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            ttnn.deallocate(gate, force=False)
            ttnn.deallocate(up, force=False)

            # 5. Down projection → L1 WIDTH_SHARDED output.
            down_cfg = get_dram_sharded_matmul_config(
                m=_DS_BATCH, k=K_down, n=N_down,
                input_num_shards=inner_cores, output_num_shards=output_cores,
            )
            result = ttnn.linear(
                x_ff, w_down,
                program_config=down_cfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=_ds_ckc,
            )
            ttnn.deallocate(x_ff, force=False)

            # 6. Move final result to DRAM (needed for all_reduce / residual add).
            result_dram = ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(result, force=False)
            return result_dram

    def _attn_linear(a: ttnn.Tensor, b: ttnn.Tensor, *, force_no_tp: bool = False) -> ttnn.Tensor:
        """Attention projection linear (2D weights only). Uses DRAM-sharded when enabled.

        When force_no_tp=True, skip mesh_partition and all_reduce (weight is replicated).
        """
        use_tp = tp_enabled and not force_no_tp
        if dram_sharded_attn:
            if use_tp:
                a_tp = ttnn.mesh_partition(a, dim=3, cluster_axis=tp_axis)
                out = _dram_sharded_linear(a_tp, b)
                ttnn.deallocate(a_tp, force=False)
                out_reduced = ttnn.all_reduce(
                    out, num_links=1, topology=ttnn.Topology.Linear,
                    cluster_axis=tp_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(out, force=False)
                return out_reduced
            else:
                return _dram_sharded_linear(a, b)
        else:
            if use_tp:
                return _tp_row_parallel_linear_from_replicated(a, b)
            else:
                return _mlp_linear(a, b)

    residual = x_embed_tok
    t0 = time.perf_counter() if profile is not None else 0.0
    x = w.input_layernorm(x_embed_tok, mode="decode")  # [1,1,B,hidden]
    _profile_add(profile, "norm_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- KVPE for new token -> update cache at cur_pos ----
    skip_kv_update = os.environ.get("GLM4_MOE_LITE_SKIP_KV_UPDATE", "").strip() == "1"
    q_a = None
    qkv = None
    if not skip_kv_update:
        t0 = time.perf_counter() if profile is not None else 0.0
        kv = None
        qkv = None
        w_q_kv_a = getattr(w, "w_q_kv_a", None)
        if w_q_kv_a is not None:
            qkv = _attn_linear(x, w_q_kv_a, force_no_tp=attn_dp)  # [1,1,B,q_lora_rank+kvpe_dim]

            if skip_defensive_clones:
                # Skip clone: use slice results directly; keep qkv alive until q_a and kv are consumed.
                q_a = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, batch, int(hparams.q_lora_rank)])
                kv = ttnn.slice(
                    qkv,
                    [0, 0, 0, int(hparams.q_lora_rank)],
                    [1, 1, batch, int(hparams.q_lora_rank) + kvpe_dim],
                )
                # qkv stays alive — deallocated later after q_a and kv are consumed
            else:
                # `slice` may return a view that aliases the `qkv` buffer (no refcounting).
                # Materialize slices before freeing `qkv` to avoid intermittent corruption.
                q_a_view = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, batch, int(hparams.q_lora_rank)])
                kv_view = ttnn.slice(
                    qkv,
                    [0, 0, 0, int(hparams.q_lora_rank)],
                    [1, 1, batch, int(hparams.q_lora_rank) + kvpe_dim],
                )
                q_a = ttnn.clone(q_a_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                kv = ttnn.clone(kv_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                # NOTE: `q_a_view`/`kv_view` may alias `qkv`; do not deallocate them separately.
                ttnn.deallocate(qkv, force=False)
                qkv = None
        else:
            kv = _attn_linear(x, w.w_kv_a, force_no_tp=attn_dp)  # [1,1,B,kvpe_dim]

        # `slice` may alias `kv`. Clone slices before freeing `kv`.
        if skip_defensive_clones:
            kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, batch, int(hparams.kv_lora_rank)])
            kv_rope = ttnn.slice(kv, [0, 0, 0, int(hparams.kv_lora_rank)], [1, 1, batch, kvpe_dim])
            # kv stays alive — deallocated after kv_nope and kv_rope are consumed by layernorm/RoPE/concat
        else:
            kv_nope_view = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, batch, int(hparams.kv_lora_rank)])
            kv_rope_view = ttnn.slice(kv, [0, 0, 0, int(hparams.kv_lora_rank)], [1, 1, batch, kvpe_dim])
            kv_nope = ttnn.clone(kv_nope_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kv_rope = ttnn.clone(kv_rope_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # NOTE: `kv_nope_view`/`kv_rope_view` may alias `kv`; do not deallocate them separately.
            ttnn.deallocate(kv, force=False)
            kv = None

        kv_nope = w.kv_a_layernorm(kv_nope, mode="decode")

        # RoPE op requires BF16.
        if kv_rope.dtype != ttnn.bfloat16:
            kv_rope = ttnn.typecast(kv_rope, dtype=ttnn.bfloat16)
        if use_decode_rope:
            kv_rope = _rope_decode(kv_rope, heads=1)
        else:
            kv_rope = ttnn.experimental.rotary_embedding_llama(
                kv_rope,
                cos_batch,
                sin_batch,
                trans_matrix,
                is_decode_mode=False,
            )  # [1,1,B,rope_dim]

        # Deferred kv deallocation: when skip_defensive_clones is True, kv_nope/kv_rope
        # were views of kv. Now that layernorm and RoPE have consumed them, kv can be freed.
        if kv is not None:
            ttnn.deallocate(kv, force=False)
            kv = None

        kvpe_new = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1,1,B,kvpe_dim]
        ttnn.deallocate(kv_nope, force=False)
        ttnn.deallocate(kv_rope, force=False)

        # Important: paged_update_cache requires the update tensor to be BF16/FP32,
        # even if the cache itself is BF8.
        kvpe_new_sharded = _shard_kvpe_update_tensor(device=device, kvpe_new=kvpe_new, batch=batch, kvpe_dim=kvpe_dim, skip_defensive_clones=skip_defensive_clones)

        # Multi-device correctness: when operating on a MeshDevice, ensure the
        # update is applied on all mesh coordinates that hold a replica of the
        # KV cache. DeepSeek passes `mesh_coords` explicitly; without it we've
        # observed KV block boundary corruption when the second KV page is first
        # touched (pos >= block_size).
        mesh_coords = None
        if device.__class__.__name__ == "MeshDevice":
            try:
                mesh_rows, mesh_cols = int(device.shape[0]), int(device.shape[1])
                mesh_coords = {ttnn.MeshCoordinate(r, c) for r in range(mesh_rows) for c in range(mesh_cols)}
            except Exception:
                mesh_coords = None

        if mesh_coords is None:
            ttnn.experimental.paged_update_cache(
                kvpe_cache,
                kvpe_new_sharded,
                update_idxs_tensor=tt_positions,
                page_table=page_table_tt,
            )
        else:
            ttnn.experimental.paged_update_cache(
                kvpe_cache,
                kvpe_new_sharded,
                update_idxs_tensor=tt_positions,
                page_table=page_table_tt,
                mesh_coords=mesh_coords,
            )
        if os.environ.get("GLM4_MOE_LITE_SYNC_AFTER_KV_UPDATE", "").strip() == "1":
            # Debug-only: enforce a barrier between KV cache update and subsequent
            # attention reads to rule out cross-queue hazards.
            ttnn.synchronize_device(device)
        ttnn.deallocate(kvpe_new_sharded, force=False)
        # NOTE: `ttnn.pad` can return a view which may alias the `kvpe_new` buffer.
        # Keep `kvpe_new` alive until after the update kernel is enqueued to avoid
        # use-after-free on some runtimes.
        ttnn.deallocate(kvpe_new, force=False)
        _profile_add(profile, "kv_cache_update_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- Q path ----
    t0 = time.perf_counter() if profile is not None else 0.0
    if q_a is None:
        q_a = _attn_linear(x, w.w_q_a, force_no_tp=attn_dp)  # [1,1,B,q_lora_rank]
    q_a = w.q_a_layernorm(q_a, mode="decode")
    # Deferred qkv deallocation: when skip_defensive_clones is True, q_a was a
    # view of qkv. Now that q_a_layernorm has consumed it, qkv can be freed.
    if qkv is not None:
        ttnn.deallocate(qkv, force=False)
        qkv = None
    q = _attn_linear(q_a, w.w_q_b, force_no_tp=attn_dp)  # [1,1,B,num_heads*qk_head_dim]
    ttnn.deallocate(q_a, force=False)

    q = ttnn.reshape(q, (1, batch, int(hparams.num_attention_heads), int(hparams.qk_head_dim)))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [1,H,B,qk_head_dim]

    # `slice` may alias `q` (no refcounting). Clone slices before freeing `q`.
    if skip_defensive_clones:
        q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, int(hparams.num_attention_heads), batch, int(hparams.qk_nope_head_dim)])
        q_rope = ttnn.slice(
            q,
            [0, 0, 0, int(hparams.qk_nope_head_dim)],
            [1, int(hparams.num_attention_heads), batch, int(hparams.qk_head_dim)],
        )
        # q stays alive — deallocated after q_nope and q_rope are consumed by kv_b1 matmul/RoPE
    else:
        q_nope_view = ttnn.slice(q, [0, 0, 0, 0], [1, int(hparams.num_attention_heads), batch, int(hparams.qk_nope_head_dim)])
        q_rope_view = ttnn.slice(
            q,
            [0, 0, 0, int(hparams.qk_nope_head_dim)],
            [1, int(hparams.num_attention_heads), batch, int(hparams.qk_head_dim)],
        )
        q_nope = ttnn.clone(q_nope_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_rope = ttnn.clone(q_rope_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # NOTE: `q_nope_view`/`q_rope_view` may alias `q`; do not deallocate them separately.
        ttnn.deallocate(q, force=False)
        q = None

    use_tp_kv_b1 = tp_enabled
    if use_tp_kv_b1:
        qk_nope = int(hparams.qk_nope_head_dim)
        qk_nope_per_shard = qk_nope // max(1, int(tp_size))
        if qk_nope % max(1, int(tp_size)) != 0 or qk_nope_per_shard % int(ttnn.TILE_SIZE) != 0:
            use_tp_kv_b1 = False
    if use_tp_kv_b1:
        q_nope = _tp_row_parallel_linear_from_replicated(q_nope, w.w_kv_b1)  # [1,H,B,kv_lora_rank]
    else:
        q_nope = _mlp_linear(q_nope, w.w_kv_b1)  # [1,H,B,kv_lora_rank]

    if q_rope.dtype != ttnn.bfloat16:
        q_rope = ttnn.typecast(q_rope, dtype=ttnn.bfloat16)
    if use_decode_rope:
        q_rope = _rope_decode(q_rope, heads=int(hparams.num_attention_heads))
    else:
        q_rope = ttnn.experimental.rotary_embedding_llama(
            q_rope,
            cos_batch,
            sin_batch,
            trans_matrix,
            is_decode_mode=False,
        )  # [1,H,B,rope_dim]

    if use_decode_rope:
        if owns_decode_rope_inputs:
            assert cos_decode is not None and sin_decode is not None and trans_decode is not None
            ttnn.deallocate(cos_decode, force=False)
            ttnn.deallocate(sin_decode, force=False)
            ttnn.deallocate(trans_decode, force=False)

    # Deferred q deallocation: when skip_defensive_clones is True, q_nope/q_rope
    # were views of q. Now that kv_b1 matmul and RoPE have consumed them, q can be freed.
    if q is not None:
        ttnn.deallocate(q, force=False)
        q = None

    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [1,H,B,kvpe_dim]
    ttnn.deallocate(q_nope, force=False)
    ttnn.deallocate(q_rope, force=False)

    # x no longer needed.
    ttnn.deallocate(x, force=False)

    # ---- FlashMLA decode ----
    # `permute` may return a view that aliases `q_kvpe` (no refcounting). Clone the
    # permuted tensor before freeing `q_kvpe` to avoid use-after-free in attention.
    if skip_defensive_clones:
        q_for_decode = ttnn.permute(q_kvpe, (0, 2, 1, 3))  # [1,B,H,kvpe_dim]
        # q_kvpe may be aliased; deallocate after FlashMLA consumes q_for_decode
    else:
        q_for_decode_view = ttnn.permute(q_kvpe, (0, 2, 1, 3))  # [1,B,H,kvpe_dim]
        q_for_decode = ttnn.clone(q_for_decode_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # NOTE: `q_for_decode_view` may alias `q_kvpe`; do not deallocate it separately.
        ttnn.deallocate(q_kvpe, force=False)
        q_kvpe = None
    _profile_add(profile, "q_path_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # MLA attention scores are computed from dot(q_kvpe, kvpe) where the dot-product
    # dimension is `kvpe_dim = kv_lora_rank + qk_rope_head_dim` (576 for GLM-4.7-Flash).
    #
    # Correctness: HF DeepseekV3Attention scales by 1/sqrt(qk_head_dim). Using the same
    # default here avoids decode drift that can eventually flip greedy tokens.
    # Keep `GLM4_MOE_LITE_MLA_SCALE_MODE=kvpe` as an escape hatch for experiments.
    # Keep scale consistent with layer0_tt + HF DeepseekV3Attention.
    # Default: 1/sqrt(qk_head_dim). Optional escape hatch: kvpe.
    scale_mode = os.environ.get("GLM4_MOE_LITE_MLA_SCALE_MODE", "qk").strip().lower()
    if scale_mode == "kvpe":
        scale = float(int(kvpe_dim) ** -0.5)
    else:
        scale = float(int(hparams.qk_head_dim) ** -0.5)
    # NOTE: k_chunk_size=128 has been observed to cause nondeterministic /
    # corrupted greedy decode on some stacks. Default to the smaller chunk
    # size for correctness and allow opt-in tuning via env var.
    try:
        k_chunk_size = int(os.environ.get("GLM4_MOE_LITE_MLA_K_CHUNK_SIZE", "64").strip() or "64")
    except ValueError:
        k_chunk_size = 64
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,  # not used in decode
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    mla_fidelity = _parse_math_fidelity(
        os.environ.get("GLM4_MOE_LITE_MLA_FIDELITY", ""),
        default=ttnn.MathFidelity.HiFi4,
    )
    mla_approx = os.environ.get("GLM4_MOE_LITE_MLA_APPROX", "0").strip() != "0"
    mla_fp32_acc_req = os.environ.get("GLM4_MOE_LITE_MLA_FP32_ACC", "").strip() == "1"
    # Known issue (bring-up): FP32 dest accumulation in FlashMLA decode has been
    # observed to corrupt greedy decode exactly when the 2nd KV block is first
    # touched (pos >= block_size). Keep it disabled unless explicitly overridden.
    mla_fp32_acc = mla_fp32_acc_req
    if mla_fp32_acc_req and os.environ.get("GLM4_MOE_LITE_UNSAFE_ALLOW_FP32_MLA", "").strip() != "1":
        logger.warning(
            "GLM4_MOE_LITE_MLA_FP32_ACC=1 is currently unsafe for FlashMLA decode (KV block boundary corruption). "
            "Forcing fp32_dest_acc_en=0. Set GLM4_MOE_LITE_UNSAFE_ALLOW_FP32_MLA=1 to override."
        )
        mla_fp32_acc = False
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=mla_fidelity,
        math_approx_mode=mla_approx,
        fp32_dest_acc_en=mla_fp32_acc,
        packer_l1_acc=_env_bool("GLM4_MOE_LITE_PACKER_L1_ACC", default=False),
    )

    # Prefer direct KVPE cache usage when supported by the kernel to avoid per-step
    # V-cache slicing overhead. Keep an opt-in fallback for older runtimes.
    t0 = time.perf_counter() if profile is not None else 0.0
    use_v_cache_slice = os.environ.get("GLM4_MOE_LITE_MLA_USE_V_CACHE_SLICE", "").strip() == "1"
    # Kernel contract: when Q is not sharded, FlashMLA decode currently requires
    # DRAM-interleaved Q. For bring-up, we support an opt-in sharded-Q path that
    # matches the DeepSeek MLA decode pattern more closely.
    flash_mla_memcfg = ttnn.DRAM_MEMORY_CONFIG
    shard_q = os.environ.get("GLM4_MOE_LITE_MLA_SHARD_Q", "").strip() == "1"
    if shard_q:
        grid_size = device.compute_with_storage_grid_size()
        num_cores = int(grid_size.x) * int(grid_size.y)
        height = int(batch) * int(hparams.num_attention_heads)
        width = int(kvpe_dim)
        kv_lora_rank = int(hparams.kv_lora_rank)

        # Shard along the flattened (B*H) dimension. Use as many cores as there are
        # head tiles, capped by the available core count.
        tiles_h = max(1, (height + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
        q_num_cores = min(tiles_h, max(1, num_cores))
        shard_h = (height + q_num_cores - 1) // q_num_cores
        shard_h = ((shard_h + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE

        q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)
        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(int(shard_h), int(width)),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        flash_mla_out_memcfg = ttnn.create_sharded_memory_config(
            shape=(int(shard_h), int(kv_lora_rank)),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        if skip_defensive_clones:
            q_for_decode = ttnn.to_memory_config(q_for_decode, q_mem_config)
        else:
            q_for_decode_view = ttnn.to_memory_config(q_for_decode, q_mem_config)
            q_for_decode_sharded = ttnn.clone(q_for_decode_view, memory_config=q_mem_config)
            ttnn.deallocate(q_for_decode, force=False)
            q_for_decode = q_for_decode_sharded
        flash_mla_memcfg = flash_mla_out_memcfg
    if os.environ.get("GLM4_MOE_LITE_DISABLE_FLASH_MLA_DECODE", "").strip() == "1":
        # Debug-only: bypass the FlashMLA decode kernel to isolate nondeterminism.
        # This produces incorrect outputs but should be deterministic if the
        # corruption is coming from the attention kernel.
        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None
        heads_padded = ((int(hparams.num_attention_heads) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        attn_latent = ttnn.from_torch(
            torch.zeros((1, batch, heads_padded, int(hparams.kv_lora_rank)), dtype=torch.bfloat16, device="cpu"),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            # from_torch does not support sharded output configs; allocate in DRAM.
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
    elif use_v_cache_slice:
        v_cache = ttnn.slice(
            kvpe_cache,
            [0, 0, 0, 0],
            [int(kvpe_cache.shape[0]), 1, int(kvpe_cache.shape[2]), int(hparams.kv_lora_rank)],
        )
        attn_latent = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_for_decode,
            kvpe_cache,
            v_cache,
            head_dim_v=int(hparams.kv_lora_rank),
            page_table_tensor=page_table_tt,
            cur_pos_tensor=tt_positions,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=flash_mla_memcfg,
        )  # [1,B,H_padded,kv_lora_rank]
        ttnn.deallocate(v_cache, force=False)
    else:
        attn_latent = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_for_decode,
            kvpe_cache,
            head_dim_v=int(hparams.kv_lora_rank),
            page_table_tensor=page_table_tt,
            cur_pos_tensor=tt_positions,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=flash_mla_memcfg,
        )  # [1,B,H_padded,kv_lora_rank]
    ttnn.deallocate(q_for_decode, force=False)
    # Deferred q_kvpe deallocation: when skip_defensive_clones is True, q_for_decode
    # was a permute view of q_kvpe. Now that FlashMLA has consumed it, q_kvpe can be freed.
    if q_kvpe is not None:
        ttnn.deallocate(q_kvpe, force=False)
        q_kvpe = None
    _profile_add(profile, "flash_mla_decode_s", time.perf_counter() - t0 if profile is not None else 0.0)

    if shard_q and os.environ.get("GLM4_MOE_LITE_DISABLE_FLASH_MLA_DECODE", "").strip() != "1":
        # Bring-up convenience: reshard FlashMLA output back to DRAM interleaved
        # before slicing/permuting. This is not the fastest path, but it keeps
        # the downstream code unchanged while we validate correctness.
        if skip_defensive_clones:
            attn_latent = ttnn.to_memory_config(attn_latent, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            attn_latent_view = ttnn.to_memory_config(attn_latent, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn_latent_dram = ttnn.clone(attn_latent_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_latent, force=False)
            attn_latent = attn_latent_dram

    # Slice padded heads back to num_heads.
    #
    # Correctness: `ttnn.slice` may return a view without refcounting. If we
    # keep both the padded tensor and the sliced view alive and deallocate both,
    # we can hit use-after-free or double-free issues depending on whether the
    # slice aliases the source buffer. Materialize the slice before freeing the
    # padded output.
    attn_latent_padded = attn_latent
    if skip_defensive_clones:
        attn_latent = ttnn.slice(
            attn_latent_padded,
            [0, 0, 0, 0],
            [1, batch, int(hparams.num_attention_heads), int(hparams.kv_lora_rank)],
        )
        # attn_latent_padded stays alive — permute and kv_b2 matmul consume the slice view
    else:
        attn_latent_view = ttnn.slice(
            attn_latent_padded,
            [0, 0, 0, 0],
            [1, batch, int(hparams.num_attention_heads), int(hparams.kv_lora_rank)],
        )
        attn_latent = ttnn.clone(attn_latent_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # NOTE: attn_latent_view may alias attn_latent_padded; do not deallocate it separately.
        ttnn.deallocate(attn_latent_padded, force=False)
    attn_latent = ttnn.permute(attn_latent, (0, 2, 1, 3))  # [1,H,B,kv_lora_rank]

    t0 = time.perf_counter() if profile is not None else 0.0
    if tp_enabled and not attn_dp:
        v = _tp_row_parallel_linear_from_replicated(attn_latent, w.w_kv_b2)  # [1,H,B,v_head_dim]
    else:
        v = _mlp_linear(attn_latent, w.w_kv_b2)  # [1,H,B,v_head_dim]
    ttnn.deallocate(attn_latent, force=False)
    # Deferred attn_latent_padded deallocation: when skip_defensive_clones is True,
    # attn_latent was a chain of views from attn_latent_padded. Now consumed by kv_b2 matmul.
    if skip_defensive_clones:
        try:
            ttnn.deallocate(attn_latent_padded, force=False)
        except Exception:
            pass

    if concat_heads:
        v = ttnn.transformer.concatenate_heads(v)  # [1, H, B, v_head_dim] -> [1, B, H*v_head_dim]
        v = ttnn.reshape(v, (1, 1, batch, int(hparams.num_attention_heads * hparams.v_head_dim)))
    else:
        v = ttnn.permute(v, (0, 2, 1, 3))  # [1,B,H,v_head_dim]
        v = ttnn.reshape(v, (1, batch, 1, int(hparams.num_attention_heads * hparams.v_head_dim)))
        v = ttnn.permute(v, (0, 2, 1, 3))  # [1,1,B,H*v_head_dim]

    attn_out = _attn_linear(v, w.w_o)  # [1,1,B,hidden]
    ttnn.deallocate(v, force=False)

    x_attn_out = residual + attn_out
    ttnn.deallocate(attn_out, force=False)
    _profile_add(profile, "attn_out_s", time.perf_counter() - t0 if profile is not None else 0.0)

    if os.environ.get("GLM4_MOE_LITE_DISABLE_MLP", "").strip() == "1":
        # Debug-only: bypass the post-attention MLP to isolate nondeterminism in
        # later dense/MoE compute. Returns the attention residual output.
        _profile_add(profile, "total_s", time.perf_counter() - t_layer0 if profile is not None else 0.0)
        return x_attn_out

    # ---- MLP (dense for layer0; MoE for routed layers) ----
    residual = x_attn_out  # [1,1,B,H]
    t0 = time.perf_counter() if profile is not None else 0.0
    x = w.post_attention_layernorm(x_attn_out, mode="decode")  # [1,1,B,H]
    _profile_add(profile, "mlp_norm_s", time.perf_counter() - t0 if profile is not None else 0.0)

    use_moe = moe_runtime is not None and getattr(w, "moe", None) is not None
    if not use_moe:
        t0 = time.perf_counter() if profile is not None else 0.0
        if dram_sharded_mlp:
            mlp_out = _dram_sharded_mlp(x, w.w_mlp_gate, w.w_mlp_up, w.w_mlp_down)
            ttnn.deallocate(x, force=False)
        else:
            gate = _mlp_linear(x, w.w_mlp_gate)
            up = _mlp_linear(x, w.w_mlp_up)
            ttnn.deallocate(x, force=False)
            x_ff = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            ttnn.deallocate(gate, force=False)
            ttnn.deallocate(up, force=False)
            mlp_out = _mlp_linear(x_ff, w.w_mlp_down)
            ttnn.deallocate(x_ff, force=False)
        if tp_enabled:
            mlp_out_reduced = ttnn.all_reduce(
                mlp_out,
                num_links=1,
                topology=ttnn.Topology.Linear,
                cluster_axis=tp_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(mlp_out, force=False)
            mlp_out = mlp_out_reduced

        x_mlp_out = residual + mlp_out
        ttnn.deallocate(mlp_out, force=False)
        ttnn.deallocate(residual, force=False)
        _profile_add(profile, "mlp_dense_s", time.perf_counter() - t0 if profile is not None else 0.0)
        _profile_add(profile, "total_s", time.perf_counter() - t_layer0 if profile is not None else 0.0)
        return x_mlp_out

    # MoE path:
    # - shared_experts MLP (dense) + routed experts MLP
    from models.demos.glm4_moe_lite.tt.moe_tt import (
        moe_dense_experts_forward_decode_tt,
        moe_dense_experts_forward_prefill_tt,
        moe_packed_experts_forward_prefill_tt,
        moe_sparse_experts_forward_tt,
        moe_topk_cpu_reference,
        moe_topk_tt,
    )

    tokens = int(x.shape[2])
    experts_impl = os.environ.get("GLM4_MOE_LITE_MOE_EXPERTS_IMPL", "sparse").strip().lower()
    use_dense_decode = experts_impl in {"dense_decode", "dense-decode"} and tokens == 1
    dense_prefill = _env_bool("GLM4_MOE_LITE_MOE_DENSE_PREFILL", default=False)
    packed_prefill = _env_bool("GLM4_MOE_LITE_MOE_PACKED_PREFILL", default=False)
    # Packed prefill reads routing indices to CPU, which is forbidden during trace
    # capture. Decode batches (tokens <= max_batch=32) can trigger tokens>1 but are
    # traced, so only use packed prefill for genuine prefill (tokens > 32).
    _PACKED_PREFILL_MIN_TOKENS = 33
    use_packed_prefill = packed_prefill and tokens >= _PACKED_PREFILL_MIN_TOKENS
    use_dense_prefill = dense_prefill and not use_packed_prefill and tokens >= 33
    moe_decode_mc = getattr(moe_runtime, "decode_memory_config", ttnn.DRAM_MEMORY_CONFIG)

    # Pad tokens dim for sparse expert kernels (decode tokens are often 1).
    # Dense prefill path uses ttnn.linear (no block alignment needed), so skip padding.
    pad_tokens = 0
    if not use_dense_decode and not use_dense_prefill and not use_packed_prefill:
        sparse_multiple = _moe_sparse_tokens_multiple(device=device, moe_runtime=moe_runtime)
        pad_tokens = (-tokens) % sparse_multiple
        if pad_tokens:
            # IMPORTANT: `ttnn.pad` can return a *view* that aliases the input buffer
            # (no refcounting). Materialize with `ttnn.clone` before deallocating the
            # original tensor, otherwise we get use-after-free in decode.
            if skip_defensive_clones:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
            else:
                x_padded_view = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
                x_padded = ttnn.clone(x_padded_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                # NOTE: x_padded_view may alias `x`; do not deallocate it separately.
                ttnn.deallocate(x, force=False)
                x = x_padded

    # Shared expert (dense MLP).
    # When fuse_mlp_moe_reduce is True, skip the per-branch all_reduce and instead
    # add the local partial results first, then do ONE all_reduce on the sum.
    _skip_shared_reduce = fuse_mlp_moe_reduce and tp_enabled
    t0 = time.perf_counter() if profile is not None else 0.0
    _use_dram_mlp = dram_sharded_mlp and int(x.shape[2]) == _DS_BATCH
    if _use_dram_mlp:
        shared_out = _dram_sharded_mlp(x, w.w_mlp_gate, w.w_mlp_up, w.w_mlp_down)
    else:
        gate_shared = _mlp_linear(x, w.w_mlp_gate)
        up_shared = _mlp_linear(x, w.w_mlp_up)
        x_ff_shared = ttnn.mul(gate_shared, up_shared, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate_shared, force=False)
        ttnn.deallocate(up_shared, force=False)
        shared_out = _mlp_linear(x_ff_shared, w.w_mlp_down, memory_config=moe_decode_mc)
        ttnn.deallocate(x_ff_shared, force=False)
    if tp_enabled and not _skip_shared_reduce:
        shared_out_reduced = ttnn.all_reduce(
            shared_out,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=tp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(shared_out, force=False)
        shared_out = shared_out_reduced
    _profile_add(profile, "moe_shared_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # Routed experts.
    t0 = time.perf_counter() if profile is not None else 0.0
    router_impl = os.environ.get("GLM4_MOE_LITE_MOE_ROUTER_IMPL", "tt").strip().lower()
    if router_impl == "cpu":
        topk_weights, topk_indices = moe_topk_cpu_reference(device=device, x=x, moe_w=w.moe, hparams=hparams)
    else:
        topk_weights, topk_indices = moe_topk_tt(
            x=x,
            moe_w=w.moe,
            hparams=hparams,
            compute_kernel_config=mlp_compute_kernel_config,
        )
    _profile_add(profile, "moe_router_s", time.perf_counter() - t0 if profile is not None else 0.0)

    t0 = time.perf_counter() if profile is not None else 0.0
    if use_dense_decode:
        routed_out = moe_dense_experts_forward_decode_tt(
            device=device,
            hidden_states=x,  # consumed
            topk_expert_indices=topk_indices,  # consumed
            topk_expert_weights=topk_weights,  # consumed
            moe_w=w.moe,
            hparams=hparams,
            memory_config=moe_decode_mc,
            compute_kernel_config=mlp_compute_kernel_config,
            skip_defensive_clones=skip_defensive_clones,
        )
    elif use_packed_prefill:
        routed_out = moe_packed_experts_forward_prefill_tt(
            device=device,
            hidden_states=x,  # consumed
            topk_expert_indices=topk_indices,  # consumed
            topk_expert_weights=topk_weights,  # consumed
            moe_w=w.moe,
            hparams=hparams,
            memory_config=moe_decode_mc,
            compute_kernel_config=mlp_compute_kernel_config,
            skip_defensive_clones=skip_defensive_clones,
        )
    elif use_dense_prefill:
        routed_out = moe_dense_experts_forward_prefill_tt(
            device=device,
            hidden_states=x,  # consumed
            topk_expert_indices=topk_indices,  # consumed
            topk_expert_weights=topk_weights,  # consumed
            moe_w=w.moe,
            hparams=hparams,
            memory_config=moe_decode_mc,
            compute_kernel_config=mlp_compute_kernel_config,
            skip_defensive_clones=skip_defensive_clones,
        )
    else:
        routed_out = moe_sparse_experts_forward_tt(
            device=device,
            hidden_states=x,  # consumed
            topk_expert_indices=topk_indices,  # consumed
            topk_expert_weights=topk_weights,  # consumed
            moe_w=w.moe,
            rt=moe_runtime,
            memory_config=moe_decode_mc,
            skip_defensive_clones=skip_defensive_clones,
            skip_final_reduce=_skip_shared_reduce,
        )
    _profile_add(profile, "moe_experts_s", time.perf_counter() - t0 if profile is not None else 0.0)

    t0 = time.perf_counter() if profile is not None else 0.0
    mlp_out = ttnn.add(shared_out, routed_out, memory_config=moe_decode_mc)
    ttnn.deallocate(shared_out, force=False)
    ttnn.deallocate(routed_out, force=False)

    # When fuse_mlp_moe_reduce is True, do ONE fused all_reduce on the combined output.
    if _skip_shared_reduce:
        mlp_out_reduced = ttnn.all_reduce(
            mlp_out,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=tp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(mlp_out, force=False)
        mlp_out = mlp_out_reduced

    # Slice back to the real token count if we padded.
    if pad_tokens:
        # `slice` may return a view that aliases `mlp_out` (no refcounting).
        # Materialize before freeing the padded tensor to avoid decode corruption.
        if skip_defensive_clones:
            mlp_out = ttnn.slice(mlp_out, [0, 0, 0, 0], [1, 1, tokens, int(hparams.hidden_size)])
        else:
            mlp_out_view = ttnn.slice(mlp_out, [0, 0, 0, 0], [1, 1, tokens, int(hparams.hidden_size)])
            mlp_out_sliced = ttnn.clone(mlp_out_view, memory_config=moe_decode_mc)
            ttnn.deallocate(mlp_out, force=False)
            mlp_out = mlp_out_sliced

    x_mlp_out = residual + mlp_out
    ttnn.deallocate(mlp_out, force=False)
    ttnn.deallocate(residual, force=False)
    _profile_add(profile, "moe_merge_s", time.perf_counter() - t0 if profile is not None else 0.0)
    _profile_add(profile, "total_s", time.perf_counter() - t_layer0 if profile is not None else 0.0)
    return x_mlp_out


@torch.no_grad()
def run_decoder_layer_prefill_update_cache_tt(
    *,
    device: Any,
    x_embed: ttnn.Tensor,  # [1, 1, S, hidden] tiled (S = B*S_pad when batched)
    page_table_tt: ttnn.Tensor,  # [B, max_num_blocks_per_req] int32 row-major on device
    kvpe_cache: ttnn.Tensor,  # [num_blocks, 1, block_size, kvpe_dim] on device
    cos_matrix: ttnn.Tensor,  # [1, 1, S_pad, rope_dim] bf16
    sin_matrix: ttnn.Tensor,  # [1, 1, S_pad, rope_dim] bf16
    trans_matrix: ttnn.Tensor,
    w: Any,
    hparams: Glm4MoeLiteHParams,
    prompt_len: int,  # number of valid tokens to write into KV cache (<= S_pad)
    batch: int = 1,  # number of requests batched in the S dimension
    prompt_lens: list[int] | None = None,  # per-request prompt lengths (required when batch > 1)
    moe_runtime: Any | None = None,
    profile: dict[str, float] | None = None,
) -> ttnn.Tensor:
    """Run prefill for a single decoder layer and fill its paged KVPE cache.

    This is the sequence-length (S>1) counterpart to
    `run_decoder_layer_decode_one_step_update_cache_tt`.

    When ``batch > 1``, the token dimension of ``x_embed`` is ``B * S_pad``
    (all requests concatenated along dim-2).  Token-wise ops (norms, linears,
    MoE) operate on the flat ``[1,1,B*S_pad,hidden]`` shape.  For RoPE and
    FlashMLA the tensors are reshaped to ``[B,...,S_pad,...]`` so that
    positional encoding and causal masking are per-request.
    """
    if len(x_embed.shape) != 4:
        raise ValueError(f"expected x_embed rank 4, got shape={tuple(x_embed.shape)}")
    total_seq = int(x_embed.shape[2])
    if total_seq <= 0:
        raise ValueError("prefill seq_len must be > 0")

    batch = int(batch)
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")

    if batch > 1:
        if total_seq % batch != 0:
            raise ValueError(
                f"x_embed dim-2 ({total_seq}) must be divisible by batch ({batch})"
            )
        seq_len = total_seq // batch
        if prompt_lens is None:
            raise ValueError("prompt_lens required when batch > 1")
        if len(prompt_lens) != batch:
            raise ValueError(f"prompt_lens length {len(prompt_lens)} != batch {batch}")
    else:
        seq_len = total_seq
        if prompt_lens is None:
            prompt_lens = [int(prompt_len)]

    prompt_len = int(prompt_len)
    if prompt_len <= 0 or prompt_len > seq_len:
        raise ValueError(f"invalid prompt_len={prompt_len} for seq_len={seq_len}")

    kvpe_dim = int(hparams.kv_lora_rank + hparams.qk_rope_head_dim)
    num_heads = int(hparams.num_attention_heads)
    t_layer0 = time.perf_counter() if profile is not None else 0.0

    # Optional precision knob for MLP/router matmuls during bring-up. Controlled
    # by the same env var as decode.
    mlp_compute_kernel_config = None
    if os.environ.get("GLM4_MOE_LITE_MOE_FP32_ACC", "").strip() == "1":
        mlp_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _mlp_linear(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        if mlp_compute_kernel_config is None:
            return ttnn.linear(a, b)
        return ttnn.linear(a, b, compute_kernel_config=mlp_compute_kernel_config)

    tp_axis = _tp_cluster_axis(device)
    tp_enabled = tp_axis is not None and os.environ.get("GLM4_MOE_LITE_TP", "").strip() == "1"
    mesh_rows, mesh_cols = _mesh_shape(device)
    tp_size = int((mesh_rows, mesh_cols)[tp_axis]) if tp_axis is not None else 1

    def _tp_row_parallel_linear_from_replicated(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        a_tp = ttnn.mesh_partition(a, dim=3, cluster_axis=tp_axis)
        out = _mlp_linear(a_tp, b)
        ttnn.deallocate(a_tp, force=False)
        out_reduced = ttnn.all_reduce(
            out,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=tp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(out, force=False)
        return out_reduced

    residual = x_embed
    t0 = time.perf_counter() if profile is not None else 0.0
    x = w.input_layernorm(x_embed, mode="prefill")  # [1,1,S,hidden]
    _profile_add(profile, "norm_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- Q path ----
    t0 = time.perf_counter() if profile is not None else 0.0
    q_a = None
    kv = None
    # Token-wise linears operate on [1,1,B*S_pad,...] (total_seq tokens).
    w_q_kv_a = getattr(w, "w_q_kv_a", None)
    if w_q_kv_a is not None:
        if tp_enabled:
            qkv = _tp_row_parallel_linear_from_replicated(x, w_q_kv_a)  # [1,1,T,q_lora_rank+kvpe_dim]
        else:
            qkv = _mlp_linear(x, w_q_kv_a)  # [1,1,T,q_lora_rank+kvpe_dim]
        q_a = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, total_seq, int(hparams.q_lora_rank)])
        kv = ttnn.slice(
            qkv,
            [0, 0, 0, int(hparams.q_lora_rank)],
            [1, 1, total_seq, int(hparams.q_lora_rank) + kvpe_dim],
        )
        ttnn.deallocate(qkv, force=False)
    else:
        if tp_enabled:
            q_a = _tp_row_parallel_linear_from_replicated(x, w.w_q_a)  # [1,1,T,q_lora_rank]
        else:
            q_a = _mlp_linear(x, w.w_q_a)  # [1,1,T,q_lora_rank]
    q_a = w.q_a_layernorm(q_a, mode="prefill")
    if tp_enabled:
        q = _tp_row_parallel_linear_from_replicated(q_a, w.w_q_b)  # [1,1,T,H*qk_head_dim]
    else:
        q = _mlp_linear(q_a, w.w_q_b)  # [1,1,T,H*qk_head_dim]
    ttnn.deallocate(q_a, force=False)

    # Reshape Q from flat token dim to [B, S_pad, H, qk_head_dim], then permute
    # to [B, H, S_pad, qk_head_dim] for attention.
    q = ttnn.reshape(q, (batch, seq_len, num_heads, int(hparams.qk_head_dim)))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [B,H,S_pad,qk_head_dim]
    q_nope = ttnn.slice(q, [0, 0, 0, 0], [batch, num_heads, seq_len, int(hparams.qk_nope_head_dim)])
    q_rope = ttnn.slice(q, [0, 0, 0, int(hparams.qk_nope_head_dim)], [batch, num_heads, seq_len, int(hparams.qk_head_dim)])
    ttnn.deallocate(q, force=False)

    # Project q_nope into KV latent space (per-head).
    # TTNN non-bcast matmul requires dim-0==1 for 4D×2D. When batch>1, reshape
    # [B,H,S,D] → [1,B*H,S,D] so dim-0 is 1, then reshape back after.
    use_tp_kv_b1 = tp_enabled
    if use_tp_kv_b1:
        qk_nope = int(hparams.qk_nope_head_dim)
        qk_nope_per_shard = qk_nope // max(1, int(tp_size))
        if qk_nope % max(1, int(tp_size)) != 0 or qk_nope_per_shard % int(ttnn.TILE_SIZE) != 0:
            use_tp_kv_b1 = False
    # w_kv_b1 is a small per-head matmul (qk_nope_head_dim → kv_lora_rank).
    # TTNN non-bcast matmul doesn't broadcast across dim-0 when B>1.
    # Loop over batch: each [1,H,S,D] matmul matches the serial-path shape exactly.
    # This is fast because the matmul is small — the big batching wins come from
    # the token-wise linears and MoE which stay on flat [1,1,B*S,hidden].
    if batch > 1:
        kv_b1_fn = _tp_row_parallel_linear_from_replicated if use_tp_kv_b1 else _mlp_linear
        q_nope_parts = []
        for bi in range(batch):
            q_bi = ttnn.slice(q_nope, [bi, 0, 0, 0], [bi + 1, num_heads, seq_len, int(hparams.qk_nope_head_dim)])
            q_bi = kv_b1_fn(q_bi, w.w_kv_b1)
            q_nope_parts.append(q_bi)
        ttnn.deallocate(q_nope, force=False)
        q_nope = ttnn.concat(q_nope_parts, dim=0)  # [B,H,S,kv_lora_rank]
        for p in q_nope_parts:
            ttnn.deallocate(p, force=False)
    else:
        if use_tp_kv_b1:
            q_nope = _tp_row_parallel_linear_from_replicated(q_nope, w.w_kv_b1)
        else:
            q_nope = _mlp_linear(q_nope, w.w_kv_b1)
    _profile_add(profile, "q_path_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- KVPE for the prompt -> fill cache ----
    t0 = time.perf_counter() if profile is not None else 0.0
    if kv is None:
        if tp_enabled:
            kv = _tp_row_parallel_linear_from_replicated(x, w.w_kv_a)  # [1,1,B*S_pad,kvpe_dim]
        else:
            kv = _mlp_linear(x, w.w_kv_a)  # [1,1,B*S_pad,kvpe_dim]
    ttnn.deallocate(x, force=False)

    # Reshape KV from flat [1,1,B*S_pad,...] to [B,1,S_pad,...] for per-request
    # RoPE and KV cache fill.
    kv = ttnn.reshape(kv, (batch, 1, seq_len, kvpe_dim))

    kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [batch, 1, seq_len, int(hparams.kv_lora_rank)])
    kv_rope = ttnn.slice(kv, [0, 0, 0, int(hparams.kv_lora_rank)], [batch, 1, seq_len, kvpe_dim])
    ttnn.deallocate(kv, force=False)

    kv_nope = w.kv_a_layernorm(kv_nope, mode="prefill")

    # RoPE ops require BF16.
    if q_rope.dtype != ttnn.bfloat16:
        q_rope = ttnn.typecast(q_rope, dtype=ttnn.bfloat16)
    if kv_rope.dtype != ttnn.bfloat16:
        kv_rope = ttnn.typecast(kv_rope, dtype=ttnn.bfloat16)

    # RoPE: cos/sin are [1,1,S_pad,rope_dim] and broadcast across batch.
    q_rope = ttnn.experimental.rotary_embedding_llama(
        q_rope,
        cos_matrix,
        sin_matrix,
        trans_matrix,
        is_decode_mode=False,
    )  # [B,H,S_pad,rope_dim]
    kv_rope = ttnn.experimental.rotary_embedding_llama(
        kv_rope,
        cos_matrix,
        sin_matrix,
        trans_matrix,
        is_decode_mode=False,
    )  # [B,1,S_pad,rope_dim]

    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [B,H,S_pad,kvpe_dim]
    ttnn.deallocate(q_nope, force=False)
    ttnn.deallocate(q_rope, force=False)

    kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [B,1,S_pad,kvpe_dim]
    ttnn.deallocate(kv_nope, force=False)
    ttnn.deallocate(kv_rope, force=False)

    # Fill KV cache for each request. paged_fill_cache operates on one batch
    # item at a time (no batched API), so we loop over the batch dimension.
    for bi in range(batch):
        plen = int(prompt_lens[bi])
        kvpe_bi = ttnn.slice(kvpe, [bi, 0, 0, 0], [bi + 1, 1, plen, kvpe_dim])

        if kvpe_bi.dtype != kvpe_cache.dtype:
            kvpe_bi_cast = ttnn.typecast(kvpe_bi, dtype=kvpe_cache.dtype)
        else:
            kvpe_bi_cast = kvpe_bi

        ttnn.experimental.paged_fill_cache(kvpe_cache, kvpe_bi_cast, page_table=page_table_tt, batch_idx=bi)

        if kvpe_bi_cast is not kvpe_bi:
            ttnn.deallocate(kvpe_bi_cast, force=False)
        ttnn.deallocate(kvpe_bi, force=False)

    _profile_add(profile, "kv_cache_fill_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- FlashMLA prefill ----
    # MLA attention scores are computed from dot(q_kvpe, kvpe) with dot-product
    # dimension `kvpe_dim`.
    #
    # Correctness: match HF DeepseekV3Attention default scaling of 1/sqrt(qk_head_dim).
    # Keep `GLM4_MOE_LITE_MLA_SCALE_MODE=kvpe` as an escape hatch for experiments.
    scale_mode = os.environ.get("GLM4_MOE_LITE_MLA_SCALE_MODE", "qk").strip().lower()
    if scale_mode == "kvpe":
        scale = float(int(kvpe_dim) ** -0.5)
    else:
        scale = float(int(hparams.qk_head_dim) ** -0.5)
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,  # heads padded up to 32
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    mla_fidelity = _parse_math_fidelity(
        os.environ.get("GLM4_MOE_LITE_MLA_FIDELITY", ""),
        default=ttnn.MathFidelity.HiFi4,
    )
    mla_approx = os.environ.get("GLM4_MOE_LITE_MLA_APPROX", "0").strip() != "0"
    mla_fp32_acc_req = os.environ.get("GLM4_MOE_LITE_MLA_FP32_ACC", "").strip() == "1"
    mla_fp32_acc = mla_fp32_acc_req
    if mla_fp32_acc_req and os.environ.get("GLM4_MOE_LITE_UNSAFE_ALLOW_FP32_MLA", "").strip() != "1":
        logger.warning(
            "GLM4_MOE_LITE_MLA_FP32_ACC=1 is currently unsafe for FlashMLA prefill/bring-up. "
            "Forcing fp32_dest_acc_en=0. Set GLM4_MOE_LITE_UNSAFE_ALLOW_FP32_MLA=1 to override."
        )
        mla_fp32_acc = False
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=mla_fidelity,
        math_approx_mode=mla_approx,
        fp32_dest_acc_en=mla_fp32_acc,
        packer_l1_acc=False,
    )

    # FlashMLA prefill with batch dimension: q_kvpe [B,H,S_pad,kvpe_dim],
    # kvpe [B,1,S_pad,kvpe_dim].  is_causal=True applies per-batch causal mask.
    t0 = time.perf_counter() if profile is not None else 0.0
    attn_latent = ttnn.transformer.flash_mla_prefill(
        q_kvpe,
        kvpe,
        head_dim_v=int(hparams.kv_lora_rank),
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        attn_mask=None,
        is_causal=True,
    )  # [B,H_padded,S_pad,kv_lora_rank]
    ttnn.deallocate(q_kvpe, force=False)
    ttnn.deallocate(kvpe, force=False)
    _profile_add(profile, "flash_mla_prefill_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # flash_mla_prefill pads heads up to q_chunk_size. Slice back to num_heads.
    attn_latent = ttnn.slice(attn_latent, [0, 0, 0, 0], [batch, num_heads, seq_len, int(hparams.kv_lora_rank)])

    t0 = time.perf_counter() if profile is not None else 0.0
    # Same per-batch loop as w_kv_b1 for w_kv_b2 (small per-head matmul).
    if batch > 1:
        kv_b2_fn = _tp_row_parallel_linear_from_replicated if tp_enabled else _mlp_linear
        v_parts = []
        for bi in range(batch):
            a_bi = ttnn.slice(attn_latent, [bi, 0, 0, 0], [bi + 1, num_heads, seq_len, int(hparams.kv_lora_rank)])
            v_bi = kv_b2_fn(a_bi, w.w_kv_b2)
            ttnn.deallocate(a_bi, force=False)
            v_parts.append(v_bi)
        ttnn.deallocate(attn_latent, force=False)
        v = ttnn.concat(v_parts, dim=0)  # [B,H,S,v_head_dim]
        for p in v_parts:
            ttnn.deallocate(p, force=False)
    else:
        if tp_enabled:
            v = _tp_row_parallel_linear_from_replicated(attn_latent, w.w_kv_b2)
        else:
            v = _mlp_linear(attn_latent, w.w_kv_b2)
        ttnn.deallocate(attn_latent, force=False)

    # Flatten back from [B,H,S_pad,v_head_dim] to [1,1,B*S_pad,H*v_head_dim]
    # for the output projection (token-wise linear).
    v = ttnn.permute(v, (0, 2, 1, 3))  # [B,S_pad,H,v_head_dim]
    v = ttnn.reshape(v, (1, 1, total_seq, int(num_heads * hparams.v_head_dim)))  # [1,1,B*S_pad,H*v_head_dim]
    if tp_enabled:
        attn_out = _tp_row_parallel_linear_from_replicated(v, w.w_o)  # [1,1,B*S_pad,hidden]
    else:
        attn_out = _mlp_linear(v, w.w_o)  # [1,1,B*S_pad,hidden]
    ttnn.deallocate(v, force=False)

    x_attn_out = residual + attn_out
    ttnn.deallocate(attn_out, force=False)
    _profile_add(profile, "attn_out_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- MLP (dense or shared-expert-as-dense) ----
    residual = x_attn_out
    t0 = time.perf_counter() if profile is not None else 0.0
    x = w.post_attention_layernorm(x_attn_out, mode="prefill")
    _profile_add(profile, "mlp_norm_s", time.perf_counter() - t0 if profile is not None else 0.0)

    use_moe = moe_runtime is not None and getattr(w, "moe", None) is not None
    if not use_moe:
        t0 = time.perf_counter() if profile is not None else 0.0
        gate = _mlp_linear(x, w.w_mlp_gate)
        up = _mlp_linear(x, w.w_mlp_up)
        ttnn.deallocate(x, force=False)

        x_ff = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate, force=False)
        ttnn.deallocate(up, force=False)

        mlp_out = _mlp_linear(x_ff, w.w_mlp_down)
        ttnn.deallocate(x_ff, force=False)
        if tp_enabled:
            mlp_out_reduced = ttnn.all_reduce(
                mlp_out,
                num_links=1,
                topology=ttnn.Topology.Linear,
                cluster_axis=tp_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(mlp_out, force=False)
            mlp_out = mlp_out_reduced
        _profile_add(profile, "mlp_dense_s", time.perf_counter() - t0 if profile is not None else 0.0)
    else:
        # MoE path:
        # - shared_experts MLP (dense) + routed experts MLP
        from models.demos.glm4_moe_lite.tt.moe_tt import (
            moe_dense_experts_forward_prefill_tt,
            moe_packed_experts_forward_prefill_tt,
            moe_sparse_experts_forward_tt,
            moe_topk_cpu_reference,
            moe_topk_tt,
        )

        dense_prefill = _env_bool("GLM4_MOE_LITE_MOE_DENSE_PREFILL", default=False)
        packed_prefill = _env_bool("GLM4_MOE_LITE_MOE_PACKED_PREFILL", default=False)

        # Pad tokens to the minimum legal sparse multiple for this mesh.
        # Dense/packed prefill paths use ttnn.linear (no block alignment needed), so skip padding.
        tokens = int(x.shape[2])
        _PACKED_PREFILL_MIN_TOKENS = 33
        use_packed_prefill_here = packed_prefill and tokens >= _PACKED_PREFILL_MIN_TOKENS
        pad_tokens = 0
        if not dense_prefill and not use_packed_prefill_here:
            sparse_multiple = _moe_sparse_tokens_multiple(device=device, moe_runtime=moe_runtime)
            pad_tokens = (-tokens) % sparse_multiple
            if pad_tokens:
                # IMPORTANT: `ttnn.pad` can return a view that aliases the input buffer.
                # Materialize with `ttnn.clone` before deallocating the original tensor.
                x_padded_view = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
                x_padded = ttnn.clone(x_padded_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                # NOTE: x_padded_view may alias `x`; do not deallocate it separately.
                ttnn.deallocate(x, force=False)
                x = x_padded

        # Shared expert (dense MLP).
        t0 = time.perf_counter() if profile is not None else 0.0
        gate_shared = _mlp_linear(x, w.w_mlp_gate)
        up_shared = _mlp_linear(x, w.w_mlp_up)
        x_ff_shared = ttnn.mul(gate_shared, up_shared, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate_shared, force=False)
        ttnn.deallocate(up_shared, force=False)
        shared_out = _mlp_linear(x_ff_shared, w.w_mlp_down)
        ttnn.deallocate(x_ff_shared, force=False)
        if tp_enabled:
            shared_out_reduced = ttnn.all_reduce(
                shared_out,
                num_links=1,
                topology=ttnn.Topology.Linear,
                cluster_axis=tp_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(shared_out, force=False)
            shared_out = shared_out_reduced
        _profile_add(profile, "moe_shared_s", time.perf_counter() - t0 if profile is not None else 0.0)

        # Routed experts.
        t0 = time.perf_counter() if profile is not None else 0.0
        router_impl = os.environ.get("GLM4_MOE_LITE_MOE_ROUTER_IMPL", "tt").strip().lower()
        if router_impl == "cpu":
            topk_weights, topk_indices = moe_topk_cpu_reference(device=device, x=x, moe_w=w.moe, hparams=hparams)
        else:
            topk_weights, topk_indices = moe_topk_tt(
                x=x,
                moe_w=w.moe,
                hparams=hparams,
                compute_kernel_config=mlp_compute_kernel_config,
            )
        _profile_add(profile, "moe_router_s", time.perf_counter() - t0 if profile is not None else 0.0)
        t0 = time.perf_counter() if profile is not None else 0.0
        if use_packed_prefill_here:
            routed_out = moe_packed_experts_forward_prefill_tt(
                device=device,
                hidden_states=x,  # consumed
                topk_expert_indices=topk_indices,  # consumed
                topk_expert_weights=topk_weights,  # consumed
                moe_w=w.moe,
                hparams=hparams,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=mlp_compute_kernel_config,
            )
        elif dense_prefill:
            routed_out = moe_dense_experts_forward_prefill_tt(
                device=device,
                hidden_states=x,  # consumed
                topk_expert_indices=topk_indices,  # consumed
                topk_expert_weights=topk_weights,  # consumed
                moe_w=w.moe,
                hparams=hparams,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=mlp_compute_kernel_config,
            )
        else:
            routed_out = moe_sparse_experts_forward_tt(
                device=device,
                hidden_states=x,  # consumed
                topk_expert_indices=topk_indices,  # consumed
                topk_expert_weights=topk_weights,  # consumed
                moe_w=w.moe,
                rt=moe_runtime,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        _profile_add(profile, "moe_experts_s", time.perf_counter() - t0 if profile is not None else 0.0)

        t0 = time.perf_counter() if profile is not None else 0.0
        mlp_out = shared_out + routed_out
        ttnn.deallocate(shared_out, force=False)
        ttnn.deallocate(routed_out, force=False)

        # Slice back to the real token count if we padded.
        if pad_tokens:
            # `slice` may return a view that aliases `mlp_out` (no refcounting).
            # Materialize before freeing the padded tensor to avoid prefill corruption.
            mlp_out_view = ttnn.slice(mlp_out, [0, 0, 0, 0], [1, 1, tokens, int(hparams.hidden_size)])
            mlp_out_sliced = ttnn.clone(mlp_out_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(mlp_out, force=False)
            mlp_out = mlp_out_sliced
        _profile_add(profile, "moe_merge_s", time.perf_counter() - t0 if profile is not None else 0.0)

    x_mlp_out = residual + mlp_out
    ttnn.deallocate(mlp_out, force=False)
    ttnn.deallocate(residual, force=False)
    _profile_add(profile, "total_s", time.perf_counter() - t_layer0 if profile is not None else 0.0)

    return x_mlp_out


__all__ = [
    "prepare_decode_rope_and_positions_tt",
    "run_decoder_layer_decode_one_step_update_cache_tt",
    "run_decoder_layer_prefill_update_cache_tt",
]

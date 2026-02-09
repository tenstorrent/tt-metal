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
    ttnn.deallocate(rot_idxs)

    cos_batch = ttnn.unsqueeze_to_4D(cos_rows)  # [1,1,B_pad,D]
    sin_batch = ttnn.unsqueeze_to_4D(sin_rows)  # [1,1,B_pad,D]
    ttnn.deallocate(cos_rows)
    ttnn.deallocate(sin_rows)

    if padded_batch != batch:
        cos_sliced = ttnn.slice(cos_batch, [0, 0, 0, 0], [1, 1, batch, rope_dim])
        sin_sliced = ttnn.slice(sin_batch, [0, 0, 0, 0], [1, 1, batch, rope_dim])
        ttnn.deallocate(cos_batch)
        ttnn.deallocate(sin_batch)
        cos_batch = cos_sliced
        sin_batch = sin_sliced

    return tt_positions, cos_batch, sin_batch


def _shard_kvpe_update_tensor(
    *,
    device: Any,
    kvpe_new: ttnn.Tensor,
    batch: int,
    kvpe_dim: int,
) -> ttnn.Tensor:
    """Transform KVPE update tensor into the sharded layout required by paged_update_cache."""
    # `paged_update_cache` expects the update tensor to be sharded.
    kvpe_new = ttnn.pad(kvpe_new, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)  # [1,32,B,kvpe_dim]
    kvpe_new = ttnn.permute(kvpe_new, (0, 2, 1, 3))  # [1,B,32,kvpe_dim]

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
    return ttnn.to_memory_config(kvpe_new, kvpe_sharded_cfg)


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

    def _mlp_linear(a: ttnn.Tensor, b: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig | None = None) -> ttnn.Tensor:
        kwargs: dict[str, object] = {}
        if memory_config is not None:
            kwargs["memory_config"] = memory_config
        if mlp_compute_kernel_config is not None:
            kwargs["compute_kernel_config"] = mlp_compute_kernel_config
        return ttnn.linear(a, b, **kwargs)

    tp_axis = _tp_cluster_axis(device)
    tp_enabled = tp_axis is not None and os.environ.get("GLM4_MOE_LITE_TP", "").strip() == "1"
    mesh_rows, mesh_cols = _mesh_shape(device)
    tp_size = int((mesh_rows, mesh_cols)[tp_axis]) if tp_axis is not None else 1
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
        ttnn.deallocate(a_tp)
        out_reduced = ttnn.all_reduce(
            out,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=tp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(out)
        return out_reduced

    residual = x_embed_tok
    t0 = time.perf_counter() if profile is not None else 0.0
    x = w.input_layernorm(x_embed_tok, mode="decode")  # [1,1,B,hidden]
    _profile_add(profile, "norm_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- KVPE for new token -> update cache at cur_pos ----
    t0 = time.perf_counter() if profile is not None else 0.0
    q_a = None
    kv = None
    w_q_kv_a = getattr(w, "w_q_kv_a", None)
    if w_q_kv_a is not None:
        if tp_enabled:
            qkv = _tp_row_parallel_linear_from_replicated(x, w_q_kv_a)  # [1,1,B,q_lora_rank+kvpe_dim]
        else:
            qkv = _mlp_linear(x, w_q_kv_a)  # [1,1,B,q_lora_rank+kvpe_dim]
        q_a = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, batch, int(hparams.q_lora_rank)])
        kv = ttnn.slice(
            qkv,
            [0, 0, 0, int(hparams.q_lora_rank)],
            [1, 1, batch, int(hparams.q_lora_rank) + kvpe_dim],
        )
        ttnn.deallocate(qkv)
    else:
        if tp_enabled:
            kv = _tp_row_parallel_linear_from_replicated(x, w.w_kv_a)  # [1,1,B,kvpe_dim]
        else:
            kv = _mlp_linear(x, w.w_kv_a)  # [1,1,B,kvpe_dim]
    kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, batch, int(hparams.kv_lora_rank)])
    kv_rope = ttnn.slice(kv, [0, 0, 0, int(hparams.kv_lora_rank)], [1, 1, batch, kvpe_dim])
    ttnn.deallocate(kv)

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

    kvpe_new = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1,1,B,kvpe_dim]
    ttnn.deallocate(kv_nope)
    ttnn.deallocate(kv_rope)

    # Important: paged_update_cache requires the update tensor to be BF16/FP32,
    # even if the cache itself is BF8.
    kvpe_new_sharded = _shard_kvpe_update_tensor(device=device, kvpe_new=kvpe_new, batch=batch, kvpe_dim=kvpe_dim)

    ttnn.experimental.paged_update_cache(
        kvpe_cache,
        kvpe_new_sharded,
        update_idxs_tensor=tt_positions,
        page_table=page_table_tt,
    )
    ttnn.deallocate(kvpe_new_sharded)
    # NOTE: `ttnn.pad` can return a view which may alias the `kvpe_new` buffer.
    # Keep `kvpe_new` alive until after the update kernel is enqueued to avoid
    # use-after-free on some runtimes.
    ttnn.deallocate(kvpe_new)
    _profile_add(profile, "kv_cache_update_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- Q path ----
    t0 = time.perf_counter() if profile is not None else 0.0
    if q_a is None:
        if tp_enabled:
            q_a = _tp_row_parallel_linear_from_replicated(x, w.w_q_a)  # [1,1,B,q_lora_rank]
        else:
            q_a = _mlp_linear(x, w.w_q_a)  # [1,1,B,q_lora_rank]
    q_a = w.q_a_layernorm(q_a, mode="decode")
    if tp_enabled:
        q = _tp_row_parallel_linear_from_replicated(q_a, w.w_q_b)  # [1,1,B,num_heads*qk_head_dim]
    else:
        q = _mlp_linear(q_a, w.w_q_b)  # [1,1,B,num_heads*qk_head_dim]
    ttnn.deallocate(q_a)

    q = ttnn.reshape(q, (1, batch, int(hparams.num_attention_heads), int(hparams.qk_head_dim)))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [1,H,B,qk_head_dim]
    q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, int(hparams.num_attention_heads), batch, int(hparams.qk_nope_head_dim)])
    q_rope = ttnn.slice(
        q,
        [0, 0, 0, int(hparams.qk_nope_head_dim)],
        [1, int(hparams.num_attention_heads), batch, int(hparams.qk_head_dim)],
    )
    ttnn.deallocate(q)

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
            ttnn.deallocate(cos_decode)
            ttnn.deallocate(sin_decode)
            ttnn.deallocate(trans_decode)

    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [1,H,B,kvpe_dim]
    ttnn.deallocate(q_nope)
    ttnn.deallocate(q_rope)

    # x no longer needed.
    ttnn.deallocate(x)

    # ---- FlashMLA decode ----
    q_for_decode = ttnn.permute(q_kvpe, (0, 2, 1, 3))  # [1,B,H,kvpe_dim]
    ttnn.deallocate(q_kvpe)
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
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,  # not used in decode
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    mla_fidelity = _parse_math_fidelity(
        os.environ.get("GLM4_MOE_LITE_MLA_FIDELITY", ""),
        default=ttnn.MathFidelity.HiFi4,
    )
    mla_approx = os.environ.get("GLM4_MOE_LITE_MLA_APPROX", "0").strip() != "0"
    mla_fp32_acc = os.environ.get("GLM4_MOE_LITE_MLA_FP32_ACC", "").strip() == "1"
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=mla_fidelity,
        math_approx_mode=mla_approx,
        fp32_dest_acc_en=mla_fp32_acc,
        packer_l1_acc=False,
    )

    # Prefer direct KVPE cache usage when supported by the kernel to avoid per-step
    # V-cache slicing overhead. Keep an opt-in fallback for older runtimes.
    t0 = time.perf_counter() if profile is not None else 0.0
    use_v_cache_slice = os.environ.get("GLM4_MOE_LITE_MLA_USE_V_CACHE_SLICE", "").strip() == "1"
    flash_mla_memcfg = ttnn.DRAM_MEMORY_CONFIG
    if use_v_cache_slice:
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
        ttnn.deallocate(v_cache)
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
    ttnn.deallocate(q_for_decode)
    _profile_add(profile, "flash_mla_decode_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # Slice padded heads back to num_heads.
    attn_latent = ttnn.slice(attn_latent, [0, 0, 0, 0], [1, batch, int(hparams.num_attention_heads), int(hparams.kv_lora_rank)])
    attn_latent = ttnn.permute(attn_latent, (0, 2, 1, 3))  # [1,H,B,kv_lora_rank]

    t0 = time.perf_counter() if profile is not None else 0.0
    if tp_enabled:
        v = _tp_row_parallel_linear_from_replicated(attn_latent, w.w_kv_b2)  # [1,H,B,v_head_dim]
    else:
        v = _mlp_linear(attn_latent, w.w_kv_b2)  # [1,H,B,v_head_dim]
    ttnn.deallocate(attn_latent)

    v = ttnn.permute(v, (0, 2, 1, 3))  # [1,B,H,v_head_dim]
    v = ttnn.reshape(v, (1, batch, 1, int(hparams.num_attention_heads * hparams.v_head_dim)))
    v = ttnn.permute(v, (0, 2, 1, 3))  # [1,1,B,H*v_head_dim]

    if tp_enabled:
        attn_out = _tp_row_parallel_linear_from_replicated(v, w.w_o)  # [1,1,B,hidden]
    else:
        attn_out = _mlp_linear(v, w.w_o)  # [1,1,B,hidden]
    ttnn.deallocate(v)

    x_attn_out = residual + attn_out
    ttnn.deallocate(attn_out)
    _profile_add(profile, "attn_out_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- MLP (dense for layer0; MoE for routed layers) ----
    residual = x_attn_out  # [1,1,B,H]
    t0 = time.perf_counter() if profile is not None else 0.0
    x = w.post_attention_layernorm(x_attn_out, mode="decode")  # [1,1,B,H]
    _profile_add(profile, "mlp_norm_s", time.perf_counter() - t0 if profile is not None else 0.0)

    use_moe = moe_runtime is not None and getattr(w, "moe", None) is not None
    if not use_moe:
        t0 = time.perf_counter() if profile is not None else 0.0
        gate = _mlp_linear(x, w.w_mlp_gate)
        up = _mlp_linear(x, w.w_mlp_up)
        ttnn.deallocate(x)
        gate = ttnn.silu(gate)
        x_ff = gate * up
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        mlp_out = _mlp_linear(x_ff, w.w_mlp_down)
        ttnn.deallocate(x_ff)
        if tp_enabled:
            mlp_out_reduced = ttnn.all_reduce(
                mlp_out,
                num_links=1,
                topology=ttnn.Topology.Linear,
                cluster_axis=tp_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(mlp_out)
            mlp_out = mlp_out_reduced

        x_mlp_out = residual + mlp_out
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)
        _profile_add(profile, "mlp_dense_s", time.perf_counter() - t0 if profile is not None else 0.0)
        _profile_add(profile, "total_s", time.perf_counter() - t_layer0 if profile is not None else 0.0)
        return x_mlp_out

    # MoE path:
    # - shared_experts MLP (dense) + routed experts MLP
    from models.demos.glm4_moe_lite.tt.moe_tt import (
        moe_dense_experts_forward_decode_tt,
        moe_sparse_experts_forward_tt,
        moe_topk_cpu_reference,
        moe_topk_tt,
    )

    tokens = int(x.shape[2])
    experts_impl = os.environ.get("GLM4_MOE_LITE_MOE_EXPERTS_IMPL", "sparse").strip().lower()
    use_dense_decode = experts_impl in {"dense_decode", "dense-decode"} and tokens == 1

    # Pad tokens dim for sparse expert kernels (decode tokens are often 1).
    # Use the minimum legal multiple for the current dispatch width to avoid
    # inflating decode work on multi-device meshes.
    pad_tokens = 0
    if not use_dense_decode:
        sparse_multiple = _moe_sparse_tokens_multiple(device=device, moe_runtime=moe_runtime)
        pad_tokens = (-tokens) % sparse_multiple
        if pad_tokens:
            # IMPORTANT: `ttnn.pad` can return a *view* that aliases the input buffer
            # (no refcounting). Materialize with `ttnn.clone` before deallocating the
            # original tensor, otherwise we get use-after-free in decode.
            x_padded_view = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
            x_padded = ttnn.clone(x_padded_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # NOTE: x_padded_view may alias `x`; do not deallocate it separately.
            ttnn.deallocate(x)
            x = x_padded

    # Shared expert (dense MLP).
    t0 = time.perf_counter() if profile is not None else 0.0
    gate_shared = _mlp_linear(x, w.w_mlp_gate)
    up_shared = _mlp_linear(x, w.w_mlp_up)
    gate_shared = ttnn.silu(gate_shared)
    x_ff_shared = gate_shared * up_shared
    ttnn.deallocate(gate_shared)
    ttnn.deallocate(up_shared)
    shared_out = _mlp_linear(x_ff_shared, w.w_mlp_down)
    ttnn.deallocate(x_ff_shared)
    if tp_enabled:
        shared_out_reduced = ttnn.all_reduce(
            shared_out,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=tp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(shared_out)
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
    ttnn.deallocate(shared_out)
    ttnn.deallocate(routed_out)

    # Slice back to the real token count if we padded.
    if pad_tokens:
        mlp_out_sliced = ttnn.slice(mlp_out, [0, 0, 0, 0], [1, 1, tokens, int(hparams.hidden_size)])
        ttnn.deallocate(mlp_out)
        mlp_out = mlp_out_sliced

    x_mlp_out = residual + mlp_out
    ttnn.deallocate(mlp_out)
    ttnn.deallocate(residual)
    _profile_add(profile, "moe_merge_s", time.perf_counter() - t0 if profile is not None else 0.0)
    _profile_add(profile, "total_s", time.perf_counter() - t_layer0 if profile is not None else 0.0)
    return x_mlp_out


@torch.no_grad()
def run_decoder_layer_prefill_update_cache_tt(
    *,
    device: Any,
    x_embed: ttnn.Tensor,  # [1, 1, S, hidden] tiled
    page_table_tt: ttnn.Tensor,  # [1, max_num_blocks_per_req] int32 row-major on device
    kvpe_cache: ttnn.Tensor,  # [num_blocks, 1, block_size, kvpe_dim] on device
    cos_matrix: ttnn.Tensor,  # [1, 1, S, rope_dim] bf16
    sin_matrix: ttnn.Tensor,  # [1, 1, S, rope_dim] bf16
    trans_matrix: ttnn.Tensor,
    w: Any,
    hparams: Glm4MoeLiteHParams,
    prompt_len: int,  # number of valid tokens to write into KV cache (<= S)
    moe_runtime: Any | None = None,
    profile: dict[str, float] | None = None,
) -> ttnn.Tensor:
    """Run prefill for a single decoder layer and fill its paged KVPE cache.

    This is the sequence-length (S>1) counterpart to
    `run_decoder_layer_decode_one_step_update_cache_tt`.
    """
    if len(x_embed.shape) != 4:
        raise ValueError(f"expected x_embed rank 4, got shape={tuple(x_embed.shape)}")
    seq_len = int(x_embed.shape[2])
    if seq_len <= 0:
        raise ValueError("prefill seq_len must be > 0")

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

    def _tp_row_parallel_linear_from_replicated(a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        a_tp = ttnn.mesh_partition(a, dim=3, cluster_axis=tp_axis)
        out = _mlp_linear(a_tp, b)
        ttnn.deallocate(a_tp)
        out_reduced = ttnn.all_reduce(
            out,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=tp_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(out)
        return out_reduced

    residual = x_embed
    t0 = time.perf_counter() if profile is not None else 0.0
    x = w.input_layernorm(x_embed, mode="prefill")  # [1,1,S,hidden]
    _profile_add(profile, "norm_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- Q path ----
    t0 = time.perf_counter() if profile is not None else 0.0
    q_a = None
    kv = None
    w_q_kv_a = getattr(w, "w_q_kv_a", None)
    if w_q_kv_a is not None:
        if tp_enabled:
            qkv = _tp_row_parallel_linear_from_replicated(x, w_q_kv_a)  # [1,1,S,q_lora_rank+kvpe_dim]
        else:
            qkv = _mlp_linear(x, w_q_kv_a)  # [1,1,S,q_lora_rank+kvpe_dim]
        q_a = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, seq_len, int(hparams.q_lora_rank)])
        kv = ttnn.slice(
            qkv,
            [0, 0, 0, int(hparams.q_lora_rank)],
            [1, 1, seq_len, int(hparams.q_lora_rank) + kvpe_dim],
        )
        ttnn.deallocate(qkv)
    else:
        if tp_enabled:
            q_a = _tp_row_parallel_linear_from_replicated(x, w.w_q_a)  # [1,1,S,q_lora_rank]
        else:
            q_a = _mlp_linear(x, w.w_q_a)  # [1,1,S,q_lora_rank]
    q_a = w.q_a_layernorm(q_a, mode="prefill")
    if tp_enabled:
        q = _tp_row_parallel_linear_from_replicated(q_a, w.w_q_b)  # [1,1,S,H*qk_head_dim]
    else:
        q = _mlp_linear(q_a, w.w_q_b)  # [1,1,S,H*qk_head_dim]
    ttnn.deallocate(q_a)

    q = ttnn.reshape(q, (1, seq_len, num_heads, int(hparams.qk_head_dim)))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [1,H,S,qk_head_dim]
    q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, num_heads, seq_len, int(hparams.qk_nope_head_dim)])
    q_rope = ttnn.slice(q, [0, 0, 0, int(hparams.qk_nope_head_dim)], [1, num_heads, seq_len, int(hparams.qk_head_dim)])
    ttnn.deallocate(q)

    # Project q_nope into KV latent space (per-head).
    use_tp_kv_b1 = tp_enabled
    if use_tp_kv_b1:
        qk_nope = int(hparams.qk_nope_head_dim)
        qk_nope_per_shard = qk_nope // max(1, int(tp_size))
        if qk_nope % max(1, int(tp_size)) != 0 or qk_nope_per_shard % int(ttnn.TILE_SIZE) != 0:
            use_tp_kv_b1 = False
    if use_tp_kv_b1:
        q_nope = _tp_row_parallel_linear_from_replicated(q_nope, w.w_kv_b1)  # [1,H,S,kv_lora_rank]
    else:
        q_nope = _mlp_linear(q_nope, w.w_kv_b1)  # [1,H,S,kv_lora_rank]
    _profile_add(profile, "q_path_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # ---- KVPE for the prompt -> fill cache ----
    t0 = time.perf_counter() if profile is not None else 0.0
    if kv is None:
        if tp_enabled:
            kv = _tp_row_parallel_linear_from_replicated(x, w.w_kv_a)  # [1,1,S,kvpe_dim]
        else:
            kv = _mlp_linear(x, w.w_kv_a)  # [1,1,S,kvpe_dim]
    ttnn.deallocate(x)

    kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, seq_len, int(hparams.kv_lora_rank)])
    kv_rope = ttnn.slice(kv, [0, 0, 0, int(hparams.kv_lora_rank)], [1, 1, seq_len, kvpe_dim])
    ttnn.deallocate(kv)

    kv_nope = w.kv_a_layernorm(kv_nope, mode="prefill")

    # RoPE ops require BF16.
    if q_rope.dtype != ttnn.bfloat16:
        q_rope = ttnn.typecast(q_rope, dtype=ttnn.bfloat16)
    if kv_rope.dtype != ttnn.bfloat16:
        kv_rope = ttnn.typecast(kv_rope, dtype=ttnn.bfloat16)

    q_rope = ttnn.experimental.rotary_embedding_llama(
        q_rope,
        cos_matrix,
        sin_matrix,
        trans_matrix,
        is_decode_mode=False,
    )  # [1,H,S,rope_dim]
    kv_rope = ttnn.experimental.rotary_embedding_llama(
        kv_rope,
        cos_matrix,
        sin_matrix,
        trans_matrix,
        is_decode_mode=False,
    )  # [1,1,S,rope_dim]

    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [1,H,S,kvpe_dim]
    ttnn.deallocate(q_nope)
    ttnn.deallocate(q_rope)

    kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1,1,S,kvpe_dim]
    ttnn.deallocate(kv_nope)
    ttnn.deallocate(kv_rope)

    # Fill KV cache only for the valid prefix tokens (prompt_len). Padding after
    # the prompt must not be written unless vLLM actually allocated those blocks.
    kvpe_fill = kvpe
    if prompt_len != seq_len:
        kvpe_fill = ttnn.slice(kvpe, [0, 0, 0, 0], [1, 1, prompt_len, kvpe_dim])

    if kvpe_fill.dtype != kvpe_cache.dtype:
        kvpe_fill_cast = ttnn.typecast(kvpe_fill, dtype=kvpe_cache.dtype)
    else:
        kvpe_fill_cast = kvpe_fill

    ttnn.experimental.paged_fill_cache(kvpe_cache, kvpe_fill_cast, page_table=page_table_tt, batch_idx=0)

    if kvpe_fill_cast is not kvpe_fill:
        ttnn.deallocate(kvpe_fill_cast)
    if kvpe_fill is not kvpe:
        ttnn.deallocate(kvpe_fill)
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
    mla_fp32_acc = os.environ.get("GLM4_MOE_LITE_MLA_FP32_ACC", "").strip() == "1"
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=mla_fidelity,
        math_approx_mode=mla_approx,
        fp32_dest_acc_en=mla_fp32_acc,
        packer_l1_acc=False,
    )

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
    )  # [1,H_padded,S,kv_lora_rank]
    ttnn.deallocate(q_kvpe)
    ttnn.deallocate(kvpe)
    _profile_add(profile, "flash_mla_prefill_s", time.perf_counter() - t0 if profile is not None else 0.0)

    # flash_mla_prefill pads heads up to q_chunk_size. Slice back to num_heads.
    attn_latent = ttnn.slice(attn_latent, [0, 0, 0, 0], [1, num_heads, seq_len, int(hparams.kv_lora_rank)])

    t0 = time.perf_counter() if profile is not None else 0.0
    if tp_enabled:
        v = _tp_row_parallel_linear_from_replicated(attn_latent, w.w_kv_b2)  # [1,H,S,v_head_dim]
    else:
        v = _mlp_linear(attn_latent, w.w_kv_b2)  # [1,H,S,v_head_dim]
    ttnn.deallocate(attn_latent)

    v = ttnn.permute(v, (0, 2, 1, 3))  # [1,S,H,v_head_dim]
    v = ttnn.reshape(v, (1, 1, seq_len, int(num_heads * hparams.v_head_dim)))  # [1,1,S,H*v_head_dim]
    if tp_enabled:
        attn_out = _tp_row_parallel_linear_from_replicated(v, w.w_o)  # [1,1,S,hidden]
    else:
        attn_out = _mlp_linear(v, w.w_o)  # [1,1,S,hidden]
    ttnn.deallocate(v)

    x_attn_out = residual + attn_out
    ttnn.deallocate(attn_out)
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
        ttnn.deallocate(x)

        gate = ttnn.silu(gate)
        x_ff = gate * up
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        mlp_out = _mlp_linear(x_ff, w.w_mlp_down)
        ttnn.deallocate(x_ff)
        if tp_enabled:
            mlp_out_reduced = ttnn.all_reduce(
                mlp_out,
                num_links=1,
                topology=ttnn.Topology.Linear,
                cluster_axis=tp_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(mlp_out)
            mlp_out = mlp_out_reduced
        _profile_add(profile, "mlp_dense_s", time.perf_counter() - t0 if profile is not None else 0.0)
    else:
        # MoE path:
        # - shared_experts MLP (dense) + routed experts MLP
        from models.demos.glm4_moe_lite.tt.moe_tt import (
            moe_sparse_experts_forward_tt,
            moe_topk_cpu_reference,
            moe_topk_tt,
        )

        # Pad tokens to the minimum legal sparse multiple for this mesh.
        tokens = int(x.shape[2])
        sparse_multiple = _moe_sparse_tokens_multiple(device=device, moe_runtime=moe_runtime)
        pad_tokens = (-tokens) % sparse_multiple
        if pad_tokens:
            # IMPORTANT: `ttnn.pad` can return a view that aliases the input buffer.
            # Materialize with `ttnn.clone` before deallocating the original tensor.
            x_padded_view = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_tokens), (0, 0)], 0.0)
            x_padded = ttnn.clone(x_padded_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # NOTE: x_padded_view may alias `x`; do not deallocate it separately.
            ttnn.deallocate(x)
            x = x_padded

        # Shared expert (dense MLP).
        t0 = time.perf_counter() if profile is not None else 0.0
        gate_shared = _mlp_linear(x, w.w_mlp_gate)
        up_shared = _mlp_linear(x, w.w_mlp_up)
        gate_shared = ttnn.silu(gate_shared)
        x_ff_shared = gate_shared * up_shared
        ttnn.deallocate(gate_shared)
        ttnn.deallocate(up_shared)
        shared_out = _mlp_linear(x_ff_shared, w.w_mlp_down)
        ttnn.deallocate(x_ff_shared)
        if tp_enabled:
            shared_out_reduced = ttnn.all_reduce(
                shared_out,
                num_links=1,
                topology=ttnn.Topology.Linear,
                cluster_axis=tp_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(shared_out)
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
        ttnn.deallocate(shared_out)
        ttnn.deallocate(routed_out)

        # Slice back to the real token count if we padded.
        if pad_tokens:
            mlp_out_sliced = ttnn.slice(mlp_out, [0, 0, 0, 0], [1, 1, tokens, int(hparams.hidden_size)])
            ttnn.deallocate(mlp_out)
            mlp_out = mlp_out_sliced
        _profile_add(profile, "moe_merge_s", time.perf_counter() - t0 if profile is not None else 0.0)

    x_mlp_out = residual + mlp_out
    ttnn.deallocate(mlp_out)
    ttnn.deallocate(residual)
    _profile_add(profile, "total_s", time.perf_counter() - t_layer0 if profile is not None else 0.0)

    return x_mlp_out


__all__ = [
    "prepare_decode_rope_and_positions_tt",
    "run_decoder_layer_decode_one_step_update_cache_tt",
    "run_decoder_layer_prefill_update_cache_tt",
]

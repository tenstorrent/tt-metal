import os
from dataclasses import dataclass

import torch

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt import tp_common as tpc
from models.demos.blackhole.qwen3_5_9b.utils.general_utils import get_cache_file_name


@dataclass(frozen=True)
class Qwen35GDNWeights:
    dt_bias: ttnn.Tensor
    neg_A_log_exp: ttnn.Tensor
    w_taps: list[ttnn.Tensor]
    w_norm: ttnn.Tensor
    wo: ttnn.Tensor
    wqkv: ttnn.Tensor
    wz: ttnn.Tensor
    wb: ttnn.Tensor
    wa: ttnn.Tensor


def load_gdn_weights(mesh_device, state_dict, args, dtype=ttnn.bfloat16, tensor_cache_path=None) -> Qwen35GDNWeights:
    """Load and per-head-shard one GDN layer's weights across a (1, tp) mesh.

    One loader for every device count. At tp=1 it degenerates to the original
    single-device load — the head reorders below become the identity permutation and
    a shard across a single device just places the whole tensor — so there is no
    separate single-device path that can silently drift out of sync with this one.

    The head split is the whole game: ``prepare_gdn_qkv`` / ``prepare_conv_taps``
    (tp_common) reorder the fused qkv + depthwise-conv weights so a contiguous
    ShardTensorToMesh slice hands each device exactly its (num_k_heads/tp) Q heads,
    (num_k_heads/tp) K heads and (num_v_heads/tp) V heads. Because that reorder keeps
    each device's V heads contiguous and IN original order, the per-V-head tensors
    (z, a, b, A_log, dt_bias, out_proj rows) shard with a plain column/row split — no
    reorder needed. The recurrence is per-value-head, so this shards everything BY HEAD
    with no cross-device comms inside the layer; only the row-parallel out_proj needs
    an all-reduce, which the forward runs afterward. Projections are kept SEPARATE
    (wqkv / wz / wa / wb) to match the my_gdn forward instead of fusing them, mirroring
    the proven gdn/tp.py loader.

    ``tensor_cache_path`` is only safe with REAL (deterministic) weights — pass None
    for random-weight tests to avoid a stale cross-run cache.
    """
    tp = mesh_device.get_num_devices()
    num_k_heads, head_k_dim = args.linear_num_key_heads, args.linear_key_head_dim
    num_v_heads, head_v_dim = args.linear_num_value_heads, args.linear_value_head_dim
    key_dim, value_dim = args.linear_q_dim, args.linear_v_dim
    conv_kernel_size = args.linear_conv_kernel_dim

    if tensor_cache_path is not None:
        os.makedirs(tensor_cache_path, exist_ok=True)

    def cache(name):
        return get_cache_file_name(tensor_cache_path, name)

    # original in_proj_qkv.weight has dimensionality [8192, hidden]
    # row 0.      ┌──────────────────────────────┐
    #             │ Q head 0   (128 rows)        │
    #             │ Q head 1                     │   Q block
    #             │   ...                        │   16 heads → 2048 rows
    #             │ Q head 15                    │
    # row 2048.   ├──────────────────────────────┤
    #             │ K head 0                     │   K block
    #             │   ...                        │   16 heads → 2048 rows
    #             │ K head 15                    │
    # row 4096.   ├──────────────────────────────┤
    #             │ V head 0                     │
    #             │   ...                        │   V block
    #             │ V head 31                    │   32 heads → 4096 rows
    # row 8191    └──────────────────────────────┘
    # tpc.prepare_gdn_qkv splits this into separate Q/K/V tensors
    # reorders them so that shard_w across the final dimension
    # gives each device a slice of Q/K/V heads.
    #            ┌──────────────────────────────┐
    #   Device 0 │ Q heads  0–3    (512 rows)   │
    #            │ K heads  0–3    (512 rows)   │   2048 rows total
    #            │ V heads  0–7    (1024 rows)  │
    #            ├──────────────────────────────┤
    #   Device 1 │ Q heads  4–7                 │
    #            │ K heads  4–7                 │   2048 rows
    #            │ V heads  8–15                │
    #            ├──────────────────────────────┤
    #   Device 2 │ Q heads  8–11                │
    #            │ K heads  8–11                │   2048 rows
    #            │ V heads 16–23                │
    #            ├──────────────────────────────┤
    #   Device 3 │ Q heads 12–15                │
    #            │ K heads 12–15                │   2048 rows
    #            │ V heads 24–31                │
    #            └──────────────────────────────┘
    qkv_re = tpc.prepare_gdn_qkv(
        state_dict["in_proj_qkv.weight"], key_dim, value_dim, num_k_heads, head_k_dim, num_v_heads, head_v_dim, tp
    )
    wqkv = tpc.shard_w(
        qkv_re, mesh_device, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=cache("wqkv"), dtype=dtype
    )

    # in_proj_z has [4096, hidden] shape
    #   row 0     ┌──────────────────────┐
    #             │ z head 0   (128)     │
    #             │ z head 1   (128)     │
    #             │   ...                │   32 heads → 4096 rows
    #             │ z head 31  (128)     │
    #   row 4095  └──────────────────────┘
    # sharding across the rows, provides even shards to each device
    wz = tpc.shard_w(
        state_dict["in_proj_z.weight"],
        mesh_device,
        dim=-1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=cache("wz"),
        dtype=dtype,
    )
    #   in_proj_a.weight   [32, hidden]          in_proj_b.weight   [32, hidden]
    #     row 0  │ a head 0  │                      row 0  │ b head 0  │
    #     row 1  │ a head 1  │                      row 1  │ b head 1  │
    #      ...   │   ...     │   (32 rows)           ...   │   ...     │   (32 rows)
    #     row 31 │ a head 31 │                      row 31 │ b head 31 │
    wa = tpc.shard_w(
        state_dict["in_proj_a.weight"],
        mesh_device,
        dim=-1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=cache("wa"),
        dtype=dtype,
    )
    wb = tpc.shard_w(
        state_dict["in_proj_b.weight"],
        mesh_device,
        dim=-1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=cache("wb"),
        dtype=dtype,
    )

    # ── out_proj (row-parallel): shard the INPUT (value) dim so each device multiplies
    # its local V-head slice; the per-device partial-hidden outputs are summed by the
    # reduce-scatter the forward runs afterward. ──
    wo = tpc.shard_w(
        state_dict["out_proj.weight"],
        mesh_device,
        dim=0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=cache("wo"),
        dtype=dtype,
    )

    # ── Depthwise conv taps: K per-channel taps, reordered to the same per-device
    # Q/K/V grouping as the qkv weight, then sharded over the (local) conv channels. ──
    taps_re = tpc.prepare_conv_taps(
        state_dict["conv1d.weight"], key_dim, num_k_heads, head_k_dim, num_v_heads, head_v_dim, conv_kernel_size, tp
    )
    w_taps = [
        tpc.shard_small(taps_re[j], mesh_device, cache(f"tap{j}"), dim=-1, dtype=dtype) for j in range(conv_kernel_size)
    ]

    # ── Per-V-head scalars: sharded over the value-head dim to match the recurrence's
    # local heads. neg_A_log_exp must stay TRUE fp32: exp() is taken in fp32 here (it
    # overflows to -inf in bf16) and these decay rates feed the long recurrent scan, so
    # we shard it with a direct as_tensor rather than tpc.shard_small — shard_small
    # rounds its host tensor to bf16 before storing, which would silently drop the fp32
    # precision the single-device load kept. The [1,1,N] shape broadcasts over the
    # [B,1,seq,N] activations exactly as the original 1-D tensor did. dt_bias is bf16
    # either way, so it rides tpc.shard_small. ──
    neg_A_log = -(state_dict["A_log"].float().exp())
    neg_A_log_exp = ttnn.as_tensor(
        neg_A_log.unsqueeze(0).unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache("neg_A_log_exp"),
    )
    dt_bias = tpc.shard_small(state_dict["dt_bias"], mesh_device, cache("dt_bias"), dim=-1, dtype=dtype)

    # ── norm.weight is per-(head_v_dim) feature, shared by every head, so it is
    # REPLICATED. as_tensor (not from_torch) so it can use the weight cache; keep its
    # 1-D shape (no unsqueeze) so the gated-RMSNorm broadcast in forward is identical
    # to the single-device path. ──
    w_norm = ttnn.as_tensor(
        state_dict["norm.weight"].to(torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache("w_norm"),
    )

    return Qwen35GDNWeights(
        dt_bias=dt_bias,
        neg_A_log_exp=neg_A_log_exp,
        w_taps=w_taps,
        w_norm=w_norm,
        wo=wo,
        wqkv=wqkv,
        wz=wz,
        wb=wb,
        wa=wa,
    )

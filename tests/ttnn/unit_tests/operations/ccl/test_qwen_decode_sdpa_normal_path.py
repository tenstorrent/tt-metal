# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone fused decode SDPA repro aligned with **Qwen Galaxy `TtLlamaAttention.forward_decode`**.

Per-device tensors (32-chip 8×4 mesh, 8 global KV heads) match `llama_attention.py`:
  - `n_local_heads = n_heads // n_kv_heads = 8`
  - `n_local_kv_heads = 1`  (MQA 8:1 on each chip — **not** global GQA 8:8)
  - `batch_size_per_device_group = max_batch_size // num_device_groups = 8`

Uses **`paged_scaled_dot_product_attention_decode`** (default in `test_qwen_attention.py`), with
`PAGED_SDPA_DECODE_PROGCFG` / `SDPA_DECODE_COMPUTE_PROGCFG` from `qwen_model_config.py`, L1 height-sharded
Q in and sharded SDPA out (`SCORES_BATCHED_MM_OUTPUT_MEMCFG`), KV in DRAM as `bfloat8_b`.

Golden: torch SDPA with KV repeated `nh // n_kv` times (MQA reference).
"""

import math

import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    fa_rand,
    get_chunk_size,
    nearest_n,
    nearest_pow_2,
)

CLUSTER_SHAPE = (8, 4)
MESH_NUM_DEVICES = CLUSTER_SHAPE[0] * CLUSTER_SHAPE[1]

# Global Qwen3-32B Galaxy head counts
N_HEADS = 64
N_KV_HEADS = 8
HEAD_DIM = 128

# Per-device decode layout (see TtLlamaAttention.__init__)
NUM_DEVICES_PER_GROUP = N_KV_HEADS
NUM_DEVICE_GROUPS = MESH_NUM_DEVICES // N_KV_HEADS
N_LOCAL_HEADS = N_HEADS // NUM_DEVICES_PER_GROUP
N_LOCAL_KV_HEADS = N_KV_HEADS // NUM_DEVICES_PER_GROUP
MAX_BATCH_SIZE = 32
BATCH_PER_DEVICE_GROUP = max(MAX_BATCH_SIZE // NUM_DEVICE_GROUPS, 1)

# Match test_qwen_attention.py decode unit test
MAX_SEQ_LEN = 256
PAGE_BLOCK_SIZE = 64
DECODE_CUR_POS = 127

START_CORE = ttnn.CoreCoord(1, 0)
_MIN_PCC = 0.99
_TORCH_SEED = 91357


def _galaxy_sub_core_grids(max_y: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, max_y)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, max_y)),
        ]
    )


def _q_l1_height_sharded(batch: int, padded_q_heads: int, sub_grids: ttnn.CoreRangeSet) -> ttnn.MemoryConfig:
    shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(START_CORE, batch, sub_grids, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, (padded_q_heads, HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _sdpa_out_memcfg_qwen(batch: int, sub_grids: ttnn.CoreRangeSet) -> ttnn.MemoryConfig:
    """`qwen_model_config.SCORES_BATCHED_MM_OUTPUT_MEMCFG` for n_local_heads=8."""
    return ttnn.create_sharded_memory_config(
        shape=(math.ceil(N_LOCAL_HEADS / 32) * 32, HEAD_DIM),
        core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(START_CORE, batch, sub_grids, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _paged_sdpa_decode_progcfg_qwen(sub_grids: ttnn.CoreRangeSet, is_blackhole: bool) -> ttnn.SDPAProgramConfig:
    """`qwen_model_config.PAGED_SDPA_DECODE_PROGCFG` (auto chunk sizes)."""
    num_cores = 40 if is_blackhole else 48
    grid = (8, 5) if is_blackhole else (8, 6)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(START_CORE, num_cores, sub_grids, row_wise=True),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
    )


def _to_paged_cache(kv_bnsd: torch.Tensor, batch: int, n_kv: int, block_size: int) -> torch.Tensor:
    """[B, n_kv, S, D] -> [B * num_blocks_per_seq, n_kv, block_size, D] (attention paged pool layout)."""
    seq_len = kv_bnsd.shape[2]
    blocks_per_seq = seq_len // block_size
    return (
        kv_bnsd.reshape(batch, n_kv, blocks_per_seq, block_size, HEAD_DIM)
        .transpose(1, 2)
        .reshape(batch * blocks_per_seq, n_kv, block_size, HEAD_DIM)
    )


def _repeat_kv_to_nh(kv_bnkv_sd: torch.Tensor, nh: int, n_kv: int) -> torch.Tensor:
    """[B, n_kv, S, D] -> [B, nh, S, D] (MQA/GQA golden)."""
    g = nh // n_kv
    return torch.cat([kv_bnkv_sd[:, i : i + 1, :, :].repeat(1, g, 1, 1) for i in range(n_kv)], dim=1)


def _read_sdpa_output(tt_out: ttnn.Tensor, nh: int) -> torch.Tensor:
    """Read [1, B, nh, D] from mesh; prefer `to_torch` on replicated height-sharded output."""
    try:
        out = ttnn.to_torch(tt_out)
    except Exception:
        shards = ttnn.get_device_tensors(tt_out.cpu())
        assert shards
        out = ttnn.to_torch(shards[0])
    return out[:, :, :nh, :]


@pytest.mark.parametrize("mesh_device", [pytest.param(CLUSTER_SHAPE, id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_qwen_decode_paged_scaled_dot_product_attention_decode_normal_path(mesh_device):
    """Fused paged decode SDPA vs torch under per-device Qwen MQA shapes (nh=8, n_kv=1, b=8)."""

    assert mesh_device.get_num_devices() == MESH_NUM_DEVICES
    assert N_LOCAL_KV_HEADS == 1, "Qwen Galaxy decode SDPA is MQA per device (one KV head per chip group)"
    assert N_LOCAL_HEADS % N_LOCAL_KV_HEADS == 0

    nh = N_LOCAL_HEADS
    n_kv = N_LOCAL_KV_HEADS
    b = BATCH_PER_DEVICE_GROUP
    s = MAX_SEQ_LEN
    block_size = PAGE_BLOCK_SIZE
    assert s % block_size == 0

    sub_grid_y = 7 if ttnn.get_arch_name().lower() == "blackhole" else 9
    is_blackhole = sub_grid_y == 7
    grids = _galaxy_sub_core_grids(sub_grid_y)
    gsz = mesh_device.compute_with_storage_grid_size()
    bb = grids.bounding_box()
    if bb.end.x >= gsz.x or bb.end.y >= gsz.y:
        pytest.skip(f"subcore grids exceed worker grid ({gsz}) bounding_box_end=({bb.end.x},{bb.end.y})")

    padded_q_heads = nearest_pow_2(nearest_n(nh, n=32))
    prog = _paged_sdpa_decode_progcfg_qwen(grids, is_blackhole)
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    dram = ttnn.DRAM_MEMORY_CONFIG
    q_mem = _q_l1_height_sharded(b, padded_q_heads, grids)
    out_mem = _sdpa_out_memcfg_qwen(b, grids)

    torch.manual_seed(_TORCH_SEED)
    K = fa_rand(b, n_kv, s, HEAD_DIM)
    V = fa_rand(b, n_kv, s, HEAD_DIM)
    Q = fa_rand(1, b, nh, HEAD_DIM)

    blocks_per_seq = s // block_size
    max_num_blocks = b * blocks_per_seq
    paged_k = _to_paged_cache(K, b, n_kv, block_size)
    paged_v = _to_paged_cache(V, b, n_kv, block_size)
    permutation = torch.randperm(max_num_blocks)
    page_table = torch.argsort(permutation).reshape(b, blocks_per_seq)
    paged_k = paged_k[permutation]
    paged_v = paged_v[permutation]

    start_indices = [DECODE_CUR_POS] * b
    k_chunk = get_chunk_size(max(start_indices) + 1, s)
    padded_layer_len = nearest_n(max(start_indices) + 1, n=k_chunk)
    scale = HEAD_DIM**-0.5

    attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
    for bi, pos in enumerate(start_indices):
        attn_mask[bi, :, :, pos + 1 :] = torch.finfo(torch.float32).min

    tt_K = ttnn.from_torch(
        paged_k,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
        mesh_mapper=replicate,
    )
    tt_V = ttnn.from_torch(
        paged_v,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
        mesh_mapper=replicate,
    )
    tt_Q = ttnn.from_torch(
        Q[:, :, :nh],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem,
        mesh_mapper=replicate,
    )
    tt_page_table = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate,
    )
    cur_pos_tt = ttnn.from_torch(
        torch.tensor(start_indices, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        tt_page_table,
        cur_pos_tensor=cur_pos_tt,
        scale=scale,
        program_config=prog,
        compute_kernel_config=ck,
        memory_config=out_mem,
    )

    got = _read_sdpa_output(tt_out, nh)

    q_bnh1d = Q[:, :, :nh, :].permute(1, 2, 0, 3)
    k_bnhsd = _repeat_kv_to_nh(K[:, :, :padded_layer_len, :], nh, n_kv)
    v_bnhsd = _repeat_kv_to_nh(V[:, :, :padded_layer_len, :], nh, n_kv)
    attn_bnh1s = attn_mask[:, :nh, :, :]
    golden = (
        torch.nn.functional.scaled_dot_product_attention(
            q_bnh1d,
            k_bnhsd,
            v_bnhsd,
            attn_bnh1s,
            scale=scale,
            is_causal=False,
        )
        .squeeze(2)
        .unsqueeze(0)
    )

    ok, info = comp_pcc(golden, got, _MIN_PCC)
    assert ok, f"paged SDPA decode vs torch golden failed {info}"

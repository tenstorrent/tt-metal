# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone fused **decode SDPA** repro aligned with Qwen Galaxy demos (no `models/`, no checkpoints).

Golden and tensor layout match **`sdpa_test_utils.run_test_sdpa_decode_single_iter`** (GQA, causal mask built
from **`cur_pos_tensor`**). **`SDPAProgramConfig`** mirrors **`qwen_model_config.py` → `SDPA_DECODE_PROGCFG`**
(non-paged: 8×4 compute grid, **q_chunk / k_chunk 256**, 32 cores from **`CoreCoord(1, 0)`** on the BH/WH
galaxy sub-grid).

Mesh: **`ReplicateTensorToMesh(mesh_device)`** on Q/K/V and **`cur_pos`**; **`device_params`** /
**`fabric_config`** resolved under **`operations/ccl/conftest.py`**. Output is **DRAM** (GQA disallows HEIGHT_SHARDED
output on device). Host readback **`tt_out.cpu()`** then **`get_device_tensors` → `to_torch`** —
same convention as **`tests/ttnn/distributed/test_multidevice_TG.py`**.
"""


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
MAX_BATCH_USERS = 32
N_KV_HEADS = 8
N_HEADS = 64
HEAD_DIM = 128
_DECODE_SEQ_CAP = 2048
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


def _q_l1_height_sharded(
    batch: int, padded_q_heads: int, sub_grids: ttnn.CoreRangeSet
) -> ttnn.MemoryConfig:
    shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        START_CORE, batch, sub_grids, row_wise=True
    )
    shard_spec = ttnn.ShardSpec(shard_grid, (padded_q_heads, HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _sdpa_decode_progcfg_qwen_demo(sub_grids: ttnn.CoreRangeSet) -> ttnn.SDPAProgramConfig:
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
            START_CORE, 32, sub_grids, row_wise=True
        ),
        exp_approx_mode=False,
        q_chunk_size=256,
        k_chunk_size=256,
    )


def _repeat_kv_to_nh(kv_bnkv_sd: torch.Tensor, nh: int, n_kv: int) -> torch.Tensor:
    """[B, n_kv, S, D] → [B, nh, S, D] with standard GQA head grouping."""
    g = nh // n_kv
    return torch.cat([kv_bnkv_sd[:, i : i + 1, :, :].repeat(1, g, 1, 1) for i in range(n_kv)], dim=1)


def _maybe_assert_heads_match_q(shard_tt, nh: int) -> None:
    fn = getattr(shard_tt, "logical_shape", None)
    if fn is None:
        return
    raw = fn() if callable(fn) else fn
    ls = tuple(raw)
    assert ls[2] == nh, f"logical head dim must stay {nh}, got logical={ls} padded={tuple(shard_tt.shape)}"


def _replicated_shard0_torch(tt_mesh_out: ttnn.Tensor, nh: int) -> torch.Tensor:
    shards = ttnn.get_device_tensors(tt_mesh_out.cpu())
    assert shards
    shard0_tt = shards[0]
    _maybe_assert_heads_match_q(shard0_tt, nh)
    tt_cpu = ttnn.to_torch(shard0_tt)
    return tt_cpu[:, :, :nh, :]


@pytest.mark.parametrize("mesh_device", [pytest.param(CLUSTER_SHAPE, id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
def test_qwen_decode_scaled_dot_product_attention_decode_normal_path(mesh_device):
    """Fused `scaled_dot_product_attention_decode` vs torch SDPA under Qwen-ish GQA decode shapes on 8×4 mesh."""

    mesh_n = CLUSTER_SHAPE[0] * CLUSTER_SHAPE[1]
    assert mesh_device.get_num_devices() == mesh_n

    nh = N_HEADS // N_KV_HEADS
    n_kv = N_KV_HEADS
    b = MAX_BATCH_USERS // (mesh_n // N_KV_HEADS)
    s = _DECODE_SEQ_CAP

    padded_q_heads = nearest_pow_2(nearest_n(nh, n=32))
    sub_grid_y = 7 if ttnn.get_arch_name().lower() == "blackhole" else 9
    grids = _galaxy_sub_core_grids(sub_grid_y)
    gsz = mesh_device.compute_with_storage_grid_size()
    bb = grids.bounding_box()
    if bb.end.x >= gsz.x or bb.end.y >= gsz.y:
        pytest.skip(f"subcore grids exceed worker grid ({gsz}) bounding_box_end=({bb.end.x},{bb.end.y})")

    prog = _sdpa_decode_progcfg_qwen_demo(grids)
    ck_wormhole = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    dram = ttnn.DRAM_MEMORY_CONFIG
    q_mem = _q_l1_height_sharded(b, padded_q_heads, grids)

    torch.manual_seed(_TORCH_SEED)
    K = fa_rand(b, n_kv, s, HEAD_DIM)
    V = fa_rand(b, n_kv, s, HEAD_DIM)
    Q = fa_rand(1, b, nh, HEAD_DIM)

    start_indices = [s // 2 for _ in range(b)]
    k_chunk = get_chunk_size(max(start_indices) + 1, s)
    padded_layer_len = nearest_n(max(start_indices) + 1, n=k_chunk)
    scale = HEAD_DIM**-0.5

    attn_mask = torch.zeros((b, nh, 1, padded_layer_len))
    for bi, pos in enumerate(start_indices):
        attn_mask[bi, :, :, pos + 1 :] = torch.finfo(torch.float32).min

    tt_K = ttnn.from_torch(K, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate)
    tt_V = ttnn.from_torch(V, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=replicate)
    tt_Q = ttnn.from_torch(
        Q[:, :, :nh],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem,
        mesh_mapper=replicate,
    )
    cur_pos_tt = ttnn.from_torch(
        torch.tensor(start_indices, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=replicate,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_out = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        cur_pos=start_indices,
        cur_pos_tensor=cur_pos_tt,
        scale=scale,
        program_config=prog,
        compute_kernel_config=ck_wormhole,
        memory_config=dram,
    )

    got = _replicated_shard0_torch(tt_out, nh)

    q_bnh1d = Q[:, :, :nh, :].permute(1, 2, 0, 3)
    k_bnhsd = _repeat_kv_to_nh(K[:, :, :padded_layer_len, :], nh, n_kv)
    v_bnhsd = _repeat_kv_to_nh(V[:, :, :padded_layer_len, :], nh, n_kv)
    attn_bnh1s = attn_mask[:, :nh, :, :]
    golden = torch.nn.functional.scaled_dot_product_attention(
        q_bnh1d,
        k_bnhsd,
        v_bnhsd,
        attn_bnh1s,
        scale=scale,
        is_causal=False,
    ).squeeze(2).unsqueeze(0)

    ok, info = comp_pcc(golden, got, _MIN_PCC)
    assert ok, f"SDPA decode vs torch golden failed {info}"

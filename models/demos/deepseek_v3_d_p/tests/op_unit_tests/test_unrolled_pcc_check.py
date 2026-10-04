# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the V-unrolled ring-SDPA harness.

Q and K stay in latent space (576); only V is materialised per head (128), i.e. W_UV moved from
after the SDPA to before it. Cache is laid out slab-major, matching the op-owner reference in
tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::kv_pad_rotation_destinations.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_ring_sdpa_unrolled_sweep import (
    LATENT_DIM,
    NUM_HEADS,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    SP,
    TP,
    V_HEAD_DIM,
    _slab_major,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import create_global_semaphores, per_axis_topology


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (SP, TP),
            torus_xy_device_params(trace_region_size=4 * 1024 * 1024),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(SP, TP), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("kv_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["kv_bf16", "kv_bf8"])
@pytest.mark.timeout(600)
def test_v_unrolled_matches_torch(mesh_device, device_params, kv_dtype):
    chunk, prefix, q_chunk, k_chunk = 2048, 2048, 128, 256
    chunk_local, heads_local = chunk // SP, NUM_HEADS // TP
    logical_n = capacity = prefix + chunk
    scale = (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5

    torch.manual_seed(7)
    q_log = torch.randn(1, NUM_HEADS, chunk, LATENT_DIM)
    k_log = torch.randn(1, 1, capacity, LATENT_DIM)  # shared latent, nhk=1
    v_log = torch.randn(1, NUM_HEADS, capacity, V_HEAD_DIM)  # materialised per head

    grid = mesh_device.compute_with_storage_grid_size()
    sdpa_grid = ttnn.CoreCoord(grid.x - 1, grid.y)
    sp_topology, _ = per_axis_topology()

    def up(t, dtype, dims):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(SP, TP), dims=dims),
        )

    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        up(q_log, ttnn.bfloat16, [2, 1]),
        up(_slab_major(k_log, chunk, capacity), kv_dtype, [2, None]),
        up(_slab_major(v_log, chunk, capacity), kv_dtype, [2, 1]),
        None,
        None,
        None,
        persistent_output_buffer_k=up(torch.zeros(1, 1, capacity, LATENT_DIM), kv_dtype, [None, None]),
        persistent_output_buffer_v=up(torch.zeros(1, NUM_HEADS, capacity, V_HEAD_DIM), kv_dtype, [None, 1]),
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_grid,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        ),
        scale=scale,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
        dim=2,
        multi_device_global_semaphore=create_global_semaphores(
            mesh_device,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}),
            0,
        ),
        num_links=2,
        cluster_axis=0,
        mesh_device=mesh_device,
        topology=sp_topology,
        ccl_core_grid_offset=(grid.x - 1, 0),
        use_column_major_ccl=True,
        is_causal=True,
        is_balanced=False,
        kv_cache_batch_idx=0,
        kv_actual_isl=prefix,
    )
    ttnn.synchronize_device(mesh_device)

    shards = [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(out)]
    kpos = torch.arange(capacity).view(1, -1)
    worst, worst_at, best = 1.0, None, -2.0
    for s in range(SP):
        q_abs = prefix + s * chunk_local  # this device's absolute Q start
        qpos = q_abs + torch.arange(chunk_local).view(-1, 1)
        mask = (kpos > qpos).view(1, 1, chunk_local, capacity)
        for t in range(TP):
            qd = q_log[:, t * heads_local : (t + 1) * heads_local, s * chunk_local : (s + 1) * chunk_local].float()
            kd = k_log.expand(1, heads_local, capacity, LATENT_DIM).float()
            vd = v_log[:, t * heads_local : (t + 1) * heads_local].float()
            sc = (qd @ kd.transpose(-2, -1)) * scale
            exp = torch.softmax(sc.masked_fill(mask, float("-inf")), dim=-1) @ vd
            p = _pcc(shards[s * TP + t], exp)
            best = max(best, p)
            if p < worst:
                worst, worst_at = p, (s, t)
    print(f"\n  kv_dtype={kv_dtype}: worst PCC {worst:.6f} (sp={worst_at[0]},tp={worst_at[1]})  best {best:.6f}")
    # Floor is set by bf16 Q + HiFi2 accumulation over a 4096-long softmax, not by the K/V dtype:
    # bf16 and bf8 K/V agree to five decimals (0.998653 vs 0.998660). A structural error shows up
    # far lower and non-uniform -- the contiguous-cache bug this test caught scored 0.581.
    floor = 0.998
    assert worst > floor, f"worst PCC {worst:.6f} < {floor}: V-unrolled harness is not computing full causal attention"

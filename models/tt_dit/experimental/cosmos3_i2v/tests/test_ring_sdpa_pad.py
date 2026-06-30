# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Targeted ring_joint_scaled_dot_product_attention probe at cosmos3 production shape.

The existing unit test (`tests/unit/test_ring_joint_attention.py`) pads K with
zeros and relies on `logical_n` to mask the pad. It passes at Wan/Mochi/Flux
shapes. But cosmos3 at 128x128x5 production has a much tinier real-vs-pad ratio
(real K = 32 + 69, padded = 256 + 128 = 384) and 64 Q heads / 8 KV heads with
head_dim=128 — significantly different from the Wan family.

This test calls `ring_joint_scaled_dot_product_attention` directly with random
Q/K/V at exactly the cosmos3 per-chip shape, zero-pads K (matching the actual
runtime behavior), and compares against torch SDPA on the *unpadded* logical
inputs. If the ring op's `logical_n` truly masks K beyond logical_n, output
PCC should be ≥ 0.99. If not, we have the kernel-level bug confirmed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.tt_dit.utils.test import line_params

_PARENT_MESH = (4, 8)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(_PARENT_MESH, line_params, id="bh_galaxy_parent")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "submesh_shape",
    [pytest.param((2, 8), id="sp2_tp8"), pytest.param((4, 8), id="sp4_tp8")],
)
@pytest.mark.timeout(300)
def test_ring_sdpa_zero_k_pad_at_cosmos3_shape(mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int]) -> None:
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(submesh.shape)
    sp_axis = 0
    tp_axis = 1
    sp_factor = mesh_shape[sp_axis]
    tp_factor = mesh_shape[tp_axis]

    # Cosmos3 production per-chip shapes after TP=8 split:
    #   global: 64 Q heads, 8 KV heads, head_dim=128
    #   per chip: nh = 8 (since 64/8 TP), n_kv = 1, kv broadcast to nh=8
    # base_seq_len = N_gen logical (32), padded = 256 (k_chunk*sp), per-chip padded = 128.
    nh = 8
    head_dim = 128
    base_n_gen = 32
    # Use small chunk sizes so logical_n=32 sits at a chunk boundary. Theory:
    # the ring kernel may mask only whole chunks beyond logical_n; if so,
    # smaller k_chunk_size should improve K-pad masking.
    # k_chunk/q_chunk < 128 hang at this mesh+shape combo — keep production values.
    q_chunk_size = 128
    k_chunk_size = 128
    padded_n_gen = k_chunk_size * sp_factor  # 128 * 2 = 256
    per_chip_n_gen = padded_n_gen // sp_factor  # = 128
    joint_padded = 128
    base_n_und = 69

    torch.manual_seed(42)

    # Reference (unpadded) Q/K/V — gen Q attends to gen K + und K, no zero-K-pad.
    Q_gen = torch.randn(1, nh, base_n_gen, head_dim, dtype=torch.bfloat16)
    K_gen = torch.randn(1, nh, base_n_gen, head_dim, dtype=torch.bfloat16)
    V_gen = torch.randn(1, nh, base_n_gen, head_dim, dtype=torch.bfloat16)
    Q_und = torch.randn(1, nh, base_n_und, head_dim, dtype=torch.bfloat16)
    K_und = torch.randn(1, nh, base_n_und, head_dim, dtype=torch.bfloat16)
    V_und = torch.randn(1, nh, base_n_und, head_dim, dtype=torch.bfloat16)

    # Pad gen Q/K/V — Q/V with zeros; K with -1e4 to test the workaround. If the
    # ring kernel's logical_n was working, K pad value wouldn't matter. With the bug,
    # -1e4 K pad should suppress softmax weight at pad positions and recover PCC.
    pad_gen = padded_n_gen - base_n_gen
    K_PAD_VALUE = -1.0e4
    Q_gen_pad = torch.cat([Q_gen, torch.zeros(1, nh, pad_gen, head_dim, dtype=torch.bfloat16)], dim=2)
    K_gen_pad = torch.cat([K_gen, torch.full((1, nh, pad_gen, head_dim), K_PAD_VALUE, dtype=torch.bfloat16)], dim=2)
    V_gen_pad = torch.cat([V_gen, torch.zeros(1, nh, pad_gen, head_dim, dtype=torch.bfloat16)], dim=2)

    # Pad und Q/K/V — K pad uses -1e4 (matches _pad_for_joint behavior).
    pad_und = joint_padded - base_n_und
    Q_und_pad = torch.cat([Q_und, torch.zeros(1, nh, pad_und, head_dim, dtype=torch.bfloat16)], dim=2)
    K_und_pad = torch.cat([K_und, torch.full((1, nh, pad_und, head_dim), -1.0e4, dtype=torch.bfloat16)], dim=2)
    V_und_pad = torch.cat([V_und, torch.zeros(1, nh, pad_und, head_dim, dtype=torch.bfloat16)], dim=2)

    # Ground truth: torch SDPA on UN-padded inputs. Gen Q attends to (gen_K, und_K).
    full_Q = torch.cat([Q_gen, Q_und], dim=2)
    full_K = torch.cat([K_gen, K_und], dim=2)
    full_V = torch.cat([V_gen, V_und], dim=2)
    gt_full = F.scaled_dot_product_attention(full_Q, full_K, full_V, is_causal=False)
    gt_gen = gt_full[:, :, :base_n_gen, :]

    # Set up ring SDPA op exactly like cosmos3 attention does.
    grid = ttnn.CoreCoord(11, 10)
    ring_grid = ttnn.CoreCoord(grid.x, max(grid.y - 1, 1))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ring_grid,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        submesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group([worker_sub_device_id])

    ccl_semaphore = [ttnn.create_global_semaphore(submesh, ccl_sub_device_crs, 0) for _ in range(2)]

    # Q/K/V on device: spatial (gen) is sp-sharded on seq dim (dim=2 in BHNE).
    sdpa_input_shard_dims = [None, None]
    sdpa_input_shard_dims[sp_axis] = 2
    sdpa_joint_shard_dims = [None, None]  # joint (und) is REPLICATED

    def shard(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=mesh_shape, dims=sdpa_input_shard_dims),
        )

    def replicate(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=mesh_shape, dims=sdpa_joint_shard_dims),
        )

    tt_Q_gen = shard(Q_gen_pad)
    tt_K_gen = shard(K_gen_pad)
    tt_V_gen = shard(V_gen_pad)
    tt_Q_und = replicate(Q_und_pad)
    tt_K_und = replicate(K_und_pad)
    tt_V_und = replicate(V_und_pad)

    pp_k = ttnn.from_torch(
        torch.zeros(1, nh, padded_n_gen, head_dim),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=mesh_shape, dims=[None, None]),
    )
    pp_v = ttnn.from_torch(
        torch.zeros(1, nh, padded_n_gen, head_dim),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=mesh_shape, dims=[None, None]),
    )

    tt_out, _joint_out, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_Q_gen,
        tt_K_gen,
        tt_V_gen,
        tt_Q_und,
        tt_K_und,
        tt_V_und,
        persistent_output_buffer_k=pp_k,
        persistent_output_buffer_v=pp_v,
        joint_strategy="rear",
        logical_n=base_n_gen,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_semaphore,
        num_links=1,
        cluster_axis=sp_axis,
        mesh_device=submesh,
        topology=ttnn.Topology.Linear,
        subdevice_id=worker_sub_device_id,
        ccl_core_grid_offset=(0, ring_grid.y),
    )

    # Gather sp-sharded output across sp_axis (row-major: chip i = row sp_idx, col 0 has the slice
    # for sp_idx; same row's other cols are TP-replicated copies). Concat slices on dim 2.
    devs = ttnn.get_device_tensors(tt_out)
    per_sp = [ttnn.to_torch(devs[sp_idx * tp_factor]) for sp_idx in range(sp_factor)]
    full = torch.cat(per_sp, dim=2)
    out_gen = full[:, :, :base_n_gen, :]

    # Compute PCC + RMSE/sigma + std.
    a = gt_gen.detach().to(torch.float32).flatten()
    b = out_gen.detach().to(torch.float32).flatten()
    am, bm = a.mean(), b.mean()
    pcc_val = float(((a - am) * (b - bm)).sum() / ((((a - am) ** 2).sum() * ((b - bm) ** 2).sum()).sqrt() + 1e-12))
    rmse_sigma = float(((a - b) ** 2).mean().sqrt() / a.std())
    print(
        f"\n[ring_sdpa probe nh={nh} base={base_n_gen} pad={pad_gen} joint={base_n_und}/{pad_und}] "
        f"PCC={pcc_val:.6f} RMSE/sigma={rmse_sigma:.4f} "
        f"gt_std={gt_gen.to(torch.float32).std().item():.4f} tt_std={out_gen.to(torch.float32).std().item():.4f}",
        flush=True,
    )
    # Dump for cross-test comparison with the std-SDPA baseline.
    _dump_dir = "/tmp/cosmos3_sdpa_compare"
    import os as _os

    _os.makedirs(_dump_dir, exist_ok=True)
    torch.save(out_gen.detach().cpu(), f"{_dump_dir}/ring_sp{sp_factor}_out.pt")
    torch.save(gt_gen.detach().cpu(), f"{_dump_dir}/torch_ref_out.pt")
    assert pcc_val > 0.5, f"Ring SDPA PCC at cosmos3 shape = {pcc_val:.4f}, expected near 0.999"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(_PARENT_MESH, line_params, id="bh_galaxy_parent")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "submesh_shape",
    [pytest.param((2, 8), id="sp2_tp8"), pytest.param((4, 8), id="sp4_tp8")],
)
@pytest.mark.timeout(300)
def test_ring_sdpa_unpadded_und_at_cosmos3_shape(mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int]) -> None:
    """Hypothesis: the joint (und) K-pad is the bug, not the gen K-pad.

    The current `_pad_for_joint` pre-pads und Q/K/V on seq to a multiple of
    `k_chunk_size` (128). That makes the kernel see `L = 128`, with
    `L % (Sk_chunk_t * TILE) == 0`, so `joint_has_padding=false` and no joint mask
    is emitted. The und K-pad rows then go through softmax with K filled by `-1e4`
    — which does NOT make `q·k_pad` strongly negative (signed q sum gives
    high-variance Gaussian scores), so pad rows dominate softmax and output
    collapses.

    Fix: pass und at its logical length (L=69). ttnn auto-tile-pads to 96 (3 tiles).
    Kernel computes `joint_has_padding = (69 % 128) != 0 = true`, partial-tile
    mask covers rows 69..127 of the joint k_chunk. Expect PCC ≥ 0.99.
    """
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    mesh_shape = tuple(submesh.shape)
    sp_axis = 0
    tp_axis = 1
    sp_factor = mesh_shape[sp_axis]
    tp_factor = mesh_shape[tp_axis]

    nh = 8
    head_dim = 128
    base_n_gen = 32
    q_chunk_size = 128
    k_chunk_size = 128
    padded_n_gen = k_chunk_size * sp_factor
    base_n_und = 69  # NOT padded — passed as logical L=69.

    torch.manual_seed(42)
    Q_gen = torch.randn(1, nh, base_n_gen, head_dim, dtype=torch.bfloat16)
    K_gen = torch.randn(1, nh, base_n_gen, head_dim, dtype=torch.bfloat16)
    V_gen = torch.randn(1, nh, base_n_gen, head_dim, dtype=torch.bfloat16)
    Q_und = torch.randn(1, nh, base_n_und, head_dim, dtype=torch.bfloat16)
    K_und = torch.randn(1, nh, base_n_und, head_dim, dtype=torch.bfloat16)
    V_und = torch.randn(1, nh, base_n_und, head_dim, dtype=torch.bfloat16)

    # Gen Q/K/V padded the same way as before (gen K-pad masking is kernel-handled).
    pad_gen = padded_n_gen - base_n_gen
    Q_gen_pad = torch.cat([Q_gen, torch.zeros(1, nh, pad_gen, head_dim, dtype=torch.bfloat16)], dim=2)
    K_gen_pad = torch.cat([K_gen, torch.zeros(1, nh, pad_gen, head_dim, dtype=torch.bfloat16)], dim=2)
    V_gen_pad = torch.cat([V_gen, torch.zeros(1, nh, pad_gen, head_dim, dtype=torch.bfloat16)], dim=2)

    full_Q = torch.cat([Q_gen, Q_und], dim=2)
    full_K = torch.cat([K_gen, K_und], dim=2)
    full_V = torch.cat([V_gen, V_und], dim=2)
    gt_full = F.scaled_dot_product_attention(full_Q, full_K, full_V, is_causal=False)
    gt_gen = gt_full[:, :, :base_n_gen, :]

    grid = ttnn.CoreCoord(11, 10)
    ring_grid = ttnn.CoreCoord(grid.x, max(grid.y - 1, 1))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ring_grid,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        submesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )

    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group([worker_sub_device_id])

    ccl_semaphore = [ttnn.create_global_semaphore(submesh, ccl_sub_device_crs, 0) for _ in range(2)]

    sdpa_input_shard_dims = [None, None]
    sdpa_input_shard_dims[sp_axis] = 2
    sdpa_joint_shard_dims = [None, None]

    def shard(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=mesh_shape, dims=sdpa_input_shard_dims),
        )

    def replicate(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=mesh_shape, dims=sdpa_joint_shard_dims),
        )

    tt_Q_gen = shard(Q_gen_pad)
    tt_K_gen = shard(K_gen_pad)
    tt_V_gen = shard(V_gen_pad)
    # KEY DIFFERENCE: pass und tensors at logical length 69. ttnn TILE-pads to 96.
    tt_Q_und = replicate(Q_und)
    tt_K_und = replicate(K_und)
    tt_V_und = replicate(V_und)

    pp_k = ttnn.from_torch(
        torch.zeros(1, nh, padded_n_gen, head_dim),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=mesh_shape, dims=[None, None]),
    )
    pp_v = ttnn.from_torch(
        torch.zeros(1, nh, padded_n_gen, head_dim),
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=mesh_shape, dims=[None, None]),
    )

    tt_out, _joint_out, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_Q_gen,
        tt_K_gen,
        tt_V_gen,
        tt_Q_und,
        tt_K_und,
        tt_V_und,
        persistent_output_buffer_k=pp_k,
        persistent_output_buffer_v=pp_v,
        joint_strategy="rear",
        logical_n=base_n_gen,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_semaphore,
        num_links=1,
        cluster_axis=sp_axis,
        mesh_device=submesh,
        topology=ttnn.Topology.Linear,
        subdevice_id=worker_sub_device_id,
        ccl_core_grid_offset=(0, ring_grid.y),
    )

    devs = ttnn.get_device_tensors(tt_out)
    per_sp = [ttnn.to_torch(devs[sp_idx * tp_factor]) for sp_idx in range(sp_factor)]
    full = torch.cat(per_sp, dim=2)
    out_gen = full[:, :, :base_n_gen, :]

    a = gt_gen.detach().to(torch.float32).flatten()
    b = out_gen.detach().to(torch.float32).flatten()
    am, bm = a.mean(), b.mean()
    pcc_val = float(((a - am) * (b - bm)).sum() / ((((a - am) ** 2).sum() * ((b - bm) ** 2).sum()).sqrt() + 1e-12))
    rmse_sigma = float(((a - b) ** 2).mean().sqrt() / a.std())
    print(
        f"\n[unpadded_und probe nh={nh} base={base_n_gen} und_logical={base_n_und}] "
        f"PCC={pcc_val:.6f} RMSE/sigma={rmse_sigma:.4f} "
        f"gt_std={gt_gen.to(torch.float32).std().item():.4f} tt_std={out_gen.to(torch.float32).std().item():.4f}",
        flush=True,
    )
    assert pcc_val > 0.99, f"Unpadded-und ring SDPA PCC = {pcc_val:.4f}, expected ≥ 0.99 if joint-pad mask works"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(_PARENT_MESH, line_params, id="bh_galaxy_parent")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(120)
def test_std_sdpa_at_cosmos3_shape(mesh_device: ttnn.MeshDevice) -> None:
    """Baseline: ttnn.transformer.scaled_dot_product_attention (no ring) on the SAME
    Q/K/V the ring test uses. Single chip, no sharding, no padding mask. If this
    matches torch SDPA at PCC ≈ 0.999, then the ring path is the bug — not the
    op family or our shape. Result dumped to /tmp/cosmos3_sdpa_compare/std_out.pt
    so it can be diffed against ring outputs."""
    nh = 8
    head_dim = 128
    base_n_gen = 32
    base_n_und = 69

    torch.manual_seed(42)
    Q_gen = torch.randn(1, nh, base_n_gen, head_dim, dtype=torch.bfloat16)
    K_gen = torch.randn(1, nh, base_n_gen, head_dim, dtype=torch.bfloat16)
    V_gen = torch.randn(1, nh, base_n_gen, head_dim, dtype=torch.bfloat16)
    Q_und = torch.randn(1, nh, base_n_und, head_dim, dtype=torch.bfloat16)
    K_und = torch.randn(1, nh, base_n_und, head_dim, dtype=torch.bfloat16)
    V_und = torch.randn(1, nh, base_n_und, head_dim, dtype=torch.bfloat16)

    # Same torch ground truth as the ring test.
    full_Q = torch.cat([Q_gen, Q_und], dim=2)
    full_K = torch.cat([K_gen, K_und], dim=2)
    full_V = torch.cat([V_gen, V_und], dim=2)
    gt_full = F.scaled_dot_product_attention(full_Q, full_K, full_V, is_causal=False)
    gt_gen = gt_full[:, :, :base_n_gen, :]

    # The model's no-ring branch concatenates K/V along the seq dim and calls standard SDPA.
    # Pad gen Q to nearest multiple of q_chunk_size=32 (101 → 128). K/V do NOT need k_chunk pad
    # for the non-ring op since N=101 (gen+und).
    q_chunk_size = 32
    k_chunk_size = 32
    n_total = base_n_gen + base_n_und  # 101
    pad_total = ((n_total + k_chunk_size - 1) // k_chunk_size) * k_chunk_size - n_total
    Q_for_op = torch.cat([Q_gen, Q_und, torch.zeros(1, nh, pad_total, head_dim, dtype=torch.bfloat16)], dim=2)
    K_for_op = torch.cat([K_gen, K_und, torch.zeros(1, nh, pad_total, head_dim, dtype=torch.bfloat16)], dim=2)
    V_for_op = torch.cat([V_gen, V_und, torch.zeros(1, nh, pad_total, head_dim, dtype=torch.bfloat16)], dim=2)

    def to_dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    tt_Q = to_dev(Q_for_op)
    tt_K = to_dev(K_for_op)
    tt_V = to_dev(V_for_op)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    tt_out = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=False,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
    # Recover the gen prefix (first base_n_gen rows of the gen+und ordering used above).
    out_gen = out[:, :, :base_n_gen, :]

    a = gt_gen.detach().to(torch.float32).flatten()
    b = out_gen.detach().to(torch.float32).flatten()
    am, bm = a.mean(), b.mean()
    pcc_val = float(((a - am) * (b - bm)).sum() / ((((a - am) ** 2).sum() * ((b - bm) ** 2).sum()).sqrt() + 1e-12))
    rmse_sigma = float(((a - b) ** 2).mean().sqrt() / a.std())
    print(
        f"\n[std_sdpa probe nh={nh} base={base_n_gen} joint={base_n_und}] "
        f"PCC={pcc_val:.6f} RMSE/sigma={rmse_sigma:.4f} "
        f"gt_std={gt_gen.to(torch.float32).std().item():.4f} tt_std={out_gen.to(torch.float32).std().item():.4f}",
        flush=True,
    )
    import os as _os

    _dump_dir = "/tmp/cosmos3_sdpa_compare"
    _os.makedirs(_dump_dir, exist_ok=True)
    torch.save(out_gen.detach().cpu(), f"{_dump_dir}/std_sp1_out.pt")
    torch.save(gt_gen.detach().cpu(), f"{_dump_dir}/torch_ref_out.pt")
    assert pcc_val > 0.99, f"Std SDPA PCC at cosmos3 shape = {pcc_val:.4f}, expected ≥ 0.99"

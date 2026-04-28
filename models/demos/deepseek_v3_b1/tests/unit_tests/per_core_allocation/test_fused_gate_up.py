# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit-test reproducer for the K-split gate→up back-to-back hang in MoE.

Runs MatmulExpertCompressedDRAM::Op twice in the same kernel invocation —
gate_proj's CTArgs then up_proj's. Both ops share cb_in0 (activation), cb_index,
and cb_in1 L1 region (anchored at gate's backing tensor via ResetCBIn1=true).
Configured for K-split (k_parallel_per_bank=2, primary_at_last_offset=True).
"""

import torch

from models.demos.deepseek_v3_b1.fused_ops.fused_gate_up.op import FusedGateUp
from models.demos.deepseek_v3_b1.tests.unit_tests.per_core_allocation.test_matmul_expert import (
    _build_activation_tensor,
    _build_assigner,
    _build_dram_experts,
    _build_dram_output,
    _build_expert_flags,
    _build_index_tensor,
    _build_weight_tensors,
    _compute_dram_matmul_params,
    _setup_core_grids,
    _validate_dram_output,
)


def _run_fused_gate_up(
    mesh_device,
    M,
    K,
    N,
    num_experts,
    dram_expert_ids,
    active_expert_ids,
    formats_per_device,
    num_subblocks_k,
    num_subblocks_n,
    n_parallel_per_bank,
    k_parallel_per_bank,
    primary_at_last_offset,
    dram_fuse_silu=False,
    pcc_threshold=0.97,
    fmt_distribution="random",
    fmt_ratios=None,
):
    cores_per_dram_bank = n_parallel_per_bank * k_parallel_per_bank
    tile_w = 32

    grids = _setup_core_grids(
        mesh_device, cores_per_dram_bank, num_sram_cores=0, sram_cores_override=None, has_sram=False
    )
    dram_core_grid = grids["dram_core_grid"]
    compute_core_grid = grids["compute_core_grid"]
    num_dram_cores = len(grids["dram_cores_list"])
    num_devices = mesh_device.get_num_devices()
    num_banks = num_dram_cores // cores_per_dram_bank
    num_cores = compute_core_grid.num_cores()

    Kt, dram_per_core_N, subblock_k, subblock_n, N_dram_per_device = _compute_dram_matmul_params(
        K,
        N,
        tile_w,
        num_banks,
        num_dram_cores,
        num_dram_cores,
        cores_per_dram_bank,
        num_subblocks_k,
        num_subblocks_n,
        k_parallel_per_bank=k_parallel_per_bank,
    )

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    dram_meta_flags, is_dram_flags = _build_expert_flags(num_experts, [], dram_expert_ids)
    assigner = _build_assigner(formats_per_device)

    # Build TWO sets of weights — gate and up — using different seeds via
    # the existing helper (each call re-seeds per-expert via `eidx * 1000 + 42`,
    # so to get distinct gate vs up tensors we build them once each but reuse
    # the helper's seeding scheme; since the test only cares about not hanging,
    # PCC validation is best-effort).
    torch_b_gate = _build_weight_tensors(
        num_experts, K, N, N_dram_per_device, set(), formats_per_device, num_devices, fmt_distribution, fmt_ratios
    )
    torch_b_up = _build_weight_tensors(
        num_experts, K, N, N_dram_per_device, set(), formats_per_device, num_devices, fmt_distribution, fmt_ratios
    )

    gate_dram_cts, gate_dram_meta = _build_dram_experts(
        dram_expert_ids,
        torch_b_gate,
        assigner,
        mesh_device,
        K,
        dram_per_core_N,
        N_dram_per_device,
        num_banks,
        cores_per_dram_bank,
        num_experts,
        dram_meta_flags,
        mesh_device.shape[0],
        mesh_device.shape[1],
        subblock_k,
        subblock_n,
        Kt,
        tile_w,
        k_parallel_per_bank=k_parallel_per_bank,
        primary_at_last_offset=primary_at_last_offset,
    )
    up_dram_cts, up_dram_meta = _build_dram_experts(
        dram_expert_ids,
        torch_b_up,
        assigner,
        mesh_device,
        K,
        dram_per_core_N,
        N_dram_per_device,
        num_banks,
        cores_per_dram_bank,
        num_experts,
        dram_meta_flags,
        mesh_device.shape[0],
        mesh_device.shape[1],
        subblock_k,
        subblock_n,
        Kt,
        tile_w,
        k_parallel_per_bank=k_parallel_per_bank,
        primary_at_last_offset=primary_at_last_offset,
    )

    a_tensor = _build_activation_tensor(torch_a, mesh_device, compute_core_grid, num_cores, M, K, tile_w)
    index_tensor = _build_index_tensor(active_expert_ids, mesh_device, compute_core_grid, num_cores, is_dram_flags)

    num_active_experts = len(active_expert_ids)
    gate_out_tensor = _build_dram_output(
        mesh_device,
        M,
        dram_per_core_N,
        num_active_experts,
        num_dram_cores,
        num_devices,
        dram_core_grid,
        tile_w,
        dram_fuse_silu=dram_fuse_silu,
    )
    up_out_tensor = _build_dram_output(
        mesh_device,
        M,
        dram_per_core_N,
        num_active_experts,
        num_dram_cores,
        num_devices,
        dram_core_grid,
        tile_w,
        dram_fuse_silu=False,  # up never has silu
    )

    FusedGateUp.op(
        a_tensor=a_tensor,
        gate_dram_cts=gate_dram_cts,
        up_dram_cts=up_dram_cts,
        gate_output_tensor=gate_out_tensor,
        up_output_tensor=up_out_tensor,
        index_tensor=index_tensor,
        gate_dram_meta_tensors=gate_dram_meta,
        up_dram_meta_tensors=up_dram_meta,
        num_active_experts=num_active_experts,
        subblock_k=subblock_k,
        subblock_n=subblock_n,
        dram_per_core_n=dram_per_core_N,
        dram_core_grid=dram_core_grid,
        cores_per_dram_bank=cores_per_dram_bank,
        k_parallel_per_bank=k_parallel_per_bank,
        primary_at_last_offset=primary_at_last_offset,
        dram_fuse_silu=dram_fuse_silu,
    )

    # PCC validation — log gate's PCCs first, then up's, then assert.
    # Each _validate_dram_output asserts on the first failure; we wrap so we
    # see the full picture for both ops before failing.
    from loguru import logger

    active_dram = [eid for eid in active_expert_ids if eid in set(dram_expert_ids)]
    failures = []
    logger.info("=== gate PCCs ===")
    try:
        _validate_dram_output(
            gate_out_tensor,
            torch_a,
            torch_b_gate,
            active_dram,
            num_active_experts,
            dram_per_core_N,
            num_dram_cores,
            pcc_threshold,
            dram_fuse_silu,
            tile_w,
            M=M,
            tp_expert=True,
            cores_per_dram_bank=cores_per_dram_bank,
            k_parallel_per_bank=k_parallel_per_bank,
            primary_at_last_offset=primary_at_last_offset,
        )
    except AssertionError as e:
        failures.append(f"gate: {e}")
        logger.error(f"gate validation failed: {e}")
    logger.info("=== up PCCs ===")
    try:
        _validate_dram_output(
            up_out_tensor,
            torch_a,
            torch_b_up,
            active_dram,
            num_active_experts,
            dram_per_core_N,
            num_dram_cores,
            pcc_threshold,
            False,  # up has no silu
            tile_w,
            M=M,
            tp_expert=True,
            cores_per_dram_bank=cores_per_dram_bank,
            k_parallel_per_bank=k_parallel_per_bank,
            primary_at_last_offset=primary_at_last_offset,
        )
    except AssertionError as e:
        failures.append(f"up: {e}")
        logger.error(f"up validation failed: {e}")
    if failures:
        raise AssertionError("PCC failures:\n" + "\n".join(failures))


def test_fused_gate_up_kspilt_primary_at_last_offset(device):
    """Repro for the K-split gate→up back-to-back hang in MoE.

    Same K-split config as test_benchmark_gate_proj_primary_at_last_offset, but
    runs gate AND up in the SAME kernel invocation so we can isolate any
    state carryover between the two ops (shared cb_in1, leftover TRISC state, etc).
    bfp4-only matches the actual MoE config.
    """
    _run_fused_gate_up(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=8,
        dram_expert_ids=list(range(8)),
        active_expert_ids=[2, 3, 4, 5, 6, 7, 1, 0],
        formats_per_device=[["bfp4"]],
        num_subblocks_k=4,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        primary_at_last_offset=True,
        dram_fuse_silu=True,  # gate has silu, up doesn't (handled inside op.py)
        fmt_distribution="uniform",
        fmt_ratios={"bfp4": 100.0},
    )

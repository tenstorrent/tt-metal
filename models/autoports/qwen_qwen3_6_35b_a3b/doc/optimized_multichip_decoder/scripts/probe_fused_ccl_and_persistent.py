# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe fused CCL+matmul and persistent CCL buffers for Qwen multichip shapes.

This script is an optimized-multichip-decoder evidence artifact.  It does not
modify the model path.  It uses the target 2x2 mesh and Qwen decode-sized
tensors to compare:

* current row-parallel matmul plus TP all-reduce;
* all-gather-minimal-matmul plus output all-gather with adapted output-sharded
  weights and persistent gather buffers;
* explicit reduce-scatter plus all-gather with and without persistent buffers
  for TP and EP axes.
"""

from __future__ import annotations

import argparse
import builtins
import functools
import statistics
import time
import traceback

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import _col_mapper, _row_mapper
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    get_tt_ccl,
)
from models.common.utility_functions import comp_pcc

print = functools.partial(builtins.print, flush=True)


def _mesh_bf16(tensor: torch.Tensor, mesh_device, mesh_mapper) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _mesh_weight(tensor: torch.Tensor, mesh_device, mesh_mapper, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _first_local_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    local = ttnn.get_device_tensors(tensor)[0]
    return local.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().float()


def _timed_median_ms(mesh_device, fn, iterations: int) -> tuple[float, ttnn.Tensor]:
    out = fn()
    ttnn.synchronize_device(mesh_device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), out


def _minimal_matmul_config() -> ttnn.MinimalMatmulConfig:
    return ttnn.MinimalMatmulConfig(
        M_block_size=1,
        K_block_size=8,
        N_block_size=4,
        subblock_h=1,
        subblock_w=1,
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 2),
    )


def _public_all_reduce(tensor: ttnn.Tensor, *, cluster_axis: int, num_links: int = 2) -> ttnn.Tensor:
    return ttnn.all_reduce(tensor, num_links=num_links, topology=ttnn.Topology.Ring, cluster_axis=cluster_axis)


def _run_fused_agmm(
    mesh_device,
    iterations: int,
    weight_dtype,
    *,
    use_persistent: bool,
    force_transpose: bool,
    num_links: int,
    num_workers_per_link: int,
) -> None:
    torch.manual_seed(7361)
    shape = (1, 1, 32, 2048)
    activation = (torch.randn(shape, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    weight = (torch.randn((1, 1, 2048, 2048), dtype=torch.float32) * 0.01).to(torch.bfloat16)

    act_tt = _mesh_bf16(activation, mesh_device, _col_mapper(mesh_device))
    weight_row = _mesh_weight(weight, mesh_device, _row_mapper(mesh_device), weight_dtype)
    weight_col = _mesh_weight(weight, mesh_device, _col_mapper(mesh_device), weight_dtype)
    ag_activation_buffer = _mesh_bf16(
        torch.zeros(shape, dtype=torch.bfloat16), mesh_device, ttnn.ReplicateTensorToMesh(mesh_device)
    )
    ag_output_buffer = _mesh_bf16(
        torch.zeros(shape, dtype=torch.bfloat16), mesh_device, ttnn.ReplicateTensorToMesh(mesh_device)
    )
    tt_ccl = get_tt_ccl(mesh_device)
    config = _minimal_matmul_config()

    print(f"fused_agmm_weight_dtype={weight_dtype}")
    print(f"fused_agmm_use_persistent={use_persistent}")
    print(f"fused_agmm_force_transpose={force_transpose}")
    print(f"fused_agmm_num_links={num_links}")
    print(f"fused_agmm_activation_shape={shape} activation_mem={act_tt.memory_config()}")
    print(f"fused_agmm_row_weight_mem={weight_row.memory_config()}")
    print(f"fused_agmm_col_weight_mem={weight_col.memory_config()}")
    print(f"fused_agmm_config={config}")
    print(f"fused_agmm_num_workers_per_link={num_workers_per_link}")

    def current_path():
        partial = ttnn.linear(act_tt, weight_row, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return _public_all_reduce(partial, cluster_axis=1, num_links=num_links)

    def fused_path():
        out_parts = ttnn.experimental.all_gather_minimal_matmul_async(
            input_tensor=act_tt,
            weight_tensor=weight_col,
            config=config,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            persistent_output_buffer=ag_activation_buffer if use_persistent else None,
            num_links=num_links,
            cluster_axis=1,
            force_transpose=force_transpose,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        local_out = out_parts[0] if isinstance(out_parts, (tuple, list)) else out_parts
        return ttnn.experimental.all_gather_async(
            local_out,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
            num_links=num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
            barrier_semaphore=None if use_persistent else tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
            persistent_output_buffer=ag_output_buffer if use_persistent else None,
        )

    try:
        current_ms, current_out = _timed_median_ms(mesh_device, current_path, iterations)
        fused_ms, fused_out = _timed_median_ms(mesh_device, fused_path, iterations)
    except Exception as exc:  # noqa: BLE001 - this is an evidence probe.
        print("fused_agmm_status=failed")
        print(f"fused_agmm_failure_type={type(exc).__name__}")
        print(f"fused_agmm_failure={exc}")
        traceback.print_exc()
        return

    expected = _first_local_torch(current_out)
    actual = _first_local_torch(fused_out)
    ok, pcc = comp_pcc(expected.float(), actual.float(), pcc=0.995)
    max_abs = torch.max(torch.abs(expected.float() - actual.float())).item()
    print("fused_agmm_status=ran")
    print(f"current_matmul_all_reduce_ms_median={current_ms:.6f}")
    print(f"fused_agmm_output_all_gather_ms_median={fused_ms:.6f}")
    print(f"fused_agmm_pcc_ok={ok} {pcc}")
    print(f"fused_agmm_max_abs={max_abs:.8f}")
    print(f"fused_agmm_speedup_vs_current={current_ms / fused_ms:.6f}")


def _rsag_all_reduce(
    input_tensor: ttnn.Tensor,
    *,
    mesh_device,
    axis: int,
    persistent: bool,
    persistent_intermediate: ttnn.Tensor | None,
    persistent_reduced: ttnn.Tensor | None,
    persistent_gathered: ttnn.Tensor | None,
) -> ttnn.Tensor:
    tt_ccl = get_tt_ccl(mesh_device)
    reduced = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor=input_tensor,
        persistent_output_buffers=[persistent_intermediate, persistent_reduced] if persistent else None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(axis),
        barrier_semaphore=None if persistent else tt_ccl.get_and_cycle_barrier_semaphore_handle(axis),
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        cluster_axis=axis,
        chunks_per_sync=CCL_CHUNKS_PER_SYNC,
        num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
        num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
    )
    gathered = ttnn.experimental.all_gather_async(
        reduced,
        persistent_output_buffer=persistent_gathered if persistent else None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(axis),
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
        cluster_axis=axis,
        barrier_semaphore=None if persistent else tt_ccl.get_and_cycle_barrier_semaphore_handle(axis),
        chunks_per_sync=CCL_CHUNKS_PER_SYNC,
        num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
        num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
    )
    if not persistent:
        reduced.deallocate(True)
    return gathered


def _run_persistent_rsag(mesh_device, iterations: int, axis: int) -> None:
    torch.manual_seed(8123 + axis)
    shape = (1, 1, 32, 2048)
    reduced_shape = (1, 1, 32, 1024)
    activation = (torch.randn(shape, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    input_tensor = _mesh_bf16(activation, mesh_device, ttnn.ReplicateTensorToMesh(mesh_device))
    persistent_intermediate = _mesh_bf16(
        torch.zeros(shape, dtype=torch.bfloat16), mesh_device, ttnn.ReplicateTensorToMesh(mesh_device)
    )
    persistent_reduced = _mesh_bf16(
        torch.zeros(reduced_shape, dtype=torch.bfloat16), mesh_device, ttnn.ReplicateTensorToMesh(mesh_device)
    )
    persistent_gathered = _mesh_bf16(
        torch.zeros(shape, dtype=torch.bfloat16), mesh_device, ttnn.ReplicateTensorToMesh(mesh_device)
    )

    print(f"persistent_rsag_axis={axis}")
    print(f"persistent_rsag_input_shape={shape} reduced_shape={reduced_shape}")
    print(f"persistent_rsag_intermediate_mem={persistent_intermediate.memory_config()}")
    print(f"persistent_rsag_reduced_mem={persistent_reduced.memory_config()}")
    print(f"persistent_rsag_gathered_mem={persistent_gathered.memory_config()}")

    def nonpersistent_path():
        return _rsag_all_reduce(
            input_tensor,
            mesh_device=mesh_device,
            axis=axis,
            persistent=False,
            persistent_intermediate=None,
            persistent_reduced=None,
            persistent_gathered=None,
        )

    def persistent_path():
        return _rsag_all_reduce(
            input_tensor,
            mesh_device=mesh_device,
            axis=axis,
            persistent=True,
            persistent_intermediate=persistent_intermediate,
            persistent_reduced=persistent_reduced,
            persistent_gathered=persistent_gathered,
        )

    try:
        public = _public_all_reduce(input_tensor, cluster_axis=axis)
        ttnn.synchronize_device(mesh_device)
        nonpersistent_ms, nonpersistent_out = _timed_median_ms(mesh_device, nonpersistent_path, iterations)
        persistent_ms, persistent_out = _timed_median_ms(mesh_device, persistent_path, iterations)
    except Exception as exc:  # noqa: BLE001 - this is an evidence probe.
        print("persistent_rsag_status=failed")
        print(f"persistent_rsag_failure_type={type(exc).__name__}")
        print(f"persistent_rsag_failure={exc}")
        traceback.print_exc()
        return

    public_torch = _first_local_torch(public)
    nonpersistent_torch = _first_local_torch(nonpersistent_out)
    persistent_torch = _first_local_torch(persistent_out)
    nonpersistent_ok, nonpersistent_pcc = comp_pcc(public_torch.float(), nonpersistent_torch.float(), pcc=0.995)
    persistent_ok, persistent_pcc = comp_pcc(public_torch.float(), persistent_torch.float(), pcc=0.995)
    print("persistent_rsag_status=ran")
    print(f"persistent_rsag_nonpersistent_ms_median={nonpersistent_ms:.6f}")
    print(f"persistent_rsag_persistent_ms_median={persistent_ms:.6f}")
    print(f"persistent_rsag_nonpersistent_pcc_ok={nonpersistent_ok} {nonpersistent_pcc}")
    print(f"persistent_rsag_persistent_pcc_ok={persistent_ok} {persistent_pcc}")
    print(f"persistent_rsag_speedup_vs_nonpersistent={nonpersistent_ms / persistent_ms:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--only", choices=("all", "fused", "persistent-rsag"), default="all")
    parser.add_argument("--fused-weight-dtype", choices=("bf8", "bf16", "both"), default="both")
    parser.add_argument("--fused-use-persistent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-force-transpose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-num-links", type=int, default=2)
    parser.add_argument("--fused-workers-per-link", type=int, default=4)
    parser.add_argument("--rsag-axis", choices=("tp", "ep", "both"), default="both")
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), trace_region_size=32_000_000)
    try:
        if args.only in ("all", "fused"):
            if args.fused_weight_dtype == "bf8":
                weight_dtypes = (ttnn.bfloat8_b,)
            elif args.fused_weight_dtype == "bf16":
                weight_dtypes = (ttnn.bfloat16,)
            else:
                weight_dtypes = (ttnn.bfloat8_b, ttnn.bfloat16)
            for weight_dtype in weight_dtypes:
                _run_fused_agmm(
                    mesh_device,
                    args.iterations,
                    weight_dtype,
                    use_persistent=args.fused_use_persistent,
                    force_transpose=args.fused_force_transpose,
                    num_links=args.fused_num_links,
                    num_workers_per_link=args.fused_workers_per_link,
                )
        if args.only in ("all", "persistent-rsag"):
            axes = {"tp": (1,), "ep": (0,), "both": (1, 0)}[args.rsag_axis]
            for axis in axes:
                _run_persistent_rsag(mesh_device, args.iterations, axis)
    finally:
        ttnn.synchronize_device(mesh_device)
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

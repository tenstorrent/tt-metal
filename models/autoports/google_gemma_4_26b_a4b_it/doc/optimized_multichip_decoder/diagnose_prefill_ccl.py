# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replay captured prefill rank inputs through separate native RS and AG ops."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager


class FullGridCCLManager(CCLManager):
    def _init_subdevice(self):
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)


def rows(mask):
    return mask.reshape(-1, *mask.shape[-2:]).sum(dim=(0, 2)).tolist()


def compare(value, reference):
    difference = value - reference
    return dict(
        changed=int((value != reference).sum()),
        changed_per_row=rows(value != reference),
        nonfinite=int((~torch.isfinite(value)).sum()),
        max_abs=float(difference.abs().max()),
        relative_frobenius=float(difference.norm() / reference.norm().clamp_min(1e-20)),
        first_indices=(value != reference).nonzero()[:12].tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True, help="Boundary JSON from diagnose_prefill_stack")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--program-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-semaphore-grid", action="store_true")
    parser.add_argument("--ag-workers", type=int, choices=[1, 2, 4])
    parser.add_argument("--rs-workers", type=int, choices=[1, 2, 4])
    parser.add_argument("--ag-manager-grid", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    torch.set_num_threads(8)
    captured_report = json.loads(args.capture.read_text())
    captured = torch.load(args.capture.with_suffix(".tensors.pt"), weights_only=True)
    dtype_map = {"DataType.BFLOAT16": ttnn.bfloat16, "DataType.BFLOAT8_B": ttnn.bfloat8_b}
    fixtures = []
    for boundary in captured_report["boundaries"]:
        if not boundary["label"].endswith("/collective_local_input"):
            continue
        label = boundary["label"]
        layer_text, role, _ = label.split("/")
        key = f"{boundary['index']:03d}/{label}"
        fixtures.append(
            dict(
                layer=int(layer_text.removeprefix("layer")),
                role=role,
                parts=captured[key],
                dtype=dtype_map[boundary["dtype"]],
                dtype_name=boundary["dtype"],
            )
        )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        if args.program_cache:
            mesh.enable_program_cache()
        else:
            mesh.disable_and_clear_program_cache()
        manager_type = FullGridCCLManager if args.full_semaphore_grid else CCLManager
        managers = {layer: manager_type(mesh, 1, ttnn.Topology.Linear) for layer in captured_report["layers"]}
        for fixture in fixtures:
            fixture["device"] = ttnn.from_torch(
                torch.cat(fixture["parts"], dim=0),
                device=mesh,
                dtype=fixture["dtype"],
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        # Confirm captured quantized values survived host packing, before the
        # uninterrupted collective sequence starts.
        roundtrips = []
        for fixture in fixtures:
            parts = [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(fixture["device"])]
            changed = [int((actual != original).sum()) for actual, original in zip(parts, fixture["parts"])]
            roundtrips.append(dict(layer=fixture["layer"], role=fixture["role"], changed_per_rank=changed))
            if any(changed):
                raise AssertionError(f"Capture upload changed values: {roundtrips[-1]}")

        recorded = []
        for iteration in range(args.iterations):
            for fixture in fixtures:
                manager = managers[fixture["layer"]]
                rs_options = {} if args.rs_workers is None else {"num_workers_per_link": args.rs_workers}
                scattered = ttnn.experimental.reduce_scatter_minimal_async(
                    fixture["device"],
                    dim=3,
                    multi_device_global_semaphore=manager.get_rs_ping_pong_semaphore(),
                    barrier_semaphore=manager.get_barrier_semaphore(),
                    num_links=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                    cluster_axis=1,
                    **rs_options,
                )
                ag_options = {} if args.ag_workers is None else {"num_workers_per_link": args.ag_workers}
                if args.ag_manager_grid:
                    ag_options["sub_core_grid"] = manager.ccl_cores
                gathered = ttnn.experimental.all_gather_async(
                    scattered,
                    dim=3,
                    cluster_axis=1,
                    mesh_device=mesh,
                    topology=ttnn.Topology.Linear,
                    multi_device_global_semaphore=manager.get_ag_ping_pong_semaphore(),
                    barrier_semaphore=manager.get_barrier_semaphore(),
                    num_links=1,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    **ag_options,
                )
                recorded.append((iteration, fixture, scattered, gathered))

        results, tensors = [], {}
        for iteration, fixture, scattered, gathered in recorded:
            label = f"iteration{iteration}/layer{fixture['layer']}/{fixture['role']}"
            rs_parts = [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(scattered)]
            ag_parts = [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(gathered)]
            # AG is a copy operation: concatenating the actual RS outputs is an
            # exact oracle, independent of reduction precision or reduction order.
            ag_oracle = torch.cat(rs_parts, dim=-1)
            sum_oracle = torch.stack(fixture["parts"]).sum(dim=0)
            shard_oracles = sum_oracle.chunk(4, dim=-1)
            entry = dict(
                label=label,
                dtype=fixture["dtype_name"],
                shape=list(fixture["device"].shape),
                rs_vs_float_sum=[compare(actual, expected) for actual, expected in zip(rs_parts, shard_oracles)],
                ag_vs_actual_rs_concat=[compare(actual, ag_oracle) for actual in ag_parts],
                ag_vs_rank0=[compare(actual, ag_parts[0]) for actual in ag_parts],
            )
            results.append(entry)
            tensors[label] = dict(rs=rs_parts, ag=ag_parts, rs_concat=ag_oracle, float_sum=sum_oracle)
            print(
                "CCL_BOUNDARY",
                json.dumps(
                    dict(
                        label=label,
                        ag_copy_mismatches=[value["changed"] for value in entry["ag_vs_actual_rs_concat"]],
                        replica_mismatches=[value["changed"] for value in entry["ag_vs_rank0"]],
                        rs_max_abs=[value["max_abs"] for value in entry["rs_vs_float_sum"]],
                    )
                ),
                flush=True,
            )
        worker_grid = mesh.compute_with_storage_grid_size()
        report = dict(
            capture=str(args.capture),
            program_cache=args.program_cache,
            full_semaphore_grid=args.full_semaphore_grid,
            device_worker_grid=[worker_grid.x, worker_grid.y],
            semaphore_grid=[worker_grid.x, worker_grid.y] if args.full_semaphore_grid else [8, 8],
            ag_workers_override=args.ag_workers,
            rs_workers_override=args.rs_workers,
            ag_manager_grid=args.ag_manager_grid,
            upload_roundtrips=roundtrips,
            no_host_reads_between_collectives=True,
            persistent_buffers=False,
            ag_exact=all(value["changed"] == 0 for result in results for value in result["ag_vs_actual_rs_concat"]),
            results=results,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, args.output.with_suffix(".tensors.pt"))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("CCL_DIAGNOSTIC_COMPLETE", args.output, "AG exact:", report["ag_exact"], flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 B32 residual-topology comparison for the Qwen3.6-27B decoder.

This shape-faithful probe compares the decoder's replicated residual control
against a coherent fractured-residual boundary:

  control:   all-reduce -> local RMSNorm -> column-parallel Q projection
  candidate: reduce-scatter(dim=3) -> distributed RMSNorm
             -> gather at the column-parallel consumer -> Q projection

The candidate deliberately keeps the 1280-wide residual fractured through the
norm.  Its only hidden-width gather is at the next consumer, where a 1-D column
parallel matmul requires the complete K dimension.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc

BATCH = 32
HIDDEN = 5120
TP = 4
LOCAL_HIDDEN = HIDDEN // TP
Q_WIDTH = 6144
LOCAL_Q_WIDTH = Q_WIDTH // TP
EPSILON = 1.0e-6
MESH_SHAPE = ttnn.MeshShape(1, TP)
TOPOLOGY = ttnn.Topology.Ring
FABRIC = ttnn.FabricConfig.FABRIC_1D_RING


def _upload_replicated(value, mesh):
    return ttnn.from_torch(
        value.contiguous(),
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _upload_width_sharded(value, mesh):
    return ttnn.from_torch(
        value.contiguous(),
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _median_ms(samples):
    return float(statistics.median(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--result-json",
        default="models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/artifacts/residual_topology_b32.json",
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 3:
        raise ValueError("use at least one warmup and three timed iterations")

    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260813)

    # Each rank starts with a distinct full-width row-parallel partial.  Packing
    # them along width lets ShardTensorToMesh give every rank [1,1,32,5120].
    residual = (torch.randn(1, 1, BATCH, HIDDEN) * 0.2).bfloat16()
    noise = [(torch.randn_like(residual.float()) * 0.01).bfloat16() for _ in range(TP - 1)]
    partials = [(residual.float() / TP + value.float()).bfloat16() for value in noise]
    partials.append((residual.float() / TP - sum(value.float() for value in noise)).bfloat16())
    packed_partials = torch.cat(partials, dim=3)
    gamma = (1.0 + torch.randn(1, 1, 1, HIDDEN) * 0.02).bfloat16()
    # Q projection is the exact global/local width family: 6144 -> 1536/rank.
    q_weight = (torch.randn(1, 1, HIDDEN, Q_WIDTH) / HIDDEN**0.5).bfloat16()

    ttnn.set_fabric_config(FABRIC)
    mesh = ttnn.open_mesh_device(MESH_SHAPE, trace_region_size=0)
    try:
        tt_partials = _upload_width_sharded(packed_partials, mesh)
        tt_gamma = _upload_replicated(gamma, mesh)
        tt_gamma_local = _upload_width_sharded(gamma, mesh)
        tt_q_weight = _upload_width_sharded(q_weight, mesh)

        def control():
            reduced = ttnn.all_reduce(
                tt_partials,
                num_links=1,
                topology=TOPOLOGY,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            normalized = ttnn.rms_norm(
                reduced,
                epsilon=EPSILON,
                weight=tt_gamma,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.linear(
                normalized,
                tt_q_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def candidate():
            fractured = ttnn.reduce_scatter(
                tt_partials,
                dim=3,
                num_links=1,
                topology=TOPOLOGY,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            stats = ttnn.rms_norm_pre_all_gather(fractured, dtype=ttnn.bfloat16)
            gathered_stats = ttnn.all_gather(
                stats,
                dim=3,
                num_links=1,
                topology=TOPOLOGY,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            normalized_fractured = ttnn.rms_norm_post_all_gather(
                fractured,
                gathered_stats,
                epsilon=EPSILON,
                weight=tt_gamma_local,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # This gather belongs to the column-parallel consumer.  There is no
            # gather between reduce-scatter and the distributed norm.
            consumer_input = ttnn.all_gather(
                normalized_fractured,
                dim=3,
                num_links=1,
                topology=TOPOLOGY,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.linear(
                consumer_input,
                tt_q_weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Compile both complete families before correctness reads and timing.
        control_out = control()
        candidate_out = candidate()
        ttnn.synchronize_device(mesh)

        control_local = [ttnn.to_torch(value).float() for value in ttnn.get_device_tensors(control_out)]
        candidate_local = [ttnn.to_torch(value).float() for value in ttnn.get_device_tensors(candidate_out)]
        rank_pcc = []
        for expected, actual in zip(control_local, candidate_local):
            passed, message = comp_pcc(expected, actual, 0.999)
            if not passed:
                raise AssertionError(message)
            rank_pcc.append(str(message))

        def time_family(fn):
            for _ in range(args.warmup):
                fn()
            ttnn.synchronize_device(mesh)
            samples = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                fn()
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - started) * 1000.0)
            return samples

        control_samples = time_family(control)
        candidate_samples = time_family(candidate)
        control_ms = _median_ms(control_samples)
        candidate_ms = _median_ms(candidate_samples)
        result = {
            "batch": BATCH,
            "hidden": HIDDEN,
            "local_hidden": LOCAL_HIDDEN,
            "mesh": [1, TP],
            "hardware_strategy": "TP4 FABRIC_1D_RING / Ring",
            "consumer": "column-parallel Q projection 5120x6144, local output width 1536",
            "control_sequence": "all_reduce -> replicated RMSNorm -> column-parallel linear",
            "candidate_sequence": "reduce_scatter(dim=3) -> distributed RMSNorm pre/stats-all-gather/post -> consumer all-gather -> column-parallel linear",
            "fractured_residual_preserved_through_norm": True,
            "fallback_audit": True,
            "rank_pcc": rank_pcc,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "control_median_ms": control_ms,
            "candidate_median_ms": candidate_ms,
            "candidate_over_control": candidate_ms / control_ms,
            "control_samples_ms": control_samples,
            "candidate_samples_ms": candidate_samples,
        }
        print("MULTICHIP_RESIDUAL_TOPOLOGY", json.dumps(result, sort_keys=True))
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()

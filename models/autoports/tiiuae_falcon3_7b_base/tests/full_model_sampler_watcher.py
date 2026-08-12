# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal Watcher gate for the TP4 split-greedy candidate gather."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.modules.tt_ccl import get_tt_ccl


def collect(
    output: Path,
    *,
    iterations: int = 8,
    force_iterations: int = 2,
    hidden_rows: int = 128,
    hidden_workers: int = 2,
    hidden_iterations: int = 3,
) -> dict:
    mesh_device = None
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    try:
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
        tt_ccl = get_tt_ccl(mesh_device)
        # This is the exact two-tile FP32 packet used by split greedy: BF16
        # candidate values and global token indices, both losslessly promoted.
        host = torch.arange(4 * 64, dtype=torch.float32).reshape(1, 4, 64, 1)
        local = ttnn.from_torch(
            host,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=(1, 4)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        exact = True
        observed_shape = None
        for _ in range(iterations):
            gathered = ttnn.all_gather(
                local,
                dim=1,
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=None,
                topology=ttnn.Topology.Linear,
            )
            ttnn.synchronize_device(mesh_device)
            observed = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0]).float()
            observed_shape = list(observed.shape)
            exact = exact and torch.equal(observed, host)
            ttnn.deallocate(gathered, True)
        ttnn.deallocate(local, True)

        hidden_host = torch.arange(hidden_rows * 3072, dtype=torch.bfloat16).reshape(1, 1, hidden_rows, 3072)
        hidden_local = ttnn.from_torch(
            hidden_host,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(1, 4)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        predecessor_ring_gather_exact = True
        for _ in range(hidden_iterations):
            hidden_gathered = ttnn.experimental.all_gather_async(
                hidden_local,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=1,
                topology=ttnn.Topology.Ring,
                chunks_per_sync=10,
                num_workers_per_link=hidden_workers,
                num_buffers_per_channel=2,
            )
            ttnn.synchronize_device(mesh_device)
            hidden_observed = ttnn.to_torch(ttnn.get_device_tensors(hidden_gathered)[0])
            predecessor_ring_gather_exact = predecessor_ring_gather_exact and torch.equal(hidden_observed, hidden_host)
            ttnn.deallocate(hidden_gathered, True)
        ttnn.deallocate(hidden_local, True)

        vocab_size = 131072
        logits_host = torch.full((1, 1, 32, vocab_size), -4.0, dtype=torch.bfloat16)
        expected_tokens = (torch.arange(32, dtype=torch.int64) * 4099 + 37) % vocab_size
        logits_host[0, 0, torch.arange(32), expected_tokens] = 8.0
        logits = ttnn.from_torch(
            logits_host,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(1, 4)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sampler = Sampling1D.from_config(
            Sampling1DConfig(
                vocab_size=vocab_size,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                max_batch_size=32,
                max_top_k=32,
                num_gather_links=2,
                allow_force_argmax=True,
                num_argmax_gather_links=2,
                ag_topology=ttnn.Topology.Ring,
                pad_to_power_of_2=True,
            )
        )
        sampled_exact = True
        sampled_tokens = None
        for _ in range(iterations):
            sampled = sampler.greedy_decode_forward(logits)
            ttnn.synchronize_device(mesh_device)
            sampled_tokens = ttnn.to_torch(ttnn.get_device_tensors(sampled)[0]).reshape(-1)[:32].to(torch.int64)
            sampled_exact = sampled_exact and torch.equal(sampled_tokens, expected_tokens)
            ttnn.deallocate(sampled, True)

        # Exercise the exact rejected comparison path used by the full-model
        # evidence gate.  On TP4 this gathers the complete BFP8 vocabulary
        # with a two-link Linear async collective before the device argmax.
        force_argmax_exact = True
        force_argmax_tokens = None
        for _ in range(force_iterations):
            force_sampled, _ = sampler.decode_forward(logits)
            ttnn.synchronize_device(mesh_device)
            force_argmax_tokens = (
                ttnn.to_torch(ttnn.get_device_tensors(force_sampled)[0]).reshape(-1)[:32].to(torch.int64)
            )
            force_argmax_exact = force_argmax_exact and torch.equal(force_argmax_tokens, expected_tokens)
            ttnn.deallocate(force_sampled, True)
        ttnn.deallocate(logits, True)
        result = {
            "mesh": "1x4 Blackhole FABRIC_1D_RING",
            "packet_shape_per_rank": [1, 1, 64, 1],
            "dtype": "FP32 (BF16 candidate values plus exact global indices)",
            "num_links": 2,
            "packet_gather_topology": "Linear (canonical Sampling1D 1D fallback)",
            "predecessor_hidden_gather": (
                f"{hidden_iterations} two-link Rings, "
                f"{hidden_workers} worker{'s' if hidden_workers != 1 else ''}/link, "
                f"global shape [1, 1, {hidden_rows}, 3072]"
            ),
            "predecessor_hidden_gather_exact": predecessor_ring_gather_exact,
            "iterations": iterations,
            "observed_shape": observed_shape,
            "packet_gather_exact": exact,
            "split_greedy_expected_tokens": expected_tokens.tolist(),
            "split_greedy_observed_tokens": sampled_tokens.tolist(),
            "split_greedy_exact": sampled_exact,
            "force_argmax_gather": "two-link Linear, full BFP8 vocabulary, comparison-only",
            "force_argmax_iterations": force_iterations,
            "force_argmax_observed_tokens": force_argmax_tokens.tolist(),
            "force_argmax_exact": force_argmax_exact,
            "passed": exact and predecessor_ring_gather_exact and sampled_exact and force_argmax_exact,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    finally:
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--force-iterations", type=int, default=2)
    parser.add_argument("--hidden-rows", type=int, default=128)
    parser.add_argument("--hidden-workers", type=int, default=2)
    parser.add_argument("--hidden-iterations", type=int, default=3)
    args = parser.parse_args()
    result = collect(
        args.output,
        iterations=args.iterations,
        force_iterations=args.force_iterations,
        hidden_rows=args.hidden_rows,
        hidden_workers=args.hidden_workers,
        hidden_iterations=args.hidden_iterations,
    )
    if not result["passed"]:
        raise SystemExit("split-greedy gather Watcher gate failed")


if __name__ == "__main__":
    main()

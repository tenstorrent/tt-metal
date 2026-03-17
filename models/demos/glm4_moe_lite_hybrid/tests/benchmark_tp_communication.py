# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""T3K microbenchmark: all_reduce vs reduce_scatter_minimal_async for TP linears.

Measures the exact claim: does reduce_scatter_minimal_async give a speedup over
all_reduce for tensor-parallel matmul communication on a multi-device mesh?

Runs three tests:
  Test 1: Isolated communication ops (no matmul, just reduce)
  Test 2: TP linear (matmul + reduce) — simulates one attention projection
  Test 3: Full layer simulation (multiple TP linears back-to-back)

Requirements:
  - T3K (8-chip) or N300 (2-chip) system
  - Set MESH_COLS to the number of devices (default: auto-detect)

Usage:
  # On T3K:
  python3 models/demos/glm4_moe_lite_hybrid/tests/benchmark_tp_communication.py --mesh-cols 8

  # On N300:
  python3 models/demos/glm4_moe_lite_hybrid/tests/benchmark_tp_communication.py --mesh-cols 2
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

import ttnn


def _open_mesh(mesh_cols: int):
    if mesh_cols > 1:
        is_galaxy = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
        fabric = ttnn.FabricConfig.FABRIC_1D_RING if is_galaxy else ttnn.FabricConfig.FABRIC_1D
        ttnn.set_fabric_config(fabric, ttnn.FabricReliabilityMode.STRICT_INIT)
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, mesh_cols),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
    )


def _warm_and_measure(fn, device, warmup=5, measure=20):
    for _ in range(warmup):
        fn()
        ttnn.synchronize_device(device)
    times = []
    for _ in range(measure):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def _print_timing(label, times_ms):
    mean = statistics.mean(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    std = statistics.stdev(times_ms) if len(times_ms) > 1 else 0
    print(f"  {label:<55s}  mean={mean:8.3f} ms  min={mn:8.3f}  max={mx:8.3f}  std={std:6.3f}")
    return mean


def _print_comparison(label_a, mean_a, label_b, mean_b):
    if mean_a > 0 and mean_b > 0:
        speedup = mean_a / mean_b
        delta_pct = (mean_a - mean_b) / mean_a * 100
        print(f"  --> {label_b} is {speedup:.2f}x vs {label_a} ({delta_pct:+.1f}%)")
    print()


# ============================================================================
# Test 1: Isolated communication ops
# ============================================================================
def test_isolated_communication(device, mesh_cols):
    print(f"\n{'='*72}")
    print(f"  Test 1: Isolated Communication Ops (no matmul)")
    print(f"  Tensor shape: [1, 1, 32, 2048] — simulates one decode-step activation")
    print(f"{'='*72}\n")

    host = torch.randn(1, 1, 32, 2048, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        host,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    # all_reduce
    def run_all_reduce():
        return ttnn.all_reduce(
            x,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    times_ar = _warm_and_measure(run_all_reduce, device)
    mean_ar = _print_timing("all_reduce [1,1,32,2048]", times_ar)

    # reduce_scatter_minimal_async
    def run_reduce_scatter():
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            scatter_dim=3,
            topology=ttnn.Topology.Ring,
            num_links=1,
            num_workers_per_link=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    times_rs = _warm_and_measure(run_reduce_scatter, device)
    mean_rs = _print_timing("reduce_scatter_minimal_async [1,1,32,2048]", times_rs)

    _print_comparison("all_reduce", mean_ar, "reduce_scatter", mean_rs)

    # Larger tensor
    host_big = torch.randn(1, 1, 32, 10240, dtype=torch.bfloat16)
    x_big = ttnn.from_torch(
        host_big,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    def run_all_reduce_big():
        return ttnn.all_reduce(
            x_big,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def run_reduce_scatter_big():
        return ttnn.experimental.reduce_scatter_minimal_async(
            x_big,
            scatter_dim=3,
            topology=ttnn.Topology.Ring,
            num_links=1,
            num_workers_per_link=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    times_ar_big = _warm_and_measure(run_all_reduce_big, device)
    mean_ar_big = _print_timing("all_reduce [1,1,32,10240]", times_ar_big)

    times_rs_big = _warm_and_measure(run_reduce_scatter_big, device)
    mean_rs_big = _print_timing("reduce_scatter_minimal_async [1,1,32,10240]", times_rs_big)

    _print_comparison("all_reduce", mean_ar_big, "reduce_scatter", mean_rs_big)

    ttnn.deallocate(x)
    ttnn.deallocate(x_big)

    return {
        "isolated_ar_2048_ms": mean_ar,
        "isolated_rs_2048_ms": mean_rs,
        "isolated_ar_10240_ms": mean_ar_big,
        "isolated_rs_10240_ms": mean_rs_big,
    }


# ============================================================================
# Test 2: TP linear (matmul + reduce)
# ============================================================================
def test_tp_linear(device, mesh_cols):
    print(f"\n{'='*72}")
    print(f"  Test 2: TP Linear (matmul + reduce)")
    print(f"  Simulates one attention projection: [1,1,1,2048] x [1,1,256,2048]")
    print(f"  (input partitioned across {mesh_cols} devices, weight local)")
    print(f"{'='*72}\n")

    hidden = 2048
    local_input = hidden // mesh_cols

    x_host = torch.randn(1, 1, 1, hidden, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        x_host,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    w_host = torch.randn(1, 1, local_input, hidden, dtype=torch.bfloat16)
    w = ttnn.from_torch(
        w_host,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    # Path A: mesh_partition + matmul + all_reduce (agentic pattern)
    def run_tp_all_reduce():
        x_tp = ttnn.mesh_partition(x, dim=3, cluster_axis=1)
        out = ttnn.linear(x_tp, w)
        out_reduced = ttnn.all_reduce(
            out,
            num_links=1,
            topology=ttnn.Topology.Linear,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_tp, force=False)
        ttnn.deallocate(out, force=False)
        return out_reduced

    times_ar = _warm_and_measure(run_tp_all_reduce, device)
    mean_ar = _print_timing("TP linear + all_reduce", times_ar)

    # Path B: mesh_partition + matmul + reduce_scatter (hybrid pattern)
    def run_tp_reduce_scatter():
        x_tp = ttnn.mesh_partition(x, dim=3, cluster_axis=1)
        out = ttnn.linear(x_tp, w)
        out_reduced = ttnn.experimental.reduce_scatter_minimal_async(
            out,
            scatter_dim=3,
            topology=ttnn.Topology.Ring,
            num_links=1,
            num_workers_per_link=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_tp, force=False)
        ttnn.deallocate(out, force=False)
        return out_reduced

    times_rs = _warm_and_measure(run_tp_reduce_scatter, device)
    mean_rs = _print_timing("TP linear + reduce_scatter", times_rs)

    _print_comparison("all_reduce", mean_ar, "reduce_scatter", mean_rs)

    ttnn.deallocate(x)
    ttnn.deallocate(w)

    return {
        "tp_linear_ar_ms": mean_ar,
        "tp_linear_rs_ms": mean_rs,
    }


# ============================================================================
# Test 3: Simulated full layer (multiple TP linears)
# ============================================================================
def test_simulated_layer(device, mesh_cols):
    print(f"\n{'='*72}")
    print(f"  Test 3: Simulated Decoder Layer ({mesh_cols} TP linears back-to-back)")
    print(f"  7 TP projections: q_a, q_b, kv_a, kv_b2, w_o, mlp_gate, mlp_down")
    print(f"{'='*72}\n")

    hidden = 2048
    local_input = hidden // mesh_cols
    projections = [
        ("q_a", hidden, 768),
        ("q_b", 768, hidden),
        ("kv_a", hidden, 576),
        ("kv_b2", 512, 128),
        ("w_o", hidden, hidden),
        ("mlp_gate", hidden, 10240 // mesh_cols),
        ("mlp_down", 10240 // mesh_cols, hidden),
    ]

    x_host = torch.randn(1, 1, 1, hidden, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        x_host,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    weights = []
    for name, k_dim, n_dim in projections:
        w_local_k = max(32, k_dim // mesh_cols)
        w_local_n = max(32, n_dim)
        w_host = torch.randn(1, 1, w_local_k, w_local_n, dtype=torch.bfloat16)
        w = ttnn.from_torch(
            w_host,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        weights.append(w)

    # Path A: all_reduce after each projection
    def run_layer_all_reduce():
        for w in weights:
            x_tp = ttnn.mesh_partition(x, dim=3, cluster_axis=1)
            out = ttnn.linear(x_tp, w)
            ttnn.all_reduce(
                out,
                num_links=1,
                topology=ttnn.Topology.Linear,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_tp, force=False)
            ttnn.deallocate(out, force=False)

    times_ar = _warm_and_measure(run_layer_all_reduce, device)
    mean_ar = _print_timing(f"7 TP projections + all_reduce", times_ar)

    # Path B: reduce_scatter after each projection
    def run_layer_reduce_scatter():
        for w in weights:
            x_tp = ttnn.mesh_partition(x, dim=3, cluster_axis=1)
            out = ttnn.linear(x_tp, w)
            ttnn.experimental.reduce_scatter_minimal_async(
                out,
                scatter_dim=3,
                topology=ttnn.Topology.Ring,
                num_links=1,
                num_workers_per_link=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_tp, force=False)
            ttnn.deallocate(out, force=False)

    times_rs = _warm_and_measure(run_layer_reduce_scatter, device)
    mean_rs = _print_timing(f"7 TP projections + reduce_scatter", times_rs)

    _print_comparison("all_reduce", mean_ar, "reduce_scatter", mean_rs)

    per_proj_ar = mean_ar / 7
    per_proj_rs = mean_rs / 7
    full_model_ar = per_proj_ar * 7 * 47
    full_model_rs = per_proj_rs * 7 * 47
    print(f"  Projected 47-layer impact:")
    print(f"    all_reduce:    {per_proj_ar:.3f} ms/proj x 7 projs x 47 layers = {full_model_ar:.1f} ms/decode")
    print(f"    reduce_scatter: {per_proj_rs:.3f} ms/proj x 7 projs x 47 layers = {full_model_rs:.1f} ms/decode")
    print(
        f"    Savings: {full_model_ar - full_model_rs:.1f} ms/decode ({(full_model_ar - full_model_rs) / full_model_ar * 100:.1f}%)"
    )
    print()

    for w in weights:
        ttnn.deallocate(w, force=False)
    ttnn.deallocate(x)

    return {
        "layer_ar_ms": mean_ar,
        "layer_rs_ms": mean_rs,
        "projected_47layer_ar_ms": full_model_ar,
        "projected_47layer_rs_ms": full_model_rs,
    }


# ============================================================================
# Summary
# ============================================================================
def print_summary(results, mesh_cols):
    print(f"\n{'='*72}")
    print(f"  SUMMARY: TP Communication Benchmark ({mesh_cols}-device mesh)")
    print(f"{'='*72}\n")

    print(f"  {'Test':<50s} {'all_reduce':>12s} {'reduce_scatter':>15s} {'Speedup':>10s}")
    print(f"  {'-'*50} {'-'*12} {'-'*15} {'-'*10}")

    tests = [
        ("Isolated comm [2048]", "isolated_ar_2048_ms", "isolated_rs_2048_ms"),
        ("Isolated comm [10240]", "isolated_ar_10240_ms", "isolated_rs_10240_ms"),
        ("TP linear (1 proj)", "tp_linear_ar_ms", "tp_linear_rs_ms"),
        ("Simulated layer (7 projs)", "layer_ar_ms", "layer_rs_ms"),
        ("Projected 47 layers", "projected_47layer_ar_ms", "projected_47layer_rs_ms"),
    ]

    for label, ar_key, rs_key in tests:
        ar = results.get(ar_key, 0)
        rs = results.get(rs_key, 0)
        speedup = ar / rs if rs > 0 else 0
        unit = "ms"
        print(f"  {label:<50s} {ar:>10.3f} {unit} {rs:>13.3f} {unit} {speedup:>9.2f}x")

    print(f"\n  Verdict: ", end="")
    layer_ar = results.get("layer_ar_ms", 1)
    layer_rs = results.get("layer_rs_ms", 1)
    speedup = layer_ar / layer_rs if layer_rs > 0 else 1
    if speedup > 1.03:
        print(f"reduce_scatter is {speedup:.2f}x faster — claim VERIFIED")
    elif speedup > 0.97:
        print(f"No significant difference ({speedup:.2f}x) — claim NOT VERIFIED at this config")
    else:
        print(f"all_reduce is faster ({speedup:.2f}x) — claim REFUTED at this config")
    print()


def main():
    ap = argparse.ArgumentParser(description="Benchmark TP communication: all_reduce vs reduce_scatter")
    ap.add_argument("--mesh-cols", type=int, default=0, help="Number of devices (0=auto-detect)")
    args = ap.parse_args()

    mesh_cols = args.mesh_cols
    if mesh_cols == 0:
        try:
            n = ttnn.GetNumAvailableDevices()
        except AttributeError:
            n = 1
        mesh_cols = max(1, n)
        print(f"Auto-detected {mesh_cols} device(s)")

    if mesh_cols < 2:
        print("ERROR: This benchmark requires at least 2 devices (N300 or T3K).")
        print("On a single N150, TP communication does not exist — both paths are no-ops.")
        print("Run with: --mesh-cols 2 (N300) or --mesh-cols 8 (T3K)")
        return 1

    print(f"{'='*72}")
    print(f"  TP Communication Benchmark: all_reduce vs reduce_scatter_minimal_async")
    print(f"  Mesh: (1, {mesh_cols}) — {mesh_cols} devices")
    print(f"{'='*72}")

    device = _open_mesh(mesh_cols)
    grid = device.compute_with_storage_grid_size()
    print(f"  Compute grid per device: {grid.x}x{grid.y}")

    results = {}
    try:
        r = test_isolated_communication(device, mesh_cols)
        results.update(r)

        r = test_tp_linear(device, mesh_cols)
        results.update(r)

        r = test_simulated_layer(device, mesh_cols)
        results.update(r)

        print_summary(results, mesh_cols)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
    finally:
        ttnn.close_mesh_device(device)
        print("Device closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

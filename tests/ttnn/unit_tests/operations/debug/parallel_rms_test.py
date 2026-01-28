# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance comparison: ttnn.rms_norm (direct) vs launch_composite() with single/multiple branches.

This test compares device kernel durations for:
- Q-only using ttnn.rms_norm (direct)
- Q-only using launch_composite([q])
- KV-only using ttnn.rms_norm (direct)
- KV-only using launch_composite([kv])
- Q+KV using launch_composite([q,kv])

Run with device profiling (Tracy):
    python -m tracy -r -m pytest tests/ttnn/unit_tests/operations/debug/parallel_rms_test.py::test_parallel_rms_performance -v -s

The two RMS norm operations:
    - q_norm: L1 width sharded on cores (0,0)-(3,3), shard [32,96], tensor [32,1536]
    - kv_norm: L1 width sharded on cores (5,0)-(6,7), shard [32,32], tensor [32,512]
"""

import pytest
import torch
import ttnn
import pandas as pd
from pathlib import Path

from tracy import signpost


def torch_rms_norm(x, gamma, eps=1e-5):
    """Reference RMS norm implementation in PyTorch."""
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    x_normed = x / rms
    if gamma is not None:
        x_normed = x_normed * gamma
    return x_normed


def create_q_norm_tensors(device):
    """
    Create q_norm tensors:
    - L1 width sharded on cores (0,0) to (3,3) = 16 cores
    - Shard shape: [32, 96]
    - Total tensor shape: [1, 1, 32, 1536] (16 cores × 96 = 1536)
    - Weights are DRAM interleaved
    """
    torch.manual_seed(42)

    # Core range: (0,0) to (3,3) = 4x4 = 16 cores
    q_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    num_cores = 16
    shard_width = 96
    total_width = num_cores * shard_width  # 1536

    # Tensor shapes
    input_shape = (1, 1, 32, total_width)
    weight_shape = (1, 1, 1, total_width)

    # Create torch tensors
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    # Create sharded memory config for input (L1 width sharded)
    shard_spec = ttnn.ShardSpec(q_cores, [32, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Move input to device with sharded memory config
    input_tensor = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
    )

    # Weights are DRAM interleaved
    weight_tensor = ttnn.from_torch(
        torch_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Program config for sharded RMS norm
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 4),  # 4x4 grid
        subblock_w=shard_width // 32,  # tiles per shard width = 96/32 = 3
        block_h=1,  # 32/32 = 1 tile high
        block_w=shard_width // 32,  # 3 tiles wide per core
        inplace=False,
    )

    return {
        "input": input_tensor,
        "weight": weight_tensor,
        "cores": q_cores,
        "memory_config": sharded_mem_config,
        "program_config": program_config,
        "torch_input": torch_input,
        "torch_weight": torch_weight,
        "name": "q_norm",
    }


def create_kv_norm_tensors(device):
    """
    Create kv_norm tensors:
    - L1 width sharded on cores (5,0) to (6,7) = 2x8 = 16 cores
    - Shard shape: [32, 32]
    - Total tensor shape: [1, 1, 32, 512] (16 cores × 32 = 512)
    - Weights are DRAM interleaved
    """
    torch.manual_seed(123)

    # Core range: (5,0) to (6,7) = 2 cols × 8 rows = 16 cores
    kv_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 7))])
    num_cores = 16
    shard_width = 32
    total_width = num_cores * shard_width  # 512

    # Tensor shapes
    input_shape = (1, 1, 32, total_width)
    weight_shape = (1, 1, 1, total_width)

    # Create torch tensors
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    # Create sharded memory config for input (L1 width sharded)
    shard_spec = ttnn.ShardSpec(kv_cores, [32, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Move input to device with sharded memory config
    input_tensor = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
    )

    # Weights are DRAM interleaved
    weight_tensor = ttnn.from_torch(
        torch_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Program config for sharded RMS norm
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 8),  # 2 cols × 8 rows
        subblock_w=shard_width // 32,  # tiles per shard width = 32/32 = 1
        block_h=1,  # 32/32 = 1 tile high
        block_w=shard_width // 32,  # 1 tile wide per core
        inplace=False,
    )

    return {
        "input": input_tensor,
        "weight": weight_tensor,
        "cores": kv_cores,
        "memory_config": sharded_mem_config,
        "program_config": program_config,
        "torch_input": torch_input,
        "torch_weight": torch_weight,
        "name": "kv_norm",
    }


def run_single_q_direct(device, q_tensors, num_iterations=1, use_signpost=False):
    """Run RMS norm on Q cores only using direct ttnn.rms_norm()."""
    if use_signpost:
        signpost("single_q_direct_start")

    for _ in range(num_iterations):
        output = ttnn.rms_norm(
            q_tensors["input"],
            epsilon=1e-5,
            weight=q_tensors["weight"],
            memory_config=q_tensors["memory_config"],
            program_config=q_tensors["program_config"],
        )

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("single_q_direct_stop")

    return output


def run_single_q_via_composite(device, q_tensors, num_iterations=1, use_signpost=False):
    """Run RMS norm on Q cores only using launch_composite() with single-element list."""
    if use_signpost:
        signpost("single_q_composite_start")

    for _ in range(num_iterations):
        q_branch = ttnn.experimental.programs.rms_norm(
            q_tensors["input"],
            epsilon=1e-5,
            weight=q_tensors["weight"],
            memory_config=q_tensors["memory_config"],
            core_range_set=q_tensors["cores"],
        )
        output = ttnn.experimental.launch_composite([q_branch])[0]

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("single_q_composite_stop")

    return output


def run_single_kv_direct(device, kv_tensors, num_iterations=1, use_signpost=False):
    """Run RMS norm on KV cores only using direct ttnn.rms_norm()."""
    if use_signpost:
        signpost("single_kv_direct_start")

    for _ in range(num_iterations):
        output = ttnn.rms_norm(
            kv_tensors["input"],
            epsilon=1e-5,
            weight=kv_tensors["weight"],
            memory_config=kv_tensors["memory_config"],
            program_config=kv_tensors["program_config"],
        )

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("single_kv_direct_stop")

    return output


def run_single_kv_via_composite(device, kv_tensors, num_iterations=1, use_signpost=False):
    """Run RMS norm on KV cores only using launch_composite() with single-element list."""
    if use_signpost:
        signpost("single_kv_composite_start")

    for _ in range(num_iterations):
        kv_branch = ttnn.experimental.programs.rms_norm(
            kv_tensors["input"],
            epsilon=1e-5,
            weight=kv_tensors["weight"],
            memory_config=kv_tensors["memory_config"],
            core_range_set=kv_tensors["cores"],
        )
        output = ttnn.experimental.launch_composite([kv_branch])[0]

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("single_kv_composite_stop")

    return output


def run_composite_qkv(device, q_tensors, kv_tensors, num_iterations=1, use_signpost=False):
    """Run q_norm and kv_norm in parallel using launch_composite()."""
    # Create branches ONCE outside the loop (for optimal caching)
    q_branch = ttnn.experimental.programs.rms_norm(
        q_tensors["input"],
        epsilon=1e-5,
        weight=q_tensors["weight"],
        memory_config=q_tensors["memory_config"],
        core_range_set=q_tensors["cores"],
    )

    kv_branch = ttnn.experimental.programs.rms_norm(
        kv_tensors["input"],
        epsilon=1e-5,
        weight=kv_tensors["weight"],
        memory_config=kv_tensors["memory_config"],
        core_range_set=kv_tensors["cores"],
    )

    if use_signpost:
        signpost("composite_qkv_start")

    # Execute in parallel - launch_composite handles caching automatically
    for _ in range(num_iterations):
        q_output, kv_output = ttnn.experimental.launch_composite([q_branch, kv_branch])
        outputs = [q_output, kv_output]

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("composite_qkv_stop")

    return outputs


def verify_outputs(q_tensors, kv_tensors, q_output, kv_output):
    """Verify outputs against PyTorch reference."""
    q_expected = torch_rms_norm(q_tensors["torch_input"], q_tensors["torch_weight"])
    kv_expected = torch_rms_norm(kv_tensors["torch_input"], kv_tensors["torch_weight"])

    q_actual = ttnn.to_torch(ttnn.from_device(q_output))
    kv_actual = ttnn.to_torch(ttnn.from_device(kv_output))

    def calc_pcc(expected, actual):
        expected_flat = expected.flatten().float()
        actual_flat = actual.flatten().float()
        return torch.corrcoef(torch.stack([expected_flat, actual_flat]))[0, 1].item()

    q_pcc = calc_pcc(q_expected, q_actual)
    kv_pcc = calc_pcc(kv_expected, kv_actual)

    q_close = torch.allclose(q_expected, q_actual, rtol=1e-2, atol=1e-2)
    kv_close = torch.allclose(kv_expected, kv_actual, rtol=1e-2, atol=1e-2)

    q_max_err = torch.max(torch.abs(q_expected - q_actual)).item()
    kv_max_err = torch.max(torch.abs(kv_expected - kv_actual)).item()

    return q_close, kv_close, q_pcc, kv_pcc, q_max_err, kv_max_err


@pytest.mark.parametrize("num_iterations", [50])
def test_parallel_rms_performance(device, num_iterations):
    """
    Compare ttnn.rms_norm (direct) vs launch_composite() with single/multiple branches.
    """
    print(f"\n{'='*80}")
    print("LAUNCH_COMPOSITE() COMPARISON TEST")
    print(f"{'='*80}")

    # Clear program cache first
    device.clear_program_cache()
    device.enable_program_cache()

    # Create tensors
    q_tensors = create_q_norm_tensors(device)
    kv_tensors = create_kv_norm_tensors(device)

    print(f"\nConfiguration:")
    print(f"  q_norm:  cores (0,0)-(3,3), shard [32,96], tensor [32,1536]")
    print(f"  kv_norm: cores (5,0)-(6,7), shard [32,32], tensor [32,512]")
    print(f"  Iterations per measurement: {num_iterations}")

    # Warmup
    print("\nWarming up...")
    warmup_iters = 10
    for _ in range(warmup_iters):
        _ = run_single_q_direct(device, q_tensors)
        _ = run_single_q_via_composite(device, q_tensors)
        _ = run_single_kv_direct(device, kv_tensors)
        _ = run_single_kv_via_composite(device, kv_tensors)
        _ = run_composite_qkv(device, q_tensors, kv_tensors)

    # Verify correctness
    print(f"\n{'='*60}")
    print("CORRECTNESS VERIFICATION")
    print(f"{'='*60}")

    device.clear_program_cache()
    device.enable_program_cache()

    q_tensors_fresh = create_q_norm_tensors(device)
    kv_tensors_fresh = create_kv_norm_tensors(device)

    composite_outputs = run_composite_qkv(device, q_tensors_fresh, kv_tensors_fresh, num_iterations=1)
    q_close, kv_close, q_pcc, kv_pcc, q_max_err, kv_max_err = verify_outputs(
        q_tensors_fresh, kv_tensors_fresh, composite_outputs[0], composite_outputs[1]
    )

    print(f"  q_norm:")
    print(f"    allclose: {q_close} (rtol=1e-2, atol=1e-2)")
    print(f"    PCC:      {q_pcc:.6f}")
    print(f"    Max error: {q_max_err:.6e}")
    print(f"  kv_norm:")
    print(f"    allclose: {kv_close} (rtol=1e-2, atol=1e-2)")
    print(f"    PCC:      {kv_pcc:.6f}")
    print(f"    Max error: {kv_max_err:.6e}")

    assert q_close, f"q_norm failed allclose check (max error: {q_max_err:.6e})"
    assert kv_close, f"kv_norm failed allclose check (max error: {kv_max_err:.6e})"
    print("  ✓ All outputs verified correct with torch.allclose!")

    # Profile device kernel durations
    print(f"\n{'='*60}")
    print("PROFILING DEVICE KERNEL DURATIONS")
    print(f"{'='*60}")

    print("\nProfiling Q-ONLY case via direct ttnn.rms_norm()...")
    _ = run_single_q_direct(device, q_tensors, num_iterations, use_signpost=True)

    print("Profiling Q-ONLY case via launch_composite()...")
    _ = run_single_q_via_composite(device, q_tensors, num_iterations, use_signpost=True)

    print("Profiling KV-ONLY case via direct ttnn.rms_norm()...")
    _ = run_single_kv_direct(device, kv_tensors, num_iterations, use_signpost=True)

    print("Profiling KV-ONLY case via launch_composite()...")
    _ = run_single_kv_via_composite(device, kv_tensors, num_iterations, use_signpost=True)

    print("Profiling COMPOSITE Q+KV case via launch_composite()...")
    _ = run_composite_qkv(device, q_tensors, kv_tensors, num_iterations, use_signpost=True)

    print(f"\n{'='*60}")
    print("PROFILING COMPLETE")
    print(f"{'='*60}\n")

    # Extract device kernel durations from Tracy CSV
    def find_latest_tracy_csv():
        reports_dir = Path("generated/profiler/reports")
        if not reports_dir.exists():
            return None
        csv_files = list(reports_dir.glob("*/ops_perf_results_*.csv"))
        if not csv_files:
            return None
        return str(max(csv_files, key=lambda p: p.stat().st_mtime))

    def extract_ops_between_signposts(csv_path, op_name):
        if not Path(csv_path).exists():
            return {}
        df = pd.read_csv(csv_path)
        results = {}
        signpost_rows = df[df["OP TYPE"] == "signpost"]
        signpost_indices = list(signpost_rows.index)
        for i in range(0, len(signpost_indices), 2):
            if i + 1 < len(signpost_indices):
                start_idx = signpost_indices[i]
                stop_idx = signpost_indices[i + 1]
                start_op_code = df.loc[start_idx]["OP CODE"]
                if start_op_code.endswith("_start"):
                    region_name = start_op_code[:-6]
                    region_ops = df.loc[start_idx + 1 : stop_idx - 1]
                    matching_ops = region_ops[
                        (region_ops["OP CODE"] == op_name) & (region_ops["DEVICE KERNEL DURATION [ns]"].notna())
                    ]
                    if len(matching_ops) > 0:
                        durations = matching_ops["DEVICE KERNEL DURATION [ns]"].tolist()
                        if region_name in results:
                            results[region_name].extend([float(d) for d in durations])
                        else:
                            results[region_name] = [float(d) for d in durations]
        return results

    csv_path = find_latest_tracy_csv()
    if csv_path:
        print(f"Extracting device kernel durations from: {csv_path}")

        # Extract both GenericOpDeviceOperation (for launch_composite) and LayerNormDeviceOperation (for direct)
        durations_generic = extract_ops_between_signposts(csv_path, "GenericOpDeviceOperation")
        durations_layernorm = extract_ops_between_signposts(csv_path, "LayerNormDeviceOperation")

        # Combine both operation types
        def combine_durations(region_name):
            generic = durations_generic.get(region_name, [])
            layernorm = durations_layernorm.get(region_name, [])
            return generic + layernorm

        q_direct_durations = combine_durations("single_q_direct")
        q_composite_durations = durations_generic.get("single_q_composite", [])
        kv_direct_durations = combine_durations("single_kv_direct")
        kv_composite_durations = durations_generic.get("single_kv_composite", [])
        composite_qkv_durations = durations_generic.get("composite_qkv", [])

        if (
            q_direct_durations
            and q_composite_durations
            and kv_direct_durations
            and kv_composite_durations
            and composite_qkv_durations
        ):
            q_direct_avg = sum(q_direct_durations) / len(q_direct_durations)
            q_composite_avg = sum(q_composite_durations) / len(q_composite_durations)
            kv_direct_avg = sum(kv_direct_durations) / len(kv_direct_durations)
            kv_composite_avg = sum(kv_composite_durations) / len(kv_composite_durations)
            composite_qkv_avg = sum(composite_qkv_durations) / len(composite_qkv_durations)

            max_single_composite = max(q_composite_avg, kv_composite_avg)
            max_single_direct = max(q_direct_avg, kv_direct_avg)
            efficiency_vs_composite = (max_single_composite / composite_qkv_avg) * 100 if composite_qkv_avg > 0 else 0
            efficiency_vs_direct = (max_single_direct / composite_qkv_avg) * 100 if composite_qkv_avg > 0 else 0

            print(f"\n{'='*80}")
            print("DEVICE KERNEL DURATION COMPARISON TABLE")
            print(f"{'='*80}")
            print(f"{'Operation':<30} {'Method':<30} {'Duration (µs)':<18} {'Runs':<8}")
            print(f"{'-'*86}")
            print(
                f"{'Q-only':<30} {'ttnn.rms_norm (direct)':<30} {q_direct_avg/1e3:>15.2f} {len(q_direct_durations):>8}"
            )
            print(
                f"{'Q-only':<30} {'launch_composite([q])':<30} {q_composite_avg/1e3:>15.2f} {len(q_composite_durations):>8}"
            )
            print(
                f"{'KV-only':<30} {'ttnn.rms_norm (direct)':<30} {kv_direct_avg/1e3:>15.2f} {len(kv_direct_durations):>8}"
            )
            print(
                f"{'KV-only':<30} {'launch_composite([kv])':<30} {kv_composite_avg/1e3:>15.2f} {len(kv_composite_durations):>8}"
            )
            print(
                f"{'Q+KV composite':<30} {'launch_composite([q,kv])':<30} {composite_qkv_avg/1e3:>15.2f} {len(composite_qkv_durations):>8}"
            )
            print(f"{'-'*86}")
            print(f"\nParallelism Efficiency:")
            print(
                f"  vs max single (composite): {efficiency_vs_composite:.1f}% (max: {max_single_composite/1e3:.2f} µs)"
            )
            print(f"  vs max single (direct):     {efficiency_vs_direct:.1f}% (max: {max_single_direct/1e3:.2f} µs)")
            print(f"\nOverhead Analysis:")
            q_overhead = ((q_composite_avg / q_direct_avg) - 1.0) * 100 if q_direct_avg > 0 else 0
            kv_overhead = ((kv_composite_avg / kv_direct_avg) - 1.0) * 100 if kv_direct_avg > 0 else 0
            print(f"  Q-only: launch_composite overhead vs direct: {q_overhead:+.1f}%")
            print(f"  KV-only: launch_composite overhead vs direct: {kv_overhead:+.1f}%")
            print(f"{'='*80}\n")
        else:
            print(f"Could not extract all durations. Found:")
            print(f"  Q-direct: {len(q_direct_durations)} entries")
            print(f"  Q-composite: {len(q_composite_durations)} entries")
            print(f"  KV-direct: {len(kv_direct_durations)} entries")
            print(f"  KV-composite: {len(kv_composite_durations)} entries")
            print(f"  Composite Q+KV: {len(composite_qkv_durations)} entries")
            print(f"\nNote: CSV may not be generated yet. Check manually: {csv_path}\n")


@pytest.mark.parametrize("num_iterations", [1])
def test_parallel_rms_correctness(device, num_iterations):
    """Quick correctness test for parallel RMS norm."""
    device.clear_program_cache()
    device.enable_program_cache()

    q_tensors = create_q_norm_tensors(device)
    kv_tensors = create_kv_norm_tensors(device)

    # Run parallel
    composite_outputs = run_composite_qkv(device, q_tensors, kv_tensors, num_iterations=1)
    q_close, kv_close, q_pcc, kv_pcc, q_max_err, kv_max_err = verify_outputs(
        q_tensors, kv_tensors, composite_outputs[0], composite_outputs[1]
    )

    print(f"\nq_norm:")
    print(f"  allclose: {q_close} (rtol=1e-2, atol=1e-2)")
    print(f"  PCC:      {q_pcc:.6f}")
    print(f"  Max error: {q_max_err:.6e}")
    print(f"kv_norm:")
    print(f"  allclose: {kv_close} (rtol=1e-2, atol=1e-2)")
    print(f"  PCC:      {kv_pcc:.6f}")
    print(f"  Max error: {kv_max_err:.6e}")

    assert q_close, f"q_norm failed allclose check (max error: {q_max_err:.6e})"
    assert kv_close, f"kv_norm failed allclose check (max error: {kv_max_err:.6e})"

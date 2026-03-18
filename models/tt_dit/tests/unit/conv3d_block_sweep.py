# SPDX-FileCopyrightText: © 2025 TenstorreAnt AI ULC

# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict

import pytest
import torch

import ttnn

from ....utils.conv3d import aligned_channels


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 4),
        (2, 4),
        (4, 8),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "target_mesh_shape",
    [
        (4, 8),
        (4, 32),
    ],
)
def test_conv3d_blocking_sweep(mesh_device, target_mesh_shape, target_H, target_W):
    """
    Sweep over blocking configurations for conv3d to find optimal settings.
    """

    # Per-device input/weight shapes and baseline blockings from VAE decoder
    # (T, H, W, C_in, kernel_size, C_out, (T_block, H_block, W_block, C_in_block, C_out_block))
    if target_mesh_shape == (4, 8):
        # On 4x8 mesh final shape is (184, 160) + conv padding = (186, 162), since H is padded so latent size (90, 160) divides (4, 8).
        conv_configs = [
            (3, 25, 22, 32, (3, 3, 3), 384, (1, 8, 8, 32, 384)),
            (3, 25, 22, 384, (3, 3, 3), 384, (1, 8, 2, 128, 128)),
            (3, 48, 42, 192, (3, 3, 3), 384, (1, 32, 1, 96, 128)),
            (3, 48, 42, 384, (3, 3, 3), 384, (1, 8, 2, 128, 128)),
            (3, 94, 82, 192, (3, 3, 3), 192, (1, 8, 4, 96, 96)),
            (3, 186, 162, 96, (3, 3, 3), 96, (1, 8, 8, 96, 96)),
            (3, 186, 162, 96, (3, 3, 3), 32, (1, 16, 8, 96, 32)),
            (3, 23, 20, 384, (3, 1, 1), 768, (1, 1, 1, 32, 1)),
            (4, 48, 42, 192, (3, 3, 3), 384, (1, 32, 1, 96, 128)),
            (4, 48, 42, 384, (3, 3, 3), 384, (1, 8, 2, 128, 128)),
            (4, 46, 40, 384, (3, 1, 1), 768, (1, 1, 1, 32, 1)),
            (6, 94, 82, 192, (3, 3, 3), 192, (1, 8, 4, 96, 96)),
            (6, 186, 162, 96, (3, 3, 3), 96, (1, 8, 8, 96, 96)),
            (6, 186, 162, 96, (3, 3, 3), 32, (1, 16, 8, 96, 32)),
        ]
    elif target_mesh_shape == (4, 32):
        conv_configs = [
            (3, 25, 7, 32, (3, 3, 3), 384, (1, 8, 1, 32, 384)),
            (3, 25, 7, 384, (3, 3, 3), 384, (1, 8, 2, 128, 128)),
            (3, 48, 12, 192, (3, 3, 3), 384, (1, 8, 4, 96, 128)),
            (3, 48, 12, 384, (3, 3, 3), 384, (1, 8, 2, 128, 128)),
            (3, 94, 22, 192, (3, 3, 3), 192, (1, 8, 4, 96, 96)),
            (3, 186, 42, 96, (3, 3, 3), 96, (1, 4, 8, 96, 96)),
            (3, 186, 42, 96, (3, 3, 3), 32, (1, 16, 8, 96, 32)),
            (3, 23, 5, 384, (3, 1, 1), 768, (1, 1, 1, 32, 1)),
            (4, 48, 12, 192, (3, 3, 3), 384, (1, 8, 4, 96, 128)),
            (4, 48, 12, 384, (3, 3, 3), 384, (1, 8, 2, 128, 128)),
            (4, 46, 10, 384, (3, 1, 1), 768, (1, 1, 1, 32, 1)),
            (6, 94, 22, 192, (3, 3, 3), 192, (1, 8, 4, 96, 96)),
            (6, 186, 42, 96, (3, 3, 3), 96, (1, 4, 8, 96, 96)),
            (6, 186, 42, 96, (3, 3, 3), 32, (1, 16, 8, 96, 32)),
        ]

    grid_size = mesh_device.compute_with_storage_grid_size()

    # Use same compute kernel config as WanCausalConv3d (HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # Generate H_out_block and W_out_block variations
    hw_variations = [
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (1, 16),
        (1, 32),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (2, 16),
        (2, 32),
        (4, 1),
        (4, 2),
        (4, 4),
        (4, 8),
        (4, 16),
        (4, 32),
        (8, 1),
        (8, 2),
        (8, 4),
        (8, 8),
        (8, 16),
        (8, 32),
        (16, 1),
        (16, 2),
        (16, 4),
        (16, 8),
        (16, 16),
        (32, 1),
        (32, 2),
        (32, 4),
        (32, 8),
    ]

    def get_cin_cout_variations(C_in, C_out, C_in_block_base, C_out_block_base, kernel_size):
        """Generate C_in_block/C_out_block variations centered on baseline."""
        padded_C_in = aligned_channels(C_in)
        kT, kH, kW = kernel_size
        kernel_vol = kT * kH * kW

        def valid_cin(c):
            """C_in_block must divide padded_C_in and patch_size must be tile-aligned."""
            return c >= 32 and c <= padded_C_in and padded_C_in % c == 0 and (kernel_vol * c) % 32 == 0

        def valid_cout(c):
            """C_out_block must be a multiple of 32 and divide C_out evenly."""
            return c >= 32 and c <= C_out and C_out % c == 0 and c % 32 == 0

        # Enumerate all valid C_in_block and C_out_block values (multiples of 32)
        valid_cins = [c for c in range(32, padded_C_in + 1, 32) if valid_cin(c)]
        valid_couts = [c for c in range(32, C_out + 1, 32) if valid_cout(c)]

        variations = set()
        variations.add((C_in_block_base, C_out_block_base))

        # All C_in_block values with baseline C_out_block
        for c_in in valid_cins:
            variations.add((c_in, C_out_block_base))

        # All C_out_block values with baseline C_in_block
        for c_out in valid_couts:
            variations.add((C_in_block_base, c_out))

        # Cross-product of non-baseline values (capped to avoid explosion)
        non_base_cins = [c for c in valid_cins if c != C_in_block_base]
        non_base_couts = [c for c in valid_couts if c != C_out_block_base]
        for c_in in non_base_cins:
            for c_out in non_base_couts:
                variations.add((c_in, c_out))

        # Sort with baseline first
        result = [(C_in_block_base, C_out_block_base)]
        for v in sorted(variations):
            if v != (C_in_block_base, C_out_block_base):
                result.append(v)
        return result

    results = []

    print("\n" + "=" * 100)
    print("CONV3D BLOCKING SWEEP")
    print("=" * 100)

    for T, H, W, C_in, kernel_size, C_out, baseline in conv_configs:
        config_key = (C_in, C_out, kernel_size, T, H, W)
        T_base, H_base, W_base, C_in_block_base, C_out_block_base = baseline

        # Get C_in/C_out variations (~1-5 options)
        cin_cout_variations = get_cin_cout_variations(C_in, C_out, C_in_block_base, C_out_block_base, kernel_size)

        print(f"\n{'='*80}")
        print(f"Config: C_in={C_in}, C_out={C_out}, kernel={kernel_size}, shape=({T},{H},{W})")
        print(f"Baseline: Cin_blk={C_in_block_base}, Cout_blk={C_out_block_base}, T={T_base}, H={H_base}, W={W_base}")
        print(f"Cin/Cout variations: {cin_cout_variations}")
        print(f"{'='*80}")

        # WanCausalConv3d applies all padding externally:
        #   - Temporal: causal zero-pad or cache concat (T already includes this)
        #   - Spatial: neighbor_pad_async halo (H, W already include this)
        # Conv3d is called with internal_padding=(0,0,0).
        padding = (0, 0, 0)

        # Create weights and input ONCE per config (outside C_in/C_out loop)
        torch_weight = torch.randn(C_out, C_in, *kernel_size, dtype=torch.float32)
        torch_bias = torch.randn(C_out, dtype=torch.float32)

        padded_C_in = aligned_channels(C_in)

        # T, H, W are the per-device shapes conv3d actually sees (post all external padding)
        torch_input = torch.randn(1, T, H, W, padded_C_in, dtype=torch.float32)

        # Create input tensor
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Prepare weights (same transform as prepare_conv3d_weights with single C_in block)
        w = torch_weight.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C_in, C_out
        if padded_C_in != C_in:
            w = torch.nn.functional.pad(w, (0, 0, 0, padded_C_in - C_in))
        w = w.reshape(-1, C_out)

        tt_weight = ttnn.from_torch(
            w, device=mesh_device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, pad_value=0
        )
        tt_bias = ttnn.from_torch(
            torch_bias.reshape(1, -1),
            device=mesh_device,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            pad_value=0,
        )

        # Track best result for this config
        best_time = float("inf")
        best_blocking = None
        baseline_time = None

        # Build list of all blockings to test
        blockings_to_test = []
        for C_in_block, C_out_block in cin_cout_variations:
            for h, w in hw_variations:
                blockings_to_test.append((C_in_block, C_out_block, h, w, False))
        print(f"Testing {len(blockings_to_test)} blocking configurations...")

        for C_in_block, C_out_block, H_out_block, W_out_block, is_baseline in blockings_to_test:
            blocking = (C_in_block, C_out_block, T_base, H_out_block, W_out_block)

            try:
                # Create conv config
                conv_config = ttnn.Conv3dConfig(
                    weights_dtype=ttnn.bfloat16,
                    output_layout=ttnn.ROW_MAJOR_LAYOUT,
                    T_out_block=T_base,
                    W_out_block=W_out_block,
                    H_out_block=H_out_block,
                    C_out_block=C_out_block,
                    C_in_block=C_in_block,
                    compute_with_storage_grid_size=grid_size,
                )

                # Warmup
                out = ttnn.experimental.conv3d(
                    input_tensor=tt_input,
                    weight_tensor=tt_weight,
                    bias_tensor=tt_bias,
                    config=conv_config,
                    output_channels=C_out,
                    kernel_size=kernel_size,
                    stride=(1, 1, 1),
                    padding=padding,
                    padding_mode="zeros",
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=compute_kernel_config,
                )
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(out)

                # Timed runs
                times = []
                for _ in range(3):
                    start = time.perf_counter()
                    out = ttnn.experimental.conv3d(
                        input_tensor=tt_input,
                        weight_tensor=tt_weight,
                        bias_tensor=tt_bias,
                        config=conv_config,
                        output_channels=C_out,
                        kernel_size=kernel_size,
                        stride=(1, 1, 1),
                        padding=padding,
                        padding_mode="zeros",
                        dtype=ttnn.bfloat16,
                        compute_kernel_config=compute_kernel_config,
                    )
                    ttnn.synchronize_device(mesh_device)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                    ttnn.deallocate(out)

                avg_ms = sum(times) / len(times) * 1000
                min_ms = min(times) * 1000

                # Track best
                if avg_ms < best_time:
                    best_time = avg_ms
                    best_blocking = blocking

                if is_baseline:
                    baseline_time = avg_ms
                    marker = " [BASELINE]"
                else:
                    marker = ""

                # Print result immediately
                cin_cout_info = (
                    ""
                    if (C_in_block == C_in_block_base and C_out_block == C_out_block_base)
                    else f" [Cin={C_in_block}, Cout={C_out_block}]"
                )
                print(
                    f"  H={H_out_block:2d}, W={W_out_block:2d}: avg={avg_ms:7.2f}ms, min={min_ms:7.2f}ms{marker}{cin_cout_info}"
                )

                results.append(
                    {
                        "config": config_key,
                        "shape": (T, H, W),
                        "blocking": blocking,
                        "avg_ms": avg_ms,
                        "min_ms": min_ms,
                        "is_baseline": is_baseline,
                        "status": "OK",
                    }
                )

            except Exception as e:
                cin_cout_info = (
                    ""
                    if (C_in_block == C_in_block_base and C_out_block == C_out_block_base)
                    else f" [Cin={C_in_block}, Cout={C_out_block}]"
                )
                print(f"  H={H_out_block:2d}, W={W_out_block:2d}: FAILED - {str(e)[:50]}{cin_cout_info}")
                results.append(
                    {
                        "config": config_key,
                        "shape": (T, H, W),
                        "blocking": blocking,
                        "avg_ms": None,
                        "min_ms": None,
                        "is_baseline": is_baseline,
                        "status": f"FAILED",
                    }
                )

        # Cleanup tensors for this config
        ttnn.deallocate(tt_input)
        ttnn.deallocate(tt_weight)
        if tt_bias is not None:
            ttnn.deallocate(tt_bias)

        # Print summary for this config
        if baseline_time and best_blocking:
            print(f"\n  Summary: baseline={baseline_time:.2f}ms, best={best_time:.2f}ms @ {best_blocking}")
            if best_time < baseline_time:
                print(f"  Speedup: {baseline_time/best_time:.2f}x")

    # Final summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    successful = [r for r in results if r["status"] == "OK"]
    print(f"Total: {len(successful)}/{len(results)} configurations succeeded")

    # Group by config and find best for each
    by_config = defaultdict(list)
    for r in successful:
        by_config[r["config"]].append(r)

    for config_key, config_results in by_config.items():
        baseline_r = next((r for r in config_results if r["is_baseline"]), None)
        best_r = min(config_results, key=lambda x: x["avg_ms"])
        print(f"\n{config_key}:")
        if baseline_r:
            b = baseline_r["blocking"]
            print(f"  Baseline: Cin={b[0]}, Cout={b[1]}, H={b[3]}, W={b[4]} -> {baseline_r['avg_ms']:.2f}ms")
        b = best_r["blocking"]
        print(f"  Best:     Cin={b[0]}, Cout={b[1]}, H={b[3]}, W={b[4]} -> {best_r['avg_ms']:.2f}ms")
        if baseline_r and best_r["avg_ms"] < baseline_r["avg_ms"]:
            print(f"  Speedup:  {baseline_r['avg_ms']/best_r['avg_ms']:.2f}x")

    assert len(successful) > 0, "All blocking configurations failed!"

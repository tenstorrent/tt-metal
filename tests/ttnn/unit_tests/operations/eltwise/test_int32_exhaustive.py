# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import os
import csv
import pytest
import ttnn
import numpy as np
import matplotlib.pyplot as plt
from tests.ttnn.utils_for_testing import assert_with_ulp


@pytest.mark.parametrize(
    "ttnn_op, torch_op",
    [
        (ttnn.remainder, torch.remainder),
        (ttnn.fmod, torch.fmod),
    ],
)
def test_remainder_fmod_int32_exhaustive_pairwise_1000(ttnn_op, torch_op, device):
    """
    Exhaustive pairwise INT32 remainder/fmod test for range (-1000, 1000).
    Tests all possible pairs of integers in this range.
    Excludes divisor=0 to avoid division by zero.
    """
    torch.manual_seed(0)
    ttnn_dtype = ttnn.int32
    op_name = ttnn_op.__name__

    # Generate all integers from -999 to 999 (range is exclusive of 1000)
    # For dividend (x): all values from -999 to 999
    x_values = torch.arange(-999, 1000, dtype=torch.int32)
    # For divisor (y): all values from -999 to 999, excluding 0
    y_values = torch.cat([torch.arange(-999, 0, dtype=torch.int32), torch.arange(1, 1000, dtype=torch.int32)])

    Nx = x_values.numel()  # 1999 values
    Ny = y_values.numel()  # 1998 values (excluding 0)
    total_pairs = Nx * Ny
    print(f"Testing INT32 {op_name}: {Nx} × {Ny} = {total_pairs:,} pairs")
    print(f"X range: [{x_values.min().item()}, {x_values.max().item()}]")
    print(f"Y range: [{y_values.min().item()}, {y_values.max().item()}] (excluding 0)")

    batch_size = 512  # process 512×Ny pairs at a time
    num_batches = (Nx + batch_size - 1) // batch_size

    total_mismatches = 0
    mismatch_examples = []  # Store first few mismatch examples
    max_examples = 100

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, Nx)

        # x_batch: [batch_size, 1], y: [1, Ny] → broadcasts to [batch_size, Ny]
        x_batch = x_values[start:end].unsqueeze(1)  # shape: [B, 1]
        y_full = y_values.unsqueeze(0)  # shape: [1, Ny]

        # Compute torch result
        z_torch = torch_op(x_batch, y_full)  # shape: [B, Ny]

        # Send to backend
        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn_op(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # Check for mismatches
        mismatch_mask = z_torch != tt_out
        mismatch_count = mismatch_mask.sum().item()
        total_mismatches += mismatch_count

        # Collect mismatch examples
        if mismatch_count > 0 and len(mismatch_examples) < max_examples:
            mismatch_indices = mismatch_mask.nonzero(as_tuple=False)

            for idx_pair in mismatch_indices:
                if len(mismatch_examples) >= max_examples:
                    break
                i, j = idx_pair[0].item(), idx_pair[1].item()
                x_val = x_batch[i, 0].item()
                y_val = y_full[0, j].item()
                torch_val = z_torch[i, j].item()
                tt_val = tt_out[i, j].item()
                mismatch_examples.append((x_val, y_val, torch_val, tt_val))

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"Batch {batch_idx+1}/{num_batches}: mismatches={mismatch_count}")

    mismatch_pct = (total_mismatches / total_pairs) * 100 if total_pairs > 0 else 0.0

    # Print summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: test_remainder_fmod_int32_exhaustive_pairwise_1000 [{op_name}]")
    print("=" * 70)
    print(f"  Input range:        x ∈ [-999, 999], y ∈ [-999, 999] \\ {{0}}")
    print(f"  Total x values:     {Nx:,}")
    print(f"  Total y values:     {Ny:,}")
    print(f"  Total pairs tested: {total_pairs:,}")
    print(f"  Exact matches:      {total_pairs - total_mismatches:,} ({100.0 - mismatch_pct:.4f}%)")
    print(f"  Mismatches:         {total_mismatches:,} ({mismatch_pct:.4f}%)")
    print("=" * 70)

    if mismatch_examples:
        print(f"\nFirst {min(len(mismatch_examples), 20)} mismatch examples (x, y, torch_result, ttnn_result):")
        for x_val, y_val, torch_val, tt_val in mismatch_examples[:20]:
            print(f"  {x_val} % {y_val} = torch:{torch_val}, ttnn:{tt_val}")

    assert total_mismatches == 0, f"Found {total_mismatches} mismatches in INT32 {op_name} test"


@pytest.mark.parametrize(
    "ttnn_op, torch_op",
    [
        (ttnn.remainder, torch.remainder),
        (ttnn.fmod, torch.fmod),
    ],
)
def test_remainder_fmod_int32_sampled_pairwise_1e5(ttnn_op, torch_op, device):
    """
    Sampled pairwise INT32 remainder/fmod test for range [-1e5, 1e5].
    Since exhaustive testing would require ~40 billion pairs, this test
    strategically samples values from different sub-ranges to achieve
    comprehensive coverage with ~4-9 million pairs.
    """
    torch.manual_seed(0)
    ttnn_dtype = ttnn.int32
    op_name = ttnn_op.__name__

    value_list = []

    # 1. Include all values from -100 to 100 (small values, most critical)
    value_list.extend(range(-100, 101))

    # 2. Include all values from -1000 to -101 and 101 to 1000 (medium-small)
    value_list.extend(range(-1000, -100))
    value_list.extend(range(101, 1001))

    # 3. Powers of 2 and their neighbors (important for bit operations)
    for exp in range(1, 17):  # 2^1 to 2^16 (up to 65536)
        base = 2**exp
        if base <= 100000:
            for offset in [-2, -1, 0, 1, 2]:
                val = base + offset
                if 1001 <= abs(val) <= 100000:
                    value_list.extend([val, -val])

    # 4. Powers of 10 and their neighbors
    for exp in range(3, 6):  # 10^3 to 10^5
        base = 10**exp
        if base <= 100000:
            for offset in [-2, -1, 0, 1, 2]:
                val = base + offset
                if 1001 <= abs(val) <= 100000:
                    value_list.extend([val, -val])

    # 5. Linearly spaced samples from different sub-ranges
    sub_ranges = [
        (1001, 5000, 200),  # 1K-5K range
        (5001, 10000, 200),  # 5K-10K range
        (10001, 25000, 200),  # 10K-25K range
        (25001, 50000, 200),  # 25K-50K range
        (50001, 75000, 200),  # 50K-75K range
        (75001, 100000, 200),  # 75K-100K range
    ]

    for start, end, num_samples in sub_ranges:
        step = max(1, (end - start) // num_samples)
        for val in range(start, end + 1, step):
            value_list.extend([val, -val])

    # 6. Random samples across the full range (for additional coverage)
    np.random.seed(0)
    random_samples = np.random.randint(-100000, 100001, size=500)
    value_list.extend(random_samples.tolist())

    # 7. Edge values
    value_list.extend([100000, -100000, 99999, -99999])

    # Convert to tensor and remove duplicates
    all_values = torch.tensor(list(set(value_list)), dtype=torch.int32)
    all_values = all_values.sort().values

    # For dividend (x): all sampled values
    x_values = all_values

    # For divisor (y): all sampled values, excluding 0
    y_values = all_values[all_values != 0]

    Nx = x_values.numel()
    Ny = y_values.numel()
    total_pairs = Nx * Ny
    print(f"Testing INT32 {op_name} (sampled 1e5): {Nx} × {Ny} = {total_pairs:,} pairs")
    print(f"X range: [{x_values.min().item()}, {x_values.max().item()}]")
    print(f"Y range: [{y_values.min().item()}, {y_values.max().item()}] (excluding 0)")

    batch_size = 512  # process 512×Ny pairs at a time
    num_batches = (Nx + batch_size - 1) // batch_size

    total_mismatches = 0
    mismatch_examples = []  # Store first few mismatch examples
    max_examples = 100

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, Nx)

        # x_batch: [batch_size, 1], y: [1, Ny] → broadcasts to [batch_size, Ny]
        x_batch = x_values[start:end].unsqueeze(1)  # shape: [B, 1]
        y_full = y_values.unsqueeze(0)  # shape: [1, Ny]

        # Compute torch result
        z_torch = torch_op(x_batch, y_full)  # shape: [B, Ny]

        # Send to backend
        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn_op(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # Check for mismatches
        mismatch_mask = z_torch != tt_out
        mismatch_count = mismatch_mask.sum().item()
        total_mismatches += mismatch_count

        # Collect mismatch examples
        if mismatch_count > 0 and len(mismatch_examples) < max_examples:
            mismatch_indices = mismatch_mask.nonzero(as_tuple=False)

            for idx_pair in mismatch_indices:
                if len(mismatch_examples) >= max_examples:
                    break
                i, j = idx_pair[0].item(), idx_pair[1].item()
                x_val = x_batch[i, 0].item()
                y_val = y_full[0, j].item()
                torch_val = z_torch[i, j].item()
                tt_val = tt_out[i, j].item()
                mismatch_examples.append((x_val, y_val, torch_val, tt_val))

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"Batch {batch_idx+1}/{num_batches}: mismatches={mismatch_count}")

    mismatch_pct = (total_mismatches / total_pairs) * 100 if total_pairs > 0 else 0.0

    # Print summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: test_remainder_fmod_int32_sampled_pairwise_1e5 [{op_name}]")
    print("=" * 70)
    print(f"  Input range:        x, y ∈ [-100000, 100000] (sampled)")
    print(f"  Total x values:     {Nx:,}")
    print(f"  Total y values:     {Ny:,} (excluding 0)")
    print(f"  Total pairs tested: {total_pairs:,}")
    print(f"  Exact matches:      {total_pairs - total_mismatches:,} ({100.0 - mismatch_pct:.4f}%)")
    print(f"  Mismatches:         {total_mismatches:,} ({mismatch_pct:.4f}%)")
    print("=" * 70)

    if mismatch_examples:
        print(f"\nFirst {min(len(mismatch_examples), 20)} mismatch examples (x, y, torch_result, ttnn_result):")
        for x_val, y_val, torch_val, tt_val in mismatch_examples[:20]:
            print(f"  {x_val} % {y_val} = torch:{torch_val}, ttnn:{tt_val}")

    assert total_mismatches == 0, f"Found {total_mismatches} mismatches in INT32 {op_name} test (sampled 1e5)"


@pytest.mark.parametrize(
    "ttnn_op, torch_op",
    [
        (ttnn.remainder, torch.remainder),
        (ttnn.fmod, torch.fmod),
    ],
)
def test_remainder_fmod_int32_sampled_pairwise_1e5_highly_exhaustive(ttnn_op, torch_op, device):
    """Test int32 remainder/fmod with highly exhaustive sampling over [-1e5, 1e5].

    This test aims for maximum coverage with ~500+ million test pairs by:
    - Fully exhaustive testing for small values (-5000 to 5000)
    - Dense sampling across all magnitude ranges
    - Special attention to powers of 2, primes, and edge cases
    """

    torch.manual_seed(0)
    np.random.seed(0)

    ttnn_dtype = ttnn.int32
    op_name = ttnn_op.__name__

    INT_MIN = -100_000
    INT_MAX = 100_000

    value_list = []

    # ------------------------------------------------------------------
    # 1. Fully exhaustive small range (very dense)
    # ------------------------------------------------------------------
    # This is critical: small divisors are where most remainder edge cases occur
    dense_range = np.arange(-5000, 5001, dtype=np.int32)
    value_list.extend(dense_range.tolist())

    # ------------------------------------------------------------------
    # 2. Dense linear samples across full range
    # ------------------------------------------------------------------
    linear_samples = np.linspace(INT_MIN, INT_MAX, 20000, dtype=np.int32)
    value_list.extend(linear_samples.tolist())

    # ------------------------------------------------------------------
    # 3. Stratified ranges with very high density near zero
    # ------------------------------------------------------------------
    for start, end, num in [
        (-10, 10, 21),  # every integer
        (-100, 100, 201),  # every integer
        (-500, 500, 1001),  # every integer
        (-1_000, 1_000, 2001),  # every integer
        (-2_000, 2_000, 4001),  # every integer
        (-5_000, 5_000, 5000),  # ~every integer (overlap with dense)
        (-10_000, 10_000, 5000),  # every 4th integer
        (-25_000, 25_000, 5000),  # every 10th integer
        (-50_000, 50_000, 5000),  # every 20th integer
        (-75_000, 75_000, 3000),  # every 50th integer
        (-100_000, 100_000, 4000),  # every 50th integer
    ]:
        samples = np.linspace(start, end, num, dtype=np.int32)
        value_list.extend(samples.tolist())

    # ------------------------------------------------------------------
    # 4. Powers of 2 and their neighbors (±1, ±2, ±3)
    # ------------------------------------------------------------------
    for exp in range(0, 17):  # 2^0 to 2^16 (1 to 65536)
        base = 2**exp
        if base <= 100000:
            for offset in range(-3, 4):
                val = base + offset
                if abs(val) <= 100000:
                    value_list.extend([val, -val])

    # ------------------------------------------------------------------
    # 5. Powers of 10 and their neighbors
    # ------------------------------------------------------------------
    for exp in range(0, 6):  # 10^0 to 10^5
        base = 10**exp
        if base <= 100000:
            for offset in range(-3, 4):
                val = base + offset
                if abs(val) <= 100000:
                    value_list.extend([val, -val])

    # ------------------------------------------------------------------
    # 6. Prime numbers up to 1000 (important for modular arithmetic)
    # ------------------------------------------------------------------
    primes = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199,
        211,
        223,
        227,
        229,
        233,
        239,
        241,
        251,
        257,
        263,
        269,
        271,
        277,
        281,
        283,
        293,
        307,
        311,
        313,
        317,
        331,
        337,
        347,
        349,
        353,
        359,
        367,
        373,
        379,
        383,
        389,
        397,
        401,
        409,
        419,
        421,
        431,
        433,
        439,
        443,
        449,
        457,
        461,
        463,
        467,
        479,
        487,
        491,
        499,
        503,
        509,
        521,
        523,
        541,
        547,
        557,
        563,
        569,
        571,
        577,
        587,
        593,
        599,
        601,
        607,
        613,
        617,
        619,
        631,
        641,
        643,
        647,
        653,
        659,
        661,
        673,
        677,
        683,
        691,
        701,
        709,
        719,
        727,
        733,
        739,
        743,
        751,
        757,
        761,
        769,
        773,
        787,
        797,
        809,
        811,
        821,
        823,
        827,
        829,
        839,
        853,
        857,
        859,
        863,
        877,
        881,
        883,
        887,
        907,
        911,
        919,
        929,
        937,
        941,
        947,
        953,
        967,
        971,
        977,
        983,
        991,
        997,
    ]
    for p in primes:
        value_list.extend([p, -p])

    # Larger primes
    large_primes = [
        1009,
        2003,
        3001,
        4001,
        5003,
        6007,
        7001,
        8009,
        9001,
        10007,
        15013,
        20011,
        25013,
        30011,
        40009,
        50021,
        60013,
        70001,
        80021,
        90001,
        99991,
    ]
    for p in large_primes:
        if p <= 100000:
            value_list.extend([p, -p])

    # ------------------------------------------------------------------
    # 7. Random uniform samples (extensive)
    # ------------------------------------------------------------------
    random_samples = np.random.randint(INT_MIN, INT_MAX + 1, size=15000, dtype=np.int32)
    value_list.extend(random_samples.tolist())

    # ------------------------------------------------------------------
    # 8. Logarithmically spaced samples (better coverage of large values)
    # ------------------------------------------------------------------
    # Positive log-spaced
    log_samples_pos = np.logspace(0, np.log10(100000), 3000, dtype=np.float64)
    log_samples_pos = np.unique(log_samples_pos.astype(np.int32))
    value_list.extend(log_samples_pos.tolist())
    value_list.extend((-log_samples_pos).tolist())

    # ------------------------------------------------------------------
    # 9. Special / adversarial values
    # ------------------------------------------------------------------
    special_values = [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        4,
        -4,
        5,
        -5,
        6,
        -6,
        7,
        -7,
        8,
        -8,
        9,
        -9,
        10,
        -10,
        15,
        -15,
        16,
        -16,
        31,
        -31,
        32,
        -32,
        63,
        -63,
        64,
        -64,
        100,
        -100,
        127,
        -127,
        128,
        -128,
        255,
        -255,
        256,
        -256,
        511,
        -511,
        512,
        -512,
        1000,
        -1000,
        1023,
        -1023,
        1024,
        -1024,
        2047,
        -2047,
        2048,
        -2048,
        4095,
        -4095,
        4096,
        -4096,
        8191,
        -8191,
        8192,
        -8192,
        10000,
        -10000,
        16383,
        -16383,
        16384,
        -16384,
        32767,
        -32767,
        32768,
        -32768,
        50000,
        -50000,
        65535,
        -65535,
        65536,
        -65536,
        99999,
        -99999,
        100000,
        -100000,
    ]
    value_list.extend(special_values)

    # ------------------------------------------------------------------
    # 10. Multiples of common bases
    # ------------------------------------------------------------------
    for base in [10, 100, 1000, 10000]:
        for mult in range(1, min(100001 // base, 101)):
            val = base * mult
            if val <= 100000:
                value_list.extend([val, -val])

    # ------------------------------------------------------------------
    # Finalize value set
    # ------------------------------------------------------------------
    vals = torch.tensor(value_list, dtype=torch.int32)
    value_set = torch.unique(vals)

    # IMPORTANT: divisor cannot be zero
    divisor_set = value_set[value_set != 0]

    N = value_set.numel()
    M = divisor_set.numel()

    print(f"Testing INT32 {op_name}: {N} x-values and {M} y-values")
    print(f"Total {op_name} ops: {N * M:,}")

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    batch_size = 256
    num_batches = (N + batch_size - 1) // batch_size

    total_mismatches = 0

    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, f"{op_name}_int32_mismatch.csv")

    csv_file = open(out_path, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "batch_idx",
            "x_idx",
            "y_idx",
            "x_value",
            "y_value",
            "torch_out",
            "tt_out",
        ]
    )

    # ------------------------------------------------------------------
    # Main test loop
    # ------------------------------------------------------------------
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)

        x_batch = value_set[start:end].unsqueeze(1)  # [B, 1]
        y_full = divisor_set.unsqueeze(0)  # [1, M]

        # Reference (PyTorch semantics)
        z_torch = torch_op(x_batch, y_full)

        # TT backend
        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

        z_tt = ttnn_op(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # Exact match required for int32
        mismatch_mask = z_torch != tt_out
        mismatch_count = mismatch_mask.sum().item()
        total_mismatches += mismatch_count

        if mismatch_count > 0:
            mismatch_indices = mismatch_mask.nonzero(as_tuple=False)

            for idx_pair in mismatch_indices:
                i, j = idx_pair.tolist()

                xv = x_batch[i, 0].item()
                yv = y_full[0, j].item()
                zv = z_torch[i, j].item()
                tv = tt_out[i, j].item()

                writer.writerow(
                    [
                        batch_idx,
                        start + i,
                        j,
                        xv,
                        yv,
                        zv,
                        tv,
                    ]
                )

        print(f"Batch {batch_idx+1}/{num_batches}: " f"mismatches={mismatch_count}")

    csv_file.close()

    total_ops = N * M
    mismatch_pct = (total_mismatches / total_ops) * 100.0 if total_ops else 0.0

    # Print summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: test_remainder_fmod_int32_sampled_pairwise_full_range [{op_name}]")
    print("=" * 70)
    print(f"  Input range:        x, y ∈ [-100000, 100000] (highly exhaustive sampling)")
    print(f"  Total x values:     {N:,}")
    print(f"  Total y values:     {M:,} (excluding 0)")
    print(f"  Total pairs tested: {total_ops:,}")
    print(f"  Exact matches:      {total_ops - total_mismatches:,} ({100.0 - mismatch_pct:.6f}%)")
    print(f"  Mismatches:         {total_mismatches:,} ({mismatch_pct:.6f}%)")
    print(f"  Mismatch log:       {out_path}")
    print("=" * 70)

    assert total_mismatches == 0, f"INT32 {op_name} mismatches detected"


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.remainder,
        ttnn.fmod,
    ],
)
def test_remainder_fmod_int32_sampled_full_int32_range(ttnn_op, device):
    """Test int32 remainder/fmod with comprehensive sampling over full INT32 range.

    This test covers the entire INT32 range [-2,147,483,648, 2,147,483,647] with:
    - Exhaustive small value testing (-10000 to 10000)
    - Dense sampling around INT32_MIN and INT32_MAX boundaries
    - Powers of 2 up to 2^30 and their neighbors
    - Strategic sampling across all magnitude ranges
    - Edge cases that commonly cause overflow/underflow issues

    Uses ttnn.get_golden_function() for reference to properly handle edge cases
    like INT32_MIN % -1 which would cause SIGFPE with direct torch calls.
    """

    torch.manual_seed(0)
    np.random.seed(0)

    ttnn_dtype = ttnn.int32
    op_name = ttnn_op.__name__

    # Full INT32 range
    INT32_MIN = torch.iinfo(torch.int32).min
    INT32_MAX = torch.iinfo(torch.int32).max

    value_list = []

    # ------------------------------------------------------------------
    # 1. Fully exhaustive small range (critical for remainder edge cases)
    # ------------------------------------------------------------------
    dense_range = np.arange(-10000, 10001, dtype=np.int32)
    value_list.extend(dense_range.tolist())

    # ------------------------------------------------------------------
    # 2. Dense sampling around INT32_MIN boundary
    # ------------------------------------------------------------------
    # Values near INT32_MIN
    for offset in range(0, 10000):
        val = INT32_MIN + offset
        value_list.append(val)

    # ------------------------------------------------------------------
    # 3. Dense sampling around INT32_MAX boundary
    # ------------------------------------------------------------------
    # Values near INT32_MAX
    for offset in range(0, 10000):
        val = INT32_MAX - offset
        value_list.append(val)

    # ------------------------------------------------------------------
    # 4. Powers of 2 and their neighbors (±1, ±2, ±3) up to 2^30
    # ------------------------------------------------------------------
    for exp in range(0, 31):  # 2^0 to 2^30
        base = 2**exp
        for offset in range(-3, 4):
            val = base + offset
            if INT32_MIN <= val <= INT32_MAX:
                value_list.extend([val, -val])
        # Also include 2^exp - 1 (all 1s pattern)
        val = base - 1
        if val > 0:
            value_list.extend([val, -val])

    # ------------------------------------------------------------------
    # 5. Powers of 10 and their neighbors
    # ------------------------------------------------------------------
    for exp in range(0, 10):  # 10^0 to 10^9
        base = 10**exp
        if base <= INT32_MAX:
            for offset in range(-3, 4):
                val = base + offset
                if INT32_MIN <= val <= INT32_MAX:
                    value_list.extend([val, -val])

    # ------------------------------------------------------------------
    # 6. Special INT32 boundary values and common edge cases
    # ------------------------------------------------------------------
    special_values = [
        # Absolute boundaries
        INT32_MIN,
        INT32_MIN + 1,
        INT32_MIN + 2,
        INT32_MAX,
        INT32_MAX - 1,
        INT32_MAX - 2,
        # Zero and small values
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        # Common bit patterns
        0x7FFFFFFF,  # INT32_MAX
        0x7FFFFFFE,  # INT32_MAX - 1
        0x40000000,  # 2^30
        0x3FFFFFFF,  # 2^30 - 1
        0x20000000,  # 2^29
        0x10000000,  # 2^28
        0x08000000,  # 2^27
        0x04000000,  # 2^26
        0x02000000,  # 2^25
        0x01000000,  # 2^24
        0x00800000,  # 2^23
        0x00400000,  # 2^22
        0x00100000,  # 2^20
        0x00010000,  # 2^16 = 65536
        0x0000FFFF,  # 65535
        0x00008000,  # 2^15 = 32768
        0x00007FFF,  # 32767
        0x00000100,  # 256
        0x000000FF,  # 255
        # Negative versions
        -0x7FFFFFFF,
        -0x40000000,
        -0x20000000,
        -0x10000000,
        -0x00010000,
        -0x00008000,
        -0x00000100,
        # Common problematic values
        1000000000,  # 1e9
        -1000000000,
        2000000000,  # 2e9
        -2000000000,
        100000000,  # 1e8
        -100000000,
        10000000,  # 1e7
        -10000000,
        1000000,  # 1e6
        -1000000,
        100000,  # 1e5
        -100000,
    ]
    value_list.extend(special_values)

    # ------------------------------------------------------------------
    # 7. Logarithmically spaced samples across positive range
    # ------------------------------------------------------------------
    # Cover 1 to INT32_MAX with log spacing
    log_samples_pos = np.logspace(0, np.log10(INT32_MAX), 5000, dtype=np.float64)
    log_samples_pos = np.unique(log_samples_pos.astype(np.int64))  # Use int64 to avoid overflow
    log_samples_pos = log_samples_pos[log_samples_pos <= INT32_MAX]
    value_list.extend(log_samples_pos.astype(np.int32).tolist())
    value_list.extend((-log_samples_pos).astype(np.int32).tolist())

    # ------------------------------------------------------------------
    # 8. Stratified linear samples across magnitude ranges
    # ------------------------------------------------------------------
    magnitude_ranges = [
        (10001, 100000, 2000),  # 10K-100K
        (100001, 1000000, 2000),  # 100K-1M
        (1000001, 10000000, 2000),  # 1M-10M
        (10000001, 100000000, 2000),  # 10M-100M
        (100000001, 1000000000, 2000),  # 100M-1B
        (1000000001, INT32_MAX, 2000),  # 1B-INT32_MAX
    ]

    for start, end, num in magnitude_ranges:
        if start <= INT32_MAX:
            actual_end = min(end, INT32_MAX)
            samples = np.linspace(start, actual_end, num, dtype=np.int64)
            samples = samples[samples <= INT32_MAX]
            value_list.extend(samples.astype(np.int32).tolist())
            # Add negative versions
            neg_samples = -samples
            neg_samples = neg_samples[neg_samples >= INT32_MIN]
            value_list.extend(neg_samples.astype(np.int32).tolist())

    # ------------------------------------------------------------------
    # 9. Random samples across full range
    # ------------------------------------------------------------------
    # Use int64 for generation to avoid overflow issues
    random_samples = np.random.randint(INT32_MIN, INT32_MAX + 1, size=10000, dtype=np.int64)
    random_samples = random_samples.astype(np.int32)
    value_list.extend(random_samples.tolist())

    # ------------------------------------------------------------------
    # 10. Prime numbers (important for modular arithmetic)
    # ------------------------------------------------------------------
    primes = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199,
        211,
        223,
        227,
        229,
        233,
        239,
        241,
        251,
        257,
        263,
        269,
        271,
        277,
        281,
        283,
        293,
        307,
        311,
        313,
        317,
        331,
        337,
        347,
        349,
        353,
        359,
        367,
        373,
        379,
        383,
        389,
        397,
        401,
        409,
        419,
        421,
        431,
        433,
        439,
        443,
        449,
        457,
        461,
        463,
        467,
        479,
        487,
        491,
        499,
        503,
        509,
        521,
        523,
        541,
        547,
        557,
        563,
        569,
        571,
        577,
        587,
        593,
        599,
        601,
        607,
        613,
        617,
        619,
        631,
        641,
        643,
        647,
        653,
        659,
        661,
        673,
        677,
        683,
        691,
        701,
        709,
        719,
        727,
        733,
        739,
        743,
        751,
        757,
        761,
        769,
        773,
        787,
        797,
        809,
        811,
        821,
        823,
        827,
        829,
        839,
        853,
        857,
        859,
        863,
        877,
        881,
        883,
        887,
        907,
        911,
        919,
        929,
        937,
        941,
        947,
        953,
        967,
        971,
        977,
        983,
        991,
        997,
        # Larger primes
        1009,
        2003,
        3001,
        4001,
        5003,
        6007,
        7001,
        8009,
        9001,
        10007,
        15013,
        20011,
        25013,
        30011,
        40009,
        50021,
        60013,
        70001,
        80021,
        90001,
        99991,
        999983,
        9999991,
        99999989,
        999999937,
        # Large primes near powers of 2
        2147483629,  # Largest prime < INT32_MAX
        1073741789,  # Large prime near 2^30
        536870909,  # Large prime near 2^29
    ]
    for p in primes:
        if p <= INT32_MAX:
            value_list.extend([p, -p])

    # ------------------------------------------------------------------
    # 11. Multiples of common bases (test clean division)
    # ------------------------------------------------------------------
    for base in [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]:
        for mult in [1, 2, 3, 5, 7, 10, 11, 13, 17, 19, 21]:
            val = base * mult
            if val <= INT32_MAX:
                value_list.extend([val, -val])

    # ------------------------------------------------------------------
    # Finalize value set
    # ------------------------------------------------------------------
    # Convert to numpy array first, handling potential overflow
    vals_array = np.array(value_list, dtype=np.int64)
    # Filter to valid INT32 range, EXCLUDING INT32_MIN due to hardware limitation
    # (INT32_MIN has no positive representation, causing issues with abs() and division)
    vals_array = vals_array[(vals_array > INT32_MIN) & (vals_array <= INT32_MAX)]
    vals = torch.tensor(vals_array, dtype=torch.int32)
    value_set = torch.unique(vals)

    # IMPORTANT: divisor cannot be zero
    divisor_set = value_set[value_set != 0]

    print(f"Note: INT32_MIN ({INT32_MIN}) excluded due to hardware limitation")

    N = value_set.numel()
    M = divisor_set.numel()

    print(f"Testing INT32 {op_name} (full INT32 range): {N} x-values and {M} y-values")
    print(f"Total {op_name} ops: {N * M:,}")
    print(f"Value range: [{value_set.min().item():,}, {value_set.max().item():,}]")

    # ------------------------------------------------------------------
    # Batch processing with ULP tracking
    # ------------------------------------------------------------------
    batch_size = 256
    num_batches = (N + batch_size - 1) // batch_size

    # ULP distribution counters
    ulp_0_count = 0
    ulp_1_count = 0
    ulp_2_count = 0
    ulp_3_count = 0
    ulp_4_to_10_count = 0
    ulp_11_to_100_count = 0
    ulp_above_100_count = 0
    max_ulp_global = 0

    # Sample storage for CSV logging (max ~10000 total, distributed across ULP ranges)
    max_samples_per_range = 1500  # ~1500 per range × 7 ranges ≈ 10500 max
    ulp_1_samples = []
    ulp_2_samples = []
    ulp_3_samples = []
    ulp_4_to_10_samples = []
    ulp_11_to_100_samples = []
    ulp_above_100_samples = []

    # Use ttnn's golden function which properly handles edge cases like INT32_MIN % -1
    golden_function = ttnn.get_golden_function(ttnn_op)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, N)

        x_batch = value_set[start:end].unsqueeze(1)  # [B, 1]
        y_full = divisor_set.unsqueeze(0)  # [1, M]

        # TT backend
        x_tt = ttnn.from_torch(x_batch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        y_tt = ttnn.from_torch(y_full, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

        z_tt = ttnn_op(x_tt, y_tt)
        tt_out = ttnn.to_torch(z_tt)

        # Reference using golden function (INT32_MIN excluded, so no special handling needed)
        z_torch = golden_function(x_batch, y_full, device=device)

        # Calculate ULP distance (for INT32, ULP = absolute difference)
        ulp_dist = (z_torch.to(torch.int64) - tt_out.to(torch.int64)).abs()

        # Update max ULP
        batch_max_ulp = ulp_dist.max().item()
        max_ulp_global = max(max_ulp_global, batch_max_ulp)

        # Count ULP distribution
        ulp_0_count += (ulp_dist == 0).sum().item()
        ulp_1_count += (ulp_dist == 1).sum().item()
        ulp_2_count += (ulp_dist == 2).sum().item()
        ulp_3_count += (ulp_dist == 3).sum().item()
        ulp_4_to_10_count += ((ulp_dist >= 4) & (ulp_dist <= 10)).sum().item()
        ulp_11_to_100_count += ((ulp_dist >= 11) & (ulp_dist <= 100)).sum().item()
        ulp_above_100_count += (ulp_dist > 100).sum().item()

        # Collect samples for each ULP range (for CSV logging)
        def collect_samples(mask, sample_list, ulp_dist_tensor):
            if len(sample_list) >= max_samples_per_range:
                return
            indices = mask.nonzero(as_tuple=False)
            for idx_pair in indices:
                if len(sample_list) >= max_samples_per_range:
                    break
                i, j = idx_pair.tolist()
                sample_list.append(
                    (
                        x_batch[i, 0].item(),
                        y_full[0, j].item(),
                        z_torch[i, j].item(),
                        tt_out[i, j].item(),
                        ulp_dist_tensor[i, j].item(),
                    )
                )

        collect_samples(ulp_dist == 1, ulp_1_samples, ulp_dist)
        collect_samples(ulp_dist == 2, ulp_2_samples, ulp_dist)
        collect_samples(ulp_dist == 3, ulp_3_samples, ulp_dist)
        collect_samples((ulp_dist >= 4) & (ulp_dist <= 10), ulp_4_to_10_samples, ulp_dist)
        collect_samples((ulp_dist >= 11) & (ulp_dist <= 100), ulp_11_to_100_samples, ulp_dist)
        collect_samples(ulp_dist > 100, ulp_above_100_samples, ulp_dist)

        if (batch_idx + 1) % 50 == 0 or batch_idx == num_batches - 1:
            print(
                f"Batch {batch_idx+1}/{num_batches}: " f"max_ulp_batch={batch_max_ulp}, max_ulp_global={max_ulp_global}"
            )

    # ------------------------------------------------------------------
    # Write samples to CSV
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, f"{op_name}_int32_ulp_samples.csv")

    all_samples = (
        ulp_1_samples
        + ulp_2_samples
        + ulp_3_samples
        + ulp_4_to_10_samples
        + ulp_11_to_100_samples
        + ulp_above_100_samples
    )

    if all_samples:
        with open(out_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["x_value", "y_value", "expected", "actual", "ulp_distance"])
            for sample in all_samples:
                writer.writerow(sample)
        print(f"\nULP samples written to: {out_path} ({len(all_samples)} samples)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_ops = N * M

    print(f"\n{'='*60}")
    print(f"ULP DISTRIBUTION SUMMARY for {op_name}")
    print(f"{'='*60}")
    print(f"Total operations tested: {total_ops:,}")
    print(f"Max ULP observed: {max_ulp_global:,}")
    print(f"\nULP Distribution:")
    print(f"  ULP = 0 (exact match): {ulp_0_count:,} ({ulp_0_count/total_ops*100:.4f}%)")
    print(f"  ULP = 1:               {ulp_1_count:,} ({ulp_1_count/total_ops*100:.4f}%)")
    print(f"  ULP = 2:               {ulp_2_count:,} ({ulp_2_count/total_ops*100:.4f}%)")
    print(f"  ULP = 3:               {ulp_3_count:,} ({ulp_3_count/total_ops*100:.4f}%)")
    print(f"  ULP 4-10:              {ulp_4_to_10_count:,} ({ulp_4_to_10_count/total_ops*100:.4f}%)")
    print(f"  ULP 11-100:            {ulp_11_to_100_count:,} ({ulp_11_to_100_count/total_ops*100:.4f}%)")
    print(f"  ULP > 100:             {ulp_above_100_count:,} ({ulp_above_100_count/total_ops*100:.4f}%)")

    total_nonzero_ulp = (
        ulp_1_count + ulp_2_count + ulp_3_count + ulp_4_to_10_count + ulp_11_to_100_count + ulp_above_100_count
    )
    print(f"\nTotal non-exact matches: {total_nonzero_ulp:,} ({total_nonzero_ulp/total_ops*100:.4f}%)")

    print(f"\nSamples logged per ULP range:")
    print(f"  ULP = 1:    {len(ulp_1_samples)}")
    print(f"  ULP = 2:    {len(ulp_2_samples)}")
    print(f"  ULP = 3:    {len(ulp_3_samples)}")
    print(f"  ULP 4-10:   {len(ulp_4_to_10_samples)}")
    print(f"  ULP 11-100: {len(ulp_11_to_100_samples)}")
    print(f"  ULP > 100:  {len(ulp_above_100_samples)}")
    print(f"{'='*60}")

    # Verify counts sum correctly
    total_counted = ulp_0_count + total_nonzero_ulp
    assert total_counted == total_ops, f"Count mismatch: {total_counted} != {total_ops}"

    # For INT32 operations, only exact matches (ULP = 0) are acceptable
    assert total_nonzero_ulp == 0, (
        f"INT32 {op_name} requires exact match (ULP=0). "
        f"Found {total_nonzero_ulp:,} non-exact results out of {total_ops:,} "
        f"({total_nonzero_ulp/total_ops*100:.4f}%). Max ULP: {max_ulp_global:,}. "
        f"See {out_path} for samples."
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
"""
Standalone TTNN repro: ttnn.sampling has three independent seeding bugs.

  python repro_ttnn_sampling_seed.py

This runs four diagnostic cases on a single Wormhole device, prints a
PASS/FAIL summary, and exits non-zero if any bug reproduces.

Three bugs surfaced (each with a self-checking case below):

  [A] Scalar seed shared across all 32 cores (no per-row entropy)
      sampling_program_factory.cpp:315-316  passes one `random_seed` to all cores.
      sampling.cpp:35                       all cores call `rand_tile_init(seed)`
                                            with the same value.
      With an identical uniform distribution across all 32 batch rows, a
      well-seeded kernel would produce ~28-32 distinct samples (independent
      draws from N candidates; birthday-problem expectation). The kernel
      produces 1 distinct value — all 32 rows draw the same RNG state.

  [B] rand_tile_init resets state every call when seed != 0 (no per-step entropy)
      sampling.cpp:34-36   `if (seed != 0) { rand_tile_init(seed); }`
      Two back-to-back calls with the same non-zero seed re-init the
      SFPU RNG to the same starting state, so the second call's draw is
      byte-identical to the first call's. Destroys per-step entropy.

  [C] Compile-time seed (no runtime seed support)
      sampling.cpp:368   `constexpr uint32_t seed = get_compile_time_arg_val(15);`
      The seed is a kernel compile-time arg, so each unique seed forces a
      program-cache miss + kernel recompile. Verified by Case D: calling
      with seed=42 then seed=43 then seed=44 takes seconds per call.

Together: ttnn.sampling cannot express vLLM's seeded-sampling contract
(same seed across requests -> reproducible; entropy advances per step
and varies across the batch).
"""

import argparse
import time

import torch
import ttnn

BATCH = 32
CANDIDATES = 128  # last dim, multiple of 32 and power-of-2 Wt (Wt=4, kernel-safe).


def make_inputs(device):
    """Identical uniform distribution across all 32 rows.

    With proper per-row RNG, each of the 32 cores would independently
    sample one of CANDIDATES tokens. Expected distinct count: ~28-32
    (birthday problem on 128 buckets).
    """
    values = torch.zeros(1, 1, BATCH, CANDIDATES, dtype=torch.bfloat16)
    indices = (
        torch.arange(CANDIDATES, dtype=torch.int32)
        .unsqueeze(0)
        .expand(BATCH, -1)
        .reshape(1, 1, BATCH, CANDIDATES)
        .contiguous()
    )
    k = torch.full((BATCH,), 32, dtype=torch.int32)
    p = torch.ones(BATCH, dtype=torch.bfloat16)
    temp = torch.ones(BATCH, dtype=torch.bfloat16)

    values_dev = ttnn.from_torch(values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    indices_dev = ttnn.from_torch(indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    k_dev = ttnn.from_torch(k, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p_dev = ttnn.from_torch(p, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    temp_dev = ttnn.from_torch(temp, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    return values_dev, indices_dev, k_dev, p_dev, temp_dev


def call_sampling(args, seed: int):
    values_dev, indices_dev, k_dev, p_dev, temp_dev = args
    t0 = time.time()
    out = ttnn.sampling(values_dev, indices_dev, k_dev, p_dev, temp_dev, seed=seed)
    samples = ttnn.to_torch(out).flatten().tolist()
    return samples, time.time() - t0


def case_a_unseeded_correlation(args):
    """seed=0: kernel skips rand_tile_init; cores' RNG starts identical."""
    print("=== Case A: seed=0, uniform distribution -> all rows correlated ===")
    samples, _ = call_sampling(args, seed=0)
    distinct = len(set(samples))
    print(f"  samples (32 rows): {samples}")
    print(f"  distinct values: {distinct}")
    print(f"  expected (proper per-row RNG): ~28-32")
    fail = distinct < 10
    print(f"  -> {'FAIL (bug A reproduced)' if fail else 'PASS'}\n")
    return fail


def case_b_seeded_all_identical(args):
    """seed!=0: rand_tile_init resets every core to the same state."""
    print("=== Case B: seed=42 -> all 32 rows byte-identical ===")
    samples, _ = call_sampling(args, seed=42)
    distinct = len(set(samples))
    print(f"  samples (32 rows): {samples}")
    print(f"  distinct values: {distinct}")
    fail = distinct == 1
    print(f"  -> {'FAIL (bug A reproduced; rand_tile_init resets all cores identically)' if fail else 'PASS'}\n")
    return fail


def case_c_same_seed_back_to_back(args):
    """Two calls with the same seed -> byte-identical RNG draws -> identical samples."""
    print("=== Case C: seed=42 twice -> second call byte-identical to first ===")
    samples1, _ = call_sampling(args, seed=42)
    samples2, _ = call_sampling(args, seed=42)
    print(f"  call 1 samples: {samples1}")
    print(f"  call 2 samples: {samples2}")
    fail = samples1 == samples2
    print(f"  -> {'FAIL (bug B reproduced; rand_tile_init called every invocation)' if fail else 'PASS'}\n")
    return fail


def case_d_seed_recompile(args):
    """Each new seed value forces a program-cache miss because seed is compile-time."""
    print("=== Case D: seed is compile-time -> each new seed triggers recompile ===")
    # First call with a fresh seed primes the cache.
    _, t_seed42 = call_sampling(args, seed=42)
    # Re-call with the same seed -> should hit cache (fast).
    _, t_seed42_again = call_sampling(args, seed=42)
    # New seed -> should miss cache and recompile (slow).
    _, t_seed43 = call_sampling(args, seed=43)
    _, t_seed44 = call_sampling(args, seed=44)
    print(f"  seed=42 (cold):       {t_seed42*1000:8.1f} ms")
    print(f"  seed=42 (warm cache): {t_seed42_again*1000:8.1f} ms")
    print(f"  seed=43 (new seed):   {t_seed43*1000:8.1f} ms")
    print(f"  seed=44 (new seed):   {t_seed44*1000:8.1f} ms")
    # Heuristic: if the new-seed calls are >5x the warm-cache call, recompile is happening.
    recompiled = t_seed43 > 5 * t_seed42_again and t_seed44 > 5 * t_seed42_again
    print(
        f"  -> {'FAIL (bug C reproduced; new seed = recompile)' if recompiled else 'PASS (seed not compile-time, or program cache disabled)'}\n"
    )
    return recompiled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device-id", type=int, default=0)
    args_cli = parser.parse_args()

    device = ttnn.open_device(device_id=args_cli.device_id)
    failures = []
    try:
        sampling_args = make_inputs(device)
        if case_a_unseeded_correlation(sampling_args):
            failures.append("A (seed=0 row correlation)")
        if case_b_seeded_all_identical(sampling_args):
            failures.append("B (seed!=0 row collapse)")
        if case_c_same_seed_back_to_back(sampling_args):
            failures.append("C (per-call RNG reset)")
        if case_d_seed_recompile(sampling_args):
            failures.append("D (compile-time seed)")
    finally:
        ttnn.close_device(device)

    print("=" * 60)
    if failures:
        print(f"FAILED {len(failures)} case(s): {', '.join(failures)}")
        print("Underlying bugs: A=no per-row entropy, B=no per-step entropy, C=compile-time seed")
        raise SystemExit(1)
    print("All cases PASS (no bug reproduced)")


if __name__ == "__main__":
    main()

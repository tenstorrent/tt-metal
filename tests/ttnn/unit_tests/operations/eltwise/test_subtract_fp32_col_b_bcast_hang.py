# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-device reproducer for a `ttnn.subtract` hang.

`ttnn.subtract(bf16_lhs, bf16_rhs, dtype=ttnn.float32)` deadlocks on Blackhole
when the RHS has W=1 (COL_B broadcast) and the LHS innermost dim is W_tiles*32
for W_tiles ∈ {3, 5, 7, ≥8}. W_tiles ∈ {1, 2, 4, 6} pass.

Triggered in production by `vocab_parallel_cross_entropy_loss` (TinyLlama TP=4,
V=32000 → V_per_shard=8000 → W_tiles_per_shard=250). The C++ multi-device
reproducer at `tt-train/tests/ops/distributed/subtract_fp32_col_b_bcast_test.cpp`
was simplified down to this single-chip version once we confirmed the bug had
nothing to do with mesh sharding — it's purely a kernel / program-factory bug.

Required conditions for the hang:
  1. BF16 inputs
  2. FP32 output dtype override (triggers binary_ng's auto-injected
     TYPECAST(BF16, FP32) in the post-activations chain)
  3. RHS has innermost dim = 1 (COL_B broadcast)
  4. W_tiles on LHS innermost dim ∈ {3, 5, 7, ≥8}

Diagnostic signature on hang: `ttnn.subtract` returns (dispatch enqueued), the
subsequent `ttnn.synchronize_device` blocks forever.

Run e.g.:
    pytest -s --timeout=60 \\
        tests/ttnn/unit_tests/operations/eltwise/test_subtract_fp32_col_b_bcast_hang.py
"""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


TILE_W = 32

# 1-2 pass, 3 hangs, 4 passes, 5 hangs, 6 passes, 7 hangs, 8 hangs (and ≥8).
# Default sweep covers the full non-monotonic pattern; use `-k W_3` etc. to
# probe a single value.
W_TILES_SWEEP = [1, 2, 3, 4, 5, 6, 7, 8]

# Subset known to hang. Useful for `-k hangs` to only run the bad cases.
W_TILES_HANG = [3, 5, 7, 8]

# Subset known to pass.
W_TILES_PASS = [1, 2, 4, 6]


def _run_subtract_fp32_col_b_bcast(device, w_tiles, *, b=5, s=256):
    """Run the canonical bf16 - bf16 → fp32 + COL_B-bcast subtract.

    lhs: [B, 1, S, W_tiles*32] bfloat16, all ones
    rhs: [B, 1, S, 1] bfloat16, all ones
    out: [B, 1, S, W_tiles*32] float32, expected all zeros

    Returns the to_torch'd output (a real fp32 read implies the op completed).
    """
    v = w_tiles * TILE_W

    lhs_torch = torch.ones(b, 1, s, v, dtype=torch.float32)
    rhs_torch = torch.ones(b, 1, s, 1, dtype=torch.float32)

    lhs = ttnn.from_torch(lhs_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rhs = ttnn.from_torch(rhs_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    print(
        f"[repro] sync BEFORE before subtract (W_tiles={w_tiles}, V={v}, B={b}, S={s})",
        flush=True,
    )
    ttnn.synchronize_device(device)
    print("[repro] sync AFTER  before subtract", flush=True)

    out = ttnn.subtract(lhs, rhs, dtype=ttnn.float32)
    print(f"[repro] dispatch returned (W_tiles={w_tiles})", flush=True)

    print("[repro] sync BEFORE after  subtract  ← hangs at W∈{3,5,7,≥8}", flush=True)
    ttnn.synchronize_device(device)
    print("[repro] sync AFTER  after  subtract", flush=True)

    assert out.dtype == ttnn.float32, f"expected fp32 output, got {out.dtype}"
    return ttnn.to_torch(out)


@pytest.mark.parametrize("w_tiles", W_TILES_SWEEP, ids=lambda w: f"W_{w}")
def test_subtract_fp32_col_b_bcast_sweep(device, w_tiles):
    """Full W sweep. Cases 3, 5, 7, 8 hang on device — use `--timeout=60` to
    bound them. Cases 1, 2, 4, 6 pass and produce all-zero output."""
    out = _run_subtract_fp32_col_b_bcast(device, w_tiles)
    # Bug is a hang, not a wrong-result; the assertion below is just a
    # correctness sanity check for the passing W values.
    assert torch.equal(
        out, torch.zeros_like(out)
    ), f"W_tiles={w_tiles}: expected zeros, got nonzero values (max abs = {out.abs().max()})"


def test_subtract_fp32_col_b_bcast_smallest_hang(device):
    """Smallest known hanging configuration (W_tiles=3 → V=96 columns).

    This is the canonical one-test bug repro. Run with a timeout:
        pytest -s --timeout=60 -k smallest_hang ...
    Expected outcome on broken builds: pytest times out after 60s.
    """
    _run_subtract_fp32_col_b_bcast(device, w_tiles=3)


# ─── Controls (no dtype mismatch) ───────────────────────────────────────────
# These two tests use the EXACT same shape (W=3, B=5, S=256) and broadcast
# pattern as the canonical hanging test above, but with matching input/output
# dtypes (no narrow→wide conversion). They exist to answer:
#
#   "Does the bug live in the col_bcast kernel itself for these shapes, or
#    specifically in the dtype-mismatch packer-reconfig path?"
#
# Run with profile_this.py to see which compute kernel the program factory
# picks for each. Compare against the BF16→FP32 case (which picks
# `eltwise_binary_col_bcast.cpp` with the post-bcast pack_reconfig).


def test_subtract_bf16_in_bf16_out_W3(device):
    """Control: BF16 inputs, BF16 output, W_tiles=3.

    Expected: same compute kernel path as the hanging BF16→FP32 case
    (`eltwise_binary_col_bcast.cpp`, is_sfpu=false), but no FP32 packer
    reconfig. If this PASSES at W=3, the kernel itself is fine for that shape
    — the bug is specifically the BF16→FP32 reconfig.
    """
    b, s, w_tiles = 5, 256, 3
    v = w_tiles * TILE_W

    lhs_torch = torch.ones(b, 1, s, v, dtype=torch.float32)
    rhs_torch = torch.ones(b, 1, s, 1, dtype=torch.float32)

    lhs = ttnn.from_torch(lhs_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rhs = ttnn.from_torch(rhs_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    print(f"[ctrl bf16→bf16] W_tiles={w_tiles}, V={v}", flush=True)
    ttnn.synchronize_device(device)
    out = ttnn.subtract(lhs, rhs)  # no dtype override
    print(f"[ctrl bf16→bf16] dispatch returned", flush=True)
    ttnn.synchronize_device(device)
    print(f"[ctrl bf16→bf16] post-op sync done", flush=True)

    assert out.dtype == ttnn.bfloat16, f"expected bf16 output, got {out.dtype}"
    out_torch = ttnn.to_torch(out)
    assert torch.equal(out_torch, torch.zeros_like(out_torch))


def test_subtract_fp32_in_fp32_out_W3(device):
    """Control: FP32 inputs, FP32 output, W_tiles=3.

    Expected: may pick a different compute kernel (likely the SFPU variant
    `eltwise_binary_sfpu_col_bcast.cpp`, since Blackhole's FPU doesn't
    natively support FP32 binary ops). If this PASSES at W=3, it tells us
    the SFPU path handles W=3 fine — narrowing the bug further to the FPU
    compute kernel's interaction with the FP32 output packer reconfig.
    """
    b, s, w_tiles = 5, 256, 3
    v = w_tiles * TILE_W

    lhs_torch = torch.ones(b, 1, s, v, dtype=torch.float32)
    rhs_torch = torch.ones(b, 1, s, 1, dtype=torch.float32)

    lhs = ttnn.from_torch(lhs_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    rhs = ttnn.from_torch(rhs_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    print(f"[ctrl fp32→fp32] W_tiles={w_tiles}, V={v}", flush=True)
    ttnn.synchronize_device(device)
    out = ttnn.subtract(lhs, rhs)  # output dtype follows input
    print(f"[ctrl fp32→fp32] dispatch returned", flush=True)
    ttnn.synchronize_device(device)
    print(f"[ctrl fp32→fp32] post-op sync done", flush=True)

    assert out.dtype == ttnn.float32, f"expected fp32 output, got {out.dtype}"
    out_torch = ttnn.to_torch(out)
    assert torch.equal(out_torch, torch.zeros_like(out_torch))

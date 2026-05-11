# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Blackhole diagnostic harness for the fft_inner cross-core path.
#
# WHEN TO RUN
#   * Whenever ``test_fft_blackhole_fp32_pow2`` fails on Blackhole at
#     N >= 8192 — this script tells you (a) which N first goes wrong,
#     (b) which output region (and therefore which partner-exchange
#     stage) is corrupted, and (c) the per-core error spectrum.
#
# HOW IT HELPS
#   The cross-core stages in fft_reader.cpp send tiles between partner
#   cores via NoC + semaphore. Each stage k contributes a different
#   "stride" in the output indexing. By looking at WHERE the error
#   concentrates, you can pinpoint WHICH cross-core stage the patch
#   broke (or the original kernel never got right on BH).
#
# OUTPUT LAYOUT (illustrative)
#   N=8192 P=8 LOG2P=3   rel=8.2e-04          ← clean
#   N=16384 P=16 LOG2P=4 rel=1.6e-01          ← BREAKS HERE
#       core 0  rel=0.05    core 8  rel=0.31
#       core 1  rel=0.06    core 9  rel=0.30
#       ...
#       (uniform error across cores → late stage)
#       (one cluster of cores bad  → specific partner exchange)
#
# This is purely a diagnostic — it doesn't gate CI. Run with -s to see
# the printed table.

import pytest
import torch
import ttnn


def _is_blackhole(device):
    arch = getattr(device, "arch", None)
    if callable(arch):
        arch = arch()
    return str(arch).lower().endswith("blackhole")


def _rel(a, b):
    """L2 relative error, with a 1e-12 floor to avoid 0/0."""
    return (torch.linalg.norm(a - b)
            / torch.linalg.norm(b).clamp_min(1e-12)).item()


def _run_fft(device, N):
    torch_in = torch.randn(N, dtype=torch.float32)
    tt_in = ttnn.from_torch(
        torch_in, dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )
    re, im = ttnn.experimental.fft(tt_in)
    got = torch.complex(
        ttnn.to_torch(re).reshape(-1).to(torch.float32),
        ttnn.to_torch(im).reshape(-1).to(torch.float32),
    )
    ref = torch.fft.fft(torch_in.to(torch.complex64))
    return got, ref


# ── Bisect N to find the first failing size ─────────────────────────────────
@pytest.mark.parametrize("N", [1024, 2048, 4096, 8192, 12288, 16384, 32768, 65536])
def test_diag_bisect_n(device, N):
    """Run a sweep of pow2 + non-pow2 N values. Print rel err per N.
    Skips assertion — run with -s to read the output table."""
    if not _is_blackhole(device):
        pytest.skip("Blackhole-only diagnostic")

    P = max(1, N // 1024)
    log2p = max(0, (P - 1).bit_length())  # log2 round-up
    got, ref = _run_fft(device, N)
    rel = _rel(got, ref)
    print(f"\n[diag] N={N:>6}  P={P:>3}  LOG2P={log2p}  rel={rel:.2e}",
          flush=True)


# ── Per-core error fingerprint (only meaningful for P > 1) ──────────────────
@pytest.mark.parametrize("N", [4096, 8192, 16384, 32768])
def test_diag_per_core_error(device, N):
    """Split the output into P chunks of N/P samples and print rel err
    per chunk. Pattern interpretation:
       * uniform errors across all chunks  → late cross-core stage bug
       * one cluster of chunks bad         → specific partner pair bug
       * even-indexed cores bad only       → bit-0 partner exchange (stage 0)
       * cores in {0,1,2,3} bad            → bit-2 partner exchange (stage 2)
    """
    if not _is_blackhole(device):
        pytest.skip("Blackhole-only diagnostic")
    if N <= 1024:
        pytest.skip(f"N={N} uses P=1, no cross-core stages to bisect")

    P = N // 1024
    chunk = N // P
    got, ref = _run_fft(device, N)

    print(f"\n[diag] N={N} P={P}  per-core error breakdown:")
    for c in range(P):
        sl = slice(c * chunk, (c + 1) * chunk)
        rel = _rel(got[sl], ref[sl])
        bar = "#" * min(int(rel * 50), 50)
        print(f"  core {c:>2}  rel={rel:.2e}  {bar}", flush=True)


# ── Repeatability: same input twice — should be bit-identical on BH ─────────
@pytest.mark.parametrize("N", [16384, 32768])
def test_diag_repeatability(device, N):
    """Run the same FFT twice with identical input. If results differ,
    the kernel has a non-determinism bug (race-on-data, not just
    race-on-correctness). Critical signal — non-determinism narrows the
    fix dramatically."""
    if not _is_blackhole(device):
        pytest.skip("Blackhole-only diagnostic")

    torch.manual_seed(42)
    torch_in = torch.randn(N, dtype=torch.float32)
    tt_in = ttnn.from_torch(
        torch_in, dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
    )

    def one_run():
        re, im = ttnn.experimental.fft(tt_in)
        return torch.complex(
            ttnn.to_torch(re).reshape(-1).to(torch.float32),
            ttnn.to_torch(im).reshape(-1).to(torch.float32),
        )

    a = one_run()
    b = one_run()
    delta = _rel(a, b)
    print(f"\n[diag] N={N} run-to-run delta = {delta:.2e}", flush=True)
    if delta > 1e-7:
        pytest.fail(
            f"BH non-determinism at N={N}: two runs differ by {delta:.2e}. "
            "This indicates a NoC/semaphore race in fft_reader.cpp's "
            "cross-core block (data corruption depends on timing, not "
            "input). Increase the BH-only barriers / cache-invalidates."
        )

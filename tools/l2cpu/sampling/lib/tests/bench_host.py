# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Host time per row of the reference library at the Qwen3 vocabulary size (V = 151936)."""
import os
import sys
import time


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from replay import synthetic_rows  # noqa: E402
from x280s_ref import X280S  # noqa: E402


def main():
    lib = X280S()
    n = 64
    for bf16 in (False, True):
        rows = synthetic_rows(n, bf16=bf16, seed=7)
        name = "bf16" if bf16 else "f32 "
        for label, (T, k, p) in (
            ("greedy", (0.0, 0, 1.0)),
            ("k=50 ", (0.7, 50, 0.9)),
            ("k=20 ", (0.6, 20, 0.95)),
            ("k=0  ", (1.0, 0, 1.0)),
        ):
            reps = 5
            best = 1e9
            for _ in range(reps):
                t0 = time.perf_counter()
                for i in range(n):
                    lib.sample(rows[i], T, k, p, 1, step=i)
                best = min(best, (time.perf_counter() - t0) / n)
            print("%s %s: %7.1f us/row (best of %d x %d rows, incl. ~2 us ctypes)" % (name, label, best * 1e6, reps, n))


if __name__ == "__main__":
    main()

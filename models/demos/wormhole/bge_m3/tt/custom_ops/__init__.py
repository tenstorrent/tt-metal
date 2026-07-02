# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 custom (generic_op) micro-ops.

This package hosts model-specific fused/tweaked ops that have *not* yet been
graduated into the production attention/MLP modules. Each sub-package is
self-contained and matches the layout:

    custom_ops/<op_name>/
        op.py              — Python wrapper around `ttnn.generic_op`
        kernels/           — .cpp kernels invoked from op.py
        __init__.py        — public surface

Ops here are only imported by tests under `tests/perf/sweep_*.py`. They are
*not* wired into `attention.py`, `mlp.py`, or `model.py` until a sweep
demonstrates a real device-time win and PCC stays ≥ 0.9999.

See `models/demos/wormhole/bge_m3/tests/perf/BGE_M3_B1S512_OPTIMIZATION_PLAN.md`
(in the upstream tree) and the sweep files in `tests/perf/` for the
methodology.
"""

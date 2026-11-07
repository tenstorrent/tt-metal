# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""HPCG model-local ops built with ``ttnn.generic_op``.

    custom_ops/<op_name>/
        op.py              - Python wrapper around `ttnn.generic_op`
        kernels/           - .cpp kernels that op.py runs
        __init__.py        - public surface

  - ``sparse_rows``: the persistent sparse-row launch. Symmetric Gauss-Seidel (16 colour phases in one launch),
    the residual and CG SpMV, and the multigrid restriction and prolongation gathers. ``plan.py`` decides which
    rows each core owns and which x window pieces it holds.
  - ``cg_vector``: the fused CG vector work. Dot products (per-core partials), p = z + beta p, and
    x += alpha p with r -= alpha Ap and the partial r.r in one pass; DotFinishOp adds the partials.
  - ``stencil_matrix``: setup. The device writes a level's matrix values, column window offsets and b.

Building the program from Python means a change to a descriptor or a kernel needs no ``_ttnn.so`` rebuild.
"""

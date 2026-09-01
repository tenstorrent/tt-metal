# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 model-local ops built with ``ttnn.generic_op``.

The data-parallel serving path imports these ops from ``attention.py``. Treat
them as production code.

    custom_ops/<op_name>/
        op.py              - Python wrapper around `ttnn.generic_op`
        kernels/           - .cpp kernels that op.py runs
        __init__.py        - public surface

  - ``encoder_sdpa``: the encoder SDPA. Reads BF4 K and V, masks from compact
    valid lengths, and writes the concat-head order.
  - ``fused_qkv_heads``: splits a fused QKV tensor into Q, K, and V heads.
  - ``fused_concat_heads``: concatenates attention heads.
  - ``qkv_scatter_matmul``: fuses the QKV projection, the head split, and the
    BF4 conversion into one program. The serving path calls this one.

Building the program from Python means a change to a descriptor or a kernel
needs no ``_ttnn.so`` rebuild.
"""

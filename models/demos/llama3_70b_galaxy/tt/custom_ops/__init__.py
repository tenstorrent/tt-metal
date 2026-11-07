# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Llama3-70B Galaxy (Qwen3-32B) model-local ops built with ``ttnn.generic_op``.

    custom_ops/<op_name>/
        op.py              - Python wrapper around `ttnn.generic_op`
        kernels/           - .cpp kernels that op.py runs
        __init__.py        - public surface

  - ``qwen_q_prep``: Qwen long-prefill Q prep (bf8 -> bf16, q_norm, RoPE, bf16 -> bf8) on the
    mesh column's two ring-SDPA slices, written in place into the bf8 Q buffer.
  - ``qwen_width_norm``: Qwen context-parallel local RMSNorm of the [S/32, 5120] residual, the width split over the
    grid columns with a per-row partial-sum exchange, bf8 out.
  - ``qwen_cp_sdpa``: Qwen context-parallel causal SDPA (streaming compute): both ring slices of a mesh column in one
    program on all grid cores, bit-identical to the two stock chunked SDPA calls; plus ``chunked_sdpa_port``, a
    Python port of the stock chunked SDPA program (the bit-identity reference).

Building the program from Python means a change to a descriptor or a kernel needs no
``_ttnn.so`` rebuild.
"""

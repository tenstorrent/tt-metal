# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Vendored Sol-Attn reference (NVIDIA Sol-H3), as the algorithm of record.

``preprocess.py``, ``fwd.py`` and ``interface.py`` come from github.com/NVlabs/Sana,
branch ``sol-engine`` at ``8e0db4fa562d`` (2026-09-11), path
``models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn/``. They are
semantically unmodified; the repo's black/isort pass reformatted them, so diff
against that commit for byte-exact upstream. The release
is Apache-2.0; these three files are Sol's own and carry none of the
FlashAttention / cuDNN-frontend provenance its ``THIRD_PARTY_NOTICES.md``
attaches to other parts of that tree.

**These files do not import in this environment and are not meant to.** They are
Triton kernels -- ``@triton.jit`` decorates at import and the kernels need a CUDA
device, and the tt-metal python env has neither. They are here to be read and
cited, not run.

The executable oracle is ``sol_attn_oracle.py`` one directory up: a pure-torch
transcription of these kernels. Its fidelity rests on reading plus the
invariants in ``test_sol_attn_oracle_minimax_h3.py`` -- most usefully the
``tau -> -inf`` case, where every block routes exact and the result must equal
dense attention. That is the same degenerate setting Sol's own correctness gate
uses (``tau=-1000.0`` in ``h3_runtime/sparse_attention.py``).
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts of the Kimi-K3.1 device Blocks' readbacks (``blocks.py``): the MoE routing readback (m06n.11.4) and
the DSA selection readback (m06n.11.5). CPU only.

Each block's tests, including their imports, stay inside their own section below.
"""

from __future__ import annotations

# ==== MoE (m06n.11.4) ===============================================================================================
# ==== end of MoE ====================================================================================================

# ==== DSA (m06n.11.5) ===============================================================================================
# isort: off
import torch

from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.blocks import device_selection
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.cases import DSA_CASES, DSA_DIMS

# isort: on


def test_dsa_selection_readback_marks_the_no_key_sentinel() -> None:
    """The indexer's uint32 no-key sentinel (read back as 0xFFFFFFFF) becomes -1; real keys stay int64."""
    got = device_selection(torch.tensor([[0xFFFFFFFF, 3], [0, 2**31]], dtype=torch.int64))
    assert got.dtype == torch.int64 and got.tolist() == [[-1, 3], [0, 2**31]]


def test_dsa_case_runs_whole_chunks_past_top_k() -> None:
    """Every DSA context length is a multiple of the chunk (no zero-padded chunk) and spans rows past top-k (every
    window has scored rows)."""
    (case,) = DSA_CASES
    assert all(rows % case.chunk == 0 and rows > DSA_DIMS.topk for rows in case.context_lengths), case.context_lengths


# ==== end of DSA ====================================================================================================

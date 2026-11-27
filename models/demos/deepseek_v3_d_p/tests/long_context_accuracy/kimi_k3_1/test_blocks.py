# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts of the Kimi-K3.1 device Blocks' readbacks (``blocks.py``): the MoE routing readback (m06n.11.4) and
the DSA selection readback (m06n.11.5). CPU only.

Each block's tests, including their imports, stay inside their own section below.
"""

from __future__ import annotations

# ==== MoE (m06n.11.4) ===============================================================================================
# isort: split
import json
from dataclasses import dataclass

from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    MANIFEST,
    ROOT_ENV,
    GoldenMismatch,
    golden_root,
    open_golden,
)
from models.demos.common.prefill.long_context_accuracy.run import run_case
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.blocks import KimiK31MoeBlock, owner_positions
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.cases import MOE_CASES


def test_owner_positions_hand_case() -> None:
    """6 real rows a SP row plus 2 padding rows, 2 row blocks of 4, TP 2: column 0 owns block rows 0-1 of each block
    (chunk rows 6 r + [0, 1, 4, 5]); column 1 owns 2-3 of each, but rows 6-7 are padding, so only [2, 3]."""
    owned, pos = owner_positions(1, 0, rows_per_sp_row=6, pad_rows=2, row_blocks=2, tp=2)
    assert owned.tolist() == [0, 1, 2, 3] and pos.tolist() == [6, 7, 10, 11]
    owned, pos = owner_positions(1, 1, rows_per_sp_row=6, pad_rows=2, row_blocks=2, tp=2)
    assert owned.tolist() == [0, 1] and pos.tolist() == [8, 9]


def test_moe_cases_load_the_weights_their_golden_was_built_from() -> None:
    """Each MoE case's checkpoint weights (the layer's digest) and products are its registered golden's. Reads the
    checkpoint (skips without it) and the golden's manifest."""
    for case in MOE_CASES:
        golden = open_golden(case.golden_id, reader=case.golden_reader)
        prepared = case.prepare(golden, case.checkpoint(), case.context_lengths[0])
        golden.require(
            identity=dict(prepared.identity.identity),
            weights_sha256=dict(prepared.identity.weights_sha256),
            products=[*prepared.input_tensors, *prepared.scorer.golden_tensors],
        )


@dataclass
class _Mesh:
    shape: tuple = (4, 4)


def test_a_moe_weight_digest_mismatch_fails_before_device_work(tmp_path, monkeypatch, expect_error) -> None:
    """The layer-1 case against a copy of its golden's manifest whose ``layers.1`` digest differs: ``run_case`` raises
    ``GoldenMismatch`` naming it and never builds the device Block."""
    (case,) = [c for c in MOE_CASES if c.id.endswith("layer1")]
    checkpoint = case.checkpoint()
    manifest = json.loads((golden_root() / case.golden_id / MANIFEST).read_text())
    manifest["identity"]["weights_sha256"]["layers.1"] = "0" * 64
    (tmp_path / case.golden_id).mkdir()
    (tmp_path / case.golden_id / MANIFEST).write_text(json.dumps(manifest))
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    built = []
    monkeypatch.setattr(KimiK31MoeBlock, "build", lambda self, *args: built.append(args))
    with expect_error(GoldenMismatch, "weights layers.1"):
        run_case(case, case.context_lengths[0], _Mesh(), checkpoint)
    assert built == []


# ==== end of MoE ====================================================================================================

# ==== DSA (m06n.11.5) ===============================================================================================
# isort: off
import math

import torch

from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.blocks import device_selection
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.cases import (
    DSA_CASES,
    DSA_DIMS,
    dsa_context_bars,
)

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


def test_dsa_unset_bar_is_refused(expect_error) -> None:
    """A NaN bar would pass every window (no comparison with NaN fails), so a context length without values is
    refused."""
    with expect_error(ValueError, "no bar value"):
        dsa_context_bars({"attn_nrmse": math.nan})


# ==== end of DSA ====================================================================================================

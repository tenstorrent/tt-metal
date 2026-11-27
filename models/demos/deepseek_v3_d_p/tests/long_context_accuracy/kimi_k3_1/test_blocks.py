# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts of the Kimi-K3.1 cases and Blocks that a device run would not show: a MoE case on a golden of other
weights is refused before device work, the DSA readback marks the indexer's no-key sentinel, and an unset DSA bar is
refused. CPU only."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch

from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    MANIFEST,
    ROOT_ENV,
    GoldenMismatch,
    golden_root,
)
from models.demos.common.prefill.long_context_accuracy.run import run_case
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.blocks import KimiK31MoeBlock, device_selection
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.cases import MOE_CASES, dsa_context_bars


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


def test_dsa_selection_readback_marks_the_no_key_sentinel() -> None:
    """The indexer's uint32 no-key sentinel (read back as 0xFFFFFFFF) becomes -1; real keys stay int64."""
    got = device_selection(torch.tensor([[0xFFFFFFFF, 3], [0, 2**31]], dtype=torch.int64))
    assert got.dtype == torch.int64 and got.tolist() == [[-1, 3], [0, 2**31]]


def test_dsa_unset_bar_is_refused(expect_error) -> None:
    """A NaN bar would pass every window (no comparison with NaN fails), so a context length without values is
    refused."""
    with expect_error(ValueError, "no bar value"):
        dsa_context_bars({"attn_nrmse": math.nan})

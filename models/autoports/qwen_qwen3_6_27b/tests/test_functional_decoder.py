# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CI smoke uses real config dimensions and deterministic weight statistics.

Run serialized with no other device owner. The standalone evidence runners add
full context, resume controls, real weights, allocation tracking and watcher.
"""
import sys

import pytest

from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import main


@pytest.mark.parametrize("layer", [0, 3])
@pytest.mark.parametrize("batch", [1, 2])
def test_synthetic_decoder(layer, batch, monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_decoder",
            "--synthetic",
            "--layer",
            str(layer),
            "--batch",
            str(batch),
            "--length",
            "33",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    main()

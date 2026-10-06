# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline checkpoint-free regression with real config and tensor statistics."""

import pytest

from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import main


@pytest.mark.parametrize("layer", [0, 5], ids=["sliding_attention", "full_attention"])
def test_statistical_weights(layer, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_decoder",
            "--layer",
            str(layer),
            "--length",
            "33",
            "--decode",
            "--steps",
            "4",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    main()

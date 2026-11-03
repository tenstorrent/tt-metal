# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized-path correctness on the pinned checkpoint and HF activations."""

import pytest

from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.validate_optimized_decoder import validate


@pytest.mark.parametrize("layer", [0, 3], ids=["dense", "mova_moe"])
@pytest.mark.parametrize("batch", [1, 2])
def test_optimized_decoder(layer, batch):
    report = validate(layer=layer, batch=batch, lengths=[31, 32, 33, 257], synthetic=False, tag="pytest")
    assert (
        report["decoder_class"] == "models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.optimized_decoder.OptimizedDecoder"
    )

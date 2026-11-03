# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Default tests synthesize real-shaped weights from committed tensor statistics.

Only the small pinned HF config/model code must be present in HF's cache; these
tests never load the safetensors checkpoint. Real-weight/long-context commands
are separate evidence runners documented in the stage README.
"""

import pytest

from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.validate_decoder import validate


@pytest.mark.parametrize("layer", [0, 3], ids=["dense", "mova_moe"])
def test_functional_decoder(layer):
    validate(layer=layer, batch=2, lengths=[31, 32, 33, 257], synthetic=True)

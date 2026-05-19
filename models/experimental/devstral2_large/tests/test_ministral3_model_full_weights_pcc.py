# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Deprecated entry point.

Partial-depth Hub-weight PCC is in ``test_ministral3_model.py`` (prefill + decode).
Full-checkpoint validation should run via the demo
(``models/experimental/devstral2_large/reference/devstral2_123b_inference.py``).
"""

import pytest


def test_replaced_by_random_weight_pcc() -> None:
    pytest.skip(
        "Replaced by tests/test_ministral3_model.py::"
        "test_ministral3_model_pcc_devstral2_large_partial_weights_one_layer_prefill and "
        "test_ministral3_model_pcc_devstral2_large_partial_weights_one_layer_decode"
    )

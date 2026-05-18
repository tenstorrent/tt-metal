# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Deprecated entry point.

End-to-end PCC for the Devstral-2 / Ministral3 model is now covered by
``test_ministral3_model.py::test_full_model_prefill_pcc_random_weights``. This file is kept as a
stub so pytest discovery + CI links don't break.
"""

import pytest


def test_replaced_by_test_ministral3_model() -> None:
    pytest.skip("Replaced by tests/test_ministral3_model.py::test_full_model_prefill_pcc_random_weights")

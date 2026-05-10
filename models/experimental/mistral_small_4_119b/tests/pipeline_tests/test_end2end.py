# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Full multimodal generation (vision + dense ``Transformer``) is not enabled for Small 4 yet:
the language model is Mistral4 (MLA + MoE), not the path used by ``mistral_24b`` E2E tests.

When TTNN Mistral4 support lands, copy/adapt
``models/experimental/mistral_24b/tests/pipeline_tests/test_end2end.py`` here and drop the skip.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "Mistral Small 4 119B text stack (Mistral4) not in TTNN; use test_vision_tower / "
        "test_vision_model for device PCC. See models/experimental/mistral_24b/tests/pipeline_tests/test_end2end.py."
    )
)


def test_full_multimodal_e2e_placeholder():
    """Reserved: wire Generator + Mistral4 when the text stack is implemented."""

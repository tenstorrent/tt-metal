# SPDX-License-Identifier: Apache-2.0
"""HF gate for the optional native RMS-based L2 normalization rewrite."""

from unittest.mock import patch

import pytest

from models.autoports.qwen_qwen3_8_flash_next.tests.fusion_candidates import RMSL2GDN
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedGatedDeltaNet

if __name__ == "__main__":
    with patch.object(FusedGatedDeltaNet, "_l2norm", staticmethod(RMSL2GDN._l2norm)):
        raise SystemExit(
            pytest.main(
                [
                    "-s",
                    "-q",
                    "--timeout=0",
                    "models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py",
                    "-k",
                    "layer0",
                ]
            )
        )

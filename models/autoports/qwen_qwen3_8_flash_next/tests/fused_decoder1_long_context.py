# SPDX-License-Identifier: Apache-2.0
"""Apply the established coupled full-context harness to the fused decoder."""

import sys
from unittest.mock import patch

from models.autoports.qwen_qwen3_8_flash_next.tests import decoder1_long_context as contract
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder

if __name__ == "__main__":
    if "--output" not in sys.argv:
        sys.argv += [
            "--output",
            "models/autoports/qwen_qwen3_8_flash_next/doc/fused_decoder/decoder1_long_context.json",
        ]
    with patch.object(contract, "FunctionalDecoder", FusedDecoder):
        contract.main()

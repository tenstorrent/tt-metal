# SPDX-License-Identifier: Apache-2.0
"""Exercise one experimental rewrite through the real HF decoder contract."""

import sys

import pytest

from models.autoports.qwen_qwen3_8_flash_next.tests.late_fusion_candidates import install

if __name__ == "__main__":
    if sys.argv[1] != "default":
        install(sys.argv[1])
    raise SystemExit(
        pytest.main(
            [
                "-s",
                "-q",
                "--timeout=0",
                "models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py",
                *sys.argv[2:],
            ]
        )
    )

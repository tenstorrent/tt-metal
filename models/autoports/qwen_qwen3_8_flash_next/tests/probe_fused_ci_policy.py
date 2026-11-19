# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Discriminating CI control using the same fixture and unchanged assertions."""

import argparse
from unittest.mock import patch

import pytest

from models.autoports.qwen_qwen3_8_flash_next.tests.run_fused_probe import POLICIES
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=[name for name, policy in POLICIES.items() if policy is not None], required=True
    )
    parser.add_argument("--junitxml", required=True)
    parser.add_argument("--all", action="store_true", help="Run the complete inherited fused synthetic matrix")
    args = parser.parse_args()
    factory = FusedDecoder.from_state_dict

    def selected(*inputs, **kwargs):
        return factory(*inputs, **kwargs, **POLICIES[args.variant])

    with patch.object(FusedDecoder, "from_state_dict", selected):
        result = pytest.main(
            [
                "-q",
                (
                    "models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py"
                    if args.all
                    else "models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py::test_fused_prefill_and_traced_decode[gdn-1]"
                ),
                "--junitxml",
                args.junitxml,
            ]
        )
    raise SystemExit(result)


if __name__ == "__main__":
    main()

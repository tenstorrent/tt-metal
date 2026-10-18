# SPDX-License-Identifier: Apache-2.0
"""Isolate short-continuation normalization and reshape deltas against real HF."""

import argparse
from unittest.mock import patch

import pytest

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedGatedDeltaNet, _precise_recurrence
from models.autoports.qwen_qwen3_8_flash_next.tt.gdn import GatedDeltaNet


def recurrence(self, q, k, v, g, beta):
    b, seq, _ = q.shape
    q = self._l2norm(self._heads(q, b, seq, 16))
    k = self._l2norm(self._heads(k, b, seq, 16))
    v = ttnn.typecast(self._heads(v, b, seq, 48), ttnn.float32)
    return _precise_recurrence(self, q, k, v, g, beta, query_scale=128**-0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["separate_scale", "primitive_l2", "rm_heads"])
    args = parser.parse_args()
    from contextlib import ExitStack

    with ExitStack() as stack:
        if args.mode != "rm_heads":
            stack.enter_context(patch.object(FusedGatedDeltaNet, "_recurrence", recurrence))
        if args.mode == "primitive_l2":
            stack.enter_context(patch.object(FusedGatedDeltaNet, "_l2norm", staticmethod(GatedDeltaNet._l2norm)))
        if args.mode == "rm_heads":
            stack.enter_context(patch.object(FusedGatedDeltaNet, "_heads", staticmethod(GatedDeltaNet._heads)))
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

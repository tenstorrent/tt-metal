# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU admission checks: malformed evidence must never reach mesh creation."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests import probe_multichip_decoder as probe


class ProbeAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.options = SimpleNamespace(
            mode="mesh",
            snapshot=Path("/checkpoint"),
            layer=0,
            length=33,
            capacity=128,
            batch=0,
            tails=False,
            components=False,
            trace=False,
            profile=False,
            stack=False,
            repeats=16,
            clean_timing=False,
            reserve_bytes=0,
            expert_dtype="bfloat4_b",
        )
        values = [torch.zeros(1, 1, n, 10240, dtype=torch.bfloat16) for n in (33, 1, 1, 1)]
        self.reference = dict(
            metadata=probe.reference_metadata(self.options),
            inputs=values,
            outputs=values,
            states=[dict(conv=torch.empty(0), recurrent=torch.empty(0)) for _ in range(4)],
        )

    def test_valid_request(self):
        probe.validate_options(self.options)
        probe.validate_reference(self.options, self.reference)

    def test_invalid_requests(self):
        for changes in (
            dict(layer=2),
            dict(capacity=129),
            dict(length=126),
            dict(batch=33),
            dict(tails=True, length=31),
            dict(components=True, batch=1),
            dict(trace=True, repeats=2),
            dict(mode="baseline", reserve_bytes=2048),
        ):
            with self.subTest(changes=changes):
                options = copy.copy(self.options)
                vars(options).update(changes)
                with self.assertRaises(ValueError):
                    probe.validate_options(options)

    def test_batch_requires_trace_tracking_even_without_trace_flag(self):
        self.options.batch = 1
        with patch.object(probe.trace_allocation_tracker, "TRACE_ALLOC_TRACKING", False):
            with self.assertRaisesRegex(ValueError, "TRACE_ALLOC_TRACKING"):
                probe.validate_options(self.options)

    def test_wrong_layer_reference_rejected(self):
        self.reference["metadata"]["layer"] = 3
        with self.assertRaisesRegex(ValueError, "metadata"):
            probe.validate_reference(self.options, self.reference)

    def test_missing_evidence_rejected(self):
        for feature in ("batch", "tails", "stack", "components"):
            with self.subTest(feature=feature):
                options = copy.copy(self.options)
                setattr(options, feature, 32 if feature == "batch" else True)
                with self.assertRaises(ValueError):
                    probe.validate_reference(options, self.reference)
        self.reference["states"] = []
        with self.assertRaisesRegex(ValueError, "state buffer"):
            probe.validate_reference(self.options, self.reference)

    def test_nonfinite_reference_rejected(self):
        self.reference["outputs"][0][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite BF16"):
            probe.validate_reference(self.options, self.reference)


if __name__ == "__main__":
    unittest.main()

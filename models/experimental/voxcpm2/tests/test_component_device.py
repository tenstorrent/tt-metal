# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-device PCC test. Missing configured oracle/hardware is an explicit skip."""

import argparse
import os
from pathlib import Path
import tempfile
import unittest

from models.experimental.voxcpm2.validation.replay_component import COMPONENTS, replay


class ComponentDeviceTest(unittest.TestCase):
    def test_cuda_reference_component_on_tt(self):
        keys = ("VOXCPM2_CUDA_CAPTURE", "VOXCPM2_CHECKPOINT", "VOXCPM2_DEVICE_ID")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest(
                "Requires native CUDA capture, checkpoint, and reserved TT device"
            )
        component = os.environ.get("VOXCPM2_COMPONENT", "feat_encoder.forward")
        if component not in COMPONENTS:
            self.fail("Unknown VOXCPM2_COMPONENT")
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                reference=Path(os.environ[keys[0]]),
                checkpoint=Path(os.environ[keys[1]]),
                device_id=int(os.environ[keys[2]]),
                component=component,
                event_index=int(os.environ.get("VOXCPM2_EVENT_INDEX", "0")),
                dtype=os.environ.get("VOXCPM2_DTYPE", "bfloat16"),
                min_pcc=float(os.environ.get("VOXCPM2_MIN_PCC", ".99")),
                max_relative_rms=None,
                max_abs=None,
                output=Path(directory) / "report.json",
            )
            self.assertEqual(replay(args), 0)


if __name__ == "__main__":
    unittest.main()

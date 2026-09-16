# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import unittest

from tt_metal.fabric.debug.visualizer.decode.liveness import classify_liveness, decode_lifecycle

HEARTBEAT = {"magic": 0xDCBA0000, "magic_mask": 0xFFFF0000}
ENUMS = {
    "EDMStatus": {
        "READY_FOR_TRAFFIC": 0xA3B3C3D3,
        "TERMINATED": 0xA4B4C4D4,
        "INITIALIZATION_STARTED": 0xB0C0D0E0,
    },
    "TerminationSignal": {"KEEP_RUNNING": 0, "IMMEDIATELY_TERMINATE": 2},
    "RunMsg": {"RUN_MSG_GO": 0x80, "RUN_MSG_DONE": 0},
}


def sample(values, *, status="ok", reset=False):
    return {
        "status": status,
        "health": {"reset_bits": {"erisc0": reset}},
        "liveness": [
            {"t": f"2026-09-15T00:00:0{index}Z", "heartbeat": value}
            for index, value in enumerate(values)
        ],
    }


class LivenessTest(unittest.TestCase):
    def test_advancing_static_and_mixed_base_firmware(self):
        advancing = classify_liveness(sample([0xDCBA0040, 0xDCBA0080, 0xDCBA00C0]), HEARTBEAT)
        self.assertEqual(advancing["classification"], "advancing")
        static = classify_liveness(sample([0xDCBA0040] * 3), HEARTBEAT)
        self.assertEqual(static["classification"], "static")

        mixed = classify_liveness(sample([0xDCBA0040, 0xABCD0001, 0xABCD0002]), HEARTBEAT)
        self.assertEqual(mixed["classification"], "insufficient")
        self.assertEqual(mixed["fabric_samples"], 1)
        self.assertEqual(mixed["base_fw_samples"], 2)

    def test_reset_and_unreadable_take_precedence(self):
        self.assertEqual(
            classify_liveness(sample([0xDCBA0040, 0xDCBA0080], reset=True), HEARTBEAT)["classification"],
            "reset",
        )
        self.assertEqual(
            classify_liveness(sample([0xDCBA0040, 0xDCBA0080], status="unreadable"), HEARTBEAT)[
                "classification"
            ],
            "unknown",
        )

    def test_lifecycle_exit_states(self):
        base = {"edm_status": 0xA3B3C3D3, "termination_signal": 0, "go_signal": 0x80}
        self.assertEqual(
            decode_lifecycle({"lifecycle": base}, ENUMS)["exit_state"],
            "running_or_host_gone",
        )
        teardown = {**base, "termination_signal": 2}
        self.assertEqual(decode_lifecycle({"lifecycle": teardown}, ENUMS)["exit_state"], "teardown_stuck")
        terminated = {**teardown, "edm_status": 0xA4B4C4D4}
        self.assertEqual(decode_lifecycle({"lifecycle": terminated}, ENUMS)["exit_state"], "orderly_exit")
        initializing = {**base, "edm_status": 0xB0C0D0E0}
        self.assertEqual(decode_lifecycle({"lifecycle": initializing}, ENUMS)["exit_state"], "initializing")
        wiped = {**base, "edm_status": 0}
        self.assertEqual(decode_lifecycle({"lifecycle": wiped}, ENUMS)["exit_state"], "wiped_or_never_ran")


if __name__ == "__main__":
    unittest.main()

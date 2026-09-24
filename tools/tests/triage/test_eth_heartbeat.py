# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Test heartbeat checks with device reads and time replaced. No hardware is used."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load_check_eth_status():
    # Import the real checker without the device libraries or the triage session.
    path = Path(__file__).resolve().parents[2] / "triage" / "check_eth_status.py"
    spec = importlib.util.spec_from_file_location("eth_status_for_heartbeat_test", path)
    module = importlib.util.module_from_spec(spec)
    dependencies = {
        "run_checks": SimpleNamespace(run=Mock()),
        "triage": SimpleNamespace(
            ScriptConfig=Mock(), triage_field=lambda *args: None, log_check_location=Mock(), run_script=Mock()
        ),
        "ttexalens": SimpleNamespace(read_word_from_device=Mock()),
        "ttexalens.context": SimpleNamespace(Context=object),
        "ttexalens.device": SimpleNamespace(Device=object, OnChipCoordinate=object),
        "utils": SimpleNamespace(ERROR=Mock()),
        spec.name: module,
    }
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


check_eth_status = load_check_eth_status()


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TestEthHeartbeat(unittest.TestCase):
    def check_samples(self, core_type, samples, read_delay=0.0):
        clock = FakeClock()
        core = core_type(location=object(), context=object())
        reads = []

        def read(location, address, *, context):
            self.assertIs(location, core.location)
            self.assertEqual(address, core.eth_core_definitions.heartbeat)
            self.assertIs(context, core.context)
            value = samples[min(len(reads), len(samples) - 1)]
            reads.append(value)
            clock.now += read_delay
            return value

        with (
            patch.object(check_eth_status, "read_word_from_device", side_effect=read),
            patch.object(check_eth_status, "log_check_location") as log,
            patch.object(check_eth_status, "monotonic", clock.monotonic),
            patch.object(check_eth_status, "sleep", clock.sleep),
        ):
            result = core.check_for_heartbeat()
        return result, reads, log, clock.now

    def test_frozen_samples_fail_after_timeout(self):
        cases = (
            (check_eth_status.WormholeEthCore, 0),
            (check_eth_status.WormholeEthCore, 0xABCD0201),
            (check_eth_status.WormholeEthCore, 0xAABB0201),
            (check_eth_status.BlackholeEthCore, 0),
            (check_eth_status.BlackholeEthCore, 1),
            (check_eth_status.BlackholeEthCore, 0xDCBA0040),
        )
        for core_type, sample in cases:
            with self.subTest(core=core_type.__name__, sample=hex(sample)):
                result, reads, log, elapsed = self.check_samples(core_type, [sample])
                self.assertFalse(result)
                self.assertGreaterEqual(len(reads), 2)
                self.assertAlmostEqual(elapsed, check_eth_status.HEARTBEAT_TIMEOUT_SECONDS)
                log.assert_called_once()
                self.assertEqual(log.call_args.args[1:], (False, "No heartbeat detected"))

    def test_advancing_samples_require_two_reads(self):
        cases = (
            (check_eth_status.WormholeEthCore, [0xABCD0201, 0xABCD0202]),
            (check_eth_status.WormholeEthCore, [0xAABB0201, 0xAABB0202]),
            (check_eth_status.BlackholeEthCore, [0, 1]),
            (check_eth_status.BlackholeEthCore, [0xDCBA0040, 0xDCBA0080]),
        )
        for core_type, samples in cases:
            with self.subTest(core=core_type.__name__, samples=samples):
                result, reads, log, elapsed = self.check_samples(core_type, samples)
                self.assertTrue(result)
                self.assertEqual(reads, samples)
                self.assertGreater(elapsed, 0)
                log.assert_not_called()

    def test_counter_wrap_passes(self):
        cases = (
            (check_eth_status.WormholeEthCore, [0xABCDFFFF, 0xABCD0000]),
            (check_eth_status.WormholeEthCore, [0xAABBFFFF, 0xAABB0000]),
            (check_eth_status.BlackholeEthCore, [0xFFFFFFFF, 0]),
            (check_eth_status.BlackholeEthCore, [0xDCBAFFC0, 0xDCBA0000]),
        )
        for core_type, samples in cases:
            with self.subTest(core=core_type.__name__, samples=samples):
                result, reads, log, _ = self.check_samples(core_type, samples)
                self.assertTrue(result)
                self.assertEqual(reads, samples)
                log.assert_not_called()

    def test_fast_reads_allow_time_for_heartbeat(self):
        samples = [0xABCD0201] * 31 + [0xABCD0202]
        result, reads, log, elapsed = self.check_samples(check_eth_status.WormholeEthCore, samples)
        self.assertTrue(result)
        self.assertEqual(reads, samples)
        self.assertGreaterEqual(elapsed, 0.03)
        log.assert_not_called()

    def test_read_time_counts_toward_timeout(self):
        result, reads, log, elapsed = self.check_samples(
            check_eth_status.WormholeEthCore, [0xABCD0201], read_delay=0.01
        )
        self.assertFalse(result)
        self.assertLessEqual(len(reads), 6)
        self.assertGreaterEqual(elapsed, check_eth_status.HEARTBEAT_TIMEOUT_SECONDS)
        self.assertLess(elapsed, check_eth_status.HEARTBEAT_TIMEOUT_SECONDS + 0.01)
        log.assert_called_once()

    def test_wormhole_zero_to_frozen_value_fails(self):
        result, _, log, _ = self.check_samples(check_eth_status.WormholeEthCore, [0, 0xABCD0201])
        self.assertFalse(result)
        log.assert_called_once()

    def test_wormhole_zero_to_advancing_value_passes(self):
        samples = [0, 0xABCD0201, 0xABCD0202]
        result, reads, log, _ = self.check_samples(check_eth_status.WormholeEthCore, samples)
        self.assertTrue(result)
        self.assertEqual(reads, samples)
        log.assert_not_called()

    def test_invalid_wormhole_signature_fails(self):
        for samples in ([0xDEAD0001, 0xDEAD0002], [0xABCD0201, 0xDEAD0001], [0xDEAD0001, 0xABCD0201]):
            with self.subTest(samples=samples):
                result, _, log, _ = self.check_samples(check_eth_status.WormholeEthCore, samples)
                self.assertFalse(result)
                log.assert_called_once()
                self.assertFalse(log.call_args.args[1])
                self.assertIn("Invalid heartbeat signature", log.call_args.args[2])

    def test_read_error_is_not_accepted_as_heartbeat(self):
        for core_type in (check_eth_status.WormholeEthCore, check_eth_status.BlackholeEthCore):
            with self.subTest(core=core_type.__name__):
                core = core_type(location=object(), context=object())
                with patch.object(check_eth_status, "read_word_from_device", side_effect=OSError("Read failed")):
                    with self.assertRaisesRegex(OSError, "Read failed"):
                        core.check_for_heartbeat()


if __name__ == "__main__":
    unittest.main()

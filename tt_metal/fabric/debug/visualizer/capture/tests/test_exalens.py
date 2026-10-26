# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cases for reading router cores through ttexalens, with fake ttexalens objects.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.capture.tests.test_exalens
"""

import contextlib
import io
import unittest

from tt_metal.fabric.debug.visualizer.capture.exalens import ExalensDevice
from tt_metal.fabric.debug.visualizer.capture.reads import get_read_plan
from tt_metal.fabric.debug.visualizer.capture.snapshot import CaptureError, OpenedRouter
from tt_metal.fabric.debug.visualizer.capture.tests.test_reads import MANIFEST

E1 = get_read_plan(MANIFEST).routers[0]  # translated (25, 16), logical (0, 0), eth_chan 0
CHIP_ID = 3


class FakeCore:
    """A ttexalens OnChipCoordinate on a core of `core_type` at `logical`."""

    def __init__(self, logical, core_type="eth"):
        self.logical = logical
        self.core_type = core_type

    def to(self, output_type):
        if output_type != f"logical-{self.core_type}":
            raise ValueError(f"the core isn't {output_type}")
        return self.logical


class FakeChip:
    def __init__(self, translated=None, eth=(), wormhole=True):
        self.id = CHIP_ID
        self.translated = translated or {}  # translated (x, y) -> FakeCore
        self.eth = list(eth)  # the eth cores, in ttexalens's block order
        self.wormhole = wormhole

    def is_wormhole(self):
        return self.wormhole

    def is_blackhole(self):
        return not self.wormhole

    def get_block_locations(self, block_type):
        assert block_type == "eth"
        return self.eth


class FakeContext:
    def __init__(self, chips):
        self.device_by_unique_id = chips


def coordinate(x, y, input_type, chip):
    assert input_type == "translated"
    if (x, y) not in chip.translated:
        raise KeyError(f"no core at translated ({x}, {y})")
    return chip.translated[(x, y)]


class Reader:
    """A fake read_from_device that records its calls."""

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def __call__(self, location, addr, device_id, num_bytes, context):
        self.calls.append((location, addr, device_id, num_bytes, context))
        if self.error is not None:
            raise self.error
        return bytes(range(num_bytes))


def make_device(chip=None, reader=None) -> ExalensDevice:
    chips = {} if chip is None else {E1.asic_id: chip}
    return ExalensDevice(FakeContext(chips), coordinate, reader or Reader(), "fake 1.0")


class OpenRouter(unittest.TestCase):
    def test_the_translated_core(self):
        core = FakeCore((0, 0))
        opened = make_device(FakeChip(translated={(25, 16): core})).open_router(E1, "wormhole_b0")
        self.assertEqual(opened, OpenedRouter(handle=core, ttexalens_id=CHIP_ID))

    def test_an_unknown_chip(self):
        with self.assertRaisesRegex(CaptureError, "ttexalens has no chip with unique id 0x000000036190302b"):
            make_device().open_router(E1, "wormhole_b0")

    def test_a_chip_of_another_arch(self):
        device = make_device(FakeChip(translated={(25, 16): FakeCore((0, 0))}))
        with self.assertRaisesRegex(CaptureError, "ttexalens's chip 3 isn't blackhole, the manifest's run.arch"):
            device.open_router(E1, "blackhole")
        with self.assertRaisesRegex(CaptureError, "capture can't check that a chip is quasar"):
            device.open_router(E1, "quasar")

    def test_the_eth_core_when_the_translated_core_is_elsewhere(self):
        eth0 = FakeCore((0, 0))
        cases = {
            "is logical (0, 1)": {(25, 16): FakeCore((0, 1))},
            "raises KeyError": {},
            "raises ValueError: the core isn't logical-eth": {(25, 16): FakeCore((1, 1), core_type="tensix")},
        }
        for why, translated in cases.items():
            with self.subTest(why), contextlib.redirect_stderr(io.StringIO()) as stderr:
                opened = make_device(FakeChip(translated=translated, eth=[eth0])).open_router(E1, "wormhole_b0")
                self.assertIs(opened.handle, eth0)
                self.assertIn(f"M0/C0/E1: translated (25, 16) {why}", stderr.getvalue())
                self.assertIn("; reading eth core 0\n", stderr.getvalue())

    def test_no_core_at_the_logical_core(self):
        chip = FakeChip(translated={(25, 16): FakeCore((0, 1))}, eth=[FakeCore((0, 2))])
        with self.assertRaisesRegex(
            CaptureError,
            r"no core is at logical \(0, 0\): "
            r"translated \(25, 16\) is logical \(0, 1\); eth core 0 is logical \(0, 2\)$",
        ):
            make_device(chip).open_router(E1, "wormhole_b0")
        with self.assertRaisesRegex(CaptureError, "eth core 0 raises IndexError"):
            make_device(FakeChip()).open_router(E1, "wormhole_b0")


class Read(unittest.TestCase):
    def test_a_read(self):
        reader = Reader()
        device = make_device(reader=reader)
        opened = OpenedRouter(handle="core", ttexalens_id=CHIP_ID)
        self.assertEqual(device.read(opened, 0x1F80, 4), bytes([0, 1, 2, 3]))
        self.assertEqual(reader.calls, [("core", 0x1F80, CHIP_ID, 4, device.context)])

    def test_a_failed_read(self):
        device = make_device(reader=Reader(error=RuntimeError("timed out")))
        with self.assertRaisesRegex(CaptureError, "reading 4 bytes at 0x1f80 raises RuntimeError: timed out"):
            device.read(OpenedRouter(handle="core", ttexalens_id=CHIP_ID), 0x1F80, 4)


if __name__ == "__main__":
    unittest.main()

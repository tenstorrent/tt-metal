# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cases for a router's health facts, on hand-written liveness samples.

Run from the repository root: python3 -m unittest tt_metal.fabric.debug.visualizer.decode.tests.test_health
"""

import copy
import pathlib
import unittest

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError
from tt_metal.fabric.debug.visualizer.decode.health import EriscReset, get_health
from tt_metal.fabric.debug.visualizer.decode.inputs import Input
from tt_metal.fabric.debug.visualizer.decode.routers import DecodedRouter

PATH = "M0/C0/E0"
ROUTER_MAGIC = 0xDCBA0000
BASE_FW_WORD = 0xABCD1234
ERISC0, ERISC1 = 1 << 11, 1 << 12


def manifest(eriscs, masks=(ERISC0, ERISC1), heartbeat=True):
    """A Blackhole-like manifest with one router, whose eriscs map erisc<n> to a physical ERISC."""
    area = {"address": 0x7CC70, "size": 4, "schema": "u32", "magic": ROUTER_MAGIC, "magic_mask": 0xFFFF0000}
    return {
        "run": {"arch": "blackhole"},
        "archs": {
            "blackhole": {
                "areas": {"heartbeat": area if heartbeat else None},
                "core_registers": [
                    {
                        "name": "soft_reset",
                        "address": 0xFFB121B0,
                        "size": 4,
                        "schema": "u32",
                        "reset_masks": list(masks),
                    },
                    {"name": "wall_clock_low", "address": 0xFFB121F0, "size": 4, "schema": "u32"},
                    {"name": "wall_clock_high", "address": 0xFFB121F8, "size": 4, "schema": "u32"},
                ],
            }
        },
        "meshes": {
            "M0": {
                "chips": {
                    "C0": {
                        "routers": {
                            "E0": {
                                "eriscs": {
                                    key: {"processor": f"riscv_{index}", "physical_erisc": physical}
                                    for index, (key, physical) in enumerate(eriscs.items())
                                }
                            }
                        }
                    }
                }
            }
        },
    }


def registers(soft_reset=0, low=0, high=0):
    return {"soft_reset": soft_reset, "wall_clock_low": low, "wall_clock_high": high}


def sample(t_ns, heartbeat, **kwargs):
    return {"t_ns": t_ns, "heartbeat": heartbeat, "core_registers": registers(**kwargs)}


# Two liveness rounds, then the main pass at 3000 ns.
ENTRY = {
    "liveness": [
        sample(1000, ROUTER_MAGIC | 1, soft_reset=0x47000, low=1000),
        sample(2000, ROUTER_MAGIC | 9, soft_reset=0x47000, low=2000),
    ],
    "started_ns": 3000,
    "core_registers": registers(soft_reset=0x47000, low=3000),
    "error": None,
}


def health(entry=ENTRY, eriscs=None, main_pass_heartbeat=ROUTER_MAGIC | 17, manifest_change=None, **manifest_options):
    areas = {} if main_pass_heartbeat is None else {"heartbeat": main_pass_heartbeat}
    described = manifest(eriscs or {"erisc0": 0, "erisc1": 1}, **manifest_options)
    if manifest_change is not None:
        manifest_change(described)
    loaded = Input(
        pathlib.Path("snapshot.json"),
        "0" * 64,
        {"routers": {PATH: entry}},
        b"",
        pathlib.Path("manifest.json"),
        described,
        (),
    )
    return get_health(loaded, DecodedRouter(PATH, entry["error"], areas, {}, {}, {}))


def with_samples(*samples, started_ns=None, core_registers=None):
    entry = copy.deepcopy(ENTRY)
    entry.update(liveness=list(samples), started_ns=started_ns, core_registers=core_registers or {})
    return entry


class ResetTest(unittest.TestCase):
    def test_each_erisc_reads_its_own_bit_and_ignores_the_rest(self):
        # 0x47000 has bits 12 to 14 and 18 set: ERISC1's bit, and bits for RISCs an Ethernet core doesn't have.
        self.assertEqual(
            health().eriscs,
            {
                "erisc0": EriscReset(0, ERISC0, (False, False, False)),
                "erisc1": EriscReset(1, ERISC1, (True, True, True)),
            },
        )

    def test_a_processor_on_another_physical_erisc_reads_that_erisc_s_bit(self):
        # Blackhole single-ERISC mode: the one processor runs on ERISC1.
        entry = with_samples(sample(1000, None, soft_reset=ERISC0), sample(2000, None, soft_reset=ERISC1))
        self.assertEqual(health(entry, eriscs={"erisc0": 1}).eriscs["erisc0"].in_reset, (False, True))

    def test_a_soft_reset_word_that_wasnt_read_is_unknown(self):
        entry = with_samples({"t_ns": 1000, "heartbeat": None, "core_registers": {}})
        self.assertEqual(health(entry).eriscs["erisc0"].in_reset, (None,))

    def test_a_physical_erisc_with_no_mask_is_refused(self):
        with self.assertRaisesRegex(ManifestError, r"eriscs\.erisc0\.physical_erisc is 1; soft_reset has 1 reset"):
            health(eriscs={"erisc0": 1}, masks=(ERISC0,))


class WallClockTest(unittest.TestCase):
    def test_the_words_join_and_the_rate_is_cycles_per_microsecond(self):
        wall_clock = health().wall_clock
        self.assertEqual(wall_clock.cycles, (1000, 2000, 3000))
        self.assertTrue(wall_clock.rises)
        self.assertEqual(wall_clock.mhz, 1000.0)

    def test_the_low_word_wrapping_still_rises(self):
        entry = with_samples(
            sample(1000, None, low=0xFFFFFF00, high=7),
            sample(2000, None, low=0x100, high=8),
        )
        self.assertEqual(health(entry).wall_clock.cycles, (7 << 32 | 0xFFFFFF00, 8 << 32 | 0x100))
        self.assertTrue(health(entry).wall_clock.rises)

    def test_a_high_word_that_didnt_latch_doesnt_rise(self):
        # The low word wrapped between the two reads of one sample, but the high word wasn't latched with it.
        entry = with_samples(sample(1000, None, low=0xFFFFFF00, high=7), sample(2000, None, low=0x100, high=7))
        self.assertFalse(health(entry).wall_clock.rises)

    def test_fewer_than_two_values_say_nothing(self):
        wall_clock = health(with_samples(sample(1000, None, low=5))).wall_clock
        self.assertEqual((wall_clock.cycles, wall_clock.rises, wall_clock.mhz), ((5,), None, None))


class HeartbeatTest(unittest.TestCase):
    def test_the_main_pass_word_comes_from_the_heartbeat_area(self):
        heartbeat = health().heartbeat
        self.assertEqual(heartbeat.words, (ROUTER_MAGIC | 1, ROUTER_MAGIC | 9, ROUTER_MAGIC | 17))
        self.assertEqual(heartbeat.router_magic, (True, True, True))
        self.assertTrue(heartbeat.changed)

    def test_words_another_writer_left_are_not_the_router_s(self):
        entry = with_samples(sample(1000, BASE_FW_WORD), sample(2000, ROUTER_MAGIC | 5))
        heartbeat = health(entry, main_pass_heartbeat=ROUTER_MAGIC | 5).heartbeat
        self.assertEqual(heartbeat.router_magic, (False, True))
        self.assertIs(heartbeat.changed, None)

    def test_router_words_that_didnt_change(self):
        entry = with_samples(sample(1000, ROUTER_MAGIC | 5), sample(2000, ROUTER_MAGIC | 5))
        self.assertIs(health(entry).heartbeat.changed, False)

    def test_a_heartbeat_the_main_pass_didnt_read_is_unknown(self):
        heartbeat = health(main_pass_heartbeat=None).heartbeat
        self.assertEqual((heartbeat.words[-1], heartbeat.router_magic[-1]), (None, None))

    def test_a_null_heartbeat_area_has_no_heartbeat(self):
        self.assertIsNone(health(heartbeat=False).heartbeat)


class SamplesTest(unittest.TestCase):
    def test_the_liveness_rounds_then_the_main_pass(self):
        self.assertEqual(health().t_ns, (1000, 2000, 3000))

    def test_a_router_capture_couldnt_open_has_no_samples(self):
        facts = health(with_samples(), main_pass_heartbeat=None)
        self.assertEqual(facts.t_ns, ())
        self.assertEqual(facts.eriscs["erisc0"].in_reset, ())
        self.assertEqual((facts.wall_clock.rises, facts.heartbeat.changed), (None, None))

    def test_a_manifest_without_soft_reset_is_refused(self):
        with self.assertRaisesRegex(ManifestError, r"core_registers has no soft_reset"):
            health(manifest_change=lambda described: described["archs"]["blackhole"]["core_registers"].pop(0))


if __name__ == "__main__":
    unittest.main()

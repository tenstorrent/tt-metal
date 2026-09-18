"""Regression for the reviewed stdlib packed-BFP8 stale-C mutant; no numerical golden."""
import copy
import json
import unittest
from functools import lru_cache
from pathlib import Path

from page_io import PAGE, range_keys
from range_pages import PageEffect, validate_new_content_fixture


# Exact packed()/decode() bodies from the independent review reproduction. Decode
# actual valid values: exponent/padding byte changes alone are not write evidence.
def packed(valid_rows):
    tiles = []
    for tile in range(4):
        exponents = []
        mantissas = []
        for face_y in range(2):
            for face_x in range(2):
                for row in range(16):
                    valid = face_y * 16 + row < valid_rows
                    exponents.append(129 if valid else 0)
                    mantissas.extend([80 if valid else 0] * 16)
        tiles.append(bytes(exponents + mantissas))
    result = b"".join(tiles)
    assert len(result) == PAGE
    return result


def decode(raw):
    rows = []
    for row in range(32):
        values = []
        for col in range(128):
            tile = col // 32
            within = col % 32
            face = (row // 16) * 2 + within // 16
            index = face * 16 + row % 16
            exponent = raw[tile * 1088 + index]
            mantissa = raw[tile * 1088 + 64 + index * 16 + within % 16]
            value = (mantissa & 127) * (2.0 ** (exponent - 133))
            values.append(-value if mantissa & 128 else value)
        rows.append(values)
    return rows


decode = lru_cache(maxsize=32)(decode)
DOC = json.loads(Path(__file__).with_name("scenario.json").read_bytes())
FIXTURE = json.loads(Path(__file__).with_name("fixtures.json").read_bytes())["tokens"]
OLD = packed(32)


def changed(raw, row=0, col=0, magnitude=96):
    result = bytearray(raw)
    face = (row // 16) * 2 + (col % 32) // 16
    index = face * 16 + row % 16
    result[(col // 32) * 1088 + 64 + index * 16 + col % 16] = magnitude
    return bytes(result)


def alternate_same_values(raw):
    result = bytearray(raw)
    for tile in range(4):
        for index in range(64):
            offset = tile * 1088 + index
            if result[offset]:
                result[offset] += 1
                for col in range(16):
                    result[tile * 1088 + 64 + index * 16 + col] //= 2
    return bytes(result)


def source_inventory(phase, before, after):
    call = phase["compute_calls"][-1]
    checker = PageEffect("source", phase=phase, call=call, decode=decode)
    for slot in (0, 1):
        for key in range_keys(slot, 0, 2048):
            checker.accept(key, before(key), after(key))
    return checker


class ValidProgressTests(unittest.TestCase):
    # This is the exact reviewed failure: all33 C104 values remain stale A, and only padding is cleared.
    # The source must reject it although a correct transport can copy the stale bytes exactly.
    def test_exact_review_mutant_rejected(self):
        phase = DOC["phases"][4]
        pad_only = packed(1)
        self.assertTrue(all(a != c for a, c in zip(FIXTURE["A"][:33], FIXTURE["C"][:33])))
        after = lambda key: pad_only if key[1] == 0 and key[3] == 32 else OLD
        checker = source_inventory(phase, lambda key: OLD, after)
        self.assertEqual(checker.count, 65536)
        passive = PageEffect("passive", phase=phase)
        for slot in (0, 1):
            for key in range_keys(slot, 0, 2048):
                selected = slot == 1 and key[3] in (0, 32)
                raw = (pad_only if key[3] == 32 else OLD) if selected else OLD
                passive.accept(key, OLD, raw, raw if selected else None)
        self.assertEqual(passive.finish()["pages"], 65536)
        with self.assertRaisesRegex(RuntimeError, "valid-region progress"):
            checker.finish()

    # One changed valid scalar per config/layer is enough; equal rows, pages and other values are permitted.
    def test_one_valid_scalar_per_group_is_enough(self):
        phase = DOC["phases"][4]
        after = lambda key: (changed(OLD) if key[3] == 0 else packed(1)) if key[1] == 0 and key[3] in (0, 32) else OLD
        result = source_inventory(phase, lambda key: OLD, after).finish()
        self.assertEqual(result["valid_change_groups"], 512)

    # A single wholly stale config/layer is not hidden by progress in the other511 groups.
    def test_one_stale_group_rejected(self):
        phase = DOC["phases"][4]

        def after(key):
            config, slot, layer, position = key
            if slot != 0 or position not in (0, 32):
                return OLD
            if position == 32:
                return packed(1)
            return OLD if (config, layer) == (15, 31) else changed(OLD)

        with self.assertRaisesRegex(RuntimeError, "valid-region progress"):
            source_inventory(phase, lambda key: OLD, after).finish()

    # Re-encoding unchanged valid values with other exponent/mantissa bytes must not count as progress.
    def test_exponent_encoding_change_is_not_progress(self):
        phase = DOC["phases"][4]
        equivalent = alternate_same_values(OLD)
        self.assertNotEqual(equivalent, OLD)
        self.assertEqual(decode(equivalent), decode(OLD))
        after = (
            lambda key: (equivalent if key[3] == 0 else alternate_same_values(packed(1)))
            if key[1] == 0 and key[3] in (0, 32)
            else OLD
        )
        with self.assertRaisesRegex(RuntimeError, "valid-region progress"):
            source_inventory(phase, lambda key: OLD, after).finish()

    # C105 must add valid rows33..64; modifying only the already-valid overlap row32 is insufficient.
    def test_overlap_row_alone_is_not_progress(self):
        phase = DOC["phases"][5]
        after = lambda key: (changed(OLD) if key[3] == 32 else packed(1)) if key[1] == 0 and key[3] in (32, 64) else OLD
        with self.assertRaisesRegex(RuntimeError, "valid-region progress"):
            source_inventory(phase, lambda key: OLD, after).finish()

    # A change to one newly valid C105 value in each group establishes structural progress without a golden.
    def test_continuation_new_valid_scalar_progress(self):
        phase = DOC["phases"][5]
        after = (
            lambda key: (changed(OLD, row=1) if key[3] == 32 else packed(1))
            if key[1] == 0 and key[3] in (32, 64)
            else OLD
        )
        result = source_inventory(phase, lambda key: OLD, after).finish()
        self.assertEqual(result["progress_begin"], 33)
        self.assertEqual(result["valid_change_groups"], 512)

    # Only the two identified deterministic A tail replays may preserve every valid value.
    def test_exact_named_a_replays_exempt(self):
        for phase in DOC["phases"][1:3]:
            old = lambda key: packed(9) if key[1] == 0 and key[3] == 1024 else OLD
            result = source_inventory(phase, old, old).finish()
            self.assertFalse(result["valid_change_required"])
            self.assertEqual(result["replay_phase"], phase["name"])

    # A100's first tail write is new content, despite sharing the A name and coordinates with later replay.
    def test_a_initial_tail_is_not_exempt(self):
        phase = DOC["phases"][0]
        old = lambda key: packed(9) if key[1] == 0 and key[3] == 1024 else OLD
        with self.assertRaisesRegex(RuntimeError, "valid-region progress"):
            source_inventory(phase, old, old).finish()

    # The new-content claim is tied to frozen token content, not merely another request ID.
    def test_fixture_precondition_rejects_same_content(self):
        fixture = copy.deepcopy(FIXTURE)
        fixture["C"][:33] = fixture["A"][:33]
        with self.assertRaises(RuntimeError):
            validate_new_content_fixture(fixture)

    # A label cannot grant replay to different tokens, UUID or compute-call identity.
    def test_forged_replay_identity_rejected(self):
        phase = copy.deepcopy(DOC["phases"][1])
        phase["source_command"]["tokens"][0] += 1
        with self.assertRaises(RuntimeError):
            PageEffect("source", phase=phase, call=phase["compute_calls"][0], decode=decode)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Small host oracles for capacity sequencing, selected pages and failure ordering."""

import copy
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import capacity_execution as c
import capacity_pages as p
from range_contract import AckRecorder
from runner_support import CapturedCompletionSink


def phase(value, slot=0):
    end = value - 32 * slot
    return dict(
        source_slot=slot,
        destination_slot=1 - slot,
        valid_prompt_tokens=end,
        source_command={"from": end - 1024, "to": end},
    )


class Table:
    def __init__(self, mapping=None, bad=None):
        self.mapping = mapping or {0: 0, 1: 1}
        self.bad = bad

    def raw(self, key):
        config, slot, layer, pos = key
        slot = self.mapping[slot]
        word = pos // 32 + (layer << 11) + (slot << 16) + (config << 17)
        if self.bad == key:
            word ^= 1
        return struct.pack("<I", word)


class CapacityTests(unittest.TestCase):
    # The complete allocation scales; selected readback never becomes a full-cache dump.
    def test_requested_matrix(self):
        for value, chunks, entries, cache in [
            (4096, 8, 131072, 570425344),
            (8192, 16, 262144, 1140850688),
            (16384, 32, 524288, 2281701376),
            (32768, 64, 1048576, 4563402752),
            (65536, 128, 2097152, 9126805504),
        ]:
            row = c.resources(value)
            self.assertEqual(
                (row["full32_chunk_calls"], row["table_entries_per_endpoint"], row["packed_cache_bytes_per_endpoint"]),
                (chunks, entries, cache),
            )
            self.assertEqual(row["selected_bytes"], 142606336)
            self.assertEqual(row["capacity_warmup_full32_calls"], chunks)
            self.assertEqual(row["total_full32_calls"], 2 + 2 * chunks)
            self.assertEqual(row["packed_cache_bytes_per_chip"] * 32, cache)
            self.assertGreater(row["sentinel_read_bytes"], 0)
        for value in (True, 2048, 3072, 131072, 4096.0):
            with self.assertRaises(ValueError):
                c.resources(value)

    # High-end requests start at zero and advance contiguously, including the final ragged32 rows.
    def test_configured_fixture_cases(self):
        from checks_fixtures import token_manifest

        for value in c.CAPACITIES:
            with token_manifest((value,)) as (path, digest):
                doc, fixtures = c.make_cases(value, path, digest)
            self.assertEqual(len(doc["phases"]), 2)
            for slot, row in enumerate(doc["phases"]):
                calls = row["compute_calls"]
                self.assertEqual(calls[0]["begin"], 0)
                self.assertEqual(calls[-1]["end"], value - slot * 32)
                self.assertEqual(len(calls), value // 1024)
                self.assertTrue(all(a["end"] == b["begin"] for a, b in zip(calls, calls[1:])))
                self.assertEqual(len(c.selected_keys(row)), 16384)
                self.assertEqual(row["destination_slot"], 1 - slot)
                self.assertEqual(row["source_command"]["tokens"], fixtures["slot" + str(slot)])

    # Both sides of an in-bounds range and the opposite slot are checked; allocation end is not readable.
    def test_sentinels_cover_adjacent_and_other_slot(self):
        for value in c.CAPACITIES:
            phases = [phase(value, s) for s in (0, 1)]
            for row in phases:
                keys = set(c.sentinel_keys(value, row))
                spec = row["source_command"]
                dst = row["destination_slot"]
                self.assertIn((0, dst, 0, spec["from"] - 32), keys)
                if spec["to"] < value:
                    self.assertIn((0, dst, 0, spec["to"]), keys)
                self.assertIn((0, 1 - dst, 0, spec["to"] - 32), keys)
                self.assertTrue(all(0 <= k[3] < value for k in keys))
                final = set(c.sentinel_keys(value, row, phases))
                self.assertFalse(final & set(k for other in phases for k in c.destination_keys(other)))

    # A small byte representation preserves independent page identity while avoiding a136MiB host test.
    def test_stream_crosses_chunk_and_preserves_exact_crossed_mapping(self):
        row = phase(4096, 1)
        with tempfile.TemporaryDirectory() as temp, patch.object(p, "PAGE", 4), patch.object(
            p, "read_page", lambda t, k: t.raw(k)
        ):
            writer = p.SelectedWriter(Path(temp) / "pages", row)
            self.assertEqual(writer.capture(Table(), 0, 2048), 0)
            self.assertEqual(writer.capture(Table(), 2048, 3072), 512)
            self.assertEqual(writer.capture(Table(), 3072, 4064), 15872)
            receipt = writer.finish()
            self.assertEqual(receipt["pages"], 16384)
            self.assertEqual(p.live_selected_group_hashes(Table(), row), p.saved_selected_group_hashes(receipt))
            self.assertEqual(len(p.selected_config_hashes(receipt)), 16)
            self.assertTrue(p.compare_selected(receipt, row, Table(), destination=False)["exact"])
            self.assertTrue(p.compare_selected(receipt, row, Table({0: 1, 1: 0}), destination=True)["exact"])
            with self.assertRaises(ValueError):
                p.compare_selected(receipt, row, Table(), destination=True)
            bad = copy.deepcopy(receipt)
            bad["keys"][0] = bad["keys"][1]
            with self.assertRaises(ValueError):
                p.compare_selected(bad, row, Table(), destination=False)
            first = tuple(receipt["keys"][0])
            with self.assertRaises(ValueError):
                p.compare_selected(receipt, row, Table(bad=first), destination=False)

    # Missing/duplicate pages and corrupted saved bytes cannot satisfy complete coverage.
    def test_missing_duplicate_and_changed_capture_rejected(self):
        row = phase(4096)
        with tempfile.TemporaryDirectory() as temp, patch.object(p, "PAGE", 4), patch.object(
            p, "read_page", lambda t, k: t.raw(k)
        ):
            writer = p.SelectedWriter(Path(temp) / "pages", row)
            writer.capture(Table(), 3072, 3104)
            with self.assertRaises(ValueError):
                writer.capture(Table(), 3072, 3104)
            with self.assertRaises(ValueError):
                writer.finish()
            writer.close()
            writer = p.SelectedWriter(Path(temp) / "whole", row)
            writer.capture(Table(), 3072, 4096)
            receipt = writer.finish()
            path = Path(receipt["path"])
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                p.compare_selected(receipt, row, Table(), destination=False)

    # Sentinel content is bound page-by-page, rather than accepted from coverage counts alone.
    def test_sentinel_mutation(self):
        keys = c.sentinel_keys(4096, phase(4096))
        with patch.object(p, "read_page", lambda t, k: t.raw(k)):
            saved = p.sample_pages(Table(), keys)
            self.assertEqual(p.check_samples(Table(), saved)["pages"], len(keys))
            with self.assertRaises(ValueError):
                p.check_samples(Table(bad=keys[-1]), saved)

    # A failed or incomplete selected capture cannot publish even the first readiness acknowledgement.
    def test_capture_precedes_real_ack(self):
        recorder = AckRecorder([dict(slot=0, begin=3072, end=4096)])
        recorder.begin(0, 0, 3072, 4096, 10)
        recorder.synchronized(20)
        events = []

        def failed(request):
            events.append("capture")
            raise ValueError("missing selected bytes")

        sink = CapturedCompletionSink(recorder, failed, lambda *args: events.append("ack"), clock=lambda: 30)
        with self.assertRaises(ValueError):
            sink(0, 0)
        self.assertEqual(events, ["capture"])
        self.assertEqual(recorder.rows, [])


if __name__ == "__main__":
    unittest.main()

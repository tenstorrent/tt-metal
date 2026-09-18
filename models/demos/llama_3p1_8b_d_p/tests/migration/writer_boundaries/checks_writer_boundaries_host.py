# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import writer_boundaries as contract
from writer_boundaries import (
    BF8_PAGE_BYTES,
    CONFIG_NAMES,
    MAX_SEQ_LEN,
    NUM_LAYERS,
    NUM_SLOTS,
    PAGE_TOKENS,
    WRITER_CASES,
    PageKey,
    classify_page_rows,
    decoder_function_source,
    device_major_positions,
    expected_page,
    iter_all_keys,
    seed_row,
    snapshot_specs,
    touched_keys,
    write_row,
)


class WriterBoundaryContractTests(unittest.TestCase):
    # The fixed matrix must independently imply exactly twelve physical positions
    # for every K/V head, rather than trusting a total supplied by the device test.
    def test_fixed_cases_derive_exact_touched_and_untouched_counts(self):
        self.assertEqual(
            [(case.layer, case.slot, case.start, case.end) for case in WRITER_CASES],
            [
                (0, 0, 0, 31),
                (1, 1, 0, 32),
                (7, 0, 0, 33),
                (8, 1, 224, 255),
                (15, 0, 224, 256),
                (16, 1, 224, 257),
                (23, 0, 992, 1023),
                (24, 1, 992, 1024),
                (31, 0, 992, 1025),
                (30, 1, 2016, 2047),
            ],
        )
        keys = touched_keys()
        self.assertEqual(len(keys), (1 + 1 + 2) * 3 * 16 + 16)
        self.assertEqual(len(keys), 208)
        self.assertEqual(len(list(iter_all_keys())) - len(keys), 65_328)
        self.assertEqual(len(keys) * BF8_PAGE_BYTES, 905_216)

    # Valid rows and tile padding are separate acceptance regions, including the
    # three end+1 cases that spill into a second physical page.
    def test_page_row_classification_is_absolute_and_padding_aware(self):
        case = WRITER_CASES[2]
        first = classify_page_rows(case, 0)
        second = classify_page_rows(case, 32)
        self.assertEqual(first.valid_positions, tuple(range(0, 32)))
        self.assertEqual(first.padding_positions, ())
        self.assertEqual(second.valid_positions, (32,))
        self.assertEqual(second.padding_positions, tuple(range(33, 64)))
        self.assertEqual(classify_page_rows(WRITER_CASES[3], 224).valid_positions, tuple(range(224, 255)))

    # Tags use exact powers of two in independent fields so every coordinate and
    # write phase can be distinguished without relying on lossy scalar packing.
    def test_bfp8_exact_tags_distinguish_every_required_coordinate(self):
        base = write_row("k", 0, 0, 0, 0)
        variants = [
            write_row("v", 0, 0, 0, 0),
            write_row("k", 1, 0, 0, 0),
            write_row("k", 0, 1, 0, 0),
            write_row("k", 0, 0, 1, 0),
            write_row("k", 0, 0, 0, 1),
            seed_row("k", 0, 0, 0, 0),
        ]
        self.assertTrue(all(row != base for row in variants))
        for row in [base, *variants]:
            self.assertEqual(len(row), 128)
            self.assertTrue(set(row).issubset({-64.0, -32.0, -16.0, 16.0, 32.0, 64.0}))

    # Every valid row in the fixed ten-call matrix has a distinct exact tag
    # across K/V, head, layer, slot, and absolute position.
    def test_all_written_rows_have_distinct_coordinate_tags(self):
        tags = {
            write_row(kind, head, case.layer, case.slot, position)[:23]
            for case in WRITER_CASES
            for kind in ("k", "v")
            for head in range(8)
            for position in range(case.start, case.end)
        }
        self.assertEqual(len(tags), sum(case.end - case.start for case in WRITER_CASES) * 16)
        self.assertEqual(len(tags), 5_104)

    # The expected page is produced from logical coordinates and never consults
    # a table address, so swapping a slot, head, or page cannot be self-consistent.
    def test_expected_page_rejects_wrong_slot_head_and_page_mapping(self):
        case = WRITER_CASES[8]
        key = PageKey("v", 7, case.layer, case.slot, 1024)
        correct = expected_page(key, case)
        self.assertNotEqual(correct, expected_page(key._replace(slot=1 - key.slot), None))
        self.assertNotEqual(correct, expected_page(key._replace(head=6), case))
        self.assertNotEqual(correct, expected_page(key._replace(position=992), case))

    # Physical input ordering stays balanced across all four SP rows even when
    # the 1K source window crosses a cache-chunk boundary at start 992.
    def test_device_major_positions_cover_cross_chunk_input_once(self):
        positions = device_major_positions(992)
        self.assertEqual(len(positions), 1024)
        self.assertEqual(set(positions), set(range(992, 2016)))
        self.assertEqual(
            [sum((position % 1024) // 256 == owner for position in positions) for owner in range(4)],
            [256, 256, 256, 256],
        )
        with self.assertRaises(ValueError):
            device_major_positions(1)
        with self.assertRaises(ValueError):
            device_major_positions(1025)

    # The final 31-token continuation keeps its valid prefix at the start of SP3,
    # pads row 2047, and carries distinct nonwrapping poison tags beyond capacity.
    def test_final_capacity_input_order_and_poison_tags(self):
        positions = device_major_positions(2016)
        groups = [positions[index * 256 : (index + 1) * 256] for index in range(4)]
        self.assertEqual([len(group) for group in groups], [256, 256, 256, 256])
        self.assertEqual(groups[3][:32], tuple(range(2016, 2048)))
        self.assertEqual(groups[3][31], 2047)
        valid = contract.input_row("write", "k", 7, 30, 1, 2016)
        poison_a = contract.input_row("write", "k", 7, 30, 1, 2048)
        poison_b = contract.input_row("write", "k", 7, 30, 1, 2049)
        self.assertEqual(valid, write_row("k", 7, 30, 1, 2016))
        self.assertNotEqual(poison_a, write_row("k", 7, 30, 1, 0))
        self.assertNotEqual(poison_a, poison_b)
        self.assertTrue(all(value != 0 for value in poison_a))
        with self.assertRaises(ValueError):
            device_major_positions(2048)

    # The final-capacity page has 31 exact valid rows and one zero padding row;
    # the same layer in the other slot remains a seeded untouched page.
    def test_final_capacity_oracle_separates_valid_padding_and_other_slot(self):
        case = WRITER_CASES[-1]
        key = PageKey("v", 7, case.layer, case.slot, 2016)
        page = expected_page(key, case)
        self.assertEqual(
            page[:31],
            tuple(write_row("v", 7, case.layer, case.slot, p) for p in range(2016, 2047)),
        )
        self.assertEqual(page[31], (0.0,) * 128)
        other_slot = key._replace(slot=0)
        self.assertNotIn(other_slot, touched_keys())
        self.assertEqual(
            expected_page(other_slot, None),
            tuple(seed_row("v", 7, case.layer, 0, p) for p in range(2016, 2048)),
        )

    # Every beyond-capacity physical row has a unique poison tag disjoint from
    # ordinary cache-position tags, so the harness never wraps an actual position.
    def test_beyond_capacity_poison_tags_are_unique_and_nonwrapping(self):
        poison = {contract.input_row("write", "k", 0, 30, 1, p) for p in range(2048, 3040)}
        valid = {write_row("k", 0, 30, 1, p) for p in range(992)}
        self.assertEqual(len(poison), 992)
        self.assertTrue(poison.isdisjoint(valid))
        self.assertTrue(all(all(value != 0 for value in row) for row in poison))
        with self.assertRaises(ValueError):
            contract.input_row("write", "k", 0, 30, 1, 3040)

    # The global key order covers the complete table geometry exactly once and
    # matches the accepted K-then-V table configuration convention.
    def test_all_keys_cover_full_table_geometry(self):
        keys = list(iter_all_keys())
        self.assertEqual(CONFIG_NAMES, tuple([f"k_h{i}" for i in range(8)] + [f"v_h{i}" for i in range(8)]))
        self.assertEqual(len(keys), 16 * NUM_LAYERS * NUM_SLOTS * (MAX_SEQ_LEN // PAGE_TOKENS))
        self.assertEqual(len(keys), 65_536)
        self.assertEqual(len(set(keys)), len(keys))

    # Snapshot planning requires one complete file per slot for each phase and
    # accounts for seed validation separately from the two comparison snapshots.
    def test_snapshot_plan_is_complete_and_costs_are_separate(self):
        specs = snapshot_specs(Path("/evidence"))
        self.assertEqual(
            [(s.phase, s.slot, s.begin, s.end) for s in specs],
            [
                ("before", 0, 0, 2048),
                ("before", 1, 0, 2048),
                ("after", 0, 0, 2048),
                ("after", 1, 0, 2048),
            ],
        )
        self.assertEqual(sum(s.expected_bytes for s in specs if s.phase == "before"), 285_212_672)
        self.assertEqual(sum(s.expected_bytes for s in specs if s.phase == "after"), 285_212_672)

    # Pin the formatted publication helpers and the unchanged production decoder.
    # Original accepted hashes and source-equivalence details are in source-origins.json.
    def test_accepted_helper_and_decoder_source_identity(self):
        here = Path(__file__).resolve().parent
        expected = json.loads((here / "publication-helper-hashes.json").read_text())
        for relative, digest in expected.items():
            source = here / relative
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)

    # Decoder extraction is pinned by file digest and function name; altered or
    # missing accepted source fails before any device work can begin.
    def test_decoder_extraction_is_fail_closed_and_import_free(self):
        source = "def _decode_bfp8_chunk(raw, head_dim):\n    return raw, head_dim\n"
        digest = hashlib.sha256(source.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "producer.py"
            path.write_text(source)
            node_source = decoder_function_source(path, digest)
            self.assertIn("def _decode_bfp8_chunk", node_source)
            with self.assertRaisesRegex(ValueError, "digest"):
                decoder_function_source(path, "0" * 64)
            path.write_text("value = 1\n")
            with self.assertRaisesRegex(ValueError, "_decode_bfp8_chunk"):
                decoder_function_source(path, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()

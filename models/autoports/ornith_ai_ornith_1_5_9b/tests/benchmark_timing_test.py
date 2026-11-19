# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of async timeline partitioning and request exclusion."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODEL = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("collector", MODEL / "tools/collect_benchmark_roofline.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def prefill(req, begin, end, ident=0):
    return dict(
        phase="prefill",
        start=begin,
        end=end,
        id=ident,
        request_ids=[req],
        starts=[0],
        ends=[4096],
        original_prompt_lens=[4096],
        discarded=[False],
        completed=True,
    )


def decode(req, begin, end, ident=1):
    return dict(
        phase="decode",
        start=begin,
        end=end,
        id=ident,
        request_ids=[req],
        positions=[4096],
        active=[True],
        completed=True,
    )


class TimingTest(unittest.TestCase):
    def test_async_decode_overlap_counted_once_with_host_gaps(self):
        events = [
            prefill("a", 1, 2),
            decode("a", 2.2, 3, 1),
            decode("a", 2.8, 4, 2),
            prefill("b", 4.2, 5, 3),
            decode("b", 5.2, 6, 4),
        ]
        result = collector.phase_intervals(events)
        self.assertAlmostEqual(result["prefill_seconds"], 1.8)
        self.assertAlmostEqual(result["decode_seconds"], 3.2)
        self.assertAlmostEqual(result["total_wall_seconds"], 5)

    def test_prefill_scheduler_gap_not_charged_to_empty_decode(self):
        result = collector.phase_intervals([prefill("a", 1, 2), prefill("b", 3, 4, 1), decode("b", 5, 6, 2)])
        self.assertEqual(result["prefill_seconds"], 3)
        self.assertEqual(result["decode_seconds"], 2)

    def test_ambiguous_phase_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            collector.phase_intervals([prefill("a", 1, 3), decode("a", 2, 4)])

    def test_probe_and_warmup_excluded_using_detailed_clock(self):
        events = [
            prefill("probe", 1, 2),
            decode("probe", 2, 3),
            prefill("measured", 11, 12, 2),
            decode("measured", 12.1, 13, 3),
        ]
        raw = dict(completed=1, start_times=[10], e2els=[4], total_input_tokens=4096, total_output_tokens=2)
        selected, mapping = collector.identify_requests(raw, events)
        self.assertEqual(mapping["request_ids"], ["measured"])
        self.assertEqual(len(selected), 2)
        raw["completed"] = 2
        with self.assertRaises(ValueError):
            collector.identify_requests(raw, events)

    def test_missing_decode_completion_rejected(self):
        raw = dict(completed=1, start_times=[10], e2els=[4], total_input_tokens=4096, total_output_tokens=128)
        with self.assertRaisesRegex(ValueError, "Missing decode"):
            collector.identify_requests(raw, [prefill("a", 11, 12), decode("a", 12, 13)])

    def test_accounting_ready_without_events_and_weights_shared_by_batch(self):
        cfg = __import__("json").loads((MODEL / "tests/target_config.json").read_text())["text_config"]
        tensor = {"stored_bytes_by_rank": [1088] * 4, "shard_shapes": [[4096, 3136]] * 4}
        layers = []
        for i in range(32):
            kind = "full_attention" if i % 4 == 3 else "linear_attention"
            layer = dict(index=i, kind=kind, weights={"packed_input": tensor}, norms={})
            if kind == "linear_attention":
                layer.update(recurrent=tensor, conv=tensor)
            layers.append(layer)
        policy = dict(id="bfp8_all_lofi", compute_fidelities={"test": "LoFi"})
        metadata = dict(
            layers=layers,
            config=cfg,
            precision={"requested": policy},
            participating_devices=[0, 1, 2, 3],
            head=tensor,
            compute_grid=[11, 10],
            clock_hz=1.35e9,
            dram_bytes_per_second_per_chip=512e9,
            peak_source="test",
        )
        self.assertGreater(collector.account(metadata, [])["prefill_peak_flops_per_second"], 0)
        single = collector.account(metadata, [prefill("a", 1, 2), decode("a", 2, 3)])
        event = decode("a", 2, 3)
        event.update(positions=[4096, 4096], active=[True, True])
        batch = collector.account(metadata, [prefill("a", 1, 2), event])
        self.assertEqual(
            single["decode_bytes_by_kind"]["projection_and_head_weights"],
            batch["decode_bytes_by_kind"]["projection_and_head_weights"],
        )
        self.assertEqual(
            2 * single["decode_bytes_by_kind"]["kv_page_reads"], batch["decode_bytes_by_kind"]["kv_page_reads"]
        )

    def test_snapshot_endpoint_serializes_buffer_without_device(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            with patch.object(collector._timing, "model_accounting", return_value={"test": True}):
                recorder = collector._timing.PhaseRecorder(None, path)
            recorder.events.append(prefill("test", 1, 2))
            result = collector.snapshot(path)
            self.assertEqual(result["events"][0]["request_ids"], ["test"])
            self.assertTrue(result["metadata"]["test"])


if __name__ == "__main__":
    unittest.main()

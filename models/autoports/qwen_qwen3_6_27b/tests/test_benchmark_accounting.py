# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only standalone tests: python tests/test_benchmark_accounting.py."""

import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


timing = module("phase_timing", ROOT / "tt/benchmark_timing.py")
collector = module("collector", ROOT / "doc/benchmark/collect_roofline.py")


class AccountingTests(unittest.TestCase):
    def metadata(self):
        return dict(
            model_config=json.loads((ROOT / "doc/hf_config.json").read_text())["text_config"],
            precision=json.loads((ROOT / "doc/datatype_sweep/selected_precision_config.json").read_text()),
            weights=[dict(name="weight", dtype="DataType.BFLOAT8_B", padded_shape=[32, 32])],
        )

    def test_storage_exponents(self):
        self.assertEqual(collector.tensor_bytes(dict(dtype="bfloat8_b", padded_shape=[32, 32])), 1088)
        self.assertEqual(collector.tensor_bytes(dict(dtype="bfloat4_b", padded_shape=[32, 32])), 576)

    def test_fidelity_and_state_accounting(self):
        metadata = self.metadata()
        rows = [dict(start=0, end=4096)]
        flops, ideal, groups = collector.prefill_work(rows, metadata)
        self.assertGreater(flops, 2 * 20e9 * 4096)
        self.assertEqual(sum(groups.values()), flops)
        self.assertGreater(flops / ideal, 4 * 166e12)
        self.assertLess(flops / ideal, 4 * 664e12)
        _, one = collector.decode_work(dict(execution_rows=1, rows=[dict(position=4096)]), metadata)
        _, eight = collector.decode_work(dict(execution_rows=8, rows=[dict(position=4096)] * 8), metadata)
        self.assertEqual(one["weights"], eight["weights"])
        self.assertEqual(one["recurrent_state"] * 8 * 9 / 2, eight["recurrent_state"])
        self.assertEqual(one["kv"] * 8, eight["kv"])

    def test_inactive_state_preservation_and_physical_embedding_rows(self):
        metadata = self.metadata()
        _, full = collector.decode_work(dict(execution_rows=32, rows=[dict(position=4096)] * 32), metadata)
        _, sparse = collector.decode_work(dict(execution_rows=32, rows=[dict(position=4096)]), metadata)
        volume = 48 * 32 * 48 * 128 * 128 * 2
        self.assertEqual(full["recurrent_state"], 9 * volume)
        self.assertEqual(sparse["recurrent_state"], full["recurrent_state"])
        conv = 48 * 32 * 3 * 10240 * 2
        conv_tile = 48 * 32 * 32 * 10240 * 2
        self.assertEqual(full["convolution_state"], 9 * conv + 6 * conv_tile)
        self.assertGreater(full["state_preservation_masks"], 0)
        self.assertEqual(full["embedding_rows"], 32 * 5120 * 2)
        self.assertEqual(sparse["embedding_rows"], full["embedding_rows"])
        self.assertEqual(sparse["kv"] * 32, full["kv"])

    def test_terminal_decode_inventory_uses_replicated_norm(self):
        def tensor(width):
            return SimpleNamespace(shape=[1, 1, width], padded_shape=[1, 32, width], dtype="float32", layout="TILE")

        layer = SimpleNamespace(
            layer_idx=0,
            kind="full_attention",
            weights={"final_norm": tensor(5120)},
            shard_norms={"final_norm": tensor(1280)},
        )
        model = SimpleNamespace(layers=[layer], lm_head=tensor(32), embedding=tensor(32))
        inventory = timing.tensor_inventory(model, SimpleNamespace(get_device_tensors=lambda t: [t] * 4))
        norms = [item for item in inventory if item["name"] == "final_norm"]
        self.assertEqual(len(norms), 4)
        self.assertEqual([item["shape"][-1] for item in norms], [5120] * 4)

    def test_completion_waits_nonblocking_and_orders_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "phase.jsonl"
            gate = {"done": False}
            tracker = timing.PhaseTiming(log, 2, lambda event: gate["done"])
            p = tracker.prefill([0], [512], [0], [512], [True], [[1, 2]], time.perf_counter())
            d = tracker.decode([512, -1], None, True, 2, time.perf_counter())
            # Deliberately register callback 2 before completion marker 1.
            tracker.read_decode(d, None)
            tracker.complete_decode(d, time.perf_counter())
            tracker.complete_prefill(p, object())
            collector.flush(log)
            self.assertFalse([e for e in collector.read_events(log) if e["kind"] == "complete"])
            gate["done"] = True
            collector.flush(log)
            done = [e for e in collector.read_events(log) if e["kind"] == "complete"]
            self.assertEqual([e["serials"] for e in done], [[1], [2]])
            self.assertLess(done[0]["end"], done[1]["end"])
            # Resident decode increments shadow positions, state remap follows source slots.
            d2 = tracker.decode([], [1, 0], False, 2, time.perf_counter())
            self.assertEqual(d2["rows"][0]["position"], 513)
            self.assertEqual(d2["rows"][0]["slot"], 1)

    def test_mixed_chunks_and_warmup_exclusion(self):
        metadata = self.metadata()
        events = []
        specs = [
            (
                1,
                "prefill",
                1.0,
                2.0,
                [dict(request="probe", start=0, end=512, sampled=True, prefix_sha256="p")],
            ),
            (
                2,
                "prefill",
                10.1,
                11.0,
                [dict(request="a", start=0, end=256, sampled=False, prefix_sha256="a")],
            ),
            (
                3,
                "prefill",
                11.1,
                12.0,
                [dict(request="b", start=0, end=512, sampled=True, prefix_sha256="b")],
            ),
            (4, "decode", 11.9, 12.5, [dict(request="b", position=512)]),
            (
                5,
                "prefill",
                12.4,
                13.0,
                [dict(request="a", start=256, end=512, sampled=True, prefix_sha256=None)],
            ),
            (6, "decode", 13.1, 13.5, [dict(request="a", position=512)]),
        ]
        for serial, phase, start, end, rows in specs:
            events.append(dict(kind="submit", serial=serial, phase=phase, start=start, rows=rows, execution_rows=2))
            events.append(
                dict(
                    kind="complete",
                    serials=[serial],
                    phase=phase,
                    end=end,
                    host_completion=end,
                    boundary="test_existing_completion",
                )
            )
            if phase == "decode":
                events.append(dict(kind="host_complete", serial=serial, start=end - 0.01, end=end))
        raw = dict(
            completed=2,
            start_times=[10.0, 10.02],
            ttfts=[3, 2],
            itls=[[0.6], [0.6]],
            input_lens=[512, 512],
            output_lens=[2, 2],
            total_input_tokens=1024,
            total_output_tokens=4,
        )
        evidence, phases = collector.collect(raw, events, metadata)
        self.assertEqual(evidence["request_ids"], ["a", "b"])
        self.assertAlmostEqual(sum(phase["seconds"] for phase in phases.values()), 13.5 - 10.1)
        self.assertEqual(evidence["extra_executed_outputs"], 0)
        self.assertEqual(len(evidence["timeline"]), 5)
        delayed = [
            dict(e, start=12.7, end=12.9) if e["kind"] == "host_complete" and e["serial"] == 4 else e for e in events
        ]
        overlay, reassigned = collector.collect(raw, delayed, metadata)
        self.assertAlmostEqual(overlay["prefill_seconds_reassigned_to_decode_host"], 0.2)
        self.assertAlmostEqual(sum(phase["seconds"] for phase in reassigned.values()), 13.5 - 10.1)
        broken = [e for e in events if not (e["kind"] == "complete" and e["serials"] == [4])]
        with self.assertRaisesRegex(ValueError, "Missing real completion"):
            collector.collect(raw, broken, metadata)


if __name__ == "__main__":
    unittest.main()

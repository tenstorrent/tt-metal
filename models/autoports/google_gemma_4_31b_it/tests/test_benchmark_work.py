# SPDX-License-Identifier: Apache-2.0
"""Host-only accounting regression checks, with no TT imports."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_work import causal_pairs, decode_work, prefill_work
from collect_benchmark_roofline import account


class BenchmarkAccountingTests(unittest.TestCase):
    def test_chunked_useful_attention_is_additive(self):
        for window in (None, 1024):
            self.assertEqual(
                causal_pairs(0, 4096, window), sum(causal_pairs(s, s + 1024, window) for s in range(0, 4096, 1024))
            )
        full = prefill_work([0], [4096])
        chunks = [prefill_work([s], [s + 1024]) for s in range(0, 4096, 1024)]
        self.assertEqual(full["linear_flops"], sum(c["linear_flops"] for c in chunks))
        self.assertEqual(full["attention_flops"], sum(c["attention_flops"] for c in chunks))

    def test_weights_once_per_decode_batch_not_per_user(self):
        one, eight = decode_work([4096], 1), decode_work([4096] * 8, 8)
        self.assertEqual(one["weight_bytes"], eight["weight_bytes"])
        self.assertEqual(one["kv_read_bytes"] * 8, eight["kv_read_bytes"])
        self.assertEqual(one["physical_rows"], 32)
        self.assertEqual(eight["physical_rows"], 32)

    def test_disjoint_boundaries_include_host_gaps_and_tail(self):
        raw = dict(
            start_times=[1.0],
            ttfts=[0.2],
            itls=[[0.3]],
            duration=0.501,
            total_input_tokens=4096,
            total_output_tokens=2,
            completed=1,
        )
        start = dict(clock="perf_counter_ns", boundary_ns=900000000, pending_steps=0, errors=[])
        end = dict(start, boundary_ns=2000000000, events=[])
        for sid, phase, submit, complete, shape in [
            (1, "prefill", 1010000000, 1150000000, dict(starts=[0], ends=[4096])),
            (2, "decode", 1250000000, 1400000000, dict(effective_positions=[4096])),
        ]:
            for event, ts in [("submit", submit), ("complete", complete)]:
                end["events"].append(
                    dict(
                        step_id=sid,
                        phase=phase,
                        event=event,
                        monotonic_ns=ts,
                        sampling_mode="device",
                        batch_size=1,
                        **shape,
                    )
                )
        result = account(raw, start, end)
        self.assertAlmostEqual(result["seconds"]["prefill"], 0.15)
        self.assertAlmostEqual(result["seconds"]["decode"], 0.35)
        self.assertAlmostEqual(sum(result["seconds"].values()), 0.5)
        end["events"].pop()
        with self.assertRaisesRegex(ValueError, "completion"):
            account(raw, start, end)


if __name__ == "__main__":
    unittest.main()

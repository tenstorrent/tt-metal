# SPDX-License-Identifier: Apache-2.0
"""Host-only accounting regression checks, with no TT imports."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_work import (
    BFP4_BYTES,
    BFP8_BYTES,
    KV_BYTES,
    LAYERS,
    H,
    V,
    causal_pairs,
    decode_work,
    prefill_work,
    projection_elements,
    projection_group_elements,
)
from collect_benchmark_roofline import account


class BenchmarkAccountingTests(unittest.TestCase):
    @staticmethod
    def precision(*, attention="bfp4", mlp="bfp4", down="bfp4", lm_head="bfp4", kv="bfloat8_b"):
        return {
            "weight_groups": {
                "attention": attention,
                "mlp": mlp,
                "down": down,
                "lm_head": lm_head,
            },
            "layer_exceptions": {},
            "kv_cache_dtype": kv,
        }

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

    def test_baseline_and_head_only_preserve_existing_weight_arithmetic(self):
        decoder_elements = sum(
            n * projection_elements(d, kv, shared, physical=True) for n, d, kv, _window, shared in LAYERS
        )
        for head, expected_head_bytes in (("bfp4", BFP4_BYTES), ("bfp8_lofi", BFP8_BYTES), ("bf16", 2.0)):
            expected = BFP4_BYTES * decoder_elements + expected_head_bytes * H * V
            for positions in ([0], [0, 127, 4095]):
                with self.subTest(head=head, positions=positions):
                    implicit = decode_work(positions, 1)
                    explicit = decode_work(positions, 1, self.precision(lm_head=head))
                    self.assertEqual(implicit["weight_bytes"], BFP4_BYTES * decoder_elements + BFP4_BYTES * H * V)
                    self.assertEqual(explicit["weight_bytes"], expected)
                    for key in (
                        "kv_read_bytes",
                        "kv_write_bytes",
                        "norm_bytes",
                        "residual_staging_bytes",
                        "logit_bytes",
                        "embedding_bytes",
                    ):
                        self.assertEqual(explicit[key], implicit[key])
                    self.assertEqual(
                        explicit["dram_bytes"] - implicit["dram_bytes"],
                        expected - implicit["weight_bytes"],
                    )

    def test_group_split_reconstructs_physical_projection_elements(self):
        for n, d, kv, _window, shared in LAYERS:
            groups = projection_group_elements(d, kv, shared, physical=True)
            self.assertEqual(sum(groups.values()), projection_elements(d, kv, shared, physical=True))

    def test_each_projection_group_uses_its_own_storage_width(self):
        positions = [0, 17]
        baseline = decode_work(positions, 1, self.precision())
        for group in ("attention", "mlp", "down"):
            policy = self.precision(**{group: "bfp8_lofi"})
            changed = decode_work(positions, 1, policy)
            elements = sum(
                n * projection_group_elements(d, kv, shared, physical=True)[group]
                for n, d, kv, _window, shared in LAYERS
            )
            expected_delta = (BFP8_BYTES - BFP4_BYTES) * elements
            self.assertEqual(changed["weight_bytes"] - baseline["weight_bytes"], expected_delta)
            self.assertEqual(changed["dram_bytes"] - baseline["dram_bytes"], expected_delta)

    def test_bf16_kv_changes_both_read_and_write_byte_terms(self):
        positions = [0, 3, 1023, 1024]
        bfp8 = decode_work(positions, 1, self.precision(kv="bfloat8_b"))
        bf16 = decode_work(positions, 1, self.precision(kv="bfloat16"))
        ratio = KV_BYTES["bfloat16"] / KV_BYTES["bfloat8_b"]
        self.assertEqual(bf16["kv_read_bytes"], bfp8["kv_read_bytes"] * ratio)
        self.assertEqual(bf16["kv_write_bytes"], bfp8["kv_write_bytes"] * ratio)
        self.assertEqual(bf16["weight_bytes"], bfp8["weight_bytes"])
        self.assertEqual(
            bf16["dram_bytes"] - bfp8["dram_bytes"],
            (bf16["kv_read_bytes"] - bfp8["kv_read_bytes"]) + (bf16["kv_write_bytes"] - bfp8["kv_write_bytes"]),
        )

    def test_full_bf16_weights_and_bf16_kv_are_supported(self):
        policy = self.precision(attention="bf16", mlp="bf16", down="bf16", lm_head="bf16", kv="bfloat16")
        result = decode_work([0, 4], 1, policy)
        expected = (
            sum(2.0 * n * projection_elements(d, kv, shared, physical=True) for n, d, kv, _window, shared in LAYERS)
            + 2.0 * H * V
        )
        self.assertEqual(result["weight_bytes"], expected)
        self.assertGreater(result["kv_read_bytes"], 0)
        self.assertGreater(result["kv_write_bytes"], 0)

    def test_unsupported_exceptions_groups_labels_and_cache_dtype_are_rejected(self):
        invalid = self.precision()
        invalid["layer_exceptions"] = {"0": {"down": "bfp4"}}
        with self.assertRaisesRegex(ValueError, "layer exceptions"):
            decode_work([0], 1, invalid)
        for edit, message in (
            (lambda p: p.pop("layer_exceptions"), "four supported weight groups and no layer exceptions"),
            (lambda p: p["weight_groups"].pop("mlp"), "four supported weight groups"),
            (lambda p: p["weight_groups"].update(extra="bfp4"), "four supported weight groups"),
            (lambda p: p["weight_groups"].update(down="int4"), "Unsupported precision label"),
            (lambda p: p.update(kv_cache_dtype="bfloat4_b"), "BFP8 or BF16 KV cache"),
        ):
            invalid = self.precision()
            edit(invalid)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                decode_work([], 1, invalid)

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

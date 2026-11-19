# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only negative controls for comparator selection; no TTNN/Torch import."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import check_fused_evidence as checker

E = checker.ROOT / "doc/fused_decoder/endurance"
GOOD = E / "v6_direct_coeff_layer34_seed154.json"
SOURCE = "c823e6f6f8480a9550276856a7aa660b9308ff598c6565a38c2bc54460440ece"


class ComparatorEligibility(unittest.TestCase):
    def test_existing_cached_reference_is_eligible_for_selection(self):
        result = checker.check_comparator_endurance(GOOD, source_sha=SOURCE, layer=34)
        self.assertTrue(result["cached_hf_reference"])
        self.assertEqual(result["seed"], 154)

    def test_corrupted_or_intervened_evidence_rejected(self):
        original = json.loads(GOOD.read_text())
        metadata = json.loads(Path(str(GOOD) + ".provenance.json").read_text())
        cases = {
            "bad_contribution": lambda d, p: d["results"][0].update(contribution_pcc=0.912681),
            "nonfinite_pcc": lambda d, p: d["results"][0].update(pcc=float("nan")),
            "duplicate_decode": lambda d, p: d["results"][0].update(position=d["results"][1]["position"]),
            "duplicate_continuation": lambda d, p: d["continuation_results"][0].update(
                position=d["continuation_results"][1]["position"]
            ),
            "missing_row": lambda d, p: d["results"].pop(),
            "false_tracking": lambda d, p: d.update(tracking=False),
            "cache_excluded": lambda d, p: p["env"].update(TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="1"),
            "oracle_assisted": lambda d, p: d.update(oracle_assisted=True),
            "intervention_harness": lambda d, p: p.update(command=["v6_prefix_state_control.py"]),
            "wrong_source": lambda d, p: p.update(measured_decoder_source_sha256="0" * 64),
            "wrong_layer": lambda d, p: d.update(layer=0),
            "wrong_prefill": lambda d, p: d.update(prefill_pcc=0.99),
            "failed_rows": lambda d, p: d.update(failed_rows=[128]),
            "wrong_label": lambda d, p: p.update(decoder="functional"),
            "oracle_parameters": lambda d, p: d["cpu_oracle_key"]["parameters"].update(seed=42),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            for name, mutate in cases.items():
                with self.subTest(name=name):
                    evidence, provenance = copy.deepcopy(original), copy.deepcopy(metadata)
                    mutate(evidence, provenance)
                    path.write_text(json.dumps(evidence))
                    Path(str(path) + ".provenance.json").write_text(json.dumps(provenance))
                    with self.assertRaises(AssertionError):
                        checker.check_comparator_endurance(path, source_sha=SOURCE, layer=34)

    def test_historical_full_generic_label_rejected(self):
        path = E / "v4_padded_views_stress_3_seed913.json"
        _, source = checker.artifact_identity(path)
        with self.assertRaises(AssertionError):
            checker.check_comparator_endurance(path, source_sha=source, layer=3)

    def test_manifest_cannot_reduce_seed_minimum(self):
        for seeds in ([], [913], [42, 154], [913, 42]):
            with self.subTest(seeds=seeds), self.assertRaises(AssertionError):
                checker.comparator_requirements(
                    dict(required_endurance_seeds=seeds, representative_comparator_correctness=["layer34.json"]),
                    layer=0,
                    kind="linear_attention",
                )
        required, representatives = checker.comparator_requirements(
            dict(required_endurance_seeds=[913, 42, 154, 7], representative_comparator_correctness=["layer34.json"]),
            layer=0,
            kind="linear_attention",
        )
        self.assertEqual(required, {913, 42, 154, 7})
        self.assertEqual(representatives, ["layer34.json"])
        self.assertEqual(
            checker.comparator_requirements(dict(required_endurance_seeds=[913]), layer=3, kind="full_attention"),
            ({913}, []),
        )

    def test_manifest_representative_layer_is_separate_and_mandatory(self):
        # Synthetic metadata tests manifest control flow only. Per-row validation
        # is independently exercised above; nothing here is hardware evidence.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "control.py.snapshot"
            source.write_text("# synthetic comparator metadata fixture\n")
            control_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            candidate_sha = "1" * 64
            for name, digest, latency in (("candidate", candidate_sha, 1.0), ("control", control_sha, 2.0)):
                report = dict(
                    decoder="FusedDecoder",
                    deterministic=True,
                    runtime_fallback_audit="pass",
                    latency_tracking_enabled=False,
                    prefill_pcc=1.0,
                    prefill_contribution_pcc=1.0,
                    decode_eager_pcc=1.0,
                    decode_traced_pcc=1.0,
                    prefill_warmed_ms=1.0,
                    decode_traced_warmed_ms=latency,
                    prefill_samples_ms=[1.0],
                    decode_samples_ms=[latency],
                    layer=0,
                    kind="linear_attention",
                    length=256,
                    context=8192,
                )
                path = root / (name + ".json")
                path.write_text(json.dumps(report))
                Path(str(path) + ".provenance.json").write_text(
                    json.dumps(
                        dict(
                            source_sha256={"tt/fused_decoder.py": digest},
                            env={},
                            regime="serial",
                            device={"id": "synthetic"},
                            versions={"fixture": True},
                        )
                    )
                )
            item = dict(
                candidate="candidate.json",
                comparator="control.json",
                comparator_source=source.name,
                comparator_correctness=["seed913.json", "seed42.json", "seed154.json"],
                required_endurance_seeds=[913, 42, 154],
                representative_comparator_correctness=["representative.json"],
            )
            manifest = root / "manifest.json"

            def write():
                manifest.write_text(
                    json.dumps(dict(selection_rationale="Synthetic host-only gate test", comparisons=[item]))
                )

            def validated(path, *, source_sha, layer):
                self.assertEqual(source_sha, control_sha)
                return dict(layer=layer, seed=154 if path.name == "representative.json" else int(path.stem[4:]))

            write()
            with patch.object(checker, "check_comparator_endurance", side_effect=validated) as call:
                result = checker.check_best_correct(manifest, candidate_sha)
                self.assertEqual(result["comparisons"][0]["layer"], 0)
                self.assertEqual(result["comparisons"][0]["representative_comparator_checks"][0]["layer"], 34)
                self.assertEqual(call.call_args_list[-1].kwargs["layer"], 34)
            item["representative_comparator_correctness"] = []
            write()
            with self.assertRaises(AssertionError):
                checker.check_best_correct(manifest, candidate_sha)
            item["representative_comparator_correctness"] = ["representative.json"]
            write()

            def wrong_seed(path, *, source_sha, layer):
                result = validated(path, source_sha=source_sha, layer=layer)
                if layer == 34:
                    result["seed"] = 42
                return result

            with patch.object(checker, "check_comparator_endurance", side_effect=wrong_seed), self.assertRaises(
                AssertionError
            ):
                checker.check_best_correct(manifest, candidate_sha)


if __name__ == "__main__":
    unittest.main()

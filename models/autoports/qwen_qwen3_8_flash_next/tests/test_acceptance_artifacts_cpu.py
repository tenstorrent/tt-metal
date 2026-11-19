# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only EOS artifact and blind-rating provenance controls; no TTNN import."""

import ast
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def normalization_function():
    source = Path(__file__).with_name("run_full_model_acceptance.py")
    tree = ast.parse(source.read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("save", "normalize_autoregressive_artifacts")
    ]
    namespace = dict(Path=Path, json=json, MODEL_ID="test/model")
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(source), "exec"), namespace
    )
    return namespace["normalize_autoregressive_artifacts"]


class AcceptanceArtifactsCPU(unittest.TestCase):
    def test_actual_final_status_call_preserves_event_and_completed_phase(self):
        source = Path(__file__).with_name("run_full_model_acceptance.py")
        tree = ast.parse(source.read_text())
        marker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "marker")
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "marker"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "ACCEPTANCE_DRIVER_DONE"
        )
        namespace = dict(
            json=json, phase="qualitative", report={"passed": True}, args=SimpleNamespace(output="result.json")
        )
        module = ast.Module(body=[marker, ast.Expr(value=call)], type_ignores=[])
        output = io.StringIO()
        with redirect_stdout(output):
            exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "phase": "ACCEPTANCE_DRIVER_DONE",
                "passed": True,
                "completed_phase": "qualitative",
                "output": "result.json",
            },
        )

    def test_eos_normalization_preserves_raw_and_checker_schema(self):
        normalize = normalization_function()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "meta": root / "autoregressive_meta.json",
                "hf_completion": root / "hf_completion.txt",
                "tt_completion": root / "tt_completion.txt",
            }
            original = {
                "hf_model_id": "test/model",
                "max_new_tokens": 128,
                "hf": {"token_ids": [4, 99], "num_tokens": 2},
                "tt": {"token_ids": [5, 98, 7, 99, 8], "num_tokens": 5},
            }
            paths["meta"].write_text(json.dumps(original))
            tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: ",".join(map(str, ids)))
            normalize(paths, tokenizer, {"eos_token_ids": [98, 99], "max_new_tokens": 128, "source": "pinned_fixture"})
            output = json.loads(paths["meta"].read_text())
            self.assertEqual(output["tt"]["token_ids"], [5, 98])
            self.assertEqual(output["tt"]["num_tokens"], 2)
            self.assertEqual(output["tt"]["raw_token_ids"], original["tt"]["token_ids"])
            self.assertEqual(output["hf"]["token_ids"], [4, 99])
            self.assertEqual(json.loads((root / "autoregressive_raw.json").read_text()), original)
            self.assertEqual(paths["tt_completion"].read_text(), "5,98")
            self.assertEqual((root / "tt_completion_raw.txt").read_text(), "5,98,7,99,8")

    def test_no_eos_keeps_full_length(self):
        normalize = normalization_function()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "meta": root / "autoregressive_meta.json",
                "hf_completion": root / "hf_completion.txt",
                "tt_completion": root / "tt_completion.txt",
            }
            original = {
                "hf_model_id": "test/model",
                "max_new_tokens": 128,
                "hf": {"token_ids": [1, 2], "num_tokens": 2},
                "tt": {"token_ids": [3, 4], "num_tokens": 2},
            }
            paths["meta"].write_text(json.dumps(original))
            normalize(
                paths,
                SimpleNamespace(decode=lambda ids, **kw: str(ids)),
                {"eos_token_ids": [99], "max_new_tokens": 128},
            )
            result = json.loads(paths["meta"].read_text())
            for backend in ("hf", "tt"):
                self.assertEqual(result[backend]["token_ids"], original[backend]["token_ids"])
                self.assertIsNone(result[backend]["first_eos_index"])


class BlindProvenanceCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = ROOT / "doc/full_model/qualitative/blind_compare.py"
        spec = importlib.util.spec_from_file_location("blind_compare_cpu_fixture", source)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module.ROOT = self.root
        self.write("prompts.json", [{"id": "p1", "messages": [{"role": "user", "content": "fixture"}]}])
        self.write("rubric.json", {"rule": "synthetic rubric"})
        stop = {"eos_token_ids": [99], "source": "pinned_fixture", "max_new_tokens": 128}
        self.write("stopping_config.json", stop)
        rows = [
            dict(
                prompt_id="p1",
                mode="sampled",
                seed=seed,
                stopping=stop,
                raw_generated_token_ids=[seed, 99, 8],
                generated_token_ids=[seed, 99],
                completion=f"synthetic{seed}",
            )
            for seed in range(10)
        ]
        self.write("hf_outputs.json", rows)
        self.write("tt_outputs.json", rows)
        self.module.prepare(self.root / "tt_outputs.json")
        envelope = json.loads((self.root / "blind_ratings_template.json").read_text())
        for row in envelope["ratings"]:
            row.update(failed=False, reason="Synthetic CPU control")
        self.write("ratings.json", envelope)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_matching_frozen_packet_passes(self):
        self.module.compare(self.root / "ratings.json")
        result = json.loads((self.root / "blind_comparison.json").read_text())
        self.assertEqual(result["verdict"], "within_matched_control_margin")
        self.assertIn("ratings_sha256", result)

    def test_same_ids_with_stale_packet_hash_rejected(self):
        envelope = json.loads((self.root / "ratings.json").read_text())
        envelope["packet_sha256"] = "0" * 64
        self.write("ratings.json", envelope)
        with self.assertRaisesRegex(ValueError, "exact packet and rubric hashes"):
            self.module.compare(self.root / "ratings.json")

    def test_modified_packet_rubric_or_labels_rejected(self):
        for name in ("blind_packet.json", "rubric.json", "blind_labels_DO_NOT_GIVE_TO_JUDGE.json"):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_text()
                self.write(name, {"changed": True})
                with self.assertRaisesRegex(ValueError, "changed after preparation"):
                    self.module.compare(self.root / "ratings.json")
                path.write_text(original)

    def test_prepare_cannot_overwrite_different_packet(self):
        packet = (self.root / "blind_packet.json").read_text()
        rows = json.loads((self.root / "tt_outputs.json").read_text())
        rows[0]["completion"] = "different synthetic fixture"
        self.write("tt_outputs.json", rows)
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.module.prepare(self.root / "tt_outputs.json")
        self.assertEqual((self.root / "blind_packet.json").read_text(), packet)

    def test_repeat_prepare_does_not_reset_template(self):
        template = self.root / "blind_ratings_template.json"
        template.write_text((self.root / "ratings.json").read_text())
        before = template.read_text()
        self.module.prepare(self.root / "tt_outputs.json")
        self.assertEqual(template.read_text(), before)


if __name__ == "__main__":
    unittest.main()

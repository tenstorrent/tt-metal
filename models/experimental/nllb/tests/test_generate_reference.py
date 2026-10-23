# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests: actual portable fixture consumer, policy and publication failures."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "reference"
spec = importlib.util.spec_from_file_location("reference_prototype", HERE / "generate_reference.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
PACKAGE = Path(os.environ.get("NLLB_PACKAGE_DIRECTORY", str(HERE.parent)))
h = g.load_helper(PACKAGE, g.HELPER_SHA256)
CONFIG = dict(vocab_size=50, decoder_start_token_id=2, eos_token_id=2, pad_token_id=1)


def tiny_assets(root):
    files = {name: b"{}" for name in g.TOKENIZER_FILES | {"config.json", "generation_config.json"}}
    files["pytorch_model.bin"] = b"not loaded: unit fixture only"
    for name, value in files.items():
        (root / name).write_bytes(value)
    entry = dict(
        repo_id="facebook/nllb-200-distilled-600M",
        revision="a" * 40,
        weight_files=["pytorch_model.bin"],
        weight_index=None,
        files={k: dict(sha256=g.sha(root / k), bytes=len(v)) for k, v in files.items()},
    )
    manifest = root / "manifest.data"
    manifest.write_text(json.dumps(dict(schema="nllb-local-reference-assets-v1", models={"600m": entry})))
    return manifest, entry


def synthetic_requests():
    inputs = dict(input_ids=np.ones((4, 256), dtype=np.int64), attention_mask=np.ones((4, 256), dtype=np.int64))
    meta = dict(source_lengths=[255, 256, 7, 2], target_id=9)
    outputs, events = {}, {}
    for case in h.CASES:
        for name, ids, mask, cap in h.requests(case, inputs, meta["source_lengths"]):
            tokens = np.full((len(ids), cap + 1), 4, dtype=np.int64)
            tokens[:, :2] = [2, 9]
            if case == "base":
                tokens[2:, 3] = 2
                tokens[2:, 4:] = 1
            key = case + "__" + name
            outputs[key] = tokens
            events[key] = [[n, len(ids)] for n in range(1, cap + 1)]
    return inputs, meta, outputs, events


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_actual_portable_consumer_accepts_generated_schema(self):
        inputs, meta, outputs, events = synthetic_requests()
        config = self.root / "config.json"
        config.write_text(json.dumps(CONFIG))
        identity = g.make_identity(h, config, inputs, meta, {"pytorch_model.bin": "a" * 64})
        qualified = g.qualify(h, outputs, events, inputs, meta, CONFIG)
        self.assertTrue(qualified["covered"])
        fixture = g.make_fixture(identity, outputs, dict(qualification=qualified))
        path = self.root / "fixture.json"
        g.publish(path, fixture, True)
        for precision in ("bf16", "bfp8_b"):
            result = h.compare_oracle(path, g.sha(path), identity, outputs, 9, CONFIG, precision)
            self.assertTrue(result["exact"])
            self.assertEqual(result["reference_precision"], "fp32")

    def test_fixture_metadata_excludes_publication_state(self):
        _, _, outputs, _ = synthetic_requests()
        report = dict(completed=False, qualification={"covered": True}, precision="fp32")
        fixture = g.make_fixture({}, outputs, report)
        path = self.root / "fixture.json"
        g.publish(path, fixture, True)
        report.update(completed=True, fixture_sha256=g.sha(path))
        metadata = json.loads(path.read_text())["metadata"]
        self.assertNotIn("completed", metadata)
        self.assertNotIn("fixture_sha256", metadata)
        self.assertTrue(metadata["qualification"]["covered"])
        self.assertEqual(metadata["precision"], "fp32")
        self.assertTrue(report["completed"])
        self.assertNotIn("completed", fixture["metadata"])

    def test_tokenizer_explicit_fast_local_policy(self):
        factory = Mock()
        actual = g.load_tokenizer(factory, "/official/tokenizer")
        self.assertIs(actual, factory.from_pretrained.return_value)
        factory.from_pretrained.assert_called_once_with(
            "/official/tokenizer",
            local_files_only=True,
            trust_remote_code=False,
            token=False,
            src_lang="eng_Latn",
            use_fast=True,
        )

    def test_consumer_rejects_identity_or_archive_replacement(self):
        inputs, meta, outputs, _ = synthetic_requests()
        config = self.root / "config.json"
        config.write_text(json.dumps(CONFIG))
        identity = g.make_identity(h, config, inputs, meta, {"pytorch_model.bin": "a" * 64})
        path = self.root / "fixture.json"
        g.publish(path, g.make_fixture(identity, outputs, {}), True)
        pin = g.sha(path)
        for field, changed in [
            ("target_id", 8),
            ("weight_sha256", {"pytorch_model.bin": "b" * 64}),
            ("config_sha256", "c" * 64),
            ("input_hashes", {}),
            ("requests", {}),
        ]:
            altered = dict(identity, **{field: changed})
            with self.subTest(field=field), self.assertRaises(ValueError):
                h.compare_oracle(path, pin, altered, outputs, 9, CONFIG)
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "external SHA256"):
            h.compare_oracle(path, pin, identity, outputs, 9, CONFIG)

    def test_asset_hash_and_optional_tokenizer_files_fail_closed(self):
        manifest, _ = tiny_assets(self.root)
        pin = g.sha(manifest)
        g.validate_assets(manifest, pin, "600m", self.root, self.root)
        (self.root / "added_tokens.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "untracked loader"):
            g.validate_assets(manifest, pin, "600m", self.root, self.root)
        (self.root / "added_tokens.json").unlink()
        for name in ["config.json", "generation_config.json", "tokenizer.json", "pytorch_model.bin"]:
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"x")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "mismatch"):
                g.validate_assets(manifest, pin, "600m", self.root, self.root)
            path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "manifest SHA256"):
            g.validate_assets(manifest, "0" * 64, "600m", self.root, self.root)

    def test_external_symlink_and_path_traversal_rejected(self):
        manifest, entry = tiny_assets(self.root)
        outside = self.root.parent / (self.root.name + "-outside")
        outside.write_bytes((self.root / "pytorch_model.bin").read_bytes())
        self.addCleanup(outside.unlink)
        (self.root / "pytorch_model.bin").unlink()
        (self.root / "pytorch_model.bin").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "outside"):
            g.validate_assets(manifest, g.sha(manifest), "600m", self.root, self.root)
        entry["files"]["../config.json"] = entry["files"].pop("config.json")
        manifest.write_text(json.dumps(dict(schema="nllb-local-reference-assets-v1", models={"600m": entry})))
        with self.assertRaises(ValueError):
            g.validate_assets(manifest, g.sha(manifest), "600m", self.root, self.root)

    def test_index_and_manifest_duplicate_keys_rejected(self):
        manifest, entry = tiny_assets(self.root)
        index = self.root / "pytorch_model.bin.index.json"
        index.write_text('{"weight_map":{"a":"pytorch_model.bin","a":"else.bin"}}')
        entry["weight_index"] = index.name
        entry["files"][index.name] = dict(bytes=index.stat().st_size, sha256=g.sha(index))
        manifest.write_text(json.dumps(dict(schema="nllb-local-reference-assets-v1", models={"600m": entry})))
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            g.validate_assets(manifest, g.sha(manifest), "600m", self.root, self.root)
        index.write_text('{"weight_map":{"a":"else.bin"}}')
        entry["files"][index.name] = dict(bytes=index.stat().st_size, sha256=g.sha(index))
        manifest.write_text(json.dumps(dict(schema="nllb-local-reference-assets-v1", models={"600m": entry})))
        with self.assertRaisesRegex(ValueError, "shard inventory"):
            g.validate_assets(manifest, g.sha(manifest), "600m", self.root, self.root)

    def test_loader_uses_safe_local_fp32_api(self):
        torch = SimpleNamespace(float32="float32", device=lambda kind, index: (kind, index))
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.parameters.return_value = [SimpleNamespace(dtype="float32", device=SimpleNamespace(type="cuda"))]
        factory = Mock()
        factory.from_pretrained.return_value = (model, {})
        actual, _ = g.load_fp32_model(factory, torch, "/official/checkpoint", 0)
        self.assertIs(actual, model)
        kwargs = factory.from_pretrained.call_args.kwargs
        self.assertEqual(
            kwargs,
            dict(
                local_files_only=True,
                trust_remote_code=False,
                token=False,
                use_safetensors=False,
                weights_only=True,
                torch_dtype="float32",
                attn_implementation="eager",
                output_loading_info=True,
            ),
        )
        model.to.assert_called_once_with(("cuda", 0))
        model.eval.assert_called_once()

    def test_loader_rejects_missing_weights_cpu_and_wrong_precision(self):
        torch = SimpleNamespace(float32="float32", device=lambda kind, index: (kind, index))
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        factory = Mock()
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            factory.from_pretrained.return_value = (model, {key: ["defect"]})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "load exactly"):
                g.load_fp32_model(factory, torch, "/official", 0)
        model.to.assert_not_called()
        factory.from_pretrained.return_value = (model, {})
        for dtype, device in [("float16", "cuda"), ("float32", "cpu")]:
            model.parameters.return_value = [SimpleNamespace(dtype=dtype, device=SimpleNamespace(type=device))]
            with self.subTest(dtype=dtype, device=device), self.assertRaisesRegex(ValueError, "CUDA FP32"):
                g.load_fp32_model(factory, torch, "/official", 0)

    def loading_factory(self):
        torch = SimpleNamespace(float32="float32", device=lambda kind, index: (kind, index))
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.parameters.return_value = [SimpleNamespace(dtype="float32", device=SimpleNamespace(type="cuda"))]
        # Transformers 5.17 LoadStateDictInfo.to_dict() returns these exact types.
        loading = dict(missing_keys=set(), unexpected_keys=set(), mismatched_keys=set(), error_msgs=[])
        factory = Mock()
        factory.from_pretrained.return_value = (model, loading)
        return factory, torch, loading

    def test_realistic_loading_sets_fixture_and_report_are_serializable(self):
        factory, torch, original = self.loading_factory()
        _, loading = g.load_fp32_model(factory, torch, "/official", 0)
        _, _, outputs, _ = synthetic_requests()
        report = dict(loading_info=loading, completed=False)
        fixture = g.make_fixture({}, outputs, report)
        path = self.root / "fixture.json"
        g.publish(path, fixture, True)
        self.assertEqual(
            json.loads(path.read_text())["metadata"]["loading_info"],
            dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[]),
        )
        self.assertEqual(json.loads(json.dumps(report))["loading_info"], loading)
        self.assertIsInstance(original["missing_keys"], set)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            bad = copy.deepcopy(original)
            bad[key] = {("weight", (2,), (3,))} if key == "mismatched_keys" else {"defect"}
            factory.from_pretrained.return_value = (factory.from_pretrained.return_value[0], bad)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "load exactly"):
                g.load_fp32_model(factory, torch, "/official", 0)

    def test_loading_metadata_deterministic_and_no_unknown_coercion(self):
        raw = dict(keys={"z", "a"}, shapes={("z", (3,), (4,)), ("a", (1,), (2,))})
        normalized = g.loading_metadata(raw)
        self.assertEqual(normalized, dict(keys=["a", "z"], shapes=[["a", [1], [2]], ["z", [3], [4]]]))
        self.assertEqual(json.dumps(normalized), json.dumps(g.loading_metadata(raw)))
        for value in (object(), b"bytes", {1: "not a string key"}, float("nan")):
            with self.subTest(value=type(value).__name__), self.assertRaises(TypeError):
                g.loading_metadata(value)

    def test_main_serializes_loaded_sets_on_success_and_qualification_failure(self):
        for qualified in (True, False):
            with self.subTest(qualified=qualified):
                output = self.root / ("success.json" if qualified else "failed.json")
                argv = [
                    "--model",
                    "600m",
                    "--checkpoint",
                    "unused",
                    "--package-directory",
                    str(PACKAGE),
                    "--execution",
                    "direct",
                    "--output",
                    str(output),
                ]

                def fake_run(args, report):
                    factory, torch, _ = self.loading_factory()
                    _, report["loading_info"] = g.load_fp32_model(factory, torch, "/official", 0)
                    _, _, outputs, _ = synthetic_requests()
                    report["qualification"] = dict(covered=qualified)
                    g.publish(args.output, g.make_fixture({}, outputs, report), qualified)
                    report.update(completed=True, fixture_sha256=g.sha(args.output))

                with patch.object(g, "run", side_effect=fake_run):
                    self.assertEqual(g.main(argv), 0 if qualified else 2)
                report = json.loads(output.with_name(output.name + ".report.json").read_text())
                self.assertEqual(report["completed"], qualified)
                self.assertEqual(report["loading_info"]["missing_keys"], [])
                self.assertEqual(output.exists(), qualified)
                if not qualified:
                    self.assertIn("coverage incomplete", report["error"])

    def test_none_generation_defaults_inspected_without_mutation(self):
        generation = SimpleNamespace(
            **{k: CONFIG[k] for k in ("decoder_start_token_id", "eos_token_id", "pad_token_id")},
            min_length=None,
            repetition_penalty=None,
            no_repeat_ngram_size=None,
            encoder_no_repeat_ngram_size=None,
        )
        defaults = dict(min_length=0, repetition_penalty=1.0, no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0)
        generation._get_default_generation_params = lambda: defaults.copy()
        before = vars(generation).copy()
        policy = g.generation_policy(generation, CONFIG)
        self.assertIsNone(policy["raw"]["repetition_penalty"])
        self.assertEqual(policy["effective"]["repetition_penalty"], 1.0)
        self.assertEqual(vars(generation), before)
        for key, value in [
            ("forced_eos_token_id", 2),
            ("repetition_penalty", 1.1),
            ("min_length", 100),
            ("suppress_tokens", [4]),
        ]:
            altered = copy.copy(generation)
            setattr(altered, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                g.generation_policy(altered, CONFIG)
        del generation._get_default_generation_params
        with self.assertRaises(ValueError):
            g.generation_policy(generation, CONFIG)

    def test_no_cpu_or_login_inference(self):
        g.require_compute_host("direct", "gpu-a", {}, True)
        g.require_compute_host("slurm", "compute-42", {"SLURM_JOB_ID": "1"}, True)
        for args in [
            ("direct", "gpu-a", {}, False),
            ("direct", "login-gpu", {}, True),
            ("direct", "lo-42", {}, True),
            ("slurm", "gpu-a", {}, True),
        ]:
            with self.subTest(args=args), self.assertRaises(RuntimeError):
                g.require_compute_host(*args)

    def test_missing_actual_prefix_or_natural_eos_never_publishes(self):
        inputs, meta, outputs, events = synthetic_requests()
        bad = copy.deepcopy(events)
        bad["base__cap64"].pop()
        with self.assertRaisesRegex(ValueError, "observations"):
            g.qualify(h, outputs, bad, inputs, meta, CONFIG)
        outputs["base__cap64"][2:, 2:] = 4
        q = g.qualify(h, outputs, events, inputs, meta, CONFIG)
        self.assertFalse(q["covered"])
        path = self.root / "fixture.json"
        with self.assertRaisesRegex(ValueError, "coverage incomplete"):
            g.publish(path, g.make_fixture({}, outputs, {}), q["covered"])
        self.assertFalse(path.exists())

    def test_early_eos_does_not_claim_boundary(self):
        inputs, meta, outputs, events = synthetic_requests()
        outputs["base__cap64"][0, 3] = 2
        outputs["base__cap64"][0, 4:] = 1
        self.assertFalse(g.qualify(h, outputs, events, inputs, meta, CONFIG)["covered"])

    def test_publication_exclusive_and_failed_write_cleanup(self):
        path = self.root / "fixture.json"
        path.write_text("earlier fixture")
        with self.assertRaises(FileExistsError):
            g.publish(path, {}, True)
        self.assertEqual(path.read_text(), "earlier fixture")
        self.assertFalse(path.with_name(path.name + ".pending").exists())
        path.unlink()
        with patch.object(g.os, "link", side_effect=OSError("disk error")), self.assertRaises(OSError):
            g.publish(path, {}, True)
        self.assertFalse(path.exists())

    def test_pending_publication_not_removed_by_second_writer(self):
        path = self.root / "fixture.json"
        pending = path.with_name(path.name + ".pending")
        pending.write_text("another writer")
        with self.assertRaises(FileExistsError):
            g.publish(path, {}, True)
        self.assertEqual(pending.read_text(), "another writer")
        self.assertFalse(path.exists())

    def test_main_failure_durable_no_fixture_and_no_retry(self):
        output = self.root / "fixture.json"
        argv = [
            "--model",
            "600m",
            "--checkpoint",
            "unused",
            "--package-directory",
            str(PACKAGE),
            "--execution",
            "direct",
            "--output",
            str(output),
        ]
        with patch.object(g, "run", side_effect=RuntimeError("no CUDA")):
            self.assertEqual(g.main(argv), 2)
        self.assertFalse(output.exists())
        report = json.loads(output.with_name(output.name + ".report.json").read_text())
        self.assertFalse(report["completed"])
        self.assertIn("no CUDA", report["error"])
        with self.assertRaises(FileExistsError):
            g.main(argv)

    def test_fresh_cli_help_does_not_import_torch_or_tt(self):
        result = subprocess.run(
            [sys.executable, str(HERE / "generate_reference.py"), "--help"], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--execution", result.stdout)
        self.assertIn("--manifest-sha256", result.stdout)
        code = f"import runpy,sys;runpy.run_path({str(HERE / 'generate_reference.py')!r},run_name='library');assert 'torch' not in sys.modules;assert 'ttnn' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""CPU-only ownership injection; these tests claim no hardware coverage."""

import ast
import builtins
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_runtime():
    spec = importlib.util.spec_from_file_location("runtime_under_test", ROOT / "tt/runtime_setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.runtime = load_runtime()
        self.events = []
        self.capture_checks = []
        self.device = types.SimpleNamespace(enable_program_cache=lambda: self.events.append("cache"))
        self.ttnn = types.SimpleNamespace(
            open_device=lambda **kw: self.open(kw),
            close_device=lambda device: self.events.append(("close", device)),
            is_trace_capture_active=lambda device: self.capture_checks.append(device) or False,
        )
        self.modules = patch.dict(
            sys.modules,
            {
                "ttnn": self.ttnn,
                "models.experimental.nllb.tt.backend": types.SimpleNamespace(
                    DEVICE_OPTIONS={"trace_region_size": 64 * 1024 * 1024}
                ),
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.env = patch.dict(
            os.environ,
            {
                "TT_METAL_TRACE_ALLOC_TRACKING": "1",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def open(self, options):
        self.events.append(options)
        return self.device

    def test_tracking_before_import(self):
        with patch.dict(sys.modules):
            del sys.modules["ttnn"]
            with patch.dict(os.environ, {}, clear=True):
                self.runtime.configure_tracking()
                self.assertNotIn("ttnn", sys.modules)
                self.assertEqual(os.environ["TT_METAL_TRACE_ALLOC_TRACKING"], "1")
                self.assertEqual(os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"], "0")

    def test_late_tracking_rejected(self):
        with patch.dict(os.environ, {"TT_METAL_TRACE_ALLOC_TRACKING": "0"}):
            with self.assertRaisesRegex(RuntimeError, "before importing"):
                self.runtime.configure_tracking()
        self.assertEqual(self.events, [])

    def test_owned_options_cache_close(self):
        with self.runtime.RuntimeOwner() as owner:
            self.assertIs(owner.open(7), self.device)
        self.assertEqual(
            self.events, [{"device_id": 7, "trace_region_size": 67108864}, "cache", ("close", self.device)]
        )

    def test_caller_device_untouched(self):
        with self.runtime.RuntimeOwner(self.device) as owner:
            owner.bind(types.SimpleNamespace(_trace_failures=[]))
        self.assertEqual(self.events, [])

    def test_clean_failure_closes_preserves_primary(self):
        primary = ValueError("request")
        with self.assertRaises(ValueError) as caught:
            with self.runtime.RuntimeOwner() as owner:
                owner.open(7)
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertEqual(self.events[-1], ("close", self.device))

    def test_close_and_reporting_failures_preserve_primary(self):
        class Primary(ValueError):
            def add_note(self, note):
                raise RuntimeError("reporting")

        primary = Primary("request")

        def bad_close(device):
            raise RuntimeError("close")

        self.ttnn.close_device = bad_close
        with self.assertRaises(Primary) as caught:
            with self.runtime.RuntimeOwner() as owner:
                owner.open(7)
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIs(self.runtime.retained_owners()[0], owner)

    def test_unresolved_borrowed_retains_model_blocks_outer_close(self):
        primary = ValueError("native")
        model = types.SimpleNamespace(_trace_failures=[object()])
        with self.assertRaises(ValueError) as caught:
            with self.runtime.RuntimeOwner() as outer:
                outer.open(7)
                with self.runtime.RuntimeOwner(self.device) as inner:
                    inner.bind(model)
                    raise primary
        self.assertIs(caught.exception, primary)
        self.assertIn(inner, self.runtime.retained_owners())
        self.assertIs(inner.models[0], model)
        self.assertNotIn(("close", self.device), self.events)

    def test_unresolved_without_primary_reports_failure(self):
        with self.assertRaisesRegex(RuntimeError, "retained"):
            with self.runtime.RuntimeOwner(self.device) as owner:
                owner.bind(types.SimpleNamespace(_trace_failures=[object()]))
        self.assertEqual(self.events, [])

    def test_trace_cleanup_failure_retains(self):
        def fail():
            raise RuntimeError("release")

        with self.assertRaisesRegex(RuntimeError, "release"):
            with self.runtime.RuntimeOwner(self.device) as owner:
                owner.bind(types.SimpleNamespace(_trace_failures=[], _decode_trace=types.SimpleNamespace(close=fail)))
        self.assertIn(owner, self.runtime.retained_owners())

    def test_success_releases_active_trace(self):
        trace = types.SimpleNamespace(close=lambda: self.events.append("release"), unresolved=False)
        with self.runtime.RuntimeOwner() as owner:
            owner.open(7)
            owner.bind(types.SimpleNamespace(_trace_failures=[], _decode_trace=trace))
        self.assertEqual(self.events[-2:], ["release", ("close", self.device)])

    def test_cache_setup_failure_closes(self):
        def fail():
            raise ValueError("cache")

        self.device.enable_program_cache = fail
        with self.assertRaisesRegex(ValueError, "cache"):
            self.runtime.RuntimeOwner().open(7)
        self.assertEqual(self.events[-1], ("close", self.device))

    def test_script_and_package_import(self):
        with patch.dict(sys.modules):
            sys.modules["models.experimental.nllb.tt.runtime_setup"] = self.runtime
            for package in (False, True):
                name = "portable_nllb.translate" if package else "portable_translate"
                if package:
                    pkg = types.ModuleType("portable_nllb")
                    pkg.__path__ = [str(ROOT)]
                    sys.modules["portable_nllb"] = pkg
                spec = importlib.util.spec_from_file_location(name, ROOT / "demo/translate.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertTrue(callable(module.translate))
                self.assertTrue(callable(module.main))
        self.assertEqual(self.events, [])

    def test_import_has_no_tracking_or_ttnn_side_effect(self):
        with patch.dict(sys.modules):
            del sys.modules["ttnn"]
            with patch.dict(os.environ, {}, clear=True):
                load_runtime()
                self.assertNotIn("ttnn", sys.modules)
                self.assertNotIn("TT_METAL_TRACE_ALLOC_TRACKING", os.environ)

    def test_open_sets_tracking_before_first_ttnn_import(self):
        real_import = builtins.__import__
        observed = []

        def checked_import(name, *args, **kwargs):
            if name == "ttnn":
                observed.append(True)
                self.assertEqual(os.environ["TT_METAL_TRACE_ALLOC_TRACKING"], "1")
                self.assertEqual(os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"], "0")
                return self.ttnn
            return real_import(name, *args, **kwargs)

        with patch.dict(sys.modules):
            del sys.modules["ttnn"]
            with patch.dict(os.environ, {}, clear=True), patch("builtins.__import__", checked_import):
                with self.runtime.RuntimeOwner() as owner:
                    owner.open(7)
        self.assertTrue(observed)
        self.assertTrue(owner.closed)

    def test_exact_backend_device_options(self):
        tree = ast.parse((ROOT / "tt/backend.py").read_text())
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "DEVICE_OPTIONS" for t in node.targets)
        )
        # Check the exact constant expression without executing parsed source.
        expected = ast.parse('{"trace_region_size": 64 * 1024 * 1024}', mode="eval").body
        self.assertEqual(ast.dump(assignment.value), ast.dump(expected))
        options = {"trace_region_size": 67108864}
        sys.modules["models.experimental.nllb.tt.backend"].DEVICE_OPTIONS = options
        with self.runtime.RuntimeOwner() as owner:
            owner.open(7)
        self.assertEqual(self.events[0], {"device_id": 7, **options})
        self.assertEqual(self.events[1], "cache")
        self.assertEqual(self.capture_checks, [self.device, self.device])

    def test_empty_registry_active_capture_retained_no_retry(self):
        self.ttnn.is_trace_capture_active = lambda device: True
        owner = self.runtime.RuntimeOwner()
        owner.open(7)
        with self.assertRaisesRegex(RuntimeError, "retained"):
            owner.finish()
        self.assertEqual(owner.models, [])
        self.assertFalse(owner.closed)
        self.assertIs(owner.device, self.device)
        self.assertIn(owner, self.runtime.retained_owners())
        self.ttnn.is_trace_capture_active = lambda device: False
        with self.assertRaisesRegex(RuntimeError, "retained"):
            owner.finish()
        with self.assertRaisesRegex(RuntimeError, "retained"):
            owner.open(7)
        self.assertEqual(self.events, [{"device_id": 7, "trace_region_size": 67108864}, "cache"])

    def test_capture_observation_error_preserves_primary(self):
        primary = ValueError("request")

        def fail(device):
            raise RuntimeError("capture observation")

        self.ttnn.is_trace_capture_active = fail
        with self.assertRaises(ValueError) as caught:
            with self.runtime.RuntimeOwner() as owner:
                owner.open(7)
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIn(owner, self.runtime.retained_owners())
        self.assertEqual(str(owner.cleanup_errors[0]), "capture observation")
        self.assertNotIn(("close", self.device), self.events)

    def test_bound_source96_unresolved_owner_survives_cleared_registry_slot(self):
        from models.experimental.nllb.tt import trace_decode

        native = patch.object(trace_decode, "ttnn", self.ttnn)
        native.start()
        self.addCleanup(native.stop)
        model = types.SimpleNamespace(_trace_failures=[], _decode_trace=None, _last_warmup_owners=set())
        trace = trace_decode.DecoderTrace.__new__(trace_decode.DecoderTrace)
        trace.model, trace.device = model, self.device
        trace.trace_id, trace.unresolved = 123, False
        trace.events, trace.inputs = [], [object()]
        trace.output, trace.bucket = object(), 32

        def fail_release(device, trace_id):
            self.events.append("release_failed")
            raise RuntimeError("native release")

        self.ttnn.release_trace = fail_release
        owner = self.runtime.RuntimeOwner()
        owner.open(7)
        owner.bind(model)
        trace.close(preserve_exception=True)
        self.assertIs(model._trace_failures[0], trace)
        self.assertIsNone(model._decode_trace)
        with self.assertRaisesRegex(RuntimeError, "retained"):
            owner.finish()
        self.assertIn(owner, self.runtime.retained_owners())
        self.assertIs(owner.models[0], model)
        self.assertIsNotNone(trace.output)
        self.assertEqual(trace.trace_id, 123)
        self.assertEqual(len(trace.inputs), 1)
        self.assertEqual(self.events, [{"device_id": 7, "trace_region_size": 67108864}, "cache", "release_failed"])

    def test_bound_trace_unresolved_flag_blocks_release_and_close(self):
        trace = types.SimpleNamespace(unresolved=True, close=lambda: self.fail("must not retry unresolved release"))
        with self.assertRaisesRegex(RuntimeError, "retained"):
            with self.runtime.RuntimeOwner() as owner:
                owner.open(7)
                owner.bind(types.SimpleNamespace(_trace_failures=[], _decode_trace=trace))
        self.assertIn(owner, self.runtime.retained_owners())
        self.assertNotIn(("close", self.device), self.events)

    def test_orphaned_preparation_owner_cleanup(self):
        trace = types.SimpleNamespace(unresolved=False, close=lambda: self.events.append("release"))
        model = types.SimpleNamespace(_trace_failures=[], _decode_trace=None, _last_warmup_owners=[trace])
        with self.runtime.RuntimeOwner() as owner:
            owner.open(7)
            owner.bind(model)
        self.assertEqual(self.events[-2:], ["release", ("close", self.device)])

    def test_trace_release_and_reporting_errors_preserve_primary(self):
        class Primary(ValueError):
            def add_note(self, message):
                raise RuntimeError("reporting")

        primary = Primary("request")

        def fail():
            raise RuntimeError("release")

        with self.assertRaises(Primary) as caught:
            with self.runtime.RuntimeOwner() as owner:
                owner.open(7)
                owner.bind(types.SimpleNamespace(_trace_failures=[], _decode_trace=types.SimpleNamespace(close=fail)))
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIn(owner, self.runtime.retained_owners())
        self.assertEqual(str(owner.cleanup_errors[0]), "release")
        self.assertNotIn(("close", self.device), self.events)

    def test_capture_activated_during_cleanup_prevents_close(self):
        active = [False]
        self.ttnn.is_trace_capture_active = lambda device: active[0]
        trace = types.SimpleNamespace(close=lambda: active.__setitem__(0, True), unresolved=False)
        with self.assertRaisesRegex(RuntimeError, "retained"):
            with self.runtime.RuntimeOwner() as owner:
                owner.open(7)
                owner.bind(types.SimpleNamespace(_trace_failures=[], _decode_trace=trace))
        self.assertIn(owner, self.runtime.retained_owners())
        self.assertNotIn(("close", self.device), self.events)


class TranslateRuntimeTests(unittest.TestCase):
    """Public entry wiring with mock assets/backend/native calls; no inference."""

    open = RuntimeIntegrationTests.open

    def setUp(self):
        RuntimeIntegrationTests.setUp(self)
        import io
        import json
        import tempfile

        self.io = io
        self.json = json
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.checkpoint = Path(self.tmp.name) / "weights.bin"
        self.config_path = Path(self.tmp.name) / "explicit-config.json"
        self.tokenizer_path = Path(self.tmp.name) / "explicit-tokenizer"
        self.config = dict(
            d_model=4,
            vocab_size=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=1,
            decoder_attention_heads=1,
            encoder_ffn_dim=8,
            decoder_ffn_dim=8,
            max_position_embeddings=128,
        )
        self.config_path.write_text(json.dumps(self.config))
        self.owners = []
        original_bind = self.runtime.RuntimeOwner.bind

        def bind(owner, model):
            self.owners.append(owner)
            self.events.append("bind")
            return original_bind(owner, model)

        self.runtime.RuntimeOwner.bind = bind
        self.primary = None
        self.unresolved = False
        self.release_error = False
        self.model = types.SimpleNamespace(_trace_failures=[], precision_policy={"mode": "bf16"})

        def release():
            self.events.append("release")
            if self.release_error:
                raise RuntimeError("release")

        self.model._decode_trace = types.SimpleNamespace(close=release, unresolved=False)
        self.rows = [[2, 7, 8, 2]]
        self.tokens = types.SimpleNamespace(tolist=lambda: self.rows)

        def generate(ids, mask, target, cap):
            self.assertIs(self.owners[-1].models[0], self.model)
            self.assertEqual((ids, mask, target, cap), ("ids", "mask", 7, 4))
            self.events.append("generate")
            if self.unresolved:
                self.model._trace_failures.append(object())
            if self.primary:
                raise self.primary
            return self.tokens

        self.model.generate = generate

        def create(checkpoint, config, device, *, precision):
            self.assertEqual(Path(checkpoint), self.checkpoint)
            self.assertIs(device, self.device)
            self.assertEqual(precision, "bf16")
            self.events.append("create")
            return self.model

        self.ttnn.synchronize_device = lambda device: self.events.append("sync")
        self.backend = types.SimpleNamespace(DEVICE_OPTIONS={"trace_region_size": 67108864}, create_backend=create)
        sys.modules.update(
            {
                "models.experimental.nllb.tt.backend": self.backend,
                "models.experimental.nllb.tt.runtime_setup": self.runtime,
                "torch": types.SimpleNamespace(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None),
                "numpy": types.SimpleNamespace(bool_=type("MockNumpyBool", (), {})),
            }
        )
        # Execute the unchanged validation module; only unused tensor libraries
        # are mocked. Actual config/integer validation runs below.
        spec = importlib.util.spec_from_file_location("nllb_validation", ROOT / "tt/nllb_validation.py")
        self.validation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.validation)
        sys.modules["models.experimental.nllb.tt.nllb_validation"] = self.validation

    def entry(self, package=False):
        name = "portable_translate"
        for name in (
            "models",
            "models.experimental",
            "models.experimental.nllb",
            "models.experimental.nllb.tt",
            "models.experimental.nllb.demo",
        ):
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT)]
            sys.modules[name] = module
        prefix = "models.experimental.nllb"
        sys.modules[prefix + ".tt.runtime_setup"] = self.runtime
        sys.modules[prefix + ".tt.backend"] = self.backend
        sys.modules[prefix + ".tt.nllb_validation"] = self.validation
        name = prefix + ".demo.translate" if package else "portable_translate"
        spec = importlib.util.spec_from_file_location(name, ROOT / "demo/translate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def inputs(checkpoint, config, source, target, texts, *, tokenizer_directory):
            self.assertEqual(Path(checkpoint), self.checkpoint)
            self.assertEqual(Path(tokenizer_directory), self.tokenizer_path)
            self.assertEqual((source, target, texts), ("eng_Latn", "fra_Latn", ["Hello"]))
            self.events.append("inputs")

            def decode(tokens, *, skip_special_tokens):
                self.assertIs(tokens, self.tokens)
                self.assertTrue(skip_special_tokens)
                return ["Bonjour"]

            return types.SimpleNamespace(batch_decode=decode), "ids", "mask", 7

        module.load_text_inputs = inputs
        return module

    def api(self, module, **kwargs):
        return module.translate(
            self.checkpoint,
            self.config,
            self.device,
            "eng_Latn",
            "fra_Latn",
            ["Hello"],
            max_new_tokens=kwargs.get("cap", 4),
            tokenizer_directory=self.tokenizer_path,
        )

    def main(self, module):
        output = self.io.StringIO()
        result = module.main(
            [
                "--checkpoint",
                str(self.checkpoint),
                "--config",
                str(self.config_path),
                "--tokenizer-directory",
                str(self.tokenizer_path),
                "--source-language",
                "eng_Latn",
                "--target-language",
                "fra_Latn",
                "--device",
                "7",
                "--max-new-tokens",
                "4",
                "--text",
                "Hello",
            ],
            result_stream=output,
        )
        self.assertEqual(result, 0)
        return self.json.loads(output.getvalue())

    def test_public_api_script_import_borrowed_success(self):
        result = self.api(self.entry())
        self.assertEqual(result["token_ids"], self.rows)
        self.assertEqual(result["translations"], ["Bonjour"])
        self.assertLess(self.events.index("bind"), self.events.index("generate"))
        self.assertEqual(self.events[-1], "release")
        self.assertFalse(any(isinstance(e, (tuple, dict)) for e in self.events))

    def test_public_api_package_import_borrowed_success(self):
        self.assertEqual(self.api(self.entry(True))["token_ids"], self.rows)
        self.assertEqual(self.events[-1], "release")

    def test_public_cli_script_import_owned_success(self):
        self.assertEqual(self.main(self.entry())["token_ids"], self.rows)
        self.assertIn({"device_id": 7, "trace_region_size": 67108864}, self.events)
        self.assertIn("cache", self.events)
        self.assertEqual(self.events[-2:], ["release", ("close", self.device)])

    def test_public_cli_package_import_owned_success(self):
        self.assertEqual(self.main(self.entry(True))["translations"], ["Bonjour"])
        self.assertEqual(self.events[-1], ("close", self.device))

    def test_public_api_clean_failure_keeps_borrowed_device(self):
        self.primary = ValueError("request")
        with self.assertRaises(ValueError) as caught:
            self.api(self.entry())
        self.assertIs(caught.exception, self.primary)
        self.assertEqual(self.events[-1], "release")
        self.assertNotIn(("close", self.device), self.events)

    def test_public_cli_clean_failure_closes_owned_device(self):
        self.primary = ValueError("request")
        with self.assertRaises(ValueError) as caught:
            self.main(self.entry())
        self.assertIs(caught.exception, self.primary)
        self.assertEqual(self.events[-2:], ["release", ("close", self.device)])

    def test_public_cli_unresolved_retains_bound_model_and_device(self):
        self.unresolved = True
        with self.assertRaisesRegex(RuntimeError, "retained"):
            self.main(self.entry())
        self.assertNotIn(("close", self.device), self.events)
        retained = self.runtime.retained_owners()
        self.assertEqual(len(retained), 2)
        self.assertIs(retained[0].models[0], self.model)
        self.assertTrue(all(owner.device is self.device for owner in retained))

    def test_public_api_unresolved_retains_borrowed_model(self):
        self.unresolved = True
        with self.assertRaisesRegex(RuntimeError, "retained"):
            self.api(self.entry(True))
        self.assertIs(self.runtime.retained_owners()[0].models[0], self.model)
        self.assertNotIn(("close", self.device), self.events)

    def test_public_cli_primary_release_reporting_failures(self):
        class Primary(ValueError):
            def add_note(self, note):
                raise RuntimeError("report")

        self.primary = Primary("request")
        self.release_error = True
        with self.assertRaises(Primary) as caught:
            self.main(self.entry())
        self.assertIs(caught.exception, self.primary)
        self.assertEqual(str(self.runtime.retained_owners()[0].cleanup_errors[0]), "release")
        self.assertNotIn(("close", self.device), self.events)

    def test_public_api_original_integer_validation(self):
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            self.api(self.entry(), cap=True)
        self.assertEqual(self.events, [])

    def test_public_entry_import_has_no_tracking_side_effect(self):
        with patch.dict(sys.modules), patch.dict(os.environ, {}, clear=True):
            del sys.modules["ttnn"]
            self.entry(True)
            self.assertNotIn("ttnn", sys.modules)
            self.assertNotIn("TT_METAL_TRACE_ALLOC_TRACKING", os.environ)

    def test_public_cli_tracking_precedes_ttnn_import(self):
        module = self.entry()
        real_import = builtins.__import__
        observed = []

        def checked(name, *args, **kwargs):
            if name == "ttnn":
                observed.append(True)
                self.assertEqual(os.environ["TT_METAL_TRACE_ALLOC_TRACKING"], "1")
                self.assertEqual(os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"], "0")
                return self.ttnn
            return real_import(name, *args, **kwargs)

        with patch.dict(sys.modules), patch.dict(os.environ, {}, clear=True):
            del sys.modules["ttnn"]
            with patch("builtins.__import__", checked):
                self.main(module)
        self.assertTrue(observed)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Host-only deployment checks; device imports are stubbed and no model is built."""

import builtins
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "models.autoports.google_gemma_4_31b_it.tt."


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def missing_readiness(name, *args, **kwargs):
    if name == "readiness_check.contract":
        raise ModuleNotFoundError("No module named 'readiness_check'", name="readiness_check")
    return ORIGINAL_IMPORT(name, *args, **kwargs)


ORIGINAL_IMPORT = builtins.__import__


def generator_stubs(contract):
    imports = {
        "ttnn": {},
        "transformers": {"AutoTokenizer": object, "GenerationConfig": object},
        PREFIX + "generator_contract": {"Generator": contract},
        PREFIX + "batched_prefill": {"PreparedBatchedPrefill": object},
        PREFIX + "model": {"Gemma4Model": object},
        PREFIX + "optimized_decoder": {"HF_MODEL": "unused", "HF_REVISION": "unused"},
        PREFIX + "prefill": {"PreparedPrefill": object},
        "models.common.sampling.generator": {"SamplingParams": object, "format_sampling_params": object},
        "models.common.sampling.tt_sampling": {"TTSampling": object},
    }
    modules = {}
    for name, attrs in imports.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        modules[name] = module
    return modules


class SourcePackagingChecks(unittest.TestCase):
    def test_doc_free_policy_exactness_and_overrides(self):
        selected = json.loads((ROOT / "doc/datatype_sweep/selected_precision_config.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "tt"
            shutil.copytree(ROOT / "tt", runtime, ignore=shutil.ignore_patterns("__pycache__"))
            self.assertFalse((runtime.parent / "doc").exists())
            precision = load("isolated_precision", runtime / "precision.py")
            self.assertEqual(precision.load_precision(), selected)
            policy = precision.load_precision()
            policy["weight_groups"]["lm_head"] = "bf16"
            self.assertEqual(precision.load_precision(), selected)
            self.assertEqual(precision.load_precision(precision.BASELINE), precision.BASELINE)
            self.assertEqual(precision.load_precision(runtime / "selected_precision_config.json"), selected)
            with self.assertRaises(FileNotFoundError):
                precision.load_precision(runtime / "absent.json")

    def test_source_only_generator_import_and_abstract_enforcement(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "tt"
            shutil.copytree(ROOT / "tt", runtime, ignore=shutil.ignore_patterns("__pycache__"))
            with patch("builtins.__import__", side_effect=missing_readiness):
                contract = load("standalone_contract", runtime / "generator_contract.py").Generator
                with patch.dict(sys.modules, generator_stubs(contract)):
                    module = load("isolated_generator", runtime / "generator.py")
                    phases = load(PREFIX + "benchmark_phases", runtime / "benchmark_phases.py")
                    with patch.dict(sys.modules, {PREFIX + "generator": module, PREFIX + "benchmark_phases": phases}):
                        adapter = load(PREFIX + "generator_vllm", runtime / "generator_vllm.py")
                    self.assertIs(adapter.Gemma4Generator, module.Gemma4Generator)
                    self.assertEqual(Path(adapter.__file__).parent, runtime)
            self.assertFalse(inspect.isabstract(module.Gemma4Generator))
            self.assertTrue(issubclass(module.Gemma4Generator, contract))
            with self.assertRaises(TypeError):
                contract()
            with self.assertRaises(NotImplementedError):
                contract.prefill_logits_continuation(None, [], 0)
            self.assertEqual(
                contract.__abstractmethods__,
                {"prefill_forward", "decode_forward", "prefill_logits", "generate", "reset"},
            )

    def test_installed_package_errors_propagate(self):
        for missing in ("readiness_check.contract", "missing_dependency"):

            def fail(name, *args, **kwargs):
                if name == "readiness_check.contract":
                    raise ModuleNotFoundError(name=missing)
                return ORIGINAL_IMPORT(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fail):
                with self.assertRaises(ModuleNotFoundError) as caught:
                    load("broken_contract", ROOT / "tt/generator_contract.py")
            self.assertEqual(caught.exception.name, missing)

    def test_installed_real_abc_and_fallback_signatures(self):
        contract_path = os.environ.get("READINESS_CONTRACT_PATH")
        if contract_path is None:
            self.skipTest("Set READINESS_CONTRACT_PATH to check the installed readiness ABC")
        installed = load("installed_readiness_contract", Path(contract_path))
        with patch.dict(sys.modules, {"readiness_check.contract": installed}):
            resolved = load("installed_compatibility", ROOT / "tt/generator_contract.py").Generator
            self.assertIs(resolved, installed.Generator)
            with patch.dict(sys.modules, generator_stubs(resolved)):
                generator = load("installed_generator", ROOT / "tt/generator.py").Gemma4Generator
            self.assertTrue(issubclass(generator, installed.Generator))
            self.assertFalse(inspect.isabstract(generator))
        with patch("builtins.__import__", side_effect=missing_readiness):
            fallback = load("fallback_signatures", ROOT / "tt/generator_contract.py").Generator
        names = installed.Generator.__abstractmethods__ | {"prefill_logits_continuation"}
        for name in names:
            real = inspect.signature(getattr(installed.Generator, name), eval_str=True)
            local = inspect.signature(getattr(fallback, name), eval_str=True)
            self.assertEqual(real, local, name)


if __name__ == "__main__":
    unittest.main()

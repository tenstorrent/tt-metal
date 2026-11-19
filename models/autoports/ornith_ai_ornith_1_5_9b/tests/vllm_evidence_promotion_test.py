# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of extracted validation functions; never invoke promotion."""

import ast
import json
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "doc/vllm_integration/promote_final_evidence.py"
NAMES = {
    "require",
    "file",
    "read",
    "contents",
    "resolved",
    "identity",
    "require_source",
    "exit_code",
    "launch",
    "tree_files",
    "sampling_origin",
}


class SamplingReuseValidation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.tracked = self.model / "tracked"
        self.original = self.tracked / "first_attempt"
        self.loaded = self.tracked / "loaded_model_source"
        self.observer = self.tracked / "retry_native_observer"
        self.files = ("sampling_tests.log", "full_sampling_runner.log", "vllm_qualitative_outputs.json")
        tree = ast.parse(SOURCE.read_text())
        extracted = ast.Module(
            body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES], type_ignores=[]
        )
        self.ns = {"Path": Path, "json": json, "REPO": self.root, "MODEL": self.model, "SAMPLING_FILES": self.files}
        exec(compile(extracted, str(SOURCE), "exec"), self.ns)  # noqa: S102 - selected local validators only
        self.launch = {
            "server_command": ["python", "serve", "--async-scheduling"],
            "hf_model": "ornith-ai/Ornith-1.5-9B",
            "mesh_device": "P300x2",
            "max_num_seqs": 32,
            "max_model_len": 262144,
            "block_size": 32,
            "tt_config": {"sample_on_device_mode": "all", "trace_mode": "decode_only"},
            "enable_prefix_caching": False,
        }
        self.env = {"plugin_sha": "unchanged", "environment": {"tracking": "1"}}
        self.manifest = {
            "no_claim_of_single_combined_runner_pass": True,
            "reason": "coverage",
            "reuse_scope": "same production",
            **{
                key: str(self.original / name)
                for key, name in (
                    ("preserved_server", "server.log"),
                    ("preserved_launch_summary", "vllm_run_summary.json"),
                    ("sampling_log", "sampling_tests.log"),
                    ("qualitative_source", "vllm_qualitative_outputs.json"),
                )
            },
        }
        self.write(self.tracked / "completed_sampling_reuse.json", self.manifest)
        for folder in (self.original, self.tracked):
            self.write(folder / "vllm_run_summary.json", self.launch)
            self.write(folder / "server.log", "server evidence")
            for name in self.files:
                self.write(folder / name, "identical evidence")
        self.write(self.original / "execution/environment.json", self.env)
        self.write(self.original / "execution/sampling_qualitative.exit", "0")
        self.write(self.original / "execution/shutdown.log", "closed")
        self.write(self.original / "execution/commands.sh", "runner --sampling-profile full")
        self.write(
            self.original / "mixed_trace.json",
            {"checks": {"concurrent_decode_during_chunks": False, "greedy_exact_across_order_and_modes": True}},
        )
        native = {
            key: {"identity": "same"}
            for key in (
                "repository",
                "python_module",
                "loaded_libraries",
                "installed_sha256_declared_by_rebuild",
                "native_source_sha256",
                "rebuild_provenance_origin",
                "rebuild_provenance_sha256",
                "observation",
            )
        }
        for folder in (self.loaded, self.observer):
            self.write(folder / "native_sampling_provenance.json", native)
            self.write(folder / "native_main_rebuild_provenance.json", {"rebuild": "same"})
            self.write(folder / "native_sampling/kernel.cpp.source", "same kernel")
        for name in (
            "generator.py",
            "generator_vllm.py",
            "prefill_contract.py",
            "prefill_runtime.py",
            "sampling_state.py",
        ):
            self.write(self.loaded / f"{name}.source", "same model")
            self.write(self.model / "tt" / name, "same model")
        for name in ("generator.py", "tt_sampling.py"):
            self.write(self.loaded / f"common_{name}.source", "same common")
            self.write(self.root / "models/common/sampling" / name, "same common")

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))

    def validate(self):
        return self.ns["sampling_origin"](self.tracked, self.launch, self.env)

    def test_valid_reuse_records_distinct_origins(self):
        origin, evidence = self.validate()
        self.assertEqual(origin, self.original)
        self.assertTrue(evidence["no_claim_of_single_combined_runner_pass"])
        self.assertNotEqual(
            evidence["sampling_and_qualitative"]["server_log"], evidence["remaining_tracked_probes"]["server_log"]
        )

    def test_unchanged_default(self):
        fresh = self.model / "fresh"
        self.assertEqual(self.ns["sampling_origin"](fresh, self.launch, self.env), (fresh, None))

    def test_missing_manifest_rejected(self):
        (self.tracked / "completed_sampling_reuse.json").unlink()
        with self.assertRaisesRegex(ValueError, "explicit reuse manifest"):
            self.validate()

    def test_changed_reused_files_rejected(self):
        for name in self.files:
            with self.subTest(name=name):
                self.write(self.tracked / name, "changed")
                with self.assertRaisesRegex(ValueError, "Reused artifact changed"):
                    self.validate()
                self.write(self.tracked / name, "identical evidence")

    def test_changed_launch_rejected(self):
        self.launch["server_command"].append("--different")
        with self.assertRaisesRegex(ValueError, "Reuse launch changed"):
            self.validate()

    def test_changed_plugin_rejected(self):
        self.env["plugin_sha"] = "new"
        with self.assertRaisesRegex(ValueError, "Plugin changed"):
            self.validate()

    def test_changed_native_source_rejected(self):
        self.write(self.observer / "native_sampling/kernel.cpp.source", "different kernel")
        with self.assertRaisesRegex(ValueError, "Native source bytes changed"):
            self.validate()

    def test_changed_native_identity_rejected(self):
        path = self.observer / "native_sampling_provenance.json"
        record = json.loads(path.read_text())
        record["loaded_libraries"] = {"identity": "different"}
        self.write(path, record)
        with self.assertRaisesRegex(ValueError, "Native reuse identity changed"):
            self.validate()

    def test_changed_model_source_rejected(self):
        self.write(self.model / "tt/generator.py", "changed model")
        with self.assertRaisesRegex(ValueError, "Production source changed"):
            self.validate()

    def test_noncoverage_failure_rejected(self):
        self.write(
            self.original / "mixed_trace.json",
            {"checks": {"concurrent_decode_during_chunks": False, "greedy_exact_across_order_and_modes": False}},
        )
        with self.assertRaisesRegex(ValueError, "not limited"):
            self.validate()


if __name__ == "__main__":
    unittest.main()

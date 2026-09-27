# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct host tests for profile planning and observed configuration rejection."""

import ast
import copy
import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_server_config import MODEL, REVISION, profile_plan, validate_observed_config


def observed(slots):
    args = profile_plan(slots)["command"]
    return {
        "vllm_config": {
            "model_config": {
                "model": MODEL,
                "revision": REVISION,
                "tokenizer_revision": REVISION,
                "max_model_len": 262144,
            },
            "cache_config": {"block_size": 32, "enable_prefix_caching": False},
            "scheduler_config": {"max_num_seqs": slots, "enable_chunked_prefill": False, "async_scheduling": True},
            "additional_config": json.loads(args[args.index("--additional-config") + 1]),
        }
    }


class ServerConfigurationTests(unittest.TestCase):
    def test_profiles_preserve_context_and_pin_revisions(self):
        for slots in (1, 32):
            plan = profile_plan(slots)
            args = plan["command"]
            for name, value in (
                ("--max-num-seqs", str(slots)),
                ("--max-model-len", "262144"),
                ("--revision", REVISION),
                ("--tokenizer-revision", REVISION),
            ):
                self.assertEqual(args[args.index(name) + 1], value)
            self.assertIn("--no-enable-prefix-caching", args)
            self.assertEqual(validate_observed_config(observed(slots), slots)["max_num_seqs"], slots)

    def test_effective_configuration_route_enabled_for_both_profiles(self):
        installed = Path("/home/container_app_user/tt-metal/python_env/lib/python3.10/site-packages/vllm")
        source = installed / "entrypoints/openai/api_server.py"
        tree = ast.parse(source.read_text())
        gate = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "envs.VLLM_SERVER_DEV_MODE"
        )
        registration = installed / "entrypoints/serve/__init__.py"
        dev_tree = ast.parse(registration.read_text())
        function = next(
            node
            for node in dev_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "register_vllm_dev_api_routers"
        )
        package = "vllm.entrypoints.serve"
        names = ("cache", "rlhf", "rpc", "server_info", "sleep")
        modules = {}
        routes = []
        for name in names:
            stub = types.ModuleType(f"{package}.dev.{name}.api_router")
            stub.attach_router = lambda app, name=name: routes.append(name)
            modules[stub.__name__] = stub
        namespace = {
            "__package__": package,
            "FastAPI": object,
            "logger": types.SimpleNamespace(warning=lambda *args: None),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(registration), "exec"), namespace)
        serve = types.ModuleType(package)
        serve.register_vllm_dev_api_routers = namespace[function.name]
        modules[package] = serve
        for enabled in (False, True):
            routes.clear()
            with patch.dict("sys.modules", modules):
                exec(
                    compile(ast.Module(body=[gate], type_ignores=[]), str(source), "exec"),
                    {"envs": types.SimpleNamespace(VLLM_SERVER_DEV_MODE=enabled), "app": object()},
                )
            self.assertEqual("server_info" in routes, enabled)
        for slots in (1, 32):
            plan = profile_plan(slots)
            self.assertEqual(plan["environment_overrides"].get("VLLM_SERVER_DEV_MODE"), "1")
            args = plan["command"]
            self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")

    def test_drift_rejected(self):
        cases = [
            ("scheduler_config", "max_num_seqs", 32),
            ("model_config", "max_model_len", 8192),
            ("model_config", "tokenizer_revision", None),
            ("cache_config", "enable_prefix_caching", True),
        ]
        for section, key, value in cases:
            evidence = copy.deepcopy(observed(1))
            evidence["vllm_config"][section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_observed_config(evidence, 1)

    def test_wrong_trace_rejected(self):
        evidence = observed(1)
        evidence["vllm_config"]["additional_config"]["tt"]["trace_mode"] = "all"
        with self.assertRaises(ValueError):
            validate_observed_config(evidence, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

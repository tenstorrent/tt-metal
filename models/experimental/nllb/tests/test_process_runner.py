# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the subprocess data/code boundary; no native device."""

import argparse
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from models.experimental.nllb.tests import process_runner as runner


class ProcessBoundaryTests(unittest.TestCase):
    def test_untrusted_values_never_enter_command(self):
        value = "--device=99; $(touch NEVER_CREATED) 'quoted'\ntext"
        options = {"checkpoint": value, "device": 0, "precision": "bf16"}
        result = SimpleNamespace(returncode=0)
        with patch.object(runner.subprocess, "run", return_value=result) as launch:
            self.assertIs(runner.run_task("reusable", options, timeout=30), result)
        args, kwargs = launch.call_args
        self.assertEqual(args[0], [sys.executable, "-m", "models.experimental.nllb.tests.process_runner"])
        self.assertNotIn(value, args[0])
        self.assertFalse(kwargs["shell"])
        self.assertEqual(json.loads(kwargs["input"]), {"task": "reusable", "options": options})
        self.assertEqual(kwargs["cwd"], Path(runner.__file__).resolve().parents[4])

    def test_unknown_task_and_options_rejected_before_launch(self):
        requests = [
            ("os.system", {}),
            ("component", {"component": "../../evil.py", "device": 0}),
            ("component", {"component": "fused_attention", "device": 0, "module": "os"}),
            ("reusable", {"checkpoint": "local", "device": 0, "precision": "bf16", "command": "bad"}),
        ]
        for task, options in requests:
            with self.subTest(task=task, options=options), patch.object(runner.subprocess, "run") as launch:
                with self.assertRaises(ValueError):
                    runner.run_task(task, options, timeout=30)
                launch.assert_not_called()

    def test_scalar_validation(self):
        valid = dict(checkpoint="local", device=0, precision="bf16")
        for field, values in {
            "checkpoint": [None, "", "bad\0path", ["local"]],
            "device": [True, -1, 256, 0.5, "0;echo bad"],
            "precision": ["fp8", "--help"],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    runner.validate_request({"task": "reusable", "options": dict(valid, **{field: value})})

    def test_invalid_timeouts_and_oversize_payload_never_launch(self):
        with patch.object(runner.subprocess, "run") as launch:
            for timeout in [True, 0, -1, 1201, float("nan"), float("inf"), "10"]:
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    runner.run_task("component", {"component": "precision_storage", "device": 0}, timeout=timeout)
            with self.assertRaises(ValueError):
                runner.run_task("reusable", {"checkpoint": "a" * 16385, "device": 0, "precision": "bf16"}, timeout=30)
            launch.assert_not_called()

    def test_failure_and_timeout_propagate(self):
        for error in [runner.subprocess.TimeoutExpired("fixed-runner", 30), OSError("cannot start")]:
            with self.subTest(error=type(error)), patch.object(runner.subprocess, "run", side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    runner.run_task("component", {"component": "generation_projection", "device": 0}, timeout=30)
                self.assertIs(caught.exception, error)

    def test_child_revalidates_and_rejects_duplicate_or_oversize_data(self):
        payloads = [
            b'{"task":"component","task":"os.system","options":{}}',
            b'{"task":"component","options":{"component":"os.system","device":0}}',
            b"x" * (runner.MAX_REQUEST_BYTES + 1),
            b"[]",
        ]
        for payload in payloads:
            with self.subTest(payload=payload[:80]), patch.object(
                runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(payload))
            ):
                with self.assertRaises(ValueError):
                    runner.main()

    def test_all_public_tasks_have_literal_dispatch(self):
        from models.experimental.nllb.tt import runtime_setup

        cases = {
            "translate": (
                "models.experimental.nllb.demo.translate",
                dict(
                    checkpoint="local",
                    config="config",
                    device=0,
                    source_language="eng_Latn",
                    target_language="fra_Latn",
                    text="--device=99; bad",
                    max_new_tokens=4,
                    precision="bf16",
                    output="out",
                ),
            ),
            "reusable": (
                "models.experimental.nllb.tests.test_reusable_port",
                dict(checkpoint="local", device=0, precision="bf16"),
            ),
            "packed": (
                "models.experimental.nllb.tests.probe_packed_integration",
                dict(checkpoint="local", config="config", tokenizer_directory="tokens", device=0, output="out"),
            ),
            "envelope": (
                "models.experimental.nllb.reference.envelope_regression",
                dict(checkpoint="local", device=0, output="out", precision="bf16", timeout=30),
            ),
            "trained": (
                "models.experimental.nllb.tests.test_trained_masks_recovery",
                dict(checkpoint="local", device=0, precision="bf16", mode="masks"),
            ),
        }
        for task, (module, options) in cases.items():
            calls = []
            stub = SimpleNamespace(main=lambda argv: calls.append(argv) or 0, cli=lambda argv: calls.append(argv) or 0)
            with self.subTest(task=task), patch.dict(sys.modules, {module: stub}), patch.object(
                runtime_setup, "configure_tracking"
            ):
                self.assertEqual(runner.dispatch({"task": task, "options": options}), 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], ["--" + k.replace("_", "-") + "=" + str(v) for k, v in options.items()])

    def test_translation_option_looking_text_is_data(self):
        from models.experimental.nllb.demo import translate

        options = dict(
            checkpoint="local",
            config="config",
            device=0,
            source_language="eng_Latn",
            target_language="fra_Latn",
            text="--device=99; bad",
            max_new_tokens=4,
            precision="bf16",
            output="out",
        )
        original = argparse.ArgumentParser.parse_args
        parsed = []
        stop = RuntimeError("stop after parsing, before importing native runtime")

        def capture(parser, args=None, namespace=None):
            parsed.append(original(parser, args, namespace))
            raise stop

        with patch.object(argparse.ArgumentParser, "parse_args", capture):
            with self.assertRaises(RuntimeError) as caught:
                translate.main(["--" + k.replace("_", "-") + "=" + str(v) for k, v in options.items()])
        self.assertIs(caught.exception, stop)
        self.assertEqual(parsed[0].text, [options["text"]])
        self.assertEqual(parsed[0].device, 0)

    def test_each_component_uses_fixed_check_and_closes_owner(self):
        from models.experimental.nllb.tt import runtime_setup

        for component in ("fused_attention", "generation_projection", "precision_storage"):
            owner = MagicMock()
            device = object()
            owner.__enter__.return_value = owner
            owner.open.return_value = device
            check = MagicMock()
            module = "models.experimental.nllb.tests.test_" + component
            with self.subTest(component=component), patch.dict(
                sys.modules, {module: SimpleNamespace(check=check), "torch": MagicMock()}
            ), patch.object(runtime_setup, "configure_tracking"), patch.object(
                runtime_setup, "RuntimeOwner", return_value=owner
            ):
                self.assertEqual(
                    runner.dispatch({"task": "component", "options": {"component": component, "device": 3}}), 0
                )
            owner.open.assert_called_once_with(3)
            check.assert_called_once_with(device)
            owner.__exit__.assert_called_once_with(None, None, None)

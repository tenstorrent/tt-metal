# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of actual profile helper control flow, without TTNN import."""

import argparse
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class ProfileScopeTest(unittest.TestCase):
    def test_capacity_parser_and_propagation(self):
        source = Path(__file__).with_name("profile_full_model_terminal.py")
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "profile_capacity")
        context = {"argparse": argparse}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), context)
        parser = argparse.ArgumentParser()
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        capacity_argument = next(
            n
            for n in ast.walk(main)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_argument"
            and n.args
            and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == "--capacity"
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=capacity_argument)], type_ignores=[])),
                str(source),
                "exec",
            ),
            {**context, "parser": parser},
        )
        self.assertEqual(parser.parse_args([]).capacity, 128)
        for value in (128, 256, 262144):
            args = parser.parse_args(["--capacity", str(value)])
            for name, key in (("report", "capacity"), ("options", "max_seq_len")):
                assignment = next(
                    n
                    for n in ast.walk(main)
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
                )
                expression = next(
                    v
                    for k, v in zip(assignment.value.keys, assignment.value.values)
                    if isinstance(k, ast.Constant) and k.value == key
                )
                actual = eval(compile(ast.Expression(body=expression), str(source), "eval"), {"args": args})
                self.assertEqual(actual, value)
        for value in (0, 1, 127, 129, 262145, 262272):
            with self.assertRaises(argparse.ArgumentTypeError):
                context["profile_capacity"](str(value))

    def helpers(self):
        source = Path(__file__).with_name("profile_full_model_terminal.py")
        tree = ast.parse(source.read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("prepare_eager_profile", "time_region")
        ]
        events = []
        context = {
            "ttnn": SimpleNamespace(
                full=lambda shape, fill, **kw: (events.append(("allocate", shape, fill)), "inactive")[1],
                bfloat16="bf16",
                TILE_LAYOUT="tile",
                synchronize_device=lambda mesh: events.append("sync"),
                ReadDeviceProfiler=lambda mesh: events.append("drain"),
            ),
            "sampler_body": lambda gen, sampler: events.append(("sample", sampler.name)),
            "signpost": lambda label: events.append(label),
            "time": SimpleNamespace(perf_counter=lambda: 1.0),
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), context)
        return context, events

    def test_exact_eager_warmup_requires_inactive_rows_and_seed_initialization(self):
        helpers, events = self.helpers()
        gen = SimpleNamespace(
            prepared=False,
            max_batch_size=1,
            factored_decode=None,
            traces={},
            prefill_traces={},
            sampling_traces={},
            mesh="mesh",
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=248320)),
            sample_logits="logits",
            decode_tokens="tokens",
            _check_cache_setup=lambda: events.append("cache_check"),
            seed_state=SimpleNamespace(prepare=lambda: events.append("seed_prepare")),
            token_feedback=SimpleNamespace(prepare=lambda: events.append("feedback_prepare")),
            reset=lambda: events.append("reset"),
        )

        def decode(slots):
            self.assertEqual(slots, (0,))
            self.assertEqual(gen.inactive_logits, "inactive")
            self.assertIn("seed_prepare", events)
            self.assertIn("feedback_prepare", events)
            self.assertIn("reset", events)
            events.append("decode")

        gen._decode_body = decode
        samplers = []
        for name in ("split", "force"):

            def precompile(logits, *, tt_out_tok, all_configs, compile_token_update, name=name):
                self.assertEqual((logits, tt_out_tok), ("logits", "tokens"))
                self.assertFalse(all_configs)
                self.assertTrue(compile_token_update)
                events.append(("compile", name))

            samplers.append(SimpleNamespace(name=name, precompile=precompile))
        # No prepare/prefill/remap methods exist on this strict fake: expanding
        # scope or using the public serving preparation would fail this test.
        helpers["prepare_eager_profile"](gen, samplers)
        self.assertFalse(gen.prepared)
        self.assertEqual(events.count("decode"), 1)
        self.assertEqual(events.count("drain"), 4)
        self.assertIn(("allocate", (1, 1, 1, 62080), -100.0), events)
        for name in ("split", "force"):
            self.assertLess(events.index(("compile", name)), events.index(("sample", name)))

    def test_profiler_drains_warmups_outside_measurement(self):
        helpers, events = self.helpers()
        helpers["time_region"]("mesh", lambda: events.append("op"), 2, warmup=2, label="PERF", profiler=True)
        start, end = events.index("PERF"), events.index("PERF_END")
        self.assertEqual(events[:start].count("drain"), 3)
        self.assertEqual(events[start:end].count("op"), 2)
        self.assertNotIn("drain", events[start:end])
        self.assertEqual(events[end + 1 :], ["drain"])

    def test_eager_profile_rejects_batched_or_factored_scope_before_ops(self):
        for batch, factored in ((2, None), (1, object())):
            helpers, events = self.helpers()
            gen = SimpleNamespace(prepared=False, max_batch_size=batch, factored_decode=factored)
            with self.assertRaises(AssertionError):
                helpers["prepare_eager_profile"](gen, [])
            self.assertEqual(events, [])

    def test_clean_timing_has_no_profiler_reads(self):
        helpers, events = self.helpers()
        helpers["time_region"]("mesh", lambda: events.append("op"), 2, warmup=2)
        self.assertNotIn("drain", events)
        self.assertEqual(events.count("op"), 4)


if __name__ == "__main__":
    unittest.main()

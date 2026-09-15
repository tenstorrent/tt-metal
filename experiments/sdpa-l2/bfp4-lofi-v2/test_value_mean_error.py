# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standard-library replay/immutability tests; no Torch, TTNN or device import.

AST-load the actual driver helpers with fake tensor hashes/trace operations.
This checks host control flow, NOT numerical correctness or hardware replay.
"""
import ast
from pathlib import Path
import unittest


DRIVER = Path(__file__).with_name("value_mean_error_fullchip.py")


class TraceDouble:
    def __init__(self, case, output, originals, inputs):
        self.case = case
        self.output = output
        self.originals = originals
        self.inputs = inputs
        self.replays = 0
        self.releases = 0
        self.captures = 0
        self.ends = 0

    def to_torch(self, tensor):
        return tensor[0]

    def begin_trace_capture(self, device, cq_id):
        self.captures += 1
        return 17

    def end_trace_capture(self, device, trace, cq_id):
        assert trace == 17
        self.ends += 1

    def execute_trace(self, device, trace, cq_id, blocking):
        assert trace == 17 and cq_id == 0 and blocking is True
        self.replays += 1
        if self.replays == 2:
            if self.case == "output":
                self.output[0] = "changed-output-bits"
            elif self.case == "device":
                self.originals[0][0] = "changed-device-input-bits"
            elif self.case == "cpu":
                self.inputs[0] = "changed-cpu-input-bits"

    def release_trace(self, device, trace):
        assert trace == 17
        self.releases += 1


def load_helpers(fake):
    tree = ast.parse(DRIVER.read_text())
    names = {"check_input_immutability", "qualify_trace_replays"}
    selected = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    assert len(selected.body) == len(names)
    namespace = {"ttnn": fake, "tensor_sha256": lambda tensor: tensor}
    exec(compile(selected, str(DRIVER), "exec"), namespace)
    return namespace


class ReplayTests(unittest.TestCase):
    def run_case(self, case):
        originals = [["q"], ["k"], ["v"]]
        inputs = ["q", "k", "v"]
        output = ["output"]
        fake = TraceDouble(case, output, originals, inputs)
        helpers = load_helpers(fake)
        captured_invocations = []

        def combined():
            captured_invocations.append(1)

        def qualify():
            return helpers["qualify_trace_replays"](
                None, combined, output, "output", originals, inputs, ["q", "k", "v"]
            )

        if case == "pass":
            result = qualify()
            self.assertEqual(result["explicit_combined_trace_replays"], 2)
            self.assertEqual(result["invocations_per_replay"], 1)
            self.assertEqual(result["replay_output_sha256"], ["output", "output"])
            for key in ("output_bitwise_equal", "cpu_inputs_unchanged", "device_inputs_unchanged"):
                self.assertIs(result[key], True)
        else:
            message = {
                "output": "output bits",
                "cpu": "CPU BF16 inputs changed",
                "device": "device BF16 inputs differ",
            }[case]
            with self.assertRaisesRegex(AssertionError, message):
                qualify()
        self.assertEqual(fake.captures, 1)
        self.assertEqual(fake.ends, 1)
        self.assertEqual(fake.replays, 2)
        self.assertEqual(fake.releases, 1)
        self.assertEqual(captured_invocations, [1])

    def test_two_replays_without_timing_configuration(self):
        # The helper has no args/iters input: correctness does not depend on timing.
        self.run_case("pass")

    def test_second_replay_output_corruption_rejected_and_trace_released(self):
        self.run_case("output")

    def test_cpu_original_mutation_rejected(self):
        self.run_case("cpu")

    def test_device_original_mutation_rejected(self):
        self.run_case("device")

    def test_main_calls_qualification_unconditionally_before_timing(self):
        tree = ast.parse(DRIVER.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        block = next(node for node in main.body if isinstance(node, ast.Try)).body
        qualification = [
            index for index, node in enumerate(block)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "qualify_trace_replays"
        ]
        timing = [
            index for index, node in enumerate(block)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "timings" for target in node.targets)
        ]
        self.assertEqual(len(qualification), 1)
        self.assertEqual(len(timing), 1)
        self.assertLess(qualification[0], timing[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Check the bridge against the target source without loading a device runtime."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "pipelines/minimax_h3"


def klass(path, name):
    return next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == name
    )


class TargetContractTests(unittest.TestCase):
    def test_renderer_keywords_fit_target_call(self):
        target = klass(ROOT / "pipeline_minimax_h3.py", "MiniMaxH3Pipeline")
        signature = next(node for node in target.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
        accepted = {arg.arg for arg in signature.args.args + signature.args.kwonlyargs}
        bridge = klass(ROOT / "streaming_pipeline.py", "PipelineRenderer")
        calls = [
            node
            for node in ast.walk(bridge)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "pipeline"
        ]
        self.assertEqual(len(calls), 1)
        self.assertTrue({kw.arg for kw in calls[0].keywords} <= accepted)

    def test_rolling_reference_keywords_fit_target_dataclass(self):
        target = klass(ROOT / "packing_ref2va.py", "MiniMaxH3Reference")
        accepted = {node.target.id for node in target.body if isinstance(node, ast.AnnAssign)}
        bridge = klass(ROOT / "streaming_pipeline.py", "PipelineRenderer")
        calls = [
            node
            for node in ast.walk(bridge)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "reference_factory"
        ]
        self.assertEqual(len(calls), 1)
        self.assertTrue({kw.arg for kw in calls[0].keywords} <= accepted)

    def test_target_output_carries_bridge_metadata(self):
        target = klass(ROOT / "pipeline_minimax_h3.py", "MiniMaxH3Output")
        accepted = {node.target.id for node in target.body if isinstance(node, ast.AnnAssign)}
        bridge = klass(ROOT / "streaming_pipeline.py", "PipelineRenderer")
        used = {
            node.attr
            for node in ast.walk(bridge)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "out"
        }
        self.assertTrue(used)
        self.assertTrue(used <= accepted)


if __name__ == "__main__":
    unittest.main()

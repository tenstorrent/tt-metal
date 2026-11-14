# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import torch

from models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch import run_case


class HostGenerator:
    model = SimpleNamespace(precision_config={"config_id": "host_test"})
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "x" * 1200)
    logits = None

    def __init__(self):
        self.calls = []

    def _host_logits(self, unused):
        return torch.tensor([[0.0, 1.0, 2.0]])

    def generate(self, prompt, length, **kwargs):
        self.calls.append(kwargs)
        for i in range(length):
            if kwargs.get("next_input"):
                kwargs["next_input"](i, 2)
        return [2] * length


def test_shared_generator_branches_probabilities_and_seeded_trials(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "prompt_token_ids": [1],
                "continuations": {"chosen": [2, 2], "correct": [1]},
                "metadata": {"sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95}, "seed": 42},
            }
        )
    )
    generator = HostGenerator()
    job = {"case": str(path), "branches": ["chosen"], "free_run_tokens": 4, "free_run_trials": 3}
    first = tmp_path / "served.json"
    run_case(generator, job, first)
    data = json.loads(first.read_text())
    assert list(data["continuations"]) == ["chosen"]
    assert data["continuations"]["chosen"]["positions"][0]["forced_token_probability"] > 0.66
    assert [r["sampling"]["seed"] for r in data["free_running"]] == [42, 43, 44]
    assert all(r["flag"] for r in data["free_running"])
    assert all(c["enable_trace"] for c in generator.calls)
    assert all("host_sampling" not in c for c in generator.calls[1:])
    assert first.stat().st_mode & 0o777 == 0o600
    second = tmp_path / "full.json"
    run_case(generator, job, second, first)
    assert json.loads(second.read_text())["comparison"]["chosen"]["top1_agreement"] == 1.0


def test_cleanup_executes_actual_generator_release_api_then_closes():
    import ast
    from pathlib import Path

    from models.autoports.google_gemma_4_31b_it.tools.precision_repro_probe import close_generator_mesh

    source = Path(__file__).parents[1] / "tt/generator.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Gemma4Generator")
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "teardown" for n in cls.body)
    release = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "release_persistent_capture")
    events = []
    env = {
        "ttnn": SimpleNamespace(release_trace=lambda mesh, trace: events.append((mesh, trace))),
        "json": json,
        "print": lambda *args, **kwargs: None,
    }
    exec(compile(ast.Module(body=[release], type_ignores=[]), str(source), "exec"), env)
    obj = SimpleNamespace(mesh="mesh", model_trace=1, sample_trace=2, buffered_sample_trace=None)
    obj.release_persistent_capture = lambda: env["release_persistent_capture"](obj)
    close_generator_mesh(obj, "mesh", lambda mesh, fabric: events.append((mesh, fabric)))
    assert events == [("mesh", 1), ("mesh", 2), ("mesh", "FABRIC_1D_RING")]
    assert obj.model_trace is obj.sample_trace is None


def test_cleanup_closes_mesh_even_if_trace_release_raises(expect_error):
    from models.autoports.google_gemma_4_31b_it.tools.precision_repro_probe import close_generator_mesh

    def fail():
        raise RuntimeError("release failure")

    closed = []
    with expect_error(RuntimeError, "release failure"):
        close_generator_mesh(
            SimpleNamespace(release_persistent_capture=fail), "mesh", lambda *args: closed.append(args)
        )
    assert closed == [("mesh", "FABRIC_1D_RING")]

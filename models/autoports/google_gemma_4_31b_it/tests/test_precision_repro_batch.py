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
        self.counters = {"model_replays": 0, "sample_replays": 0, "readbacks": 0}

    def _replay(self):
        for key in self.counters:
            self.counters[key] += 1
        return 2

    def _host_logits(self, unused):
        return torch.tensor([[0.0, 1.0, 2.0]])

    def generate(self, prompt, length, **kwargs):
        self.calls.append(kwargs)
        for i in range(length):
            if kwargs.get("next_input"):
                kwargs["next_input"](i, 2)
            elif i > 0:
                self._replay()
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


def test_decode_timing_excludes_prefill_and_includes_host_gaps():
    from models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch import timed_free_generate

    class Clocked(HostGenerator):
        now = 0.0

        def _replay(self):
            self.now += 2.0  # existing synchronous completion
            return super()._replay()

        def generate(self, prompt, length, **kwargs):
            self.now += 100.0  # prefill and its first output token
            tokens = [2]
            for _ in range(length - 1):
                tokens.append(self._replay())
                self.now += 3.0  # Python loop work, including after last completion
            return tokens

    generator = Clocked()
    ids, timing = timed_free_generate(generator, [1], 4, {}, clock=lambda: generator.now)
    assert len(ids) == 4
    assert timing["decode_tokens"] == 3
    assert timing["decode_wall_s"] == 12.0  # 2+3+2+3+2, excludes 100 prefill and last gap
    assert timing["decode_tok_s_user"] == 0.25
    assert timing["trace_counters_delta"] == {"model_replays": 3, "sample_replays": 3, "readbacks": 3}
    assert "_replay" not in generator.__dict__


def test_decode_timing_restores_override_on_generation_failure(expect_error):
    from models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch import timed_free_generate

    generator = HostGenerator()
    original = generator._replay
    generator._replay = original

    def fail(*args, **kwargs):
        raise RuntimeError("generate failed")

    generator.generate = fail
    with expect_error(RuntimeError, "generate failed"):
        timed_free_generate(generator, [1], 4, {})
    assert generator._replay is original


def test_chosen_only_case_validates_without_invented_correct_branch(tmp_path, monkeypatch, expect_error):
    from models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch import main

    case = tmp_path / "case.json"
    case.write_text(json.dumps({"prompt_token_ids": [1], "continuations": {"chosen": [2]}}))
    jobs = tmp_path / "jobs.json"
    job = {"name": "onset", "case": str(case), "branches": ["chosen"]}
    jobs.write_text(json.dumps([job]))
    kwargs = tmp_path / "kwargs.json"
    kwargs.write_text(json.dumps({"batch_size": 1}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--cases",
            str(jobs),
            "--build-kwargs-file",
            str(kwargs),
            "--output-dir",
            str(tmp_path / "out"),
            "--validate-only",
        ],
    )
    main()
    job["branches"].append("correct")
    jobs.write_text(json.dumps([job]))
    with expect_error(ValueError, "Missing branch"):
        main()

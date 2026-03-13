# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json

from models.demos.deepseek_v3.demo import demo as demo_module
from models.demos.deepseek_v3.demo.make_lmeval_prompts import resolve_task_name as resolve_prompt_task_name
from models.demos.deepseek_v3.demo.score_lmeval_outputs import extract_gpqa_choice, load_generations_by_index


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token) for token in token_ids)


class _FakeMeshDevice:
    def get_submeshes(self):
        return []


class _FakeGenerator:
    def __init__(self, **kwargs):
        self.tokenizer = kwargs["tokenizer"]
        self.generate_kwargs = None

    def generate(self, prompts, **kwargs):
        self.generate_kwargs = kwargs
        on_user_finished = kwargs["on_user_finished"]
        if on_user_finished is not None:
            on_user_finished(0, [11, 12])
        return [[11, 12], [21, 22, 23]], {"decode_t/s": 1.0}

    def cleanup_all(self):
        return None


def test_create_parser_stop_at_eos_flag():
    args = demo_module.create_parser().parse_args(["--model-path", "/tmp/model", "--cache-dir", "/tmp/cache"])
    assert args.stop_at_eos is False

    args = demo_module.create_parser().parse_args(
        ["--model-path", "/tmp/model", "--cache-dir", "/tmp/cache", "--stop-at-eos"]
    )
    assert args.stop_at_eos is True


def test_run_demo_writes_checkpoint_jsonl(monkeypatch, tmp_path):
    fake_generator = _FakeGenerator(tokenizer=_FakeTokenizer())

    monkeypatch.setenv("MESH_DEVICE", "TG")
    monkeypatch.setattr(demo_module, "validate_model_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo_module, "load_tokenizer", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr(demo_module, "system_name_to_mesh_shape", lambda *args, **kwargs: (1, 1))
    monkeypatch.setattr(demo_module, "get_fabric_config", lambda: "fake-fabric")
    monkeypatch.setattr(demo_module.ttnn, "set_fabric_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo_module.ttnn, "open_mesh_device", lambda *args, **kwargs: _FakeMeshDevice())
    monkeypatch.setattr(demo_module.ttnn, "synchronize_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo_module.ttnn, "close_mesh_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(demo_module, "DeepseekGeneratorDP", lambda **kwargs: fake_generator)

    checkpoint_path = tmp_path / "partial.jsonl"
    result = demo_module.run_demo(
        prompts=["prompt-0", "prompt-1"],
        model_path=tmp_path / "model",
        cache_dir=tmp_path / "cache",
        checkpoint_jsonl=checkpoint_path,
        stop_at_eos=True,
    )

    assert fake_generator.generate_kwargs["stop_at_eos"] is True
    assert callable(fake_generator.generate_kwargs["on_user_finished"])
    assert result["generations"][0]["text"] == "11 12"
    assert result["generations"][1]["text"] == "21 22 23"

    lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"index": 1, "prompt": "prompt-0", "text": "11 12"},
        {"index": 2, "prompt": "prompt-1", "text": "21 22 23"},
    ]


def test_lmeval_helpers():
    task_names = {"aime24", "gpqa_diamond"}
    assert resolve_prompt_task_name("aime24", task_names) == "aime24"
    assert resolve_prompt_task_name("r1_aime24", task_names) == "aime24"
    assert resolve_prompt_task_name("r1_gpqa_diamond", task_names) == "gpqa_diamond"
    assert resolve_prompt_task_name("missing", task_names) is None

    assert extract_gpqa_choice(r"Some work \boxed{b}") == "(B)"
    assert extract_gpqa_choice("The answer is c") == "(C)"


def test_load_generations_by_index_jsonl(tmp_path):
    output_path = tmp_path / "partial.jsonl"
    output_path.write_text(
        json.dumps({"index": 2, "text": "second"}) + "\n" + json.dumps({"index": 1, "text": "first"}) + "\n",
        encoding="utf-8",
    )

    assert load_generations_by_index(output_path) == {1: "first", 2: "second"}

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the LTX prompt-enhancer scaffold (no device)."""

import json
import os

import pytest
import torch

from models.tt_dit.pipelines.ltx import prompt_enhancer as pe
from models.tt_dit.pipelines.ltx.prompt_enhancer import (
    DEFAULT_ENHANCER_PATH,
    ENHANCER_MAX_NEW_TOKENS,
    GEMMA4_EOS_IDS,
    I2V_SYSTEM_PROMPT,
    T2V_SYSTEM_PROMPT,
    DevicePromptEnhancer,
    HostPromptEnhancer,
    PromptEnhancer,
    apply_prompt_enhancer,
    bind_prompt_enhancer_mesh,
    build_messages,
    build_prompt_enhancer,
    resolve_enhancer_cache_dir,
    resolve_stop_tokens,
)


class _FakeEnhancer(PromptEnhancer):
    name = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.loads = 0
        self.calls = []

    def ensure_loaded(self) -> None:
        self.loads += 1

    def is_loaded(self) -> bool:
        return self.loads > 0

    def enhance(self, prompt, *, mode="t2v", image_path=None, seed=10):
        self.calls.append((prompt, mode, image_path, seed))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.mark.parametrize("setting", [None, "", "0", "off", "OFF", "none", "false", "  "])
def test_build_prompt_enhancer_off(setting):
    assert build_prompt_enhancer(setting) is None


def test_build_prompt_enhancer_host_defaults(monkeypatch):
    monkeypatch.delenv("LTX_ENHANCER_PATH", raising=False)
    enhancer = build_prompt_enhancer("host")
    assert isinstance(enhancer, HostPromptEnhancer)
    assert enhancer.model_path == DEFAULT_ENHANCER_PATH
    assert enhancer.max_new_tokens == ENHANCER_MAX_NEW_TOKENS
    assert enhancer.name == f"host:{DEFAULT_ENHANCER_PATH}"


def test_build_prompt_enhancer_host_path_resolution(monkeypatch):
    monkeypatch.setenv("LTX_ENHANCER_PATH", "/weights/from-env")
    assert build_prompt_enhancer("host").model_path == "/weights/from-env"
    assert build_prompt_enhancer("host", model_path="/weights/explicit").model_path == "/weights/explicit"


def test_build_prompt_enhancer_device_is_unbound_by_default(expect_error):
    enhancer = build_prompt_enhancer("device")
    assert isinstance(enhancer, DevicePromptEnhancer)
    assert enhancer.mesh_device is None
    assert enhancer.name == f"device:{enhancer.model_path}"
    with expect_error(RuntimeError, "no mesh handle"):
        enhancer.ensure_loaded()


def test_build_prompt_enhancer_rejects_unknown_backend(expect_error):
    with expect_error(ValueError, "LTX_PROMPT_ENHANCER"):
        build_prompt_enhancer("gpu")


@pytest.mark.parametrize("mode,system", [("t2v", T2V_SYSTEM_PROMPT), ("i2v", I2V_SYSTEM_PROMPT)], ids=["t2v", "i2v"])
def test_build_messages(mode, system):
    messages = build_messages("a cat on a windowsill", mode)
    assert messages == [
        {"role": "system", "content": system},
        {"role": "user", "content": "user prompt: a cat on a windowsill"},
    ]


def test_build_messages_rejects_unknown_mode(expect_error):
    with expect_error(ValueError, "unknown enhance mode"):
        build_messages("a cat", "v2v")


def test_apply_is_identity_without_enhancer():
    assert apply_prompt_enhancer(None, "a cat") == "a cat"


def test_apply_rewrites_and_routes_mode_by_image():
    fake = _FakeEnhancer("Style: realistic. A cat is sitting on a windowsill as birds chirp outside.")
    out = apply_prompt_enhancer(fake, "a cat", image_path="/frames/first.png", seed=3)
    assert out == fake.reply
    assert fake.calls == [("a cat", "i2v", "/frames/first.png", 3)]
    assert fake.loads == 1

    apply_prompt_enhancer(fake, "a dog", seed=4)
    assert fake.calls[-1] == ("a dog", "t2v", None, 4)

    # An explicit mode wins over the image-derived default.
    apply_prompt_enhancer(fake, "a bird", image_path="/frames/first.png", mode="t2v", seed=5)
    assert fake.calls[-1] == ("a bird", "t2v", "/frames/first.png", 5)


def test_apply_falls_back_to_raw_prompt_on_empty_rewrite():
    fake = _FakeEnhancer("  \n")
    assert apply_prompt_enhancer(fake, "a cat") == "a cat"


def test_apply_falls_back_to_raw_prompt_when_too_long():
    # The rewriter's context limit must never fail a request the encoder could still take.
    fake = _FakeEnhancer(pe.PromptTooLongError("prompt of 9000 tokens does not fit"))
    assert issubclass(pe.PromptTooLongError, ValueError)
    assert apply_prompt_enhancer(fake, "a very long cat") == "a very long cat"
    assert fake.calls == [("a very long cat", "t2v", None, pe.ENHANCER_SEED)]


@pytest.mark.skipif(not os.environ.get("LTX_ENHANCER_SMOKE"), reason="set LTX_ENHANCER_SMOKE=1 to run the CPU rewriter")
def test_host_enhancer_smoke():
    enhancer = HostPromptEnhancer(max_new_tokens=48)
    out = enhancer.enhance("a cat on a windowsill", seed=10)
    assert isinstance(out, str) and out
    assert out != "a cat on a windowsill"
    # Same seed, same text: the rewrite is reproducible per (prompt, seed).
    assert enhancer.enhance("a cat on a windowsill", seed=10) == out


# --- device backend through fakes ------------------------------------------------------------------

VOCAB = 16
BOS, EOS = 2, 1
PROMPT_IDS = [BOS, 3, 4, 5]


class _FakeTensor:
    """Stands in for ttnn.Tensor: tracks whether it is allocated."""

    def __init__(self):
        self.allocated = True

    def is_allocated(self):
        return self.allocated

    def deallocate(self, force=False):
        assert self.allocated, "double deallocate"
        self.allocated = False


class _FakeLayer:
    __module__ = "models.fake.layer"

    def __init__(self):
        self.weight = _FakeTensor()
        self.kv_cache = [_FakeTensor(), _FakeTensor()]


class _FakeModel:
    __module__ = "models.fake.model"

    def __init__(self):
        self.layers = [_FakeLayer(), _FakeLayer()]
        self.layers[0].parent = self  # a reference cycle, like the real model graph


class _FakeTokenizer:
    bos_token = "<bos>"
    bos_token_id = BOS
    eos_token_id = EOS
    stop_tokens = [EOS]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "<bos>" + messages[-1]["content"]

    def encode(self, text, add_special_tokens=True):
        return list(PROMPT_IDS)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)


class _FakeGenerator:
    """Emits a scripted token per step through its logits; records every call."""

    __module__ = "models.fake.generator"

    def __init__(self, script, *, peaked=True):
        self.script = list(script)
        self.peaked = peaked
        self.model = [_FakeModel()]
        self.model_args = ["args"]
        self.prefill_calls = []
        self.decode_calls = []
        self.released = 0

    def _logits(self, step):
        logits = torch.zeros(1, 1, VOCAB)
        if self.peaked:
            logits[0, 0, self.script[min(step, len(self.script) - 1)]] = 50.0
        else:
            logits[0, 0] = torch.linspace(0.0, 1.5, VOCAB)
        return logits

    def prefill_forward_text(self, tokens, **kwargs):
        self.prefill_calls.append((tokens.clone(), kwargs))
        return self._logits(0)

    def decode_forward(self, out_tok, current_pos, **kwargs):
        self.decode_calls.append((int(out_tok.reshape(-1)[0]), int(current_pos.reshape(-1)[0]), dict(kwargs)))
        return self._logits(len(self.decode_calls)), None

    def release_persistent_capture(self):
        self.released += 1


def _device_enhancer(monkeypatch, script, *, peaked=True, tokenizer=None, real_tokenize=False, **kwargs):
    gen = _FakeGenerator(script, peaked=peaked)
    kv = [[layer.kv_cache for layer in gen.model[0].layers]]
    tok = tokenizer or _FakeTokenizer()
    monkeypatch.setattr(DevicePromptEnhancer, "_build_generator", lambda self: (gen, kv, tok))
    if not real_tokenize:
        monkeypatch.setattr(DevicePromptEnhancer, "_tokenize_prompt", lambda self, text, n: list(PROMPT_IDS))
    monkeypatch.setattr(DevicePromptEnhancer, "_device_tensor_type", staticmethod(lambda: _FakeTensor))
    enhancer = DevicePromptEnhancer("/fake/gemma", mesh_device="mesh", **kwargs)
    return enhancer, gen


class _PrefillPreprocessRecorder:
    """Stands in for ``preprocess_inputs_prefill``: returns the scripted ids and records the call."""

    def __init__(self, ids):
        self.ids = list(ids)
        self.calls = []

    def __call__(self, prompts, tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=None):
        self.calls.append(
            dict(
                prompts=prompts,
                tokenizer=tokenizer,
                model_args=model_args,
                instruct=instruct,
                max_generated_tokens=max_generated_tokens,
                max_prefill_len=max_prefill_len,
            )
        )
        return None, [list(self.ids)], None, None


def test_device_enhancer_stops_on_stop_token(monkeypatch):
    enhancer, gen = _device_enhancer(monkeypatch, [7, 8, EOS, 9], temperature=0.0, max_new_tokens=32)
    assert enhancer.enhance("a cat") == "t7 t8"
    # The prefill produced t7; one decode each for t8 and the stop token, none after it.
    assert [c[0] for c in gen.decode_calls] == [7, 8]
    assert enhancer.last_stats["new_tokens"] == 2
    assert enhancer.last_stats["prompt_tokens"] == len(PROMPT_IDS)


def test_device_enhancer_caps_at_max_new_tokens(monkeypatch):
    enhancer, gen = _device_enhancer(monkeypatch, [7], temperature=0.0, max_new_tokens=5)
    assert enhancer.enhance("a cat") == " ".join(["t7"] * 5)
    assert len(gen.decode_calls) == 4


def test_device_enhancer_greedy_at_temperature_zero(monkeypatch):
    # Flat-ish logits with a unique maximum at the last id: argmax every step, no RNG involved.
    enhancer, _ = _device_enhancer(monkeypatch, [], peaked=False, temperature=0.0, max_new_tokens=3)
    torch.manual_seed(1)
    assert enhancer.enhance("a cat", seed=1) == " ".join([f"t{VOCAB - 1}"] * 3)
    assert enhancer.enhance("a cat", seed=2) == " ".join([f"t{VOCAB - 1}"] * 3)


def test_device_enhancer_sampling_is_seeded(monkeypatch):
    enhancer, _ = _device_enhancer(monkeypatch, [], peaked=False, temperature=1.0, top_p=1.0, max_new_tokens=12)
    first = enhancer.enhance("a cat", seed=10)
    torch.manual_seed(999)  # outside state must not leak into the next call
    assert enhancer.enhance("a cat", seed=10) == first
    others = {enhancer.enhance("a cat", seed=s) for s in range(11, 31)}
    assert others != {first}


def test_device_enhancer_decodes_only_new_tokens_and_feeds_back_positions(monkeypatch):
    enhancer, gen = _device_enhancer(monkeypatch, [7, 8, 9], temperature=0.0, max_new_tokens=3)
    out = enhancer.enhance("a cat")
    assert out == "t7 t8 t9"
    assert all(f"t{i}" not in out.split() for i in PROMPT_IDS)
    tokens, prefill_kwargs = gen.prefill_calls[0]
    assert tokens.tolist() == [PROMPT_IDS] and tokens.dtype == torch.int32
    assert prefill_kwargs["prompt_lens"] == [len(PROMPT_IDS)]
    assert prefill_kwargs["warmup_prefill"] is False and prefill_kwargs["enable_trace"] is False
    assert prefill_kwargs["page_table"].tolist() == [list(range(enhancer.num_page_blocks))]
    assert [(tok, pos) for tok, pos, _ in gen.decode_calls] == [(7, len(PROMPT_IDS)), (8, len(PROMPT_IDS) + 1)]
    for _, _, kwargs in gen.decode_calls:
        assert kwargs["reload_inputs"] is True
        assert kwargs["enable_trace"] is False
        assert kwargs["sampling_params"] is None
        assert kwargs["kv_cache"][0][0] is gen.model[0].layers[0].kv_cache


def test_device_enhancer_warmup_is_short_greedy_per_bucket_and_once(monkeypatch):
    enhancer, gen = _device_enhancer(monkeypatch, [7], temperature=0.9, max_new_tokens=100)
    temperatures = []
    real_sample = DevicePromptEnhancer._sample
    monkeypatch.setattr(
        DevicePromptEnhancer, "_sample", lambda self, logits, t: temperatures.append(t) or real_sample(self, logits, t)
    )
    rng_before = torch.get_rng_state()
    enhancer.warmup()
    # One short rewrite per warm prompt (one per prefill bucket), each greedy: the configured temperature
    # is overridden to 0 and the global RNG is never touched.
    assert len(gen.prefill_calls) == len(pe.WARMUP_PROMPTS)
    assert len(gen.decode_calls) == len(pe.WARMUP_PROMPTS) * (pe.WARMUP_NEW_TOKENS - 1)
    assert temperatures and set(temperatures) == {0.0}
    assert torch.equal(rng_before, torch.get_rng_state())
    enhancer.warmup()
    assert len(gen.prefill_calls) == len(pe.WARMUP_PROMPTS)


def test_device_enhancer_unload_is_idempotent_and_frees_tensors(monkeypatch):
    enhancer, gen = _device_enhancer(monkeypatch, [7], temperature=0.0, max_new_tokens=2)
    enhancer.unload()  # nothing loaded yet
    enhancer.ensure_loaded()
    tensors = [layer.weight for layer in gen.model[0].layers] + [
        t for layer in gen.model[0].layers for t in layer.kv_cache
    ]
    assert all(t.allocated for t in tensors)
    assert enhancer.is_loaded()
    enhancer.unload()
    assert not any(t.allocated for t in tensors)
    assert gen.released == 1
    assert not enhancer.is_loaded()
    enhancer.unload()
    assert gen.released == 1
    # Loading again after an unload is an ordinary load.
    enhancer.ensure_loaded()
    assert enhancer.is_loaded()


class _Topology:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return f"Topology.{self._name}"


def test_bind_prompt_enhancer_mesh():
    unbound = DevicePromptEnhancer("/fake/gemma")
    assert bind_prompt_enhancer_mesh(unbound, "pipeline-mesh", ccl_topology=_Topology("Linear")) is True
    assert unbound.mesh_device == "pipeline-mesh"
    assert pe.topology_env_value(unbound.ccl_topology) == "linear"
    # An already-bound enhancer keeps its handle and topology; host enhancers and None are ignored.
    assert bind_prompt_enhancer_mesh(unbound, "other-mesh", ccl_topology=_Topology("Ring")) is False
    assert unbound.mesh_device == "pipeline-mesh"
    assert pe.topology_env_value(unbound.ccl_topology) == "linear"
    host = HostPromptEnhancer("/fake/gemma")
    assert bind_prompt_enhancer_mesh(host, "pipeline-mesh") is False
    assert not hasattr(host, "mesh_device")
    assert bind_prompt_enhancer_mesh(None, "pipeline-mesh") is False


def test_cache_dir_resolution_order(monkeypatch, tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setattr(pe, "SHARED_ENHANCER_CACHE_DIR", str(shared))
    monkeypatch.setattr(pe, "USER_ENHANCER_CACHE_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("LTX_ENHANCER_CACHE_DIR", str(tmp_path / "env"))

    assert resolve_enhancer_cache_dir(str(tmp_path / "arg")) == str(tmp_path / "arg")
    assert resolve_enhancer_cache_dir() == str(tmp_path / "env")
    monkeypatch.delenv("LTX_ENHANCER_CACHE_DIR")
    assert resolve_enhancer_cache_dir() == str(shared)
    monkeypatch.setattr(os, "access", lambda path, mode: False)
    assert resolve_enhancer_cache_dir() == str(tmp_path / "user")
    monkeypatch.setattr(os, "access", lambda path, mode: True)
    shared.rmdir()
    assert resolve_enhancer_cache_dir() == str(tmp_path / "user")
    assert DevicePromptEnhancer("/fake/gemma", cache_dir=str(tmp_path / "arg")).cache_dir == str(tmp_path / "arg")


def test_topology_env_value(expect_error):
    assert pe.topology_env_value(_Topology("Ring")) == "ring"
    assert pe.topology_env_value("Linear") == "linear"
    with expect_error(ValueError, "unsupported CCL topology"):
        pe.topology_env_value(_Topology("Mesh"))


def test_gemma4_env_is_scoped_to_the_load(monkeypatch, tmp_path):
    seen = {}

    def record_build_env(monkeypatch):
        fake_build = DevicePromptEnhancer._build_generator

        def recording_build(self):
            seen["cache"] = os.environ.get("TT_CACHE_PATH")
            seen["topology"] = os.environ.get("GEMMA4_CCL_TOPOLOGY")
            return fake_build(self)

        monkeypatch.setattr(DevicePromptEnhancer, "_build_generator", recording_build)

    # Unset before: both variables exist only while gemma4 builds the model.
    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    enhancer, _ = _device_enhancer(
        monkeypatch, [7], temperature=0.0, max_new_tokens=2, cache_dir=str(tmp_path / "a"), ccl_topology="Linear"
    )
    record_build_env(monkeypatch)
    enhancer.ensure_loaded()
    assert seen == {"cache": str(tmp_path / "a"), "topology": "linear"}
    assert "TT_CACHE_PATH" not in os.environ and "GEMMA4_CCL_TOPOLOGY" not in os.environ

    # Preset before: the rewriter's own cache dir still wins for the load, an explicit gemma4 topology is
    # kept, and both are restored afterwards.
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path / "other-model"))
    monkeypatch.setenv("GEMMA4_CCL_TOPOLOGY", "ring")
    other, _ = _device_enhancer(
        monkeypatch, [7], temperature=0.0, max_new_tokens=2, cache_dir=str(tmp_path / "b"), ccl_topology="Linear"
    )
    record_build_env(monkeypatch)
    other.ensure_loaded()
    assert seen == {"cache": str(tmp_path / "b"), "topology": "ring"}
    assert os.environ["TT_CACHE_PATH"] == str(tmp_path / "other-model")
    assert os.environ["GEMMA4_CCL_TOPOLOGY"] == "ring"


def test_resolve_model_dir(monkeypatch, tmp_path):
    import huggingface_hub

    assert pe.resolve_model_dir(str(tmp_path)) == str(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, local_files_only: f"/snap/{repo}")
    assert pe.resolve_model_dir("google/gemma-4-E2B-it") == "/snap/google/gemma-4-E2B-it"

    def missing(repo, local_files_only):
        raise FileNotFoundError(repo)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    assert pe.resolve_model_dir("google/gemma-4-E2B-it") == "google/gemma-4-E2B-it"


def test_resolve_stop_tokens_reads_generation_config(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 106, 50]}))
    assert resolve_stop_tokens(str(tmp_path), _FakeTokenizer()) == [1, 50, 106]
    # No readable config: the known Gemma-4 ids fill in.
    assert resolve_stop_tokens(str(tmp_path / "missing"), _FakeTokenizer()) == sorted(set(GEMMA4_EOS_IDS) | {EOS})


def test_strip_template_bos():
    class _DoubleBosTokenizer(_FakeTokenizer):
        def encode(self, text, add_special_tokens=True):
            return [BOS, BOS, 3, 4] if text.startswith("<bos>") else [BOS, 3, 4]

    text, ids = pe._strip_template_bos("<bos>hello", _DoubleBosTokenizer())
    assert (text, ids) == ("hello", [BOS, 3, 4])
    text, ids = pe._strip_template_bos("<bos>hello", _FakeTokenizer())
    assert (text, ids) == ("<bos>hello", PROMPT_IDS)


def test_device_enhancer_rejects_overlong_budget(expect_error):
    with expect_error(AssertionError, "max_new_tokens"):
        DevicePromptEnhancer("/fake/gemma", max_new_tokens=2048, max_seq_len=2048)


def test_device_enhancer_tokenizes_through_prefill_preprocessing(monkeypatch):
    import models.tt_transformers.tt.common as ttt_common

    enhancer, gen = _device_enhancer(monkeypatch, [7, EOS], temperature=0.0, max_new_tokens=8, real_tokenize=True)
    recorder = _PrefillPreprocessRecorder(PROMPT_IDS)
    monkeypatch.setattr(ttt_common, "preprocess_inputs_prefill", recorder)
    assert enhancer.enhance("a cat") == "t7"
    (call,) = recorder.calls
    # Already-templated text goes through the non-instruct path with the full context as the budget; the
    # single template <bos> is kept because this tokenizer adds none of its own.
    assert call["prompts"] == ["<bos>user prompt: a cat"]
    assert call["tokenizer"] is enhancer._tokenizer and call["model_args"] is gen.model_args
    assert call["instruct"] is False
    assert call["max_generated_tokens"] == 8 and call["max_prefill_len"] == enhancer.max_seq_len
    assert gen.prefill_calls[0][0].tolist() == [PROMPT_IDS]


def test_device_enhancer_tokenization_strips_a_doubled_bos(monkeypatch):
    import models.tt_transformers.tt.common as ttt_common

    class _DoubleBosTokenizer(_FakeTokenizer):
        def encode(self, text, add_special_tokens=True):
            return [BOS] + list(PROMPT_IDS) if text.startswith("<bos>") else list(PROMPT_IDS)

    enhancer, _ = _device_enhancer(
        monkeypatch, [7], temperature=0.0, max_new_tokens=8, tokenizer=_DoubleBosTokenizer(), real_tokenize=True
    )
    enhancer.ensure_loaded()
    recorder = _PrefillPreprocessRecorder(PROMPT_IDS)
    monkeypatch.setattr(ttt_common, "preprocess_inputs_prefill", recorder)
    assert enhancer._tokenize_prompt("<bos>user prompt: a cat", 8) == PROMPT_IDS
    assert recorder.calls[0]["prompts"] == ["user prompt: a cat"]


def test_device_enhancer_rejects_a_prompt_preprocessing_would_clip(monkeypatch, expect_error):
    import models.tt_transformers.tt.common as ttt_common

    enhancer, gen = _device_enhancer(monkeypatch, [7], temperature=0.0, max_new_tokens=8, real_tokenize=True)
    # Left-clipping returns fewer ids than the prompt has: the system prompt's head would be gone.
    monkeypatch.setattr(ttt_common, "preprocess_inputs_prefill", _PrefillPreprocessRecorder(PROMPT_IDS[1:]))
    with expect_error(pe.PromptTooLongError, "does not fit"):
        enhancer.enhance("a cat")
    assert gen.prefill_calls == []
    # Through apply_prompt_enhancer the request survives with the raw prompt.
    assert apply_prompt_enhancer(enhancer, "a cat") == "a cat"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only contract for the HunyuanVideo-1.5 text-conditioning cache.

No device is opened here. The bar these tests enforce is that a cache hit is
byte-identical to a miss for the same key, and that anything which can move a
conditioning tensor moves the key instead of being silently reused.
"""

import types

import pytest
import torch

from models.demos.hf_eager.hunyuanvideo_1_5.tt import text_conditioning as tc
from models.demos.hf_eager.hunyuanvideo_1_5.tt.text_conditioning import (
    PromptCacheError,
    cache_descriptor,
    conditioning_fingerprint,
    encode_prompt_pair,
)


class _Transformer:
    config = types.SimpleNamespace()
    dtype = torch.bfloat16


class _Encoder:
    """Stands in for a host HF encoder: carries a config, dtype and no mesh."""

    def __init__(self, name, dtype=torch.bfloat16):
        self.config = types.SimpleNamespace(_name_or_path=name)
        self.dtype = dtype


class _DeviceEncoder(_Encoder):
    """Stands in for a TT adapter: identical proxied attributes, plus a mesh."""

    def __init__(self, name, mesh_shape, zero_padding=True):
        super().__init__(name)
        self._device = types.SimpleNamespace(shape=mesh_shape)
        self._zero_padding = zero_padding


class FakePipe:
    """A pipeline exposing exactly the surface ``encode_prompt_pair`` reads.

    Each ``encode_prompt`` call returns a distinct tensor value, so a cache hit
    that silently re-encoded, or returned another prompt's tuple, would fail an
    equality assertion rather than pass by coincidence.
    """

    tokenizer_max_length = 1000
    tokenizer_2_max_length = 256
    system_message = "system"
    prompt_template_encode_start_idx = 108
    _execution_device = torch.device("cpu")

    def __init__(self, *, enabled=True, conditions=2, text_encoder=None, text_encoder_2=None):
        self.calls = 0
        self.transformer = _Transformer()
        self.text_encoder = text_encoder or _Encoder("qwen-test")
        self.text_encoder_2 = text_encoder_2 or _Encoder("byt5-test")
        self.tokenizer = types.SimpleNamespace(name_or_path="tok", vocab_size=151936)
        self.tokenizer_2 = types.SimpleNamespace(name_or_path="tok2", vocab_size=1510)
        self.guider = types.SimpleNamespace(_enabled=enabled, num_conditions=conditions)

    def encode_prompt(self, prompt, **kwargs):
        self.calls += 1
        value = float(self.calls)
        return (
            torch.full((1, 4, 3), value, dtype=torch.bfloat16),
            torch.ones(1, 4, dtype=torch.bfloat16),
            torch.full((1, 2, 2), value, dtype=torch.bfloat16),
            torch.ones(1, 2, dtype=torch.bfloat16),
        )


@pytest.fixture(autouse=True)
def _isolate_memory_cache():
    tc._MEMORY_CACHE.clear()
    yield
    tc._MEMORY_CACHE.clear()


def _keys(pipe, prompt="a prompt", negative=""):
    descriptor = cache_descriptor(pipe, prompt, negative)
    return tc._serialize(descriptor)


# --------------------------------------------------------------------- hit == miss


def test_a_warm_hit_is_byte_identical_to_the_cold_encode(tmp_path):
    pipe = FakePipe()
    cold, cold_hit = encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    tc._MEMORY_CACHE.clear()  # force the persisted artifact to be re-read
    warm, warm_hit = encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)

    assert (cold_hit, warm_hit) == (False, True)
    assert pipe.calls == 2, "the warm run must not have called either encoder"
    assert conditioning_fingerprint(cold) == conditioning_fingerprint(warm)
    for name in cold:
        assert torch.equal(cold[name], warm[name])
        assert cold[name].dtype == warm[name].dtype


def test_a_memory_hit_is_byte_identical_and_skips_both_encoders(tmp_path):
    pipe = FakePipe()
    cold, _ = encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    warm, hit = encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    assert hit and pipe.calls == 2
    assert conditioning_fingerprint(cold) == conditioning_fingerprint(warm)


def test_the_returned_tuple_is_a_copy_so_a_caller_cannot_poison_the_cache(tmp_path):
    pipe = FakePipe()
    first, _ = encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    first["prompt_embeds"].fill_(-99.0)
    second, hit = encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    assert hit
    assert not torch.equal(first["prompt_embeds"], second["prompt_embeds"])
    assert torch.all(second["prompt_embeds"] == 1.0)


def test_the_verify_gate_recomputes_and_accepts_a_matching_hit(tmp_path):
    class Deterministic(FakePipe):
        def encode_prompt(self, prompt, **kwargs):
            self.calls += 1
            return (
                torch.full((1, 4, 3), 7.0, dtype=torch.bfloat16),
                torch.ones(1, 4, dtype=torch.bfloat16),
                torch.full((1, 2, 2), 7.0, dtype=torch.bfloat16),
                torch.ones(1, 2, dtype=torch.bfloat16),
            )

    pipe = Deterministic()
    encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    values, hit = encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path, verify=True)
    assert hit and pipe.calls == 4  # 2 cold + 2 verification encodes
    assert torch.all(values["prompt_embeds"] == 7.0)


def test_the_verify_gate_rejects_a_hit_that_is_not_byte_identical(tmp_path):
    pipe = FakePipe()  # every call returns a different value
    encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    with pytest.raises(PromptCacheError, match="not byte-identical"):
        encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path, verify=True)


def test_use_cache_false_never_reads_or_writes_an_artifact(tmp_path):
    pipe = FakePipe()
    encode_prompt_pair(pipe, "a prompt", "", use_cache=False, cache_dir=tmp_path)
    _, hit = encode_prompt_pair(pipe, "a prompt", "", use_cache=False, cache_dir=tmp_path)
    assert not hit
    assert not list(tmp_path.rglob("*.pt"))


# --------------------------------------------------------------------- key coverage


def test_the_prompt_and_negative_prompt_separate_keys():
    pipe = FakePipe()
    base = _keys(pipe)
    assert _keys(pipe, prompt="another prompt") != base
    assert _keys(pipe, negative="blurry") != base


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda p: setattr(p, "tokenizer_max_length", 512), id="qwen_max_length"),
        pytest.param(lambda p: setattr(p, "tokenizer_2_max_length", 128), id="byt5_max_length"),
        pytest.param(lambda p: setattr(p, "system_message", "other"), id="prompt_template"),
        pytest.param(lambda p: setattr(p, "prompt_template_encode_start_idx", 0), id="crop_start"),
        pytest.param(lambda p: setattr(p.transformer, "dtype", torch.float32), id="transformer_dtype"),
        pytest.param(lambda p: setattr(p.text_encoder, "dtype", torch.float32), id="text_encoder_dtype"),
        pytest.param(lambda p: setattr(p.text_encoder_2, "dtype", torch.float32), id="text_encoder_2_dtype"),
        pytest.param(lambda p: setattr(p.guider, "num_conditions", 1), id="conditions"),
        pytest.param(lambda p: setattr(p.guider, "_enabled", False), id="guidance_enabled"),
        pytest.param(lambda p: setattr(p.tokenizer, "name_or_path", "other"), id="tokenizer"),
        pytest.param(lambda p: setattr(p.tokenizer_2, "vocab_size", 9), id="tokenizer_2"),
        pytest.param(
            lambda p: setattr(p.text_encoder, "config", types.SimpleNamespace(_name_or_path="other")),
            id="qwen_checkpoint",
        ),
        pytest.param(
            lambda p: setattr(p.text_encoder_2, "config", types.SimpleNamespace(_name_or_path="other")),
            id="byt5_checkpoint",
        ),
    ],
)
def test_every_field_that_can_move_an_embedding_moves_the_key(mutate):
    pipe = FakePipe()
    before = _keys(pipe)
    mutate(pipe)
    assert _keys(pipe) != before


def test_a_device_encoder_never_shares_a_key_with_the_host_encoder():
    host = FakePipe()
    device = FakePipe(text_encoder=_DeviceEncoder("qwen-test", (8, 4)))
    assert _keys(host) != _keys(device)
    assert cache_descriptor(host, "p", "")["placement"]["qwen"]["where"] == "host"
    assert cache_descriptor(device, "p", "")["placement"]["qwen"]["where"] == "device"


def test_the_device_mesh_shape_is_keyed_because_it_sets_the_tensor_parallel_fracture():
    small = FakePipe(text_encoder=_DeviceEncoder("qwen-test", (1, 2)))
    large = FakePipe(text_encoder=_DeviceEncoder("qwen-test", (8, 4)))
    assert _keys(small) != _keys(large)


def test_the_byt5_padding_neutralization_choice_is_keyed():
    on = FakePipe(text_encoder_2=_DeviceEncoder("byt5-test", (1, 2), zero_padding=True))
    off = FakePipe(text_encoder_2=_DeviceEncoder("byt5-test", (1, 2), zero_padding=False))
    assert _keys(on) != _keys(off)


@pytest.mark.parametrize(
    "name,value",
    [
        ("HY_TT_QWEN", "1"),
        ("HY_TT_QWEN_SHARED", "1"),
        ("HY_TT_BYT5", "1"),
        ("HY_QWEN_ZERO_PAD", "0"),
        ("HY_BYT5_ZERO_PAD", "0"),
        ("HY_CFG_PADDING_POLICY", "masked"),
        ("HY_MESH", "8,4"),
    ],
)
def test_every_keyed_environment_flag_separates_the_key(monkeypatch, name, value):
    pipe = FakePipe()
    before = _keys(pipe)
    monkeypatch.setenv(name, value)
    assert _keys(pipe) != before


def test_two_prompts_never_collide_on_disk(tmp_path):
    pipe = FakePipe()
    first, _ = encode_prompt_pair(pipe, "prompt one", "", cache_dir=tmp_path)
    second, hit = encode_prompt_pair(pipe, "prompt two", "", cache_dir=tmp_path)
    assert not hit
    assert not torch.equal(first["prompt_embeds"], second["prompt_embeds"])
    assert len(list(tmp_path.rglob("*.pt"))) == 2


# --------------------------------------------------------------------- payload shape


def test_guidance_disabled_stores_only_the_positive_tuple(tmp_path):
    pipe = FakePipe(enabled=False)
    values, _ = encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    assert tuple(sorted(values)) == tuple(sorted(tc.POSITIVE_NAMES))
    assert pipe.calls == 1

    warm, hit = encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    assert hit and tuple(sorted(warm)) == tuple(sorted(tc.POSITIVE_NAMES))


def test_a_positive_only_artifact_cannot_be_served_to_a_guided_run(tmp_path):
    """The failure this guards against: guidance was off when the artifact was
    written, so it carries no negative tuple; a later guided run must miss."""
    unguided = FakePipe(enabled=False)
    encode_prompt_pair(unguided, "a prompt", "", cache_dir=tmp_path)

    guided = FakePipe(enabled=True)
    values, hit = encode_prompt_pair(guided, "a prompt", "", cache_dir=tmp_path)
    assert not hit
    assert tuple(sorted(values)) == tuple(sorted(tc.POSITIVE_NAMES + tc.NEGATIVE_NAMES))


def test_an_artifact_whose_descriptor_disagrees_is_rejected_rather_than_used(tmp_path):
    pipe = FakePipe()
    encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    tc._MEMORY_CACHE.clear()

    (path,) = list(tmp_path.rglob("*.pt"))
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    artifact["descriptor"] = artifact["descriptor"].replace("a prompt", "a different prompt")
    torch.save(artifact, path)

    with pytest.raises(PromptCacheError, match="different conditioning descriptor"):
        encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)


def test_a_schema_1_artifact_is_rejected_rather_than_silently_reinterpreted(tmp_path):
    pipe = FakePipe()
    encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    tc._MEMORY_CACHE.clear()

    (path,) = list(tmp_path.rglob("*.pt"))
    legacy = torch.load(path, map_location="cpu", weights_only=True)["tensors"]
    torch.save(legacy, path)  # the old bare-dict layout

    with pytest.raises(PromptCacheError, match="predates schema"):
        encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)


def test_a_truncated_tensor_set_is_rejected(tmp_path):
    pipe = FakePipe()
    encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)
    tc._MEMORY_CACHE.clear()

    (path,) = list(tmp_path.rglob("*.pt"))
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    artifact["tensors"].pop("negative_prompt_embeds")
    torch.save(artifact, path)

    with pytest.raises(PromptCacheError, match="holds"):
        encode_prompt_pair(pipe, "a prompt", "", cache_dir=tmp_path)


def test_the_fingerprint_separates_value_shape_and_dtype():
    base = {"a": torch.ones(2, 2, dtype=torch.bfloat16)}
    assert conditioning_fingerprint(base) == conditioning_fingerprint({"a": torch.ones(2, 2, dtype=torch.bfloat16)})
    assert conditioning_fingerprint(base) != conditioning_fingerprint({"a": torch.zeros(2, 2, dtype=torch.bfloat16)})
    assert conditioning_fingerprint(base) != conditioning_fingerprint({"a": torch.ones(4, dtype=torch.bfloat16)})
    assert conditioning_fingerprint(base) != conditioning_fingerprint({"a": torch.ones(2, 2, dtype=torch.float32)})

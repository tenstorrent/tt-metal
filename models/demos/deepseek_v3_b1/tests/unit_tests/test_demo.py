# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.demo.cli import (
    FIRST_K_DENSE_REPLACE,
    SYSTEM_MESH_ID_EMBEDDING,
    SYSTEM_MESH_ID_LM_HEAD,
    load_weights_from_cache,
)
from models.demos.deepseek_v3_b1.demo.runner import run_generation
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
    save_decoder_layer,
    save_embedding_weights,
    save_lm_head_weights,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_prepare_weights import (
    NUM_ROUTED_EXPERTS,
    _add_global_weights,
    _deallocate_layer,
    _layer_state_dict,
    _skip_unless_4x2_mesh,
)


class FakeTokenizer:
    def __init__(self, *, prompt_tokens: list[int], bos_token_id: int | None = 1):
        self._prompt_tokens = prompt_tokens
        self.bos_token_id = bos_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del text, add_special_tokens
        return list(self._prompt_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return f"<{token_ids[0]}>"


class FakeModel:
    def __init__(self, decode_outputs: list[int]):
        self._decode_outputs = list(decode_outputs)
        self.started = False
        self.stopped = False
        self.prefill_inputs: list[int] = []
        self.decode_inputs: list[int] = []

    def start(self) -> None:
        self.started = True

    def prefill(self, prompt_tokens: list[int]) -> int:
        self.prefill_inputs = list(prompt_tokens)
        return prompt_tokens[-1]

    def decode_step(self, input_tensor: int) -> int:
        self.decode_inputs.append(input_tensor)
        return self._decode_outputs.pop(0)

    def stop(self) -> None:
        self.stopped = True


def test_run_generation_streams_decode_text_and_tracks_token_flow():
    tokenizer = FakeTokenizer(prompt_tokens=[11, 12], bos_token_id=1)
    model = FakeModel(decode_outputs=[101, 102, 103])
    streamed_chunks: list[str] = []

    result = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt="hello",
        max_new_tokens=3,
        make_input_tensor=lambda token_id: token_id,
        extract_token_id=lambda output: output,
        write_text=streamed_chunks.append,
    )

    assert model.started
    assert model.stopped
    assert model.prefill_inputs == [11, 12]
    assert model.decode_inputs == [12, 101, 102]
    assert result.prompt_token_ids == [11, 12]
    assert result.generated_token_ids == [101, 102, 103]
    assert result.generated_text == "<101><102><103>"
    assert "".join(streamed_chunks) == "<101><102><103>"


def test_run_generation_uses_bos_for_empty_prompt():
    tokenizer = FakeTokenizer(prompt_tokens=[], bos_token_id=7)
    model = FakeModel(decode_outputs=[8, 9])

    result = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt="",
        max_new_tokens=2,
        make_input_tensor=lambda token_id: token_id,
        extract_token_id=lambda output: output,
    )

    assert model.prefill_inputs == [7]
    assert model.decode_inputs == [7, 8]
    assert result.prompt_token_ids == [7]
    assert result.generated_token_ids == [8, 9]


def test_run_generation_stops_model_if_decode_raises():
    tokenizer = FakeTokenizer(prompt_tokens=[3], bos_token_id=1)

    class FailingModel(FakeModel):
        def decode_step(self, input_tensor: int) -> int:
            super().decode_step(input_tensor)
            raise RuntimeError("decode failed")

    model = FailingModel(decode_outputs=[4])

    with pytest.raises(RuntimeError, match="decode failed"):
        run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt="x",
            max_new_tokens=1,
            make_input_tensor=lambda token_id: token_id,
            extract_token_id=lambda output: output,
        )

    assert model.started
    assert model.stopped


def test_run_generation_rejects_negative_max_new_tokens():
    tokenizer = FakeTokenizer(prompt_tokens=[1], bos_token_id=1)
    model = FakeModel(decode_outputs=[])

    with pytest.raises(ValueError, match="max_new_tokens must be >= 0"):
        run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt="x",
            max_new_tokens=-1,
            make_input_tensor=lambda token_id: token_id,
            extract_token_id=lambda output: output,
        )

    assert not model.started
    assert not model.stopped


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "layer_offset",
    [
        SYSTEM_MESH_ID_EMBEDDING,
        FIRST_K_DENSE_REPLACE - 1,
        FIRST_K_DENSE_REPLACE,
        SYSTEM_MESH_ID_LM_HEAD,
    ],
)
def test_load_weights_from_cache(bh_2d_mesh_device, tmp_path, layer_offset):
    """Load weights from cache for embedding, last dense, first MoE, and LM head (4x2 submesh)."""
    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    if not is_slow_dispatch():
        pytest.skip("load_weights_from_cache requires slow dispatch")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    manifest_kw = dict(
        hf_model_name="test-demo-load",
        hf_state_dict_name="test.safetensors",
        device_mesh_shape=(4, 2),
    )

    if layer_offset == SYSTEM_MESH_ID_EMBEDDING or layer_offset == SYSTEM_MESH_ID_LM_HEAD:
        state = {}
        _add_global_weights(state)
        embedding_weights = prepare_embedding_weights(state, submesh)
        lm_head_weights = prepare_lm_head_weights(state, submesh)
        save_embedding_weights(embedding_weights, tmp_path, **manifest_kw)
        save_lm_head_weights(lm_head_weights, tmp_path, **manifest_kw)
        ttnn.deallocate(embedding_weights.embedding, force=True)
        ttnn.deallocate(lm_head_weights.lm_head, force=True)
        ttnn.deallocate(lm_head_weights.final_norm, force=True)
    elif layer_offset == FIRST_K_DENSE_REPLACE - 1:
        state = _layer_state_dict(0, is_moe=False)
        bdw = BlitzDecodeWeights(submesh)
        layer = prepare_dense_layer_weights(bdw, state, 0)
        save_decoder_layer(layer, tmp_path, 1, **manifest_kw)
        _deallocate_layer(layer)
    elif layer_offset == FIRST_K_DENSE_REPLACE:
        state = _layer_state_dict(0, is_moe=True, seed=43)
        bdw = BlitzDecodeWeights(submesh)
        layer = prepare_moe_layer_weights(bdw, state, 0, num_routed_experts=NUM_ROUTED_EXPERTS)
        save_decoder_layer(layer, tmp_path, 2, **manifest_kw)
        _deallocate_layer(layer)

    result = load_weights_from_cache(tmp_path, submesh, layer_offset)

    if layer_offset == SYSTEM_MESH_ID_EMBEDDING:
        assert isinstance(result, DeepSeekV3EmbeddingLayerWeights)
        assert result.embedding is not None
    elif layer_offset == SYSTEM_MESH_ID_LM_HEAD:
        assert isinstance(result, DeepSeekV3LMHeadWeights)
        assert result.lm_head is not None
        assert result.final_norm is not None
    elif layer_offset == FIRST_K_DENSE_REPLACE - 1:
        assert isinstance(result, DeepSeekV3DenseLayerWeights)
    else:
        assert layer_offset == FIRST_K_DENSE_REPLACE
        assert isinstance(result, DeepSeekV3MoELayerWeights)


class _LoopbackTokenizer:
    bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del text, add_special_tokens
        return [self.bos_token_id]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del token_ids, skip_special_tokens
        return ""


@pytest.mark.slow
@pytest.mark.skip_post_commit
@pytest.mark.timeout(3600)
def test_demo_decode_stress_64k_tokens(mesh_device) -> None:
    from models.common.utility_functions import is_slow_dispatch
    from models.demos.deepseek_v3_b1.demo.runtime import TokenCodec, create_model

    if not is_slow_dispatch():
        pytest.skip("Skipping stress test in fast dispatch mode")

    max_new_tokens = 65536
    batch_size = 1
    tokenizer = _LoopbackTokenizer()
    token_codec = TokenCodec(batch_size=batch_size)
    model = create_model(mesh_device=mesh_device, batch_size=batch_size, loopback_mode=True)

    result = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt="",
        max_new_tokens=max_new_tokens,
        make_input_tensor=token_codec.make_input,
        extract_token_id=token_codec.extract_token_id,
        write_text=lambda _: None,
    )

    assert len(result.generated_token_ids) == max_new_tokens
    assert result.generated_token_ids[0] == tokenizer.bos_token_id
    assert result.generated_token_ids[-1] == tokenizer.bos_token_id
    assert model.position == 1 + max_new_tokens

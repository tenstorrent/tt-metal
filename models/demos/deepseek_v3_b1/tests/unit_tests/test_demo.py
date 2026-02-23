# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.cli import FIRST_K_DENSE_REPLACE, enable_fast_dispatch_mode
from models.demos.deepseek_v3_b1.demo.runner import run_generation
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3Weights,
    deallocate_weights,
    load_layer,
    load_moe_routed_experts_from_cache,
    prepare_weights,
    save_layer,
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


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
def test_demo_weight_loading_from_cache_4x2(bh_2d_mesh_device, tmp_path):
    """Test the demo's two-phase weight loading: create a one-layer dense cache, then run the same load path as run_demo and assert weights are loaded."""
    from models.demos.deepseek_v3_b1.tests.unit_tests.test_prepare_weights import (
        _layer_state_dict,
        _skip_unless_4x2_mesh,
    )

    _skip_unless_4x2_mesh(bh_2d_mesh_device)
    if not is_slow_dispatch():
        pytest.skip("Demo weight loading requires slow dispatch")

    # Same as CLI: full mesh split into (4, 2) submeshes; layer i goes on submeshes[i]
    submeshes = bh_2d_mesh_device.create_submeshes(ttnn.MeshShape(4, 2))
    assert len(submeshes) >= 1, "Need at least one (4,2) submesh"
    submesh0 = submeshes[0]
    cache_path = tmp_path

    # Create minimal cache: one dense layer on submesh 0 (same pattern as test_prepare_weights)
    state = _layer_state_dict(0, is_moe=False)
    weights = prepare_weights(
        state,
        submesh0,
        num_layers=1,
        first_k_dense_replace=1,
    )
    save_layer(
        weights.layers[0],
        cache_path,
        0,
        hf_model_name="test-demo-weight-load",
        hf_state_dict_name="test.safetensors",
        device_mesh_shape=(4, 2),
    )
    deallocate_weights(weights)

    # Run the same two-phase loading logic as run_demo() (each layer on its submesh)
    num_layers = 1
    preloaded_experts = {}
    with enable_fast_dispatch_mode(bh_2d_mesh_device):
        for layer_idx in range(FIRST_K_DENSE_REPLACE, num_layers):
            preloaded_experts[layer_idx] = load_moe_routed_experts_from_cache(
                cache_path, submeshes[layer_idx], layer_idx
            )

    layers = []
    for layer_idx in range(num_layers):
        layer = load_layer(
            cache_path,
            submeshes[layer_idx],
            layer_idx,
            preloaded_routed_experts=preloaded_experts.get(layer_idx),
        )
        layers.append(layer)
    loaded_weights = DeepSeekV3Weights(layers=layers)

    assert len(loaded_weights.layers) == 1
    assert isinstance(loaded_weights.layers[0], DeepSeekV3DenseLayerWeights)
    layer0 = loaded_weights.layers[0]
    assert layer0.q_a_proj.tensor_shape == (3584, 3072)
    assert layer0.o_proj.tensor_shape == (8192, 7168)

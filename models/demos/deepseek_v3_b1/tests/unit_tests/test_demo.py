# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from models.demos.deepseek_v3_b1.demo.runner import run_generation


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

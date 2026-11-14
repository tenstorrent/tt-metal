# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral3ForConditionalGeneration.prefill_forward must give the vllm-tt-plugin images to the model.

vllm-tt-plugin passes images as ``pixel_values``: one entry per request, None or a list with one
[C, H, W] tensor cut to the image's size. The adapter used to read only an ``images`` keyword,
which the plugin does not send, so vLLM served every image request as text only.

A batch with images must not use the batched prefill of the generator: that path gives the images
of the first request to every request.

Host-only: the generator runs with a stub model, so no device is opened.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

from models.common.tests.vllm_stubs import stub_vllm_if_missing
from models.ttt_compat.tt.generator import Generator

PAD_TOKEN_ID = 11


def _adapter(monkeypatch):
    stub_vllm_if_missing(monkeypatch)
    adapter = importlib.import_module("models.ttt_compat.tt.generator_vllm").Mistral3ForConditionalGeneration
    wrapper = adapter.__new__(adapter)
    wrapper.model_args = [SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=PAD_TOKEN_ID))]
    wrapper.data_parallel = 1  # read by Generator.__del__
    return adapter, wrapper


def _prefill_calls(monkeypatch, **plugin_kwargs):
    """Run prefill_forward with Generator.prefill_forward_text replaced; return the result and each call's kwargs."""
    adapter, wrapper = _adapter(monkeypatch)
    calls = []

    def prefill_forward_text(self, **kwargs):
        calls.append(kwargs)
        # One logit per request: the first token of its row.
        return kwargs["tokens"][:, :1].unsqueeze(-1).float()

    monkeypatch.setattr(Generator, "prefill_forward_text", prefill_forward_text)
    result = adapter.prefill_forward(
        wrapper,
        tokens=torch.arange(1, 13).reshape(3, 4),
        prompt_lens=[4, 2, 3],
        page_table=torch.arange(6, dtype=torch.int32).reshape(3, 2),
        kv_cache="kv_cache",
        **plugin_kwargs,
    )
    return result, calls


@pytest.mark.parametrize("enable_trace", [True, False])
def test_text_only_prefill_keeps_the_plugin_trace_mode(enable_trace, monkeypatch):
    result, calls = _prefill_calls(
        monkeypatch,
        enable_trace=enable_trace,
        pixel_values=[None, None, None],
        image_grid_thw=[None, None, None],
    )

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["enable_trace"] is enable_trace
    assert kwargs["pixel_values"] is None
    assert kwargs["image_sizes"] is None
    assert "empty_slots" not in kwargs
    assert kwargs["page_table"].tolist() == [[0, 1], [2, 3], [4, 5]]
    assert kwargs["kv_cache"] == "kv_cache"
    assert kwargs["prompt_lens"] == [4, 2, 3]
    torch.testing.assert_close(
        kwargs["tokens"],
        torch.tensor([[1, 2, 3, 4], [5, 6, PAD_TOKEN_ID, PAD_TOKEN_ID], [9, 10, 11, PAD_TOKEN_ID]]),
    )
    assert result.flatten().tolist() == [1, 5, 9]


def test_image_batch_prefills_one_request_per_call(monkeypatch):
    image = torch.rand(3, 28, 42)
    other_image = torch.rand(3, 14, 56)

    result, calls = _prefill_calls(
        monkeypatch,
        enable_trace=True,
        pixel_values=[None, [image], [other_image]],
    )

    assert len(calls) == 3
    for i, kwargs in enumerate(calls):
        # A traced prefill cannot take image features, so a batch with images runs without trace.
        assert kwargs["enable_trace"] is False
        assert kwargs["empty_slots"] == [i]
        assert kwargs["page_table"].tolist() == [[2 * i, 2 * i + 1]]
        assert kwargs["kv_cache"] == "kv_cache"
    assert [kwargs["prompt_lens"] for kwargs in calls] == [[4], [2], [3]]
    assert [kwargs["tokens"].tolist() for kwargs in calls] == [
        [[1, 2, 3, 4]],
        [[5, 6, PAD_TOKEN_ID, PAD_TOKEN_ID]],
        [[9, 10, 11, PAD_TOKEN_ID]],
    ]
    assert calls[0]["pixel_values"] == [None]
    assert calls[0]["image_sizes"] == [None]
    torch.testing.assert_close(calls[1]["pixel_values"][0], image.unsqueeze(0))
    assert calls[1]["image_sizes"] == [[[28, 42]]]
    torch.testing.assert_close(calls[2]["pixel_values"][0], other_image.unsqueeze(0))
    assert calls[2]["image_sizes"] == [[[14, 56]]]
    assert result.flatten().tolist() == [1, 5, 9]


def test_a_request_with_two_images_is_refused(monkeypatch, expect_error):
    # The vision tower gives every image the patches of the first image (#64646), so the
    # processor allows one image per request and the adapter refuses more.
    with expect_error(ValueError, "one image per request"):
        _prefill_calls(monkeypatch, pixel_values=[None, [torch.rand(3, 28, 28), torch.rand(3, 28, 28)], None])


class _DeviceLogits:
    """Stands in for one prefill's untilized device logits, which the generator reads back with cpu()."""

    def __init__(self, logits):
        self.logits = logits

    def cpu(self, blocking=True):
        return self.logits


@pytest.mark.parametrize("with_image", [False, True], ids=["text_only", "text_and_image"])
def test_each_request_reaches_the_model_with_its_own_image(with_image, monkeypatch):
    """Run the real generator, with a stub model, on two prompts of the same padded length.

    Text only, the generator prefills both prompts in one batched call. With an image in the
    second request, a batched call would get only the first request's (empty) image input, so
    each request must reach the model in its own call, with its own image.
    """
    from models.ttt_compat.tt import generator as generator_module

    adapter, wrapper = _adapter(monkeypatch)
    wrapper.model_args = [
        SimpleNamespace(
            tokenizer=SimpleNamespace(pad_token_id=PAD_TOKEN_ID),
            max_batch_size=32,
            max_prefill_chunk_size=32768,
            vocab_size=1,
            batched_prefill_compact_rows=False,
            can_enable_trace=lambda *args: False,
        )
    ]
    wrapper.model = [
        SimpleNamespace(
            mesh_device=object(),
            # Every token of a row has the same value, so the "logit" of a request is its token.
            process_logits_after_prefill_trace=lambda logits, last: logits[:, :, last : last + 1],
            process_output_prefill=lambda logits, **kwargs: logits.reshape(-1)[:1].reshape(1, 1),
        )
    ]
    wrapper.already_warmed_up_prefill = True
    wrapper.warmup_model_prefill = lambda **kwargs: None
    wrapper._defer_trace_recording = False
    wrapper._defer_prefill_recording = False
    wrapper._will_row_shard_prefill = lambda *args: False
    model_calls = []

    def prefill_forward_single_user_text(tokens, **kwargs):
        model_calls.append((tokens.clone(), kwargs))
        return tokens.float().unsqueeze(-1)

    wrapper.prefill_forward_single_user_text = prefill_forward_single_user_text
    monkeypatch.setattr(generator_module.ttnn, "reshape", torch.reshape)
    monkeypatch.setattr(generator_module.ttnn, "to_layout", lambda tensor, *args, **kwargs: tensor)
    monkeypatch.setattr(generator_module.ttnn, "untilize", lambda tensor, **kwargs: _DeviceLogits(tensor))
    monkeypatch.setattr(generator_module.ttnn, "synchronize_device", lambda *args: None)

    image = torch.rand(3, 28, 42)
    result = adapter.prefill_forward(
        wrapper,
        tokens=torch.tensor([[21] * 6, [22] * 6]),
        prompt_lens=[6, 6],
        page_table=torch.tensor([[10, 99], [20, 99]], dtype=torch.int32),
        kv_cache=[[(SimpleNamespace(shape=(100, 1, 64, 128)), None)]],
        enable_trace=False,
        pixel_values=[None, [image] if with_image else None],
    )

    assert result.flatten().tolist() == [21, 22]
    if not with_image:
        assert len(model_calls) == 1
        tokens, kwargs = model_calls[0]
        assert tokens[:, 0].tolist() == [21, 22]
        assert kwargs["pixel_values"] is None
        return
    assert len(model_calls) == 2
    (first_tokens, first), (second_tokens, second) = model_calls
    assert first_tokens[:, 0].tolist() == [21]
    assert first["batch_size"] == 1
    assert first["pixel_values"] is None
    assert first["image_sizes"] is None
    assert second_tokens[:, 0].tolist() == [22]
    assert second["batch_size"] == 1
    torch.testing.assert_close(second["pixel_values"], image.unsqueeze(0))
    assert second["image_sizes"] == [[28, 42]]
    assert second["page_table"][0, 0].item() == 20

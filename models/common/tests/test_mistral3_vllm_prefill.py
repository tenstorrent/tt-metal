# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral3ForConditionalGeneration.prefill_forward must give the vllm-tt-plugin images to the model.

vllm-tt-plugin passes images as ``pixel_values``: one entry per request, None or a list of
[C, H, W] tensors cut to each image's size. The adapter used to read only an ``images`` keyword,
which the plugin does not send, so vLLM served every image request as text only.

Host-only: Generator.prefill_forward_text is replaced, so no device is opened.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

from models.common.tests.vllm_stubs import stub_vllm_if_missing
from models.tt_transformers.tt.generator import Generator

PAD_TOKEN_ID = 11


def _forwarded_prefill_kwargs(monkeypatch, **plugin_kwargs):
    stub_vllm_if_missing(monkeypatch)
    adapter = importlib.import_module("models.tt_transformers.tt.generator_vllm").Mistral3ForConditionalGeneration
    forwarded = []
    monkeypatch.setattr(
        Generator, "prefill_forward_text", lambda self, *args, **kwargs: forwarded.append(kwargs) or "logits"
    )
    wrapper = adapter.__new__(adapter)
    wrapper.model_args = [SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=PAD_TOKEN_ID))]
    wrapper.data_parallel = 1  # read by Generator.__del__

    result = adapter.prefill_forward(
        wrapper,
        tokens=torch.arange(1, 13).reshape(3, 4),
        prompt_lens=[4, 2, 3],
        page_table="page_table",
        kv_cache="kv_cache",
        **plugin_kwargs,
    )

    assert result == "logits"
    assert len(forwarded) == 1
    return forwarded[0]


@pytest.mark.parametrize("enable_trace", [True, False])
def test_text_only_prefill_keeps_the_plugin_trace_mode(enable_trace, monkeypatch):
    kwargs = _forwarded_prefill_kwargs(
        monkeypatch,
        enable_trace=enable_trace,
        pixel_values=[None, None, None],
        image_grid_thw=[None, None, None],
    )

    assert kwargs["enable_trace"] is enable_trace
    assert kwargs["pixel_values"] is None
    assert kwargs["image_sizes"] is None
    assert kwargs["page_table"] == "page_table"
    assert kwargs["kv_cache"] == "kv_cache"
    assert kwargs["prompt_lens"] == [4, 2, 3]
    torch.testing.assert_close(
        kwargs["tokens"],
        torch.tensor([[1, 2, 3, 4], [5, 6, PAD_TOKEN_ID, PAD_TOKEN_ID], [9, 10, 11, PAD_TOKEN_ID]]),
    )


def test_image_prefill_gets_per_user_pixel_values_and_sizes(monkeypatch):
    one_image = torch.rand(3, 28, 42)
    wide = torch.rand(3, 14, 56)
    tall = torch.rand(3, 42, 28)

    kwargs = _forwarded_prefill_kwargs(
        monkeypatch,
        enable_trace=True,
        pixel_values=[None, [one_image], [wide, tall]],
    )

    # A traced prefill cannot take image features, so a batch with images runs eagerly.
    assert kwargs["enable_trace"] is False
    assert kwargs["image_sizes"] == [None, [[28, 42]], [[14, 56], [42, 28]]]
    pixel_values = kwargs["pixel_values"]
    assert pixel_values[0] is None
    torch.testing.assert_close(pixel_values[1], one_image.unsqueeze(0))
    # Images of one request are padded at the bottom and the right to the largest height and width.
    assert pixel_values[2].shape == (2, 3, 42, 56)
    torch.testing.assert_close(pixel_values[2][0, :, :14, :], wide)
    torch.testing.assert_close(pixel_values[2][1, :, :, :28], tall)
    assert not pixel_values[2][0, :, 14:, :].any()
    assert not pixel_values[2][1, :, :, 28:].any()

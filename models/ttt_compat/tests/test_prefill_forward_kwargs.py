# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every prefill path hands ``ttnn_prefill_forward`` only the keywords it takes.

Callers pass keywords meant for ``prepare_inputs_prefill``: the generator adds ``global_user_id`` for
row-sharded models, multimodal models pass their image inputs, and a serving plugin may pass its own
(``new_request_rows``). The single-chunk path always forwarded only ``page_tables_per_layer`` to the model;
the chunked and prefix-cached paths forwarded everything, so ``Transformer.ttnn_prefill_forward`` raised a
TypeError on every prompt longer than one prefill chunk and on every prefix-cache hit. Host only, no device.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from models.ttt_compat.tt.generator import Generator
from models.ttt_compat.tt.model import Transformer

BLOCK_SIZE = 32
MAX_PREFILL_CHUNK_SIZE = 2048
FORWARD_SIGNATURE = inspect.signature(Transformer.ttnn_prefill_forward)


class _RecordingModel:
    """Takes arguments exactly like ``Transformer.ttnn_prefill_forward`` and records what each call got."""

    def __init__(self):
        self.prepare_calls = []
        self.forward_calls = []

    def prepare_inputs_prefill(self, tokens, **kwargs):
        self.prepare_calls.append(kwargs)
        return "x", "rot_mats_global", "rot_mats_local", "page_table", "chunk_page_table", "chunk_start_idx"

    def ttnn_prefill_forward(self, *args, **kwargs):
        FORWARD_SIGNATURE.bind(self, *args, **kwargs)  # raises TypeError on a keyword the model does not take
        self.forward_calls.append(kwargs)
        return "logits"


def _generator(model):
    generator = Generator.__new__(Generator)  # host-only: no mesh device, no weights
    generator.model = [model]
    generator.data_parallel = 1
    generator.model_args = [SimpleNamespace(max_prefill_chunk_size=MAX_PREFILL_CHUNK_SIZE)]
    return generator


@pytest.mark.parametrize(
    "seq_len,num_cached_tokens,expected_forward_calls",
    [
        (128, 0, 1),  # one chunk
        (2 * MAX_PREFILL_CHUNK_SIZE, 0, 2),  # chunked prefill
        (128, 256, 1),  # prefix-cache hit
    ],
    ids=["single_chunk", "chunked", "prefix_cached"],
)
@pytest.mark.parametrize("with_page_tables_per_layer", [False, True])
def test_prefill_forwards_only_the_keywords_the_model_takes(
    seq_len, num_cached_tokens, expected_forward_calls, with_page_tables_per_layer
):
    model = _RecordingModel()
    total_len = num_cached_tokens + seq_len
    num_blocks = total_len // BLOCK_SIZE
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    kv_cache = [[SimpleNamespace(shape=(num_blocks, 1, BLOCK_SIZE, 64))]]
    extra_kwargs = {"global_user_id": 3, "new_request_rows": [0]}
    if with_page_tables_per_layer:
        extra_kwargs["page_tables_per_layer"] = ["per-layer page table"]

    logits = _generator(model).prefill_forward_single_user_text(
        torch.zeros(1, seq_len, dtype=torch.long),
        page_table=page_table,
        user_id=0,
        last_token_idx=total_len - 1,
        kv_cache=kv_cache,
        num_cached_tokens=num_cached_tokens,
        **extra_kwargs,
    )

    assert logits == "logits"
    assert len(model.forward_calls) == expected_forward_calls
    for call in model.forward_calls:
        assert "global_user_id" not in call and "new_request_rows" not in call
        assert call.get("page_tables_per_layer") == extra_kwargs.get("page_tables_per_layer")
    # prepare_inputs_prefill still gets every caller keyword, chunk by chunk
    for call in model.prepare_calls:
        assert {k: call[k] for k in extra_kwargs} == extra_kwargs

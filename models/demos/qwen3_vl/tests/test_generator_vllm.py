# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from models.demos.qwen3_vl.tt import generator_vllm


def test_single_user_prefill_receives_sliced_multi_user_page_table():
    class FakeTTGenerator:
        def __init__(self):
            self.calls = []

        def _get_prefill_user_page_table(self, page_table, kv_cache, prefill_len):
            self.calls.append((page_table, kv_cache, prefill_len))
            return page_table[:, :17]

    class FakeVLLMGenerator:
        def __init__(self):
            self._ttt_generator = FakeTTGenerator()
            self.prefill_kwargs = None

        def prefill_forward_single_user_text(self, input_prefill, **kwargs):
            self.prefill_kwargs = kwargs
            return input_prefill

    generator = FakeVLLMGenerator()
    input_prefill = object()
    page_table = torch.arange(128, dtype=torch.int32).reshape(2, 64)
    kv_cache = object()
    rot_mats = object()
    deepstack_visual_embeds = object()

    result = generator_vllm._prefill_single_user_with_sliced_page_table(
        generator,
        input_prefill,
        page_table,
        user_id=1,
        decoding_pos=1043,
        rot_mats=rot_mats,
        kv_cache=kv_cache,
        deepstack_visual_embeds=deepstack_visual_embeds,
    )

    assert result is input_prefill
    assert generator._ttt_generator.calls == [(page_table, kv_cache, 1043)]
    sliced_page_table = generator.prefill_kwargs.pop("page_table")
    assert torch.equal(sliced_page_table, page_table[:, :17])
    assert generator.prefill_kwargs == {
        "user_id": 1,
        "last_token_idx": 1042,
        "rot_mats": rot_mats,
        "kv_cache": kv_cache,
        "deepstack_visual_embeds": deepstack_visual_embeds,
    }

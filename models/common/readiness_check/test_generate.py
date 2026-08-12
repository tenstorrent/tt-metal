# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from models.common.readiness_check.generate import _chat_or_plain_prompt_tokens


class _BatchEncodingTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return {"input_ids": torch.tensor([[11, 22, 33]])}


def test_chat_template_accepts_batch_encoding_input_ids():
    assert _chat_or_plain_prompt_tokens(_BatchEncodingTokenizer(), "prompt", chat_template=True) == [11, 22, 33]

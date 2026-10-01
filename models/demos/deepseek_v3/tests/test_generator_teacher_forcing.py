# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from models.demos.deepseek_v3.tt.generator import _apply_teacher_forcing_tokens


class _TeacherForcingStub:
    def __init__(self, forced_tokens: list[int]):
        self.forced_tokens = forced_tokens
        self.calls: list[tuple[int, int]] = []

    def collect_predicted_tokens(self, tt_pred_token: int, *, user_idx: int = 0) -> int:
        self.calls.append((user_idx, int(tt_pred_token)))
        return self.forced_tokens[user_idx]


def test_apply_teacher_forcing_tokens_updates_each_prompt_user():
    tokens = torch.tensor([10, 20, 30], dtype=torch.long)
    teacher_forcing = _TeacherForcingStub([101, 201, 301])

    _apply_teacher_forcing_tokens(tokens, teacher_forcing, num_of_prompts=3)

    assert tokens.tolist() == [101, 201, 301]
    assert teacher_forcing.calls == [(0, 10), (1, 20), (2, 30)]


def test_apply_teacher_forcing_tokens_uses_prompt_user_ids():
    tokens = torch.tensor([10, -1, 20, -1, 30, -1], dtype=torch.long)
    teacher_forcing = _TeacherForcingStub([101, 201, 301])
    prompt_user_ids = torch.tensor([0, 2, 4], dtype=torch.long)

    _apply_teacher_forcing_tokens(tokens, teacher_forcing, num_of_prompts=3, prompt_user_ids=prompt_user_ids)

    assert tokens.tolist() == [101, -1, 201, -1, 301, -1]
    assert teacher_forcing.calls == [(0, 10), (1, 20), (2, 30)]

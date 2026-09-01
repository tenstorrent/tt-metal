# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.common.readiness_check.run_teacher_forcing import _run_one_entry
from models.common.readiness_check.schema import Reference, ReferenceEntry
from models.common.readiness_check.teacher_forcing import TokenAccuracy


class _CallbackGenerator:
    def __init__(self, predictions):
        self.predictions = predictions

    def generate(self, prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True):
        assert enable_trace is True
        emitted = []
        for step, prediction in enumerate(self.predictions[:max_new_tokens]):
            emitted.append(prediction)
            if next_input is not None:
                next_input(step, prediction)
        return emitted


class _NoCallbackGenerator:
    def generate(self, prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True):
        assert enable_trace is True
        return [11, 12, 13][:max_new_tokens]


def _make_accuracy() -> TokenAccuracy:
    reference = Reference(
        k=5,
        hf_model_id="unit-test",
        token_ids_meta={},
        entries=[
            ReferenceEntry(
                prompt_text="prompt",
                prompt_tokens=torch.tensor([[101, 102]], dtype=torch.int64),
                generated_tokens=torch.tensor([[11, 12, 13]], dtype=torch.int64),
                topk_tokens=torch.tensor(
                    [
                        [11, 1, 2, 3, 4],
                        [12, 1, 2, 3, 4],
                        [13, 1, 2, 3, 4],
                    ],
                    dtype=torch.int32,
                ),
                tf_prompt_len=2,
            )
        ],
    )
    return TokenAccuracy(reference)


def test_run_one_entry_scores_full_teacher_forcing_run():
    stats = _run_one_entry(generator=_CallbackGenerator([11, 12, 13]), acc=_make_accuracy(), entry_idx=0)

    assert stats["total"] == 3
    assert stats["matches_top1"] == 3
    assert stats["top1"] == 1.0


def test_run_one_entry_fails_when_generate_stops_before_reference_length():
    with pytest.raises(RuntimeError, match="produced 2/3 predictions"):
        _run_one_entry(generator=_CallbackGenerator([11, 12]), acc=_make_accuracy(), entry_idx=0)


def test_run_one_entry_fails_when_generate_never_calls_next_input():
    with pytest.raises(RuntimeError, match="produced 0/3 predictions"):
        _run_one_entry(generator=_NoCallbackGenerator(), acc=_make_accuracy(), entry_idx=0)

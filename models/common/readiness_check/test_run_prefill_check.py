# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.common.readiness_check.run_prefill_check import _run_one_entry_prefill
from models.common.readiness_check.schema import Reference, ReferenceEntry


def _reference() -> Reference:
    return Reference(
        k=5,
        hf_model_id="unit-test",
        entries=[
            ReferenceEntry(
                prompt_text="prompt",
                prompt_tokens=torch.tensor([[101, 102]], dtype=torch.int64),
                generated_tokens=torch.tensor([[11, 12, 13]], dtype=torch.int64),
                topk_tokens=torch.tensor(
                    [[11, 1, 2, 3, 4], [12, 1, 2, 3, 4], [13, 1, 2, 3, 4]],
                    dtype=torch.int32,
                ),
                tf_prompt_len=2,
            )
        ],
    )


class _PrefillGenerator:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.received_tokens = None

    def prefill_logits(self, prompt_token_ids):
        self.received_tokens = prompt_token_ids
        return self.logits


def test_run_one_entry_prefill_uses_generator_owned_high_level_path():
    reference = _reference()
    logits = torch.full((1, 5, 20), -100.0)
    logits[0, 1, 11] = 1.0
    logits[0, 2, 12] = 1.0
    logits[0, 3, 13] = 1.0
    generator = _PrefillGenerator(logits)

    stats = _run_one_entry_prefill(generator=generator, entry=reference.entries[0], reference=reference)

    assert generator.received_tokens == [101, 102, 11, 12, 13]
    assert stats["matches_top1"] == 3
    assert stats["top1"] == 1.0


def test_run_one_entry_prefill_rejects_padded_or_truncated_logit_shape():
    reference = _reference()
    generator = _PrefillGenerator(torch.zeros((1, 6, 20)))

    with pytest.raises(RuntimeError, match=r"logical shape \[1, 5, vocab\]"):
        _run_one_entry_prefill(generator=generator, entry=reference.entries[0], reference=reference)

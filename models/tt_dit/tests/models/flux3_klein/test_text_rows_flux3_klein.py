# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Klein text encoder rows (host): warm-up runs the encoder at every row count a prompt can reach."""

import pytest

from models.tt_dit.pipelines.flux3_klein import prompt_encoder
from models.tt_dit.pipelines.flux3_klein.packing import TEXT_BUCKETS, TEXT_PAD_MULTIPLE
from models.tt_dit.pipelines.flux3_klein.prompt_encoder import ENCODER_ROW_STEP, Flux3KleinPromptEncoder, encoder_rows


@pytest.mark.parametrize("bucket", TEXT_BUCKETS)
def test_warmup_covers_every_prompt(bucket: int, monkeypatch) -> None:
    monkeypatch.setattr(prompt_encoder.ttnn, "deallocate", lambda tensor: None)
    encoder, warmed = object.__new__(Flux3KleinPromptEncoder), []

    def record(input_ids, attention_mask, bucket_rows):
        rows, real = attention_mask.shape[-1], int(attention_mask.sum())
        valid = -(-real // TEXT_PAD_MULTIPLE) * TEXT_PAD_MULTIPLE
        assert input_ids.shape == attention_mask.shape and bucket_rows == bucket
        assert encoder_rows(valid, bucket) == rows, (real, valid, rows)
        warmed.append(rows)

    encoder._encode = record
    encoder.warmup(bucket)
    assert len(warmed) == bucket // ENCODER_ROW_STEP
    reachable = {encoder_rows(valid, bucket) for valid in range(TEXT_PAD_MULTIPLE, bucket + 1, TEXT_PAD_MULTIPLE)}
    assert reachable | {encoder_rows(bucket, bucket)} == set(warmed)

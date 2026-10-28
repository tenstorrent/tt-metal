# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Serving wrapper (demo/generator_vllm.py) as tt-inference-server drives it.

The server builds `EmbeddingGemma2ForEmbedding(device, max_batch_size, max_seq_len, model_name)`
and calls `forward(input_ids, attention_mask)` on right-padded [B, L] batches. Every served
(batch, bucket) is captured in the constructor; rows are grouped by the bucket of their own
length, and each group is padded up to a ladder batch and replayed. The embedding of each row must match HF fp32 and come back in input order.
"""

import pytest
import torch

from models.demos.embedding_gemma2.demo.generator_vllm import (
    EMBEDDING_DIM,
    EmbeddingGemma2ForEmbedding,
    batch_ladder,
    group_rows_by_bucket,
    plan_chunks,
    serving_shapes,
)
from models.demos.embedding_gemma2.tests.common import COSINE_EMBEDDING, case_id, cosine_rows, log_result

MAX_BATCH = 8
MAX_SEQ_LEN = 512

# request batches: one row, a padded batch (4 -> ladder 8), a full batch of mixed lengths, and one
# long row among short ones (rows run grouped by the bucket of their own length)
CASES = (
    (100,),
    (300, 37, 512, 5),
    (512, 480, 400, 300, 257, 129, 64, 7),
    (20, 500, 9, 31, 300),
)


def test_serving_plan():
    ladder = batch_ladder(32)
    assert ladder == (1, 8, 32)
    assert batch_ladder(4) == (1, 4)
    shapes = serving_shapes(ladder, (128, 512, 2048, 8192), 8192, 8192)
    assert (1, 8192) in shapes and (32, 128) in shapes and (8, 2048) not in shapes and (8, 512) in shapes
    # 20 rows at 512: cap is the largest ladder batch with batch * 512 <= 8192, i.e. 8
    assert plan_chunks(20, 512, ladder, 8192) == [(0, 8, 8), (8, 16, 8), (16, 20, 8)]
    assert plan_chunks(3, 128, ladder, 8192) == [(0, 3, 8)]
    assert plan_chunks(1, 8192, ladder, 8192) == [(0, 1, 1)]


def test_group_rows_by_bucket():
    lengths = torch.tensor([20, 500, 9, 31, 300])
    mask = (torch.arange(512)[None, :] < lengths[:, None]).long()
    groups = [(idx.tolist(), n) for idx, n in group_rows_by_bucket(mask, 512)]
    assert groups == [([0, 2, 3], 31), ([1, 4], 500)]
    # a mask that is not right-padded stays one group at full length
    mask[0] = torch.flip(mask[0], dims=[0])
    ((idx, n),) = group_rows_by_bucket(mask, 512)
    assert idx.tolist() == [0, 1, 2, 3, 4] and n == 512


@pytest.fixture(scope="module")
def wrapper(eg2_device, eg2_snapshot):
    model = EmbeddingGemma2ForEmbedding(
        device=eg2_device, max_batch_size=MAX_BATCH, max_seq_len=MAX_SEQ_LEN, model_name=str(eg2_snapshot)
    )
    yield model
    model.release()


@pytest.mark.parametrize("lengths", CASES, ids=[case_id(c) for c in CASES])
def test_forward_matches_hf(lengths, wrapper, reference_for):
    ids, mask, ref = reference_for(lengths)
    emb = wrapper.forward(ids, attention_mask=mask)
    assert emb.shape == (len(lengths), EMBEDDING_DIM) and emb.dtype == torch.float32
    assert torch.allclose(emb.norm(dim=-1), torch.ones(len(lengths)), atol=1e-3)
    cos = cosine_rows(ref.embedding, emb)
    for b in range(len(lengths)):
        assert log_result(
            "serving_embedding_cosine", float(cos[b]), COSINE_EMBEDDING, case=list(lengths), row=b
        ), f"row {b} (len {lengths[b]}) cosine {float(cos[b])}"

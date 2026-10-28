# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced text forward: replay == eager, bucket reuse across logical lengths, determinism.

One trace is captured per (batch, bucket). Replays refresh the persistent input tensors with
`copy_host_to_device_tensor`, so a new prompt of a different logical length in the same bucket
reuses the trace.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.embedding_gemma2.tests.common import COSINE_EMBEDDING, case_id, cosine_rows, log_result, make_batch
from models.demos.embedding_gemma2.tt.text_encoder import TextEncoder

N_REPEAT = 5

# (capture lengths, replay lengths in the same bucket)
TRACE_CASES = (
    ((100,), (77,)),
    ((1025,), (2000,)),
    ((1025, 513, 31), (1500, 33, 700)),
    ((512,) * 8, (512, 480, 400, 300, 257, 129, 64, 7)),
)


@pytest.fixture(scope="module")
def encoder(eg2_device, tt_text_model):
    enc = TextEncoder(tt_text_model, eg2_device)
    # every shape this module runs, eager or traced, is compiled + captured up front
    enc.capture_all([(1, 128), (1, 2048), (3, 2048), (8, 512), (1, 8192)])
    yield enc
    enc.release_traces()


def _eager(encoder, ids, mask):
    return encoder.encode(ids, mask, traced=False, return_last_hidden=True)


def _replay(encoder, ids, mask):
    emb, last = encoder.encode(ids, mask, traced=True, return_last_hidden=True)
    return emb.clone(), last.clone()


@pytest.mark.parametrize("first,second", TRACE_CASES, ids=[case_id(c[0]) for c in TRACE_CASES])
def test_trace_matches_eager(first, second, encoder, eg2_snapshot, reference_for):
    for lengths in (first, second):
        ids, mask = make_batch(eg2_snapshot, list(lengths))
        e_emb, e_last = _eager(encoder, ids, mask)
        t_emb, t_last = _replay(encoder, ids, mask)
        valid = mask.bool()
        d_emb = (e_emb - t_emb).abs().max().item()
        d_last = (e_last[valid] - t_last[valid]).abs().max().item()
        logger.info(f"{case_id(lengths)} replay vs eager: max|d emb| {d_emb:.3e}, max|d last| {d_last:.3e}")
        log_result("trace_vs_eager_max_abs_emb", d_emb, 0.0, case=list(lengths))
        assert torch.equal(e_emb, t_emb), f"trace replay embedding differs from eager (max {d_emb})"
        assert torch.equal(e_last[valid], t_last[valid]), f"trace replay last hidden differs (max {d_last})"

        # replay output also meets the HF bar
        _, _, ref = reference_for(lengths)
        cos = cosine_rows(ref.embedding, t_emb)
        for b in range(len(lengths)):
            assert log_result(
                "trace_embedding_cosine", float(cos[b]), COSINE_EMBEDDING, case=list(lengths), row=b
            ), f"row {b} cosine {float(cos[b])}"
    assert len(encoder._traces) >= 1


@pytest.mark.parametrize("lengths", [(1025, 513, 31), (8192,)], ids=["b3_pad", "b1_s8192"])
def test_determinism(lengths, encoder, eg2_snapshot):
    ids, mask = make_batch(eg2_snapshot, list(lengths))
    base_emb, base_last = _replay(encoder, ids, mask)
    for _ in range(N_REPEAT - 1):
        emb, last = _replay(encoder, ids, mask)
        assert torch.equal(base_emb, emb), "traced replay is not deterministic (embedding)"
        assert torch.equal(base_last, last), "traced replay is not deterministic (last hidden)"
    for _ in range(2):
        emb, last = _eager(encoder, ids, mask)
        assert torch.equal(base_emb, emb), "eager run differs from replay (embedding)"
    ttnn.synchronize_device(encoder.device)
    logger.info(f"determinism {case_id(lengths)}: {N_REPEAT} replays + 2 eager runs bit-identical")

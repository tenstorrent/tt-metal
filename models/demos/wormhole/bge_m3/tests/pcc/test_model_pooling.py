# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC/score tests for the BgeM3Model `pooling=` heads (dense, sparse, ColBERT).

Mirrors ``test_generator_vllm.py`` but drives the core model directly via
``create_tt_model(pooling=...)`` instead of the ``BgeM3ForEmbedding`` wrapper.

The model emits the *raw* head output for each pooling mode:
  * ``pooling="cls"``     -> [B, 1, 1, D]   (first-token hidden)
  * ``pooling="colbert"`` -> [B, 1, S, D]   (colbert_linear projection)
  * ``pooling="sparse"``  -> [B, 1, S, 1]   (sparse_linear token weights)

All downstream post-processing (CLS crop, attention masking, L2 normalize,
vocab scatter, special-token masking) is done here in torch, then scored with
the same ``m3_scores`` helpers and checked against the same fixed reference
tensors as the vLLM test.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.demo.generator_vllm import _crop_hidden_state_ttnn, _flatten_sparse_token_weights_ttnn
from models.demos.wormhole.bge_m3.demo.m3_scores import (
    _get_special_token_ids,
    _sparse_embedding_scatter_ttnn,
    compute_colbert_score_torch,
    compute_dense_score_torch,
    compute_sparse_score_torch,
)
from models.demos.wormhole.bge_m3.tests.test_utils import require_single_device
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_NAME = "BAAI/bge-m3"
MAX_MODEL_LEN = 512

# Same example queries/documents and fixed reference tensors as test_generator_vllm.py.
sentences_1 = ["What is BGE M3?", "Definition of BM25"]
sentences_2 = [
    "BGE M3 is an embedding model supporting dense retrieval, " "lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set "
    "of documents based on the query terms appearing in each document",
]

similarity_reference = torch.tensor([[0.6259, 0.3474], [0.3309, 0.6734]], dtype=torch.float32)
lexical_score_reference = [0.19554901123046875, 0.0]
colbert_score_reference = [0.7797, 0.4620]
corner_case_token_id = 2673
corner_case_token_weight = 0.26710861921310425


def _dense_allclose_kwargs() -> dict:
    # Blackhole drifts slightly more than Wormhole (matches vLLM test BH tolerance).
    return {"rtol": 0.035, "atol": 1e-2}


_COLBERT_REL = 0.025  # Blackhole rel tolerance for scalar colbert scores
_SPARSE_REL = 0.025
_CORNER_SPARSE_REL = 0.04


@pytest.fixture(scope="module")
def model_artifacts(model_location_generator):
    # Resolve the local checkpoint path only; the full state_dict (including the
    # colbert_linear / sparse_linear heads) is built by ModelArgs.load_state_dict
    # inside create_tt_model, which reads the already-cached head .pt files
    # (local_files_only) -- no download.
    model_id_or_path = str(model_location_generator(MODEL_NAME, download_if_ci_v2=True, ci_v2_timeout_in_s=1800))
    return model_id_or_path


def _build_pooled_model(device, model_id_or_path, pooling):
    require_single_device(device)
    # state_dict=None -> ModelArgs.load_state_dict() populates base encoder +
    # colbert_linear + sparse_linear heads from the local cache.
    model_args, model, _ = create_tt_model(
        mesh_device=device,
        max_batch_size=max(len(sentences_1), len(sentences_2), 1),
        max_seq_len=MAX_MODEL_LEN,
        dtype=ttnn.bfloat8_b,
        state_dict=None,
        hf_model_name=model_id_or_path,
        pooling=pooling,
    )
    return model_args, model


def _encode(model_args, sentences):
    enc = model_args.encode_prompts(sentences, attention_mask_4d=False)
    return enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))


def _forward_pooled_ttnn(model, model_args, device, sentences):
    """Run a single pooled forward; returns (raw_pooled_ttnn, input_ids, attention_mask).

    Keeps the head output as a ttnn tensor so the sparse path can run the exact
    same ttnn scatter the vLLM wrapper uses (matching its numerics bit-for-bit).
    """
    input_ids, attention_mask, token_type_ids = _encode(model_args, sentences)
    out = model.forward(
        input_ids=ttnn.from_torch(
            input_ids.to(torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        attention_mask=None,
        token_type_ids=ttnn.from_torch(
            token_type_ids.to(torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
    )
    return out, input_ids, attention_mask


def _forward_pooled(model, model_args, device, sentences):
    """Run a single pooled forward; returns (raw_pooled_torch, input_ids, attention_mask)."""
    out, input_ids, attention_mask = _forward_pooled_ttnn(model, model_args, device, sentences)
    pooled = to_torch_auto_compose(out).to(torch.float32)
    return pooled, input_ids, attention_mask


# ── torch post-processing (replicates the vLLM wrapper, but host-side) ──


def _dense_from_cls(pooled: torch.Tensor) -> torch.Tensor:
    """cls head output [B,1,1,D] -> normalized dense vec [B, D]."""
    vec = pooled
    while vec.dim() > 2:
        vec = vec.squeeze(1)
    return F.normalize(vec.float(), dim=-1)


def _colbert_from_head(pooled: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """colbert head output [B,1,S,D] -> CLS-dropped, masked, normalized [B, S-1, D]."""
    vec = pooled
    if vec.dim() == 4 and vec.shape[1] == 1:
        vec = vec.squeeze(1)  # [B, S, D]
    S = vec.shape[1]
    vec = vec[:, 1:S].float()  # drop CLS
    mask = attention_mask[:, 1:S].unsqueeze(-1).to(vec.dtype)
    vec = vec * mask
    return F.normalize(vec, dim=-1)


def _sparse_from_head_ttnn(out_tt, input_ids, vocab_size, unused_ids, device):
    """sparse head ttnn output [B,1,S,1] -> [B, vocab] sparse vec.

    Runs the EXACT same ttnn pipeline the vLLM wrapper uses
    (_crop_hidden_state_ttnn -> _flatten_sparse_token_weights_ttnn ->
    _sparse_embedding_scatter_ttnn) so the result matches the wrapper bit-for-bit
    on the same silicon -- not a more-precise host reimplementation.
    """
    batch_size, seq_len = input_ids.shape
    tw = _crop_hidden_state_ttnn(out_tt, batch_size, seq_len)
    tw = _flatten_sparse_token_weights_ttnn(tw)
    ids_tt = ttnn.from_torch(
        input_ids.long().to(torch.int32),
        device=device,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    sparse_tt = _sparse_embedding_scatter_ttnn(device, tw, ids_tt, vocab_size, unused_ids)
    return to_torch_auto_compose(sparse_tt).to(torch.float32)


# ── tests (one model build per pooling mode) ──


def test_model_pooling_dense_embedding(device, model_artifacts, reset_seeds):
    model_id_or_path = model_artifacts
    model_args, model = _build_pooled_model(device, model_id_or_path, "cls")

    p1, _, _ = _forward_pooled(model, model_args, device, sentences_1)
    p2, _, _ = _forward_pooled(model, model_args, device, sentences_2)
    q = _dense_from_cls(p1)
    p = _dense_from_cls(p2)

    similarity = compute_dense_score_torch(q, p)
    assert torch.allclose(similarity, similarity_reference, **_dense_allclose_kwargs())


def test_model_pooling_colbert_embedding(device, model_artifacts, reset_seeds):
    model_id_or_path = model_artifacts
    model_args, model = _build_pooled_model(device, model_id_or_path, "colbert")

    p1, _, am1 = _forward_pooled(model, model_args, device, sentences_1)
    p2, _, am2 = _forward_pooled(model, model_args, device, sentences_2)
    q = _colbert_from_head(p1, am1)
    p = _colbert_from_head(p2, am2)

    colbert_scores = compute_colbert_score_torch(q, p, q_mask=am1)
    assert float(colbert_scores[0, 0]) == pytest.approx(colbert_score_reference[0], rel=_COLBERT_REL)
    assert float(colbert_scores[0, 1]) == pytest.approx(colbert_score_reference[1], rel=_COLBERT_REL)


def test_model_pooling_sparse_embedding(device, model_artifacts, reset_seeds):
    model_id_or_path = model_artifacts
    model_args, model = _build_pooled_model(device, model_id_or_path, "sparse")
    vocab_size = model_args.vocab_size
    unused = _get_special_token_ids(model_args.tokenizer, vocab_size)

    o1, ids1, _ = _forward_pooled_ttnn(model, model_args, device, sentences_1)
    sp1 = _sparse_from_head_ttnn(o1, ids1, vocab_size, unused, device)
    o2, ids2, _ = _forward_pooled_ttnn(model, model_args, device, sentences_2)
    sp2 = _sparse_from_head_ttnn(o2, ids2, vocab_size, unused, device)
    sp1 = torch.tensor(sp1)
    sp2 = torch.tensor(sp2)

    cross = compute_sparse_score_torch(sp1, sp2)
    self_scores = compute_sparse_score_torch(sp1[:1], sp1[1:2])
    assert float(cross[0, 0]) == pytest.approx(lexical_score_reference[0], rel=_SPARSE_REL)
    assert float(self_scores[0, 0]) == pytest.approx(lexical_score_reference[1], abs=1e-3)


def test_model_pooling_sparse_corner_case(device, model_artifacts, reset_seeds):
    model_id_or_path = model_artifacts
    model_args, model = _build_pooled_model(device, model_id_or_path, "sparse")
    vocab_size = model_args.vocab_size
    unused = _get_special_token_ids(model_args.tokenizer, vocab_size)

    out, ids, _ = _forward_pooled_ttnn(model, model_args, device, ["Hi"])
    sparse = _sparse_from_head_ttnn(out, ids, vocab_size, unused, device)
    corner_weight = float(sparse[0, corner_case_token_id])
    assert corner_weight == pytest.approx(corner_case_token_weight, rel=_CORNER_SPARSE_REL)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

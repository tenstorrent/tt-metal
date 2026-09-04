# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced data-parallel path of the BGE-M3 serving wrapper, on a 1x2 N300.

The tier-2 command names its files one by one, and this one is not on that list.
It needs two chips, and the registered BGE-M3 tests run on a single card. Run it
by hand:

    TT_VISIBLE_DEVICES=0 HF_MODEL=BAAI/bge-m3 \
    pytest models/demos/wormhole/bge_m3/tests/pcc/generator_vllm_dp.py -s

``tests/pcc/model_dp.py`` drives ``BgeM3Model`` directly. These tests drive
``BgeM3ForEmbedding``, the class vLLM and tt-media-server load, so they cover the
wrapper: trace capture, buffer refill, batch sharding, on-device pooling, and
replay.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.wormhole.bge_m3.demo.generator_vllm import BgeM3ForEmbedding

MODEL_ID = "BAAI/bge-m3"
DP_BATCH_SIZE = 12
DP_SEQ_LEN = 8192
EMBED_DIM = 1024

# Cosine against the HF reference. The serving path runs BF8 weights with BF4 K
# and V, so it does not match HF to PCC 0.99. model_dp.py gates the numerics;
# this bound catches a wrong mask or a stale buffer, which move it far more.
COSINE_FLOOR = 0.90

MESH_PARAMS = pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["dp2_n300"])
DEVICE_PARAMS = pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 50_000_000}],
    indirect=True,
)


@pytest.fixture(scope="module")
def prompts():
    """Twelve prompts of different lengths, so every row has its own valid length."""
    base = [
        "Artificial intelligence changes how people work with computers.",
        "Quantum computing solves problems that classical machines cannot.",
        "Machine learning finds patterns in very large collections of data.",
        "Renewable energy costs fall every year across most markets.",
        "The library held thousands of manuscripts from many centuries.",
        "Neural networks learn a representation from examples alone.",
    ]
    return [text * (1 + index % 3) for index, text in enumerate(base * 2)]


@pytest.fixture(scope="module")
def encoded(prompts):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    return tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=DP_SEQ_LEN,
        return_tensors="pt",
    )


@pytest.fixture(scope="module")
def reference(encoded):
    """Mean-pooled HF output for the same tokens."""
    transformers = pytest.importorskip("transformers")
    hf_model = transformers.AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        hidden = hf_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        ).last_hidden_state
    keep = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)


def _build(mesh_device, **kwargs):
    return BgeM3ForEmbedding(
        device=mesh_device,
        max_batch_size=DP_BATCH_SIZE,
        max_seq_len=DP_SEQ_LEN,
        dtype=ttnn.bfloat8_b,
        **kwargs,
    )


def _fixed_length_batch(valid_length):
    """One batch where every row holds ``valid_length`` real tokens."""
    input_ids = torch.zeros((DP_BATCH_SIZE, DP_SEQ_LEN), dtype=torch.long)
    attention_mask = torch.zeros((DP_BATCH_SIZE, DP_SEQ_LEN), dtype=torch.long)
    torch.manual_seed(valid_length)
    for row in range(DP_BATCH_SIZE):
        input_ids[row, :valid_length] = torch.randint(5, 1000, (valid_length,))
        attention_mask[row, :valid_length] = 1
    return input_ids, attention_mask


@MESH_PARAMS
@DEVICE_PARAMS
def test_wrapper_dp2_matches_hf(mesh_device, encoded, reference, reset_seeds):
    """The wrapper selects the data-parallel path and agrees with HF."""
    assert tuple(mesh_device.shape) == (2, 1)

    model = _build(mesh_device)
    try:
        first = model.forward(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )["dense_vecs"]

        assert model._data_parallel, "the wrapper did not select the data-parallel path"
        assert model._traced_inputs is not None, "the wrapper did not capture a trace"
        assert first.shape == (DP_BATCH_SIZE, EMBED_DIM)
        assert torch.isfinite(first).all(), "the embeddings hold non-finite values"

        cosine = F.cosine_similarity(first.float(), reference, dim=-1)
        print(f"GATE_COSINE_WRAPPER_DP2 mean={cosine.mean():.6f} min={cosine.min():.6f}")
        assert cosine.min() > COSINE_FLOOR, f"cosine against HF fell to {cosine.min():.4f}"
    finally:
        model.release()


@MESH_PARAMS
@DEVICE_PARAMS
def test_replay_follows_the_valid_lengths(mesh_device, reset_seeds):
    """A second request with different valid lengths pools over its own tokens.

    The pooling helpers hold the valid length of each row. An earlier version built
    them once at capture, so this second request reused the first request's lengths
    and pooled over the wrong span. The trace replay must refill them.
    """
    model = _build(mesh_device)
    try:
        short_ids, short_mask = _fixed_length_batch(64)
        long_ids, long_mask = _fixed_length_batch(2048)

        # Capture on the short batch, then replay on the long one.
        model.forward(input_ids=short_ids, attention_mask=short_mask)
        after_long = model.forward(input_ids=long_ids, attention_mask=long_mask)["dense_vecs"].clone()

        # A fresh wrapper pools the long batch with no earlier state to inherit.
        model.release()
        reference_model = _build(mesh_device)
        try:
            expected = reference_model.forward(input_ids=long_ids, attention_mask=long_mask)["dense_vecs"].clone()
        finally:
            reference_model.release()

        cosine = F.cosine_similarity(after_long.float(), expected.float(), dim=-1)
        print(f"GATE_COSINE_REPLAY_LENGTHS min={cosine.min():.6f}")
        assert (
            cosine.min() > 0.999
        ), f"replay pooled over the wrong tokens: cosine {cosine.min():.4f} against a fresh wrapper"
    finally:
        model.release()


@MESH_PARAMS
@DEVICE_PARAMS
def test_release_frees_the_trace(mesh_device, reset_seeds):
    """release() returns the trace region, so the wrapper can be rebuilt."""
    input_ids, attention_mask = _fixed_length_batch(128)

    for _ in range(3):
        model = _build(mesh_device)
        output = model.forward(input_ids=input_ids, attention_mask=attention_mask)["dense_vecs"]
        assert output.shape == (DP_BATCH_SIZE, EMBED_DIM)
        assert model._traced_inputs is not None

        model.release()
        assert model._traced_inputs is None
        assert model._pool_mask is None
        assert model._pool_counts is None
        model.release()  # calling it twice is safe


@MESH_PARAMS
@DEVICE_PARAMS
def test_heads_match_the_base_path(mesh_device, expect_error, reset_seeds):
    """The traced path returns the heads the caller asks for, and rejects the rest."""
    input_ids, attention_mask = _fixed_length_batch(64)

    model = _build(mesh_device, return_dense=False)
    try:
        assert model.forward(input_ids=input_ids, attention_mask=attention_mask) == {}
    finally:
        model.release()

    for method in ("cls", "mean"):
        model = _build(mesh_device, sentence_pooling_method=method)
        try:
            output = model.forward(input_ids=input_ids, attention_mask=attention_mask)["dense_vecs"]
            assert output.shape == (DP_BATCH_SIZE, EMBED_DIM), method
            assert torch.isfinite(output).all(), method
        finally:
            model.release()

    # The base path serves last_token. The traced path must say so, not return a
    # mean embedding and report success.
    model = _build(mesh_device, sentence_pooling_method="last_token")
    try:
        with expect_error(NotImplementedError, "cls and mean"):
            model.forward(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        model.release()

    for return_head in ("return_sparse", "return_colbert"):
        model = _build(mesh_device, **{return_head: True})
        try:
            with expect_error(NotImplementedError, "dense vectors only"):
                model.forward(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            model.release()

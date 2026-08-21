# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 token embedding — ``Gemma4Model.embed_tokens`` / ``raw_embed``.

Embedding is the one block whose correctness was previously inferred from
downstream logits: ``test_model`` calls ``embed_tokens`` only to *feed* the
model and asserts on the final logits, so a fault here surfaces as a whole-stack
PCC drop rather than a located failure. These tests gate the block directly.

What is specific to Gemma4 (and therefore what these tests pin):

* The ``sqrt(hidden_size)`` scale is **baked into the device table at load**, so
  ``embed_tokens`` is a single lookup with no BinaryNg mul — while the tied
  lm_head and the host ``_embed_weight_cpu`` deliberately stay *unscaled*.
  Nothing else asserts that the device table carries the scale and the host copy
  does not.
* Embedding is **column-parallel**: each device holds ``[vocab, hidden/TP]`` and
  an all-gather reconstructs the hidden dim after the lookup. A wrong mapper or
  a missing gather is invisible at TP=1.
* ``layout=ttnn.TILE_LAYOUT`` tilizes *inside* the embedding kernel instead of
  emitting ROW_MAJOR for a separate ``to_layout``. That was adopted as a win on
  the embed+all-gather path claiming bit-identical output;
  ``test_embedding_tile_layout_is_bit_exact`` is what holds that claim.

Small-vocab random weights cover the wiring; ``test_embedding_real_weights``
covers the shipped 262144-row table, where the token indices exceed uint16 and
the trained values exercise the bf16 cast at real dynamic range.

    pytest -k "1x1"   # single card (TP=1, no all-gather)
    pytest -k "1x8"   # T3K (TP=8, column-parallel + all-gather)
    HF_MODEL=google/gemma-4-31B-it pytest models/demos/gemma4/tests/unit/test_embedding.py -k real_weights
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.model import Gemma4Model
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

from ...tests.test_factory import (
    _get_model_path,
    compare_tensors,
    get_pcc_threshold,
    load_real_substate,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
)

# Small enough to build in a second, wide enough that a column-parallel shard of
# the hidden dim is exercised at every TP the suite runs at.
_RANDOM_VOCAB = 2048


def _embedding_only_text_config(vocab_size=None):
    """HF text config truncated to the smallest model that still owns an embedding.

    ``Gemma4Model`` always builds at least one decoder layer (``num_layers or
    num_hidden_layers``, so 0 means "all"), but with no layer weights in the
    state dict that layer falls back to random weights and is never called here.
    KV sharing and per-layer inputs are zeroed: both are orthogonal to the
    embedding and only add allocation.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(_get_model_path(), trust_remote_code=True)
    tc = getattr(config, "text_config", config)
    if vocab_size is not None:
        tc.vocab_size = vocab_size
    tc.num_hidden_layers = 1
    tc.num_kv_shared_layers = 0
    tc.hidden_size_per_layer_input = 0
    tc._attn_implementation = "eager"
    return tc


def _build_embedding_model(mesh_device, embed_weight, vocab_size=None):
    """A ``Gemma4Model`` whose only real weight is the embedding table.

    Goes through the shipped ``__init__`` rather than reconstructing the tensor
    locally, so the ``embed_scale`` baking and the column-parallel mapper are
    part of what is under test — not part of the test.
    """
    text_config = _embedding_only_text_config(vocab_size)
    model_args = Gemma4ModelArgs.from_hf_config(text_config)

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict={"model.embed_tokens.weight": embed_weight},
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=2048,
        max_local_batch_size=1,
        num_layers=1,
        create_kv_cache=False,
    )
    return model


def _tokens_to_device(tokens, mesh_device):
    """[1, N] int64 token ids -> the uint32 ROW_MAJOR tensor ``embed_tokens`` expects."""
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.from_torch(
        tokens.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )


def _to_torch_2d(tt_tensor, mesh_device, hidden_size):
    """Device-0 view of an embedding output, flattened to ``[N, hidden]``.

    The op returns ``[1, N, hidden]`` at TP=1 and ``[1, 1, N, hidden]`` after the
    all-gather's ``unsqueeze_to_4D``; flattening keeps the assertions shape-agnostic.
    """
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    torch_out = ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0]) if is_mesh else ttnn.to_torch(tt_tensor)
    return torch_out.reshape(-1, hidden_size)


# ── Random-weight PCC ──────────────────────────────────────────────────────


@parametrize_mesh_with_fabric()
@parametrize_batch_seq(
    configs=[(1, 1), (1, 32), (1, 128), (1, 1024)],
    ids=["decode", "decode_batch32", "prefill_128", "prefill_1024"],
)
def test_embedding(batch_size, seq_len, mesh_device, reset_seeds, request):
    """``embed_tokens`` vs HF ``Gemma4TextScaledWordEmbedding``.

    Covers the baked ``embed_scale`` and, at TP>1, the column-parallel shard plus
    the all-gather that rebuilds the hidden dim. ``decode_batch32`` is the decode
    contract (32 users, one token each), which is the shape the demo actually
    embeds every step.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

    text_config = _embedding_only_text_config(_RANDOM_VOCAB)
    hidden_size = text_config.hidden_size

    hf_embed = Gemma4TextScaledWordEmbedding(
        _RANDOM_VOCAB,
        hidden_size,
        padding_idx=getattr(text_config, "pad_token_id", 0),
        embed_scale=hidden_size**0.5,
    )
    hf_embed.eval()
    embed_weight = hf_embed.weight.data.clone().to(torch.bfloat16)

    tokens = torch.randint(0, _RANDOM_VOCAB, (1, seq_len), dtype=torch.long)
    with torch.no_grad():
        ref = hf_embed(tokens).to(torch.bfloat16).reshape(-1, hidden_size)

    model = _build_embedding_model(mesh_device, embed_weight, vocab_size=_RANDOM_VOCAB)
    tt_out = model.embed_tokens(_tokens_to_device(tokens, mesh_device), layout=ttnn.TILE_LAYOUT)
    tt_torch = _to_torch_2d(tt_out, mesh_device, hidden_size)

    passing, pcc_msg = compare_tensors(tt_torch, ref, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"Embedding PCC too low: {pcc_msg}"


@parametrize_mesh_with_fabric()
def test_embedding_applies_embed_scale(mesh_device, reset_seeds):
    """The device table must carry ``sqrt(hidden)``; the host copy must not.

    A PCC check cannot see this — dropping the scale is a uniform gain, which
    leaves PCC at 1.0 while every downstream norm and logit shifts. Assert on the
    magnitude ratio instead, and pin that ``_embed_weight_cpu`` (feeding the tied
    lm_head and the host PLI/parity paths) stayed unscaled.
    """
    text_config = _embedding_only_text_config(_RANDOM_VOCAB)
    hidden_size = text_config.hidden_size
    expected_scale = hidden_size**0.5

    embed_weight = torch.randn(_RANDOM_VOCAB, hidden_size, dtype=torch.bfloat16)
    tokens = torch.randint(0, _RANDOM_VOCAB, (1, 32), dtype=torch.long)

    # ``Gemma4Model`` stashes ``_embed_weight_cpu = state_dict[key]`` by reference, so
    # comparing it against ``embed_weight`` would compare a tensor with itself and pass
    # unconditionally. Hold a pristine copy taken before the model touches anything.
    pristine = embed_weight.clone()

    model = _build_embedding_model(mesh_device, embed_weight, vocab_size=_RANDOM_VOCAB)

    # Host copy is the raw table — the tied lm_head must never see the scale.
    assert model._embed_weight_cpu is not None, "state_dict embedding was not stashed on host"
    assert torch.equal(
        model._embed_weight_cpu.float(), pristine.float()
    ), "_embed_weight_cpu must stay unscaled (tied lm_head reads it)"
    assert model.embed_scale == pytest.approx(expected_scale)

    tt_out = model.embed_tokens(_tokens_to_device(tokens, mesh_device), layout=ttnn.TILE_LAYOUT)
    tt_torch = _to_torch_2d(tt_out, mesh_device, hidden_size).float()

    unscaled = F.embedding(tokens, embed_weight.float()).reshape(-1, hidden_size)
    ratio = tt_torch.norm() / (unscaled.norm() * expected_scale)
    logger.info(f"embed_scale={expected_scale:.4f} |TT| / (sqrt(h)*|raw|) = {float(ratio):.5f}")
    assert ratio == pytest.approx(1.0, abs=2e-2), (
        f"embed_tokens output is {float(ratio):.4f}x the scaled reference — "
        f"sqrt(hidden)={expected_scale:.2f} is not baked into the device table"
    )


@parametrize_mesh_with_fabric()
def test_raw_embed_undoes_embed_scale(mesh_device, reset_seeds, request):
    """``raw_embed`` must return the *unscaled* table lookup.

    The device table is pre-scaled, so ``raw_embed`` divides it back out. The
    drafter cross-attention path depends on that inverse being exact; today the
    only check on it is a ``logger.info`` inside an env-gated spec-decode probe.
    """
    text_config = _embedding_only_text_config(_RANDOM_VOCAB)
    hidden_size = text_config.hidden_size

    embed_weight = torch.randn(_RANDOM_VOCAB, hidden_size, dtype=torch.bfloat16)
    tokens = torch.randint(0, _RANDOM_VOCAB, (1, 32), dtype=torch.long)
    ref_raw = F.embedding(tokens, embed_weight.float()).reshape(-1, hidden_size)

    model = _build_embedding_model(mesh_device, embed_weight, vocab_size=_RANDOM_VOCAB)
    tt_raw = _to_torch_2d(model.raw_embed(_tokens_to_device(tokens, mesh_device)), mesh_device, hidden_size).float()

    ratio = tt_raw.norm() / ref_raw.norm()
    logger.info(f"raw_embed |TT| / |raw table| = {float(ratio):.5f} (must be ~1, not sqrt(h))")
    assert ratio == pytest.approx(1.0, abs=2e-2), (
        f"raw_embed is {float(ratio):.4f}x the raw table — the 1/embed_scale " f"inverse is missing or doubled"
    )

    passing, pcc_msg = compare_tensors(tt_raw, ref_raw, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"raw_embed PCC too low: {pcc_msg}"


@parametrize_mesh_with_fabric()
@parametrize_batch_seq(configs=[(1, 32), (1, 128)], ids=["decode_batch32", "prefill_128"])
def test_embedding_tile_layout_is_bit_exact(batch_size, seq_len, mesh_device, reset_seeds):
    """Fused in-kernel tilize must equal ROW_MAJOR + a separate ``to_layout``.

    ``layout=ttnn.TILE_LAYOUT`` was adopted as a win on the embed+all-gather path
    specifically because it was bit-identical. That is a bit-exactness claim, so
    assert ``torch.equal`` — a PCC gate would pass on a genuine regression here.
    """
    text_config = _embedding_only_text_config(_RANDOM_VOCAB)
    hidden_size = text_config.hidden_size

    embed_weight = torch.randn(_RANDOM_VOCAB, hidden_size, dtype=torch.bfloat16)
    tokens = torch.randint(0, _RANDOM_VOCAB, (1, seq_len), dtype=torch.long)

    model = _build_embedding_model(mesh_device, embed_weight, vocab_size=_RANDOM_VOCAB)

    fused = model.embed_tokens(_tokens_to_device(tokens, mesh_device), layout=ttnn.TILE_LAYOUT)
    row_major = model.embed_tokens(_tokens_to_device(tokens, mesh_device))
    tiled_after = ttnn.to_layout(row_major, ttnn.TILE_LAYOUT)

    t_fused = _to_torch_2d(fused, mesh_device, hidden_size)
    t_after = _to_torch_2d(tiled_after, mesh_device, hidden_size)

    mismatches = int((t_fused != t_after).sum())
    assert torch.equal(t_fused, t_after), (
        f"fused TILE_LAYOUT embedding diverged from ROW_MAJOR+to_layout in "
        f"{mismatches}/{t_fused.numel()} elements — the in-kernel tilize is no "
        f"longer bit-identical"
    )


# ── Real-weight PCC ────────────────────────────────────────────────────────


@parametrize_mesh_with_fabric()
@parametrize_batch_seq(configs=[(1, 32), (1, 128)], ids=["decode_batch32", "prefill_128"])
def test_embedding_real_weights(batch_size, seq_len, mesh_device, reset_seeds, request):
    """``embed_tokens`` against the checkpoint's trained table at the full vocab.

    The random-weight tests use a 2048-row table, which keeps every token id in
    uint16 range and every value near unit scale. The shipped table is 262144
    rows — ids past 65535 and trained magnitudes the bf16 cast has to carry.
    Reads the one tensor out of the safetensors shards (``load_real_substate``)
    rather than materializing the model, for the reasons documented on
    ``test_lm_head._real_lm_head_weight``.
    """
    embed_weight = load_real_substate("embed_tokens")["weight"]  # skips unless real weights
    vocab_size, hidden_size = embed_weight.shape
    logger.info(f"Real embed_tokens.weight: [{vocab_size}, {hidden_size}] dtype={embed_weight.dtype}")

    # Bias sampling toward the high end of the vocab so ids beyond uint16 range are
    # actually exercised, not just the low ids a uniform draw would favour. Every shipped
    # Gemma4 variant is 262144 rows; the guard keeps a narrower table from tripping
    # torch.randint's empty-range error rather than silently skipping the intent.
    uint16_max = 65536
    if vocab_size > uint16_max:
        half = seq_len // 2
        tokens = torch.cat(
            [
                torch.randint(0, uint16_max, (half,), dtype=torch.long),
                torch.randint(uint16_max, vocab_size, (seq_len - half,), dtype=torch.long),
            ]
        ).unsqueeze(0)
    else:
        tokens = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    ref = (F.embedding(tokens, embed_weight.float()) * (hidden_size**0.5)).reshape(-1, hidden_size)

    model = _build_embedding_model(mesh_device, embed_weight)
    assert model.vocab_size == vocab_size, f"config vocab {model.vocab_size} != checkpoint vocab {vocab_size}"

    tt_out = model.embed_tokens(_tokens_to_device(tokens, mesh_device), layout=ttnn.TILE_LAYOUT)
    tt_torch = _to_torch_2d(tt_out, mesh_device, hidden_size).float()

    passing, pcc_msg = compare_tensors(tt_torch, ref, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"Embedding real-weight PCC too low: {pcc_msg}"

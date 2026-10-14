# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel vocab padding, in both places that implement it.

The rule is one invariant with two implementations -- ``tp_padded_vocab_size`` in the
qwen3 example and ``Llama.padded_vocab_size`` in ``ttml.models.llama`` -- so both are
covered here.

The padded vocab must be divisible by ``tp_size`` (ColumnParallelLinear and
VocabParallelEmbedding both shard dim 2 across TP) *and* each resulting shard must be
a multiple of 32 for the tile layout. That means a multiple of ``32 * tp_size``.

``lcm(32, tp_size)`` looks like it satisfies both but does not -- it only makes the
*global* size divisible by both. At TP=8 it is 32, so Qwen3's 151936 came back
unpadded with an 18992-row shard; ``make_empty_on_device`` then rounded each shard
to 19008 and the HF loader partitioned at 19008, while VocabParallelEmbedding kept
deriving ownership offsets from ``151936 // 8 = 18992``. Every TP rank after the
first silently read embedding rows shifted by ``16 * rank``.

The tests run at TP=2 with a vocab chosen to reproduce that exact arithmetic
(96 % 32 == 0 but 96 / 2 = 48 is not tile-aligned), and cover it at three levels:
the padding formula, the structural invariant the model must uphold, and the
end-to-end row identity through the real HF loader.
"""

import os
import sys

import numpy as np
import pytest
import torch
import ttml
import ttnn

pytestmark = pytest.mark.requires_device

TP_AXIS_SIZE = 2

# 96 % 32 == 0 (so the naive lcm formula pads nothing) but 96 / 2 = 48 is not a
# multiple of 32 -- the shape of Qwen3's 151936 at TP=8.
MISALIGNED_VOCAB = 96
HIDDEN = 64

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_QWEN3_EXAMPLE = os.path.abspath(os.path.join(_REPO_ROOT, "sources", "examples", "qwen3"))
# Force the qwen3 example dir to the front of sys.path so ``import utils`` resolves
# here even if a sibling example dir is already on the path.
if _QWEN3_EXAMPLE in sys.path:
    sys.path.remove(_QWEN3_EXAMPLE)
sys.path.insert(0, _QWEN3_EXAMPLE)

# The qwen3 example ships a top-level ``utils`` package, and so do sibling example
# dirs (e.g. examples/grpo, imported by test_grpo_trainer). In a single pytest session
# another test module may have already imported its own ``utils`` first, caching it in
# sys.modules and shadowing ours. Evict any cached ``utils*`` so the imports below
# resolve against _QWEN3_EXAMPLE, then restore the sibling's modules.
#
# Everything is imported *here* rather than lazily inside the tests: after the restore
# below, a runtime ``from utils...`` would resolve against the sibling package again.
_saved_utils = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "utils" or k.startswith("utils.")}
try:
    from ttml.models.qwen3 import Qwen3Config  # noqa: E402
    from ttml.models.llama import EmbeddingPlacement, Llama, LlamaConfig  # noqa: E402
    from utils.context_managers import empty_init  # noqa: E402
    from model_qwen3_distributed import (  # noqa: E402
        DistributedQwen3ForCausalLM,
        load_weights_from_hf_distributed,
        tp_padded_vocab_size,
    )
finally:
    for _k in [k for k in list(sys.modules) if k == "utils" or k.startswith("utils.")]:
        del sys.modules[_k]
    sys.modules.update(_saved_utils)


def _tied_model(vocab):
    """The real tied-TP model, built the way model_factory builds it (empty_init)."""
    cfg = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=TP_AXIS_SIZE,
        num_key_value_heads=TP_AXIS_SIZE,
        head_dim=32,
        vocab_size=vocab,
        max_position_embeddings=32,
    )
    with empty_init():
        model = DistributedQwen3ForCausalLM(cfg, tie_word_embeddings=True, shard_dim=1)
    return model, cfg


# ---------------------------------------------------------------------------
# 1. The padding formula, in both implementations
# ---------------------------------------------------------------------------


def _assert_shards_tile_aligned(padded, vocab, tp):
    assert padded >= vocab, "padding must not truncate the vocabulary"
    assert padded % tp == 0, f"{padded} not divisible by tp_size {tp}"
    assert (padded // tp) % 32 == 0, f"shard {padded // tp} is not tile-aligned"
    assert padded - vocab < 32 * tp, "padded more than one alignment block"


@pytest.mark.parametrize("vocab", [32, 64, 96, 100, 151936])
def test_qwen3_padded_vocab_gives_tile_aligned_shards(tp_mesh, vocab):
    """Every shard of the padded vocab must be whole AND a multiple of 32.

    Fails under ``lcm(32, tp_size)``: at TP=2 that pads 96 to 96, whose 48-row
    shards are not tile-aligned.
    """
    _assert_shards_tile_aligned(tp_padded_vocab_size(vocab), vocab, tp_mesh.axis_size("tp"))


@pytest.mark.parametrize("vocab", [64, 96, 151936])
def test_llama_padded_vocab_gives_tile_aligned_shards(tp_mesh, vocab):
    """``ttml.models.llama`` carries the same rule and had the same ``lcm`` bug.

    Latent rather than live there -- llama shards a global tensor, so the per-device
    logical shape stays ``V / tp`` and self-consistent -- but the padding still has to
    honour the tile constraint it claims to.
    """
    cfg = LlamaConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=TP_AXIS_SIZE,
        num_key_value_heads=TP_AXIS_SIZE,
        intermediate_size=128,
        vocab_size=vocab,
        max_position_embeddings=32,
        use_tp=True,
        embedding_placement=EmbeddingPlacement.VocabParallel,
    )
    _assert_shards_tile_aligned(Llama(cfg).padded_vocab_size, vocab, tp_mesh.axis_size("tp"))


# ---------------------------------------------------------------------------
# 2. The structural invariant in the built model
# ---------------------------------------------------------------------------


def test_tied_model_ownership_stride_matches_allocated_rows(tp_mesh):
    """The embedding's per-rank window must equal the rows actually allocated.

    This is the invariant the bug broke: 18992 assumed vs 19008 allocated. With a
    misaligned padded vocab the VocabParallelEmbedding guard raises during
    construction, so a regression surfaces here either as a raise or a mismatch.
    """
    model, _ = _tied_model(MISALIGNED_VOCAB)

    emb = model.model.embed_tokens
    params = model.parameters()
    lm_name = next(n for n in params if n.endswith("lm_head/weight"))
    allocated_rows = int(params[lm_name].shape()[2])

    assert emb.num_embeddings_per_partition == allocated_rows
    # ...and the tied weight really is one shared tensor, not a copy.
    assert emb.weight.tensor.shape()[2] == allocated_rows


# ---------------------------------------------------------------------------
# 3. End-to-end row identity through the real HF loader
# ---------------------------------------------------------------------------


def test_every_rank_reads_its_own_vocab_rows(tp_mesh):
    """Token id -> embedding row must be identity on every TP rank.

    Writes a row-index ramp through ``load_weights_from_hf_distributed`` and decodes
    which row each probe id actually gathered.
    """
    model, cfg = _tied_model(MISALIGNED_VOCAB)
    emb = model.model.embed_tokens
    stride = emb.num_embeddings_per_partition
    width = stride * TP_AXIS_SIZE

    # Row i encodes i + 1 in base 128 across columns 0 and 1 -- both < 128, so bf16
    # represents them exactly and the row index survives the round trip.
    #
    # The ramp is vocab_size rows, not the padded width: the loader validates the
    # incoming shape against the config-implied one and pads the rest with zeros. So
    # a shifted read can land in that zero padding, and the +1 is what keeps that
    # distinguishable -- encoding i directly would make padding decode as row 0, i.e.
    # identical to a correct read of row 0. Offsetting by one sends padding to -1,
    # which is not a valid id, so any shift is caught no matter where it lands.
    ramp = np.zeros((MISALIGNED_VOCAB, HIDDEN), dtype=np.float32)
    idx = np.arange(MISALIGNED_VOCAB) + 1
    ramp[:, 0] = idx % 128
    ramp[:, 1] = idx // 128
    load_weights_from_hf_distributed(
        model,
        {"model.embed_tokens.weight": torch.from_numpy(ramp)},
        cfg,
        tie_word_embeddings=True,
        shard_dim=1,
    )

    seq = 32
    probes = [r * stride + off for r in range(TP_AXIS_SIZE) for off in (0, 1, 17)]
    probes = [p for p in probes if p < width][:seq]

    ids = np.zeros((1, 1, 1, seq), dtype=np.uint32)
    ids[0, 0, 0, : len(probes)] = probes
    out = emb(ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32))

    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    got = out.to_numpy(composer=composer).astype(np.float32)[0, 0]

    # -1 undoes the +1 above; a read that fell in the zero padding decodes to -1.
    rows_read = [int(round(got[i, 1])) * 128 + int(round(got[i, 0])) - 1 for i in range(len(probes))]
    shifted = [(p, r) for p, r in zip(probes, rows_read) if p != r]
    assert not shifted, f"(id, row_read) pairs disagree, -1 means the zero padding: {shifted}"

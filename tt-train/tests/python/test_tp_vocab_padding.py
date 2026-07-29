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
from typing import Optional

import numpy as np
import pytest
import torch
import ttml
import ttnn

pytestmark = pytest.mark.requires_device

TP_AXIS_SIZE = 2
MESH_SHAPE = (1, TP_AXIS_SIZE)

# 96 % 32 == 0 (so the naive lcm formula pads nothing) but 96 / 2 = 48 is not a
# multiple of 32 -- the shape of Qwen3's 151936 at TP=8.
MISALIGNED_VOCAB = 96
HIDDEN = 64

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MGD_FOR_ARCH_AND_SHAPE = {
    ("blackhole", MESH_SHAPE): os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_1_2_line_line.textproto"),
    ("wormhole_b0", MESH_SHAPE): os.path.join(_REPO_ROOT, "configs", "mgd", "n300_1_2_line_line.textproto"),
}

sys.path.insert(0, os.path.join(_REPO_ROOT, "sources", "examples", "qwen3"))


# ---------------------------------------------------------------------------
# Mesh fixture (same shape/skip conventions as test_vocab_parallel_embedding.py)
# ---------------------------------------------------------------------------
def _detect_arch() -> Optional[str]:
    try:
        name = ttnn.get_arch_name().lower()
    except Exception:  # noqa: BLE001
        return None
    if "blackhole" in name:
        return "blackhole"
    if "wormhole_b0" in name:
        return "wormhole_b0"
    return None


def _close_device_mesh_quietly() -> None:
    try:
        ttml.close_device_mesh()
    except Exception:  # noqa: BLE001
        pass


def _ensure_mgd_path(shape) -> Optional[str]:
    previous = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    if previous:
        return previous
    arch = _detect_arch()
    if arch is None:
        return previous
    candidate = _MGD_FOR_ARCH_AND_SHAPE.get((arch, shape))
    if candidate and os.path.isfile(candidate):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = candidate
    return previous


def _restore_mgd_path(previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
    else:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous


@pytest.fixture(scope="module")
def tp_mesh():
    """Open a ``[1, TP_AXIS_SIZE]`` mesh with axes ``("dp", "tp")``; skip if unavailable."""
    previous_mgd = _ensure_mgd_path(MESH_SHAPE)
    _close_device_mesh_quietly()
    try:
        ttml.open_device_mesh(ttml.Mesh(MESH_SHAPE, ("dp", "tp")))
        ttml.autograd.AutoContext.get_instance().initialize_parallelism_context(
            ttml.autograd.DistributedConfig(enable_ddp=False, enable_tp=True)
        )
    except Exception as e:  # noqa: BLE001
        _restore_mgd_path(previous_mgd)
        pytest.skip(f"qwen3 tied-TP vocab tests need {TP_AXIS_SIZE} devices on the 'tp' axis: {e}")

    yield ttml.mesh()

    _close_device_mesh_quietly()
    _restore_mgd_path(previous_mgd)


def _tied_model(vocab):
    """The real tied-TP model, built the way model_factory builds it (empty_init)."""
    from ttml.models.qwen3 import Qwen3Config
    from utils.context_managers import empty_init
    from model_qwen3_distributed import DistributedQwen3ForCausalLM

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
    from model_qwen3_distributed import tp_padded_vocab_size

    _assert_shards_tile_aligned(tp_padded_vocab_size(vocab), vocab, tp_mesh.axis_size("tp"))


@pytest.mark.parametrize("vocab", [64, 96, 151936])
def test_llama_padded_vocab_gives_tile_aligned_shards(tp_mesh, vocab):
    """``ttml.models.llama`` carries the same rule and had the same ``lcm`` bug.

    Latent rather than live there -- llama shards a global tensor, so the per-device
    logical shape stays ``V / tp`` and self-consistent -- but the padding still has to
    honour the tile constraint it claims to.
    """
    from ttml.models.llama import EmbeddingPlacement, Llama, LlamaConfig

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
    which row each probe id actually gathered. Under the bug, rank 0 was correct and
    every later rank read rows shifted by the tile pad.
    """
    from model_qwen3_distributed import load_weights_from_hf_distributed

    model, cfg = _tied_model(MISALIGNED_VOCAB)
    emb = model.model.embed_tokens
    stride = emb.num_embeddings_per_partition
    width = stride * TP_AXIS_SIZE

    # Row i encodes i in base 128 across columns 0 and 1 -- both < 128, so bf16
    # represents them exactly and the row index survives the round trip.
    ramp = np.zeros((width, HIDDEN), dtype=np.float32)
    idx = np.arange(width)
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

    rows_read = [int(round(got[i, 1])) * 128 + int(round(got[i, 0])) for i in range(len(probes))]
    shifted = [(p, r) for p, r in zip(probes, rows_read) if p != r]
    assert not shifted, f"(id, row_read) pairs disagree: {shifted}"

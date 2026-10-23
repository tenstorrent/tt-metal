# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Image input to the V4.1 LLM: an image span reaches the model as token ids past the vocabulary (rows of an
extended embedding table, see ``tt/vision.py``), from which the model derives the reference's ``image_mask``.

Host only:

* ``test_span_ids`` -- the span's ids: the three learned delimiters, then the IMAGE rows in reading order.
* ``test_engram_masking`` -- ``EngramHostLookup`` hashes a prompt holding an image span as the checkpoint's
  ``NgramHashState`` with ``token_mask`` does, and the decode step after it too.

On the device (the full mesh opened as the server opens it, then a submesh of it):

* ``test_image_routing`` -- ``DeepSeekV41PrefillMoE.route`` with an image mask against the checkpoint's ``Gate``
  (``bias_vl`` for image tokens): the ranking's top 6 overlap, the per-token bias rows exactly the bf16 table's,
  and an all-text mask ranking bit-identically to no mask.
* ``test_extended_embedding`` -- ``image_mask`` on device, and rows written by ``write_embedding_rows`` into an
  ``extend_embedding`` table read back by ``ttnn.embedding``.

    pytest models/experimental/deepseek_v41_flash/tests/test_image_input.py
"""

import threading
import types

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.reference import load_reference, reference_moe
from models.experimental.deepseek_v41_flash.tt import engram_lookup
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, load_config, moe_weights
from models.experimental.deepseek_v41_flash.tt.prefill.moe import DeepSeekV41PrefillMoE
from models.experimental.deepseek_v41_flash.tt.vision import (
    DELIMITER_ROWS,
    FIRST_IMAGE_ROW,
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    extend_embedding,
    image_mask,
    span_ids,
    write_embedding_rows,
)
from models.experimental.deepseek_v4_flash.demo.chat_cli import open_mesh_device
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

TP_SIZE = 4
# Mean fraction of each token's 6 experts the bf16 ranking shares with the fp32 reference's (tests/prefill/test_moe.py).
MIN_ROUTING_OVERLAP = 0.9


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


needs_checkpoint = pytest.mark.skipif(
    _snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}"
)


def _full_mesh():
    """The whole system as the server opens it (rows x TP4)."""
    chips = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()
    return open_mesh_device(None, mesh_shape=(chips // TP_SIZE, TP_SIZE))


def test_span_ids(expect_error):
    vocab = 1000
    types_ = torch.tensor([IMAGE_START, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE_END])
    ids = span_ids(types_, torch.tensor([7, 3, 9, 4]), vocab)
    row = vocab + FIRST_IMAGE_ROW
    assert ids.tolist() == [
        vocab + DELIMITER_ROWS[IMAGE_START],
        row + 7,
        row + 3,
        vocab + DELIMITER_ROWS[IMAGE_NEW_LINE],
        row + 9,
        row + 4,
        vocab + DELIMITER_ROWS[IMAGE_NEW_LINE],
        vocab + DELIMITER_ROWS[IMAGE_END],
    ]
    with expect_error(ValueError, "slots for a span of"):
        span_ids(types_, torch.tensor([1, 2]), vocab)


@needs_checkpoint
def test_engram_masking():
    from transformers import AutoTokenizer

    snapshot = _snapshot()
    config = load_config(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    module = engram_lookup._engram_module(snapshot)
    args = types.SimpleNamespace(
        engram_layer_ids=config.engram_layer_ids,
        engram_num_embeddings=config.engram_num_embeddings,
        engram_max_ngram_size=config.engram_max_ngram_size,
        engram_vocab_size=config.engram_vocab_size,
        engram_n_heads=config.engram_n_heads,
        engram_head_dim=config.engram_head_dim,
        engram_compressed_vocab_size=config.engram_compressed_vocab_size,
        engram_pad_id=config.engram_pad_token_id,
        max_batch_size=1,
        max_seq_len=64,
    )
    layout = module.EngramLayout.from_args(args)
    reference = module.NgramHashState(args, layout, tokenizer)
    # The lookup without its ~200 GB of tables: only its hashing is under test.
    lookup = engram_lookup.EngramHostLookup.__new__(engram_lookup.EngramHostLookup)
    lookup.hash, lookup.vocab_size, lookup._lock = (
        module.NgramHashState(args, layout, tokenizer),
        config.vocab_size,
        threading.Lock(),
    )

    vocab = config.vocab_size
    text = tokenizer("Describe the picture in a few words, please", add_special_tokens=False)["input_ids"]
    span = [vocab + DELIMITER_ROWS[IMAGE_START], vocab + FIRST_IMAGE_ROW, vocab + FIRST_IMAGE_ROW + 1]
    span += [vocab + DELIMITER_ROWS[IMAGE_NEW_LINE], vocab + DELIMITER_ROWS[IMAGE_END]]
    ids = torch.tensor([text[:4] + span + text[4:]])
    text_mask = ids < vocab
    # The reference's prompt: image_token_id at every span position, the span masked out.
    expected = reference(torch.where(text_mask, ids, 129264), 0, token_mask=text_mask)
    with lookup._lock:
        got = lookup._hash(ids, 0)
    assert torch.equal(expected, got)
    step = torch.tensor([[text[5]]])
    with lookup._lock:
        assert torch.equal(reference(step, ids.shape[1]), lookup._hash(step, ids.shape[1]))


@needs_checkpoint
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("layer", (2, 39))
@torch.no_grad()
def test_image_routing(layer):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    reference = reference_moe(load_reference(snapshot), snapshot, loader, layer)
    assert reference.gate.bias_vl is not None, "the reference has no bias_vl (vision_n_layers == 0?)"
    seq_len = 256

    with _full_mesh() as mesh_device:
        device = mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        # The router only: the block checks just these of its routed experts at construction.
        experts = types.SimpleNamespace(
            tp_size=1, top_k=config.num_experts_per_tok, gate_up_weights=[None] * config.n_routed_experts
        )
        moe = DeepSeekV41PrefillMoE(
            config,
            moe_weights(loader, layer),
            device,
            experts=experts,
            cache=WeightCache(CACHE_DIR).sub(f"layers.{layer}.ffn"),
        )
        torch.manual_seed(layer)
        hidden = torch.randn(seq_len, config.hidden_size).bfloat16().float()
        in_image = torch.zeros(seq_len, dtype=torch.bool)
        in_image[40:200] = True  # one span in the middle of the chunk

        def upload_mask(mask):
            return ttnn.from_torch(
                mask.to(torch.int32).reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )

        x = ttnn.from_torch(
            hidden.reshape(1, 1, seq_len, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        def ranking(mask):
            routing = moe.route(x, image_mask=None if mask is None else upload_mask(mask))
            return ttnn.to_torch(routing.ranking).reshape(seq_len, -1).float()

        with_image = ranking(in_image)
        picked = with_image.topk(config.num_experts_per_tok, dim=-1).indices
        _, expected = reference.gate(hidden, in_image)
        overlap = torch.tensor(
            [len(set(a.tolist()) & set(b.tolist())) / config.num_experts_per_tok for a, b in zip(picked, expected)]
        )
        _, text_routes = reference.gate(hidden)
        moved = torch.tensor([set(a.tolist()) != set(b.tolist()) for a, b in zip(expected, text_routes)])
        logger.info(
            f"layer {layer}: routing overlap {overlap[in_image].mean():.4f} (image tokens), "
            f"{overlap[~in_image].mean():.4f} (text); bias_vl moves {moved[in_image].float().mean():.2f} of the image "
            "tokens' expert sets"
        )
        assert overlap[in_image].mean() >= MIN_ROUTING_OVERLAP
        assert overlap[~in_image].mean() >= MIN_ROUTING_OVERLAP

        # An all-text mask gathers row 0 of the table, which holds the bf16 values of the text bias.
        assert torch.equal(ranking(torch.zeros(seq_len, dtype=torch.bool)), ranking(None))

        # Text tokens gather exactly the router's bias row; image tokens the shifted bias_vl (to bf16 spacing).
        rows = ttnn.to_torch(moe.gate.ranking_bias(upload_mask(in_image))).reshape(seq_len, -1).float()
        text_bias = ttnn.to_torch(moe.gate.e_score_correction_bias).reshape(1, -1).float()
        assert torch.equal(rows[~in_image], text_bias.expand(int((~in_image).sum()), -1))
        bias_vl = reference.gate.bias_vl.float()
        assert torch.allclose(rows[in_image], (bias_vl - bias_vl.max()).expand(int(in_image.sum()), -1), atol=0.07)


@pytest.mark.timeout(1800)
@torch.no_grad()
def test_extended_embedding(expect_error):
    vocab, extra, hidden = 1000, 64, 512
    with _full_mesh() as mesh_device:
        device = mesh_device.create_submesh(ttnn.MeshShape(1, TP_SIZE), ttnn.MeshCoordinate(0, 0))
        replicate = ttnn.ReplicateTensorToMesh(device)

        def upload(t, dtype, shape=None):
            return ttnn.from_torch(
                t if shape is None else t.reshape(shape),
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )

        torch.manual_seed(0)
        table_host = torch.randn(vocab, hidden).bfloat16()
        extended = extend_embedding(upload(table_host, ttnn.bfloat16), extra, vocab)

        # 40 rows: a full block and a short one (padded by repeating its last row).
        new_rows = torch.randn(40, hidden).bfloat16()
        at = torch.randperm(extra)[:40] + vocab
        write_embedding_rows(extended, new_rows, at, vocab)
        with expect_error(ValueError, "embedding row ids must lie in the image rows"):
            write_embedding_rows(extended, new_rows[:1], [vocab - 1], vocab)
        ids = torch.cat([torch.randint(0, vocab, (24,)), at])[torch.randperm(64)]
        ids_dev = upload(ids.to(torch.int32).reshape(1, -1), ttnn.uint32)
        got = ttnn.to_torch(ttnn.get_device_tensors(ttnn.embedding(ids_dev, extended, layout=ttnn.ROW_MAJOR_LAYOUT))[0])
        want = torch.cat([table_host, torch.zeros(extra, hidden, dtype=torch.bfloat16)])
        want[at] = new_rows
        assert torch.equal(got.reshape(64, hidden), want[ids])

        mask = ttnn.to_torch(ttnn.get_device_tensors(image_mask(ids_dev, vocab))[0]).reshape(-1)
        assert torch.equal(mask.long(), (ids >= vocab).long())

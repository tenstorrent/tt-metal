# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests of the vendored DeepSeek-V4.1 reference on a small random config (host only, no device).

The small schedule has one backbone layer of each of the six V4.1 block types, Engram on an SWA and a
KV-source layer, and one DSpark layer; weights are stored in the checkpoint dtypes, so every CPU kernel
port runs on the real model path.
"""

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.image_processor import TEXT, ImageInput, image_token_types
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.testing import (
    SmallScheduleConfig,
    build_small_model,
    prefill,
    small_model_args,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import V41BlockType

SEQ = 41  # > window (16), odd (ratio-2 remainder carried), > candidate blocks x block size


def _tokens(seq: int, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, 500, (1, seq), generator=torch.Generator().manual_seed(seed))


def test_small_schedule_covers_six_block_types_and_matches_model_roles():
    cfg = SmallScheduleConfig
    types = [cfg.block_type(layer) for layer in range(cfg.NUM_LAYERS)]
    assert set(types) == set(V41BlockType)
    model = build_small_model()
    for layer, block_type in enumerate(types):
        attn = model.layers[layer].attn
        assert attn.compress_ratio == cfg.compress_ratio(layer)
        assert (attn.compressor is not None) == (
            block_type in {V41BlockType.KV_INDEX_SOURCE, V41BlockType.CANDIDATE_SOURCE}
        )
        owns_indexer = block_type not in {
            V41BlockType.SWA_ONLY,
            V41BlockType.CONSUMER_RATIO2,
            V41BlockType.CONSUMER_RATIO1,
        }
        assert (attn.indexer is not None) == owns_indexer
        if attn.indexer is not None:
            assert attn.indexer.is_candidate_source == (block_type == V41BlockType.CANDIDATE_SOURCE)
            assert attn.indexer.uses_candidates == (block_type == V41BlockType.CANDIDATE_INDEX_SOURCE)
    assert [layer.engram is not None for layer in model.layers] == [i in cfg.ENGRAM_LAYER_IDS for i in range(6)]
    assert len(model.mtp) == cfg.NUM_DSPARK_LAYERS


def test_small_model_storage_dtypes():
    model = build_small_model()
    attn = model.layers[1].attn
    assert attn.wq_a.weight.dtype == torch.float8_e4m3fn and attn.wq_a.scale.dtype == torch.float8_e8m0fnu
    assert attn.wq_a.scale.shape == (64 // 32, 256 // 32)
    assert attn.wo_a.weight.dtype == torch.bfloat16
    assert attn.compressor.wkv.weight.dtype == torch.float32  # ratio 2 pools in fp32
    assert model.layers[3].attn.compressor.wkv.weight.dtype == torch.bfloat16  # ratio 1
    expert = model.layers[0].ffn.experts[0]
    assert expert.w1.weight.dtype == torch.float4_e2m1fn_x2 and expert.w1.weight.shape == (128, 128)
    assert expert.w1.scale.dtype == torch.float8_e8m0fnu and expert.w1.scale.shape == (128, 256 // 32)
    assert model.layers[0].ffn.shared_experts.w1.weight.dtype == torch.float8_e4m3fn
    embed = model.layers[0].engram.embed
    assert embed.weight.dtype == torch.float8_e4m3fn and embed.scale.dtype == torch.float8_e8m0fnu
    assert model.head.weight.dtype == torch.float32


def test_small_model_prefill_finite_and_deterministic():
    model = build_small_model(seed=0)
    ids = _tokens(SEQ)
    logits, main_hidden = prefill(model, ids)
    assert logits.shape == (1, 512) and logits.dtype == torch.float32
    assert torch.isfinite(logits).all() and logits.std() > 0
    assert main_hidden.shape == (1, SEQ, 256 * len(SmallScheduleConfig.DSPARK_TARGET_LAYER_IDS))
    # the candidate source published a mask and the last index source a top-k over compressed rows
    assert v41.shared_attn.candidates.shape == (1, SEQ, SEQ)
    assert v41.shared_attn.topk_idxs.shape == (1, SEQ, small_model_args().index_topk)
    # DSpark prefill seeded its window cache
    assert model.mtp[0].attn.window_kv_cache.abs().sum() > 0

    # bit-identical on repeat (stateful caches are fully rewritten by a start_pos-0 prefill) and on rebuild
    again, _ = prefill(model, ids)
    assert torch.equal(logits, again)
    rebuilt, _ = prefill(build_small_model(seed=0), ids)
    assert torch.equal(logits, rebuilt)
    other, _ = prefill(build_small_model(seed=1), ids)
    assert not torch.equal(logits, other)


def test_small_model_prefill_short_prompts():
    """Prompts shorter than the ratio / window: seqlen 1 gives compress_len 0 on ratio-2 layers
    (indexer skipped, empty top-k published); seqlen 16 fills the window exactly."""
    model = build_small_model()
    for seq in (1, 2, 3, 16, 17):
        logits, _ = prefill(model, _tokens(seq, seed=seq))
        assert torch.isfinite(logits).all(), seq


def test_small_model_prefill_with_image_span():
    args = small_model_args(vision_n_layers=1, vision_dim=64, vision_n_heads=2, vision_inter_dim=64, image_token_id=510)
    model = build_small_model(args)
    types = image_token_types(1, 1)  # 3x3 ViT patches -> one aligner row: START, IMAGE, NEW_LINE, END
    start, seq = 5, 20
    ids = _tokens(seq)
    ids[0, start : start + types.numel()] = args.image_token_id
    token_types = torch.full((1, seq), TEXT)
    token_types[0, start : start + types.numel()] = types
    patches = torch.randn(9, 3, 14, 14, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    image = ImageInput(start, patches, 3, 3, types)
    with v41.set_dtype(torch.bfloat16):
        _, logits, _ = model(ids, 0, images=[[image]], token_types=token_types)
        _, again, _ = model(ids, 0, images=[[image]], token_types=token_types)
        _, text_only, _ = model(ids, 0)
    assert torch.isfinite(logits).all() and torch.equal(logits, again)
    assert not torch.equal(logits, text_only)  # the image span changes the result

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Small, randomly initialized DeepSeek-V4.1 reference models for CPU tests.

``small_model_args`` builds a ``ModelArgs`` whose 6 backbone layers cover the six V4.1 block types
(schedule in ``SmallScheduleConfig``, typed by the canonical ``DeepSeekV41FlashConfig.block_type``),
plus Engram on an SWA and a KV-source layer and one DSpark layer. ``build_small_model`` constructs the
vendored ``Transformer`` with a stub tokenizer (the Engram hash needs a compressed token map) and fills
every parameter deterministically from a seed, in its checkpoint storage dtype: FP8 e4m3 weights with
E8M0 32x32 block scales, FP4 (float4_e2m1fn_x2) routed experts with E8M0 per-32 scales, FP8 Engram
tables with E8M0 per-32 scales, bf16/fp32 elsewhere.
"""

import math
from dataclasses import replace

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.engram import EngramLayout
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.kernel_cpu import FP8_MAX, fast_round_scale
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig

EMBED_STD = 0.12  # RMS of the released V4.1 embedding (measured on the checkpoint, layer-0 input)


class SmallScheduleConfig(DeepSeekV41FlashConfig):
    """The V4.1 layer-role rules on a 6-layer schedule: one layer of each block type, in the order
    SWA_ONLY, KV_INDEX_SOURCE (ratio 2), CONSUMER_RATIO2, CANDIDATE_SOURCE (ratio 1), CONSUMER_RATIO1,
    CANDIDATE_INDEX_SOURCE; then one ratio-0 DSpark layer."""

    NUM_LAYERS = 6
    NUM_DSPARK_LAYERS = 1
    COMPRESS_RATIOS = (0, 2, 2, 1, 1, 1, 0)
    KV_SOURCE_LAYERS = (1, 3)
    INDEX_SOURCE_LAYERS = (1, 3, 5)
    CANDIDATE_SOURCE_LAYER = 3
    ENGRAM_LAYER_IDS = (0, 1)
    DSPARK_TARGET_LAYER_IDS = (4, 5)


class StubTokenizer:
    """Just enough of a HF tokenizer for ``engram.build_compressed_token_map``: token 2k decodes to
    "w{k}" and 2k+1 to " W{k}", which normalize alike, so the compressed vocab is vocab_size // 2."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.backend_tokenizer = self

    def __len__(self) -> int:
        return self.vocab_size

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        (token_id,) = ids
        return f"w{token_id // 2}" if token_id % 2 == 0 else f" W{token_id // 2}"

    def id_to_token(self, token_id: int) -> str:
        return self.decode([token_id])


def small_model_args(**overrides) -> v41.ModelArgs:
    """ModelArgs for a small model; attention/indexer/engram sizes keep every quantized dim a multiple of 32."""
    cfg = SmallScheduleConfig
    vocab_size = 512
    args = v41.ModelArgs(
        max_batch_size=1,
        max_seq_len=128,
        temperature=0.0,
        vocab_size=vocab_size,
        dim=256,
        moe_inter_dim=128,
        n_layers=cfg.NUM_LAYERS,
        n_mtp_layers=cfg.NUM_DSPARK_LAYERS,
        n_heads=4,
        n_routed_experts=4,
        n_activated_experts=2,
        score_func=cfg.SCORE_FUNC,
        route_scale=cfg.ROUTE_SCALE,
        swiglu_limit=cfg.SWIGLU_LIMIT,
        q_lora_rank=64,
        head_dim=64,
        rope_head_dim=16,
        norm_eps=cfg.RMS_NORM_EPS,
        o_groups=2,
        o_lora_rank=64,
        window_size=16,
        compress_ratios=cfg.COMPRESS_RATIOS,
        kv_source_layers=cfg.KV_SOURCE_LAYERS,
        index_source_layers=cfg.INDEX_SOURCE_LAYERS,
        compress_rope_theta=cfg.COMPRESS_ROPE_THETA,
        original_seq_len=32,  # < max_seq_len so the compressed layers' YaRN ramp is active
        rope_theta=cfg.ROPE_THETA,
        rope_factor=4,
        beta_fast=cfg.ROPE_SCALING_BETA_FAST,
        beta_slow=cfg.ROPE_SCALING_BETA_SLOW,
        index_n_heads=4,
        index_head_dim=64,
        index_topk=8,
        candidate_source_layer=cfg.CANDIDATE_SOURCE_LAYER,
        candidate_topk_blocks=3,  # 3 blocks x 4 >= index_topk, so reachable candidates always fill top-k
        candidate_block_size=4,
        hc_mult=cfg.HC_MULT,
        hc_sinkhorn_iters=cfg.HC_SINKHORN_ITERS,
        hc_eps=cfg.HC_EPS,
        engram_layer_ids=cfg.ENGRAM_LAYER_IDS,
        engram_max_ngram_size=cfg.ENGRAM_MAX_NGRAM_SIZE,
        engram_vocab_size=100,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_pad_id=cfg.ENGRAM_PAD_ID,
        engram_compressed_vocab_size=vocab_size // 2,
        dspark_block_size=2,
        dspark_noise_token_id=vocab_size - 1,
        dspark_target_layer_ids=cfg.DSPARK_TARGET_LAYER_IDS,
        dspark_markov_rank=32,
    )
    args = replace(args, **overrides)
    if args.engram_layer_ids and not args.engram_num_embeddings:
        # table rows = the summed prime bucket ranges, as in the released config
        layout = EngramLayout.from_args(replace(args, engram_num_embeddings=(0,) * len(args.engram_layer_ids)))
        rows = tuple(sum(p for per_ngram in layer for p in per_ngram) for layer in layout.primes)
        args = replace(args, engram_num_embeddings=rows)
    return args


def quantize_fp8_blocks(w: torch.Tensor, block: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 [R, C] -> (float8_e4m3fn [R, C], E8M0 [ceil(R/block), C/block]) with one power-of-2 scale per
    block x block tile (the checkpoint's dense FP8 format). Rows are padded to a block multiple."""
    rows, cols = w.shape
    pad = -rows % block
    wp = torch.cat([w, w.new_zeros(pad, cols)]) if pad else w
    tiles = wp.unflatten(0, (-1, block)).unflatten(-1, (-1, block))  # [Rb, block, Cb, block]
    amax = tiles.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    s = fast_round_scale(amax, torch.tensor(1.0 / FP8_MAX, dtype=torch.float32))
    q = (tiles / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.flatten(2, 3).flatten(0, 1)[:rows].contiguous(), s.to(torch.float8_e8m0fnu)


def quantize_fp8_rows(w: torch.Tensor, block: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 [R, C] -> (float8_e4m3fn [R, C], E8M0 [R, C/block]): per-row groups (Engram table format)."""
    groups = w.unflatten(-1, (-1, block))
    s = fast_round_scale(groups.abs().amax(dim=-1).clamp_min(1e-4), torch.tensor(1.0 / FP8_MAX))
    q = (groups / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.flatten(-2), s.to(torch.float8_e8m0fnu)


@torch.no_grad()
def init_weights(model: torch.nn.Module, seed: int = 0) -> None:
    """Fill every parameter from a seeded generator, in named_parameters order, in its storage dtype."""
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    modules = dict(model.named_modules())
    for name, p in model.named_parameters():
        owner_name, _, leaf = name.rpartition(".")
        owner = modules[owner_name]
        if p.dtype == torch.float8_e8m0fnu:
            continue  # written together with its weight below
        if p.dtype == torch.float8_e4m3fn:
            w = randn(*p.shape)
            if isinstance(owner, v41.ParallelEngramEmbedding):
                q, s = quantize_fp8_rows(w, owner.block_size)
            else:
                q, s = quantize_fp8_blocks(w * p.size(1) ** -0.5)
            p.copy_(q)
            owner.scale.copy_(s)
        elif p.dtype == torch.float4_e2m1fn_x2:
            # Random E2M1 codes (both nibbles) with one power-of-2 scale per 32 inputs, drawn directly in
            # the checkpoint format: quantizing a gaussian costs minutes per layer at real dims. Uniform
            # codes have RMS ~3.2, so the scale targets an RMS of fan_in^-0.5 like the other weights.
            fan_in = 2 * p.size(1)
            codes = torch.randint(0, 256, p.shape, generator=gen, dtype=torch.int32).to(torch.uint8)
            p.view(torch.uint8).copy_(codes)
            exp = round(math.log2(fan_in**-0.5 / 3.2))
            jitter = torch.randint(-1, 2, owner.scale.shape, generator=gen, dtype=torch.int32)
            owner.scale.copy_(torch.pow(2.0, (exp + jitter).float()).to(torch.float8_e8m0fnu))
        elif leaf == "weight" and "norm" in owner_name.rsplit(".", 1)[-1]:
            p.copy_(1 + 0.1 * randn(*p.shape))
        elif isinstance(owner, v41.ParallelEmbedding):
            # the released embedding has RMS ~0.12 (real layer-0 input); fan-in scaling would give ~0.014 and
            # make the first block of a synthetic stack far more sensitive than the real model
            p.copy_(randn(*p.shape) * EMBED_STD)
        elif p.dim() >= 2:
            p.copy_(randn(*p.shape) * p.size(-1) ** -0.5)
        else:
            p.copy_(0.5 * randn(*p.shape))


def build_small_model(args: v41.ModelArgs | None = None, seed: int = 0) -> v41.Transformer:
    """Construct the vendored Transformer (bf16 default dtype, as generate.py does) and seed its weights."""
    args = args or small_model_args()
    with v41.set_dtype(torch.bfloat16):
        model = v41.Transformer(args, StubTokenizer(args.vocab_size))
    init_weights(model, seed)
    return model.eval()


def prefill(model: v41.Transformer, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One single-shot prefill (start_pos 0) under bf16 default dtype, as generate.py runs it: the
    backbone forward, then the DSpark prefill that seeds the DSpark window caches. Returns full-vocab
    logits of the last position [b, vocab] (fp32) and the DSpark main hidden (None without DSpark)."""
    with v41.set_dtype(torch.bfloat16):
        output_ids, logits, main_hidden = model(input_ids, 0)
        if len(model.mtp):
            assert model.forward_spec(output_ids, main_hidden, 0) is None
    return logits, main_hidden

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU oracle for the DeepSeek-V4.1 §4 prototype: real V4.1 dims, synthetic weights.

The prototype covers every attention-sharing role with a 5-layer schedule standing in for V4.1 layers
2 -> 3 (ratio-2 KV+index source, consumer) and 20 -> 21 -> 24 (ratio-1 candidate source, consumer,
candidate-constrained index source). Every dimension is the released model's; only the schedule and
``CANDIDATE_TOPK_BLOCKS`` (96 instead of 2048, so candidate selection is not trivially "all blocks" at a
CPU-feasible 2048-token prompt) differ.

``oracle()`` runs the vendored reference single-shot and records, per block, its inputs (streams,
pre_mix), outputs, the attention input/output and the MoE output. Results are cached on disk keyed by
the configuration, since the full-dims CPU run takes minutes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.kernel_cpu import unpack_fp4
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.testing import StubTokenizer, init_weights
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig

SEQ = 2048
SEED = 0
CACHE_DIR = Path(os.environ.get("TT_V41_ORACLE_CACHE", Path.home() / ".cache" / "tt-v41-oracle"))


class PrototypeScheduleConfig(DeepSeekV41FlashConfig):
    """Prototype layers 0..4 stand for V4.1 layers 2, 3, 20, 21, 24."""

    NUM_LAYERS = 5
    NUM_DSPARK_LAYERS = 0
    COMPRESS_RATIOS = (2, 2, 1, 1, 1)
    KV_SOURCE_LAYERS = (0, 2)
    INDEX_SOURCE_LAYERS = (0, 2, 4)
    CANDIDATE_SOURCE_LAYER = 2
    CANDIDATE_TOPK_BLOCKS = 96
    ENGRAM_LAYER_IDS = ()
    ENGRAM_NUM_EMBEDDINGS = ()
    DSPARK_TARGET_LAYER_IDS = ()


V41_LAYER = {0: 2, 1: 3, 2: 20, 3: 21, 4: 24}


def model_args(cfg=PrototypeScheduleConfig, seq: int = SEQ) -> v41.ModelArgs:
    return v41.ModelArgs(
        max_batch_size=1,
        max_seq_len=seq,
        temperature=0.0,
        vocab_size=cfg.VOCAB_SIZE,
        dim=cfg.EMB_SIZE,
        moe_inter_dim=cfg.MOE_INTERMEDIATE_SIZE,
        n_layers=cfg.NUM_LAYERS,
        n_mtp_layers=cfg.NUM_DSPARK_LAYERS,
        n_heads=cfg.NUM_ATTENTION_HEADS,
        n_routed_experts=cfg.NUM_ROUTED_EXPERTS,
        n_activated_experts=cfg.NUM_EXPERTS_PER_TOKEN,
        score_func=cfg.SCORE_FUNC,
        route_scale=cfg.ROUTE_SCALE,
        swiglu_limit=cfg.SWIGLU_LIMIT,
        q_lora_rank=cfg.Q_LORA_RANK,
        head_dim=cfg.HEAD_DIM,
        rope_head_dim=cfg.QK_ROPE_HEAD_DIM,
        norm_eps=cfg.RMS_NORM_EPS,
        o_groups=cfg.O_GROUPS,
        o_lora_rank=cfg.O_LORA_RANK,
        window_size=cfg.SLIDING_WINDOW,
        compress_ratios=cfg.COMPRESS_RATIOS,
        kv_source_layers=cfg.KV_SOURCE_LAYERS,
        index_source_layers=cfg.INDEX_SOURCE_LAYERS,
        compress_rope_theta=cfg.COMPRESS_ROPE_THETA,
        original_seq_len=cfg.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS,
        rope_theta=cfg.ROPE_THETA,
        rope_factor=cfg.ROPE_SCALING_FACTOR,
        beta_fast=cfg.ROPE_SCALING_BETA_FAST,
        beta_slow=cfg.ROPE_SCALING_BETA_SLOW,
        index_n_heads=cfg.INDEX_N_HEADS,
        index_head_dim=cfg.INDEX_HEAD_DIM,
        index_topk=cfg.INDEX_TOPK,
        candidate_source_layer=cfg.CANDIDATE_SOURCE_LAYER,
        candidate_topk_blocks=cfg.CANDIDATE_TOPK_BLOCKS,
        candidate_block_size=cfg.CANDIDATE_BLOCK_SIZE,
        hc_mult=cfg.HC_MULT,
        hc_sinkhorn_iters=cfg.HC_SINKHORN_ITERS,
        hc_eps=cfg.HC_EPS,
    )


def build_reference(args: v41.ModelArgs, seed: int = SEED) -> v41.Transformer:
    """The seeded reference model; its weights are cached on disk (initializing real dims takes minutes)."""
    with v41.set_dtype(torch.bfloat16):
        model = v41.Transformer(args, StubTokenizer(args.vocab_size))
    path = CACHE_DIR / f"prototype-{_cache_key(args, seed)}-weights.pt"
    if path.is_file():
        model.load_state_dict(torch.load(path, mmap=True), assign=True)
        # assign=True drops the `.scale` attribute that Linear attaches to its weight
        for mod in model.modules():
            if isinstance(mod, v41.Linear) and mod.scale is not None:
                mod.weight.scale = mod.scale
    else:
        init_weights(model, seed)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    return model.eval()


def tokens(args: v41.ModelArgs, seed: int = SEED) -> torch.Tensor:
    return torch.randint(0, args.vocab_size, (1, args.max_seq_len), generator=torch.Generator().manual_seed(seed + 1))


def _cache_key(args: v41.ModelArgs, seed: int) -> str:
    blob = json.dumps({"args": asdict(args), "seed": seed, "v": 1}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@torch.no_grad()
def run_oracle(model: v41.Transformer, input_ids: torch.Tensor) -> dict:
    """Single-shot prefill with per-block captures (bf16/fp32 host tensors)."""
    rec: dict = {}
    hooks = []
    for i, layer in enumerate(model.layers):

        def block_pre(mod, args, i=i):
            rec[f"block{i}.x_in"] = args[0][0].clone()
            rec[f"block{i}.pre_in"] = args[2][0].clone()

        def block_post(mod, args, out, i=i):
            rec[f"block{i}.x_out"] = out[0][0].clone()
            rec[f"block{i}.pre_out"] = out[1][0].clone()

        def attn_pre(mod, args, i=i):
            rec[f"block{i}.attn_in"] = args[0][0].clone()

        def attn_post(mod, args, out, i=i):
            rec[f"block{i}.attn_out"] = out[0].clone()

        def ffn_pre(mod, args, i=i):
            rec[f"block{i}.ffn_in"] = args[0][0].clone()

        def ffn_post(mod, args, out, i=i):
            rec[f"block{i}.ffn_out"] = out[0].clone()

        hooks += [
            layer.register_forward_pre_hook(block_pre),
            layer.register_forward_hook(block_post),
            layer.attn.register_forward_pre_hook(attn_pre),
            layer.attn.register_forward_hook(attn_post),
            layer.ffn.register_forward_pre_hook(ffn_pre),
            layer.ffn.register_forward_hook(ffn_post),
        ]
    try:
        with v41.set_dtype(torch.bfloat16):
            _, logits, _ = model(input_ids, 0)
    finally:
        for h in hooks:
            h.remove()
    rec["logits"] = logits[0].float()
    return rec


def oracle(model: v41.Transformer, args: v41.ModelArgs, seed: int = SEED) -> dict:
    path = CACHE_DIR / f"prototype-{_cache_key(args, seed)}.pt"
    if path.is_file():
        return torch.load(path)
    rec = run_oracle(model, tokens(args, seed))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(rec, path)
    return rec


def _dequant(linear) -> torch.Tensor:
    """A reference Linear's weight as bf16 [out, in] (FP8 32x32 / FP4 per-32 dequantization is exact)."""
    w = linear.weight
    if w.dtype == torch.float8_e4m3fn:
        s = linear.scale.float()
        full = s.repeat_interleave(32, 0)[: w.shape[0]].repeat_interleave(32, 1)[:, : w.shape[1]]
        return (w.float() * full).to(torch.bfloat16)
    if w.dtype == torch.float4_e2m1fn_x2:
        return (unpack_fp4(w) * linear.scale.float().repeat_interleave(32, 1)).to(torch.bfloat16)
    return w.detach().to(torch.bfloat16)


@torch.no_grad()
def device_weights(model: v41.Transformer, layer: int) -> dict:
    """The TtV41Block ``weights`` dict for reference layer ``layer``."""
    blk = model.layers[layer]
    attn, ffn = blk.attn, blk.ffn
    experts = [
        {"gate_proj": _dequant(e.w1), "up_proj": _dequant(e.w3), "down_proj": _dequant(e.w2)} for e in ffn.experts
    ]
    se = ffn.shared_experts
    return {
        "attn": {
            "wq_a": _dequant(attn.wq_a),
            "q_norm": attn.q_norm.weight.detach(),
            "wq_b": _dequant(attn.wq_b),
            "wkv": _dequant(attn.wkv),
            "kv_norm": attn.kv_norm.weight.detach(),
            "wo_a": _dequant(attn.wo_a),
            "wo_b": _dequant(attn.wo_b),
            "attn_sink": attn.attn_sink.detach(),
        },
        "attn_norm": blk.attn_norm.weight.detach(),
        "ffn_norm": blk.ffn_norm.weight.detach(),
        "hc_attn": (blk.hc_attn_fn.detach(), blk.hc_attn_base.detach(), blk.hc_attn_scale.detach()),
        "hc_ffn": (blk.hc_ffn_fn.detach(), blk.hc_ffn_base.detach(), blk.hc_ffn_scale.detach()),
        "gate_weights": {
            "weight": ffn.gate.weight.detach().to(torch.bfloat16),
            "e_score_correction_bias": ffn.gate.bias.detach().float(),
        },
        "routed_expert_weights": experts,
        "shared_expert_weights": {
            "gate_proj": _dequant(se.w1),
            "up_proj": _dequant(se.w3),
            "down_proj": _dequant(se.w2),
        },
    }

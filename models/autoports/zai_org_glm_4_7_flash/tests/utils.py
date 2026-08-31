# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared harness for GLM-4.7-Flash functional-decoder tests.

Provides: real-weight per-layer state dict loading from the HF snapshot,
deterministic synthetic state dicts from recorded tensor stats, the HF fp32
reference decoder layer, paged-cache torch helpers, and PCC utilities.
All torch usage lives here (explicit test boundary); the TTNN runtime path is
in tt/functional_decoder.py.
"""

import hashlib
import json
import os
from pathlib import Path

import torch

SNAPSHOT = Path(
    os.environ.get(
        "GLM47_FLASH_SNAPSHOT",
        "/home/stisi/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash"
        "/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67",
    )
)
STATS_PATH = Path(__file__).resolve().parent / "weight_stats.json"

LAYER_KINDS = {"dense": 0, "moe": 1}


def hf_config():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(SNAPSHOT)
    cfg._attn_implementation = "eager"
    return cfg


def layer_weight_keys(cfg, layer_idx):
    """Canonical per-layer HF checkpoint keys (relative to model.layers.<i>.)."""
    keys = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_a_proj.weight",
        "self_attn.q_a_layernorm.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight",
    ]
    if cfg.mlp_layer_types[layer_idx] == "sparse":
        keys += ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]
        for e in range(cfg.n_routed_experts):
            keys += [f"mlp.experts.{e}.{p}_proj.weight" for p in ("gate", "up", "down")]
        keys += [f"mlp.shared_experts.{p}_proj.weight" for p in ("gate", "up", "down")]
    else:
        keys += [f"mlp.{p}_proj.weight" for p in ("gate", "up", "down")]
    return keys


def load_real_layer_state_dict(cfg, layer_idx):
    """Load one layer's tensors from the safetensors shards (fp32)."""
    from safetensors import safe_open

    with open(SNAPSHOT / "model.safetensors.index.json") as f:
        index = json.load(f)["weight_map"]
    prefix = f"model.layers.{layer_idx}."
    by_shard = {}
    for key in layer_weight_keys(cfg, layer_idx):
        shard = index[prefix + key]
        by_shard.setdefault(shard, []).append(key)
    sd = {}
    for shard, keys in by_shard.items():
        with safe_open(str(SNAPSHOT / shard), framework="pt") as f:
            for key in keys:
                sd[key] = f.get_tensor(prefix + key).to(torch.float32)
    return sd


def synth_layer_state_dict(cfg, layer_idx, stats=None):
    """Deterministic synthetic weights with real shapes/stats.

    Per-tensor N(mean, std) from recorded real-weight stats, seeded from the
    tensor name so any subset regenerates identically.
    """
    stats = stats or json.loads(STATS_PATH.read_text())
    kind = "moe" if cfg.mlp_layer_types[layer_idx] == "sparse" else "dense"
    tensor_stats = stats["layers"][kind]["tensors"]
    sd = {}
    for key in layer_weight_keys(cfg, layer_idx):
        st = tensor_stats[key]
        gen = torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        t = torch.randn(st["shape"], generator=gen) * st["std"] + st["mean"]
        if "layernorm" in key:
            # Norm weights are tightly clustered around their mean; keep them positive.
            t = t.abs()
        sd[key] = t.to(torch.float32)
    return sd


def synth_activations(cfg, layer_idx, seq_len, batch=1, seed=0, stats=None):
    """Synthetic decoder-layer inputs at the recorded real activation scale."""
    stats = stats or json.loads(STATS_PATH.read_text())
    kind = "moe" if cfg.mlp_layer_types[layer_idx] == "sparse" else "dense"
    st = stats["activations"][kind]
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq_len, cfg.hidden_size, generator=gen)
    return (x * st["std"] + st["mean"]).to(torch.float32)


def build_hf_layer(cfg, layer_idx, sd):
    """HF fp32 reference layer from a canonical per-layer state dict."""
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer

    layer = Glm4MoeLiteDecoderLayer(cfg, layer_idx).to(torch.float32).eval()
    converted = dict(sd)
    if cfg.mlp_layer_types[layer_idx] == "sparse":
        E = cfg.n_routed_experts
        gate_up = torch.stack(
            [
                torch.cat(
                    [
                        converted.pop(f"mlp.experts.{e}.gate_proj.weight"),
                        converted.pop(f"mlp.experts.{e}.up_proj.weight"),
                    ],
                    dim=0,
                )
                for e in range(E)
            ]
        )
        down = torch.stack([converted.pop(f"mlp.experts.{e}.down_proj.weight") for e in range(E)])
        converted["mlp.experts.gate_up_proj"] = gate_up
        converted["mlp.experts.down_proj"] = down
    missing, unexpected = layer.load_state_dict(converted, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not [m for m in missing if "rotary" not in m], f"missing keys: {missing}"
    return layer


@torch.no_grad()
def hf_forward(cfg, layer, x, position_ids=None):
    """Full-sequence fp32 reference forward. x: [B, S, H]. Returns [B, S, H].

    Causal rows i of the output are exact references for decode steps at
    position i, so decode references reuse this via full-sequence forward.
    """
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteRotaryEmbedding

    B, S, _ = x.shape
    if position_ids is None:
        position_ids = torch.arange(S).unsqueeze(0)
    rotary = Glm4MoeLiteRotaryEmbedding(cfg)
    cos, sin = rotary(x, position_ids)
    mask = torch.full((S, S), torch.finfo(torch.float32).min).triu(1)[None, None]
    out = layer(x, attention_mask=mask, position_ids=position_ids, position_embeddings=(cos.float(), sin.float()))
    if isinstance(out, tuple):
        out = out[0]
    return out


# ------------------------------------------------------------------ paged cache helpers


def make_page_table(batch, blocks_per_user, seed=0):
    """Random permuted page table [batch, blocks_per_user] (int32)."""
    gen = torch.Generator().manual_seed(seed)
    total = batch * blocks_per_user
    return torch.randperm(total, generator=gen, dtype=torch.int32).reshape(batch, blocks_per_user)


def gather_user_cache(paged_cache_torch, page_table, user_id, seq_len, block_size):
    """Reassemble user rows [seq_len, kvpe] from a torch copy of the paged cache."""
    n_blocks = -(-seq_len // block_size)
    blocks = page_table[user_id, :n_blocks].long()
    rows = paged_cache_torch[blocks, 0].reshape(n_blocks * block_size, -1)
    return rows[:seq_len]


# ------------------------------------------------------------------ torch MLA reference pieces


@torch.no_grad()
def torch_latent_cache_reference(cfg, sd, x):
    """Exact (linear-only) fp32 reference of the latent cache contents for x [S, H]
    (x = raw decoder-layer input; the input_layernorm is applied here).

    Returns [S, 576] = [rmsnorm(kv_nope), rope(kv_rope)] — what the TTNN cache
    should contain (meta-interleaved rope layout matches HF interleaved pairs).
    """
    S = x.shape[0]
    gin = sd["input_layernorm.weight"]
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * gin
    w = sd["self_attn.kv_a_proj_with_mqa.weight"]
    kv_a = x @ w.T
    nope, rope_part = kv_a[:, : cfg.kv_lora_rank], kv_a[:, cfg.kv_lora_rank :]
    g = sd["self_attn.kv_a_layernorm.weight"]
    var = nope.pow(2).mean(-1, keepdim=True)
    nope = nope * torch.rsqrt(var + 1e-6) * g
    dim = cfg.qk_rope_head_dim
    inv = 1.0 / (cfg.rope_parameters["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = torch.outer(torch.arange(S, dtype=torch.float32), inv)
    cos, sin = ang.cos(), ang.sin()
    r1, r2 = rope_part[:, 0::2], rope_part[:, 1::2]
    roped = torch.stack((r1 * cos - r2 * sin, r2 * cos + r1 * sin), dim=-1).flatten(-2)
    return torch.cat([nope, roped], dim=-1)


@torch.no_grad()
def router_tie_positions(cfg, layer, x, ulp_factor=4.0):
    """Positions whose HF top-4 selection is a sub-bf16-ulp tie.

    The TTNN router computes scores in fp32 but must run ttnn.topk in bf16
    (on per-token mean-centered biased scores). Selection can only differ from
    the fp32 reference when the 4th-vs-5th biased-score gap is within a few
    bf16 rounding quanta at the centered score magnitude: 0.5 spacing typecast
    rounding per candidate plus the perturbation of the fp32 scores caused by
    the bf16 rounding of the layer input itself (the HF reference consumes the
    exact fp32 activations). NOTE on units: the "ulp" computed below is
    2^(floor(log2 |x|) - 8), which is HALF the conventional bf16 spacing
    2^(e-7); ulp_factor=4 therefore admits gaps within 2 conventional bf16
    ULPs. Measured flips to date sit at 0.25-1.1 bf16 spacings.
    Returns {pos: gap} for those tokens.
    """
    if not hasattr(layer.mlp, "route_tokens_to_experts"):
        return {}
    captured = {}
    orig = layer.mlp.route_tokens_to_experts

    def hook(router_logits):
        captured["scores"] = router_logits.sigmoid() + layer.mlp.gate.e_score_correction_bias
        return orig(router_logits)

    layer.mlp.route_tokens_to_experts = hook
    try:
        hf_forward(cfg, layer, x)
    finally:
        layer.mlp.route_tokens_to_experts = orig
    s = captured["scores"]  # [tokens, E]
    centered = s - s.mean(-1, keepdim=True)
    top = torch.topk(centered, 5, dim=-1).values
    gap = top[:, 3] - top[:, 4]
    mag = top[:, 3].abs().clamp_min(1e-30)
    ulp = 2.0 ** (mag.log2().floor() - 8)
    ties = gap < ulp_factor * ulp
    return {int(i): float(gap[i]) for i in ties.nonzero().flatten()}


@torch.no_grad()
def torch_absorbed_window_reference(cfg, sd, hf_layer, x, kvpe, rows, return_tie_mask=False, return_parts=False):
    """fp32 reference of the full decoder-layer output for query rows `rows`
    (an iterable of absolute positions), given the full input x [S, H] and the
    reference latent cache kvpe [S, 576]. Mathematically identical to the HF
    layer (absorbed MLA is an exact refactoring); tractable at 200k context
    because only |rows| query positions are evaluated.
    """
    nh = cfg.num_attention_heads
    dnope, drope, dkv = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.kv_lora_rank
    qk_head = dnope + drope
    rows = torch.tensor(sorted(rows))
    xr = x[rows]  # [R, H]

    def rms(v, g, eps):
        return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps) * g

    h = rms(xr, sd["input_layernorm.weight"], cfg.rms_norm_eps)
    q = rms(h @ sd["self_attn.q_a_proj.weight"].T, sd["self_attn.q_a_layernorm.weight"], 1e-6)
    q = (q @ sd["self_attn.q_b_proj.weight"].T).view(-1, nh, qk_head)
    q_nope, q_rope = q[..., :dnope], q[..., dnope:]
    # interleaved rope at absolute positions
    inv = 1.0 / (cfg.rope_parameters["rope_theta"] ** (torch.arange(0, drope, 2, dtype=torch.float32) / drope))
    ang = rows[:, None].float() * inv[None]
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    r1, r2 = q_rope[..., 0::2], q_rope[..., 1::2]
    q_rope = torch.stack((r1 * cos - r2 * sin, r2 * cos + r1 * sin), dim=-1).flatten(-2)
    kv_b = sd["self_attn.kv_b_proj.weight"].view(nh, dnope + cfg.v_head_dim, dkv)
    q_lat = torch.einsum("rhd,hdk->rhk", q_nope, kv_b[:, :dnope])
    q_abs = torch.cat([q_lat, q_rope], dim=-1)  # [R, nh, 576]

    scale = qk_head**-0.5
    out_rows = []
    for i, pos in enumerate(rows.tolist()):
        keys = kvpe[: pos + 1]  # [pos+1, 576]
        att = torch.softmax((q_abs[i] @ keys.T) * scale, dim=-1)  # [nh, pos+1]
        lat = att @ keys[:, :dkv]  # [nh, 512]
        v = torch.einsum("hk,hvk->hv", lat, kv_b[:, dnope:])  # [nh, v_head]
        out_rows.append(v.reshape(-1))
    attn = torch.stack(out_rows) @ sd["self_attn.o_proj.weight"].T
    res = xr + attn
    h2 = rms(res, sd["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    mlp = hf_layer.mlp(h2.unsqueeze(0)).squeeze(0)
    out = res + mlp
    if not (return_tie_mask or return_parts):
        return out
    # Sub-bf16-ulp router tie flags for the window rows (same rule as
    # router_tie_positions, computed from the reference h2 rows).
    if "mlp.gate.weight" in sd:
        scores = torch.sigmoid(h2 @ sd["mlp.gate.weight"].T) + sd["mlp.gate.e_score_correction_bias"]
        centered = scores - scores.mean(-1, keepdim=True)
        top = torch.topk(centered, 5, dim=-1).values
        gap = top[:, 3] - top[:, 4]
        ulp = 2.0 ** (top[:, 3].abs().clamp_min(1e-30).log2().floor() - 8)
        tie = gap < 4.0 * ulp
    else:
        tie = torch.zeros(len(rows), dtype=torch.bool)
    if return_parts:
        return out, tie, res, h2
    return out, tie


@torch.no_grad()
def manual_moe_row(cfg, sd, h2_row, expert_set):
    """fp32 MoE output for one normed row under an explicit expert set
    (HF math: sigmoid-score weights over the set, normalized + 1e-20, x
    routed_scaling_factor, plus the shared expert)."""

    def swiglu(v, prefix):
        g = v @ sd[f"{prefix}.gate_proj.weight"].T
        u = v @ sd[f"{prefix}.up_proj.weight"].T
        return (torch.nn.functional.silu(g) * u) @ sd[f"{prefix}.down_proj.weight"].T

    scores = torch.sigmoid(h2_row @ sd["mlp.gate.weight"].T)
    picked = scores[list(expert_set)]
    w = picked / (picked.sum() + 1e-20) * cfg.routed_scaling_factor
    routed = sum(w[i] * swiglu(h2_row, f"mlp.experts.{e}") for i, e in enumerate(expert_set))
    return routed + swiglu(h2_row, "mlp.shared_experts")


@torch.no_grad()
def explain_row_as_routing_flip(cfg, sd, h2_row, res_row, got_row, bar):
    """Test whether a below-bar TTNN row is exactly an alternate top-4 routing.

    Tries every 4-subset of the reference top-6 (by biased score); returns
    (best_pcc, expert_set) if some subset's fp32 reference output matches the
    TTNN row at >= bar, else (best_pcc, None).
    """
    from itertools import combinations

    biased = torch.sigmoid(h2_row @ sd["mlp.gate.weight"].T) + sd["mlp.gate.e_score_correction_bias"]
    top6 = torch.topk(biased, 6).indices.tolist()
    best = (-1.0, None)
    for subset in combinations(top6, 4):
        out_alt = res_row + manual_moe_row(cfg, sd, h2_row, subset)
        p = pcc(out_alt, got_row)
        if p > best[0]:
            best = (p, subset)
    return best if best[0] >= bar else (best[0], None)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return float("nan")
    return float((a @ b) / denom)

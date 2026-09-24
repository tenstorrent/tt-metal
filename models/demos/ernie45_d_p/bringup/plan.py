# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Sharding plan for ERNIE-4.5-21B-A3B prefill on a 1x4 Blackhole P150 mesh (single source of truth).

Residual stream: replicated [S, 2560] bf16 on every chip.
  attention : Q/K/V column-parallel by head (5 Q heads + 1 KV head per chip), local chunked SDPA,
              O row-parallel -> all_reduce
  dense MLP : gate/up column-parallel (3072 per chip), down row-parallel -> all_reduce
  MoE       : router replicated (every chip computes the same routing), routed experts expert-parallel
              (16 of 64 per chip, full 1536 FFN), shared experts TP (768 of 3072 per chip);
              routed + shared partials summed locally -> one all_reduce
  embedding : replicated table; LM head vocab-sharded (25856 per chip), logits gathered on host
  KV cache  : one KV head per chip (TP col == KV head, matches the prefill-server contract)
"""

from __future__ import annotations

from models.demos.ernie45_d_p.reference.ernie_ref import ErnieConfig

NUM_CHIPS = 4
CHIP_DRAM_BYTES = 32 * 2**30  # p150b: 32 GB GDDR6 (8 banks)
BF16 = 2
BF8 = 1088 / 1024  # bfloat8_b: 1 byte + shared exponent per 16 values (tile = 1088 B / 1024 elems)
FP32 = 4


def plan(cfg: ErnieConfig | None = None, max_seq: int = 56320, num_users: int = 1, kv_bytes: float = BF16) -> dict:
    cfg = cfg or ErnieConfig()
    H, D, L = cfg.hidden_size, cfg.head_dim, cfg.num_hidden_layers
    n = NUM_CHIPS
    n_moe = sum(cfg.is_moe_layer(i) for i in range(L))
    n_dense = L - n_moe
    q_heads = cfg.num_attention_heads // n
    kv_heads = cfg.num_key_value_heads // n
    e_per = cfg.moe_num_experts // n
    shared_I = cfg.moe_intermediate_size * cfg.moe_num_shared_experts

    per_chip = {
        "attention (Q,K,V,O)": L * (H * (q_heads + 2 * kv_heads) * D + q_heads * D * H) * BF16,
        "dense MLP (layer 0)": n_dense * 3 * H * (cfg.intermediate_size // n) * BF16,
        "routed experts": n_moe * e_per * 3 * H * cfg.moe_intermediate_size * BF16,
        "shared experts": n_moe * 3 * H * (shared_I // n) * BF16,
        "router + bias": n_moe * (cfg.moe_num_experts * H + cfg.moe_num_experts) * FP32,
        "norms": (2 * L + 1) * H * BF16,
        "embedding (replicated)": cfg.vocab_size * H * BF16,
        "LM head (vocab-sharded)": (cfg.vocab_size // n) * H * BF16,
        "KV cache": num_users * L * 2 * kv_heads * max_seq * D * kv_bytes,
    }
    total = sum(per_chip.values())
    chips = []
    for c in range(n):
        chips.append(
            {
                "chip": c,
                "q_heads": list(range(c * q_heads, (c + 1) * q_heads)),
                "kv_heads": list(range(c * kv_heads, (c + 1) * kv_heads)),
                "experts": [c * e_per, (c + 1) * e_per - 1],
                "dense_mlp_cols": [c * cfg.intermediate_size // n, (c + 1) * cfg.intermediate_size // n - 1],
                "shared_mlp_cols": [c * shared_I // n, (c + 1) * shared_I // n - 1],
                "vocab": [c * cfg.vocab_size // n, (c + 1) * cfg.vocab_size // n - 1],
            }
        )
    return {
        "mesh": [1, n],
        "chip_dram_gb": CHIP_DRAM_BYTES / 2**30,
        "max_seq": max_seq,
        "num_users": num_users,
        "per_chip_bytes": {k: int(v) for k, v in per_chip.items()},
        "per_chip_total_gb": total / 2**30,
        "fits": total < 0.85 * CHIP_DRAM_BYTES,
        "chips": chips,
        "ccl_per_layer": ["all_reduce after O proj", "all_reduce after MLP/MoE"],
        "activation_bytes_5k_chunk_est": int(5120 * cfg.moe_num_experts // n * cfg.moe_intermediate_size * 2 * BF16),
    }

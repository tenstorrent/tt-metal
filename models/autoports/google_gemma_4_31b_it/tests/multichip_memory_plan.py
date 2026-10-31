"""Analytical full-stack capacity; estimates are distinct from allocator evidence."""

import json
from collections import Counter
from pathlib import Path

from transformers import AutoConfig

from models.autoports.google_gemma_4_31b_it.tt.optimized_decoder import HF_MODEL, HF_REVISION

c = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
counts = Counter(c.layer_types)
h, i = c.hidden_size, c.intermediate_size
rows = {}
for kind, count in counts.items():
    d = 256 if kind == "sliding_attention" else 512
    nk = 16 if kind == "sliding_attention" else 4
    qkv = (32 + nk * (2 if kind == "sliding_attention" else 1)) * d
    params = h * qkv / 4 + (32 * d / 4) * 5632 + 2 * h * 5632 + 5632 * 5632
    rows[kind] = dict(
        layers=count,
        weight_bytes_per_device_layer=int(params * 576 / 1024 + 4 * h * 64 + 2 * d * 64),
        kv_bytes_per_token_device=int(2 * (nk / 4) * d * 1088 / 1024),
    )
weights = sum(r["layers"] * r["weight_bytes_per_device_layer"] for r in rows.values())
# Embedding and LM-head reserve conservatively uses BF16 TP4, including two copies.
embedding = 2 * c.vocab_size * h * 2 // 4
kv = sum(r["layers"] * r["kv_bytes_per_token_device"] for r in rows.values())
plan = dict(
    layer_types=rows,
    decoder_weights_per_device=weights,
    embedding_head_reserve_per_device=embedding,
    full_cache_bytes_per_token_device=kv,
    full_context_kv_per_device=kv * c.max_position_embeddings,
    trace_activation_reserve_per_device=2 * 1024**3,
    capacity_basis="Analytical; allocator reconciliation pending. Full prefix caches retained for every layer; windowed ring storage not assumed.",
    advertised_context=c.max_position_embeddings,
)
plan["total_estimate_per_device"] = (
    weights + embedding + kv * c.max_position_embeddings + plan["trace_activation_reserve_per_device"]
)
plan[
    "capacity_basis"
] = "Analytical BFP4 tile payload (576 bytes/1024 elements), BFP8 KV (1088 bytes/1024), BF16 norms. Sliding cyclic pool2048; global full prefix. Allocator reconciliation is separate."
plan["sliding_physical_tokens"] = 2048
plan["windowed_kv_per_device"] = sum(
    r["layers"] * r["kv_bytes_per_token_device"] * (2048 if k == "sliding_attention" else c.max_position_embeddings)
    for k, r in rows.items()
)
plan["windowed_total_per_device"] = (
    weights + embedding + plan["windowed_kv_per_device"] + plan["trace_activation_reserve_per_device"]
)
plan["batch_context_capacity"] = [
    dict(
        batch=b,
        context=t,
        estimated_bytes=weights
        + embedding
        + plan["trace_activation_reserve_per_device"]
        + b
        * sum(
            r["layers"] * r["kv_bytes_per_token_device"] * (2048 if k == "sliding_attention" else t)
            for k, r in rows.items()
        ),
    )
    for b in [1, 8, 16, 32]
    for t in [32768, 262144]
]
Path("models/autoports/google_gemma_4_31b_it/doc/multichip_decoder/memory_plan.json").write_text(
    json.dumps(plan, indent=2) + "\n"
)
print(json.dumps(plan, indent=2))

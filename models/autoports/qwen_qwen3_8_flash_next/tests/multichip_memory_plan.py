"""Host-only full-stack capacity arithmetic for the fixed TP4 decoder policy."""

import json, math, os
from pathlib import Path


def tiled(shape, bytes_per_tile):
    *batch, h, w = shape
    return math.prod(batch) * math.ceil(h / 32) * math.ceil(w / 32) * bytes_per_tile


def plan(context=262144, batch=1, requests=2):
    rows = {}
    rows["routed_experts_48"] = 48 * 128 * (2 * tiled((2560, 640), 576) + tiled((640, 2560), 576))
    rows["ple_tables"] = 32 * tiled((2500012, 160), 576)
    rows["gdn_projection_weights_36"] = 36 * (tiled((2560, 4160), 1088) + tiled((1536, 2560), 1088))
    rows["qsa_projection_weights_12"] = 12 * (tiled((2560, 4224), 1088) + tiled((1536, 2560), 1088))
    rows["qsa_decode_projection_weights_12"] = 12 * tiled((2560, 4224), 576) if batch == 1 else 0
    rows["attention_column_wo_48"] = 48 * tiled((6144, 640), 1088)
    rows["hyperconnection_weights_96"] = 96 * (
        tiled((2560, 352), 1088) + tiled((320, 2560), 1088) + tiled((4, 640), 2048)
    )
    rows["shared_router_weights_48"] = 48 * (
        tiled((2560, 544), 2048) + tiled((2560, 320), 2048) + tiled((160, 2560), 2048)
    )
    rows["ple_projection_norm_conv"] = tiled((2560, 3200), 1088) + 3 * tiled((4, 640), 4096) + tiled((4, 2560), 4096)
    rows["embedding_and_lm_head_reserve"] = 2 * tiled((248320, 640), 2048)
    rows["small_weight_metadata_reserve"] = 64 * 1024**2
    weights = sum(rows.values())
    cache = {}
    cache["qsa_kv_12"] = 12 * batch * context * 512
    cache["qsa_raw_index_keys_12"] = 12 * batch * context * 128 * 2
    cache["qsa_pooled_index_keys_12"] = 12 * (batch + requests) * (context // 4) * 128 * 2
    cache["qsa_rope_12"] = 12 * 2 * context * 64 * 2
    cache["gdn_state_36"] = 36 * ((2 * batch + requests) * 12 * 128 * 128 * 4 + (2 * batch + requests) * 32 * 2560 * 2)
    cache["ple_state"] = requests * (9 * 2560 * 2 + 128)
    cache["attention_decode_gather_48"] = 48 * tiled((1, 6144), 1088)
    # One shared BF16 HC-partial gather buffer per layer. The FP32 prefill
    # variant remains live and is covered by the existing activation reserve.
    cache["hc_bf16_decode_gather_48"] = 48 * tiled((4, 1, 352), 2048)
    cache["hc_bf16_norm_stats_gather_48"] = 48 * tiled((4, 128), 2048)
    cache["native_ep_decode_capacity_48"] = 48 * 128 * 32 * 2560 * 2
    cache["native_ep_routing_metadata_reserve_48"] = 48 * 1024**2
    reserve = 1024**3
    return dict(
        mesh=[1, 4],
        physical_device_ids=[3, 2, 1, 0],
        context=context,
        batch=batch,
        max_requests=requests,
        weight_bytes_per_device=weights,
        weight_breakdown=rows,
        cache_breakdown=cache,
        cache_bytes_per_device=sum(cache.values()),
        reserved_trace_activation_bytes_per_device=reserve,
        total_bytes_per_device=weights + sum(cache.values()) + reserve,
        physical_bytes_per_device=32 * 1024**3,
        remaining_bytes_per_device=32 * 1024**3 - weights - sum(cache.values()) - reserve,
        caveat="Calculated payload/tile padding plus explicit reserves; allocator bank padding and physical DRAM must be confirmed on target. Embedding/head reservation is planning only, not full-model implementation.",
    )


if __name__ == "__main__":
    p = plan()
    dest = Path("models/autoports/qwen_qwen3_8_flash_next") / os.environ.get("QWEN38_STAGE_DIR", "doc/multichip_decoder") / "memory_capacity_plan.json"
    dest.write_text(json.dumps(p, indent=2) + "\n")
    print(json.dumps(p, indent=2))

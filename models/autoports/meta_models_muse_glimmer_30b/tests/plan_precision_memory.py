# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recompute precision-specific physical memory accounting without opening TT devices."""

import argparse
import copy
import json
import math
from pathlib import Path

from models.autoports.meta_models_muse_glimmer_30b.tt.precision import decoder_policy, load_precision_config

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"
TILE_BYTES = {"bfloat16": 2048, "bfloat8_b": 1088, "bfloat4_b": 576}


def plan(config):
    old = json.loads((ROOT / "doc/optimized_full_model/memory_capacity_plan.json").read_text())
    terms = copy.deepcopy(old["per_device_terms"])
    shapes = {
        "qkv": (6656, 2304, 3),
        "self_attn.o_proj": (1024, 6656, 3),
        "mlp.gate_up": (6656, 9984, 2),
        "mlp.down_proj": (4992, 6656, 2),
    }
    hf = json.loads((ROOT / "tests/assets/config.json").read_text())["text_config"]
    weight_bytes = 0
    for idx, kind in enumerate(hf["layer_types"]):
        policy = decoder_policy(config, kind, idx)
        for name, (k, n, readers) in shapes.items():
            interleaved = k * n // 1024
            sharded = k * math.ceil(n / (8 * 32 * readers)) * (8 * 32 * readers) // 1024
            weight_bytes += (interleaved + sharded) * TILE_BYTES[policy[name + ".dtype"]]
    terms["decoder_matrices_both_selected_layouts"] = weight_bytes
    for key in list(terms):
        if key.startswith("lm_head_dram_sharded_"):
            del terms[key]
    terminal = config["terminal"]
    local_vocab = math.ceil(hf["vocab_size"] / 128) * 32
    split_width = math.ceil(local_vocab / (32 * terminal["splits"])) * 32
    head_tiles = sum(
        6656 * math.ceil(min(split_width, local_vocab - start) / (8 * 32 * 2)) * (8 * 32 * 2) // 1024
        for start in range(0, local_vocab, split_width)
    )
    terms["lm_head_dram_sharded_selected"] = head_tiles * TILE_BYTES[terminal["weight_dtype"]]
    kv_per_page = 52 * 2 * 32 * 128 // 1024 * TILE_BYTES[config["kv_cache_dtype"]]
    dram = old["per_device_dram_bytes"]
    max_pages = (dram - sum(terms.values())) // kv_per_page
    pool = config["default_physical_pages"]
    return dict(
        classification="modeled",
        config_id=config["config_id"],
        mesh=[1, 4],
        kv_cache_dtype=config["kv_cache_dtype"],
        supported_context=config["max_context"],
        max_batch=32,
        default_physical_pages=pool,
        capability_reduction=None,
        per_device_terms=terms,
        per_device_dram_bytes=dram,
        per_device_kv_bytes_per_page_all_layers=kv_per_page,
        per_device_kv_bytes_default_pool=kv_per_page * pool,
        per_device_default_total_bytes=sum(terms.values()) + kv_per_page * pool,
        per_device_default_margin_bytes=dram - sum(terms.values()) - kv_per_page * pool,
        modeled_max_pages_with_reservations=max_pages,
        modeled_full_context_concurrency=max_pages // 4096,
        largest_feasible_per_request_context=min(131072, max_pages * 32),
        assumptions="Inherited stage7 reservations; BFP exponent and DRAM-bank padding included. Concurrency is modeled, not tested. HF limits each request to131072; physical pages are shared by at most32 logical slots.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", action="store_true")
    args = parser.parse_args()
    plans = {p.stem: plan(load_precision_config(p)) for p in sorted((DOC / "configs").glob("*.json"))}
    (DOC / "memory_capacity_plans.json").write_text(json.dumps(plans, indent=2) + "\n")
    path = ROOT / "doc/context_contract.json"
    contract = json.loads(path.read_text())
    contract["datatype_sweep_candidate_memory_plans"] = {
        k: v
        for k, v in plans.items()
        if k.startswith("kv_") or k.startswith("baseline_") or v["kv_cache_dtype"] != "bfloat8_b"
    }
    if args.selected:
        selected = plan(load_precision_config())
        assert selected["largest_feasible_per_request_context"] == 131072
        contract["datatype_sweep_stage_context_preservation"] = selected
        (DOC / "memory_capacity_plan.json").write_text(json.dumps(selected, indent=2) + "\n")
    path.write_text(json.dumps(contract, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {
                    f: v[f]
                    for f in ("kv_cache_dtype", "per_device_default_margin_bytes", "modeled_full_context_concurrency")
                }
                for k, v in plans.items()
            },
            indent=2,
        )
    )

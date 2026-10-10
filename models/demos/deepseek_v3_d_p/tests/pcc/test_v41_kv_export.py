# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill -> decode hand-off, prefill side, on hardware (tt-blaze DS41F-0037 M2): run the prefill
runtime over a real prompt (``V41PrefillRuntime``, chunked, exports after every chunk), build the contract's 3-config KV
chunk table, then read EVERY chunk the decode ring would ask for back DEVICE-LESSLY through the table
(``read_dram_umd`` + the producer's chunk decoder) and compare it with model.py's state at the same position, laid out
the ring's way (tt-blaze ``seed_from_snapshot.py``): window rings at p % 256, entries at 256 + w, index keys; the
decoder layers 21..39 get layer 20's entries / keys and an empty window.

    V41_PREFILL_TRACE=<prefill_trace dir: model.py's prefill of ids[:S] over layers 0..20>
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.common.prefill.runners.migration import serialize_device_map
from models.demos.common.prefill.runners.prefill_producer import _decode_kv_chunk, _resolve_unique_id
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v41.config import V41Config
from models.demos.deepseek_v3_d_p.tt.v41.kv_export import GROUPS, RING, table_rows
from models.demos.deepseek_v3_d_p.tt.v41.runtime import V41PrefillRuntime
from models.demos.deepseek_v3_d_p.tt.v41.weights import checkpoint

TRACE = os.environ.get("V41_PREFILL_TRACE", "/mnt/tt-data/sdawle/dsv41_golden/prefill_trace_n4096_1536")
CHUNK = int(os.environ.get("V41_TEST_CHUNK", "512"))
CACHE = os.environ.get("V41_WEIGHT_CACHE")
MIN_PCC = float(os.environ.get("V41_TEST_PCC", "0.97"))
MESH = tuple(int(v) for v in os.environ.get("V41_TEST_MESH", "8,4").split(","))

_MESH_CONFIGS = [
    pytest.param(
        MESH,
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        id=f"fabric2d-mesh-{MESH[0]}x{MESH[1]}",
    ),
]


def _expected(gold: dict, cfg, S: int) -> dict:
    """{(group, layer): [rows, width]} as the ring holds it after a prefill of S tokens (seed_from_snapshot's layout)."""
    out = {}
    for L in range(cfg.n_layers):
        r = cfg.role(L)
        win = torch.zeros(RING, 512)
        if L in gold:
            w = gold[L]["window_kv_cache"][0].float()
            for q in range(max(0, S - 128), S):
                win[q % RING] = w[q % 128]
        if not r.compress_ratio:
            out[("swa_window", L)] = win
            continue
        src = gold[r.kv_source]
        out[("csa_unified", L)] = torch.cat([win, src["compress_kv_cache"][0].float()], 0)
        if r.index_source == L:
            out[("index_k", L)] = src["index_k_cache"][0].float()
    return out


@pytest.mark.parametrize("mesh_device, device_params", _MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_v41_kv_export_through_the_table(mesh_device, device_params, tmp_path):
    cfg = V41Config.load()
    meta = json.load(open(os.path.join(TRACE, "meta.json")))
    S = int(meta["S"])
    ids = [int(v) for v in open(meta["prompt_ids_file"]).read().replace("\n", ",").split(",") if v.strip()][:S]
    assert S % CHUNK == 0, (S, CHUNK)
    rt = V41PrefillRuntime(
        mesh_device, cfg, checkpoint(), chunk_size=CHUNK, max_seq_len=max(S, 2048), weight_cache_path=CACHE
    )
    exp = rt.allocate_kv_cache()
    for start in range(0, S, CHUNK):
        rt.prefill_chunk(ids[start : start + CHUNK], exp, slot_id=0, actual_start=start, actual_end=start + CHUNK)
    ttnn.synchronize_device(mesh_device)

    table_path = rt.build_kv_chunk_table(exp, str(tmp_path / "kv_table.pb"))
    map_path = serialize_device_map(mesh_device, str(tmp_path / "device_map.json"))
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(table_path)
    with open(map_path) as f:
        device_map = {tuple(int(x) for x in k.split(":")): int(v) for k, v in json.load(f).items()}
    assert table.num_configs() == len(GROUPS), table.num_configs()

    widths = {"swa_window": 512, "csa_unified": 512, "index_k": 128}
    via: dict = {}
    for g, L, pos, _t, _b, _row in table_rows(exp, cfg, slot=0, prompt_tokens=S):
        loc = table.lookup(L, pos, 0, GROUPS.index(g))
        uid = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = ttnn.experimental.disaggregation.read_dram_umd(uid, loc.noc_addr, loc.size_bytes)
        via.setdefault((g, L), []).append(_decode_kv_chunk(bytes(raw), widths[g]).float())
    via = {k: torch.cat(v, 0) for k, v in via.items()}

    gold = torch.load(os.path.join(TRACE, "cache.pt"))["layers"]
    want = _expected(gold, cfg, S)
    assert set(via) == set(want), sorted(set(via) ^ set(want))
    worst = 1.0
    for (g, L), ref in sorted(want.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        got = via[(g, L)][: ref.shape[0]]
        nz = ref.abs().sum(-1) > 0
        if (g, L)[0] != "index_k" and L > cfg.first_decoder_layer:
            win_zero = bool((got[:RING] == 0).all())
            assert win_zero, f"layer {L}: the decoder window chunks must read zeros"
        _, p = comp_pcc(ref[nz], got[nz])
        stray = float(got[~nz].abs().max()) if (~nz).any() else 0.0
        logger.info(
            f"[v41 kv export] {g:12s} layer {L:2d}: {int(nz.sum())}/{ref.shape[0]} rows, PCC {p:.6f}, "
            f"max |row| where model.py has none {stray:.3g}"
        )
        worst = min(worst, p)
    assert worst >= MIN_PCC, f"worst export PCC {worst:.6f} < {MIN_PCC}"

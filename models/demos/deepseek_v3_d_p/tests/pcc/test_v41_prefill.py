# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill (encoder layers 0..19 + layer 20 KV-only, ``tt/v41/prefill.py``) vs the checkpoint's own
``model.py``, REAL weights, REAL prompt: tt-blaze's per-layer prefill golden (``golden/prefill_trace.py``,
``V41_PREFILL_TRACE``) holds every block's input / output streams and the end-of-prompt cache snapshot.

Checks per layer: the output streams (PCC over [S, hc, d]) and the next block's ``pre`` (max abs error); at the end the
decode-side state the ring is seeded with -- window rings, compressed entries, index keys -- vs the golden ``cache.pt``.
``V41_PREFILL_OUT`` (optional): where to save the device snapshot (``cache_snapshot`` format) for tt-blaze's seed writer.

    PYTHONPATH=<this worktree>:<tt-blaze cpu_stub> pytest models/demos/deepseek_v3_d_p/tests/pcc/test_v41_prefill.py
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v41.config import V41Config
from models.demos.deepseek_v3_d_p.tt.v41.prefill import V41Prefill
from models.demos.deepseek_v3_d_p.tt.v41.weights import checkpoint

TRACE = os.environ.get("V41_PREFILL_TRACE", "/mnt/tt-data/sdawle/dsv41_golden/prefill_trace_n1200")
N_LAYERS = int(os.environ.get("V41_TEST_LAYERS", "20"))
CHUNK = int(os.environ.get("V41_TEST_CHUNK", "2048"))
OUT = os.environ.get("V41_PREFILL_OUT")
CACHE = os.environ.get("V41_WEIGHT_CACHE")  # .tensorbin dir (per mesh shape)
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


def _pcc(a, b) -> float:
    return float(comp_pcc(a.float(), b.float())[1])


def _prompt_ids() -> list[int]:
    meta = json.load(open(os.path.join(TRACE, "meta.json")))
    f = meta["prompt_ids_file"]
    ids = [int(v) for v in open(f).read().replace("\n", ",").split(",") if v.strip()]
    return ids[: int(meta["S"])]


@pytest.mark.parametrize("mesh_device, device_params", _MESH_CONFIGS, indirect=["mesh_device", "device_params"])
def test_v41_prefill_vs_model_py(mesh_device, device_params):
    from blaze.models.deepseek_v4_1_flash.engram_host import EngramHost

    cfg = V41Config.load()
    ids = _prompt_ids()
    S = len(ids)
    sp = mesh_device.shape[0]
    comp2d = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
    results = {}

    def on_hidden(name, t, pre=None):
        if not name.endswith(":out"):
            return
        L = int(name.split(":")[0])
        got = torch.stack([ttnn.to_torch(s, mesh_composer=comp2d)[0, 0, :S].float() for s in t], dim=1)  # [S, hc, d]
        pre_got = ttnn.to_torch(pre, mesh_composer=comp2d)[0, 0, :S, :4].float()
        gold = torch.load(os.path.join(TRACE, f"layer_{L}.pt"))
        x_pcc = _pcc(got, gold["x_out"][:S])
        upd_pcc = _pcc(got - gold["x_in"][:S].float(), gold["x_out"][:S].float() - gold["x_in"][:S].float())
        pre_err = float((pre_got - gold["pre_out"][:S]).abs().max())
        results[L] = (x_pcc, upd_pcc, pre_err)
        logger.info(
            f"[v41 prefill L{L}] x_out PCC {x_pcc:.6f}, update PCC {upd_pcc:.6f}, pre_out max|err| {pre_err:.4g}"
        )

    pf = V41Prefill(
        mesh_device,
        cfg,
        checkpoint(),
        max_seq_len=-(-S // CHUNK) * CHUNK,
        chunk_tokens=min(CHUNK, -(-S // (V41Prefill.CHUNK_ALIGN * sp)) * V41Prefill.CHUNK_ALIGN * sp),
        n_layers=N_LAYERS,
        kv_only=N_LAYERS == cfg.first_decoder_layer,
        weight_cache_path=CACHE,
    )
    host = EngramHost(max_seq_len=S + 64, table_dir=os.environ.get("DSV41_ENGRAM_TABLE_DIR") or None)
    pf.prefill(ids, host, on_hidden=on_hidden)
    snap = pf.cache_snapshot()
    if OUT:
        torch.save(snap, OUT)
        logger.info(f"[v41 prefill] snapshot -> {OUT}")
    gold = torch.load(os.path.join(TRACE, "cache.pt"))
    worst = 1.0
    for L, d in snap["layers"].items():
        g = gold["layers"][L]
        for k, v in d.items():
            n = min(v.shape[1], g[k].shape[1])
            p = _pcc(v[:, :n], g[k][:, :n])
            logger.info(f"[v41 prefill cache L{L}] {k} {tuple(v.shape)} vs golden {tuple(g[k].shape)}: PCC {p:.6f}")
            worst = min(worst, p)
    last = max(results)
    assert results[last][0] >= MIN_PCC, f"layer {last} output PCC {results[last][0]:.6f} < {MIN_PCC}"
    assert worst >= MIN_PCC, f"cache PCC {worst:.6f} < {MIN_PCC}"

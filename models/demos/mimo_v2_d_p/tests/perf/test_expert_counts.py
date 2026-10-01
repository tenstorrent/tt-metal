# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-expert routing counts of a real prefill: a long document chunk by chunk through the model (all-gather MoE
block), and after every MoE layer the route plan's on-device counts read back (tokens routed to each local expert of
each chip). Saved as counts[chunk, moe_layer, device, local_expert] (+ gids) to MIMO_EC_OUT; a short per-layer
summary (per-chip load spread, hottest expert) is logged.

Run with --profile (a few chunks: MIMO_EC_SEQ=8192) to also get each chunk's per-chip FlatRoutedExpert kernel times
(signpost chunk{c}).

    MIMO_EC_SEQ (57344), MIMO_EC_CHUNK (4096), MIMO_EC_LAYERS (48), MIMO_EC_PROMPT (text file; default the golden
    docs_prompt.txt), MIMO_EC_OUT (generated/mimo_expert_counts/counts_<mesh>.pt)
"""

import collections
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.mimo_v2_d_p.reference import hf
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tests.golden import GOLDEN_DIR
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.mimo_v2_d_p.tt.model import TtMiMoModel
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

SEQ = int(os.environ.get("MIMO_EC_SEQ", "57344"))
CHUNK = int(os.environ.get("MIMO_EC_CHUNK", "4096"))
N_LAYERS = int(os.environ.get("MIMO_EC_LAYERS", "48"))
PROMPT = os.environ.get("MIMO_EC_PROMPT", str(GOLDEN_DIR / "docs_prompt.txt"))
# MIMO_EC_TIMES=1 (+ TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1):
# per-chip FlatRoutedExpert kernel time of every MoE layer, read in process after each layer (the longest full-grid
# device program of the layer's MoE part)
TIMES = os.environ.get("MIMO_EC_TIMES") == "1"
MOE_FROM = 20  # program index past the attention block (33-34 programs per MoE layer)


@pytest.mark.timeout(14400)
@MESH_PARAMS
def test_expert_counts(mesh_device, device_params):
    cfg = MiMoTextConfig.from_json()
    rows, cols = tuple(mesh_device.shape)
    n_dev = rows * cols
    epc = cfg.n_routed_experts // n_dev
    model = TtMiMoModel(
        mesh_device,
        cfg,
        lambda i: layer_state(i, cfg),
        fabric_config=device_params["fabric_config"],
        max_seq_len=SEQ,
        chunk_size=CHUNK,
        layers=list(range(N_LAYERS)),
        global_state=global_state,
        options=MiMoRuntimeOptions.from_env(),
    )
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    gids = torch.tensor([[int(g) for g in table[c, r]] for r in range(rows) for c in range(cols)])  # [n_dev, epc]
    moe_layers = [i for i in range(N_LAYERS) if cfg.is_moe(i)]
    by_layer = {i: model.layers[k].ffn.ag for k, i in enumerate(model.layer_ids) if cfg.is_moe(i)}
    # each layer's placement (MIMO_EXPERT_PLACEMENT moves experts per layer; else the EP table's): [moe_layer, dev, epc]
    ffns = [model.layers[k].ffn for k, i in enumerate(model.layer_ids) if cfg.is_moe(i)]
    layer_gids = torch.stack([torch.tensor(f.gids) if f.gids is not None else gids for f in ffns])
    n_chunks = SEQ // CHUNK
    counts = torch.zeros(n_chunks, len(moe_layers), n_dev, epc, dtype=torch.int32)
    times = torch.full((n_chunks, len(moe_layers), n_dev), float("nan"))
    n_progs = collections.Counter()
    chip_ids = list(mesh_device.get_device_ids())  # mesh (row-major) order -> chip id
    if TIMES:
        ttnn.ReadDeviceProfiler(mesh_device)

    def read_times(c, layer_idx):
        ttnn.ReadDeviceProfiler(mesh_device)
        data = ttnn.get_latest_programs_perf_data()
        if layer_idx not in by_layer:
            return
        m = moe_layers.index(layer_idx)
        for d, chip in enumerate(chip_ids):
            progs = sorted(data.get(chip, []), key=lambda p: p.program_execution_uid.runtime_id)
            n_progs[len(progs)] += 1
            # the expert: the longest full-grid program after the attention block (the local reduces are full-grid too,
            # but ~10x shorter; the collectives that wait for it run on a few cores)
            dur = lambda p: p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration / 1e3
            tail = [p for p in progs[MOE_FROM:] if p.core_count == p.num_available_cores]
            if tail:
                times[c, m, d] = max(dur(p) for p in tail)

    def capture(layer_idx, x, kv_actual):
        if TIMES:
            read_times(kv_actual // CHUNK, layer_idx)
        if layer_idx not in by_layer:
            return
        c, m = kv_actual // CHUNK, moe_layers.index(layer_idx)
        dts = ttnn.get_device_tensors(by_layer[layer_idx].plan_op.counts)
        for d in range(n_dev):
            counts[c, m, d] = ttnn.to_torch(dts[d]).reshape(-1)[layer_gids[m, d]].to(torch.int32)

    ids = hf.tokenize_prompt(SEQ, PROMPT)
    for c in range(n_chunks):
        signpost(f"chunk{c}_start")
        out = model.prefill_chunk(ids[c * CHUNK : (c + 1) * CHUNK], c * CHUNK, capture=capture)
        out.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        signpost(f"chunk{c}_end")
        if os.environ.get("MIMO_EC_DRAIN"):  # long --profile runs; tracy -r post-processing rejects a mid-run read
            ttnn.ReadDeviceProfiler(mesh_device)
        tot = counts[c].sum((1, 2))
        assert torch.all(tot == CHUNK * cfg.num_experts_per_tok), (c, tot)  # every (token, k) pair landed once
        per_dev = counts[c].sum(2).float()  # [moe_layer, dev]
        logger.info(
            f"chunk {c}: per-chip load max/mean over layers {(per_dev.max(1).values / per_dev.mean(1)).mean():.2f}x, "
            f"hottest expert {counts[c].max()} tokens (mean {counts[c].float().mean():.0f})"
        )

    out_path = Path(
        os.environ.get(
            "MIMO_EC_OUT",
            Path(__file__).parents[5] / "generated" / "mimo_expert_counts" / f"counts_{mesh_id(mesh_device)}.pt",
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if TIMES:
        ok = times.isfinite()
        logger.info(f"expert times: {int(ok.sum())} of {ok.numel()} samples; programs per layer window {dict(n_progs)}")
    torch.save(
        {
            "counts": counts,
            "times": times,
            "gids": gids,
            "layer_gids": layer_gids,
            "moe_layers": moe_layers,
            "chunk": CHUNK,
            "mesh": (rows, cols),
        },
        out_path,
    )
    logger.info(f"saved {tuple(counts.shape)} to {out_path}")

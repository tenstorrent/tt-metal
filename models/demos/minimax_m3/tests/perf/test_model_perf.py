# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Full-model prefill perf (DP-attention + EP=32), traced + device-bound.

The full forward is traceable (on-device EP bridge + MiniMax device router — no host ops
mid-forward), so we capture a ttnn trace of `prefill_forward` and measure the replay.

`test_model_fwd` (THE headline path — run this directly): build model on (4,8) (DP-attn +
EP=32), prepare a batched-prefill input, warmup, capture trace, then measure the BLOCKING
replay by host wall-clock looped PERF_REPS times. Logs ms/prefill + tokens/s. Wall-clock is
device-bound here (matches Tracy device-kernel to <1% on a small run) and, unlike the Tracy
device-perf harness, does NOT choke on the real model (its per-op CSV overflows at 62 layers
x 256 experts). `signpost(start/stop)` are kept so Tracy still works on small/random configs.

`test_model_device_perf`: the Tracy `run_device_perf` wrapper — usable for small/random configs
only; it FAILS on the real 62-layer model ("Op N not present in cpp_device_perf_report.csv").

Env: PERF_LAYERS / PERF_EXPERTS / PERF_SEQ / PERF_REPS ; REAL=1 → real weights (HF_MODEL);
TRACE_REGION (bytes, bump to ~1e9 for the 62-layer real run). Set TT_MESH_GRAPH_DESC_PATH to
single_bh_galaxy_4x8. If a run hangs, kill the whole tracy process tree (it holds CHIP_IN_USE).

Run (validate trace, fast random):
  PERF_LAYERS=4 PERF_EXPERTS=64 PERF_SEQ=512 pytest models/demos/minimax_m3/tests/perf/test_model_perf.py::test_model_fwd -s
Run (real headline):
  REAL=1 PERF_SEQ=2048 TRACE_REGION=1000000000 HF_MODEL=/data/vmelnykov/MiniMax-M3 pytest ...::test_model_fwd -s
"""

import json
import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn

_THIS = "models/demos/minimax_m3/tests/perf/test_model_perf.py"


def _build_random_model(mesh, layers, experts, seq):
    from types import SimpleNamespace

    from models.demos.minimax_m3.config import MeshConfig
    from models.demos.minimax_m3.tt.ccl import CCLManager
    from models.demos.minimax_m3.tt.model import Model
    from models.demos.minimax_m3.utils.general_utils import get_default_num_links

    p = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "MiniMax-M3", "config.json")
    c = json.load(open(p))
    c.update(num_hidden_layers=layers, num_local_experts=experts, use_qk_norm=False, attn_type_list=[1] * layers)
    cfg = SimpleNamespace(**c)
    H, I, E, V = cfg.hidden_size, cfg.intermediate_size, experts, cfg.vocab_size
    qd, kvd = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
    g = torch.Generator().manual_seed(0)
    rn = lambda *s: (torch.randn(*s, generator=g) * 0.02).to(torch.bfloat16)
    sd = {
        "model.embed_tokens.weight": rn(V, H),
        "model.norm.weight": torch.ones(H, dtype=torch.bfloat16),
        "lm_head.weight": rn(V, H),
    }
    for i in range(layers):
        pf = f"model.layers.{i}."
        sd[pf + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[pf + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[pf + "self_attn.q_proj.weight"], sd[pf + "self_attn.k_proj.weight"] = rn(qd, H), rn(kvd, H)
        sd[pf + "self_attn.v_proj.weight"], sd[pf + "self_attn.o_proj.weight"] = rn(kvd, H), rn(H, qd)
        sd[pf + "block_sparse_moe.gate.weight"] = rn(E, H)
        sd[pf + "block_sparse_moe.e_score_correction_bias"] = torch.zeros(E, dtype=torch.bfloat16)
        for e in range(E):
            ep = pf + f"block_sparse_moe.experts.{e}."
            sd[ep + "w1.weight"], sd[ep + "w3.weight"], sd[ep + "w2.weight"] = rn(I, H), rn(I, H), rn(H, I)
    mesh_config = MeshConfig((4, 8), tp=8)
    ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)
    model = Model(
        mesh_device=mesh,
        hf_config=cfg,
        state_dict=sd,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        max_local_batch_size=1,
        users_row_sharded=True,
        use_ep_moe=True,
        ep_seq_len_per_chip=seq,
    )
    return model, cfg


def _build_real_model(mesh, seq):
    from transformers import AutoConfig

    from models.demos.minimax_m3.config import MeshConfig
    from models.demos.minimax_m3.tt.ccl import CCLManager
    from models.demos.minimax_m3.tt.model import Model
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.utils.general_utils import get_default_num_links

    ma = ModelArgs(mesh_device=mesh)
    cfg = AutoConfig.from_pretrained(ma.model_path, trust_remote_code=True)
    sd = ModelArgs.load_state_dict(ma.weights_path)
    mesh_config = MeshConfig((4, 8), tp=8)
    ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)
    model = Model(
        mesh_device=mesh,
        hf_config=cfg,
        state_dict=sd,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        tensor_cache_path=ma.weight_cache_path(ttnn.bfloat8_b),
        max_local_batch_size=1,
        users_row_sharded=True,
        use_ep_moe=True,
        ep_seq_len_per_chip=seq,
    )
    del sd
    return model, cfg


@pytest.mark.timeout(0)  # real-weights load (~45 min over NFS) exceeds the default 300s
def test_model_fwd():
    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config

    fabric_payload_size = 6144  # M3 hidden_size (max fabric packet payload)

    seq = int(os.getenv("PERF_SEQ", "512"))
    layers = int(os.getenv("PERF_LAYERS", "4"))
    experts = int(os.getenv("PERF_EXPERTS", "64"))
    real = os.getenv("REAL", "0") == "1"
    torch.manual_seed(0)

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        create_fabric_router_config(max_payload_size=fabric_payload_size),
    )
    trace_region = int(os.getenv("TRACE_REGION", "300000000"))  # bump for 62-layer real run
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8), trace_region_size=trace_region)
    try:
        model, cfg = _build_real_model(mesh, seq) if real else _build_random_model(mesh, layers, experts, seq)
        logger.info(
            f"[model-perf] built ({'REAL' if real else 'random'}) layers={cfg.num_hidden_layers} experts={cfg.num_local_experts} seq={seq}"
        )
        toks = torch.randint(0, cfg.vocab_size, (4, seq), dtype=torch.int32)
        host_out = model.prepare_inputs_prefill(toks, batched_prefill=True)
        last = ((seq - 1) // 32) * 32
        # The decoder layer frees its OWN input buffer (layer.py: residual.deallocate after the
        # residual add), so each forward consumes host_out[0]. Keep it PERSISTENT and run the
        # forward on a fresh clone — the original survives capture and stays at a stable address
        # for execute_trace's replay.
        x_persist = host_out[0]
        fwd = lambda: model.prefill_forward(
            ttnn.clone(x_persist),
            rot_mats_global=host_out[1],
            rot_mats_local=host_out[2],
            kv_cache=None,
            batch_size=1,
            get_last_token=last,
        )
        fwd()  # warmup/compile
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = fwd()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        logger.info(f"[model-perf] TRACED full-model replay captured: layers={cfg.num_hidden_layers} seq={seq}")

        # Headline measurement: host wall-clock of the BLOCKING traced replay. The trace is
        # pure device execution (no host ops), so wall ~= device time + one tiny dispatch.
        # We loop K times so the fixed dispatch cost amortizes to ~0. This sidesteps the Tracy
        # device-perf CSV (its per-op matching overflows at 62 layers * 256-expert MoE), and is
        # exactly the end-to-end prefill latency we care about. Signposts kept for optional Tracy.
        import time

        K = int(os.getenv("PERF_REPS", "10"))
        signpost(header="start")
        t0 = time.perf_counter()
        for _ in range(K):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        t1 = time.perf_counter()
        signpost(header="stop")
        ttnn.deallocate(out)

        wall = (t1 - t0) / K  # seconds per full-model prefill
        total_tokens = 4 * seq  # 4 prompts/rows (DP), seq tokens each
        logger.info(
            f"\n[model-perf] FULL MODEL ({'REAL' if real else 'random'}) "
            f"layers={cfg.num_hidden_layers} experts={cfg.num_local_experts} seq={seq} "
            f"tokens={total_tokens} reps={K}\n"
            f"  wall-clock / prefill (traced replay): {wall*1e3:.2f} ms\n"
            f"  prefill throughput: {total_tokens / wall:,.0f} tokens/s"
        )
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.mark.timeout(0)
def test_model_device_perf():
    from models.perf.device_perf_utils import run_device_perf

    seq = int(os.getenv("PERF_SEQ", "512"))
    layers = int(os.getenv("PERF_LAYERS", "4"))
    real = os.getenv("REAL", "0") == "1"
    eff_layers = 62 if real else layers
    total_tokens = 4 * seq  # 4 prompts/rows
    res = run_device_perf(
        command=f"pytest {_THIS}::test_model_fwd -s",
        subdir="minimax_model",
        num_iterations=1,
        cols=["DEVICE KERNEL"],
        batch_size=1,
        has_signposts=True,
    )
    ns = res["AVG DEVICE KERNEL DURATION [ns]"]
    logger.info(
        f"\n[model-perf] FULL MODEL ({'REAL 62L' if real else f'{layers}L random'}) seq={seq} tokens={total_tokens}\n"
        f"  device-kernel time (summed/32 chips): {ns/1e6:.1f} ms  (~{ns/32/1e6:.2f} ms/chip)\n"
        f"  prefill throughput: {total_tokens / (ns/32/1e9):,.0f} tokens/s (per-chip-parallel basis)"
    )

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated MoE PCC tests: top-k gate, expert FFN, full MoE layer, mesh parallel.
# Real layer-0 checkpoint weights; lean ISL S=1, 32 (+ S=4096 slow for router).
#
# Run (fast):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_moe.py -m "not slow" -v

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[5]
PCC_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PCC_DIR) not in sys.path:
    sys.path.insert(0, str(PCC_DIR))

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.moe.gate import HunyuanTopKGate
from models.experimental.hunyuan_image_3_0.ref.moe.mlp import HunyuanMLP as RefMLP
from models.experimental.hunyuan_image_3_0.ref.moe.moe import HunyuanMoE as RefMoE
from models.experimental.hunyuan_image_3_0.ref.weights import (
    load_prefixed_state_dict,
    load_tensors,
    resolve_base_model_dir,
)
from models.experimental.hunyuan_image_3_0.tt.moe.gate import HunyuanTtTopKGate
from models.experimental.hunyuan_image_3_0.tt.moe.mlp import HunyuanTtMLP
from models.experimental.hunyuan_image_3_0.tt.moe.moe import HunyuanTtMoE
from models.experimental.hunyuan_image_3_0.tt.moe.moe_parallel import HunyuanTtMoEParallel
from models.tt_dit.parallel.manager import CCLManager
from pcc_common import (
    MOE_ISL_FAST,
    MOE_ISL_PRODUCTION,
    MOE_ISL_SLOW,
    MOE_PARALLEL_PCC,
    MOE_PARALLEL_PCC_BF8,
    MOE_SET_MATCH,
    MOE_TIE_EPS,
    MOE_WEIGHT_PCC,
    PCC_STRICT,
    pcc_metrics,
    transformer_cfg,
)

LAYER = 0
LAYER_PREFIX = f"model.layers.{LAYER}"
MLP_PREFIX = f"{LAYER_PREFIX}.mlp"
WG_KEY = f"{MLP_PREFIX}.gate.wg.weight"
BATCH = 1

MOE_MODULE_CASES = [
    ("expert_ffn", MOE_ISL_FAST),
    ("moe_layer", MOE_ISL_FAST),
]


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="function")
def device_params():
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _router_run(device, seq_len: int):
    c = transformer_cfg()
    wg = load_tensors(resolve_base_model_dir(), [WG_KEY])[WG_KEY]
    state_dict = {WG_KEY: wg}

    ref_gate = HunyuanTopKGate(
        hidden_size=c["H"], num_experts=c["E"], moe_topk=c["K"], norm_topk_prob=c["NORM_TOPK"]
    ).eval()
    ref_gate.wg.weight = torch.nn.Parameter(wg.float().clone())

    torch.manual_seed(0)
    x = torch.randn(BATCH, seq_len, c["H"], dtype=torch.bfloat16)
    with torch.no_grad():
        ref_w, ref_idx = ref_gate(x, topk_impl="easy")
        ref_gates = torch.softmax(ref_gate.wg(x.reshape(-1, c["H"]).float()), dim=-1)

    tt_gate = HunyuanTtTopKGate(device, c["H"], c["E"], c["K"], state_dict, WG_KEY, norm_topk_prob=c["NORM_TOPK"])
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_w_t, tt_idx_t = tt_gate(x_tt)
    tt_w = ttnn.to_torch(tt_w_t).reshape(BATCH * seq_len, c["K"])
    tt_idx = ttnn.to_torch(tt_idx_t).reshape(BATCH * seq_len, c["K"]).long()
    x_tt.deallocate(True)

    ref_sets = [set(r.tolist()) for r in ref_idx]
    tt_sets = [set(r.tolist()) for r in tt_idx]
    same = sum(int(a == b) for a, b in zip(ref_sets, tt_sets))
    total = len(ref_sets)

    sorted_gates, _ = torch.sort(ref_gates, dim=-1, descending=True)
    boundary_gap = (sorted_gates[:, c["K"] - 1] - sorted_gates[:, c["K"]]).abs()
    tie_explained = sum(
        1 for t, (a, b) in enumerate(zip(ref_sets, tt_sets)) if a != b and boundary_gap[t].item() <= MOE_TIE_EPS
    )
    genuine = total - same - tie_explained
    effective_rate = (same + tie_explained) / total

    def aligned(idx, w):
        out = torch.zeros(idx.shape[0], c["E"])
        out.scatter_(1, idx, w.float())
        return out

    weight_pcc, _ = pcc_metrics(aligned(ref_idx, ref_w), aligned(tt_idx, tt_w), MOE_WEIGHT_PCC)
    return effective_rate, weight_pcc, genuine


def _expert_ffn_run(device, seq_len: int):
    c = transformer_cfg()
    expert_prefix = f"{MLP_PREFIX}.experts.0"
    exp_sd = load_prefixed_state_dict(resolve_base_model_dir(), f"{expert_prefix}.")

    torch.manual_seed(0)
    x = torch.randn(BATCH, seq_len, c["H"], dtype=torch.bfloat16)
    ref_mlp = RefMLP(c["H"], c["MOE_INTER"], is_moe=True)
    ref_mlp.load_state_dict({k: v.float() for k, v in exp_sd.items()})
    ref_mlp.eval()
    with torch.no_grad():
        ref_out = ref_mlp(x.float())

    tt_sd = {f"{expert_prefix}.{k}": v for k, v in exp_sd.items()}
    tt_mlp = HunyuanTtMLP(device, c["H"], tt_sd, expert_prefix)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.to_torch(tt_mlp(x_tt))
    x_tt.deallocate(True)
    return pcc_metrics(ref_out, tt_out, PCC_STRICT)


def _moe_layer_run(device, seq_len: int):
    c = transformer_cfg()
    sd = load_prefixed_state_dict(resolve_base_model_dir(), f"{LAYER_PREFIX}.")
    moe_sd = {k[len("mlp.") :]: v for k, v in sd.items() if k.startswith("mlp.")}

    torch.manual_seed(0)
    x = torch.randn(BATCH, seq_len, c["H"], dtype=torch.bfloat16)
    ref_moe = RefMoE(
        c["H"],
        c["MOE_INTER"],
        num_experts=c["E"],
        moe_topk=c["K"],
        num_shared_expert=c["NUM_SHARED"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
    )
    ref_moe.load_state_dict({k: v.float() for k, v in moe_sd.items()})
    ref_moe.eval()
    with torch.no_grad():
        ref_out = ref_moe(x.float())

    tt_moe_sd = {f"{MLP_PREFIX}.{k}": v for k, v in moe_sd.items()}
    tt_moe = HunyuanTtMoE(
        device,
        c["H"],
        c["E"],
        c["K"],
        tt_moe_sd,
        MLP_PREFIX,
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        stream_experts=True,
    )
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.to_torch(tt_moe(x_tt))
    x_tt.deallocate(True)
    return pcc_metrics(ref_out, tt_out, PCC_STRICT)


# ---------------------------------------------------------------------------
# Top-k router
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len,label", MOE_ISL_FAST)
def test_moe_router_pcc(device, seq_len, label):
    effective_rate, weight_pcc, genuine = _router_run(device, seq_len)
    print(
        f"MoE router [{label}] S={seq_len}: effective={100 * effective_rate:.2f}%  "
        f"weight_PCC={weight_pcc:.8f}  genuine_mismatch={genuine}"
    )
    assert genuine == 0
    assert effective_rate >= MOE_SET_MATCH
    assert weight_pcc >= MOE_WEIGHT_PCC


@pytest.mark.slow
@pytest.mark.parametrize("seq_len,label", MOE_ISL_SLOW)
def test_moe_router_large_isl_pcc(device, seq_len, label):
    effective_rate, weight_pcc, genuine = _router_run(device, seq_len)
    assert genuine == 0 and effective_rate >= MOE_SET_MATCH and weight_pcc >= MOE_WEIGHT_PCC


# ---------------------------------------------------------------------------
# Expert FFN + full MoE layer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module,seq_len,label", [(module, seq_len, label) for module, cases in MOE_MODULE_CASES for seq_len, label in cases]
)
def test_moe_module_pcc(device, module, seq_len, label):
    if module == "expert_ffn":
        p, d = _expert_ffn_run(device, seq_len)
    else:
        p, d = _moe_layer_run(device, seq_len)
    print(f"MoE {module} [{label}] S={seq_len}: PCC={p:.8f}  max|diff|={d:.6f}")
    assert p >= PCC_STRICT


@pytest.mark.slow
@pytest.mark.parametrize(
    "module,seq_len,label",
    [(module, seq_len, label) for module in ("expert_ffn", "moe_layer") for seq_len, label in MOE_ISL_PRODUCTION],
)
def test_moe_module_production_pcc(device, module, seq_len, label):
    """MoE submodule PCC at production decode (S=1) and prefill (S=4160)."""
    if module == "expert_ffn":
        p, d = _expert_ffn_run(device, seq_len)
    else:
        p, d = _moe_layer_run(device, seq_len)
    phase = "decode" if seq_len == 1 else "prefill"
    print(f"MoE {module} production {phase} [{label}] S={seq_len}: PCC={p:.8f}  max|diff|={d:.6f}")
    assert p >= PCC_STRICT


@pytest.mark.slow
@pytest.mark.parametrize("seq_len,label", MOE_ISL_PRODUCTION)
def test_moe_router_production_pcc(device, seq_len, label):
    """Top-k router at production decode (S=1) and prefill (S=4160)."""
    effective_rate, weight_pcc, genuine = _router_run(device, seq_len)
    phase = "decode" if seq_len == 1 else "prefill"
    print(
        f"MoE router production {phase} [{label}] S={seq_len}: effective={100 * effective_rate:.2f}%  "
        f"weight_PCC={weight_pcc:.8f}"
    )
    assert genuine == 0 and effective_rate >= MOE_SET_MATCH and weight_pcc >= MOE_WEIGHT_PCC


# ---------------------------------------------------------------------------
# Mesh expert-parallel vs dense
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_moe_parallel_vs_dense(mesh_device):
    mesh_device.enable_program_cache()
    c = transformer_cfg()
    sd = {k: v for k, v in load_prefixed_state_dict(resolve_base_model_dir(), f"{MLP_PREFIX}.").items()}
    sd = {f"{MLP_PREFIX}.{k}": v for k, v in sd.items()}
    seq_len = 32

    torch.manual_seed(0)
    x = torch.randn(BATCH, seq_len, c["H"]) * 0.05
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    dense = HunyuanTtMoE(
        mesh_device,
        c["H"],
        c["E"],
        c["K"],
        sd,
        MLP_PREFIX,
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        weight_dtype=ttnn.bfloat16,
        stream_experts=False,
    )
    y_dense = dense.forward(x_tt)

    par_dtype = ttnn.bfloat8_b if os.environ.get("HY_MOE_DTYPE", "bf16") == "bf8" else ttnn.bfloat16
    thr = MOE_PARALLEL_PCC_BF8 if par_dtype == ttnn.bfloat8_b else MOE_PARALLEL_PCC
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    par = HunyuanTtMoEParallel(
        mesh_device,
        ccl,
        sd,
        MLP_PREFIX,
        num_experts=c["E"],
        hidden_size=c["H"],
        moe_topk=c["K"],
        norm_topk_prob=c["NORM_TOPK"],
        use_mixed_mlp_moe=c["MIXED"],
        mesh_axis=1,
        weight_dtype=par_dtype,
    )
    y_par = par.forward(x_tt)

    d0 = ttnn.to_torch(y_dense, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:BATCH].float()
    p0 = ttnn.to_torch(y_par, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:BATCH].float()
    passing, pcc = comp_pcc(d0, p0, thr)
    logger.info(f"moe_parallel({par_dtype}) vs dense(bf16) PCC: {pcc:.6f}")
    assert passing, f"PCC {pcc:.6f} < {thr}"

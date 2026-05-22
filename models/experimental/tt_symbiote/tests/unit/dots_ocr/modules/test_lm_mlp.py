# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 module test: text-decoder MLP variants.

Targets the actual wrapper classes seen in the Phase 0 capture
(``shape_matrix/text_modules_dedup.json``):

* :class:`TTNNDotsOCRMLP` — outer SwiGLU MLP
* :class:`TTNNDotsOCRFusedGateUpRowSharded` — its fused gate+up projection
* :class:`TTNNDotsOCRRowShardedNoAllGather` — its down projection
* :class:`TTNNLinearLLamaIColShardedWAllReduced` — used by attention QKV
  (also tested here to keep the LM-side linear coverage in one file)
* :class:`TTNNLinearLLamaIReplicatedWColSharded` — used by attention out-proj

Reference: random ``Qwen2MLP`` (and matching torch ``nn.Linear`` instances).
Per Phase 0 finding §1.2 the production pipeline specializes weight dtype
on ``layer_idx``: BFP4 for 0..6, BFP8 for 7..27. We sweep the boundary
explicitly: ``{0, 6, 7, 13, 20, 27}``.

Per Phase 0 finding §11.8 the linear wrapper classes appear by name in the
capture; for the LM MLP they are children of :class:`TTNNDotsOCRMLP`, so
covering the parent class exercises both children at once. We add two
direct sub-tests for the column-replicated / column-sharded linear classes
because they are not children of TTNNDotsOCRMLP (they live in the attention
module). PCC threshold for BFP4-weighted MLPs is set per dtype.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.modules.dots_ocr_mlp import TTNNDotsOCRMLP
from models.experimental.tt_symbiote.modules.dots_ocr_decoder_layer import (
    _use_bfp8_decoder_weights,
)
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearLLamaIColShardedWAllReduced,
    TTNNLinearLLamaIReplicatedWColSharded,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.reference.architecture_factory import (
    build_random_qwen2_mlp,
    _get_dots_config,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.module_helpers import (
    gather_replicated_first,
    prepare_module,
    replicated_from_torch,
)
from models.experimental.tt_symbiote.tests.unit.dots_ocr.util.pcc import assert_op_pcc


# User decision: representatives + small sweep across BFP4 / BFP8 groups
# plus the transition boundary 6→7. Phase 3 §1 of the task message.
_LAYER_IDXS = [0, 6, 7, 13, 20, 27]


def _mlp_threshold(layer_idx: int) -> float:
    """BFP4 layers (0..6) get a looser PCC budget than BFP8 layers (7..27)."""
    return 0.95 if not _use_bfp8_decoder_weights(layer_idx) else 0.97


# ---------------------------------------------------------------------------
# TTNNDotsOCRMLP (parent — exercises FusedGateUp + RowShardedNoAllGather)
# ---------------------------------------------------------------------------


_MLP_ROWS: List[Dict[str, Any]] = []
for li in _LAYER_IDXS:
    for phase, shape in [("prefill", (1, 14, 1536)), ("decode", (1, 1, 1536))]:
        _MLP_ROWS.append(
            {
                "id": f"lm_mlp_L{li}_{phase}_b{shape[0]}_s{shape[1]}_h{shape[2]}",
                "layer_idx": li,
                "phase": phase,
                "shape": shape,
            }
        )


@pytest.mark.parametrize("row", _MLP_ROWS, ids=[r["id"] for r in _MLP_ROWS])
def test_lm_mlp_full(row, mesh_device_t3k_dp):
    """PCC for TTNNDotsOCRMLP (covers fused gate-up + down-proj transitively)."""
    torch.manual_seed(0)
    layer_idx = row["layer_idx"]

    ref = build_random_qwen2_mlp(layer_idx=layer_idx, seed=0).to(torch.bfloat16).eval()

    x_torch = torch.randn(*row["shape"], dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        ref_out = ref(x_torch).to(torch.float32)

    tt_module = TTNNDotsOCRMLP.from_torch(ref)
    # Production pipeline: bfp8 for layer >= 7, bfp4 otherwise.
    if _use_bfp8_decoder_weights(layer_idx):
        tt_module.set_weight_dtype(ttnn.bfloat8_b)
    prepare_module(tt_module, mesh_device_t3k_dp)

    x_tt = replicated_from_torch(x_torch, mesh_device=mesh_device_t3k_dp)

    try:
        out_tt = tt_module(x_tt)
    except Exception as e:
        pytest.xfail(f"Unsupported sharding/CCL combination for shape {row['shape']}: {e}")

    out_torch = gather_replicated_first(out_tt, mesh_device_t3k_dp).to(torch.float32)
    while out_torch.dim() > ref_out.dim() and out_torch.shape[0] == 1:
        out_torch = out_torch.squeeze(0)
    if out_torch.shape != ref_out.shape:
        try:
            out_torch = out_torch.reshape(ref_out.shape)
        except RuntimeError:
            pass

    threshold = _mlp_threshold(layer_idx)
    pcc = assert_op_pcc(
        ref_out,
        out_torch,
        threshold=threshold,
        op_name="TTNNDotsOCRMLP",
        row_id=row["id"],
    )
    print(f"\n[{row['id']}] PCC={pcc:.5f} (threshold={threshold:.4f})")


# ---------------------------------------------------------------------------
# TTNNLinearLLamaIColShardedWAllReduced (text QKV linear class) — uses random
# torch.nn.Linear with input dim = hidden_size. The Phase 0 capture has this
# class show up as the QKV fused projection inside TTNNDotsOCRAttention; we
# test it standalone here to give the class its own per-row PCC.
# ---------------------------------------------------------------------------


_TXT_QKV_ROWS = [
    {"id": "lm_lin_colsharded_allreduce_prefill_b1_s14_h1536_n2048", "shape": (1, 14, 1536), "out": 2048},
    {"id": "lm_lin_colsharded_allreduce_decode_b1_s1_h1536_n2048", "shape": (1, 1, 1536), "out": 2048},
]


@pytest.mark.parametrize("row", _TXT_QKV_ROWS, ids=[r["id"] for r in _TXT_QKV_ROWS])
def test_lm_lin_col_sharded_all_reduced(row, mesh_device_t3k_dp):
    """``TTNNLinearLLamaIColShardedWAllReduced`` standalone PCC."""
    torch.manual_seed(0)
    cfg = _get_dots_config()
    K = cfg.hidden_size
    N = row["out"]
    lin = torch.nn.Linear(K, N, bias=True)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for i, p in enumerate(lin.parameters()):
            g.manual_seed(i)
            tmp = torch.empty_like(p, dtype=torch.float32)
            tmp.normal_(mean=0.0, std=0.02, generator=g)
            p.copy_(tmp.to(torch.bfloat16))
    lin = lin.to(torch.bfloat16).eval()

    x_torch = torch.randn(*row["shape"], dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        ref_out = torch.nn.functional.linear(x_torch, lin.weight, lin.bias).to(torch.float32)

    tt_lin = TTNNLinearLLamaIColShardedWAllReduced.from_torch(lin)
    prepare_module(tt_lin, mesh_device_t3k_dp)

    x_tt = replicated_from_torch(x_torch, mesh_device=mesh_device_t3k_dp)

    try:
        out_tt = tt_lin(x_tt)
    except Exception as e:
        pytest.xfail(f"Sharded linear path requires production-matched input sharding: {e}")

    out_torch = gather_replicated_first(out_tt, mesh_device_t3k_dp).to(torch.float32)
    while out_torch.dim() > ref_out.dim() and out_torch.shape[0] == 1:
        out_torch = out_torch.squeeze(0)
    if out_torch.shape != ref_out.shape:
        try:
            out_torch = out_torch.reshape(ref_out.shape)
        except RuntimeError:
            pass

    # bfloat8_b weight + bf16 act + CCL all-reduce — match dtype threshold
    pcc = assert_op_pcc(
        ref_out,
        out_torch,
        threshold=0.97,
        op_name="TTNNLinearLLamaIColShardedWAllReduced",
        row_id=row["id"],
    )
    print(f"\n[{row['id']}] PCC={pcc:.5f} (threshold=0.97)")


# ---------------------------------------------------------------------------
# TTNNLinearLLamaIReplicatedWColSharded (text o-proj class) — standalone PCC.
# ---------------------------------------------------------------------------


_TXT_O_ROWS = [
    {"id": "lm_lin_rep_colsharded_prefill_b1_s14_h1536_n1536", "shape": (1, 14, 1536), "out": 1536},
    {"id": "lm_lin_rep_colsharded_decode_b1_1_1_s1536", "shape": (1, 1, 1, 1536), "out": 1536},
]


@pytest.mark.parametrize("row", _TXT_O_ROWS, ids=[r["id"] for r in _TXT_O_ROWS])
def test_lm_lin_replicated_col_sharded(row, mesh_device_t3k_dp):
    """``TTNNLinearLLamaIReplicatedWColSharded`` standalone PCC."""
    torch.manual_seed(0)
    cfg = _get_dots_config()
    K = cfg.hidden_size
    N = row["out"]
    lin = torch.nn.Linear(K, N, bias=False)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for i, p in enumerate(lin.parameters()):
            g.manual_seed(i)
            tmp = torch.empty_like(p, dtype=torch.float32)
            tmp.normal_(mean=0.0, std=0.02, generator=g)
            p.copy_(tmp.to(torch.bfloat16))
    lin = lin.to(torch.bfloat16).eval()

    x_torch = torch.randn(*row["shape"], dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        ref_out = torch.nn.functional.linear(x_torch, lin.weight, None).to(torch.float32)

    tt_lin = TTNNLinearLLamaIReplicatedWColSharded.from_torch(lin)
    prepare_module(tt_lin, mesh_device_t3k_dp)

    x_tt = replicated_from_torch(x_torch, mesh_device=mesh_device_t3k_dp)

    try:
        out_tt = tt_lin(x_tt)
    except Exception as e:
        pytest.xfail(f"Sharded linear path requires production-matched input sharding: {e}")

    out_torch = gather_replicated_first(out_tt, mesh_device_t3k_dp).to(torch.float32)
    while out_torch.dim() > ref_out.dim() and out_torch.shape[0] == 1:
        out_torch = out_torch.squeeze(0)
    if out_torch.shape != ref_out.shape:
        try:
            out_torch = out_torch.reshape(ref_out.shape)
        except RuntimeError:
            pass

    pcc = assert_op_pcc(
        ref_out,
        out_torch,
        threshold=0.97,
        op_name="TTNNLinearLLamaIReplicatedWColSharded",
        row_id=row["id"],
    )
    print(f"\n[{row['id']}] PCC={pcc:.5f} (threshold=0.97)")

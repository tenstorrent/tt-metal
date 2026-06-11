# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for ``models.common.modules.moe.tt_moe_gate.TTMoEGate``.

Modeled on ``test_tt_moe_decode.py``: load each YAML model config, derive the gate
sizes (``num_routed_experts`` / ``select_experts_k`` / ``hidden_size``), build torch
hidden states + a router weight, run ``TTMoEGate`` to produce ``(scores, indices)``,
and verify against the torch golden. ``TTMoEGate`` produces exactly the routing
``(tt_scores, tt_indices)`` that ``test_tt_moe_decode`` currently fakes with random
tensors, so the two compose into the full router→MoE path.

Scaffold scope: ``num_routed_experts == 256`` (``n_group`` 1 → generalized op, 8 → deepseek
grouped op). Other sizes (64/128 pad-to-256, 512 combine) are skipped for now.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from loguru import logger

import ttnn
from models.common.modules.moe.tt_moe_decode_config import TTMoEDecodeConfig
from models.common.modules.moe.tt_moe_gate import TTMoEGate
from tests.ttnn.utils_for_testing import comp_pcc

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "modules/moe/configs"
CONFIG_PATHS = sorted(CONFIGS_DIR.glob("*.yaml"))
assert CONFIG_PATHS, f"no YAML configs found in {CONFIGS_DIR}"


def _config_id(path: Path) -> str:
    return path.stem


def _gate_params_for(raw: dict) -> dict:
    """Gate semantics read straight from the model yaml. Fields (defaults = softmax-no-bias /
    ungrouped): ``n_group`` (1 → generalized op, 8 → deepseek grouped op), ``score_func``
    (softmax|sigmoid), ``routed_scaling_factor``."""
    return {
        "n_group": raw.get("n_group", 1),
        "score_func": raw.get("score_func", "softmax"),
        "scaling_factor": raw.get("routed_scaling_factor", 1.0),
    }


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=_config_id)
@pytest.mark.parametrize("seed", [42])
def test_tt_moe_gate(device, config_path: Path, seed: int):
    yaml_text = config_path.read_text()
    config = TTMoEDecodeConfig.from_yaml(yaml_text, topology=ttnn.Topology.Linear)
    raw = yaml.safe_load(yaml_text)
    gp = _gate_params_for(raw)
    matmul_cc = raw.get("gate_matmul_compute")  # optional, per-model router-matmul fidelity (TTMoEGate-only)

    num_experts = config.num_routed_experts
    k = config.select_experts_k
    hidden = config.hidden_size

    if num_experts > 512 or (gp["n_group"] == 8 and num_experts != 256):
        pytest.skip(
            f"supports ≤512 experts; n_group=8 (deepseek grouped) needs 256. n_group=1 with k∈(4,6,8) uses "
            f"the kernel op, other k (e.g. 10) uses the ttnn fallback. got N={num_experts}, "
            f"n_group={gp['n_group']}, k={k}"
        )

    batch = 32  # decode: one token per core, tile-aligned. (forward() pads/loops for other batch sizes.)
    logger.info(f"[{config_path.stem}] gate: N={num_experts} k={k} hidden={hidden} batch={batch} {gp}")
    # per-model caveats about steps TTMoEGate does NOT cover (e.g. gemma4's RMSNorm + per-dim input scale
    # and per-expert output scale) — the caller must apply these around TTMoEGate.
    if raw.get("gate_notes"):
        logger.warning(f"[{config_path.stem}] gate_notes: {raw['gate_notes']}")

    # --- torch inputs ---
    torch.manual_seed(seed)
    hidden_states = (2 * torch.rand((batch, hidden), dtype=torch.bfloat16)) - 1
    gate_weight = ((2 * torch.rand((hidden, num_experts), dtype=torch.bfloat16)) - 1) * 0.1
    # score-correction bias (deepseek/noaux_tc sigmoid+sqrtsoftplus models): present iff the yaml declares
    # `score_correction_bias: true` — EXPLICIT, not inferred from score_func. Added to scores for SELECTION
    # only (the output weights stay unbiased). None → TTMoEGate feeds the op a zeros bias (a no-op).
    gate_bias = (2 * torch.rand((num_experts,)) - 1) if raw.get("score_correction_bias") else None
    # router LINEAR bias (gpt-oss, yaml router_bias: true): logits = Wx + b, flows into selection + weights.
    proj_bias = (2 * torch.rand((num_experts,)) - 1) if raw.get("router_bias") else None

    # --- device module + inputs ---
    gate = TTMoEGate(
        device,
        num_experts=num_experts,
        select_experts_k=k,
        hidden_size=hidden,
        torch_gate_weight=gate_weight,
        torch_gate_bias=gate_bias,
        torch_gate_proj_bias=proj_bias,
        matmul_compute_config=matmul_cc,
        **gp,
    )
    tt_x = ttnn.from_torch(
        hidden_states.reshape(1, 1, batch, hidden),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # --- golden (hidden -> logits -> gate -> (scores[batch,k], indices[batch,k])) ---
    gold_scores, gold_idx = TTMoEGate.golden(
        hidden_states, gate_weight, gate_bias, select_experts_k=k, gate_proj_bias=proj_bias, **gp
    )

    tt_scores, tt_indices = gate.forward(tt_x)
    dev_scores = ttnn.to_torch(tt_scores).reshape(batch, k).float()
    dev_idx = ttnn.to_torch(tt_indices).reshape(batch, k).to(torch.int64)

    # --- verify (torch fp32 golden; the device matmul runs HiFi2 + fp32 accumulation, so its residual
    #     bf16 noise only nudges rank-boundary experts — the checks below tolerate that) ---
    logits = hidden_states.float() @ gate_weight.float()
    if proj_bias is not None:  # gpt-oss router linear bias: logits = Wx + b (into selection AND weights)
        logits = logits + proj_bias.float()
    bias = gate_bias.float() if gate_bias is not None else torch.zeros(num_experts)
    # the per-expert score the op ranks/weights with, per score_func (softmax ranks by the raw logit):
    if gp["score_func"] == "sigmoid":
        score = torch.sigmoid(logits)
    elif gp["score_func"] == "sqrtsoftplus":
        score = torch.sqrt(torch.nn.functional.softplus(logits))
    else:  # softmax
        score = logits
    gold_idx = gold_idx.to(torch.int64)
    logger.info(f"dev_idx=\n{dev_idx}\ngold_idx=\n{gold_idx}")
    assert dev_idx.min() >= 0 and dev_idx.max() < num_experts, f"out-of-range expert id:\n{dev_idx}"

    # (1) PRIMARY check — score self-consistency: the device's output weights are the correct
    #     (softmax/linear) normalization of the score at the experts IT selected. Tie-robust (uses
    #     dev's OWN selection), so it validates the module wiring + normalize regardless of any
    #     selection ambiguity. (bf16-at-scale: exp/softmax over large logits amplifies bf16 noise → loose.)
    dev_sel = torch.gather(score, -1, dev_idx)
    weights = torch.exp(dev_sel) if gp["score_func"] == "softmax" else dev_sel  # softmax→exp-over-selected; else linear
    expected = weights / (weights.sum(-1, keepdim=True) + 1e-20) * gp["scaling_factor"]
    assert torch.allclose(
        dev_scores.sort(-1).values, expected.sort(-1).values, atol=5e-2
    ), f"gate scores not consistent with the device's own selection.\n dev={dev_scores}\n expected={expected}"

    # (2) SELECTION vs golden.
    if gp["n_group"] == 1:
        # ungrouped: dev's selected experts form a valid global top-k (ranking-key multiset matches
        # golden). key = score + bias for every score_func (softmax has bias=0, so key=logit there;
        # sigmoid→sigmoid+bias; sqrtsoftplus→sqrt(softplus)+bias). bf16-at-logit-scale noise only swaps
        # rank-(k-1)/k boundary experts; a real mis-selection is off by >>0.05.
        key = score + bias
        dev_key = torch.gather(key, -1, dev_idx).sort(-1).values
        gold_key = torch.gather(key, -1, gold_idx).sort(-1).values
        assert torch.allclose(dev_key, gold_key, atol=5e-2), (
            f"gate selection not a valid top-{k}.\n dev_idx={dev_idx}\n gold_idx={gold_idx}\n"
            f" dev_key={dev_key}\n gold_key={gold_key}"
        )
    else:
        # grouped (deepseek, n_group=8): 8 groups of 32 → top-2-sum per group → top-4 groups → top-8.
        # test_moe_gate.py-style: sort weights desc, gather indices to the same order, PCC the sorted
        # weights + position-wise index accuracy. The remaining ~1% selection diff (and the lower position
        # accuracy) is genuine bf16 ties at the top-8 boundary — the swapped experts have near-equal
        # weights, so PCC/overlap stay high while exact index positions shift.
        ref_sorted_w, ref_si = torch.sort(gold_scores.float(), dim=-1, descending=True, stable=True)
        ref_sorted_i = torch.gather(gold_idx, -1, ref_si)
        tt_sorted_w, tt_si = torch.sort(dev_scores, dim=-1, descending=True, stable=True)
        tt_sorted_i = torch.gather(dev_idx, -1, tt_si)
        pcc_ok, pcc_msg = comp_pcc(ref_sorted_w, tt_sorted_w, 0.97)
        accuracy = tt_sorted_i.eq(ref_sorted_i).float().mean().item()
        overlap = torch.stack([torch.isin(dev_idx[b], gold_idx[b]).float().mean() for b in range(batch)]).mean().item()
        logger.info(f"grouped: weights {pcc_msg} | index accuracy={accuracy:.3f} | mean overlap={overlap:.3f}")
        # Regression guards: a wrong grouping/wiring tanks both (the 16×16-vs-8×32 golden bug gave PCC 0.87 /
        # overlap 0.71); a correct 8×32 grouped op lands ~0.99 even on random data. Index *position*
        # accuracy is left as a log (boundary ties move positions without being a bug).
        assert pcc_ok, f"grouped weights PCC below 0.97 — likely a grouping/wiring bug: {pcc_msg}"
        assert overlap >= 0.9, f"grouped selection overlaps golden only {overlap:.3f} (< 0.9) — likely a wiring bug"

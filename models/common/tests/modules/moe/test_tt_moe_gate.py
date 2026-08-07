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

Coverage: every model YAML in ``configs/`` runs. n_group=1 spans the kernel op (k∈{4,6,8}, ≤512 experts:
64/128/160 pad-to-256, 256 single-face, 384/512 2-block combine) and the ttnn fallback (any other k, e.g.
512-experts top-10); n_group=8 is the deepseek grouped op (256 select-8). The only skips are configs
``TTMoEGate`` doesn't wire — n_group∉{1,8}, or n_group=8 not at exactly 256/k8 (mirrors __init__'s guards).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from loguru import logger

import ttnn
from models.common.modules.moe.tt_moe_gate import TTMoEGate
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig
from tests.ttnn.utils_for_testing import comp_pcc

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "modules/moe/configs"
CONFIG_PATHS = sorted(CONFIGS_DIR.glob("*.yaml"))
assert CONFIG_PATHS, f"no YAML configs found in {CONFIGS_DIR}"


def _config_id(path: Path) -> str:
    return path.stem


@pytest.mark.parametrize(
    # 4×8 = 32-chip mesh (TG/Galaxy). TTMoEGate is a PER-DEVICE gate (no cross-chip comms): its weight +
    # op buffers replicate to every chip, so each chip runs the same one-token-per-core routing independently.
    # The mesh_device fixture auto-skips when fewer chips are available. (More shapes can be added here in the
    # pytest.param form used by test_tt_moe_decode.py.)
    "mesh_device",
    [pytest.param((4, 8), id="4x8")],
    indirect=True,
)
@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=_config_id)
@pytest.mark.parametrize("seed", [42])
def test_tt_moe_gate(mesh_device, config_path: Path, seed: int):
    yaml_text = config_path.read_text()
    gate_config = TTMoEGateConfig.from_yaml(yaml_text)
    raw = yaml.safe_load(yaml_text)  # only for gate_notes (a doc field, not part of the config)

    num_experts = gate_config.num_routed_experts
    k = gate_config.select_experts_k
    hidden = gate_config.hidden_size
    n_group = gate_config.n_group
    score_func = gate_config.score_func
    scaling = gate_config.routed_scaling_factor

    # Skip only what TTMoEGate genuinely can't build (mirrors its __init__ guards):
    #   • n_group ∈ {1, 8} only.
    #   • n_group=8 = the deepseek grouped op, HARDWIRED to 256 experts select-8 (8 groups × 32 → top-8) — so
    #     EXACTLY N==256, k==8 (not "≤256": a smaller N has ≠32 experts/group, which the kernel can't do).
    #   • n_group=1 has NO expert ceiling: k∈{4,6,8} & N≤512 → kernel op; any other k OR N>512 → ttnn fallback.
    if n_group not in (1, 8):
        pytest.skip(f"only n_group 1 (generalized/ungrouped) / 8 (deepseek grouped) are wired; got n_group={n_group}")
    if n_group == 8 and (num_experts != 256 or k != 8):
        pytest.skip(f"n_group=8 (deepseek grouped op) is hardwired to 256 experts select-8; got N={num_experts}, k={k}")

    batch = gate_config.batch_per_device or 32  # PER-DEVICE batch (one token per core); replicated to every chip
    logger.info(
        f"[{config_path.stem}] gate: N={num_experts} k={k} hidden={hidden} batch={batch} "
        f"score_func={score_func} mesh={tuple(mesh_device.shape)}"
    )
    # per-model caveats about steps TTMoEGate does NOT cover (e.g. gemma4's RMSNorm + per-dim input scale
    # and per-expert output scale) — the caller must apply these around TTMoEGate.
    if raw.get("gate_notes"):
        logger.warning(f"[{config_path.stem}] gate_notes: {raw['gate_notes']}")

    # --- torch inputs ---
    torch.manual_seed(seed)
    hidden_states = (2 * torch.rand((batch, hidden), dtype=torch.bfloat16)) - 1
    gate_weight = ((2 * torch.rand((hidden, num_experts), dtype=torch.bfloat16)) - 1) * 0.1
    # score-correction bias (deepseek/noaux_tc): present iff config.score_correction_bias (EXPLICIT per-model).
    # Added to scores for SELECTION only (output weights stay unbiased). None → TTMoEGate feeds the op a zeros bias.
    gate_bias = (2 * torch.rand((num_experts,)) - 1) if gate_config.score_correction_bias else None
    # router LINEAR bias (gpt-oss, config.gate_proj_bias): logits = Wx + b, flows into selection + weights.
    proj_bias = (2 * torch.rand((num_experts,)) - 1) if gate_config.gate_proj_bias else None

    # --- device module + inputs (config-driven entry point, mirrors TTMoEDecode) ---
    gate = TTMoEGate(
        mesh_device,
        gate_config,
        torch_gate_weight=gate_weight,
        torch_gate_bias=gate_bias,
        torch_gate_proj_bias=proj_bias,
    )
    # replicate the hidden states to every chip — matches TTMoEGate's replicated weight/buffers, so each chip
    # routes the same batch (ReplicateTensorToMesh is a no-op on a 1-chip mesh, so this also works at 1×1).
    tt_x = ttnn.from_torch(
        hidden_states.reshape(1, 1, batch, hidden),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # --- golden (hidden -> logits -> gate -> (scores[batch,k], indices[batch,k])) ---
    gold_scores, gold_idx = TTMoEGate.golden(
        hidden_states,
        gate_weight,
        gate_bias,
        select_experts_k=k,
        score_func=score_func,
        softmax_position=gate_config.softmax_position,
        scaling_factor=scaling,
        eps=gate_config.eps,
        n_group=n_group,
        gate_proj_bias=proj_bias,
    )

    tt_scores, tt_indices = gate.forward(tt_x)
    # every chip computed the same routing (replicated inputs); concat over the mesh and keep the first chip.
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    dev_scores = ttnn.to_torch(tt_scores, mesh_composer=composer).reshape(-1, k)[:batch].float()
    dev_idx = ttnn.to_torch(tt_indices, mesh_composer=composer).reshape(-1, k)[:batch].to(torch.int64)

    # --- verify (torch fp32 golden; the device matmul runs HiFi2 + fp32 accumulation, so its residual
    #     bf16 noise only nudges rank-boundary experts — the checks below tolerate that) ---
    logits = hidden_states.float() @ gate_weight.float()
    if proj_bias is not None:  # gpt-oss router linear bias: logits = Wx + b (into selection AND weights)
        logits = logits + proj_bias.float()
    bias = gate_bias.float() if gate_bias is not None else torch.zeros(num_experts)
    # the per-expert score the op ranks/weights with, per score_func (softmax ranks by the raw logit):
    if score_func == "sigmoid":
        score = torch.sigmoid(logits)
    elif score_func == "sqrtsoftplus":
        score = torch.sqrt(torch.nn.functional.softplus(logits))
    else:  # softmax
        score = logits
    gold_idx = gold_idx.to(torch.int64)
    logger.info(f"dev_idx=\n{dev_idx}\ngold_idx=\n{gold_idx}")
    assert dev_idx.min() >= 0 and dev_idx.max() < num_experts, f"out-of-range expert id:\n{dev_idx}"

    # (1) PRIMARY check — score self-consistency: the device's output weights are the correct
    #     (softmax/linear) normalization of the score at the experts IT selected. Tie-robust (uses
    #     dev's OWN selection), so it validates the module wiring + normalize regardless of any
    #     selection ambiguity.
    dev_sel = torch.gather(score, -1, dev_idx)
    weights = torch.exp(dev_sel) if score_func == "softmax" else dev_sel  # softmax→exp-over-selected; else linear
    expected = weights / (weights.sum(-1, keepdim=True) + 1e-20) * scaling
    # Kernel-op configs hold at the tight 1e-2. The pure-ttnn FALLBACK (n_group=1 with k∉{4,6,8} or N>512 —
    # today only qwen35_397b, and the one path with no op-level test behind it) runs matmul→topk→softmax,
    # whose bf16 noise the softmax exp-over-logits amplifies, so it alone needs 5e-2. Predicate mirrors
    # TTMoEGate.use_fallback (tt_moe_gate.py).
    use_fallback = n_group == 1 and (k not in (4, 6, 8) or num_experts > 512)
    score_atol = 5e-2 if use_fallback else 1e-2
    assert torch.allclose(
        dev_scores.sort(-1).values, expected.sort(-1).values, atol=score_atol
    ), f"gate scores not consistent with the device's own selection.\n dev={dev_scores}\n expected={expected}"

    # (2) SELECTION vs golden.
    if n_group == 1:
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
        pcc_ok, pcc_msg = comp_pcc(ref_sorted_w, tt_sorted_w, 0.99)
        accuracy = tt_sorted_i.eq(ref_sorted_i).float().mean().item()
        overlap = torch.stack([torch.isin(dev_idx[b], gold_idx[b]).float().mean() for b in range(batch)]).mean().item()
        logger.info(f"grouped: weights {pcc_msg} | index accuracy={accuracy:.3f} | mean overlap={overlap:.3f}")
        # Regression guards: a wrong grouping/wiring tanks both (the 16×16-vs-8×32 golden bug gave PCC 0.87 /
        # overlap 0.71); a correct 8×32 grouped op lands ~0.99 even on random data. Index *position*
        # accuracy is left as a log (boundary ties move positions without being a bug).
        assert pcc_ok, f"grouped weights PCC below 0.99 — likely a grouping/wiring bug: {pcc_msg}"
        assert overlap >= 0.9, f"grouped selection overlaps golden only {overlap:.3f} (< 0.9) — likely a wiring bug"

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Prefill-shape perf probe for ttnn.experimental.deepseek_prefill.moe_grouped_topk.

One case is the MoE gate of one production model on ONE chip, called exactly as
TtMoEGatePrefill._device_grouped_gate_fp32 calls it: [PREFILL_CHUNK_TOKENS_PER_CHIP, n_experts]
bf16 logits in L1 (the all-reduce output), the per-expert correction bias broadcast to the same
shape in DRAM, a right-padding config, stable_sort=True and ROW_MAJOR weights.

The op is called ``warmup + measured`` times, each call on a fresh random input of the same shape;
the device kernel duration is read from the profiler by the caller. Accuracy vs an fp32 golden is
pooled over all calls (~8k tokens per case) and logged as a DREAM_METRIC line: ``miss_pct`` (% of
golden experts not selected), recall, exact sorted rows, weight PCC and max error. The asserts are
loose floors against a broken op.
"""

from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    build_padding_config,
    calculate_average_recall,
    grouped_gate_golden_act,
)
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP

_WARMUP_ITERS = 3
_MEASURE_ITERS = 10
_EPSILON = 1e-20


def _distinct_bf16_logits(seq_len: int, n_experts: int) -> torch.Tensor:
    """Tie-free bf16 logits: each row is a random subset of the distinct bf16 values in [-6, 6]
    (|x| >= 1/16), enough for Kimi-K3's 896 experts. Returned as fp32 holding exact bf16 values."""
    candidates = torch.linspace(-6.0, 6.0, 200000).to(torch.bfloat16).unique().float()
    candidates = candidates[candidates.abs() >= 2**-4]
    assert candidates.numel() >= n_experts
    return torch.stack([candidates[torch.randperm(candidates.numel())[:n_experts]] for _ in range(seq_len)])


@dataclass(frozen=True)
class GateCase:
    """One production model's gate. Every field is read off the model config."""

    model_cfg: type

    @property
    def n_experts(self) -> int:
        return self.model_cfg.NUM_ROUTED_EXPERTS

    @property
    def k(self) -> int:
        return self.model_cfg.NUM_EXPERTS_PER_TOKEN

    @property
    def n_groups(self) -> int:
        return self.model_cfg.NUM_EXPERT_GROUPS

    @property
    def topk_groups(self) -> int:
        return self.model_cfg.NUM_LIMITED_GROUPS

    @property
    def summed_experts_per_group(self) -> int:
        # Same expression as TtMoEGatePrefill._device_grouped_gate_fp32.
        return self.n_groups // self.topk_groups

    @property
    def route_scale(self) -> float:
        return self.model_cfg.ROUTE_SCALE

    @property
    def score_func(self) -> str:
        return getattr(self.model_cfg, "SCORE_FUNC", "sigmoid")


def _case(model_cfg: type, name: str):
    case = GateCase(model_cfg)
    return pytest.param(case, id=f"{name}-{case.n_groups}g{case.n_experts}e-k{case.k}")


_CASES = [
    _case(DeepSeekV3Config, "deepseek-v3"),
    _case(KimiK27Config, "kimi-k2-7"),
    _case(KimiK3Config, "kimi-k3"),
    _case(GLM53Config, "glm-5-3"),
    _case(DeepSeekV4FlashConfig, "dsv4-flash"),
    _case(DeepSeekV4ProConfig, "dsv4-pro"),
]


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("seq_len", [PREFILL_CHUNK_TOKENS_PER_CHIP], ids=lambda s: f"{s}tok")
def test_moe_grouped_topk_prefill_perf(device, case: GateCase, seq_len: int):
    torch.manual_seed(0)
    n_experts, k = case.n_experts, case.k
    expert_bias = (torch.randn(n_experts) * 0.1).to(torch.bfloat16).float()
    bias = expert_bias.repeat(seq_len).view(seq_len, n_experts)
    tt_bias = ttnn.from_torch(
        bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    padding_config = build_padding_config(device, seq_len)

    all_tt_idx, all_tt_w, all_ref_idx, all_ref_w = [], [], [], []
    for _ in range(_WARMUP_ITERS + _MEASURE_ITERS):
        # bf16-exact inputs so the fp32 golden sees exactly what the device sees (the op upcasts).
        logits = _distinct_bf16_logits(seq_len, n_experts)
        ref_indices, ref_weights = grouped_gate_golden_act(
            logits.view(1, 1, seq_len, n_experts),
            bias.view(1, 1, seq_len, n_experts),
            case.route_scale,
            _EPSILON,
            case.n_groups,
            case.summed_experts_per_group,
            case.topk_groups,
            k,
            case.score_func,
            stable=True,
        )
        tt_logits = ttnn.from_torch(
            logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        tt_weights, tt_indices = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            tt_logits,
            tt_bias,
            n_groups=case.n_groups,
            summed_experts_per_group=case.summed_experts_per_group,
            topk_groups=case.topk_groups,
            n_activated_experts=k,
            route_scale=case.route_scale,
            stable_sort=True,
            epsilon=_EPSILON,
            score_func=case.score_func,
            padding_config=padding_config,
            weights_layout=ttnn.ROW_MAJOR_LAYOUT if k <= 32 else ttnn.TILE_LAYOUT,
        )
        all_tt_w.append(ttnn.to_torch(tt_weights).float().reshape(-1, tt_weights.shape[-1])[:seq_len, :k])
        all_tt_idx.append(ttnn.to_torch(tt_indices).long().reshape(-1, tt_indices.shape[-1])[:seq_len, :k])
        all_ref_idx.append(ref_indices.reshape(-1, k).long())
        all_ref_w.append(ref_weights.reshape(-1, k).float())
        for t in (tt_logits, tt_weights, tt_indices):
            ttnn.deallocate(t)

    tt_indices, tt_weights = torch.cat(all_tt_idx), torch.cat(all_tt_w)
    ref_indices, ref_weights = torch.cat(all_ref_idx), torch.cat(all_ref_w)
    tokens = tt_indices.shape[0]

    recall = calculate_average_recall(tt_indices, ref_indices)
    miss_pct = 100.0 * (1.0 - recall)
    exact_rows = (tt_indices == ref_indices).all(dim=-1).float().mean().item()
    _, weights_pcc = comp_pcc(ref_weights, tt_weights)
    weights_max_abs = (ref_weights - tt_weights).abs().max().item()
    # Weight accuracy alone: rows whose expert SET matches, with weights aligned by expert id.
    tt_sorted_idx, tt_perm = tt_indices.sort(dim=-1)
    ref_sorted_idx, ref_perm = ref_indices.sort(dim=-1)
    same_set = (tt_sorted_idx == ref_sorted_idx).all(dim=-1)
    set_weights_max_abs = (
        (tt_weights.gather(-1, tt_perm)[same_set] - ref_weights.gather(-1, ref_perm)[same_set]).abs().max().item()
    )

    logger.info(
        f"moe_grouped_topk prefill probe model={case.model_cfg.__name__} seq_len={seq_len} tokens={tokens} "
        f"experts={n_experts} groups={case.n_groups} topk_groups={case.topk_groups} k={k} "
        f"score_func={case.score_func} miss_pct={miss_pct:.6f} recall={recall:.6f} exact_rows={exact_rows:.6f} "
        f"weights_pcc={weights_pcc:.6f} weights_max_abs={weights_max_abs:.6f} "
        f"set_weights_max_abs={set_weights_max_abs:.6f}"
    )
    print(
        f"DREAM_METRIC miss_pct={miss_pct:.6f} recall={recall:.6f} exact_rows={exact_rows:.6f} "
        f"weights_pcc={weights_pcc:.6f} weights_max_abs={weights_max_abs:.6f} "
        f"set_weights_max_abs={set_weights_max_abs:.6f}",
        flush=True,
    )

    # Loose floors that only catch a broken op; the campaign scores and gates accuracy itself.
    assert recall >= 0.98, f"recall {recall:.6f} < 0.98"
    assert weights_pcc >= 0.9, f"weights PCC {weights_pcc:.6f} < 0.9"

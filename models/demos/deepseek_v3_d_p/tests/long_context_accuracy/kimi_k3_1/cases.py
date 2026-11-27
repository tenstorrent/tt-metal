# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 long-context accuracy cases (``block.LongContextCase``) against the FP64 chain golden ``k31-l0-3-fp64-1m``
(reader ``golden.KimiK31ChainGolden``), run by ``test_long_context_accuracy.py``: MoE layers 1 and 2 (``moe_case``) and
DSA layer 3 (``DSA_CASE``).

Every case runs its production device Block (``blocks.py``) on one 4x4 K3.1 stage (``K31_PLACEMENT``, a Galaxy tray
half) in the stage's ``K31_CHUNK``-row chunks, teacher-fed from the chain's products. Identity: the digest of the
layer's non-routed weights as the chain read them (``golden.layer_weights``) from ``k31_checkpoint()`` must be the
golden's ``layers.L``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar
from models.demos.common.prefill.long_context_accuracy.block import (
    ContextBars,
    GoldenIdentity,
    LongContextCase,
    PreparedCase,
)
from models.demos.common.prefill.long_context_accuracy.dsa.fp64_attention import WEIGHTS, Dims
from models.demos.common.prefill.long_context_accuracy.dsa.scorer import DsaScorer, GoldenNames
from models.demos.common.prefill.long_context_accuracy.fp64_golden import Fp64Golden
from models.demos.common.prefill.long_context_accuracy.moe.scorer import MoeScorer
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.blocks import KimiK31DsaBlock, KimiK31MoeBlock
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden import (
    DSA_LAYER,
    K31_CHAIN_1M,
    MODEL,
    KimiK31ChainGolden,
    layer_weights,
)

K31_PLACEMENT = next(p for p in k31_stage.PLACEMENT if p.id == "4x4-sp1")
K31_CHUNK = k31_stage.CHUNK_4X4


def k31_checkpoint() -> Path:
    """The K3.1 checkpoint the cases digest and the MoE Block reads its router, shared expert and latent pair from
    (``moe_block_checkpoint``: ``$KIMI_K31_HF_MODEL``, then ``$PREFILL_HF_MODEL``, then the adapter's default); skips
    without one."""
    path = Path(k31_stage.ADAPTER.moe_block_checkpoint())
    if not (path / "model.safetensors.index.json").is_file():
        pytest.skip(f"no K3.1 checkpoint at {path} (set ${k31_stage.ADAPTER.env_var})")
    return path


# MoE layers 1 and 2: the production K3.1 MoE (``blocks.KimiK31MoeBlock``) fed the chain's ``moe_input_L`` (after
# ``post_attention_layernorm``, rounded once to bf16) and scored by ``MoeScorer`` against ``moe_output_L`` and the FP64
# top-16 ``routing_ids_L`` / ``routing_weights_L``. The routed experts come from the TTNN weight cache and stay
# unchecked: the chain's ``layers.L.experts`` digest covers the checkpoint's experts, not the cache.
#
# Bars (``MOE_BARS``), on every 16K window of the run (none per chunk: a chunk's 2560 rows are a smaller sample, so
# the window values would stop runs whose windows pass; a chunk stops only on a broken contract): the study's MoE
# teacher-feed bars (``moe_teacher_feed.BARS``, branch ``mvasilijevic/k31/accuracy-harness-consumers``), proposed there
# against the vLLM trace over 9 layers at 128K. Its growth bar is left out: the MoE carries no state across chunks (a
# reverse-order run is bit-identical), so its error versus position follows the input statistics, not accumulation; it
# is gated by the absolute per-window bars instead. Measured at 1M (commit 9bf2a291b5b, one 4x4 tray half), layer 1 /
# layer 2: pooled NRMSE 0.088 / 0.078, median row NRMSE 0.083 / 0.071, mean recall 0.994 / 0.992 (worst 16K window
# 0.9939 / 0.9916), mean dropped weight 0.0038 / 0.0053 (worst window 0.0041 / 0.0059), no missed expert in the top
# half; the RMS error's worst window is 1.02x / 1.03x its first.
MOE_WINDOW = 16384  # rows per scored window, aligned to row 0 (a shorter run is one partial window)
MOE_BARS = (
    AcceptanceBar("moe_row_nrmse_median", "max", 0.20),
    AcceptanceBar("recall_mean", "min", 0.98),
    AcceptanceBar("miss_top_half_fraction", "max", 0.01),
    AcceptanceBar("dropped_weight_mean", "max", 0.01),
)
# Context lengths in rows, the last the golden's whole prompt (408 chunks); 256k ends in a partial chunk. Measured on
# one tray half: 1M ~4 min a layer (~50 s weight load, then ~0.45 s a chunk), 256k ~100 s, 5k-20k ~15 s warm.
MOE_CONTEXT_LENGTHS = (5120, 20480, 262144, 1044480)


def _prepare_moe(layer: int, golden: Fp64Golden, checkpoint: Path, context_len: int) -> PreparedCase:
    _, digest = layer_weights(checkpoint, layer)  # (the tensors are only digested; the Block reads its own)
    moe_input = f"moe_input_{layer}"
    return PreparedCase(
        identity=GoldenIdentity({"model": MODEL}, digest),
        scorer=MoeScorer(
            output=f"moe_output_{layer}",
            expert_ids=f"routing_ids_{layer}",
            expert_weights=f"routing_weights_{layer}",
            window=MOE_WINDOW,
        ),
        input_tensors=(moe_input,),
        block_input=lambda lo, hi, rows: rows[moe_input].to(torch.bfloat16),
        device_block=KimiK31MoeBlock(layer),
    )


def moe_case(layer: int) -> LongContextCase:
    return LongContextCase(
        id=f"k31-moe-layer{layer}",
        golden_id=K31_CHAIN_1M,
        golden_reader=KimiK31ChainGolden,
        placement=K31_PLACEMENT,
        chunk=K31_CHUNK,
        bars={n: ContextBars(run=MOE_BARS) for n in MOE_CONTEXT_LENGTHS},
        checkpoint=k31_checkpoint,
        prepare=lambda golden, checkpoint, context_len: _prepare_moe(layer, golden, checkpoint, context_len),
    )


# DSA layer 3: ``ttMLA`` (``blocks.KimiK31DsaBlock``) fed the FP64 ``input_layernorm`` of the chain's ``dsa_input_3``
# rounded once to bf16, and scored by ``DsaScorer`` against the chain's FP64 selection and keys and an FP64 attention
# recompute (``dsa/fp64_attention.py``, from the layer's checkpoint weights).
DSA_DIMS = Dims(
    heads=KimiK31Config.NUM_ATTENTION_HEADS,
    nope=KimiK31Config.QK_NOPE_HEAD_DIM,
    rope=KimiK31Config.QK_ROPE_HEAD_DIM,
    v=KimiK31Config.V_HEAD_DIM,
    lora=KimiK31Config.KV_LORA_RANK,
    topk=KimiK31Config.INDEX_TOPK,
    eps=KimiK31Config.RMS_NORM_EPS,  # (the golden's model.eps)
)
DSA_GOLDEN_NAMES = GoldenNames(
    input=f"dsa_input_{DSA_LAYER}",
    latent="latent",
    kpe="kpe",
    index_key="index_key",
    selection=f"topk_indices_{DSA_LAYER}",
    selection_scores=f"topk_scores_{DSA_LAYER}",
    sampled_rows="sampled_rows",
    sampled_output=f"attn_output_{DSA_LAYER}",
)

# Per-window maxima of the gated metrics (metric tier 1), per context length (rows; each a multiple of the 2560-row
# chunk, so none ends in a zero-padded chunk). PROPOSED, pending the owner's decision (epic m06n): 1.25x the worst
# window measured on main's production DSA (4x4-sp1), rounded up at two significant figures. Measured worst windows:
#   5k:   sel_nrmse_median 0.00518, attn_nrmse 0.0280, attn_rms_err 5.68e-4 (one window, 49 scored rows)
#   128k: sel_nrmse_median 0.0299,  attn_nrmse 0.0476, attn_rms_err 9.69e-4 (rows [98304, 114688) / [114688, 128000))
#   1M:   sel_nrmse_median 0.0299,  attn_nrmse 0.0461, attn_rms_err 1.20e-3 (rows [98304, ...), [114688, ...),
#         [999424, ...)); the absolute error grows 1.8x over 1M (reference RMS 1.33x): DSA's own selection-driven
#         growth, accepted by these ceilings, not gated by a growth bar.
# Reruns are bit-identical on full windows; choosing other scored rows moved a window's metrics by 3-4%. The gate is
# the selection NRMSE (m06n.11.7); missed-key max is a metric tier 2 diagnostic. No future key is a per-chunk contract
# (DsaScorer.score_chunk).
DSA_BARS: dict[int, dict[str, float]] = {
    5120: {"sel_nrmse_median": 0.0065, "attn_nrmse": 0.036, "attn_rms_err": 7.1e-4},
    128000: {"sel_nrmse_median": 0.038, "attn_nrmse": 0.060, "attn_rms_err": 1.3e-3},
    1044480: {"sel_nrmse_median": 0.038, "attn_nrmse": 0.058, "attn_rms_err": 1.6e-3},
}


def dsa_context_bars(maxima: Mapping[str, float]) -> ContextBars:
    """Run bars of one context length: each metric at most its value in every window. A non-finite value (a bar not
    yet set) raises: ``check`` would pass it, since no comparison with NaN fails. ``DSA_CASE`` builds its bars at
    import, so an unset bar fails the collection of every registered case."""
    unset = [metric for metric, value in maxima.items() if not math.isfinite(value)]
    if unset:
        raise ValueError(f"no bar value for {unset}")
    return ContextBars(run=tuple(AcceptanceBar(metric, "max", value) for metric, value in maxima.items()))


def _prepare_dsa(golden: Fp64Golden, checkpoint: Path, context_len: int) -> PreparedCase:
    """Layer 3's FP64 attention weights and the digest of the tensor set the chain read for the layer; the scorer,
    which reads the golden's sampled rows."""
    tensors, digest = layer_weights(checkpoint, DSA_LAYER)
    weights = {n: tensors[n].double() for n in WEIGHTS}
    scorer = DsaScorer(golden, context_len, DSA_GOLDEN_NAMES, weights, DSA_DIMS)
    return PreparedCase(
        identity=GoldenIdentity({"model": MODEL}, digest),
        scorer=scorer,
        input_tensors=(DSA_GOLDEN_NAMES.input,),
        block_input=lambda lo, hi, rows: scorer.device_input(rows),
        device_block=KimiK31DsaBlock(DSA_LAYER),
    )


DSA_CASE = LongContextCase(
    id=f"k31-l{DSA_LAYER}-dsa",
    golden_id=K31_CHAIN_1M,
    golden_reader=KimiK31ChainGolden,
    placement=K31_PLACEMENT,
    chunk=K31_CHUNK,
    bars={rows: dsa_context_bars(maxima) for rows, maxima in DSA_BARS.items()},
    checkpoint=k31_checkpoint,
    prepare=_prepare_dsa,
)

MOE_CASES: tuple[LongContextCase, ...] = (moe_case(1), moe_case(2))
CASES: tuple[LongContextCase, ...] = (*MOE_CASES, DSA_CASE)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 long-context accuracy cases (``block.LongContextCase``) against the FP64 chain golden ``k31-l0-3-fp64-1m``
(reader ``golden.KimiK31ChainGolden``), run by ``test_long_context_accuracy.py`` and checked against the golden by
``test_cases_match_goldens.py``: the MoE layer 1 and 2 cases (m06n.11.4) and the DSA layer 3 case (m06n.11.5).

Each block's cases, including their imports, stay inside their own section below.
"""

from __future__ import annotations

from models.demos.common.prefill.long_context_accuracy.block import LongContextCase

# ==== MoE (m06n.11.4) ===============================================================================================
MOE_CASES: tuple[LongContextCase, ...] = ()
# ==== end of MoE ====================================================================================================

# ==== DSA (m06n.11.5) ===============================================================================================
import math
import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar
from models.demos.common.prefill.long_context_accuracy.block import ContextBars, GoldenIdentity, PreparedCase
from models.demos.common.prefill.long_context_accuracy.dsa.fp64_attention import WEIGHTS, Dims
from models.demos.common.prefill.long_context_accuracy.dsa.scorer import DsaScorer, GoldenNames
from models.demos.common.prefill.long_context_accuracy.fp64_golden import Fp64Golden
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import k31_stage
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.blocks import KimiK31DsaBlock
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden import (
    DSA_LAYER,
    K31_CHAIN_1M,
    MODEL,
    KimiK31ChainGolden,
    layer_weights,
)

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
# (DsaScorer.check_chunk).
DSA_BARS: dict[int, dict[str, float]] = {
    5120: {"sel_nrmse_median": 0.0065, "attn_nrmse": 0.036, "attn_rms_err": 7.1e-4},
    128000: {"sel_nrmse_median": 0.038, "attn_nrmse": 0.060, "attn_rms_err": 1.3e-3},
    1044480: {"sel_nrmse_median": 0.038, "attn_nrmse": 0.058, "attn_rms_err": 1.6e-3},
}


def dsa_context_bars(maxima: Mapping[str, float]) -> ContextBars:
    """Run bars of one context length: each metric at most its value in every window. A non-finite value (a bar not
    yet set) raises: ``check`` would pass it, since no comparison with NaN fails."""
    unset = [metric for metric, value in maxima.items() if not math.isfinite(value)]
    if unset:
        raise ValueError(f"no bar value for {unset}")
    return ContextBars(run=tuple(AcceptanceBar(metric, "max", value) for metric, value in maxima.items()))


def dsa_checkpoint() -> Path:
    """The K3.1 checkpoint (``$KIMI_K31_HF_MODEL`` or the adapter's default); skips without one."""
    adapter = k31_stage.ADAPTER
    path = Path(os.environ.get(adapter.env_var) or adapter.hf_model_path)
    if not (path / "model.safetensors.index.json").is_file():
        pytest.skip(f"no K3.1 checkpoint at {path} (set ${adapter.env_var})")
    return path


def prepare_dsa(golden: Fp64Golden, checkpoint: Path, context_len: int) -> PreparedCase:
    """Layer 3's FP64 attention weights and the digest of the tensor set the chain read for the layer; the scorer,
    which reads the golden's sampled rows."""
    tensors, digest = layer_weights(checkpoint, DSA_LAYER)
    weights = {n: tensors[n].double() for n in WEIGHTS}
    scorer = DsaScorer(golden, context_len, DSA_GOLDEN_NAMES, weights, DSA_DIMS)
    return PreparedCase(
        identity=GoldenIdentity(identity={"model": MODEL}, weights_sha256=digest),
        scorer=scorer,
        input_tensors=(DSA_GOLDEN_NAMES.input,),
        block_input=lambda lo, hi, rows: scorer.device_input(rows),
        device_block=KimiK31DsaBlock(DSA_LAYER),
    )


DSA_CASES: tuple[LongContextCase, ...] = (
    LongContextCase(
        id=f"k31-l{DSA_LAYER}-dsa",
        golden_id=K31_CHAIN_1M,
        golden_reader=KimiK31ChainGolden,
        placement=next(p for p in k31_stage.PLACEMENT if p.id == "4x4-sp1"),
        chunk=k31_stage.CHUNK_4X4,
        bars={rows: dsa_context_bars(maxima) for rows, maxima in DSA_BARS.items()},
        checkpoint=dsa_checkpoint,
        prepare=prepare_dsa,
    ),
)
# ==== end of DSA ====================================================================================================

CASES = MOE_CASES + DSA_CASES

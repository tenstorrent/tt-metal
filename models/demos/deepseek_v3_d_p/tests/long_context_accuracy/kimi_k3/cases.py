# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 long-context accuracy cases (``block.LongContextCase``), run by ``test_long_context_accuracy.py``.

``KDA_LAYER0``: the KDA layer of ``model.layers.0`` (device Block ``blocks.KimiK3KdaBlock``, scorer
``common/prefill/long_context_accuracy/kda/scorer.KdaScorer``) on a full Galaxy (SP8 x TP4, the 2D fabric KDA tests
open on 8x4), teacher-fed the real code tokens of golden ``kimi_k3_code_debug_1m_cpu_fp64_l0_kda`` (reader
``golden.KimiK3KdaGolden``) in production-size 5,120-row prefill calls. The input is built from the token ids with the
checkpoint's embedding and input norm (``FirstLayerInput``), and the weights and input are checked against the
golden's digests before any device time. Needs ``KIMI_K3_CKPT`` (layers.0 and the embedding shards); skips without it.

Bars per context length, NRMSE against the golden: per call (with PCC 0.999), over the run, per 1,024-row window
(drift with context), per row (the golden's floor), and for the final recurrent state per head (one drifting head is
not averaged away; its PCC bar is looser, since the worst head drifts with context).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar
from models.demos.common.prefill.long_context_accuracy.block import (
    ContextBars,
    GoldenIdentity,
    LongContextCase,
    PreparedCase,
)
from models.demos.common.prefill.long_context_accuracy.kda.scorer import KdaScorer
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256, load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.first_layer_input import FIRST_LAYER_INPUT, FirstLayerInput
from models.demos.deepseek_v3_d_p.tests.kda.utils import kda_device_params
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.blocks import KimiK3KdaBlock
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden import KimiK3KdaGolden
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3.golden_producer import K3_LAYER0_GOLDEN
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS

_TP_AXIS = 1
_CALL_PCC = 0.999
# PCC is the structural sanity gate; NRMSE carries precision. The worst head's state drifts with context on main
# (PCC 0.9975 and NRMSE 0.16 after 1M rows), so the state's per-head PCC bound is looser than the output's.
_STATE_PCC = 0.99


def _bars(call: float, output: float, window: float, token: float, state_head: float) -> ContextBars:
    return ContextBars(
        chunk=(AcceptanceBar("call_pcc", "min", _CALL_PCC), AcceptanceBar("call_nrmse", "max", call)),
        run=(
            AcceptanceBar("output_nrmse", "max", output),
            AcceptanceBar("window_nrmse", "max", window),
            AcceptanceBar("output_token_worst", "max", token),
            AcceptanceBar("state_head_final_worst", "max", state_head),
            AcceptanceBar("state_head_final_pcc_min", "min", _STATE_PCC),
        ),
    )


def _checkpoint() -> Path:
    value = os.getenv("KIMI_K3_CKPT")
    if value is None:
        pytest.skip("set KIMI_K3_CKPT to the pinned Kimi-K3 checkpoint subset")
    return Path(value)


def _prepare_kda_layer0(golden: KimiK3KdaGolden, checkpoint: Path, context_len: int) -> PreparedCase:
    config = kimi_k3_kda_config()
    weights = load_kda_layer_state_dict(checkpoint, 0, config)
    weights_sha256 = kda_state_dict_sha256(weights)
    build_input = FirstLayerInput(checkpoint)
    token_ids = golden.token_ids()
    identity = {
        "model": "Kimi-K3",
        "layer": "model.layers.0",
        "input": FIRST_LAYER_INPUT,
        "weights_sha256": weights_sha256,
        "input_weights_sha256": build_input.weights_sha256(),
    }
    return PreparedCase(
        identity=GoldenIdentity(identity=identity),
        scorer=KdaScorer(golden.manifest["token_floor"]),
        input_tensors=(),
        block_input=lambda lo, hi, rows: build_input(token_ids[lo:hi])[0],
        device_block=KimiK3KdaBlock(config, weights, weights_sha256, _TP_AXIS),
    )


KDA_LAYER0 = LongContextCase(
    id="k3-l0-kda",
    golden_id=K3_LAYER0_GOLDEN,
    golden_reader=KimiK3KdaGolden,
    placement=pytest.param((8, 4), kda_device_params((8, 4)), id="SP8xTP4"),
    chunk=PREFILL_CHUNK_TOKENS,
    bars={
        5120: _bars(call=0.015, output=0.015, window=0.02, token=0.03, state_head=0.06),
        20480: _bars(call=0.02, output=0.02, window=0.02, token=0.03, state_head=0.15),
        1044480: _bars(call=0.025, output=0.02, window=0.03, token=0.14, state_head=0.2),
    },
    checkpoint=_checkpoint,
    prepare=_prepare_kda_layer0,
)

CASES: tuple[LongContextCase, ...] = (KDA_LAYER0,)

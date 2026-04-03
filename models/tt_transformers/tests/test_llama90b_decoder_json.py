# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Llama 3.2 90B decoder JSON must match ModelOptimizations baseline except LI_QKV_DECODE (issue #36378)."""
from pathlib import Path

import pytest

from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelOptimizations,
    OpGroup,
    TensorGroup,
    parse_decoder_json,
)

_MN = "Llama-3.2-90B-Instruct"
_TT_ROOT = Path(__file__).resolve().parents[1]
_ACC = _TT_ROOT / "model_params" / _MN / "accuracy_decoder_config.json"
_PERF = _TT_ROOT / "model_params" / _MN / "performance_decoder_config.json"


def _tensor_map(m: ModelOptimizations):
    return {
        k.name: m.tensor_dtype_settings[k].name
        for k in TensorGroup
        if k in m.tensor_dtype_settings and m.tensor_dtype_settings[k] is not None
    }


def _op_map(m: ModelOptimizations):
    return {k.name: m.op_fidelity_settings[k].name for k in OpGroup if k in m.op_fidelity_settings}


def _assert_json_matches_baseline_except_qkv_decode(json_path: Path, opt_fn):
    baseline = opt_fn(_MN)
    b_tensor = _tensor_map(baseline)
    b_op = _op_map(baseline)
    loaded = parse_decoder_json(json_path, default_optimization=opt_fn)
    n = len(loaded.decoder_optimizations)
    assert n == 80
    for di in range(n):
        j = loaded.decoder_optimizations[di]
        j_tensor = _tensor_map(j)
        j_op = _op_map(j)
        assert j_tensor == b_tensor, f"decoder {di} tensor_dtype mismatch"
        for ok, vb in b_op.items():
            jv = j_op.get(ok)
            if ok == OpGroup.LI_QKV_DECODE.name:
                assert (
                    vb == MathFidelitySetting.HIFI2.name and jv == MathFidelitySetting.HIFI2_NOL1ACC.name
                ), f"decoder {di} LI_QKV_DECODE expected HIFI2->HIFI2_NOL1ACC, got {vb}->{jv}"
            else:
                assert jv == vb, f"decoder {di} op {ok} baseline={vb} json={jv}"


@pytest.mark.skipif(not _ACC.is_file(), reason="accuracy_decoder_config.json not present")
def test_llama90b_accuracy_decoder_json_matches_baseline_except_qkv_decode():
    _assert_json_matches_baseline_except_qkv_decode(_ACC, ModelOptimizations.accuracy)


@pytest.mark.skipif(not _PERF.is_file(), reason="performance_decoder_config.json not present")
def test_llama90b_performance_decoder_json_matches_baseline_except_qkv_decode():
    _assert_json_matches_baseline_except_qkv_decode(_PERF, ModelOptimizations.performance)


def test_llama90b_decoders_precision_without_json_layers_match_accuracy_baseline():
    """When no JSON is used, DecodersPrecision repeats optimization_level(model_name) per layer."""
    base = ModelOptimizations.accuracy(_MN)
    dp = DecodersPrecision(80, _MN, base)
    for i in range(80):
        assert dp.decoder_optimizations[i]._full_name == base._full_name

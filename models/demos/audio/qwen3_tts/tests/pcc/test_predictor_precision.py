# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the code predictor's precision levels."""

import pytest
import torch

import ttnn
from models.demos.audio.qwen3_tts import weights as checkpoint
from models.demos.audio.qwen3_tts.reference.qwen import transformers_compat  # noqa: F401  (registers rope "default")
from models.demos.audio.qwen3_tts.reference.qwen3_code_predictor_ref import CodePredictorReference
from models.demos.audio.qwen3_tts.tests.reference_helpers import code_predictor_prompt
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor import CODE_PREDICTOR_PREFIX
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_code_predictor_decode import (
    PRECISION_LEVELS,
    TtCodePredictorCachedDecoder,
    predictor_mlp_dtype,
    preprocess_cached_predictor_parameters,
)
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker_decode import MLP_WEIGHT_DTYPE

DEVICE_PARAMS = [{"l1_small_size": 65536, "trace_region_size": 90_000_000}]

MLP = ("gate_proj", "up_proj", "down_proj")


@pytest.fixture(scope="module")
def frame():
    return code_predictor_prompt()


def _predictor(device, precision):
    parameters = preprocess_cached_predictor_parameters(device, mlp_dtype=predictor_mlp_dtype(precision))
    return TtCodePredictorCachedDecoder(device, parameters, precision=precision)


def _forced_logits(device, precision, frame, traced):
    talker_hidden, first_code, codes, _ = frame
    predictor = _predictor(device, precision)
    if traced:
        predictor.capture()
    rows, forced = [], iter(codes)
    predictor.generate(talker_hidden, first_code, pick=lambda row: (rows.append(row.double().clone()), next(forced))[1])
    if traced:
        predictor.release()
    return torch.stack(rows)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_levels_rank_by_precision(device, frame):
    reference = CodePredictorReference(dtype=torch.float64).teacher_forced(frame[3].double())[0]
    error, agree = {}, {}
    for precision in PRECISION_LEVELS:
        logits = _forced_logits(device, precision, frame, traced=False)
        error[precision] = float((logits - reference).norm() / reference.norm())
        agree[precision] = int((logits.argmax(-1) == reference.argmax(-1)).sum())
        print(f"{precision}: logits error {error[precision]:.3e}, top-1 agreement {agree[precision]}/{len(reference)}")
    assert error["accurate"] < 0.75 * error["main"]
    assert error["max"] < 0.75 * error["main"]


@pytest.mark.parametrize("precision", PRECISION_LEVELS)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_traced_level_replays_the_eager_level(device, frame, precision):
    eager = _forced_logits(device, precision, frame, traced=False)
    traced = _forced_logits(device, precision, frame, traced=True)
    assert torch.equal(eager, traced)


@pytest.mark.parametrize("precision", PRECISION_LEVELS)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_levels_load_the_checkpoint_mlp_weights(device, precision):
    parameters = preprocess_cached_predictor_parameters(device, mlp_dtype=predictor_mlp_dtype(precision))
    state = checkpoint.load_prefixed(CODE_PREDICTOR_PREFIX)
    for index, layer in enumerate(parameters["layers"]):
        for name in MLP:
            weight = state[f"model.layers.{index}.mlp.{name}.weight"].t().contiguous()
            stored = ttnn.to_torch(layer[name])
            if precision == "max":
                assert layer[name].dtype == ttnn.bfloat16
                assert torch.equal(stored, weight.to(torch.bfloat16)), f"layer {index} {name}"
            else:
                assert layer[name].dtype == MLP_WEIGHT_DTYPE
                packed = ttnn.to_torch(ttnn.from_torch(weight, dtype=MLP_WEIGHT_DTYPE, layout=ttnn.TILE_LAYOUT))
                assert torch.equal(stored, packed), f"layer {index} {name}"

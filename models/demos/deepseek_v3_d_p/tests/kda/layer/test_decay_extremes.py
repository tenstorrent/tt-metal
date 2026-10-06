# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA layer accuracy at the decay-gate extremes against an FP64 CPU reference (tt_metal_tracker-g1b.7.1).

One crafted head per case (tests/kda/decay_extremes.py) on one device, with the production Kimi-K3 recurrence
configuration at T=1280 (grouped scan, 20-chunk summary groups, BF16 summaries, HiFi2 prefix). Calls are chained
through the device-resident state. Besides PCC, the test gates the localized and scale errors the decay path can
introduce: the worst per-token output error and the final recurrent state's worst per-key-row error and norm.
Real-text cases (tt_metal_tracker-g1b.7.2) feed the model's own text input to its most extreme heads instead.

References are prepared without a device: python -m models.demos.deepseek_v3_d_p.tests.kda.decay_extremes
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.kda.cases import compute_on_cache_miss
from models.demos.deepseek_v3_d_p.tests.kda.decay_extremes import (
    DECAY_EXTREME_CASES,
    case_config,
    decay_extreme_reference,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.timeout(900)]

_PCC_THRESHOLD = 0.9995  # the K3 layer acceptance threshold
# Targeted gates at twice the worst value of the clean controls (synthetic-control and glm-control, device run
# 2026-10-05): output token error/RMS 3.3e-2, final-state key-row error/RMS 0.13, state norm ratio 0.985.
_OUTPUT_TOKEN_SCALE_ERROR = 0.066  # worst token error / RMS token norm
_STATE_ROW_SCALE_ERROR = 0.27  # worst key-row error / RMS key-row norm
_STATE_NORM_RATIO_ERROR = 0.03


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected, actual = expected.double().flatten(), actual.double().flatten()
    return float(torch.corrcoef(torch.stack((expected, actual)))[0, 1])


def _relative_rows(expected: torch.Tensor, actual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row error of the last dimension relative to (own row norm, RMS row norm of the tensor).

    The scale-relative form is gated: a row whose own norm is tiny (a token with a small output, a strongly
    decayed key row) has a large own-relative error that does not matter downstream.
    """
    expected, actual = expected.double(), actual.double()
    error = (actual - expected).norm(dim=-1)
    norms = expected.norm(dim=-1)
    return error / norms.clamp_min(1e-30), error / norms.square().mean().sqrt()


def _dump(case_name: str, name: str, tensor: torch.Tensor) -> None:
    directory = os.environ.get("KDA_DECAY_EXTREMES_DUMP")
    if directory:
        path = Path(directory) / f"{case_name}-{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)


# Failures measured on today's code (Blackhole, 2026-10-05, tt_metal_tracker-g1b.7.1). Each passes the K3 acceptance
# output PCC (0.9995); the targeted gates catch them.
_MEASURED_FAILURES = {
    # Remaining failures after the weak-decay fix (tt_metal_tracker-g1b.7: complement-form decay, FP32 state carry),
    # the FP32 gate chain (tt_metal_tracker-g1b.4.13) and the suffix-sum k_dec_t with the FP32 q/k/v convolution
    # carry (tt_metal_tracker-g1b.4.18), measured on device 2026-10-06. Cases fixed by them (synthetic-weak, glm-weak,
    # real-text k3 h1/h24/h28/h36, glm h10/h18; the strong-band state cases synthetic-strong, glm-strong and glm h32,
    # whose error was the anchored k_dec_t) pass and carry no mark. k3-strong-saturated (in-domain crafted, g1b.4.17)
    # failed on long-memory rows of a saturated head (key-row error/RMS 0.31) through the HiFi2 recurrent scan; it
    # passes at the HiFi3 scan (0.12, tt_metal_tracker-g1b.4.19). k3-control and k3-weak failed only on crafted inputs
    # outside the input_layernorm domain (BF16 decay rank at |f_a x| RMS ~75); in-domain they pass (g1b.4.17).
    "synthetic-strong-saturated-h0-1-T1280x1": "strong decay |G_last|=160 (g=-5): output token 31 of a chunk error/RMS "
    "1.0e-1 (control 3.3e-2); output PCC 0.99995; no KDA strong-end fix by decision (tt_metal_tracker-g1b.4.16)",
}


def _case_ids() -> list:
    return [
        pytest.param(
            name,
            id=name,
            marks=[pytest.mark.xfail(strict=True, raises=AssertionError, reason=_MEASURED_FAILURES[name])]
            if name in _MEASURED_FAILURES
            else [],
        )
        for name in DECAY_EXTREME_CASES
    ]


@pytest.mark.parametrize("case_name", _case_ids())
def test_kda_layer_decay_extremes(device: ttnn.Device, case_name: str) -> None:
    case = DECAY_EXTREME_CASES[case_name]
    reference = decay_extreme_reference(case, compute_missing=compute_on_cache_miss())
    config = case_config(case)
    logger.info(f"{case_name}: {reference.metadata}")
    layer = ttKDA(
        device,
        config,
        reference.weights,
        program_config=kimi_k3_program_config(
            active_seq_len_local=case.chunk_tokens, tp_ccl_topology=ttnn.Topology.Linear
        ),
        active_seq_len=case.chunk_tokens,
    )
    actual_start = make_actual_start(device)
    state = layer.allocate_state()
    failures = []
    for call in range(case.num_calls):
        hidden = reference.hidden[:, call * case.chunk_tokens : (call + 1) * case.chunk_tokens]
        hidden_tt = ttnn.from_torch(
            hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output_tt, next_state = layer.forward(hidden_tt, state, actual_start)
        ttnn.deallocate(hidden_tt)
        ttnn.deallocate(state.recurrent)
        ttnn.deallocate(state.convolution)
        state = next_state
        output = ttnn.to_torch(output_tt).reshape(reference.outputs[call].shape)
        ttnn.deallocate(output_tt)
        expected = reference.outputs[call]
        _dump(case_name, f"output{call}", output)
        own_error, token_error = _relative_rows(expected[0], output[0])
        worst = int(token_error.argmax())
        output_pcc = _pcc(expected, output)
        logger.info(
            f"{case_name} call {call}: output PCC={output_pcc:.6f} token error/RMS max={float(token_error.max()):.3e} "
            f"at token {worst} (mod 32 = {worst % 32}) median={float(token_error.median()):.3e}; own-relative max="
            f"{float(own_error.max()):.3e} median={float(own_error.median()):.3e}; "
            f"norm ratio={float(output.double().norm() / expected.norm()):.5f}"
        )
        if not output_pcc >= _PCC_THRESHOLD:
            failures.append(f"call {call} output PCC {output_pcc:.6f} < {_PCC_THRESHOLD}")
        if not float(token_error.max()) <= _OUTPUT_TOKEN_SCALE_ERROR:
            failures.append(
                f"call {call} output token {worst} error/RMS {float(token_error.max()):.3e} > {_OUTPUT_TOKEN_SCALE_ERROR}"
            )

    recurrent = ttnn.to_torch(state.recurrent).reshape(reference.states[-1].shape)
    _dump(case_name, "state", recurrent)
    expected_state = reference.states[-1]
    own_row_error, row_error = _relative_rows(expected_state[0], recurrent[0])  # [H, K]
    norm_ratio = float(recurrent.double().norm() / expected_state.norm())
    state_pcc = _pcc(expected_state, recurrent)
    worst_row = tuple(int(i) for i in torch.nonzero(row_error == row_error.max())[0])
    logger.info(
        f"{case_name} final state: PCC={state_pcc:.6f} norm ratio={norm_ratio:.5f} key-row error/RMS "
        f"max={float(row_error.max()):.3e} at (head, key) {worst_row} median={float(row_error.median()):.3e}; "
        f"own-relative max={float(own_row_error.max()):.3e} median={float(own_row_error.median()):.3e}"
    )
    ttnn.deallocate(state.recurrent)
    ttnn.deallocate(state.convolution)
    if not state_pcc >= _PCC_THRESHOLD:
        failures.append(f"final state PCC {state_pcc:.6f} < {_PCC_THRESHOLD}")
    if not abs(norm_ratio - 1.0) <= _STATE_NORM_RATIO_ERROR:
        failures.append(f"final state norm ratio {norm_ratio:.5f} outside 1 +- {_STATE_NORM_RATIO_ERROR}")
    if not float(row_error.max()) <= _STATE_ROW_SCALE_ERROR:
        failures.append(
            f"final state key row {worst_row} error/RMS {float(row_error.max()):.3e} > {_STATE_ROW_SCALE_ERROR}"
        )
    assert not failures, f"{case_name}: " + "; ".join(failures)

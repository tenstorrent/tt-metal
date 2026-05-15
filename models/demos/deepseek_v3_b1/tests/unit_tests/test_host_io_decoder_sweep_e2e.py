# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end on-device PCC validation for the HostIoDecoderStage sweep.

Each parametrized case opens a BH 2D mesh, instantiates the decoder for one
layer, sweeps a reference trace through it, and asserts the collected output
PCC's >= ``pcc_threshold`` against the reference's ``output``. PCC failure -->
``AssertionError`` (from ``run_sweep`` Phase 3b).

This file doubles as a **status board for what's actually runnable today**.
Each ``pytest.mark.skip`` cites the specific blocker and what would unblock it.
As blockers fall (Kimi weights staged, gate kernel ported, etc.), flip the
``skip`` to ``skipif`` against a real path check, or remove the marker outright.

Run on a single BH galaxy (slow dispatch required)::

    TT_METAL_SLOW_DISPATCH_MODE=1 \\
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_host_io_decoder_sweep_e2e.py -v

Tests auto-skip on non-Blackhole arches (via the ``bh_2d_mesh_device``
fixture's ``silicon_arch_blackhole`` requirement) and when slow dispatch is
not enabled.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import create_fabric_router_config
from models.demos.deepseek_v3_b1.tests.unit_tests.host_io_decoder_harness import HostIoDecoderSweepConfig, run_sweep

# ---------------------------------------------------------------------------
# Device params — ported verbatim from run_host_io_decoder_sweep.py so the
# pytest invocation matches the CLI's production fabric / worker-L1 config.
# ---------------------------------------------------------------------------
_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D_TORUS_X
_FABRIC_ROUTER_MAX_PAYLOAD_BYTES = 15232
_WORKER_L1_SIZE = 1431568


def _device_params() -> dict:
    return {
        "fabric_config": _FABRIC_CONFIG,
        "fabric_router_config": create_fabric_router_config(_FABRIC_ROUTER_MAX_PAYLOAD_BYTES),
        "worker_l1_size": _WORKER_L1_SIZE,
    }


# ---------------------------------------------------------------------------
# Reference trace roots.
# Override via env vars to point at a non-default checkout / staging path.
# ---------------------------------------------------------------------------
_BIT_SCULPT_ROOT_ENV = "BIT_SCULPT_ROOT"
_PIPECLEAN_TRACES_ENV = "DEEPSEEK_PIPECLEAN_TRACES_DIR"
_KIMI_WEIGHTS_ENV = "KIMI_K26_BF16_DEQUANT_PATH"

_DEFAULT_BIT_SCULPT_ROOT = Path("/workspace/bit_sculpt")
_DEFAULT_PIPECLEAN_TRACES_DIR = Path("/data/asaigal/pipeclean_traces")
_DEFAULT_KIMI_WEIGHTS_PATH = Path("/workspace/models/moonshotai/Kimi-K2.6-bf16-dequant")


def _kimi_weights_path() -> Path:
    return Path(os.environ.get(_KIMI_WEIGHTS_ENV, str(_DEFAULT_KIMI_WEIGHTS_PATH)))


def _bit_sculpt_trace_root() -> Path:
    return Path(os.environ.get(_BIT_SCULPT_ROOT_ENV, str(_DEFAULT_BIT_SCULPT_ROOT))) / "results"


def _pipeclean_traces_dir() -> Path:
    return Path(os.environ.get(_PIPECLEAN_TRACES_ENV, str(_DEFAULT_PIPECLEAN_TRACES_DIR)))


def _deepseek_trace_dir() -> Path:
    return _bit_sculpt_trace_root() / "deepseek-r1-0528/debug_trace"


def _kimi_trace_dir() -> Path:
    return _bit_sculpt_trace_root() / "moonshotai-kimi-k26/debug_trace"


# ---------------------------------------------------------------------------
# Parametrized cases. Each entry is (sweep config dict, pytest id, skip-mark).
# ---------------------------------------------------------------------------


def _build_param(
    *,
    case_id: str,
    config_kwargs: dict,
    skip: pytest.MarkDecorator | None = None,
) -> pytest.param:
    """Build a pytest.param so skip reasons live next to the config they describe."""
    marks = [skip] if skip is not None else []
    return pytest.param(config_kwargs, id=case_id, marks=marks)


SWEEP_CASES = [
    # --- DeepSeek, original .pt path (back-compat) ----------------------------
    # Runnable today on a BH galaxy iff the pipeclean traces + R1 dequant
    # weights are staged at the expected paths. The CLI driver has used this
    # case in production since the pytest -> CLI migration; this entry just
    # codifies it as a real pytest.
    _build_param(
        case_id="deepseek_pt_pipeclean_8192_layer4_modeA",
        config_kwargs=dict(
            decoder_layer_idx=4,
            hidden_states_dir=_pipeclean_traces_dir(),
            prompt_names=("pipeclean_seq_8192",),
            trace_format="pt",
        ),
        skip=pytest.mark.skipif(
            not (_pipeclean_traces_dir() / "pipeclean_seq_8192.pt").exists(),
            reason=(
                f"Pipeclean .pt trace not staged. Expected "
                f"{_pipeclean_traces_dir()}/pipeclean_seq_8192.pt (override via "
                f"${_PIPECLEAN_TRACES_ENV})."
            ),
        ),
    ),
    # --- DeepSeek, safetensors flat layout ------------------------------------
    # Existing checked-in DeepSeek bit_sculpt traces (aho_*, tt_moconnor*, ...)
    # use the *old bundled* on-disk format (`hidden_states.safetensors` with all
    # layers in one file). The loader implements the *new per-layer* format
    # (decoder_input_layer_{L}.safetensors / decoder_output_layer_{L}.safetensors).
    # To unblock: regenerate a small trace with the current bit_sculpt
    # `run_debug_trace.py --capture-group-a` (which emits per-layer files), drop
    # it under bit_sculpt's results dir, and point this case at it.
    _build_param(
        case_id="deepseek_safetensors_flat_layer4_modeA",
        config_kwargs=dict(
            decoder_layer_idx=4,
            hidden_states_dir=_deepseek_trace_dir(),
            prompt_names=("PLACEHOLDER_per_layer_trace",),
            trace_format="safetensors",
        ),
        skip=pytest.mark.skip(
            reason=(
                "Needs a DeepSeek-R1-0528 trace in bit_sculpt's *new* per-layer "
                "safetensors format. All currently checked-in DeepSeek traces "
                "(aho_*, tt_moconnor*, ...) use the *old bundled* format "
                "(hidden_states.safetensors with all layers in one file). "
                "Regenerate with current run_debug_trace.py + --capture-group-a, "
                "then update the prompt name above."
            ),
        ),
    ),
    # --- DeepSeek, safetensors per-step decode layout -------------------------
    # Same as above — aho_sus_decode_10/step_0/ contains hidden_states.safetensors
    # (old bundle), not per-layer files. Regenerate with current bit_sculpt.
    _build_param(
        case_id="deepseek_safetensors_per_step_layer4_modeA",
        config_kwargs=dict(
            decoder_layer_idx=4,
            hidden_states_dir=_deepseek_trace_dir(),
            prompt_names=("PLACEHOLDER_per_step_trace",),
            trace_format="safetensors",
        ),
        skip=pytest.mark.skip(
            reason=(
                "Needs a DeepSeek decode-mode trace in per-step + per-layer format. "
                "aho_sus_decode_10 and demo_1K_debug_decode_10000 step_*/ dirs "
                "currently hold the old bundled format. Regenerate with current "
                "run_debug_trace.py --decode-steps N --capture-group-a, then update "
                "the prompt name above."
            ),
        ),
    ),
    # --- Kimi K2.6 layer 0 (dense) --------------------------------------------
    # Unblocked by the `weight_key_prefix="language_model."` knob added to
    # CacheWeightProvider in this PR. Skip now requires only the BF16-dequant
    # snapshot to be staged at $KIMI_K26_BF16_DEQUANT_PATH (default
    # /workspace/models/moonshotai/Kimi-K2.6-bf16-dequant). Produce the
    # snapshot via bit_sculpt's scripts/dequant_compressed_tensors_streaming.py.
    _build_param(
        case_id="kimi_safetensors_flat_dense_layer0_modeA",
        config_kwargs=dict(
            decoder_layer_idx=0,
            hidden_states_dir=_kimi_trace_dir(),
            prompt_names=("smoke_kimi_k26_2026-05-14_bf16",),
            trace_format="safetensors",
            hf_model_path=_kimi_weights_path(),
            weight_key_prefix="language_model.",
        ),
        skip=pytest.mark.skipif(
            not (_kimi_weights_path() / "config.json").exists(),
            reason=(
                f"Kimi K2.6 BF16-dequant snapshot not staged. Expected "
                f"{_kimi_weights_path()}/config.json (override via ${_KIMI_WEIGHTS_ENV}). "
                f"Produce via bit_sculpt scripts/dequant_compressed_tensors_streaming.py."
            ),
        ),
    ),
    # --- Kimi K2.6 MoE layers (1-60) ------------------------------------------
    # Blocked on the 384-expert / ungrouped-top-8 gate kernel. The b1 demo's
    # DeepseekMoeGateSingleCore (micro_ops/deepseek_moe_gate/op.py) hardcodes
    # (16, 16) input tile == 256 grouped routing. Kimi has 1 group of 384, plain
    # topk. Tracked on origin/ddjekic/kimi26_bringup (full Kimi demo on the
    # `_d_p` path) and origin/gchoudhary/41826-generalize-moe_compute-shape-support.
    _build_param(
        case_id="kimi_safetensors_flat_moe_layer1_modeA",
        config_kwargs=dict(
            decoder_layer_idx=1,
            hidden_states_dir=_kimi_trace_dir(),
            prompt_names=("smoke_kimi_k26_2026-05-14_bf16",),
            trace_format="safetensors",
            hf_model_path=_kimi_weights_path(),
            weight_key_prefix="language_model.",
        ),
        skip=pytest.mark.skip(
            reason=(
                "Kimi K2.6 MoE layers (1-60) need a 384-expert / ungrouped-top-8 "
                "gate kernel. b1's DeepseekMoeGateSingleCore hardcodes 16x16=256 "
                "grouped routing. Tracked on origin/ddjekic/kimi26_bringup and "
                "origin/gchoudhary/41826-generalize-moe_compute-shape-support. "
                "See KIMI_K26_PORT_NOTES section in host_io_decoder_harness.py "
                "for the full breakdown."
            ),
        ),
    ),
    _build_param(
        case_id="kimi_safetensors_per_step_moe_layer30_modeA",
        config_kwargs=dict(
            decoder_layer_idx=30,
            hidden_states_dir=_kimi_trace_dir(),
            prompt_names=("q_what_is_python",),
            trace_format="safetensors",
            hf_model_path=_kimi_weights_path(),
            weight_key_prefix="language_model.",
        ),
        skip=pytest.mark.skip(
            reason=(
                "Same gate-kernel blocker as kimi_safetensors_flat_moe_layer1_modeA. "
                "Kept as a separate case so when the gate kernel lands, both layouts "
                "(flat + per-step) are exercised on the validation path."
            ),
        ),
    ),
]


# ---------------------------------------------------------------------------
# Test body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize("config_kwargs", SWEEP_CASES)
def test_host_io_decoder_sweep_e2e(bh_2d_mesh_device, config_kwargs):
    """One PCC-validated decoder-layer sweep per parametrized case.

    Mode A (single replicated slot) is enough to validate cross-trace PCC —
    no slot-replication validation needed for correctness — and runs in well
    under the BH galaxy CI per-case budget.
    """
    if not is_slow_dispatch():
        pytest.skip(
            "run_sweep requires TT_METAL_SLOW_DISPATCH_MODE=1 (H2D/D2H sockets do not "
            "function under fast dispatch). Re-run with the env var set."
        )

    config = HostIoDecoderSweepConfig(
        num_replication_slots=1,
        validate_metadata_roundtrip=True,
        validate_hidden_states_cross_slot=False,  # no-op at num_replication_slots=1
        validate_kv_cache_cross_slot=False,  # ditto
        validate_hidden_states_cross_trace=True,
        pcc_threshold=0.97,
        dump_hidden_states=False,
        dump_kv_cache=False,
        **config_kwargs,
    )
    run_sweep(config, bh_2d_mesh_device)

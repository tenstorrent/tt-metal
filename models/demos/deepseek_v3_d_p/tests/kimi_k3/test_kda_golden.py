# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3's KDA layer against the real vLLM trace. Host only — no TTNN, no device.

`reference/kda/` is the oracle every KDA device gate is scored against, and until now it has only
ever been scored against itself: `reference/kda/tests/` checks its ops against hand-written
semantics, and the device tests check TT against it. Nothing checked it against activations a real
Kimi-K3 actually produced. The 100k trace captures layer 0's KDA input, output and per-640-token
state, so it can.

Two things are pinned:

* **The forward.** `kda_forward_reference(kda_input_layer_0)` reproduces `kda_output_layer_0`. The
  input is already post-`input_layernorm` (see `test_golden_contract.py`), which is exactly what
  the reference expects, so this is the module in isolation with nothing else in the way.
* **The carry, and its layout.** Feeding the sequence in 640-token segments and threading
  `KDAReferenceState` reproduces the golden snapshot at *every* boundary. This is the property the
  whole chunked-prefill story rests on — a chunk boundary must be invisible — established on host
  before any device work.

  The layout note is the load-bearing part. The golden stores the recurrent state as
  `[heads, v_dim, k_dim]`; `KDAReferenceState.recurrent` is `[batch, heads, k_dim, v_dim]`. Both
  are `[96, 128, 128]`, so a comparison that forgets the transpose type-checks, runs, and reports
  **PCC 0.012** — indistinguishable from a broken recurrence. Anyone wiring the device carry
  against this trace will hit that, which is why it is asserted here with the wrong orientation as
  an explicit control.
"""

from pathlib import Path

import pytest
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kda.layer import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace

# Host fp32 against a trace whose activations are bf16 and whose state is fp32 from a different
# accumulation order. The observed values are 1.0000000 (output) and 0.99998 (state); a wrong
# orientation or a dropped carry lands at 0.01 and 0.70 respectively, so there is no grey zone.
KDA_PCC = 0.9999

# The trace snapshots state every 640 tokens, so this is the trace's cadence, not a choice. Four
# boundaries is enough to show the carry compounds correctly rather than merely starting right.
STATE_STRIDE = 640
NUM_CHUNKS = 4

# Layer 0 is the only KDA layer the 100k trace instruments.
KDA_LAYER = 0


@pytest.fixture(scope="module")
def kda_layer_weights():
    checkpoint = resolve_checkpoint()
    if checkpoint is None:
        pytest.skip("no Kimi-K3 checkpoint on this host; set KIMI_K3_HF_MODEL")
    # KDA weights are bf16 in both published checkpoints (`quantization_config.ignore` covers
    # `re:.*self_attn.*`), so either root works and the loader resolves which.
    return load_kda_layer_state_dict(Path(checkpoint), KDA_LAYER, kimi_k3_kda_config())


@pytest.fixture(scope="module")
def trace():
    found = resolve_trace(TRACE_100K)
    if found is None:
        pytest.skip("no Kimi-K3 100k golden trace on this host; set KIMI_K3_GOLDEN_TRACE")
    return found


def test_kda_reference_matches_the_trace(trace, kda_layer_weights):
    """The KDA module in isolation, against activations the real model produced."""
    tokens = STATE_STRIDE
    hidden = trace.rows("kda", f"kda_input_layer_{KDA_LAYER}", 0, tokens).unsqueeze(0)
    golden = trace.rows("kda", f"kda_output_layer_{KDA_LAYER}", 0, tokens)

    output, _ = kda_forward_reference(hidden, kda_layer_weights, kimi_k3_kda_config())

    passed, message = comp_pcc(golden, output.squeeze(0), KDA_PCC)
    logger.info(f"KDA layer {KDA_LAYER} output over {tokens} tokens: {message}")
    assert passed, f"kda_forward_reference != kda_output_layer_{KDA_LAYER}: {message}"


def test_kda_carry_survives_chunk_boundaries(trace, kda_layer_weights):
    """Segmented prefill reproduces the golden state at every 640-token boundary.

    The golden's `[heads, v_dim, k_dim]` is the reference's `[heads, k_dim, v_dim]` transposed;
    both are `[96, 128, 128]`, so the wrong orientation is silent. It is asserted below as a
    control precisely because it cannot be caught by shape.
    """
    config = kimi_k3_kda_config()
    hidden = trace.rows("kda", f"kda_input_layer_{KDA_LAYER}", 0, STATE_STRIDE * NUM_CHUNKS).unsqueeze(0)

    state = None
    for chunk in range(NUM_CHUNKS):
        segment = hidden[:, chunk * STATE_STRIDE : (chunk + 1) * STATE_STRIDE]
        _, state = kda_forward_reference(segment, kda_layer_weights, config, state)

        # Row `c` is the state after position `(c + 1) * 640`, per the trace's own tensor_mapping.
        golden = trace.rows("kda", f"kda_recurrent_state_layer_{KDA_LAYER}", chunk, chunk + 1).squeeze(0)
        carried = state.recurrent.squeeze(0).transpose(-1, -2)

        passed, message = comp_pcc(golden, carried, KDA_PCC)
        logger.info(f"KDA carry after chunk {chunk} (position {(chunk + 1) * STATE_STRIDE}): {message}")
        assert passed, f"KDA carry diverged at chunk boundary {chunk}: {message}"

        untransposed_passed, untransposed_message = comp_pcc(golden, state.recurrent.squeeze(0), 0.9)
        assert not untransposed_passed, (
            "the untransposed state also matches, so this test would not catch a layout error: "
            f"{untransposed_message}"
        )

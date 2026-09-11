# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is the Kimi-K3 KDA failure local to one chip, or does it live in the distributed prefix?

At 8x4 the layer runs sequence-parallel over 8 ranks, so every result so far mixes the per-chip
chunk scan with the cross-rank affine prefix that stitches the ranks together. Both ops that can be
scored against their own oracles are correct — `prepare_chunk_recurrence` matches on real layer-1
tensors and `recurrent_chunk_scan` matches at every magnitude — so what is left is the composition,
and the composition has a sequence-parallel half that a single chip does not exercise at all.

One chip settles it. With SP=1 and TP=1 there is no halo, no affine summary, no cross-rank exchange
and no `tt_ccl`: just the local chunk scan. If layer 1 is correct here, the defect is in the
distributed prefix; if it is wrong here too, the defect is local and the 8x4 result was never about
sharding.

Layer 0 runs alongside as the control, since it is the one KDA layer that works at 8x4.

Measured: layer 0 scores 0.99994 on one chip and layer 1 scores 0.69629, already 0.77954 over the
first 256 tokens. So the defect is LOCAL — the 8x4 result of 0.004 is the distributed prefix
compounding a per-chip error, not causing it — and it reproduces on a single Blackhole card with no
Galaxy, no mesh and no fabric.

This is the combination no existing test covers, and the reason is worth stating precisely.
`test_real_weights.py::test_kimi_k3_layer_1_real_weights_pcc` uses this exact layer's real weights
and passes at 0.9995, but `make_kimi_k3_test_case` feeds it `torch.randn` hidden states. Real
weights, synthetic activations. The op suites underneath cover the opposite corner: real op
semantics, but a gate pinned to [-0.051, -0.001] where the model's spans (-5, 0). Nothing exercises
real weights against real activations, which is the only place the two meet.
"""
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.layer import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA

SEQ = 1024


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_kda_single_device_matches_reference(device, device_params, layer_idx):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    config = kimi_k3_kda_config()
    weights = load_kda_layer_state_dict(Path(checkpoint), layer_idx, config)
    hidden = trace.rows("kda", "kda_input_layer_0", 0, SEQ)
    want, _ = kda_forward_reference(hidden.float().unsqueeze(0), weights, config)

    layer = ttKDA(device, config, weights, layer_idx=layer_idx, sp_axis=0, tp_axis=1)
    state = layer.allocate_state(batch_size=1)
    hidden_tt = ttnn.from_torch(
        hidden.unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output, _ = layer.forward(hidden_tt, state)
    got = ttnn.to_torch(output).reshape(SEQ, -1)

    pcc = float(str(comp_pcc(want.squeeze(0), got, 0.99)[1]).split()[-1])
    segment = SEQ // 4
    per_position = "  ".join(
        f"[{i * segment}] "
        f"{float(str(comp_pcc(want.squeeze(0)[i * segment:(i + 1) * segment], got[i * segment:(i + 1) * segment], 0.99)[1]).split()[-1]):.5f}"
        for i in range(4)
    )
    logger.info(f"  layer {layer_idx} single chip: PCC {pcc:.5f}   per-position {per_position}")

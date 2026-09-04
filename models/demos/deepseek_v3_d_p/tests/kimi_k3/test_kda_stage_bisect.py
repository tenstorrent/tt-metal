# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which stage of ttKDA.forward diverges on Kimi-K3's real layer-1 weights?

The single-chip reproduction narrows the defect to one card and one layer: layer 0 scores 0.99994
against the torch reference and layer 1 scores 0.69629, with no mesh, no fabric and no CCL in play.
`forward` is six stages, each with an exact torch counterpart, so the divergence can be attributed
rather than guessed at.
"""
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import causal_depthwise_conv_reference, kda_gate_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import load_kda_layer_state_dict
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA

SEQ = 1024


def _pcc(want, got):
    got = ttnn.to_torch(got).float().reshape(-1, want.shape[-1]) if not isinstance(got, torch.Tensor) else got
    return float(str(comp_pcc(want.float().reshape(-1, want.shape[-1]), got, 0.99)[1]).split()[-1])


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_stage_bisect(device, device_params, layer_idx):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    config = kimi_k3_kda_config()
    w = {k: v.float() for k, v in load_kda_layer_state_dict(Path(checkpoint), layer_idx, config).items()}
    hidden = trace.rows("kda", "kda_input_layer_0", 0, SEQ)
    h = hidden.float().unsqueeze(0)
    zero = h.new_zeros(1, config.conv_kernel_size - 1, config.q_dim)

    # --- torch, stage by stage -------------------------------------------------------------------
    t_q, _ = causal_depthwise_conv_reference(F.linear(h, w["q_proj.weight"]), w["q_conv1d.weight"], zero)
    t_k, _ = causal_depthwise_conv_reference(F.linear(h, w["k_proj.weight"]), w["k_conv1d.weight"], zero)
    t_v, _ = causal_depthwise_conv_reference(F.linear(h, w["v_proj.weight"]), w["v_conv1d.weight"], zero)
    t_raw_gate = F.linear(F.linear(h, w["f_a_proj.weight"]), w["f_b_proj.weight"])
    t_gate = kda_gate_reference(
        t_raw_gate.reshape(1, SEQ, config.num_heads, config.head_k_dim),
        w["A_log"],
        w["dt_bias"],
        config.gate_lower_bound,
    ).reshape(1, SEQ, config.q_dim)
    t_beta = torch.sigmoid(F.linear(h, w["b_proj.weight"]))
    t_output_gate = F.linear(h, w["g_proj.weight"])

    # --- device, stage by stage ------------------------------------------------------------------
    layer = ttKDA(
        device,
        config,
        {k: v for k, v in load_kda_layer_state_dict(Path(checkpoint), layer_idx, config).items()},
        layer_idx=layer_idx,
        sp_axis=0,
        tp_axis=1,
    )
    state = layer.allocate_state(batch_size=1)
    hidden_tt = ttnn.from_torch(
        hidden.unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    projected = layer._project_inputs(hidden_tt)
    d_q, d_k, d_v, _ = layer._convolve_qkv(projected.qkv, state.convolution, sequence=SEQ)
    inputs = layer._compute_gates(
        d_q, d_k, d_v, beta=projected.beta, decay_rank=projected.decay_rank, output_gate=projected.output_gate
    )

    rows = [
        ("q (conv+silu)", t_q, inputs.q),
        ("k (conv+silu)", t_k, inputs.k),
        ("v (conv+silu)", t_v, inputs.v),
        ("beta", t_beta, inputs.beta),
        ("gate (log decay)", t_gate, inputs.decay),
        ("output_gate", t_output_gate, inputs.output_gate),
    ]
    logger.info(f"  layer {layer_idx} stages:")
    for name, want, got in rows:
        logger.info(f"    {name:20s} {_pcc(want, got):9.5f}")

    # The gate is the one stage with a nonlinearity whose device and torch spellings could differ,
    # so report its range too: `-5 * sigmoid(z)` is only as good as sigmoid is over z's actual span.
    d_gate = ttnn.to_torch(inputs.decay).float().reshape(-1)
    logger.info(
        f"    gate range torch [{float(t_gate.min()):7.4f}, {float(t_gate.max()):8.5f}]  "
        f"device [{float(d_gate.min()):7.4f}, {float(d_gate.max()):8.5f}]  "
        f"max abs diff {float((t_gate.reshape(-1) - d_gate).abs().max()):.5f}"
    )

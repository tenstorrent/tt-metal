# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prepare or scan? A 2x2 on Kimi-K3's real layer-1 tensors.

Every input to the recurrence is already cleared: on one chip the six stages feeding `_kda_prefill`
— q, k, v after conv+SiLU, beta, the log-decay gate and the output gate — all score >= 0.99996
against torch for both layer 0 and layer 1. So whatever goes wrong is inside the recurrence, which
is `prepare_chunk_recurrence` followed by `recurrent_chunk_scan`.

Crossing the two stages between device and torch attributes it exactly. The torch/torch corner is
the reference; whichever substitution moves away from it is the faulty stage.
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
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    device_protocol,
    recurrent_oracle,
    run_recurrent,
    to_device,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.test_prepare_chunk_recurrence import (
    OUTPUT_NAMES,
    _device_inputs,
    _oracle,
    _run,
)

SEQ = 1024
BF16_MASK = 0x26


@run_for_blackhole()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_prepare_versus_scan(device, device_params, layer_idx):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    config = kimi_k3_kda_config()
    w = {k: v.float() for k, v in load_kda_layer_state_dict(Path(checkpoint), layer_idx, config).items()}
    h = trace.rows("kda", "kda_input_layer_0", 0, SEQ).float().unsqueeze(0)
    zero = h.new_zeros(1, config.conv_kernel_size - 1, config.q_dim)

    def conv(name):
        out, _ = causal_depthwise_conv_reference(
            F.linear(h, w[f"{name}_proj.weight"]), w[f"{name}_conv1d.weight"], zero
        )
        return out

    q, k, v = conv("q"), conv("k"), conv("v")
    gate = kda_gate_reference(
        F.linear(F.linear(h, w["f_a_proj.weight"]), w["f_b_proj.weight"]).reshape(
            1, SEQ, config.num_heads, config.head_k_dim
        ),
        w["A_log"],
        w["dt_bias"],
        config.gate_lower_bound,
    ).reshape(1, SEQ, config.q_dim)
    beta = torch.sigmoid(F.linear(h, w["b_proj.weight"]))
    beta = beta.reshape(SEQ // 32, 32, config.num_heads).permute(2, 0, 1).unsqueeze(-1).contiguous()
    inputs = (q, k, v, gate, beta)

    try:
        torch_protocol = _oracle(inputs, config.num_heads, BF16_MASK)
    except Exception as error:
        logger.info(f"  layer {layer_idx}: torch prepare unrepresentable -- {type(error).__name__}")
        return
    device_protocol_tt = _run(_device_inputs(inputs, device), config.num_heads, output_bf16_mask=BF16_MASK)
    device_prepared = tuple(ttnn.to_torch(t).float() for t in device_protocol_tt)

    for name, a, b in zip(OUTPUT_NAMES, torch_protocol, device_prepared, strict=True):
        line = f"    prepare {name:14s} {float(str(comp_pcc(a.float(), b, 0.99)[1]).split()[-1]):9.5f}"
        if name == "t_inv":
            # t_inv is I + strictly-lower, so whole-tensor PCC is dominated by the diagonal and
            # stays near 1 even when the off-diagonals — the entire content of the UT transform —
            # are wrong. Score the strictly-lower part on its own.
            lo = torch.tril(torch.ones(32, 32), diagonal=-1).bool()
            wa, wb = a.float()[..., lo], b.float()[..., lo]
            line += (
                f"   strictly-lower only {float(str(comp_pcc(wa, wb, 0.99)[1]).split()[-1]):9.5f}"
                f"   |N|inf torch {float(a.float().abs().sum(-1).max()):8.3f}"
            )
        logger.info(line)

    # Which precision does the kernel's inversion actually achieve? `invert_doubling` evaluates the
    # exact nilpotent identity (I-N)^-1 = (I+N)(I+N^2)(I+N^4)(I+N^8)(I+N^16), which is stable in fp32
    # but not if the intermediate powers round to bf16 — N^16 is O(1e2) here while the true inverse
    # is bounded by 1. Matching the device against both settles which it is getting.
    eye = torch.eye(32)
    lower = torch.tril(torch.ones(32, 32), diagonal=-1).bool()

    def doubling(neg_n, bf16):
        r = (lambda x: x.to(torch.bfloat16).float()) if bf16 else (lambda x: x)
        total, power = eye + neg_n, r(neg_n @ neg_n)
        for _ in range(4):
            total = r(total + r(power @ total))
            power = r(power @ power)
        return total

    exact = torch_protocol[6].float().reshape(-1, 32, 32)
    # Recover N from the oracle's own t_inv: it inverted (I + tril(akk,-1)), so N = -tril(akk,-1)
    # and (I - N) = t_inv^-1.
    neg_n = torch.linalg.inv(exact.double()).float() - eye
    neg_n = torch.tril(neg_n, diagonal=-1)
    device_t_inv = device_prepared[6].reshape(-1, 32, 32)
    for label, candidate in (("fp32 doubling", doubling(neg_n, False)), ("bf16 doubling", doubling(neg_n, True))):
        pcc = float(str(comp_pcc(candidate[..., lower], device_t_inv[..., lower], 0.99)[1]).split()[-1])
        logger.info(
            f"    device t_inv vs {label:15s} strictly-lower PCC {pcc:9.5f}"
            f"   max|err| {float((candidate - device_t_inv).abs().max()):9.4f}"
        )
    logger.info(
        f"    ||N||inf max {float(neg_n.abs().sum(-1).max()):8.3f}   true |t_inv|max {float(exact.abs().max()):7.3f}"
    )

    heads, chunks = torch_protocol[0].shape[0], torch_protocol[0].shape[1]
    state = torch.zeros(heads, config.head_k_dim, config.head_v_dim)
    reference, _ = recurrent_oracle(torch_protocol, state)

    def score(label, protocol, on_device):
        if on_device:
            out = run_recurrent(device_protocol(protocol, device), to_device(state, device))
            got = ttnn.to_torch(out[0]).float()
        else:
            got, _ = recurrent_oracle(protocol, state)
            got = got.float()
        pcc = float(str(comp_pcc(reference.float(), got, 0.99)[1]).split()[-1])
        logger.info(f"    {label:34s} {pcc:9.5f}")

    logger.info(f"  layer {layer_idx} 2x2 (vs torch prepare + torch scan):")
    score("torch prepare -> device scan", torch_protocol, True)
    score("device prepare -> torch scan", device_prepared, False)
    score("device prepare -> device scan", device_prepared, True)

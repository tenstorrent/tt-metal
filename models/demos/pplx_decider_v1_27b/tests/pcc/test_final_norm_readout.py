# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Final RMSNorm and the 5120x255 readout vs HF, real weights, real last-layer hidden states."""

import pytest
import torch

from models.demos.pplx_decider_v1_27b.reference import hf_reference as ref
from models.demos.pplx_decider_v1_27b.tests.test_utils import (
    DEVICE_PARAMS,
    SEQ_LENS,
    bf16_round,
    build_optimizations,
    check_pcc,
    golden_tensor,
    model_args,
    reader,
    seq_ids,
    to_device,
    to_host,
)
from models.demos.pplx_decider_v1_27b.tt.norm import PplxRMSNorm
from models.demos.pplx_decider_v1_27b.tt.readout import PplxReadout
from models.demos.pplx_decider_v1_27b.tt.weight_adapter import build_readout_weight, zero_centred_norm


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=seq_ids())
def test_final_norm(device, seq_len):
    hf = ref.build_final_norm(reader())
    x = bf16_round(golden_tensor("final_input", seq_len))
    with torch.no_grad():
        expected = hf(x)
    opts = build_optimizations(device)
    tt = PplxRMSNorm(
        zero_centred_norm(reader().final_norm_weight(), opts.policy).weight, model_args().rms_norm_eps, opts.norm
    )
    out = tt(to_device(x, device))
    check_pcc(expected, to_host(out, expected.shape), module="final_norm", seq_len=seq_len)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=seq_ids())
def test_readout(device, seq_len):
    with torch.no_grad():
        hidden = ref.build_final_norm(reader())(bf16_round(golden_tensor("final_input", seq_len)))
        expected = ref.build_readout(reader())(hidden[:, -1:, :])  # last_hidden_state[:, -1] -> readout
    opts = build_optimizations(device)
    tt = PplxReadout(build_readout_weight(reader().readout_weight(), opts.policy), opts.linear, mesh_device=device)
    out = tt(to_device(hidden, device))
    check_pcc(expected, to_host(out, expected.shape), module="readout", seq_len=seq_len)

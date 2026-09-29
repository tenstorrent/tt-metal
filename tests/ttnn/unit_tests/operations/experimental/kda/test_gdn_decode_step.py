# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness and state-lifecycle coverage for experimental GDN decode."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical


NV, NK, DK, DV = 12, 4, 128, 128
KD, VD = NK * DK, NV * DV
C = 2 * KD + VD
QKVZ = C + VD
W = QKVZ + 32
SCALE = DK**-0.5

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device({"l1_small_size": 24576})]


def _pack_rows(rows, *, parity=0, both=False):
    repeat_factor = NV // NK
    packed = torch.zeros(NV, 4, 32, 32, dtype=torch.bfloat16)
    for head in range(NV):
        key_head = head // repeat_factor
        for tap, row in enumerate(rows):
            row = row.reshape(-1).to(torch.bfloat16)
            chunks = torch.cat(
                [
                    row[key_head * DK : (key_head + 1) * DK],
                    row[KD + key_head * DK : KD + (key_head + 1) * DK],
                    row[2 * KD + head * DV : 2 * KD + (head + 1) * DV],
                ]
            ).reshape(-1, 32)
            for selected_parity in (0, 1) if both else (parity,):
                packed[head, tap, selected_parity : 2 * chunks.shape[0] + selected_parity : 2, :] = chunks
    return packed


def _reference(rows, histories, taps, state, dt_bias, neg_exp_a, weight):
    outputs = []
    states = []
    repeat_factor = NV // NK
    for user in range(rows.shape[0]):
        window = torch.cat((histories[user, 1:4], rows[user, :C].unsqueeze(0)), dim=0)
        conv = torch.nn.functional.silu(sum(taps[tap] * window[tap] for tap in range(4)))
        q = conv[:KD].reshape(NK, DK).repeat_interleave(repeat_factor, 0)
        k = conv[KD : 2 * KD].reshape(NK, DK).repeat_interleave(repeat_factor, 0)
        v = conv[2 * KD : C].reshape(NV, DV)
        z = rows[user, C:QKVZ].reshape(NV, DV)
        a = rows[user, QKVZ : QKVZ + NV]
        b = rows[user, QKVZ + NV : QKVZ + 2 * NV]
        beta = torch.sigmoid(b)
        decay = neg_exp_a * torch.nn.functional.softplus(a + dt_bias, beta=1.0, threshold=20.0)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        updated = state[user] * torch.exp(decay)[:, None, None]
        value_read = torch.einsum("hk,hkv->hv", k, updated)
        delta = beta[:, None] * (v - value_read)
        updated = updated + torch.einsum("hk,hv->hkv", k, delta)
        out = torch.einsum("hk,hkv->hv", q, updated)
        out = out / torch.sqrt((out * out).mean(-1, keepdim=True) + 1e-6) * weight[None, :]
        outputs.append((out * torch.nn.functional.silu(z)).reshape(-1))
        states.append(updated)
    return torch.stack(outputs), torch.stack(states)


class DecodeCase:
    def __init__(self, device, batch, seed):
        torch.manual_seed(seed)
        self.device = device
        self.batch = batch
        self.rows = (0.5 * torch.randn(batch, W)).bfloat16().float()
        self.histories = (0.5 * torch.randn(batch, 4, C)).bfloat16().float()
        self.taps = (0.3 * torch.randn(4, C)).bfloat16().float()
        self.state = 0.05 * torch.randn(batch, NV, DK, DV)
        self.dt_bias = (0.1 * torch.randn(NV)).bfloat16().float()
        self.neg_exp_a = (-torch.exp(0.2 * torch.randn(NV))).bfloat16().float()
        self.weight = (1.0 + 0.1 * torch.randn(DV)).bfloat16().float()
        self.expected_out, self.expected_state = _reference(
            self.rows, self.histories, self.taps, self.state, self.dt_bias, self.neg_exp_a, self.weight
        )

        def device_tensor(tensor, dtype):
            return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        self.qkv = device_tensor(self.rows.reshape(1, batch, W), ttnn.bfloat16)
        self.state_tt = device_tensor(self.state, ttnn.float32)
        self.history_host = torch.stack([_pack_rows(self.histories[user], parity=user & 1) for user in range(batch)])
        self.history_tt = device_tensor(self.history_host, ttnn.bfloat16)
        self.taps_tt = device_tensor(_pack_rows(self.taps, both=True), ttnn.bfloat16)
        self.dt_bias_tt = device_tensor(self.dt_bias.reshape(1, 1, NV), ttnn.float32)
        self.neg_exp_a_tt = device_tensor(self.neg_exp_a.reshape(1, 1, NV), ttnn.float32)
        self.weight_tt = device_tensor(self.weight.reshape(1, 1, DV), ttnn.bfloat16)

    def reset(self):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self.state, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), self.state_tt
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(self.history_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self.history_tt
        )

    def run(self):
        return ttnn.experimental.kda.gdn_decode_step(
            self.qkv,
            self.dt_bias_tt,
            self.neg_exp_a_tt,
            self.state_tt,
            self.weight_tt,
            NV,
            NK,
            DK,
            DV,
            scale=SCALE,
            output_dtype=ttnn.float32,
            conv_hist=self.history_tt,
            conv_taps=self.taps_tt,
            qkvz_dim=QKVZ,
        )

    def check(self, output, suffix=""):
        actual_out = ttnn.to_torch(output).reshape(-1, VD).float()[: self.batch]
        actual_state = ttnn.to_torch(self.state_tt).reshape(self.batch, NV, DK, DV).float()
        actual_history = ttnn.to_torch(self.history_tt).reshape(self.batch, NV, 4, 32, 32)
        assert_accurate(self.expected_out, actual_out, name=f"output{suffix}", pcc_threshold=0.999)
        assert_accurate(self.expected_state, actual_state, name=f"state{suffix}", pcc_threshold=0.9999)
        for user in range(self.batch):
            expected_history = _pack_rows([*self.histories[user, 1:4], self.rows[user, :C]], parity=user & 1)
            assert_bit_identical(expected_history, actual_history[user], name=f"history user {user}{suffix}")
        return actual_out, actual_state, actual_history


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_gdn_decode_step_fused_conv(device, batch):
    case = DecodeCase(device, batch, seed=3100 + batch)
    output = case.run()
    first = case.check(output)

    case.reset()
    second_output = case.run()
    second = case.check(second_output, suffix=" repeat")
    for name, first_value, second_value in zip(("output", "state", "history"), first, second, strict=True):
        assert_bit_identical(first_value, second_value, name=f"deterministic {name}")


def test_gdn_decode_step_trace_replay(device):
    case = DecodeCase(device, 1, seed=3201)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output = case.run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    case.reset()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    case.check(output, suffix=" trace")
    ttnn.release_trace(device, trace_id)

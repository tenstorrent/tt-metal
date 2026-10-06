# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weak-decay precision of the KDA recurrence over long chunk chains (tt_metal_tracker-g1b.7, T4/T5).

Long-memory key channels decay by exp(G_last) with |G_last| far below the BF16/TF32 resolution of 1. The scan
carries final_decay in complement form (expm1) and the state in FP32; summaries carry A - I. These tests chain
10240 chunks and compare against an FP64 oracle per decay class, with a magnitude (norm ratio) gate besides the
relative error, because a lost decay is a scale error that PCC cannot see.

Inputs isolate the decay path: kd = 0 (no delta-rule erase), t_inv = I, so each chunk adds k_dec_t @ v_beta.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    CHUNK_SIZE,
    to_device,
)

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]

_DIM = 128
# Per-chunk log decay G_last per class of 16 key rows: from far below BF16/TF32 resolution to strong.
_DECAY_CLASSES = (-1e-6, -1e-5, -1e-4, -(2.0**-10), -1e-3, -1e-2, -0.1, -1.0)
_ROWS_PER_CLASS = _DIM // len(_DECAY_CLASSES)
_STATE_RELATIVE_ERROR = 1e-2
_STATE_NORM_RATIO_ERROR = 1e-2
_OUTPUT_RELATIVE_ERROR = 2e-2


def _compute_config(device: ttnn.Device, fidelity: ttnn.MathFidelity) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


def _chain_protocol(heads: int, chunks: int, *, seed: int) -> tuple[torch.Tensor, ...]:
    """Seven-tensor scan protocol (production BF16 subset) isolating the decay path."""
    generator = torch.Generator().manual_seed(seed)
    base = (heads, chunks)
    v_beta = (0.08 * torch.randn(*base, CHUNK_SIZE, _DIM, generator=generator)).bfloat16()
    kd = torch.zeros(*base, CHUNK_SIZE, _DIM).bfloat16()
    q_decay = (0.06 * torch.randn(*base, CHUNK_SIZE, _DIM, generator=generator)).bfloat16()
    intra = torch.zeros(*base, CHUNK_SIZE, CHUNK_SIZE)
    k_dec_t = (0.025 * torch.randn(*base, _DIM, CHUNK_SIZE, generator=generator)).bfloat16()
    g_last = torch.tensor(_DECAY_CLASSES, dtype=torch.float64).repeat_interleave(_ROWS_PER_CLASS)
    final_decay = torch.expm1(g_last).reshape(1, 1, _DIM, 1).expand(*base, _DIM, 1).bfloat16()
    t_inv = torch.eye(CHUNK_SIZE).expand(*base, CHUNK_SIZE, CHUNK_SIZE).clone()
    return v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv


def _oracle(protocol: tuple[torch.Tensor, ...], state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 chain over the given (already quantized) protocol values. Returns (last-chunk output, final state)."""
    v_beta, _, q_decay, _, k_dec_t, final_decay, _ = (t.double() for t in protocol)
    state = state.double().clone()
    output = None
    for chunk in range(v_beta.shape[1]):
        output = torch.matmul(q_decay[:, chunk], state)  # intra = 0
        state = state + (state * final_decay[:, chunk] + torch.matmul(k_dec_t[:, chunk], v_beta[:, chunk]))
    return output, state


def _assert_state_classes(expected: torch.Tensor, actual: torch.Tensor, context: str) -> None:
    expected, actual = expected.double(), actual.double()
    failures = []
    for index, g_last in enumerate(_DECAY_CLASSES):
        rows = slice(index * _ROWS_PER_CLASS, (index + 1) * _ROWS_PER_CLASS)
        reference = expected[..., rows, :]
        error = float((actual[..., rows, :] - reference).norm() / reference.norm())
        norm_ratio = float(actual[..., rows, :].norm() / reference.norm())
        logger.info(f"{context} G_last={g_last:.1e}: state rel error {error:.3e}, norm ratio {norm_ratio:.5f}")
        if not error <= _STATE_RELATIVE_ERROR:
            failures.append(f"G_last={g_last:.1e} rel error {error:.3e} > {_STATE_RELATIVE_ERROR}")
        if not abs(norm_ratio - 1.0) <= _STATE_NORM_RATIO_ERROR:
            failures.append(f"G_last={g_last:.1e} norm ratio {norm_ratio:.5f}")
    assert not failures, f"{context}: " + "; ".join(failures)


def _assert_output(expected: torch.Tensor, actual: torch.Tensor, context: str) -> None:
    error = float((actual.double() - expected.double()).norm() / expected.double().norm())
    logger.info(f"{context}: last-chunk output rel error {error:.3e}")
    assert error <= _OUTPUT_RELATIVE_ERROR, f"{context}: last-chunk output rel error {error:.3e}"


def test_recurrent_chunk_scan_weak_decay_chain(device: ttnn.Device) -> None:
    """T4: one direct scan over 10240 chunks keeps every decay class, from 1e-6 to 1, against FP64."""
    chunks = 10240
    protocol = _chain_protocol(1, chunks, seed=4401)
    state = 0.04 * torch.randn(1, _DIM, _DIM, generator=torch.Generator().manual_seed(4402))
    expected_output, expected_state = _oracle(protocol, state)
    inputs = tuple(to_device(tensor, device) for tensor in protocol)
    state_tt = to_device(state, device)
    actual_start = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int64), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    output, final_state = ttnn.experimental.kda.recurrent_chunk_scan(
        *inputs,
        state_tt,
        tail_entry_states=state_tt,
        actual_start=actual_start,
        compute_kernel_config=_compute_config(device, ttnn.MathFidelity.HiFi3),
    )
    _assert_state_classes(expected_state, ttnn.to_torch(final_state), "direct scan 10240 chunks")
    _assert_output(expected_output, ttnn.to_torch(output)[:, -1], "direct scan 10240 chunks")


def test_grouped_summary_prefix_weak_decay_chain(device: ttnn.Device) -> None:
    """T5: summaries (A - I), the group prefix and the grouped scan chained over 64 calls of 8 x 20 chunks.

    Production fidelities (KDARecurrenceProgramConfig): summary HiFi4, affine prefix HiFi3, scan HiFi3. A HiFi2
    prefix truncates E to 7 significant bits in the matmul (SrcB) and measured 1.26% state error, norm ratio 1.012,
    for the |G_last| = 1e-3 class; HiFi3 and HiFi4 both measured 0.33% / 0.32% for that class, 0.42% worst class
    (tt_metal_tracker-g1b.7; tt_metal_tracker-g1b.4.15 .build-logs/g1b.4.15/device-e3-t5probe-20261006-001731.log).
    """
    groups, group_chunks, calls = 8, 20, 64
    protocol = _chain_protocol(groups, group_chunks, seed=4501)
    state = 0.04 * torch.randn(1, _DIM, _DIM, generator=torch.Generator().manual_seed(4502))
    # The same call input repeats; the oracle walks the groups of every call in order.
    sequential = tuple(t.reshape(1, groups * group_chunks, *t.shape[2:]) for t in protocol)
    expected_state = state
    expected_output = None
    for _ in range(calls):
        expected_output, expected_state = _oracle(sequential, expected_state)
    inputs = tuple(to_device(tensor, device) for tensor in protocol)
    state_tt = to_device(state, device)
    actual_start = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int64), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    summary_config = _compute_config(device, ttnn.MathFidelity.HiFi4)
    prefix_config = _compute_config(device, ttnn.MathFidelity.HiFi3)
    scan_config = _compute_config(device, ttnn.MathFidelity.HiFi3)
    output = None
    for _ in range(calls):
        a, b, tail_a, tail_b = ttnn.experimental.kda.summarize_chunk_recurrence(
            *inputs, groups_per_head=groups, actual_start=actual_start, compute_kernel_config=summary_config
        )
        entries = ttnn.experimental.kda.affine_exclusive_scan(
            a,
            b,
            state_tt,
            groups,
            local_rows=groups * group_chunks * CHUNK_SIZE,
            tail_a=a,
            tail_b=b,
            tail_entry_states=state_tt,
            actual_start=actual_start,
            compute_kernel_config=prefix_config,
        )
        if output is not None:
            ttnn.deallocate(output)
        output, final_states = ttnn.experimental.kda.recurrent_chunk_scan(
            *inputs,
            entries,
            groups_per_head=groups,
            tail_entry_states=state_tt,
            actual_start=actual_start,
            compute_kernel_config=scan_config,
        )
        last = ttnn.slice(final_states, (groups - 1, 0, 0), (groups, _DIM, _DIM))
        for tensor in (a, b, tail_a, tail_b, entries, final_states, state_tt):
            ttnn.deallocate(tensor)
        state_tt = last
    _assert_state_classes(expected_state, ttnn.to_torch(state_tt), "grouped 64 x 8 x 20 chunks")
    _assert_output(expected_output, ttnn.to_torch(output)[groups - 1 :, -1], "grouped 64 x 8 x 20 chunks")

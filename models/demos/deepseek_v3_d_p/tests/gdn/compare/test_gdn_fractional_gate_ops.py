# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-level accuracy of the two GDN recurrences under fractional FP32 gates of growing chunk decay (g1b.5.10).

The current op (``ttnn.transformer.chunk_gated_delta_rule``) builds the intra-chunk decay matrix as
``exp(G_i - G_j)`` from the running chunk sum ``G`` read through FPU broadcasts (``lmask_fused``), which keep about
10 mantissa bits of ``G``: emulated (tt-work artifacts/scripts/gdn_s4_form_emulation.py) the matrix error is
6.5e-3 at ``|G_last| = 16``, 0.1 at 160 and up to 1.0 at 8635 when one token carries most of the chunk's decay. The
GDN-on-KDA recurrence (``KDARecurrence(scalar_decay=True)``, prepare_chunk_recurrence S4 difference form) never forms a
large ``G`` operand. This measures both on identical inputs against FP64 token recurrences: Qwen3.8-27B TP4 heads
(4 K heads, 12 V heads, K = V = 128), 128 tokens (four chunks), per (chunk, V head) gate profiles ``spread`` (Dirichlet
split), ``spike`` (one token carries the rest of the decay) and ``const`` at ``|G_last|`` per 32 tokens from 1e-3 to
8635 (the largest measured on Qwen layer 0, g1b.5.12). Measurement, not a gate: it prints
``GDN_FRACTIONAL_GATE=<json>`` and asserts only finite results.

References: the current op reads q / k as given (normalized upstream, scale ``K^-0.5``); the KDA recurrence normalizes
q / k itself. Each implementation is compared with its own FP64 recurrence of exactly the bf16 values it receives.
"""

from __future__ import annotations

import json

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tt.kda import recurrence
from models.demos.deepseek_v3_d_p.tt.kda.config import KDARecurrenceProgramConfig
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.timeout(600)]

_CHUNK = 32
_TOKENS = 128
_KEY_HEADS, _VALUE_HEADS, _DIM = 4, 12, 128
_G_LAST = (1e-3, 16.0, 160.0, 1000.0, 2931.0, 8635.0)
_PROFILES = ("spread", "spike", "const")
_IMPLEMENTATIONS = ("current-fused", "current-phased", "kda-scalar")


def _chunk_gates(profile: str, g_last: float, generator: torch.Generator) -> torch.Tensor:
    """Log decays of one 32-token chunk summing to ``-g_last`` (FP32, fractional)."""
    if profile == "const":
        return torch.full((_CHUNK,), -g_last / _CHUNK)
    if profile == "spread":
        weights = torch.distributions.Dirichlet(torch.ones(_CHUNK)).sample()
        return -(g_last * weights)
    small = -0.5 * torch.rand(_CHUNK, generator=generator) * min(1.0, g_last / 16)
    spike = int(torch.randint(1, _CHUNK - 1, (1,), generator=generator))
    small[spike] = -(g_last - float(-small.sum()) + float(small[spike]))
    return small


def _gates(profile: str, g_last: float, seed: int) -> torch.Tensor:
    """``[1, T, HV]`` FP32 log decays, an independent profile draw per (chunk, V head)."""
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # Dirichlet draws use the global generator
    chunks = _TOKENS // _CHUNK
    gates = torch.stack(
        [torch.cat([_chunk_gates(profile, g_last, generator) for _ in range(chunks)]) for _ in range(_VALUE_HEADS)],
        dim=-1,
    )
    return gates.float()[None]


def _recurrence_fp64(q, k, v, gate, beta, scale):
    """Token-ordered gated delta rule, ``[B, T, HV, D]`` inputs (q / k already expanded to HV), FP64."""
    q, k, v, gate, beta = (t.double() for t in (q, k, v, gate, beta))
    batch, tokens, heads, key_dim = q.shape
    state = torch.zeros(batch, heads, key_dim, v.shape[-1], dtype=torch.float64)
    output = torch.empty(batch, tokens, heads, v.shape[-1], dtype=torch.float64)
    for t in range(tokens):
        state = state * gate[:, t].exp()[..., None, None]
        residual = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], beta[:, t, :, None] * residual)
        output[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, state)
    return output, state


def _metrics(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    expected, actual = expected.double().flatten(), actual.double().flatten()
    error = actual - expected
    rms = expected.pow(2).mean().sqrt()
    pcc = torch.corrcoef(torch.stack([expected, actual]))[0, 1]
    return {
        "pcc": float(pcc),
        "rel_rmse": float(error.pow(2).mean().sqrt() / rms),
        "max_abs_err_over_rms": float(error.abs().max() / rms),
    }


def _head_state_rel_rmse(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected, actual = expected.double(), actual.double()
    error = (actual - expected).pow(2).mean((-1, -2)).sqrt()
    scale = expected.pow(2).mean((-1, -2)).sqrt().clamp_min(1e-30)
    return float((error / scale).max())


def _const_tiles(device):
    """The op's constant tiles, as test_chunk_gated_delta_rule.py builds them (trace-safe explicit inputs)."""
    eye = torch.eye(_CHUNK)
    tril = torch.tril(torch.ones(_CHUNK, _CHUNK))
    ones = torch.ones(_CHUNK, _CHUNK)
    ii, jj = torch.arange(32).unsqueeze(1), torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)
    return tuple(
        ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        for t in (eye, tril, ones, masks)
    )


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


@pytest.mark.parametrize("implementation", _IMPLEMENTATIONS)
@pytest.mark.parametrize("profile", _PROFILES)
@pytest.mark.parametrize("g_last", _G_LAST, ids=lambda value: f"G{value:g}")
@pytest.mark.use_module_device
def test_gdn_fractional_gate_accuracy(device: ttnn.Device, implementation: str, profile: str, g_last: float) -> None:
    seed = 20261006
    generator = torch.Generator().manual_seed(seed)
    group = _VALUE_HEADS // _KEY_HEADS
    q = _l2(torch.randn(1, _TOKENS, _KEY_HEADS, _DIM, generator=generator)).bfloat16()
    k = _l2(torch.randn(1, _TOKENS, _KEY_HEADS, _DIM, generator=generator)).bfloat16()
    v = torch.randn(1, _TOKENS, _VALUE_HEADS, _DIM, generator=generator).bfloat16()
    beta = torch.sigmoid(torch.randn(1, _TOKENS, _VALUE_HEADS, generator=generator)).float()
    gate = _gates(profile, g_last, seed)

    def dev(tensor: torch.Tensor, dtype) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    if implementation.startswith("current"):
        # Fused is what the layer dispatches at the Qwen TP4 shapes (g1b.5.2); phased is the alternative path.
        program_config = (
            ttnn.ChunkGdnPhasedProgramConfig()
            if implementation == "current-phased"
            else ttnn.ChunkGdnFusedProgramConfig()
        )
        eye, tril, ones, masks = _const_tiles(device)
        output_tt, state_tt = ttnn.transformer.chunk_gated_delta_rule(
            dev(q, ttnn.bfloat16),
            dev(k, ttnn.bfloat16),
            dev(v, ttnn.bfloat16),
            dev(gate, ttnn.float32),
            dev(beta, ttnn.float32),
            output_final_state=True,
            chunk_size=_CHUNK,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
            program_config=program_config,
        )
        output = ttnn.to_torch(output_tt).float().reshape(1, _TOKENS, _VALUE_HEADS, _DIM)
        state = ttnn.to_torch(state_tt).float().reshape(1, _VALUE_HEADS, _DIM, _DIM)
        q_ref, k_ref = q.float(), k.float()
    else:
        executor = recurrence.KDARecurrence(
            device,
            KDARecurrenceProgramConfig(local_scan_strategy="direct"),
            sequence_parallel_axis=0,
            local_rows=_TOKENS,
            heads=_VALUE_HEADS,
            key_heads=_KEY_HEADS,
            key_dim=_DIM,
            value_dim=_DIM,
            scalar_decay=True,
        )
        start = make_actual_start(device, 0)
        result = executor(
            q=dev(q.reshape(1, _TOKENS, -1), ttnn.bfloat16),
            k=dev(k.reshape(1, _TOKENS, -1), ttnn.bfloat16),
            v=dev(v.reshape(1, _TOKENS, -1), ttnn.bfloat16),
            gate=dev(gate, ttnn.float32),
            beta=dev(beta, ttnn.float32),
            initial_state=dev(torch.zeros(1, _VALUE_HEADS, _DIM, _DIM), ttnn.float32),
            actual_start=start,
        )
        output = ttnn.to_torch(result.output).float().reshape(1, _VALUE_HEADS, _TOKENS, _DIM).permute(0, 2, 1, 3)
        state = ttnn.to_torch(result.final_state).float().reshape(1, _VALUE_HEADS, _DIM, _DIM)
        q_ref, k_ref = _l2(q.double()), _l2(k.double())
    expected_output, expected_state = _recurrence_fp64(
        q_ref.repeat_interleave(group, dim=2),
        k_ref.repeat_interleave(group, dim=2),
        v.float(),
        gate,
        beta,
        _DIM**-0.5,
    )
    chunk_sums = gate[0].reshape(_TOKENS // _CHUNK, _CHUNK, _VALUE_HEADS).sum(1)
    result = {
        "implementation": implementation,
        "profile": profile,
        "g_last": g_last,
        "max_abs_chunk_g_sum": float(chunk_sums.abs().max()),
        "max_abs_token_g": float(gate.abs().max()),
        "output": _metrics(expected_output, output),
        "state": _metrics(expected_state, state),
        "state_worst_head_rel_rmse": _head_state_rel_rmse(expected_state, state),
    }
    print("GDN_FRACTIONAL_GATE=" + json.dumps(result, sort_keys=True))
    assert torch.isfinite(output).all() and torch.isfinite(state).all()

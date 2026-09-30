# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""SDPAProgramConfig opt-in modes of the streaming SDPA kernel: per-phase math fidelity and fixed-offset softmax."""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_and_get_pcc

B, NH, S, D = 1, 8, 1024, 128
Q_CHUNK, K_CHUNK = 256, 512
# q and k are scaled by this, so the scaled logits have std ~ amp^2 and a block max near +20.
LOGIT_AMP = 2.0
# Two whole tile rows inside the second Q chunk, driven ~40 below the other rows in the dead-row test.
DEAD_ROWS = slice(320, 384)
DEAD_SHIFT = 40.0

blackhole_only = pytest.mark.skipif(
    not is_blackhole(), reason="fixed_offset is folded into the Blackhole exp macro only"
)


def make_inputs(amp, seed=0):
    """bf16-rounded fp32 q, k, v so the torch reference and the device see the same values."""
    torch.manual_seed(seed)
    q, k, v = torch.randn(B, NH, S, D) * amp, torch.randn(B, NH, S, D) * amp, torch.randn(B, NH, S, D)
    return tuple(t.bfloat16().float() for t in (q, k, v))


def scaled_logits(q, k):
    return (q @ k.transpose(-1, -2)) / D**0.5


def reference(q, k, v, is_causal=False):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def to_device(device, *tensors):
    return tuple(ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in tensors)


def program_config(device, **modes):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=Q_CHUNK,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
        **modes,
    )


def run_sdpa(device, tensors, pc, is_causal=False, fidelity=ttnn.MathFidelity.HiFi2):
    """Streaming path (fp32_dest_acc_en=False); returns the fp32 host copy of the bf16 output."""
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=False
    )
    tq, tk, tv = tensors
    out = ttnn.transformer.scaled_dot_product_attention(
        tq, tk, tv, is_causal=is_causal, program_config=pc, compute_kernel_config=compute_kernel_config
    )
    return ttnn.to_torch(out).float()


def assert_pcc(ref, out, threshold):
    passing, msg, pcc = comp_and_get_pcc(ref, out, threshold)
    assert passing, msg
    return float(pcc)


@pytest.mark.parametrize(
    "qk_fidelity, pv_fidelity, min_pcc",
    [
        (None, None, 0.999),
        (ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi, 0.9994),
        (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, 0.9990),
        (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.LoFi, 0.998),
    ],
    ids=["compute-config", "hifi2-qk-lofi-pv", "lofi-qk-hifi2-pv", "lofi-both"],
)
def test_sdpa_per_phase_math_fidelity(device, qk_fidelity, pv_fidelity, min_pcc):
    """A per-phase fidelity overrides the HiFi2 compute config for QK^T and/or PV; None keeps it."""
    q, k, v = make_inputs(1.0)
    ref = reference(q, k, v)
    pc = program_config(device, qk_math_fidelity=qk_fidelity, pv_math_fidelity=pv_fidelity)
    out = run_sdpa(device, to_device(device, q, k, v), pc)
    assert torch.isfinite(out).all(), "SDPA returned nonfinite values"
    assert_pcc(ref, out, min_pcc)


def test_sdpa_per_phase_math_fidelity_program_cache(device):
    """Each distinct fidelity pair compiles its own program; repeating one hits the cache."""
    device.enable_program_cache()
    tensors = to_device(device, *make_inputs(1.0))
    configs = [
        program_config(device),
        program_config(device, pv_math_fidelity=ttnn.MathFidelity.LoFi),
        program_config(device, qk_math_fidelity=ttnn.MathFidelity.LoFi),
        program_config(device, qk_math_fidelity=ttnn.MathFidelity.LoFi, pv_math_fidelity=ttnn.MathFidelity.LoFi),
    ]
    entries = device.num_program_cache_entries()
    for pc in configs:
        run_sdpa(device, tensors, pc)
        entries += 1
        assert device.num_program_cache_entries() == entries, f"{pc} did not add exactly one program cache entry"
    for pc in configs:
        run_sdpa(device, tensors, pc)
    assert device.num_program_cache_entries() == entries, "Repeated configs must hit the program cache"


@blackhole_only
@pytest.mark.parametrize("offset_above_max", [5.0, 40.0], ids=["max-plus-5", "max-plus-40"])
def test_sdpa_fixed_offset_softmax_matches_standard(device, offset_above_max):
    """With the offset at or above the max scaled logit the fixed mode tracks the running-max path."""
    q, k, v = make_inputs(LOGIT_AMP)
    logit_max = scaled_logits(q, k).max().item()
    assert 15.0 < logit_max < 30.0, f"fixture drifted: max scaled logit {logit_max:.2f}"
    ref = reference(q, k, v)
    tensors = to_device(device, q, k, v)

    standard_pcc = assert_pcc(ref, run_sdpa(device, tensors, program_config(device)), 0.999)
    pc = program_config(device, fixed_offset_softmax=True, fixed_offset=logit_max + offset_above_max)
    out = run_sdpa(device, tensors, pc)
    assert torch.isfinite(out).all(), "fixed-offset SDPA returned nonfinite values"
    fixed_pcc = assert_pcc(ref, out, 0.999)
    assert abs(fixed_pcc - standard_pcc) <= 0.001, f"fixed PCC {fixed_pcc:.5f} vs standard {standard_pcc:.5f}"


@blackhole_only
def test_sdpa_fixed_offset_softmax_dead_rows(device):
    """Rows whose max logit sits far below the offset come out as zeros, not NaN; other rows are unaffected."""
    q, k, v = make_inputs(LOGIT_AMP)
    # Feature 0 carries a per-row constant shift: k0 = 1 everywhere, q0 = -DEAD_SHIFT * sqrt(D) on the dead rows.
    k[..., 0] = 1.0
    q[..., 0] = 0.0
    q[:, :, DEAD_ROWS, 0] = -DEAD_SHIFT * D**0.5
    q = q.bfloat16().float()
    row_max = scaled_logits(q, k).amax(-1)
    fixed_offset = row_max.max().item() + 60.0
    live = torch.ones(S, dtype=torch.bool)
    live[DEAD_ROWS] = False
    assert (row_max[:, :, ~live] < fixed_offset - 87).all(), "fixture drifted: a dead row is above the exp floor"
    assert (row_max[:, :, live] >= fixed_offset - 80).all(), "fixture drifted: a live row is near the exp floor"
    ref = reference(q, k, v)

    pc = program_config(device, fixed_offset_softmax=True, fixed_offset=fixed_offset)
    out = run_sdpa(device, to_device(device, q, k, v), pc)
    assert torch.isfinite(out).all(), "fixed-offset SDPA returned nonfinite values"
    zero_rows = out.abs().amax(-1) == 0
    assert zero_rows[:, :, ~live].all(), "every dead row must come out as zeros"
    assert not zero_rows[:, :, live].any(), "no live row may be zeroed"
    assert_pcc(ref[:, :, live], out[:, :, live], 0.999)


@blackhole_only
def test_sdpa_fixed_offset_softmax_causal(device):
    """Causal chunks stamp the mask and take the fixed mode's fallback pass; the result must still match torch."""
    q, k, v = make_inputs(LOGIT_AMP)
    fixed_offset = scaled_logits(q, k).max().item() + 5.0
    ref = reference(q, k, v, is_causal=True)
    pc = program_config(device, fixed_offset_softmax=True, fixed_offset=fixed_offset)
    out = run_sdpa(device, to_device(device, q, k, v), pc, is_causal=True)
    assert torch.isfinite(out).all(), "fixed-offset causal SDPA returned nonfinite values"
    assert_pcc(ref, out, 0.999)


@blackhole_only
def test_sdpa_fixed_offset_softmax_far_negative_logits(device):
    """Every scaled logit in [-1, 1] with offset 30: the exp only ever sees inputs near -30."""
    q, k, v = make_inputs(0.3)
    assert scaled_logits(q, k).abs().max().item() <= 1.0, "fixture drifted: logits leave [-1, 1]"
    ref = reference(q, k, v)
    pc = program_config(device, fixed_offset_softmax=True, fixed_offset=30.0)
    out = run_sdpa(device, to_device(device, q, k, v), pc)
    assert torch.isfinite(out).all(), "fixed-offset SDPA returned nonfinite values"
    assert_pcc(ref, out, 0.999)


def test_sdpa_fixed_offset_requires_fixed_offset_softmax(device):
    """A non-zero fixed_offset without fixed_offset_softmax is rejected by the program factory."""
    tensors = to_device(device, *make_inputs(1.0))
    pc = program_config(device, fixed_offset=1.0)
    with pytest.raises(
        RuntimeError, match="fixed_offset requires fixed_offset_softmax"
    ):  # allow-pytest.raises: the TT_FATAL text is the contract
        run_sdpa(device, tensors, pc)

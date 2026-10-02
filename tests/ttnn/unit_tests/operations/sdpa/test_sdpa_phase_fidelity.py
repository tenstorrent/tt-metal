# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""SDPAProgramConfig.qk_math_fidelity / pv_math_fidelity: per-phase matmul fidelity of the streaming SDPA kernel."""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_and_get_pcc

B, NH, S, D = 1, 8, 1024, 128
Q_CHUNK, K_CHUNK = 256, 512


def make_inputs():
    """bf16-rounded fp32 q, k, v so the torch reference and the device see the same values."""
    torch.manual_seed(0)
    q, k, v = torch.randn(B, NH, S, D), torch.randn(B, NH, S, D), torch.randn(B, NH, S, D)
    return tuple(t.bfloat16().float() for t in (q, k, v))


def to_device(device, *tensors):
    return tuple(ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in tensors)


def program_config(device, **fidelity):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=Q_CHUNK,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
        **fidelity,
    )


def run_sdpa(device, tensors, pc):
    """HiFi2 compute config on the streaming path, non-causal like the torch reference; fp32 host copy of the output."""
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    tq, tk, tv = tensors
    out = ttnn.transformer.scaled_dot_product_attention(
        tq, tk, tv, is_causal=False, program_config=pc, compute_kernel_config=compute_kernel_config
    )
    return ttnn.to_torch(out).float()


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
    q, k, v = make_inputs()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    pc = program_config(device, qk_math_fidelity=qk_fidelity, pv_math_fidelity=pv_fidelity)
    out = run_sdpa(device, to_device(device, q, k, v), pc)
    assert torch.isfinite(out).all(), "SDPA returned nonfinite values"
    passing, msg, _ = comp_and_get_pcc(ref, out, min_pcc)
    assert passing, msg


def test_sdpa_per_phase_math_fidelity_program_cache(device):
    """Each distinct fidelity pair compiles its own program; repeating one hits the cache."""
    device.enable_program_cache()
    tensors = to_device(device, *make_inputs())
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

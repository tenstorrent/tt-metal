# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Legacy dense-SDPA compatibility checks, not qualification of the new recipes."""

import hashlib
import math

import pytest
import torch

import ttnn


def _inputs(device, seed):
    generator = torch.Generator().manual_seed(seed)
    host = [torch.randn((1, 2, 1024, 128), generator=generator).bfloat16() for _ in range(3)]
    tensors = [
        ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG) for x in host
    ]
    return host, tensors


def _program(exp_approx, fp32_dst):
    kwargs = {} if exp_approx is None else {"exp_approx_mode": exp_approx}
    # The legacy non-streaming FP32 path exceeds Blackhole L1 at Q256/K512.
    # These are compatibility checks, not a matched-geometry performance test.
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        q_chunk_size=128 if fp32_dst else 256,
        k_chunk_size=512,
        **kwargs,
    )


def _run(inputs, config, exp_approx):
    return ttnn.transformer.scaled_dot_product_attention(
        *inputs,
        is_causal=False,
        program_config=_program(exp_approx, config is not None and config.fp32_dest_acc_en),
        compute_kernel_config=config,
    )


def test_sdpa_python_empty_config_contract():
    # Unlike C++ ComputeKernelConfig{}, the legacy Python binding constructs
    # MathFidelity.Invalid. Do not launch it or silently reinterpret it as LoFi.
    assert ttnn.WormholeComputeKernelConfig().math_fidelity == ttnn.MathFidelity.Invalid


@pytest.mark.parametrize("explicit_lofi", [False, True], ids=["omitted", "explicit-lofi"])
def test_sdpa_legacy_defaults(device, explicit_lofi, record_property):
    """Omitted SDPA config means HiFi2; an explicitly selected LoFi stays LoFi."""
    _, inputs = _inputs(device, 20260921)
    actual_config = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi) if explicit_lofi else None
    equivalent_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi if explicit_lofi else ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    actual = _run(inputs, actual_config, None)
    explicit = _run(inputs, equivalent_config, True)
    actual_host = ttnn.to_torch(actual)
    assert torch.equal(actual_host, ttnn.to_torch(explicit))
    record_property("output_sha256", hashlib.sha256(actual_host.contiguous().view(torch.uint8).numpy()).hexdigest())


@pytest.mark.parametrize("fp32_dst", [False, True])
@pytest.mark.parametrize("math_approx", [False, True])
@pytest.mark.parametrize("exp_approx", [False, True])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16777216}], indirect=True)
def test_sdpa_legacy_numerics_cache_and_trace(device, fp32_dst, math_approx, exp_approx, record_property):
    """Independent knobs survive cache hits, fresh addresses and actual trace replay."""
    device.enable_program_cache()
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=math_approx,
        fp32_dest_acc_en=fp32_dst,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    # Keep the first set alive so the second call must update input addresses.
    _, first_inputs = _inputs(device, 20260921)
    first_output = _run(first_inputs, config, exp_approx)
    ttnn.to_torch(first_output)
    host, inputs = _inputs(device, 20260922)
    entries = device.num_program_cache_entries()
    output = _run(inputs, config, exp_approx)
    actual = ttnn.to_torch(output)
    record_property("output_sha256", hashlib.sha256(actual.contiguous().view(torch.uint8).numpy()).hexdigest())
    assert device.num_program_cache_entries() == entries
    assert torch.isfinite(actual).all()

    q, k, v = [x.double() for x in host]
    reference = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1) @ v
    relative_l2 = (torch.linalg.vector_norm(actual.double() - reference) / torch.linalg.vector_norm(reference)).item()
    pcc = torch.corrcoef(torch.stack([actual.double().flatten(), reference.flatten()]))[0, 1].item()
    record_property("relative_l2", relative_l2)
    record_property("pcc", pcc)
    # Broad legacy smoke thresholds; these are NOT the frozen recipe accuracy gates.
    assert relative_l2 < 0.1
    assert pcc > 0.99

    trace = ttnn.begin_trace_capture(device, cq_id=0)
    traced_output = _run(inputs, config, exp_approx)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert torch.equal(ttnn.to_torch(traced_output), actual)
        for original, tensor in zip(host, inputs):
            assert torch.equal(ttnn.to_torch(tensor), original)
    finally:
        ttnn.release_trace(device, trace)

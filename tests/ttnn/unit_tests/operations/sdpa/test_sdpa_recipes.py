# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Production SDPA recipe contracts, independent of the research adapter."""

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import load_baseline, VARIANTS, digest, make_inputs, metrics, prepare, reference, run


@pytest.mark.parametrize("precision", ["FAST", "COMPENSATED", "BALANCED", "ACCURATE"])
@pytest.mark.parametrize("k_length", [512, 1024, 1536])
def test_sdpa_recipe_smoke(device, precision, k_length, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    generator = torch.Generator().manual_seed(20260919)
    q = torch.randn((1, 1, 256, 128), generator=generator).bfloat16()
    k = torch.randn((1, 1, k_length, 128), generator=generator).bfloat16()
    v = torch.randn(k.shape, generator=generator).bfloat16()
    tensors = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (q, k, v)]
    output = ttnn.transformer.scaled_dot_product_attention(
        *tensors, is_causal=False, precision=getattr(ttnn.transformer.SDPAPrecision, precision)
    )
    actual = ttnn.to_torch(output).double()
    reference = torch.softmax(q.double() @ k.double().transpose(-1, -2) / 128**0.5, -1) @ v.double()
    l2 = ((actual - reference).norm() / reference.norm()).item()
    pcc = torch.corrcoef(torch.stack([actual.flatten(), reference.flatten()]))[0, 1].item()
    record_property("relative_l2", l2)
    record_property("pcc", pcc)
    assert torch.isfinite(actual).all()
    assert l2 < {"FAST": 0.06, "COMPENSATED": 0.04, "BALANCED": 0.01, "ACCURATE": 0.003}[precision]
    assert pcc > 0.995


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("distribution", ["normal", "uniform", "changed_max", "constant_v", "zero_v"])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 4194304}], indirect=True)
def test_sdpa_recipe_state_cache_trace(device, variant, distribution, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    device.enable_program_cache()
    # Three jobs/head on two cores/head makes unequal chain lengths. Odd K3
    # exercises the compensated final-group flush and per-query state reset.
    host = make_inputs(1536, distribution, q_length=768, heads=2)
    expected = reference(*host)
    inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host]
    prepared = prepare(inputs, variant)
    hashes = [digest(ttnn.to_torch(x)) for x in prepared]
    output = run(prepared, variant, cores=4)
    actual = ttnn.to_torch(output)
    observed = metrics(actual, expected)
    for name, value in observed.items():
        record_property(name, value)
    if distribution == "zero_v":
        assert observed["max_abs"] <= 1e-6
    else:
        # Boundary smoke limits. The wider frozen suite owns the tighter
        # per-distribution regression budgets, including visible stress cases.
        limits = {"A": 8, "B": 8, "C": 2, "D": 0.4, "E_bf16": 8, "E_bfp8": 8, "E_bfp4": 35}
        assert observed["l2_pct"] < limits[variant]
    entries = device.num_program_cache_entries()
    second_inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in (host[0], host[1], -host[2])]
    second_prepared = prepare(second_inputs, variant)
    second_output = run(second_prepared, variant, cores=4)
    assert inputs[0].buffer_address() != second_inputs[0].buffer_address()
    assert device.num_program_cache_entries() == entries
    # Negating V is an independent address-sensitive oracle. Compare with
    # its FP64 reference instead of assuming signed-zero or round-tie symmetry.
    second_metrics = metrics(ttnn.to_torch(second_output), -expected)
    assert second_metrics["max_abs"] <= max(1e-6, 1.05 * observed["max_abs"])
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    traced = run(prepared, variant, cores=4)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert digest(ttnn.to_torch(traced)) == digest(actual)
    finally:
        ttnn.release_trace(device, trace)
    assert [digest(ttnn.to_torch(x)) for x in inputs] == [digest(x) for x in host]
    assert [digest(ttnn.to_torch(x)) for x in prepared] == hashes


@pytest.mark.parametrize(
    "invalid",
    [
        "causal",
        "mask",
        "sink",
        "sliding",
        "compute",
        "exp",
        "chunks",
        "subgrid",
        "scale",
        "scale_bf16",
        "scale_nan",
        "scale_inf",
        "prepared",
        "unprepared_e",
        "kv_type",
        "dimension",
        "output_l1",
    ],
)
def test_sdpa_recipe_rejects_unsupported(device, invalid):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = make_inputs(512, "normal")
    if invalid == "dimension":
        host = [x[..., :64].contiguous() for x in host]
    inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host]
    kwargs = dict(is_causal=False, precision=ttnn.SDPAPrecision.ACCURATE)
    if invalid == "causal":
        kwargs["is_causal"] = True
    elif invalid == "mask":
        kwargs["attn_mask"] = inputs[0]
    elif invalid == "sink":
        kwargs["attention_sink"] = inputs[0]
    elif invalid == "sliding":
        kwargs["sliding_window_size"] = 512
    elif invalid == "compute":
        kwargs["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)
    elif invalid in ("exp", "chunks", "subgrid"):
        cfg = dict(compute_with_storage_grid_size=(1, 1), q_chunk_size=256, k_chunk_size=512)
        if invalid == "exp":
            cfg["exp_approx_mode"] = False
        elif invalid == "chunks":
            # Q128-Q320 and K256/K384/K512 are supported.
            cfg["k_chunk_size"] = 1024
        else:
            cfg["sub_core_grids"] = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        kwargs["program_config"] = ttnn.SDPAProgramConfig(**cfg)
    elif invalid == "scale":
        kwargs["scale"] = 0.5
    elif invalid.startswith("scale_"):
        kwargs["scale"] = {
            "scale_bf16": torch.tensor(128**-0.5, dtype=torch.bfloat16).item(),
            "scale_nan": float("nan"),
            "scale_inf": float("inf"),
        }[invalid]
    elif invalid == "prepared":
        kwargs["inputs_prepared"] = True
    elif invalid == "unprepared_e":
        kwargs["precision"] = ttnn.SDPAPrecision.LOW_PRECISION
    elif invalid == "kv_type":
        inputs[1] = ttnn.from_torch(host[1], device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    elif invalid == "output_l1":
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
    before = device.num_program_cache_entries()
    with pytest.raises(RuntimeError, match="SDPA|recipe|precision"):
        ttnn.transformer.scaled_dot_product_attention(*inputs, **kwargs)
    assert device.num_program_cache_entries() == before, "Unsupported recipes must fail before device dispatch"


@pytest.mark.parametrize("k_length", [512, 1536, 32768])
def test_sdpa_fast_matches_legacy_streaming(device, k_length):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    host = make_inputs(k_length, "normal", q_length=768, heads=5)
    inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in host]
    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(5, 2), q_chunk_size=256, k_chunk_size=512)
    baseline = ttnn.transformer.scaled_dot_product_attention(*inputs, is_causal=False, program_config=cfg)
    named = ttnn.transformer.scaled_dot_product_attention(
        *inputs, is_causal=False, program_config=cfg, precision=ttnn.SDPAPrecision.FAST
    )
    assert digest(ttnn.to_torch(named)) == digest(ttnn.to_torch(baseline))


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4194304}], indirect=True)
def test_sdpa_recipe_switching_cache_identity(device, record_property):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    device.enable_program_cache()
    baseline = load_baseline()
    case = next(c for c in baseline["cases"] if c["k_length"] == 4096 and c["distribution"] == "normal")
    inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in make_inputs(4096, "normal")]
    prepared = {variant: prepare(inputs, variant) for variant in VARIANTS}
    outputs = {variant: run(prepared[variant], variant) for variant in VARIANTS}
    expected = reference(*make_inputs(4096, "normal"))
    for variant, output in outputs.items():
        actual = ttnn.to_torch(output)
        frozen = case["variants"][variant]
        record_property(f"{variant}_frozen_output_equal", digest(actual) == frozen["output_sha256"])
        assert metrics(actual, expected)["l2_pct"] <= 1.05 * frozen["metrics"]["l2_pct"] + 0.0001
    entries = device.num_program_cache_entries()
    # Same shapes and addresses, reversed recipe order: numerical policy and
    # prepared KV format must distinguish cached programs, not just tensor shape.
    for variant in reversed(VARIANTS):
        actual = run(prepared[variant], variant)
        assert digest(ttnn.to_torch(actual)) == digest(ttnn.to_torch(outputs[variant]))
    assert device.num_program_cache_entries() == entries
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    traced = {variant: run(prepared[variant], variant) for variant in VARIANTS}
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            for variant, output in traced.items():
                assert digest(ttnn.to_torch(output)) == digest(ttnn.to_torch(outputs[variant]))
    finally:
        ttnn.release_trace(device, trace)


@pytest.mark.parametrize("scale", [128**-0.5, torch.tensor(128**-0.5, dtype=torch.float32).item()])
def test_sdpa_recipe_default_scale(device, scale):
    if not is_blackhole():
        pytest.skip("Named recipes initially target Blackhole")
    inputs = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in make_inputs(512, "normal")]
    kwargs = dict(is_causal=False, precision=ttnn.SDPAPrecision.ACCURATE)
    expected = ttnn.transformer.scaled_dot_product_attention(*inputs, **kwargs)
    actual = ttnn.transformer.scaled_dot_product_attention(*inputs, scale=scale, **kwargs)
    assert digest(ttnn.to_torch(actual)) == digest(ttnn.to_torch(expected))

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from ttnn._experimental.auto_config import _install as install_ops
from ttnn._experimental.auto_config import matmul as auto_matmul

from models.common.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_numeric_metrics

pytestmark = pytest.mark.use_module_device


@pytest.fixture(autouse=True)
def clear_runtime_records():
    auto_matmul.AutoMatmulCache.clear_runtime()
    yield
    auto_matmul.AutoMatmulCache.clear_runtime()


def _ok_candidate_entries(result):
    return [entry for entry in result["candidate_timings_us"] if entry["status"] == "ok"]


def test_install_passthrough_wrapper_preserves_doc_and_golden_function():
    def wrapper():
        return None

    wrapped = install_ops._install_passthrough_wrapper(
        wrapper,
        doc="passthrough-doc",
        golden_function=_ok_candidate_entries,
    )

    assert wrapped is wrapper
    assert wrapped.__doc__ == "passthrough-doc"
    assert wrapped.golden_function is _ok_candidate_entries


def test_matmul_wrapper_impl_prefers_queue_id_over_cq_id(monkeypatch):
    import ttnn._experimental.auto_config.matmul as auto_matmul_module

    captured_kwargs = {}

    def fake_dispatch_matmul(**kwargs):
        captured_kwargs.update(kwargs)
        return "result"

    monkeypatch.setattr(auto_matmul_module, "dispatch_matmul", fake_dispatch_matmul)

    result = install_ops._matmul_wrapper_impl("lhs", "rhs", queue_id=3, cq_id=7)

    assert result == "result"
    assert captured_kwargs["queue_id"] == 3
    assert "cq_id" not in captured_kwargs


def test_linear_auto_config_accepts_host_rhs_and_bias_when_disabled(device):
    torch.manual_seed(0)

    input_shape_a = (1, 1, 32, 64)
    input_shape_b = (64, 64)
    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    torch_bias = torch_random((64,), -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.nn.functional.linear(
        torch_input_tensor_a,
        torch_input_tensor_b.T.contiguous(),
        bias=torch_bias,
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn.linear(
        input_tensor_a,
        torch_input_tensor_b,
        bias=torch_bias,
        auto_config=False,
        dtype=ttnn.bfloat16,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        atol=0.001 * input_shape_b[0],
        rtol=0.016 * input_shape_b[0],
        frobenius_threshold=0.001 * input_shape_b[0],
        pcc_threshold=0.999,
        check_ulp=False,
    )


def test_explain_matmul_force_retune_picks_fastest_candidate_and_reuses_cache(monkeypatch, tmp_path, device):
    monkeypatch.setenv("TTNN_AUTO_MATMUL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TTNN_AUTO_MATMUL_VERSION", "unit-test-version")
    monkeypatch.setenv("TTNN_AUTO_MATMUL_FORCE_RETUNE", "1")

    torch.manual_seed(0)
    torch_input_tensor_a = torch_random((1, 1, 32, 64), -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random((64, 64), -0.1, 0.1, dtype=torch.float32)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    tuned = ttnn.experimental.auto_config.explain_matmul(
        input_tensor_a,
        input_tensor_b,
        dtype=ttnn.bfloat16,
        allow_tuning=True,
    )

    ok_candidates = _ok_candidate_entries(tuned)
    assert ok_candidates, "Expected at least one successfully measured candidate"

    winner_entry = next(entry for entry in ok_candidates if entry["descriptor"] == tuned["winner"])
    winner_average_us = winner_entry["average_us"]
    assert winner_average_us == min(entry["average_us"] for entry in ok_candidates)

    default_entries = [entry for entry in ok_candidates if entry["descriptor"]["kind"] == "default_matmul"]
    if default_entries:
        assert default_entries[0]["average_us"] >= winner_average_us

    monkeypatch.delenv("TTNN_AUTO_MATMUL_FORCE_RETUNE", raising=False)
    cached = ttnn.experimental.auto_config.explain_matmul(
        input_tensor_a,
        input_tensor_b,
        dtype=ttnn.bfloat16,
        allow_tuning=False,
    )

    assert cached["cache_hit"] is True
    assert cached["winner"] == tuned["winner"]


def test_place_weight_single_device_places_host_weight_and_matmul_is_correct(device):
    # The reviewer ask: hand the module a host torch weight and let it place the
    # weight on device.  On a single device that is a replicated placement, and
    # the returned tensor must be a correct drop-in for ttnn.linear.
    torch.manual_seed(0)
    torch_input_tensor_a = torch_random((1, 1, 32, 64), -0.1, 0.1, dtype=torch.float32)
    torch_weight = torch_random((64, 128), -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_input_tensor_a @ torch_weight

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    placement = ttnn.experimental.auto_config.place_weight(torch_weight, activation=input_tensor_a)

    assert placement.strategy == "replicate"
    assert placement.output_is_sharded is False
    assert isinstance(placement.tensor, ttnn.Tensor)

    output_tensor = ttnn.to_torch(ttnn.matmul(input_tensor_a, placement.tensor))

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        atol=0.001 * 64,
        rtol=0.016 * 64,
        frobenius_threshold=0.001 * 64,
        pcc_threshold=0.999,
        check_ulp=False,
    )


# (label, K, N) for the core GPT-OSS GEMMs plus a large generic shape. GPT-OSS
# (both 20b and 120b) uses hidden=2880, heads*head_dim=64*64=4096, and a fused
# QKV width of 4096 + 512 + 512 = 5120.
_MODEL_SHAPES = [
    ("qkv_proj", 2880, 5120),
    ("o_proj", 4096, 2880),
    ("mlp_w", 2880, 2880),
    ("generic_4k", 4096, 4096),
]
# M = batch * seq: a decode-like and a prefill-like token count.
_M_VALUES = [32, 512]
# (name, dtype, correctness pcc threshold)
_DTYPES = [("bf16", ttnn.bfloat16, 0.99), ("bf8", ttnn.bfloat8_b, 0.97)]


def _measured_default_and_winner(result):
    """Extract the measured default candidate and the selected winner from an explain report.

    Both are measured in the same on-device session, so ``winner_us <= default_us`` is a
    consistency invariant (the winner is the fastest measured candidate) and the ratio is
    the per-shape default-vs-auto-tuned speedup.
    """
    ok = [entry for entry in result["candidate_timings_us"] if entry["status"] == "ok"]
    winner = next((entry for entry in ok if entry["descriptor"] == result["winner"]), None)
    default = next((entry for entry in ok if str(entry["descriptor"].get("kind", "")).startswith("default")), None)
    return default, winner, ok


@pytest.mark.skipif(
    auto_matmul._tuning_bypassed_by_environment(),
    reason="Measured auto-config tuning needs fast-dispatch hardware: it is invalid in slow dispatch/profiler-sync "
    "and unrunnable on the functional simulator (op timeout). Runs on fast-dispatch hardware CI.",
)
@pytest.mark.parametrize("dname, dtype, pcc", _DTYPES, ids=lambda p: p if isinstance(p, str) else "")
@pytest.mark.parametrize("label, K, N", _MODEL_SHAPES)
@pytest.mark.parametrize("M", _M_VALUES)
@pytest.mark.parametrize("is_linear", [False, True], ids=["matmul", "linear"])
def test_auto_config_model_shapes_correct_and_not_slower_than_default(
    device, M, label, K, N, dname, dtype, pcc, is_linear
):
    """Exhaustive matmul/linear matrix over real GPT-OSS shapes.

    For every shape/dtype/op it (1) checks the public auto-config entrypoint against a
    torch golden and (2) proves via a single ``explain_matmul`` report that the selected
    winner is never slower than the plain default candidate, printing the measured
    default-vs-auto-tuned timings so the per-shape speedup is visible.
    """
    torch.manual_seed(0)
    torch_a = torch_random((1, 1, M, K), -0.1, 0.1, dtype=torch.float32)
    torch_b = torch_random((K, N), -0.1, 0.1, dtype=torch.float32)
    torch_bias = torch_random((N,), -0.1, 0.1, dtype=torch.float32) if is_linear else None

    golden = torch_a @ torch_b
    if torch_bias is not None:
        golden = golden + torch_bias

    a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(torch_b, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = (
        ttnn.from_torch(torch_bias.reshape(1, N), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        if torch_bias is not None
        else None
    )

    # (1) Correctness through the public auto-config entrypoint.
    if is_linear:
        out = ttnn.to_torch(ttnn.linear(a, b, bias=tt_bias, dtype=dtype))
    else:
        out = ttnn.to_torch(ttnn.matmul(a, b, dtype=dtype))
    assert_numeric_metrics(golden, out, pcc_threshold=pcc, check_allclose=False, check_frobenius=False, check_ulp=False)

    # (2) Measured default vs auto-tuned, from one benchmarking session.
    result = ttnn.experimental.auto_config.explain_matmul(
        a, b, bias=tt_bias, is_linear=is_linear, dtype=dtype, allow_tuning=True
    )
    default, winner, ok = _measured_default_and_winner(result)
    assert winner is not None, "selector returned no measured winner"
    assert ok, "expected at least one successfully measured candidate"

    if default is not None:
        # Winner is the fastest measured candidate, so it must not be slower than default.
        assert winner["average_us"] <= default["average_us"] + 1e-9
        speedup = default["average_us"] / winner["average_us"]
        print(
            f"PERF {label} M={M} K={K} N={N} dtype={dname} op={'linear' if is_linear else 'matmul'} "
            f"default_us={default['average_us']:.3f} tuned_us={winner['average_us']:.3f} "
            f"speedup={speedup:.3f}x winner={result['winner'].get('kind')}"
        )

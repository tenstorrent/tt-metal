# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from tt_bfp_quant import factor_hessian, gptq_search, search_linear, search_packed, to_bf16_exact, validate_repacking
from tt_bfp_quant.native import build_native
from tt_bfp_quant.rounding import search_numpy


@pytest.fixture(scope="module")
def native_backend(tmp_path_factory, request):
    # Opt in to compiler-dependent tests; pure Python installs still test CPU math.
    import os

    if os.environ.get("TT_BFP_TEST_NATIVE") != "1":
        pytest.skip("set TT_BFP_TEST_NATIVE=1 to compile and check the native backend")
    return build_native(openmp="auto")


def assert_bits(a, b):
    np.testing.assert_array_equal(a.numpy().view(np.uint32), b.numpy().view(np.uint32))


def test_known_clipping_win_and_ties():
    x = np.array([[1.001] + [0.124] * 15], dtype=np.float32)
    q, counts = search_numpy(x, 4)
    assert counts == {"0": 0, "-1": 1}
    np.testing.assert_array_equal(q, [[0.875] + [0.125] * 15])
    zero, counts = search_numpy(np.zeros((1, 16), np.float32), 4)
    assert counts == {"0": 1, "-1": 0}
    assert (zero == 0).all()


def test_rne_ties_saturation_and_denormals():
    x = np.array(
        [[1.0, 0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875, -1.875, 0.0, -0.0, 1e-40, -1e-40, 0.25, -0.25]],
        np.float32,
    )
    actual, _ = search_numpy(x, 4, (0,))
    expected = [[1.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 1.75, -1.75, 0.0, -0.0, 0.0, -0.0, 0.25, -0.25]]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("deltas", [(0,), (0, -1), (0, -1, -2), (0, 1), (0, -1, 1, -2)])
def test_native_matches_numpy(native_backend, bits, deltas):
    rng = np.random.default_rng(50)
    # More than 8192 groups also exercises OpenMP on Linux when available.
    a = np.ldexp(rng.normal(size=(2049, 80)).astype(np.float32), rng.integers(-30, 30, size=(2049, 80)))
    a[:3] = 0
    a[1, 0] = -0.0
    expected, ei = search_packed(torch.from_numpy(a), bits, deltas, backend="numpy", row_chunk=2049)
    actual, ai = search_packed(torch.from_numpy(a), bits, deltas, backend="native", threads=2, row_chunk=2049)
    assert_bits(actual, expected)
    assert ai["group_choices"] == ei["group_choices"]


@pytest.mark.parametrize("bits", [4, 8])
def test_search_never_worse_by_group_and_repack(bits):
    torch.manual_seed(5)
    x = torch.randn(33, 47)
    base, _ = search_packed(x, bits, (0,), backend="numpy")
    q, _ = search_packed(x, bits, backend="numpy")
    for start in range(0, 47, 16):
        naive = (x[:, start : start + 16].double() - base[:, start : start + 16]).square().sum(-1)
        improved = (x[:, start : start + 16].double() - q[:, start : start + 16]).square().sum(-1)
        assert torch.all(improved <= naive)
    validate_repacking(q, bits, layout="packed")
    assert torch.equal(to_bf16_exact(q).float(), q)


def test_linear_transpose_and_non_aligned_shard_boundaries():
    torch.manual_seed(9)
    w = torch.randn(80, 33)
    w[40:48] *= 12
    actual, _ = search_linear(w, output_splits=[40, 40], backend="numpy")
    expected = torch.cat([search_packed(s.T, backend="numpy")[0].T for s in w.chunk(2)])
    assert_bits(actual, expected.contiguous())
    global_q, _ = search_linear(w, backend="numpy")
    assert not torch.equal(global_q, actual), "case must detect forgetting to restart groups"
    validate_repacking(actual, output_splits=[40, 40])


def test_gptq_diagonal_reduces_to_search_and_does_not_mutate():
    torch.manual_seed(3)
    w = torch.randn(65, 33)
    original = w.clone()
    h = torch.eye(33)
    q, _ = gptq_search(w, h, block_size=17, output_splits=[40, 25], backend="numpy")
    expected, _ = search_linear(w, exponent_deltas=(0, -1, -2), output_splits=[40, 25], backend="numpy")
    assert_bits(q, expected)
    assert torch.equal(w, original) and torch.equal(h, torch.eye(33))


@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_native_exact_partial_blocks_dead_channels(native_backend, act_order):
    torch.manual_seed(3)
    x = torch.randn(256, 129)
    x[:, 7] = 0
    w = torch.randn(80, 129)
    factor = factor_hessian(x.T @ x / len(x), act_order=act_order)
    before = w.clone()
    expected, ei = gptq_search(w, factor=factor, output_splits=[40, 40], block_size=32, backend="numpy")
    actual, ai = gptq_search(w, factor=factor, output_splits=[40, 40], block_size=32, backend="native", threads=2)
    assert_bits(actual, expected)
    assert ei["group_choices"] == ai["group_choices"]
    assert torch.count_nonzero(actual[:, 7]) == 0
    assert torch.equal(w, before)
    validate_repacking(actual, output_splits=[40, 40])


def test_gptq_improves_heldout_reconstruction_on_correlated_inputs():
    torch.manual_seed(8)
    w = torch.randn(64, 64) * 0.02
    mixing = torch.randn(8, 64)
    train = torch.randn(2048, 8) @ mixing + 0.1 * torch.randn(2048, 64)
    test = torch.randn(1024, 8) @ mixing + 0.1 * torch.randn(1024, 64)
    naive, _ = search_linear(w, exponent_deltas=(0,), backend="numpy")
    q, _ = gptq_search(w, train.T @ train / len(train), backend="numpy")
    error = lambda v: float((test @ (w - v).T).square().mean())
    assert error(q) < error(naive) * 0.5


@pytest.mark.parametrize(
    "bad", [torch.full((16, 16), float("nan")), torch.full((16, 16), float("inf")), torch.ones(16)]
)
def test_invalid_weights_rejected(bad):
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only tests.
        search_linear(bad)


def test_invalid_calibration_and_options_rejected():
    with pytest.raises(ValueError, match="No|energy"):  # allow-pytest.raises: CPU-only tests.
        factor_hessian(torch.zeros(8, 8))
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only tests.
        factor_hessian(torch.eye(8), damping=0)
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only tests.
        search_linear(torch.ones(20, 10), output_splits=[16, 16])
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only tests.
        search_numpy(np.zeros((1, 16)), exponent_deltas=(-1,))
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only tests.
        gptq_search(torch.ones(16, 16), torch.eye(8))

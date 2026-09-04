# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""`depthwise_tap_filter` picks its conv1d formulation (full-C, C-chunked, or MAC) by asking conv1d's own
DRAM slicer up front instead of by failed attempts. These tests pin that prediction to what actually happens
on the device, so the log-free path chooses exactly what the trial loop used to find."""

import pytest
import torch

import ttnn

from ...layers import audio_ops
from ...layers.audio_ops import TAP_PATH_CANDIDATES, _predict_tap_path, _tap_conv1d_fits, depthwise_tap_filter

# (C, K, stride, T_pad): the depthwise resample filters MiniMax-H3's BigVGAN runs at a 5 s clip
# (K=12 anti-alias taps at stride 1 up / stride 2 down per band, K=7 in the LTX vocoder), from the
# latent rate (T_pad ~ 40-200, where full C cannot fit and C is chunked) to the audio rate.
SHAPES = [
    (512, 12, 1, 166),
    (512, 12, 2, 416),
    (512, 7, 1, 40),
    (256, 12, 1, 832),
    (256, 7, 1, 80),
    (128, 12, 2, 1664),
    (64, 12, 1, 3328),
    (64, 7, 1, 166),
    (32, 12, 2, 3328),
]
SINGLE_DEVICE_PARAMS = [{"l1_small_size": 65536}]


def _candidates_for(C):
    return [c for c in TAP_PATH_CANDIDATES if c == "direct" or (c != "mac" and C % c == 0 and c < C)]


# ---------------------------------------------------------------- pure: selection logic


def _fake_fits(fitting):
    """Stand-in for `_tap_conv1d_fits`: `fitting` is the set of channel counts that fit, or None for unknown."""

    def fits(x_BTC, weight, *, C, **kwargs):
        return None if fitting is None else C in fitting

    return fits


@pytest.mark.parametrize(
    ("C", "fitting", "expected"),
    [
        (512, {512, 128, 64, 32}, "direct"),  # full C fits: no chunking
        (512, {128, 64, 32}, 128),  # widest chunk that fits
        (512, {32}, 32),
        (512, set(), "mac"),  # nothing fits
        (96, {32}, 32),  # 128 >= C and 64 does not divide 96: both skipped, not asked
        (32, {32}, "direct"),  # every chunk >= C: direct or MAC only
        (32, set(), "mac"),
        (512, None, None),  # not a DRAM-path input: prediction unavailable
    ],
)
def test_predict_tap_path_selection(monkeypatch, C, fitting, expected):
    asked = []
    fake = _fake_fits(fitting)

    def recording(x_BTC, weight, *, C, **kwargs):
        asked.append(C)
        return fake(x_BTC, weight, C=C, **kwargs)

    monkeypatch.setattr(audio_ops, "_tap_conv1d_fits", recording)
    got = _predict_tap_path(
        None,
        None,
        B=1,
        C=C,
        T_pad=64,
        K=7,
        stride=1,
        mesh_device=None,
        dtype=None,
        conv_config=None,
        compute_config=None,
    )
    assert got == expected
    # Candidates that cannot apply to this C are never asked about.
    assert all(c == C or (C % c == 0 and c < C) for c in asked)
    if expected not in ("mac", None):
        assert asked[-1] == (C if expected == "direct" else expected)


# ---------------------------------------------------------------- device: prediction == reality


def _inputs(C, K, T_pad, mesh_device):
    torch.manual_seed(0)
    x = torch.randn(1, T_pad, C, dtype=torch.float32)
    x_dev = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    taps = [float(t) for t in torch.randn(K)]
    return x, x_dev, taps


def _weight_for_query(C, K):
    """A host weight tensor of the conv1d's shape: the fit query reads only its dtype."""
    return ttnn.from_torch(torch.zeros(C, 1, K), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)


def _reference(x, taps, stride):
    K, C = len(taps), x.shape[-1]
    weight = torch.tensor(taps, dtype=torch.float32).view(1, 1, K).expand(C, 1, K).contiguous()
    return torch.nn.functional.conv1d(x.transpose(1, 2), weight, stride=stride, groups=C).transpose(1, 2)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", SINGLE_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(("C", "K", "stride", "T_pad"), SHAPES)
def test_tap_fit_prediction_matches_device(mesh_device, C, K, stride, T_pad):
    """For every applicable candidate, the up-front fit answer equals whether conv1d actually runs it.

    Seeding the path cache with a candidate makes `depthwise_tap_filter` try it first, and the trial loop
    only records a candidate as the winner if it succeeded -- so `cache[path_key] == candidate` is the
    ground truth the prediction must reproduce.
    """
    x, x_dev, taps = _inputs(C, K, T_pad, mesh_device)
    B = 1
    path_key = ("tap_path", B, T_pad, C, stride, K)

    # One call to build the shared compute-config entry the fit query needs.
    cache: dict = {}
    depthwise_tap_filter(x_dev, taps, stride, mesh_device=mesh_device, dtype=ttnn.float32, cache=cache)
    weight = _weight_for_query(C, K)
    conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.float32, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

    for candidate in _candidates_for(C):
        channels = C if candidate == "direct" else candidate
        predicted = _tap_conv1d_fits(
            x_dev,
            weight,
            B=B,
            C=channels,
            T_pad=T_pad,
            K=K,
            stride=stride,
            mesh_device=mesh_device,
            dtype=ttnn.float32,
            conv_config=conv_config,
            compute_config=cache["cc"],
        )
        assert predicted is not None, "input is a DRAM-path tensor, so the fit must be decidable"

        trial: dict = {"cc": cache["cc"], path_key: candidate}
        depthwise_tap_filter(x_dev, taps, stride, mesh_device=mesh_device, dtype=ttnn.float32, cache=trial)
        actually_ran = trial[path_key] == candidate
        assert predicted == actually_ran, (
            f"C={C} K={K} stride={stride} T_pad={T_pad} candidate={candidate}: "
            f"predicted fit={predicted} but conv1d {'ran' if actually_ran else 'did not run'} it"
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", SINGLE_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(("C", "K", "stride", "T_pad"), SHAPES)
def test_tap_filter_predicted_path_is_correct_and_cached(mesh_device, C, K, stride, T_pad):
    """The predicted path is what runs, it is cached under the shape-specific key, and it is numerically right."""
    x, x_dev, taps = _inputs(C, K, T_pad, mesh_device)
    cache: dict = {}
    out = depthwise_tap_filter(x_dev, taps, stride, mesh_device=mesh_device, dtype=ttnn.float32, cache=cache)
    path_key = ("tap_path", 1, T_pad, C, stride, K)
    assert path_key in cache, "the winner must be cached per shape (B, T_pad, C, stride, K)"

    predicted = _predict_tap_path(
        x_dev,
        _weight_for_query(C, K),
        B=1,
        C=C,
        T_pad=T_pad,
        K=K,
        stride=stride,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
        conv_config=ttnn.Conv1dConfig(weights_dtype=ttnn.float32, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        compute_config=cache["cc"],
    )
    assert cache[path_key] == predicted, "the trial loop must have run the predicted candidate first and kept it"

    actual = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    expected = _reference(x, taps, stride)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4), f"path {cache[path_key]} is numerically wrong"

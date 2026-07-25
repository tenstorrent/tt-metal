# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cross-canvas-position correlation of the device Gumbel draw (#48291).

``sample_gumbel_noise_with_permuted_vocab`` exists to keep the vocab axis off the ``ttnn.rand``
innermost axis, because QB2's rand shows last-dimension correlation. But it collapses every
non-vocab axis into ONE trailing axis and draws ``ttnn.rand((vocab, inner))`` -- and for the
production logits shape ``(1, 1, 256, vocab)`` that ``inner`` **is the 256 canvas positions**.
So the known last-dim correlation is not removed, it is relocated onto the canvas-position axis.

The existing gate (``test_device_canvas_sampling_dist.py``) cannot see this: it averages over a
sample axis into per-position marginals, and correlation *between* positions leaves every
marginal correct. This module tests the axis that gate integrates away.

Why it matters functionally: with correlated noise across positions, one unlucky draw pushes the
SAME token to the top at many positions at once, instead of the independent rare flips IID
Gumbel gives. That is the observed degeneration texture (synchronized same-token bursts), and it
is worst where the logits are flattest -- the deep-block regime.

Three arms in one run so the numbers are interpretable:
  * ``host``     -- torch Gumbel, the IID control that calibrates every metric;
  * ``permuted`` -- the production device path;
  * ``plain``    -- ``sample_gumbel_noise``, vocab-innermost, kept as a diagnostic.
"""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.tt import sampling as TS

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
    ),
    pytest.mark.use_module_device,
]

CANVAS_LEN = 256
# The production canvas/vocab geometry; PROBE_VOCAB keeps the default run cheap while staying
# far above the canvas length so the IID null is tight (off-diagonal |r| ~ 1/sqrt(vocab)).
PROD_VOCAB = 262144
PROBE_VOCAB = 16384


def _host_gumbel(shape, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    uniform = torch.rand(shape, generator=generator, dtype=torch.float32)
    return -torch.log(-torch.log(uniform.clamp_min(torch.finfo(torch.float32).tiny)))


def _draw(arm: str, shape, *, device, seed: int) -> torch.Tensor:
    """Return the arm's Gumbel noise as a host ``[canvas_len, vocab]`` matrix."""
    if arm == "host":
        return _host_gumbel(shape, seed=seed).reshape(shape[-2], shape[-1])
    generator = TS.sample_gumbel_noise_with_permuted_vocab if arm == "permuted" else TS.sample_gumbel_noise
    noise = generator(shape, device=device, seed=seed)
    host = ttnn.to_torch(noise).float()
    if hasattr(noise, "deallocate"):
        noise.deallocate(True)
    # A mesh draw comes back with the replicated devices concatenated; keep the first replica.
    return host.reshape(-1, shape[-1])[: shape[-2], :]


def _position_correlation(noise: torch.Tensor) -> dict:
    """Correlation ACROSS canvas positions, using the vocab axis as the sample axis."""
    vocab = noise.shape[1]
    corr = torch.corrcoef(noise.double())
    corr = torch.nan_to_num(corr, nan=0.0)
    off_diagonal = ~torch.eye(corr.shape[0], dtype=torch.bool)
    magnitudes = corr[off_diagonal].abs()
    # Under IID, off-diagonal r is ~N(0, 1/sqrt(vocab)); 5 sigma is a generous per-pair bound.
    sigma = 1.0 / (vocab**0.5)
    return {
        "max_abs_r": float(magnitudes.max()),
        "mean_abs_r": float(magnitudes.mean()),
        "sigma": sigma,
        "max_r_in_sigmas": float(magnitudes.max()) / sigma,
        "frac_pairs_over_5sigma": float((magnitudes > 5.0 * sigma).float().mean()),
    }


def _argmax_burst(noise: torch.Tensor) -> dict:
    """Flat-logits winner multiplicity: the deep-block regime, where noise decides alone.

    With flat logits the winner at each position is ``argmax`` of that position's noise row.
    Under IID over a vocab this large, 256 winners essentially never collide (expected
    collisions = C(256,2)/vocab), so any sizeable tie group is a cross-position dependency.
    """
    winners = noise.argmax(dim=-1)
    counts = torch.bincount(winners)
    max_multiplicity = int(counts.max())
    return {
        "distinct_winners": int((counts > 0).sum()),
        "num_positions": int(winners.numel()),
        "max_multiplicity": max_multiplicity,
        "positions_in_largest_group": max_multiplicity,
        "expected_collisions_iid": round(winners.numel() * (winners.numel() - 1) / 2 / noise.shape[1], 4),
    }


def _measure(arm: str, *, device, vocab: int, seed: int) -> dict:
    noise = _draw(arm, (1, 1, CANVAS_LEN, vocab), device=device, seed=seed)
    assert noise.shape == (CANVAS_LEN, vocab), noise.shape
    return {"arm": arm, "vocab": vocab, **_position_correlation(noise), **_argmax_burst(noise)}


def _report(label: str, stats: dict) -> None:
    print(
        f"[gumbel-pos-corr] {label:9s} vocab={stats['vocab']} "
        f"max|r|={stats['max_abs_r']:.5f} ({stats['max_r_in_sigmas']:.1f} sigma) "
        f"mean|r|={stats['mean_abs_r']:.5f} "
        f"pairs>5sigma={stats['frac_pairs_over_5sigma']:.4f} "
        f"distinct_winners={stats['distinct_winners']}/{stats['num_positions']} "
        f"max_multiplicity={stats['max_multiplicity']} "
        f"(iid expected collisions {stats['expected_collisions_iid']})"
    )


def test_host_control_calibrates_the_metrics():
    """The IID arm must pass both bounds, or the bounds are measuring the wrong thing."""
    stats = _measure("host", device=None, vocab=PROBE_VOCAB, seed=48291)
    _report("host", stats)
    assert stats["max_r_in_sigmas"] < 6.0
    assert stats["max_multiplicity"] <= 2
    assert stats["distinct_winners"] >= CANVAS_LEN - 2


def test_production_device_gumbel_is_independent_across_canvas_positions(device):
    """The gate the marginal test cannot express: no cross-position dependency."""
    stats = _measure("permuted", device=device, vocab=PROBE_VOCAB, seed=48291)
    _report("permuted", stats)
    assert stats["max_multiplicity"] <= 4, (
        "canvas positions are picking the same token far more often than IID noise allows -- "
        f"largest synchronized group is {stats['max_multiplicity']} positions: {stats}"
    )
    assert (
        stats["frac_pairs_over_5sigma"] < 0.01
    ), f"cross-position correlation is widespread, not a tail artefact: {stats}"


def test_diagnostic_plain_path_for_comparison(device):
    """Not a gate -- records the vocab-innermost path so the two axes can be compared."""
    _report("plain", _measure("plain", device=device, vocab=PROBE_VOCAB, seed=48291))


@pytest.mark.skipif(
    os.environ.get("DG_GUMBEL_CORR_FULL_VOCAB") != "1",
    reason="set DG_GUMBEL_CORR_FULL_VOCAB=1 for the 262144-vocab production geometry",
)
def test_production_vocab_geometry(device):
    """The real shipped geometry: [1, 1, 256, 262144] -> rand((262144, 256))."""
    for arm in ("host", "permuted"):
        stats = _measure(arm, device=device if arm != "host" else None, vocab=PROD_VOCAB, seed=48291)
        _report(arm, stats)
        if arm == "permuted":
            assert stats["max_multiplicity"] <= 4, stats

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity for the composed LTX-2.5 DiffVAE decoder, on shipped weights.

The halves are covered elsewhere — ``test_diffvae_det`` for the deterministic stages against
captured activations, ``test_diffvae_stage5`` for the diffusion stage against upstream's
modules. What neither covers is the seam: the trailing-ghost pad and crop that bracket the
deterministic stages, the context handoff into stage 5, and whether the two halves agree on
layout and channel order. That is what fails when parts verified separately are joined.

Ground truth is ``capture_stages.py``'s dump, which also supplies the stage-5 noise: it is an
input to a single-step x0 prediction, not an implementation detail, so matching pixels
requires using the reference's own noise rather than reseeding.

  PYTHONPATH=/tmp/LTX-2/packages/ltx-core/src:. python capture_stages.py \
      latents/latent_0_1x128x4x34x60.pt --crop 10 --out stages/crop10.safetensors
"""

import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import ttnn
from models.tt_dit.layers.na3d import window_bounds
from models.tt_dit.models.vae.diffvae_ltx import DiffVAEDecoder, decoder_config
from models.tt_dit.models.vae.diffvae_ltx_stage5 import _bands
from models.tt_dit.utils.check import assert_quality


def _gate_ccl(mesh_device):
    """CCLManager for the gates. Defaults are the historical Linear/1-link the committed baseline
    was recorded with, so an unset environment reproduces it exactly; DIFFVAE_TOPOLOGY /
    DIFFVAE_NUM_LINKS let a gate run also cover the collective config the runner actually ships
    (ring + 2 links). An all-gather only moves bytes, so this should not shift any PCC -- which is
    the point of being able to check.
    """
    from models.tt_dit.parallel.manager import CCLManager

    topology = (
        ttnn.Topology.Ring if os.environ.get("DIFFVAE_TOPOLOGY", "linear").lower() == "ring" else ttnn.Topology.Linear
    )
    return CCLManager(mesh_device, num_links=int(os.environ.get("DIFFVAE_NUM_LINKS", 1)), topology=topology)


CAPTURE = Path(
    os.environ.get(
        "DIFFVAE_CAPTURE",
        os.path.expanduser("~/ltx25_diffvae/stages/crop10.safetensors"),
    )
)
CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)


def _captured(*names: str) -> tuple[torch.Tensor, ...]:
    """Load tensors from the capture, skipping if it was written without them.

    ``capture_stages.py --pixels-only`` keeps only the endpoints, which is how captures at
    resolutions above the smallest tile are produced — the intermediates would be tens of GB.
    """
    with safe_open(str(CAPTURE), "pt") as handle:
        available = set(handle.keys())
        if missing := [name for name in names if name not in available]:
            pytest.skip(f"{CAPTURE.name} lacks {missing}; regenerate without --pixels-only")
        return tuple(handle.get_tensor(name).float() for name in names)


@pytest.fixture
def decoder(device):
    if not CAPTURE.exists():
        pytest.skip(f"missing {CAPTURE}; run capture_stages.py first")
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    model = DiffVAEDecoder(decoder_config(CHECKPOINT), mesh_device=device)
    model.load_checkpoint(CHECKPOINT)
    return model


@pytest.mark.diffvae_gate
def test_context_matches_upstream(*, decoder):
    """All four deterministic stages plus the ghost pad and crop, latent to stage-5 context.

    The frame arithmetic is the point: 4 latent frames become 6 by trailing replication, grow
    to 41 through four upsamples, then crop back to 25. An off-by-one in any of those three
    steps lands here rather than in a stage's own test.
    """
    latent, expected = _captured("input.latent", "stage4.context")

    context, dims = decoder.forward_context(latent)
    assert dims == tuple(expected.shape[1:4]), f"dims {dims} != capture {tuple(expected.shape[1:4])}"
    assert decoder.context_frames(latent.shape[2]) == expected.shape[1]

    actual = ttnn.to_torch(context).reshape(1, *dims, expected.shape[-1])
    assert_quality(expected, actual, pcc=0.99)


@pytest.mark.parametrize("t", [12, 25, 145])
@pytest.mark.parametrize("frames", [1, 3, 8, 16, 64])
@pytest.mark.diffvae_gate
def test_band_halo_covers_every_window(t: int, frames: int):
    """A band's local windows must be the volume's own, shifted by the band's halo.

    Stage 5 runs long videos as frame bands and attends each one as a standalone volume, so a
    query whose window reaches outside its band would quietly attend to the wrong frames: wrong
    pixels, no error. The inward window shift makes the bound easy to get wrong, since a query
    near either end reaches ``kernel - 1`` frames the other way rather than ``kernel // 2``.

    Needs no device or capture: it is arithmetic against the same ``window_bounds`` the attention
    plan is built from.
    """
    kernel = 11
    starts, ends = window_bounds(t, kernel)
    bands = _bands(t, frames=frames, kernel=kernel)

    assert bands[0].lo == 0 and bands[-1].hi == t, f"bands do not cover {t} frames: {bands}"
    for before, after in zip(bands, bands[1:], strict=False):
        assert before.hi == after.lo, f"bands are not contiguous: {before} then {after}"

    for band in bands:
        local_starts, local_ends = window_bounds(band.pad_frames, kernel)
        for q in range(band.lo, band.hi):
            local = q - band.pad_lo
            assert local_starts[local] + band.pad_lo == starts[q], f"frame {q} of {band}: start moved"
            assert local_ends[local] + band.pad_lo == ends[q], f"frame {q} of {band}: end moved"


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp_rows", "sp_cols"])
def test_decode_stage5_wsp_matches_replicated(*, mesh_device, sp_axis):
    """Full decode with stage 5 run under spatial-W SP matches the replicated decode, on shipped
    weights. The deterministic stages are identical (replicated) in both; only stage 5 differs --
    its sequence, context and RoPE are W-sharded and the output gathered back -- so this is the
    end-to-end check that the full-stage-SP forward plumbing (reshard, sharded upload, gather)
    reassembles to the same pixels. Same latent and seed, so both draw the same x0 noise.
    """

    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    config = decoder_config(CHECKPOINT)
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 2, 8, 8)

    replicated = DiffVAEDecoder(config, mesh_device=mesh_device)
    replicated.load_checkpoint(CHECKPOINT)
    pixels_rep = replicated.decode(latent, seed=0)

    ccl_manager = _gate_ccl(mesh_device)
    sharded = DiffVAEDecoder(
        config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        stage5_na3d_backend="op_sp_w_sharded",
        stage5_sp_axis=sp_axis,
    )
    sharded.load_checkpoint(CHECKPOINT)
    pixels_sp = sharded.decode(latent, seed=0)

    assert tuple(pixels_sp.shape) == tuple(pixels_rep.shape), f"{tuple(pixels_sp.shape)} != {tuple(pixels_rep.shape)}"
    assert_quality(pixels_rep, pixels_sp, pcc=0.999)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp_rows", "sp_cols"])
def test_decode_full_wsp_matches_replicated(*, mesh_device, sp_axis):
    """Full decode with BOTH the deterministic stages and stage 5 W-sharded matches the replicated
    decode, on shipped weights. The det stages shard from stage 1 (stage 0 replicated) and hand the
    context to stage 5 W-sharded directly (same sp_axis) -- no gather-to-replicated round trip. Same
    latent and seed, so both draw the same x0 noise. End-to-end check of det-stage SP + the handoff."""

    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    config = decoder_config(CHECKPOINT)
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 2, 8, 8)

    replicated = DiffVAEDecoder(config, mesh_device=mesh_device)
    replicated.load_checkpoint(CHECKPOINT)
    pixels_rep = replicated.decode(latent, seed=0)

    ccl_manager = _gate_ccl(mesh_device)
    # DIFFVAE_TP_HEADS=1 also enables TP-over-heads on the orthogonal (size-4) axis -- only valid when
    # W shards the size-8 axis (sp_axis=1), since num_heads=4 must divide the TP axis size.
    tp_axis = 0 if (os.environ.get("DIFFVAE_TP_HEADS") == "1" and sp_axis == 1) else None
    sharded = DiffVAEDecoder(
        config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        stage5_na3d_backend="op_sp_w_sharded",
        stage5_sp_axis=sp_axis,
        stage5_tp_axis=tp_axis,
        stages_na3d_backend="op_sp_w_sharded",
        stages_sp_axis=sp_axis,
        stages_tp_axis=tp_axis,
    )
    sharded.load_checkpoint(CHECKPOINT)
    pixels_sp = sharded.decode(latent, seed=0)

    assert tuple(pixels_sp.shape) == tuple(pixels_rep.shape), f"{tuple(pixels_sp.shape)} != {tuple(pixels_rep.shape)}"
    assert_quality(pixels_rep, pixels_sp, pcc=0.999)


@pytest.mark.diffvae_gate
def test_decode_matches_upstream(*, decoder):
    """Latent to pixels through the whole decoder, against upstream's own pixels."""
    latent, noise, expected = _captured("input.latent", "stage5.noise", "output.pixels")

    pixels = decoder.decode(latent, noise=noise)

    assert tuple(pixels.shape) == tuple(expected.shape), f"{tuple(pixels.shape)} != {tuple(expected.shape)}"
    if out := os.environ.get("DIFFVAE_DUMP_PIXELS"):
        # A PCC number says the port is right; it does not say the video looks right. Keeping
        # the device pixels lets them be viewed against the reference and the conv decoder.
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(pixels.cpu(), out)
        print(f"\nwrote device pixels {tuple(pixels.shape)} to {out}")
    assert_quality(expected, pixels, pcc=0.99)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("sp_axis", [1], ids=["sp_cols"])
# latent_t 2 puts stage 5 at t=11, exactly the window extent -- every brick then reads as clamped
# (brick_window_is_unclamped returns false when window >= volume on an axis), which is a different
# path from production's t=84. latent_t 4 gives t=21, comfortably clear of it. Covering both is
# what distinguishes "broken at degenerate t" from "broken generally".
@pytest.mark.parametrize("latent_t", [2, 4], ids=["t_at_window", "t_clear_of_window"])
@pytest.mark.diffvae_gate
def test_decode_stage5_bricked_matches_replicated(*, mesh_device, sp_axis, latent_t):
    """Full decode with stage 5 on the BRICKED backend matches the replicated decode.

    The end-to-end companion to test_stage5_parity_w_sharded_bricked. Same shape as
    test_decode_stage5_wsp_matches_replicated -- deterministic stages replicated and identical in
    both arms, only stage 5 differs -- but exercising ``bricked_sp_w_sharded``: halo exchange
    instead of a full-W all-gather, the bricked layout, and the in-kernel neighborhood gather.

    sp_axis is fixed to 1. The bricked path halo-exchanges whole bricks and needs a local width of
    at least ``halo_sites(11, 2) == 6``; only the size-8 axis leaves enough width here, and sharding
    the size-4 axis would give a local width the brick chooser rejects outright.

    Note this decodes the same (2, 8, 8) latent as the other decoder tests, which puts stage 5 at
    W = 64 and so a local width of 8 -- just above the halo. A smaller latent will not build a plan.
    """

    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    config = decoder_config(CHECKPOINT)
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], latent_t, 8, 8)

    replicated = DiffVAEDecoder(config, mesh_device=mesh_device)
    replicated.load_checkpoint(CHECKPOINT)
    pixels_rep = replicated.decode(latent, seed=0)

    # Ring collectives, matching the runner and the FABRIC_1D_RING fabric above. The halo exchange
    # stays Linear regardless: _halo_exchange in neighborhood_attention.py pins it, because
    # neighbor_pad_async deadlocks on Ring. _gate_ccl would give Linear for everything, which is a
    # configuration no production path runs.
    from models.tt_dit.parallel.manager import CCLManager

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Ring)
    bricked = DiffVAEDecoder(
        config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        stage5_na3d_backend="bricked_sp_w_sharded",
        stage5_sp_axis=sp_axis,
    )
    bricked.load_checkpoint(CHECKPOINT)
    pixels_bricked = bricked.decode(latent, seed=0)

    assert tuple(pixels_bricked.shape) == tuple(
        pixels_rep.shape
    ), f"{tuple(pixels_bricked.shape)} != {tuple(pixels_rep.shape)}"
    assert_quality(pixels_rep, pixels_bricked, pcc=0.999)

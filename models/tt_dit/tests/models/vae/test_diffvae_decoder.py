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
@pytest.mark.parametrize("tp", [False, True], ids=["tp_off", "tp4"])
def test_decode_full_bricked_matches_replicated(*, mesh_device, tp):
    """Full decode with the deterministic stages AND stage 5 on the BRICKED executor matches the
    replicated decode, on shipped weights.

    Both deterministic stages and stage 5 on ``bricked_sp_w_sharded``: stages
    1-3 shard over W from stage 1 on, run the halo exchange + neighborhood op with a width-1 brick
    where the shard is 15 wide, hand the W-band straight to stage 5 (same sp_axis), and stage 5 runs
    its own bricked path. ``tp4`` adds TP over heads on the size-4 axis, which is production and puts
    4/2/2 heads per chip through the executor's flat (B, NH, S, HD) handoff; ``tp_off`` keeps every
    head on one chip, so the K/V halo stick is at its widest (1024 channels -> 16 sub-columns).

    Latent W is 16, not the usual 8: at 8 the stage-2 shard is 2 columns wide against a halo of 3
    and no brick can plan. At 16 the stages sit at W_local 4/4/8 (stage 5 at 16), the tightest
    shards the chooser accepts, so a seam error shows up at its worst.
    """

    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    config = decoder_config(CHECKPOINT)
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 2, 8, 16)

    replicated = DiffVAEDecoder(config, mesh_device=mesh_device)
    replicated.load_checkpoint(CHECKPOINT)
    pixels_rep = replicated.decode(latent, seed=0)

    # Ring collectives as the runner ships; the halo exchange pins itself to Linear regardless.
    from models.tt_dit.parallel.manager import CCLManager

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Ring)
    tp_axis = 0 if tp else None
    bricked = DiffVAEDecoder(
        config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        stage5_na3d_backend="bricked_sp_w_sharded",
        stage5_sp_axis=1,
        stage5_tp_axis=tp_axis,
        stages_na3d_backend="bricked_sp_w_sharded",
        stages_sp_axis=1,
        stages_tp_axis=tp_axis,
    )
    bricked.load_checkpoint(CHECKPOINT)
    pixels_bricked = bricked.decode(latent, seed=0)

    assert tuple(pixels_bricked.shape) == tuple(
        pixels_rep.shape
    ), f"{tuple(pixels_bricked.shape)} != {tuple(pixels_rep.shape)}"
    if out := os.environ.get("DIFFVAE_DUMP_PIXELS"):
        path = Path(out).with_name(f"{Path(out).stem}_full_bricked_tp{int(tp)}{Path(out).suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"replicated": pixels_rep.cpu(), "bricked": pixels_bricked.cpu()}, path)
        print(f"\nwrote both arms {tuple(pixels_rep.shape)} to {path}")
    assert_quality(pixels_rep, pixels_bricked, pcc=0.999)


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
# latent W 8 puts stage 5 at W=64, a local width of 8 against a 6-site halo -- the tightest shard the
# brick chooser accepts. W 16 doubles the local width to 16, so a miss that appears only at 8 is
# the near-full-width halo exchange, not the op.
@pytest.mark.parametrize("latent_w", [8, 16], ids=["w_local8", "w_local16"])
@pytest.mark.diffvae_gate
def test_decode_stage5_bricked_matches_replicated(*, mesh_device, sp_axis, latent_t, latent_w):
    """Full decode with stage 5 on the BRICKED backend matches the replicated decode.

    The end-to-end companion to test_stage5_parity_w_sharded_bricked. Same shape as
    the deterministic stages replicated and identical in both arms, only stage 5 differs, exercising ``bricked_sp_w_sharded``: halo exchange
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
    latent = torch.randn(1, config["in_channels"], latent_t, 8, latent_w)

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
    if out := os.environ.get("DIFFVAE_DUMP_PIXELS"):
        # Both arms, so a miss can be localised (a seam error sits at the W-shard boundaries,
        # a math error is everywhere). Suffixed by latent_t and latent_w so the params do not overwrite.
        path = Path(out).with_name(f"{Path(out).stem}_t{latent_t}_w{latent_w}{Path(out).suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"replicated": pixels_rep.cpu(), "bricked": pixels_bricked.cpu()}, path)
        print(f"\nwrote both arms {tuple(pixels_rep.shape)} to {path}")
    assert_quality(pixels_rep, pixels_bricked, pcc=0.999)

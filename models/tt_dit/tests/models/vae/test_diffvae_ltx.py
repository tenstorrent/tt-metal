# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``models/tt_dit/models/vae/diffvae_ltx.py``: the deterministic NABlock and its fast
paths, DeterministicStages, and the composed DiffVAEDecoder.

Parity gates compare against upstream on shipped weights. Ground truth comes from
``capture_stages.py``, which drives upstream's own decoder one stage at a time with the shipped
checkpoint and injected noise, so per-block targets exist: a 24-block network compared only at its
output tells you something is wrong but never where. The decoder gates check the seam the halves
cannot -- the trailing-ghost pad and crop, the context handoff into stage 5, and the layout and
channel order the two halves must agree on. The capture also supplies the stage-5 noise: it is an
input to a single-step x0 prediction, not an implementation detail.

Generate the capture first (host only, no device):

  PYTHONPATH=$LTX_CORE_SRC:. python models/tt_dit/tests/models/vae/capture_stages.py \\
      latents/latent_0_1x128x4x34x60.pt --crop 10 --out stages/crop10.safetensors

The arms tests compare each ``DetBlockOptions`` fast path against the unflagged block on identical
weights, and the timing tests are the instrument those paths were tuned with; neither is a gate.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import ttnn
from models.tt_dit.layers import feedforward
from models.tt_dit.layers.neighborhood_attention_plan import build_device_plan, plan_na3d
from models.tt_dit.models.vae import diffvae_ltx
from models.tt_dit.models.vae.diffvae_ltx import (
    DetBlockOptions,
    DeterministicStages,
    DiffVAEDecoder,
    DiffVAEOptions,
    NABlock,
    decoder_config,
    rope_tables,
)
from models.tt_dit.models.vae.diffvae_rope import default_rope_dim_split
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tools import diffvae_bench as bench
from models.tt_dit.tools.diffvae_bench import ARMS, BASELINE, CHECKPOINT, HEAD_DIM, STAGE1, STAGE1_ARMS, STAGES

#: Both halves W-sharded over the columns axis on the bricked executor, no TP, nothing fused.
SHARDED = DiffVAEOptions(
    stage5_backend="bricked_sp_w_sharded",
    stage5_sp_axis=1,
    stages_backend="bricked_sp_w_sharded",
    stages_sp_axis=1,
)
from models.tt_dit.utils.check import assert_quality

CAPTURE = Path(
    os.environ.get(
        "DIFFVAE_CAPTURE",
        os.path.expanduser("~/ltx25_diffvae/stages/crop10.safetensors"),
    )
)

# Stage 1 of the shipped config: 2048 channels, kernel (3,7,7), 4 blocks.
STAGE_DIM = 2048
STAGE_KERNEL = (3, 7, 7)


def _require_capture() -> None:
    if not CAPTURE.exists():
        pytest.skip(f"missing {CAPTURE}; run capture_stages.py first")


def _require_checkpoint() -> None:
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")


def _captured(*names: str) -> tuple[torch.Tensor, ...]:
    """Load tensors from the capture, skipping if it was written without them.

    ``capture_stages.py --pixels-only`` keeps only the endpoints, which is how captures at
    resolutions above the smallest tile are produced -- the intermediates would be tens of GB.
    """
    with safe_open(str(CAPTURE), "pt") as handle:
        available = set(handle.keys())
        if missing := [name for name in names if name not in available]:
            pytest.skip(f"{CAPTURE.name} lacks {missing}; regenerate without --pixels-only")
        return tuple(handle.get_tensor(name).float() for name in names)


def _block_weights(stage: int, block: int) -> dict[str, torch.Tensor]:
    prefix = f"decoder.det_stages.{stage}.{block}."
    with safe_open(str(CHECKPOINT), "pt") as handle:
        return {key[len(prefix) :]: handle.get_tensor(key).float() for key in handle.keys() if key.startswith(prefix)}


# ---------------------------------------------------------------------------
# NABlock against upstream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fused", [False, True], ids=["halves", "fused_rope"])
@pytest.mark.parametrize("block_index", [0, 1])
@pytest.mark.diffvae_gate
def test_na_block_matches_upstream(*, device, block_index, fused):
    """One deterministic NA block, real weights, real activations.

    ``fused_rope`` is the rotation the W-sharded stages run in production: fused qkv, whose TILE
    output is what ``rotary_embedding_hf`` needs, then that op instead of the halves form. The two
    travel together here as they do in the production options.
    """
    _require_capture()
    _require_checkpoint()
    options = DetBlockOptions(fused_qkv=fused, fused_rope=fused)

    source = "stage0.conv_in" if block_index == 0 else f"det0.block{block_index - 1}"
    hidden, expected = _captured(source, f"det0.block{block_index}")
    _, t, h, w, dim = hidden.shape
    assert dim == STAGE_DIM, f"capture has dim {dim}, expected {STAGE_DIM}"

    block = NABlock(STAGE_DIM, STAGE_KERNEL, head_dim=HEAD_DIM, mesh_device=device, options=options)
    block.load_state_dict(_block_weights(0, block_index))

    tokens = t * h * w
    tt_hidden = ttnn.from_torch(
        hidden.reshape(tokens, dim), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    cos, sin = rope_tables((t, h, w), default_rope_dim_split(HEAD_DIM), mesh_device=device)
    plan = build_device_plan(plan_na3d((t, h, w), STAGE_KERNEL), mesh_device=device)

    actual = block(tt_hidden, dims=(t, h, w), cos=cos, sin=sin, device_plan=plan)
    actual = ttnn.to_torch(actual).reshape(1, t, h, w, dim)

    assert_quality(expected, actual, pcc=0.99)


@pytest.mark.diffvae_gate
def test_row_chunking_is_exact(*, device, monkeypatch):
    """Chunking the pointwise parts changes nothing about the result.

    A block's peak is set by its SwiGLU, whose hidden width is 4x the activation, so at 6s
    1920x1088 its three intermediates come to 30 GiB. Running those in row chunks is only sound
    because they are pointwise in the site axis, and that is what this pins: same weights, same
    input, chunked against whole, bit for bit. Sizes here are far below :data:`CHUNK_BYTES`, so
    the budget is shrunk to force the loop, with a row count that leaves a short final chunk.
    """
    torch.manual_seed(0)
    dim, head_dim, kernel = 128, 64, (3, 3, 3)
    hidden = (int(dim * 4.0) + 15) // 16 * 16
    dims = (5, 8, 7)
    tokens = dims[0] * dims[1] * dims[2]  # 280, not a multiple of TILE
    assert tokens % diffvae_ltx.TILE != 0, "want a ragged final chunk"

    weights = {
        "norm1.weight": (dim,),
        "norm2.weight": (dim,),
        "attn.qkv.weight": (3 * dim, dim),
        "attn.qkv.bias": (3 * dim,),
        "attn.proj.weight": (dim, dim),
        "attn.proj.bias": (dim,),
        "attn.q_norm.weight": (head_dim,),
        "attn.k_norm.weight": (head_dim,),
        "mlp.w_gate.weight": (hidden, dim),
        "mlp.w_up.weight": (hidden, dim),
        "mlp.w_down.weight": (dim, hidden),
    }
    state = {name: torch.randn(shape) * 0.1 for name, shape in weights.items()}
    hidden_states = torch.randn(tokens, dim)

    cos, sin = rope_tables(dims, default_rope_dim_split(head_dim), mesh_device=device)
    plan = build_device_plan(plan_na3d(dims, kernel), mesh_device=device)

    def run() -> torch.Tensor:
        block = NABlock(dim, kernel, head_dim=head_dim, mesh_device=device)
        block.load_state_dict({key: value.clone() for key, value in state.items()})
        tt_hidden = ttnn.from_torch(hidden_states, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(block(tt_hidden, dims=dims, cos=cos, sin=sin, device_plan=plan))

    whole = run()
    monkeypatch.setattr(feedforward, "CHUNK_BYTES", 2 * diffvae_ltx.TILE * hidden * 2)
    chunked = run()

    assert torch.equal(whole, chunked), (whole - chunked).abs().max()


# ---------------------------------------------------------------------------
# NABlock fast paths: every DetBlockOptions arm against the unflagged block
# ---------------------------------------------------------------------------


def _state(dim: int, seed: int) -> dict[str, torch.Tensor]:
    """Checkpoint-shaped weights: fused ``attn.qkv`` and split ``mlp.w_gate``/``w_up``.

    Every arm loads this same dict; only ``_prepare_torch_state`` differs between them, so a
    divergence is the arm's fault and not the input's.
    """
    g = torch.Generator().manual_seed(seed)
    hidden = (int(dim * 4.0) + 15) // 16 * 16

    def rn(*shape):
        return torch.randn(*shape, generator=g) * (shape[-1] ** -0.5)

    return {
        "norm1.weight": 1.0 + 0.05 * torch.randn(dim, generator=g),
        "norm2.weight": 1.0 + 0.05 * torch.randn(dim, generator=g),
        "attn.qkv.weight": rn(3 * dim, dim),
        "attn.qkv.bias": 0.02 * torch.randn(3 * dim, generator=g),
        "attn.proj.weight": rn(dim, dim),
        "attn.proj.bias": 0.02 * torch.randn(dim, generator=g),
        "attn.q_norm.weight": 1.0 + 0.05 * torch.randn(HEAD_DIM, generator=g),
        "attn.k_norm.weight": 1.0 + 0.05 * torch.randn(HEAD_DIM, generator=g),
        "mlp.w_gate.weight": rn(hidden, dim),
        "mlp.w_up.weight": rn(hidden, dim),
        "mlp.w_down.weight": rn(dim, hidden),
    }


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True, ids=["1d"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("arm", list(ARMS), ids=list(ARMS))
@pytest.mark.parametrize("stage", STAGES, ids=[s[0] for s in STAGES])
def test_det_nablock_arm_matches_baseline(*, mesh_device, device_params, arm, stage):
    """Every fast path reproduces the unflagged block on the same weights."""
    _, dim, kernel, dims, _ = stage
    state = _state(dim, seed=11)
    tokens, local, cos, sin = bench.det_inputs(mesh_device, dim, dims)
    x_t = torch.randn(tokens, dim, generator=torch.Generator().manual_seed(5))

    def run(arm):
        block = bench.build_det_block(mesh_device, dim, kernel, arm)
        block.load_torch_state_dict(dict(state))
        x = ttnn.from_torch(x_t, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = block(x, dims=local, cos=cos, sin=sin, device_plan=None)
        # Chip 0's W-band suffices: the arms differ only in per-chip head bookkeeping.
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()

    reference = run(BASELINE)
    assert_quality(reference, run(ARMS[arm]), pcc=0.999)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True, ids=["1d"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("arm", list(STAGE1_ARMS), ids=list(STAGE1_ARMS))
def test_det_stage1_arm_matches_baseline(*, mesh_device, device_params, arm):
    """Every stage-1 fast path reproduces the unflagged replicated block on the same weights."""
    dim, kernel, dims, _ = STAGE1
    state = _state(dim, seed=11)
    t, h, w = dims
    tokens = t * h * w  # replicated: no W shard
    cos, sin = rope_tables(dims, default_rope_dim_split(HEAD_DIM), mesh_device=mesh_device)
    plan = bench.stage1_plan(mesh_device, dims, kernel)
    x_t = torch.randn(tokens, dim, generator=torch.Generator().manual_seed(5))

    def run(arm):
        block = bench.build_stage1_block(mesh_device, arm)
        block.load_torch_state_dict(dict(state))
        x = ttnn.from_torch(x_t, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = block(x, dims=dims, cos=cos, sin=sin, device_plan=plan)
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()

    reference = run(BASELINE)
    assert_quality(reference, run(STAGE1_ARMS[arm]), pcc=0.999)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True, ids=["1d"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("arm", ["baseline", *ARMS], ids=["baseline", *ARMS])
def test_det_nablock_arm_timing(*, mesh_device, device_params, arm, bench_options):
    """Per-block device time for one arm, summed over the W-sharded stages. Run with ``-s``.

    A change inside one block is invisible against a whole decode; this is where the fusions were
    tuned. ``python -m models.tt_dit.tools.time_module det_nablock --arm <arm>`` is the same
    measurement outside pytest.
    """
    iters = bench_options.iterations()
    total = 0.0
    for stage in STAGES:
        b = bench.det_block_bench(mesh_device, stage, bench.arm_options(arm))
        ms = bench.timed(b, mesh_device, iters)
        total += ms * b.depth
        print(f"\n[{arm}/{b.label}] {ms:8.2f} ms/block  x{b.depth} = {ms * b.depth:8.1f} ms", flush=True)
        b.close()
    print(f"\n[{arm}] W-sharded det blocks total: {total:8.1f} ms\n", flush=True)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True, ids=["1d"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("arm", ["baseline", *STAGE1_ARMS], ids=["baseline", *STAGE1_ARMS])
def test_det_stage1_arm_timing(*, mesh_device, device_params, arm, bench_options):
    """Per-block device time for the replicated stage-1 block under one arm. Run with ``-s``.

    Stage 1 carries ~2.2x the per-chip matmul FLOPs of stages 2-4 combined: it is the widest stage
    and the only one that runs on every chip. ``time_module det_stage1`` is the CLI twin.
    """
    b = bench.stage1_bench(mesh_device, bench.arm_options(arm, STAGE1_ARMS))
    ms = bench.timed(b, mesh_device, bench_options.iterations())
    print(f"\n[{arm}/{b.label}] {ms:8.2f} ms/block  x{b.depth} = {ms * b.depth:8.1f} ms\n", flush=True)
    b.close()


# The stage-5 grid is --diffvae-grid TxHxW, defaulting to the shipped 1080p 25-frame geometry,
# whose host-side context and noise tensors are several GB; shrink T first when validating.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True, ids=["1d"])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
def test_diff_block_timing(*, mesh_device, device_params, bench_options):
    """Per-block device time for one stage-5 DiffusionNABlock in the production SP x TP config.

    Stage 5 is the largest block in the decoder and the parity tests run a toy grid, so this is
    the A/B instrument for stage-5 work. Under ``TT_DIT_STAGE_TIMING=1`` the block's sections are
    printed too. ``time_module diff_block`` is the CLI twin.
    """
    iters = bench_options.iterations()
    b = bench.diff_block_bench(mesh_device, bench_options.stage5_grid(bench.STAGE5_GRID))
    with bench.timing_tree.span(mesh_device, "diff_block_timing", root=True):
        ms = bench.timed(b, mesh_device, iters)
    print("\n" + bench.block_line(b, ms), flush=True)
    if sections := bench.block_sections(iters + 2):
        print(bench.render_sections(sections), flush=True)


# ---------------------------------------------------------------------------
# DeterministicStages against upstream
# ---------------------------------------------------------------------------


@pytest.mark.diffvae_gate
def test_det_stages_match_upstream(*, device):
    """Stages 1-3 end to end: 14 blocks, three kernels, three upsample strides.

    Starts from the normalized latent rather than ``conv_in``'s output, so the per-channel
    statistics folded into ``conv_in`` are exercised too. Stopping at ``det2.upsampled`` is
    what the capture offers -- stage 4's boundary has upstream's trailing-ghost crop applied,
    which is a tiling concern handled at the pipeline level, not here.
    """
    _require_capture()
    _require_checkpoint()

    latent, expected = _captured("input.latent_padded", "det2.upsampled")
    config = decoder_config(CHECKPOINT)

    stages = DeterministicStages(
        in_channels=config["in_channels"],
        stage_channels=config["stage_channels"],
        stage_depths=config["stage_depths"],
        stage_kernels=config["stage_kernels"],
        upsamples=config["upsamples"],
        head_dim=config["head_dim"],
        mesh_device=device,
    )
    stages.load_checkpoint(CHECKPOINT)

    # Channels-last, flattened to tokens: the decoder's own first move on the latent.
    _, channels, t, h, w = latent.shape
    tokens = latent.permute(0, 2, 3, 4, 1).reshape(t * h * w, channels)
    tt_latent = ttnn.from_torch(tokens, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    actual, dims = stages(tt_latent, dims=(t, h, w), stages=3)
    assert dims == tuple(expected.shape[1:4]), f"dims {dims} != capture {tuple(expected.shape[1:4])}"
    actual = ttnn.to_torch(actual).reshape(1, *dims, expected.shape[-1])

    assert_quality(expected, actual, pcc=0.99)


# ---------------------------------------------------------------------------
# DiffVAEDecoder: the composed decoder against upstream, and W-sharded against replicated
# ---------------------------------------------------------------------------


@pytest.fixture
def decoder(device):
    _require_capture()
    _require_checkpoint()
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


@pytest.mark.diffvae_gate
def test_decode_matches_upstream(*, decoder):
    """Latent to pixels through the whole decoder, against upstream's own pixels.

    The one gate that checks the seam end to end: the deterministic stages, the ghost pad and
    crop, the context handoff and stage 5 on the reference's own noise, as a single number.
    """
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
@pytest.mark.parametrize("tp", [False, True], ids=["tp_off", "tp4"])
def test_decode_full_bricked_matches_replicated(*, mesh_device, tp):
    """Full decode with the deterministic stages AND stage 5 on the bricked executor matches the
    replicated decode, on shipped weights.

    Stages 1-3 shard over W from stage 1 on, run the halo exchange + neighborhood op with a width-1
    brick where the shard is 15 wide, hand the W-band straight to stage 5 (same sp_axis), and stage 5
    runs its own bricked path. ``tp4`` adds TP over heads on the size-4 axis, which is production and
    puts 4/2/2 heads per chip through the executor's flat (B, NH, S, HD) handoff; ``tp_off`` keeps
    every head on one chip, so the K/V halo stick is at its widest (1024 channels -> 16 sub-columns).

    Latent W is 16, not the usual 8: at 8 the stage-2 shard is 2 columns wide against a halo of 3
    and no brick can plan. At 16 the stages sit at W_local 4/4/8 (stage 5 at 16), the tightest
    shards the chooser accepts, so a seam error shows up at its worst.
    """
    _require_checkpoint()
    config = decoder_config(CHECKPOINT)
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 2, 8, 16)

    replicated = DiffVAEDecoder(config, mesh_device=mesh_device)
    replicated.load_checkpoint(CHECKPOINT)
    pixels_rep = replicated.decode(latent, seed=0)

    # Ring collectives as the runner ships; the halo exchange pins itself to Linear regardless.
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Ring)
    tp_axis = 0 if tp else None
    bricked = DiffVAEDecoder(
        config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        options=dataclasses.replace(SHARDED, stage5_tp_axis=tp_axis, stages_tp_axis=tp_axis),
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
    """Full decode with stage 5 on the bricked backend matches the replicated decode.

    The end-to-end companion to test_stage5_parity_w_sharded_bricked: the deterministic stages
    replicated and identical in both arms, only stage 5 differs, exercising ``bricked_sp_w_sharded``:
    halo exchange instead of a full-W all-gather, the bricked layout, and the in-kernel neighborhood
    gather.

    sp_axis is fixed to 1. The bricked path halo-exchanges whole bricks and needs a local width of
    at least ``halo_sites(11, 2) == 6``; only the size-8 axis leaves enough width here, and sharding
    the size-4 axis would give a local width the brick chooser rejects outright.
    """
    _require_checkpoint()
    config = decoder_config(CHECKPOINT)
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], latent_t, 8, latent_w)

    replicated = DiffVAEDecoder(config, mesh_device=mesh_device)
    replicated.load_checkpoint(CHECKPOINT)
    pixels_rep = replicated.decode(latent, seed=0)

    # Ring collectives, matching the runner and the FABRIC_1D_RING fabric above. The halo exchange
    # stays Linear regardless: _halo_exchange in neighborhood_attention.py pins it, because
    # neighbor_pad_async deadlocks on Ring.
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Ring)
    bricked = DiffVAEDecoder(
        config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        options=DiffVAEOptions(stage5_backend="bricked_sp_w_sharded", stage5_sp_axis=sp_axis),
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


# ---------------------------------------------------------------------------
# Decode timing instruments (not gates)
# ---------------------------------------------------------------------------


# (deterministic-stages backend, stage-5 backend), single chip, replicated. "linear_order+bricked5" runs
# the linear-order executor where it fits (the smaller early stages) and the unsharded bricked executor for
# stage 5. Set TT_DIT_STAGE_TIMING=1 for the per-stage breakdown.
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "backends",
    [("linear_order", "linear_order"), ("linear_order", "bricked")],
    ids=["linear_order", "linear_order+bricked5"],
)
@pytest.mark.parametrize("latent_hw", [(16, 16), (34, 60)], ids=["s16", "s34x60"])
def test_decode_timing(*, mesh_device, backends, latent_hw):
    _require_checkpoint()
    stages_b, stage5_b = backends
    config = decoder_config(CHECKPOINT)
    lh, lw = latent_hw
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 4, lh, lw)

    dec = DiffVAEDecoder(
        config, mesh_device=mesh_device, options=DiffVAEOptions(stages_backend=stages_b, stage5_backend=stage5_b)
    )
    dec.load_checkpoint(CHECKPOINT)

    px, dt = bench.timed_decode(dec, latent, mesh_device)
    tag = f"stages={stages_b},stage5={stage5_b}"
    print(f"\n[decode {tag}] latent(1,{config['in_channels']},4,{lh},{lw}) -> {tuple(px.shape)}: {dt:8.0f} ms\n")


# The production configuration: both halves W-sharded across the mesh on the bricked executor
# (stage 1 stays replicated), as the ``diffvae_options`` fixture builds it.
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("latent_hw", [(16, 16), (34, 60)], ids=["s16", "s34x60"])
def test_decode_wsp_timing(*, mesh_device, latent_hw, timing_tree, diffvae_options, bench_options):
    """Whole-decode wall time in the runner's configuration; ``time_module decoder`` is the CLI twin.

    The decoder is built from the ``--diffvae-*`` options (``diffvae_options``) and the run from
    ``--diffvae-latent-t`` (default 19, the 145-frame target; 4 gives a quick 25-frame run),
    ``--diffvae-topology`` / ``--diffvae-num-links`` (``bench_options``). Note that the plain
    ``bricked`` stage-5 backend does not W-shard, so it runs the full volume on every chip: honest
    about speed, not about memory. The stages' axis swap is priced here and rejected in
    NEIGHBORHOOD_ATTENTION.md ("The retired block-permute path").
    """
    _require_checkpoint()
    config = decoder_config(CHECKPOINT)
    t_lat = bench_options.latent_frames()
    latent = bench.latent(config, t_lat, latent_hw)
    dec = bench.production_decoder(mesh_device, config, bench_options.ccl(mesh_device), diffvae_options)
    dec.load_checkpoint(CHECKPOINT)

    px, dt = bench.timed_decode(dec, latent, mesh_device)
    lh, lw = latent_hw
    print(
        f"\n[decode {bench.describe(diffvae_options)} 4x8] latent(1,{config['in_channels']},{t_lat},{lh},{lw})"
        f" -> {tuple(px.shape)}: {dt:8.0f} ms\n"
    )


# The decode tail: host unpatchify + pull (float) against on-device YUV 4:2:0 (yuv). Both run the
# identical device graph; only the tail differs, so the delta is the transfer. TT_DIT_STAGE_TIMING=1
# also isolates the tail timer from the rest of the decode. ``time_module decoder --output float,yuv``
# is the CLI twin.
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
def test_decode_tail_timing(*, mesh_device, diffvae_options, bench_options):
    _require_checkpoint()
    dec, config = bench.loaded_production_decoder(mesh_device, diffvae_options, bench_options.ccl(mesh_device))
    latent = bench.latent(config, bench_options.latent_frames(), seed=3)
    for kind in ("float", "yuv"):
        out, dt = bench.timed_decode(dec, latent, mesh_device, output_type=kind)
        print(f"\n[{kind:5s}] {dt:9.1f} ms  out={tuple(out.shape)}", flush=True)
        del out


# Trace capture and replay of the whole decode (``decode``) or the deterministic stages alone
# (``det_context``). The upload stays outside the captured region, so the raw latent is uploaded
# once into a buffer the capture reads and the replays read again. TT_DIT_STAGE_TIMING must be
# unset. ``time_module decoder --trace`` / ``time_module det_stages --trace`` are the CLI twins.
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": bench.TRACE_REGION_SIZE}],
    indirect=True,
    ids=["ring"],
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("region", ["decode", "det_context"])
def test_decode_trace_timing(*, mesh_device, device_params, region, diffvae_options, bench_options):
    _require_checkpoint()
    dec, config = bench.loaded_production_decoder(mesh_device, diffvae_options, bench_options.ccl(mesh_device))
    latent = bench.latent(config, bench_options.latent_frames(4))
    raw = bench.upload_latent(dec, latent, mesh_device)
    run = bench.trace_region(dec, region, latent, raw)
    report = bench.trace_replay(mesh_device, run, bench_options.iterations(3), probe_dispatch=region == "decode")
    assert report.identical, f"replay differs from eager by {report.max_abs_diff}"


# The pipeline's traced path: the pipeline flips ``_vae_traced`` after warm-up and ``forward``
# captures its device half on the first call. Needs ``device_boundaries`` (the production options
# have it) and TT_DIT_STAGE_TIMING unset. ``time_module decoder --vae-traced`` is the CLI twin.
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": bench.TRACE_REGION_SIZE}],
    indirect=True,
    ids=["ring"],
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("output_type", ["float", "yuv"])
def test_decode_traced_forward_matches_eager(
    *, mesh_device, device_params, output_type, diffvae_options, bench_options
):
    _require_checkpoint()
    dec, config = bench.loaded_production_decoder(mesh_device, diffvae_options, bench_options.ccl(mesh_device))
    latent = bench.latent(config, bench_options.latent_frames(4))
    report = bench.traced_forward_check(dec, latent, mesh_device, output_type=output_type)
    assert report.identical, "traced forward differs from eager"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": bench.TRACE_REGION_SIZE}],
    indirect=True,
    ids=["ring"],
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
def test_trace_reexecutes(*, mesh_device, device_params, diffvae_options, bench_options):
    """A captured trace of the deterministic stages re-executes rather than leaving a stale buffer.

    A bit-identical replay is not evidence on its own: the capture's output allocation can reuse
    the address the eager output was just freed from. Poisoning the output and swapping the input in
    place after capture are what tell a replay apart from a no-op. ``time_module trace_check`` is the
    CLI twin.
    """
    _require_checkpoint()
    dec, config = bench.loaded_production_decoder(mesh_device, diffvae_options, bench_options.ccl(mesh_device))
    t_lat = bench_options.latent_frames(4)
    result = bench.trace_validate(
        dec, mesh_device, bench.latent(config, t_lat, seed=1), bench.latent(config, t_lat, seed=2)
    )
    assert result.eager_reproducible
    assert result.replay_matches_a, "poisoned output not rewritten by the replay"
    assert result.replay_follows_input, "replay ignored the swapped input"
    assert result.replay_matches_b, "replay on the swapped input does not match eager"


# Stage 5 on the linear-order executor across the mesh: its shard splits QUERY tiles across all 32
# chips (K/V stay replicated per chip), so the per-call gather shrinks 32x. The activation stays
# replicated (full memory), so this is the 25-frame path, not the 6s path.
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("latent_hw", [(16, 16), (34, 60)], ids=["s16", "s34x60"])
def test_decode_gather_mesh_timing(*, mesh_device, latent_hw, bench_options):
    _require_checkpoint()
    config = decoder_config(CHECKPOINT)
    lh, lw = latent_hw
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 4, lh, lw)
    ccl = bench_options.ccl(mesh_device)
    dec = DiffVAEDecoder(config, mesh_device=mesh_device, ccl_manager=ccl, options=DiffVAEOptions())
    dec.load_checkpoint(CHECKPOINT)

    px, dt = bench.timed_decode(dec, latent, mesh_device)
    print(
        f"\n[decode gather-mesh 4x8] latent(1,{config['in_channels']},4,{lh},{lw}) -> {tuple(px.shape)}: {dt:8.0f} ms\n"
    )


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# PCC and runtime at 1080p 25 frames for the no-TP sharded path and the TP + column-parallel qkv
# sharded path, each against the gather-mesh decode as the reference (the dense-masked linear-order
# backend, the highest-fidelity path that fits at 1080p; single-chip replicated OOMs at the tail).
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
def test_decode_1080p_tp_pcc(*, mesh_device, bench_options):
    _require_checkpoint()
    from loguru import logger

    config = decoder_config(CHECKPOINT)
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 4, 34, 60)  # 1080p, 25 frames
    ccl = bench_options.ccl(mesh_device)

    def sharded(tp_axis, tp_proj):
        dec = DiffVAEDecoder(
            config,
            mesh_device=mesh_device,
            ccl_manager=ccl,
            options=dataclasses.replace(
                SHARDED, stage5_tp_axis=tp_axis, stages_tp_axis=tp_axis, stage5_tp_proj=tp_proj
            ),
        )
        dec.load_checkpoint(CHECKPOINT)
        px, dt = bench.timed_decode(dec, latent, mesh_device)
        return px.float(), dt

    ref_dec = DiffVAEDecoder(config, mesh_device=mesh_device, ccl_manager=ccl, options=DiffVAEOptions())
    ref_dec.load_checkpoint(CHECKPOINT)
    ref, t_ref = bench.timed_decode(ref_dec, latent, mesh_device)
    ref = ref.float()

    no_tp, t_no = sharded(None, False)
    tp_full, t_tp = sharded(0, True)

    logger.info(f"[1080p-pcc] gather-mesh reference:          runtime {t_ref:8.0f} ms")
    logger.info(f"[1080p-pcc] no-TP sharded:  PCC {_pcc(no_tp, ref) * 100:.4f} %   runtime {t_no:8.0f} ms")
    logger.info(f"[1080p-pcc] TP + col-qkv:   PCC {_pcc(tp_full, ref) * 100:.4f} %   runtime {t_tp:8.0f} ms")


# Does W-sharding stage 5 change what it computes? Window placement must stay GLOBAL: a query
# within half a context window of a shard seam still needs a full window, clamped only at the
# true volume boundary. Getting that wrong truncates the receptive field along every internal
# edge and still returns plausible video, so this compares pixels rather than trusting it.
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("latent_hw", [(16, 16)], ids=["s16"])
def test_decode_wsp_shard_equivalence(*, mesh_device, latent_hw, bench_options):
    _require_checkpoint()
    config = decoder_config(CHECKPOINT)
    lh, lw = latent_hw
    torch.manual_seed(0)
    t_lat = bench_options.latent_frames(8)
    latent = torch.randn(1, config["in_channels"], t_lat, lh, lw)
    ccl = bench_options.ccl(mesh_device, default_links=2)

    pixels = {}
    for backend in ("bricked", "bricked_sp_w_sharded"):
        decoder = DiffVAEDecoder(
            config,
            mesh_device=mesh_device,
            ccl_manager=ccl,
            options=DiffVAEOptions(stage5_backend=backend, stage5_sp_axis=1),
        )
        decoder.load_checkpoint(CHECKPOINT)
        pixels[backend] = decoder.decode(latent, seed=0).float()
        ttnn.synchronize_device(mesh_device)

    replicated, sharded = pixels["bricked"], pixels["bricked_sp_w_sharded"]
    correlation = _pcc(replicated, sharded)
    spread = (replicated.max() - replicated.min()).item()
    drift = ((replicated - sharded).abs().mean() / spread * 100).item()
    print(f"\n[shard equivalence] PCC {correlation:.6f}   mean |difference| {drift:.4f}% of range\n")
    assert correlation > 0.999, f"W-sharded stage 5 disagrees with replicated: PCC {correlation:.6f}"

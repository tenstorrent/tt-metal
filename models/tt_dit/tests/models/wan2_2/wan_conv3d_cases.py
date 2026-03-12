# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Wan Conv3d sweep case manifest.

Single source of truth for all production-relevant conv3d shapes, resolutions,
mesh families, axis layouts, dtypes, and cache modes used by the Wan2.2 VAE.

Each case is a dict with keys:
    case_id        – unique human-readable label
    path           – symbolic location in the model (e.g. "t2v.decoder.conv_in")
    kind           – "decoder" or "encoder"
    B, C_in, C_out – batch/channel dims (C_in is the raw, unpadded value)
    T, H, W        – temporal/spatial dims at 480p (see _scale_to_720p)
    kernel_size    – int or tuple
    stride         – int
    padding        – int or tuple
    resolution     – "480p" or "720p"
    cache_modes    – list of cache_len values to sweep (None = no cache)

The manifest is resolution-parametric: base shapes are 480p and _scale_to_720p()
derives 720p dimensions automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TILE_WIDTH = 32


def aligned_channels(channels: int) -> int:
    """Pad channels up to the next multiple of 32."""
    remainder = channels % TILE_WIDTH
    return channels if remainder == 0 else channels + (TILE_WIDTH - remainder)


def _scale_to_720p(case: dict) -> dict:
    """Return a copy of a 480p case with H/W scaled to 720p."""
    assert case["resolution"] == "480p"
    # 480p base: H=720/W=1280 at the output end; internal dims scale proportionally
    # Scale factor: 720/480 = 1.5 for H, 1280/832 ~= 1.538 for W
    # Actual 720p dims from the model: (720,1280) vs (480,832) -> factor ~1.5
    c = dict(case)
    c["H"] = int(round(case["H"] * 720 / 480))
    c["W"] = int(round(case["W"] * 1280 / 832))
    c["resolution"] = "720p"
    c["case_id"] = case["case_id"].replace("480p", "720p")
    return c


# ---------------------------------------------------------------------------
# 480p decoder conv3d shapes (9 shapes from test_wan_conv3d)
# ---------------------------------------------------------------------------

DECODER_CONV3D_CASES_480P = [
    dict(
        case_id="dec_conv_in_480p",
        path="t2v.decoder.conv_in",
        kind="decoder",
        B=1,
        C_in=16,
        C_out=384,
        T=1,
        H=90,
        W=160,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        resolution="480p",
        cache_modes=[None],
    ),
    dict(
        case_id="dec_mid_conv1_480p",
        path="t2v.decoder.mid_block.resnets.0.conv1",
        kind="decoder",
        B=1,
        C_in=384,
        C_out=384,
        T=1,
        H=90,
        W=160,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        resolution="480p",
        cache_modes=[None, 1, 2],
    ),
    dict(
        case_id="dec_up1_conv1_480p",
        path="t2v.decoder.up_blocks.1.resnets.0.conv1",
        kind="decoder",
        B=1,
        C_in=192,
        C_out=384,
        T=2,
        H=180,
        W=320,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        resolution="480p",
        cache_modes=[None, 1, 2],
    ),
    dict(
        case_id="dec_up1_conv2_480p",
        path="t2v.decoder.up_blocks.1.resnets.0.conv2",
        kind="decoder",
        B=1,
        C_in=384,
        C_out=384,
        T=2,
        H=180,
        W=320,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        resolution="480p",
        cache_modes=[None, 1, 2],
    ),
    dict(
        case_id="dec_up2_conv1_480p",
        path="t2v.decoder.up_blocks.2.resnets.0.conv1",
        kind="decoder",
        B=1,
        C_in=192,
        C_out=192,
        T=4,
        H=360,
        W=640,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        resolution="480p",
        cache_modes=[None, 1, 2],
    ),
    dict(
        case_id="dec_up3_conv1_480p",
        path="t2v.decoder.up_blocks.3.resnets.0.conv1",
        kind="decoder",
        B=1,
        C_in=96,
        C_out=96,
        T=4,
        H=720,
        W=1280,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        resolution="480p",
        cache_modes=[None, 1, 2],
    ),
    dict(
        case_id="dec_conv_out_480p",
        path="t2v.decoder.conv_out",
        kind="decoder",
        B=1,
        C_in=96,
        C_out=3,
        T=4,
        H=720,
        W=1280,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        resolution="480p",
        cache_modes=[None, 1, 2],
    ),
    # Time-conv (1D temporal) shapes — (3,1,1) kernel
    dict(
        case_id="dec_resample_time_conv_t1_480p",
        path="t2v.decoder.up_blocks.0.upsamplers.0.time_conv",
        kind="decoder",
        B=1,
        C_in=384,
        C_out=768,
        T=1,
        H=90,
        W=160,
        kernel_size=(3, 1, 1),
        stride=1,
        padding=(1, 0, 0),
        resolution="480p",
        cache_modes=[None],
    ),
    dict(
        case_id="dec_resample_time_conv_t2_480p",
        path="t2v.decoder.up_blocks.0.upsamplers.0.time_conv",
        kind="decoder",
        B=1,
        C_in=384,
        C_out=768,
        T=2,
        H=180,
        W=320,
        kernel_size=(3, 1, 1),
        stride=1,
        padding=(1, 0, 0),
        resolution="480p",
        cache_modes=[None, 1, 2],
    ),
]

# ---------------------------------------------------------------------------
# 480p encoder conv3d shapes (4 shapes from encoder path)
# Encoder shapes: conv_in (3->96), down_block residuals, conv_out (384->z_dim*2=32)
# ---------------------------------------------------------------------------

ENCODER_CONV3D_CASES_480P = [
    dict(
        case_id="enc_conv_in_480p",
        path="t2v.encoder.conv_in",
        kind="encoder",
        B=1,
        C_in=3,
        C_out=96,
        T=4,
        H=720,
        W=1280,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        resolution="480p",
        cache_modes=[None, 1, 2],
    ),
    dict(
        case_id="enc_res_96_192_480p",
        path="t2v.encoder.down_blocks.residual_96_192",
        kind="encoder",
        B=1,
        C_in=96,
        C_out=192,
        T=4,
        H=360,
        W=640,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        resolution="480p",
        cache_modes=[None, 1, 2],
    ),
    dict(
        case_id="enc_res_192_384_480p",
        path="t2v.encoder.down_blocks.residual_192_384",
        kind="encoder",
        B=1,
        C_in=192,
        C_out=384,
        T=2,
        H=180,
        W=320,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        resolution="480p",
        cache_modes=[None, 1, 2],
    ),
    dict(
        case_id="enc_conv_out_480p",
        path="t2v.encoder.conv_out",
        kind="encoder",
        B=1,
        C_in=384,
        C_out=32,
        T=1,
        H=90,
        W=160,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=1,
        resolution="480p",
        cache_modes=[None],
    ),
]

# ---------------------------------------------------------------------------
# All 480p cases combined
# ---------------------------------------------------------------------------

ALL_CASES_480P = DECODER_CONV3D_CASES_480P + ENCODER_CONV3D_CASES_480P

# Derive 720p variants
ALL_CASES_720P = [_scale_to_720p(c) for c in ALL_CASES_480P]

ALL_CASES = ALL_CASES_480P + ALL_CASES_720P


# ---------------------------------------------------------------------------
# Production mesh / axis layout targets
# ---------------------------------------------------------------------------


@dataclass
class MeshTarget:
    """A production-relevant mesh + axis layout."""

    mesh_id: str
    mesh_shape: tuple[int, int]
    h_axis: int
    w_axis: int

    @property
    def h_factor(self) -> int:
        return self.mesh_shape[self.h_axis]

    @property
    def w_factor(self) -> int:
        return self.mesh_shape[self.w_axis]


# Meshes actually used by pipeline/perf tests (test_pipeline_wan.py, test_performance_wan.py)
PRODUCTION_MESH_TARGETS = [
    # 1x8 — low-cost loudbox validation (WH LB with h_axis=0, w_axis=1)
    MeshTarget(mesh_id="1x8_h0_w1", mesh_shape=(1, 8), h_axis=0, w_axis=1),
    # 2x4 — WH T3K / BH LB (sp_axis=0, tp_axis=1 → h_axis=0, w_axis=1)
    MeshTarget(mesh_id="2x4_h0_w1", mesh_shape=(2, 4), h_axis=0, w_axis=1),
    # 4x8 — Galaxy (sp_axis=1, tp_axis=0 → h_axis=0, w_axis=1 for the VAE)
    MeshTarget(mesh_id="4x8_h0_w1", mesh_shape=(4, 8), h_axis=0, w_axis=1),
]

# BH-specific axis layout used by i2v pipeline (2x4 sp1tp0)
BH_MESH_TARGETS = [
    MeshTarget(mesh_id="2x4_h1_w0", mesh_shape=(2, 4), h_axis=1, w_axis=0),
]

ALL_MESH_TARGETS = PRODUCTION_MESH_TARGETS + BH_MESH_TARGETS


# ---------------------------------------------------------------------------
# Sweep stages — ordered from cheapest to most expensive
# ---------------------------------------------------------------------------


@dataclass
class SweepStage:
    """A sweep stage: a specific mesh + resolution combination."""

    stage_id: str
    mesh_target: MeshTarget
    resolution: str
    priority: int  # lower = run first

    @property
    def mesh_id(self) -> str:
        return self.mesh_target.mesh_id


SWEEP_STAGES = [
    # Phase 1: cheapest validation
    SweepStage("validate_1x8_480p", PRODUCTION_MESH_TARGETS[0], "480p", priority=1),
    # Phase 2: production 2x4
    SweepStage("prod_2x4_480p", PRODUCTION_MESH_TARGETS[1], "480p", priority=2),
    # Phase 3: production 4x8
    SweepStage("prod_4x8_480p", PRODUCTION_MESH_TARGETS[2], "480p", priority=3),
    # Phase 4: 720p on 4x8
    SweepStage("prod_4x8_720p", PRODUCTION_MESH_TARGETS[2], "720p", priority=4),
]

SWEEP_STAGES_BY_ID = {s.stage_id: s for s in SWEEP_STAGES}


# ---------------------------------------------------------------------------
# Candidate blocking generation (mesh-aware)
# ---------------------------------------------------------------------------


def compute_local_dims(case: dict, mesh_target: MeshTarget) -> dict:
    """
    Compute per-device local output dimensions for a case on a given mesh.

    Returns dict with H_out_local, W_out_local, aligned_C_in, C_out.
    """
    kernel = case["kernel_size"]
    if isinstance(kernel, int):
        k_h, k_w = kernel, kernel
        pad_h, pad_w = case["padding"], case["padding"]
    else:
        _, k_h, k_w = kernel if len(kernel) == 3 else (kernel[0], kernel[0], kernel[0])
        pad = case["padding"]
        if isinstance(pad, int):
            pad_h, pad_w = pad, pad
        else:
            _, pad_h, pad_w = pad if len(pad) == 3 else (pad[0], pad[0], pad[0])

    stride = case["stride"] if isinstance(case["stride"], int) else case["stride"][1]

    # Global output spatial dims
    H_out = (case["H"] + 2 * pad_h - k_h) // stride + 1
    W_out = (case["W"] + 2 * pad_w - k_w) // stride + 1

    # Pad H to be divisible by h_factor before fracturing
    h_factor = mesh_target.h_factor
    w_factor = mesh_target.w_factor
    H_padded = math.ceil(H_out / h_factor) * h_factor
    W_padded = math.ceil(W_out / w_factor) * w_factor

    return dict(
        H_out_local=H_padded // h_factor,
        W_out_local=W_padded // w_factor,
        aligned_C_in=aligned_channels(case["C_in"]),
        C_out=case["C_out"],
        H_out_global=H_out,
        W_out_global=W_out,
    )


def generate_candidates(case: dict, mesh_target: MeshTarget, current_default: tuple | None = None) -> list[tuple]:
    """
    Generate legal blocking candidates for a case on a given mesh.

    Each candidate is (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block).

    Strategy:
    - Start from current default (if any) as baseline
    - Search spatial (H_out_block, W_out_block) first
    - Then C_out_block
    - Then C_in_block
    - Keep candidate space pruned for practical sweep times
    """
    local = compute_local_dims(case, mesh_target)
    H_local = local["H_out_local"]
    W_local = local["W_out_local"]
    C_in = local["aligned_C_in"]
    C_out = local["C_out"]
    C_out_padded = aligned_channels(C_out)

    # Legal H_out_block values: divisors of H_local, plus powers of 2 up to H_local
    h_candidates = sorted(
        set(
            [h for h in range(1, H_local + 1) if H_local % h == 0]
            + [2**i for i in range(int(math.log2(max(H_local, 1))) + 1) if 2**i <= H_local]
        )
    )

    # Legal W_out_block values: divisors of W_local, skip 13 (known hang)
    w_candidates = sorted(
        set(
            [w for w in range(1, W_local + 1) if W_local % w == 0 and w != 13]
            + [2**i for i in range(int(math.log2(max(W_local, 1))) + 1) if 2**i <= W_local and 2**i != 13]
        )
    )

    # Legal C_out_block values: multiples of TILE_WIDTH up to C_out_padded
    c_out_candidates = sorted(set([TILE_WIDTH * i for i in range(1, C_out_padded // TILE_WIDTH + 1)]))

    # Legal C_in_block values: multiples of TILE_WIDTH up to C_in (but also C_in itself)
    c_in_candidates = sorted(set([TILE_WIDTH * i for i in range(1, C_in // TILE_WIDTH + 1)]))

    # T_out_block is always 1 for these models
    T_out_block = 1

    return dict(
        h_candidates=h_candidates,
        w_candidates=w_candidates,
        c_out_candidates=c_out_candidates,
        c_in_candidates=c_in_candidates,
        T_out_block=T_out_block,
        local_dims=local,
    )


def get_current_default(case: dict) -> tuple | None:
    """
    Look up the current production default blocking for a case from conv3d.py's table.

    Returns (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block) or None.
    """
    C_in_aligned = aligned_channels(case["C_in"])
    C_out = case["C_out"]
    kernel = case["kernel_size"]
    if isinstance(kernel, int):
        kernel = (kernel, kernel, kernel)
    elif len(kernel) == 3:
        kernel = tuple(kernel)

    # This mirrors the lookup in conv3d.py get_conv3d_config()
    BF16_DEFAULTS = {
        (96, 3, (3, 3, 3)): (96, 32, 1, 16, 8),
        (96, 32, (3, 3, 3)): (96, 32, 1, 16, 8),
        (192, 96, (1, 3, 3)): (192, 96, 1, 4, 8),
        (96, 96, (3, 3, 3)): (96, 96, 1, 8, 8),
        (384, 192, (1, 3, 3)): (192, 96, 1, 32, 4),
        (192, 192, (3, 3, 3)): (96, 96, 1, 8, 4),
        (32, 384, (3, 3, 3)): (32, 384, 1, 8, 8),
        (192, 384, (3, 3, 3)): (96, 128, 1, 32, 1),
        (384, 384, (3, 3, 3)): (128, 128, 1, 8, 2),
        (384, 768, (3, 3, 3)): (128, 128, 1, 16, 2),
    }

    return BF16_DEFAULTS.get((C_in_aligned, C_out, kernel), None)


# ---------------------------------------------------------------------------
# Sweep manifest builder
# ---------------------------------------------------------------------------


def build_sweep_manifest(
    stage: SweepStage,
    cases: list[dict] | None = None,
    cache_modes_override: list | None = None,
) -> list[dict]:
    """
    Build a flat list of sweep entries for a given stage.

    Each entry includes the case dict plus mesh/stage metadata.
    If cache_modes_override is given, it replaces each case's cache_modes.
    """
    if cases is None:
        # Filter cases to the stage's resolution
        cases = [c for c in ALL_CASES if c["resolution"] == stage.resolution]

    manifest = []
    for case in cases:
        if case["resolution"] != stage.resolution:
            continue

        cache_modes = cache_modes_override if cache_modes_override is not None else case["cache_modes"]

        for cache_len in cache_modes:
            cache_suffix = f"cache_{cache_len}" if cache_len is not None else "cache_none"
            entry = dict(
                **case,
                stage_id=stage.stage_id,
                mesh_target=stage.mesh_target,
                cache_len=cache_len,
                sweep_id=f"{case['case_id']}__{cache_suffix}",
                current_default=get_current_default(case),
            )
            manifest.append(entry)

    return manifest

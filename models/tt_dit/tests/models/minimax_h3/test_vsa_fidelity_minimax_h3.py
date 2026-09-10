# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""How far VSA moves the 4-forward FastH3 output, in PCC and MSE, against the dense reference.

Two comparisons, because one number cannot answer both questions:

- **attention only** -- the VSA student's adapter run with VSA on vs off, the off arm through
  ``MiniMaxH3VSAConfig.bypass`` so the gates still load. The adapter weights are held fixed, so the
  delta is the attention path alone. That arm is off-distribution -- a trained
  ``to_gate_compress`` sits idle with no compressed branch to gate -- which is the price of
  isolating the variable.
- **shipping delta** -- the VSA student under VSA against the *dense* student under dense
  attention. Both arms are in-distribution and both are shippable, but the adapter weights differ
  as well as the attention, and a 4-forward distilled sampler is chaotic under weight
  perturbation. A low number here is not evidence about VSA.

Scored on the denoised rows first, both modalities, and on the decoded uint8 frames second. The
VAE is nonlinear, so a pixel PCC reports the decoder's sensitivity folded in with the latent
difference under test.

Nothing here is gated: there is no calibrated bar for how far a sparse-attention student may drift
from a dense one. The arms run as separate tests so each gets a fresh mesh, and hand their rows to
the report through ``artifact_dir``; run the whole file in one invocation.

Requires ``MINIMAX_H3_LORA_DIR`` pointing at a FastH3 adapter bundle -- the directory holding
``adapter_manifest.json`` and one subdirectory per variant.
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSAConfig
from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from .common import GALAXY_MESHES
from .common_av import CALIBRATED_FOX_PROMPT, artifact_dir, to_uint8_frames, weights_dir

NUM_INFERENCE_STEPS = 5  # five sigma grid points, four transformer forwards
SEED = 0
ASPECT_RATIO = (16, 9)
DURATION_S = 15.0
VSA_SPARSITY = 0.9  # the sparsity the VSA students were distilled against (vsa-h3-90pct-tile64)

# arm -> (variant slug, VSA config). `bypass` loads the gates and runs dense anyway, which is what
# makes the vsa_off arm a same-weights comparison rather than a different model.
ARMS = {
    "vsa": ("vsa-datafree", MiniMaxH3VSAConfig(sparsity=VSA_SPARSITY)),
    "vsa_off": ("vsa-datafree", MiniMaxH3VSAConfig(sparsity=VSA_SPARSITY, bypass=True)),
    "dense": ("dense-datafree", None),
}

# (arm, reference arm, what the pair isolates)
PAIRS = (
    ("vsa", "vsa_off", "attention only: same adapter, VSA vs dense"),
    ("vsa", "dense", "shipping delta: VSA student vs dense student"),
)

FIELDS = ("video_rows", "audio_rows", "frames")


def _bundle() -> Path:
    directory = os.environ.get("MINIMAX_H3_LORA_DIR")
    if not directory:
        pytest.skip("set MINIMAX_H3_LORA_DIR to a FastH3 adapter bundle directory")
    return Path(directory)


def _adapter_path(bundle: Path, slug: str) -> Path:
    """Resolve a variant slug through the bundle manifest.

    Through the manifest rather than by globbing, so the log carries which variant an arm actually
    ran -- attention, training source and step -- next to its numbers. Two of the three variants
    are 5.3 GB files whose names differ by a directory, and a swapped arm is otherwise invisible.
    """
    manifest = json.loads((bundle / "adapter_manifest.json").read_text())
    variants = {variant["slug"]: variant for variant in manifest["variants"]}
    assert slug in variants, f"{slug} is not in the bundle; it carries {sorted(variants)}"
    variant = variants[slug]
    path = bundle / variant["adapter_path"]
    assert path.exists(), f"{slug} is in the manifest but {path} is missing"
    logger.info(
        f"{slug}: {variant['display_name']}, attention {variant['attention']}, "
        f"requires_vsa={variant['requires_vsa']}, step {variant['training_step']}"
    )
    return path


def _arm_file(arm: str) -> Path:
    return artifact_dir("h3_vsa_fidelity") / f"{arm}_{DURATION_S:.0f}s_seed{SEED}.pt"


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """PCC, MSE and RMSE relative to the reference's own spread; ``b`` is the reference.

    Accumulated in chunks rather than through ``torch.cov``, which would need two float64 copies of
    the whole tensor -- 37 GB for a 15 s frame stack.
    """
    a, b = a.detach().flatten(), b.detach().flatten()
    n = a.numel()
    moments = torch.zeros(5, dtype=torch.float64)
    for start in range(0, n, 1 << 24):
        x = a[start : start + (1 << 24)].to(torch.float64)
        y = b[start : start + (1 << 24)].to(torch.float64)
        moments += torch.stack([x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()])
    sum_a, sum_b, sum_aa, sum_bb, sum_ab = (value.item() for value in moments)
    mean_a, mean_b = sum_a / n, sum_b / n
    var_a, var_b = sum_aa / n - mean_a**2, sum_bb / n - mean_b**2
    covariance = sum_ab / n - mean_a * mean_b
    mse = (sum_aa - 2 * sum_ab + sum_bb) / n
    std_a, std_b = math.sqrt(max(var_a, 0.0)), math.sqrt(max(var_b, 0.0))
    return {
        "pcc": covariance / (std_a * std_b) if std_a and std_b else float("nan"),
        "mse": mse,
        "relative_rmse": math.sqrt(max(mse, 0.0)) / std_b if std_b else float("nan"),
        "std_ref": std_b,
    }


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("arm", list(ARMS), ids=list(ARMS))
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES[:1], indirect=["mesh_device", "device_params"])
def test_denoise_arm(mesh_device, reset_seeds, arm):
    """One arm of the A/B: generate at the shared working point and hand the rows to the report."""
    slug, vsa_config = ARMS[arm]
    vsa_on = vsa_config is not None and not vsa_config.bypass
    bundle = _bundle()
    lora_path = _adapter_path(bundle, slug)

    if not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning("TT_DIT_CACHE_DIR is unset; every weight load reads safetensors and the run will drag")

    height, width = resolve_canvas_size(*ASPECT_RATIO)
    num_frames = align_num_frames(round(DURATION_S * MINIMAX_H3_FPS))

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir("transformer", "text_encoder", "vae", "audio_vae"),
        lora_path=str(lora_path),
        lora_strength=float(os.environ.get("FASTH3_LORA_STRENGTH", 1.0)),
        vsa_config=vsa_config,
    )

    # Cold, not warm: the arms are compared against each other and a warm run computes the same
    # thing as a cold one, so a warmup generation per arm would double the device time for nothing.
    output = pipeline(
        CALIBRATED_FOX_PROMPT,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    report = pipeline._lora_report
    assert report is not None and report.bound, f"{arm}: the transformer was built without an adapter bound"
    if vsa_config is not None:
        # A gate that fails to land is not a crash: the run completes through the ungated compressed
        # branch and looks like a success, which would silently make this the wrong measurement.
        assert len(report.replaced) == pipeline.transformer_config["num_layers"], (
            f"{arm}: {len(report.replaced)} gates assigned for "
            f"{pipeline.transformer_config['num_layers']} blocks; VSA is running partly ungated"
        )

    video_rows, audio_rows = pipeline.last_video_rows, pipeline.last_audio_rows
    assert video_rows is not None and audio_rows is not None, f"{arm}: the pipeline exposed no denoised rows"
    frames = torch.from_numpy(to_uint8_frames(output))
    payload = {
        "arm": arm,
        "slug": slug,
        "vsa": vsa_on,
        "sparsity": VSA_SPARSITY if vsa_on else None,
        "bypass": vsa_config is not None and vsa_config.bypass,
        "seed": SEED,
        "duration_s": DURATION_S,
        "padded_len": pipeline.last_padded_len,
        "video_rows": video_rows,
        "audio_rows": audio_rows,
        "frames": frames,
    }
    torch.save(payload, _arm_file(arm))
    logger.info(
        f"{arm}: video rows {tuple(video_rows.shape)}, audio rows {tuple(audio_rows.shape)}, "
        f"frames {tuple(frames.shape)}, padded_len={pipeline.last_padded_len} -> {_arm_file(arm)}"
    )


def test_fidelity_report():
    """PCC and MSE for every pair, from the arm files. Host only; run after the arms in one session."""
    missing = [arm for arm in ARMS if not _arm_file(arm).exists()]
    if missing:
        pytest.skip(f"arms {missing} have not run; run the whole file in one invocation")

    arms = {arm: torch.load(_arm_file(arm), weights_only=False) for arm in ARMS}
    # VSA tile order pads the packed sequence to its own geometry, so the arms need not agree here;
    # the denoised rows are the logical rows either way and their shapes are checked per field.
    for arm, payload in arms.items():
        logger.info(
            f"{arm}: variant {payload['slug']}, vsa={payload['vsa']}, bypass={payload['bypass']}, "
            f"padded_len={payload['padded_len']}"
        )

    for arm, reference, what in PAIRS:
        logger.info(f"=== {arm} vs {reference} -- {what}")
        for field in FIELDS:
            a, b = arms[arm][field], arms[reference][field]
            assert a.shape == b.shape, f"{field}: {tuple(a.shape)} != {tuple(b.shape)}"
            found = _metrics(a, b)
            logger.info(
                f"  {field:<11} PCC = {found['pcc'] * 100:8.4f} %   MSE = {found['mse']:.6g}   "
                f"RMSE/σ_ref = {found['relative_rmse'] * 100:6.2f} %   (σ_ref = {found['std_ref']:.4g})"
            )

    # Structural only: the numbers above are recorded, not gated.
    for arm, payload in arms.items():
        for field in FIELDS:
            assert torch.isfinite(payload[field].float()).all(), f"{arm}: {field} is not finite"

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The input set both sides of the PCC comparison receive, byte-identical.

Why this file exists
--------------------
This checkpoint has no processor, tokenizer or feature extractor of its own --
it is the *transformer* subfolder of FLUX.2 Klein 9B, and its real input is
whatever the enclosing latent-diffusion pipeline hands it. Source A's own
``diffusers.Flux2Pipeline`` defines every piece, so that recipe is reproduced
here rather than invented:

* ``latents``  = ``randn(B, 128, H//16, W//16)`` then ``_pack_latents`` ->
  ``[B, S_img, 128]``   (``prepare_latents``, pipeline_flux2.py:621-652)
* ``img_ids``  = ``cartesian_prod(arange(1), arange(h), arange(w), arange(1))``
  (``_prepare_latent_ids``, pipeline_flux2.py:375-404)
* ``txt_ids``  = ``cartesian_prod(arange(1), arange(1), arange(1), arange(L))``
  (``_prepare_text_ids``, pipeline_flux2.py:356-372)
* ``timestep`` = ``t / 1000`` with ``t`` from the flow-match schedule; the model's
  own ``forward`` multiplies by 1000 again (transformer_flux2.py:1231)
* ``guidance`` = ``None`` -- ``guidance_embeds`` is false in this config, so the
  guidance branch of ``time_guidance_embed`` is dead code

The one piece with no Source-A recipe available *inside this component* is
``encoder_hidden_states``: it is produced by the text encoder, a SEPARATE model
(a Qwen3 in the sibling ``flux_2_klein_9b_text_encoder`` component) that is not
shipped here. It is therefore a seeded stand-in of exactly the right shape and
dtype ``[B, L, 12288]``. That is honest because both the TT pipeline and the HF
golden consume the same tensor, so the PCC still measures this component.

Determinism
-----------
One ``torch.Generator`` seeded once, drawn in a fixed order (latents, then the
prompt-embedding stand-in). Same seed -> same bytes, on any host, in any process,
which is what makes the demo, the e2e test and the trace seams comparable.

The captured goldens, and where they disagree with the recipe
-------------------------------------------------------------
Two pieces are anchored to Source B's ``_captured/`` recordings of a real
reference forward rather than guessed. :func:`captured_timestep` reads
``timesteps/args.pt`` = ``500.0``, which is the input of the sinusoidal
``time_proj`` and therefore ``t*1000``: it confirms the pipeline-level timestep
was 0.5, exactly the ``timestep=t/1000`` convention, and it is the default this
builder uses.

:func:`captured_txt_ids` reads ``flux2_pos_embed/args.pt`` = an ``[8, 4]``
float32 tensor. MEASURED: that tensor is **all zeros**, not
``cartesian_prod(arange(1), arange(1), arange(1), arange(8))``. It was produced
by the bring-up capture driver
(``scripts/tt_hw_planner/learned_drivers/flux2transformer2dmodel.py``, whose
``make_ids(txt_seq_len, spatial=False)`` returns ``torch.zeros(seq_len, 4)``),
i.e. it is a synthetic probe that pinned only the SHAPE, RANK, DTYPE and axis
count of the txt id tensor -- a degenerate case of the real pattern in which the
one varying axis happens to be constant.

So the capture agrees with the recipe on structure and disagrees on values, and
this builder deliberately follows Source A's ``_prepare_text_ids`` (positions
0..L-1 on the last axis), because that is what the transformer is handed in
production and what makes text RoPE non-degenerate. Both the TT pipeline and the
HF golden receive this same tensor, so the PCC stays meaningful either way; the
recipe is the one that also stays meaningful at L=512.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .stubs import REPO_ROOT, SOURCE_B

# From e2e_plan.json / config.json: in_channels 128 (= 32 VAE channels x 2x2
# packing) and joint_attention_dim 12288 (the text encoder's hidden width).
LATENT_CHANNELS = 128
JOINT_ATTENTION_DIM = 12288

# vae_scale_factor 8 x 2x2 latent packing = 16 pixels per packed token per axis.
PIXELS_PER_LATENT_TOKEN = 16

_CAPTURED = REPO_ROOT / SOURCE_B / "_captured"

# Horizon safety cap (e2e_plan.json horizon.safety_cap): a diffusion transformer
# has no stop token, so the only thing bounding the loop is this clamp.
MIN_STEPS = 1
MAX_STEPS = 50

_CAPTURE_CACHE: dict[str, Any] = {}


# ----------------------------------------------------------------- geometry


def latent_grid(height: int, width: int) -> tuple[int, int]:
    """Source A recipe: ``(height // 16, width // 16)``.

    Source A spells it as ``2 * (H // (vae_scale_factor * 2))`` then ``// 2``,
    which for vae_scale_factor 8 is exactly ``H // 16`` -- and the ``2 *``/``// 2``
    round trip is what forces the even latent size the 2x2 packing needs.
    """
    grid_h = int(height) // PIXELS_PER_LATENT_TOKEN
    grid_w = int(width) // PIXELS_PER_LATENT_TOKEN
    if grid_h < 1 or grid_w < 1:
        raise ValueError(
            f"height/width must be at least {PIXELS_PER_LATENT_TOKEN}px "
            f"(got {height}x{width} -> latent grid {grid_h}x{grid_w})"
        )
    return grid_h, grid_w


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W] -> [B, H*W, C]`` (Source A ``_pack_latents``, pipeline_flux2.py:472-482)."""
    batch, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).permute(0, 2, 1)


# ------------------------------------------------------------- captured reads


def _load_captured_args(component: str) -> tuple:
    """The recorded positional args of one captured reference forward."""
    if component not in _CAPTURE_CACHE:
        path = _CAPTURED / component / "args.pt"
        if not path.is_file():
            raise FileNotFoundError(f"no captured args for {component!r} at {path}")
        # weights_only=True is the safe default and is enough here: the captures are
        # plain tuples of tensors. Older/newer torch defaults differ, so it is explicit.
        _CAPTURE_CACHE[component] = torch.load(path, map_location="cpu", weights_only=True)
    args = _CAPTURE_CACHE[component]
    return tuple(args) if isinstance(args, (tuple, list)) else (args,)


def captured_timestep() -> float:
    """The timestep the bring-up harness captured, as the model-level ``t*1000``.

    ``_captured/timesteps/args.pt`` is the input of the sinusoidal ``time_proj``,
    i.e. the value *after* ``forward``'s ``timestep * 1000``. So 500.0 here means
    the pipeline-level timestep (and hence the flow-match sigma) was 0.5.
    """
    for arg in _load_captured_args("timesteps"):
        if torch.is_tensor(arg) and arg.numel() == 1:
            return float(arg.reshape(-1)[0])
    raise ValueError("captured timesteps/args.pt holds no 1-element tensor")


def captured_txt_ids() -> torch.Tensor:
    """The ``[8, 4]`` text position ids captured at ``pos_embed``.

    Returned verbatim, which means all zeros: see the module docstring. The capture
    is a synthetic bring-up probe (8 tokens, 4 axes, float32) that fixes the shape
    and dtype ``pos_embed`` really was called with, not Source A's positional
    pattern. :func:`build_inputs` uses the Source-A recipe instead and matches this
    tensor's rank/width/dtype, which is the part of the capture that is load-bearing.
    """
    for arg in _load_captured_args("flux2_pos_embed"):
        if torch.is_tensor(arg) and arg.ndim == 2 and arg.shape[-1] == 4:
            return arg.to(torch.float32).clone()
    raise ValueError("captured flux2_pos_embed/args.pt holds no [S, 4] ids tensor")


# ---------------------------------------------------------------- input set


def build_inputs(
    height: int = 256,
    width: int = 256,
    txt_len: int = 64,
    batch: int = 1,
    seed: int = 0,
    timestep: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """The deterministic, Source-A-recipe input set for one denoise forward.

    Both the TT pipeline and the HF golden are handed this exact dict, so any PCC
    difference is the port's, not the input's.
    """
    if batch < 1 or txt_len < 1:
        raise ValueError(f"batch and txt_len must be >= 1 (got batch={batch}, txt_len={txt_len})")

    grid_h, grid_w = latent_grid(height, width)
    s_img = grid_h * grid_w
    s_txt = int(txt_len)

    # One generator, drawn in a fixed order. Both draws are made in float32 and
    # only then cast: that way `dtype` selects the *precision handed to the model*
    # without also changing which random numbers were drawn, so a bfloat16 run and
    # a float32 run are the same input at two precisions.
    generator = torch.Generator().manual_seed(int(seed))

    latents = torch.randn(batch, LATENT_CHANNELS, grid_h, grid_w, generator=generator, dtype=torch.float32)
    hidden_states = _pack_latents(latents).to(dtype)

    # Stand-in for the separate text encoder. The 0.5 scale keeps it in the range a
    # real Qwen3 last-hidden-state occupies, so the modulation/attention numerics
    # the PCC measures are exercised at a realistic magnitude rather than saturating.
    encoder_hidden_states = (
        torch.randn(batch, s_txt, JOINT_ATTENTION_DIM, generator=generator, dtype=torch.float32) * 0.5
    ).to(dtype)

    # Position ids stay float32 regardless of `dtype`: they are consumed by the RoPE
    # table build, where the phase `ids * inv_freq` reaches ~S radians and bfloat16
    # would cost ~0.25 rad of phase error (see the graduated flux2_pos_embed stub).
    img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(grid_h), torch.arange(grid_w), torch.arange(1)).to(
        torch.float32
    )
    txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(s_txt)).to(
        torch.float32
    )

    # Model-level timestep (pre *1000). Default = the captured 500.0 / 1000 = 0.5, so
    # the e2e input is anchored to a real recorded forward. Kept float32 even when
    # `dtype` is lower precision: `forward` casts it to the hidden dtype anyway, and
    # a rounded sigma would silently change the schedule Call 2 walks.
    t_value = captured_timestep() / 1000.0 if timestep is None else float(timestep)
    timestep_tensor = torch.full((batch,), t_value, dtype=torch.float32)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep_tensor,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        # guidance_embeds=false in this config: passing a tensor here would drive a
        # branch the checkpoint has no weights for.
        "guidance": None,
        "meta": {
            "height": int(height),
            "width": int(width),
            "txt_len": s_txt,
            "S_img": s_img,
            "S_txt": s_txt,
            "S_joint": s_txt + s_img,
            "grid": (grid_h, grid_w),
            "seed": int(seed),
            "batch": int(batch),
        },
    }


# ------------------------------------------------------------ flow-match schedule


def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Source A's ``compute_empirical_mu``, reproduced exactly.

    Copied from ``python_env/lib/python3.10/site-packages/diffusers/pipelines/flux2/
    pipeline_flux2.py`` lines 157-175 (itself taken from black-forest-labs/flux2
    ``src/flux2/sampling.py``:L251). The two calibration lines are fits of the
    resolution-dependent shift at 10 and 200 steps; below 4300 image tokens the
    result is interpolated between them in ``num_steps``.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def sigma_schedule(num_steps: int, image_seq_len: int) -> list[float]:
    """Source A's flow-match Euler sigma schedule, as host-side python floats.

    Reproduces, for the ``mu is not None`` path, what
    ``FlowMatchEulerDiscreteScheduler.set_timesteps`` actually does
    (``python_env/lib/python3.10/site-packages/diffusers/schedulers/
    scheduling_flow_match_euler_discrete.py`` lines 282-385):

    1. sigmas arrive as the pipeline's ``np.linspace(1.0, 1/N, N)``
       (pipeline_flux2.py:934) and are cast to float32 (scheduler L341-343);
    2. ``use_dynamic_shifting`` is on for FLUX, so L346-347 applies
       ``time_shift(mu, 1.0, sigmas)``, and ``time_shift_type`` defaults to
       ``"exponential"`` -> ``_time_shift_exponential`` (L648-649):
       ``exp(mu) / (exp(mu) + (1/t - 1) ** 1.0)``;
    3. ``shift_terminal`` / karras / exponential / beta conversions are all unset
       for this pipeline, so L350-360 are no-ops;
    4. ``invert_sigmas`` is false, so L376-379 appends a single trailing 0.0.

    Hence the returned list has length ``N + 1`` and step ``i`` runs
    ``sigmas[i] -> sigmas[i+1]``. ``num_steps`` is clamped to [1, 50]: there is no
    stop token on a denoise loop, so the cap is the only bound.

    float32 is preserved (rather than computing in float64) so the numbers are the
    ones diffusers itself would hand the model -- the schedule is shared by the TT
    loop and the HF golden, so it must be one list, computed once.
    """
    steps = max(MIN_STEPS, min(MAX_STEPS, int(num_steps)))
    mu = _compute_empirical_mu(int(image_seq_len), steps)

    sigmas = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
    exp_mu = math.exp(mu)
    # `** 1.0` is `sigma=1.0` from `time_shift(mu, 1.0, sigmas)`; kept explicit so
    # the line reads as the scheduler's formula rather than a simplification of it.
    shifted = exp_mu / (exp_mu + (1.0 / sigmas - 1.0) ** 1.0)

    return [float(value) for value in np.asarray(shifted, dtype=np.float32)] + [0.0]


# --------------------------------------------------------------- unpacking


def unpack_latents(latents: torch.Tensor, img_ids: torch.Tensor) -> torch.Tensor:
    """Source A's ``_unpack_latents_with_ids``: ``[B, S, C] -> [B, C, h, w]``.

    Reproduced from pipeline_flux2.py:483-506. Pure shape work -- a scatter driven
    by the position ids, which is what lets a token order other than raster order
    still land in the right cell (the ids, not the sequence index, define position).

    Source A always holds ids as ``[B, S, 4]``; this component's contract carries
    them as ``[S, 4]`` (the same shape ``forward`` reduces them to internally), so a
    rank-2 tensor is broadcast over the batch here.
    """
    ids = img_ids
    if ids.ndim == 2:
        ids = ids.unsqueeze(0).expand(latents.shape[0], -1, -1)

    unpacked = []
    for data, pos in zip(latents, ids):
        _, channels = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        height = int(torch.max(h_ids)) + 1
        width = int(torch.max(w_ids)) + 1
        flat_ids = h_ids * width + w_ids

        out = torch.zeros((height * width, channels), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, channels), data)
        unpacked.append(out.view(height, width, channels).permute(2, 0, 1))

    return torch.stack(unpacked, dim=0)

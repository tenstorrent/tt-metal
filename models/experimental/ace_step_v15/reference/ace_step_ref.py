# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU (fp32) reference harness for the ACE-Step 1.5 TTNN bringup — Block 0.

This is a *thin* wrapper around the `diffusers` implementation. Nothing is
re-implemented here: `diffusers` 0.38.0 ships ACE-Step 1.5 natively
(`AceStepTransformer1DModel`, `AceStepConditionEncoder`, `AutoencoderOobleck`,
`AceStepPipeline`) and that is the golden. All this module adds is

  1. `load_pipeline()`   — fp32 CPU pipeline load from a diffusers-format directory,
  2. `capture()`         — a context manager that registers forward hooks on named
                           submodules and returns a flat `{name: tensor}` dict,
  3. `HookSpec` + the per-block spec builders (`dit_specs`, `cond_specs`,
     `vae_specs`, `solver_specs`) that name every hook point the bringup blocks
     need to drive their submodule in isolation,
  4. `run_reference()`   — one seeded pipeline invocation with hooks installed.

Everything is host-only. **This module must never import ttnn or open a device.**

Determinism: verified 2026-07-31 that two runs of this reference at seed 1234
produce `max|diff| = 0.0`, so "delete golden/, re-dump, compare" is a real gate
(see `dump_goldens.py --verify`).

Geometry (see ACE_STEP_1_5_BRINGUP.md §3.2):
    latent frame rate  = 48000 / 1920 = 25 Hz
    T (latent frames)  = ceil(duration * 25)
    S (DiT tokens)     = ceil(T / patch_size) = T // 2 for even T
    audio samples out  = T * 1920

Reference durations (§5b): 2.56 s -> S=32, 10.24 s -> S=128, 20.48 s -> S=256,
61.44 s -> S=768.

Usage:
    from models.experimental.ace_step_v15.reference import ace_step_ref as ref
    pipe = ref.load_pipeline()
    specs = ref.dit_specs(pipe)
    with ref.capture(pipe, specs) as cap:
        out = ref.run_reference(pipe, duration=2.56)
    tensors = cap.tensors      # {"transformer.layers.0.out": Tensor, ...}
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import torch

# --------------------------------------------------------------------------------------
# Constants / geometry
# --------------------------------------------------------------------------------------

# Diffusers-format ACE-Step 1.5 turbo (2 B) pipeline, converted locally. Override with
# $ACE_STEP_PIPELINE. There is no public 2 B `-diffusers` repo (see master doc §2), so
# there is deliberately no HF-hub fallback here.
DEFAULT_PIPELINE_PATH = os.getenv("ACE_STEP_PIPELINE", "/localdev/acicovic/ace_step_diffusers")

GOLDEN_SEED = 1234
LATENTS_PER_SECOND = 25.0  # 48000 / prod([2,4,4,6,10]) = 48000 / 1920
PATCH_SIZE = 2
SAMPLES_PER_LATENT = 1920

# The four reference durations from ACE_STEP_1_5_BRINGUP.md §5b. Each is 2.56*k seconds
# so that S = 32*k is tile-aligned.
REFERENCE_DURATIONS = (2.56, 10.24, 20.48, 61.44)

# Fixed golden prompt/lyrics. Short and ASCII-only so the tokenization is stable across
# tokenizer revisions; keep these frozen or every golden changes.
GOLDEN_PROMPT = "melodic techno, driving analog bassline, warm synth pads, instrumental"
GOLDEN_LYRICS = "[verse]\nneon lines across the floor\n[chorus]\nhold the night a little more\n"
GOLDEN_CALL_KWARGS = dict(
    vocal_language="en",
    num_inference_steps=8,  # turbo: 8 steps
    guidance_scale=1.0,  # turbo: guidance distilled into the weights, no CFG
    shift=3.0,  # diffusers / generate_audio default (upstream CLI default is 1.0!)
    task_type="text2music",
    bpm=124,
    keyscale="A minor",
    timesignature="4",
    max_text_length=256,
    max_lyric_length=2048,
)


def latent_frames_for_duration(duration: float) -> int:
    """T — number of 25 Hz VAE latent frames, matching `pipeline.prepare_latents`."""
    return math.ceil(duration * LATENTS_PER_SECOND)


def dit_tokens_for_duration(duration: float) -> int:
    """S — DiT sequence length after the patch-2 collapse."""
    return math.ceil(latent_frames_for_duration(duration) / PATCH_SIZE)


def duration_for_dit_tokens(s: int) -> float:
    """Inverse of `dit_tokens_for_duration` on the tile-aligned reference grid."""
    return (s * PATCH_SIZE) / LATENTS_PER_SECOND


# --------------------------------------------------------------------------------------
# Pipeline loading
# --------------------------------------------------------------------------------------


def load_pipeline(path: Optional[str] = None, dtype: torch.dtype = torch.float32):
    """Load the ACE-Step 1.5 pipeline on CPU in `dtype` (fp32 for goldens).

    Kept deliberately dumb: no offloading, no tiling, no attention-backend override, so
    the reference is the plain eager/SDPA torch path. In particular VAE tiling stays
    **off** — the pipeline's step-11 comment claims tiling is "enabled on pipeline init"
    but `__init__` never calls `enable_tiling()`, so the golden waveform is an unsplit
    single-shot decode (see FINDING-4 in ACE_STEP_1_5_GOLDENS.md).
    """
    from diffusers import AceStepPipeline

    path = path or DEFAULT_PIPELINE_PATH
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"ACE-Step diffusers pipeline not found at {path!r}. "
            f"Set $ACE_STEP_PIPELINE to a diffusers-format ACE-Step 1.5 directory."
        )
    pipe = AceStepPipeline.from_pretrained(path, dtype=dtype)
    pipe.to("cpu")
    for name in ("transformer", "condition_encoder", "vae", "text_encoder"):
        mod = getattr(pipe, name, None)
        if mod is not None:
            mod.eval()
    assert pipe.transformer.dtype == dtype, f"transformer dtype {pipe.transformer.dtype} != {dtype}"
    assert not getattr(pipe.vae, "use_slicing", False), "VAE slicing must be off for the golden"
    assert not getattr(pipe.vae, "use_tiling", False), "VAE tiling must be off for the golden"
    return pipe


# --------------------------------------------------------------------------------------
# Hook plumbing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HookSpec:
    """One hook point.

    module      dotted module path relative to the pipeline, e.g.
                "transformer.layers.0.self_attn.to_q". `.N` indexes into
                nn.ModuleList / nn.Sequential exactly like `named_modules()` does.
    keep_calls  which invocations to record, by 0-based per-module call index.
                `None` records every call. The DiT and everything below it runs
                once per denoising step (8x for turbo), so the default (0,) keeps
                only the first step and keeps the golden dir a manageable size.
    inputs      record the forward's tensor arguments (positional -> `.in{i}`,
                keyword -> `.kw_{name}`).
    output      record the forward's tensor output(s) (`.out`, or `.out{i}` for a
                tuple/list return).
    alias       golden-name prefix; defaults to `module`.
    keys        if set, keep only the suffixes that start with one of these, e.g.
                `("kw_hidden_states", "out")`. Used to avoid dumping 24 identical
                copies of `encoder_hidden_states` / the `[1,1,S,S]` mask, one per
                DiT layer.
    """

    module: str
    keep_calls: Optional[tuple] = (0,)
    inputs: bool = True
    output: bool = True
    alias: Optional[str] = None
    keys: Optional[tuple] = None

    @property
    def prefix(self) -> str:
        return self.alias or self.module

    def wants(self, suffix: str) -> bool:
        return self.keys is None or any(suffix.startswith(k) for k in self.keys)


def _resolve(pipe, dotted: str) -> torch.nn.Module:
    obj = pipe
    for part in dotted.split("."):
        if part.isdigit() and hasattr(obj, "__getitem__"):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _flatten_tensors(prefix: str, value, out: dict) -> None:
    """Record `value` under `prefix`, descending into tuples/lists/mappings.

    Floating tensors are normalised to fp32 (the golden dtype); integer and bool tensors
    keep their dtype — casting `input_ids` or an attention mask to fp32 would silently
    corrupt it.
    """
    if isinstance(value, torch.Tensor):
        t = value.detach()
        if t.is_floating_point():
            t = t.to(torch.float32)
        out[prefix] = t.clone().contiguous()
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            _flatten_tensors(f"{prefix}{i}", v, out)
    elif isinstance(value, dict):  # covers transformers/diffusers ModelOutput (an OrderedDict)
        for k, v in value.items():
            _flatten_tensors(f"{prefix}.{k}", v, out)
    # anything else (None, scalars, config objects) is intentionally dropped


@dataclass
class Capture:
    """Result container for `capture()`."""

    tensors: dict = field(default_factory=dict)
    call_counts: dict = field(default_factory=dict)

    def subset(self, prefix: str) -> dict:
        return {k: v for k, v in self.tensors.items() if k.startswith(prefix)}

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors.values())


@contextlib.contextmanager
def capture(pipe, specs: Sequence[HookSpec]):
    """Register forward hooks for `specs`; yields a `Capture` filled in as the model runs.

    Names are `<prefix>.in{i}` / `<prefix>.kw_{name}` / `<prefix>.out` for a single kept
    call, and gain a `.call{k}` infix when more than one call is kept.
    """
    cap = Capture()
    handles = []

    def make_hook(spec: HookSpec):
        counter = {"n": 0}

        def hook(module, args, kwargs, output):
            idx = counter["n"]
            counter["n"] += 1
            cap.call_counts[spec.prefix] = counter["n"]
            if spec.keep_calls is not None and idx not in spec.keep_calls:
                return
            multi = spec.keep_calls is None or len(spec.keep_calls) > 1
            base = f"{spec.prefix}.call{idx}" if multi else spec.prefix
            local: dict = {}
            if spec.inputs:
                for i, a in enumerate(args):
                    _flatten_tensors(f"in{i}", a, local)
                for k, v in kwargs.items():
                    _flatten_tensors(f"kw_{k}", v, local)
            if spec.output:
                _flatten_tensors("out", output, local)
            for suffix, tensor in local.items():
                if spec.wants(suffix):
                    cap.tensors[f"{base}.{suffix}"] = tensor

        return hook

    try:
        for spec in specs:
            mod = _resolve(pipe, spec.module)
            handles.append(mod.register_forward_hook(make_hook(spec), with_kwargs=True))
        yield cap
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------------------
# Per-block hook specs
# --------------------------------------------------------------------------------------

# The per-layer sub-blocks of AceStepTransformerBlock.
#
# NOTE `self_attn.to_out` is an `nn.ModuleList([Linear, Dropout])` and nn.ModuleList has
# no `forward`, so a hook on `...to_out` would NEVER fire. The processor calls
# `attn.to_out[0](...)` directly, so the hookable module is `to_out.0`. We alias it back
# to `...to_out` in the golden names. (Same trap applies to `vae.decoder.block`, which is
# also a bare ModuleList — hook `block.0`, not `block`.)
_ATTN_SUBMODULES = ("to_q", "to_k", "to_v", "norm_q", "norm_k")

# Which DiT layers get full intra-layer detail. Layer 0 is `sliding_attention`
# (|i-j| <= 128 band), layer 1 is `full_attention` (mask=None) — one of each is enough to
# PCC-gate the two code paths, and dumping all 24 in detail would 12x the golden size.
DETAIL_LAYERS = (0, 1)


# `temb`, `position_embeddings`, `encoder_hidden_states` and the `[1,1,S,S]` sliding mask
# are byte-identical for every one of the 24 layers, so only the detail layers dump them.
# Non-detail layers keep just the residual-stream in/out.
_LAYER_BOUNDARY_KEYS = ("kw_hidden_states", "out")


def _layer_specs(layer_idx: int, detail: bool) -> list:
    p = f"transformer.layers.{layer_idx}"
    if not detail:
        return [HookSpec(module=p, keys=_LAYER_BOUNDARY_KEYS)]
    specs = [HookSpec(module=p)]  # full in/out: temb, RoPE cos/sin, mask, enc_hs
    specs.append(HookSpec(module=f"{p}.self_attn_norm"))
    specs.append(HookSpec(module=f"{p}.self_attn"))
    for sub in _ATTN_SUBMODULES:
        specs.append(HookSpec(module=f"{p}.self_attn.{sub}"))
    specs.append(HookSpec(module=f"{p}.self_attn.to_out.0", alias=f"{p}.self_attn.to_out"))
    specs.append(HookSpec(module=f"{p}.cross_attn_norm"))
    specs.append(HookSpec(module=f"{p}.cross_attn"))
    for sub in _ATTN_SUBMODULES:
        specs.append(HookSpec(module=f"{p}.cross_attn.{sub}"))
    specs.append(HookSpec(module=f"{p}.cross_attn.to_out.0", alias=f"{p}.cross_attn.to_out"))
    specs.append(HookSpec(module=f"{p}.mlp_norm"))
    specs.append(HookSpec(module=f"{p}.mlp"))
    for sub in ("gate_proj", "up_proj", "down_proj"):
        specs.append(HookSpec(module=f"{p}.mlp.{sub}"))
    return specs


def dit_specs(pipe, detail_layers: Optional[Iterable[int]] = None, duration: Optional[float] = None) -> list:
    """Hook points for Block 1 (the DiT).

    `detail_layers` defaults to `DETAIL_LAYERS` (= (0, 1), one sliding + one full layer)
    for durations up to 10.24 s and to `(0,)` above that, where the intra-layer detail
    tensors are 6-19 MB apiece.
    """
    if detail_layers is None:
        detail_layers = DETAIL_LAYERS if (duration is None or duration <= 10.24) else (0,)
    n_layers = pipe.transformer.config.num_hidden_layers
    detail = set(detail_layers)
    specs = [
        # Whole-model in/out. `hidden_states`/`timestep`/`timestep_r`/
        # `encoder_hidden_states`/`context_latents` all arrive as kwargs from the
        # pipeline, so they land under `transformer.kw_*`.
        HookSpec(module="transformer"),
        HookSpec(module="transformer.proj_in_conv"),  # NCL [B,192,T] -> [B,2048,S]
        HookSpec(module="transformer.condition_embedder"),
        HookSpec(module="transformer.time_embed"),
        HookSpec(module="transformer.time_embed_r"),
        HookSpec(module="transformer.norm_out"),
        HookSpec(module="transformer.proj_out_conv"),  # NCL [B,2048,S] -> [B,64,2S]
    ]
    for i in range(n_layers):
        specs += _layer_specs(i, detail=i in detail)
    return specs


def cond_specs(pipe) -> list:
    """Hook points for Block 2 (the condition encoder). Runs once per generation."""
    return [
        HookSpec(module="condition_encoder"),
        HookSpec(module="condition_encoder.text_projector"),
        HookSpec(module="condition_encoder.lyric_encoder"),
        HookSpec(module="condition_encoder.timbre_encoder"),
        HookSpec(module="text_encoder", alias="qwen3_text_encoder"),
    ]


_VAE_BLOCK_SUBMODULES = ("snake1", "conv_t1", "res_unit1", "res_unit2", "res_unit3")

# Which decoder blocks get intra-block detail. Blocks 3 and 4 are structurally identical
# to 0-2 (Snake -> ConvTranspose1d -> 3x ResidualUnit) but 8x/16x longer in T, so their
# intermediates are 250-500 MB each at T=512. Detail on 0-2 covers every code path.
VAE_DETAIL_BLOCKS = (0, 1, 2)


def vae_specs(pipe, n_blocks: int = 5, mode: str = "detail", detail_blocks=VAE_DETAIL_BLOCKS) -> list:
    """Hook points for Block 3 (the Oobleck VAE decoder).

    `vae.decoder.block` is an `nn.ModuleList` — hook `block.N`, never `block` (ModuleList
    has no `forward`, so a hook on it never fires).

    mode:
      "detail"    decoder in/out + conv1/snake1/conv2 + every block boundary + the
                  sub-modules of `detail_blocks`.
      "boundary"  decoder in/out + conv1/snake1/conv2 + every block boundary.
      "io"        decoder in/out only — a whole-decoder end-to-end golden.

    Sub-module hooks record **outputs only**: the decoder is a strict chain, so each
    sub-module's input is the previous one's output and dumping both would double the
    (already large) VAE golden for no extra information.
    """
    specs = [HookSpec(module="vae.decoder", alias="vae.decoder")]
    if mode == "io":
        return specs
    specs += [
        HookSpec(module="vae.decoder.conv1"),
        HookSpec(module="vae.decoder.snake1", inputs=False),
        HookSpec(module="vae.decoder.conv2", inputs=False),
    ]
    for i in range(n_blocks):
        p = f"vae.decoder.block.{i}"
        specs.append(HookSpec(module=p))
        if mode == "detail" and i in detail_blocks:
            for sub in _VAE_BLOCK_SUBMODULES:
                specs.append(HookSpec(module=f"{p}.{sub}", inputs=False))
    return specs


def solver_specs(pipe) -> list:
    """Hook points for Block 4 (solver + pipeline): every denoising step's DiT in/out.

    `keep_calls=None` records all 8 turbo steps, filtered down to the small tensors:
    `kw_hidden_states` ([1,T,64] x_t), `kw_timestep`/`kw_timestep_r` and `out` (the
    [1,T,64] velocity). `encoder_hidden_states` / `context_latents` are step-invariant
    and already dumped once by `dit_specs`, so they are filtered out here.
    """
    return [
        HookSpec(
            module="transformer",
            keep_calls=None,
            alias="solver.transformer",
            keys=("kw_hidden_states", "kw_timestep", "out"),
        )
    ]


# --------------------------------------------------------------------------------------
# Reference invocation
# --------------------------------------------------------------------------------------


def make_generator(seed: int = GOLDEN_SEED) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


@torch.no_grad()
def run_reference(
    pipe,
    duration: float,
    seed: int = GOLDEN_SEED,
    output_type: str = "pt",
    collect_step_latents: bool = True,
    **overrides,
) -> dict:
    """One seeded fp32 CPU generation. Returns the pipeline output plus loop bookkeeping.

    Seeding is belt-and-braces: an explicit `torch.Generator` for `prepare_latents` (the
    only sampling site on the text2music path) *and* a global `torch.manual_seed`, so any
    stray unseeded op is still reproducible.
    """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # CPU fp32 eager is already deterministic

    kwargs = dict(GOLDEN_CALL_KWARGS)
    kwargs.update(overrides)

    step_latents: list = []
    timesteps: list = []

    def on_step_end(_pipe, step_idx, timestep, callback_kwargs):
        timesteps.append(float(timestep))
        if collect_step_latents:
            step_latents.append(callback_kwargs["latents"].detach().clone())
        return {}

    out = pipe(
        prompt=GOLDEN_PROMPT,
        lyrics=GOLDEN_LYRICS,
        audio_duration=duration,
        generator=make_generator(seed),
        output_type=output_type,
        return_dict=True,
        callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=("latents",),
        **kwargs,
    )

    T = latent_frames_for_duration(duration)
    return {
        "audio": out.audios,
        "duration": duration,
        "latent_frames": T,
        "dit_tokens": dit_tokens_for_duration(duration),
        "timesteps": torch.tensor(timesteps, dtype=torch.float32),
        "step_latents": step_latents,
        "call_kwargs": kwargs,
        "seed": seed,
    }


@torch.no_grad()
def run_with_capture(pipe, duration: float, specs: Sequence[HookSpec], seed: int = GOLDEN_SEED, **overrides):
    """`run_reference` with `specs` hooked. Returns `(result, Capture)`."""
    with capture(pipe, specs) as cap:
        result = run_reference(pipe, duration, seed=seed, **overrides)
    return result, cap


def describe(tensors: dict) -> str:
    """One line per golden: name, shape, dtype. Used by dump_goldens' inventory."""
    lines = []
    for name in sorted(tensors):
        t = tensors[name]
        lines.append(f"{name:78s} {str(tuple(t.shape)):26s} {str(t.dtype).replace('torch.', '')}")
    return "\n".join(lines)


if __name__ == "__main__":  # tiny smoke test: shapes only, shortest duration
    _pipe = load_pipeline()
    _res, _cap = run_with_capture(_pipe, 2.56, dit_specs(_pipe)[:8] + cond_specs(_pipe))
    print(describe(_cap.tensors))
    print(f"audio {tuple(_res['audio'].shape)}  timesteps {_res['timesteps'].tolist()}")

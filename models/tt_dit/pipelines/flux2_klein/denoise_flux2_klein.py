# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein Euler sampler on device: the denoising loop of the diffusers ``Flux2KleinPipeline.__call__``
(``pipeline_flux2_klein.py:826-880``, cited as ``P``; the scheduler ``scheduling_flow_match_euler_discrete.py``
``S``; BFL ``sampling.py`` ``SA`` as the second citation) on one :class:`Flux2KleinTransformer`, with the step
captured as ONE ttnn trace. The trace contract and the elementwise chains are those of the FLUX 3 Klein sampler
(``pipelines/flux3_klein/denoise_flux3_klein.py``), kept verbatim where the model does not differ.

Reference math, per step ``i``, and how each piece is reproduced here (bf16 operands, fp32 accumulation; every
operation outside the DiT is bit-equal to torch):

* ``timestep = t.expand(B).to(bf16)``, the DiT gets ``timestep / 1000`` and rescales by ``1000`` in bf16
  (``P:841``, ``P:853``, ``transformer_flux2.py:1284``): the host computes that rounding chain once per step
  (``scheduler.Flux2KleinStep.embedding_t``) and writes it into the fp32 ``[1, 1]`` ``t_embed`` slot.
* ``latents`` enter the DiT in bf16 (``P:851``): ``ttnn.typecast(x, bf16)`` of the fp32 state slot, which holds
  bf16-valued numbers between steps (below), so the cast is exact.
* CFG of the base model ``neg + g * (pos - neg)`` on bf16 tensors with a Python float ``g`` (``P:877``): torch rounds
  each op to bf16. Device: ``c`` / ``u`` widened to fp32, ``subtract`` -> typecast bf16 -> fp32, ``multiply`` by the
  fp32 ``[1, 1]`` guidance SLOT -> bf16 -> fp32, ``add`` -> bf16 (:func:`combine_cfg`; measured bit-equal on the
  FLUX 3 Klein port for ``g`` in 0.1 .. 7.7).
* ``scheduler.step`` (``S:493-526``): ``sample.float()``; ``dt = sigmas[i + 1] - sigmas[i]`` (a 0-dim fp32 tensor,
  negative); ``prev = sample + dt * model_output`` where ``dt * v`` stays bf16 under torch promotion AND the 0-dim
  ``dt`` is cast to bf16 first (``Flux2KleinStep.dt_rounded``; unlike the Python-float ``dt`` of BFL / FLUX 3);
  ``prev.to(bf16)``: THE STATE IS bf16 BETWEEN STEPS. Device: ``typecast(v, fp32)`` x the fp32 ``[1, 1]`` dt SLOT
  (holding ``bf16(dt)``) -> typecast bf16 -> fp32,
  times the fp32 ``{1, 0}`` row mask, fp32 ``add`` (:func:`euler_update`, the FLUX 3 chain), then
  ``typecast(bf16) -> typecast(fp32)`` of the new state (the ``prev.to(bf16)`` the FLUX 3 model did not have).
* ``noise_pred = noise_pred[:, : latents.size(1)]`` (``P:862``): the reference rows and the alignment rows of the
  state slot are constant. The slot holds ``[image | reference tokens | zero pad]`` (``P:846-848``
  ``cat([latents, image_latents], dim=1)``) and the row mask is ``1.0`` on the image rows only, so the Euler update
  leaves the reference rows (bf16-valued, exact through the bf16 round trip) and the pad rows (``+0``) untouched.

Forms that were measured WRONG on the FLUX 3 port and must never return: ``ttnn.multiply(v_bf16, dt_python_float)``
(the scalar is rounded to bf16); a mixed-dtype ``multiply`` / ``add`` (the fp32 side is downcast); bf16-in /
bf16-out ``subtract`` / ``add`` with the default ``fast_and_approximate_mode``; any Python scalar for ``dt`` or the
guidance inside the traced step (also rejected by the Tracer across replays). No binary op in the chain mixes dtypes.

Trace contract (``utils/tracing.py``):

* :meth:`Flux2KleinDenoiser._step` is PURE: it reads the fp32 state slot ``x`` and returns ``(x_new, velocity)``;
  the host ``ttnn.copy``'s ``x_new`` into the slot after the call (same command queue, so it is ordered after the
  replay) and deallocates ``x_new`` only in eager mode (in traced mode it is the trace's own output buffer).
* Slots, one set per layout key ``(sp, S_img, S_cond, L_pad, cfg)``, live at fixed addresses for the life of the
  denoiser: ``x`` fp32 ``[1, L_pad, 128]`` (``mesh_axes=[None, sp, None]``), ``t_embed`` fp32 ``[1, 1]``, ``dt`` fp32
  ``[1, 1]``, ``guidance`` fp32 ``[1, 1]`` (CFG), the step's 17 modulation chunks + ``silu(temb)`` as bf16
  ``[1, 1, D]`` slots (``mod``, computed on the host by ``Flux2KleinHostTime`` bit-equal to the reference; the device
  time module is the fallback when the transformer has no host path), the constant ``mask``, and per prompt the
  hoisted ``prompt_embedded`` (``context_embedder`` once per request, ``transformer_flux2.py:1319``). Every refresh
  is a ``ttnn.copy`` into the slot (``StateTensor.update(..., traced=True)`` once bound; :func:`tree_copy_into` for
  the prompt): no per-step value is ever baked into the trace, and a second request with the same layout reuses the
  captured trace.
* The plan (:class:`Flux2KleinPlan`: the RoPE tables of the layout and the ids) is NOT a slot: ``prepare`` takes it
  from the transformer's plan cache (``Flux2KleinTransformer.plan_for``: the SAME object for the same canvas,
  model-owned, never copied into or freed here), so a request of a seen canvas rebuilds and uploads no table. A
  cache entry is a long-lived buffer: a miss releases every live trace before it allocates (and before the eviction
  that makes room), and a refresh that hands a key's slot a DIFFERENT plan object releases that key's trace (it read
  the old plan's addresses; the next traced step recaptures). :meth:`_step_inputs` refuses a plan the cache evicted
  meanwhile (more than its ``maxsize`` distinct canvases prepared since the request's ``prepare``).
* EVERY step constant travels ONCE per distinct value, not per step (FLUX 3's rule that the step constants are
  uploaded in ``prepare``, taken one step further: FLUX 3 still uploads its two scalars per step). The denoiser keeps
  two caches shared by all layout keys: ``_mod_cache`` (``embedding_t`` -> the 17 + 1 bf16 ``[1, 1, D / TP]`` ``mod``
  tensors of that timestep, TP-fractured / ``time_embed`` replicated exactly as the slots) and ``_scalar_cache``
  (fp32 value -> one replicated fp32 ``[1, 1]`` tensor, used for both ``t_embed`` = ``embedding_t`` and ``dt`` =
  ``dt_rounded``). :meth:`prepare` uploads the entries of every step of the request's schedule the caches do not hold
  yet (a few hundred KB per schedule); an EAGER step then hands its timestep's cached ``mod`` tree and the cached
  scalars straight to :meth:`_step` and uploads and copies NOTHING (the values are the same tensors, so the step is
  bit-identical to the slot-refresh form; the schedule is fixed per ``(num_steps, image_seq_len)``, so a repeated size
  hits). A TRACED step keeps refreshing the ``t_embed`` / ``dt`` / ``mod`` SLOTS by ``ttnn.copy`` (the trace reads
  fixed addresses). Cache entries are long-lived buffers, so they are only allocated while no trace is live
  (``prepare`` releases the captured traces first when it has entries to add; a new key has already released them);
  they are freed by :meth:`clear_step_constants` (``deallocate()`` of every key does it).
* A replay writes the trace's intermediate buffers at the addresses they had during the capture; those addresses are
  FREE after the capture, so any buffer allocated later may land on them and be overwritten by the replay (Tracer
  caveat 1). The sampler therefore keeps this invariant: EVERY long-lived buffer a replay could hit exists before the
  trace is captured. :meth:`prepare` of a NEW key first releases every captured trace (recaptured on the next traced
  step), then allocates the key's slots, runs ONE eager step on them so every program of the layout is compiled (a
  program's kernel binaries live in a DRAM buffer allocated at compile time) and :func:`warm_copy`'s the copy program
  of every slot tensor (a ``CopyDeviceOperation`` compiled after a capture was measured to leave such a buffer where
  the replay overwrites it; ``TT_METAL_TRACE_ALLOC_TRACKING=1`` fails).
* Several keys may hold a captured trace at once (at most ``max_live_traces``, least recently replayed released
  first). A trace's two output buffers are allocated during its capture, i.e. possibly inside an OLDER live trace's
  intermediate region: that older trace's replay may overwrite them, which is harmless because a step's outputs are
  consumed before any other replay (``x_new`` is copied into the state slot on the same command queue right after the
  replay; the velocity is read before the next step). The outputs are therefore acknowledged to the trace allocation
  tracker as corruptible; nothing else is.
* Replay is non-blocking (``tracer_blocking_execution=False``); the host reads only through :meth:`read_state` /
  :meth:`read_velocity`, which synchronize the device first. Nothing may be allocated on device between replays
  except the slot-refresh temporaries (freed before the next replay) and the transients of an eager step (freed
  inside it).
* Release order: traces (:meth:`Flux2KleinPreparedRequest.release` / :meth:`release_traces`) before the slots are
  freed, before the model's weights are freed and before the mesh closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

import ttnn

from ...models.transformers.flux2_klein.time_flux2_klein import Flux2KleinModulation
from ...models.transformers.flux2_klein.transformer_flux2_klein import (
    Flux2KleinPlan,
    Flux2KleinSequenceLayout,
    Flux2KleinTransformer,
)
from ...utils import tensor
from ...utils.tracing import StateTensor, traced_function
from .packing import FLUX2_KLEIN_IN_CHANNELS, FLUX2_KLEIN_TEXT_LENGTH, pad_rows
from .scheduler import Flux2KleinScheduler, Flux2KleinStep

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

LayoutKey = tuple
"""``(sp_factor, image_rows, cond_rows, padded_rows, cfg)``."""

_F32 = torch.float32
_BF16 = torch.bfloat16


# --- host helpers -----------------------------------------------------------------------------------------------------


def layout_key(layout: Flux2KleinSequenceLayout, cfg: bool) -> LayoutKey:
    """The hashable identity of a slot set and its trace: the row layout of the spatial stream and whether the step
    runs the CFG pair (the prompt stream is always ``FLUX2_KLEIN_TEXT_LENGTH`` rows, so it does not key)."""
    return (int(layout.sp_factor), int(layout.image), int(layout.cond), int(layout.padded), bool(cfg))


def image_row_mask(num_rows: int, num_rows_padded: int) -> torch.Tensor:
    """fp32 ``[1, num_rows_padded, 128]``: ``1.0`` on the ``num_rows`` image rows, ``0.0`` on the reference and
    alignment rows (``P:862``: only the latents are updated)."""
    if not 0 < num_rows <= num_rows_padded:
        raise ValueError(f"need 0 < num_rows <= num_rows_padded, got {num_rows}, {num_rows_padded}")
    mask = torch.zeros(1, num_rows_padded, FLUX2_KLEIN_IN_CHANNELS, dtype=_F32)
    mask[:, :num_rows] = 1.0
    return mask


def tree_leaves(tree) -> Iterator:
    """Depth-first leaves of nested dicts / tuples / lists, skipping ``None``."""
    if tree is None:
        return
    if isinstance(tree, dict):
        for value in tree.values():
            yield from tree_leaves(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from tree_leaves(value)
    else:
        yield tree


def tree_copy_into(new, old) -> None:
    """``ttnn.copy`` every leaf of ``new`` into the matching leaf of ``old`` (same structure, shape, dtype, layout)."""
    for source, target in zip(tree_leaves(new), tree_leaves(old), strict=True):
        if source.shape != target.shape or source.dtype != target.dtype or source.layout != target.layout:
            raise ValueError(f"slot refresh mismatch: {source.shape} {source.dtype} vs {target.shape} {target.dtype}")
        ttnn.copy(source, target)


def warm_copy(t: ttnn.Tensor) -> None:
    """Compile and cache the copy program of ``t``'s spec before any trace is captured (see the module docstring)."""
    tmp = ttnn.clone(t)
    ttnn.copy(tmp, t)
    ttnn.deallocate(tmp)


# --- the exact elementwise chain --------------------------------------------------------------------------------------


def combine_cfg(conditional: ttnn.Tensor, unconditional: ttnn.Tensor, guidance: ttnn.Tensor) -> ttnn.Tensor:
    """``u + g * (c - u)`` of two bf16 velocities with ``g`` an fp32 ``[1, 1]`` slot, each op rounded to bf16 exactly
    as torch rounds it (``P:877``, ``Flux2KleinScheduler.cfg_velocity``). Returns bf16; the inputs are left to the
    caller."""
    c32 = ttnn.typecast(conditional, ttnn.float32)
    u32 = ttnn.typecast(unconditional, ttnn.float32)
    d = ttnn.typecast(ttnn.typecast(ttnn.subtract(c32, u32), ttnn.bfloat16), ttnn.float32)
    ttnn.deallocate(c32)
    m = ttnn.typecast(ttnn.typecast(ttnn.multiply(d, guidance), ttnn.bfloat16), ttnn.float32)
    ttnn.deallocate(d)
    v = ttnn.typecast(ttnn.add(u32, m), ttnn.bfloat16)
    ttnn.deallocate(u32)
    ttnn.deallocate(m)
    return v


def euler_update(x: ttnn.Tensor, velocity: ttnn.Tensor, dt: ttnn.Tensor, mask: ttnn.Tensor) -> ttnn.Tensor:
    """``x + bf16(dt * v)`` with ``x`` fp32, ``v`` bf16, ``dt`` an fp32 ``[1, 1]`` slot and ``mask`` the fp32 ``{1, 0}``
    row mask (``S:507-520``). Returns a new fp32 tensor; the masked rows of the result equal those of ``x``
    (``+0 + (-0) = +0`` keeps zero pad rows exactly zero)."""
    v32 = ttnn.typecast(velocity, ttnn.float32)
    p = ttnn.typecast(ttnn.typecast(ttnn.multiply(v32, dt), ttnn.bfloat16), ttnn.float32)
    ttnn.deallocate(v32)
    delta = ttnn.multiply(p, mask)
    ttnn.deallocate(p)
    x_new = ttnn.add(x, delta)
    ttnn.deallocate(delta)
    return x_new


def round_state_to_bf16(x: ttnn.Tensor) -> ttnn.Tensor:
    """``prev_sample.to(bf16)`` (``S:526``) on the fp32 state: ``typecast(bf16) -> typecast(fp32)``. Consumes ``x``;
    returns a new fp32 tensor holding bf16-valued numbers."""
    x_bf16 = ttnn.typecast(x, ttnn.bfloat16)
    ttnn.deallocate(x)
    x_rounded = ttnn.typecast(x_bf16, ttnn.float32)
    ttnn.deallocate(x_bf16)
    return x_rounded


# --- requests ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Flux2KleinDenoiseRequest:
    """One image to sample. Host tensors in the pipeline's dtypes (``packing.py``): ``noise`` ``[S_img, 128]`` fp32
    (bf16-valued, the state entering step 0), ``image_ids`` ``[S_img, 4]`` int64, ``text`` ``([512, joint] bf16,
    [512, 4] int64)``; ``guidance`` is the CFG scale of the base model (two forwards per step, ``text_null`` = the
    empty prompt's encoding is then required) and ``None`` for the distilled model; ``image_cond`` ``([N, 128] bf16,
    [N, 4] int64)`` are the reference tokens shared by both CFG samples.

    The text rows may instead arrive ON THE DEVICE (``prompt_encoder.Flux2KleinDeviceText.pair``): a ``ttnn.Tensor``
    ``[1, 512, joint]`` bf16 replicated, or SP-partitioned on dim 1 to ``[1, 512 / SP, joint]`` per device, in the
    placement ``Flux2KleinTransformer.embed_prompt`` takes (sharded under ``shard_prompt``, replicated otherwise;
    ``prepare`` checks the per-device rows). ``prepare`` consumes it: the tensor is deallocated once the prompt is
    embedded (and on a failed ``prepare``), so a device encoding serves ONE ``prepare``."""

    noise: torch.Tensor
    image_ids: torch.Tensor
    text: tuple[torch.Tensor | ttnn.Tensor, torch.Tensor]
    num_steps: int
    guidance: float | None = None
    text_null: tuple[torch.Tensor | ttnn.Tensor, torch.Tensor] | None = None
    image_cond: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def cfg(self) -> bool:
        return self.guidance is not None

    @property
    def image_seq_len(self) -> int:
        """The latent token count (``P:817``: the schedule's ``image_seq_len``, references excluded)."""
        return int(self.noise.shape[0])

    @property
    def cond_rows(self) -> int:
        return 0 if self.image_cond is None else int(self.image_cond[0].shape[0])

    def text_of(self, unconditional: bool = False) -> tuple[torch.Tensor | ttnn.Tensor, torch.Tensor]:
        if unconditional:
            if self.text_null is None:
                raise ValueError("the unconditional sample needs text_null")
            return self.text_null
        return self.text

    def device_text(self) -> list[ttnn.Tensor]:
        """The text tensors handed over on the device (``text`` / ``text_null`` whose rows are a ``ttnn.Tensor``)."""
        return [
            pair[0] for pair in (self.text, self.text_null) if pair is not None and isinstance(pair[0], ttnn.Tensor)
        ]

    def spatial_ids(self) -> torch.Tensor:
        """int64 ``[S_img + S_cond, 4]``: the latent ids followed by the reference ids (``P:846-848``)."""
        if self.image_cond is None:
            return self.image_ids
        return torch.cat([self.image_ids, self.image_cond[1]])

    def spatial_rows(self) -> torch.Tensor:
        """fp32 ``[S_img + S_cond, 128]``: the noise followed by the reference tokens (bf16-valued)."""
        if self.image_cond is None:
            return self.noise.to(_F32)
        return torch.cat([self.noise.to(_F32), self.image_cond[0].to(_F32)])


# --- slots ------------------------------------------------------------------------------------------------------------


@dataclass
class _SampleSlots:
    """One prompt's step-invariant device tensors: the hoisted prompt embedding (a fixed-address slot, refreshed by
    ``ttnn.copy``) and the plan the transformer's cache handed out (model-owned references, module docstring)."""

    plan: Flux2KleinPlan
    prompt: ttnn.Tensor

    def tree(self) -> dict:
        return {"plan": self.plan.tree(), "prompt": self.prompt}

    def plan_leaves(self) -> list[ttnn.Tensor]:
        return list(tree_leaves(self.plan.tree()))


@dataclass
class _LayoutSlots:
    """Everything a layout key's trace reads, owned by the denoiser."""

    key: LayoutKey
    layout: Flux2KleinSequenceLayout
    cond: _SampleSlots
    uncond: _SampleSlots | None
    x: StateTensor
    t_embed: StateTensor
    dt: StateTensor
    guidance: StateTensor | None
    mask: ttnn.Tensor
    num_image_rows: int
    num_rows_padded: int
    mod: dict[str, StateTensor] | None = None

    def mod_tree(self) -> dict[str, ttnn.Tensor] | None:
        return None if self.mod is None else {name: state.value for name, state in self.mod.items()}

    def tensors(self) -> list[ttnn.Tensor]:
        """Every slot tensor that is refreshed by ``ttnn.copy`` (the mask is constant; the plans belong to the
        transformer's cache)."""
        out = [self.cond.prompt]
        if self.uncond is not None:
            out.append(self.uncond.prompt)
        states = [self.x, self.t_embed, self.dt, self.guidance]
        if self.mod is not None:
            states.extend(self.mod.values())
        for state in states:
            if state is not None and state.value is not None:
                out.append(state.value)
        return out


class Flux2KleinPreparedRequest:
    """A request bound to a denoiser's slot set; ``release()`` frees its trace and slots (in that order)."""

    def __init__(
        self,
        denoiser: Flux2KleinDenoiser,
        request: Flux2KleinDenoiseRequest,
        key: LayoutKey,
        scheduler: Flux2KleinScheduler,
        slots: _LayoutSlots,
    ) -> None:
        self._denoiser = denoiser
        self.request = request
        self.key = key
        self.scheduler = scheduler
        self.slots = slots

    @property
    def num_steps(self) -> int:
        return self.scheduler.num_steps

    def release(self) -> None:
        self._denoiser.release_trace(self.key)
        self._denoiser.deallocate(self.key)


# --- the sampler ------------------------------------------------------------------------------------------------------


class Flux2KleinDenoiser:
    """Euler sampling on ``transformer`` (see the module docstring). ``sp_axis`` is the mesh axis the spatial rows are
    sharded over (``parallel_config.sequence_parallel.mesh_axis``); ``max_live_traces`` bounds the captured traces
    held at once (the least recently replayed is released first)."""

    def __init__(
        self, transformer: Flux2KleinTransformer, *, device: ttnn.MeshDevice, sp_axis: int, max_live_traces: int = 1
    ) -> None:
        if max_live_traces < 1:
            raise ValueError(f"max_live_traces must be at least 1, got {max_live_traces}")
        self._transformer = transformer
        self._device = device
        self._sp_axis = sp_axis
        self._max_live_traces = max_live_traces
        self._slots: dict[LayoutKey, _LayoutSlots] = {}
        self._captured: list[LayoutKey] = []
        """Keys holding a captured trace, least recently replayed first."""
        self._mod_cache: dict[float, dict[str, ttnn.Tensor]] = {}
        """``embedding_t`` -> the step's ``mod`` tree uploaded once; shared by every layout key (module docstring)."""
        self._scalar_cache: dict[float, ttnn.Tensor] = {}
        """fp32 value -> replicated fp32 ``[1, 1]`` tensor (the eager steps' ``t_embed`` / ``dt``), uploaded once."""

    @property
    def transformer(self) -> Flux2KleinTransformer:
        return self._transformer

    @property
    def captured_keys(self) -> tuple[LayoutKey, ...]:
        """The layout keys holding a captured trace, least recently replayed first."""
        return tuple(self._captured)

    def _tracer(self, key: LayoutKey):
        tracers = type(self)._step._tracers_keyed.get(self)
        return tracers.get(key) if tracers else None

    # --- preparation ------------------------------------------------------------------------------------------------

    def _validate(self, request: Flux2KleinDenoiseRequest) -> None:
        if (request.text_null is not None) != request.cfg:
            raise ValueError("text_null is required by, and only by, a CFG request")
        if request.noise.ndim != 2 or request.noise.shape[0] == 0 or request.noise.shape[1] != FLUX2_KLEIN_IN_CHANNELS:
            raise ValueError(f"noise must be [S_img, 128] with S_img > 0, got {tuple(request.noise.shape)}")
        if tuple(request.image_ids.shape) != (request.noise.shape[0], 4):
            raise ValueError(f"image_ids must be [{request.noise.shape[0]}, 4], got {tuple(request.image_ids.shape)}")
        for name, pair in (("text", request.text), ("text_null", request.text_null)):
            if pair is None:
                continue
            tokens, ids = pair
            if isinstance(tokens, ttnn.Tensor):
                # The encoder's handoff: per-device [1, rows, joint] bf16 in embed_prompt's placement.
                rows = self._prompt_rows_per_device()
                shape = tuple(tokens.shape)
                if len(shape) != 3 or shape[0] != 1 or shape[1] != rows or tokens.dtype != ttnn.bfloat16:
                    raise ValueError(
                        f"{name} on the device must be bf16 [1, {rows}, joint] per device "
                        f"({'SP-partitioned' if self._transformer.shard_prompt else 'replicated'} "
                        f"{FLUX2_KLEIN_TEXT_LENGTH} rows), got {tokens.dtype} {shape}"
                    )
                if tuple(ids.shape) != (FLUX2_KLEIN_TEXT_LENGTH, 4):
                    raise ValueError(f"{name} ids must be [{FLUX2_KLEIN_TEXT_LENGTH}, 4], got {tuple(ids.shape)}")
                continue
            if (
                tokens.ndim != 2
                or tokens.shape[0] != FLUX2_KLEIN_TEXT_LENGTH
                or tuple(ids.shape) != (tokens.shape[0], 4)
            ):
                raise ValueError(
                    f"{name} must be ([{FLUX2_KLEIN_TEXT_LENGTH}, joint], [{FLUX2_KLEIN_TEXT_LENGTH}, 4]), got "
                    f"{tuple(tokens.shape)}, {tuple(ids.shape)}"
                )
        if request.image_cond is not None:
            tokens, ids = request.image_cond
            if (
                tokens.ndim != 2
                or tokens.shape[1] != FLUX2_KLEIN_IN_CHANNELS
                or tuple(ids.shape) != (tokens.shape[0], 4)
            ):
                raise ValueError(
                    f"image_cond must be ([N, 128], [N, 4]), got {tuple(tokens.shape)}, {tuple(ids.shape)}"
                )
        if request.num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {request.num_steps}")

    def _layout(self, request: Flux2KleinDenoiseRequest) -> Flux2KleinSequenceLayout:
        return self._transformer.sequence_layout(request.image_seq_len, request.cond_rows)

    def _prompt_rows_per_device(self) -> int:
        """The prompt rows each device holds: ``512 / SP`` under ``shard_prompt``, else all 512."""
        if not self._transformer.shard_prompt:
            return FLUX2_KLEIN_TEXT_LENGTH
        return FLUX2_KLEIN_TEXT_LENGTH // int(tuple(self._device.shape)[self._sp_axis])

    def _upload_prompt_rows(self, rows: torch.Tensor) -> ttnn.Tensor:
        """``[512, joint]`` -> bf16 ``[1, 512, joint]``, SP-sharded on dim 1 under ``shard_prompt`` else replicated
        (the ``embed_prompt`` input contract of ``transformer_flux2_klein.py``)."""
        sp_axis = self._sp_axis if self._transformer.shard_prompt else None
        return tensor.from_torch(
            rows.to(_BF16)[None].contiguous(), device=self._device, dtype=ttnn.bfloat16, mesh_axes=[None, sp_axis, None]
        )

    @staticmethod
    def _release_device_text(request: Flux2KleinDenoiseRequest) -> None:
        """Free the device text a failed ``prepare`` did not consume."""
        for t in request.device_text():
            if t.is_allocated():
                ttnn.deallocate(t)

    def _fresh_sample(
        self, request: Flux2KleinDenoiseRequest, layout: Flux2KleinSequenceLayout, unconditional: bool
    ) -> _SampleSlots:
        """The cached plan and a fresh hoisted prompt embedding of one prompt. The plan comes from the transformer's
        plan cache (a miss releases every live trace first, module docstring). Host rows are uploaded in
        ``embed_prompt``'s placement; a device tensor (the encoder's handoff, validated by ``_validate``) is used as
        is. Either way the ``[1, rows, joint]`` input is freed once the prompt is embedded."""
        tokens_in, text_ids = request.text_of(unconditional)
        plan = self._transformer.plan_for(layout, request.spatial_ids(), text_ids, on_miss=self.release_traces)
        tokens = tokens_in if isinstance(tokens_in, ttnn.Tensor) else self._upload_prompt_rows(tokens_in)
        prompt = self._transformer.embed_prompt(tokens)
        ttnn.deallocate(tokens)
        return _SampleSlots(plan=plan, prompt=prompt)

    def _state_host(self, request: Flux2KleinDenoiseRequest, num_rows_padded: int) -> torch.Tensor:
        """fp32 ``[1, L_pad, 128]``: ``[noise | reference tokens | zero pad]``."""
        return pad_rows(request.spatial_rows(), num_rows_padded)[0][None].contiguous()

    def _scalar_slot(self, state: StateTensor, value: torch.Tensor) -> None:
        state.update(value, True, dtype=ttnn.float32, mesh_axes=[None, None], device=self._device)

    def _mod_mesh_axes(self, name: str) -> list[int | None]:
        """``time_embed`` is replicated, every chunk is TP-fractured (``Flux2KleinModulation``)."""
        return [None, None, None if name == "time_embed" else self._transformer._tp_axis]

    def _write_step_slots(self, slots: _LayoutSlots, step: Flux2KleinStep, *, mod: bool = True) -> None:
        """Refresh the per-step SLOTS (the traced form, and the compile step of a new key): the two fp32 scalars, and
        with ``mod`` the host-exact modulation chunks by ``ttnn.copy``. Eager steps never call this: they read the
        denoiser's caches (:meth:`_step_constants`)."""
        self._scalar_slot(slots.t_embed, torch.full((1, 1), step.embedding_t, dtype=_F32))
        if mod and slots.mod is not None:
            for name, value in self._transformer.host_time(step.embedding_t).slots().items():
                slots.mod[name].update(
                    value.reshape(1, 1, -1),
                    True,
                    dtype=ttnn.bfloat16,
                    mesh_axes=self._mod_mesh_axes(name),
                    device=self._device,
                )
        # bf16(dt), not dt: torch rounds the scheduler's 0-dim fp32 dt to bf16 before dt * v (Flux2KleinStep.dt_rounded).
        self._scalar_slot(slots.dt, torch.full((1, 1), step.dt_rounded, dtype=_F32))

    def _upload_step_mod(self, embedding_t: float) -> dict[str, ttnn.Tensor]:
        """The ``mod`` tree of one timestep as fresh device tensors (same values, dtype and sharding as the slots)."""
        return {
            name: tensor.from_torch(
                value.reshape(1, 1, -1),
                device=self._device,
                dtype=ttnn.bfloat16,
                mesh_axes=self._mod_mesh_axes(name),
            )
            for name, value in self._transformer.host_time(embedding_t).slots().items()
        }

    def _upload_scalar(self, value: float) -> ttnn.Tensor:
        """One replicated fp32 ``[1, 1]`` tensor (the placement of the ``t_embed`` / ``dt`` / ``guidance`` slots)."""
        return tensor.from_torch(
            torch.full((1, 1), value, dtype=_F32), device=self._device, dtype=ttnn.float32, mesh_axes=[None, None]
        )

    def _ensure_step_constants(self, scheduler: Flux2KleinScheduler) -> None:
        """Upload every step constant of ``scheduler`` the caches do not hold yet: the ``mod`` tree per distinct
        ``embedding_t`` (host time path only) and the fp32 scalars ``embedding_t`` / ``dt_rounded`` per distinct value
        (module docstring: long-lived buffers are allocated only while no trace is live)."""
        mods: list[float] = []
        scalars: list[float] = []
        for index in range(scheduler.num_steps):
            step = scheduler.step(index)
            t = float(step.embedding_t)
            if self._transformer.host_time is not None and t not in self._mod_cache and t not in mods:
                mods.append(t)
            for value in (t, float(step.dt_rounded)):
                if value not in self._scalar_cache and value not in scalars:
                    scalars.append(value)
        if not mods and not scalars:
            return
        if self._captured:
            self.release_traces()
        for t in mods:
            self._mod_cache[t] = self._upload_step_mod(t)
        for value in scalars:
            self._scalar_cache[value] = self._upload_scalar(value)

    def _step_constants(self, step: Flux2KleinStep) -> tuple[ttnn.Tensor, ttnn.Tensor, dict[str, ttnn.Tensor] | None]:
        """``(t_embed, dt, mod)`` of ``step`` from the caches for an eager :meth:`_step` (``mod`` is ``None`` without
        a host time path)."""
        try:
            t_embed = self._scalar_cache[float(step.embedding_t)]
            dt = self._scalar_cache[float(step.dt_rounded)]
            mod = None if self._transformer.host_time is None else self._mod_cache[float(step.embedding_t)]
        except KeyError:
            raise KeyError(
                f"no step constants cached for embedding_t {step.embedding_t} / dt {step.dt_rounded}; prepare() uploads them"
            ) from None
        return t_embed, dt, mod

    def step_constants_info(self) -> dict[str, int]:
        """Cache occupancy: distinct timesteps holding a ``mod`` tree, distinct fp32 scalars."""
        return {"mods": len(self._mod_cache), "scalars": len(self._scalar_cache)}

    def clear_step_constants(self) -> None:
        """Free every cached step constant (the next ``prepare`` uploads what its schedule needs again)."""
        for tree in self._mod_cache.values():
            for t in tree.values():
                ttnn.deallocate(t)
        self._mod_cache.clear()
        for t in self._scalar_cache.values():
            ttnn.deallocate(t)
        self._scalar_cache.clear()

    def prepare(self, request: Flux2KleinDenoiseRequest) -> Flux2KleinPreparedRequest:
        """Bind ``request`` to the slot set of its layout key: take the plans from the transformer's cache, build the
        hoisted prompt embeddings, put the noise and the reference tokens into the fp32 state slot. A NEW key first
        releases every captured trace
        (recaptured on the next traced step of each key), allocates the slots, compiles every program of the layout
        with one eager step and warms every copy program, so that nothing a later replay writes can alias them (module
        docstring); an existing key is refreshed in place by ``ttnn.copy`` so its captured trace keeps working. Either
        way the step constants of the schedule (modulation per timestep, the fp32 scalars per value) the denoiser's
        caches do not hold yet are uploaded once (the eager steps read them). A refresh whose plan is a different
        cache entry than the slot held re-points the slot and releases the key's trace (module docstring).
        Text handed over on the device (``Flux2KleinDenoiseRequest`` docstring) is consumed here."""
        try:
            self._validate(request)
            layout = self._layout(request)
            key = layout_key(layout, request.cfg)
            scheduler = Flux2KleinScheduler(request.num_steps, request.image_seq_len)
            num_rows, num_rows_padded = layout.image, layout.padded
            x_host = self._state_host(request, num_rows_padded)
            spatial_axes = [None, self._sp_axis, None]

            if key not in self._slots:
                # Nothing may be allocated for a new layout while a trace is live (module docstring).
                self.release_traces()
            fresh_c = self._fresh_sample(request, layout, False)
            fresh_u = self._fresh_sample(request, layout, True) if request.cfg else None
        except BaseException:
            self._release_device_text(request)
            raise
        if key not in self._slots:
            slots = _LayoutSlots(
                key=key,
                layout=layout,
                cond=fresh_c,
                uncond=fresh_u,
                x=StateTensor(),
                t_embed=StateTensor(),
                dt=StateTensor(),
                guidance=StateTensor() if request.cfg else None,
                mod=(
                    {name: StateTensor() for name in self._transformer.host_time(scheduler.step(0).embedding_t).slots()}
                    if self._transformer.host_time is not None
                    else None
                ),
                mask=tensor.from_torch(
                    image_row_mask(num_rows, num_rows_padded),
                    device=self._device,
                    dtype=ttnn.float32,
                    mesh_axes=spatial_axes,
                ),
                num_image_rows=num_rows,
                num_rows_padded=num_rows_padded,
            )
            slots.x.update(x_host, True, dtype=ttnn.float32, mesh_axes=spatial_axes, device=self._device)
            self._write_step_slots(slots, scheduler.step(0))
            if request.cfg:
                self._scalar_slot(slots.guidance, torch.tensor([[request.guidance]], dtype=_F32))
            self._slots[key] = slots
            # Compile every program of the layout before any capture; the pure step leaves the state slot untouched.
            x_new, velocity = self._step(**self._step_inputs(slots, key), traced=False)
            ttnn.deallocate(x_new)
            ttnn.deallocate(velocity)
            for t in slots.tensors():
                warm_copy(t)
            ttnn.synchronize_device(self._device)
        else:
            slots = self._slots[key]
            for fresh, slot in ((fresh_c, slots.cond), (fresh_u, slots.uncond)):
                if fresh is None:
                    continue
                tree_copy_into(fresh.prompt, slot.prompt)
                ttnn.deallocate(fresh.prompt)
                if fresh.plan is not slot.plan:
                    # The captured trace of this key read the old plan's addresses.
                    self.release_trace(key)
                    slot.plan = fresh.plan
            slots.x.update(x_host, True, dtype=ttnn.float32, mesh_axes=spatial_axes, device=self._device)
            if request.cfg:
                self._scalar_slot(slots.guidance, torch.tensor([[request.guidance]], dtype=_F32))
        # The schedule's step constants, once per distinct value (the eager steps read them; module docstring).
        self._ensure_step_constants(scheduler)
        return Flux2KleinPreparedRequest(self, request, key, scheduler, slots)

    # --- stepping ---------------------------------------------------------------------------------------------------

    @traced_function(device=lambda self: self._device, clone_prep_inputs=False)
    def _step(
        self,
        *,
        x: ttnn.Tensor,
        t_embed: ttnn.Tensor,
        dt: ttnn.Tensor,
        guidance: ttnn.Tensor | None,
        mask: ttnn.Tensor,
        key: LayoutKey,
        cond: dict,
        uncond: dict | None,
        mod: dict | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """One pure Euler step (the module docstring's chain): ``(x_new, velocity)`` from the slots."""
        layout = self._slots[key].layout
        img = ttnn.typecast(x, ttnn.bfloat16)
        time_kwargs = (
            {"t_embed": t_embed}
            if mod is None
            else {
                "mod": Flux2KleinModulation(
                    img=tuple(mod[f"img{i}"] for i in range(6)),
                    txt=tuple(mod[f"txt{i}"] for i in range(6)),
                    single=tuple(mod[f"single{i}"] for i in range(3)),
                    norm_out_scale_plus_one=mod["out_scale"],
                    norm_out_shift=mod["out_shift"],
                    time_embed=mod["time_embed"],
                )
            }
        )
        c = self._transformer(
            spatial=img,
            prompt_embedded=cond["prompt"],
            plan=Flux2KleinPlan.from_tree(layout, cond["plan"]),
            **time_kwargs,
        )
        if uncond is not None:
            u = self._transformer(
                spatial=img,
                prompt_embedded=uncond["prompt"],
                plan=Flux2KleinPlan.from_tree(layout, uncond["plan"]),
                **time_kwargs,
            )
            ttnn.deallocate(img)
            v = combine_cfg(c, u, guidance)
            ttnn.deallocate(c)
            ttnn.deallocate(u)
        else:
            ttnn.deallocate(img)
            v = c
        return round_state_to_bf16(euler_update(x, v, dt, mask)), v

    def _step_inputs(self, slots: _LayoutSlots, key: LayoutKey) -> dict:
        for sample in (slots.cond, slots.uncond):
            if sample is not None and not all(t.is_allocated() for t in sample.plan_leaves()):
                raise RuntimeError(
                    f"the plan of layout key {key} was evicted from the transformer's plan cache (more than its "
                    f"maxsize distinct canvases prepared since this request's prepare); prepare the request again"
                )
        return {
            "x": slots.x.value,
            "t_embed": slots.t_embed.value,
            "dt": slots.dt.value,
            "guidance": slots.guidance.value if slots.guidance is not None else None,
            "mask": slots.mask,
            "key": key,
            "cond": slots.cond.tree(),
            "uncond": slots.uncond.tree() if slots.uncond is not None else None,
            "mod": slots.mod_tree(),
        }

    def step(self, prepared: Flux2KleinPreparedRequest, index: int, *, traced: bool) -> ttnn.Tensor:
        """Advance the state slot by step ``index`` of the prepared request's schedule. Returns the combined bf16
        velocity ``[1, L_pad, 128]`` (``mesh_axes=[None, sp, None]``; the reference and pad rows carry velocities the
        update discards): in eager mode the caller owns it; in traced mode it is the trace's output buffer, to be read
        before the next step and never deallocated. The first traced step of a key captures its trace (releasing the
        least recently replayed one when ``max_live_traces`` are held); later traced steps replay it. An eager step
        uploads nothing: its ``t_embed`` / ``dt`` scalars and its modulation come from the denoiser's caches (filled by
        :meth:`prepare`); a traced step refreshes the ``t_embed`` / ``dt`` / ``mod`` slots by ``ttnn.copy`` (module
        docstring)."""
        slots = prepared.slots
        key = prepared.key
        step = prepared.scheduler.step(index)
        inputs = self._step_inputs(slots, key)
        if traced:
            # The trace reads the fixed-address slots: refresh them.
            self._write_step_slots(slots, step, mod=True)
        else:
            inputs["t_embed"], inputs["dt"], inputs["mod"] = self._step_constants(step)
        capturing = False
        if traced:
            tracer = self._tracer(key)
            capturing = tracer is None or not tracer.trace_captured
            if capturing:
                while len(self._captured) >= self._max_live_traces:
                    self.release_trace(self._captured[0])
            else:
                self._captured.remove(key)
                self._captured.append(key)
        x_new, velocity = self._step(**inputs, traced=traced, tracer_blocking_execution=False, tracer_trace_key=key)
        if capturing:
            self._captured.append(key)
            # The outputs may sit where an older live trace's replay writes; they are consumed before that (docstring).
            TraceAllocationTracker.acknowledge_corruptible(x_new)
            TraceAllocationTracker.acknowledge_corruptible(velocity)
        ttnn.copy(x_new, slots.x.value)
        if not traced:
            ttnn.deallocate(x_new)
        return velocity

    def denoise_on_device(
        self,
        prepared: Flux2KleinPreparedRequest,
        *,
        traced: bool,
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> ttnn.Tensor:
        """Run every step and return the fp32 state SLOT ``[1, L_pad, 128]`` (``mesh_axes=[None, sp, None]``, rows
        ``[image | references | zero pad]``) after ``ttnn.synchronize_device``, without a host read: the pipeline hands
        it to the device VAE. The slot belongs to the denoiser (never deallocate it; the next ``prepare`` of the key
        refreshes it). ``on_step`` as in :meth:`denoise`."""
        if on_step is not None:
            on_step(0, self.read_state(prepared))
        for index in range(prepared.num_steps):
            velocity = self.step(prepared, index, traced=traced)
            if not traced:
                ttnn.deallocate(velocity)
            if on_step is not None:
                on_step(index + 1, self.read_state(prepared))
        ttnn.synchronize_device(self._device)
        return prepared.slots.x.value

    def denoise(
        self,
        prepared: Flux2KleinPreparedRequest,
        *,
        traced: bool,
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Run every step; returns the final fp32 ``[S_img, 128]`` tokens on the host. ``on_step(i, x_i)`` sees the
        state ENTERING step ``i`` for ``i`` in ``0..num_steps`` (``i == num_steps`` is the final state) as a host
        clone; each hook read synchronizes the device. The loop is :meth:`denoise_on_device` followed by
        :meth:`read_state` (bit-identical path)."""
        self.denoise_on_device(prepared, traced=traced, on_step=on_step)
        return self.read_state(prepared)

    def has_layout(self, request: Flux2KleinDenoiseRequest) -> bool:
        """True when ``request``'s layout key already owns a slot set (``prepare`` will refresh in place rather than
        allocate; a pipeline holding OTHER live traces must release them before a ``prepare`` that allocates, by the
        aliasing rule of the module docstring)."""
        return layout_key(self._layout(request), request.cfg) in self._slots

    def write_state(self, prepared: Flux2KleinPreparedRequest, x: torch.Tensor) -> None:
        """Overwrite the image rows of the fp32 state slot with ``x`` ``[S_img, 128]`` by ``ttnn.copy`` (the reference
        rows are taken from the prepared request, the pad rows stay zero): a teacher-forced step for the tests
        (``prepare`` does the same with the request's noise)."""
        slots = prepared.slots
        if tuple(x.shape) != (slots.num_image_rows, FLUX2_KLEIN_IN_CHANNELS):
            raise ValueError(f"state must be [{slots.num_image_rows}, 128], got {tuple(x.shape)}")
        rows = x.to(_F32)
        if prepared.request.image_cond is not None:
            rows = torch.cat([rows, prepared.request.image_cond[0].to(_F32)])
        padded = pad_rows(rows, slots.num_rows_padded)[0][None].contiguous()
        slots.x.update(padded, True, dtype=ttnn.float32, mesh_axes=[None, self._sp_axis, None], device=self._device)

    # --- host reads -------------------------------------------------------------------------------------------------

    def read_padded_state(self, prepared: Flux2KleinPreparedRequest) -> torch.Tensor:
        """All ``L_pad`` rows of the fp32 state (reference rows constant, alignment rows exactly zero)."""
        ttnn.synchronize_device(self._device)
        return tensor.to_torch(prepared.slots.x.value, mesh_axes=[None, self._sp_axis, None])[0].clone()

    def read_state(self, prepared: Flux2KleinPreparedRequest) -> torch.Tensor:
        """The fp32 ``[S_img, 128]`` state (bf16-valued; synchronizes the device first)."""
        return self.read_padded_state(prepared)[: prepared.slots.num_image_rows].clone()

    def read_velocity(self, prepared: Flux2KleinPreparedRequest, velocity: ttnn.Tensor) -> torch.Tensor:
        """The image rows of a velocity returned by :meth:`step`, bf16 ``[S_img, 128]`` on the host (``P:862``)."""
        ttnn.synchronize_device(self._device)
        rows = tensor.to_torch(velocity, mesh_axes=[None, self._sp_axis, None])[0]
        return rows[: prepared.slots.num_image_rows].clone()

    # --- release ----------------------------------------------------------------------------------------------------

    def release_trace(self, key: LayoutKey) -> None:
        """Release the captured trace of ``key`` (no-op when none was captured); the key's slots stay."""
        tracers = type(self)._step._tracers_keyed.get(self)
        tracer = tracers.pop(key, None) if tracers else None
        if tracer is not None:
            tracer.release_trace()
        if key in self._captured:
            self._captured.remove(key)

    def release_traces(self) -> None:
        for key in list(self._slots):
            self.release_trace(key)

    def deallocate(self, key: LayoutKey | None = None) -> None:
        """Free the slot set of ``key`` (all keys when ``None``, together with the step-constant caches); its trace is
        released first. The caches are shared by the keys, so freeing one key keeps them."""
        for k in [key] if key is not None else list(self._slots):
            self.release_trace(k)
            slots = self._slots.pop(k, None)
            if slots is None:
                continue
            for t in slots.tensors():
                ttnn.deallocate(t)
            ttnn.deallocate(slots.mask)
        if key is None:
            self.clear_step_constants()


__all__ = [
    "Flux2KleinDenoiseRequest",
    "Flux2KleinDenoiser",
    "Flux2KleinPreparedRequest",
    "LayoutKey",
    "combine_cfg",
    "euler_update",
    "image_row_mask",
    "layout_key",
    "round_state_to_bf16",
    "tree_copy_into",
    "tree_leaves",
    "warm_copy",
]

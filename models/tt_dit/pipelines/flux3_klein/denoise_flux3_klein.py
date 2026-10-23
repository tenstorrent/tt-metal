# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein Euler sampler on device: the whole denoising loop of ``reference/flux3_klein/sampling.py``
(``denoise`` for the base model with classifier-free guidance, ``denoise_flow_map`` for the distilled model) on
one :class:`Flux3KleinTransformer`, with the step captured as ONE ttnn trace. Design record: ``docs/M4_DESIGN.md``.

Reference math, per step ``i`` (``sampling.py:36-54``) and how each piece is reproduced here (PLAN decision 10:
exact fidelity; the DiT itself carries the M3 floors, every other operation is bit-equal to torch):

* ``t = schedule[i].item()``, ``dt = (schedule[i+1] - schedule[i]).item()`` in fp32: :class:`Flux3KleinScheduler`
  (its module docstring pins the three rounding facts); the DiT sees ``t`` and ``t_next`` rounded to bf16 and
  ``1000 * t`` rounded again, computed on the host by ``modulation_flux3_klein.timestep_inputs``.
* autocast casts the fp32 image state to bf16 at ``emb_in`` (``sampling.py:40,54``): ``ttnn.typecast(x, bf16)`` is
  value-equal to torch's round-to-nearest-even cast on every probed input (random with exponent spread, 65544 exact
  ties, carry into the next exponent, near-max -> inf; measured 2026-10-17, M4 design and verification probes) with
  ONE exception: a ``-0.0`` input comes back as ``+0.0`` (torch keeps ``-0.0``). A ``-0.0`` state element can only
  arise from ``-0 + -0`` (never from N(0, 1) noise plus a bf16 product), and the DiT sees ``+0`` and ``-0`` alike, so
  this is documented, not gated.
* CFG ``u + g * (c - u)`` on bf16 tensors with a Python float ``g``: torch rounds each op to bf16. Device:
  ``c``/``u`` widened to fp32, ``subtract`` -> typecast bf16 -> fp32, ``multiply`` by the fp32 ``[1, 1]`` guidance
  SLOT -> bf16 -> fp32, ``add`` -> bf16 (:func:`combine_cfg`). Bit-equal to ``Flux3KleinScheduler.cfg_velocity``
  for ``g`` in 0.1 / 1.0 / 2.0 / 3.3 / 3.5 / 7.7 on exponent-gap data (measured 2026-10-17).
* ``x_fp32 + dt * v``: the product in bf16 (``v`` is the DiT's bf16 output), widened exactly, fp32 add. Device:
  ``typecast(v, fp32)`` x the fp32 ``[1, 1]`` dt SLOT -> typecast bf16 -> fp32, times the fp32 ``{1, 0}`` row mask
  (pad rows become +/-0), fp32 ``add`` (:func:`euler_update`); ``+0 + (-0) = +0`` keeps the pad rows exactly zero.
  Bit-equal to the goldens' ``step{k+1}.x_t`` / ``final_tokens`` when fed the golden velocities.

Forms that were measured WRONG and must never return (counts are mismatching elements vs torch):
``ttnn.multiply(v_bf16, dt_python_float)`` / ``multiply_`` (the scalar is rounded to bf16: 221570 / 524288 for a
schedule dt); the mixed ``ttnn.multiply(d_bf16, g_slot_fp32)`` (21465 / 262144 at g = 3.5, 5631 at 3.3, 2694 at
7.7); bf16-in / bf16-out ``subtract`` / ``add`` with the default ``fast_and_approximate_mode=True`` (17796-29117);
``ttnn.add(x_fp32, p_bf16)`` mixed (downcasts ``x``: 510937 / 524288); any Python scalar for ``dt`` or the guidance
inside the traced step (also rejected by the Tracer across replays). No binary op in the chain mixes dtypes.

Trace contract (``utils/tracing.py``):

* :meth:`Flux3KleinDenoiser._step` is PURE: it reads the fp32 state slot ``x`` and returns ``(x_new, velocity)``;
  the host ``ttnn.copy``'s ``x_new`` into the slot after the call (same command queue, so it is ordered after the
  replay) and deallocates ``x_new`` only in eager mode (in traced mode it is the trace's own output buffer).
  Tracer defaults: the first traced call runs the function once (prep run, nothing is mutated), captures, executes.
* Slots, one set per layout key ``(flow_map, sp, cond lengths, uncond lengths)``, live at fixed addresses for the
  life of the denoiser: ``x`` fp32 ``[1, L_img_pad, 128]`` (``mesh_axes=[None, sp, None]``, pad rows zero), ``t_embed``
  fp32 ``[3, 1]``, ``t_ref_embed`` fp32 ``[1, 1]`` (flow map), ``dt`` fp32 ``[1, 1]``, ``guidance`` fp32 ``[1, 1]``
  (CFG), the constant ``mask``, and per prompt the plan tree (:class:`Flux3KleinSequencePlan` flattened to nested
  dicts / tuples), the hoisted ``text_embedded`` and ``image_cond_embedded``. Every refresh is a ``ttnn.copy`` into
  the slot (``StateTensor.update(..., traced=True)`` once bound; :func:`tree_copy_into` for the trees): no per-step
  value is ever baked into the trace, and a second request with the same layout reuses the captured trace.
* A replay writes the trace's intermediate buffers at the addresses they had during the capture; those addresses are
  FREE after the capture, so any buffer allocated later may land on them and be overwritten by the replay (Tracer
  caveat 1). The sampler therefore keeps this invariant: EVERY long-lived buffer a replay could hit exists before the
  trace is captured. :meth:`prepare` of a NEW key first releases every captured trace (they are recaptured on their
  next traced step), then allocates the key's slots, runs ONE eager step on them so every program of the layout is
  compiled (a program's kernel binaries live in a DRAM buffer allocated at compile time) and :func:`warm_copy`'s the
  copy program of every slot tensor (a ``CopyDeviceOperation`` compiled after a capture was measured to leave such a
  buffer where the replay overwrites it; ``TT_METAL_TRACE_ALLOC_TRACKING=1`` fails). Without the release, a second
  layout's slots were measured corrupted by the first layout's replay (its mask and state rows, 2026-10-17).
* Several keys may hold a captured trace at once (at most ``max_live_traces``, least recently replayed released
  first; the trace region is ``device_params.TRACE_REGION_SIZE``). A trace's two output buffers are allocated during
  its capture, i.e. possibly inside an OLDER live trace's intermediate region: that older trace's replay may overwrite
  them, which is harmless because a step's outputs are consumed before any other replay (``x_new`` is copied into the
  state slot on the same command queue right after the replay; the velocity is read before the next step). The
  outputs are therefore acknowledged to the trace allocation tracker as corruptible; nothing else is.
* Replay is non-blocking (``tracer_blocking_execution=False``); the host reads only through :meth:`read_state` /
  :meth:`read_velocity`, which synchronize the device first. Nothing may be allocated on device between replays
  except the slot-refresh temporaries (freed before the next replay) and the transients of an eager step (freed
  inside it); an eager step of a key whose programs are compiled allocates nothing long-lived.
* Release order: traces (:meth:`Flux3KleinPreparedRequest.release` / :meth:`release_traces`) before the slots are
  freed, before the model's weights are freed and before the mesh closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

import ttnn

from ...models.transformers.flux3_klein.layout_flux3_klein import Flux3KleinSequenceLayout
from ...models.transformers.flux3_klein.modulation_flux3_klein import timestep_inputs
from ...models.transformers.flux3_klein.transformer_flux3_klein import Flux3KleinSequencePlan, Flux3KleinTransformer
from ...utils import tensor
from ...utils.tracing import StateTensor, traced_function
from .packing import FLUX3_KLEIN_IN_CHANNELS, pad_rows, padded_length
from .scheduler import FLUX3_KLEIN_ALPHA, Flux3KleinScheduler, Flux3KleinStep

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

LayoutKey = tuple
"""``(flow_map, sp, tuple(sorted(cond.lengths.items())), None | tuple(sorted(uncond.lengths.items())))``."""

_F32 = torch.float32
_BF16 = torch.bfloat16


# --- host helpers -----------------------------------------------------------------------------------------------------


def layout_key(flow_map: bool, cond: Flux3KleinSequenceLayout, uncond: Flux3KleinSequenceLayout | None) -> LayoutKey:
    """The hashable identity of a slot set and its trace (``Flux3KleinSequenceLayout`` holds a dict)."""
    return (
        bool(flow_map),
        int(cond.sp_factor),
        tuple(sorted(cond.lengths.items())),
        None if uncond is None else tuple(sorted(uncond.lengths.items())),
    )


def image_row_mask(num_rows: int, num_rows_padded: int) -> torch.Tensor:
    """fp32 ``[1, num_rows_padded, 128]``: ``1.0`` on the ``num_rows`` real rows, ``0.0`` on the alignment rows."""
    if not 0 < num_rows <= num_rows_padded:
        raise ValueError(f"need 0 < num_rows <= num_rows_padded, got {num_rows}, {num_rows_padded}")
    mask = torch.zeros(1, num_rows_padded, FLUX3_KLEIN_IN_CHANNELS["image"], dtype=_F32)
    mask[:, :num_rows] = 1.0
    return mask


def plan_tree(plan: Flux3KleinSequencePlan) -> dict:
    """The device tensors of a plan as nested dicts / tuples (a Tracer-valid input; the layout stays on the host)."""
    return {
        "early_rope": {mode: tuple(pair) for mode, pair in plan.early_rope.items()},
        "early_mod_indices": {mode: tuple(triple) for mode, triple in plan.early_mod_indices.items()},
        "single_rope": tuple(plan.single_rope),
        "single_mod_indices": tuple(plan.single_mod_indices),
        "assembly_indices": plan.assembly_indices,
        "image_out_indices": plan.image_out_indices,
    }


def plan_from_tree(layout: Flux3KleinSequenceLayout, tree: dict) -> Flux3KleinSequencePlan:
    """Inverse of :func:`plan_tree` for ``layout``."""
    return Flux3KleinSequencePlan(layout=layout, **tree)


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
    as torch rounds it (``sampling.py:41``, ``Flux3KleinScheduler.cfg_velocity``). Returns bf16; the inputs are left
    to the caller."""
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
    row mask (``sampling.py:42-43,54``). Returns a new fp32 tensor; pad rows of the result are exactly ``+0`` when the
    pad rows of ``x`` are."""
    v32 = ttnn.typecast(velocity, ttnn.float32)
    p = ttnn.typecast(ttnn.typecast(ttnn.multiply(v32, dt), ttnn.bfloat16), ttnn.float32)
    ttnn.deallocate(v32)
    delta = ttnn.multiply(p, mask)
    ttnn.deallocate(p)
    x_new = ttnn.add(x, delta)
    ttnn.deallocate(delta)
    return x_new


# --- requests ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Flux3KleinDenoiseRequest:
    """One image to sample. Host tensors in the pipeline's dtypes (``packing.py``): ``noise`` ``[L_img, 128]`` fp32
    (bf16-valued, the state entering step 0), ``image_ids`` ``[L_img, 4]`` fp32 (the scaled ids ``pack`` receives),
    ``text`` ``([L_txt, 20480] bf16, [L_txt, 4] int64)``; ``guidance`` is the CFG scale of the base model (two
    forwards per step, ``text_null`` = the empty prompt's encoding is then required) and ``None`` for the distilled
    flow map; ``image_cond`` ``([N, 128] bf16, [N, 4] fp32)`` is shared by both CFG samples."""

    noise: torch.Tensor
    image_ids: torch.Tensor
    text: tuple[torch.Tensor, torch.Tensor]
    num_steps: int
    guidance: float | None = None
    text_null: tuple[torch.Tensor, torch.Tensor] | None = None
    image_cond: tuple[torch.Tensor, torch.Tensor] | None = None
    alpha: float = FLUX3_KLEIN_ALPHA

    @property
    def cfg(self) -> bool:
        return self.guidance is not None

    def text_of(self, unconditional: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if unconditional:
            if self.text_null is None:
                raise ValueError("the unconditional sample needs text_null")
            return self.text_null
        return self.text

    def lengths(self, unconditional: bool = False) -> dict[str, int]:
        return {
            "image": int(self.noise.shape[0]),
            "image_cond": 0 if self.image_cond is None else int(self.image_cond[0].shape[0]),
            "text": int(self.text_of(unconditional)[0].shape[0]),
        }

    def ids(self, unconditional: bool = False) -> dict[str, torch.Tensor]:
        ids = {"image": self.image_ids, "text": self.text_of(unconditional)[1]}
        if self.image_cond is not None:
            ids["image_cond"] = self.image_cond[1]
        return ids


# --- slots ------------------------------------------------------------------------------------------------------------


@dataclass
class _SampleSlots:
    """One prompt's step-invariant device tensors (fixed addresses)."""

    layout: Flux3KleinSequenceLayout
    plan: dict
    text: ttnn.Tensor
    image_cond: ttnn.Tensor | None

    def tree(self) -> dict:
        return {"plan": self.plan, "text": self.text, "image_cond": self.image_cond}


@dataclass
class _LayoutSlots:
    """Everything a layout key's trace reads, owned by the denoiser."""

    key: LayoutKey
    cond: _SampleSlots
    uncond: _SampleSlots | None
    x: StateTensor
    t_embed: StateTensor
    t_ref_embed: StateTensor | None
    dt: StateTensor
    guidance: StateTensor | None
    mask: ttnn.Tensor
    num_image_rows: int
    num_image_rows_padded: int

    def tensors(self) -> list[ttnn.Tensor]:
        """Every slot tensor that is refreshed by ``ttnn.copy`` (the mask is constant)."""
        out = list(tree_leaves(self.cond.tree()))
        if self.uncond is not None:
            out.extend(tree_leaves(self.uncond.tree()))
        for state in (self.x, self.t_embed, self.t_ref_embed, self.dt, self.guidance):
            if state is not None and state.value is not None:
                out.append(state.value)
        return out


class Flux3KleinPreparedRequest:
    """A request bound to a denoiser's slot set; ``release()`` frees its trace and slots (in that order)."""

    def __init__(
        self,
        denoiser: Flux3KleinDenoiser,
        request: Flux3KleinDenoiseRequest,
        key: LayoutKey,
        scheduler: Flux3KleinScheduler,
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


class Flux3KleinDenoiser:
    """Euler sampling on ``transformer`` (see the module docstring). ``sp_axis`` is the mesh axis the image rows are
    sharded over (``parallel_config.sequence_parallel.mesh_axis``); ``max_live_traces`` bounds the captured traces
    held at once (the least recently replayed is released first; a 4x8 ring base CFG T2I trace is 25.6 MB of the
    64 MB trace region)."""

    def __init__(
        self, transformer: Flux3KleinTransformer, *, device: ttnn.MeshDevice, sp_axis: int, max_live_traces: int = 2
    ) -> None:
        if max_live_traces < 1:
            raise ValueError(f"max_live_traces must be at least 1, got {max_live_traces}")
        self._transformer = transformer
        self._device = device
        self._sp_axis = sp_axis
        self._max_live_traces = max_live_traces
        self._flow_map = bool(transformer.flow_map)
        self._layouts: dict[LayoutKey, tuple[Flux3KleinSequenceLayout, Flux3KleinSequenceLayout | None]] = {}
        self._slots: dict[LayoutKey, _LayoutSlots] = {}
        self._captured: list[LayoutKey] = []
        """Keys holding a captured trace, least recently replayed first."""

    @property
    def flow_map(self) -> bool:
        return self._flow_map

    @property
    def captured_keys(self) -> tuple[LayoutKey, ...]:
        """The layout keys holding a captured trace, least recently replayed first."""
        return tuple(self._captured)

    def _tracer(self, key: LayoutKey):
        tracers = type(self)._step._tracers_keyed.get(self)
        return tracers.get(key) if tracers else None

    # --- preparation ------------------------------------------------------------------------------------------------

    def _validate(self, request: Flux3KleinDenoiseRequest) -> None:
        if request.cfg == self._flow_map:
            raise ValueError(
                f"guidance {request.guidance} on a {'flow-map' if self._flow_map else 'CFG'} transformer: "
                "the base model needs a guidance, the distilled flow map none"
            )
        if (request.text_null is not None) != request.cfg:
            raise ValueError("text_null is required by, and only by, a CFG request")
        if request.noise.ndim != 2 or request.noise.shape[0] == 0:
            raise ValueError(f"noise must be [L_img, 128] with L_img > 0, got {tuple(request.noise.shape)}")
        if request.num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {request.num_steps}")

    def _upload_rows(self, rows: torch.Tensor) -> ttnn.Tensor:
        """``[L, C]`` -> bf16 ``[1, L_pad, C]`` ``mesh_axes=[None, sp, None]`` with zero alignment rows."""
        padded = pad_rows(rows.to(_BF16), padded_length(rows.shape[0], self._transformer.sp_factor))[0]
        return tensor.from_torch(
            padded[None], device=self._device, dtype=ttnn.bfloat16, mesh_axes=[None, self._sp_axis, None]
        )

    def _fresh_sample(self, request: Flux3KleinDenoiseRequest, unconditional: bool) -> _SampleSlots:
        """Plan and hoisted projections of one prompt (fresh device tensors)."""
        layout = self._transformer.sequence_layout(request.lengths(unconditional))
        plan = self._transformer.prepare_plan(layout, request.ids(unconditional))
        tokens = self._upload_rows(request.text_of(unconditional)[0])
        text = self._transformer.embed_text(tokens)
        ttnn.deallocate(tokens)
        image_cond = None
        if request.image_cond is not None:
            tokens = self._upload_rows(request.image_cond[0])
            image_cond = self._transformer.embed_image_cond(tokens)
            ttnn.deallocate(tokens)
        return _SampleSlots(layout=layout, plan=plan_tree(plan), text=text, image_cond=image_cond)

    def _scalar_slot(self, state: StateTensor, value: torch.Tensor) -> None:
        state.update(value, True, dtype=ttnn.float32, mesh_axes=[None, None], device=self._device)

    def _write_step_slots(self, slots: _LayoutSlots, step: Flux3KleinStep) -> None:
        t_embed, t_ref_embed = timestep_inputs(step, self._flow_map)
        self._scalar_slot(slots.t_embed, t_embed)
        if self._flow_map:
            self._scalar_slot(slots.t_ref_embed, t_ref_embed)
        self._scalar_slot(slots.dt, torch.full((1, 1), step.dt, dtype=_F32))

    def prepare(self, request: Flux3KleinDenoiseRequest) -> Flux3KleinPreparedRequest:
        """Bind ``request`` to the slot set of its layout key: build the layouts, plans and hoisted projections, put the
        noise into the fp32 state slot. A NEW key first releases every captured trace (recaptured on the next traced
        step of each key), allocates the slots, compiles every program of the layout with one eager step and warms
        every copy program, so that nothing a later replay writes can alias them (module docstring); an existing key
        is refreshed in place by ``ttnn.copy`` so its captured trace keeps working."""
        self._validate(request)
        layout_c = self._transformer.sequence_layout(request.lengths())
        layout_u = self._transformer.sequence_layout(request.lengths(True)) if request.cfg else None
        key = layout_key(self._flow_map, layout_c, layout_u)
        scheduler = Flux3KleinScheduler(request.num_steps, request.alpha)
        num_rows, num_rows_padded = layout_c.lengths["image"], layout_c.padded["image"]
        x_host = pad_rows(request.noise.to(_F32), num_rows_padded)[0][None]
        image_axes = [None, self._sp_axis, None]

        if key not in self._slots:
            # Nothing may be allocated for a new layout while a trace is live (module docstring).
            self.release_traces()
        fresh_c = self._fresh_sample(request, False)
        fresh_u = self._fresh_sample(request, True) if request.cfg else None
        if key not in self._slots:
            slots = _LayoutSlots(
                key=key,
                cond=fresh_c,
                uncond=fresh_u,
                x=StateTensor(),
                t_embed=StateTensor(),
                t_ref_embed=StateTensor() if self._flow_map else None,
                dt=StateTensor(),
                guidance=StateTensor() if request.cfg else None,
                mask=tensor.from_torch(
                    image_row_mask(num_rows, num_rows_padded),
                    device=self._device,
                    dtype=ttnn.float32,
                    mesh_axes=image_axes,
                ),
                num_image_rows=num_rows,
                num_image_rows_padded=num_rows_padded,
            )
            slots.x.update(x_host, True, dtype=ttnn.float32, mesh_axes=image_axes, device=self._device)
            self._write_step_slots(slots, scheduler.step(0))
            if request.cfg:
                self._scalar_slot(slots.guidance, torch.tensor([[request.guidance]], dtype=_F32))
            self._layouts[key] = (layout_c, layout_u)
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
                tree_copy_into(fresh.tree(), slot.tree())
                for t in tree_leaves(fresh.tree()):
                    ttnn.deallocate(t)
            slots.x.update(x_host, True, dtype=ttnn.float32, mesh_axes=image_axes, device=self._device)
            if request.cfg:
                self._scalar_slot(slots.guidance, torch.tensor([[request.guidance]], dtype=_F32))
        return Flux3KleinPreparedRequest(self, request, key, scheduler, slots)

    # --- stepping ---------------------------------------------------------------------------------------------------

    @traced_function(device=lambda self: self._device, clone_prep_inputs=False)
    def _step(
        self,
        *,
        x: ttnn.Tensor,
        t_embed: ttnn.Tensor,
        t_ref_embed: ttnn.Tensor | None,
        dt: ttnn.Tensor,
        guidance: ttnn.Tensor | None,
        mask: ttnn.Tensor,
        key: LayoutKey,
        cond: dict,
        uncond: dict | None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """One pure Euler step (the module docstring's chain): ``(x_new, velocity)`` from the slots."""
        layout_c, layout_u = self._layouts[key]
        img = ttnn.typecast(x, ttnn.bfloat16)
        c = self._transformer(
            image_tokens=img,
            text_embedded=cond["text"],
            image_cond_embedded=cond["image_cond"],
            t_embed=t_embed,
            t_ref_embed=t_ref_embed,
            plan=plan_from_tree(layout_c, cond["plan"]),
        )
        if uncond is not None:
            u = self._transformer(
                image_tokens=img,
                text_embedded=uncond["text"],
                image_cond_embedded=uncond["image_cond"],
                t_embed=t_embed,
                t_ref_embed=t_ref_embed,
                plan=plan_from_tree(layout_u, uncond["plan"]),
            )
            ttnn.deallocate(img)
            v = combine_cfg(c, u, guidance)
            ttnn.deallocate(c)
            ttnn.deallocate(u)
        else:
            ttnn.deallocate(img)
            v = c
        return euler_update(x, v, dt, mask), v

    def _step_inputs(self, slots: _LayoutSlots, key: LayoutKey) -> dict:
        return {
            "x": slots.x.value,
            "t_embed": slots.t_embed.value,
            "t_ref_embed": slots.t_ref_embed.value if self._flow_map else None,
            "dt": slots.dt.value,
            "guidance": slots.guidance.value if slots.guidance is not None else None,
            "mask": slots.mask,
            "key": key,
            "cond": slots.cond.tree(),
            "uncond": slots.uncond.tree() if slots.uncond is not None else None,
        }

    def step(self, prepared: Flux3KleinPreparedRequest, index: int, *, traced: bool) -> ttnn.Tensor:
        """Advance the state slot by step ``index`` of the prepared request's schedule. Returns the combined bf16
        velocity ``[1, L_img_pad, 128]`` (pad rows are copies of row 0): in eager mode the caller owns it; in traced
        mode it is the trace's output buffer, to be read before the next step and never deallocated. The first traced
        step of a key captures its trace (releasing the least recently replayed one when ``max_live_traces`` are
        held); later traced steps replay it."""
        slots = prepared.slots
        key = prepared.key
        step = prepared.scheduler.step(index)
        self._write_step_slots(slots, step)
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
        x_new, velocity = self._step(
            **self._step_inputs(slots, key), traced=traced, tracer_blocking_execution=False, tracer_trace_key=key
        )
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
        prepared: Flux3KleinPreparedRequest,
        *,
        traced: bool,
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> ttnn.Tensor:
        """Run every step and return the fp32 state SLOT ``[1, L_img_pad, 128]`` (``mesh_axes=[None, sp, None]``,
        alignment rows zero) after ``ttnn.synchronize_device``, without a host read: the M7 pipeline hands it to the
        device VAE. The slot belongs to the denoiser (never deallocate it; the next ``prepare`` of the key refreshes
        it). ``on_step`` as in :meth:`denoise`."""
        cache = getattr(self._transformer, "cache", None)
        if cache is not None:
            if traced:
                raise ValueError("DBCache is eager-only: a cached step is a different program sequence; run with traced=False")
            cache.begin_request()
        if on_step is not None:
            on_step(0, self.read_state(prepared))
        for index in range(prepared.num_steps):
            if cache is not None:
                cache.begin_step(index)
            velocity = self.step(prepared, index, traced=traced)
            if not traced:
                ttnn.deallocate(velocity)
            if on_step is not None:
                on_step(index + 1, self.read_state(prepared))
        ttnn.synchronize_device(self._device)
        return prepared.slots.x.value

    def denoise(
        self,
        prepared: Flux3KleinPreparedRequest,
        *,
        traced: bool,
        on_step: Callable[[int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Run every step; returns the final fp32 ``[L_img, 128]`` tokens on the host. ``on_step(i, x_i)`` sees the
        state ENTERING step ``i`` for ``i`` in ``0..num_steps`` (``i == num_steps`` is the final state) as a host
        clone; each hook read synchronizes the device. The loop is :meth:`denoise_on_device` followed by
        :meth:`read_state` (bit-identical path)."""
        self.denoise_on_device(prepared, traced=traced, on_step=on_step)
        return self.read_state(prepared)

    def has_layout(self, request: Flux3KleinDenoiseRequest) -> bool:
        """True when ``request``'s layout key already owns a slot set (``prepare`` will refresh in place rather than
        allocate; a pipeline holding OTHER live traces must release them before a ``prepare`` that allocates, by the
        aliasing rule of the module docstring)."""
        layout_c = self._transformer.sequence_layout(request.lengths())
        layout_u = self._transformer.sequence_layout(request.lengths(True)) if request.cfg else None
        return layout_key(self._flow_map, layout_c, layout_u) in self._slots

    def write_state(self, prepared: Flux3KleinPreparedRequest, x: torch.Tensor) -> None:
        """Overwrite the fp32 state slot with ``x`` ``[L_img, 128]`` (pad rows zero) by ``ttnn.copy``: a teacher-forced
        step for the tests (``prepare`` does the same with the request's noise)."""
        slots = prepared.slots
        if tuple(x.shape) != (slots.num_image_rows, FLUX3_KLEIN_IN_CHANNELS["image"]):
            raise ValueError(f"state must be [{slots.num_image_rows}, 128], got {tuple(x.shape)}")
        padded = pad_rows(x.to(_F32), slots.num_image_rows_padded)[0][None]
        slots.x.update(padded, True, dtype=ttnn.float32, mesh_axes=[None, self._sp_axis, None], device=self._device)

    # --- host reads -------------------------------------------------------------------------------------------------

    def read_padded_state(self, prepared: Flux3KleinPreparedRequest) -> torch.Tensor:
        """All ``L_img_pad`` rows of the fp32 state (the alignment rows must read exactly zero)."""
        ttnn.synchronize_device(self._device)
        return tensor.to_torch(prepared.slots.x.value, mesh_axes=[None, self._sp_axis, None])[0].clone()

    def read_state(self, prepared: Flux3KleinPreparedRequest) -> torch.Tensor:
        """The fp32 ``[L_img, 128]`` state (synchronizes the device first)."""
        return self.read_padded_state(prepared)[: prepared.slots.num_image_rows].clone()

    def read_velocity(self, prepared: Flux3KleinPreparedRequest, velocity: ttnn.Tensor) -> torch.Tensor:
        """The real rows of a velocity returned by :meth:`step`, bf16 ``[L_img, 128]`` on the host."""
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
        """Free the slot set of ``key`` (all keys when ``None``); its trace is released first."""
        for k in [key] if key is not None else list(self._slots):
            self.release_trace(k)
            slots = self._slots.pop(k, None)
            self._layouts.pop(k, None)
            if slots is None:
                continue
            for t in slots.tensors():
                ttnn.deallocate(t)
            ttnn.deallocate(slots.mask)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein DiT (``reference/flux3_klein/dit.py::DiT``) on a tt_dit mesh: one denoising step.

The reference forward (``dit.py:186-239``) for one sample, and where each piece lives here::

    vec[m]    = time_in(timestep_embedding(t_m)) (+ ref_time_in(t_next) for the flow map)   -> modulation (Flux3KleinModulation)
    early[m]  = early_stream_modulations[m](vec[m]); single[m] = single_stream_modulations[m](vec[m])
    x[m]      = txt_in / emb_in.image / emb_in.image_cond (data[m])                          -> txt_in, emb_in_image, emb_in_image_cond
    x[m]      = content_mode_blocks[m][0..D) / txt_mode_blocks[0..D) (x[m], rope(ids[m]), early[m])   -> early (Flux3KleinEarlyStacks)
    h         = cat([x[m] for non-empty m in MODES order]); per-token single[m] rows          -> assemble_single_stage + gathers
    h         = single_blocks[0..S)(h, rope(cat ids), mods)                                  -> single_blocks
    velocity  = final_layer.image(h[image rows], vec[image])  # linear((1 + scale) * LN(x) + shift)   -> norm_out + proj_out

Design record: ``docs/M2_M3_DESIGN.md`` (decisions 1-12). Only the ``depth`` / ``depth_single`` block counts and the
optional ``ref_time_in`` embedder (``flow_map=True``, the distilled checkpoint) distinguish the two models.

Tensor contract (``mesh_axes`` as ``utils.tensor.from_torch`` / ``to_torch`` take them; ``sp`` / ``tp`` are the
sequence- and tensor-parallel mesh axes of ``parallel_config``; ``L_m_pad = packing.padded_length(L_m, SP)``):

* ``image_tokens``: ``[1, L_img_pad, 128]`` bf16, ``mesh_axes=[None, sp, None]``, rows ``[L_img, L_img_pad)`` ZERO.
* ``text_embedded`` = :meth:`Flux3KleinTransformer.embed_text` of ``[1, L_txt_pad, 20480]`` bf16 ``[None, sp, None]``
  text tokens (zero pad rows) -> ``[1, L_txt_pad, 3072]`` ``[None, sp, tp]``; ``image_cond_embedded`` likewise from
  :meth:`embed_image_cond` of ``[1, L_cond_pad, 128]`` tokens, or ``None`` for a text-to-image request. Both
  projections are bias-free, so zero pad rows stay zero, and both are HOISTED: they do not depend on the timestep,
  so a request runs them once and reuses the results for every step; only the image rows are re-projected per
  step (``emb_in_image`` inside ``forward``).
* ``t_embed``: fp32 ``[3, 1]`` replicated, ``[[embedding_t], [0], [0]]`` (``modulation_flux3_klein.timestep_inputs``);
  ``t_ref_embed``: fp32 ``[1, 1]`` ``[[embedding_t_next]]`` for a flow-map instance, ``None`` otherwise.
* ``plan``: a :class:`Flux3KleinSequencePlan` from :meth:`prepare_plan`, the step-invariant constants of one
  sample layout (RoPE tables, modulation and assembly index tensors); built once per request and reused by every
  step and by both CFG forwards of the same prompt.
* Output: the velocity ``[1, L_img_pad, 128]`` bf16 ``[None, sp, None]`` in the IMAGE stack's own SP layout (rows
  ``[L_img, L_img_pad)`` are copies of row 0 and are discarded): ``tensor.to_torch(v, mesh_axes=[None, sp, None])[0, :L_img]``.
  M4's Euler update and the next step's ``emb_in_image`` are therefore device-local.

Rules the dataflow relies on:

* Every pad row of every input must be exactly zero. Pad rows of both stages carry the modulation
  ``(shift, scale_plus_one, gate) = (0, 1, 0)``, gathered from the bias-only row of the modulation table
  (``modulation_flux3_klein``), so a zero row stays exactly zero through every block and the single-stage assembly
  (``stacks_flux3_klein.assemble_single_stage``) can copy one of them as the sequence's pad rows. The ring SDPA masks
  every pad KEY row through ``logical_length`` (= the real row count of the tensor).
* An empty mode is skipped: the reference gives it one zero phantom token (``dit.py:192-198``) that never enters the
  single stage and is never attended, so the device runs no projection, no stack and no concat part for it (the
  layout's ``modes`` lists the non-empty modes; M0 and M2 pin the independence).
* CFG (base model): the two samples never attend to each other, so the step is TWO forwards on the same instance,
  one per prompt (unconditional: the empty prompt's 80 text tokens), each with its own plan and ``text_embedded``;
  the image tokens and ``t_embed`` are shared (``Flux3KleinScheduler.cfg_velocity`` combines the results).
* On a Ring topology a block's direct output aliases the CCLManager's reduce-scatter ping-pong buffer for that
  shard shape (two buffers alternate), so the early stacks return a private copy and the single stack only CHAINS
  its blocks: nothing here retains or deallocates a block's direct output (``stacks_flux3_klein.py`` has the
  measurement). ``trace`` (debug only) reads stage outputs synchronously for the same reason.

Loader (decision 10): :meth:`_prepare_torch_state` routes the reference ``DiT.state_dict()`` (base 357 keys,
distilled 359) by prefix only; each child's own ``_prepare_torch_state`` understands the reference naming of its
part, so every part is strictly loadable on its own. ``load_torch_state_dict(strict=True)`` reports a base state
fed to a flow-map instance as missing ``modulation.ref_time_in.*`` and a distilled state fed to a base instance as
unexpected ``modulation.ref_time_in.*``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import ttnn

from ....layers.linear import ColParallelLinear, Linear
from ....layers.module import Module, ModuleList
from ....layers.normalization import DistributedLayerNorm
from ....pipelines.flux3_klein.packing import FLUX3_KLEIN_IN_CHANNELS, FLUX3_KLEIN_MODES
from ....utils.substate import rename_substate
from .layout_flux3_klein import Flux3KleinSequenceLayout
from .modulation_flux3_klein import Flux3KleinModulation, gather_modulation, modulation_table_indices
from .rope_flux3_klein import rope_cos_sin, rope_to_device
from .stacks_flux3_klein import (
    FLUX3_KLEIN_EARLY_DEPTH,
    Flux3KleinEarlyStacks,
    assemble_single_stage,
    gather_rows_over_sp,
    indices_to_device,
)
from .transformer_block_flux3_klein import FLUX3_KLEIN_HIDDEN, Flux3KleinTransformerBlock

if TYPE_CHECKING:
    from collections.abc import Callable

    from ....parallel.config import DiTParallelConfig
    from ....parallel.manager import CCLManager

FLUX3_KLEIN_DEPTH = FLUX3_KLEIN_EARLY_DEPTH  # dit.py:21
FLUX3_KLEIN_DEPTH_SINGLE = 28  # dit.py:22
FLUX3_KLEIN_LAYERNORM_EPS = 1e-6  # dit.py:75

# Reference state prefix -> this module's child prefix (decision 10). Pop-and-rename only when present; everything
# else (txt_in.*, single_blocks.{i}.*) keeps the reference name because the child is named the same.
_STATE_PREFIXES = (
    ("emb_in.image", "emb_in_image"),
    ("emb_in.image_cond", "emb_in_image_cond"),
    ("time_in", "modulation.time_in"),
    ("ref_time_in", "modulation.ref_time_in"),
    ("early_stream_modulations", "modulation.early_stream_modulations"),
    ("single_stream_modulations", "modulation.single_stream_modulations"),
    ("final_layer.image.adaLN_modulation", "modulation.final_layer.image.adaLN_modulation"),
    ("content_mode_blocks", "early.content_mode_blocks"),
    ("txt_mode_blocks", "early.txt_mode_blocks"),
    ("final_layer.image.linear", "proj_out"),
)


@dataclass
class Flux3KleinSequencePlan:
    """Step-invariant device constants of ONE sample layout (see :meth:`Flux3KleinTransformer.prepare_plan`)."""

    layout: Flux3KleinSequenceLayout
    early_rope: dict[str, tuple[ttnn.Tensor, ttnn.Tensor]]  # per non-empty mode, length padded[m]
    early_mod_indices: dict[str, tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]]  # per mode: (shift, scale, gate) rows
    single_rope: tuple[ttnn.Tensor, ttnn.Tensor]  # length total_padded, G order
    single_mod_indices: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]
    assembly_indices: ttnn.Tensor  # indices_to_device(layout.assembly_indices())
    image_out_indices: ttnn.Tensor  # indices_to_device(layout.image_out_indices())

    def deallocate(self) -> None:
        """Free every device tensor of the plan (rope tables, modulation and assembly index tensors)."""
        for pair in (*self.early_rope.values(), self.single_rope):
            for t in pair:
                ttnn.deallocate(t)
        for triple in (*self.early_mod_indices.values(), self.single_mod_indices):
            for t in triple:
                ttnn.deallocate(t)
        ttnn.deallocate(self.assembly_indices)
        ttnn.deallocate(self.image_out_indices)


class Flux3KleinTransformer(Module):
    """The Klein DiT; see the module docstring for the tensor contract."""

    def __init__(
        self,
        *,
        flow_map: bool = False,
        depth: int = FLUX3_KLEIN_DEPTH,
        depth_single: int = FLUX3_KLEIN_DEPTH_SINGLE,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        sp_axis = parallel_config.sequence_parallel.mesh_axis
        self.flow_map = flow_map
        self.depth = depth
        self.depth_single = depth_single
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._tp_axis = tp_axis
        self._sp_axis = sp_axis
        self._sp_factor = parallel_config.sequence_parallel.factor

        # Input projections (dit.py:167-168), bias-free: zero pad rows in, zero pad rows out. ColParallelLinear on a
        # TP-replicated input runs the per-device matmul and leaves the output column-fractured.
        self.txt_in = ColParallelLinear(
            FLUX3_KLEIN_IN_CHANNELS["text"],
            FLUX3_KLEIN_HIDDEN,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )
        self.emb_in_image = ColParallelLinear(
            FLUX3_KLEIN_IN_CHANNELS["image"],
            FLUX3_KLEIN_HIDDEN,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )
        self.emb_in_image_cond = ColParallelLinear(
            FLUX3_KLEIN_IN_CHANNELS["image_cond"],
            FLUX3_KLEIN_HIDDEN,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )
        self.modulation = Flux3KleinModulation(
            flow_map=flow_map, device=device, ccl_manager=ccl_manager, parallel_config=parallel_config
        )
        self.early = Flux3KleinEarlyStacks(
            depth=depth, device=device, ccl_manager=ccl_manager, parallel_config=parallel_config, is_fsdp=is_fsdp
        )
        self.single_blocks = ModuleList(
            Flux3KleinTransformerBlock(
                device=device, ccl_manager=ccl_manager, parallel_config=parallel_config, is_fsdp=is_fsdp
            )
            for _ in range(depth_single)
        )
        # Final layer (dit.py:128-136, decision 8): LN without affine, (1 + scale) * LN(x) + shift as dynamic affine,
        # TP all-gather, then one replicated 3072 -> 128 matmul (fp32 accumulation over the whole K).
        self.norm_out = DistributedLayerNorm(
            FLUX3_KLEIN_HIDDEN,
            norm_eps=FLUX3_KLEIN_LAYERNORM_EPS,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=device,
            ccl_manager=ccl_manager,
        )
        self.proj_out = Linear(FLUX3_KLEIN_HIDDEN, FLUX3_KLEIN_IN_CHANNELS["image"], bias=False, mesh_device=device)

    # --- loading ----------------------------------------------------------------------------------------------------

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Prefix routing of the reference ``DiT.state_dict()`` onto the children (decision 10); each child then
        handles the reference naming of its own part. Keys are moved only when present."""
        for src, dst in _STATE_PREFIXES:
            rename_substate(state, src, dst)

    # --- per-request preparation ------------------------------------------------------------------------------------

    @property
    def sp_factor(self) -> int:
        """The mesh's sequence-parallel factor, which every ``Flux3KleinSequenceLayout`` of this model must carry."""
        return self._sp_factor

    def sequence_layout(self, lengths: dict[str, int]) -> Flux3KleinSequenceLayout:
        """The row layout of one sample with real lengths ``lengths`` (all three modes; 0 = empty) on this mesh."""
        return Flux3KleinSequenceLayout.from_lengths(lengths, self._sp_factor)

    def prepare_plan(self, layout: Flux3KleinSequenceLayout, ids: dict[str, torch.Tensor]) -> Flux3KleinSequencePlan:
        """Upload the step-invariant constants of ``layout``: per non-empty mode the RoPE tables of ``ids[m]``
        (``[L_m, 4]``, ``packing.image_ids`` / ``reference_ids`` / ``text_ids``) at ``padded[m]`` and the early
        modulation gather indices, the single-stage RoPE of the ``[image | image_cond | text]`` ids at
        ``total_padded`` and its gather indices, the assembly map and the image-layout output map."""
        if layout.sp_factor != self._sp_factor:
            raise ValueError(f"layout built for SP {layout.sp_factor}, the mesh runs SP {self._sp_factor}")
        if layout.lengths["image"] == 0:
            raise ValueError("a sample always has image rows")
        device, sp_axis = self._device, self._sp_axis
        upload_indices = lambda idx: indices_to_device(idx, device=device, sp_axis=sp_axis)  # noqa: E731
        early_rope, early_mod_indices = {}, {}
        for mode in layout.modes:
            length = layout.lengths[mode]
            if mode not in ids or ids[mode].shape[0] < length:
                raise ValueError(f"{mode}: ids for {length} rows are required")
            cos, sin = rope_cos_sin(ids[mode][:length], layout.padded[mode])
            early_rope[mode] = rope_to_device(cos, sin, device=device, sp_axis=sp_axis)
            early_mod_indices[mode] = tuple(
                upload_indices(idx) for idx in modulation_table_indices(layout.mode_tail_ids(mode), "early")
            )
        cos, sin = rope_cos_sin(layout.sequence_ids(ids), layout.total_padded)
        return Flux3KleinSequencePlan(
            layout=layout,
            early_rope=early_rope,
            early_mod_indices=early_mod_indices,
            single_rope=rope_to_device(cos, sin, device=device, sp_axis=sp_axis),
            single_mod_indices=tuple(
                upload_indices(idx) for idx in modulation_table_indices(layout.sequence_mode_ids(), "single")
            ),
            assembly_indices=upload_indices(layout.assembly_indices()),
            image_out_indices=upload_indices(layout.image_out_indices()),
        )

    def embed_text(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """``txt_in``: ``[1, L_txt_pad, 20480]`` bf16 ``[None, sp, None]`` (zero pad rows) -> ``[1, L_txt_pad, 3072]``
        ``[None, sp, tp]``. Once per request."""
        return self.txt_in(tokens, parallel_config=self._parallel_config)

    def embed_image_cond(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """``emb_in.image_cond``: ``[1, L_cond_pad, 128]`` bf16 ``[None, sp, None]`` -> ``[1, L_cond_pad, 3072]``
        ``[None, sp, tp]``. Once per request."""
        return self.emb_in_image_cond(tokens, parallel_config=self._parallel_config)

    # --- one step ---------------------------------------------------------------------------------------------------

    def forward(
        self,
        *,
        image_tokens: ttnn.Tensor,
        text_embedded: ttnn.Tensor | None,
        image_cond_embedded: ttnn.Tensor | None,
        t_embed: ttnn.Tensor,
        t_ref_embed: ttnn.Tensor | None,
        plan: Flux3KleinSequencePlan,
        trace: Callable[[str, ttnn.Tensor], None] | None = None,
    ) -> ttnn.Tensor:
        """One DiT evaluation; returns the velocity ``[1, L_img_pad, 128]`` bf16 ``[None, sp, None]`` in the image
        layout (see the module docstring). ``trace(name, tensor)`` (debug only) is called with the early-stack
        output of every mode, the assembled sequence and every single block's output; it must read synchronously
        and must not retain the tensor."""
        layout = plan.layout
        inputs = {"image": image_tokens, "image_cond": image_cond_embedded, "text": text_embedded}
        for mode in FLUX3_KLEIN_MODES:
            if (inputs[mode] is None) != (layout.lengths[mode] == 0):
                raise ValueError(f"{mode}: the plan has {layout.lengths[mode]} rows but the input is {inputs[mode]}")

        # Modulation of this step: the 60-row table (per-token sources of both stages) and the final layer's rows.
        table, (final_shift, final_scale_plus_one) = self.modulation(t_embed, t_ref_embed)

        # Early stage: image projection per step; text / image_cond arrive projected; per-mode stacks.
        x = {mode: inputs[mode] for mode in layout.modes}
        x["image"] = self.emb_in_image(image_tokens, parallel_config=self._parallel_config)
        for mode in layout.modes:
            mod = tuple(gather_modulation(table, idx) for idx in plan.early_mod_indices[mode])
            x[mode] = self.early(
                mode, x[mode], rope=plan.early_rope[mode], mod=mod, logical_length=layout.lengths[mode]
            )
            if trace is not None:
                trace(f"early.{mode}", x[mode])

        # Single stage: [image | image_cond | text | pad] in the reference order, per-token modulation.
        h = assemble_single_stage(
            [x[mode] for mode in layout.modes],
            plan.assembly_indices,
            ccl_manager=self._ccl_manager,
            sp_axis=self._sp_axis,
        )
        for mode in layout.modes:
            if x[mode] is not inputs[mode]:  # the stacks' outputs are ours; a depth-0 stack hands back the input
                ttnn.deallocate(x[mode])
        if trace is not None:
            trace("assembled", h)
        mod = tuple(gather_modulation(table, idx) for idx in plan.single_mod_indices)
        for i, block in enumerate(self.single_blocks):
            h = block(h, rope=plan.single_rope, mod=mod, logical_length=layout.total)
            if trace is not None:
                trace(f"single.{i}", h)

        # Final layer on every row of the shard (row-local), then the image rows back in the image layout.
        y = ttnn.squeeze(
            self.norm_out(ttnn.unsqueeze(h, 0), dynamic_weight=final_scale_plus_one, dynamic_bias=final_shift), 0
        )
        y = self._ccl_manager.all_gather(y, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=False)
        y = self.proj_out(y)
        return gather_rows_over_sp(y, plan.image_out_indices, ccl_manager=self._ccl_manager, sp_axis=self._sp_axis)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein DiT (diffusers ``Flux2Transformer2DModel``, ``transformer_flux2.py:1226-1434``; BFL ``Flux2``,
``model.py:115-168``) on a tt_dit mesh: one denoising step.

The reference forward and where each piece lives here::

    temb       = time_guidance_embed(timestep * 1000)              (:1284-1288)   -> time (Flux2KleinTimeModulation)
    mods       = *_modulation(temb)                                (:1290-1293)   -> time
    x, c       = x_embedder(latents), context_embedder(text)       (:1318-1319)   -> x_embedder, embed_prompt (hoisted)
    rope       = pos_embed(img_ids), pos_embed(txt_ids)            (:1322-1332)   -> prepare_plan (rope_flux2_klein)
    c, x       = transformer_blocks[0..N)(x, c, mods, rope)        (:1352-1373)   -> transformer_blocks (TransformerBlock)
    h          = cat([c, x]); single_transformer_blocks[0..S)(h)   (:1378-1414)   -> ttnn.concat + single_transformer_blocks
    velocity   = proj_out(norm_out(h[image rows], temb))           (:1420-1424)   -> ttnn.slice + norm_out + proj_out

Built from main's own FLUX.2 parts (``transformer_flux2.py`` is untouched): the dual ``TransformerBlock``
(``blocks/transformer_block_opt.py``) exactly as ``transformer_flux2.py:235-254`` constructs it, the merged 17-chunk
modulation (``time_flux2_klein``), and ``Flux2KleinSingleBlock`` (``block_flux2_klein``) on ONE sequence per device,
the per-device concatenation ``[prompt_d | spatial_d]`` (the FLUX 3 Klein single stage; main's block concatenates
``[spatial_d | prompt_d]``, which would put the spatial pad rows in the middle of the gathered sequence). Three things
the Klein checkpoint changes against ``Flux2Transformer``: no guidance embedder (``guidance_embeds`` false,
``:1016-1021``), the spatial stream may carry tile-alignment rows at its global tail (every 16-px canvas; masked
through ``logical_n``), and the sinusoid angle is computed in fp32 with only the sinusoid rounded to bf16
(``embeddings.py:56-73``, ``:1025``). Every ``Attention`` instance (dual and single) takes the exact profile of
``block_flux2_klein.apply_exact_attention_profile``.

Single stage (one sequence): after the dual blocks ``h = ttnn.concat([prompt, x], dim=1)`` once per step, per device
``[P / SP | L_pad / SP]`` rows, so the ring-gathered order is ``[p_0 s_0 ... p_{SP-1} s_{SP-1}]`` and the pad rows
(the tail of the last spatial shard) are the global tail: ``logical_n = P + L_real`` masks exactly them. The one
modulation of the single blocks applies to every row (the reference modulates the concatenation). RoPE arrives as
ONE table in the same per-device order (``plan.single_rope``, ``rope_flux2_klein.single_stream_rope``). After the
last block ``ttnn.slice`` drops the ``P / SP`` prompt rows of every shard (row-local); the final layer runs on the
spatial rows as before. Requires the prompt SP-sharded (``shard_prompt``, the pipeline default) or SP 1: with a
replicated prompt the concatenation would repeat the prompt on every shard.

Tensor contract (``mesh_axes`` as ``utils.tensor.from_torch`` / ``to_torch`` take them; ``sp`` / ``tp`` are the
sequence- and tensor-parallel mesh axes of ``parallel_config``; P = 512 prompt rows):

* ``spatial``: ``[1, L_pad / SP, 128]`` bf16 ``mesh_axes=[None, sp, None]``, rows ``[image | references | pad]``
  (``pipeline_flux2_klein.py:846-848`` ``cat([latents, image_latents], dim=1)``), pad rows ``[L_real, L_pad)`` exactly
  zero, ``L_pad = packing.padded_length(L_real, SP)`` (a multiple of ``32 * SP``).
* ``prompt_embedded`` = :meth:`Flux2KleinTransformer.embed_prompt` of the ``[1, P / SP, joint_dim]`` bf16
  ``[None, sp, None]`` text rows -> ``[1, P / SP, D / TP]`` ``[None, sp, tp]`` (replicated ``[1, P, D / TP]`` without
  ``shard_prompt``): the ``context_embedder`` does not depend on the timestep, so a request runs it ONCE
  (``transformer_flux2.py:362-364``).
* ``t_embed``: fp32 ``[1, 1]`` replicated, the reference's bf16-rounded ``timestep * 1000``
  (``pipelines/flux2_klein/scheduler.py`` ``embedding_t``).
* ``plan``: a :class:`Flux2KleinPlan` from :meth:`plan_for` (cached) or :meth:`prepare_plan` (fresh), the
  step-invariant constants of one sequence layout (the dual blocks' two RoPE tables, the single blocks' merged table
  and the three logical lengths); reused by every step and by both CFG forwards of the same layout. The tables depend
  only on the layout and the ids (the spatial ids on ``(h, w, reference grids)``, the text ids are constant), so the
  model keeps the uploaded plans in :class:`Flux2KleinPlanCache` (``FLUX2_KLEIN_PLAN_CACHE_SIZE`` entries, LRU,
  keyed by the layout and the sha256 of the ids' bytes) and the prompt pair per ``(prompt rows, text ids)``, shared
  by every plan: a request of a seen canvas rebuilds and uploads nothing (before: ~2.5 ms host at 1K, 8.5 ms at 2K
  with three references, plus two fp32 ``[1, 1, L_pad, 128]`` uploads per request). A cached plan belongs to the
  model (never deallocate it; :meth:`clear_plan_cache` frees them all); a plan from :meth:`prepare_plan` belongs to
  the caller.
* Output: the velocity ``[1, L_pad / SP, 128]`` bf16 ``[None, sp, None]`` in the input's own layout, a FRESH buffer
  (``norm_out`` -> TP all-gather -> replicated ``Linear(D, 128)``, the FLUX 3 Klein final layer), so the denoiser may
  hold it across the CFG pair; rows beyond the image rows (references, pads) are discarded by the Euler mask
  (``pipeline_flux2_klein.py:862`` slices them off on the host).

Rules the dataflow relies on: pad KEY rows are masked by the ring SDPA through ``spatial_length = L_real`` (dual
blocks) and ``single_length = P + L_real`` (single blocks); pad QUERY rows produce finite garbage the caller
discards; on Ring every block's output aliases a CCLManager reduce-scatter ping-pong buffer of its own shard shape
(``[L_pad / SP]`` and ``[P / SP]`` rows in the dual blocks, ``[(P + L_pad) / SP]`` in the single blocks, three
distinct keys), so the blocks only CHAIN and nothing here retains a block's direct output.

Loader: :meth:`_prepare_torch_state` is ``transformer_flux2.py:309-343`` with the time path routed into ``time``
(``time_guidance_embed.timestep_embedder -> time.time_in``, the four modulation prefixes -> ``time.<prefix>``);
each child's ``_prepare_torch_state`` handles the diffusers naming of its own part (``transformer_blocks.{i}.*`` via
``TransformerBlock``, ``single_transformer_blocks.{i}.*`` via ``Flux2KleinSingleBlock``). The checkpoint
(``transformer/diffusion_pytorch_model.safetensors``, 169 bf16 tensors for 4B) has no biases.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import ttnn

from ....blocks.transformer_block_opt import TransformerBlock
from ....layers.linear import ColParallelLinear, Linear
from ....layers.module import Module, ModuleList
from ....layers.normalization import DistributedLayerNorm
from ....pipelines.flux2_klein.packing import padded_length
from ....utils.substate import rename_substate
from .block_flux2_klein import (
    Flux2KleinSingleBlock,
    apply_exact_attention_profile,
    apply_linear_math_fidelity,
    exact_matmul_config,
)
from .config_flux2_klein import Flux2KleinConfig
from .rope_flux2_klein import rope_cos_sin, rope_to_device, single_stream_rope
from .time_flux2_klein import (
    FLUX2_KLEIN_TIME_MOD_SOURCES,
    Flux2KleinHostTime,
    Flux2KleinModulation,
    Flux2KleinTimeModulation,
)

if TYPE_CHECKING:
    from ....parallel.config import DiTGParallelConfigNoCFG
    from ....parallel.manager import CCLManager

_TILE = 32

# Uploaded plans a model keeps (LRU over distinct (layout, ids)); an entry is at most ~7.5 MB per device at SP 8
# (2048x2048 with three references: the spatial and the merged fp32 tables of 28672 rows).
FLUX2_KLEIN_PLAN_CACHE_SIZE = 16


@dataclass(frozen=True)
class Flux2KleinSequenceLayout:
    """Row layout of ONE request at sequence-parallel factor ``sp_factor`` (the denoiser's layout key): ``image``
    latent rows, ``cond`` reference rows, ``real = image + cond``, ``padded = padded_length(real, sp_factor)`` and the
    ``prompt`` rows (512, no pad)."""

    image: int
    cond: int
    real: int
    padded: int
    prompt: int
    sp_factor: int

    def __post_init__(self) -> None:
        if self.sp_factor < 1:
            raise ValueError(f"sp_factor must be positive, got {self.sp_factor}")
        if self.image < 1 or self.cond < 0:
            raise ValueError(
                f"a request has image rows and a non-negative reference count, got {self.image}, {self.cond}"
            )
        if self.real != self.image + self.cond:
            raise ValueError(f"real {self.real} != image {self.image} + cond {self.cond}")
        if self.padded < self.real or self.padded % (_TILE * self.sp_factor):
            raise ValueError(
                f"padded {self.padded} must cover {self.real} rows in multiples of {_TILE * self.sp_factor}"
            )
        if self.prompt < 1 or self.prompt % _TILE:
            raise ValueError(f"prompt rows {self.prompt} must be a positive multiple of {_TILE}")

    @property
    def rows(self) -> int:
        """Spatial rows per device."""
        return self.padded // self.sp_factor

    @property
    def single_length(self) -> int:
        """Real rows of the single blocks' one sequence: ``prompt + real`` (its ``logical_n``)."""
        return self.prompt + self.real


@dataclass(frozen=True)
class Flux2KleinPlan:
    """Step-invariant device constants of ONE sequence layout (see :meth:`Flux2KleinTransformer.prepare_plan`):
    the spatial RoPE ``(cos, sin)`` fp32 ``[1, 1, L_pad / SP, 128]`` SP-sharded, the prompt RoPE ``[1, 1, P / SP, 128]``
    SP-sharded (``[1, 1, P, 128]`` replicated without ``shard_prompt``), the single blocks' merged RoPE
    ``[1, 1, (P + L_pad) / SP, 128]`` SP-sharded in the per-device ``[prompt_d | spatial_d]`` order
    (``rope_flux2_klein.single_stream_rope``) and the three logical lengths."""

    spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor]
    prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor]
    single_rope: tuple[ttnn.Tensor, ttnn.Tensor]
    spatial_length: int
    prompt_length: int
    single_length: int

    def tree(self) -> dict:
        """The device tensors of the plan as nested dicts / tuples (a Tracer-valid input; the lengths stay on the host)."""
        return {
            "spatial_rope": tuple(self.spatial_rope),
            "prompt_rope": tuple(self.prompt_rope),
            "single_rope": tuple(self.single_rope),
        }

    @classmethod
    def from_tree(cls, layout: Flux2KleinSequenceLayout, tree: dict) -> Flux2KleinPlan:
        """Inverse of :meth:`tree` for ``layout``."""
        return cls(
            spatial_rope=tuple(tree["spatial_rope"]),
            prompt_rope=tuple(tree["prompt_rope"]),
            single_rope=tuple(tree["single_rope"]),
            spatial_length=layout.real,
            prompt_length=layout.prompt,
            single_length=layout.single_length,
        )

    def deallocate(self, *, prompt: bool = True) -> None:
        """Free the plan's device tensors (a plan of :meth:`Flux2KleinTransformer.prepare_plan`; never a cached one).
        ``prompt=False`` keeps the prompt pair (the cache shares it between plans)."""
        pairs = (
            (self.spatial_rope, self.prompt_rope, self.single_rope) if prompt else (self.spatial_rope, self.single_rope)
        )
        for pair in pairs:
            for t in pair:
                ttnn.deallocate(t)


def ids_digest(ids: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    """``(shape, dtype, sha256 of the bytes)`` of an ids tensor: equal digests give equal RoPE tables
    (:func:`rope_flux2_klein.rope_cos_sin` is a pure function of the values and the dtype)."""
    return (
        tuple(ids.shape),
        str(ids.dtype),
        hashlib.sha256(ids.detach().contiguous().cpu().numpy().tobytes()).hexdigest(),
    )


PlanKey = tuple[Flux2KleinSequenceLayout, tuple[tuple[int, ...], str, str], tuple[tuple[int, ...], str, str]]


def plan_key(layout: Flux2KleinSequenceLayout, spatial_ids: torch.Tensor, text_ids: torch.Tensor) -> PlanKey:
    """The cache key of a plan: the layout (which fixes ``padded`` / ``prompt`` / SP) and the two ids digests."""
    return layout, ids_digest(spatial_ids), ids_digest(text_ids)


class Flux2KleinPlanCache:
    """LRU of uploaded :class:`Flux2KleinPlan` keyed by :func:`plan_key`; ``build(layout, spatial_ids, text_ids)``
    uploads a missing plan, ``free(plan)`` frees an evicted or cleared one. :meth:`get` returns the SAME plan object
    for the same key (callers compare identity to see whether their plan changed) and bumps it to most recent.
    ``on_miss`` of :meth:`get` runs before ``build`` allocates (the denoiser releases its live traces there: a cache
    entry is a long-lived buffer, allocated only while no trace is live, and an eviction frees one, so both happen
    only on a miss, after that release)."""

    def __init__(
        self,
        build: Callable[[Flux2KleinSequenceLayout, torch.Tensor, torch.Tensor], Flux2KleinPlan],
        free: Callable[[Flux2KleinPlan], None],
        *,
        maxsize: int = FLUX2_KLEIN_PLAN_CACHE_SIZE,
    ) -> None:
        if maxsize < 1:
            raise ValueError(f"the plan cache holds at least one plan, got maxsize {maxsize}")
        self._build = build
        self._free = free
        self._maxsize = maxsize
        self._plans: OrderedDict[PlanKey, Flux2KleinPlan] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def __len__(self) -> int:
        return len(self._plans)

    def __contains__(self, key: PlanKey) -> bool:
        return key in self._plans

    def get(
        self,
        layout: Flux2KleinSequenceLayout,
        spatial_ids: torch.Tensor,
        text_ids: torch.Tensor,
        *,
        on_miss: Callable[[], None] | None = None,
    ) -> Flux2KleinPlan:
        key = plan_key(layout, spatial_ids, text_ids)
        plan = self._plans.get(key)
        if plan is not None:
            self.hits += 1
            self._plans.move_to_end(key)
            return plan
        self.misses += 1
        if on_miss is not None:
            on_miss()
        while len(self._plans) >= self._maxsize:
            _, evicted = self._plans.popitem(last=False)
            self._free(evicted)
        plan = self._build(layout, spatial_ids, text_ids)
        self._plans[key] = plan
        return plan

    def info(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._plans), "maxsize": self._maxsize}

    def clear(self) -> None:
        """Free every cached plan."""
        while self._plans:
            _, plan = self._plans.popitem(last=False)
            self._free(plan)


class Flux2KleinTransformer(Module):
    """The Klein DiT; see the module docstring for the tensor contract."""

    def __init__(
        self,
        config: Flux2KleinConfig,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTGParallelConfigNoCFG,
        shard_prompt: bool = True,
        is_fsdp: bool = False,
        sdpa_exp_approx: bool | None = None,
        math_fidelity: ttnn.MathFidelity | None = None,
        plan_cache_size: int = FLUX2_KLEIN_PLAN_CACHE_SIZE,
    ) -> None:
        """``sdpa_exp_approx``: the ring SDPA exp of every block (``None``:
        ``block_flux2_klein.FLUX2_KLEIN_SDPA_EXP_APPROX_DEFAULT``). ``math_fidelity``: the matmul fidelity of every DiT
        linear (``None``: ``block_flux2_klein.FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT``, HiFi4; HiFi2 is the opt-in); the
        ring SDPA compute stays HiFi4 either way (:func:`apply_math_fidelity` re-points a built model).
        ``plan_cache_size``: the uploaded plans :meth:`plan_for` keeps (module docstring; at least 1)."""
        super().__init__()
        # Uploaded plans by (layout, ids) and the prompt RoPE pair they share, by (prompt rows, text ids digest):
        # (host (cos, sin), device (cos, sin)). Both belong to the model (plan_for / clear_plan_cache).
        self._plan_cache = Flux2KleinPlanCache(
            self._build_plan, lambda plan: plan.deallocate(prompt=False), maxsize=plan_cache_size
        )
        self._prompt_rope: dict[
            tuple[int, tuple[tuple[int, ...], str, str]],
            tuple[tuple[torch.Tensor, torch.Tensor], tuple[ttnn.Tensor, ttnn.Tensor]],
        ] = {}
        self.register_eviction_hook(self.clear_plan_cache)
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        self.sdpa_exp_approx = sdpa_exp_approx
        self.math_fidelity = math_fidelity
        self.config = config
        self.shard_prompt = shard_prompt
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._tp_axis = tp_axis
        self._sp_axis = parallel_config.sequence_parallel.mesh_axis
        self._sp_factor = parallel_config.sequence_parallel.factor
        self._tp_factor = parallel_config.tensor_parallel.factor
        config.local_shapes(self._tp_factor)
        if not shard_prompt and self._sp_factor > 1:
            raise ValueError(
                f"the single blocks run on the per-device [prompt | spatial] concatenation, which needs the prompt "
                f"SP-sharded (shard_prompt=True) at SP {self._sp_factor}"
            )

        # Input projections (transformer_flux2.py:1182-1183), bias-free: zero pad rows in, zero pad rows out.
        # ColParallelLinear on a TP-replicated input runs the per-device matmul and leaves the output column-fractured.
        exact = exact_matmul_config(device, math_fidelity=math_fidelity)
        self.x_embedder = ColParallelLinear(
            config.in_channels,
            config.hidden,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            compute_kernel_config=exact,
        )
        self.context_embedder = ColParallelLinear(
            config.joint_dim,
            config.hidden,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            compute_kernel_config=exact,
        )
        self.time = Flux2KleinTimeModulation(
            config, device=device, ccl_manager=ccl_manager, parallel_config=parallel_config
        )
        # The reference time path on the host (bit-exact bf16 torch); attached by the weight loader, which has the
        # checkpoint. ``forward`` takes its chunks through ``mod``; without it the device ``time`` module runs.
        self.host_time: Flux2KleinHostTime | None = None

        # Dual blocks exactly as transformer_flux2.py:235-254 builds them (blocks/transformer_block_opt.py unchanged).
        self.transformer_blocks = ModuleList(
            TransformerBlock(
                dim=config.hidden,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                context_pre_only=False,
                attention_proj_bias=False,
                ff_activation_fn="swiglu",
                ff_mult=config.mlp_ratio,
                ff_bias=False,
                time_norm_affine=False,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=None,
                mesh_device=device,
                is_fsdp=is_fsdp,
                shard_prompt=shard_prompt,
            )
            for _ in range(config.num_layers)
        )
        for block in self.transformer_blocks:
            apply_exact_attention_profile(
                block.attn, device, sdpa_exp_approx=sdpa_exp_approx, math_fidelity=math_fidelity
            )
            # The shared block's feed-forward config (HiFi2, fp32 acc, packer L1 acc): the DiT linear fidelity.
            block.ff_compute_kernel_config = exact_matmul_config(
                device, packer_l1_acc=True, math_fidelity=math_fidelity
            )

        self.single_transformer_blocks = ModuleList(
            Flux2KleinSingleBlock(
                config,
                device=device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
                shard_prompt=shard_prompt,
                sdpa_exp_approx=sdpa_exp_approx,
                math_fidelity=math_fidelity,
            )
            for _ in range(config.num_single_layers)
        )

        # Final layer (transformer_flux2.py:1216-1219): LN without affine, (1 + scale) * LN(x) + shift as the dynamic
        # affine, TP all-gather, then one replicated D -> 128 matmul (fp32 accumulation over the whole K).
        self.norm_out = DistributedLayerNorm(
            config.hidden,
            norm_eps=config.eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=device,
            ccl_manager=ccl_manager,
        )
        self.proj_out = Linear(
            config.hidden, config.out_channels, bias=False, mesh_device=device, compute_kernel_config=exact
        )

    def apply_math_fidelity(self, math_fidelity: ttnn.MathFidelity | None = None) -> None:
        """Re-point every DiT linear (the constructor's sites: ``x_embedder``, ``context_embedder``, the dual blocks'
        projections and feed-forward, the single blocks, the final ``proj_out``) at ``math_fidelity`` (``None``: the
        module default); the ring SDPA compute stays HiFi4. Plain attributes, so valid on a built model: the next
        render recompiles those matmuls (the fidelity A/B swaps it in place in one process)."""
        self.math_fidelity = math_fidelity
        exact = exact_matmul_config(self._device, math_fidelity=math_fidelity)
        self.x_embedder.compute_config = exact
        self.context_embedder.compute_config = exact
        for block in self.transformer_blocks:
            apply_linear_math_fidelity(block.attn, self._device, math_fidelity)
            block.ff_compute_kernel_config = exact_matmul_config(
                self._device, packer_l1_acc=True, math_fidelity=math_fidelity
            )
        for block in self.single_transformer_blocks:
            block.apply_math_fidelity(math_fidelity)
        self.proj_out.compute_config = exact

    # --- loading ----------------------------------------------------------------------------------------------------

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """``transformer_flux2.py:309-343`` with the time path routed into ``time``; keys are moved only when present
        so a strict load reports what is missing under the diffusers names."""
        rename_substate(state, "time_guidance_embed.timestep_embedder", "time.time_in")
        for prefix, _, _ in FLUX2_KLEIN_TIME_MOD_SOURCES:
            rename_substate(state, prefix, f"time.{prefix}")

        for i in range(len(self.transformer_blocks)):
            prefix = f"transformer_blocks.{i}."

            # swiglu implementation uses different ordering (Flux2SwiGLU gates the FIRST half, transformer_flux2.py:295-298)
            w = state.pop(f"{prefix}ff.linear_in.weight", None)
            if w is not None:
                state[f"{prefix}ff.net.0.proj.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)
            w = state.pop(f"{prefix}ff_context.linear_in.weight", None)
            if w is not None:
                state[f"{prefix}ff_context.net.0.proj.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)

            rename_substate(state, f"{prefix}ff.linear_out", f"{prefix}ff.net.2")
            rename_substate(state, f"{prefix}ff_context.linear_out", f"{prefix}ff_context.net.2")

    # --- per-request preparation ------------------------------------------------------------------------------------

    @property
    def sp_factor(self) -> int:
        """The mesh's sequence-parallel factor, which every layout of this model must carry."""
        return self._sp_factor

    @property
    def tp_factor(self) -> int:
        """The mesh's tensor-parallel factor (the weight cache key's TP)."""
        return self._tp_factor

    def sequence_layout(
        self, image_rows: int, cond_rows: int, prompt_rows: int | None = None
    ) -> Flux2KleinSequenceLayout:
        """The row layout of one request with ``image_rows`` latent rows and ``cond_rows`` reference rows on this mesh
        (``prompt_rows`` defaults to the config's 512)."""
        prompt = self.config.text_length if prompt_rows is None else prompt_rows
        real = image_rows + cond_rows
        return Flux2KleinSequenceLayout(
            image=image_rows,
            cond=cond_rows,
            real=real,
            padded=padded_length(real, self._sp_factor),
            prompt=prompt,
            sp_factor=self._sp_factor,
        )

    def _check_plan_inputs(
        self, layout: Flux2KleinSequenceLayout, spatial_ids: torch.Tensor, text_ids: torch.Tensor
    ) -> None:
        if layout.sp_factor != self._sp_factor:
            raise ValueError(f"layout built for SP {layout.sp_factor}, the mesh runs SP {self._sp_factor}")
        if self.shard_prompt and layout.prompt % (_TILE * self._sp_factor):
            raise ValueError(f"prompt rows {layout.prompt} do not shard over SP {self._sp_factor} in tile rows")
        if spatial_ids.shape[0] != layout.real:
            raise ValueError(f"spatial ids {tuple(spatial_ids.shape)} do not cover the {layout.real} real rows")
        if text_ids.shape[0] != layout.prompt:
            raise ValueError(f"text ids {tuple(text_ids.shape)} do not cover the {layout.prompt} prompt rows")

    def _prompt_rope_pair(
        self, prompt_rows: int, text_ids: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[ttnn.Tensor, ttnn.Tensor]]:
        """The prompt RoPE ``(host (cos, sin), device (cos, sin))`` of ``text_ids`` at ``prompt_rows``: a fresh host
        build and upload (``[1, 1, P, 128]`` fp32, SP-sharded under ``shard_prompt``)."""
        host = rope_cos_sin(text_ids, prompt_rows, theta=self.config.rope_theta, axes_dim=self.config.axes_dim)
        device = rope_to_device(*host, device=self._device, sp_axis=self._sp_axis, sharded=self.shard_prompt)
        return host, device

    def _shared_prompt_rope_pair(
        self, prompt_rows: int, text_ids: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[ttnn.Tensor, ttnn.Tensor]]:
        """The model's prompt pair of ``(prompt_rows, text_ids)``, uploaded on first use (the text ids are the
        constant ``(0, 0, 0, l)`` rows in practice: one pair per model; freed by :meth:`clear_plan_cache`)."""
        key = (prompt_rows, ids_digest(text_ids))
        pair = self._prompt_rope.get(key)
        if pair is None:
            pair = self._prompt_rope[key] = self._prompt_rope_pair(prompt_rows, text_ids)
        return pair

    def _build_plan(
        self,
        layout: Flux2KleinSequenceLayout,
        spatial_ids: torch.Tensor,
        text_ids: torch.Tensor,
        prompt: tuple[tuple[torch.Tensor, torch.Tensor], tuple[ttnn.Tensor, ttnn.Tensor]] | None = None,
    ) -> Flux2KleinPlan:
        """Host build + upload of the plan (:meth:`prepare_plan`'s contract) around ``prompt`` = the prompt pair to
        use (the model's shared pair when ``None``: the cache's builder)."""
        self._check_plan_inputs(layout, spatial_ids, text_ids)
        prompt_host, prompt_rope = self._shared_prompt_rope_pair(layout.prompt, text_ids) if prompt is None else prompt
        spatial_host = rope_cos_sin(
            spatial_ids, layout.padded, theta=self.config.rope_theta, axes_dim=self.config.axes_dim
        )
        spatial_rope = rope_to_device(*spatial_host, device=self._device, sp_axis=self._sp_axis, sharded=True)
        # The single blocks' one sequence: shard d of the upload is [prompt rows of shard d | spatial rows of shard d],
        # the order of the per-device ttnn.concat in forward (a replicated prompt only exists at SP 1, where the
        # single shard is the whole prompt).
        single_host = single_stream_rope(prompt_host, spatial_host, sp_factor=self._sp_factor)
        single_rope = rope_to_device(*single_host, device=self._device, sp_axis=self._sp_axis, sharded=True)
        return Flux2KleinPlan(
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            single_rope=single_rope,
            spatial_length=layout.real,
            prompt_length=layout.prompt,
            single_length=layout.single_length,
        )

    def prepare_plan(
        self, layout: Flux2KleinSequenceLayout, spatial_ids: torch.Tensor, text_ids: torch.Tensor
    ) -> Flux2KleinPlan:
        """Upload the step-invariant constants of ``layout`` as a FRESH plan the caller owns (``deallocate()`` frees
        it; :meth:`plan_for` is the cached form): the RoPE tables of ``spatial_ids`` ``[L_real, 4]`` (the
        ``[latent_ids | reference_ids]`` rows, ``pipeline_flux2_klein.py:287-367``) at ``layout.padded`` and of
        ``text_ids`` ``[P, 4]`` (``:267-283``) at ``layout.prompt`` for the dual blocks, and their per-device
        ``[prompt_d | spatial_d]`` interleave for the single blocks (built on the host from the same two tables, so
        every row's angle is the one its id gives)."""
        return self._build_plan(layout, spatial_ids, text_ids, prompt=self._prompt_rope_pair(layout.prompt, text_ids))

    def plan_for(
        self,
        layout: Flux2KleinSequenceLayout,
        spatial_ids: torch.Tensor,
        text_ids: torch.Tensor,
        *,
        on_miss: Callable[[], None] | None = None,
    ) -> Flux2KleinPlan:
        """The plan of ``(layout, spatial_ids, text_ids)`` from the model's cache (:class:`Flux2KleinPlanCache`;
        :meth:`prepare_plan`'s contract, built and uploaded on the first request of the key and handed out as the
        SAME object afterwards). The plan belongs to the model: never deallocate it. ``on_miss`` runs before a miss
        allocates (and before the eviction that makes room)."""
        self._check_plan_inputs(layout, spatial_ids, text_ids)
        return self._plan_cache.get(layout, spatial_ids, text_ids, on_miss=on_miss)

    def plan_cache_info(self) -> dict[str, int]:
        """``hits`` / ``misses`` / ``size`` / ``maxsize`` of the plan cache and ``prompt_pairs`` it shares."""
        return {**self._plan_cache.info(), "prompt_pairs": len(self._prompt_rope)}

    def clear_plan_cache(self) -> None:
        """Free every cached plan and the shared prompt pairs (any plan handed out by :meth:`plan_for` is dead)."""
        self._plan_cache.clear()
        while self._prompt_rope:
            _, (_, device_pair) = self._prompt_rope.popitem()
            for t in device_pair:
                ttnn.deallocate(t)

    def embed_prompt(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """``context_embedder``: ``[1, P / SP, joint_dim]`` bf16 ``[None, sp, None]`` (``[1, P, joint_dim]`` replicated
        without ``shard_prompt``) -> ``[1, P / SP, D / TP]`` ``[None, sp, tp]``. Once per request."""
        return self.context_embedder(tokens, parallel_config=self._parallel_config)

    # --- one step ---------------------------------------------------------------------------------------------------

    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt_embedded: ttnn.Tensor,
        plan: Flux2KleinPlan,
        t_embed: ttnn.Tensor | None = None,
        mod: Flux2KleinModulation | None = None,
    ) -> ttnn.Tensor:
        """One DiT evaluation; returns the velocity ``[1, L_pad / SP, 128]`` bf16 ``[None, sp, None]`` in the input's
        layout (see the module docstring). The step's modulation comes either precomputed as ``mod`` (the denoiser's
        host-exact chunks, ``time_flux2_klein.Flux2KleinHostTime``) or from ``t_embed`` through the device ``time``
        module."""
        if (mod is None) == (t_embed is None):
            raise ValueError("pass exactly one of mod (precomputed chunks) and t_embed (device time path)")
        if spatial.shape[1] != plan.spatial_rope[0].shape[2]:
            raise ValueError(
                f"spatial rows per device {spatial.shape[1]} do not match the plan's {plan.spatial_rope[0].shape[2]}"
            )
        if prompt_embedded.shape[1] != plan.prompt_rope[0].shape[2]:
            raise ValueError(
                f"prompt rows per device {prompt_embedded.shape[1]} do not match the plan's {plan.prompt_rope[0].shape[2]}"
            )
        prompt_rows = prompt_embedded.shape[1]
        if plan.single_rope[0].shape[2] != prompt_rows + spatial.shape[1]:
            raise ValueError(
                f"single-stage rows per device {plan.single_rope[0].shape[2]} != prompt {prompt_rows} + spatial "
                f"{spatial.shape[1]}"
            )

        if mod is None:
            mod = self.time(t_embed)

        x = self.x_embedder(spatial, parallel_config=self._parallel_config)
        prompt = prompt_embedded

        for block in self.transformer_blocks:
            x, prompt = block.forward(
                spatial=x,
                prompt=prompt,
                time_embed=mod.time_embed,
                spatial_rope=plan.spatial_rope,
                prompt_rope=plan.prompt_rope,
                temb_mod_params_img=mod.img,
                temb_mod_params_txt=mod.txt,
                spatial_sequence_length=plan.spatial_length,
                prompt_sequence_length=plan.prompt_length,
                skip_time_embed_activation_fn=True,
            )

        # Single stage on ONE sequence per device, [prompt_d | spatial_d] (module docstring): one concat per step,
        # then every block runs its norm / to_qkv / SDPA / proj_mlp / fused proj_out once over all rows.
        h = ttnn.concat([prompt, x], dim=1)
        for block in self.single_transformer_blocks:
            h = block(h, mod=mod.single, rope=plan.single_rope, logical_length=plan.single_length)

        # The spatial rows of every shard back out (row-local, tile-aligned: P / SP is a multiple of 32).
        x = ttnn.slice(h, [0, prompt_rows, 0], [1, h.shape[1], h.shape[2]])

        # Final layer on every spatial row of the shard (row-local).
        y = ttnn.squeeze(
            self.norm_out(
                ttnn.unsqueeze(x, 0), dynamic_weight=mod.norm_out_scale_plus_one, dynamic_bias=mod.norm_out_shift
            ),
            0,
        )
        y = self._ccl_manager.all_gather(y, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=False)
        return self.proj_out(y)

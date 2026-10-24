# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein time path and the merged modulation heads.

Reference (diffusers ``transformer_flux2.py``): ``temb = time_guidance_embed(timestep * 1000, guidance=None)``
(``:1284-1288``) = ``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)`` -> ``.to(timestep.dtype)`` ->
``TimestepEmbedding`` ``linear_1 (256 -> D) -> SiLU -> linear_2 (D -> D)``, no biases, no guidance embedder for Klein
(``:1011-1033``, ``guidance_embeds`` false); then ``double_stream_modulation_img`` / ``_txt`` (``Linear(D, 6 D)``,
chunks ``(shift, scale, gate) x (attn, ff)``), ``single_stream_modulation`` (``Linear(D, 3 D)``), each ``linear(silu(temb))``
(``:1036-1056``, ``:1290-1293``), and the final ``AdaLayerNormContinuous`` ``scale, shift = linear(silu(temb)).chunk(2)``
(``normalization.py:346-351``, SCALE first). BFL (``model.py:115-134``, ``:683-731``) is the same math with the
``MLPEmbedder`` and ``Modulation`` names.

Device form (the exact profile): the host supplies ``embedding_t`` = the reference's bf16-rounded ``timestep * 1000``
(``pipelines/flux2_klein/scheduler.py``) as a replicated fp32 ``[1, 1]``; ``Timesteps`` (fp32, cos first, 256 channels,
max period 10000) computes the sinusoid in fp32 like ``get_timestep_embedding`` (``embeddings.py:56-73``), ``ttnn.typecast``
to bf16 is the reference's ``.to(timestep.dtype)`` (``:1025``), and ``TimestepEmbedding`` with bf16 weights (HiFi2,
fp32 accumulation) is the reference's bf16 GEMMs. The four modulation heads are one ``ColParallelLinear(D, 17 D,
chunks=17)`` in :data:`FLUX2_KLEIN_TIME_MOD_SOURCES` order with a synthesized bias of 1.0 on every scale chunk, the
``transformer_flux2.py:33-39, 273-280, 309-327`` convention: the chunks come out directly in the ``(shift, 1 + scale,
gate)`` form ``blocks/transformer_block_opt.py:284-319`` and the single block consume, ONE rounding ``bf16(acc + 1)``
instead of the reference's ``bf16(bf16(acc) + 1)`` (the Flux2 convention; always the value closer to ``acc + 1``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F

import ttnn

from ....layers.embeddings import TimestepEmbedding, Timesteps
from ....layers.linear import ColParallelLinear, prepare_chunked_linear_output
from ....layers.module import Module

if TYPE_CHECKING:
    from ....parallel.config import DiTGParallelConfigNoCFG
    from ....parallel.manager import CCLManager
    from .config_flux2_klein import Flux2KleinConfig

FLUX2_KLEIN_TIME_MAX_PERIOD = 10000

# (diffusers key prefix, chunk count, offsets of the scale chunks within the head). Row order is the chunk order of
# the merged weight and the order `forward` unpacks (transformer_flux2.py:33-38).
FLUX2_KLEIN_TIME_MOD_SOURCES = (
    ("double_stream_modulation_img.linear", 6, (1, 4)),  # (shift, scale, gate) x attn, ff
    ("double_stream_modulation_txt.linear", 6, (1, 4)),
    ("single_stream_modulation.linear", 3, (1,)),  # (shift, scale, gate)
    ("norm_out.linear", 2, (0,)),  # (scale, shift), normalization.py:349
)
FLUX2_KLEIN_TIME_MOD_CHUNKS = sum(count for _, count, _ in FLUX2_KLEIN_TIME_MOD_SOURCES)  # 17


class Flux2KleinModulation(NamedTuple):
    """One step's modulation, every tensor bf16 ``[1, 1, D / TP]`` ``mesh_axes=[None, None, tp]``: ``img`` / ``txt``
    ``(shift_attn, scale_attn + 1, gate_attn, shift_ff, scale_ff + 1, gate_ff)`` for the dual blocks, ``single``
    ``(shift, scale + 1, gate)`` for the single blocks, the final layer's ``(scale + 1, shift)``, and ``time_embed``
    = ``silu(temb)`` bf16 ``[1, 1, D]`` replicated (the dual blocks take it positionally; with ``time_norm_affine``
    False they do not read it)."""

    img: tuple[ttnn.Tensor, ...]
    txt: tuple[ttnn.Tensor, ...]
    single: tuple[ttnn.Tensor, ...]
    norm_out_scale_plus_one: ttnn.Tensor
    norm_out_shift: ttnn.Tensor
    time_embed: ttnn.Tensor


class Flux2KleinHostModulation(NamedTuple):
    """The host twin of :class:`Flux2KleinModulation`: bf16 torch ``[D]`` chunks in the same order and form (``img`` /
    ``txt`` ``(shift_attn, 1 + scale_attn, gate_attn, shift_ff, 1 + scale_ff, gate_ff)``, ``single`` ``(shift,
    1 + scale, gate)``, the final layer's ``(1 + scale, shift)``) and ``time_embed`` = ``silu(temb)`` ``[D]``."""

    img: tuple[torch.Tensor, ...]
    txt: tuple[torch.Tensor, ...]
    single: tuple[torch.Tensor, ...]
    norm_out_scale_plus_one: torch.Tensor
    norm_out_shift: torch.Tensor
    time_embed: torch.Tensor

    def slots(self) -> dict[str, torch.Tensor]:
        """Flat ``name -> [D]`` view (the denoiser's slot names)."""
        out = {f"img{i}": t for i, t in enumerate(self.img)}
        out.update({f"txt{i}": t for i, t in enumerate(self.txt)})
        out.update({f"single{i}": t for i, t in enumerate(self.single)})
        out["out_scale"] = self.norm_out_scale_plus_one
        out["out_shift"] = self.norm_out_shift
        out["time_embed"] = self.time_embed
        return out


# The six checkpoint tensors of the time path (diffusers names).
FLUX2_KLEIN_TIME_WEIGHT_KEYS = (
    "time_guidance_embed.timestep_embedder.linear_1.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight",
    *(f"{prefix}.weight" for prefix, _, _ in FLUX2_KLEIN_TIME_MOD_SOURCES),
)


class Flux2KleinHostTime:
    """The reference time path on the host, op for op in bf16 torch, so the per-step modulation is BIT-EQUAL to the
    diffusers model's (``transformer_flux2.py`` ``Flux2TimestepGuidanceEmbeddings`` / ``Flux2Modulation`` /
    ``AdaLayerNormContinuous`` on a bf16 CPU model): the fp32 sinusoid (``embeddings.py get_timestep_embedding``,
    ``flip_sin_to_cos`` -> cos first, shift 0) of the bf16 ``timestep * 1000`` -> ``.to(bf16)`` -> ``linear_1`` ->
    ``SiLU`` -> ``linear_2`` (``temb``) -> per head ``linear(silu(temb))`` -> chunks, and the consumers' ``1 + scale``
    as the reference's bf16 add (``bf16(1 + bf16(acc))``). Why the host: the device ``Flux2KleinTimeModulation``
    reproduces the sinusoid exactly but its three bf16 GEMMs (HiFi2 at the time of measurement) left every chunk
    0.7-2.6 % rel_rmse off the reference, a per-channel error applied to every token in every block; the host chunks
    cost 17 ``[D, D]`` bf16 matvecs per distinct timestep (memoized below) and travel as 17 + 1 small bf16 tensors
    uploaded ONCE per distinct timestep per process by the denoiser's ``prepare`` into its ``_mod_cache`` (shared by
    every layout key; eager steps read that cache and upload nothing; a traced step refreshes the 17 + 1 slots by
    ``ttnn.copy``; ``denoise_flux2_klein``).
    Measured 2026-10-23 (job 146747, 4B 1024x1024 step 0, whole DiT vs diffusers bf16 CPU): velocity rel_rmse
    0.0517 -> 0.0452 from this alone, 0.0433 -> 0.0328 on top of HiFi4 matmuls + HiFi4 SDPA. ``state`` holds
    :data:`FLUX2_KLEIN_TIME_WEIGHT_KEYS` as bf16 ``[out, in]`` tensors (``weights_dit_flux2_klein.load_time_torch_state``).
    """

    def __init__(self, state: dict[str, torch.Tensor], config: Flux2KleinConfig) -> None:
        missing = [k for k in FLUX2_KLEIN_TIME_WEIGHT_KEYS if k not in state]
        if missing:
            raise KeyError(f"time path weights missing: {missing}")
        bf16 = torch.bfloat16
        self.hidden = config.hidden
        self.time_channels = config.time_channels
        self.w1 = state[FLUX2_KLEIN_TIME_WEIGHT_KEYS[0]].to(bf16).contiguous()  # [D, 256]
        self.w2 = state[FLUX2_KLEIN_TIME_WEIGHT_KEYS[1]].to(bf16).contiguous()  # [D, D]
        self.heads = tuple(
            (state[f"{prefix}.weight"].to(bf16).contiguous(), count, scale_offsets)
            for prefix, count, scale_offsets in FLUX2_KLEIN_TIME_MOD_SOURCES
        )
        if self.w1.shape != (self.hidden, self.time_channels) or self.w2.shape != (self.hidden, self.hidden):
            raise ValueError(
                f"time MLP shapes {tuple(self.w1.shape)} / {tuple(self.w2.shape)} do not match {config.size}"
            )
        for weight, count, _ in self.heads:
            if weight.shape != (count * self.hidden, self.hidden):
                raise ValueError(
                    f"modulation head {tuple(weight.shape)} is not [{count} x {self.hidden}, {self.hidden}]"
                )
        # One result per distinct timestep: a schedule has num_steps of them (CFG shares), so the matvecs run once per
        # step per process; the first call also pages the weights in, hence the warm-up by the loader.
        self._memo: dict[float, Flux2KleinHostModulation] = {}

    def sinusoid(self, embedding_t: float) -> torch.Tensor:
        """``get_timestep_embedding(bf16 timestep, 256, flip_sin_to_cos=True, downscale_freq_shift=0)`` -> bf16 ``[256]``
        (``embeddings.py:56-73``, ``transformer_flux2.py`` ``time_proj`` + ``.to(timestep.dtype)``)."""
        half = self.time_channels // 2
        exponent = -math.log(FLUX2_KLEIN_TIME_MAX_PERIOD) * torch.arange(0, half, dtype=torch.float32) / half
        t = torch.tensor([embedding_t], dtype=torch.bfloat16)  # the reference's bf16 timestep * 1000
        emb = t[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = torch.cat([emb[:, half:], emb[:, :half]], dim=-1)  # flip_sin_to_cos: cos first
        return emb.to(torch.bfloat16)[0]

    def temb(self, embedding_t: float) -> torch.Tensor:
        """``timestep_embedder``: ``linear_2(silu(linear_1(sinusoid)))`` in bf16, ``[D]``."""
        with torch.inference_mode():
            h = F.linear(self.sinusoid(embedding_t)[None], self.w1)
            return F.linear(F.silu(h), self.w2)[0]

    def __call__(self, embedding_t: float) -> Flux2KleinHostModulation:
        key = float(embedding_t)
        hit = self._memo.get(key)
        if hit is None:
            hit = self._memo[key] = self._compute(key)
        return hit

    def _compute(self, embedding_t: float) -> Flux2KleinHostModulation:
        with torch.inference_mode():
            temb = self.temb(embedding_t)
            act = F.silu(temb)[None]  # Flux2Modulation.act_fn / AdaLayerNormContinuous.silu, once per head
            chunks: list[torch.Tensor] = []
            for weight, count, scale_offsets in self.heads:
                raw = F.linear(act, weight)[0].chunk(count, dim=-1)
                chunks.extend((1 + c) if i in scale_offsets else c for i, c in enumerate(raw))  # bf16(1 + bf16(acc))
            chunks = [c.contiguous() for c in chunks]
        return Flux2KleinHostModulation(
            img=tuple(chunks[0:6]),
            txt=tuple(chunks[6:12]),
            single=tuple(chunks[12:15]),
            norm_out_scale_plus_one=chunks[15],
            norm_out_shift=chunks[16],
            time_embed=act[0].contiguous(),
        )


class Flux2KleinTimeModulation(Module):
    """``time_guidance_embed`` + the four modulation heads of one Klein DiT (see the module docstring).

    Parameters (after :meth:`_prepare_torch_state`): ``time_in.linear_1.weight`` ``[256, D]``, ``time_in.linear_2.weight``
    ``[D, D]``, ``mod.weight`` (TP-fractured, chunk-major per device) and ``mod.bias``. The diffusers keys of this part
    arrive as ``time_in.linear_1 / linear_2`` (renamed by the transformer from ``time_guidance_embed.timestep_embedder``)
    and the four ``<prefix>.weight`` of :data:`FLUX2_KLEIN_TIME_MOD_SOURCES`.
    """

    def __init__(
        self,
        config: Flux2KleinConfig,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTGParallelConfigNoCFG,
    ) -> None:
        super().__init__()
        self._hidden = config.hidden
        self._parallel_config = parallel_config
        self._tp_factor = parallel_config.tensor_parallel.factor

        self.time_proj = Timesteps(
            num_channels=config.time_channels,
            cos_first=True,
            max_period=FLUX2_KLEIN_TIME_MAX_PERIOD,
            dtype=ttnn.float32,
            mesh_device=device,
        )
        self.time_in = TimestepEmbedding(
            config.time_channels,
            config.hidden,
            act_fn="silu",
            dtype=ttnn.bfloat16,
            bias=False,
            mesh_device=device,
        )
        self.mod = ColParallelLinear(
            config.hidden,
            FLUX2_KLEIN_TIME_MOD_CHUNKS * config.hidden,
            bias=True,
            mesh_device=device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            chunks=FLUX2_KLEIN_TIME_MOD_CHUNKS,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Merge the four modulation projections into `mod` by stacking their rows in FLUX2_KLEIN_TIME_MOD_SOURCES
        # order (transformer_flux2.py:309-327). Never a partial merge: with any head missing, mod.weight / mod.bias
        # stay absent so the strict loader reports them.
        weights = [state.pop(f"{prefix}.weight", None) for prefix, _, _ in FLUX2_KLEIN_TIME_MOD_SOURCES]
        if any(state.pop(f"{prefix}.bias", None) is not None for prefix, _, _ in FLUX2_KLEIN_TIME_MOD_SOURCES):
            raise ValueError("the merged modulation projection assumes bias-free heads")
        if all(w is not None for w in weights):
            state["mod.weight"] = torch.cat(weights, dim=0)
            # The bias carries the consumers' `+1` on each scale chunk.
            state["mod.bias"] = torch.cat(
                [
                    torch.full([self._hidden], 1.0 if i in scale_offsets else 0.0)
                    for _, count, scale_offsets in FLUX2_KLEIN_TIME_MOD_SOURCES
                    for i in range(count)
                ]
            )
        prepare_chunked_linear_output(
            state, prefix="mod", device_count=self._tp_factor, chunks=FLUX2_KLEIN_TIME_MOD_CHUNKS
        )

    def sinusoid_bf16(self, t_embed: ttnn.Tensor) -> ttnn.Tensor:
        """``time_proj(timestep).to(dtype)`` of the reference: bf16 ``[1, 1, 1, 256]`` (cos first) for the replicated
        fp32 ``[1, 1]`` ``embedding_t``; the angle is computed in fp32 like ``get_timestep_embedding``."""
        return ttnn.typecast(self.time_proj(t_embed), ttnn.bfloat16)

    def forward(self, t_embed: ttnn.Tensor) -> Flux2KleinModulation:
        """The modulation of one step from the replicated fp32 ``[1, 1]`` ``embedding_t``."""
        temb = self.time_in(self.sinusoid_bf16(t_embed))
        # silu(temb) once for every head (transformer_flux2.py:1041-1042 per head; :376-378 shared).
        time_embed = ttnn.reshape(ttnn.silu(temb), (1, 1, self._hidden))
        chunks = self.mod(time_embed, parallel_config=self._parallel_config, dtype=ttnn.bfloat16)
        return Flux2KleinModulation(
            img=tuple(chunks[0:6]),
            txt=tuple(chunks[6:12]),
            single=tuple(chunks[12:15]),
            norm_out_scale_plus_one=chunks[15],
            norm_out_shift=chunks[16],
            time_embed=time_embed,
        )

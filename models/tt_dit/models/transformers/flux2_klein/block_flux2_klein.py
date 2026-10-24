# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""One FLUX.2 Klein single-stream block on ONE sequence, and the exact attention profile every DiT attention takes.

Reference: diffusers ``Flux2SingleTransformerBlock`` (``transformer_flux2.py:807-873``) over ``Flux2ParallelSelfAttention``
(``:723-788``) with ``Flux2ParallelSelfAttnProcessor`` (``:572-630``); BFL ``SingleStreamBlock`` (``model.py:437-521``)::

    x_mod = (1 + scale) * LN(x, eps=1e-6, no affine) + shift                    # :853-856
    q, k, v, mlp = to_qkv_mlp_proj(x_mod).split([D, D, D, 2 * 3 D])             # :587-596
    q, k = rope(rmsnorm(q)), rope(rmsnorm(k))                                     # per head over 128, :604-609
    attn = sdpa(q, k, v)                                                          # full attention, no mask
    mlp = silu(mlp[:, :3 D]) * mlp[:, 3 D:]                                       # Flux2SwiGLU, :295-298
    x = x + gate * to_out(cat([attn, mlp]))                                       # :627-628, :865

The composition is ``Flux2SingleTransformerBlock`` of ``transformer_flux2.py:63-142`` (the same ``Attention`` with
``pre_only`` / ``use_spatial_weights_for_prompt`` / ``per_head_norm``, ``DistributedLayerNorm`` without affine,
``ColParallelLinear(activation_fn="swiglu")`` and ``RowParallelLinear`` over the ``[attn | mlp]`` concatenation, the same
``_prepare_torch_state``), and the forward is the FLUX 3 Klein block's (``transformer_block_flux3_klein.py``
``Flux3KleinTransformerBlock.forward``): ONE sequence per device, so one norm, one ``to_qkv``, one q-norm + RoPE, one
k-norm + RoPE, one v head-split, one ring SDPA with no joint stream, one ``concatenate_heads``, one ``proj_mlp`` and
one fused ``proj_out`` + reduce-scatter + addcmul per block. The reference runs the block on ``cat([txt, img])``
(``:1378``); the DiT (``transformer_flux2_klein``) hands this block the per-device concatenation ``[prompt_d |
spatial_d]`` so that the spatial pad rows ``[L_real, L_pad)`` of the last shard stay the GLOBAL tail of the
ring-gathered sequence, which the ring SDPA masks through ``logical_n = logical_length = P + L_real``
(``attention_opt.py:615-646``). Attention is permutation-equivariant, so the order ``[prompt | spatial]`` versus the
reference's ``[txt, img]`` changes nothing beyond the fp32 accumulation order. (The first FLUX.2 port kept the two
streams apart through ``Attention.forward(sequence_2=prompt)``, which cost a second ``to_qkv``, two added norms, two
prompt K/V all-gathers, a second ``concatenate_heads``, a second ``proj_mlp`` and a second ``proj_out`` per block;
main's block concatenates ``[spatial_d | prompt_d]``, which puts the pad rows mid-sequence.)

Tensor contract (SP / TP the sequence- and tensor-parallel factors of ``parallel_config``):

* ``x``: ``[1, (P + L_pad) / SP, D / TP]`` bf16 ``mesh_axes=[None, sp, tp]``, per device ``[prompt_d | image |
  references | pad]`` (the pad rows only on the last shard), P = 512 prompt rows SP-sharded. Output: same shape and
  distribution, so blocks chain without a relayout.
* ``rope``: fp32 ``(cos, sin)`` ``[1, 1, (P + L_pad) / SP, 128]`` SP-sharded in the same per-device order
  (``rope_flux2_klein.single_stream_rope``); ``logical_length = P + L_real`` masks the tail's KEY rows.
* ``mod = (shift, scale_plus_one, gate)``, bf16 ``[1, 1, D / TP]`` ``mesh_axes=[None, None, tp]`` (``time_flux2_klein``),
  one row for every token (the reference modulates the concatenation with the single-block modulation).
* Pad QUERY rows produce finite garbage the caller discards (the denoiser's Euler mask keeps the state's pad rows
  zero); no pad row is ever a key.
* Residual: on Ring the fused matmul + reduce-scatter + addcmul (``RowParallelLinear.forward_fused_addcmul``,
  ``transformer_flux2.py:192-194``), whose output is one of the CCLManager's two alternating reduce-scatter
  ping-pong buffers for the ``[(P + L_pad) / SP, D / TP]`` shard shape; chaining the same-shape blocks alternates
  them exactly as main's single block does. The dual blocks' two streams use the ``[L_pad / SP]`` and ``[P / SP]``
  shapes, so no buffer is shared across the two stages.

:func:`apply_exact_attention_profile` is the FLUX 3 Klein profile (``transformer_block_flux3_klein.py:138-167``)
applied to every ``Attention`` of the DiT (dual and single blocks): fp32-accumulating ring SDPA (the shared default
accumulates in bf16, ``attention_opt.py:150-155``), the approximate SFPU exp inside that SDPA by default (the deviation the
2026-10-21 decision approved and the 2026-10-23 user decision kept; the precise exp is the ``sdpa_exp_approx=False`` opt-in; A/B numbers at
:data:`FLUX2_KLEIN_SDPA_EXP_APPROX_DEFAULT`) and the q chunk clamped to 128 because the fp32 accumulation enlarges
the kernel's circular buffers beyond L1 at the shared (256, 512) fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn

from ....blocks.attention_opt import Attention
from ....layers.linear import ColParallelLinear, RowParallelLinear
from ....layers.module import Module
from ....layers.normalization import DistributedLayerNorm
from ....pipelines.flux2_klein.packing import FLUX2_KLEIN_TEXT_LENGTH
from ....utils.tensor import prepare_weight_for_concatenated_input
from .config_flux2_klein import FLUX2_KLEIN_SDPA_Q_CHUNK_MAX

if TYPE_CHECKING:
    from ....parallel.config import DiTGParallelConfigNoCFG
    from ....parallel.manager import CCLManager
    from .config_flux2_klein import Flux2KleinConfig


# The matmul fidelity of every DiT linear with bf16 weights (x_embedder, context_embedder, to_qkv / add_qkv_proj,
# to_out / to_add_out, the dual feed-forward, proj_mlp, proj_out, the final proj_out) when nothing is passed: HiFi4
# (all mantissa bits of BOTH bf16 operands, 4 FPU passes per product). HiFi2 (2 passes: the low mantissa bits of one
# operand are dropped) is the layers/linear.py default FLUX 3's exact profile runs on and the opt-in here
# (``FLUX2_KLEIN_DIT_FIDELITY=hifi2`` / ``Flux2KleinPipeline(dit_math_fidelity=ttnn.MathFidelity.HiFi2)``). Measured
# 2026-10-23 (jobs 146727 / 146747, 4B 1024x1024, teacher-forced step 0 vs diffusers bf16 on CPU): HiFi2 -> HiFi4 takes
# context_embedder rel_rmse 0.0031 -> 0.0011 and the whole-DiT step-0 velocity 0.0433 -> 0.0328 (with the SDPA at
# HiFi4 as well). The ring SDPA's own compute (QK^T, PV) is not under this knob: it stays HiFi4
# (:data:`FLUX2_KLEIN_SDPA_MATH_FIDELITY`). The A/B of the two fidelities (same box, same process, warm eager) is in
# the README's exact-profile table.
FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT = ttnn.MathFidelity.HiFi4
# The ring SDPA compute (QK^T, PV): HiFi4 took the dual-block SDPA output from rel_rmse 0.0193 to 0.0144 vs the CPU
# reference (2026-10-23, job 146747); the kernel is bound by the exp and the fp32 accumulation, not by matmul fidelity
# (FLUX 3 ledger 2026-10-21: LoFi changed nothing), so there is nothing to trade.
FLUX2_KLEIN_SDPA_MATH_FIDELITY = ttnn.MathFidelity.HiFi4


def exact_matmul_config(
    device: ttnn.MeshDevice, *, packer_l1_acc: bool = False, math_fidelity: ttnn.MathFidelity | None = None
):
    """The matmul compute config of the exact profile: ``math_fidelity`` (``None``:
    :data:`FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT`, HiFi4; HiFi2 is the opt-in), exact SFPU math, fp32 destination
    accumulation. Measured 2026-10-23 on the 4B DiT at 1024x1024, teacher-forced step 0 vs diffusers bf16 on CPU:
    context_embedder rel_rmse 0.0031 -> 0.0011; whole-DiT velocity 0.0517 -> 0.0509 alone, and 0.0433 -> 0.0328 when
    the SDPA's matmuls go HiFi4 too (``apply_exact_attention_profile``). ``packer_l1_acc`` is kept where the shared
    block had it (the dual block's feed-forward)."""
    if math_fidelity is None:
        math_fidelity = FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )


def apply_linear_math_fidelity(
    attn: Attention, device: ttnn.MeshDevice, math_fidelity: ttnn.MathFidelity | None = None
) -> None:
    """The matmul fidelity of one ``Attention``'s projections: ``to_qkv`` / ``add_qkv_proj`` run on the attention's
    own matmul config (HiFi2, approx SFPU mode, packer L1 accumulation in the shared default; here exact SFPU math and
    ``math_fidelity``), ``to_out`` / ``to_add_out`` (the dual block's ColParallelLinear with the fused gated residual;
    None / shared on the single block) on the plain exact config. The SDPA compute is not touched. The attributes are
    plain, so this also re-points a built model (the fidelity A/B swaps it in place; the next render recompiles the
    matmuls)."""
    attn.mm_compute_kernel_config = exact_matmul_config(device, packer_l1_acc=True, math_fidelity=math_fidelity)
    for proj in (attn.to_out, attn.to_add_out):
        if proj is not None and getattr(proj, "compute_config", None) is not None:
            proj.compute_config = exact_matmul_config(device, math_fidelity=math_fidelity)


# The DiT ring SDPA's softmax exp: True = the SFPU's approximate exp (the default: the user's 2026-10-23 23:3x
# decision 'the approx looks good enough' re-approved the 2026-10-21 deviation for this port), False = the precise exp
# (``FLUX2_KLEIN_SDPA_EXP_APPROX=0`` or ``Flux2KleinPipeline(sdpa_exp_approx=False)``). A/B 2026-10-23, 4B man_dog 1024 seed 7 4 steps eager sp8tp4, jobs
# 146815 (approx) / 146817 (precise), same box: teacher-forced velocity rel_rmse 0.033/0.079/0.052/0.042 ->
# 0.032/0.054/0.046/0.037, golden-text e2e 28.84 -> 31.02 dB, device-text e2e 25.26 -> 26.79 dB; warm denoise
# 1024x1024 0.524 -> 0.520 s (no cost), 2048x2048 0.590 -> 0.713 s (+21 %); 4096x4096 not measured here (FLUX 3's
# kernel measurement was 2.6x at 4K), so the approximate exp stays one switch away for the large canvases.
FLUX2_KLEIN_SDPA_EXP_APPROX_DEFAULT = True


def apply_exact_attention_profile(
    attn: Attention,
    device: ttnn.MeshDevice,
    *,
    sdpa_exp_approx: bool | None = None,
    math_fidelity: ttnn.MathFidelity | None = None,
) -> None:
    """The exact profile of one ``Attention`` instance (``transformer_block_flux3_klein.py:142-167``, with the SDPA
    matmul fidelity raised to HiFi4 and the projections on ``math_fidelity``, ``None``:
    :data:`FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT`): the attributes are plain, overwritten after
    ``Attention.__init__``; the shared class tables are untouched. ``sdpa_exp_approx`` picks the ring SDPA's exp
    (``None``: :data:`FLUX2_KLEIN_SDPA_EXP_APPROX_DEFAULT`)."""
    if sdpa_exp_approx is None:
        sdpa_exp_approx = FLUX2_KLEIN_SDPA_EXP_APPROX_DEFAULT
    # Exact profile: fp32 accumulation everywhere. Attention's SDPA config is the one shared default that
    # accumulates in bf16 (attention_opt.py:150-155, "Set to True if there's a correctness issue"); it is a
    # plain attribute, overwritten here the way pipelines/wan/quant_config.py:252 does. HiFi4 for the QK^T and PV
    # matmuls: the ring SDPA is bound by the exp and the fp32 accumulation, not by matmul fidelity (FLUX 3 ledger
    # 2026-10-21: LoFi changed nothing), and HiFi4 took the dual-block SDPA output from rel_rmse 0.0193 to 0.0144
    # (approx exp) vs the CPU reference (2026-10-23, job 146747).
    attn.sdpa_compute_kernel_config = exact_matmul_config(device, math_fidelity=FLUX2_KLEIN_SDPA_MATH_FIDELITY)
    # The projections (to_qkv / add_qkv_proj, to_out / to_add_out) on the DiT linear fidelity.
    apply_linear_math_fidelity(attn, device, math_fidelity)
    # The softmax exponential of the ring SDPA: approximate by default (FLUX2_KLEIN_SDPA_EXP_APPROX_DEFAULT, numbers
    # there); the precise exp is the opt-in. Measured 2026-10-21 on FLUX 3 Klein (eager, 16 steps): 4096x4096 step
    # 1.87 -> 0.72 s, 2048x2048 0.182 -> 0.139 s approximate vs precise; the kernel is bound by that exp and the fp32
    # destination accumulation, not by matmul fidelity. bf16 accumulation (30.27 dB) was rejected, so the fp32
    # accumulation stays.
    attn.ring_sdpa_exp_approx = bool(sdpa_exp_approx)
    # fp32 accumulation enlarges the ring SDPA's intermediate circular buffers: with the shared (4, 8) table's
    # (256, 512) fallback chunks the kernel's static CBs measured 1 645 760 B against the 1 572 864 B L1 for
    # L = 80 / 320 / 4496 (the swept 4096 entry, (128, 256), and the (2, 2) table's (128, 512) fit). This
    # instance therefore gets its own copy of the table with every q chunk clamped to 128; the shared class
    # tables are untouched. Correctness-safe, perf untuned.
    attn.ring_sdpa_chunk_size_map = {
        key: {length: (min(q, FLUX2_KLEIN_SDPA_Q_CHUNK_MAX), k) for length, (q, k) in table.items()}
        for key, table in Attention.ring_sdpa_chunk_size_map.items()
    }
    # The single blocks key the table on the merged length P + L_real (one sequence per block). The shared (4, 8)
    # table already carries the merged 1K / 2K entries (4608, (4096 + 128) * 4); the other layouts' swept spatial
    # entries are extended by the 512 prompt rows so a merged length keeps its canvas's chunking instead of
    # falling to the default.
    for table in attn.ring_sdpa_chunk_size_map.values():
        for length, chunks in list(table.items()):
            if length > 0:
                table.setdefault(length + FLUX2_KLEIN_TEXT_LENGTH, chunks)
    attn.default_ring_sdpa_chunk_size = {
        length: (min(q, FLUX2_KLEIN_SDPA_Q_CHUNK_MAX), k)
        for length, (q, k) in Attention.default_ring_sdpa_chunk_size.items()
    }


class Flux2KleinSingleBlock(Module):
    """``Flux2SingleTransformerBlock`` on one sequence; see the module docstring for the tensor contract.
    ``shard_prompt`` is the ``Attention`` flag of the dual blocks' contract; this block never passes a joint stream,
    so it has no effect here."""

    def __init__(
        self,
        config: Flux2KleinConfig,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTGParallelConfigNoCFG,
        is_fsdp: bool = False,
        shard_prompt: bool = True,
        sdpa_exp_approx: bool | None = None,
        math_fidelity: ttnn.MathFidelity | None = None,
    ) -> None:
        """``math_fidelity``: the matmul fidelity of ``to_qkv``, ``proj_mlp`` and ``proj_out`` (``None``:
        :data:`FLUX2_KLEIN_DIT_MATH_FIDELITY_DEFAULT`); the ring SDPA compute stays HiFi4."""
        super().__init__()

        dim = config.hidden
        mlp_hidden_dim = config.mlp_hidden
        self.math_fidelity = math_fidelity
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        # 24 / 32 heads divide every TP factor in use, so no head padding (config.local_shapes).
        self.attn = Attention(
            query_dim=dim,
            head_dim=config.head_dim,
            heads=config.num_heads,
            out_dim=dim,
            added_kv_proj_dim=0,
            pre_only=True,
            proj_bias=False,
            eps=config.eps,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=None,
            use_spatial_weights_for_prompt=True,
            per_head_norm=True,
            is_fsdp=is_fsdp,
            shard_prompt=shard_prompt,
        )
        apply_exact_attention_profile(self.attn, device, sdpa_exp_approx=sdpa_exp_approx, math_fidelity=math_fidelity)

        # No affine parameters: the modulation arrives as the dynamic weight / bias at forward.
        self.norm = DistributedLayerNorm(
            dim,
            norm_eps=config.eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=device,
            ccl_manager=ccl_manager,
        )

        # Shard output, since size of input dimension << size of output dimension. The swiglu flag doubles
        # out_features to 2 * mlp_hidden_dim and fuses silu(gate) * up into the matmul.
        self.proj_mlp = ColParallelLinear(
            dim,
            mlp_hidden_dim,
            bias=False,
            activation_fn="swiglu",
            mesh_device=device,
            mesh_axis=tp_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
            compute_kernel_config=exact_matmul_config(device, math_fidelity=math_fidelity),
        )

        # Shard input, since size of input dimension >> size of output dimension: one matmul over the
        # concatenated [attn | mlp] input is the reference's to_out over cat([attn, mlp]).
        self.proj_out = RowParallelLinear(
            dim + mlp_hidden_dim,
            dim,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
            compute_kernel_config=exact_matmul_config(device, math_fidelity=math_fidelity),
        )

        self._dim = dim
        self._mlp_hidden_dim = mlp_hidden_dim
        self._tp_factor = parallel_config.tensor_parallel.factor
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._mesh_device = device

    def apply_math_fidelity(self, math_fidelity: ttnn.MathFidelity | None = None) -> None:
        """Re-point ``to_qkv``, ``proj_mlp`` and ``proj_out`` at ``math_fidelity`` (``None``: the module default);
        the SDPA compute is untouched. Plain attributes: valid on a built block, the next forward recompiles."""
        self.math_fidelity = math_fidelity
        apply_linear_math_fidelity(self.attn, self._mesh_device, math_fidelity)
        self.proj_mlp.compute_config = exact_matmul_config(self._mesh_device, math_fidelity=math_fidelity)
        self.proj_out.compute_config = exact_matmul_config(self._mesh_device, math_fidelity=math_fidelity)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """``transformer_flux2.py:125-142``: the fused ``attn.to_qkv_mlp_proj`` rows ``[q | k | v | gate | up]`` become
        ``attn.to_q / to_k / to_v`` (head-interleaved per TP shard by ``Attention``) and ``proj_mlp`` rolled by half
        (the reference's ``[gate | up]`` is tt_dit's ``[up | gate]``); ``attn.to_out`` ``[D, D + 3 D]`` is re-laid so
        device d holds ``[attn cols_d | mlp cols_d]`` for the ``[attn, mlp]`` list input. Keys are popped only when
        present so a strict load reports what is missing."""
        qkv_mlp_weight = state.pop("attn.to_qkv_mlp_proj.weight", None)
        if qkv_mlp_weight is not None:
            state["attn.to_q.weight"] = qkv_mlp_weight[: self._dim]
            state["attn.to_k.weight"] = qkv_mlp_weight[self._dim : 2 * self._dim]
            state["attn.to_v.weight"] = qkv_mlp_weight[2 * self._dim : 3 * self._dim]

            w = qkv_mlp_weight[3 * self._dim :]
            # swiglu implementation uses different ordering
            state["proj_mlp.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)

        proj_out_weight = state.pop("attn.to_out.weight", None)
        if proj_out_weight is not None:
            state["proj_out.weight"] = prepare_weight_for_concatenated_input(
                proj_out_weight,
                [self._dim, self._mlp_hidden_dim],
                device_count=self._tp_factor,
            )

    def _mlp(self, x_norm: ttnn.Tensor) -> ttnn.Tensor:
        """``silu(gate) * up`` fused in the matmul: ``[1, rows, 3 D / TP]``."""
        return self.proj_mlp(
            x_norm,
            parallel_config=self._parallel_config,
            use_heuristic_mmcfg=True,
            core_grid=Attention.get_core_grid(x_norm.shape[-2], self._mesh_device.compute_with_storage_grid_size()),
        )

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        mod: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
        rope: tuple[ttnn.Tensor, ttnn.Tensor],
        logical_length: int,
    ) -> ttnn.Tensor:
        """``x`` ``[1, (P + L_pad) / SP, D / TP]`` -> same shape and distribution. See the module docstring for
        ``rope``, ``mod`` (``shift, scale_plus_one, gate``) and ``logical_length`` (the real row count, masks the
        tail)."""
        shift, scale_plus_one, gate = mod

        # (1 + scale) * LN(x) + shift, the affine applied in-op in fp32 (normalization.py:397-437).
        x_norm = ttnn.squeeze(self.norm(ttnn.unsqueeze(x, 0), dynamic_weight=scale_plus_one, dynamic_bias=shift), 0)

        # to_qkv -> per-head RMSNorm + RoPE on q and k -> ring SDPA over the real rows -> concatenated heads,
        # [1, (P + L_pad) / SP, D / TP] (attention_opt.py:496-687; no joint stream).
        attn, _ = self.attn.forward(
            sequence_1=x_norm,
            sequence_1_rope=rope,
            sequence_1_length=logical_length,
        )

        # silu(gate) * up fused in the matmul: [1, (P + L_pad) / SP, 3 D / TP].
        mlp = self._mlp(x_norm)

        # x + gate * ([attn | mlp] @ to_out^T): the gated residual fused into the reduce-scatter's final write on
        # Ring, a separate addcmul after the reduce-scatter on Linear (transformer_flux2.py:192-194).
        if self._ccl_manager.topology == ttnn.Topology.Ring:
            return self.proj_out.forward_fused_addcmul([attn, mlp], x, gate, scalar=1.0)
        return ttnn.addcmul(x, self.proj_out([attn, mlp]), gate)

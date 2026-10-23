# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""One FLUX 3 Image Klein transformer block (``Block`` of ``reference/flux3_klein/dit.py``) on a tt_dit mesh.

The same block serves the early per-mode stacks (``content_mode_blocks.*``, ``txt_mode_blocks``) and the shared
single-stream stack (``single_blocks``); only the modulation differs (one row per mode vs one row per token).
It is COMPOSED from the pieces ``Flux2SingleTransformerBlock`` uses (``transformer_flux2.py:63-115, 166-194``):
``Attention`` (pre_only, per-head QK RMSNorm fused with RoPE, ring SDPA), ``DistributedLayerNorm`` (dynamic affine
carries the modulation), ``ColParallelLinear(activation_fn="swiglu")`` and ``RowParallelLinear`` over the
``[attn | mlp]`` concatenation. Fidelity (the exact profile): bf16 operands, fp32 accumulation in every op. The
shared matmul and norm defaults already accumulate in fp32 (``linear.py`` ``fp32_dest_acc_en=True``,
``normalization.py:173-179``); the ring SDPA does NOT by default (``Attention`` builds it with
``fp32_dest_acc_en=False``, ``attention_opt.py:150-155``), so ``__init__`` replaces that one config with the
fp32-accumulating twin (same HiFi2 fidelity, exact exp).

Reference math reproduced (``dit.py:151-161``)::

    x_mod = (1 + scale) * LN(x, eps=1e-6, no affine) + shift
    q, k  = rmsnorm_rope(q_proj(x_mod)), rmsnorm_rope(k_proj(x_mod))     # per-head RMS over 128, scale, RoPE
    attn  = packed_attention(q, k, v_proj(x_mod))                         # full attention inside a sample
    gate_in, up = mlp_in(x_mod).chunk(2)                                  # rows [gate | up]
    out   = x + gate * (attn_out(attn) + mlp_out(silu(gate_in) * up))

Tensor contract (SP = sequence-parallel factor on ``sp_axis``, TP = tensor-parallel factor on ``tp_axis``; the
``mesh_axes`` below are ``utils.tensor.from_torch`` / ``to_torch`` arguments):

* ``x``: ``[1, L_pad, 3072]`` bf16, ``mesh_axes=[None, sp_axis, tp_axis]`` -> ``[1, L_pad/SP, 3072/TP]`` per
  device. ``L_pad = packing.padded_length(L_real, SP)`` (a multiple of ``32 * SP``). Rows are laid out per sample
  as ``[image | image_cond | text_real | pad]``; every text row the encoder produced (multiples of 80) is a REAL
  token, only the alignment rows ``[L_real, L_pad)`` are phantom. The output has the input's shape and
  distribution, so blocks chain without a relayout; the caller slices ``to_torch(out)[0, :L_real]``.
* ``logical_length = L_real``: forwarded as ``logical_n`` of the ring SDPA, which masks the phantom key rows
  ``[L_real, L_pad)`` at row granularity (``attention_opt.py:630``). Phantom QUERY rows still produce output rows
  that the caller discards: exactly zero in the per-token case below (pad rows gated with ``gate = 0``), but
  under broadcast modulation a zero pad row leaves the LN as ``shift`` and the block adds ``gate * (...)`` to
  it. The block does not re-zero those rows, so the CALLER owns them: a stack of early blocks must either zero
  the tail ``[L_real, L_pad)`` after each block or pass per-token modulation with ``gate = 0`` on the pad rows.
  This matters because the ring SDPA key mask is an additive ``-inf``: a non-finite pad key row would turn
  every real query's masked score into NaN.
* ``rope = (cos, sin)``: fp32 (or bf16) ``[1, 1, L_pad, 128]``, ``mesh_axes=[None, None, sp_axis, None]``. Built
  from the reference ``rope_embed(ids[None])`` ``(1, 1, L, 64, 2, 2)`` as ``cos = emb[..., 0, 0]``,
  ``sin = emb[..., 1, 0]``, each ``repeat_interleave(2, dim=-1)`` (channels ``2p, 2p+1`` share pair ``p``'s angle,
  pairs axis-major, no channel permutation). The fused op rotates with ``get_rot_transformation_mat`` so that
  ``out[2i] = x[2i] cos - x[2i+1] sin`` and ``out[2i+1] = x[2i+1] cos + x[2i] sin``, which is ``dit.py:69-70``.
  Pad rows carry ``cos = 1, sin = 0``.
* ``mod = (shift, scale_plus_one, gate)``, all rank 3 bf16. ``scale_plus_one`` is ``1 + scale`` in bf16; who adds
  the 1 is the caller's business. The block test folds ``(1 + scale).to(bf16)`` on host from the reference's bf16
  ``scale`` (bit-identical to the reference's own bf16 ``1.0 + scale``). The DiT (``modulation_flux3_klein.py``)
  gets it from the merged modulation matmul with the 1 as a fused bias, ONE rounding ``bf16(acc + 1)`` instead of
  the reference's two ``bf16(bf16(acc) + 1)``: not bit-identical (about a fifth of N(0, 0.5) scales differ by one
  bf16 ulp) but always the value closer to ``acc + 1`` (the Flux2 convention; decision 10's bf16-with-fp32-
  accumulation operands, no precision loss).
  - broadcast (early stacks): each ``[1, 1, 3072]``, ``mesh_axes=[None, None, tp_axis]``;
  - per token (single stack): each ``[1, L_pad, 3072]``, ``mesh_axes=[None, sp_axis, tp_axis]`` (the activation's
    own distribution). Rows are the per-run expansion of the three ``(shift, scale, gate)`` rows in
    ``[image | image_cond | text]`` order; pad rows carry ``shift = 0, scale_plus_one = 1, gate = 0`` so a
    zero-padded input row stays exactly zero through the block.
  ``shift``/``scale_plus_one`` are the fused LayerNorm's dynamic bias/weight (the op accepts a broadcast row or
  one row per token of the per-device input); ``gate`` is the third operand of ``addcmul`` (Linear) or of the
  fused matmul + reduce-scatter + addcmul (Ring), which accept ``[1, N/TP]`` or ``[M, N/TP]``.

Two CFG samples never attend to each other: run them as two forwards on the same instance (each with its own
``L_pad``, ``logical_length``, modulation slice and RoPE slice), which equals the reference ``packed_attention``
with ``bounds=[(0, L1), (L1, L1 + L2)]`` because every other op is row-local.

Perf notes (not correctness): one instance pins its ring-SDPA worker grid at the first call from that call's
``logical_length`` (``attention_opt.py:612-613``); SP must be > 1 (the non-ring branch of ``Attention.forward``
is not usable for this block, ``attention_opt.py:647-650``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn

from ....blocks.attention_opt import Attention
from ....layers.linear import ColParallelLinear, RowParallelLinear
from ....layers.module import Module
from ....layers.normalization import DistributedLayerNorm
from ....utils.tensor import prepare_weight_for_concatenated_input

if TYPE_CHECKING:
    from ....parallel.config import DiTParallelConfig
    from ....parallel.manager import CCLManager

FLUX3_KLEIN_HIDDEN = 3072
FLUX3_KLEIN_HEADS = 24
FLUX3_KLEIN_HEAD_DIM = 128
FLUX3_KLEIN_MLP_RATIO = 3
# Largest ring-SDPA q chunk whose fp32-accumulating kernel fits L1 next to k chunk 512 (see __init__).
FLUX3_KLEIN_SDPA_Q_CHUNK_MAX = 128


class Flux3KleinTransformerBlock(Module):
    """``reference/flux3_klein/dit.py::Block`` on a mesh; see the module docstring for the tensor contract."""

    def __init__(
        self,
        *,
        dim: int = FLUX3_KLEIN_HIDDEN,
        num_heads: int = FLUX3_KLEIN_HEADS,
        head_dim: int = FLUX3_KLEIN_HEAD_DIM,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        mlp_hidden_dim = FLUX3_KLEIN_MLP_RATIO * dim
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        # 24 heads divide every TP factor in use (1, 2, 4, 8), so no head padding.
        self.attn = Attention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            added_kv_proj_dim=0,
            pre_only=True,
            proj_bias=False,
            eps=1e-6,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=None,
            use_spatial_weights_for_prompt=True,
            per_head_norm=True,
            is_fsdp=is_fsdp,
            shard_prompt=False,
        )
        # Exact profile: fp32 accumulation everywhere. Attention's SDPA config is the one shared default that
        # accumulates in bf16 (attention_opt.py:150-155, "Set to True if there's a correctness issue"); it is a
        # plain attribute, overwritten here the way pipelines/wan/quant_config.py:252 does, keeping the default's
        # HiFi2 fidelity and exact exp. Measured effect at block level: see the test file's threshold comments.
        self.attn.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        # The softmax exponential of the ring SDPA uses the SFPU's approximate exp (the shared default is the precise
        # one). Measured 2026-10-21 (bin/probes/probe_sdpa_modes.py, eager, 16 steps): 4096x4096 step 1.87 -> 0.72 s
        # (end to end 31.9 -> 14.4 s), 2048x2048 0.182 -> 0.139 s; the kernel is bound by that exp and the fp32
        # destination accumulation, not by matmul fidelity (LoFi changes nothing). Exactness: man_dog 1024 with the
        # recorded text 31.45 dB vs the CPU golden (precise exp 32.86 dB); the images were inspected and accepted by
        # the user (the fp32 accumulation and HiFi2 stay; bf16 accumulation, 30.27 dB, was rejected).
        self.attn.ring_sdpa_exp_approx = True
        # fp32 accumulation enlarges the ring SDPA's intermediate circular buffers: with the shared (4, 8) table's
        # (256, 512) fallback chunks the kernel's static CBs measured 1 645 760 B against the 1 572 864 B L1 for
        # L = 80 / 320 / 4496 (the swept 4096 entry, (128, 256), and the (2, 2) table's (128, 512) fit). This
        # instance therefore gets its own copy of the table with every q chunk clamped to 128; the shared class
        # tables are untouched. Correctness-safe, perf untuned (the perf milestone re-sweeps these).
        self.attn.ring_sdpa_chunk_size_map = {
            key: {length: (min(q, FLUX3_KLEIN_SDPA_Q_CHUNK_MAX), k) for length, (q, k) in table.items()}
            for key, table in Attention.ring_sdpa_chunk_size_map.items()
        }
        self.attn.default_ring_sdpa_chunk_size = {
            length: (min(q, FLUX3_KLEIN_SDPA_Q_CHUNK_MAX), k)
            for length, (q, k) in Attention.default_ring_sdpa_chunk_size.items()
        }

        # No affine parameters: the modulation arrives as the dynamic weight/bias at forward.
        self.norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
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
        )

        # Shard input, since size of input dimension >> size of output dimension: one matmul over the
        # concatenated [attn | mlp] input replaces attn_out(attn) + mlp_out(mlp).
        self.proj_out = RowParallelLinear(
            dim + mlp_hidden_dim,
            dim,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        self._dim = dim
        self._mlp_hidden_dim = mlp_hidden_dim
        self._tp_axis = tp_axis
        self._tp_factor = parallel_config.tensor_parallel.factor
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._mesh_device = device

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Map the reference ``Block.state_dict()`` (8 keys, six separate linears + two QK-norm scales) onto the
        fused tt_dit layout. Keys are popped only when present so a strict load reports what is missing."""
        # q/k/v -> Attention, which head-interleaves them per TP shard into to_qkv (attention_opt.py:300-309,
        # 348-384) so device d's column block is [q heads_d | k heads_d | v heads_d], split in-kernel (chunks=3).
        for src, dst in (("q_proj", "attn.to_q"), ("k_proj", "attn.to_k"), ("v_proj", "attn.to_v")):
            w = state.pop(f"{src}.weight", None)
            if w is not None:
                state[f"{dst}.weight"] = w

        # Per-head RMSNorm scales [128]: Attention repeats them to [heads * 128] (attention_opt.py:343-346) and
        # the fused kernel applies x * rsqrt(mean(x^2) + eps) -> scale -> RoPE per head, the reference order.
        for src, dst in (
            ("norm.query_norm.scale", "attn.norm_q.weight"),
            ("norm.key_norm.scale", "attn.norm_k.weight"),
        ):
            w = state.pop(src, None)
            if w is not None:
                state[dst] = w

        # Reference rows are [gate | up] (gate_in, up = mlp_in(x).chunk(2)). ColParallelLinear(swiglu) runs
        # prepare_for_fused_swiglu with gate_is_first=False (linear.py:269-270), whose input contract is
        # [up | gate] (utils/tensor.py:29-34), so the halves are swapped; the same roll Flux2 does
        # (transformer_flux2.py:132-134).
        w = state.pop("mlp_in.weight", None)
        if w is not None:
            state["proj_mlp.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)

        # attn_out [3072, 3072] and mlp_out [3072, 9216] become one [3072, 12288] weight whose K rows are
        # re-ordered so device d holds [attn cols_d | mlp cols_d], matching the forward's [attn, mlp] list input
        # (utils/tensor.py:921-945; Flux2 precedent transformer_flux2.py:136-142). Never synthesize a partial
        # fused weight: with either source missing, strict loading reports proj_out.weight.
        attn_out = state.pop("attn_out.weight", None)
        mlp_out = state.pop("mlp_out.weight", None)
        if attn_out is not None and mlp_out is not None:
            state["proj_out.weight"] = prepare_weight_for_concatenated_input(
                torch.cat([attn_out, mlp_out], dim=1),
                [self._dim, self._mlp_hidden_dim],
                device_count=self._tp_factor,
            )

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        rope: tuple[ttnn.Tensor, ttnn.Tensor],
        mod: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
        logical_length: int,
    ) -> ttnn.Tensor:
        """``x`` ``[1, L_pad/SP, 3072/TP]`` -> same shape and distribution. See the module docstring for ``rope``,
        ``mod`` (``shift, scale_plus_one, gate``) and ``logical_length`` (the real row count, masks the tail)."""
        shift, scale_plus_one, gate = mod

        # (1 + scale) * LN(x) + shift, affine applied in-op in fp32 (normalization.py:397-437).
        x_norm = ttnn.squeeze(self.norm(ttnn.unsqueeze(x, 0), dynamic_weight=scale_plus_one, dynamic_bias=shift), 0)

        # to_qkv -> per-head RMSNorm + RoPE on q and k -> ring SDPA over the real rows -> concatenated heads,
        # [1, L_pad/SP, 3072/TP] (attention_opt.py:496-687).
        attn, _ = self.attn.forward(
            sequence_1=x_norm,
            sequence_1_rope=rope,
            sequence_1_length=logical_length,
        )

        # silu(gate) * up fused in the matmul: [1, L_pad/SP, 9216/TP].
        mlp = self.proj_mlp(
            x_norm,
            parallel_config=self._parallel_config,
            use_heuristic_mmcfg=True,
            core_grid=Attention.get_core_grid(x_norm.shape[-2], self._mesh_device.compute_with_storage_grid_size()),
        )

        # x + gate * ([attn | mlp] @ [attn_out ; mlp_out]^T): one fp32 accumulation over both segments, the gated
        # residual fused into the reduce-scatter's final write on Ring (linear.py:683-772) or a separate addcmul
        # after the reduce-scatter on Linear (transformer_flux2.py:192-194).
        if self._ccl_manager.topology == ttnn.Topology.Ring:
            return self.proj_out.forward_fused_addcmul([attn, mlp], x, gate, scalar=1.0)
        return ttnn.addcmul(x, self.proj_out([attn, mlp]), gate)

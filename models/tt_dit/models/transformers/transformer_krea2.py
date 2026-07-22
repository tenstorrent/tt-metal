# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""KREA-2 (Krea 2) single-stream MMDiT flow-matching transformer for tt_dit.

Faithful on-device port of
`diffusers/main src/diffusers/models/transformers/transformer_krea2.py`.

Supports both single-device (TP factor-1, replicated — the original PCC-verified
path) and multi-chip tensor-parallel (TP factor>1) execution. The 12B DiT does not
fit on one 32GB chip, so TP shards the weights across the tensor-parallel mesh axis.

Tensor-parallel design (activated when ``parallel_config.tensor_parallel.factor > 1``):

- The transformer residual stream ``x`` is kept **sharded on the hidden dim**
  (``hidden / tp`` per device), matching ``transformer_qwenimage.py`` /
  ``blocks/transformer_block.py``. Weights are sharded at load time by
  ``ColParallelLinear`` (shards the output dim) / ``RowParallelLinear`` (shards the
  input dim and reduce-scatters), so no single device ever holds a full weight.
- Norms over the (sharded) hidden dim use ``DistributedRMSNorm`` (RMS reduction across
  the TP axis via an all-gather of the partial stats). Norms over a replicated dim
  (the per-head ``head_dim`` q/k norms) use the plain ``Krea2RMSNorm``: heads are
  sharded but ``head_dim`` is whole on every device, so the per-head RMS is local.
- Attention (GQA 48q/12kv): ``to_q/to_k/to_v/to_gate`` are ``ColParallelLinear`` on
  the (all-gathered, replicated) normed input, sharding heads across TP
  (12 q-heads + 3 kv-heads per device at TP=4 — clean). SDPA runs on the local head
  slice; GQA kv-broadcast is per-device. RoPE is applied per shard (head_dim whole).
  The concat-heads output and the ``to_gate`` output are both column-fractured on
  hidden, so ``out * sigmoid(gate)`` stays elementwise per shard. ``to_out`` is
  ``RowParallelLinear`` (reduce-scatter back to the sharded hidden). Mirrors
  ``blocks/attention.py`` (ColParallel qkv + local-head SDPA + AG-before-out).
- adaLN modulation: ``temb_mod`` (6*hidden) is produced column-fractured by a
  ``ColParallelLinear`` time-mod projection and the per-block ``scale_shift_table``
  (6, hidden) is sharded on hidden, so each of the six modulation vectors is a
  sharded-hidden tensor applied elementwise to the sharded normed stream. The final
  layer's (2, hidden) table + raw ``temb`` are handled the same way.
- SwiGLU: ``gate``/``up`` ``ColParallelLinear`` (shard intermediate) on the
  all-gathered normed input; ``down`` ``RowParallelLinear`` (reduce-scatter to
  sharded hidden).
- Text fusion tower (dim 2560, MHA 20/20): TP-sharded with the same scheme
  (5 heads/device at TP=4). The projector (num_text_layers -> 1) is small and kept
  replicated.

Reference math notes preserved exactly:
- RMSNorm is "1 + weight" computed in float32 (see `Krea2RMSNorm` / `Krea2DistributedRMSNorm`).
- Attention: GQA (num_heads query, num_kv_heads key/value), all projections
  bias=False, q/k RMSNorm over head_dim BEFORE rope, rope on q/k for the image
  trunk only (text fusion passes rope=None), SDPA with kv broadcast to q heads,
  flatten heads, elementwise * sigmoid(to_gate(x)), then to_out[0].
- Block: modulation = temb_mod.unflatten(-1,(6,-1)) + scale_shift_table(6,H);
  attn_out = attn((1+prescale)*norm1(x)+preshift); x += pregate*attn_out;
  ff_out = ff((1+postscale)*norm2(x)+postshift); x += postgate*ff_out.
- SwiGLU: down(silu(gate(x)) * up(x)), bias=False.
- TimestepEmbedding: half=dim//2; freqs=exp(-log(1e4)*arange(half)/half);
  args=(t*1e3)[:,None,None]*freqs; emb=cat([cos,sin]); L2(gelu_tanh(L1(emb))).
- Transformer.forward: temb=time_embed(t); temb_mod=time_mod_proj(gelu_tanh(temb));
  ehs=txt_in(text_fusion(ehs,text_mask)); x=cat([ehs, img_in(hs)], 1);
  loop blocks(x, temb_mod, rope, mask); x=x[:, text_seq_len:];
  out=final_layer(x, temb)  # NOTE: raw temb, not temb_mod.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import ttnn

from ...blocks.attention import _apply_rope
from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedRMSNorm, RMSNorm
from ...utils import cache
from ...utils.substate import rename_substate
from ...utils.tensor import bf16_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager


# ======================================================================================
# Tensor-parallel helpers
# ======================================================================================
def _tp_factor(parallel_config: DiTParallelConfig | None) -> int:
    if parallel_config is None:
        return 1
    return parallel_config.tensor_parallel.factor


def _tp_axis(parallel_config: DiTParallelConfig | None) -> int | None:
    if parallel_config is None:
        return None
    return parallel_config.tensor_parallel.mesh_axis


# ======================================================================================
# RMSNorm ("1 + weight", fp32 compute)
# ======================================================================================
class Krea2RMSNorm(RMSNorm):
    """RMSNorm with the Krea-2 zero-centered scale convention: effective weight is
    ``1 + weight``. The reference stores a zero-initialised weight and normalises in
    float32.

    We reuse tt_dit's :class:`RMSNorm` (which fuses ``ttnn.experimental.dit_rms_norm_unary_fused``)
    and fold the ``+1`` into the loaded weight tensor so the on-device op sees the
    effective multiplier directly. Weights are kept in float32 to match the reference's
    fp32 norm precision.

    Used where the normalized tensor's reduction dim is *replicated* (not sharded across
    TP): the per-head q/k norms (over head_dim, whole on every device) and — at TP
    factor-1 — every hidden norm.
    """

    def __init__(self, embedding_dim, norm_eps=1e-5, mesh_device=None, dtype=ttnn.float32):
        super().__init__(
            embedding_dim=embedding_dim,
            norm_eps=norm_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            # Fold "1 + weight" and match fp32 reference precision.
            state["weight"] = (state["weight"].to(torch.float32) + 1.0).unsqueeze(0)


class Krea2DistributedRMSNorm(DistributedRMSNorm):
    """Krea-2 "1 + weight" RMSNorm over a hidden dim that is *sharded* across the TP
    axis. Reuses tt_dit's :class:`DistributedRMSNorm` (partial-sum stats + all-gather
    across the mesh axis) and folds the ``+1`` into the (sharded) weight, matching the
    fp32 reference convention.
    """

    def __init__(self, embedding_dim, norm_eps=1e-5, *, mesh_axis, mesh_device, ccl_manager):
        super().__init__(
            embedding_dim=embedding_dim,
            norm_eps=norm_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_axis=mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            # Fold "1 + weight"; keep as (1, embedding_dim) for the sharded Parameter.
            state["weight"] = (state["weight"].to(torch.float32) + 1.0).reshape(1, self.embedding_dim)

    def forward(self, x: ttnn.Tensor, **kwargs) -> ttnn.Tensor:
        # The underlying wan fused-rmsnorm op requires a rank-4 (1, 1, S, D) tensor
        # (asserts logical_shape[0]==logical_shape[1]==1). RMSNorm is independent per row
        # of the last (feature) dim, so flatten all leading dims into the S axis.
        orig_shape = list(x.shape)
        feat = orig_shape[-1]
        flat = ttnn.reshape(x, (1, 1, -1, feat))
        flat = super().forward(flat, **kwargs)
        return ttnn.reshape(flat, orig_shape)


def _make_hidden_norm(
    embedding_dim: int,
    *,
    norm_eps: float,
    mesh_device,
    parallel_config: DiTParallelConfig | None,
    ccl_manager: CCLManager | None,
) -> Module:
    """RMSNorm over a hidden dim that is sharded across TP when factor>1, else replicated."""
    if _tp_factor(parallel_config) > 1:
        return Krea2DistributedRMSNorm(
            embedding_dim,
            norm_eps=norm_eps,
            mesh_axis=_tp_axis(parallel_config),
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
    return Krea2RMSNorm(embedding_dim, norm_eps=norm_eps, mesh_device=mesh_device)


# ======================================================================================
# SwiGLU feed-forward
# ======================================================================================
class Krea2SwiGLU(Module):
    """SwiGLU FFN: ``down(silu(gate(x)) * up(x))``, all bias=False.

    A thin wrapper over three linear layers rather than :class:`ParallelFeedForward`,
    because the reference keeps ``gate`` and ``up`` as two separate projections (the
    fused-swiglu path in tt_dit interleaves a single doubled matmul, which would require
    re-fusing the two reference weights and does not map cleanly to the reference key
    names).

    TP: ``gate``/``up`` are ColParallel (shard the intermediate); ``down`` is
    RowParallel (reduce-scatter back to the sharded hidden). The ColParallel inputs must
    be replicated, so the caller passes an already-all-gathered ``x``.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        mesh_device=None,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.tp = _tp_factor(parallel_config) > 1

        if self.tp:
            tp_axis = _tp_axis(parallel_config)
            self.gate = ColParallelLinear(
                dim, hidden_dim, bias=False, mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl_manager
            )
            self.up = ColParallelLinear(
                dim, hidden_dim, bias=False, mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl_manager
            )
            self.down = RowParallelLinear(
                hidden_dim, dim, bias=False, mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl_manager
            )
        else:
            self.gate = Linear(dim, hidden_dim, bias=False, mesh_device=mesh_device)
            self.up = Linear(dim, hidden_dim, bias=False, mesh_device=mesh_device)
            self.down = Linear(hidden_dim, dim, bias=False, mesh_device=mesh_device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: replicated (full dim). gate/up -> sharded intermediate; down -> sharded dim.
        gate = ttnn.silu(self.gate(x))
        up = self.up(x)
        return self.down(gate * up)


# ======================================================================================
# Attention (GQA + q/k RMSNorm + rope + sigmoid output gate)
# ======================================================================================
class Krea2Attention(Module):
    """Self-attention with grouped-query projections, q/k RMSNorm (pre-rope), rotary
    embeddings and a per-channel sigmoid output gate. All projections are bias=False.

    forward(hidden_states, attention_mask=None, rope=None):
      # hidden_states is replicated (full hidden) — TP callers all-gather before calling.
      q = to_q(x) -> (B, num_heads, S, head_dim)         # heads sharded across TP
      k = to_k(x), v = to_v(x) -> (B, num_kv_heads, S, head_dim)
      gate = to_gate(x)                                   # (B, S, hidden/tp)
      q = norm_q(q); k = norm_k(k)                        # RMSNorm over head_dim (whole)
      if rope: q = rope(q); k = rope(k)
      o = sdpa(q, k, v, mask)  broadcasting kv over q heads (local heads only)
      o = concat_heads(o) * sigmoid(gate)                 # (B, S, hidden/tp)
      o = to_out(o)                                       # RowParallel -> (B, S, hidden/tp)
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
        padding_config=None,
        q_chunk_size: int = 128,
        k_chunk_size: int = 512,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.scale = self.head_dim**-0.5

        self.tp = _tp_factor(parallel_config)
        self.tp_axis = _tp_axis(parallel_config)
        if self.tp > 1:
            if self.num_heads % self.tp != 0:
                raise ValueError(f"num_heads={self.num_heads} must be divisible by tensor_parallel factor={self.tp}")
            if self.num_kv_heads % self.tp != 0:
                raise ValueError(
                    f"num_kv_heads={self.num_kv_heads} must be divisible by tensor_parallel factor={self.tp}"
                )
        self.local_num_heads = self.num_heads // self.tp
        self.local_num_kv_heads = self.num_kv_heads // self.tp

        if self.tp > 1:
            common = dict(mesh_device=mesh_device, mesh_axis=self.tp_axis, ccl_manager=ccl_manager)
            self.to_q = ColParallelLinear(hidden_size, self.head_dim * self.num_heads, bias=False, **common)
            self.to_k = ColParallelLinear(hidden_size, self.head_dim * self.num_kv_heads, bias=False, **common)
            self.to_v = ColParallelLinear(hidden_size, self.head_dim * self.num_kv_heads, bias=False, **common)
            self.to_gate = ColParallelLinear(hidden_size, hidden_size, bias=False, **common)
            self.to_out = RowParallelLinear(hidden_size, hidden_size, bias=False, **common)
        else:
            self.to_q = Linear(hidden_size, self.head_dim * self.num_heads, bias=False, mesh_device=mesh_device)
            self.to_k = Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False, mesh_device=mesh_device)
            self.to_v = Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False, mesh_device=mesh_device)
            self.to_gate = Linear(hidden_size, hidden_size, bias=False, mesh_device=mesh_device)
            self.to_out = Linear(hidden_size, hidden_size, bias=False, mesh_device=mesh_device)

        # q/k RMSNorm is over head_dim, which is whole (not sharded) on every device.
        self.norm_q = Krea2RMSNorm(self.head_dim, norm_eps=eps, mesh_device=mesh_device)
        self.norm_k = Krea2RMSNorm(self.head_dim, norm_eps=eps, mesh_device=mesh_device)

        # SDPA config. exp_approx_mode=False for correctness.
        grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )
        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Reference `to_out` is an nn.ModuleList([Linear, Dropout]); drop the "0." index
        # and the (weightless) dropout entry so keys map onto our single `to_out` Linear.
        rename_substate(state, "to_out.0", "to_out")

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> ttnn.Tensor:
        # hidden_states: (B, S, hidden) replicated (TP callers all-gather beforehand).
        B = hidden_states.shape[0]
        S = hidden_states.shape[-2]

        nh = self.local_num_heads
        nkv = self.local_num_kv_heads

        q = self.to_q(hidden_states)  # (B, S, nh*head_dim)   [sharded heads]
        k = self.to_k(hidden_states)  # (B, S, nkv*head_dim)
        v = self.to_v(hidden_states)
        gate = self.to_gate(hidden_states)  # (B, S, hidden/tp)

        # Reshape to (B, num_heads, S, head_dim). Reference unflattens the last dim into
        # (heads, head_dim) leaving (B, S, heads, head_dim), then attention treats dim 1
        # as sequence. We move heads to dim 1 for the ttnn SDPA layout [b, h, s, d].
        q = ttnn.reshape(q, (B, S, nh, self.head_dim))
        k = ttnn.reshape(k, (B, S, nkv, self.head_dim))
        v = ttnn.reshape(v, (B, S, nkv, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # q/k RMSNorm over head_dim (last dim), BEFORE rope.
        q = self.norm_q(q)
        k = self.norm_k(k)

        if rope is not None:
            q = _apply_rope(q, rope)
            k = _apply_rope(k, rope)

        # ttnn SDPA requires the mask query-seq dim (dim 2) to equal Q's sequence length
        # (it broadcasts only the batch/head dims, not the query-seq dim). Key-padding
        # masks are constant across query positions, so a (B, 1, 1, S_k) mask is expanded
        # to (B, 1, S, S_k) here. PyTorch SDPA broadcasts this dim implicitly; ttnn does not.
        if attention_mask is not None and attention_mask.shape[-2] == 1 and S != 1:
            reps = [1] * len(attention_mask.shape)
            reps[-2] = S
            attention_mask = ttnn.repeat(attention_mask, ttnn.Shape(reps))

        # GQA SDPA: q has nh heads, k/v have nkv heads -> kernel broadcasts kv (local).
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scale,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )  # (B, nh, S, head_dim)

        # Flatten heads: (B, nh, S, head_dim) -> (B, S, nh*head_dim) [= hidden/tp]
        out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(out, (B, S, nh * self.head_dim))

        out = out * ttnn.sigmoid(gate)  # elementwise per shard, hidden/tp
        return self.to_out(out)  # RowParallel reduce-scatter -> (B, S, hidden/tp)


# ======================================================================================
# Text fusion tower
# ======================================================================================
class Krea2TextFusionBlock(Module):
    """Pre-norm transformer block (no rope, no time modulation) for the text fusion
    stage: ``x += attn(norm1(x), mask); x += ff(norm2(x))``.

    Under TP the residual stream is sharded on ``dim``; norm1/norm2 are distributed and
    the attention/FF gather the normed input as needed.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        eps: float,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.tp = _tp_factor(parallel_config) > 1
        self.tp_axis = _tp_axis(parallel_config)

        self.norm1 = _make_hidden_norm(
            dim, norm_eps=eps, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        self.norm2 = _make_hidden_norm(
            dim, norm_eps=eps, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        self.attn = Krea2Attention(
            hidden_size=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self.ff = Krea2SwiGLU(
            dim, intermediate_size, mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config
        )

    def _gather(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if not self.tp:
            return x
        return self.ccl_manager.all_gather_persistent_buffer(x, dim=len(x.shape) - 1, mesh_axis=self.tp_axis)

    def forward(self, hidden_states: ttnn.Tensor, attention_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        normed = self.norm1(hidden_states)
        hidden_states = hidden_states + self.attn(self._gather(normed), attention_mask=attention_mask, rope=None)
        normed = self.norm2(hidden_states)
        hidden_states = hidden_states + self.ff(self._gather(normed))
        return hidden_states


class Krea2TextFusion(Module):
    """Fuses the stack of tapped text-encoder hidden states into a single sequence.

    layerwise_blocks attend across the ``num_text_layers`` axis (per token); a linear
    ``projector`` (num_text_layers -> 1, bias=False) collapses that axis; refiner_blocks
    attend across the token sequence.

    TP: the residual stream is sharded on ``dim``. The projector operates over the
    ``num_text_layers`` axis (last dim of a (B,seq,dim,L) tensor) with ``dim`` on the
    penultimate axis; it is small and kept replicated, so the (B,seq,dim,L) tensor is
    all-gathered on ``dim`` before the projector and re-sharded after (the sharding is
    re-established by the refiner blocks' distributed norms which expect a sharded dim).
    """

    def __init__(
        self,
        *,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_text_layers = num_text_layers
        self.dim = dim
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.tp = _tp_factor(parallel_config) > 1
        self.tp_axis = _tp_axis(parallel_config)

        common = dict(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self.layerwise_blocks = ModuleList(Krea2TextFusionBlock(**common) for _ in range(num_layerwise_blocks))
        # projector: Linear(num_text_layers -> 1, bias=False), kept replicated (tiny).
        self.projector = Linear(num_text_layers, 1, bias=False, mesh_device=mesh_device)
        self.refiner_blocks = ModuleList(Krea2TextFusionBlock(**common) for _ in range(num_refiner_blocks))

    def forward(self, encoder_hidden_states: ttnn.Tensor, attention_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        # encoder_hidden_states: (B, seq, num_text_layers, dim), dim sharded under TP.
        B, seq, L, dim = (
            encoder_hidden_states.shape[0],
            encoder_hidden_states.shape[1],
            encoder_hidden_states.shape[2],
            encoder_hidden_states.shape[3],
        )
        local_dim = encoder_hidden_states.shape[3]

        # layerwise: attend across the L axis, independently per (B*seq) token.
        hidden_states = ttnn.reshape(encoder_hidden_states, (B * seq, L, local_dim))
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states, attention_mask=None)

        # (B*seq, L, dim) -> (B, seq, L, dim) -> (B, seq, dim, L)
        hidden_states = ttnn.reshape(hidden_states, (B, seq, L, local_dim))
        hidden_states = ttnn.permute(hidden_states, (0, 1, 3, 2))  # (B, seq, dim, L)

        # projector operates over the last (L) axis with `dim` on the penultimate axis.
        # It is replicated, so gather the sharded `dim` axis (now dim 2) first.
        if self.tp:
            hidden_states = self.ccl_manager.all_gather_persistent_buffer(hidden_states, dim=2, mesh_axis=self.tp_axis)

        # projector: (B, seq, dim, L) @ (L, 1) -> (B, seq, dim, 1) -> squeeze -> (B, seq, dim)
        hidden_states = self.projector(hidden_states)  # (B, seq, dim, 1)
        hidden_states = ttnn.reshape(hidden_states, (B, seq, self.dim))

        # Re-shard the hidden dim across TP for the refiner blocks (distributed norms
        # expect a sharded last dim). mesh_partition slices the replicated tensor.
        if self.tp:
            hidden_states = ttnn.mesh_partition(
                hidden_states, dim=len(hidden_states.shape) - 1, cluster_axis=self.tp_axis
            )

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        return hidden_states


# ======================================================================================
# Timestep embedding
# ======================================================================================
class Krea2TimestepEmbedding(Module):
    """Sinusoidal flow-time embedding (cos-first, input scaled by 1000) + 2-layer MLP.

    The sinusoidal projection is computed on host (as in the reference) and uploaded;
    the MLP runs on device. Keeps the sequence dim at size 1 so per-block modulations
    broadcast over tokens.

    Output ``temb`` is kept **replicated** (full hidden) because it is consumed by
    ``time_mod_proj`` (which shards it) and by the final layer (small). The two Linear
    layers here stay replicated — the MLP is cheap relative to the transformer trunk.
    """

    def __init__(self, embed_dim: int, hidden_size: int, *, mesh_device=None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.mesh_device = mesh_device
        self.linear_1 = Linear(embed_dim, hidden_size, bias=True, mesh_device=mesh_device)
        self.linear_2 = Linear(hidden_size, hidden_size, bias=True, activation_fn=None, mesh_device=mesh_device)

    def forward(self, timestep: torch.Tensor) -> ttnn.Tensor:
        """timestep: torch.Tensor (B,) on host, fp32. Returns (B, 1, hidden) on device."""
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32) / half)  # (half,)
        args = (timestep.float() * 1e3)[:, None, None] * freqs  # (B, 1, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 1, embed_dim)
        emb = bf16_tensor(emb, device=self.mesh_device)

        x = self.linear_1(emb)
        x = ttnn.gelu(x, fast_and_approximate_mode=True)  # gelu_tanh
        return self.linear_2(x)


# ======================================================================================
# Text projection (txt_in)
# ======================================================================================
class Krea2TextProjection(Module):
    """Projects fused text features into the transformer width:
    ``L2(gelu_tanh(L1(rms(x))))``.

    Input ``x`` (text_hidden_dim, sharded under TP) is RMS-normalized (distributed),
    then ``linear_1`` (ColParallel, output=hidden sharded) after an all-gather, and
    ``linear_2`` (RowParallel, reduce-scatter to sharded hidden). Output is the sharded
    hidden dim, matching the transformer residual stream.
    """

    def __init__(
        self,
        text_dim: int,
        hidden_size: int,
        *,
        eps: float,
        mesh_device=None,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.tp = _tp_factor(parallel_config) > 1
        self.tp_axis = _tp_axis(parallel_config)

        self.norm = _make_hidden_norm(
            text_dim, norm_eps=eps, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        if self.tp:
            common = dict(mesh_device=mesh_device, mesh_axis=self.tp_axis, ccl_manager=ccl_manager)
            self.linear_1 = ColParallelLinear(text_dim, hidden_size, bias=True, **common)
            self.linear_2 = RowParallelLinear(hidden_size, hidden_size, bias=True, **common)
        else:
            self.linear_1 = Linear(text_dim, hidden_size, bias=True, mesh_device=mesh_device)
            self.linear_2 = Linear(hidden_size, hidden_size, bias=True, mesh_device=mesh_device)

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = self.norm(hidden_states)  # sharded text_dim under TP
        if self.tp:
            hidden_states = self.ccl_manager.all_gather_persistent_buffer(
                hidden_states, dim=len(hidden_states.shape) - 1, mesh_axis=self.tp_axis
            )
        hidden_states = self.linear_1(hidden_states)  # -> sharded hidden (Col) / full (plain)
        hidden_states = ttnn.gelu(hidden_states, fast_and_approximate_mode=True)  # gelu_tanh
        # Under TP, gelu preserves the column fracture so linear_2 (Row) reduce-scatters
        # back to the sharded hidden; at factor-1 this is a plain replicated matmul.
        return self.linear_2(hidden_states)


# ======================================================================================
# Transformer block (adaLN-single + per-block scale_shift_table)
# ======================================================================================
class Krea2TransformerBlock(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
        padding_config=None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.tp = _tp_factor(parallel_config) > 1
        self.tp_axis = _tp_axis(parallel_config)
        self.local_hidden = hidden_size // _tp_factor(parallel_config)

        from ...layers.module import Parameter

        # scale_shift_table: (6, hidden), kept REPLICATED. The full modulation
        # (temb_mod + table) is formed at full width and then sliced to the local hidden
        # shard with ttnn.mesh_partition in forward, so the six modulation vectors stay
        # correctly aligned with the sharded stream (a contiguous shard of the flat 6*hidden
        # would scramble them).
        self.scale_shift_table = Parameter(
            total_shape=[6, hidden_size],
            mesh_axes=None,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )

        self.norm1 = _make_hidden_norm(
            hidden_size,
            norm_eps=norm_eps,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        self.norm2 = _make_hidden_norm(
            hidden_size,
            norm_eps=norm_eps,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        self.attn = Krea2Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            eps=norm_eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
        )
        self.ff = Krea2SwiGLU(
            hidden_size,
            intermediate_size,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

    def _gather(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if not self.tp:
            return x
        return self.ccl_manager.all_gather_persistent_buffer(x, dim=len(x.shape) - 1, mesh_axis=self.tp_axis)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        temb: ttnn.Tensor,
        rope: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        # temb: (B, 1, 6 * hidden/tp) [sharded per chunk under TP] modulation = temb + table.
        # scale_shift_table is (6, hidden/tp) local under TP so each modulation vector is
        # a sharded-hidden tensor applied elementwise to the sharded normed stream.
        B = temb.shape[0]
        h = self.local_hidden
        H = self.hidden_size
        # temb (temb_mod) and scale_shift_table are REPLICATED at full 6*hidden width.
        # Form the full modulation, then slice the hidden dim to this device's TP shard
        # with mesh_partition so the six modulation vectors stay aligned with the sharded
        # stream (a contiguous shard of the flat 6*hidden would scramble the six vectors).
        temb6 = ttnn.reshape(temb, (B, 1, 6, H))
        table = ttnn.reshape(self.scale_shift_table.data, (1, 1, 6, H))
        modulation = temb6 + table  # (B, 1, 6, hidden) full, broadcast add
        if self.tp:
            modulation = ttnn.mesh_partition(
                modulation, dim=len(modulation.shape) - 1, cluster_axis=self.tp_axis
            )  # -> (B, 1, 6, hidden/tp) local shard

        # unbind along dim -2 into six (B, 1, hidden/tp) tensors.
        mods = [ttnn.reshape(modulation[:, :, i : i + 1, :], (B, 1, h)) for i in range(6)]
        prescale, preshift, pregate, postscale, postshift, postgate = mods

        norm1 = self.norm1(hidden_states)  # sharded hidden
        attn_in = (1.0 + prescale) * norm1 + preshift  # sharded
        attn_out = self.attn(self._gather(attn_in), attention_mask=attention_mask, rope=rope)  # sharded
        hidden_states = hidden_states + pregate * attn_out

        norm2 = self.norm2(hidden_states)  # sharded hidden
        ff_in = (1.0 + postscale) * norm2 + postshift  # sharded
        ff_out = self.ff(self._gather(ff_in))  # sharded
        hidden_states = hidden_states + postgate * ff_out
        return hidden_states


# ======================================================================================
# Final layer
# ======================================================================================
class Krea2FinalLayer(Module):
    """Final adaptive RMSNorm + output projection.

    ``modulation = temb + scale_shift_table(2, hidden); scale, shift = chunk(2);
    x = (1 + scale) * norm(x) + shift; x = linear(x)``.

    Under TP the incoming ``x`` is sharded on hidden; the norm is distributed and the
    (2, hidden) table + replicated ``temb`` are sliced to the local shard. The output
    projection (hidden -> out_channels=64) is small: gather the modulated sharded hidden
    to full and run a replicated Linear.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        *,
        eps: float,
        mesh_device=None,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.tp = _tp_factor(parallel_config) > 1
        self.tp_axis = _tp_axis(parallel_config)
        self.local_hidden = hidden_size // _tp_factor(parallel_config)

        from ...layers.module import Parameter

        self.scale_shift_table = Parameter(
            total_shape=[2, hidden_size],
            mesh_axes=[None, self.tp_axis] if self.tp else None,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )
        self.norm = _make_hidden_norm(
            hidden_size, norm_eps=eps, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
        )
        # Output projection is replicated (small out dim); consumes full hidden.
        self.linear = Linear(hidden_size, out_channels, bias=True, mesh_device=mesh_device)

    def forward(self, hidden_states: ttnn.Tensor, temb: ttnn.Tensor) -> ttnn.Tensor:
        # temb: (B, 1, hidden) replicated. Under TP slice to the local shard so it lines
        # up with the sharded table and the distributed-normed hidden.
        B = temb.shape[0]
        h = self.local_hidden
        if self.tp:
            temb = ttnn.mesh_partition(temb, dim=len(temb.shape) - 1, cluster_axis=self.tp_axis)
        table = ttnn.reshape(self.scale_shift_table.data, (1, 1, 2, h))
        temb2 = ttnn.reshape(temb, (B, 1, 1, h))
        modulation = temb2 + table  # (B, 1, 2, hidden/tp)
        scale = ttnn.reshape(modulation[:, :, 0:1, :], (B, 1, h))
        shift = ttnn.reshape(modulation[:, :, 1:2, :], (B, 1, h))

        normed = self.norm(hidden_states)  # sharded hidden under TP
        modded = (1.0 + scale) * normed + shift  # sharded

        if self.tp:
            modded = self.ccl_manager.all_gather_persistent_buffer(
                modded, dim=len(modded.shape) - 1, mesh_axis=self.tp_axis
            )
        return self.linear(modded)


# ======================================================================================
# 3D axial RoPE precompute (host)
# ======================================================================================
class Krea2RotaryPosEmbed:
    """3D axial RoPE, matching `Krea2RotaryPosEmbed` + `get_1d_rotary_pos_embed`
    (use_real=True, repeat_interleave_real=True).

    For each of the three position axes (t, h, w) with dims [32, 48, 48], computes a
    per-axis cos/sin over the axis's dim and concatenates along the last dim to form
    the full head_dim cos/sin. The result is uploaded as bf16 ttnn tensors of shape
    (seq, head_dim) for use with `_apply_rope`. RoPE is over head_dim, which is whole
    on every device, so the same (replicated) tensors are used regardless of TP.
    """

    def __init__(self, theta: float, axes_dim: Sequence[int], mesh_device=None) -> None:
        self.theta = theta
        self.axes_dim = list(axes_dim)
        self.mesh_device = mesh_device

    @staticmethod
    def _get_1d(dim: int, pos: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
        # freqs computed in fp64 (reference uses freqs_dtype=float64 on non-NPU), cast fp32.
        assert dim % 2 == 0
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))  # (dim/2,)
        freqs = torch.outer(pos.to(torch.float64), freqs)  # (S, dim/2)
        cos = freqs.cos().repeat_interleave(2, dim=1).float()  # (S, dim)
        sin = freqs.sin().repeat_interleave(2, dim=1).float()
        return cos, sin

    def __call__(self, position_ids: torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """position_ids: (seq, 3) torch tensor (t, h, w). Returns (cos, sin) ttnn tensors
        of shape (seq, head_dim)."""
        pos = position_ids.float()
        cos_out, sin_out = [], []
        for i, dim in enumerate(self.axes_dim):
            cos, sin = self._get_1d(dim, pos[:, i], self.theta)
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1)  # (seq, head_dim)
        freqs_sin = torch.cat(sin_out, dim=-1)
        cos_tt = bf16_tensor(freqs_cos, device=self.mesh_device)
        sin_tt = bf16_tensor(freqs_sin, device=self.mesh_device)
        return cos_tt, sin_tt


# ======================================================================================
# Top-level transformer
# ======================================================================================
class Krea2Transformer(Module):
    def __init__(
        self,
        *,
        in_channels: int = 64,
        num_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 12,
        intermediate_size: int = 16384,
        timestep_embed_dim: int = 256,
        text_hidden_dim: int = 2560,
        num_text_layers: int = 12,
        text_num_attention_heads: int = 20,
        text_num_key_value_heads: int = 20,
        text_intermediate_size: int = 6912,
        num_layerwise_text_blocks: int = 2,
        num_refiner_text_blocks: int = 2,
        axes_dims_rope: Sequence[int] = (32, 48, 48),
        rope_theta: float = 1000.0,
        norm_eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
        padding_config=None,
    ) -> None:
        super().__init__()

        hidden_size = attention_head_dim * num_attention_heads
        if sum(axes_dims_rope) != attention_head_dim:
            raise ValueError(
                f"sum(axes_dims_rope)={sum(axes_dims_rope)} must equal attention_head_dim={attention_head_dim}"
            )

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.tp = _tp_factor(parallel_config) > 1
        self.tp_axis = _tp_axis(parallel_config)

        if self.tp:
            common = dict(mesh_device=mesh_device, mesh_axis=self.tp_axis, ccl_manager=ccl_manager)
            # img_in: shard output hidden across TP (small input dim << output dim).
            self.img_in = ColParallelLinear(in_channels, hidden_size, bias=True, **common)
            # time_mod_proj: kept REPLICATED (full 6*hidden on every device). A ColParallel
            # contiguous shard of the flat 6*hidden output scrambles the six modulation
            # vectors when the block reshapes to (6, hidden/tp); instead each block slices
            # the full modulation to its hidden shard with ttnn.mesh_partition (see block).
            self.time_mod_proj = Linear(hidden_size, 6 * hidden_size, bias=True, mesh_device=mesh_device)
        else:
            self.img_in = Linear(in_channels, hidden_size, bias=True, mesh_device=mesh_device)
            self.time_mod_proj = Linear(hidden_size, 6 * hidden_size, bias=True, mesh_device=mesh_device)

        self.time_embed = Krea2TimestepEmbedding(timestep_embed_dim, hidden_size, mesh_device=mesh_device)
        self.text_fusion = Krea2TextFusion(
            num_text_layers=num_text_layers,
            dim=text_hidden_dim,
            num_heads=text_num_attention_heads,
            num_kv_heads=text_num_key_value_heads,
            intermediate_size=text_intermediate_size,
            num_layerwise_blocks=num_layerwise_text_blocks,
            num_refiner_blocks=num_refiner_text_blocks,
            eps=norm_eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self.txt_in = Krea2TextProjection(
            text_hidden_dim,
            hidden_size,
            eps=norm_eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self.rotary_emb = Krea2RotaryPosEmbed(theta=rope_theta, axes_dim=axes_dims_rope, mesh_device=mesh_device)

        self.transformer_blocks = ModuleList(
            Krea2TransformerBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                norm_eps=norm_eps,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
            )
            for _ in range(num_layers)
        )

        self.final_layer = Krea2FinalLayer(
            hidden_size,
            out_channels=in_channels,
            eps=norm_eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # `rotary_emb` in the reference is a parameter-free helper (host RoPE precompute
        # here), so it has no state_dict entries to remap. Nothing to do.
        pass

    def _shard_hidden(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Slice a replicated hidden-dim tensor across the TP axis (no-op at factor 1)."""
        if not self.tp:
            return x
        return ttnn.mesh_partition(x, dim=len(x.shape) - 1, cluster_axis=self.tp_axis)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Args:
            hidden_states: (B, image_seq_len, in_channels) device tensor.
            encoder_hidden_states: (B, text_seq_len, num_text_layers, text_hidden_dim) device tensor.
            timestep: (B,) host torch tensor, fp32, in [0, 1].
            position_ids: (text_seq_len + image_seq_len, 3) host torch tensor.
            encoder_attention_mask: optional host bool torch tensor (B, text_seq_len) marking
                valid text tokens (matching the reference). When None, every text token is valid
                and both attention paths run unmasked.

        Returns:
            velocity (B, image_seq_len, in_channels)
        """
        text_attention_mask = None
        trunk_attention_mask = None

        if encoder_attention_mask is not None:
            # Key-padding masks: padded text tokens are excluded as attention keys
            # everywhere. Build additive float masks on host (0 for valid, large negative
            # for padded) and upload as bf16. Reference builds boolean (B,1,1,L) masks; the
            # ttnn SDPA path consumes additive masks, so we materialise the additive form.
            batch_size = encoder_attention_mask.shape[0]
            image_seq_len = hidden_states.shape[1]
            neg = -1.0e9

            enc = encoder_attention_mask.to(torch.bool)
            text_add = torch.zeros(batch_size, 1, 1, enc.shape[1], dtype=torch.float32)
            text_add.masked_fill_(~enc[:, None, None, :], neg)

            image_valid = torch.ones(batch_size, image_seq_len, dtype=torch.bool)
            trunk_bool = torch.cat([enc, image_valid], dim=1)
            trunk_add = torch.zeros(batch_size, 1, 1, trunk_bool.shape[1], dtype=torch.float32)
            trunk_add.masked_fill_(~trunk_bool[:, None, None, :], neg)

            text_attention_mask = bf16_tensor(text_add, device=self.mesh_device)
            trunk_attention_mask = bf16_tensor(trunk_add, device=self.mesh_device)

        return self._forward(
            hidden_states,
            encoder_hidden_states,
            timestep,
            position_ids,
            trunk_attention_mask=trunk_attention_mask,
            text_attention_mask=text_attention_mask,
        )

    def _forward(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        trunk_attention_mask: ttnn.Tensor | None = None,
        text_attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        text_seq_len = encoder_hidden_states.shape[1]

        temb = self.time_embed(timestep)  # (B, 1, hidden) replicated
        # time_mod_proj: replicated -> (B,1,6*hidden/tp) sharded per chunk under TP.
        temb_mod = self.time_mod_proj(ttnn.gelu(temb, fast_and_approximate_mode=True))

        # text fusion tower: shard the text_hidden dim across TP for the tower.
        ehs = self._shard_hidden(encoder_hidden_states)
        ehs = self.text_fusion(ehs, attention_mask=text_attention_mask)  # (B, text_seq, text_dim/tp)
        ehs = self.txt_in(ehs)  # (B, text_seq, hidden/tp)

        # img_in: shard the hidden output across TP -> (B, image_seq, hidden/tp).
        x = self.img_in(hidden_states)
        x = ttnn.concat([ehs, x], dim=1)  # (B, total_seq, hidden/tp)

        rope = self.rotary_emb(position_ids)

        for block in self.transformer_blocks:
            x = block(x, temb_mod, rope, trunk_attention_mask)

        x = x[:, text_seq_len:, :]  # drop text tokens
        return self.final_layer(x, temb)  # NOTE raw temb


# ======================================================================================
# Checkpoint loader
# ======================================================================================
class Krea2Checkpoint:
    """A Krea-2 checkpoint loader mirroring Flux1Checkpoint.

    Builds a loaded :class:`Krea2Transformer` from either a diffusers
    `Krea2Transformer2DModel` pretrained repo or an in-memory (config, state_dict).
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        config: dict | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self._name = name or "krea2"
        if config is not None and state_dict is not None:
            self._config = dict(config)
            self._state_dict = state_dict
        else:
            from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel

            torch_transformer = Krea2Transformer2DModel.from_pretrained(
                name, subfolder="transformer", torch_dtype=torch.float32
            )
            torch_transformer.eval()
            self._config = dict(torch_transformer.config)
            self._state_dict = torch_transformer.state_dict()

    def build(
        self,
        *,
        mesh_device: ttnn.MeshDevice | None = None,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
        padding_config=None,
        use_cache: bool = False,
    ) -> Krea2Transformer:
        device = mesh_device if mesh_device is not None else (ccl_manager.mesh_device if ccl_manager else None)
        c = self._config

        model = Krea2Transformer(
            in_channels=c["in_channels"],
            num_layers=c["num_layers"],
            attention_head_dim=c["attention_head_dim"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"],
            intermediate_size=c["intermediate_size"],
            timestep_embed_dim=c["timestep_embed_dim"],
            text_hidden_dim=c["text_hidden_dim"],
            num_text_layers=c["num_text_layers"],
            text_num_attention_heads=c["text_num_attention_heads"],
            text_num_key_value_heads=c["text_num_key_value_heads"],
            text_intermediate_size=c["text_intermediate_size"],
            num_layerwise_text_blocks=c["num_layerwise_text_blocks"],
            num_refiner_text_blocks=c["num_refiner_text_blocks"],
            axes_dims_rope=tuple(c["axes_dims_rope"]),
            rope_theta=c["rope_theta"],
            norm_eps=c["norm_eps"],
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
        )

        if use_cache:
            cache.load_model(
                model,
                get_torch_state_dict=lambda: self._state_dict,
                model_name=self._name,
                subfolder="transformer",
                parallel_config=parallel_config,
                mesh_shape=tuple(device.shape),
            )
        else:
            model.load_torch_state_dict(self._state_dict)
        return model

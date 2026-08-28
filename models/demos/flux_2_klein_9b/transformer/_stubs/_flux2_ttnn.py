# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN building blocks for the FLUX.2 Klein 9B transformer.

Shared by the per-component stubs in this directory: the model reuses the same
few pieces (a modulated LayerNorm, a SwiGLU feed-forward, a joint dual-stream
attention, a fused parallel self-attention) in several components, so they live
here once instead of being copy-pasted per stub.

Everything in a `__call__` is ttnn: no torch math, no device->host readback.
torch appears only in the constructors, where a checkpoint weight is transposed
into ttnn's `[in, out]` layout and (when tensor-parallel) permuted so that a
contiguous mesh shard lands on the right columns, and then staged on device.

TENSOR PARALLELISM
------------------
Megatron column-then-row, adapted to this model's *fused* projections.

* A projection whose output feeds a per-element op is COLUMN-parallel: split its
  output features, no collective needed, because everything downstream (SwiGLU
  gate, QK-norm, RoPE, softmax) is per-feature or per-head.
* The projection that reduces back to the model dim is ROW-parallel: split its
  input features, then one `all_reduce` (SUM) turns each chip's partial sum into
  the single-device answer.
* Norms (all `elementwise_affine=False` here), QK-norm gammas (over head_dim,
  which is never split), RoPE tables and modulation shift/scale/gate vectors are
  elementwise or per-head, so they stay REPLICATED.

The wrinkle this model adds is that several of its linears are *fused*: one
`nn.Linear` emits several logically separate tensors that are later `chunk`ed or
`split` apart (`Flux2FeedForward.linear_in` emits SwiGLU's two halves;
`Flux2ParallelSelfAttention.to_qkv_mlp_proj` emits q, k, v and the MLP's two
halves). Sharding such a weight contiguously is WRONG: with 24576 columns over
8 chips, chips 0-3 would own all of SwiGLU's `x1` and chips 4-7 all of its `x2`,
so the per-element `silu(x1) * x2` would pair features that live on different
chips. `_regroup` fixes this at load time by reordering the columns to
`[group0_chip0, group1_chip0, ..., group0_chip1, ...]`, so a plain contiguous
`ShardTensorToMesh` hands each chip a matching slice of EVERY group. The
row-parallel `to_out` of the single-stream block gets the same treatment on its
input axis, so its rows line up with the (attn_out | mlp) concatenation each chip
actually holds.

None of this changes the arithmetic: it is the same sum of the same products,
partitioned differently.
"""

from __future__ import annotations

import torch

import ttnn

TILE = 32


# --------------------------------------------------------------------------- mesh
def mesh_shape(device):
    """(rows, cols) of `device` as a mesh, or None if it is a single device."""
    shape = getattr(device, "shape", None)
    if shape is None:
        return None
    try:
        dims = [int(d) for d in shape]
    except TypeError:
        return None
    if len(dims) == 1:
        dims = [1, dims[0]]
    if len(dims) != 2:
        return None
    return dims[0], dims[1]


def num_devices(device) -> int:
    shape = mesh_shape(device)
    return shape[0] * shape[1] if shape is not None else 1


def replicate_mapper(device):
    return ttnn.ReplicateTensorToMesh(device) if num_devices(device) > 1 else None


def shard_mapper(device, dim: int):
    return ttnn.ShardTensorToMesh(device, dim=dim) if num_devices(device) > 1 else None


def all_reduce(device, x):
    """SUM across the tensor-parallel group, leaving the full result on every chip."""
    if num_devices(device) <= 1:
        return x
    shape = mesh_shape(device)
    if shape is not None and shape[0] > 1:
        # (DP, TP) mesh: reduce only along the tensor-parallel axis.
        return ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.all_reduce(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def all_gather(device, x, dim):
    """Concatenate the per-chip shards along `dim`, leaving the full tensor on
    every chip -- the collective that closes a column-parallel projection whose
    consumer needs all of the output features."""
    if num_devices(device) <= 1:
        return x
    shape = mesh_shape(device)
    if shape is not None and shape[0] > 1:
        return ttnn.all_gather(x, dim, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.all_gather(x, dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def mesh_partition(device, x, dim):
    """Turn a REPLICATED tensor into a sharded one along `dim` -- the documented
    inverse of `all_gather`. Needed when a component's input arrives replicated
    but its work is to be split feature-wise."""
    if num_devices(device) <= 1:
        return x
    shape = mesh_shape(device)
    if shape is not None and shape[0] > 1:
        return ttnn.mesh_partition(x, dim, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.mesh_partition(x, dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def kernel_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def to_device(device, tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Accept an already-marshalled `ttnn.Tensor`, or stage a host one.

    The PCC harness marshals its inputs, so on the compute path this is a
    pass-through; the `from_torch` branch keeps the module callable with plain
    torch tensors (e.g. from the demo pipeline)."""
    if tensor is None or isinstance(tensor, ttnn.Tensor):
        return tensor
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate_mapper(device),
    )


def as_rank4(x, *, seq_last_dim=None):
    """Return `(x4, original_shape)` with `x4` shaped [B, 1, S, C]."""
    shape = list(x.shape)
    if len(shape) == 4:
        return x, shape
    if len(shape) == 3:
        return ttnn.reshape(x, [shape[0], 1, shape[1], shape[2]]), shape
    if len(shape) == 2:
        return ttnn.reshape(x, [1, 1, shape[0], shape[1]]), shape
    raise RuntimeError(f"unsupported rank {len(shape)} for {shape}")


def restore_rank(x, original_shape, channels=None):
    if len(original_shape) == 4:
        return x
    out_shape = list(original_shape[:-1]) + [channels if channels is not None else original_shape[-1]]
    return ttnn.reshape(x, out_shape)


# ------------------------------------------------------------------------ weights
def _regroup(t: torch.Tensor, dim: int, group_sizes, tp: int) -> torch.Tensor:
    """Reorder `t` along `dim` so a contiguous `tp`-way split is group-correct.

    `group_sizes` are the logically separate tensors a FUSED linear emits (or
    consumes) along `dim`, in order. The result concatenates, for chip 0 then
    chip 1 ..., that chip's slice of every group. With a single group this is
    the identity, so the plain column/row-parallel case shares the code path.
    """
    if tp <= 1:
        return t.contiguous()
    total = int(t.shape[dim])
    if sum(group_sizes) != total:
        raise RuntimeError(f"group sizes {group_sizes} do not cover axis {dim} of size {total}")
    offsets, off = [], 0
    for g in group_sizes:
        if g % tp:
            raise RuntimeError(f"group of {g} is not divisible by TP={tp}")
        offsets.append((off, g))
        off += g
    parts = [t.narrow(dim, o + c * (g // tp), g // tp) for c in range(tp) for (o, g) in offsets]
    return torch.cat(parts, dim=dim).contiguous()


class TtLinear:
    """A `nn.Linear` staged on device, optionally tensor-parallel.

    scheme='replicate' : full weight on every chip.
    scheme='column'    : output features split; returns each chip's LOCAL slice
                         (`local_groups` gives the per-group local widths).
    scheme='row'       : input features split; the caller must already hold the
                         matching input slice. Returns the all_reduced full
                         result, with the bias (if any) added AFTER the reduce --
                         adding it before would count it TP times.
    """

    def __init__(self, device, torch_linear, *, scheme="replicate", groups=None, dtype=ttnn.bfloat16):
        self.device = device
        self.tp = num_devices(device)
        self.kernel_cfg = kernel_config()

        w = torch_linear.weight.detach().to(torch.float32).t().contiguous()  # [in, out]
        self.in_features, self.out_features = int(w.shape[0]), int(w.shape[1])
        bias = getattr(torch_linear, "bias", None)
        b = None if bias is None else bias.detach().to(torch.float32).reshape(1, -1)

        self.scheme = scheme if self.tp > 1 else "replicate"
        axis_len = self.out_features if self.scheme == "column" else self.in_features
        self.groups = list(groups) if groups else [axis_len]
        self.local_groups = [g // self.tp for g in self.groups] if self.scheme != "replicate" else list(self.groups)

        if self.scheme == "column":
            w = _regroup(w, 1, self.groups, self.tp)
            if b is not None:
                b = _regroup(b, 1, self.groups, self.tp)
            w_mapper = shard_mapper(device, 3)
            b_mapper = shard_mapper(device, 3)
        elif self.scheme == "row":
            w = _regroup(w, 0, self.groups, self.tp)
            w_mapper = shard_mapper(device, 2)
            b_mapper = replicate_mapper(device)
        else:
            w_mapper = replicate_mapper(device)
            b_mapper = replicate_mapper(device)

        self.weight = ttnn.from_torch(
            w.reshape(1, 1, *w.shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=w_mapper,
        )
        self.bias = (
            None
            if b is None
            else ttnn.from_torch(
                b.reshape(1, 1, 1, -1),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=b_mapper,
            )
        )

    def __call__(self, x4, *, dtype=ttnn.bfloat16):
        # For 'row', the bias must wait until after the reduce.
        fused_bias = self.bias if self.scheme != "row" else None
        out = ttnn.linear(
            x4,
            self.weight,
            bias=fused_bias,
            compute_kernel_config=self.kernel_cfg,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.scheme == "row":
            out = all_reduce(self.device, out)
            if self.bias is not None:
                out = ttnn.add(out, self.bias)
        return out


def split_last(x4, sizes):
    """Split [B, 1, S, sum(sizes)] into contiguous chunks along the last axis."""
    shape = list(x4.shape)
    out, off = [], 0
    for size in sizes:
        out.append(ttnn.slice(x4, [0, 0, 0, off], [shape[0], shape[1], shape[2], off + size]))
        off += size
    return out


def split_seq(x4, sizes):
    """Split [B, H, sum(sizes), C] into contiguous chunks along the sequence axis."""
    shape = list(x4.shape)
    out, off = [], 0
    for size in sizes:
        out.append(ttnn.slice(x4, [0, 0, off, 0], [shape[0], shape[1], off + size, shape[3]]))
        off += size
    return out


# -------------------------------------------------------------------- modulation
def modulation_split(mod, n_params):
    """`Flux2Modulation.split`: [B, n_params * C] -> n_params tensors [B, 1, 1, C].

    Each is broadcast over the sequence axis by the caller's elementwise ops,
    exactly as torch's `[:, None, :]` / `unsqueeze(1)` does.
    """
    shape = list(mod.shape)
    batch, width = int(shape[0]), int(shape[-1])
    dim = width // n_params
    mod4 = ttnn.reshape(mod, [batch, 1, 1, width])
    return [ttnn.slice(mod4, [0, 0, 0, i * dim], [batch, 1, 1, (i + 1) * dim]) for i in range(n_params)]


def modulate(normed, scale, shift):
    """`(1 + scale) * normed + shift`, the Flux2 blocks' AdaLN-Zero modulation."""
    scaled = ttnn.mul(normed, ttnn.add(scale, 1.0))
    out = ttnn.add(scaled, shift)
    ttnn.deallocate(scaled)
    return out


# --------------------------------------------------------------------------- rope
def rope_pair_matrix(head_dim: int) -> torch.Tensor:
    """`P` with `x @ P == interleave(-x_odd, x_even)`.

    diffusers' `apply_rotary_emb(..., use_real_unbind_dim=-1)` rotates ADJACENT
    feature pairs: `out[2i] = -x[2i+1]`, `out[2i+1] = x[2i]`. ttnn's packaged RoPE
    kernels implement the other (half-split) convention, and expressing the
    interleave with reshape/unbind would need a rank-5 tensor. A constant 0/±1
    permutation matrix does it in one matmul instead -- exact in bfloat16, since
    every entry is 0 or ±1 and each output is a single input element.
    """
    p = torch.zeros(head_dim, head_dim, dtype=torch.float32)
    for i in range(head_dim // 2):
        p[2 * i + 1, 2 * i] = -1.0
        p[2 * i, 2 * i + 1] = 1.0
    return p


class TtRotary:
    def __init__(self, device, head_dim: int):
        self.device = device
        self.head_dim = head_dim
        self.pair = ttnn.from_torch(
            rope_pair_matrix(head_dim).reshape(1, 1, head_dim, head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_mapper(device),
        )
        self.kernel_cfg = kernel_config()

    def tables(self, image_rotary_emb):
        """(cos, sin) as [1, 1, S, head_dim], broadcastable over batch and heads."""
        if image_rotary_emb is None:
            return None
        cos, sin = image_rotary_emb
        cos = to_device(self.device, cos)
        sin = to_device(self.device, sin)
        seq = int(cos.shape[-2])
        return (
            ttnn.reshape(cos, [1, 1, seq, self.head_dim]),
            ttnn.reshape(sin, [1, 1, seq, self.head_dim]),
        )

    def __call__(self, x, cos, sin):
        rotated = ttnn.linear(
            x,
            self.pair,
            compute_kernel_config=self.kernel_cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        a = ttnn.mul(x, cos)
        b = ttnn.mul(rotated, sin)
        out = ttnn.add(a, b)
        ttnn.deallocate(rotated)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return out


# --------------------------------------------------------------------------- norms
class TtRmsNorm:
    """RMSNorm over the last axis, for the per-head QK norms."""

    def __init__(self, device, torch_norm):
        self.device = device
        self.kernel_cfg = kernel_config()
        self.eps = float(getattr(torch_norm, "eps", None) or 1e-6)
        weight = getattr(torch_norm, "weight", None)
        self.weight = (
            None
            if weight is None
            else ttnn.from_torch(
                weight.detach().to(torch.float32).reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_mapper(device),
            )
        )

    def __call__(self, x):
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            compute_kernel_config=self.kernel_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class TtLayerNorm:
    def __init__(self, device, torch_norm):
        self.device = device
        self.kernel_cfg = kernel_config()
        self.eps = float(getattr(torch_norm, "eps", None) or 1e-6)

        def stage(param):
            if param is None:
                return None
            return ttnn.from_torch(
                param.detach().to(torch.float32).reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate_mapper(device),
            )

        self.weight = stage(getattr(torch_norm, "weight", None))
        self.bias = stage(getattr(torch_norm, "bias", None))

    def __call__(self, x):
        return ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            bias=self.bias,
            compute_kernel_config=self.kernel_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


# ------------------------------------------------------------------------ swiglu
def swiglu(x4, half):
    """`Flux2SwiGLU`: silu(first half) * second half, along the last axis."""
    x1, x2 = split_last(x4, [half, half])
    gate = ttnn.silu(x1)
    out = ttnn.mul(gate, x2)
    ttnn.deallocate(x1)
    ttnn.deallocate(x2)
    ttnn.deallocate(gate)
    return out


class TtFlux2FeedForward:
    """`Flux2FeedForward`: linear_in -> SwiGLU -> linear_out.

    `linear_in` is COLUMN-parallel over its two SwiGLU halves (declared as two
    groups so each chip gets matching slices of both), `linear_out` is
    ROW-parallel over the single inner axis, which lines up with those slices.
    """

    def __init__(self, device, torch_ff, *, tensor_parallel=True):
        self.device = device
        self.tp = num_devices(device) if tensor_parallel else 1
        inner = int(torch_ff.linear_out.in_features)
        self.inner = inner
        column = "column" if tensor_parallel else "replicate"
        row = "row" if tensor_parallel else "replicate"
        self.linear_in = TtLinear(device, torch_ff.linear_in, scheme=column, groups=[inner, inner])
        self.linear_out = TtLinear(device, torch_ff.linear_out, scheme=row, groups=[inner])
        self.local_inner = self.linear_in.local_groups[0]

    def __call__(self, x4):
        hidden = self.linear_in(x4)
        act = swiglu(hidden, self.local_inner)
        ttnn.deallocate(hidden)
        out = self.linear_out(act)
        ttnn.deallocate(act)
        return out


# ---------------------------------------------------------------------- attention
def split_heads(x4, n_heads: int, head_dim: int):
    """[B, 1, S, n_heads * head_dim] -> [B, n_heads, S, head_dim]."""
    shape = list(x4.shape)
    batch, seq = int(shape[0]), int(shape[2])
    heads = ttnn.reshape(x4, [batch, seq, n_heads, head_dim])
    out = ttnn.permute(heads, [0, 2, 1, 3])
    ttnn.deallocate(heads)
    return out


def merge_heads(x4):
    """[B, n_heads, S, head_dim] -> [B, 1, S, n_heads * head_dim]."""
    shape = list(x4.shape)
    batch, n_heads, seq, head_dim = (int(s) for s in shape)
    seq_major = ttnn.permute(x4, [0, 2, 1, 3])
    out = ttnn.reshape(seq_major, [batch, 1, seq, n_heads * head_dim])
    ttnn.deallocate(seq_major)
    return out


class TtFlux2Attention:
    """`Flux2Attention` -- the dual-stream block's joint attention.

    q/k/v (image) and add_{q,k,v}_proj (text) are COLUMN-parallel, so each chip
    owns a contiguous set of whole heads (out features are head-major, so a
    1/TP slice is exactly `heads/TP` heads). Everything between the projections
    and the output -- QK-norm over head_dim, RoPE, softmax -- is per-head, so no
    collective is needed there. `to_out[0]` and `to_add_out` are ROW-parallel and
    each end in one all_reduce.
    """

    def __init__(self, device, torch_attn, *, tensor_parallel=True):
        self.device = device
        self.tp = num_devices(device) if tensor_parallel else 1
        self.heads = int(torch_attn.heads)
        self.head_dim = int(torch_attn.head_dim)
        self.inner_dim = int(torch_attn.inner_dim)
        self.scale = self.head_dim**-0.5
        if self.heads % self.tp:
            raise RuntimeError(f"{self.heads} heads is not divisible by TP={self.tp}")
        self.local_heads = self.heads // self.tp

        column = "column" if tensor_parallel else "replicate"
        row = "row" if tensor_parallel else "replicate"
        self.to_q = TtLinear(device, torch_attn.to_q, scheme=column)
        self.to_k = TtLinear(device, torch_attn.to_k, scheme=column)
        self.to_v = TtLinear(device, torch_attn.to_v, scheme=column)
        self.norm_q = TtRmsNorm(device, torch_attn.norm_q)
        self.norm_k = TtRmsNorm(device, torch_attn.norm_k)
        self.to_out = TtLinear(device, torch_attn.to_out[0], scheme=row)

        self.has_context = getattr(torch_attn, "added_kv_proj_dim", None) is not None
        if self.has_context:
            self.add_q_proj = TtLinear(device, torch_attn.add_q_proj, scheme=column)
            self.add_k_proj = TtLinear(device, torch_attn.add_k_proj, scheme=column)
            self.add_v_proj = TtLinear(device, torch_attn.add_v_proj, scheme=column)
            self.norm_added_q = TtRmsNorm(device, torch_attn.norm_added_q)
            self.norm_added_k = TtRmsNorm(device, torch_attn.norm_added_k)
            self.to_add_out = TtLinear(device, torch_attn.to_add_out, scheme=row)

        self.rotary = TtRotary(device, self.head_dim)
        self.kernel_cfg = kernel_config()

    def _heads(self, projected):
        out = split_heads(projected, self.local_heads, self.head_dim)
        ttnn.deallocate(projected)
        return out

    def __call__(self, hidden_states, encoder_hidden_states=None, image_rotary_emb=None, **kwargs):
        x, x_shape = as_rank4(to_device(self.device, hidden_states))
        img_len = int(x.shape[2])

        q = self._heads(self.to_q(x))
        k = self._heads(self.to_k(x))
        v = self._heads(self.to_v(x))
        q = self.norm_q(q)
        k = self.norm_k(k)

        txt_len = 0
        if self.has_context and encoder_hidden_states is not None:
            ctx, ctx_shape = as_rank4(to_device(self.device, encoder_hidden_states))
            txt_len = int(ctx.shape[2])
            eq = self.norm_added_q(self._heads(self.add_q_proj(ctx)))
            ek = self.norm_added_k(self._heads(self.add_k_proj(ctx)))
            ev = self._heads(self.add_v_proj(ctx))
            # Text tokens come FIRST in the joint sequence.
            q, k, v = (
                ttnn.concat([eq, q], dim=2),
                ttnn.concat([ek, k], dim=2),
                ttnn.concat([ev, v], dim=2),
            )
            for t in (eq, ek, ev):
                ttnn.deallocate(t)
        else:
            ctx_shape = None

        tables = self.rotary.tables(image_rotary_emb)
        if tables is not None:
            cos, sin = tables
            q_rot, k_rot = self.rotary(q, cos, sin), self.rotary(k, cos, sin)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            q, k = q_rot, k_rot

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            compute_kernel_config=self.kernel_cfg,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        merged = merge_heads(attn)
        ttnn.deallocate(attn)

        if txt_len:
            ctx_part, img_part = split_seq(merged, [txt_len, img_len])
            ttnn.deallocate(merged)
            out_img = restore_rank(self.to_out(img_part), x_shape, self.to_out.out_features)
            out_ctx = restore_rank(self.to_add_out(ctx_part), ctx_shape, self.to_add_out.out_features)
            ttnn.deallocate(img_part)
            ttnn.deallocate(ctx_part)
            return out_img, out_ctx

        out = restore_rank(self.to_out(merged), x_shape, self.to_out.out_features)
        ttnn.deallocate(merged)
        return out


class TtFlux2ParallelSelfAttention:
    """`Flux2ParallelSelfAttention` -- the single-stream block's ViT-22B-style
    parallel attention+MLP, with both fused into one input and one output linear.

    `to_qkv_mlp_proj` is COLUMN-parallel over FIVE groups (q, k, v, and SwiGLU's
    two halves) so every chip gets matching slices of all of them; `to_out` is
    ROW-parallel over TWO groups (the attention output and the MLP activation),
    which is exactly the concatenation each chip holds.
    """

    def __init__(self, device, torch_attn, *, tensor_parallel=True):
        self.device = device
        self.tp = num_devices(device) if tensor_parallel else 1
        self.heads = int(torch_attn.heads)
        self.head_dim = int(torch_attn.head_dim)
        self.inner_dim = int(torch_attn.inner_dim)
        self.mlp_hidden_dim = int(torch_attn.mlp_hidden_dim)
        self.mlp_mult_factor = int(torch_attn.mlp_mult_factor)
        self.scale = self.head_dim**-0.5
        if self.heads % self.tp:
            raise RuntimeError(f"{self.heads} heads is not divisible by TP={self.tp}")
        self.local_heads = self.heads // self.tp

        column = "column" if tensor_parallel else "replicate"
        row = "row" if tensor_parallel else "replicate"
        qkv_groups = [self.inner_dim] * 3 + [self.mlp_hidden_dim] * self.mlp_mult_factor
        self.to_qkv_mlp_proj = TtLinear(device, torch_attn.to_qkv_mlp_proj, scheme=column, groups=qkv_groups)
        self.local_splits = self.to_qkv_mlp_proj.local_groups
        self.to_out = TtLinear(device, torch_attn.to_out, scheme=row, groups=[self.inner_dim, self.mlp_hidden_dim])
        self.norm_q = TtRmsNorm(device, torch_attn.norm_q)
        self.norm_k = TtRmsNorm(device, torch_attn.norm_k)
        self.rotary = TtRotary(device, self.head_dim)
        self.kernel_cfg = kernel_config()

    def __call__(self, hidden_states, image_rotary_emb=None, **kwargs):
        x, x_shape = as_rank4(to_device(self.device, hidden_states))
        fused = self.to_qkv_mlp_proj(x)
        parts = split_last(fused, self.local_splits)
        ttnn.deallocate(fused)
        q_lin, k_lin, v_lin = parts[0], parts[1], parts[2]
        mlp_parts = parts[3:]

        q = self.norm_q(split_heads(q_lin, self.local_heads, self.head_dim))
        k = self.norm_k(split_heads(k_lin, self.local_heads, self.head_dim))
        v = split_heads(v_lin, self.local_heads, self.head_dim)
        for t in (q_lin, k_lin, v_lin):
            ttnn.deallocate(t)

        tables = self.rotary.tables(image_rotary_emb)
        if tables is not None:
            cos, sin = tables
            q_rot, k_rot = self.rotary(q, cos, sin), self.rotary(k, cos, sin)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            q, k = q_rot, k_rot

        attn = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=self.scale, compute_kernel_config=self.kernel_cfg
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        merged = merge_heads(attn)
        ttnn.deallocate(attn)

        # `Flux2SwiGLU` over the MLP half of the fused projection.
        gate = ttnn.silu(mlp_parts[0])
        mlp_act = ttnn.mul(gate, mlp_parts[1])
        ttnn.deallocate(gate)
        for t in mlp_parts:
            ttnn.deallocate(t)

        joined = ttnn.concat([merged, mlp_act], dim=3)
        ttnn.deallocate(merged)
        ttnn.deallocate(mlp_act)
        out = restore_rank(self.to_out(joined), x_shape, self.to_out.out_features)
        ttnn.deallocate(joined)
        return out


# ---------------------------------------------------------------- timestep stack
class TtTimesteps:
    """`diffusers` `Timesteps` / `get_timestep_embedding`: the sinusoidal
    timestep features.

        freq[k] = max_period ** (-k / (half - downscale_freq_shift))
        emb     = scale * timestep[:, None] * freq[None, :]
        out     = cat([cos(emb), sin(emb)])        # flip_sin_to_cos=True

    Parameterless, so nothing is sharded -- the frequency row is a lookup table
    and stays REPLICATED.

    PRECISION: run in float32. This model scales the timestep by 1000 before it
    reaches here, so the phase reaches ~1000 radians; measured on device,
    `ttnn.cos` of a bfloat16 argument is off by up to 1.6 absolute at that
    range, versus 0.0 in float32. The outer product is a `repeat` + `mul` rather
    than a K=1 matmul for the same reason (the matmul path rounds through
    bfloat16; the broadcast multiply was measured exact).
    """

    def __init__(self, device, torch_module, *, max_period: int = 10000):
        import math

        self.device = device
        self.num_channels = int(torch_module.num_channels)
        self.flip_sin_to_cos = bool(torch_module.flip_sin_to_cos)
        self.half = self.num_channels // 2
        shift = float(getattr(torch_module, "downscale_freq_shift", 0.0) or 0.0)
        scale = float(getattr(torch_module, "scale", 1) or 1)

        exponent = -math.log(max_period) * torch.arange(0, self.half, dtype=torch.float32)
        exponent = exponent / (self.half - shift)
        # `scale` multiplies the whole product, so folding it into the frequency
        # row is the same arithmetic with one op fewer.
        freq = torch.exp(exponent) * scale
        self.freq = ttnn.from_torch(
            freq.reshape(1, 1, 1, self.half),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_mapper(device),
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, timesteps, **kwargs):
        t = to_device(self.device, timesteps)
        count = 1
        for dim in list(t.shape):
            count *= int(dim)
        t4 = ttnn.reshape(t, [1, 1, count, 1])
        if t4.dtype != ttnn.float32:
            t4 = ttnn.typecast(t4, ttnn.float32)

        spread = ttnn.repeat(t4, [1, 1, 1, self.half])
        emb = ttnn.mul(spread, self.freq)
        ttnn.deallocate(spread)
        cos, sin = ttnn.cos(emb), ttnn.sin(emb)
        ttnn.deallocate(emb)

        parts = [cos, sin] if self.flip_sin_to_cos else [sin, cos]
        out = ttnn.concat(parts, dim=3)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        return ttnn.reshape(out, [count, self.num_channels])


class TtTimestepEmbedding:
    """`diffusers` `TimestepEmbedding`: linear_1 -> SiLU -> linear_2.

    Textbook column-then-row: `linear_1`'s output feeds a per-element SiLU, so
    it is COLUMN-parallel with no collective; `linear_2` reduces back to the
    model dim, so it is ROW-parallel and ends in one `all_reduce`.
    """

    def __init__(self, device, torch_module, *, tensor_parallel=True):
        self.device = device
        column = "column" if tensor_parallel else "replicate"
        row = "row" if tensor_parallel else "replicate"
        self.linear_1 = TtLinear(device, torch_module.linear_1, scheme=column)
        self.linear_2 = TtLinear(device, torch_module.linear_2, scheme=row)
        if getattr(torch_module, "cond_proj", None) is not None:
            raise RuntimeError("TimestepEmbedding.cond_proj is not part of this checkpoint")
        if getattr(torch_module, "post_act", None) is not None:
            raise RuntimeError("TimestepEmbedding.post_act is not part of this checkpoint")
        self.out_features = self.linear_2.out_features

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, sample, **kwargs):
        x4, shape = as_rank4(to_device(self.device, sample))
        if x4.dtype != ttnn.bfloat16:
            x4 = ttnn.typecast(x4, ttnn.bfloat16)
        hidden = self.linear_1(x4)
        act = ttnn.silu(hidden)
        ttnn.deallocate(hidden)
        out = self.linear_2(act)
        ttnn.deallocate(act)
        return restore_rank(out, shape, self.out_features)


class TtFlux2TimestepGuidanceEmbeddings:
    """`Flux2TimestepGuidanceEmbeddings`: sinusoidal timestep features through an
    MLP, plus the same for the guidance scale when the variant has one.

    This checkpoint sets `guidance_embeds=false`, so `guidance_embedder` is None
    and the guidance argument is ignored -- the reference returns `timesteps_emb`
    alone. The optional branch is implemented anyway so the module is correct
    for a guidance-distilled sibling.
    """

    def __init__(self, device, torch_module, *, tensor_parallel=True):
        self.device = device
        self.time_proj = TtTimesteps(device, torch_module.time_proj)
        self.timestep_embedder = TtTimestepEmbedding(
            device, torch_module.timestep_embedder, tensor_parallel=tensor_parallel
        )
        guidance = getattr(torch_module, "guidance_embedder", None)
        self.guidance_embedder = (
            None if guidance is None else TtTimestepEmbedding(device, guidance, tensor_parallel=tensor_parallel)
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(self, timestep, guidance=None, **kwargs):
        emb = self.timestep_embedder(self.time_proj(timestep))
        if guidance is not None and self.guidance_embedder is not None:
            guidance_emb = self.guidance_embedder(self.time_proj(guidance))
            out = ttnn.add(emb, guidance_emb)
            ttnn.deallocate(emb)
            ttnn.deallocate(guidance_emb)
            return out
        return emb


# ------------------------------------------------------------------------- blocks
class TtFlux2TransformerBlock:
    """`Flux2TransformerBlock` -- a dual-stream (image + text) DiT block."""

    def __init__(self, device, torch_block, *, tensor_parallel=True):
        self.device = device
        self.norm1 = TtLayerNorm(device, torch_block.norm1)
        self.norm1_context = TtLayerNorm(device, torch_block.norm1_context)
        self.norm2 = TtLayerNorm(device, torch_block.norm2)
        self.norm2_context = TtLayerNorm(device, torch_block.norm2_context)
        self.attn = TtFlux2Attention(device, torch_block.attn, tensor_parallel=tensor_parallel)
        self.ff = TtFlux2FeedForward(device, torch_block.ff, tensor_parallel=tensor_parallel)
        self.ff_context = TtFlux2FeedForward(device, torch_block.ff_context, tensor_parallel=tensor_parallel)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states=None,
        temb_mod_img=None,
        temb_mod_txt=None,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        **kwargs,
    ):
        x, x_shape = as_rank4(to_device(self.device, hidden_states))
        ctx, ctx_shape = as_rank4(to_device(self.device, encoder_hidden_states))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation_split(
            to_device(self.device, temb_mod_img), 6
        )
        (
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = modulation_split(to_device(self.device, temb_mod_txt), 6)

        norm_x = modulate(self.norm1(x), scale_msa, shift_msa)
        norm_ctx = modulate(self.norm1_context(ctx), c_scale_msa, c_shift_msa)

        attn_img, attn_ctx = self.attn(
            hidden_states=norm_x, encoder_hidden_states=norm_ctx, image_rotary_emb=image_rotary_emb
        )
        ttnn.deallocate(norm_x)
        ttnn.deallocate(norm_ctx)

        # --- image stream ---
        x = ttnn.add(x, ttnn.mul(attn_img, gate_msa))
        ttnn.deallocate(attn_img)
        ff_in = modulate(self.norm2(x), scale_mlp, shift_mlp)
        x = ttnn.add(x, ttnn.mul(self.ff(ff_in), gate_mlp))
        ttnn.deallocate(ff_in)

        # --- text stream ---
        ctx = ttnn.add(ctx, ttnn.mul(attn_ctx, c_gate_msa))
        ttnn.deallocate(attn_ctx)
        cff_in = modulate(self.norm2_context(ctx), c_scale_mlp, c_shift_mlp)
        ctx = ttnn.add(ctx, ttnn.mul(self.ff_context(cff_in), c_gate_mlp))
        ttnn.deallocate(cff_in)

        return restore_rank(ctx, ctx_shape), restore_rank(x, x_shape)


class TtFlux2SingleTransformerBlock:
    """`Flux2SingleTransformerBlock` -- a single-stream parallel DiT block."""

    def __init__(self, device, torch_block, *, tensor_parallel=True):
        self.device = device
        self.norm = TtLayerNorm(device, torch_block.norm)
        self.attn = TtFlux2ParallelSelfAttention(device, torch_block.attn, tensor_parallel=tensor_parallel)

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states=None,
        temb_mod=None,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        split_hidden_states=False,
        text_seq_len=None,
        **kwargs,
    ):
        x, x_shape = as_rank4(to_device(self.device, hidden_states))
        if encoder_hidden_states is not None:
            ctx, _ = as_rank4(to_device(self.device, encoder_hidden_states))
            text_seq_len = int(ctx.shape[2])
            joined = ttnn.concat([ctx, x], dim=2)
            ttnn.deallocate(x)
            x = joined
            x_shape = list(x_shape[:-2]) + [int(x.shape[2]), x_shape[-1]]

        shift, scale, gate = modulation_split(to_device(self.device, temb_mod), 3)
        norm_x = modulate(self.norm(x), scale, shift)
        attn_out = self.attn(hidden_states=norm_x, image_rotary_emb=image_rotary_emb)
        ttnn.deallocate(norm_x)
        x = ttnn.add(x, ttnn.mul(attn_out, gate))
        ttnn.deallocate(attn_out)

        if split_hidden_states and text_seq_len:
            ctx_part, img_part = split_seq(x, [text_seq_len, int(x.shape[2]) - text_seq_len])
            ttnn.deallocate(x)
            return ctx_part, img_part
        return restore_rank(x, x_shape)

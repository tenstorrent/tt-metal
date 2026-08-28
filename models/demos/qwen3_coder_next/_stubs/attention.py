# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Qwen3NextAttention` (`Qwen/Qwen3-Coder-Next`), tensor-parallel over TP chips.

Reference: `transformers/models/qwen3_next/modeling_qwen3_next.py::Qwen3NextAttention`.

    q_proj : hidden -> H * (2 * head_dim)      # per head: [query | gate], side by side
    k_proj : hidden -> KV * head_dim
    v_proj : hidden -> KV * head_dim
    q_norm / k_norm : RMSNorm over head_dim, gamma = (1 + weight)
    rope (partial: rotary_dim = cos.shape[-1]) -> GQA softmax attention
    out = o_proj( attn * sigmoid(gate) )

Tensor-parallel scheme (derived per the TP principles; math is unchanged, only placement):

  * q_proj / k_proj / v_proj are COLUMN-parallel: their outputs feed per-element ops (the head-wise
    RMSNorm, rope, the sigmoid gate) so each chip can own a disjoint slice of the head axis and never
    needs its neighbour's heads. q_proj's output is laid out head-major with each head's gate riding
    along inside that head's 2*head_dim block, so a plain split of the output feature axis lands
    exactly on head boundaries and keeps every query next to its own gate.
  * The GQA head mapping is what makes k/v splittable the same way: `repeat_kv` expands kv head j to
    query heads [j*n_rep, (j+1)*n_rep), contiguously. Splitting the query heads in half therefore
    splits the kv heads in half too -- chip c needs exactly kv heads [c*KV/TP, (c+1)*KV/TP), and no
    cross-chip traffic is needed to build the scores.
  * q_norm / k_norm scale head_dim, not the sharded axis, so they stay REPLICATED -- as do the rope
    cos/sin tables and the additive attention mask.
  * o_proj is the projection that reduces back to model dim, so it is ROW-parallel: its INPUT feature
    axis (H * head_dim) is split to match the query heads each chip owns, every chip produces a
    partial sum over the full model dim, and one all_reduce turns those partials into the golden
    output on every chip.

Everything from the first matmul to the collective runs on device; the only host work is one-time
weight preparation in `build` plus staging the per-call rope/mask tensors with `ttnn.from_torch`.
"""
from __future__ import annotations

import ttnn

TILE = 32


def _num_devices(device) -> int:
    fn = getattr(device, "get_num_devices", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            pass
    ids = getattr(device, "get_device_ids", None)
    if callable(ids):
        try:
            return len(ids()) or 1
        except Exception:
            pass
    return 1


def _to_device(host_tensor, device, *, mesh_mapper=None, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    kwargs = dict(dtype=dtype, layout=layout, device=device)
    if mesh_mapper is not None:
        kwargs["mesh_mapper"] = mesh_mapper
    return ttnn.from_torch(host_tensor.contiguous(), **kwargs)


def _replicate_mapper(device, num_devices):
    if num_devices <= 1:
        return None
    try:
        return ttnn.ReplicateTensorToMesh(device)
    except (AttributeError, TypeError):
        return None


def _shard_mapper(device, num_devices, dim):
    """Split a weight across the TP chips on `dim`. Returns None when there is nothing to split."""
    if num_devices <= 1:
        return None
    return ttnn.ShardTensorToMesh(device, dim=dim)


def _tile_on_heads(tensor, times):
    """Replicate a (1, n, S, X) tensor `times` along the head axis."""
    if times == 1:
        return tensor
    return ttnn.concat([tensor] * times, dim=1)


def _head_to_batch(flat, num_heads, width, seq, *, stride=None, offset=0):
    """(1, 1, S, num_heads*stride) -> (1, num_heads, S, width), taking `width` cols per head.

    Slicing the feature axis head by head and stacking on dim 1 keeps every intermediate
    tile-aligned in both of its last two dims, which a reshape-then-permute round trip would not.
    """
    stride = width if stride is None else stride
    heads = [
        ttnn.slice(flat, [0, 0, 0, h * stride + offset], [1, 1, seq, h * stride + offset + width])
        for h in range(num_heads)
    ]
    return ttnn.concat(heads, dim=1) if num_heads > 1 else heads[0]


def _batch_to_feature(x, num_heads, width, seq):
    """(1, num_heads, S, width) -> (1, 1, S, num_heads*width), head-major (HF's `.reshape(*, -1)`)."""
    if num_heads == 1:
        return x
    heads = [ttnn.slice(x, [0, h, 0, 0], [1, h + 1, seq, width]) for h in range(num_heads)]
    return ttnn.concat(heads, dim=-1)


class TtQwen3NextAttention:
    """Native ttnn Qwen3-Next attention, sharded head-wise over the TP mesh."""

    def __init__(
        self,
        device,
        *,
        wq,
        wk,
        wv,
        wo,
        q_gamma,
        k_gamma,
        hidden_size,
        head_dim,
        local_num_heads,
        local_num_kv_heads,
        eps,
        scaling,
        tp=None,
    ) -> None:
        self.device = device
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_gamma = q_gamma
        self.k_gamma = k_gamma
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.local_num_heads = local_num_heads
        self.local_num_kv_heads = local_num_kv_heads
        self.n_rep = local_num_heads // local_num_kv_heads
        self.eps = eps
        self.scaling = scaling
        self.num_devices = _num_devices(device)
        # Effective tensor-parallel degree actually used to SHARD the weights. It is not
        # always num_devices: `build` falls back to tp=1 (replicated) when the head counts
        # do not divide the mesh. The o_proj all_reduce below must key off THIS, not the
        # chip count -- replicated weights produce a COMPLETE result per chip, so reducing
        # them would sum N identical copies and scale the output by N.
        # Defaults to num_devices so any caller that does not pass tp keeps the old path.
        self.tp = self.num_devices if tp is None else max(int(tp), 1)
        self._replicate = _replicate_mapper(device, self.num_devices)
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("attention stub needs the torch reference module to source its weights")

        sd = torch_module.state_dict()
        wq_t = sd["q_proj.weight"].detach().float()  # (H * 2 * head_dim, hidden)
        wk_t = sd["k_proj.weight"].detach().float()  # (KV * head_dim, hidden)
        wv_t = sd["v_proj.weight"].detach().float()
        wo_t = sd["o_proj.weight"].detach().float()  # (hidden, H * head_dim)
        q_norm_w = sd["q_norm.weight"].detach().float()
        k_norm_w = sd["k_norm.weight"].detach().float()

        head_dim = int(q_norm_w.shape[0])
        hidden_size = int(wq_t.shape[1])
        num_heads = int(wq_t.shape[0]) // (2 * head_dim)
        num_kv_heads = int(wk_t.shape[0]) // head_dim
        eps = float(getattr(getattr(torch_module, "q_norm", None), "eps", 1e-6))
        scaling = float(getattr(torch_module, "scaling", head_dim**-0.5))

        num_devices = _num_devices(device)
        # Head-wise TP needs both head axes to divide evenly; anything else would strand a head's
        # query on one chip and its kv on another, so fall back to a replicated (TP=1) placement
        # rather than silently computing the wrong thing.
        tp = num_devices if (num_heads % num_devices == 0 and num_kv_heads % num_devices == 0) else 1

        shard_out = _shard_mapper(device, tp, -1)  # column-parallel: split the output features
        shard_in = _shard_mapper(device, tp, -2)  # row-parallel: split the input features
        replicate = _replicate_mapper(device, num_devices)

        def _as_matmul_weight(w):
            # torch nn.Linear stores (out, in); ttnn matmul wants (in, out).
            return w.t().unsqueeze(0).unsqueeze(0).contiguous()

        wq = _to_device(_as_matmul_weight(wq_t), device, mesh_mapper=shard_out)
        wk = _to_device(_as_matmul_weight(wk_t), device, mesh_mapper=shard_out)
        wv = _to_device(_as_matmul_weight(wv_t), device, mesh_mapper=shard_out)
        wo = _to_device(_as_matmul_weight(wo_t), device, mesh_mapper=shard_in)

        # Qwen3NextRMSNorm scales by (1 + weight); fold the +1 in once here so the device op is a
        # plain weighted rms_norm. `ttnn.rms_norm` wants its gamma row-major as [1, 1, D/32, 32]
        # (same staging models/common/rmsnorm.py uses).
        def _as_gamma(w):
            return (1.0 + w).view(1, 1, head_dim // TILE, TILE).contiguous()

        q_gamma = _to_device(_as_gamma(q_norm_w), device, mesh_mapper=replicate, layout=ttnn.ROW_MAJOR_LAYOUT)
        k_gamma = _to_device(_as_gamma(k_norm_w), device, mesh_mapper=replicate, layout=ttnn.ROW_MAJOR_LAYOUT)

        return cls(
            device,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_gamma=q_gamma,
            k_gamma=k_gamma,
            hidden_size=hidden_size,
            head_dim=head_dim,
            local_num_heads=num_heads // tp,
            local_num_kv_heads=max(num_kv_heads // tp, 1),
            eps=eps,
            scaling=scaling,
            tp=tp,
        )

    # -------------------------------------------------------------- helpers

    def _stage(self, host_tensor):
        return _to_device(host_tensor, self.device, mesh_mapper=self._replicate)

    def _as_resident(self, tensor, seq, width):
        """Accept a caller-side constant either already ON DEVICE or as a host torch tensor.

        The pipeline keeps rope tables and the causal mask in persistent device buffers, so the
        device branch is what the traced forward takes; the host branch exists for the
        per-component PCC harness, which hands over the golden's torch tensors.
        """
        if isinstance(tensor, ttnn.Tensor):
            return ttnn.reshape(tensor, (1, 1, seq, width))
        return self._stage(tensor.reshape(1, 1, seq, width).float())

    def _apply_rope(self, x, cos, sin, rotary_dim, num_heads, seq):
        """rope on q/k: rotate the leading `rotary_dim` channels, pass the rest through untouched."""
        head_dim = self.head_dim
        rotated = x
        passthrough = None
        if rotary_dim < head_dim:
            rotated = ttnn.slice(x, [0, 0, 0, 0], [1, num_heads, seq, rotary_dim])
            passthrough = ttnn.slice(x, [0, 0, 0, rotary_dim], [1, num_heads, seq, head_dim])

        half = rotary_dim // 2
        first = ttnn.slice(rotated, [0, 0, 0, 0], [1, num_heads, seq, half])
        second = ttnn.slice(rotated, [0, 0, 0, half], [1, num_heads, seq, rotary_dim])
        rotate_half = ttnn.concat([ttnn.neg(second), first], dim=-1)

        cos_h = _tile_on_heads(cos, num_heads)
        sin_h = _tile_on_heads(sin, num_heads)
        embed = ttnn.add(ttnn.multiply(rotated, cos_h), ttnn.multiply(rotate_half, sin_h))
        if passthrough is None:
            return embed
        return ttnn.concat([embed, passthrough], dim=-1)

    def _repeat_kv(self, x, seq):
        """Expand each local kv head `n_rep` times, contiguously -- matches HF `repeat_kv`."""
        if self.n_rep == 1:
            return x
        if self.local_num_kv_heads == 1:
            return _tile_on_heads(x, self.n_rep)
        pieces = []
        for j in range(self.local_num_kv_heads):
            head = ttnn.slice(x, [0, j, 0, 0], [1, j + 1, seq, self.head_dim])
            pieces.extend([head] * self.n_rep)
        return ttnn.concat(pieces, dim=1)

    # -------------------------------------------------------------- forward

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        head_dim = self.head_dim
        local_h = self.local_num_heads
        local_kv = self.local_num_kv_heads

        seq = int(hidden_states.shape[-2])
        x = ttnn.reshape(hidden_states, (1, 1, seq, self.hidden_size))

        # --- column-parallel projections: each chip owns a disjoint slice of the head axis ---
        qg = ttnn.linear(x, self.wq, compute_kernel_config=self.compute_config)
        query = _head_to_batch(qg, local_h, head_dim, seq, stride=2 * head_dim)
        gate = _head_to_batch(qg, local_h, head_dim, seq, stride=2 * head_dim, offset=head_dim)
        gate = _batch_to_feature(gate, local_h, head_dim, seq)
        query = ttnn.rms_norm(
            query, weight=self.q_gamma, epsilon=self.eps, compute_kernel_config=self.compute_config
        )

        k_flat = ttnn.linear(x, self.wk, compute_kernel_config=self.compute_config)
        key = _head_to_batch(k_flat, local_kv, head_dim, seq)
        key = ttnn.rms_norm(key, weight=self.k_gamma, epsilon=self.eps, compute_kernel_config=self.compute_config)

        v_flat = ttnn.linear(x, self.wv, compute_kernel_config=self.compute_config)
        value = _head_to_batch(v_flat, local_kv, head_dim, seq)

        # --- rope, from the replicated cos/sin tables the caller hands down ---
        # In the real pipeline these arrive as the graduated `rotary_embedding` stub's DEVICE
        # output, so nothing crosses the host bus here.  The torch branch is the per-component PCC
        # harness, which replays the golden's tables.
        if position_embeddings is not None:
            cos_in, sin_in = position_embeddings
            rotary_dim = int(cos_in.shape[-1])
            cos = self._as_resident(cos_in, seq, rotary_dim)
            sin = self._as_resident(sin_in, seq, rotary_dim)
            query = self._apply_rope(query, cos, sin, rotary_dim, local_h, seq)
            key = self._apply_rope(key, cos, sin, rotary_dim, local_kv, seq)

        # --- GQA attention over this chip's heads only ---
        key = self._repeat_kv(key, seq)
        value = self._repeat_kv(value, seq)

        scores = ttnn.matmul(
            query, ttnn.transpose(key, -2, -1), compute_kernel_config=self.compute_config
        )
        scores = ttnn.multiply(scores, self.scaling)

        # The additive 4-D mask is a replicated per-element tensor: it applies identically to every
        # chip's heads. A 2-D "all ones" padding mask instead adds the same constant to every logit
        # in a row, which softmax cancels, so it is skipped rather than broadcast into the wrong axis.
        if attention_mask is not None and len(attention_mask.shape) == 4:
            kv_len = int(attention_mask.shape[-1])
            mask = self._as_resident(attention_mask, seq, kv_len)
            scores = ttnn.add(scores, _tile_on_heads(mask, local_h))

        probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_config)
        context = ttnn.matmul(probs, value, compute_kernel_config=self.compute_config)
        context = _batch_to_feature(context, local_h, head_dim, seq)

        context = ttnn.multiply(context, ttnn.sigmoid(gate))

        # --- row-parallel o_proj: partial sums over the full model dim, then one all_reduce ---
        partial = ttnn.linear(context, self.wo, compute_kernel_config=self.compute_config)
        if self.tp > 1:
            partial = ttnn.all_reduce(partial)

        return ttnn.reshape(partial, (1, seq, self.hidden_size))


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtQwen3NextAttention.build(device, torch_module)


# Module-level shim with the component's lowercase slug name, kept for legacy SMOKE/PCC tests.
def attention(device, torch_module=None):
    return TtQwen3NextAttention.build(device, torch_module)

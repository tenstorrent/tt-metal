# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `attention` (Qwen3Attention) for FLUX.2-klein-9B's text encoder.

Tensor-parallel over the whole mesh, Megatron-style, derived from
`models/tt_transformers/tt/attention.py`:

  * q/k/v are COLUMN-parallel -- their outputs feed per-head work (q_norm/k_norm,
    RoPE, softmax), so each chip owns a disjoint slice of the head axis and needs
    no collective to compute it.
  * o_proj is ROW-parallel -- it reduces the head axis back to the model dim, so
    each chip owns the rows matching its own heads and produces a PARTIAL sum;
    one `ttnn.all_reduce` at the end makes the result whole and identical on
    every chip.
  * q_norm / k_norm gammas are per-head-dim (128 elements) and stay REPLICATED,
    as does everything else that isn't a big matmul weight.

The head split is exact for this config: 32 q heads / 8 kv heads over TP=8 gives
4 q heads and 1 kv head per chip, and GQA maps q head `h` to kv head `h // 4`, so
chip `d`'s q heads (4d..4d+3) all want kv head `d` -- the one it already holds.
Attention itself therefore needs no cross-chip traffic.

q/k/v are packed into one fused per-chip QKV weight laid out as
`[q_local | k_local | v_local]` so a single `ShardTensorToMesh(dim=-1)` produces
exactly that, and `nlp_create_qkv_heads` can split it in one op.

The math is unchanged from the torch reference; only placement differs, so the
gathered output still matches the single-device golden.
"""
from __future__ import annotations

import torch

import ttnn

from .r_m_s_norm import TtRMSNorm


class TtAttention:
    def __init__(
        self,
        mesh_device,
        wqkv,
        wo,
        q_norm,
        k_norm,
        n_local_heads,
        n_local_kv_heads,
        head_dim,
        hidden_size,
        norm_eps,
        scaling,
        num_devices,
    ) -> None:
        self.mesh_device = mesh_device
        self.wqkv = wqkv
        self.wo = wo
        # The per-head q/k norms ARE the `r_m_s_norm` component (it is bound to
        # `model.layers.0.self_attn.q_norm`), so they are that graduated body rather
        # than a second inline copy of the same one-call op.
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.n_local_heads = n_local_heads
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.norm_eps = norm_eps
        self.scaling = scaling
        self.num_devices = num_devices
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("attention stub needs the torch module to source its weights")

        cfg = torch_module.config
        hidden_size = torch_module.q_proj.in_features
        head_dim = torch_module.head_dim
        n_heads = cfg.num_attention_heads
        n_kv_heads = cfg.num_key_value_heads

        num_devices = _num_devices(device)
        if n_heads % num_devices or n_kv_heads % num_devices:
            raise RuntimeError(
                f"attention TP needs both head counts divisible by the mesh size: "
                f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, devices={num_devices}"
            )
        n_local_heads = n_heads // num_devices
        n_local_kv_heads = n_kv_heads // num_devices

        sd = {k: v.detach().to(torch.float32) for k, v in torch_module.state_dict().items()}

        # torch nn.Linear stores [out, in]; ttnn matmuls x @ W want [in, out].
        wq = sd["q_proj.weight"].T.reshape(hidden_size, n_heads, head_dim)
        wk = sd["k_proj.weight"].T.reshape(hidden_size, n_kv_heads, head_dim)
        wv = sd["v_proj.weight"].T.reshape(hidden_size, n_kv_heads, head_dim)

        # Fuse per chip as [q_local | k_local | v_local] so ShardTensorToMesh(dim=-1) hands each
        # chip exactly its own slice, in the layout nlp_create_qkv_heads expects.
        per_chip = []
        for d in range(num_devices):
            q_lo, kv_lo = d * n_local_heads, d * n_local_kv_heads
            per_chip.append(
                torch.cat(
                    [
                        wq[:, q_lo : q_lo + n_local_heads, :].reshape(hidden_size, -1),
                        wk[:, kv_lo : kv_lo + n_local_kv_heads, :].reshape(hidden_size, -1),
                        wv[:, kv_lo : kv_lo + n_local_kv_heads, :].reshape(hidden_size, -1),
                    ],
                    dim=-1,
                )
            )
        wqkv_torch = torch.cat(per_chip, dim=-1)

        # o_proj is row-parallel: split its INPUT features, which are the head axis, the same way
        # q was split, so chip d's rows line up with the heads chip d just computed.
        wo_torch = sd["o_proj.weight"].T.contiguous()

        wqkv = ttnn.from_torch(
            wqkv_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_shard_mapper(device, num_devices, dim=-1),
        )
        wo = ttnn.from_torch(
            wo_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_shard_mapper(device, num_devices, dim=0),
        )

        # Qwen3's per-head q/k RMSNorm IS the graduated `r_m_s_norm` component --
        # its own binding is `model.layers.0.self_attn.q_norm`. Build that body here
        # instead of re-deriving its gamma layout, so there is ONE port of the op.
        # Gamma is per-head-dim (128 elements) and stays REPLICATED either way.
        q_norm = TtRMSNorm.build(device, torch_module.q_norm)
        k_norm = TtRMSNorm.build(device, torch_module.k_norm)

        return cls(
            mesh_device=device,
            wqkv=wqkv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            n_local_heads=n_local_heads,
            n_local_kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            norm_eps=torch_module.q_norm.variance_epsilon,
            scaling=torch_module.scaling,
            num_devices=num_devices,
        )

    # -------------------------------------------------------------- forward

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        kv_cache=None,
        cur_pos=None,
        mode="prefill",
        is_causal=False,
        **kwargs,
    ):
        """One attention sublayer.

        `mode="prefill"` is the graduated path, unchanged: [B, 1, S, H] in, all S
        positions at once. `mode="decode"` is the additive autoregressive step --
        ONE token read against a RESIDENT KV cache. The TP placement is identical in
        both: the cache is indexed by this chip's OWN kv heads, which are already the
        sharded axis, so the cache inherits the graduated split for free and decode
        needs exactly the same single all_reduce after o_proj.
        """
        if mode == "decode":
            return self._decode(hidden_states, position_embeddings, kv_cache, cur_pos)
        return self._prefill(hidden_states, position_embeddings, attention_mask, kv_cache, is_causal)

    # ------------------------------------------------------------ prefill

    def _prefill(self, hidden_states, position_embeddings, attention_mask, kv_cache, is_causal):
        x = hidden_states
        seq_len = int(x.shape[-2])
        batch = int(x.shape[0]) if len(x.shape) == 4 else 1
        x = ttnn.reshape(x, (batch, 1, seq_len, self.hidden_size))

        # ---- column-parallel QKV: local heads only, no collective needed.
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # ---- Qwen3's per-head q/k RMSNorm, over the head_dim (the last axis here).
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ---- RoPE, from the (cos, sin) the caller supplies. rotary_embedding_hf is the
        # rotate_half/HF convention, matching apply_rotary_pos_emb in the reference.
        if position_embeddings is not None:
            cos, sin = self._rope_tables(position_embeddings, seq_len)
            q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
            k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)

        # ---- seed the resident KV cache, if the caller keeps one. k/v here are
        # [B, n_local_kv_heads, S, head_dim] -- already this chip's own kv heads --
        # so the cache is sharded exactly like the heads that fill it.
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            for b in range(batch):
                k_b = (
                    k
                    if batch == 1
                    else ttnn.slice(k, (b, 0, 0, 0), (b + 1, self.n_local_kv_heads, seq_len, self.head_dim))
                )
                v_b = (
                    v
                    if batch == 1
                    else ttnn.slice(v, (b, 0, 0, 0), (b + 1, self.n_local_kv_heads, seq_len, self.head_dim))
                )
                ttnn.fill_cache(k_cache, k_b, b)
                ttnn.fill_cache(v_cache, v_b, b)

        # ---- attention over this chip's own heads; GQA is handled inside SDPA.
        mask = self._attn_bias(attention_mask, batch, seq_len)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=bool(is_causal) and mask is None,
            scale=self.scaling,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # ---- row-parallel o_proj: every chip produces a PARTIAL sum over the full model dim...
        out = ttnn.linear(
            attn,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn)

        # ...and the all_reduce turns those partials into the whole answer, on every chip.
        if self.num_devices > 1:
            out = ttnn.all_reduce(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return ttnn.reshape(out, (batch, seq_len, self.hidden_size))

    # ------------------------------------------------------------- decode

    def _decode(self, hidden_states, position_embeddings, kv_cache, cur_pos):
        """One autoregressive token against the resident cache.

        Layout is the decode-mode convention: [1, 1, batch, hidden] in,
        [1, batch, heads, head_dim] through the head ops, [1, 1, batch, hidden] out.
        """
        if kv_cache is None or cur_pos is None:
            raise RuntimeError("decode mode needs a resident kv_cache and a cur_pos tensor")

        in_shape = list(hidden_states.shape)
        batch = int(in_shape[-2])
        x = ttnn.reshape(hidden_states, (1, 1, batch, self.hidden_size))

        # ---- column-parallel QKV: local heads only, no collective needed.
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
        )
        ttnn.deallocate(xqkv)

        # The create-heads op hands back height-sharded tensors; the norm and the
        # rotation want them interleaved, and `paged_update_cache` wants them sharded
        # again, so keep the op's own memory config and restore it afterwards.
        kv_mem = k.memory_config()
        q = ttnn.sharded_to_interleaved(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.sharded_to_interleaved(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.sharded_to_interleaved(v, ttnn.DRAM_MEMORY_CONFIG)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if position_embeddings is not None:
            cos, sin = self._rope_tables_decode(position_embeddings)
            q = self._rotate(q, cos, sin)
            k = self._rotate(k, cos, sin)

        # ---- write this token's k/v into the resident cache, then read the whole
        # prefix back out of it. `cur_pos` is a DEVICE tensor, so the slot index is
        # data, not a Python constant baked into the program.
        k_cache, v_cache = kv_cache
        ttnn.experimental.paged_update_cache(
            k_cache, ttnn.interleaved_to_sharded(k, kv_mem), update_idxs_tensor=cur_pos
        )
        ttnn.experimental.paged_update_cache(
            v_cache, ttnn.interleaved_to_sharded(v, kv_mem), update_idxs_tensor=cur_pos
        )
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=cur_pos,
            scale=self.scaling,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)

        # [1, batch, n_local_heads, head_dim] -> [1, 1, batch, n_local_heads * head_dim].
        # Batch is the outer axis of both layouts, so this concat-heads IS a reshape
        # (verified bit-exact against nlp_concat_heads_decode on device).
        attn = ttnn.reshape(attn, (1, 1, batch, self.n_local_heads * self.head_dim))

        # ---- row-parallel o_proj + the one all_reduce, exactly as in prefill.
        out = ttnn.linear(
            attn,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn)
        if self.num_devices > 1:
            out = ttnn.all_reduce(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, (1, 1, batch, self.hidden_size))

    def _rope_tables_decode(self, position_embeddings):
        """(cos, sin) for ONE position, as [1, 1, 1, head_dim] device tensors."""
        cos, sin = position_embeddings
        out = []
        for t in (cos, sin):
            if not isinstance(t, ttnn.Tensor):
                raise RuntimeError(
                    "decode mode needs device-resident (cos, sin); host tables would be a host op in the hot path"
                )
            out.append(ttnn.reshape(t, (1, 1, 1, self.head_dim)))
        return out[0], out[1]

    def _rotate(self, t, cos, sin):
        """HF's `apply_rotary_pos_emb`: t * cos + rotate_half(t) * sin.

        Written out rather than calling `rotary_embedding_hf`, because the decode
        layout is [1, batch, heads, head_dim] and the fused op's decode mode expects
        height-sharded operands. `ttnn.experimental.rotate_half` is bit-identical to
        the reference's `cat(-x2, x1)` (measured max error 0.0 on device), and cos/sin
        broadcast over the batch and head axes.
        """
        return ttnn.add(ttnn.multiply(t, cos), ttnn.multiply(ttnn.experimental.rotate_half(t), sin))

    # -------------------------------------------------------------- helpers

    def allocate_kv_cache(self, batch, capacity, dtype=ttnn.bfloat16):
        """A resident [batch, n_local_kv_heads, capacity, head_dim] K/V pair.

        The kv-head axis is the axis TP already split, so each chip allocates only
        its OWN kv heads and the cache needs no collective of its own."""
        zeros = torch.zeros(batch, self.n_local_kv_heads, capacity, self.head_dim, dtype=torch.bfloat16)
        return tuple(
            ttnn.from_torch(
                zeros,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate_mapper(self.mesh_device, self.num_devices),
            )
            for _ in range(2)
        )

    def _rope_tables(self, position_embeddings, seq_len):
        """(cos, sin) -> replicated [1, 1, seq_len, head_dim] device tensors.

        Already-on-device tables are passed straight through -- that is the path the
        e2e pipeline uses, where `rotary_embedding` built them on device and uploading
        anything here would be a host op in the hot path. The torch branch below is the
        per-component harness's path, which hands the captured golden tables as torch.
        """
        cos, sin = position_embeddings
        if isinstance(cos, ttnn.Tensor):
            return (
                ttnn.reshape(cos, (1, 1, seq_len, self.head_dim)),
                ttnn.reshape(sin, (1, 1, seq_len, self.head_dim)),
            )
        out = []
        for t in (cos, sin):
            t = t.detach().to(torch.float32).reshape(1, 1, -1, self.head_dim)[:, :, :seq_len, :]
            out.append(
                ttnn.from_torch(
                    t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=_replicate_mapper(self.mesh_device, self.num_devices),
                )
            )
        return out[0], out[1]

    def _attn_bias(self, attention_mask, batch, seq_len):
        """The additive [b, 1, Sq, Sk] bias SDPA wants, or None when unmasked.

        The reference hands its `attention_mask` straight to the attn implementation as a bias, so
        this port applies whatever it is given rather than assuming a causal shape of its own.
        """
        if attention_mask is None:
            return None
        if isinstance(attention_mask, ttnn.Tensor):
            # Already a device-resident additive bias -- the e2e pipeline's path.
            return attention_mask
        if not isinstance(attention_mask, torch.Tensor):
            return attention_mask
        mask = attention_mask.detach()
        if mask.dtype == torch.bool:
            # bool means "attend here"; SDPA's bias form wants 0 / big-negative.
            mask = torch.where(mask, 0.0, -1e9)
        mask = mask.to(torch.float32)
        if mask.dim() != 4:
            # Anything that isn't already a bias can't be broadcast into one unambiguously.
            raise RuntimeError(f"attention_mask must be a 4-D attention bias, got shape {tuple(mask.shape)}")
        # bf16 saturates well before -inf/finfo.min; clamp so masked logits stay finite.
        mask = mask.clamp(min=-1e9).expand(batch, -1, seq_len, seq_len).contiguous()
        return ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(self.mesh_device, self.num_devices),
        )


def _num_devices(device):
    try:
        return int(device.get_num_devices())
    except AttributeError:
        return 1


def _shard_mapper(device, num_devices, dim):
    if num_devices <= 1:
        return None
    return ttnn.ShardTensorToMesh(device, dim=dim)


def _replicate_mapper(device, num_devices):
    if num_devices <= 1:
        return None
    return ttnn.ReplicateTensorToMesh(device)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtAttention.build(device, torch_module)


# Module-level shim with the component's lowercase slug name, for legacy SMOKE/PCC tests.
def attention(device, torch_module=None):
    return TtAttention.build(device, torch_module)

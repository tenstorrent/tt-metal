# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `attention` (`MistralAttention`) for
`/localdev/lserbedzija/hf_models/voxtral-tts-backbone`.

GQA self-attention: 32 query heads / 8 KV heads, head_dim 128, hidden 3072,
no biases, RoPE applied to Q and K, additive causal mask.

The canonical `models/tt_transformers/tt/attention.py::Attention` is not usable
as-is from this per-component harness: it is built around `ModelArgs` (a full
checkpoint config + tokenizer + mesh/CCL plumbing) and its forward takes
`(x, current_pos, rot_mats, user_id, mode, page_table, ...)` with its own KV
cache, not the `(hidden_states, position_embeddings, attention_mask)` triple the
HF golden defines. So the forward below is written directly against the same
ttnn primitives that module uses.

`__init__`/`build` stage weights with torch (allowed — the weights come from an
HF checkpoint). `__call__` is pure ttnn: `models/common/native_probe.py` counts
what actually executes and a single torch op in the forward would (correctly)
disqualify this as a host reimplementation.
"""
from __future__ import annotations

import torch
import ttnn

from models.demos.voxtral_tts_backbone._stubs.decode_matmul import build_plan


def _is_mesh_device(device) -> bool:
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


def _replicate_mapper(device):
    if not _is_mesh_device(device):
        return None
    try:
        return ttnn.ReplicateTensorToMesh(device)
    except (AttributeError, TypeError):
        return None


def _stage(weight: torch.Tensor, device):
    """Stage an `nn.Linear` weight (out, in) as a ttnn matmul operand (in, out)."""
    host = weight.detach().to(torch.float32).transpose(0, 1).contiguous().to(torch.bfloat16)
    mapper = _replicate_mapper(device)
    if mapper is not None:
        return ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mapper,
        )
    return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


_TILE = 32


def _kv_write_memory_config(n_kv_heads: int, head_dim: int):
    """L1 shard config for the single-token cache write operand.

    `ttnn.experimental.paged_update_cache` — the only cache write that takes its
    row index as a DEVICE tensor, which is what keeps a traced decode step free
    of host scalars — wants a sharded `[1, batch, n_kv_heads, head_dim]` input.
    Batch is 1 here, so one core holds the whole (tile-padded) operand.
    """
    rows = max(_TILE, ((int(n_kv_heads) + _TILE - 1) // _TILE) * _TILE)
    try:
        return ttnn.create_sharded_memory_config(
            shape=(rows, int(head_dim)),
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    except Exception:  # noqa: BLE001 - fall back to the int-index cache write
        return None


class TtAttention:
    def __init__(self, device, weights, dims, scaling):
        self.device = device
        self.wq, self.wk, self.wv, self.wo = weights
        self.n_heads, self.n_kv_heads, self.head_dim = dims
        self.scaling = scaling
        self.kv_write_memory_config = _kv_write_memory_config(self.n_kv_heads, self.head_dim)
        # K and V share a shape, so one plan serves both.
        self.kv_plan = build_plan(device, int(self.wk.shape[-2]), int(self.wk.shape[-1]))
        self.o_plan = build_plan(device, int(self.wo.shape[-2]), int(self.wo.shape[-1]))
        self.q_plan = build_plan(device, int(self.wq.shape[-2]), int(self.wq.shape[-1]))
        # HiFi4 + fp32 accumulate: the projections and the SDPA feed a 0.99 PCC
        # gate, and bf16 LoFi accumulation is what usually costs those digits.
        try:
            self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        except Exception:  # noqa: BLE001 - accuracy tuning is best-effort
            self.compute_kernel_config = None

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("attention build needs the HF MistralAttention module to read weights from")
        cfg = getattr(torch_module, "config", None)
        head_dim = getattr(torch_module, "head_dim", None) or getattr(cfg, "head_dim", None)
        n_heads = getattr(cfg, "num_attention_heads", None)
        n_kv_heads = getattr(cfg, "num_key_value_heads", None) or n_heads
        q_w = torch_module.q_proj.weight
        k_w = torch_module.k_proj.weight
        v_w = torch_module.v_proj.weight
        o_w = torch_module.o_proj.weight
        if head_dim is None:
            head_dim = q_w.shape[0] // int(n_heads)
        head_dim = int(head_dim)
        # Trust the weights over the config for the head counts: a per-component
        # harness may hand us a submodule whose config was edited.
        n_heads = int(q_w.shape[0]) // head_dim
        n_kv_heads = int(k_w.shape[0]) // head_dim
        scaling = getattr(torch_module, "scaling", None) or head_dim**-0.5
        weights = (_stage(q_w, device), _stage(k_w, device), _stage(v_w, device), _stage(o_w, device))
        return cls(device, weights, (n_heads, n_kv_heads, head_dim), float(scaling))

    # -------------------------------------------------------------- forward
    def _split_heads(self, proj, n_heads):
        """(1, s, n*hd) -> (1, n, s, hd)"""
        shape = list(proj.shape)
        batch, seq = shape[0], shape[-2]
        heads = ttnn.reshape(proj, (batch, seq, n_heads, self.head_dim))
        return ttnn.permute(heads, (0, 2, 1, 3))

    def _rotate_half(self, x):
        half = self.head_dim // 2
        shape = list(x.shape)
        first = ttnn.slice(x, [0, 0, 0, 0], [shape[0], shape[1], shape[2], half])
        second = ttnn.slice(x, [0, 0, 0, half], [shape[0], shape[1], shape[2], self.head_dim])
        return ttnn.concat([ttnn.neg(second), first], dim=-1)

    def _kv_proj(self, hidden_states, weight, mm):
        """K/V projection, on the full grid for the decode shape.

        These are the NARROWEST projections in the block (3072 -> 1024, only 32
        output tiles), which is exactly the case `ttnn.linear` routes worst: at
        decode it reached 118 GB/s, less than a quarter of the device's DRAM
        bandwidth and by far the lowest of any projection here.
        """
        if self.kv_plan is not None and self.kv_plan.matches(hidden_states):
            return self.kv_plan(hidden_states, weight, self.compute_kernel_config)
        return ttnn.linear(hidden_states, weight, **mm)

    def _apply_rope(self, x, cos, sin):
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(self._rotate_half(x), sin))

    def _as_broadcastable(self, t):
        """cos/sin arrive as (1, s, hd); SDPA-shaped tensors are (1, n, s, hd)."""
        shape = list(t.shape)
        if len(shape) == 4:
            return t
        if len(shape) == 3:
            return ttnn.reshape(t, (shape[0], 1, shape[1], shape[2]))
        raise RuntimeError(f"unexpected rotary embedding rank {len(shape)} (shape={shape})")

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        kv_cache=None,
        cache_fill=False,
        cache_pos=None,
        cache_pos_tensor=None,
        **_ignored,
    ):
        """Pure-ttnn GQA self-attention.

        `kv_cache`/`cache_fill`/`cache_pos`/`cache_pos_tensor` are OPTIONAL and
        default to None/False: with them absent this is bit-for-bit the
        graduated body the per-component PCC test pinned. They exist so the
        e2e pipeline can drive ONE attention body in both modes instead of
        forking a second one:

          * `kv_cache=(k_cache, v_cache), cache_fill=True`  -> prefill. The
            rotated K/V are written into the resident cache with
            `ttnn.fill_cache` and the unchanged full-sequence SDPA runs.
          * `kv_cache=(k_cache, v_cache)` plus `cache_pos=p` (a python row
            index) or `cache_pos_tensor=<int32 [1] device tensor>` (the resident
            index, which is what a traced step needs) -> one decode step. The
            rotated K/V for the single new token are written at that row and
            attention reads the resident cache via
            `ttnn.transformer.scaled_dot_product_attention_decode`, so the step
            is O(1) and never recomputes the prompt.

        Both branches are ttnn-only; no torch op is reachable from here.
        """
        if position_embeddings is None:
            raise RuntimeError(
                "attention forward needs position_embeddings=(cos, sin); the harness must pass the "
                "same pair it gave the HF reference"
            )
        cos, sin = position_embeddings
        cos = self._as_broadcastable(cos)
        sin = self._as_broadcastable(sin)

        mm = {"compute_kernel_config": self.compute_kernel_config} if self.compute_kernel_config else {}

        if self.q_plan is not None and self.q_plan.matches(hidden_states):
            query = self._split_heads(self.q_plan(hidden_states, self.wq, self.compute_kernel_config), self.n_heads)
        else:
            query = self._split_heads(ttnn.linear(hidden_states, self.wq, **mm), self.n_heads)
        key = self._split_heads(self._kv_proj(hidden_states, self.wk, mm), self.n_kv_heads)
        value = self._split_heads(self._kv_proj(hidden_states, self.wv, mm), self.n_kv_heads)

        query = self._apply_rope(query, cos, sin)
        key = self._apply_rope(key, cos, sin)

        if kv_cache is not None and (cache_pos is not None or cache_pos_tensor is not None):
            attn = self._attend_from_cache(query, key, value, kv_cache, cache_pos, cache_pos_tensor)
        else:
            if kv_cache is not None and cache_fill:
                k_cache, v_cache = kv_cache
                # [1, n_kv, s, hd] straight into the [1, n_kv, C, hd] cache;
                # rows [0:s) are written exactly, the rest stay as they were.
                ttnn.fill_cache(k_cache, key, 0)
                ttnn.fill_cache(v_cache, value, 0)
            # ttnn's SDPA is GQA-aware (nqh vs nkv), so no explicit repeat_kv.
            # The mask the harness hands us IS the causal mask the HF golden used,
            # so honour it instead of asking the kernel for its own.
            attn = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                scale=self.scaling,
                **mm,
            )
            attn = ttnn.permute(attn, (0, 2, 1, 3))

        shape = list(attn.shape)
        attn = ttnn.reshape(attn, (shape[0], shape[1], self.n_heads * self.head_dim))
        if self.o_plan is not None and self.o_plan.matches(attn):
            return self.o_plan(attn, self.wo, self.compute_kernel_config)
        return ttnn.linear(attn, self.wo, **mm)

    def _write_cache_row(self, cache, tensor, cache_pos, cache_pos_tensor):
        """Write ONE token's K or V into the resident cache.

        With a device index tensor the write goes through
        `ttnn.experimental.paged_update_cache`, which reads the row number out of
        a `[1]` int32 tensor — so a captured trace advances by itself instead of
        being pinned to whatever row was current at capture time. It wants the
        `[1, batch, n_kv, hd]` layout, sharded, hence the permute + reshard.
        With a python `cache_pos` the plain `ttnn.update_cache` takes the
        `[1, n_kv, 1, hd]` layout `_split_heads` already produced.
        """
        if cache_pos_tensor is not None and self.kv_write_memory_config is not None:
            row = ttnn.permute(tensor, (2, 0, 1, 3))
            ttnn.experimental.paged_update_cache(
                cache,
                ttnn.to_memory_config(row, self.kv_write_memory_config),
                update_idxs_tensor=cache_pos_tensor,
            )
        else:
            ttnn.update_cache(cache, tensor, cache_pos)

    def _attend_from_cache(self, query, key, value, kv_cache, cache_pos, cache_pos_tensor):
        """One decode step against the resident cache -> (1, 1, n_heads, hd).

        The decode SDPA kernel wants `[1, batch, n_q_heads, hd]`, hence the
        (2, 0, 1, 3) permute in and the (1, 0, 2, 3) permute back out.
        """
        k_cache, v_cache = kv_cache
        self._write_cache_row(k_cache, key, cache_pos, cache_pos_tensor)
        self._write_cache_row(v_cache, value, cache_pos, cache_pos_tensor)

        q_decode = ttnn.permute(query, (2, 0, 1, 3))
        if cache_pos_tensor is not None:
            pos_kwargs = {"cur_pos_tensor": cache_pos_tensor}
        else:
            pos_kwargs = {"cur_pos": [cache_pos]}
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode,
            k_cache,
            v_cache,
            scale=self.scaling,
            **pos_kwargs,
        )
        return ttnn.permute(attn, (1, 0, 2, 3))


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtAttention.build(device, torch_module)


# Legacy slug-named shim.
def attention(device, torch_module=None):
    return TtAttention.build(device, torch_module)

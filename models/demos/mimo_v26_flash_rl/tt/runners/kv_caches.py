# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The engine-owned served KV cache of MiMo (serving_contract.md "KV cache"): one ``LayerKV`` (tt/kv_cache.py) per
served layer, the same tensors the device model reads and writes (MimoModel(kv_caches=...)).

Per chip: key [slots, local_kv_heads, 7040, 192] and value [slots, local_kv_heads, 7040, 128] bfloat8_b, ND-sharded
DRAM [1, 1, 32, width] ROUND_ROBIN_1D (one 32-token record of one head contiguous in one bank); full layers hold 1 KV
head per chip, sliding layers 2. Sliding layers also own the unserved 192-wide attention-only value cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models.demos.common.prefill.adapter import KvCaches


def is_sliding(cfg: dict, layer: int) -> bool:
    return bool(cfg["hybrid_layer_pattern"][layer])


def kv_heads(cfg: dict, layer: int) -> int:
    return cfg["swa_num_key_value_heads"] if is_sliding(cfg, layer) else cfg["num_key_value_heads"]


def widths(cfg: dict, layer: int) -> dict:
    """{state name: row width} of one layer's served cache."""
    if is_sliding(cfg, layer):
        return {"key": cfg["swa_head_dim"], "value": cfg["swa_v_head_dim"]}
    return {"key": cfg["head_dim"], "value": cfg["v_head_dim"]}


@dataclass
class MimoKvCaches(KvCaches):
    """{global layer: LayerKV} of this rank's layers and the geometry the address table walks."""

    layers: dict = field(repr=False)
    cfg: dict = field(repr=False)
    num_users: int
    max_seq_len: int
    chunk_size: int
    sp: int
    tp: int

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, layer: int):
        return self.layers[layer]

    @property
    def layer_ids(self) -> list[int]:
        return sorted(self.layers)


def cache_capacity(max_seq_len: int, chunk_size: int) -> int:
    """At least two chunks: the sliding ring SDPA runs only its chunked path (a cache longer than the chunk's rows)."""
    return max(max_seq_len, 2 * chunk_size)


def allocate_kv_caches(mesh, cfg: dict, *, layers, num_users: int, max_seq_len: int, chunk_size: int) -> MimoKvCaches:
    """Allocate (on the device, zero-filled) the served caches of ``layers``, ``cache_capacity`` positions per slot."""
    from models.demos.mimo_v26_flash_rl.tt.kv_cache import LayerKV

    sp, tp = tuple(mesh.shape)
    max_seq_len = cache_capacity(max_seq_len, chunk_size)
    if max_seq_len % chunk_size or chunk_size % (sp * 32):
        raise ValueError(f"max_seq {max_seq_len} / chunk {chunk_size} do not split into 32-row SP slabs of {sp}")
    out = {}
    for layer in layers:
        heads = kv_heads(cfg, layer)
        if heads % tp:
            raise ValueError(f"layer {layer}: {heads} KV heads do not split over TP {tp}")
        w = widths(cfg, layer)
        out[layer] = LayerKV(
            mesh,
            num_users,
            heads // tp,
            max_seq_len // sp,
            w["key"],
            w["value"],
            attn_v_dim=w["key"] if is_sliding(cfg, layer) else None,
        )
    return MimoKvCaches(
        layers=out, cfg=cfg, num_users=num_users, max_seq_len=max_seq_len, chunk_size=chunk_size, sp=sp, tp=tp
    )

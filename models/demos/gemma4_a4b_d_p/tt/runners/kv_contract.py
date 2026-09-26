# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill-server KV-cache contract for Gemma-4 26B-A4B on a 1x4 mesh (sp = 1, tp = 4).

Gemma-4 has two layer types with different KV shapes, but the same per-chip width:
  * sliding (25 layers): 8 KV heads x 256, chip c holds heads 2c, 2c+1  -> 512 values per token per chip
  * global  (5 layers):  2 KV heads x 512, chip c holds head c // 2       -> 512 values per token per chip
So the migratable cache is one uniform slab per TP column ("what chip c holds"), for K and for V:

  * per chip K and V [num_users * num_layers, 1, max_seq, 512], bfloat8_b, TILE, DRAM NdShard [1, 1, 32, 512]
    ROUND_ROBIN_1D over the DRAM banks; batch index = slot * num_layers + layer (the gpt_oss_d_p GQA substrate)
  * sliding layer: columns [0:256] = head 2c, [256:512] = head 2c+1 (nlp_concat_heads order)
  * global layer:  columns [0:512] = head c // 2 (chips 0,1 hold head 0; chips 2,3 hold head 1: replicas)
  * K is post-k_norm, post-RoPE in rotate-half (HF) order, as the golden; V is the unscaled RMS norm (no RoPE)
  * address table: configs 0..3 = K chip 0..3, 4..7 = V chip 0..3; 32-token entries of 16 bf8 tiles (17408 B)

Written from each attention's kv_sink at the end of the layer's KV computation (the attention itself reads its own
bf16 per-user caches, tt.attention.TtKVCacheSliding / TtKVCacheGlobal).
"""

from __future__ import annotations

import torch

import ttnn

SP_AXIS = 0
NUM_CHIPS = 4
SLAB = 512  # per-chip KV width per token (both layer types)


class Gemma4ContractKV:
    def __init__(self, mesh, num_layers: int, max_seq: int, num_users: int = 1, dtype=ttnn.bfloat8_b):
        from models.demos.gpt_oss_d_p.tt.attention.kv_cache import allocate_kv_cache

        assert tuple(mesh.shape) == (1, NUM_CHIPS), f"Gemma-4 contract cache is built for 1x4, got {mesh.shape}"
        self.mesh, self.num_layers, self.max_seq, self.num_users = mesh, num_layers, max_seq, num_users
        self.cache = allocate_kv_cache(
            mesh,
            num_layers=num_layers,
            max_seq_len=max_seq,
            sp_axis=SP_AXIS,
            num_users=num_users,
            head_dim=SLAB,
            cache_dtype=dtype,
        )

    def sink(self, layer: int, start: int, slot: int):
        """kv_sink(k, v) for one (layer, chunk, slot): k/v are the attention's per-chip [1, h, S, D] with h * D = 512."""
        from models.demos.gpt_oss_d_p.tt.attention.kv_cache import write_kv_chunk

        def write(k, v):
            ks, vs = _slab(k), _slab(v)
            write_kv_chunk(self.cache, ks, vs, slot_idx=slot, layer_idx=layer, kv_actual=start, sp_axis=SP_AXIS)
            if ks is not k:
                ttnn.deallocate(ks)
                ttnn.deallocate(vs)

        return write

    def address_table(self, seq_len: int, chunk_size: int):
        from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table

        # "num_kv_heads" here is the number of per-chip slabs: config c (and 4 + c) lives on TP column c.
        return build_kv_chunk_address_table(
            mesh_device=self.mesh,
            kv_cache=self.cache,
            seq_len=seq_len,
            num_layers=self.num_layers,
            mesh_shape=list(self.mesh.shape),
            sp_axis=SP_AXIS,
            num_users=self.num_users,
            chunk_size=chunk_size,
            num_kv_heads=NUM_CHIPS,
            head_dim=SLAB,
        )


def _slab(t):
    """[1, h, S, D] -> [1, 1, S, h * D] (heads side by side); a single head is returned as is."""
    if t.shape[1] == 1:
        return t
    return ttnn.experimental.nlp_concat_heads(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# ------------------------------------------------------------------ read-back (host, device-less via the table)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def read_layer_kv(table, device_map, layer: int, slot: int, length: int, cfg) -> dict:
    """Read one layer's K/V back through the address table as the golden's [nkv, length, D] (global heads from their
    first replica) plus the raw per-chip slabs {'k': [4, length, 512], 'v': ...}."""
    from models.demos.common.prefill.runners import prefill_producer as producer

    read_len = -(-length // 32) * 32
    slabs = {}
    for kind, base in (("k", 0), ("v", NUM_CHIPS)):
        slabs[kind] = torch.stack(
            [
                producer._read_kv_slice(
                    table, device_map, base + c, layer, slot, read_len, SLAB, producer._decode_bfp8_chunk
                )[:length]
                for c in range(NUM_CHIPS)
            ]
        )
    hkv, d = cfg.attn_dims(layer)
    out = {}
    for kind, name in (("k", "key"), ("v", "value")):
        s = slabs[kind]  # [4, L, 512]
        if cfg.is_sliding(layer):
            out[name] = s.reshape(NUM_CHIPS, length, SLAB // d, d).transpose(1, 2).reshape(hkv, length, d)
        else:
            rep = NUM_CHIPS // hkv
            out[name] = s[::rep]  # [hkv, L, 512]
    out["slabs"] = slabs
    return out


def read_slot_kv_and_check_pcc(table, device_map: dict, slot_id: int, real_len: int, trace_dir, num_layers: int, cfg):
    """Min K / V PCC over [0, real_len) of every layer vs the bring-up golden (trace_dir/kv_cache/layer_i.safetensors,
    keys key_cache_layer_i / value_cache_layer_i, [nkv, S, D]). Global layers also check that both replicas agree."""
    from pathlib import Path

    from loguru import logger
    from safetensors import safe_open

    mins = {"k": 1.0, "v": 1.0}
    for layer in range(num_layers):
        dev = read_layer_kv(table, device_map, layer, slot_id, real_len, cfg)
        with safe_open(str(Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as f:
            gk = f.get_slice(f"key_cache_layer_{layer}")[:, :real_len].float()
            gv = f.get_slice(f"value_cache_layer_{layer}")[:, :real_len].float()
        pk, pv = _pcc(dev["key"], gk), _pcc(dev["value"], gv)
        if not cfg.is_sliding(layer):
            rep = NUM_CHIPS // cfg.attn_dims(layer)[0]
            for kind in ("k", "v"):
                s = dev["slabs"][kind]
                for c in range(NUM_CHIPS):
                    if not torch.equal(s[c], s[c - c % rep]):
                        raise RuntimeError(f"layer {layer} {kind}: chip {c} replica differs from chip {c - c % rep}")
        mins["k"], mins["v"] = min(mins["k"], pk), min(mins["v"], pv)
        logger.info(f"  layer {layer:>2} ({'sliding' if cfg.is_sliding(layer) else 'global'}): K={pk:.5f} V={pv:.5f}")
    logger.info(f"[gemma4] slot {slot_id} KV PCC over [0,{real_len}) -> K={mins['k']:.5f} V={mins['v']:.5f}")
    return mins

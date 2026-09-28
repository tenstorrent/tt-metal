# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill-server KV-cache contract for MiMo-V2.6-Flash-RL on a 1x4 mesh (sp = 1, tp = 4).

MiMo has two KV shapes per chip (tt/attention.py):
  * full layers (0, 5, ...):     4 KV heads x 192, chip c holds head c          -> K [1, 1, S, 192]
  * sliding layers (1-4, ...):   8 KV heads x 192, chip c holds heads 2c, 2c+1  -> K [1, 2, S, 192]
V has head dim 128, x attention_value_scale. The attention keeps V at 128 (tt/attention.py; MIMO_V_PAD=1 pads it to
192 there), but this contract keeps its layout with V zero-padded to 192 per head: the padding is added only where
the contract copy is written (kv_sink -> _slab), so the slab, the address table (one 12-tile entry per 32 tokens for
K and V alike, gpt_oss_d_p kv_cache / kv_chunk_table) and every reader of it are unchanged.
The migratable cache is one uniform slab per TP column ("what chip c holds"), for K and for V:

  * per chip K and V [num_users * num_layers, 1, max_seq, 384], bfloat8_b, TILE, DRAM NdShard [1, 1, 32, 384]
    ROUND_ROBIN_1D over the DRAM banks; batch index = slot * num_layers + layer (the gpt_oss_d_p GQA substrate)
  * sliding layer: columns [0:192] = head 2c, [192:384] = head 2c+1 (nlp_concat_heads order)
  * full layer:    columns [0:192] = head c, [192:384] = zero (device pad)
  * V columns of a head: [h*192 : h*192 + 128] (the last 64 of each 192 are zero, padded at the contract write)
  * K is post-RoPE in rotate-half (HF) order, as the golden; V is x attention_value_scale, as the golden
  * address table: configs 0..3 = K chip 0..3, 4..7 = V chip 0..3; 32-token entries of 12 bf8 tiles (13056 B)

Written from each attention's kv_sink right after the chunk's K/V are computed (the attention itself reads its own
bf16 per-user caches, tt.attention.TtKVCacheFull / TtKVCacheSliding). Table layer index = the layer's position in
the cache (== global layer index for a rank that starts at layer 0).
"""

from __future__ import annotations

import torch

import ttnn

SP_AXIS = 0
NUM_CHIPS = 4
HEAD_DIM = 192  # QK head dim, and the contract's padded V head dim
V_HEAD_DIM = 128
SLAB = 2 * HEAD_DIM  # per-chip KV width per token (both layer types: sliding holds 2 heads, full 1 head + zeros)


class MiMoContractKV:
    def __init__(self, mesh, num_layers: int, max_seq: int, num_users: int = 1, dtype=ttnn.bfloat8_b):
        from models.demos.gpt_oss_d_p.tt.attention.kv_cache import allocate_kv_cache

        assert tuple(mesh.shape) == (1, NUM_CHIPS), f"MiMo contract cache is built for 1x4, got {mesh.shape}"
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
        """kv_sink(k, v) for one (cache layer, chunk, slot): k is the attention's per-chip [1, h, S, 192], h in {1, 2},
        v [1, h, S, 128] (or [1, h, S, 192] zero-padded under MIMO_V_PAD=1)."""
        from models.demos.gpt_oss_d_p.tt.attention.kv_cache import write_kv_chunk

        def write(k, v):
            ks, vs = _slab(k), _slab(v)
            write_kv_chunk(self.cache, ks, vs, slot_idx=slot, layer_idx=layer, kv_actual=start, sp_axis=SP_AXIS)
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
    """[1, h, S, 192] -> [1, 1, S, 384]: two heads side by side (sliding), or one head then 192 zero columns (full).
    A V at its real width [1, h, S, 128] is zero-padded to 192 per head first."""
    h, d = t.shape[1], t.shape[-1]
    assert d in (HEAD_DIM, V_HEAD_DIM) and h in (1, 2), t.shape
    if h == 1:
        return ttnn.pad(t, padding=[(0, 0), (0, 0), (0, 0), (0, SLAB - d)], value=0.0)
    if d == HEAD_DIM:
        return ttnn.experimental.nlp_concat_heads(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    p = ttnn.pad(t, padding=[(0, 0), (0, 0), (0, 0), (0, HEAD_DIM - d)], value=0.0)
    out = ttnn.experimental.nlp_concat_heads(p, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(p)
    return out


# ------------------------------------------------------------------ read-back (host, device-less via the table)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def read_layer_kv(table, device_map, layer: int, slot: int, length: int, sliding: bool) -> dict:
    """Read one cache layer's K/V back through the address table as the golden's key [nkv, length, 192] and
    value [nkv, length, 128], plus the raw per-chip slabs {'k': [4, length, 384], 'v': ...}."""
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
    per_chip = 2 if sliding else 1
    heads = {
        kind: [s[c, :, j * HEAD_DIM : (j + 1) * HEAD_DIM] for c in range(NUM_CHIPS) for j in range(per_chip)]
        for kind, s in slabs.items()
    }
    return {
        "key": torch.stack(heads["k"]),
        "value": torch.stack([h[:, :V_HEAD_DIM] for h in heads["v"]]),
        "slabs": slabs,
    }


def read_slot_kv_and_check_pcc(table, device_map: dict, slot_id: int, real_len: int, trace_dir, num_layers: int, cfg):
    """Min K / V PCC over [0, real_len) of every cache layer vs the bring-up golden (trace_dir/kv_cache/layer_i
    .safetensors, keys key_cache_layer_i [nkv, S, 192] / value_cache_layer_i [nkv, S, 128]). Also checks that the pad
    columns (full-layer second half, each V head's last 64) read back as zero."""
    from pathlib import Path

    from loguru import logger
    from safetensors import safe_open

    n = int(num_layers)
    mins = {"k": 1.0, "v": 1.0}
    for layer in range(n):
        sliding = cfg.is_sliding(layer)
        dev = read_layer_kv(table, device_map, layer, slot_id, real_len, sliding)
        with safe_open(str(Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as f:
            gk = f.get_slice(f"key_cache_layer_{layer}")[:, :real_len].float()
            gv = f.get_slice(f"value_cache_layer_{layer}")[:, :real_len].float()
        assert dev["key"].shape == gk.shape and dev["value"].shape == gv.shape, (dev["key"].shape, gk.shape)
        s = dev["slabs"]
        pad_cols = [slice(HEAD_DIM, SLAB)] if not sliding else []
        pad_v = [slice(j * HEAD_DIM + V_HEAD_DIM, (j + 1) * HEAD_DIM) for j in range(2 if sliding else 1)]
        for kind, cols in (("k", pad_cols), ("v", pad_cols + pad_v)):
            for c in cols:
                if s[kind][..., c].abs().max() != 0:
                    raise RuntimeError(f"layer {layer} {kind}: pad columns {c} of the contract slab are not zero")
        pk, pv = _pcc(dev["key"], gk), _pcc(dev["value"], gv)
        mins["k"], mins["v"] = min(mins["k"], pk), min(mins["v"], pv)
        logger.info(f"  layer {layer:>2} ({'sliding' if sliding else 'full'}): K={pk:.5f} V={pv:.5f}")
    logger.info(
        f"[mimo] slot {slot_id} KV PCC over [0,{real_len}) of {n} layers -> K={mins['k']:.5f} V={mins['v']:.5f}"
    )
    return mins

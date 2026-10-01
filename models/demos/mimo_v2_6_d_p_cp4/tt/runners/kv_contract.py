# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill-server KV-cache contract for MiMo-V2.6-Flash-RL on a 1x4 mesh with context parallelism CP=4, TP=1.

Every chip holds all KV heads for its own positions (tt/attention.py): chip r holds rows [r*L, (r+1)*L) of each chunk
(L = C/4, C the chunk). The migratable cache follows the attention's chunk-major ring layout, one slab per chip for K
and one for V:

  * per chip K and V [num_users * num_layers, 1, max_seq/4, 1536], bfloat8_b, TILE, DRAM NdShard [1, 1, 32, 1536]
    ROUND_ROBIN_1D over the DRAM banks; batch index = slot * num_layers + layer (the gpt_oss_d_p GQA substrate)
  * local row n*L + j on chip r holds global position n*C + r*L + j (update_padded_kv_cache, cluster_axis=1,
    kv_actual_global = chunk start: the same write the attention's ring cache uses)
  * columns: head h at [h*192, (h+1)*192) (nlp_concat_heads order); sliding layers 8 heads, full layers 4 heads and
    768 zero columns
  * V: each head's first 128 of 192 columns (the attention's V is zero-padded to the QK head dim), x
    attention_value_scale; K post-RoPE in rotate-half (HF) order, as the golden
  * address table: config 0 = K, config 1 = V; a 32-token entry (48 bf8 tiles, 52224 B) for position p lives on chip
    (p % C) // L in a one-chip device group at MeshCoordinate(0, chip); the block-cyclic period is the served chunk

Written from the attention's ring-cache write (TtKVCacheRing.kv_sink, set per chunk by the runtime), right after the
chunk's K/V are computed; the attention itself reads its own bf16 per-user ring caches.
"""

from __future__ import annotations

import socket

import torch

import ttnn

CP_AXIS = 1
NUM_CHIPS = 4
HEAD_DIM = 192  # QK head dim, and the padded V head dim
V_HEAD_DIM = 128
MAX_KV_HEADS = 8  # sliding layers; full layers have 4
SLAB = MAX_KV_HEADS * HEAD_DIM  # 1536
BLOCK = 32  # tokens per DRAM bank shard / table entry
TILE_BYTES = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}


def chip_and_row(pos: int, chunk: int, cp: int = NUM_CHIPS) -> tuple[int, int]:
    """Global position -> (chip, local row) of the chunk-major layout."""
    L = chunk // cp
    n, off = divmod(pos, chunk)
    return off // L, n * L + off % L


class MiMoContractKVCP:
    def __init__(self, mesh, num_layers: int, max_seq: int, num_users: int = 1, dtype=ttnn.bfloat8_b):
        from models.demos.common.prefill.runners.migration import get_num_dram_banks

        assert tuple(mesh.shape) == (1, NUM_CHIPS), f"MiMo CP contract cache is built for 1x4, got {mesh.shape}"
        assert max_seq % (NUM_CHIPS * BLOCK) == 0, max_seq
        self.mesh, self.num_layers, self.max_seq, self.num_users, self.dtype = (
            mesh,
            num_layers,
            max_seq,
            num_users,
            dtype,
        )
        self.local = max_seq // NUM_CHIPS
        self.num_banks = get_num_dram_banks(mesh)
        banks = [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(self.num_banks)]
        spec = ttnn.NdShardSpec(
            shard_shape=[1, 1, BLOCK, SLAB],
            grid=ttnn.CoreRangeSet(banks),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)

        def alloc():
            # Same zeroed buffer on every chip; which positions a chip holds is decided by the CP write.
            return ttnn.from_torch(
                torch.zeros(num_users * num_layers, 1, self.local, SLAB),
                dtype=dtype,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self.k, self.v = alloc(), alloc()

    def sink(self, layer: int, start: int, slot: int):
        """kv_sink(k, v) for one (cache layer, chunk, slot): k, v per chip [1, nkv, L, 192] bf16 (chip r's CP slice,
        V zero-padded past 128), nkv in {4, 8}. k and v stay owned by the caller."""
        assert 0 <= slot < self.num_users and 0 <= layer < self.num_layers, (slot, layer)

        def write(k, v):
            for cache, t in ((self.k, k), (self.v, v)):
                s = _slab(t)
                src = ttnn.typecast(s, cache.dtype) if s.dtype != cache.dtype else s
                if src is not s:
                    ttnn.deallocate(s)
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    cache,
                    src,
                    slot_idx=slot,
                    layer_idx=layer,
                    num_layers=self.num_layers,
                    kv_actual_global=int(start),
                    cluster_axis=CP_AXIS,
                )
                ttnn.deallocate(src)

        return write

    def chunk_bytes(self) -> int:
        return (SLAB // 32) * TILE_BYTES[self.dtype]

    def address_table(self, seq_len: int, chunk_size: int):
        """Config 0 = K, 1 = V; one entry per (layer, 32-token position, slot), on the chip that holds the position."""
        from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import _make_config, _stable_config_name

        D = ttnn.experimental.disaggregation
        assert seq_len <= self.max_seq and seq_len % chunk_size == 0, (seq_len, chunk_size)
        assert chunk_size % (NUM_CHIPS * BLOCK) == 0, chunk_size
        specs = [self.k, self.v]
        n = len(specs)
        nbytes = self.chunk_bytes()
        table = D.KvChunkAddressTable(
            {
                _stable_config_name(i, n): _make_config(
                    num_layers=self.num_layers, max_seq_len=seq_len, num_users=self.num_users, chunk_size_bytes=nbytes
                )
                for i in range(n)
            }
        )
        for i in range(n):
            assert table.config_name(i) == _stable_config_name(i, n)
        host = socket.gethostname()
        seen = set()
        blocks_per_batch = self.local // BLOCK
        for cid, tensor in enumerate(specs):
            base = tensor.buffer_address()
            groups = []
            for r in range(NUM_CHIPS):
                fid = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, r))
                groups.append(table.add_device_group([fid]))
                key = (int(fid.mesh_id), int(fid.chip_id))
                if key not in seen:
                    table.set_fabric_node_host(fid, host_name=host)
                    seen.add(key)
            # ROUND_ROBIN_1D: shard j (row-major over [batch, local / 32]) lives in bank j % B at offset (j // B).
            for slot in range(self.num_users):
                for layer in range(self.num_layers):
                    b = slot * self.num_layers + layer
                    for pos in range(0, seq_len, BLOCK):
                        chip, row = chip_and_row(pos, chunk_size)
                        j = b * blocks_per_batch + row // BLOCK
                        loc = D.KvCacheLocation()
                        loc.noc_addr = ((j % self.num_banks) << 32) | (base + (j // self.num_banks) * nbytes)
                        loc.size_bytes = nbytes
                        loc.device_group_index = groups[chip]
                        table.set(layer, pos, slot, loc, cid)
        return table


def _slab(t):
    """[1, nkv, L, 192] -> [1, 1, L, 1536]: heads side by side, zero columns past nkv * 192."""
    h, d = t.shape[1], t.shape[-1]
    assert d == HEAD_DIM and h in (4, 8), t.shape
    out = ttnn.experimental.nlp_concat_heads(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if h * d < SLAB:
        p = ttnn.pad(out, padding=[(0, 0), (0, 0), (0, 0), (0, SLAB - h * d)], value=0.0)
        ttnn.deallocate(out)
        out = p
    return out


# ------------------------------------------------------------------ read-back (host, device-less via the table)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def read_layer_kv(table, device_map, layer: int, slot: int, length: int, nkv: int) -> dict:
    """One cache layer's K/V through the address table as the golden's key [nkv, length, 192] and value
    [nkv, length, 128], plus the raw slabs {'k': [length, 1536], 'v': ...}."""
    from models.demos.common.prefill.runners import prefill_producer as producer

    read_len = -(-length // BLOCK) * BLOCK
    slabs = {
        kind: producer._read_kv_slice(table, device_map, cid, layer, slot, read_len, SLAB, producer._decode_bfp8_chunk)[
            :length
        ].float()
        for kind, cid in (("k", 0), ("v", 1))
    }
    heads = {kind: s[:, : nkv * HEAD_DIM].reshape(length, nkv, HEAD_DIM).transpose(0, 1) for kind, s in slabs.items()}
    return {"key": heads["k"], "value": heads["v"][..., :V_HEAD_DIM], "slabs": slabs}


def read_slot_kv_and_check_pcc(table, device_map: dict, slot_id: int, real_len: int, trace_dir, num_layers: int, cfg):
    """Min K / V PCC over [0, real_len) of every cache layer vs the bring-up golden (trace_dir/kv_cache/layer_i
    .safetensors: key_cache_layer_i [nkv, S, 192], value_cache_layer_i [nkv, S, 128]). Also checks that the pad
    columns (full layers' heads 4..7, each V head's last 64) read back as zero."""
    from pathlib import Path

    from loguru import logger
    from safetensors import safe_open

    n = int(num_layers)
    mins = {"k": 1.0, "v": 1.0}
    for layer in range(n):
        nkv = cfg.attn_dims(layer)[1]
        dev = read_layer_kv(table, device_map, layer, slot_id, real_len, nkv)
        with safe_open(str(Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as f:
            gk = f.get_slice(f"key_cache_layer_{layer}")[:, :real_len].float()
            gv = f.get_slice(f"value_cache_layer_{layer}")[:, :real_len].float()
        assert dev["key"].shape == gk.shape and dev["value"].shape == gv.shape, (dev["key"].shape, gk.shape)
        s = dev["slabs"]
        if nkv * HEAD_DIM < SLAB:
            for kind in ("k", "v"):
                if s[kind][:, nkv * HEAD_DIM :].abs().max() != 0:
                    raise RuntimeError(f"layer {layer} {kind}: pad heads of the contract slab are not zero")
        vpad = s["v"][:, : nkv * HEAD_DIM].reshape(-1, nkv, HEAD_DIM)[..., V_HEAD_DIM:]
        if vpad.abs().max() != 0:
            raise RuntimeError(f"layer {layer} v: pad columns of the V heads are not zero")
        pk, pv = _pcc(dev["key"], gk), _pcc(dev["value"], gv)
        mins["k"], mins["v"] = min(mins["k"], pk), min(mins["v"], pv)
        logger.info(f"  layer {layer:>2} ({'sliding' if cfg.is_sliding(layer) else 'full'}): K={pk:.5f} V={pv:.5f}")
    logger.info(
        f"[mimo-cp4] slot {slot_id} KV PCC over [0,{real_len}) of {n} layers -> K={mins['k']:.5f} V={mins['v']:.5f}"
    )
    return mins

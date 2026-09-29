# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill-server KV contract for GLM-5.3-Flash on a 2x2 mesh.

Only the DSA layers (3, 7, ...) own positional KV; the KDA layers carry fixed-size state (recurrent [64, 128, 128]
fp32 and the q/k/v conv tail [3, 24576]), which is not in the address table (per-slot buffers, see adapter.py).
Per DSA layer the device holds, replicated on all 4 chips (tt/mla_attention.py, tt/indexer.py):
  * kv_latent  [S, 512] bf16: the MLA latent after kv_a_layernorm (NoPE, no rope columns)
  * index_key  [S / 4, 128] bf16: the indexer's pooled keys (softmax pool of 4 tokens)

The attention reads its own per-slot caches (ROW_MAJOR latent, TILE pooled keys). The migratable copy is written from
them after the layer, into two DRAM-interleaved ROW_MAJOR slabs whose page is one 32-token block:
  * latent slab  [1, 1, users * dsa_layers * max_seq / 32, 32 * 512] bf16: page = 32 latent rows (32768 B)
  * index slab   [1, 1, users * dsa_layers * max_seq / 32, 8 * 128] bf16: page = the 8 pooled rows of those 32
    tokens (2048 B)
Row (page) j = (slot * dsa_layers + dsa_layer) * max_seq / 32 + pos / 32 lives in DRAM bank j % B at offset
(j // B) * page_bytes (interleaved), the same walk as the gpt_oss_d_p / MiMo round-robin tables.

Address table: 2 configs (0 = kv_latent, 1 = index_key), both 32-token entries; table layer = the DSA layer's index
among the served DSA layers (kv_slot_layer_ids maps it back to the global layer); one-chip device group at
MeshCoordinate(0, 0) (every chip holds the same bytes).
"""

from __future__ import annotations

import socket

import torch

import ttnn

BLOCK = 32  # tokens per table entry / slab page
MC = ttnn.DRAM_MEMORY_CONFIG
CONFIGS = ("kv_latent", "index_key")


class GlmContractKV:
    def __init__(self, mesh, cfg, num_dsa_layers: int, max_seq: int, num_users: int):
        from models.demos.common.prefill.runners.migration import get_num_dram_banks

        assert max_seq % BLOCK == 0 and num_dsa_layers > 0, (max_seq, num_dsa_layers)
        self.mesh, self.n_layers, self.max_seq, self.num_users = mesh, num_dsa_layers, max_seq, num_users
        self.kp = cfg.index_kpool
        self.lat_w, self.idx_w = cfg.kv_lora_rank, cfg.index_head_dim
        assert BLOCK % self.kp == 0
        self.rows_per_slab = max_seq // BLOCK
        rows = num_users * num_dsa_layers * self.rows_per_slab
        self.widths = {"kv_latent": BLOCK * self.lat_w, "index_key": (BLOCK // self.kp) * self.idx_w}
        self.slabs = {
            name: ttnn.zeros(
                (1, 1, rows, w), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=MC
            )
            for name, w in self.widths.items()
        }
        self.num_banks = get_num_dram_banks(mesh)

    def page_bytes(self, name: str) -> int:
        return self.widths[name] * 2

    def _row(self, slot: int, layer: int, pos: int) -> int:
        assert 0 <= slot < self.num_users and 0 <= layer < self.n_layers and pos % BLOCK == 0, (slot, layer, pos)
        return (slot * self.n_layers + layer) * self.rows_per_slab + pos // BLOCK

    def write(self, slot: int, layer: int, start: int, seq: int, latent: ttnn.Tensor, pooled: ttnn.Tensor) -> None:
        """Copy the chunk [start, start + seq) of one DSA layer from the attention's caches into the slabs.
        latent: [1, 1, max_seq, 512] ROW_MAJOR; pooled: [1, 1, max_seq / 4 + 32, 128] TILE. Device ops only."""
        assert start % BLOCK == 0 and seq % (BLOCK * self.kp) == 0 and start + seq <= self.max_seq, (start, seq)
        r0, n = self._row(slot, layer, start), seq // BLOCK
        lat = ttnn.slice(latent, (0, 0, start, 0), (1, 1, start + seq, self.lat_w), memory_config=MC)
        p0 = start // self.kp
        pk = ttnn.slice(pooled, (0, 0, p0, 0), (1, 1, p0 + seq // self.kp, self.idx_w), memory_config=MC)
        pkr = ttnn.to_layout(pk, ttnn.ROW_MAJOR_LAYOUT, memory_config=MC)
        ttnn.deallocate(pk)
        for name, t in (("kv_latent", lat), ("index_key", pkr)):
            w = self.widths[name]
            blk = ttnn.reshape(t, (1, 1, n, w))
            ttnn.experimental.slice_write(blk, self.slabs[name], [0, 0, r0, 0], [1, 1, r0 + n, w], [1, 1, 1, 1])
            ttnn.deallocate(blk)
            if t.is_allocated():
                ttnn.deallocate(t)

    def base_address(self) -> int:
        return int(self.slabs["kv_latent"].buffer_address())

    def address_table(self, seq_len: int):
        """Configs 0 kv_latent, 1 index_key; one entry per (DSA layer, 32-token position, slot)."""
        from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import _make_config, _stable_config_name

        D = ttnn.experimental.disaggregation
        assert seq_len <= self.max_seq and seq_len % BLOCK == 0, seq_len
        n = len(CONFIGS)
        table = D.KvChunkAddressTable(
            {
                _stable_config_name(i, n): _make_config(
                    num_layers=self.n_layers,
                    max_seq_len=seq_len,
                    num_users=self.num_users,
                    chunk_size_bytes=self.page_bytes(name),
                )
                for i, name in enumerate(CONFIGS)
            }
        )
        fid = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(0, 0))
        table.set_fabric_node_host(fid, host_name=socket.gethostname())
        for cid, name in enumerate(CONFIGS):
            assert table.config_name(cid) == _stable_config_name(cid, n)
            base, nbytes = int(self.slabs[name].buffer_address()), self.page_bytes(name)
            group = table.add_device_group([fid])
            for slot in range(self.num_users):
                for layer in range(self.n_layers):
                    for pos in range(0, seq_len, BLOCK):
                        j = self._row(slot, layer, pos)
                        loc = D.KvCacheLocation()
                        loc.noc_addr = ((j % self.num_banks) << 32) | (base + (j // self.num_banks) * nbytes)
                        loc.size_bytes = nbytes
                        loc.device_group_index = group
                        table.set(layer, pos, slot, loc, cid)
        return table


# ------------------------------------------------------------------ read-back (host, device-less via the table)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def _bf16_rows(raw: bytes, width: int) -> torch.Tensor:
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(-1, width).float()


def read_layer(table, device_map: dict, layer: int, slot: int, length: int, cfg) -> dict:
    """One DSA layer's contract copy through the table: kv_latent [length, 512], index_key [length / 4, 128]."""
    from models.demos.common.prefill.runners import prefill_producer as producer

    D = ttnn.experimental.disaggregation
    read_len = -(-length // BLOCK) * BLOCK
    out = {}
    for cid, (name, width) in enumerate((("kv_latent", cfg.kv_lora_rank), ("index_key", cfg.index_head_dim))):
        rows = []
        for pos in range(0, read_len, BLOCK):
            loc = table.lookup(layer, pos, slot, cid)
            uid = producer._resolve_unique_id(
                table.get_device_group(loc.device_group_index).fabric_node_ids, device_map
            )
            rows.append(_bf16_rows(D.read_dram_umd(uid, loc.noc_addr, loc.size_bytes), width))
        n = length if name == "kv_latent" else length // cfg.index_kpool
        out[name] = torch.cat(rows)[:n]
    return out


def read_slot_kv_and_check_pcc(table, device_map: dict, slot_id: int, real_len: int, trace_dir, dsa_layers, cfg):
    """Min PCC per cache over [0, real_len) of every served DSA layer (table layer k = dsa_layers[k], global) vs the
    bring-up golden (trace_dir/kv_cache/layer_i.safetensors: kv_latent_cache_layer_i [S, 512],
    index_key_cache_layer_i [S / 4, 128])."""
    from pathlib import Path

    from loguru import logger
    from safetensors import safe_open

    mins = {name: 1.0 for name in CONFIGS}
    for k, layer in enumerate(dsa_layers):
        dev = read_layer(table, device_map, k, slot_id, real_len, cfg)
        with safe_open(str(Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as f:
            gold = {
                "kv_latent": f.get_slice(f"kv_latent_cache_layer_{layer}")[:real_len].float(),
                "index_key": f.get_slice(f"index_key_cache_layer_{layer}")[: real_len // cfg.index_kpool].float(),
            }
        for name in CONFIGS:
            assert dev[name].shape == gold[name].shape, (name, dev[name].shape, gold[name].shape)
            p = _pcc(dev[name], gold[name])
            mins[name] = min(mins[name], p)
            logger.info(f"  layer {layer:>2} {name}: PCC {p:.6f}")
    logger.info(f"[glm53] slot {slot_id} KV PCC over [0,{real_len}) of DSA layers {list(dsa_layers)} -> {mins}")
    return mins

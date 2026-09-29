# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill-server KV-cache contract for Hy4 Preview on a 2x2 mesh (SP = 2 over axis 0 x TP = 2 over axis 1).

The engine-owned cache IS the model's device state: TtHy4Attention / TtHy4Indexer write and gather it in place
(``bind_cache``), so there is no copy and nothing to sync but the layer itself.

  * ``kvpe``  [num_users * L, 1, max_seq / 4, 576] bf16 ROW_MAJOR per chip: the MLA latent cache
    (kv_a_layernorm(latent 512) | RoPE(k_pe 64), interleaved RoPE in checkpoint order, as the golden kv_latent);
    batch = slot * L + layer (L = served layers)
  * ``index`` [num_users * F, 1, max_seq / 4, 128] bf16 TILE per chip: the index-key cache of the F full layers
    (LayerNorm(wk(h)) with RoPE on dims 64..127, checkpoint order, as the golden index_key); batch = slot * F + k,
    k = the full layer's rank among the served full layers. Shared layers own no index keys.

Both are ``init_kvpe_cache(tp_axis=1)`` caches: DRAM NdShard [1, 1, 32, W] ROUND_ROBIN_1D over the DRAM banks,
block-cyclic over the 4 chips with the model's chunk as the period: chip l = row * 2 + col holds positions
j * chunk + l * chunk / 4 + [0, chunk / 4) of every slab j. So the table's period is the served chunk size (the DeepSeek
builder hard-codes 5120), and this module walks it itself.

Address table: config "0" = kvpe on the layer axis 0..L-1, config "1" = index on the same layer axis (rows only for
full layers; a shared layer's index rows are unset, zero-size). One-chip device groups at MeshCoordinate(l // 2, l % 2).
Entry = one 32-token block: 32 aligned bf16 rows (kvpe) or 4 bf16 tiles (index).
"""

from __future__ import annotations

import socket

import torch

import ttnn

MESH = (2, 2)
BLOCK = 32  # tokens per DRAM bank shard / table entry
KV_WIDTH = 576
INDEX_DIM = 128


def chip_of(pos: int, chunk: int) -> tuple[int, int]:
    """Natural position -> (linear chip l, local row) in the block-cyclic cache (inverse of update_padded_kv_cache)."""
    n = MESH[0] * MESH[1]
    q = chunk // n
    slab, off = divmod(pos, chunk)
    return off // q, slab * q + off % q


class Hy4ContractKV:
    """The engine-owned caches + their address table (see the module docstring)."""

    def __init__(self, mesh, layers: list[int], full_layers: list[int], max_seq: int, chunk: int, num_users: int):
        from models.demos.common.prefill.runners.migration import get_num_dram_banks
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

        assert tuple(mesh.shape) == MESH, f"Hy4 contract cache is built for a 2x2 mesh, got {mesh.shape}"
        n = MESH[0] * MESH[1]
        assert max_seq % chunk == 0 and chunk % (BLOCK * n) == 0, (max_seq, chunk)
        self.mesh, self.layers, self.full = mesh, list(layers), list(full_layers)
        self.max_seq, self.chunk, self.num_users = max_seq, chunk, num_users
        self.num_banks = get_num_dram_banks(mesh)

        def alloc(width, rows, layout):
            return init_kvpe_cache(
                width,
                mesh,
                max_seq,
                MESH,
                0,
                rows,
                num_users=num_users,
                dtype=ttnn.bfloat16,
                layout=layout,
                tp_axis=1,
            )

        self.kvpe = alloc(KV_WIDTH, len(self.layers), ttnn.ROW_MAJOR_LAYOUT)
        self.index = alloc(INDEX_DIM, len(self.full), ttnn.TILE_LAYOUT) if self.full else None

    def kv_row(self, layer: int) -> int:
        return self.layers.index(layer)

    def index_row(self, layer: int) -> int:
        return self.full.index(layer)

    @staticmethod
    def entry_bytes(cache) -> int:
        if cache.layout == ttnn.ROW_MAJOR_LAYOUT:
            return BLOCK * cache.buffer_aligned_page_size()
        return (cache.shape[-1] // 32) * {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}[cache.dtype]

    def address_table(self, seq_len: int):
        from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import _make_config

        D = ttnn.experimental.disaggregation
        assert seq_len <= self.max_seq and seq_len % self.chunk == 0, (seq_len, self.max_seq, self.chunk)
        L = len(self.layers)
        caches = [("0", self.kvpe, self.layers, self.kv_row)]
        if self.index is not None:
            caches.append(("1", self.index, self.full, self.index_row))
        table = D.KvChunkAddressTable(
            {
                name: _make_config(
                    num_layers=L, max_seq_len=seq_len, num_users=self.num_users, chunk_size_bytes=self.entry_bytes(c)
                )
                for name, c, _, _ in caches
            }
        )
        host = socket.gethostname()
        groups = []
        for l in range(MESH[0] * MESH[1]):
            fid = self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(l // MESH[1], l % MESH[1]))
            groups.append(table.add_device_group([fid]))
            table.set_fabric_node_host(fid, host_name=host)
        blocks_per_batch = (self.max_seq // (MESH[0] * MESH[1])) // BLOCK
        for name, cache, owned, row_of in caches:
            cid = table.config_id_of(name)
            base, nbytes, rows = cache.buffer_address(), self.entry_bytes(cache), len(owned)
            for slot in range(self.num_users):
                for layer in owned:
                    b = slot * rows + row_of(layer)
                    for pos in range(0, seq_len, BLOCK):
                        chip, lr = chip_of(pos, self.chunk)
                        # ROUND_ROBIN_1D: shard j (row-major over [batch, rows / 32]) lives in bank j % B at j // B.
                        j = b * blocks_per_batch + lr // BLOCK
                        loc = D.KvCacheLocation()
                        loc.noc_addr = ((j % self.num_banks) << 32) | (base + (j // self.num_banks) * nbytes)
                        loc.size_bytes = nbytes
                        loc.device_group_index = groups[chip]
                        table.set(self.layers.index(layer), pos, slot, loc, cid)
        return table


# ---- read-back (host, device-less through the table: what a migration consumer sees)
def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def read_cache(table, device_map, config_id: int, row: int, slot: int, length: int, width: int) -> torch.Tensor:
    """[length, width] natural-order rows of table row ``row`` read through ``read_dram_umd``."""
    from models.demos.common.prefill.runners import prefill_producer as P

    out = []
    for pos in range(0, -(-length // BLOCK) * BLOCK, BLOCK):
        loc = table.lookup(row, pos, slot, config_id)
        uid = P._resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = bytes(ttnn.experimental.disaggregation.read_dram_umd(uid, loc.noc_addr, loc.size_bytes))
        if width == KV_WIDTH:  # ROW_MAJOR bf16 rows (page-aligned)
            out.append(P._decode_row_major_chunk(raw, width, torch.bfloat16))
        else:  # bf16 tiles
            out.append(P._decode_bf16_chunk(raw, width))
    return torch.cat(out, dim=0)[:length]


def read_slot_kv_and_check_pcc(table, device_map, slot_id: int, real_len: int, trace_dir, layers, full) -> dict:
    """Min PCC over [0, real_len) of every served layer's kv_latent and every full layer's index_key vs the bring-up
    golden (trace_dir/kv_cache/layer_{i}.safetensors: kv_latent_cache_layer_i [S, 576], index_key_cache_layer_i
    [S, 128], empty on shared layers). Also checks a shared layer's index rows are unset in the table."""
    from pathlib import Path

    from loguru import logger
    from safetensors import safe_open

    mins = {"kv_latent": 1.0, "index_key": 1.0}
    names = [table.config_name(i) for i in range(table.num_configs())]
    for row, layer in enumerate(layers):
        with safe_open(str(Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as f:
            gkv = f.get_slice(f"kv_latent_cache_layer_{layer}")[:real_len].float()
            gik = f.get_slice(f"index_key_cache_layer_{layer}")[:real_len].float()
        dkv = read_cache(table, device_map, names.index("0"), row, slot_id, real_len, KV_WIDTH)
        assert dkv.shape == gkv.shape, (dkv.shape, gkv.shape)
        p_kv = _pcc(dkv, gkv)
        mins["kv_latent"] = min(mins["kv_latent"], p_kv)
        msg = f"  layer {layer}: kv_latent {p_kv:.5f} (nope {_pcc(dkv[:, :512], gkv[:, :512]):.5f}"
        msg += f" pe {_pcc(dkv[:, 512:], gkv[:, 512:]):.5f})"
        if layer in full:
            dik = read_cache(table, device_map, names.index("1"), row, slot_id, real_len, INDEX_DIM)
            assert dik.shape == gik.shape, (dik.shape, gik.shape)
            p_ik = _pcc(dik, gik)
            mins["index_key"] = min(mins["index_key"], p_ik)
            msg += f" index_key {p_ik:.5f}"
        else:
            assert gik.shape[0] == 0, f"layer {layer} is shared but its golden has index keys {tuple(gik.shape)}"
            if "1" in names:
                size = table.lookup(row, 0, slot_id, names.index("1")).size_bytes
                assert size == 0, f"shared layer {layer} has an index entry of {size} B"
        logger.info(msg)
    logger.info(f"[hy4] slot {slot_id} KV PCC over [0,{real_len}) of layers {list(layers)} -> {mins}")
    return mins

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill-server KV-cache contract for Xing4.0 on a 4x2 mesh (SP = 4 over axis 0 x TP = 2 over axis 1).

The engine-owned cache IS the model's device state: TtMlaAttention writes and gathers it in place (``bind_cache``),
so there is no copy and nothing to sync but the layer itself.

  * ``kvpe`` [num_users * L, 1, max_seq / 4, 576] bf16 TILE per chip: the MLA latent cache
    (kv_a_layernorm(latent 512) | RoPE(k_pe 64), interleaved RoPE in checkpoint order, as the golden kv_latent);
    batch = slot * L + layer row (L = served layers).

It is an ``init_kvpe_cache(tp_axis=None)`` cache, the layout of the attention's own geometry cache: DRAM NdShard
[1, 1, 32, 576] ROUND_ROBIN_1D over the DRAM banks, block-cyclic over the 4 mesh rows with the model's chunk as the
period (row r holds positions j * chunk + r * chunk / 4 + [0, chunk / 4) of every slab j), replicated over the 2
columns (both columns compute the all-reduced kv and write their own copy). The table's period is the served chunk
(the DeepSeek builder hard-codes 5120), so this module walks it itself.

Address table: config "0" = kvpe on the layer axis 0..L-1. One device group per mesh row holding both column
replicas (MeshCoordinate(r, 0), (r, 1)), as the DeepSeek TP-replicated table. Entry = one 32-token block: 18 bf16
tiles (36 KB).
"""

from __future__ import annotations

import socket

import torch

import ttnn

MESH = (4, 2)
BLOCK = 32  # tokens per DRAM bank shard / table entry
KV_WIDTH = 576


def chip_of(pos: int, chunk: int) -> tuple[int, int]:
    """Natural position -> (mesh row, local row) in the block-cyclic cache (inverse of update_padded_kv_cache)."""
    q = chunk // MESH[0]
    slab, off = divmod(pos, chunk)
    return off // q, slab * q + off % q


class XingContractKV:
    """The engine-owned latent cache + its address table (see the module docstring)."""

    def __init__(self, mesh, layers: list[int], max_seq: int, chunk: int, num_users: int):
        from models.demos.common.prefill.runners.migration import get_num_dram_banks
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

        assert tuple(mesh.shape) == MESH, f"Xing contract cache is built for a 4x2 mesh, got {mesh.shape}"
        assert max_seq % chunk == 0 and chunk % (BLOCK * MESH[0]) == 0, (max_seq, chunk)
        self.mesh, self.layers = mesh, list(layers)
        self.max_seq, self.chunk, self.num_users = max_seq, chunk, num_users
        self.num_banks = get_num_dram_banks(mesh)
        self.kvpe = init_kvpe_cache(
            KV_WIDTH,
            mesh,
            max_seq,
            MESH,
            0,
            len(self.layers),
            num_users=num_users,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def kv_row(self, layer: int) -> int:
        return self.layers.index(layer)

    @staticmethod
    def entry_bytes(cache) -> int:
        return (cache.shape[-1] // 32) * {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088}[cache.dtype]

    def address_table(self, seq_len: int):
        from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import _make_config

        D = ttnn.experimental.disaggregation
        assert seq_len <= self.max_seq and seq_len % self.chunk == 0, (seq_len, self.max_seq, self.chunk)
        L = len(self.layers)
        nbytes = self.entry_bytes(self.kvpe)
        table = D.KvChunkAddressTable(
            {"0": _make_config(num_layers=L, max_seq_len=seq_len, num_users=self.num_users, chunk_size_bytes=nbytes)}
        )
        host = socket.gethostname()
        groups = []
        for r in range(MESH[0]):
            fids = [self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for c in range(MESH[1])]
            groups.append(table.add_device_group(fids))
            for fid in fids:
                table.set_fabric_node_host(fid, host_name=host)
        cid = table.config_id_of("0")
        base = self.kvpe.buffer_address()
        blocks_per_batch = (self.max_seq // MESH[0]) // BLOCK
        for slot in range(self.num_users):
            for row in range(L):
                b = slot * L + row
                for pos in range(0, seq_len, BLOCK):
                    chip_row, lr = chip_of(pos, self.chunk)
                    # ROUND_ROBIN_1D: shard j (row-major over [batch, rows / 32]) lives in bank j % B at j // B.
                    j = b * blocks_per_batch + lr // BLOCK
                    loc = D.KvCacheLocation()
                    loc.noc_addr = ((j % self.num_banks) << 32) | (base + (j // self.num_banks) * nbytes)
                    loc.size_bytes = nbytes
                    loc.device_group_index = groups[chip_row]
                    table.set(row, pos, slot, loc, cid)
        return table


# ---- read-back (host, device-less through the table: what a migration consumer sees)
def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def read_cache(table, device_map, config_id: int, row: int, slot: int, length: int, width: int = KV_WIDTH):
    """[length, width] natural-order rows of table row ``row`` read through ``read_dram_umd`` (bf16 tiles)."""
    from models.demos.common.prefill.runners import prefill_producer as P

    out = []
    for pos in range(0, -(-length // BLOCK) * BLOCK, BLOCK):
        loc = table.lookup(row, pos, slot, config_id)
        uid = P._resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = bytes(ttnn.experimental.disaggregation.read_dram_umd(uid, loc.noc_addr, loc.size_bytes))
        out.append(P._decode_bf16_chunk(raw, width))
    return torch.cat(out, dim=0)[:length]


def read_slot_kv_and_check_pcc(table, device_map, slot_id: int, real_len: int, trace_dir, layers) -> dict:
    """Min PCC over [0, real_len) of every served layer's kv_latent vs the bring-up golden
    (trace_dir/kv_cache/layer_{i}.safetensors: kv_latent_cache_layer_i [S, 576])."""
    from pathlib import Path

    from loguru import logger
    from safetensors import safe_open

    mins = {"kv_latent": 1.0}
    cid = [table.config_name(i) for i in range(table.num_configs())].index("0")
    for row, layer in enumerate(layers):
        with safe_open(str(Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as f:
            gkv = f.get_slice(f"kv_latent_cache_layer_{layer}")[:real_len].float()
        dkv = read_cache(table, device_map, cid, row, slot_id, real_len)
        assert dkv.shape == gkv.shape, (dkv.shape, gkv.shape)
        p_kv = _pcc(dkv, gkv)
        mins["kv_latent"] = min(mins["kv_latent"], p_kv)
        logger.info(
            f"  layer {layer}: kv_latent {p_kv:.5f} (nope {_pcc(dkv[:, :512], gkv[:, :512]):.5f}"
            f" pe {_pcc(dkv[:, 512:], gkv[:, 512:]):.5f})"
        )
    logger.info(f"[xing] slot {slot_id} KV PCC over [0,{real_len}) of layers {list(layers)} -> {mins}")
    return mins

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill -> decode KV hand-off (tt-blaze DS41F-0037 M2): the export caches in the decode ring's
contract layout and the ``KvChunkAddressTable`` that maps every (config, global layer, 32-row chunk, slot) the ring asks
for onto them.

The contract is the ring's (tt-blaze ``blaze/models/deepseek_v4_1_flash/kv_migration.py``), in config order::

    0  swa_window   layers 0, 1     bfp8 TILE  256-row window ring (position p at row p % 256)
    1  csa_unified  layers 2..39    bf16 RM    rows [0, 256) the window ring, row 256 + w = compressed entry w
    2  index_k      2/8/14/20 + 24/28/32/36   bfp8 TILE  row w = index key of entry w (unrotated)

What the prefill OWNS (it runs the encoder 0..19 and layer 20 KV-only):

    swa   [U * 2,  1, 256,     512] bfp8 TILE   layers 0, 1
    win   [U * 19, 1, 256,     512] bf16 RM     the window rings of layers 2..20
    zero  [1,      1, 256,     512] bf16 RM     never written: the decoder layers' (21..39) window chunks point here
    ent2  [U * 3,  1, S / 2,   512] bf16 RM     the entries of the ratio-2 KV sources 2 / 8 / 14
    ent1  [U,      1, S,       512] bf16 RM     layer 20's ratio-1 entries
    idx2  [U * 3,  1, S / 2,   128] bfp8 TILE   the index keys of 2 / 8 / 14
    idx1  [U,      1, S,       128] bfp8 TILE   layer 20's index keys

Consumers hold no rows of their own: the table points their entry chunks (and the Reindex layers' key chunks) at their
source's rows, and every decoder layer's entry chunks at layer 20's. No per-consumer copies on the prefill galaxy.

Every tensor is ``init_kvpe_cache``-allocated (DRAM ND-sharded in 32-row shards round-robin over the banks, replicated
over the mesh), so the chunk at (batch b, row r) of a tensor with ``rows`` rows per batch sits at
``flat = (b * rows + r) // 32``: bank ``flat % n_banks``, offset ``base + (flat // n_banks) * chunk_bytes`` -- the
V4-Flash ``kv_table.walk_linear`` formula. A 32-row chunk is one tile row (TILE) or 32 contiguous rows (RM).

Writes: after every chunk (the export is complete at every chunk boundary, so the per-layer acks are always truthful),
on device -- ``TtCSA._write_rm`` (update_padded_kv_cache) for the RM caches, ``fill_cache_for_user_`` for the tiles. The
window ring is the attention state's carry (the last 128 real tokens in order) at rows ``(E - 128) % 256``, E the
tokens so far; chunk boundaries are multiples of 128 (the prefill's 64-per-chip x sp alignment), so it is one block.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger

import ttnn

RING = 256
TILE = 32
HEAD_DIM = 512
INDEX_HEAD_DIM = 128
CHUNK_BYTES = {"swa_window": 16 * 1088, "csa_unified": 32 * 1024, "index_k": 4 * 1088}
GROUPS = ("swa_window", "csa_unified", "index_k")


def _roles(cfg):
    return {L: cfg.role(L) for L in range(cfg.n_layers)}


@dataclass
class V41KvExport:
    swa: object
    win: object
    zero: object
    ent2: object
    ent1: object
    idx2: object
    idx1: object
    num_users: int
    max_seq_len: int
    sources2: tuple  # (2, 8, 14)

    def tensors(self) -> dict:
        return dict(
            swa=self.swa, win=self.win, zero=self.zero, ent2=self.ent2, ent1=self.ent1, idx2=self.idx2, idx1=self.idx1
        )

    # the export batch index of each owned (slot, layer)
    def swa_batch(self, slot, L):
        return slot * 2 + L

    def win_batch(self, slot, L):
        return slot * 19 + (L - 2)

    def ent_batch(self, slot, src):
        return (slot * 3 + self.sources2.index(src)) if src in self.sources2 else slot

    def zero_all(self) -> None:
        """DS4F-0271: no row of a previous request survives into the next one (the migrate's row range is one range for
        every layer, so a ratio-2 layer's rows past its real entries are copied too)."""
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import DRAMZeroFill

        for t in self.tensors().values():
            DRAMZeroFill.op(t)


def allocate_v41_kv_export(
    mesh_device, cfg, *, max_seq_len: int, num_users: int = 1, mesh_shape=None, sp_axis: int = 0
):
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

    mesh_shape = list(mesh_shape or mesh_device.shape)
    sp = int(mesh_shape[sp_axis])
    roles = _roles(cfg)
    sources2 = tuple(L for L in range(cfg.first_decoder_layer) if roles[L].mode == "full")
    assert sources2 == (2, 8, 14), sources2
    assert max_seq_len % (2 * TILE) == 0, max_seq_len

    def alloc(rows, width, layers, dtype, layout, users=num_users):
        return init_kvpe_cache(
            kvpe_cache_head_dim=width,
            mesh_device=mesh_device,
            seq_len=rows * sp,  # replicated over SP: every chip holds `rows`
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=layers,
            num_users=users,
            dtype=dtype,
            layout=layout,
        )

    exp = V41KvExport(
        swa=alloc(RING, HEAD_DIM, 2, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        win=alloc(RING, HEAD_DIM, 19, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        zero=alloc(RING, HEAD_DIM, 1, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, users=1),
        ent2=alloc(max_seq_len // 2, HEAD_DIM, 3, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ent1=alloc(max_seq_len, HEAD_DIM, 1, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        idx2=alloc(max_seq_len // 2, INDEX_HEAD_DIM, 3, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        idx1=alloc(max_seq_len, INDEX_HEAD_DIM, 1, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        num_users=int(num_users),
        max_seq_len=int(max_seq_len),
        sources2=sources2,
    )
    logger.info(f"[v41 kv export] {', '.join(f'{k} {tuple(t.shape)}' for k, t in exp.tensors().items())}")
    return exp


# ---- per-chunk writes ----------------------------------------------------------------------------------------------------
def _carry_ring_block(state, kv_actual: int):
    """(ring row of the carry's first row, the [1, 1, 128, 512] TILE carry) -- the last 128 real tokens in order."""
    assert kv_actual % 128 == 0, f"a hand-over chunk boundary must be a multiple of 128 tokens (at {kv_actual})"
    return (kv_actual - 128) % RING, state.sliding_carry


def export_chunk(exp: V41KvExport, pf, slot: int, start: int, end: int) -> None:
    """Write the rows chunk [start, end) produced into the export caches (after ``pf._chunk``). ``pf`` is the
    ``V41Prefill``; its states already include the chunk."""
    E = int(end)
    for blk in pf.blocks:
        L, attn, st = blk.layer, blk.attn, blk.state
        r = pf.cfg.role(L)
        row0, carry = _carry_ring_block(st, E)
        if r.mode == "swa":
            blockt = carry if carry.dtype == ttnn.bfloat8_b else ttnn.typecast(carry, ttnn.bfloat8_b)
            ttnn.kv_cache.fill_cache_for_user_(exp.swa, blockt, exp.swa_batch(slot, L), update_idx=row0)
            if blockt is not carry:
                ttnn.deallocate(blockt)
            continue
        attn._write_rm(exp.win, carry, exp.win_batch(slot, L), row0)
        if r.mode == "full":
            _export_entries(exp, attn, st, slot, L, start, end, ratio=r.compress_ratio, ent=exp.ent2, idx=exp.idx2)
    if pf.kv_attn is not None:
        L, attn, st = pf.kv_only_layer, pf.kv_attn, pf.kv_state
        row0, carry = _carry_ring_block(st, E)
        attn._write_rm(exp.win, carry, exp.win_batch(slot, L), row0)
        _export_entries(exp, attn, st, slot, L, start, end, ratio=1, ent=exp.ent1, idx=exp.idx1)


def _export_entries(exp, attn, st, slot, L, start, end, *, ratio, ent, idx):
    e0, e1 = int(start) // ratio, -(-int(end) // ratio)
    n = -(-(e1 - e0) // TILE) * TILE  # whole tiles (a ragged tail's extra rows are the next chunk's to overwrite)
    b = exp.ent_batch(slot, L)
    rows = ttnn.slice(st.compressed_kv, [0, 0, e0, 0], [1, 1, e0 + n, HEAD_DIM])
    attn._write_rm(ent, rows, b, e0)
    ttnn.deallocate(rows)
    keys = ttnn.slice(st.index_k, [0, 0, e0, 0], [1, 1, e0 + n, INDEX_HEAD_DIM])
    k8 = ttnn.typecast(keys, ttnn.bfloat8_b)
    ttnn.deallocate(keys)
    ttnn.kv_cache.fill_cache_for_user_(idx, k8, b, update_idx=e0)
    ttnn.deallocate(k8)


# ---- the table -----------------------------------------------------------------------------------------------------------
def _chunk_addr(tensor, batch: int, row: int, n_banks: int, chunk_bytes: int) -> int:
    rows = int(tensor.shape[2])
    flat = (int(batch) * rows + int(row)) // TILE
    bank = flat % n_banks
    offset = int(tensor.buffer_address()) + (flat // n_banks) * int(chunk_bytes)
    return (bank << 32) | offset


def table_rows(exp: V41KvExport, cfg, *, slot: int, prompt_tokens: int | None = None):
    """Yield (group, layer, position, tensor, batch, row) for every chunk the contract asks of ``slot`` (device-free
    apart from the tensors' shapes): the logic the table and its tests share. ``prompt_tokens`` caps the entry rows (the
    table's extent is the export capacity by default)."""
    S = exp.max_seq_len if prompt_tokens is None else int(prompt_tokens)
    roles = _roles(cfg)
    for L in range(cfg.n_layers):
        r = roles[L]
        if not r.compress_ratio:
            for p in range(0, RING, TILE):
                yield "swa_window", L, p, exp.swa, exp.swa_batch(slot, L), p
            continue
        for p in range(0, RING, TILE):
            if L <= cfg.first_decoder_layer:
                yield "csa_unified", L, p, exp.win, exp.win_batch(slot, L), p
            else:
                yield "csa_unified", L, p, exp.zero, 0, p
        src = r.kv_source
        ent, idx = (exp.ent2, exp.idx2) if src in exp.sources2 else (exp.ent1, exp.idx1)
        n_rows = S // r.compress_ratio
        for w in range(0, n_rows, TILE):
            yield "csa_unified", L, RING + w, ent, exp.ent_batch(slot, src), w
        if r.index_source == L:
            for w in range(0, n_rows, TILE):
                yield "index_k", L, w, idx, exp.ent_batch(slot, src), w


def build_v41_kv_chunk_table(exp: V41KvExport, cfg, mesh_device, path: str) -> str:
    """The merged 3-config table, every chip of the mesh one device group (the exports are replicated), serialized to
    ``path`` (``serialize_prebuilt_kv_chunk_table``)."""
    from models.demos.common.prefill.runners.migration import get_num_dram_banks, serialize_prebuilt_kv_chunk_table

    D = ttnn.experimental.disaggregation
    extents = {"swa_window": RING, "csa_unified": RING + exp.max_seq_len, "index_k": exp.max_seq_len}
    configs = []
    for g in GROUPS:
        c = D.KvChunkAddressTableConfig()
        c.num_layers = int(cfg.n_layers)
        c.max_sequence_length = int(extents[g])
        c.num_slots = int(exp.num_users)
        c.chunk_n_tokens = TILE
        c.chunk_size_bytes = int(CHUNK_BYTES[g])
        configs.append(c)
    table = D.KvChunkAddressTable(configs)
    rows, cols = (int(v) for v in mesh_device.shape)
    fnids = [mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c)) for r in range(rows) for c in range(cols)]
    group = table.add_device_group(fnids)
    for fid in fnids:
        table.set_fabric_node_host(fid, host_name="host-00000000")
    n_banks = int(get_num_dram_banks(mesh_device))
    n = 0
    for slot in range(exp.num_users):
        for g, L, pos, t, b, row in table_rows(exp, cfg, slot=slot):
            loc = D.KvCacheLocation()
            loc.noc_addr = _chunk_addr(t, b, row, n_banks, CHUNK_BYTES[g])
            loc.size_bytes = int(CHUNK_BYTES[g])
            loc.device_group_index = group
            table.set(int(L), int(pos), int(slot), loc, GROUPS.index(g))
            n += 1
    logger.info(
        f"[v41 kv export] table: {n} chunk locations over {len(GROUPS)} configs, {len(fnids)} chips, {n_banks} banks"
    )
    return serialize_prebuilt_kv_chunk_table(table=table, path=path)


def read_back(exp: V41KvExport, cfg, slot: int, prompt_tokens: int) -> dict:
    """The exported rows as the ring would read them (host, first chip): {layer: {swa_window|csa_unified|index_k:
    [rows, width]}}, positions in contract order -- the device-free check of the export against a cache snapshot."""
    host = {k: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float() for k, t in exp.tensors().items()}
    by_id = {id(t): k for k, t in exp.tensors().items()}
    out: dict = {}
    for g, L, pos, t, b, row in table_rows(exp, cfg, slot=slot, prompt_tokens=prompt_tokens):
        out.setdefault(L, {}).setdefault(g, []).append(host[by_id[id(t)]][b, 0, row : row + TILE])
    return {L: {g: torch.cat(v, dim=0) for g, v in d.items()} for L, d in out.items()}

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 Flash prefill -> 48-ring KV hand-off for tt-d-gen (tt-blaze G53F-0256 G4b).

The prefill (``GLM53MeshModel`` on the 8x4 mesh) keeps its own caches in its own layouts. After every chunk this module
writes the chunk's state into EXPORT slabs in the decode ring's byte format, and ``build_table`` publishes a
``KvChunkAddressTable`` over them whose configs, chunk sizes and position axes are the ring's
(tt-blaze ``blaze/models/glm_5_3_flash_4_GLX/kv_migration.py``). The tt-d-gen KV Manager then copies chunk for chunk:

    0_dsa_kv    DSA layers 3, 7, ..., 43 (+ MTP 45)  32-position latent pages: 32 x 576 B slots (528 B scaled fp8 + scale)
    1_dsa_pool  DSA layers (+ MTP 45)                one 32-pool bfp8 tile row (128 positions): 4 tiles, 4352 B
    2_kda_s     KDA layers                           one [128, 32] fp32 V-band per (global head, band): 16384 B
    3_kda_conv  KDA layers                           one [3, 128] bf16 window per (branch, global head): 768 B

DSA: the latent rows and pool keys are read back to the host per chunk, packed with the SAME functions the ring's seed
path uses (tt-blaze ``kv_chunk_migration_helpers.pack_glm_scaled_kv_cache`` / ``glm_scaled_kv_rows_to_pages`` and the
bfp8 tile packer of ``prefill_seed_apply._bfp8_tiles``, copied here byte for byte) and raw-written over the NOC into a
per-layer slab: DSA layer i's slab lives on mesh device i (row-major), a ``[1, 1, R, page]`` uint8 row-major interleaved
DRAM tensor whose page p sits in bank ``p % N`` at ``base + (p // N) * page`` -- one page per table chunk.

KDA: the native carries (``dec.kda_batched``: recurrent ``[1, 16, 128, 128]`` fp32 TILE, conv ``[1, 3, 6144]`` per TP
column, replicated over the SP rows) are copied on device into two slabs per rank (the Kimi-K3 contract form,
``deepseek_v3_d_p/tt/kda/state_adapter.py``): recurrent ``[L, 16, 128, 128]`` fp32 ND-sharded ``[1, 1, 128, 32]`` (one
shard = one 16 KiB segment), conv ``[L, 48, 384]`` bf16 row-major interleaved (one 768-B page = one whole-head window;
the ring has one conv core per head, so Kimi-K3's 64-channel halves do not apply). Every SP row holds a replica; a table
device group is one TP column.

KDA has no token axis: one version of both states spans ``W = lcm(256, 192) * 32 = 24576`` synthetic positions
(recurrent segment G at G * 96, conv segment G at G * 128) and the table repeats the window for each of the ring's
``versions`` (``GLM53F_KDA_CACHE_MAX_LENGTH``, 2): the engine copies window ``(S' - 1) % versions`` and both windows alias
the one prefill state. Only the state after the LAST chunk is meaningful; the engine sends a KDA window once, after the
burst's last chunk (tt-d-gen ``KvmClient::finish_burst``).
"""

from __future__ import annotations

import math
import os
import socket
import time
import zlib
from dataclasses import dataclass, field

import torch
from loguru import logger

import ttnn

# ---------------------------------------------------------------------------------------------------- ring contract
NUM_LAYERS = 45
MTP_LAYER = 45
TABLE_LAYERS = NUM_LAYERS + 1
DSA_LAYERS = tuple(range(3, NUM_LAYERS, 4))
KDA_LAYERS = tuple(L for L in range(NUM_LAYERS) if L not in DSA_LAYERS)
CFG_DSA_KV, CFG_DSA_POOL, CFG_KDA_S, CFG_KDA_CONV = "0_dsa_kv", "1_dsa_pool", "2_kda_s", "3_kda_conv"

TPP = 32
LATENT_DIM = 512
LATENT_ROW_BYTES = LATENT_DIM + 16  # 528: fp8 latent + 4 fp32 block scales
LATENT_SLOT_BYTES = 576  # the ring's physical row slot (48-byte tail zero)
LATENT_PAGE_BYTES = TPP * LATENT_SLOT_BYTES  # 18432
POOL_RATIO = 4
POOL_WIDTH = 128
POOL_TILE_BYTES = 1088
POOL_ROW_BYTES = (POOL_WIDTH // 32) * POOL_TILE_BYTES  # 4352
POOL_CHUNK_TOKENS = 32 * POOL_RATIO  # 128

KDA_HEADS, KDA_HEAD_DIM, KDA_TAPS = 64, 128, 3
KDA_BAND = 32
KDA_BANDS = KDA_HEAD_DIM // KDA_BAND
KDA_BRANCHES = 3
KDA_S_SEGMENT = KDA_HEAD_DIM * KDA_BAND * 4  # 16384
KDA_CONV_SEGMENT = KDA_TAPS * KDA_HEAD_DIM * 2  # 768
KDA_S_SEGMENTS = KDA_HEADS * KDA_BANDS  # 256
KDA_CONV_SEGMENTS = KDA_BRANCHES * KDA_HEADS  # 192
KDA_WINDOW = math.lcm(KDA_S_SEGMENTS, KDA_CONV_SEGMENTS) * 32  # 24576
KDA_S_STRIDE = KDA_WINDOW // KDA_S_SEGMENTS  # 96
KDA_CONV_STRIDE = KDA_WINDOW // KDA_CONV_SEGMENTS  # 128


# ------------------------------------------------------------------- host packers (byte-identical to tt-blaze's)
def pack_latent_pages(latent: torch.Tensor, pow2_scale: bool = False) -> torch.Tensor:
    """``[n, 512]`` latent rows (n % 32 == 0) -> ``[n / 32, 18432]`` uint8 ring pages.

    tt-blaze ``pack_glm_scaled_kv_cache(rope_dim=0, reconstruct=False)`` + ``glm_scaled_kv_rows_to_pages(rows, 0)``:
    per 128-column block an fp32 scale ``amax / 448`` (clamped to [1e-4, 3e38]; rounded up to a power of two with
    ``pow2_scale``, the ring's ``GLM_KV_POW2_SCALE``), e4m3 values, then 32 rows per page in 576-byte slots."""
    n = int(latent.shape[0])
    assert n % TPP == 0 and int(latent.shape[-1]) == LATENT_DIM, tuple(latent.shape)
    blocks = latent.reshape(-1, LATENT_DIM).float().reshape(-1, LATENT_DIM // 128, 128)
    scales = blocks.abs().amax(dim=-1).clamp(1.0e-4, 3.0e38) / 448.0
    if pow2_scale:
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    fp8 = (blocks / scales.unsqueeze(-1)).reshape(-1, LATENT_DIM).to(torch.float8_e4m3fn)
    paged = torch.zeros((n // TPP, TPP, LATENT_SLOT_BYTES), dtype=torch.uint8)
    paged[:, :, :LATENT_DIM] = fp8.contiguous().view(torch.uint8).reshape(-1, TPP, LATENT_DIM)
    paged[:, :, LATENT_DIM:LATENT_ROW_BYTES] = scales.contiguous().view(torch.uint8).reshape(-1, TPP, 16)
    return paged.reshape(n // TPP, LATENT_PAGE_BYTES)


def pack_pool_rows(pools: torch.Tensor) -> torch.Tensor:
    """``[m, 128]`` pool keys (m % 32 == 0) -> ``[m / 32, 4352]`` uint8: each 32-pool tile row's 4 bfp8 tiles in column
    order (tt-blaze ``prefill_seed_apply._bfp8_tiles`` in row-major tile order, 4 consecutive device pages per row)."""
    m, w = int(pools.shape[0]), int(pools.shape[-1])
    assert m % 32 == 0 and w == POOL_WIDTH, tuple(pools.shape)
    tiles = pools.float().reshape(m // 32, 32, w // 32, 32).permute(0, 2, 1, 3).contiguous().reshape(-1).numpy()
    words = ttnn._ttnn.bfp_utils.pack_bfp8(tiles, row_major_input=True)
    raw = torch.from_numpy(words.view("uint8").copy())
    assert raw.numel() == (m // 32) * POOL_ROW_BYTES, (raw.numel(), m)
    return raw.reshape(m // 32, POOL_ROW_BYTES)


# ------------------------------------------------------------------------------------------------- raw DRAM writes
class _Banks:
    """NOC write access to the DRAM banks of the mesh's chips (``ttnn.cluster`` raw writes, the seed path's)."""

    def __init__(self, device_ids):
        self.tables = {
            int(d): {int(e["bank_id"]): e for e in ttnn.cluster.get_dram_bank_table(int(d))} for d in device_ids
        }

    def write(self, dev: int, bank: int, bank_addr: int, data: bytes) -> None:
        e = self.tables[dev][bank]
        ttnn.cluster.write_to_core(dev, int(e["noc_x"]), int(e["noc_y"]), int(e["base_addr"]) + int(bank_addr), data)

    def read(self, dev: int, bank: int, bank_addr: int, size: int) -> bytes:
        e = self.tables[dev][bank]
        return ttnn.cluster.read_from_core(
            dev, int(e["noc_x"]), int(e["noc_y"]), int(e["base_addr"]) + int(bank_addr), size
        )


def _write_pages(banks: _Banks, dev: int, base: int, nb: int, page_bytes: int, first: int, pages: torch.Tensor) -> None:
    """Pages ``first ..`` of an interleaved slab on chip ``dev``: page p in bank p % nb at base + (p // nb) * page_bytes.
    Pages of one bank in a run are adjacent in that bank, so each bank takes one write."""
    n = int(pages.shape[0])
    for k in range(min(nb, n)):
        p0 = first + k
        sel = pages[k::nb]
        banks.write(dev, p0 % nb, base + (p0 // nb) * page_bytes, sel.contiguous().numpy().tobytes())


def table_host() -> str:
    """The table's host tag: ``host-<crc32(hostname) & 0x7fffffff>``, what tt-d-gen's KV Manager derives from
    ``KV_MANAGER_TABLE_HOST=$(hostname)`` (the tag the ring's tables carry)."""
    return f"host-{zlib.crc32(socket.gethostname().encode()) & 0x7FFFFFFF:08x}"


# --------------------------------------------------------------------------------------------------- export slabs
@dataclass
class GlmRingExport:
    """The engine-owned export slabs of one prefill slot (``KvCaches`` for the common runner)."""

    mesh_device: object
    max_seq_len: int
    versions: int
    dsa_layers: tuple
    kda_layers: tuple
    latent: object = None
    pool: object = None
    kda_rec: object = None
    kda_conv: object = None
    pow2_scale: bool = False
    stats: dict = field(default_factory=dict)

    @classmethod
    def allocate(cls, mesh_device, *, max_seq_len: int, versions: int = 2, with_mtp: bool = True) -> "GlmRingExport":
        from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill
        from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import nd_dram_memory_config

        rows, cols = tuple(mesh_device.shape)
        dsa = DSA_LAYERS + ((MTP_LAYER,) if with_mtp else ())
        assert len(dsa) <= rows * cols, (len(dsa), rows, cols)
        assert max_seq_len % POOL_CHUNK_TOKENS == 0, max_seq_len
        r_lat, r_pool = max_seq_len // TPP, max_seq_len // POOL_CHUNK_TOKENS
        dram = ttnn.DRAM_MEMORY_CONFIG
        latent = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, r_lat, LATENT_PAGE_BYTES]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, mesh_device, dram
        )
        pool = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, r_pool, POOL_ROW_BYTES]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, mesh_device, dram
        )
        L = len(KDA_LAYERS)
        hl = KDA_HEADS // cols
        kda_rec = ttnn.allocate_tensor_on_device(
            ttnn.Shape([L, hl, KDA_HEAD_DIM, KDA_HEAD_DIM]),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            mesh_device,
            nd_dram_memory_config(mesh_device, [1, 1, KDA_HEAD_DIM, KDA_BAND]),
        )
        DRAMZeroFill.op(kda_rec)
        kda_conv = ttnn.zeros(
            (L, KDA_BRANCHES * hl, KDA_TAPS * KDA_HEAD_DIM),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=dram,
        )
        for name, t, page in (
            ("latent", latent, LATENT_PAGE_BYTES),
            ("pool", pool, POOL_ROW_BYTES),
            ("kda_conv", kda_conv, KDA_CONV_SEGMENT),
        ):
            assert t.buffer_aligned_page_size() == page, (name, t.buffer_aligned_page_size(), page)
        assert KDA_S_SEGMENT % kda_rec.buffer_aligned_page_size() == 0, kda_rec.buffer_aligned_page_size()
        ex = cls(
            mesh_device=mesh_device,
            max_seq_len=max_seq_len,
            versions=versions,
            dsa_layers=dsa,
            kda_layers=KDA_LAYERS,
            latent=latent,
            pool=pool,
            kda_rec=kda_rec,
            kda_conv=kda_conv,
            pow2_scale=os.environ.get("GLM_KV_POW2_SCALE", "0") == "1",
        )
        ex._dev_ids = [int(d) for d in mesh_device.get_device_ids()]
        ex._banks = _Banks(ex._dev_ids)
        ex._nb = int(mesh_device.dram_grid_size().x)
        ex.zero_dsa()
        mb = lambda t, n: n * t.buffer_aligned_page_size() / 2**20  # noqa: E731
        logger.info(
            f"[glm53f-export] slabs: latent {mb(latent, r_lat):.0f} MiB + pool {mb(pool, r_pool):.0f} MiB per chip "
            f"(DSA layers {dsa} on devices 0..{len(dsa) - 1}), KDA {L} layers rec + conv; max_seq_len {max_seq_len}, "
            f"KDA versions {versions}, pow2 scale {ex.pow2_scale}, banks {ex._nb}, host {table_host()}"
        )
        return ex

    # ---- placement
    def dsa_device(self, layer: int) -> int:
        return self.dsa_layers.index(layer)

    def zero_dsa(self) -> None:
        """Zero every DSA slab once (the MTP layer is never written: its chunks stay zero until the MTP cache is built)."""
        for slab, page in ((self.latent, LATENT_PAGE_BYTES), (self.pool, POOL_ROW_BYTES)):
            n = int(slab.shape[-2])
            for i, _layer in enumerate(self.dsa_layers):
                _write_pages(
                    self._banks,
                    self._dev_ids[i],
                    slab.buffer_address(),
                    self._nb,
                    page,
                    0,
                    torch.zeros(n, page, dtype=torch.uint8),
                )

    # ---- per-chunk export
    def export_dsa(
        self, layer: int, latent: torch.Tensor, latent_first: int, pools: torch.Tensor | None, pool_first: int
    ) -> int:
        """Latent rows [32 * latent_first, ..) (whole pages, zero-padded) and pool keys [32 * pool_first, ..) (whole tile
        rows) of DSA layer ``layer`` into its slab. Returns the bytes written."""
        dev = self._dev_ids[self.dsa_device(layer)]
        n = int(latent.shape[0])
        if n % TPP:
            pad = torch.zeros(TPP - n % TPP, LATENT_DIM, dtype=latent.dtype)
            latent = torch.cat([latent, pad])
        pages = pack_latent_pages(latent, self.pow2_scale)
        _write_pages(self._banks, dev, self.latent.buffer_address(), self._nb, LATENT_PAGE_BYTES, latent_first, pages)
        nbytes = pages.numel()
        if pools is not None and int(pools.shape[0]):
            m = int(pools.shape[0])
            if m % 32:
                pools = torch.cat([pools.float(), torch.zeros(32 - m % 32, POOL_WIDTH)])
            rows = pack_pool_rows(pools)
            _write_pages(self._banks, dev, self.pool.buffer_address(), self._nb, POOL_ROW_BYTES, pool_first, rows)
            nbytes += rows.numel()
        self._banks.read(
            dev, 0, self.latent.buffer_address(), 64
        )  # a read behind the posted writes on this chip's path
        return nbytes

    def export_kda(self, layer: int, state) -> None:
        """One KDA layer's native carry (``recurrent`` [1, 16, 128, 128] fp32 TILE; ``convolution`` [1, 3, 6144] bf16
        row-major or tap-major tiles [3, 1, 6144]) into the slabs (device ops, every chip)."""
        k = self.kda_layers.index(layer)
        rec = state.recurrent
        if int(rec.shape[0]) != 1:
            rec = ttnn.slice(rec, [0, 0, 0, 0], [1, *[int(s) for s in rec.shape[1:]]])
        ttnn.kv_cache.fill_cache_for_user_(self.kda_rec, rec, k)
        if rec is not state.recurrent:
            ttnn.deallocate(rec)
        conv = state.convolution
        tmp = []
        if conv.layout == ttnn.TILE_LAYOUT:  # tap-major tiles [taps, B, W] -> row-major [1, taps, W]
            c = ttnn.to_layout(conv, ttnn.ROW_MAJOR_LAYOUT)
            tmp.append(c)
            c = ttnn.permute(c, (1, 0, 2))
            tmp.append(c)
            conv = c
        taps, width = int(conv.shape[-2]), int(conv.shape[-1])
        assert taps == KDA_TAPS and width % KDA_HEAD_DIM == 0, tuple(conv.shape)
        if int(conv.shape[0]) != 1:
            conv = ttnn.slice(conv, [0, 0, 0], [1, taps, width])
            tmp.append(conv)
        x = ttnn.reshape(conv, (1, taps, width // KDA_HEAD_DIM, KDA_HEAD_DIM))
        x = ttnn.permute(x, (0, 2, 1, 3))
        tmp.append(x)
        rows = ttnn.reshape(x, (1, width // KDA_HEAD_DIM, taps * KDA_HEAD_DIM))
        seg, w = int(rows.shape[1]), int(rows.shape[2])
        ttnn.experimental.slice_write(rows, self.kda_conv, [k, 0, 0], [k + 1, seg, w], [1, 1, 1])
        keep = {state.convolution.buffer_address(), self.kda_conv.buffer_address()}
        for t in tmp + [rows]:
            try:
                if t.buffer_address() not in keep:
                    keep.add(t.buffer_address())
                    ttnn.deallocate(t)
            except RuntimeError:
                pass

    # ---- the table
    def build_table(self, path: str, stage_layouts=None) -> str:
        """Serialize the KvChunkAddressTable of slot 0 to ``path`` (protobuf). Single-rank prefill: every chunk's chip is
        this mesh's; device groups: one chip per DSA layer, one TP column (all SP rows, replicas) per KDA segment."""
        from models.demos.common.prefill.runners.migration import serialize_prebuilt_kv_chunk_table

        disagg = ttnn.experimental.disaggregation
        md = self.mesh_device
        rows, cols = tuple(md.shape)
        host = table_host()
        nb = self._nb

        def cfg(seq, n_tokens, nbytes):
            c = disagg.KvChunkAddressTableConfig()
            c.num_layers, c.max_sequence_length, c.num_slots = TABLE_LAYERS, seq, 1
            c.chunk_n_tokens, c.chunk_size_bytes = n_tokens, nbytes
            return c

        kda_seq = self.versions * KDA_WINDOW
        configs = {
            CFG_DSA_KV: cfg(self.max_seq_len, TPP, LATENT_PAGE_BYTES),
            CFG_DSA_POOL: cfg(self.max_seq_len, POOL_CHUNK_TOKENS, POOL_ROW_BYTES),
            CFG_KDA_S: cfg(kda_seq, KDA_S_STRIDE, KDA_S_SEGMENT),
            CFG_KDA_CONV: cfg(kda_seq, KDA_CONV_STRIDE, KDA_CONV_SEGMENT),
        }
        table = disagg.KvChunkAddressTable(configs)
        fnid = lambda r, c: md.get_fabric_node_id(ttnn.MeshCoordinate(r, c))  # noqa: E731
        groups = {}

        def group(members):
            key = tuple((int(f.mesh_id), int(f.chip_id)) for f in members)
            if key not in groups:
                groups[key] = table.add_device_group(members)
                for f in members:
                    table.set_fabric_node_host(f, host_name=host)
            return groups[key]

        def loc(bank, addr, size, g):
            x = disagg.KvCacheLocation()
            x.noc_addr = (int(bank) << 32) | int(addr)
            x.size_bytes = int(size)
            x.device_group_index = g
            return x

        n = 0
        cid = {name: table.config_id_of(name) for name in configs}
        lat_base, pool_base = int(self.latent.buffer_address()), int(self.pool.buffer_address())
        for i, layer in enumerate(self.dsa_layers):
            g = group([fnid(i // cols, i % cols)])
            for p in range(self.max_seq_len // TPP):
                table.set(
                    layer,
                    p * TPP,
                    0,
                    loc(p % nb, lat_base + (p // nb) * LATENT_PAGE_BYTES, LATENT_PAGE_BYTES, g),
                    cid[CFG_DSA_KV],
                )
                n += 1
            for r in range(self.max_seq_len // POOL_CHUNK_TOKENS):
                table.set(
                    layer,
                    r * POOL_CHUNK_TOKENS,
                    0,
                    loc(r % nb, pool_base + (r // nb) * POOL_ROW_BYTES, POOL_ROW_BYTES, g),
                    cid[CFG_DSA_POOL],
                )
                n += 1
        hl = KDA_HEADS // cols
        rec_base, conv_base = int(self.kda_rec.buffer_address()), int(self.kda_conv.buffer_address())
        for c in range(cols):
            g = group([fnid(r, c) for r in range(rows)])
            for k, layer in enumerate(self.kda_layers):
                for h in range(hl):
                    for b in range(KDA_BANDS):
                        s = (k * hl + h) * KDA_BANDS + b  # ND shard [1, 1, 128, 32] of the [L, hl, 128, 128] slab
                        G = (c * hl + h) * KDA_BANDS + b
                        x = loc(s % nb, rec_base + (s // nb) * KDA_S_SEGMENT, KDA_S_SEGMENT, g)
                        for v in range(self.versions):
                            table.set(layer, v * KDA_WINDOW + G * KDA_S_STRIDE, 0, x, cid[CFG_KDA_S])
                            n += 1
                for br in range(KDA_BRANCHES):
                    for h in range(hl):
                        p = k * KDA_BRANCHES * hl + br * hl + h  # page of the [L, 3 * hl, 384] conv slab
                        G = br * KDA_HEADS + c * hl + h
                        x = loc(p % nb, conv_base + (p // nb) * KDA_CONV_SEGMENT, KDA_CONV_SEGMENT, g)
                        for v in range(self.versions):
                            table.set(layer, v * KDA_WINDOW + G * KDA_CONV_STRIDE, 0, x, cid[CFG_KDA_CONV])
                            n += 1
        logger.info(f"[glm53f-export] table: {n} chunk entries, {len(groups)} device groups, host {host} -> {path}")
        return serialize_prebuilt_kv_chunk_table(table=table, path=path)


def kda_window_range(real_len: int, versions: int) -> tuple[int, int]:
    """The KDA positions the decode reads after a ``real_len``-token prefill: window (real_len - 1) % versions."""
    v = (int(real_len) - 1) % int(versions) if real_len else 0
    return v * KDA_WINDOW, (v + 1) * KDA_WINDOW


def log_export(stats: dict, start: int, end: int, t0: float) -> None:
    stats["chunks"] = stats.get("chunks", 0) + 1
    stats["s"] = stats.get("s", 0.0) + (time.perf_counter() - t0)
    logger.info(f"[glm53f-export] chunk [{start}, {end}) exported in {time.perf_counter() - t0:.3f} s")

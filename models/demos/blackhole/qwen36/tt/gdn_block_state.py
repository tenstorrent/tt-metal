# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN state in vLLM-owned MambaSpec blocks (QWEN36_MAMBA_SPEC=1, mamba_cache_mode "none" or "align"; prototype).

vLLM overlays its KV cache groups: block id b names the same page in the attention layer of a layer slot
(``tensor_idx``) and in each GDN layer of that slot, and one group owns a block id at a time. A GDN layer keeps the
state of a request in the K and V buffers of its slot, at the block its group owns for that request:

* recurrent state: 16 fp32 pages per value head in a float32 alias_view of K, [pages / 8, 1, Dk / 2, Dv] (dim-0
  slices of half a head). Interleaved page p lives on bank p % nb, so block b owns bytes [b * s, (b + 1) * s) of every
  bank (s = block bytes / nb) and the pages that lie wholly inside them: nb * ceil(b * s / 4096) and the next ones.
  The first hk heads go to K. The others go to the same float32 view of V (gdn_decode_step's state_pool2) when K is
  too small: BF8 [N, 1, 1024, 256] holds 8.5 fp32 pages per bank, 64 whole pages per block, 4 heads at TP=8 of 6;
* packed conv history: 4 bf16 pages per head in a bfloat16 alias_view of V, [N, pages, 32, 32], at page t0 of the
  block: 0 when the state is in K alone, else the last whole page rows of the block (after the state in V).

Decode reads and writes them through gdn_decode_step(state_page_idx=...), one persistent [2 * Nv, Bmax] index tensor
per GDN layer (device-row order): row h = first fp32 page of head h (plus the page count of the K view for a head in
V), row Nv + h = first bf16 page of its history. In this pooled mode the op keeps channel chunk c in both tile rows
2c and 2c + 1 of the packed history, so a request may change batch-row parity. Prefill keeps writing the model's
[Bmax, ...] decode buffers (now a staging row per device row) and copies the row into the request's block afterwards,
duplicating the history into both parities. Fits: BF8 2048 holds 136 fp32 pages per block (TP=8 needs 96, K alone),
BF8 4096 holds 272 (TP=4 needs 192), BF8 1024 at TP=8 splits 4 + 2 heads over K and V.

A prefill that starts at start_pos > 0 (a prefix hit in mode "align", or a later chunk of a prompt) first loads the
request's block into the B=1 prefill scratch (GdnBlockState.restore). The runner's (src, dst) block copies
(copy_kv_blocks) move whole blocks, so they carry every GDN layer's state of the block's group with them.
"""

import math
from types import SimpleNamespace

import torch

import ttnn

FP32_PAGE_BYTES = 4096
BF16_PAGE_BYTES = 2048
HALF_HEAD_PAGES = 8  # fp32 pages of [Dk / 2, Dv] = [64, 128], one dim-0 slice of the state view


def _tile_bytes(dtype):
    return {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088}[dtype]


def _dram_banks(mesh):
    return int(mesh.dram_grid_size().x)


def _first_row(b, s, page):
    """First page row of a bank wholly inside block b, which owns bytes [b * s, (b + 1) * s) of the bank."""
    return -(-b * s // page)


def _rows_per_block(s, page):
    """Fewest whole page rows any block holds on a bank (the count repeats every page / gcd(s, page) blocks)."""
    period = page // math.gcd(s, page)
    return min((b + 1) * s // page - _first_row(b, s, page) for b in range(period))


def is_mamba_entry(entry):
    """A per_layer_specs entry of a MambaSpec layer: (shapes, dtypes, tensor_idx), shapes a tuple of shapes."""
    shapes = entry[0]
    return len(shapes) > 0 and isinstance(shapes[0], (tuple, list))


class GdnBlockState:
    """Binds every GDN layer of ``model`` to page views of its layer slot's KV buffers."""

    def __init__(self, model, per_layer_specs, kv_caches):
        mesh = model.mesh_device
        self.model = model
        self.mesh = mesh
        self.bmax = int(model.args.max_batch_size)
        attn_entries = [(i, e) for i, e in enumerate(per_layer_specs) if not is_mamba_entry(e)]
        assert [i for i, _ in attn_entries] == list(
            model._attention_layer_indices
        ), "attention entries of per_layer_specs must be the model's attention layers"
        kv_by_slot = {e[2]: kv_caches[n] for n, (_, e) in enumerate(attn_entries)}
        self.kv_caches = kv_caches
        self.layers = []  # (layer index, TPGatedDeltaNet)
        self._slot_layer = {}  # layer slot -> its first GDN layer (whose views cover the slot's K and V)
        self._host_idx = {}
        for i, e in enumerate(per_layer_specs):
            if not is_mamba_entry(e):
                continue
            dn = model.layers[i].attention
            assert not model.layers[i].is_full_attention
            if not (dn._decode_op and not dn.use_fused_recurrent_decode):
                raise ValueError("QWEN36_MAMBA_SPEC needs the gdn_decode_step decode (QWEN36_GDN_DECODE_STEP_OP=1)")
            k, v = kv_by_slot[e[2]]
            dn._paged_state = self._bind(dn, k, v)
            self.layers.append((i, dn))
            self._slot_layer.setdefault(e[2], dn)
        # The gather tables and the zero block live as long as the model. Allocate them now, before the warmup
        # captures any trace: a buffer allocated after a capture and still alive at a replay can be overwritten by it.
        for _, dn in self.layers:
            self._mirror_table(dn.Nv * 4, 0)
            self._mirror_table(dn.Nv * 4, 1)
            self._unpack_table(dn)
        if kv_caches:
            zero_block(model, kv_caches[0][0])

    def _bind(self, dn, k, v):
        num_blocks, heads, block, hd = (int(x) for x in k.shape)
        nb = _dram_banks(self.mesh)
        tiles = heads * block * hd // 1024
        if tiles % nb:
            raise ValueError(f"block {block}: {tiles} tiles per block do not spread evenly over {nb} DRAM banks")
        s_k = tiles // nb * _tile_bytes(k.dtype)
        s_v = tiles // nb * _tile_bytes(v.dtype)
        head_pages = dn.Dk * dn.Dv // 1024
        hist_tiles = tiles * _tile_bytes(v.dtype) // BF16_PAGE_BYTES
        hist_rows = -(-dn.Nv * 4 // nb)
        if s_v % BF16_PAGE_BYTES or hist_tiles < dn.Nv * 4:
            raise ValueError(f"block {block}: a V block does not hold {dn.Nv * 4} whole bf16 pages")
        hk = min(dn.Nv, _rows_per_block(s_k, FP32_PAGE_BYTES) * nb // head_pages)
        hv = dn.Nv - hk
        t0 = 0
        if hv:
            # The heads in V take the first whole fp32 rows of the block, the history its last hist_rows bf16 rows.
            rv = -(-hv * head_pages // nb)
            period = FP32_PAGE_BYTES // math.gcd(s_v, FP32_PAGE_BYTES)
            if any(
                (_first_row(b, s_v, FP32_PAGE_BYTES) + rv) * FP32_PAGE_BYTES
                > (b + 1) * s_v - hist_rows * BF16_PAGE_BYTES
                for b in range(period)
            ):
                raise ValueError(
                    f"block {block}: K holds {hk} of {dn.Nv} heads and V cannot hold the other {hv} with the history"
                )
            t0 = hist_tiles - hist_rows * nb
        rec_pages_k = nb * (num_blocks * s_k // FP32_PAGE_BYTES)
        rec_pages_v = nb * (num_blocks * s_v // FP32_PAGE_BYTES)
        # Half-head slices of the views must start on 8-page boundaries.
        if nb % HALF_HEAD_PAGES and (s_k % (FP32_PAGE_BYTES * HALF_HEAD_PAGES) or hv):
            raise ValueError(f"{nb} DRAM banks: block {block} needs the state view in 8-page slices")
        rec_view = ttnn.experimental.alias_view(
            k,
            [rec_pages_k // HALF_HEAD_PAGES, 1, dn.Dk // 2, dn.Dv],
            ttnn.float32,
            check_slices=s_k % FP32_PAGE_BYTES == 0,
        )
        rec_view2 = None
        if hv:
            rec_view2 = ttnn.experimental.alias_view(
                v,
                [rec_pages_v // HALF_HEAD_PAGES, 1, dn.Dk // 2, dn.Dv],
                ttnn.float32,
                check_slices=s_v % FP32_PAGE_BYTES == 0,
            )
        hist_view = ttnn.experimental.alias_view(v, [num_blocks, hist_tiles, 32, 32], ttnn.bfloat16)
        idx = ttnn.from_torch(
            torch.zeros(2 * dn.Nv, self.bmax, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        return SimpleNamespace(
            rec_pool=rec_view,
            rec_pool2=rec_view2,
            hist_pool=hist_view,
            rec_view=rec_view,
            rec_view2=rec_view2,
            hist_view=hist_view,
            idx=idx,
            banks=nb,
            bank_bytes_k=s_k,
            bank_bytes_v=s_v,
            heads_k=hk,
            rec_pages_k=rec_pages_k,
            hist_pages=hist_tiles,
            hist_first=t0,
        )

    @staticmethod
    def _rec_pages(ps, b):
        """First fp32 page of block b in the K view and in the V view."""
        return (
            ps.banks * _first_row(b, ps.bank_bytes_k, FP32_PAGE_BYTES),
            ps.banks * _first_row(b, ps.bank_bytes_v, FP32_PAGE_BYTES),
        )

    @staticmethod
    def _block_ids(blocks, g, n):
        return [int(x) for x in blocks[g].reshape(-1)[:n].tolist()]

    def set_decode_blocks(self, blocks, group_of, row_of=None):
        """Point device row d of every GDN layer at its request's block. ``blocks``: per KV cache group, int [R]
        (entry i = the block of submitted row i); ``group_of``: {GDN layer index: its group's index in ``blocks``};
        ``row_of[i]`` = device row of submitted row i (None = identity). Rows no request occupies point at block 0,
        vLLM's null block. Copies only changed indices."""
        rows = []
        for g in range(len(blocks)):
            row = [0] * self.bmax
            for r, b in enumerate(self._block_ids(blocks, g, self.bmax)):
                row[r if row_of is None else row_of[r]] = b
            rows.append(tuple(row))
        built = {}  # the layers of one group share its blocks: one host index per group and page geometry
        for i, dn in self.layers:
            g = group_of[i]
            key = rows[g]
            prev = self._host_idx.get(i)
            if prev is not None and prev[0] == key:
                continue
            ps = dn._paged_state
            geom = (g, ps.banks, ps.bank_bytes_k, ps.bank_bytes_v, ps.heads_k, ps.rec_pages_k, ps.hist_pages)
            geom += (ps.hist_first, dn.Nv, dn.Dk * dn.Dv // 1024)
            host = built.get(geom)
            if host is None:
                host = built[geom] = self._index_host(ps, dn, list(key))
            ttnn.copy_host_to_device_tensor(host, ps.idx)
            # Held until the index changes again: the copy may read the host tensor after this call returns.
            self._host_idx[i] = (key, host)

    def _index_host(self, ps, dn, blocks):
        """Host uint32 [2 * Nv, Bmax] gdn_decode_step state_page_idx for device rows holding `blocks`."""
        hp = dn.Dk * dn.Dv // 1024
        b = torch.tensor(blocks, dtype=torch.int64)
        pk = ps.banks * (-(-b * ps.bank_bytes_k // FP32_PAGE_BYTES))
        pv = ps.rec_pages_k + ps.banks * (-(-b * ps.bank_bytes_v // FP32_PAGE_BYTES))
        h = torch.arange(dn.Nv, dtype=torch.int64).reshape(-1, 1)
        rec = torch.where(h < ps.heads_k, pk + hp * h, pv + hp * (h - ps.heads_k))
        host = torch.cat([rec, b * ps.hist_pages + ps.hist_first + 4 * h]).to(torch.int32)
        return ttnn.from_torch(
            host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )

    def write_prefill_rows(self, blocks, group_of, rows):
        """Copy device row rows[u] of every GDN layer's decode buffers (the state prefill just wrote, packed for the
        row's parity) into the block of prefilled request u (``blocks`` and ``group_of`` as for set_decode_blocks)."""
        self.model.prepare_gdn_decode_width(self.bmax)
        ids_of = [self._block_ids(blocks, g, len(rows)) for g in range(len(blocks))]
        for i, dn in self.layers:
            ids = ids_of[group_of[i]]
            ps = dn._paged_state
            nv, dk, dv, nr = dn.Nv, dn.Dk, dn.Dv, dn.Nv * 4
            halves, hk2 = 2 * nv, 2 * ps.heads_k
            rec_rows = ttnn.reshape(dn.rec_state, (dn._decode_B * halves, 1, dk // 2, dv))
            packed = ttnn.reshape(dn._conv_hist_packed, (dn._decode_B, nr, 32, 32))
            for d, blk in zip(rows, ids):
                assert blk > 0, f"layer {i}: prefill row {d} has the null block"
                pk, pv = self._rec_pages(ps, blk)
                parts = [(ps.rec_view, pk // HALF_HEAD_PAGES, 0, hk2)]
                if hk2 < halves:
                    parts.append((ps.rec_view2, pv // HALF_HEAD_PAGES, hk2, halves))
                for view, s0, lo, hi in parts:
                    # L1-sharded sources: a DRAM source makes slice_write re-seat the destination buffer.
                    src = ttnn.slice(
                        rec_rows,
                        (d * halves + lo, 0, 0, 0),
                        (d * halves + hi, 1, dk // 2, dv),
                        memory_config=self._halves_shard_memcfg(dn, hi - lo),
                    )
                    ttnn.experimental.slice_write(
                        src, view, [s0, 0, 0, 0], [s0 + hi - lo, 1, dk // 2, dv], [1, 1, 1, 1]
                    )
                    ttnn.deallocate(src)
                row = ttnn.to_layout(ttnn.slice(packed, (d, 0, 0, 0), (d + 1, nr, 32, 32)), ttnn.ROW_MAJOR_LAYOUT)
                src = ttnn.embedding(
                    self._mirror_table(nr, d & 1),
                    ttnn.reshape(row, (nr * 32, 32)),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=dn._packed_row_shard_memcfg(),
                )  # [1, nr * 32, 32]
                ttnn.deallocate(row)
                src = ttnn.reshape(src, (1, nr, 32, 32))
                t0 = ps.hist_first
                ttnn.experimental.slice_write(
                    src, ps.hist_view, [blk, t0, 0, 0], [blk + 1, t0 + nr, 32, 32], [1, 1, 1, 1]
                )
                ttnn.deallocate(src)

    def copy_blocks(self, copies):
        """vLLM's (src, dst) state block copies, in list order: whole blocks (copy_kv_blocks), then the head pages
        again wherever src and dst hold them at different offsets inside the block. When a block is not a whole number
        of fp32 pages per bank, its whole pages start at an offset that depends on the block (BF8 1024: 0 for even
        blocks, 2048 B for odd ones), and a byte copy keeps the source's offset. The bf16 history pages are whole in
        every block (17 per bank at BF8 1024), so the byte copy places them right."""
        copy_kv_blocks(self.model, copies)
        for src, dst in copies:
            src, dst = int(src), int(dst)
            if src == dst:
                continue
            for dn in self._slot_layer.values():
                ps = dn._paged_state
                (pk_s, pv_s), (pk_d, pv_d) = self._rec_pages(ps, src), self._rec_pages(ps, dst)
                parts = [(ps.rec_view, ps.bank_bytes_k, pk_s, pk_d, 2 * ps.heads_k)]
                if ps.rec_view2 is not None:
                    parts.append((ps.rec_view2, ps.bank_bytes_v, pv_s, pv_d, 2 * (dn.Nv - ps.heads_k)))
                for view, s, p_s, p_d, n in parts:
                    if (src * s) % FP32_PAGE_BYTES == (dst * s) % FP32_PAGE_BYTES:
                        continue
                    s0, d0 = p_s // HALF_HEAD_PAGES, p_d // HALF_HEAD_PAGES
                    halves = ttnn.slice(
                        view,
                        (s0, 0, 0, 0),
                        (s0 + n, 1, dn.Dk // 2, dn.Dv),
                        memory_config=self._halves_shard_memcfg(dn, n),
                    )
                    ttnn.experimental.slice_write(
                        halves, view, [d0, 0, 0, 0], [d0 + n, 1, dn.Dk // 2, dn.Dv], [1, 1, 1, 1]
                    )
                    ttnn.deallocate(halves)

    def set_prefill_blocks(self, blocks, group_of):
        """The state blocks of this prefill step (entry u = prefilled request u; ``blocks`` and ``group_of`` as for
        set_decode_blocks), for restore()."""
        self._prefill_ids = (blocks, group_of)

    def restore(self, u):
        """Load the state block of prefilled request u into the bound B=1 prefill scratch of every GDN layer
        (rec_state [1, Nv, Dk, Dv] fp32, conv_carry [1, K - 1, C] bf16), so the prefill resumes at vLLM's start_pos.
        The block holds the state after exactly start_pos tokens: the runner copied a prefix hit's boundary state into
        it (mamba_cache_mode "align"), or the previous chunk of this prompt wrote it. Same contract as
        GdnPrefixStateCache.restore (Qwen36Model._prefill_with_prefix_cache calls either)."""
        blocks, group_of = self._prefill_ids
        for i, dn in self.layers:
            blk = int(blocks[group_of[i]].reshape(-1)[u])
            if blk <= 0:
                raise ValueError(f"layer {i}: resumed prefill of request {u} has the null block {blk}")
            ps = dn._paged_state
            nv, dk, dv, nr = dn.Nv, dn.Dk, dn.Dv, dn.Nv * 4
            hk2 = 2 * ps.heads_k
            pk, pv = self._rec_pages(ps, blk)
            s0 = pk // HALF_HEAD_PAGES
            rec = ttnn.slice(ps.rec_view, (s0, 0, 0, 0), (s0 + hk2, 1, dk // 2, dv))
            if hk2 < 2 * nv:
                s1 = pv // HALF_HEAD_PAGES
                rec2 = ttnn.slice(ps.rec_view2, (s1, 0, 0, 0), (s1 + 2 * nv - hk2, 1, dk // 2, dv))
                both = ttnn.concat([rec, rec2], dim=0)
                ttnn.deallocate(rec)
                ttnn.deallocate(rec2)
                rec = both
            ttnn.copy(ttnn.reshape(rec, (1, nv, dk, dv)), dn.rec_state)
            ttnn.deallocate(rec)
            t0 = ps.hist_first
            hist = ttnn.to_layout(
                ttnn.slice(ps.hist_view, (blk, t0, 0, 0), (blk + 1, t0 + nr, 32, 32)), ttnn.ROW_MAJOR_LAYOUT
            )
            rows = ttnn.embedding(
                self._unpack_table(dn), ttnn.reshape(hist, (nr * 32, 32)), layout=ttnn.ROW_MAJOR_LAYOUT
            )  # [1, (K - 1) * C / 32, 32]
            ttnn.deallocate(hist)
            carry = ttnn.to_layout(ttnn.reshape(rows, (1, dn.K - 1, dn.qkv_dim_tp)), ttnn.TILE_LAYOUT)
            ttnn.deallocate(rows)
            ttnn.copy(carry, dn.conv_carry)
            ttnn.deallocate(carry)

    def _halves_shard_memcfg(self, dn, halves):
        """L1 height-sharded config for `halves` half-head slices [halves, 1, Dk / 2, Dv], with the core choice of
        TPGatedDeltaNet._rec_row_shard_memcfg for that row count (the same config for a whole row). slice_write writes
        into the destination view in place only from an L1-sharded source whose shards it fills exactly."""
        cfgs = self.__dict__.setdefault("_halves_shard_cfgs", {})
        key = (halves, dn.Dk, dn.Dv)
        if key not in cfgs:
            grid_size = self.mesh.compute_with_storage_grid_size()
            rows = halves * dn.Dk // 2
            n_tiles = rows // ttnn.TILE_SIZE
            num_cores = max(c for c in range(1, min(48, grid_size.x * grid_size.y) + 1) if n_tiles % c == 0)
            cfgs[key] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True),
                    (rows // num_cores, dn.Dv),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
        return cfgs[key]

    def _unpack_table(self, dn):
        """uint32 [1, (K - 1) * C / 32] gather table: 32-channel row (j, ch) of conv_carry <- packed row
        (h * 4 + j + 1) * 32 + 2c of one block's history, viewed as [Nv * 4 * 32, 32] rows. (h, c) is the head and
        chunk of channel group ch in [q | k | v], as in TPGatedDeltaNet._pack_tables "to_rm" for batch row 0; the
        pooled history holds the same chunk at parity 1."""
        tabs = self.__dict__.setdefault("_unpack_tables", {})
        key = (dn.Nv, dn.Nk, dn.Dk, dn.Dv, dn.K)
        if key not in tabs:
            tabs[key] = ttnn.from_torch(
                unpack_rows(*key).reshape(1, -1).to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        return tabs[key]

    def _mirror_table(self, nr, parity):
        """uint32 [1, nr * 32] gather table: packed tile row r of tile t <- row 2 * (r // 2) + parity of the same tile.
        The decode row's packing (chunk c in row 2c + parity, the other parity zero) becomes the pooled packing (chunk c
        in rows 2c and 2c + 1)."""
        tabs = self.__dict__.setdefault("_mirror_tables", {})
        if (nr, parity) not in tabs:
            r = torch.arange(32)
            rows = (torch.arange(nr).reshape(-1, 1) * 32 + 2 * (r // 2) + parity).reshape(1, -1)
            tabs[(nr, parity)] = ttnn.from_torch(
                rows.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        return tabs[(nr, parity)]


def unpack_rows(nv, nk, dk, dv, k):
    """int64 [K - 1, C / 32]: the packed history row of each 32-channel group of conv_carry (see
    GdnBlockState._unpack_table)."""
    rf, qn, vn = nv // nk, dk // 32, dv // 32
    kd32 = nk * dk // 32
    cpr = 2 * kd32 + nv * vn
    out = torch.zeros(k - 1, cpr, dtype=torch.int64)
    for j in range(k - 1):
        for ch in range(cpr):
            if ch < kd32:  # q of key head ch // qn, read by its first value head
                h, c = (ch // qn) * rf, ch % qn
            elif ch < 2 * kd32:  # k
                h, c = ((ch - kd32) // qn) * rf, qn + (ch - kd32) % qn
            else:  # v
                h, c = (ch - 2 * kd32) // vn, 2 * qn + (ch - 2 * kd32) % vn
            out[j, ch] = (h * 4 + j + 1) * 32 + 2 * c
    return out


def copy_kv_blocks(model, copies):
    """vLLM's (src, dst) block copies, in list order: copy the whole block in every K and V buffer of
    model._paged_kv_caches. A block id belongs to one KV cache group at a time and each layer slot holds that group's
    layer at the block, so a whole-block copy moves exactly that group's content (attention K and V, or the recurrent
    state and conv history of the group's GDN layers). paged_fill_cache is a tile copy (dataflow kernels only), so the
    fp32 and bf16 bytes inside a BF8 buffer move unchanged."""
    kv_caches = getattr(model, "_paged_kv_caches", None)
    if not copies or not kv_caches:
        return
    mesh = model.mesh_device
    for src, dst in copies:
        src, dst = int(src), int(dst)
        if src == dst:
            continue
        if src <= 0 or dst <= 0:
            raise ValueError(f"block copy {src} -> {dst} names the null block 0")
        pt = ttnn.from_torch(
            torch.tensor([[dst]], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for kv in kv_caches:
            for t in kv:
                end = [src + 1] + [int(x) for x in list(t.shape)[1:]]
                blk = ttnn.slice(t, (src, 0, 0, 0), end, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.experimental.paged_fill_cache(t, blk, pt, batch_idx=0)
                ttnn.deallocate(blk)
        ttnn.deallocate(pt)


def zero_block(model, k0):
    """model._zero_kv_block: one zero block shaped and typed like the paged buffer k0 (created on the first call)."""
    shape = [1] + [int(x) for x in list(k0.shape)[1:]]
    zeros = getattr(model, "_zero_kv_block", None)
    if zeros is None or list(zeros.shape) != shape or zeros.dtype != k0.dtype:
        mesh = model.mesh_device
        zeros = model._zero_kv_block = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=k0.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
    return zeros


def zero_kv_blocks(model, block_ids):
    """Zero blocks in every K and V buffer of model._paged_kv_caches (vLLM's new_block_ids_to_zero: a block that held
    fp32 GDN state reads as NaN / Inf in bf16 attention)."""
    kv_caches = getattr(model, "_paged_kv_caches", None)
    if not block_ids or not kv_caches:
        return
    mesh = model.mesh_device
    zeros = zero_block(model, kv_caches[0][0])
    for blk in block_ids:
        pt = ttnn.from_torch(
            torch.tensor([[blk]], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for k, v in kv_caches:
            ttnn.experimental.paged_fill_cache(k, zeros, pt, batch_idx=0)
            ttnn.experimental.paged_fill_cache(v, zeros, pt, batch_idx=0)
        ttnn.deallocate(pt)

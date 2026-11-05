# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN state in vLLM-owned MambaSpec blocks (QWEN36_MAMBA_SPEC=1, mamba_cache_mode "none" or "align"; prototype).

vLLM overlays its KV cache groups: block id b names the same page in the attention layer of a layer slot
(``tensor_idx``) and in each GDN layer of that slot, and one group owns a block id at a time. A GDN layer keeps the
state of a request in the K and V buffers of its slot, at the block its group owns for that request:

* recurrent state: the first Nv * 16 pages of block b in a float32 alias_view of K, [N * pages / 8, 1, Dk / 2, Dv]
  (dim-0 slices of half a head, so a block need not hold whole heads: BF8 2048 holds 136 pages, 17 slices);
* packed conv history: the first Nv * 4 pages of block b in a bfloat16 alias_view of V, [N, pages, 32, 32].

Decode reads and writes them through gdn_decode_step(state_page_idx=...), one persistent [2 * Nv, Bmax] index tensor
per GDN layer (device-row order): row h = first fp32 page of head h, row Nv + h = first bf16 page of its history. In
this pooled mode the op keeps channel chunk c in both tile rows 2c and 2c + 1 of the packed history, so a request may
change batch-row parity. Prefill keeps writing the model's [Bmax, ...] decode buffers (now a staging row per
device row) and copies the row into the request's block afterwards, duplicating the history into both parities. The
views use alias_view's default checks, so a block must hold whole fp32 and bf16 pages on every bank: BF8
[N, 1, 2048, 256] holds 17 fp32 pages per bank (136 per block; TP=8 needs 96), BF8 4096 holds 272 (TP=4 needs 192),
bf16 2048 holds 256.

A prefill that starts at start_pos > 0 (a prefix hit in mode "align", or a later chunk of a prompt) first loads the
request's block into the B=1 prefill scratch (GdnBlockState.restore). The runner's (src, dst) block copies
(copy_kv_blocks) move whole blocks, so they carry every GDN layer's state of the block's group with them.
"""

from types import SimpleNamespace

import torch

import ttnn

FP32_PAGE_BYTES = 4096
HALF_HEAD_PAGES = 8  # fp32 pages of [Dk / 2, Dv] = [64, 128], one dim-0 slice of the state view


def _tile_bytes(dtype):
    return {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088}[dtype]


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

    def _bind(self, dn, k, v):
        num_blocks, heads, block, hd = (int(x) for x in k.shape)
        tiles = heads * block * hd // 1024
        k_bytes = tiles * _tile_bytes(k.dtype)
        v_bytes = tiles * _tile_bytes(v.dtype)
        if k_bytes % (FP32_PAGE_BYTES * HALF_HEAD_PAGES):
            raise ValueError(f"K block of {k_bytes} B is not a whole number of 8-page fp32 slices (block {block})")
        rec_pages = k_bytes // FP32_PAGE_BYTES
        if rec_pages < dn.Nv * 2 * HALF_HEAD_PAGES:
            raise ValueError(f"K block holds {rec_pages} fp32 pages, layer needs {dn.Nv * 16}; use a larger block")
        hist_tiles = v_bytes // _tile_bytes(ttnn.bfloat16)
        if v_bytes % _tile_bytes(ttnn.bfloat16) or hist_tiles < dn.Nv * 4:
            raise ValueError(f"V block of {v_bytes} B does not hold {dn.Nv * 4} whole bf16 tiles (block {block})")
        rec_view = ttnn.experimental.alias_view(
            k, [num_blocks * rec_pages // HALF_HEAD_PAGES, 1, dn.Dk // 2, dn.Dv], ttnn.float32
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
            hist_pool=hist_view,
            rec_view=rec_view,
            hist_view=hist_view,
            idx=idx,
            rec_pages=rec_pages,
            hist_pages=hist_tiles,
        )

    def _block_ids(self, state_block_ids_per_layer, i, n):
        t = state_block_ids_per_layer[i]
        assert t is not None, f"layer {i} is a GDN layer but has no state block ids"
        return [int(x) for x in t.reshape(-1)[:n].tolist()]

    def set_decode_blocks(self, state_block_ids_per_layer, row_of=None):
        """Point device row d of every GDN layer at its request's block. ``row_of[i]`` = device row of submitted row i
        (None = identity); rows no request occupies point at block 0, vLLM's null block. Copies only changed indices."""
        for i, dn in self.layers:
            ids = self._block_ids(state_block_ids_per_layer, i, self.bmax)
            blocks = [0] * self.bmax
            for r, b in enumerate(ids):
                blocks[r if row_of is None else row_of[r]] = b
            ps = dn._paged_state
            b = torch.tensor(blocks, dtype=torch.int32)
            h = torch.arange(dn.Nv, dtype=torch.int32).reshape(-1, 1)
            host = torch.cat([b * ps.rec_pages + 16 * h, b * ps.hist_pages + 4 * h])  # [2 * Nv, Bmax]
            prev = self._host_idx.get(i)
            if prev is not None and torch.equal(prev, host):
                continue
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    host,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                ),
                ps.idx,
            )
            self._host_idx[i] = host

    def write_prefill_rows(self, state_block_ids_per_layer, rows):
        """Copy device row rows[u] of every GDN layer's decode buffers (the state prefill just wrote, packed for the
        row's parity) into the block of prefilled request u."""
        self.model.prepare_gdn_decode_width(self.bmax)
        for i, dn in self.layers:
            ids = self._block_ids(state_block_ids_per_layer, i, len(rows))
            ps = dn._paged_state
            nv, dk, dv, nr = dn.Nv, dn.Dk, dn.Dv, dn.Nv * 4
            halves, per_blk = 2 * nv, ps.rec_pages // HALF_HEAD_PAGES
            rec_rows = ttnn.reshape(dn.rec_state, (dn._decode_B * halves, 1, dk // 2, dv))
            packed = ttnn.reshape(dn._conv_hist_packed, (dn._decode_B, nr, 32, 32))
            for d, blk in zip(rows, ids):
                assert blk > 0, f"layer {i}: prefill row {d} has the null block"
                # L1-sharded sources: a DRAM source makes slice_write re-seat the destination buffer.
                src = ttnn.slice(
                    rec_rows,
                    (d * halves, 0, 0, 0),
                    ((d + 1) * halves, 1, dk // 2, dv),
                    memory_config=dn._rec_row_shard_memcfg(),
                )
                s0 = blk * per_blk
                ttnn.experimental.slice_write(
                    src, ps.rec_view, [s0, 0, 0, 0], [s0 + halves, 1, dk // 2, dv], [1, 1, 1, 1]
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
                ttnn.experimental.slice_write(src, ps.hist_view, [blk, 0, 0, 0], [blk + 1, nr, 32, 32], [1, 1, 1, 1])
                ttnn.deallocate(src)

    def set_prefill_blocks(self, state_block_ids_per_layer):
        """The state block ids of this prefill step (row u = prefilled request u), for restore()."""
        self._prefill_ids = state_block_ids_per_layer

    def restore(self, u):
        """Load the state block of prefilled request u into the bound B=1 prefill scratch of every GDN layer
        (rec_state [1, Nv, Dk, Dv] fp32, conv_carry [1, K - 1, C] bf16), so the prefill resumes at vLLM's start_pos.
        The block holds the state after exactly start_pos tokens: the runner copied a prefix hit's boundary state into
        it (mamba_cache_mode "align"), or the previous chunk of this prompt wrote it. Same contract as
        GdnPrefixStateCache.restore (Qwen36Model._prefill_with_prefix_cache calls either)."""
        for i, dn in self.layers:
            blk = self._block_ids(self._prefill_ids, i, u + 1)[u]
            if blk <= 0:
                raise ValueError(f"layer {i}: resumed prefill of request {u} has the null block {blk}")
            ps = dn._paged_state
            nv, dk, dv, nr = dn.Nv, dn.Dk, dn.Dv, dn.Nv * 4
            s0 = blk * (ps.rec_pages // HALF_HEAD_PAGES)
            rec = ttnn.slice(ps.rec_view, (s0, 0, 0, 0), (s0 + 2 * nv, 1, dk // 2, dv))
            ttnn.copy(ttnn.reshape(rec, (1, nv, dk, dv)), dn.rec_state)
            ttnn.deallocate(rec)
            hist = ttnn.to_layout(
                ttnn.slice(ps.hist_view, (blk, 0, 0, 0), (blk + 1, nr, 32, 32)), ttnn.ROW_MAJOR_LAYOUT
            )
            rows = ttnn.embedding(
                self._unpack_table(dn), ttnn.reshape(hist, (nr * 32, 32)), layout=ttnn.ROW_MAJOR_LAYOUT
            )  # [1, (K - 1) * C / 32, 32]
            ttnn.deallocate(hist)
            carry = ttnn.to_layout(ttnn.reshape(rows, (1, dn.K - 1, dn.qkv_dim_tp)), ttnn.TILE_LAYOUT)
            ttnn.deallocate(rows)
            ttnn.copy(carry, dn.conv_carry)
            ttnn.deallocate(carry)

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


def zero_kv_blocks(model, block_ids):
    """Zero blocks in every K and V buffer of model._paged_kv_caches (vLLM's new_block_ids_to_zero: a block that held
    fp32 GDN state reads as NaN / Inf in bf16 attention)."""
    kv_caches = getattr(model, "_paged_kv_caches", None)
    if not block_ids or not kv_caches:
        return
    mesh = model.mesh_device
    k0 = kv_caches[0][0]
    shape = [1] + [int(x) for x in list(k0.shape)[1:]]
    zeros = getattr(model, "_zero_kv_block", None)
    if zeros is None or list(zeros.shape) != shape or zeros.dtype != k0.dtype:
        zeros = model._zero_kv_block = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=k0.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
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

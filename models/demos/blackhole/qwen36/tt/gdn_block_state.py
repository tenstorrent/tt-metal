# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN state in vLLM-owned MambaSpec blocks (QWEN36_MAMBA_SPEC=1, mamba_cache_mode "none"; prototype).

vLLM overlays its KV cache groups: block id b names the same page in the attention layer of a layer slot
(``tensor_idx``) and in each GDN layer of that slot, and one group owns a block id at a time. A GDN layer keeps the
state of a request in the K and V buffers of its slot, at the block its group owns for that request:

* recurrent state: the first Nv * 16 fp32 pages of block b in a float32 alias_view of K, [N, pages / 16, Dk, Dv];
* packed conv history: the first Nv * 4 bf16 pages of block b in V.

Decode reads and writes them through gdn_decode_step(state_page_idx=...), one persistent [2, Bmax] index tensor per
GDN layer (device-row order). Prefill keeps writing the model's [Bmax, ...] decode buffers (now a staging row per
device row) and copies the row into the request's block afterwards. Needs whole fp32 pages per block (bf16 KV, or a
block size whose BF8 page is a whole number of fp32 pages per bank).
"""

from types import SimpleNamespace

import torch

import ttnn

FP32_PAGE_BYTES = 4096


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
        if k_bytes % FP32_PAGE_BYTES or (k_bytes // FP32_PAGE_BYTES) % 16:
            raise ValueError(f"K block of {k_bytes} B is not a whole number of 16-page fp32 heads; use bf16 KV")
        rec_heads = k_bytes // FP32_PAGE_BYTES // 16
        if rec_heads < dn.Nv:
            raise ValueError(f"K block holds {rec_heads} fp32 heads of state, layer needs {dn.Nv}; use a larger block")
        if v.dtype != ttnn.bfloat16 or tiles < dn.Nv * 4:
            raise ValueError(f"V must be bf16 with >= {dn.Nv * 4} tiles per block (got {v.dtype}, {tiles})")
        rec_view = ttnn.experimental.alias_view(k, [num_blocks, rec_heads, dn.Dk, dn.Dv], ttnn.float32)
        hist_view = ttnn.experimental.alias_view(v, [num_blocks, tiles, 32, 32], ttnn.bfloat16)
        idx = ttnn.from_torch(
            torch.zeros(2, self.bmax, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        # Decode addresses V itself as the conv-history pool: block b's tiles are pages [b * tiles, (b + 1) * tiles).
        return SimpleNamespace(
            rec_pool=rec_view,
            hist_pool=v,
            rec_view=rec_view,
            hist_view=hist_view,
            idx=idx,
            rec_pages=rec_heads * 16,
            hist_pages=tiles,
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
            host = torch.tensor(
                [[b * ps.rec_pages for b in blocks], [b * ps.hist_pages for b in blocks]], dtype=torch.int32
            )
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
            packed = ttnn.reshape(dn._conv_hist_packed, (dn._decode_B, nr, 32, 32))
            for d, blk in zip(rows, ids):
                assert blk > 0, f"layer {i}: prefill row {d} has the null block"
                # L1-sharded sources: a DRAM source makes slice_write re-seat the destination buffer.
                src = ttnn.slice(
                    dn.rec_state, (d, 0, 0, 0), (d + 1, nv, dk, dv), memory_config=dn._rec_row_shard_memcfg()
                )
                ttnn.experimental.slice_write(src, ps.rec_view, [blk, 0, 0, 0], [blk + 1, nv, dk, dv], [1, 1, 1, 1])
                ttnn.deallocate(src)
                src = ttnn.slice(packed, (d, 0, 0, 0), (d + 1, nr, 32, 32), memory_config=dn._packed_row_shard_memcfg())
                ttnn.experimental.slice_write(src, ps.hist_view, [blk, 0, 0, 0], [blk + 1, nr, 32, 32], [1, 1, 1, 1])
                ttnn.deallocate(src)


def zero_kv_blocks(model, block_ids):
    """Zero blocks in every K and V buffer of model._paged_kv_caches (vLLM's new_block_ids_to_zero: a block that held
    fp32 GDN state reads as NaN / Inf in bf16 attention)."""
    kv_caches = getattr(model, "_paged_kv_caches", None)
    if not block_ids or not kv_caches:
        return
    mesh = model.mesh_device
    k0 = kv_caches[0][0]
    shape = [1] + [int(x) for x in k0.shape[1:]]
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

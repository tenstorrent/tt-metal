# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN request state in vLLM-owned KV cache blocks (MambaSpec groups, mamba_cache_mode "none").

vLLM gives each request one state block per MambaSpec group for its whole life. A GDN layer of the group keeps the
request's recurrent state and convolution history in the K and V buffers of its layer slot, at that block
(kv_cache_spec.GdnBlockLayout). GdnBlockState binds every GDN layer to float32 and bfloat16 alias_view views of its
slot's buffers and to a persistent ``state_page_idx`` tensor, which names the pages of the block of each decode row.

* Prefill runs on the layer's batch-1 workspace (``rec_state``, ``conv_carry``) from zero state and then copies the
  end state into the request's block (write_prefill_state).
* Decode runs gdn_decode_step on the blocks directly. The adapter rewrites the page index whenever vLLM moves a row
  or gives a row another request (set_decode_blocks), so moving a row moves no state. The index tensors keep their
  addresses, so a captured decode trace reads the new index on its next replay.
* zero_blocks clears blocks that vLLM gives to the attention group: float32 state bytes read as bfloat16 can be NaN
  or Inf at positions that attention masks.

Persistent tensors are allocated in the constructor, before any trace capture: a buffer allocated after a capture and
alive at a replay can be overwritten by the replay.
"""

from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.kv_cache_spec import (
    HIST_SLOTS,
    MAX_DECODE_ROWS,
    GdnDims,
    gdn_state_shapes,
    layer_index,
    pack_history_rows,
    plan_gdn_block_layout,
)

# Tokens of zero K / V that one zero_blocks pass writes; blocks are cleared in groups of this many tokens.
_ZERO_TOKENS = 4096


def is_mamba_entry(entry):
    """True for a MambaSpec entry of allocate_kv_cache_per_layer: (shapes, dtypes, tensor_idx), shapes a tuple of
    shapes. An attention entry is (shape, dtype, tensor_idx)."""
    shapes = entry[0]
    return len(shapes) > 0 and isinstance(shapes[0], (tuple, list))


def _free_copy(view, source):
    """Free ``view`` when a reshape had to copy ``source`` instead of re-viewing it."""
    if not tpc.aliases(view, source):
        ttnn.deallocate(view)


def _dram_banks(mesh):
    grid = mesh.dram_grid_size()
    return int(grid.x * grid.y)


def _height_sharded(mesh, rows, width):
    """L1 height-sharded TILE config for a [rows, width] tensor on the most cores (at most 48) that divide its tile
    rows. slice_write writes in place into an interleaved TILE output only from a sharded input."""
    grid = mesh.compute_with_storage_grid_size()
    n_tiles = rows // ttnn.TILE_SIZE
    cores = max(c for c in range(1, min(48, grid.x * grid.y) + 1) if n_tiles % c == 0)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
            (rows // cores, width),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


class GdnBlockState:
    """The GDN layers of a Qwen36Model bound to the GDN state in vLLM KV cache blocks."""

    def __init__(self, model, per_layer_specs, kv_caches):
        args = model.args
        self.mesh = model.mesh_device
        self.kv_caches = kv_caches
        self.bmax = int(args.max_batch_size)
        if not 1 <= self.bmax <= MAX_DECODE_ROWS:
            raise ValueError(f"max_num_seqs={self.bmax}: gdn_decode_step decodes 1 to {MAX_DECODE_ROWS} rows per layer")
        attn = [(i, e) for i, e in enumerate(per_layer_specs) if not is_mamba_entry(e)]
        if [i for i, _ in attn] != list(model._attention_layer_indices):
            raise ValueError(
                f"allocate_kv_cache_per_layer names attention layers {[i for i, _ in attn]}, the model has "
                f"{list(model._attention_layer_indices)}"
            )
        kv_by_slot = {e[2]: kv_caches[n] for n, (_, e) in enumerate(attn)}
        if len(kv_by_slot) != len(attn):
            raise ValueError("two attention layers share a layer slot; each needs its own K and V buffers")
        kda = getattr(ttnn.experimental, "kda", None)
        if not (hasattr(ttnn.experimental, "alias_view") and hasattr(kda, "gdn_decode_step")):
            raise ValueError("GDN state in KV cache blocks needs ttnn.experimental.alias_view and kda.gdn_decode_step")
        k0 = kv_caches[0][0]
        self.num_blocks, kv_heads, block_size, head_dim = (int(x) for x in k0.shape)
        self.dims = GdnDims(
            nk=args.gdn_nk_tp,
            nv=args.gdn_nv_tp,
            dk=args.gdn_dk,
            dv=args.gdn_dv,
            kconv=args.gdn_conv_kernel_size,
        )
        self.layout = plan_gdn_block_layout(
            block_size=block_size,
            kv_heads=kv_heads,
            head_dim=head_dim,
            kv_bf8=k0.dtype == ttnn.bfloat8_b,
            dims=self.dims,
            banks=_dram_banks(self.mesh),
        )
        expected = gdn_state_shapes(self.dims)
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        null_index = self.layout.index_host([0] * self.bmax, self.num_blocks)
        self.layers = {}  # GDN layer index -> TPGatedDeltaNet
        views = {}  # layer slot -> (float32 view of K, float32 view of V or None, bfloat16 history view of V)
        for i, entry in enumerate(per_layer_specs):
            if not is_mamba_entry(entry):
                continue
            shapes, dtypes, slot = entry
            layer = model.layers[i]
            if layer.is_full_attention:
                raise ValueError(
                    f"layer {i} is an attention layer, but allocate_kv_cache_per_layer gives it a MambaSpec"
                )
            got = (tuple(tuple(int(d) for d in s) for s in shapes), tuple(dtypes))
            if got != expected:
                raise ValueError(
                    f"layer {i}: MambaSpec shapes/dtypes {got} do not match this model's GDN state {expected} at "
                    f"TP={args.num_devices}; MESH_DEVICE must name the mesh the model runs on"
                )
            if slot not in kv_by_slot:
                raise ValueError(f"layer {i}: layer slot {slot} has no attention layer whose buffers hold its state")
            dn = layer.attention
            if dn.rec_state is None or dn.rec_state.dtype != ttnn.float32:
                raise ValueError(
                    "GDN state in KV cache blocks needs the float32 recurrent state (QWEN35_GDN_STATE_BF16)"
                )
            if not (dn._fuse_ab and getattr(args, "proj_1d_decode", False) and "conv_taps_packed" in dn.tw):
                raise ValueError(
                    f"layer {i}: the GDN decode on KV cache blocks needs the fused [qkv | z | a | b] 1D decode "
                    "projection and the gdn_decode_step constants"
                )
            if slot not in views:
                views[slot] = self._views(*kv_by_slot[slot])
            rec_k, rec_v, hist = views[slot]
            dn.state_blocks = SimpleNamespace(
                rec_k=rec_k,
                rec_v=rec_v,
                hist=hist,
                idx=ttnn.from_torch(
                    null_index, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, mesh_mapper=rep
                ),
            )
            self.layers[i] = dn
        unbound = [i for i, layer in enumerate(model.layers) if not layer.is_full_attention and i not in self.layers]
        if unbound:
            raise ValueError(f"GDN layers {unbound} have no MambaSpec entry in allocate_kv_cache_per_layer")

        self._pack_table = ttnn.from_torch(
            pack_history_rows(self.dims).reshape(1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=rep,
        )
        hist_tiles = self.dims.nv * HIST_SLOTS
        self._hist_cfg = _height_sharded(self.mesh, hist_tiles * 32, 32)
        self._halves_cfgs = {}
        zero_blocks = max(1, _ZERO_TOKENS // block_size)
        self._zero = ttnn.from_torch(
            torch.zeros(1, kv_heads, zero_blocks * block_size, head_dim, dtype=torch.bfloat16),
            dtype=k0.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self._zero_table = ttnn.from_torch(
            torch.zeros(1, zero_blocks, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=rep,
        )
        self._zero_refs = []
        self._group_names = None
        self._group_layers = None
        self._decode_blocks = {}  # group -> blocks of the decode rows that the index tensors hold
        self._index_refs = {}
        self._warmed = False

    def _views(self, k, v):
        lay, n = self.layout, self.num_blocks
        check = lay.fp32_slices_are_blocks
        rec_k = ttnn.experimental.alias_view(k, lay.fp32_view_shape(n), ttnn.float32, check_slices=check)
        rec_v = None
        if lay.heads_v:
            rec_v = ttnn.experimental.alias_view(v, lay.fp32_view_shape(n), ttnn.float32, check_slices=check)
        hist = ttnn.experimental.alias_view(v, lay.hist_view_shape(n), ttnn.bfloat16)
        return rec_k, rec_v, hist

    def _halves_cfg(self, halves):
        if halves not in self._halves_cfgs:
            self._halves_cfgs[halves] = _height_sharded(self.mesh, halves * self.dims.dk // 2, self.dims.dv)
        return self._halves_cfgs[halves]

    def groups(self, mamba_state):
        """[(GDN layers of the group, windows)] per MambaSpec group of ``mamba_state``, in group order."""
        names = tuple(tuple(g.layer_names) for g in mamba_state.groups)
        if names != self._group_names:
            group_layers, seen = [], set()
            for group in names:
                layers = []
                for name in group:
                    i = layer_index(name)
                    if i not in self.layers:
                        raise ValueError(f"mamba_state names {name!r}, which is not a GDN layer of this model")
                    if i in seen:
                        raise ValueError(f"mamba_state names {name!r} in more than one group")
                    seen.add(i)
                    layers.append(self.layers[i])
                group_layers.append(layers)
            if seen != set(self.layers):
                raise ValueError(f"mamba_state has no group for GDN layers {sorted(set(self.layers) - seen)}")
            self._group_names, self._group_layers = names, group_layers
            self._decode_blocks.clear()
        return list(zip(self._group_layers, (g.windows for g in mamba_state.groups)))

    def set_decode_blocks(self, mamba_state):
        """Point decode row r of every GDN layer at the block of ``mamba_state`` row r of the layer's group. Rows past
        the windows, and padding rows, take block 0, vLLM's null block. Copies an index only when its blocks change:
        the decode step reads the index tensors and never writes them."""
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for g, (layers, windows) in enumerate(self.groups(mamba_state)):
            blocks = [int(b) for b in windows.reshape(-1)[: self.bmax].tolist()]
            blocks = tuple(blocks + [0] * (self.bmax - len(blocks)))
            if self._decode_blocks.get(g) == blocks:
                continue
            host = ttnn.from_torch(
                self.layout.index_host(blocks, self.num_blocks),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=rep,
            )
            for dn in layers:
                ttnn.copy_host_to_device_tensor(host, dn.state_blocks.idx)
            # The copies may read the host tensor after they return; hold it until this group's index changes again.
            self._index_refs[g] = host
            self._decode_blocks[g] = blocks

    def write_prefill_state(self, mamba_state, row):
        """Copy the end state of the request that prefill row ``row`` just prefilled (each GDN layer's batch-1
        workspace) into the request's block in every MambaSpec group."""
        for layers, windows in self.groups(mamba_state):
            block = int(windows[row, 0])
            if block <= 0:
                raise ValueError(f"prefill row {row} has the null block {block} as its GDN state block")
            for dn in layers:
                self._write_state(dn, block)

    def _write_state(self, dn, block):
        lay, nv, dk, dv = self.layout, self.dims.nv, self.dims.dk, self.dims.dv
        sb = dn.state_blocks
        # [1, Nv, Dk, Dv] as half-head slices; the tiles keep their order.
        rec = ttnn.reshape(dn.rec_state, (2 * nv, 1, dk // 2, dv))
        for pool, s0, lo, hi in lay.state_writes(block):
            src = ttnn.slice(rec, (lo, 0, 0, 0), (hi, 1, dk // 2, dv), memory_config=self._halves_cfg(hi - lo))
            view = sb.rec_k if pool == 0 else sb.rec_v
            ttnn.experimental.slice_write(src, view, [s0, 0, 0, 0], [s0 + hi - lo, 1, dk // 2, dv], [1, 1, 1, 1])
            ttnn.deallocate(src)
        _free_copy(rec, dn.rec_state)
        carry = ttnn.to_layout(dn.conv_carry, ttnn.ROW_MAJOR_LAYOUT)  # [1, K - 1, C]
        rows = ttnn.reshape(carry, ((self.dims.kconv - 1) * self.dims.conv_dim // 32, 32))
        packed = ttnn.embedding(self._pack_table, rows, layout=ttnn.TILE_LAYOUT, memory_config=self._hist_cfg)
        _free_copy(rows, carry)
        ttnn.deallocate(carry)
        tiles = ttnn.reshape(packed, (1, nv * HIST_SLOTS, 32, 32))
        t0 = lay.hist_first
        ttnn.experimental.slice_write(
            tiles, sb.hist, [block, t0, 0, 0], [block + 1, t0 + nv * HIST_SLOTS, 32, 32], [1, 1, 1, 1]
        )
        _free_copy(tiles, packed)
        ttnn.deallocate(packed)

    def zero_blocks(self, block_ids):
        """Zero blocks ``block_ids`` in every K and V buffer. A partial group of blocks is padded with block 0."""
        ids = [int(b) for b in block_ids]
        if not ids:
            return
        width = int(self._zero_table.shape[-1])
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        refs = []
        for start in range(0, len(ids), width):
            chunk = ids[start : start + width]
            host = ttnn.from_torch(
                torch.tensor([chunk + [0] * (width - len(chunk))], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(host, self._zero_table)
            refs.append(host)
            for k, v in self.kv_caches:
                ttnn.experimental.paged_fill_cache(k, self._zero, self._zero_table, batch_idx=0)
                ttnn.experimental.paged_fill_cache(v, self._zero, self._zero_table, batch_idx=0)
        self._zero_refs = refs

    def release(self):
        """Unbind the GDN layers from the blocks, before the K and V buffers are freed."""
        for dn in self.layers.values():
            dn.state_blocks = None
        self.layers = {}

    def warmup(self):
        """Compile the block write and zero programs on the null block 0, before any trace is captured."""
        if self._warmed:
            return
        for dn in self.layers.values():
            self._write_state(dn, 0)
        self.zero_blocks([0])
        ttnn.synchronize_device(self.mesh)
        self._warmed = True

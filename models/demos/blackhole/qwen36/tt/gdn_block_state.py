# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN request state in vLLM-owned KV cache blocks (MambaSpec groups, mamba_cache_mode "none" and "align").

A GDN layer of a MambaSpec group keeps a request's recurrent state and convolution history in the K and V buffers of
its layer slot, at the block that the group gives the request (kv_cache_spec.GdnBlockLayout). In mode "none" a request
keeps one state block per group for its whole life. In mode "align" vLLM's ``mamba_state`` names, per row and group,
the block that holds the start state (``start_blocks``) and the block that takes the end state (``windows``), and
vLLM keeps some blocks in its prefix cache, so a start block can be shared by several requests and must stay
unchanged. GdnBlockState binds every GDN layer to float32 and bfloat16 alias_view views of its slot's buffers and to a
persistent ``state_page_idx`` tensor, which names the pages of the block of each decode row.

* Prefill runs on the layer's batch-1 workspace (``rec_state``, ``conv_carry``). A prefill at start_pos 0 starts from
  zero state; a later one first loads its start block into the workspace (restore). The end state goes into the
  destination block. A prefill checkpoint inside the prefill's last chunk is computed by that chunk's forward next to
  the workspace (TPGatedDeltaNet.prefill_checkpoint) and written into its block by write_checkpoint; a checkpoint on
  a chunk boundary, or in an earlier chunk, is the workspace written into its block (write_state) between two parts
  of the prefill (prefill_segments). The start block is only read. Writes use paged_fill_cache: the views have one
  page or one half value head per dim-0 slice, so a persistent page table names the destination; the block id never
  reaches a program hash, and no request compiles a program.
* Decode runs gdn_decode_step on the blocks directly. The adapter rewrites the page index whenever vLLM moves a row
  or gives a row another request (set_decode_blocks), so moving a row moves no state. The index tensors keep their
  addresses, so a captured decode trace reads the new index on its next replay. A row whose start block differs from
  its destination first gets the start state copied into the destination, which the step then updates in place.
* Reads and block-to-block copies address a block through ttnn.narrow views of the alias views. narrow only moves the
  buffer address, so the programs that warmup compiles on the null block serve every block, also after a trace is
  captured. Each view starts at its own block's first whole float32 page (GdnBlockLayout.first_fp32_page), so a state
  moves correctly between blocks whose float32 pages start at different offsets: with bfloat8_b K and V a 1024-token
  block holds 8.5 float32 pages per bank, so the pages of an odd block start half a page later than those of an even
  one.
* zero_blocks clears blocks that vLLM gives to the attention group: float32 state bytes read as bfloat16 can be NaN
  or Inf at positions that attention masks.
* A speculative verify (Qwen36MTPContractForCausalLM) runs gdn_spec_step with conv_hist on the blocks: row r reads its
  full state from start_blocks[r] and writes the full state after query token t, history included, into windows[r, t]
  (kv_cache_spec.GdnBlockLayout.verify_tables). So every block holds a complete state after every step, and no model
  state outlives a step. set_verify_blocks writes the step's tables into one ctrl and one index tensor per MambaSpec
  group, which every GDN layer of the group reads.

Persistent tensors are allocated in the constructor, before any trace capture: a buffer allocated after a capture and
alive at a replay can be overwritten by the replay.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.kv_cache_spec import (
    HALF_HEAD_PAGES,
    HIST_SLOTS,
    MAX_DECODE_ROWS,
    GdnDims,
    gdn_state_shapes,
    layer_index,
    pack_history_rows,
    plan_gdn_block_layout,
    unpack_history_rows,
)

# Tokens of zero K / V that one zero_blocks pass writes; blocks are cleared in groups of this many tokens.
_ZERO_TOKENS = 4096


def is_mamba_entry(entry):
    """True for a MambaSpec entry of allocate_kv_cache_per_layer: (shapes, dtypes, tensor_idx), shapes a tuple of
    shapes. An attention entry is (shape, dtype, tensor_idx)."""
    shapes = entry[0]
    return len(shapes) > 0 and isinstance(shapes[0], (tuple, list))


@dataclass(frozen=True)
class RowBlocks:
    """The state blocks of one row of ``mamba_state``, one entry per MambaSpec group."""

    start: tuple  # block holding the full state after start_pos tokens; 0 at start_pos 0
    dest: tuple  # block that takes the end state
    checkpoint: tuple | None  # block that takes the state after checkpoint_pos tokens, or None
    checkpoint_pos: int  # token count of the prefill checkpoint, 0 for none


def row_blocks(mamba_state, row):
    """RowBlocks of row ``row`` of ``mamba_state`` (vllm_tt_plugin.mamba_state.MambaStepState)."""
    groups = mamba_state.groups
    if any(getattr(g, "start_blocks", None) is None for g in groups):
        raise ValueError(
            "mamba_state has no start_blocks; Qwen36ForCausalLM needs a vllm-tt-plugin whose MambaGroupState names "
            "the block of each row's start state"
        )
    pos = getattr(mamba_state, "checkpoint_positions", None)
    p = int(pos[row, 0]) if pos is not None else 0
    checkpoint = None
    if p > 0:
        checkpoint = tuple(int(g.checkpoint_blocks[row, 0]) for g in groups)
        if min(checkpoint) <= 0:
            raise ValueError(f"row {row} writes a prefill checkpoint after {p} tokens into the null block 0")
    return RowBlocks(
        start=tuple(int(g.start_blocks[row]) for g in groups),
        dest=tuple(int(g.windows[row, 0]) for g in groups),
        checkpoint=checkpoint,
        checkpoint_pos=p,
    )


def prefill_segments(start, end, checkpoint_pos, block_size, chunk_size):
    """[(a, b, c), ...]: the token ranges [a, b), in order, that a prefill of tokens [start, end) runs one after the
    other on the GDN workspace with Qwen36Model.prefill_traced_chunked (start_pos a, checkpoint_pos c), so that the
    state after ``checkpoint_pos`` tokens (0: none) becomes available. Each range starts on a KV block boundary.

    The prefill replays its full ``chunk_size``-token chunks from the chunk trace and runs its partial last chunk as
    one masked-bucket forward. A checkpoint inside the last chunk is computed by that chunk's forward (one range, c =
    checkpoint_pos), so the prefill runs no extra forward; a full last chunk then runs as a masked-bucket forward
    instead of a chunk-trace replay. Otherwise the
    prefill stops at the checkpoint (two ranges, c = 0), where the workspace holds its state; on a chunk boundary that
    also adds no forward. vLLM puts the checkpoint on the last block boundary before ``end``, which is one of these
    two cases when ``chunk_size`` is a multiple of ``block_size``."""
    if not 0 <= start < end:
        raise ValueError(f"prefill of tokens [{start}, {end}) is empty")
    if start % block_size:
        raise ValueError(f"prefill resumes at {start}, inside a {block_size}-token KV block")
    if not checkpoint_pos:
        return [(start, end, 0)]
    if not start < checkpoint_pos < end or checkpoint_pos % block_size:
        raise ValueError(
            f"prefill checkpoint after {checkpoint_pos} tokens is not on a {block_size}-token block boundary inside "
            f"[{start}, {end})"
        )
    last_chunk = start + (end - start - 1) // chunk_size * chunk_size
    if (checkpoint_pos - start) % chunk_size and checkpoint_pos > last_chunk:
        return [(start, end, checkpoint_pos)]
    return [(start, checkpoint_pos, 0), (checkpoint_pos, end, 0)]


def _free_copy(view, source):
    """Free ``view`` when a reshape had to copy ``source`` instead of re-viewing it."""
    if not tpc.aliases(view, source):
        ttnn.deallocate(view)


def _dram_banks(mesh):
    grid = mesh.dram_grid_size()
    return int(grid.x * grid.y)


class GdnBlockState:
    """The GDN layers of a Qwen36Model bound to the GDN state in vLLM KV cache blocks. ``verify_tokens``: T = 1 + K of a
    speculative model's verify (0: no verify)."""

    def __init__(self, model, per_layer_specs, kv_caches, verify_tokens=0):
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
        lay = self.layout
        block_bytes = 2 * lay.tiles * lay.kv_tile_bytes
        state_bytes = self.dims.nv * (self.dims.head_pages * 4096 + HIST_SLOTS * 2048)
        logger.info(
            f"GDN state in vLLM KV cache blocks: {self.num_blocks} blocks of {block_size} tokens, {block_bytes} B of K and "
            f"V per block and layer slot, {len(kv_caches)} layer slots ({block_bytes * len(kv_caches) * self.num_blocks / 2**30:.2f} "
            f"GiB per device); one GDN state uses {state_bytes} B of a block ({lay.heads_k} value heads in K, "
            f"{lay.heads_v} in V); {len(self.layers)} GDN layers"
        )

        self._pack_table = ttnn.from_torch(
            pack_history_rows(self.dims).reshape(1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=rep,
        )
        self._write_tables = [
            ttnn.from_torch(
                torch.zeros(1, len(t), dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep,
            )
            for t in self.layout.write_tables(0)
        ]
        self._write_refs = []
        self._unpack_table = ttnn.from_torch(
            unpack_history_rows(self.dims).reshape(1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=rep,
        )
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
        self.verify_tokens = int(verify_tokens)
        if self.verify_tokens:
            self._bind_verify(per_layer_specs)

    def _bind_verify(self, per_layer_specs):
        """Persistent gdn_spec_step inputs of the verify, before any trace capture: one ctrl page and one page index
        per MambaSpec group. vLLM's MambaSpec group k holds the k-th GDN layer of every layer slot (its groups hold one
        layer per slot, filled in layer order); set_verify_blocks checks that against mamba_state."""
        T, rows = self.verify_tokens, self.bmax
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        seen = {}
        self._verify_group = {}  # GDN layer index -> group whose tables it reads
        for i, entry in enumerate(per_layer_specs):
            if is_mamba_entry(entry):
                self._verify_group[i] = seen.get(entry[2], 0)
                seen[entry[2]] = self._verify_group[i] + 1
        ctrl0, idx0 = self.layout.verify_tables([], torch.zeros(0, T), self.num_blocks, rows)

        def table(host):
            return ttnn.from_torch(
                host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, mesh_mapper=rep
            )

        groups = max(self._verify_group.values()) + 1
        self._verify_ctrl = [table(ctrl0) for _ in range(groups)]
        self._verify_idx = [table(idx0) for _ in range(groups)]
        for i, dn in self.layers.items():
            k = self._verify_group[i]
            dn.state_blocks.verify = SimpleNamespace(T=T, ctrl=self._verify_ctrl[k], idx=self._verify_idx[k])
        self._verify_refs = []
        self._verify_staged = {}  # ("ctrl" | "idx", table group) -> the host table its device table holds
        self._verify_order = None  # (mamba_state group names, [table group of each mamba_state group])

    def set_verify_blocks(self, mamba_state):
        """Point the verify of every GDN layer at the blocks of ``mamba_state``: row r of group g reads its start state
        from start_blocks[r] and writes the state after query token t into windows[r, t] (verify_tables). Rows past the
        windows, and padding rows (block 0), use vLLM's null block. The tables are rewritten before every verify; the
        caller's previous verify has finished reading them (verify_traced synchronizes)."""
        T = self.verify_tokens
        groups = self.groups(mamba_state)
        if self._verify_order is None or self._verify_order[0] != self._group_names:
            order = []
            for g, (layers, _) in enumerate(groups):
                ks = {self._verify_group[layer_index(name)] for name in mamba_state.groups[g].layer_names}
                members = {i for i, k in self._verify_group.items() if k in ks}
                if len(ks) != 1 or members != {layer_index(name) for name in mamba_state.groups[g].layer_names}:
                    raise ValueError(
                        f"MambaSpec group {g} holds layers {sorted(members)}, not the k-th GDN layer of every layer "
                        "slot; the verify tables are bound per such group at allocation"
                    )
                order.append(ks.pop())
            self._verify_order = (self._group_names, order)
        tables = []
        for g, (_, windows) in enumerate(groups):
            if windows.shape[1] < T:
                raise ValueError(
                    f"a verify of {T} tokens per row needs {T} state blocks per row, got {windows.shape[1]}"
                )
            starts = getattr(mamba_state.groups[g], "start_blocks", None)
            if starts is None:
                raise ValueError("a verify needs mamba_state start_blocks")
            n = min(self.bmax, int(windows.shape[0]))
            ctrl, idx = self.layout.verify_tables(starts[:n], windows[:n, :T], self.num_blocks, self.bmax)
            k = self._verify_order[1][g]
            tables += [(("ctrl", k), ctrl, self._verify_ctrl[k]), (("idx", k), idx, self._verify_idx[k])]
        self._stage_verify_tables(tables)

    def _stage_verify_tables(self, tables):
        """Copy each (key, host table, device table) whose host table differs from the one staged last under that key: a
        window index table stays the same while its rows keep their blocks. Only this method writes the device tables.
        """
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        refs = []
        for key, host, dev in tables:
            if key in self._verify_staged and torch.equal(self._verify_staged[key], host):
                continue
            h = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
            ttnn.copy_host_to_device_tensor(h, dev)
            self._verify_staged[key] = host.clone()
            refs.append(h)
        # The copies may read the host tensors after they return; hold them until the next verify's tables.
        self._verify_refs = refs

    def clear_verify_blocks(self):
        """Point every verify row at the null block 0 (warmup and trace capture)."""
        ctrl, idx = self.layout.verify_tables([], torch.zeros(0, self.verify_tokens), self.num_blocks, self.bmax)
        self._verify_staged.clear()
        tables = []
        for k, (dev_ctrl, dev_idx) in enumerate(zip(self._verify_ctrl, self._verify_idx)):
            tables += [(("ctrl", k), ctrl, dev_ctrl), (("idx", k), idx, dev_idx)]
        self._stage_verify_tables(tables)

    def _views(self, k, v):
        lay, n = self.layout, self.num_blocks
        check = lay.fp32_slices_are_blocks
        rec_k = ttnn.experimental.alias_view(k, lay.fp32_view_shape(n), ttnn.float32, check_slices=check)
        rec_v = None
        if lay.heads_v:
            rec_v = ttnn.experimental.alias_view(v, lay.fp32_view_shape(n), ttnn.float32, check_slices=check)
        hist = ttnn.experimental.alias_view(v, lay.hist_view_shape(n), ttnn.bfloat16)
        return rec_k, rec_v, hist

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
        the decode step reads the index tensors and never writes them.

        A row whose start block differs from its destination block (mode "align": this step's token crosses into a
        new block-table column) first gets the full state of the start block copied into the destination block;
        gdn_decode_step then updates the destination in place and the start block stays unchanged."""
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for g, (layers, windows) in enumerate(self.groups(mamba_state)):
            starts = getattr(mamba_state.groups[g], "start_blocks", None)
            if starts is not None:
                for r in range(min(self.bmax, windows.shape[0])):
                    src, dst = int(starts[r]), int(windows[r, 0])
                    if src > 0 and dst > 0 and src != dst:
                        for dn in layers:
                            self._copy_state(dn, src, dst)
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
        self.write_state(mamba_state, [int(windows[row, 0]) for _, windows in self.groups(mamba_state)])

    def write_state(self, mamba_state, blocks):
        """Copy the batch-1 workspace of every GDN layer of MambaSpec group g of ``mamba_state`` into block
        ``blocks[g]``: a prefill's end state or one of its checkpoints."""
        self._write_groups(mamba_state, blocks, checkpoint=False)

    def write_checkpoint(self, mamba_state, blocks):
        """Copy the state that every GDN layer of MambaSpec group g of ``mamba_state`` computed at the prefill
        checkpoint of its last prefill forward (TPGatedDeltaNet.take_prefill_checkpoint) into block ``blocks[g]``,
        and release it (TPGatedDeltaNet.release_prefill_checkpoint)."""
        self._write_groups(mamba_state, blocks, checkpoint=True)

    def _write_groups(self, mamba_state, blocks, checkpoint):
        # paged_fill_cache may read the host page tables after it returns; hold them until the device is idle.
        if self._write_refs:
            ttnn.synchronize_device(self.mesh)
            self._write_refs = []
        for (layers, _), block in zip(self.groups(mamba_state), blocks, strict=True):
            if block <= 0:
                raise ValueError(f"a prefill row has the null block {block} as its GDN state block")
            self._set_write_tables(block)
            for dn in layers:
                if checkpoint:
                    self._write_checkpoint(dn)
                else:
                    self._write_state(dn)

    def restore(self, mamba_state, blocks):
        """Load the full state that block ``blocks[g]`` holds into the batch-1 workspace of every GDN layer of
        MambaSpec group g of ``mamba_state``, so that a prefill continues from it. The blocks are only read."""
        for (layers, _), block in zip(self.groups(mamba_state), blocks, strict=True):
            if block <= 0:
                raise ValueError(f"a prefill resumes from the null block {block}")
            for dn in layers:
                self._read_state(dn, block)

    def _block_views(self, views, block):
        """Views of the GDN state of ``block`` in the alias views (rec_k, rec_v, hist) of one layer slot: float32
        [2 * heads_k, 1, Dk / 2, Dv] of K, the same of V (None when K holds every head), and the packed history
        [4 * Nv, 1, 32, 32] bfloat16. Each view starts on bank 0, as ttnn.narrow requires: with 8 DRAM banks a half-head
        slice is one page row, and the history of a block starts on a whole page row of the bfloat16 view."""
        rec_k, rec_v, hist = views
        lay = self.layout
        s0 = lay.first_fp32_page(block) // HALF_HEAD_PAGES
        k = ttnn.narrow(rec_k, 0, s0, 2 * lay.heads_k)
        v = ttnn.narrow(rec_v, 0, s0, 2 * lay.heads_v) if lay.heads_v else None
        h = ttnn.narrow(hist, 0, lay.hist_page(block, 0), HIST_SLOTS * self.dims.nv)
        return k, v, h

    def _layer_views(self, dn):
        sb = dn.state_blocks
        return sb.rec_k, sb.rec_v, sb.hist

    def _read_state(self, dn, block):
        lay, nv, dk, dv = self.layout, self.dims.nv, self.dims.dk, self.dims.dv
        k, v, hist = self._block_views(self._layer_views(dn), block)
        # Half-head slices of the workspace: the same tile order as [1, Nv, Dk, Dv].
        rec = ttnn.experimental.alias_view(dn.rec_state, [2 * nv, 1, dk // 2, dv], ttnn.float32)
        ttnn.copy(k, ttnn.narrow(rec, 0, 0, 2 * lay.heads_k))
        if v is not None:
            ttnn.copy(v, ttnn.narrow(rec, 0, 2 * lay.heads_k, 2 * lay.heads_v))
        packed = ttnn.to_layout(hist, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rows = ttnn.reshape(packed, (nv * HIST_SLOTS * 32, 32))
        carry = ttnn.embedding(
            self._unpack_table, rows, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, (K - 1) * C / 32, 32]
        _free_copy(rows, packed)
        ttnn.deallocate(packed)
        carry_rm = ttnn.reshape(carry, (1, self.dims.kconv - 1, self.dims.conv_dim))
        tiled = ttnn.to_layout(carry_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _free_copy(carry_rm, carry)
        ttnn.deallocate(carry)
        ttnn.copy(tiled, dn.conv_carry)
        ttnn.deallocate(tiled)

    def _copy_state(self, dn, src, dst):
        """Copy the full GDN state of layer dn from block ``src`` into block ``dst``, each at its own offsets."""
        views = self._layer_views(dn)
        for a, b in zip(self._block_views(views, src), self._block_views(views, dst)):
            if a is not None:
                ttnn.copy(a, b)

    def _set_write_tables(self, block):
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for table, ids in zip(self._write_tables, self.layout.write_tables(block)):
            host = ttnn.from_torch(
                torch.tensor([ids], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
            )
            ttnn.copy_host_to_device_tensor(host, table)
            self._write_refs.append(host)

    def _write_checkpoint(self, dn):
        rec_state, conv_carry = dn.take_prefill_checkpoint()
        self._write_state(dn, rec_state, conv_carry)
        dn.release_prefill_checkpoint(rec_state, conv_carry)

    def _write_state(self, dn, rec_state=None, conv_carry=None):
        """Write a GDN state of ``dn`` into the block that the write tables name: ``rec_state`` [1, Nv, Dk, Dv] float32
        and ``conv_carry`` [1, K - 1, C] bfloat16, by default its batch-1 workspace."""
        lay, nv, dk, dv = self.layout, self.dims.nv, self.dims.dk, self.dims.dv
        sb = dn.state_blocks
        rec_state = dn.rec_state if rec_state is None else rec_state
        conv_carry = dn.conv_carry if conv_carry is None else conv_carry
        # [1, Nv, Dk, Dv] as rows of half heads; the tiles keep their order.
        rec = ttnn.reshape(rec_state, (1, 1, nv * dk, dv))
        for (pool, _, lo, hi), table in zip(lay.state_writes(0), self._write_tables):
            whole = lo == 0 and hi == 2 * nv
            src = rec if whole else ttnn.slice(rec, (0, 0, lo * dk // 2, 0), (1, 1, hi * dk // 2, dv))
            ttnn.experimental.paged_fill_cache(sb.rec_k if pool == 0 else sb.rec_v, src, table, batch_idx=0)
            if not whole:
                ttnn.deallocate(src)
        _free_copy(rec, rec_state)
        carry = ttnn.to_layout(conv_carry, ttnn.ROW_MAJOR_LAYOUT)  # [1, K - 1, C]
        rows = ttnn.reshape(carry, ((self.dims.kconv - 1) * self.dims.conv_dim // 32, 32))
        packed = ttnn.embedding(
            self._pack_table, rows, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, Nv * 4 * 32, 32]
        _free_copy(rows, carry)
        ttnn.deallocate(carry)
        tiles = ttnn.reshape(packed, (1, 1, nv * HIST_SLOTS * 32, 32))
        ttnn.experimental.paged_fill_cache(sb.hist, tiles, self._write_tables[-1], batch_idx=0)
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
        """Compile the block write, read, copy and zero programs on the null block 0, before any trace is captured,
        and the write of a prefill checkpoint when the GDN layers hold one (Qwen36Model._warmup_prefill_checkpoints).
        The reads leave the null block's bytes in the batch-1 workspace, which every prefill zeroes or restores."""
        if self._warmed:
            return
        self._set_write_tables(0)
        for dn in self.layers.values():
            self._write_state(dn)
            if dn.prefill_checkpoint is not None:
                self._write_checkpoint(dn)
            self._read_state(dn, 0)
            self._copy_state(dn, 0, 0)
        self.zero_blocks([0])
        ttnn.synchronize_device(self.mesh)
        self._warmed = True

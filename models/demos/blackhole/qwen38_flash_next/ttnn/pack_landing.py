# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Landing of a multi-user pack's users directly into their decode lanes (P3d-3), device ops only.

After a pack replay (``Qwen38TTNNTextModel.run_pack``) the landing sources of every user sit in persistent buffers
(``ttnn/pack.py``: the GDN final states and conv rows, the dense KV / compressed caches, the kept ``[v|k]`` and raw-key
rows, the PLE snapshots).  :class:`Qwen38TTNNPackLander` writes lane ``l`` of the lane state from the sources of user ``u``
so the lane holds what a single-path prefill of the same prompt (chunk prefill, hand-off, the last token as a forced step,
the importer) leaves: the layout of every family is documented in ``pack.py`` ("landing (P3d-3) host images").

Eager, outside any trace, one program per op shape: the user, the length, the bucket's lane and the pack row bases only
appear in host-written select tensors (persistent buffers written before each landing, exact 0/1 matmul operands) and in
runtime arguments (``fill_cache`` batch index, ``kv_page_scatter`` lane / source row), never in a shape.  The GDN ring
slots of every landed lane are written in ONE pass per pack (a selection write over all lanes).  Build it in the chain's
warm hook (before any capture: the select buffers are persistent) and warm it with :meth:`warm`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch
from ttnn.tools.trace_allocation_tracker import corruptible_allocation_scope

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import pack as pack_module
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import replicate_tensor_2d_mesh_mapper
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.kv_page_scatter import kv_page_scatter
from models.demos.blackhole.qwen38_flash_next.ttnn.lanes import _deallocate, _landed

PLE_WINDOW_ROWS = pack_module.PLE_HISTORY_ROWS + pack_module.PLE_BLOCK_ROWS
COMPRESSED_HEAD_ROWS = pack_module.PACK_ROWS // pack_module.BLOCK_ROWS


class Qwen38TTNNPackLander:
    """Lands pack users into lanes.  ``pager`` is the LANE pager (its same-buffer views of the lane tensors and compute
    config are reused), ``lane_state`` the model's lane state (page table, QSA KV pool), ``chain_state`` the generic state
    the pack ran on (the dense KV and compressed caches)."""

    def __init__(self, model, pack_state, chain_state, lane_state, pager) -> None:
        self.model = model
        self.mesh = mesh = model.mesh_device
        self.pack_state = pack_state
        self.chain_state = chain_state
        self.lane_state = lane_state
        self.pager = pager
        self.layout = layout = pager.layout
        self.lanes = lanes = layout.lanes
        self.compute_config = pager.compute_config
        paging = lane_state.paging
        if paging is None:
            raise ValueError("pack landing scatters into a paged lane state")
        self.page_size = paging.page_size
        self.gdn_layers = [i for i, layer in enumerate(pack_state.layers) if layer is not None]
        self.qsa_layers = [i for i, layer in enumerate(pack_state.layers) if layer is None]
        if len(self.gdn_layers) != len(layout.recurrent) or len(self.qsa_layers) != len(layout.compressed):
            raise ValueError("the pack state and the lane layout disagree on the layer counts")
        if len(layout.conv) != pack_module.CONV_RING_ROWS * len(self.gdn_layers):
            raise ValueError("the lane layout holds a different number of ring slots than four per GDN layer")
        self.owned: list[Any] = []
        self.timing: dict[str, list[float]] = {}
        replicated = replicate_tensor_2d_mesh_mapper(mesh)
        self._replicated = replicated

        def persistent(host: torch.Tensor):
            tensor = ttnn.from_torch(
                host.reshape(1, 1, *host.shape).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicated,
            )
            self.owned.append(tensor)
            return tensor

        try:
            self.conv_hits = tuple(
                persistent(pack_module.conv_hit_image(lanes, [], slot)) for slot in range(pack_module.CONV_RING_ROWS)
            )
            self.conv_keep = persistent(pack_module.conv_keep_image(lanes, []))
            self.compressed_select = {
                rows: persistent(torch.zeros(bucket.blocks_per_user, COMPRESSED_HEAD_ROWS))
                for rows, bucket in pack_state.buckets.items()
            }
            # the lane's compressed rows past the user's segment: written as zero with the landed rows, so a landed lane
            # holds no row of its previous occupant (the importer writes the whole generic cache; the rows are dead
            # either way: a decode reads only the complete blocks of its position)
            lane_rows = int(layout.compressed[0].shape[2])
            self.compressed_zero_tail = {
                rows: persistent(torch.zeros(lane_rows - bucket.blocks_per_user, qsa_module.INDEX_HEAD_DIM))
                for rows, bucket in pack_state.buckets.items()
            }
            self.staging_select = persistent(torch.zeros(pack_module.CHUNK, pack_module.PACK_ROWS))
            self.raw_select = persistent(torch.zeros(pack_module.CHUNK, pack_module.PACK_ROWS))
            self.ple_select = persistent(torch.zeros(PLE_WINDOW_ROWS, pack_module.CHUNK))
            # the pack's recurrent state buffers as [1, 1, 96 * 128, 128] rows (same pages), per GDN layer and launch
            rows = pack_state.layers[self.gdn_layers[0]].states[0].shape[0] * gdn_module.HEAD_DIM
            self.state_views = {
                index: tuple(
                    ttnn.experimental.view(state, (1, 1, rows, gdn_module.HEAD_DIM))
                    for state in pack_state.layers[index].states
                )
                for index in self.gdn_layers
            }
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        _deallocate(*self.owned)
        self.owned = []

    # -- host writes of the persistent selects ---------------------------------------------------------------------

    def _write(self, device_tensor, host: torch.Tensor) -> None:
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                host.reshape(1, 1, *host.shape).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=self._replicated,
            ),
            device_tensor,
        )

    # -- the landing -------------------------------------------------------------------------------------------------

    def land(self, assignments: Sequence[tuple[int, int, int]], rows_per_user: int) -> dict[str, float]:
        """Lane ``lane`` <- pack user ``user`` holding ``length`` tokens, for every ``(lane, user, length)`` of
        ``assignments`` (distinct lanes, distinct users; the pack's last replay must hold these users).  The page-table rows
        of the lanes must be written (``write_lane_page_table``) and the position rows / contexts are the caller's.  Queues
        everything on the command queue without a final synchronize; returns the host wall of each part in ms."""

        bucket = self.pack_state.buckets[rows_per_user]
        lanes_seen, users_seen = set(), set()
        for lane, user, length in assignments:
            if not 0 <= lane < self.lanes or not 0 <= user < bucket.users or not 1 <= length <= rows_per_user:
                raise ValueError(f"bad landing (lane {lane}, user {user}, length {length}) for bucket {rows_per_user}")
            if lane in lanes_seen or user in users_seen:
                raise ValueError("a landing writes each lane and takes each user once")
            lanes_seen.add(lane)
            users_seen.add(user)
        clock = time.perf_counter
        marks = {"conv_ms": 0.0, "user_ms": 0.0}
        with corruptible_allocation_scope(
            self.mesh
        ):  # transients are consumed before the next replay (as the importer)
            started = clock()
            self._land_conv([(lane, user) for lane, user, _ in assignments])
            marks["conv_ms"] = (clock() - started) * 1e3
            for lane, user, length in assignments:
                started = clock()
                self._land_user(lane, user, length, bucket)
                marks["user_ms"] += (clock() - started) * 1e3
        for key, value in marks.items():
            self.timing.setdefault(key, []).append(value)
        return marks

    def _land_conv(self, pairs: Sequence[tuple[int, int]]) -> None:
        """Every ring slot of every GDN layer: ``slot <- slot * keep + hit_j @ conv_rows`` for all landed lanes at once."""

        for slot_index, hit in enumerate(self.conv_hits):
            self._write(hit, pack_module.conv_hit_image(self.lanes, pairs, slot_index))
        self._write(self.conv_keep, pack_module.conv_keep_image(self.lanes, pairs))
        for index, slot in enumerate(self.layout.conv):
            layer, ring = divmod(index, pack_module.CONV_RING_ROWS)
            conv_rows = self.pack_state.layers[self.gdn_layers[layer]].conv_rows
            placed = ttnn.matmul(
                self.conv_hits[ring],
                conv_rows,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_config,
            )
            # A select (not slot * keep) keeps NaN/Inf in the admitted lane's old rows out of its new state; bit-identical for finite values.
            kept = ttnn.where(self.conv_keep, slot, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _landed(ttnn.add(kept, placed, output_tensor=slot), slot, f"pack landing conv[{index}]")
            _deallocate(placed, kept)

    def _land_user(self, lane: int, user: int, length: int, bucket: pack_module.PackBucket) -> None:
        rows = bucket.rows_per_user
        pager = self.pager
        # host-written selects of this user (one persistent buffer each: the next landing rewrites them in order)
        self._write(self.compressed_select[rows], pack_module.compressed_select_image(bucket, user, length))
        self._write(self.staging_select, pack_module.staging_select_image(bucket, user, length))
        self._write(self.raw_select, pack_module.raw_ring_select_image(bucket, user, length))
        self._write(self.ple_select, pack_module.ple_history_select_image(length))
        launch, head0 = pack_module.gdn_state_slot(bucket, user)
        row0 = head0 * gdn_module.HEAD_DIM
        row_count = pack_module.VALUE_HEADS * gdn_module.HEAD_DIM
        # GDN recurrent state: the user's 12 heads of its launch's final-state buffer
        for index, layer_index in enumerate(self.gdn_layers):
            source = ttnn.slice(
                self.state_views[layer_index][launch],
                (0, 0, row0, 0),
                (1, 1, row0 + row_count, gdn_module.HEAD_DIM),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            view = pager.views["recurrent"][index]
            _landed(ttnn.fill_cache(view, source, batch_idx=lane), view, f"pack landing recurrent[{index}]")
            _deallocate(source)
        # QSA: KV pages, compressed rows, staging tile, raw-key ring
        for index, layer_index in enumerate(self.qsa_layers):
            generic = self.chain_state.layers[layer_index].attention
            kept = self.pack_state.chunk_state.layers[layer_index].attention
            lane_attention = self.lane_state.layers[layer_index].attention
            kv_page_scatter(
                generic.packed_kv_cache,
                lane_attention.packed_kv_cache,
                self.lane_state.page_table,
                lane=lane,
                rows=length,
                page_size=self.page_size,
                src_row0=user * rows,
            )
            head = ttnn.slice(
                generic.compressed_index_cache,
                (0, 0, 0, 0),
                (1, 1, COMPRESSED_HEAD_ROWS, qsa_module.INDEX_HEAD_DIM),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            compressed = ttnn.matmul(
                self.compressed_select[rows],
                head,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_config,
            )
            whole = ttnn.concat(
                [compressed, self.compressed_zero_tail[rows]], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            view = pager.views["compressed"][index]
            _landed(ttnn.fill_cache(view, whole, batch_idx=lane), view, f"pack landing compressed[{index}]")
            _deallocate(head, compressed, whole)
            for name, select, source in (
                ("staging", self.staging_select, kept.kept_kv),
                ("ring", self.raw_select, kept.kept_raw),
            ):
                picked = ttnn.matmul(
                    select, source, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.compute_config
                )
                view = pager.views[name][index]
                _landed(ttnn.fill_cache(view, picked, batch_idx=lane), view, f"pack landing {name}[{index}]")
                _deallocate(picked)
        # PLE: the nine history rows before the first pad row of the user's snapshot window
        snapshot = self.pack_state.ple_snapshots[user]
        dram = ttnn.DRAM_MEMORY_CONFIG
        tokens_last = ttnn.permute(snapshot, (0, 2, 3, 1), memory_config=dram)  # [1, 4, 640, 137]
        picked = ttnn.matmul(
            tokens_last, self.ple_select, memory_config=dram, compute_kernel_config=self.compute_config
        )
        _deallocate(tokens_last)
        window = ttnn.permute(picked, (0, 3, 1, 2), memory_config=dram)  # [1, 32, 4, 640]
        _deallocate(picked)
        for k, view in enumerate(pager.views["ple"]):
            row = ttnn.slice(
                window,
                (0, k, 0, 0),
                (1, k + 1, ple_module.RESIDUAL_BRANCHES, ple_module.LOCAL_HIDDEN_SIZE),
                memory_config=dram,
            )
            _landed(ttnn.fill_cache(view, row, batch_idx=lane), view, f"pack landing ple[{k}]")
            _deallocate(row)
        _deallocate(window)

    # -- warm ------------------------------------------------------------------------------------------------------------

    def warm(self, lanes: Sequence[int] | None = None) -> None:
        """Compile every landing program (the chain's miss guard closes after the open): every bucket, every user slot
        (the recurrent slice offset), lengths on both sides of the tile and block edges, the given lanes (default: all).
        """

        lanes = list(range(self.lanes)) if lanes is None else list(lanes)
        for rows, bucket in self.pack_state.buckets.items():
            for user in range(bucket.users):
                lane = lanes[user % len(lanes)]
                for length in sorted({rows, max(1, rows - 5), 1}):
                    self.land([(lane, user, length)], rows)
        ttnn.synchronize_device(self.mesh)

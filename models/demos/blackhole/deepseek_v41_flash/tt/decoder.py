# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One DeepSeek-V4.1-Flash decode step on the device: token ids -> embedding -> layers (Engram at its layers) -> LM head.

``forward`` only enqueues device ops, so the whole step can be captured as one trace. Host inputs per step are the token
ids and the Engram rows (``HostEngramRows``), written into persistent device buffers before every replay.
"""

import os

import ttnn


class DSV41Decoder:
    def __init__(self, mesh_device, layers, embedding, head, engram=None, groups=None, step_states=None):
        """layers: list of (layer_id, DSV41Layer, step_inputs); engram: {layer_id: DSV41DeviceEngram}.
        step_inputs is either a dict (host-built, static) or a KEY into ``step_states`` ({key: DSV41StepState}): then the
        step tensors are derived on the device from the uploaded positions at the start of every ``forward`` (inside the trace).
        groups: legacy {key: (attention, step_inputs)} refreshed in place on the host by ``set_inputs``."""
        self.md, self.layers, self.embedding, self.head, self.engram = (
            mesh_device,
            layers,
            embedding,
            head,
            engram or {},
        )
        self.groups = groups or {}
        self.step_states = step_states or {}
        self.pos = None
        self.ctrl = None  # persistent packed control input [T, 32] int32 per device: col 0 = token id, col 1 = position
        self.rows_cat = None  # persistent packed Engram rows [T,1,1, n_engram * Kin] bf16
        self.sampled = None
        self.mesh_config = self.ccl = None  # set by ``enable_sampling``
        self.tokens = None  # persistent device input buffers, set by ``set_inputs``
        self.rows = {}

    def enable_device_loop(self, mesh_config, ccl, tokens, positions):
        """Host-free token loop: after every step the sampled token is written back into the persistent token buffer and the
        position buffer is incremented, both INSIDE the trace; replays can then run back to back. Requires the Engram rows to be
        produced on the device too (``engram_rows_fn``), since the host never sees the token. ``tokens`` [B] / ``positions`` [B]
        torch tensors: the initial state."""
        import torch

        rows, cols = tuple(self.md.shape)
        mp = ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(rows, cols))
        self.mesh_config, self.ccl = mesh_config, ccl
        self.tok_dev = ttnn.from_torch(
            tokens.reshape(-1, 1).to(torch.int32),
            device=self.md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mp,
        )
        self.pos_dev = ttnn.from_torch(
            positions.reshape(-1).to(torch.int32),
            device=self.md,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mp,
        )
        self.device_loop = True

    device_loop = False
    rows_shard = False  # Engram rows uploaded column-sharded (8x fewer host bytes) and all-gathered over the mesh columns inside the trace
    engram_rows_fn = None  # callable(tokens_u32[T,1], pos_i32[T]) -> {layer_id: rows [T,1,1,Kin] tile bf16}, built from the device Engram table

    def enable_sampling(self, mesh_config, ccl):
        """Add greedy sampling to the traced step (``forward`` then also fills ``self.sampled``)."""
        self.mesh_config, self.ccl = mesh_config, ccl

    def prepare_packed_inputs(self, token_ids, engram_rows, positions, with_ctrl=True):
        """Host half of ``set_packed_inputs`` (torch -> host tensors); safe to run on a worker thread while the device replays."""
        import torch

        rows, cols = tuple(self.md.shape)
        B = token_ids.shape[0]
        mp = ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(rows, cols))
        ctrl = torch.zeros(B, 32, dtype=torch.int32)
        ctrl[:, 0], ctrl[:, 1] = token_ids.reshape(-1).to(torch.int32), positions.reshape(-1).to(torch.int32)
        host_c = (
            ttnn.from_torch(ctrl, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp) if with_ctrl else None
        )
        order = sorted(engram_rows)
        host_r = None
        if order:
            cat = torch.cat([engram_rows[l].reshape(B, 1, 1, -1) for l in order], dim=-1).to(torch.bfloat16)
            mpr = (
                ttnn.ShardTensor2dMesh(self.md, dims=(0, 3), mesh_shape=(rows, cols)) if self.rows_shard else mp
            )  # rows_shard: each column uploads 1/8 (gathered on device)
            host_r = ttnn.from_torch(
                cat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mpr
            )  # row-major: a tile layout would pad each user's row to 32 rows (32x the bytes)
        return order, host_c, host_r

    def upload_packed_inputs(self, prepared):
        """Device half: copy the prepared host tensors into the persistent buffers (call after the previous replay finished)."""
        self.engram_order, host_c, host_r = prepared
        if self.ctrl is None:
            self.ctrl = ttnn.to_device(host_c, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.rows_cat = (
                ttnn.to_device(host_r, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG) if host_r is not None else None
            )
        else:
            ttnn.copy_host_to_device_tensor(host_c, self.ctrl)
            if host_r is not None:
                ttnn.copy_host_to_device_tensor(host_r, self.rows_cat)

    def upload_rows_only(self, prepared):
        """Device-loop mode: tokens / positions live on the device, only the Engram rows of the new token are uploaded."""
        self.engram_order, _, host_r = prepared
        if self.rows_cat is None:
            self.rows_cat = ttnn.to_device(host_r, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn.copy_host_to_device_tensor(host_r, self.rows_cat)

    def set_packed_inputs(self, token_ids, engram_rows, positions):
        """Two uploads per step instead of four: tokens + positions in one int32 tensor, all Engram rows in one bf16 tensor."""
        self.upload_packed_inputs(self.prepare_packed_inputs(token_ids, engram_rows, positions))

    def set_inputs(self, token_ids, engram_rows, positions=None):
        """token_ids [B] torch; engram_rows {layer_id: [B,1,Kin] torch bf16}. First call allocates the persistent device
        buffers, later calls overwrite them in place (trace-safe)."""
        import torch

        rows, cols = tuple(self.md.shape)
        if positions is not None:
            for attn, st in self.groups.values():
                attn.step_inputs(positions, st)
            if self.step_states:
                host_p = ttnn.from_torch(
                    positions.to(torch.int32),
                    dtype=ttnn.int32,
                    mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(rows, cols)),
                )
                if self.pos is None:
                    self.pos = ttnn.to_device(host_p, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                else:
                    ttnn.copy_host_to_device_tensor(host_p, self.pos)
        host_t = ttnn.from_torch(
            token_ids.reshape(-1, 1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(rows, cols)),
        )
        host_r = {
            lid: ttnn.from_torch(
                r.reshape(-1, 1, 1, r.shape[-1]).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(rows, cols)),
            )
            for lid, r in engram_rows.items()
        }
        if self.tokens is None:
            self.tokens = ttnn.to_device(host_t, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.rows = {
                lid: ttnn.to_device(h, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG) for lid, h in host_r.items()
            }
        else:
            ttnn.copy_host_to_device_tensor(host_t, self.tokens)
            for lid, h in host_r.items():
                ttnn.copy_host_to_device_tensor(h, self.rows[lid])

    def _l1(
        self, tag
    ):  # DSV41_L1_DIAG=1: print the L1 allocator state (debug the SDPA static-CB / persistent-buffer clash)
        ttnn.synchronize_device(self.md)
        mv = ttnn.get_memory_view(self.md, ttnn.BufferType.L1)
        print(
            f"L1DIAG {tag:28s} allocated/bank {mv.total_bytes_allocated_per_bank:8d} largest_free {mv.largest_contiguous_bytes_free_per_bank:8d}",
            flush=True,
        )

    def _rows_from_cat(self, T):
        kin = {l: e.kin for l, e in self.engram.items()}
        rows, off = {}, 0
        v2 = os.environ.get("DSV41_ENGRAM_V2") != "0"
        cat = self.rows_cat
        if self.rows_shard:
            cat = self.mesh_config.allgather(cat, self.ccl, axis=1, dim=3)
        rc = ttnn.reshape(cat, [1, 1, T, cat.shape[-1]]) if v2 else cat  # v2: one [T,Kin] tile block (M=T matmul)
        for l in self.engram_order:
            sl = (
                ttnn.slice(rc, [0, 0, 0, off], [1, 1, T, off + kin[l]])
                if v2
                else ttnn.slice(rc, [0, 0, 0, off], [T, 1, 1, off + kin[l]])
            )
            rows[l] = ttnn.to_layout(sl, ttnn.TILE_LAYOUT)
            off += kin[l]
        return rows

    def forward(self):
        """-> logits shard [1,1,T,vocab/cols] fp32 (see ``DSV41DeviceHead``)."""

        diag = os.environ.get("DSV41_L1_DIAG") == "1"
        if diag:
            self._l1("start")
        packed = self.ctrl is not None
        if self.device_loop:
            tokens, self.pos = self.tok_dev, self.pos_dev
            if self.engram_rows_fn is not None:
                self.rows = self.engram_rows_fn(tokens, self.pos)
            elif (
                self.rows_cat is not None
            ):  # token fed back on the device, Engram rows of the new token uploaded by the host
                self.rows = self._rows_from_cat(self.rows_cat.shape[0])
            else:
                self.rows = {}
        elif packed:  # tokens / positions / Engram rows arrive in two packed buffers (set_packed_inputs)
            T = self.ctrl.shape[0]
            tokens = ttnn.typecast(ttnn.slice(self.ctrl, [0, 0], [T, 1]), ttnn.uint32)
            self.pos = ttnn.reshape(ttnn.slice(self.ctrl, [0, 1], [T, 2]), [T])
            self.rows = self._rows_from_cat(T)
        else:
            tokens = self.tokens
        x, pre = self.embedding.forward(tokens)
        if diag:
            self._l1("after embedding")
        states = {k: ss.build(self.pos) for k, ss in self.step_states.items()}  # device-side, from the positions
        for lid, layer, st in self.layers:
            if not isinstance(st, dict):
                st = states[st]
            if lid in self.engram:
                eng = self.engram[lid]
                x = (
                    eng.forward(x, self.rows[lid])
                    if os.environ.get("DSV41_ENGRAM_V2") == "0"
                    else eng.forward_v2(x, self.rows[lid])
                )
                if diag:
                    self._l1(f"after engram {lid}")
            x, pre = layer.forward(x, pre, st)
            if diag:
                self._l1(f"after layer {lid}")
        logits = self.head.forward(x, pre)
        if self.device_loop:
            nxt = self.head.sample_global(logits, self.mesh_config, self.ccl)
            ttnn.copy(nxt, self.tok_dev)  # feed the sampled token to the next replay ...
            ttnn.copy(ttnn.add(self.pos_dev, 1), self.pos_dev)  # ... and advance the position
        elif self.mesh_config is not None:
            self.sampled = self.head.sample(logits, self.mesh_config, self.ccl)
        return logits

    def snapshot_states(self):
        """Step-carried device state (compressor 'previous token' buffers): lets a warm-up / compile pass run without
        corrupting the state of the real first step."""
        return [
            (layer.attention, layer.attention.snapshot_state())
            for _, layer, _ in self.layers
            if hasattr(layer.attention, "snapshot_state")
        ]

    def restore_states(self, snaps):
        for attn, snap in snaps:
            attn.restore_state(snap)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded QSA prefill preserving every logical token and prefix state.

This is a functional context implementation, not a new precision policy.
``forward_chunk`` accepts 1..32 actual tokens, projects all rows together,
updates the existing paged K/V and compressed-indexer caches, and attends only
to the device-selected <=2051 keys. Cache updates are deliberately serialized:
parallel paged_update_cache calls targeting rows of the same tile would race.

``forward_chunk`` leaves state.position unchanged, matching QSA.step. The
orchestrator must advance it once by the logical chunk length. The convenience
``prefill_forward`` owns that advancement and returns all logical outputs.
Its input is [1,1,S,2560] BF16 on device; all token-dependent work, selection,
cache reads/writes and output assembly stay on device. No host fallback occurs
inside either forward. State and page-table ownership match QSA.make_state.

Every new request needs reset state; continuation needs the real preceding
three raw indexer keys plus compressed keys and paged K/V, including when its
start is not divisible by four, a page, or a tile. Padded outputs/cache writes
never advance position or become visible to attention. The outer decoder may
use forward_chunk between hyper-connection mixing and MoE on the same rows.
"""

from .precise_projection import precise_indexer_rope


class ChunkedQSA:
    def __init__(self, attention):
        import torch

        from .qsa_selected_cache import selected_paged_kv

        self.attention = attention
        self.t = attention.ttnn
        self.gather = selected_paged_kv
        t = self.t
        device = attention.loader.mesh_device

        def tensor(value, dtype=t.float32, layout=t.TILE_LAYOUT):
            return t.from_torch(value.contiguous(), dtype=dtype, layout=layout, device=device)

        self.query_offsets = tensor(torch.arange(32).float().reshape(1, 1, 32, 1))
        self.token_offsets = tensor(torch.arange(2048).remainder(4).float().reshape(1, 1, 1, 2048))
        self.tail_offsets = tensor(torch.arange(32).float().reshape(1, 1, 1, 32))
        self.invalid_ids = tensor(torch.full((1, 1, 32, 2080), 0xFFFFFFFF, dtype=torch.int64), t.uint32)
        self.output_pages = tensor(torch.arange(attention.context // 32).reshape(1, -1), t.int32, t.ROW_MAJOR_LAYOUT)
        self.sdpa_program = t.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )

    def _rope(self, value, positions):
        """value [1,H,T,D], positions [1,1,T,1]; only first 64 dims rotate."""
        t = self.t
        count = value.shape[-2]
        indices = t.reshape(t.to_layout(t.typecast(positions, t.uint32), t.ROW_MAJOR_LAYOUT), (1, count))
        cos = t.reshape(
            t.to_layout(t.embedding(indices, self.attention.cos, layout=t.ROW_MAJOR_LAYOUT), t.TILE_LAYOUT),
            (1, 1, count, 64),
        )
        sin = t.reshape(
            t.to_layout(t.embedding(indices, self.attention.sin, layout=t.ROW_MAJOR_LAYOUT), t.TILE_LAYOUT),
            (1, 1, count, 64),
        )
        if value.shape[-1] == 128:
            return precise_indexer_rope(value, cos, sin)
        rotated = t.concat([t.neg(value[..., 32:64]), value[..., :32]], dim=-1)
        result = t.add(t.multiply(value[..., :64], cos), t.multiply(rotated, sin))
        return t.concat([result, value[..., 64:]], dim=-1)

    def advance(self, state, count):
        """Advance a device-valued position by a host-known logical row count."""
        t = self.t
        pos = t.to_layout(t.reshape(t.typecast(state.position, t.float32), (1, 1, 1, 1)), t.TILE_LAYOUT)
        updated = t.reshape(t.to_layout(t.typecast(t.add(pos, count), t.int32), t.ROW_MAJOR_LAYOUT), (1,))
        t.copy(updated, state.position)

    def _selected_ids(self, query, *, state, positions):
        t = self.t
        count = query.shape[-2]
        complete = t.floor(t.multiply(t.add(positions, 1), 0.25))
        scores = self.attention.indexer_scores(query, state.compressed_keys)
        scores = t.where(t.lt(state.block_numbers, complete), scores, float("-inf"))
        top_k = min(512, state.capacity // 4)
        _, indices = t.topk(scores, k=top_k, dim=-1, sorted=False)
        # Every coordinate is <=262175 and therefore exactly represented in
        # FP32. This also avoids the native UINT32 transpose width limitation
        # inside repeat_interleave. Never route indices through BF16, and insert
        # the UINT32_MAX sentinel only after all coordinate arithmetic.
        expanded = t.repeat_interleave(t.typecast(indices, t.float32), repeats=4, dim=3)
        ids = t.typecast(t.add(t.multiply(expanded, 4), self.token_offsets[..., : top_k * 4]), t.uint32)
        valid = t.lt(expanded, complete)
        invalid = self.invalid_ids[:, :, :count, : top_k * 4]
        ids = t.where(valid, ids, invalid)
        if top_k < 512:
            ids = t.concat([ids, self.invalid_ids[:, :, :count, top_k * 4 : 2048]], dim=-1)
        tail = t.add(t.multiply(complete, 4), self.tail_offsets)
        tail_valid = t.le(tail, positions)
        tail = t.where(tail_valid, t.typecast(tail, t.uint32), self.invalid_ids[:, :, :count, 2048:])
        return t.reshape(t.concat([ids, tail], dim=-1), (count, 2080))

    def forward_chunk(self, hidden_states, *, state, return_selected_ids=False):
        """Consume 1..32 rows, preserving state.position for caller advancement."""
        t, a = self.t, self.attention
        count = hidden_states.shape[-2]
        if tuple(hidden_states.shape) != (1, 1, count, 2560) or not 1 <= count <= 32:
            raise ValueError("Expected [1,1,1..32,2560] valid token rows")
        pos = t.to_layout(t.reshape(t.typecast(state.position, t.float32), (1, 1, 1, 1)), t.TILE_LAYOUT)
        positions = t.add(pos, self.query_offsets[:, :, :count, :])
        group = t.floor(t.multiply(positions, 0.25))
        group_starts = t.multiply(group, 4)
        write_groups = t.where(t.eq(t.subtract(positions, group_starts), 3), group, -1.0)

        q_gate = t.permute(t.reshape(a.linear(hidden_states, a.q, a.compute), (1, count, 24, 512)), (0, 2, 1, 3))
        query, gate = q_gate[..., :256], q_gate[..., 256:]
        key = t.permute(t.reshape(a.linear(hidden_states, a.k, a.compute), (1, count, 2, 256)), (0, 2, 1, 3))
        value = t.permute(t.reshape(a.linear(hidden_states, a.v, a.compute), (1, count, 2, 256)), (0, 2, 1, 3))
        query = self._rope(a.rms(query, a.q_norm, a.eps), positions)
        key = self._rope(a.rms(key, a.k_norm, a.eps), positions)

        index_qk = a.project_index_qk(hidden_states)
        index_query = t.permute(t.reshape(index_qk[..., :512], (1, count, 4, 128)), (0, 2, 1, 3))
        index_query = self._rope(a.rms(index_query, a.index_q_norm, a.eps), positions)
        raw = index_qk[..., 512:]
        history = t.concat([state.raw_history, raw], dim=2)
        # Sliding four-row sums must accumulate FP32, then round to BF16 before
        # the HF k-layernorm. Both arbitrary continuation phase and short tails
        # use exactly the same device expression.
        history_fp32 = t.typecast(history, t.float32)
        pooled = t.add(history_fp32[:, :, :count, :], history_fp32[:, :, 1 : count + 1, :])
        pooled = t.add(pooled, history_fp32[:, :, 2 : count + 2, :])
        pooled = t.add(pooled, history_fp32[:, :, 3 : count + 3, :])
        pooled = t.typecast(t.multiply(pooled, 0.25), t.bfloat16)
        pooled = self._rope(a.rms(pooled, a.index_k_norm, a.eps), group_starts)

        for row in range(count):
            position = t.reshape(
                t.to_layout(t.typecast(positions[:, :, row : row + 1, :], t.int32), t.ROW_MAJOR_LAYOUT), (1,)
            )
            group_position = t.reshape(
                t.to_layout(t.typecast(write_groups[:, :, row : row + 1, :], t.int32), t.ROW_MAJOR_LAYOUT), (1,)
            )
            a._update(state.k_cache, t.permute(key[:, :, row : row + 1, :], (0, 2, 1, 3)), position, state.page_table)
            a._update(state.v_cache, t.permute(value[:, :, row : row + 1, :], (0, 2, 1, 3)), position, state.page_table)
            a._update(state.compressed_keys, pooled[:, :, row : row + 1, :], group_position)
        t.copy(history[:, :, -3:, :], state.raw_history)

        selected_ids = self._selected_ids(index_query, state=state, positions=positions)
        selected_k, selected_v = self.gather(state.k_cache, state.v_cache, state.page_table, selected_ids)
        sentinel = t.reshape(self.invalid_ids[:, :, :count, :], (count, 2080))
        allowed = t.ne(selected_ids, sentinel)
        mask = t.reshape(t.typecast(t.where(allowed, 0.0, -1.0e9), t.bfloat16), (count, 1, 1, 2080))
        attention = t.transformer.scaled_dot_product_attention(
            t.permute(query, (2, 1, 0, 3)),
            selected_k,
            selected_v,
            is_causal=False,
            attn_mask=mask,
            scale=256**-0.5,
            program_config=self.sdpa_program,
            compute_kernel_config=a.compute,
            memory_config=t.DRAM_MEMORY_CONFIG,
        )
        attention = t.permute(attention, (2, 1, 0, 3))
        attention = t.multiply(attention, t.sigmoid(gate))
        attention = t.reshape(t.permute(attention, (0, 2, 1, 3)), (1, 1, count, 6144))
        output = a.linear(attention, a.out, a.compute)
        return (output, selected_ids) if return_selected_ids else output

    def prefill_forward(self, hidden_states, *, state):
        """All outputs for any positive logical S; continue the supplied prefix.

        Assembly uses an output-only paged buffer, so it never retains thousands
        of live chunk outputs or concatenates an unbounded Python tensor list.
        Its physical tail is local output padding, not part of request state.
        """
        t = self.t
        length = hidden_states.shape[-2]
        if tuple(hidden_states.shape) != (1, 1, length, 2560) or not 1 <= length <= self.attention.context:
            raise ValueError("Expected positive logical [1,1,S,2560] within advertised context")
        pages = (length + 31) // 32
        output = t.allocate_tensor_on_device(
            t.Shape((pages, 1, 32, 2560)),
            t.bfloat16,
            t.TILE_LAYOUT,
            self.attention.loader.mesh_device,
            t.DRAM_MEMORY_CONFIG,
        )
        for start in range(0, length, 32):
            count = min(32, length - start)
            chunk = self.forward_chunk(hidden_states[:, :, start : start + count, :], state=state)
            page_table = self.output_pages[:, start // 32 : start // 32 + 1]
            t.experimental.paged_fill_cache(output, chunk, page_table, batch_idx=0)
            self.advance(state, count)
        output = t.reshape(output, (1, 1, pages * 32, 2560))
        return output if length % 32 == 0 else output[:, :, :length, :]

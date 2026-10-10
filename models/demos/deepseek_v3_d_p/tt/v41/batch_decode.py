# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 batched decode: one step for every user of the mesh (harness; eager, or traced once).

``TtV41BatchDecoder`` runs ``TtV41Decoder``'s step on a tile whose rows are different users: row ``u`` of the chips
of mesh row ``r`` is slot ``r * users_per_rank + u`` of the :class:`V41DecodePool`, so the SP ranks decode different
users (TP splits hidden and heads as before) and the per-step host inputs are per rank: the slots' tokens, RoPE rows
at their positions, the indexer masks' visible counts, the ring slot and the compressed rows the step writes.

Per step and layer the graph is the single-token one with the position-dependent pieces made per row:

* the window row of every user goes to its layer ring at slot ``step % 128`` (``scatter_rows``; the window index
  rows are a constant per layer, the ring being the window);
* a ratio-2 source pools every row with the pool's carry buffers and writes the carry back unconditionally (the odd
  rows' pooled result is their group's row, the even rows' is junk the scatter skips: the step's write indices are
  the sentinel for rows that complete no group), a ratio-1 source writes every active row; both go straight to the
  user's row ``position // ratio`` (no scratch row, no copy after the step);
* an index source scores the query tile against each slot's index keys (``cache_batch_idx``), takes row ``u`` of
  slot ``u``'s selection, and the attention reads the pool with per-row offsets;
* the MoE sees ``users_per_rank`` real rows per chip (``padding_config``), so only they are dispatched.

Users join from a prefill state (``join``: the pool copies the state's caches into the slot) and leave by being
forgotten; a free slot's rows still run (its inputs are zeros, its writes go nowhere). The step's graph is the same
for every position, so a single capture replays every step (``trace=True``). Limits as ``TtV41Decoder``: text only,
positions from ``SLIDING_WINDOW - 1`` to below ``max_seq_len``, compressed rows within the candidate budget.
"""

import os
import time
from dataclasses import dataclass

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand
from models.demos.deepseek_v3_d_p.tt.v41.attention import TOPK_ALIGN, TtV41Attention
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.decode import MOE_ROWS, ROWS, TtV41Decoder
from models.demos.deepseek_v3_d_p.tt.v41.head_layout import o_heads, q_heads
from models.demos.deepseek_v3_d_p.tt.v41.indexer import TtV41Indexer
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.tt.v41.moe import TtV41Moe
from models.demos.deepseek_v3_d_p.tt.v41.pool import RING, V41DecodePool, scatter_rows
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_e4m3_qdq, fp8_qdq
from models.demos.deepseek_v3_d_p.tt.v41.rope import cos_sin
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

NEG_INF = float("-inf")


@dataclass
class _User:
    position: int  # the position the next token goes to
    engram_history: torch.Tensor | None = None


class TtV41BatchDecoder(TtV41Decoder):
    def __init__(
        self,
        model: TtV41Transformer,
        weight_cache_path,
        users_per_rank: int = 4,
        routed_expert_weights_dtype=ttnn.bfloat8_b,
        trace: bool = False,
        warmup: int = 2,
        max_seq_len: int | None = None,
    ):
        """``users_per_rank`` slots per mesh row (``sp * users_per_rank`` users); ``max_seq_len`` of the pool
        (default the model's). ``trace``: capture the step once after ``warmup`` eager steps and replay it."""
        super().__init__(model, weight_cache_path, routed_expert_weights_dtype, trace=trace, warmup=warmup)
        self.users = users_per_rank
        self.pool = V41DecodePool(
            self.mesh_device,
            self.config,
            max_seq_len or model.max_seq_len,
            model.layers,
            users_per_rank,
            model.kv_format,
        )
        self.slots: dict[int, _User] = {}
        # the MoE's per-rank [real rows, right padding]: the slots' rows of every rank
        self.padding_config = (
            None
            if os.environ.get("V41_BATCH_MOE_PAD", "1") == "0"
            else ttnn.from_torch(
                torch.tensor([[users_per_rank, 0]] * self.sp, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=self.mesh_device.shape),
            )
        )
        # per ratio: fp32 column indices [1, 1, ROWS, cap + 32] of an index source's score (the mask compares them
        # with the step's per-row visible counts); the sentinel tail of the pad rows' selection
        self.iota = {}
        for src, slot in self.pool.slots.items():
            if src < 0:
                continue
            ratio = self.config.compress_ratio(src)
            if ratio not in self.iota:
                cols = torch.arange(slot.cap + 32, dtype=torch.float32).view(1, 1, 1, -1).expand(1, 1, ROWS, -1)
                self.iota[ratio] = self._replicated(cols.contiguous(), ttnn.float32)
        self.pad_topk = self._replicated(
            torch.full((1, 1, ROWS - self.users, self.config.INDEX_TOPK), -1, dtype=torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.capture = None
        self.eager_count = 0
        self.join_traces: dict[tuple[int, int], SubDeviceTraceController] = {}
        self.join_buffers: dict = {}
        self.last_join: dict = {}

    # --- users -------------------------------------------------------------------------------------------------
    @property
    def num_slots(self) -> int:
        return self.sp * self.users

    def join(self, slot: int, tokens: torch.Tensor, state: V41PrefillState) -> None:
        """Slot ``slot`` takes over ``state`` after ``prefill(tokens)`` filled it: its caches move into the pool
        (``V41DecodePool.join_ops``; replayed when ``capture_join`` captured this state and slot, else eager), its
        Engram history is seeded from the last compressed ids."""
        assert 0 <= slot < self.num_slots, slot
        assert state.start >= self.config.SLIDING_WINDOW - 1, f"a prompt of {state.start}: decode needs a full window"
        user = _User(state.start)
        if self.model.engram:
            hash_ = self.model.engram_hash
            n = hash_.layout.max_ngram_size - 1
            user.engram_history = hash_.token_map[tokens[-n:].to(torch.int64)].clone()
        t0 = time.time()
        inputs = self.pool.join_inputs(slot, state)
        trace = self.join_traces.get((id(state), slot))
        if trace is None:
            idx = {k: ttnn.to_device(v, self.mesh_device) for k, v in inputs.items()}
            self.pool.join_ops(slot, state, idx)
            for t in idx.values():
                ttnn.deallocate(t)
            mode = "eager"
        else:
            self._stage_join(inputs)
            trace.replay(blocking=False)
            mode = "replay"
        self.last_join = {"mode": mode, "s": round(time.time() - t0, 4)}
        self.slots[slot] = user

    def _stage_join(self, inputs: dict) -> None:
        if not self.join_buffers:
            self.join_buffers = {k: ttnn.to_device(v, self.mesh_device) for k, v in inputs.items()}
            return
        for k, v in inputs.items():
            ttnn.copy_host_to_device_tensor(v, self.join_buffers[k])

    def capture_join(self, state: V41PrefillState, slot: int) -> None:
        """Capture the join of ``state`` into ``slot`` (their tensors are persistent; the index buffers are
        uploaded per join), so later joins of the pair replay it: a join is ~60 small ops, 20 to 80 ms issued
        eagerly."""
        key = (id(state), slot)
        if key in self.join_traces:
            return
        self._stage_join(self.pool.join_inputs(slot, state))
        controller = SubDeviceTraceController(self.mesh_device)
        ttnn.synchronize_device(self.mesh_device)
        controller.begin_capture()
        try:
            self.pool.join_ops(slot, state, self.join_buffers)
        finally:
            controller.end_capture()
        self.join_traces[key] = controller

    def leave(self, slot: int) -> None:
        self.slots.pop(slot, None)

    def position(self, slot: int) -> int:
        return self.slots[slot].position

    # --- host inputs ---------------------------------------------------------------------------------------------
    def _host_inputs(self, tokens: dict[int, int]) -> dict:
        """The step's host tensors (per rank, device layouts), keyed as the device buffers: ``tokens`` slot ->
        token of the slots that step (the others get zeros and write nothing)."""
        cfg, sp, users = self.config, self.sp, self.users
        step = self.pool.step
        ids = torch.zeros(sp, 1, ROWS, dtype=torch.int32)
        positions = torch.zeros(sp, ROWS, dtype=torch.int64)
        active = torch.zeros(sp, ROWS, dtype=torch.bool)
        for slot, token in tokens.items():
            r, u = divmod(slot, users)
            ids[r, 0, u] = token
            positions[r, u] = self.slots[slot].position
            active[r, u] = True
        inputs = {"token": self.pool.per_rank(ids, ttnn.uint32, device=False)}

        def rope(compressed: bool, pos: torch.Tensor):
            out = []
            for t in cos_sin(cfg, compressed, pos.flatten()):
                out.append(self.pool.per_rank(t.view(sp, 1, ROWS, -1), ttnn.bfloat16, ttnn.TILE_LAYOUT, device=False))
            return tuple(out)

        inputs["rope0"], inputs["ropec"] = rope(False, positions), rope(True, positions)
        inputs["ropeg"] = rope(True, positions // 2 * 2)
        inputs["ring"] = self._replicated(
            torch.full((1, 1, 1, ROWS), step % RING, dtype=torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            device=False,
        )
        for ratio in sorted(self.iota):
            visible = torch.where(active, (positions + 1) // ratio, torch.zeros_like(positions)).float()
            inputs[f"visible{ratio}"] = self.pool.per_rank(
                visible.view(sp, 1, ROWS, 1), ttnn.float32, ttnn.TILE_LAYOUT, device=False
            )
            writes = active & (positions % ratio == ratio - 1)
            comp = torch.where(writes, positions // ratio, torch.full_like(positions, -1)).to(
                torch.int32
            )  # -1: sentinel
            inputs[f"comp{ratio}"] = self.pool.per_rank(comp.view(sp, 1, 1, ROWS), ttnn.uint32, device=False)
        if self.model.engram:
            hash_ = self.model.engram_hash
            order = list(tokens)
            if order:
                ids_batch, _ = hash_.step(
                    torch.tensor([tokens[s] for s in order]),
                    torch.stack([self.slots[s].engram_history for s in order]),
                )  # [U, n_layers, cols]
            inputs["engram"] = {}
            for l, e in self.model.engram.items():
                ids = torch.zeros(sp * ROWS, hash_.n_hash_cols, dtype=torch.int64)
                for k, slot in enumerate(order):
                    r, u = divmod(slot, users)
                    ids[r * ROWS + u] = ids_batch[k, hash_.layer_index(l)]
                inputs["engram"][l] = e.prepare_host(ids, sp * ROWS)
        return inputs

    # --- one step ----------------------------------------------------------------------------------------------
    def decode(self, tokens: dict[int, int]) -> dict[int, torch.Tensor]:
        """Append ``tokens`` (slot -> token) at the slots' positions -> slot -> fp32 logits ``[vocab]`` of the next
        position; the slots advance by one, the pool's step too."""
        for slot in tokens:
            user = self.slots[slot]
            assert user.position < self.pool.max_seq_len, f"slot {slot} at {user.position}: the pool is full"
        t0 = time.time()
        inputs = self._host_inputs(tokens)
        t1 = time.time()
        self._stage(inputs)
        t2 = time.time()
        if not self.trace:
            mode = "eager"
        elif self.capture is not None:
            mode = "replay"
        elif self.eager_count >= self.warmup:
            mode = "capture"
        else:
            mode = "eager"
        self.last = {"mode": mode, "inputs_s": round(t1 - t0, 4), "stage_s": round(t2 - t1, 4)}
        if mode == "eager":
            self.eager_count += 1
            logits = self._forward(self.on_block)
        else:
            if mode == "capture":
                self.capture = self._capture_step()
            ttnn.synchronize_device(self.mesh_device)
            t3 = time.time()
            self.capture.controller.replay(blocking=False)
            t4 = time.time()
            ttnn.synchronize_device(self.mesh_device)
            t5 = time.time()
            logits = self.capture.logits
            self.last.update(sync_s=round(t3 - t2, 4), device_s=round(t5 - t3, 4), issue_s=round(t4 - t3, 4))
            if self.on_block is not None:
                for layer, x in self.capture.blocks:
                    self.on_block(layer, x)
        t6 = time.time()
        out = self._logits_to_host(logits, tokens)
        if mode == "eager":
            ttnn.deallocate(logits)
        self.last["read_s"] = round(time.time() - t6, 4)
        if self.model.engram and tokens:
            order = list(tokens)
            _, histories = self.model.engram_hash.step(
                torch.tensor([tokens[s] for s in order]), torch.stack([self.slots[s].engram_history for s in order])
            )
            for k, slot in enumerate(order):
                self.slots[slot].engram_history = histories[k]
        for slot in tokens:
            self.slots[slot].position += 1
        self.pool.step += 1
        return out

    def _logits_to_host(self, logits, tokens: dict[int, int]) -> dict[int, torch.Tensor]:
        """The gathered ``[1, 1, slots, vocab]`` fp32 logits (the same on every chip) -> slot -> ``[vocab]``."""
        rows = ttnn.to_torch(ttnn.get_device_tensors(logits)[0])[0, 0].float()
        return {slot: rows[slot] for slot in tokens}

    def replay_device_time(self, repeats: int = 3) -> dict:
        """Device time of one replay of the captured step without host work -> {"step": min seconds}."""
        if self.capture is None:
            return {}
        ttnn.synchronize_device(self.mesh_device)
        best = None
        for _ in range(repeats):
            t0 = time.time()
            self.capture.controller.replay(blocking=False)
            ttnn.synchronize_device(self.mesh_device)
            dt = time.time() - t0
            best = dt if best is None else min(best, dt)
        return {"step": round(best, 4)}

    def _capture_step(self):
        from models.demos.deepseek_v3_d_p.tt.v41.decode import _Trace

        controller = SubDeviceTraceController(self.mesh_device)
        for moe in self.moes:
            moe.set_trace_controller(controller)
        blocks = []
        ttnn.synchronize_device(self.mesh_device)
        controller.begin_capture()
        try:
            logits = self._forward(lambda layer, x: blocks.append((layer, x)))
        finally:
            controller.end_capture()
            for moe in self.moes:
                moe.set_trace_controller(None)
        return _Trace(controller, logits, 0, blocks)

    def release(self) -> None:
        if self.capture is not None:
            self.capture.controller.release()
            self.capture = None
        for controller in self.join_traces.values():
            controller.release()
        self.join_traces = {}

    @property
    def trace_segments(self) -> int:
        return self.capture.controller.num_segments if self.capture is not None else 0

    def _forward(self, on_block):
        """The step's device graph on the staged buffers -> the logits tile ``[1, 1, ROWS, vocab/tp]`` per chip
        (row u: slot u of the chip's rank); the pool is updated in place."""
        model = self.model
        masks = {ratio: self._mask(ratio) for ratio in self.iota}
        selection = {}
        h = model.embedding(self.buffers["token"])
        x = mhc_expand(ttnn.typecast(h, ttnn.float32), self.config.HC_MULT)
        pre = self.pre_mix
        for layer, block, moe in zip(model.layers, model.blocks, self.moes):
            if layer in model.engram:
                x = model.engram[layer](x, self.buffers["engram"][layer])
            x, pre = block.residual(
                x,
                pre,
                attention=lambda h, a=block.attn, n=block.attn_norm: self._attention(a, n(h), masks, selection),
                ffn=lambda h, m=moe, n=block.ffn_norm: self._ffn(m, n(h)),
            )
            if on_block is not None:
                on_block(layer, x)
        final = ttnn.typecast(model.blocks[-1].residual.final_collapse(x, pre), ttnn.bfloat16)
        logits, _ = model.head(final, 0)  # [1, 1, ROWS, vocab/tp] fp32 per chip, row u slot u of the chip's rank
        # the slots' rows, gathered over TP (the vocabulary) and SP (the ranks): one [1, 1, slots, vocab] read
        rows = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(logits)
        if self.users < ROWS:
            part = ttnn.slice(rows, [0, 0, 0, 0], [1, 1, self.users, rows.shape[3]])
            ttnn.deallocate(rows)
            rows = part
        # the ranks first (pages of vocab/tp fp32; a full-vocabulary row is beyond L1 as an input page)
        rows = self.pool.gather_sp(rows, dim=2)
        return model.blocks[0].attn.ccl.tp_all_gather(rows, dim=3)

    def _mask(self, ratio: int):
        """Additive ``[1, 1, ROWS, cap + 32]`` bf16: row u -inf at the columns at or past its visible count."""
        beyond = ttnn.ge(self.iota[ratio], self.buffers[f"visible{ratio}"])
        return ttnn.typecast(ttnn.where(beyond, NEG_INF, 0.0), ttnn.bfloat16)

    def _ffn(self, moe: TtV41Moe, h):
        """The decode MoE on the tile repeated to MOE_ROWS rows, the slots' rows real -> ROWS rows."""
        x = ttnn.concat([h] * (MOE_ROWS // ROWS), dim=2)
        if self.padding_config is None:
            out = moe(x)
        else:
            out = moe(x, actual_isl=self.users, padding_config=self.padding_config)
        return ttnn.slice(out, [0, 0, 0, 0], [1, 1, ROWS, out.shape[3]])

    # --- attention ------------------------------------------------------------------------------------------------
    def _attention(self, attn: TtV41Attention, x, masks: dict, selection: dict):
        """``TtV41Attention.forward`` for the tile of users (``x`` [1, 1, ROWS, hidden/tp] bf16)."""
        layer, pool, cfg = attn.layer, self.pool, self.config
        cos, sin = self.buffers["ropec" if attn.ratio else "rope0"]
        xq = fp8_qdq(x)
        qr = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wq_a")), weight=attn.q_norm, epsilon=attn.eps)
        q = attn._dense(fp8_qdq(qr), "wq_b")  # [1, 1, ROWS, H/tp * head_dim]
        kv = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wkv")), weight=attn.kv_norm, epsilon=attn.eps)
        kv_format = pool.kv_format if attn.ratio else MlaKvCacheFormat.BF16_RM
        reference_values = kv_format == MlaKvCacheFormat.BF16_RM
        kv = attn._rope(kv, cos, sin)
        kv_tensor = pool.kv_tensor(layer)
        rows = self._encode(fp8_qdq(kv) if reference_values else kv, kv_format)
        scatter_rows(
            rows,
            kv_tensor,
            self.buffers["ring"],
            dst_base=pool.ring_row(layer),
            stride=pool.slot_rows(layer),
            rows=self.users,
        )
        ttnn.deallocate(rows)

        compressed = None
        if attn.ratio:
            ratio, src = attn.ratio, cfg.kv_source(layer)
            if attn.is_kv_source:
                latent = self._compress(attn.compressor, x, layer)
                c_rope = self.buffers["ropeg" if ratio == 2 else "ropec"]  # the group's first position
                keys = attn.index_keys(latent, c_rope)
                comp_kv = attn._rope(latent, *c_rope)
                rows = self._encode(fp4_e4m3_qdq(comp_kv) if reference_values else comp_kv, kv_format)
                idx, n = self.buffers[f"comp{ratio}"], self.users
                scatter_rows(rows, pool.kv[src], idx, dst_base=pool.comp_row(src), stride=pool.slot_rows(layer), rows=n)
                ttnn.deallocate(rows)
                keys = ttnn.to_layout(keys, ttnn.ROW_MAJOR_LAYOUT)
                scatter_rows(keys, pool.index_k_flat[src], idx, stride=pool.slots[src].cap + 32, rows=n)
                ttnn.deallocate(keys)
            if attn.is_index_source:
                compressed, candidates = self._index(
                    attn.indexer, x, qr, src, masks[ratio], selection.get("candidates")
                )
                selection["topk"] = compressed
                if candidates is not None:
                    selection["candidates"] = candidates
            else:
                compressed = selection["topk"]
        rows = self._index_rows(layer, compressed)

        q = attn.ccl.tp_all_gather(q, dim=3)
        out = ttnn.transformer.sparse_sdpa(
            q_heads(q, cos, sin, attn.trans_mat, attn.heads, attn.rope_dim),
            kv_tensor,
            rows,
            attn.head_dim,
            kv_format=kv_format.sparse_sdpa_format,
            scale=attn.scale,
            k_chunk_size=next(c for c in (128, 64, 32) if rows.shape[-1] % c == 0),
            attention_sink=attn.sink,
        )
        out = o_heads(out, cos, ttnn.neg(sin), attn.trans_mat, attn.o_groups, attn.rope_dim)
        if self.tp > 1:
            out = ttnn.mesh_partition(out, dim=1, cluster_axis=TP_AXIS)
        return attn._o_proj(out)

    def _encode(self, rows, fmt: MlaKvCacheFormat):
        """Logical ``[1, 1, ROWS, 512]`` rows -> the row-major rows of ``fmt`` (as ``V41PrefillState._encode``)."""
        rows = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
        if fmt == MlaKvCacheFormat.BF16_RM:
            return rows
        latent, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
            rows, round_scale_to_power_of_two=True
        )
        packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(latent, scales)
        ttnn.deallocate(latent)
        ttnn.deallocate(scales)
        return packed

    def _compress(self, compressor, x, layer: int):
        """The reference ``Compressor`` for every row: ratio 1 the row's latent; ratio 2 the pool of the pool's
        carry (the previous step's projections) and this step's, the carry then replaced by this step's."""
        if compressor.ratio == 1:
            kv = compressor._project(x, compressor.wkv, ttnn.bfloat16)
            return ttnn.rms_norm(kv, weight=compressor.norm, epsilon=compressor.eps)
        xf = ttnn.typecast(x, ttnn.float32)
        kv = compressor._project(xf, compressor.wkv, ttnn.float32)
        score = compressor._project(xf, compressor.wgate, ttnn.float32)
        k0, s0 = self.pool.carry[layer]
        m = ttnn.maximum(s0, score)
        e0, e1 = ttnn.exp(ttnn.subtract(s0, m)), ttnn.exp(ttnn.subtract(score, m))
        pooled = ttnn.divide(ttnn.add(ttnn.multiply(k0, e0), ttnn.multiply(kv, e1)), ttnn.add(e0, e1))
        ttnn.copy(kv, k0)
        ttnn.copy(score, s0)
        return ttnn.rms_norm(ttnn.typecast(pooled, ttnn.bfloat16), weight=compressor.norm, epsilon=compressor.eps)

    def _index(self, indexer: TtV41Indexer, x, qr, source: int, mask, candidates):
        """The indexer of every slot's user over its index keys -> (top-k rows ``[1, 1, ROWS, INDEX_TOPK]`` uint32
        row-major, row u slot u's selection (the pad rows all sentinels); published candidates or None)."""
        cfg, pool = self.config, self.pool
        cap = pool.slots[source].cap
        assert (
            cap // cfg.CANDIDATE_BLOCK_SIZE <= cfg.CANDIDATE_TOPK_BLOCKS
        ), f"decode over {cap} compressed rows needs the candidate-block selection (not implemented)"
        cos, sin = self.buffers["ropec"]
        q = ttnn.linear(fp8_qdq(qr), indexer.wq_b, compute_kernel_config=indexer.compute_kernel_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=indexer.heads, num_kv_heads=0, transpose_k_heads=False
        )
        q = indexer.qdq(indexer.rope(q, cos, sin))
        w = indexer.ccl.tp_all_reduce(ttnn.linear(x, indexer.weights_proj))
        k = ttnn.to_layout(pool.index_k[source], ttnn.TILE_LAYOUT)  # [users, 1, cap + 32, idim]
        config = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(256, cap + 32), head_group_size=0)
        parts, published = [], None
        for u in range(self.users):
            # every row sees every key (the causal window starts past the slot's rows); the mask then hides the rows
            # past each user's visible count, and the pad tile
            score = ttnn.experimental.indexer_score_dsa(
                q, k, w, chunk_start_idx=cap, seq_shard_axes=[], program_config=config, cache_batch_idx=u
            )
            score = ttnn.to_layout(ttnn.add(ttnn.to_layout(score, ttnn.TILE_LAYOUT), mask), ttnn.ROW_MAJOR_LAYOUT)
            topk, published = indexer.select(score, None, 0, cfg.INDEX_TOPK, candidates)
            parts.append(ttnn.slice(topk, [0, 0, u, 0], [1, 1, u + 1, topk.shape[3]]))
        ttnn.deallocate(k)
        if self.users < ROWS:
            parts.append(self.pad_topk)
        return ttnn.concat(parts, dim=2), published

    def _index_rows(self, layer: int, compressed):
        """``sparse_sdpa`` index rows ``[1, 1, ROWS, K]`` uint32 row-major: the layer's window ring, then the selected
        compressed rows of each row's user (offset to its slot) with their sentinel tail."""
        pool = self.pool
        window = pool.window_rows(layer)
        if compressed is None:
            return window
        src = self.config.kv_source(layer)
        comp = ttnn.typecast(ttnn.to_layout(compressed, ttnn.TILE_LAYOUT), ttnn.int32)
        comp = ttnn.where(ttnn.eq(comp, -1), -1, ttnn.add(comp, pool.user_offsets(src)))
        comp = ttnn.to_layout(ttnn.typecast(comp, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        rows = ttnn.concat([window, comp], dim=-1)
        assert rows.shape[3] % TOPK_ALIGN == 0, tuple(rows.shape)
        return rows

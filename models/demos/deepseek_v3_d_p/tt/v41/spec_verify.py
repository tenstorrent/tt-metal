# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The verify step of speculative decode: a slot's ``1 + K`` tokens (the
committed next token, then ``K`` drafts) as ``1 + K`` rows of the batched decode tile, at consecutive positions, in
one step; the host accepts the longest matching prefix and commits.

``TtV41VerifyDecoder`` is the batched decoder (``tt/v41/batch_decode.py``) with the per-row pool addresses as step
inputs instead of the tile row = pool slot rule:

* the window ring has ``RING + ROWS`` slots per layer and a position's slot is ``position % ring_slots``
  (``V41DecodePool.position_rings``: the join places the carry by position), so the later rows' writes never land
  in an earlier row's window; each row's 128 window slots are a step input (``wslots``, the slots of its positions
  ``p - 127 .. p``) offset on device by its user's row base per layer;
* every write scatters by an absolute row (``ring<layer>``, ``comp<source>``, ``key<source>``: ``scatter_rows`` with
  stride 0 over the 32 index words, the sentinel on idle rows), the selection's pins and the attention's visible
  bounds come per row from its position, so a rejected row's writes sit past every committed pin and are rewritten
  when the position is really processed;
* a ratio-2 row pools with the previous row's projections of the same step (rows shifted by one, a block's first
  row with its slot's carry) and the step keeps every row's projections in a scratch; the next step's first ops
  scatter the committed row's into the slot's carry (``carry<source>`` index: the only state a rejection undoes);
* the index selection loops over the live rows (``range_scores`` with the row apart from the user; the candidate
  scorer with the row's pin and blocks);
* the Engram ids of row ``j`` come from the slot's history extended by the block's first ``j`` tokens.

The sampler takes every row's argmax; ``decode`` returns each slot's rows' argmaxes and leaves the slot where it
was; ``commit(slot, accepted)`` advances it (the drafts up to the first mismatch and the
bonus token are the step's output). A slot stepping with ``K = 0`` is a plain row of the same graph. One live slot
per rank at most (its rows are ``u .. u + K`` of its rank's tile, within the MoE's real rows), the captures keyed by
the live rows' layout.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand
from models.demos.deepseek_v3_d_p.tt.v41.attention import TOPK_ALIGN, TtV41Attention
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import ROWS, DeviceSampler, TtV41BatchDecoder
from models.demos.deepseek_v3_d_p.tt.v41.head_layout import o_heads, q_heads
from models.demos.deepseek_v3_d_p.tt.v41.indexer import DecodePins, TtV41Indexer
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import range_scores
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.tt.v41.pool import RING, scatter_rows
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_e4m3_qdq, fp8_qdq
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

SENTINEL32 = -1  # int32 host value of the uint32 sentinel


class TtV41VerifyDecoder(TtV41BatchDecoder):
    POOL_RING_SLOTS = RING + ROWS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.single_query, "verify rows need V41_DECODE_SINGLE_QUERY=1 (one scorer per live row)"
        assert self.packer is not None, "verify rows need V41_DECODE_HOST_INPUTS=packed"
        assert self.sample_on_device, "verify rows take every row's argmax on device"
        assert not self.c1, "verify rows run the 32-row MoE path (V41_DECODE_C1_MOE unset)"
        self.pool.position_rings = True
        self.ring_slots = self.pool.ring_slots
        self.rows: dict[int, list[int]] = {}  # slot -> its tile rows this step
        self.live: list[tuple[int, int]] = []  # (row, user) of the live rows this step, in row order
        self.histories: dict[int, list] = {}  # slot -> the Engram history after each of its block's tokens
        self.pending_carry: dict[int, int] = {}  # slot -> the scratch row to commit into its carry
        self.scratch = {
            src: tuple(self._replicated(torch.zeros(1, 1, ROWS, self.config.HEAD_DIM), ttnn.float32) for _ in range(2))
            for src in self.pool.carry
        }
        self.sentinel_topk = self._replicated(
            torch.full((1, 1, 1, self.config.INDEX_TOPK), SENTINEL32, dtype=torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self._windows: dict[int, ttnn.Tensor] = {}
        self._uoff: dict[int, ttnn.Tensor] = {}
        # with a seeder on the model: the drafter's tap of every tile row, [1, 1, ROWS, n_taps * hidden/tp] bf16 per
        # chip (the hc-mean block inputs of the tap layers, chip-major), computed inside the step and kept at a fixed
        # address per capture (``taps`` after a step: the drafter's ``stage(tap32=)`` input)
        self.taps = None
        self.capture_taps: dict[str, ttnn.Tensor] = {}

    # --- the step's layout ---------------------------------------------------------------------------------------
    def variant(self, tokens: dict) -> str:
        """The capture key: the live rows' layout (rank, user, rows) of the step."""
        parts = []
        for slot, block in sorted(tokens.items()):
            r, u = divmod(slot, self.users)
            parts.append(f"{r}.{u}.{len(block)}")
        return "v:" + ",".join(parts)

    def join(self, slot: int, tokens: torch.Tensor, state) -> None:
        super().join(slot, tokens, state)
        self.pending_carry.pop(slot, None)

    # --- host inputs ---------------------------------------------------------------------------------------------
    def _host_inputs(self, tokens: dict[int, list[int]], sampling: dict | None = None) -> dict:
        cfg, sp, users, pool = self.config, self.sp, self.users, self.pool
        tick = self._tick("in")
        ids = torch.zeros(sp, 1, ROWS, dtype=torch.int32)
        positions = torch.zeros(sp, ROWS, dtype=torch.int64)
        active = torch.zeros(sp, ROWS, dtype=torch.bool)
        user_of = torch.zeros(sp, ROWS, dtype=torch.int64)
        first = torch.zeros(sp, ROWS, dtype=torch.float32)
        carry_rows = {src: torch.full((sp, ROWS), SENTINEL32, dtype=torch.int32) for src in pool.carry}
        self.rows, self.live = {}, []
        taken = set()
        for slot in sorted(tokens):
            block = tokens[slot]
            r, u = divmod(slot, users)
            k = len(block) - 1
            assert 0 <= k and u + k < users, f"slot {slot}: {k} drafts need rows {u}..{u + k} within {users}"
            p = self.slots[slot].position
            rows = []
            for j, token in enumerate(block):
                row = u + j
                assert (r, row) not in taken, f"slot {slot}: row {row} of rank {r} already live"
                taken.add((r, row))
                ids[r, 0, row] = int(token)
                positions[r, row] = p + j
                active[r, row] = True
                user_of[r, row] = u
                first[r, row] = 1.0 if j == 0 else 0.0
                rows.append(row)
            self.rows[slot] = rows
            self.live += [(row, u) for row in rows]
            committed = self.pending_carry.pop(slot, None)
            if committed is not None:
                for src in carry_rows:
                    carry_rows[src][r, committed] = u  # scratch row ``committed`` -> the slot's carry row u
        tick("rows")
        ints = {"token": ids.view(sp, ROWS)}
        floats, rope, visible_cols = {}, {}, {}
        if self.sample_on_device:
            ints["k"] = torch.ones(sp, ROWS, dtype=torch.int32)
            floats["p"] = torch.zeros(sp, ROWS, dtype=torch.float32)
            floats["temp"] = torch.ones(sp, ROWS, dtype=torch.float32)
        rope["rope0"] = self.rope_tables.rows(False, positions)
        rope["ropec"] = self.rope_tables.rows(True, positions)
        rope["ropeg"] = self.rope_tables.rows(True, positions // 2 * 2)
        tick("rope")
        # the ring: a position's slot, as an absolute row of every layer's KV tensor (idle rows the sentinel)
        slot_of = positions % self.ring_slots
        for layer in self.model.layers:
            base = user_of * pool.slot_rows(layer) + pool.ring_row(layer)
            ints[f"ring{layer}"] = torch.where(active, base + slot_of, torch.full_like(base, SENTINEL32)).to(
                torch.int32
            )
        # the window: each row's 128 slots (positions p - 127 .. p), offset on device by the user's row base per layer
        wslots = (positions.view(sp, ROWS, 1) - (RING - 1) + torch.arange(RING).view(1, 1, RING)) % self.ring_slots
        wslots = torch.where(active.view(sp, ROWS, 1), wslots, torch.zeros_like(wslots))
        for src, slot in pool.slots.items():
            visible_cols[f"wu{src}"] = (user_of * slot.rows).float()
        for src in pool.carry:
            ints[f"carry{src}"] = carry_rows[src]
        visible_cols["first"] = first
        visible_cols["notfirst"] = 1.0 - first
        for src, slot in pool.slots.items():
            if src < 0:
                continue
            ratio = cfg.compress_ratio(src)
            visible = torch.where(active, (positions + 1) // ratio, torch.zeros_like(positions))
            newest = torch.where(visible > 0, visible - 1, torch.full_like(visible, SENTINEL32)).to(torch.int32)
            writes = active & (positions % ratio == ratio - 1)
            comp = torch.where(
                writes,
                user_of * slot.rows + pool.comp_row(src) + positions // ratio,
                torch.full_like(positions, SENTINEL32),
            )
            key = torch.where(
                writes, user_of * (slot.cap + 32) + positions // ratio, torch.full_like(positions, SENTINEL32)
            )
            ints[f"newest{ratio}"] = newest  # the same for every source of the ratio
            ints[f"comp{src}"], ints[f"key{src}"] = comp.to(torch.int32), key.to(torch.int32)
            visible_cols[f"uoff{src}"] = (user_of * slot.rows + pool.comp_row(src)).float()
        tick("addresses")
        inputs = self.packer.pack(ints, floats, rope, visible_cols)
        inputs["wslots"] = pool.per_rank(
            wslots.float().view(sp, 1, ROWS, RING), ttnn.float32, ttnn.TILE_LAYOUT, device=False
        )
        tick("pack")
        if self.model.engram:
            hash_ = self.model.engram_hash
            per_row = {}
            self.histories = {}
            for slot, block in tokens.items():
                h = self.slots[slot].engram_history.view(1, -1)
                self.histories[slot] = []
                for j, token in enumerate(block):
                    row_ids, h = hash_.step(torch.tensor([int(token)]), h)
                    per_row[(slot // users, self.rows[slot][j])] = row_ids[0]
                    self.histories[slot].append(h[0].clone())
            tick("engram_hash")
            inputs["engram"] = {}
            live = active.flatten().nonzero().flatten()
            for l, e in self.model.engram.items():
                ids_l = torch.zeros(sp * ROWS, hash_.n_hash_cols, dtype=torch.int64)
                for (r, row), row_ids in per_row.items():
                    ids_l[r * ROWS + row] = row_ids[hash_.layer_index(l)]
                inputs["engram"][l] = e.prepare_host(ids_l, sp * ROWS, live_rows=live)
            tick("engram_rows")
        return inputs

    # --- one step ------------------------------------------------------------------------------------------------
    def decode(self, tokens: dict[int, list[int]], sampling: dict | None = None) -> dict[int, list[int]]:
        """``tokens``: slot -> its block (the committed next token, then its drafts) -> slot -> the argmax of each
        of its rows. The slots do not advance: ``commit`` does."""
        for slot, block in tokens.items():
            user = self.slots[slot]
            assert user.position + len(block) <= self.pool.max_seq_len, f"slot {slot} at {user.position}: full"
        self.items = {}
        t0 = time.time()
        inputs = self._host_inputs(tokens, sampling)
        t1 = time.time()
        self._stage(inputs)
        t2 = time.time()
        variant = self.variant(tokens)
        self.last_variant = variant
        if not self.trace:
            mode = "eager"
        elif variant in self.captures:
            mode = "replay"
        elif self.eager_counts.get(variant, 0) >= self.warmup:
            mode = "capture"
        else:
            mode = "eager"
        self.last = {"mode": mode, "inputs_s": round(t1 - t0, 4), "stage_s": round(t2 - t1, 4), "variant": variant}
        if self.input_timing:
            self.last["items"] = dict(self.items)
        if mode == "eager":
            self.eager_counts[variant] = self.eager_counts.get(variant, 0) + 1
            logits = self._forward(self.on_block)
        else:
            if mode == "capture":
                self.captures[variant] = self._capture_step()
                self.capture_taps[variant] = self.taps
            capture = self.captures[variant]
            self.taps = self.capture_taps.get(variant)
            ttnn.synchronize_device(self.mesh_device)
            t3 = time.time()
            capture.controller.replay(blocking=False)
            t4 = time.time()
            ttnn.synchronize_device(self.mesh_device)
            t5 = time.time()
            logits = capture.logits
            self.last.update(sync_s=round(t3 - t2, 4), device_s=round(t5 - t3, 4), issue_s=round(t4 - t3, 4))
            if self.on_block is not None:
                for layer, x in capture.blocks:
                    self.on_block(layer, x)
        t6 = time.time()
        row = DeviceSampler.to_host(logits)
        out = {slot: [int(row[(slot // self.users) * ROWS + r]) for r in rows] for slot, rows in self.rows.items()}
        if mode == "eager":
            ttnn.deallocate(logits)
        self.last["read_s"] = round(time.time() - t6, 4)
        self.pool.step += 1
        return out

    def commit(self, slot: int, accepted: int) -> None:
        """After ``decode``: ``accepted`` of the slot's drafts matched (the rows ``0 .. accepted`` are committed):
        the slot advances by ``accepted + 1``, its Engram history follows, and the next step commits row
        ``accepted``'s ratio-2 projections as its carry."""
        rows = self.rows[slot]
        assert 0 <= accepted < len(rows), (accepted, rows)
        self.slots[slot].position += accepted + 1
        if self.model.engram:
            self.slots[slot].engram_history = self.histories[slot][accepted]
        self.pending_carry[slot] = rows[accepted]

    @staticmethod
    def accept(block: list[int], argmax: list[int]) -> tuple[int, list[int]]:
        """``spec_state``'s rule on a block ``[x, d_1 .. d_K]`` and its rows' argmaxes: the accepted count ``a`` (the
        drafts ``d_1 .. d_a`` equal the argmaxes of rows ``0 .. a - 1``) and the emitted tokens ``argmax[0 .. a]``
        (the accepted drafts and the bonus)."""
        a = 0
        while a < len(block) - 1 and block[a + 1] == argmax[a]:
            a += 1
        return a, argmax[: a + 1]

    # --- the graph -----------------------------------------------------------------------------------------------
    def _commit_carries(self) -> None:
        """The previous step's committed rows' projections into the slots' carry rows (``carry<source>``)."""
        for src, (k0, s0) in self.pool.carry.items():
            idx = self.inputs[f"carry{src}"]
            for scratch, dst in zip(self.scratch[src], (k0, s0)):
                src_rows = ttnn.to_layout(scratch, ttnn.ROW_MAJOR_LAYOUT)
                dst_rows = ttnn.to_layout(dst, ttnn.ROW_MAJOR_LAYOUT)
                scatter_rows(src_rows, dst_rows, idx, rows=ROWS)
                ttnn.copy(ttnn.to_layout(dst_rows, ttnn.TILE_LAYOUT), dst)
                ttnn.deallocate(src_rows)
                ttnn.deallocate(dst_rows)

    def _forward(self, on_block, c1: bool = False):
        assert not c1
        model = self.model
        self.inputs = dict(self.packer.unpack(self.buffers), engram=self.buffers.get("engram", {}))
        self.inputs["token"] = ttnn.reshape(self.inputs["token"], (1, 1, ROWS))
        self.inputs["wslots"] = self.buffers["wslots"]
        self._windows, self._uoff = {}, {}
        self._commit_carries()
        selection = {}
        h = model.embedding(self.inputs["token"])
        x = mhc_expand(ttnn.typecast(h, ttnn.float32), self.config.HC_MULT)
        pre = self.pre_mix
        taps = []
        for layer, block, moe in zip(model.layers, model.blocks, self.moes):
            if layer in model.engram:
                x = model.engram[layer](x, self.inputs["engram"][layer])
            if model.dspark is not None and layer in self.config.DSPARK_TARGET_LAYER_IDS:
                taps.append(model.dspark.tap(x))  # the block's input: the tap of every row
            x, pre = block.residual(
                x,
                pre,
                attention=lambda h, a=block.attn, n=block.attn_norm: self._attention(a, n(h), {}, selection),
                ffn=lambda h, m=moe, n=block.ffn_norm: self._ffn(m, n(h)),
            )
            if on_block is not None:
                on_block(layer, x)
        if taps:
            self.taps = ttnn.concat(taps, dim=3)
        final = ttnn.typecast(model.blocks[-1].residual.final_collapse(x, pre), ttnn.bfloat16)
        logits, _ = model.head(final, 0)
        return self.sampler(logits, self.inputs)

    def _window_rows(self, layer: int):
        """``sparse_sdpa`` window rows of the step ``[1, 1, ROWS, RING]`` uint32: each row's slots offset by its
        user's row base in ``layer``'s KV tensor."""
        if layer not in self._windows:
            pool = self.pool
            w = ttnn.add(self.inputs["wslots"], self.inputs[f"wu{pool.source_of(layer)}"])
            w = ttnn.add(w, float(pool.ring_row(layer)))
            self._windows[layer] = ttnn.to_layout(
                ttnn.typecast(ttnn.typecast(w, ttnn.int32), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT
            )
        return self._windows[layer]

    def _user_offsets(self, src: int):
        """``[1, 1, ROWS, 1]`` int32 tiled: each row's offset of its user's compressed row 0 in ``src``'s pool."""
        if src not in self._uoff:
            self._uoff[src] = ttnn.typecast(self.inputs[f"uoff{src}"], ttnn.int32)
        return self._uoff[src]

    def _attention(self, attn: TtV41Attention, x, masks: dict, selection: dict):
        layer, pool, cfg = attn.layer, self.pool, self.config
        cos, sin = self.inputs["ropec" if attn.ratio else "rope0"]
        xq = fp8_qdq(x)
        qr = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wq_a")), weight=attn.q_norm, epsilon=attn.eps)
        q = attn._dense(fp8_qdq(qr), "wq_b")
        kv = ttnn.rms_norm(attn.ccl.tp_all_reduce(attn._dense(xq, "wkv")), weight=attn.kv_norm, epsilon=attn.eps)
        kv_format = pool.kv_format if attn.ratio else MlaKvCacheFormat.BF16_RM
        reference_values = kv_format == MlaKvCacheFormat.BF16_RM
        kv = attn._rope(kv, cos, sin)
        kv_tensor = pool.kv_tensor(layer)
        rows = self._encode(fp8_qdq(kv) if reference_values else kv, kv_format)
        scatter_rows(rows, kv_tensor, self.inputs[f"ring{layer}"], rows=ROWS)
        ttnn.deallocate(rows)

        compressed = None
        if attn.ratio:
            ratio, src = attn.ratio, cfg.kv_source(layer)
            if attn.is_kv_source:
                latent = self._compress(attn.compressor, x, layer)
                c_rope = self.inputs["ropeg" if ratio == 2 else "ropec"]
                keys = attn.index_keys(latent, c_rope)
                comp_kv = attn._rope(latent, *c_rope)
                rows = self._encode(fp4_e4m3_qdq(comp_kv) if reference_values else comp_kv, kv_format)
                scatter_rows(rows, pool.kv[src], self.inputs[f"comp{src}"], rows=ROWS)
                ttnn.deallocate(rows)
                keys = ttnn.to_layout(keys, ttnn.ROW_MAJOR_LAYOUT)
                pool.write_index_keys(src, keys, self.inputs[f"key{src}"], rows=ROWS)
                ttnn.deallocate(keys)
            if attn.is_index_source:
                compressed, candidates = self._index(attn.indexer, x, qr, src, None, selection.get("candidates"))
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

    @staticmethod
    def _shift_rows(t):
        """``[1, 1, ROWS, W]`` tiled fp32 -> the rows shifted down by one (row i = row i - 1; row 0 = row 0)."""
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        w = rm.shape[3]
        head = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, 1, w])
        body = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, ROWS - 1, w])
        out = ttnn.to_layout(ttnn.concat([head, body], dim=2), ttnn.TILE_LAYOUT)
        for u in (rm, head, body):
            ttnn.deallocate(u)
        return out

    def _compress(self, compressor, x, layer: int):
        """As the batched step's, with a row pooling against the previous row's projections unless it is its
        block's first row (then its slot's carry), and this step's projections kept in the scratch for ``commit``."""
        if compressor.ratio == 1:
            kv = compressor._project(x, compressor.wkv, ttnn.bfloat16)
            return ttnn.rms_norm(kv, weight=compressor.norm, epsilon=compressor.eps)
        xf = ttnn.typecast(x, ttnn.float32)
        kv = compressor._project(xf, compressor.wkv, ttnn.float32)
        score = compressor._project(xf, compressor.wgate, ttnn.float32)
        k0, s0 = self.pool.carry[layer]
        first, notfirst = self.inputs["first"], self.inputs["notfirst"]
        k_prev = ttnn.add(ttnn.multiply(k0, first), ttnn.multiply(self._shift_rows(kv), notfirst))
        s_prev = ttnn.add(ttnn.multiply(s0, first), ttnn.multiply(self._shift_rows(score), notfirst))
        m = ttnn.maximum(s_prev, score)
        e0, e1 = ttnn.exp(ttnn.subtract(s_prev, m)), ttnn.exp(ttnn.subtract(score, m))
        pooled = ttnn.divide(ttnn.add(ttnn.multiply(k_prev, e0), ttnn.multiply(kv, e1)), ttnn.add(e0, e1))
        scratch_k, scratch_s = self.scratch[layer]
        ttnn.copy(kv, scratch_k)
        ttnn.copy(score, scratch_s)
        return ttnn.rms_norm(ttnn.typecast(pooled, ttnn.bfloat16), weight=compressor.norm, epsilon=compressor.eps)

    def _index(self, indexer: TtV41Indexer, x, qr, source: int, mask, candidates: dict | None):
        """The indexer of every live row over its user's keys (the candidate blocks keyed by tile row)."""
        cfg, pool = self.config, self.pool
        cap = pool.slots[source].cap
        pins = DecodePins(self.inputs[f"newest{cfg.compress_ratio(source)}"])
        cos, sin = self.inputs["ropec"]
        q = ttnn.linear(fp8_qdq(qr), indexer.wq_b, compute_kernel_config=indexer.compute_kernel_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=indexer.heads, num_kv_heads=0, transpose_k_heads=False
        )
        q = indexer.qdq(indexer.rope(q, cos, sin))
        w = indexer.ccl.tp_all_reduce(ttnn.linear(x, indexer.weights_proj))
        k = pool.index_k[source]
        published = {} if indexer.is_candidate_source else None
        blocks_of = pool.index_k_blocks.get(source)
        in_blocks = (
            blocks_of is not None and candidates is not None and any(c.ids is not None for c in candidates.values())
        )
        q_rm, w_rm = indexer.decode_query(q, w)
        parts = {}
        for row, u in self.live:
            mine = candidates[row] if candidates is not None else None
            pin_r = ttnn.slice(pins.newest, [0, 0, 0, row], [1, 1, 1, row + 1])
            if in_blocks and mine.ids is not None:
                base = u * (cap + 32) // cfg.CANDIDATE_BLOCK_SIZE
                parts[row] = indexer.select_user_in_blocks(
                    q_rm, w_rm, blocks_of, pin_r, mine.ids, row, cap, base, cfg.INDEX_TOPK
                )
                continue
            score = range_scores(q_rm, w_rm, k, pins.newest, u, cap + 32, row=row)
            topk, blocks = indexer.select_split(score, pin_r, cap + 32, cfg.INDEX_TOPK, mine)
            if published is not None:
                published[row] = blocks
            parts[row] = topk
        rows = [parts.get(i, self.sentinel_topk) for i in range(ROWS)]
        return ttnn.concat(rows, dim=2), published

    def _index_rows(self, layer: int, compressed):
        window = self._window_rows(layer)
        if compressed is None:
            return window
        src = self.config.kv_source(layer)
        comp = ttnn.typecast(ttnn.to_layout(compressed, ttnn.TILE_LAYOUT), ttnn.int32)
        comp = ttnn.where(ttnn.eq(comp, -1), -1, ttnn.add(comp, self._user_offsets(src)))
        comp = ttnn.to_layout(ttnn.typecast(comp, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        rows = ttnn.concat([window, comp], dim=-1)
        assert rows.shape[3] % TOPK_ALIGN == 0, tuple(rows.shape)
        return rows

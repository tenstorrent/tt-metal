# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The KV-cached 26-layer Voxtral-4B-TTS-2603 text decoder, batched at B=32.

    ids -> token_embed -> [layer] x 26 -> final RMSNorm

`token_embed`, `mistral_rotary_embedding` and `mistral_r_m_s_norm` (the final norm) come from
`tt/modules/`, and every decoder layer is the fused `tt/modules/layer.py` block (input norm,
attention, post-attention norm and SwiGLU MLP in one callable). `blocks` is a plain list of
`TextBlock`, one per layer, and any cap of 1..26 layers builds.

NUMERICS
float32 ACTIVATIONS with bfloat16 / bfloat8_b weights. All-bfloat16 activations put this 26-layer
stack at PCC 0.9798 -- about 0.4% per residual add over 52 of them. The PREFILL narrows q/k/v to
bfloat16 because SDPA takes nothing wider (`sdpa_device_operation.cpp:43`); the DECODE branch does
not call SDPA at all and stays float32 end to end.

THE KV CACHE
Prefill runs causal SDPA and SEEDS each layer's cache from its own post-RoPE k/v -- the cache is
that tensor with a zero tail concatenated onto the sequence axis, so there is no copy and no second
source of truth. `decode_step` then appends ONE slot with `paged_update_cache` and attends over the
resident history with an EXPLICIT masked matmul/softmax/matmul rather than
`scaled_dot_product_attention_decode`, whose output measured systematically short of the
reference's (norm ratio 0.948 at PCC 0.9999). The zero tail past the current position is masked from
a staged `[C, C]` table; a zero key scores ZERO, which is an ordinary logit rather than a negligible
one.

THE SHARED PROMPT PREFIX
Every row of a speech request carries the same voice, so the prompt's first N + 3 tokens are the
same on every row. `prefill_voiced` runs that prefix ONCE at batch 1 and only the per-row tail at
the full batch (`_stage_prefix`, `_prefill_split`).

This module NEVER opens a device.
"""
from __future__ import annotations

import torch

import ttnn
from models.demos.voxtral_4b_tts_2603.tt import common
from models.demos.voxtral_4b_tts_2603.tt.modules import layer, mistral_r_m_s_norm, mistral_rotary_embedding, token_embed

# Default resident capacity C for the sequence axis of the KV cache: the prefill's 64 plus 64
# decode slots. `kv_capacity=` overrides it.
DEFAULT_PREFILL_CAPACITY = 64
DEFAULT_DECODE_HEADROOM = 64


def _tile_ceil(value: int) -> int:
    return -(-int(value) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


class TextBlock:
    """ONE decoder layer: the fused `layer` callable plus the KV cache it owns.

    `kv` is `{"k": ..., "v": ..., "capacity": C, "filled": n, ...}`, seeded by this block's own
    prefill and read and appended to by its own decode step.
    """

    def __init__(self, index: int, run):
        self.index = index
        self.run = run
        self.kv = None

    def __call__(self, hidden_states, position_embeddings=None, position=None, decode=False, trim=None):
        """`trim`, when given, maps the post-attention residual `[1, 1, R, H]` to the rows the caller
        reads (or None for none): the FFN then runs on those rows only, the way a decode step does."""
        return self.run(
            hidden_states,
            position_embeddings=position_embeddings,
            kv_cache=self.kv,
            position=position,
            decode=decode,
            trim=trim,
        )

    def __repr__(self) -> str:
        return f"TextBlock(index={self.index})"


class TextStack:
    """embed -> rope -> N blocks -> final RMSNorm, with a resident per-layer KV cache."""

    def __init__(self, device, token_embed, rotary, blocks, final_norm, hidden_size, kv_capacity):
        self.device = device
        self.token_embed = token_embed
        self.rotary = rotary
        self.blocks = blocks
        self.final_norm = final_norm
        self.hidden_size = int(hidden_size)
        self.n_layers = len(blocks)
        self.kv_capacity = int(kv_capacity)
        self.act_dtype = ttnn.float32
        self.filled = 0
        # A split (shared-prefix) prefill leaves the cache with a `gap` of dead slots between the
        # prefix and the tail; decode then writes position p at slot p + gap through this table.
        self._slot_gap = 0
        self._slot_mask = None
        # The batch the stack is built and validated for (the program configs are sized for it).
        self.max_batch = int(common.DEFAULT_BATCH)
        # THE DECODE ATTENTION MASK, staged once. Row `p` is 0 through column `p` and a large
        # negative after it: the KV cache is allocated to the full capacity and its tail is zeros,
        # and a zero key scores ZERO, which is an ordinary logit rather than a negligible one. The
        # table is built HERE, at construction, because building it per step would put a torch
        # call in the forward and a host write inside a captured trace. `[C, C]` float32 at C=128
        # is 64 KB, shared by reference across all 26 blocks.
        rows = torch.arange(self.kv_capacity).reshape(-1, 1)
        cols = torch.arange(self.kv_capacity).reshape(1, -1)
        self._decode_mask = ttnn.from_torch(
            torch.where(cols <= rows, 0.0, -1e9).to(torch.float32).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    # ---- pieces ------------------------------------------------------------------------

    def embed(self, input_ids_tt):
        """`[B, S]` uint32 ROW_MAJOR -> `[B, 1, S, 3072]` float32."""
        table_out = self.token_embed(input_ids_tt)
        batch, seq = int(table_out.shape[0]), int(table_out.shape[-2])
        wide = ttnn.typecast(table_out, self.act_dtype)
        ttnn.deallocate(table_out)
        return ttnn.reshape(wide, [batch, 1, seq, self.hidden_size])

    def rope(self, position_ids_tt):
        """`[B, S]` uint32 ROW_MAJOR -> `(cos, sin)`, each `[B, 1, S, head_dim]`."""
        cos, sin = self.rotary(None, position_ids=position_ids_tt)
        return tuple(ttnn.reshape(t, [int(t.shape[0]), 1, int(t.shape[-2]), int(t.shape[-1])]) for t in (cos, sin))

    # ---- prefill -----------------------------------------------------------------------

    def stage_voice(self, audio_mask, voice_embedding, input_ids):
        """The speaker's voice as two PERSISTENT device constants, built ONCE, outside the forward.

        Voxtral TTS conditions on a speaker embedding carried in the prompt: a contiguous block of
        `[AUDIO]` placeholder ids whose EMBEDDINGS are replaced by that speaker's. This is input
        ENCODING -- the same category as tokenizing -- so it happens here, on the host, before the
        forward runs, and the forward then does the substitution with two device ops:

            voiced = embeds * keep + placed

        `keep` is 1 where the prompt keeps its token embedding and 0 under the placeholders;
        `placed` holds the voice rows under the placeholders and 0 elsewhere. Both are
        `[1, 1, S_pad, 3072]` float32 and broadcast over the batch, which is legal because every
        row carries the placeholder block at the same positions (`common.build_voice_prompt` lays it
        out right after `[BOS] [BEGIN_AUDIO]`). x*1+0 and x*0+v are exact, so this is bit-for-bit the
        reference's `inputs_embeds[mask] = voice`.
        """
        import torch

        if voice_embedding.shape[-1] != self.hidden_size:
            raise ValueError(
                f"voice embedding is {tuple(voice_embedding.shape)}; last dim must be the hidden "
                f"size {self.hidden_size}"
            )
        if not bool((audio_mask == audio_mask[:1]).all()):
            raise ValueError("every row must carry the [AUDIO] placeholders at the same positions")
        row = audio_mask[0]
        slots = int(row.sum())
        if slots != int(voice_embedding.shape[0]):
            raise ValueError(
                f"the prompt has {slots} [AUDIO] placeholders but the voice embedding has "
                f"{voice_embedding.shape[0]} rows"
            )
        seq = int(row.shape[0])
        padded = _tile_ceil(seq)
        keep = torch.ones(1, 1, padded, self.hidden_size, dtype=torch.float32)
        placed = torch.zeros(1, 1, padded, self.hidden_size, dtype=torch.float32)
        where = torch.zeros(padded, dtype=torch.bool)
        where[:seq] = row
        keep[0, 0, where] = 0.0
        placed[0, 0, where] = voice_embedding.to(torch.float32)
        upload = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        staged = {
            "keep": ttnn.from_torch(keep, **upload),
            "placed": ttnn.from_torch(placed, **upload),
            "seq": seq,
            "slots": slots,
            # The shared-prefix layout below is measured on THESE ids; `run_text_to_speech` refuses others.
            "ids": input_ids.clone(),
        }
        staged.update(self._stage_prefix(input_ids, keep, placed))
        return staged

    def _stage_prefix(self, input_ids, keep, placed):
        """The SHARED PROMPT PREFIX, measured on the host ids, and everything its split prefill reads.

        Every row carries the same voice, so `[BOS] [BEGIN_AUDIO] [AUDIO]*N [NEXT_AUDIO_TEXT]` is the
        same N + 3 tokens on all of them, and a causal stack computes the same hidden state at those
        positions for every row. `prefill_voiced` then runs row 0's first P = tile_ceil(t0) positions
        ONCE, at batch 1, and only the T-row TAIL -- positions [t0, R), exactly the positions the
        rows disagree on -- for every row, packed compact as one `[1, 1, B * T, dim]` sequence. For
        casual_male's 170-token prompt (N = 147) that is 20 rows per user instead of 192.

        The tail attends to the prefix's k/v through `mask`: prefix columns before t0 open, the
        prefix's own rows past t0 (row 0's tokens, not this row's) closed, the tail's columns causal.
        The KV cache is the prefix k/v followed by the tail's, so position p >= t0 lives at slot
        p + gap (gap = P - t0); `slot_mask` is the decode table for that layout, row p opening
        [0, t0) and [P, p + gap]. Nothing is staged when the rows share nothing a split can use.
        """
        if input_ids is None or int(input_ids.shape[0]) < 2:
            return {}
        ids = input_ids
        tile = ttnn.TILE_SIZE
        batch = int(ids.shape[0])
        real = int(ids.shape[-1])
        start = int((ids == ids[:1]).all(dim=0).long().cumprod(0).sum())
        if real - start < 2:
            # A TAIL OF AT LEAST TWO TOKENS. With every row identical -- what the demo builds from a single
            # `--text`, repeated to the batch -- there is no per-row tail at all, and the whole-prompt
            # prefill is not an option: at batch 32 a long prompt's fused SwiGLU circular buffers plus its
            # live input exceed L1. So the last positions become the tail. Two, not one: a one-token tail
            # (exactly one 32-row tile of compact rows) came back wrong in the last rows of the tile at the
            # prompt's final position (PCC ~0.45 for rows 29-31 against the reference), while two- and
            # four-token tails are exact (>= 0.9996 on every row).
            start = real - 2
        tail = real - start
        tail_slots = _tile_ceil(tail)
        rows = _tile_ceil(start)
        if start < tile or tail < 1 or (batch * tail) % tile:
            return {}
        if rows + tail_slots > self.kv_capacity - 1:
            # Say so instead of falling back: the whole-prompt prefill this would take does not fit L1 at
            # batch 32 for any prompt long enough to reach here.
            raise ValueError(
                f"KV capacity {self.kv_capacity} is too small for this {real}-token prompt's shared-prefix "
                f"layout ({rows} prefix + {tail_slots} tail slots); build with "
                "kv_capacity=pipeline.tts_kv_capacity(prompt_len, max_frames)"
            )
        gap = rows - start
        # The tail runs COMPACT: `[1, 1, batch * tail, dim]`, sample b's positions [t0, R) in rows
        # b * tail.., so no row pads its tail out to a tile and nothing recomputes shared positions.
        # Row (b, i) sees prefix columns before t0 and its own sample's tail columns up to i.
        sample = torch.arange(batch * tail) // tail
        step = torch.arange(batch * tail) % tail
        cols = torch.arange(rows + batch * tail).reshape(1, -1)
        own = (cols >= rows) & (((cols - rows) // tail) == sample.reshape(-1, 1))
        own = own & (((cols - rows) % tail) <= step.reshape(-1, 1))
        open_ = (cols < start) | own
        mask = torch.where(open_, 0.0, float("-inf")).reshape(1, 1, batch * tail, rows + batch * tail)
        cap = self.kv_capacity
        pos = torch.arange(cap).reshape(-1, 1)
        slot = torch.arange(cap).reshape(1, -1)
        slot_mask = torch.where((slot < start) | ((slot >= rows) & (slot <= pos + gap)), 0.0, -1e9)

        class _Rows:
            shape = (1, real, 1)

        cos, sin = (ttnn.to_torch(t).float().reshape(-1, int(t.shape[-1])) for t in self.rotary(_Rows()))
        bf16 = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        fp32 = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)

        def _compact(t):
            return t[:, :, start:real].repeat(1, 1, batch, 1).contiguous()

        return {
            "prefix": {
                "real": real,
                "rows": rows,
                "start": start,
                "tail": tail,
                "tail_slots": tail_slots,
                "batch": batch,
                "gap": gap,
                "keep_head": ttnn.from_torch(keep[:, :, :rows].contiguous(), **fp32),
                "placed_head": ttnn.from_torch(placed[:, :, :rows].contiguous(), **fp32),
                "keep_tail": ttnn.from_torch(_compact(keep), **fp32),
                "placed_tail": ttnn.from_torch(_compact(placed), **fp32),
                "rope_tail": tuple(
                    ttnn.from_torch(
                        t[start:real].repeat(batch, 1).reshape(1, 1, batch * tail, -1).to(torch.bfloat16).contiguous(),
                        **bf16,
                    )
                    for t in (cos, sin)
                ),
                "mask": ttnn.from_torch(mask.to(torch.bfloat16).contiguous(), **bf16),
                "slot_mask": ttnn.from_torch(
                    slot_mask.to(torch.float32).contiguous(),
                    dtype=ttnn.float32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                ),
            }
        }

    def prefill_voiced(self, input_ids_tt, voice, position_ids_tt=None, real_len=None, need_hidden=True):
        """Prefill with the speaker's voice substituted into the prompt's `[AUDIO]` rows, ON DEVICE.

        Substitution, not concatenation: the ids and therefore the positions and the causal mask
        are the prompt's own, and only the rows under the placeholders change. `voice` is what
        `stage_voice` returned; the forward reads only those resident buffers.
        `real_len` is the real prompt length when the caller already
        padded the ids (a trace pinned at a fixed capacity); it keeps `last` on the real prompt.
        """
        seq = int(input_ids_tt.shape[-1])
        if seq != int(voice["seq"]):
            raise ValueError(f"prompt length {seq} != the staged voice layout's {voice['seq']}")
        real_len = seq if real_len is None else int(real_len)
        split = voice.get("prefix")
        if split is not None and position_ids_tt is None and split["real"] == real_len:
            return self._prefill_split(input_ids_tt, split, need_hidden)
        padded = _tile_ceil(seq)
        if padded != seq:
            # The same tile-aligned tail `prefill` adds: causal attention hides it and `real_len`
            # keeps `last` and the first decode write on the real prompt.
            input_ids_tt = ttnn.pad(input_ids_tt, [(0, 0), (0, padded - seq)], value=0)
        embeds = self.embed(input_ids_tt)
        kept = ttnn.multiply(embeds, voice["keep"])
        ttnn.deallocate(embeds)
        voiced = ttnn.add(kept, voice["placed"])
        ttnn.deallocate(kept)
        try:
            return self.prefill_embeds(voiced, position_ids_tt=position_ids_tt, real_len=real_len)
        finally:
            ttnn.deallocate(voiced)

    def prefill(self, input_ids_tt, position_ids_tt=None):
        """ids -> `(hidden [B, 1, S, 3072], last_hidden [B, 3072])`. Seeds the KV cache.

        A REAL PROMPT IS NOT A TILE MULTIPLE. This model's prompt is `[bos] + text +
        [BEGIN_AUDIO]`, which for the package's own 32-token texts is 33 -- and the KV cache
        appends on the sequence axis, so it needs a tile-aligned prefill. The zero tail is added
        HERE, to the ids, where the tensor is ROW_MAJOR and the pad is a plain row extension
        rather than a concat across a tile boundary at row 33.

        The tail is invisible to the answer on both sides of the cache: prefill attention is
        `is_causal=True`, so no real position can see a later padded one, and decode reads
        `[0, position]` off `cur_pos` starting at `real_len`, so the slots the tail filled are
        either overwritten by the first decode step or never read.
        """
        batch, real_len = int(input_ids_tt.shape[0]), int(input_ids_tt.shape[-1])
        padded = _tile_ceil(real_len)
        if padded != real_len:
            tail = padded - real_len
            input_ids_tt = ttnn.pad(input_ids_tt, [(0, 0), (0, tail)], value=0)
            if position_ids_tt is not None:
                # The tail's positions are never read, so any in-range value does; 0 avoids
                # running the rotary table past the capacity it was staged for.
                position_ids_tt = ttnn.pad(position_ids_tt, [(0, 0), (0, tail)], value=0)
        embeds = self.embed(input_ids_tt)
        try:
            return self.prefill_embeds(embeds, position_ids_tt=position_ids_tt, real_len=real_len)
        finally:
            ttnn.deallocate(embeds)

    def prefill_embeds(self, embeds, position_ids_tt=None, real_len=None):
        """`[B, 1, S, 3072]` -> `(hidden [B, 1, real, 3072], last_hidden [B, 3072])`.

        The TTS decode feeds audio-token EMBEDDINGS rather than ids, so the embedding table is not
        always the front of this stack; both entry points run the identical block chain.

        `real_len` is the prompt length BEFORE `prefill` padded it to a tile multiple; `S` here is
        the padded length. It drives three things that must follow the real prompt and not the
        pad: which row is `last`, where the first decode step writes, and how much hidden state
        comes back. `None` means the caller's sequence is already its own real length.
        """
        batch, seq = int(embeds.shape[0]), int(embeds.shape[-2])
        real = seq if real_len is None else int(real_len)
        if not 0 < real <= seq:
            raise ValueError(f"real_len {real} must lie in [1, {seq}]")
        if batch > self.max_batch:
            raise ValueError(f"batch {batch} exceeds the {self.max_batch} rows the stack is built for")
        self._arm_cache(batch, seq)
        # The rope pair is NOT deallocated by hand: the rotary module's default branch returns a
        # `ttnn.slice` of its own build-time table, and freeing a view of that would take the
        # table with it.
        rope = self._prefill_rope(embeds, position_ids_tt)
        out = self._run_chain(embeds, rope)
        self._slot_gap = 0
        self._slot_mask = None
        # The REAL length, so the first decode step writes slot `real` and its masked `[0, real]`
        # window covers the prompt and nothing the pad wrote.
        self.filled = real
        last = self._last_row(out, real)
        if real != seq:
            # NOT deallocated: `ttnn.slice` hands back a metadata VIEW whenever it can, and
            # `last` was just sliced out of this same buffer -- freeing it here would take
            # `last` with it.
            out = ttnn.slice(out, [0, 0, 0, 0], [batch, 1, real, self.hidden_size])
        return out, last

    def _run_chain(self, embeds, rope, trim=None):
        """Every block over `embeds`, then the final norm. `embeds` itself is never freed.

        `trim` goes to the LAST block only (see `TextBlock.__call__`): its FFN and the final norm run
        on the rows `trim` keeps, and a `trim` that keeps none returns None without either."""
        hidden = embeds
        last = len(self.blocks) - 1
        try:
            for i, block in enumerate(self.blocks):
                nxt = block(hidden, position_embeddings=rope, decode=False, trim=trim if i == last else None)
                if hidden is not embeds:
                    ttnn.deallocate(hidden)
                hidden = nxt
            return None if hidden is None else self.final_norm(hidden)
        finally:
            if hidden is not None and hidden is not embeds:
                ttnn.deallocate(hidden)

    def _last_row(self, out, real):
        """Row `real - 1` of every sample as `[B, H]`, via the tile-aligned 32-row block holding it:
        slicing one row straight out of the whole tiled `[B, 1, S, H]` untilizes all of it."""
        batch = int(out.shape[0])
        top = (real - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        block = ttnn.slice(out, [0, 0, top, 0], [batch, 1, top + ttnn.TILE_SIZE, self.hidden_size])
        return ttnn.reshape(
            ttnn.slice(block, [0, 0, real - 1 - top, 0], [batch, 1, real - top, self.hidden_size]),
            [batch, self.hidden_size],
        )

    def _last_tail_rows(self, batch, tail):
        """A `trim` for the compact tail `[1, 1, batch * tail, H]` (sample-major): each sample's last
        row, as `[1, 1, batch, H]`. The rows are `tail` apart and off-tile, so they are picked ROW_MAJOR."""
        hidden = self.hidden_size

        def trim(h):
            rm = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
            per_sample = ttnn.reshape(rm, [batch, 1, tail, hidden])
            rows = ttnn.slice(per_sample, [0, 0, tail - 1, 0], [batch, 1, tail, hidden])
            ttnn.deallocate(rm)
            return ttnn.to_layout(ttnn.reshape(rows, [1, 1, batch, hidden]), ttnn.TILE_LAYOUT)

        return trim

    def _voiced(self, ids, keep, placed):
        embeds = self.embed(ids)
        kept = ttnn.multiply(embeds, keep)
        ttnn.deallocate(embeds)
        voiced = ttnn.add(kept, placed)
        ttnn.deallocate(kept)
        return voiced

    def _prefill_split(self, input_ids_tt, split, need_hidden):
        """The voiced prefill with the shared prefix run ONCE -- see `_stage_prefix` for the layout.

        Every attention layer is called twice: "stash" (row 0's P prefix rows at batch 1) keeps its
        post-RoPE k/v, and "extend" (the T tail rows at the full batch) puts that k/v in front of
        its own for every row and seeds the cache with the P + T slots.
        """
        batch = int(input_ids_tt.shape[0])
        rows, start, tail, real = split["rows"], split["start"], split["tail"], split["real"]
        if batch > self.max_batch:
            raise ValueError(f"batch {batch} exceeds the {self.max_batch} rows the stack is built for")
        if batch != split["batch"]:
            raise ValueError(f"batch {batch} != the staged compact tail's {split['batch']}")
        self._arm_cache(batch, rows + split["tail_slots"])
        # The prefix runs row 0's first `rows` = tile_ceil(start) positions. When the shared prefix ends
        # just past a tile edge, `rows` can exceed the prompt itself (7 of the 20 presets with the
        # package's 18-token texts, e.g. ar_male: start 70 -> rows 96 > real 90), so the ids are padded
        # out to `rows`. Positions past `start` are masked from the tail and from decode either way.
        width = int(input_ids_tt.shape[-1])
        pre_ids = ttnn.slice(input_ids_tt, [0, 0], [1, min(rows, width)])
        if rows > width:
            pre_ids = ttnn.pad(pre_ids, [(0, 0), (0, rows - width)], value=0)
        pre_in = self._voiced(pre_ids, split["keep_head"], split["placed_head"])
        tail_ids = ttnn.reshape(ttnn.slice(input_ids_tt, [0, start], [batch, real]), [1, batch * tail])
        tail_in = self._voiced(tail_ids, split["keep_tail"], split["placed_tail"])
        for block in self.blocks:
            block.kv["prefix_phase"] = "stash"
        # With no hidden state asked for, the prefix's last block is read only for its k/v (its
        # attention stashes them), and the tail's only for each sample's last row.
        pre_trim = None if need_hidden else (lambda h: None)
        tail_trim = None if need_hidden else self._last_tail_rows(batch, tail)
        try:
            pre_out = self._run_chain(pre_in, self._prefill_rope(pre_in, None), trim=pre_trim)
            for block in self.blocks:
                block.kv["prefix_phase"] = "extend"
                block.kv["prefix_mask"] = split["mask"]
                block.kv["prefix_compact"] = (batch, tail, split["tail_slots"])
            tail_out = self._run_chain(tail_in, split["rope_tail"], trim=tail_trim)
        finally:
            for block in self.blocks:
                block.kv.pop("prefix_phase", None)
                block.kv.pop("prefix_mask", None)
                block.kv.pop("prefix_kv", None)
                block.kv.pop("prefix_compact", None)
                block.kv["slot_offset"] = split["gap"]
            ttnn.deallocate(pre_in)
            ttnn.deallocate(tail_in)
        self.filled = real
        self._slot_gap = split["gap"]
        self._slot_mask = split["slot_mask"]
        if not need_hidden:
            return None, ttnn.reshape(tail_out, [batch, self.hidden_size])
        # Compact rows back to `[B, 1, tail, H]`, sample-major, for `last` and the hidden state.
        per_sample = ttnn.reshape(ttnn.to_layout(tail_out, ttnn.ROW_MAJOR_LAYOUT), [batch, 1, tail, self.hidden_size])
        ttnn.deallocate(tail_out)
        last = ttnn.to_layout(
            ttnn.reshape(
                ttnn.slice(per_sample, [0, 0, tail - 1, 0], [batch, 1, tail, self.hidden_size]),
                [batch, self.hidden_size],
            ),
            ttnn.TILE_LAYOUT,
        )
        # The whole prompt's hidden state, positions [0, R): row 0's first t0 prefix rows for every
        # sample, then each sample's own tail. The seam is off-tile, so it is joined ROW_MAJOR.
        head = ttnn.slice(ttnn.to_layout(pre_out, ttnn.ROW_MAJOR_LAYOUT), [0, 0, 0, 0], [1, 1, start, self.hidden_size])
        ttnn.deallocate(pre_out)
        out = ttnn.concat([ttnn.repeat(head, ttnn.Shape([batch, 1, 1, 1])), per_sample], dim=2)
        return ttnn.to_layout(out, ttnn.TILE_LAYOUT), last

    def _prefill_rope(self, embeds, position_ids_tt):
        if position_ids_tt is not None:
            return self.rope(position_ids_tt)
        # The rotary module's default: contiguous positions, which are the first rows of its table.
        # Its leading dim is 1, which broadcasts across the batch in the RoPE multiply.
        cos, sin = self.rotary(embeds)
        # Every layer's fused RoPE wants the table in bf16 and re-reads it for each (batch, head)
        # tile: narrow it ONCE here rather than once per layer, and keep the result in L1.
        return tuple(
            ttnn.typecast(
                ttnn.reshape(t, [1, 1, int(t.shape[-2]), int(t.shape[-1])]),
                ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            for t in (cos, sin)
        )

    # ---- decode ------------------------------------------------------------------------

    def decode_step(self, embeds, position):
        """ONE token per user: `[B, 1, 1, 3072]` -> `[B, 3072]`, from the RESIDENT cache.

        Nothing recomputes the prefix. Each layer appends this token's k/v to its own cache slot
        `position` and attends over positions `[0, position]` of it (an explicit masked
        matmul/softmax/matmul, not SDPA-decode -- see the module docstring).
        """
        position = int(position)
        batch = int(embeds.shape[0])
        if self.blocks and self.blocks[0].kv is None:
            raise RuntimeError("decode_step needs a seeded KV cache: run prefill first")
        if position + self._slot_gap >= self.kv_capacity:
            raise ValueError(
                f"position {position} (slot {position + self._slot_gap}) exceeds the KV capacity {self.kv_capacity}"
            )
        # FOLD ONCE, at the stack entry. `[B, 1, 1, H]` is a lie about its size in TILE layout --
        # the middle dims pad 1 -> 32, so every residual add and norm would touch 32x the data the
        # step carries. Everything between here and the exit is elementwise or reduces over the
        # last dim, so neither cares which leading axis holds the batch.
        folded = ttnn.reshape(embeds, [1, 1, batch, self.hidden_size])
        rope = self._decode_rope(batch, position)
        # The additive mask row for `position` is the same for every layer: cut it off the staged
        # table ONCE per step and hand it to each block, instead of 26 slice + tilize pairs.
        cap = int(self.kv_capacity)
        table = self._decode_mask if self._slot_mask is None else self._slot_mask
        mask_row = ttnn.to_layout(
            ttnn.reshape(ttnn.slice(table, [position, 0], [position + 1, cap]), [1, 1, 1, cap]),
            ttnn.TILE_LAYOUT,
        )
        # `rotate_half(x) * sin == cat(x2, x1) * cat(-sin1, sin2)`: the sign rides on a sin table
        # built once per step, so no layer negates half its q and k.
        sin = rope[1]
        half = int(sin.shape[-1]) // 2
        width = int(sin.shape[-1])
        sin_signed = ttnn.concat(
            [
                ttnn.neg(ttnn.slice(sin, [0, 0, 0, 0], [1, 1, 1, half])),
                ttnn.slice(sin, [0, 0, 0, half], [1, 1, 1, width]),
            ],
            dim=-1,
        )
        for block in self.blocks:
            block.kv["mask_row"] = (position, mask_row)
            block.kv["rope_signed"] = (position, sin_signed)
        hidden = folded
        try:
            for block in self.blocks:
                nxt = block(hidden, position_embeddings=rope, position=position, decode=True)
                if hidden is not folded:
                    ttnn.deallocate(hidden)
                hidden = nxt
            out = self.final_norm(hidden)
        finally:
            if hidden is not folded:
                ttnn.deallocate(hidden)
        self.filled = max(self.filled, position + 1)
        return ttnn.reshape(out, [batch, self.hidden_size])

    def _decode_rope(self, batch, position):
        """`(cos, sin)` for `position`, each `[1, 1, 1, head_dim]` float32, shared by every user.

        Routed through the SAME rotary module the prefill uses -- the position index is selected on
        device out of the staged table, so this makes no host call.
        """
        # NOTHING HERE IS DEALLOCATED BY HAND. `ttnn.slice` and `ttnn.reshape` hand back a
        # metadata VIEW whenever they can, and freeing a view frees the buffer it aliases -- here
        # that buffer would be the rotary module's own cos/sin table, which every later step reads.
        # ONE ROW, NOT A GATHER. Every user in a step shares one position, so the rotary module's
        # single-position branch hands back `[1, 1, head_dim]` float32 straight off its table.
        # The `position_ids` branch would gather instead, and `ttnn.embedding` requires a
        # bfloat16 table -- which put RoPE at bfloat16 for all 26 layers of every decode step
        # while the prefill through the same weights ran it at float32.
        cos, sin = self.rotary(None, position=position)
        # `[1, 1, 1, head_dim]` float32, INTERLEAVED. One position is shared by every user and
        # every head, so the pair broadcasts across the `[B, n_kv, heads, head_dim]` q/k the
        # layer builds -- there is nothing to replicate and nothing to shard.
        return tuple(
            ttnn.to_layout(
                ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT), [1, 1, 1, int(t.shape[-1])]),
                ttnn.TILE_LAYOUT,
            )
            for t in (cos, sin)
        )

    # ---- the cache ---------------------------------------------------------------------

    def _arm_cache(self, batch, seq):
        """Hand every block an empty slot for the prefill to seed, sized to the pinned capacity."""
        capacity = self.kv_capacity
        if capacity <= seq:
            raise ValueError(f"kv_capacity {capacity} leaves no decode room after a prefill of {seq}")
        if seq % ttnn.TILE_SIZE or capacity % ttnn.TILE_SIZE:
            raise ValueError(
                f"prefill length {seq} and capacity {capacity} must both be tile multiples "
                f"({ttnn.TILE_SIZE}): the zero tail is concatenated on the sequence axis"
            )
        self.reset_cache()
        for block in self.blocks:
            block.kv = {
                "k": None,
                "v": None,
                "capacity": capacity,
                "filled": 0,
                "batch": batch,
                # By REFERENCE: one staged table, every block, never rebuilt per step.
                "mask": self._decode_mask,
            }

    def reset_cache(self):
        """Free every block's resident cache. The next prefill re-seeds them."""
        for block in self.blocks:
            kv = block.kv
            block.kv = None
            if not kv:
                continue
            for key in ("k", "v"):
                tensor = kv.get(key)
                if tensor is None:
                    continue
                try:
                    ttnn.deallocate(tensor)
                except Exception:  # noqa: BLE001 - an already-freed buffer is fine to skip
                    pass
        self.filled = 0

    def __repr__(self) -> str:
        return f"TextStack(n_layers={self.n_layers}, hidden={self.hidden_size}, kv_capacity={self.kv_capacity})"


# ------------------------------------------------------------------------------------------
# the factory
# ------------------------------------------------------------------------------------------


def build_text_stack(device, hf_model, layers=None, kv_capacity=None) -> TextStack:
    """Build the text decoder on `device` from `hf_model`'s own weights.

    `layers=None` means EVERY layer (26) -- never 0, which a builder would read as a zero-layer
    model. A cap of 1..26 builds the first `layers` layers; a larger one is capped to 26.
    """
    text = hf_model.model
    full_depth = len(text.layers)
    if layers is None:
        depth = full_depth
    else:
        depth = int(layers)
        if depth < 1:
            raise ValueError(f"layers={layers} is not a depth; None means every layer")
        if depth > full_depth:
            print(f"[text_stack] layers={layers} capped to the model's own depth {full_depth}")
            depth = full_depth

    hidden_size = int(text.embed_tokens.embedding_dim)
    if kv_capacity is None:
        kv_capacity = DEFAULT_PREFILL_CAPACITY + DEFAULT_DECODE_HEADROOM
    # One tile of headroom for the dead slots a split (shared-prefix) prefill leaves in the cache.
    kv_capacity = _tile_ceil(kv_capacity) + ttnn.TILE_SIZE

    blocks = [TextBlock(index, layer.build(device, text.layers[index])) for index in range(depth)]
    return TextStack(
        device,
        token_embed.build(device, text.embed_tokens),
        mistral_rotary_embedding.build(device, text.rotary_emb),
        blocks,
        mistral_r_m_s_norm.build(device, text.norm),
        hidden_size,
        kv_capacity,
    )

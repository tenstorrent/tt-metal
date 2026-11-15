# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Paged-bucket (PB) prefill: a bounded prefill program cache. bench/packaging/DRAM_PREFILL_FIX_PLAN_v2.md (+ v2
ADDENDUM_1, review DRAM_PREFILL_FIX_REVIEW.md delta check D1-D6).

The shipped prefill shows the exact chunk length and the absolute start to a handful of ttnn ops, so every new prompt
length or continuation start adds program-cache entries whose kernel binaries pin DRAM for the process lifetime
(MeshWorkloadImpl::load_binaries; ProgramCache has no eviction). PB:
- F1: every row's chunk is padded to the next 128-row page (pages128), the rows past the prompt are zeroed once at the
  embedding, and the last real row is taken from a whole 32-row tile. The decoder already computes at 128-multiples,
  so no op computes more than before.
- F2: past start 0 the rope rows are gathered from the row-major copy of the same table, the full-attention SDPA reads
  its start from a device tensor (GEMMA4_PB_SDPA_FLEX=1; 0 = the shipped scalar, and the start joins the retire key),
  and the KV fill sub-table is built on the host from the same row table.
- Trace safety: traces are retired when a row's key (prefill_paged_keys.row_key) or the row count is new; everything
  else a call can create is pre-warmed at boot. A guard counts program-cache entries around every prefill; if one
  appeared while a trace was live and not retired it logs ERROR and retires every trace before returning (the
  default, a fail-safe: the kernel-binary write is enqueued on CQ0 behind any in-flight replay), or raises
  (GEMMA4_PB_GUARD=raise: P3 and churn runs, where a fire is a failure).

The methods marked with `# PB` hunks are copies of the shipped bodies (decoder.py, model.py, generator.py, extracted
by bench/packaging/dram_fix/make_prefill_paged.py); tests/test_prefill_pb_host.py checks that only marked hunks
differ. The shipped classes and files are unchanged.
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder
from models.demos.gemma4_31b_qb2.tt.generator import CacheState, Gemma4Generator
from models.demos.gemma4_31b_qb2.tt.model import Gemma4Model
from models.demos.gemma4_31b_qb2.tt.prefill_paged_keys import (
    CHUNK_LIMIT,
    MERGE_ROWS,
    PAGE,
    call_keys,
    first_buckets,
    pages128,
    start_class,
    tap_windows,
    terminal_slices,
)

SDPA_FLEX = os.environ.get("GEMMA4_PB_SDPA_FLEX", "1") == "1"
EMBED_ZERO = os.environ.get("GEMMA4_PB_EMBED_ZERO", "mul")  # mul | where | off
GUARD = os.environ.get("GEMMA4_PB_GUARD", "retire")  # retire (fail-safe: log ERROR, retire every trace) | raise (P3, churn)
LOG_EVERY = int(os.environ.get("GEMMA4_PB_LOG_EVERY", "256"))


class PBGuardError(RuntimeError):
    """The program cache grew during a prefill while a trace was live: a lazily built program escaped the key."""


def dram_free_per_chip(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return int(view.num_banks) * int(view.total_bytes_free_per_bank)


class PrefillShared:
    """Per-call device uploads shared by every layer of a prefill call: rope rows, KV fill sub-tables and the SDPA
    start tensor, keyed so each is uploaded once. release() drops them all before the call returns."""

    def __init__(self, model, host_page_tables):
        self.model, self.host = model, host_page_tables or {}
        self.sdpa_flex = getattr(model, "sdpa_flex", SDPA_FLEX)
        self.uploads = {}

    def _put(self, key, make):
        if key not in self.uploads:
            self.uploads[key] = make()
        return self.uploads[key]

    def positions(self, start, rows):
        return self._put(("pos", start, rows), lambda: self.model.tensor(
            torch.arange(start, start + rows, dtype=torch.int32).reshape(1, rows), dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT))

    def rope(self, layer, start, rows):
        """cos, sin rows [start, start + rows) of this layer kind's table, [1, 1, rows, head_dim] tiled: the same
        bf16 values the shipped slice of layer.cos / layer.sin returns (decode_cos is its row-major copy)."""
        def gather():
            pos = self.positions(start, rows)
            shape = ttnn.Shape([1, 1, rows, layer.head_dim])
            return tuple(
                ttnn.reshape(ttnn.embedding(pos, table, layout=ttnn.TILE_LAYOUT), shape, shape)
                for table in (layer.decode_cos, layer.decode_sin)
            )

        return self._put(("rope", id(layer.decode_cos), start, rows), gather)

    def fill_table(self, layer, start, length):
        """The page-table columns [start / 128, (start + length) / 128) of this layer's row, from the host row the
        uploaded table was made from (the null-page sentinel included)."""
        index = self.model.layers.index(layer)
        host = self.host[index] if index in self.host else self.host[layer.kind]
        lo, hi = start // Decoder.PAGE_SIZE, (start + length) // Decoder.PAGE_SIZE
        return self._put(("fill", id(host), lo, hi), lambda: self.model.tensor(
            host[:, lo:hi].contiguous().to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT))

    def start_tensor(self, start):
        return self._put(("start", start), lambda: self.model.tensor(
            torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT))

    def release(self):
        self.uploads.clear()


class DecoderPB(Decoder):
    """A loaded Decoder with the PB prefill. Instances are converted in place (Gemma4ModelPB.__init__): no new state."""

    def prefill_forward(
        self, x, *, page_table, kv_cache, consume_input=False, start_pos=0, history=None, return_history=False,
        shared=None,  # PB: per-call uploads (PrefillShared)
    ):
        """Prefill a page-aligned span, optionally retaining sliding history for a continuation."""
        if start_pos < 0 or start_pos % self.PAGE_SIZE:
            raise ValueError("Prefill start must be a nonnegative cache-page boundary")
        if start_pos and self.sliding and history is None:
            raise ValueError("Sliding continuation requires its preceding K/V history")
        next_history = None
        batch, one, seq, width = x.shape
        assert one == 1 and width == self.prefill_input_width(seq) and (0 < seq <= self.max_context)
        assert page_table.shape[0] == batch and page_table.shape[1] * self.PAGE_SIZE >= start_pos + seq
        if consume_input:
            assert batch == 1 and x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT
            assert x.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        users = []
        for user in range(batch):
            parts = []
            stream = seq > self.PREFILL_CHUNK
            if stream:
                output_shape = ttnn.Shape([(seq + 31) // 32, 1, 32, width])
                if consume_input:
                    # Same-width BF16 tile view. Only completed chunk pages
                    # are written; future hidden rows remain untouched.
                    output = ttnn.reshape(x, output_shape, output_shape)
                    assert output.buffer_unique_id() == x.buffer_unique_id()
                else:
                    output = ttnn.empty(
                        output_shape,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
            previous_kv = history
            pt = (
                ttnn.reshape(
                    page_table, ttnn.Shape([1, page_table.shape[1]]), ttnn.Shape([1, page_table.padded_shape[1]])
                )
                if user == 0
                else page_table[user : user + 1, :]
            )
            for start in range(0, seq, self.PREFILL_CHUNK):
                end = min(start + self.PREFILL_CHUNK, seq)
                inp = (
                    ttnn.reshape(x, ttnn.Shape([1, 1, end, width]), ttnn.Shape([1, 1, (end + 31) // 32 * 32, width]))
                    if user == 0 and start == 0
                    else x[user : user + 1, :, start:end, :]
                )
                pad = -inp.shape[2] % self.PAGE_SIZE
                if pad and self.prefill_tile_pad:
                    inp, pad = self._zero_tile_tail(inp, width)
                if pad:
                    inp = ttnn.pad(inp, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
                absolute_start = start_pos + start
                q, k, v = self._prefill_qkv(inp, absolute_start, shared)  # PB: rope rows by gather (F2-a)
                if not self.sliding:
                    q = ttnn.typecast(q, ttnn.bfloat8_b)
                sdpa_config = self._prefill_sdpa_config(q, kv_cache[0], pt, start=start_class(absolute_start))  # PB
                length = inp.shape[2]
                fill_pt = (  # PB: a host-built sub-table past start 0 (F2-c)
                    shared.fill_table(self, absolute_start, length)  # PB
                    if absolute_start and shared is not None  # PB
                    else pt[:, absolute_start // self.PAGE_SIZE : (absolute_start + length) // self.PAGE_SIZE]  # PB
                )  # PB
                ttnn.experimental.paged_fill_cache(
                    kv_cache[0], ttnn.typecast(k, kv_cache[0].dtype), fill_pt, batch_idx=0
                )
                ttnn.experimental.paged_fill_cache(
                    kv_cache[1], ttnn.typecast(v, kv_cache[1].dtype), fill_pt, batch_idx=0
                )
                if not self.sliding:
                    # Full attention consumes the paged cache, not these
                    # temporary heads. CQ0 orders cache fill before SDPA.
                    del k, v
                if self.sliding and previous_kv is not None:
                    both_k = ttnn.concat([previous_kv[0], k], dim=2)
                    both_v = ttnn.concat([previous_kv[1], v], dim=2)
                    attn = ttnn.transformer.scaled_dot_product_attention(
                        q,
                        both_k,
                        both_v,
                        attn_mask=self.history_mask[
                            :, :, : inp.shape[2], 1024 - previous_kv[0].shape[2] : 1024 + inp.shape[2]
                        ],
                        is_causal=False,
                        scale=1.0,
                        program_config=sdpa_config,
                        compute_kernel_config=self.pref_compute,
                    )
                    del both_k, both_v
                elif self.sliding:
                    attn = ttnn.transformer.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        is_causal=True,
                        scale=1.0,
                        sliding_window_size=self.window,
                        program_config=sdpa_config,
                        compute_kernel_config=self.pref_compute,
                    )
                else:
                    attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                        q,
                        kv_cache[0],
                        kv_cache[1],
                        pt,
                        **(  # PB: the start as a device tensor past 0 (F2-b), else the shipped scalar
                            {"chunk_start_idx_tensor": shared.start_tensor(absolute_start)}  # PB
                            if absolute_start and shared is not None and shared.sdpa_flex  # PB
                            else {"chunk_start_idx": absolute_start}  # PB
                        ),  # PB
                        scale=1.0,
                        program_config=sdpa_config,
                        compute_kernel_config=self.pref_compute,
                    )
                if self.sliding:
                    if return_history and end == seq:
                        # Keep one extra page so a later unaligned scheduler boundary
                        # can replay its partial page without losing the prior window.
                        tails = []
                        for index, value in enumerate((k, v)):
                            live = value[:, :, : end - start, :]
                            joined = ttnn.concat([previous_kv[index], live], dim=2) if previous_kv is not None else live
                            tail_start = max(0, joined.shape[2] - (self.window + self.PAGE_SIZE))
                            tails.append(joined[:, :, tail_start:, :])
                        next_history = tuple(tails)
                    previous_kv = (k, v)
                if attn.dtype != ttnn.bfloat16 and self.sliding:
                    attn = ttnn.typecast(attn, ttnn.bfloat16)
                joined = self._concat_heads(attn)
                del q, attn
                local = self._pref_linear(joined, "o")
                del joined
                reduced = self._reduce(local, width)
                del local
                post = self._resnorm(reduced, "post_attention_layernorm", False)
                # Reuse the earliest private owner: reduced has finished its
                # norm read on CQ0, while post and a nonaliasing input slice
                # are also disposable after this add. This lets later MLP
                # allocations coalesce without mutating caller x.
                reuse_post = width != self.hidden_size
                private_input = (reuse_post) and inp.buffer_unique_id() != x.buffer_unique_id()
                residual_output = None
                if reuse_post:
                    candidates = [reduced, post]
                    if private_input:
                        candidates.append(inp)
                    residual_output = min(candidates, key=lambda tensor: tensor.buffer_address())
                    del candidates
                mlp_input = ttnn.add(inp, post, output_tensor=residual_output, dtype=self.residual_dtype)
                del residual_output
                del inp, post, reduced
                result = self._prefill_mlp(mlp_input)
                del mlp_input
                # Keep canonical tile padding while aliasing leading output rows.
                # The storage view retains the result allocation owner.
                logical = end - start
                if stream:
                    # Raw BF16 tile copies into the final sequence allocation.
                    # A32-row block preserves exactly the canonical tile tail.
                    if consume_input:
                        assert result.buffer_unique_id() != x.buffer_unique_id()
                    if start == 0:
                        output.update_tensor_topology(result.tensor_topology())
                    tiled_shape = ttnn.Shape([1, 1, (logical + 31) // 32 * 32, width])
                    output_part = ttnn.reshape(result, tiled_shape, tiled_shape)
                    output_pages = self.prefill_output_page_table[:, start // 32 : (end + 31) // 32]
                    ttnn.experimental.paged_fill_cache(output, output_part, output_pages, batch_idx=0)
                    del output_part, output_pages
                else:
                    parts.append(
                        ttnn.reshape(
                            result,
                            ttnn.Shape([1, 1, logical, width]),
                            ttnn.Shape([1, 1, (logical + 31) // 32 * 32, width]),
                        )
                    )
                del result
            if self.sliding:
                del previous_kv, k, v
            if stream:
                users.append(
                    ttnn.reshape(
                        output, ttnn.Shape([1, 1, seq, width]), ttnn.Shape([1, 1, (seq + 31) // 32 * 32, width])
                    )
                )
                del output
            elif len(parts) > 1:
                # Concatenating a partial tile would untilize every chunk and
                # allocate extra full-sequence copies. Only the final chunk
                # has padding: join physical tile views, then restore true S.
                tiled_parts = [ttnn.reshape(part, part.padded_shape, part.padded_shape) for part in parts]
                joined_parts = ttnn.concat(tiled_parts, dim=2)
                users.append(
                    ttnn.reshape(
                        joined_parts, ttnn.Shape([1, 1, seq, width]), ttnn.Shape([1, 1, (seq + 31) // 32 * 32, width])
                    )
                )
                del tiled_parts, joined_parts
            else:
                users.append(parts[0])
        result = ttnn.concat(users, dim=0) if batch > 1 else users[0]
        return (result, next_history) if return_history else result

    def _prefill_qkv(self, x, start, shared=None):  # PB
        tied = not self.sliding
        fused = self._pref_linear(self._input_norm(x, "input_layernorm", False), "qk" if tied else "qkv")
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            fused,
            num_heads=8,
            num_kv_heads=self.kv_heads,
            transpose_k_heads=False,
            kv_tied=tied,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        del fused
        if tied:
            # Full attention derives rotated K from normalized V below.
            # The raw K output has no consumer after head creation.
            del k
        # Only a start-zero, whole-width prefix can alias the table buffer.
        rope_shape = ttnn.Shape([1, 1, x.shape[2], self.head_dim])
        if start == 0:
            cos = ttnn.reshape(self.cos, rope_shape, rope_shape)
            sin = ttnn.reshape(self.sin, rope_shape, rope_shape)
        elif shared is not None:  # PB: rows start.. gathered from the row-major copy of the same table (F2-a)
            cos, sin = shared.rope(self, start, x.shape[2])  # PB
        else:
            cos = ttnn.reshape(self.cos[start : start + x.shape[2], :], rope_shape)
            sin = ttnn.reshape(self.sin[start : start + x.shape[2], :], rope_shape)
        q = self._norm(q, "self_attn.q_norm")
        q = ttnn.experimental.rotary_embedding(q, cos, sin)
        if not self.sliding:
            v = self._norm(v)
            k = ttnn.experimental.rotary_embedding(ttnn.multiply(v, self.norms["self_attn.k_norm"]), cos, sin)
        else:
            k = self._norm(k, "self_attn.k_norm")
            k = ttnn.experimental.rotary_embedding(k, cos, sin)
            v = self._norm(v)
        return (q, k, v)


class Gemma4ModelPB(Gemma4Model):
    """The shipped model with the PB prefill. Arguments as Gemma4Model; sdpa_flex selects F2-b."""

    def __init__(self, *args, sdpa_flex=SDPA_FLEX, embed_zero=EMBED_ZERO, **kwargs):
        super().__init__(*args, **kwargs)
        for cls in DecoderPB.__mro__:
            assert "__slots__" not in vars(cls), f"{cls.__name__} defines __slots__"
        for layer in self.layers:
            assert type(layer) is Decoder, type(layer)
            layer.__class__ = DecoderPB
            assert type(layer) is DecoderPB
        self.sdpa_flex = sdpa_flex
        if embed_zero not in ("mul", "where", "off"):
            raise ValueError(f"GEMMA4_PB_EMBED_ZERO must be mul, where or off: {embed_zero}")
        self.embed_zero = embed_zero
        self.pb_warm_terminal = False

    def zero_rows(self, x, valid_rows):
        """x [1, 1, S, 1344] with rows >= valid_rows set to exact 0: a 0/1 column ([1, 1, S, 1], uploaded, released
        at return). mul: x * 1 for real rows (P3-1 must show it bitwise); where: a select."""
        if self.embed_zero == "off":
            return x
        rows = x.shape[2]
        col = (torch.arange(rows) < valid_rows).to(torch.bfloat16).reshape(1, 1, rows, 1)
        mask = self.tensor(col)
        y = ttnn.multiply(x, mask) if self.embed_zero == "mul" else ttnn.where(mask, x, 0.0)
        del mask
        return y

    def warm_terminal_slices(self, x, sequence_length):
        """Boot only: the terminal's tile-start slices of this bucket and its 32 row slices (prefill_paged_keys)."""
        for tile_start in terminal_slices(sequence_length):
            tile = x[:, :, tile_start : tile_start + 32, :]
            for row in range(32):
                tile[:, :, row : row + 1, :]
            del tile

    def embed(self, tokens, *, sequence_length=None, batch=None, valid_rows=None):  # PB
        x = ttnn.embedding(tokens, self.embedding, layout=ttnn.TILE_LAYOUT, dtype=self.activation_dtype)
        # Prefill owns this newly allocated embedding. Scalar multiplication
        # is elementwise, so it need not retain a second full sequence owner.
        output = x if (sequence_length is not None) else None
        x = ttnn.multiply(x, self.embedding_scale, output_tensor=output)
        del output
        if sequence_length is not None:
            x = ttnn.reshape(
                x,
                ttnn.Shape([1, 1, sequence_length, 1344]),
                ttnn.Shape([1, 1, (sequence_length + 31) // 32 * 32, 1344]),
            )
            if valid_rows is not None and valid_rows < sequence_length:  # PB: padded rows start at exact zero
                x = self.zero_rows(x, valid_rows)  # PB
            if self.layers[0].prefill_input_width(sequence_length) == 5376:
                x = self.gather_hidden(x)
        else:
            x = ttnn.reshape(x, [1, 1, 32, 1344])
            x = self.gather_hidden(x)
            x = ttnn.reshape(x, ttnn.Shape([1, 1, batch, 5376]), ttnn.Shape([1, 1, 32, 5376]))
            x = ttnn.to_memory_config(x, self.layers[0]._mem(5376, 28))
        return x

    def prefill_device(
        self,
        tokens,
        *,
        sequence_length,
        page_tables,
        kv_cache,
        all_logits=False,
        start_pos=0,
        histories=None,
        taps=None,
        tap_window=None,
        valid_length=None,  # PB: real rows; sequence_length is the 128-row bucket
        host_page_tables=None,  # PB: the per-row host tables the uploaded ones came from (F2-c)
    ):
        """taps (speculative drafting only): {layer index: None} is filled with output rows
        [start, end) = tap_window (default: the last min(2048, rows)) of each named layer, before
        the next layer consumes them. See capture_tap for windows that end in tile padding."""
        if len(kv_cache) != len(self.layers):
            raise ValueError("Expected one cache pair per loaded layer")
        valid = sequence_length if valid_length is None else valid_length  # PB
        shared = PrefillShared(self, host_page_tables)  # PB
        x = self.embed(tokens, sequence_length=sequence_length, valid_rows=valid)  # PB
        for i, (layer, cache) in enumerate(zip(self.layers, kv_cache)):
            # This hidden stream is private to this invocation. Completed
            # chunks can reuse it once their input reads and KV fills finish.
            x = layer.prefill_forward(
                x,
                page_table=page_tables[i] if i in page_tables else page_tables[layer.kind],
                kv_cache=cache,
                consume_input=True,
                start_pos=start_pos,
                history=histories[i] if histories is not None else None,
                return_history=histories is not None,
                shared=shared,  # PB
            )
            if histories is not None:
                x, histories[i] = x
            if taps is not None and self.layer_indices[i] in taps:
                start, end = tap_window or (max(0, valid - 2048), valid)  # PB
                taps[self.layer_indices[i]] = self.capture_tap(x, start, end)
        if not all_logits:
            # An unaligned slice begin untilizes its entire input. Copy
            # the aligned final tile first, then select its logical row.
            if self.pb_warm_terminal:  # PB: boot only: every tile start and row of this bucket
                self.warm_terminal_slices(x, sequence_length)  # PB
            tile_start = (valid - 1) // 32 * 32  # PB
            x = x[:, :, tile_start : tile_start + 32, :]  # PB: a whole tile, keyed on tile_start only
            last_row = valid - 1 - tile_start  # PB
            x = x[:, :, last_row : last_row + 1, :]
        shared.release()  # PB
        return self.terminal(x)


class GeneratorPBMixin:
    """Gemma4Generator.prefill_forward with PB keys, history alignment, the upload release and the trace guard.
    pb_trace_probe: an adapter's own live traces (sampling, draft) besides self.traces; pb_on_taint: the adapter's
    release of them (GEMMA4_PB_GUARD=retire only)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdpa_flex = getattr(self.model, "sdpa_flex", SDPA_FLEX)
        self.pb_seen = set()
        self.pb_warmed = None
        self.pb_calls = 0
        self.pb_taints = []
        self.pb_trace_probe = lambda: False
        self.pb_on_taint = lambda: None
        self.pb_guard_mode = GUARD

    def pb_novel(self, prompt_lens, starts, return_all_logits=False):
        """Whether this call needs the traces retired first (an adapter's own traces too)."""
        keys = call_keys(list(prompt_lens), list(starts), flex=self.sdpa_flex) | {("all_logits", bool(return_all_logits))}
        return bool(keys - self.pb_seen)

    def pb_traces_live(self):
        return bool(self.traces) or bool(self.pb_trace_probe())

    def pb_guard_begin(self, novel):
        return {"live": self.pb_traces_live(), "entries": self.mesh.num_program_cache_entries(), "novel": novel}

    def pb_guard_end(self, guard, keys, novel):
        entries = self.mesh.num_program_cache_entries()
        grew = entries - guard["entries"]
        self.pb_calls += 1
        if grew > 0 and guard["live"]:
            new = sorted(map(str, keys - self.pb_seen))
            message = (f"program cache grew by {grew} during a prefill with a live trace (novel={novel}, new keys "
                       f"{new}); a lazily built program escaped the PB key")
            self.pb_taints.append(message)
            if self.pb_guard_mode == "raise":
                raise PBGuardError(message)
            logger.error("PB guard: {}; retiring every trace", message)
            self._retire_traces()
            self.pb_on_taint()
        self.pb_seen |= keys
        if novel or self.pb_calls % LOG_EVERY == 0:
            logger.info("Prefill program cache: {} entries; DRAM free {} B per chip (PB, {} keys, {} calls)", entries,
                        dram_free_per_chip(self.mesh), len(self.pb_seen), self.pb_calls)

    def pb_mark_warmed(self):
        """End of boot warmup: from here on every key outside this set is a first appearance."""
        self.pb_warmed = frozenset(self.pb_seen)
        logger.info("PB warmup covers {} keys; {} program-cache entries; DRAM free {} B per chip",
                    len(self.pb_warmed), self.mesh.num_program_cache_entries(), dram_free_per_chip(self.mesh))

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        return_all_logits=False,
        slots=None,
        return_device=False,
        start_positions=None,
        history_keys=None,
        taps=None,
        tap_windows=None,
        **kwargs,
    ):
        """taps / tap_windows (speculative drafting only; None leaves every call as before): one {layer: None} dict
        and one absolute [start, end) window per prompt row. Each row's dict is filled in place with that layer's
        output rows of the window positions this call computes (tap_capture_range), as model.prefill_device does."""
        state = kv_cache
        self._check_options(kwargs)
        if taps is not None and (tap_windows is None or len(taps) != len(prompt_lens) or len(tap_windows) != len(taps)):
            raise ValueError("taps and tap_windows need one entry per prompt row")
        if not isinstance(state, CacheState):
            raise TypeError("kv_cache must be a CacheState; caller may supply its kv tensors and tables")
        slots = list(range(len(prompt_lens))) if slots is None else list(slots)
        if (
            len(slots) != len(prompt_lens)
            or len(set(slots)) != len(slots)
            or any(s < 0 or s >= state.batch for s in slots)
        ):
            raise ValueError("Expected distinct valid fixed slots, one per prompt")
        if tokens.ndim != 2 or tokens.shape[0] != len(prompt_lens) or not prompt_lens:
            raise ValueError("Expected one token row per nonempty prompt")
        self._validate_tokens(tokens)
        if any(not isinstance(n, int) or not 0 < n <= min(state.capacity, tokens.shape[1]) for n in prompt_lens):
            raise ValueError("Invalid logical prompt length")
        starts = [0] * len(prompt_lens) if start_positions is None else list(start_positions)
        if len(starts) != len(prompt_lens) or any(s < 0 or s >= n for s, n in zip(starts, prompt_lens)):
            raise ValueError("Prefill must advance each request from its current position")
        shape_key = (state.batch, state.capacity, tuple(prompt_lens), tuple(starts), bool(return_all_logits))
        keys = call_keys(prompt_lens, starts, flex=self.sdpa_flex) | {("all_logits", bool(return_all_logits))}  # PB
        novel = bool(keys - self.pb_seen)  # PB: per-row bucket keys, not the exact signature
        if novel:  # PB
            # A new prefill signature may create cached protocol semaphores or
            # norm buffers. Retire old traces before those persistent allocations.
            self._retire_traces()
        guard = self.pb_guard_begin(novel)  # PB
        self._tables(state, page_table)
        outputs = []
        for row, (slot, length) in enumerate(zip(slots, prompt_lens)):
            if not 0 < length <= state.capacity or length > tokens.shape[1]:
                raise ValueError("Invalid logical prompt length")
            start = starts[row]
            aligned_start = start // Decoder.PAGE_SIZE * Decoder.PAGE_SIZE
            chunk_length = length - aligned_start
            histories = None
            if history_keys is not None:
                key = history_keys[row]
                if start == 0:
                    histories = [None] * len(self.model.layers)
                else:
                    previous = state.prefill_history[key]
                    if previous["end"] != start:
                        raise ValueError("Prefill history does not match the scheduler continuation")
                    rewind = previous["padded_end"] - aligned_start  # PB: the tail ends at the padded end
                    if rewind not in (0, Decoder.PAGE_SIZE):  # PB
                        raise ValueError("PB history must end at most one page past the continuation's start")  # PB
                    histories = []
                    for tail in previous["layers"]:
                        if tail is None:
                            histories.append(None)
                            continue
                        valid = tail[0].shape[2] - rewind
                        if valid == 0:
                            histories.append(None)
                            continue
                        begin = max(0, valid - 1024)
                        histories.append(tuple(t[:, :, begin:valid, :] for t in tail))
            ids = torch.zeros(1, pages128(chunk_length), dtype=torch.int32)  # PB: the 128-row bucket
            ids[0, :chunk_length] = tokens[row, aligned_start:length]
            tt_ids = self._upload(ids, ttnn.uint32)
            tables, uploaded_rows, host_rows = {}, {}, {}  # PB
            host_tables = {}  # PB
            for kind, table in page_table.items():
                sentinel = state.vllm_owned and self.model.layers[kind].kind == "sliding_attention"
                key = (id(table), sentinel)
                if key not in uploaded_rows:
                    row_table = table[slot : slot + 1]
                    if sentinel:
                        # Only temporary serving prefill uses the null-page
                        # sentinel. These read-only rows may be shared; mutable
                        # persistent decode tables retain independent storage.
                        row_table = row_table.masked_fill(row_table == 0, -1)
                    uploaded_rows[key] = self._upload(row_table)
                    host_rows[key] = row_table  # PB
                    self.counters["prefill_table_uploads"] += 1
                tables[kind] = uploaded_rows[key]
                host_tables[kind] = host_rows[key]  # PB
            tap_kwargs = {}
            if taps is not None:
                captured = self.tap_capture_range(start, tap_windows[row])
                if captured is not None:
                    tap_kwargs = {
                        "taps": taps[row],
                        "tap_window": (captured[0] - aligned_start, captured[1] - aligned_start),
                    }
            out = self.model.prefill_device(
                tt_ids,
                sequence_length=pages128(chunk_length),  # PB
                valid_length=chunk_length,  # PB
                host_page_tables=host_tables,  # PB
                page_tables=tables,
                kv_cache=state.kv,
                all_logits=return_all_logits,
                start_pos=aligned_start,
                histories=histories,
                **tap_kwargs,
            )
            if history_keys is not None:
                state.prefill_history[history_keys[row]] = {  # PB
                    "end": length, "padded_end": aligned_start + pages128(chunk_length), "layers": histories  # PB
                }  # PB
            outputs.append(
                out  # PB
                if return_device  # PB
                else self._host_logits(out).reshape(1, -1, self.model.config.vocab_size)[:, :chunk_length]  # PB
            )
            # Device-return outputs retain their own owners in outputs. Host
            # mode has consumed out; do not carry it into the next prefill.
            del out, tt_ids, tables, uploaded_rows, host_rows, host_tables  # PB: no upload outlives the call
        self.prefill_shapes.add(shape_key)
        self.pb_guard_end(guard, keys, novel)  # PB
        if return_device:
            return outputs
        width = max(prompt_lens) if return_all_logits else 1
        result = torch.zeros(len(outputs), width, self.model.config.vocab_size)
        for i, out in enumerate(outputs):
            result[i, : out.shape[1]] = out[0]
        return result


class Gemma4GeneratorPB(GeneratorPBMixin, Gemma4Generator):
    """B32's generator (build_generator) with the PB prefill."""


def build_generator_pb(mesh_device, *, checkpoint=None, layer_indices=None, max_seq_len=2048, prefill_tile_pad=False):
    model = Gemma4ModelPB(mesh_device, checkpoint=checkpoint, layer_indices=layer_indices,
                          prefill_tile_pad=prefill_tile_pad)
    return Gemma4GeneratorPB(model, max_seq_len=max_seq_len)


def layer_output(model, rows, width):
    """A prefill layer output's spec: [1, 1, rows, 5376] replicated (rows <= 128) or [1, 1, rows, 1344] hidden-sharded,
    as dflash_warmup.warmup_tap_seed builds it."""
    hidden, mesh = model.config.hidden_size, model.mesh
    return ttnn.from_torch(
        torch.zeros(1, 1, rows, hidden, dtype=torch.bfloat16),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if width == hidden else ttnn.ShardTensorToMesh(mesh, dim=3),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def all_buckets():
    """Every chunk bucket: first chunks to 8,192 and a continuation that recomputes a rewind (8,320)."""
    return first_buckets(CHUNK_LIMIT + PAGE)


def prewarm_tap_windows(model, seed_rows, buckets=None):
    """D3: the real capture_tap for every (bucket, tap window) the drafter can ask for (prefill_paged_keys.tap_windows;
    65 buckets x 4 = 260), on tensors with the real layer-output spec. Before any trace capture."""
    count = 0
    for b in buckets or all_buckets():
        x = layer_output(model, b, model.layers[0].prefill_input_width(b))
        for lo, hi in tap_windows(b, seed_rows):
            model.capture_tap(x, lo, hi)
            count += 1
        del x
    ttnn.synchronize_device(model.mesh)
    return count


def prewarm_terminal(model, buckets=None):
    """The terminal tile-start and row slices of every bucket on tensors with the last layer output's spec (the
    bucket warmup builds them on real outputs too; this adds the rewind bucket 8,320). Before any trace capture."""
    for b in buckets or all_buckets():
        x = layer_output(model, b, model.layers[0].prefill_input_width(b))
        model.warm_terminal_slices(x, b)
        del x
    ttnn.synchronize_device(model.mesh)


def prewarm_merge(generator, out, logits_like, max_rows=MERGE_ROWS):
    """sample_prefill_outputs' merge for every row count (concat, pad to 32, reshape, the copy into the sampling
    trace's logits input) from one device prefill output `out`, into a tensor with the trace input's spec: the
    merge ops standalone, never through sample_prefill_outputs (its _entry would capture the traces first; review
    D7). Before any trace capture."""
    if generator.traces:
        raise RuntimeError("The merge pre-warm must run before any trace capture")
    shape = ttnn.Shape([1, 1, 32, generator.model.config.vocab_size // 4])
    for batch in range(1, max_rows + 1):
        merged = out if batch == 1 else ttnn.concat([out] * batch, dim=2)
        if 1 < batch < 32:
            merged = ttnn.pad(merged, [(0, 0), (0, 0), (0, 32 - batch), (0, 0)], value=0.0)
        ttnn.copy(ttnn.reshape(merged, shape, shape), logits_like)
        del merged
    ttnn.synchronize_device(generator.mesh)

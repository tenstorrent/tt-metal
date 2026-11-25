# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optional prepared B1/P128 first-chunk trace around bounded prefill.

The generator owns capture, replay and release. This helper owns no additional
device storage: it binds the existing bounded-prefill inputs, working recurrent
states and generator logits before the first capture. Other lengths, resumed
prefill and multi-row requests continue through BoundedPrefill.run.
"""

import torch

import ttnn


class FirstChunkPrefillTrace:
    trace_key = "prefill_128"
    length = 128

    def __init__(self, generator):
        if generator.batch_size != 1:
            raise ValueError("First-chunk trace candidate requires configured batch 1")
        if generator.traces:
            raise RuntimeError("Construct prefill trace helper before the first capture")
        if generator.max_seq_len < self.length:
            raise ValueError("First-chunk trace requires capacity for 128 prompt tokens")
        self.generator = generator
        self.engine = generator.prefill_single
        self.model = generator.model
        self.prepared = False

    def upload(self, prompt_token_ids, *, page_table=None, kv_cache=None):
        """Install first-chunk inputs outside capture, preserving buffer identity.

        The only accepted interval is [0,128), in fixed slot zero. Match the
        public prefill validation before changing any device input. Page-table
        metadata is shared with eager bounded prefill, so transitions between
        eager requests and this trace refresh the same backing tensors.
        """
        if ttnn.is_trace_capture_active(self.generator.mesh):
            raise RuntimeError("Prefill trace input uploads must be outside capture")
        tokens = torch.as_tensor(prompt_token_ids, dtype=torch.int32)
        if tokens.ndim == 1:
            tokens = tokens[None]
        if tuple(tokens.shape) != (1, self.length):
            raise ValueError("First-chunk trace accepts exactly one 128-token prompt")
        if bool(((tokens < 0) | (tokens >= self.model.config.vocab_size)).any()):
            raise ValueError("Prompt token outside vocabulary")
        gen, engine = self.generator, self.engine
        table = gen.page_host if page_table is None else torch.as_tensor(page_table, dtype=torch.int32)
        if tuple(table.shape) != tuple(gen.page_host.shape):
            raise ValueError("Prefill page table must retain the configured shape")
        cache = gen.cache if kv_cache is None else kv_cache
        gen._bind_pages(table, cache)
        gen._release_sampling_history(0)
        row_table = gen.page_host[:1]
        if not torch.equal(row_table, engine._page_host):
            engine._write(row_table, engine.full_table)
            engine._page_host = row_table.clone()
            engine.counters["page_table_refreshes"] += 1
        # B1 takes ordered paged_fill_cache regardless of bulk-page ownership.
        # Use the same first 16 entries as the aligned bounded-prefill path.
        fill_table = row_table[:, :16]
        if not torch.equal(fill_table, engine._fill_page_host):
            engine._write(fill_table, engine.fill_table)
            engine._fill_page_host = fill_table.clone()
        engine._write(tokens, engine.ids[self.length])
        engine._write(torch.arange(self.length, dtype=torch.int32)[None], engine.positions[self.length])
        engine._write(torch.zeros(1, dtype=torch.int32), engine.start)

    def graph(self):
        """Device-only equivalent of a fresh single-chunk last-logit prefill."""
        gen, engine = self.generator, self.engine
        # Fresh public prefill clears recurrent slot state; attention cache is
        # overwritten by paged_fill_cache and causally masked just as in run().
        gen._release_recurrent_slot(0)
        engine._import_state(0)
        value = self.model.embeddings(engine.ids[self.length])
        rope = self.model.rotations(engine.positions[self.length], batch=1, length=self.length)
        for layer, state in zip(self.model.layers, engine.states):
            value = engine._layer_chunk(layer, value, state, rope)
        value = self.model.terminal(value[:, -1:, :])
        value = ttnn.reshape(value, (1, 1, 1, self.model.config.vocab_size // 4))
        padded = ttnn.pad(value, ((0, 0), (0, 0), (0, 31), (0, 0)), value=0)
        ttnn.copy(padded, gen.logits)
        engine._publish_state(0)

    def prepare(self):
        """Warm the exact graph before any capture; caller restores request state."""
        if self.prepared:
            return
        if self.generator.traces:
            raise RuntimeError("Prepare first-chunk prefill before every trace capture")
        if not self.engine.prepared or self.length not in self.engine.prepared_lengths:
            raise RuntimeError("Prepare the bounded-prefill engine before this trace")
        self.upload([1] * self.length)
        self.graph()
        self.prepared = True

    def record_replay(self):
        """Mirror eager bookkeeping once after a submitted prefill replay."""
        self.engine.counters["chunks"] += 1
        self.engine.counters["ordered_fill_chunks"] += 1

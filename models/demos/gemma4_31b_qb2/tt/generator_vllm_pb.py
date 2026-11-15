# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`batch-262k`: shipped B32 (generator_vllm.Gemma4ForCausalLM) with the paged-bucket prefill (prefill_paged.py,
bench/packaging/DRAM_PREFILL_FIX_PLAN_v2.md + v2 ADDENDUM_1). Only construction and warmup differ:
- the model is Gemma4ModelPB with B32's own settings (decode weights, prefill_tile_pad False: v2 ADDENDUM_1 (d));
- warmup_model_prefill, empty in B32, builds every first-chunk bucket (128 .. 8,192), the terminal slices (to the
  rewind bucket 8,320), and the sampling merge for 1..32 rows (standalone: never through sample_prefill_outputs,
  whose _entry captures the traces) before warmup_model_decode captures them; GEMMA4_PREFILL_WARM_CONT=1 also builds
  one continuation per (bucket, start class, history length) (~800 passes, >= 8 min warm, est.);
- every prefill call goes through the PB generator's key, retirement and guard; a "Prefill program cache" line is
  logged on each new key and every GEMMA4_PB_LOG_EVERY calls.
"""

import os
import time

import torch
from loguru import logger

from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM
from models.demos.gemma4_31b_qb2.tt.prefill_paged import (
    Gemma4GeneratorPB,
    Gemma4ModelPB,
    prewarm_merge,
    prewarm_terminal,
)
from models.demos.gemma4_31b_qb2.tt.prefill_paged_keys import MERGE_ROWS, PAGE, continuation_starts, first_buckets

WARM_CONT = os.environ.get("GEMMA4_PREFILL_WARM_CONT", "0") == "1"


class Gemma4ForCausalLMPB(Gemma4ForCausalLM):
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len=262144, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1 or not 1 <= max_batch_size <= 32:
            raise ValueError("Gemma4 QB2 requires TP4, DP1, and batch 1..32")
        if not 128 <= max_seq_len <= 262144 or max_seq_len % 128:
            raise ValueError("Serving context must be 128..262144, aligned to 128-token pages")
        if hf_config.text_config.hidden_size != 5376 or hf_config.text_config.num_hidden_layers != 60:
            raise ValueError("Expected the Gemma4 31B checkpoint")
        torch.set_num_threads(8)
        # = build_generator(mesh_device, max_seq_len=...) with the PB model and generator
        generator = Gemma4GeneratorPB(Gemma4ModelPB(mesh_device), max_seq_len=max_seq_len)
        return cls(generator, max_batch_size, max_seq_len)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        cache = super().allocate_kv_cache_per_layer(per_layer_specs)
        # Allocator blocks per layer (its pool's block count), for the warmup's per-pool check.
        self.pb_layer_blocks = [shape[0] for shape, _, _ in per_layer_specs]
        return cache

    def _warm_tables(self, pages, first_block=1):
        """Per-layer [1, columns] tables naming `pages` distinct, nonzero blocks below each layer's pool size (the
        writes land in free blocks before vLLM has allocated any; sliding zeros past them become the -1 sentinel)."""
        columns = self.max_seq_len // PAGE
        tables = []
        for blocks in self.pb_layer_blocks:
            if first_block + pages > blocks:
                raise ValueError(f"The pool ({blocks} blocks) is too small for a {pages}-page warmup prefill")
            table = torch.zeros(1, columns, dtype=torch.int32)
            table[0, :pages] = first_block + torch.arange(pages, dtype=torch.int32)
            tables.append(table)
        return tables

    def _warm_prefill(self, length, *, start=0, tables=None):
        tables = tables or self._warm_tables(-(-length // PAGE))
        self.prefill_forward(
            torch.zeros(1, length, dtype=torch.int32), None, self.cache, [length],
            page_tables_per_layer=tables, start_pos=[start], empty_slots=[0],
        )
        return tables

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        if kv_cache is not self.cache:
            raise ValueError("Serving must use the cache allocated by vLLM")
        if self.generator.pb_warmed is not None:
            return
        model, gen = self.generator.model, self.generator
        if gen.traces:
            raise RuntimeError("The PB warmup must run before any trace capture")
        t0 = time.perf_counter()
        model.pb_warm_terminal = True
        try:
            for b in first_buckets(self.MAX_PREFILL_TOKENS):
                # L = b - 1: the bucket's programs, its zeroing mask and a partial last tile
                self._warm_prefill(b - 1)
                kv_cache.prefill_history.clear()
                self.prefill_slot_keys.clear()
        finally:
            model.pb_warm_terminal = False
        prewarm_terminal(model)
        t1 = time.perf_counter()
        # The sampling merge for every row count, standalone, from one device output into a tensor with the sampling
        # trace input's spec (review D7: not through sample_prefill_outputs, whose _entry would capture the traces).
        tables = self._tables(kv_cache, self._warm_tables(1), slots=[0])
        out = gen.prefill_forward(torch.zeros(1, 1, dtype=torch.int32), page_table=tables, kv_cache=kv_cache,
                                  prompt_lens=[1], slots=[0], return_device=True, start_positions=[0])[0]
        logits_like = model.tensor(torch.zeros(1, 1, 32, model.config.vocab_size), dim=3)
        prewarm_merge(gen, out, logits_like, MERGE_ROWS)
        gen.pb_seen |= {("rows", k) for k in range(1, MERGE_ROWS + 1)}
        del out, logits_like
        kv_cache.prefill_history.clear()
        self.prefill_slot_keys.clear()
        t2 = time.perf_counter()
        if WARM_CONT:
            self._warm_continuations()
        t3 = time.perf_counter()
        gen.pb_mark_warmed()
        logger.info("PB prefill warmup: {} buckets {:.1f} s, merge {:.1f} s, continuations {:.1f} s ({})",
                    len(first_buckets(self.MAX_PREFILL_TOKENS)), t1 - t0, t2 - t1, t3 - t2,
                    "on" if WARM_CONT else "off")

    def _warm_continuations(self):
        """GEMMA4_PREFILL_WARM_CONT=1: one continuation per (bucket, start class, history rows): a first chunk of
        `a` rows, then a `b`-row chunk from `a` (prefill_paged_keys.continuation_starts)."""
        for a in continuation_starts():
            for b in first_buckets(self.MAX_PREFILL_TOKENS):
                tables = self._warm_tables((a + b) // PAGE)
                self._warm_prefill(a, tables=tables)
                self._warm_prefill(a + b, start=a, tables=tables)
                self.cache.prefill_history.clear()
                self.prefill_slot_keys.clear()

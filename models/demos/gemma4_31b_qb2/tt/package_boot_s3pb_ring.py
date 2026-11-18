# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package's `batch` class for update 4: Gemma4ResidentMultiS32PBRingForCausalLM (generator_resident_multi_s3_2_pb_
ring_vllm: S3-PB with option C's KV layout, update 4 item A, m5-serving) that first makes sure the resident weight cache
exists (package_boot.ensure_resident_cache, as package_boot_s3pb does for update 3's class), plus the continuation
pre-warm (update 4 item B; contwarm.py; bench/packaging/update4/B_PREREG.md; LEAD_DECISIONS_PKG P106: the full warm).

Under `batch` a continuation's (start class, history rows) depends on how the prompts sharing a prefill-only step split
its 8,192-token budget (m5-serving; plugin ca7a182 scheduler.py:914-938), and B0 showed that the pair compiles its own
kernels at a fixed bucket. The warm builds every reachable key: the 10 (class, history) pairs of
prefill_paged_keys.continuation_starts() (starts 128 .. 1,280) x the 64 buckets 128 .. 8,192 = 640 keys (8,320 is
unreachable under the plugin's 128 alignment). Each pass: an a-row first chunk, then a b-row chunk from a, through the
PB generator directly (m5-serving: not the adapter's prefill_forward, which would admit SlotTable state), with tables
from the ring class's own _tables(..., slots=[0], chunks=[(start, end)]) (only the full layer's table is used; every
pass writes slot 0's ring, which a request's own prefill rewrites before its first decode reads it: option C's reuse
invariant), one history key per pass, the history cleared after each pass, and one _l1_guard after the block. It runs
at the end of the first warmup_model_prefill, before warmup_model_decode captures the sampling trace, and only when
contwarm finds no valid marker for this kernel cache and code (GEMMA4_CONT_PREWARM=auto)."""

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_31b_qb2.tt import contwarm
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_ring_vllm import (
    Gemma4ResidentMultiS32PBRingForCausalLM,
)
from models.demos.gemma4_31b_qb2.tt.package_boot import ensure_resident_cache
from models.demos.gemma4_31b_qb2.tt.prefill_paged_keys import CHUNK_LIMIT, PAGE, continuation_starts, first_buckets


def warm_tables(adapter, pages):
    """One source table per layer from a table per KV group, `pages` blocks each in distinct nonzero blocks (the pool is
    empty at boot); only the full layer's table is read by the ring class's _tables."""
    columns = adapter.max_seq_len // PAGE
    stride = pages + 1
    source = []
    for g in range(max(adapter.groups) + 1):
        table = torch.zeros(1, columns, dtype=torch.int32)
        table[0, :pages] = 1 + g * stride + torch.arange(pages, dtype=torch.int32)
        source.append(table)
    return [source[g] for g in adapter.groups]


class Gemma4ResidentMultiS32PBRingPackaged(Gemma4ResidentMultiS32PBRingForCausalLM):
    """The ring class that first makes sure the resident weight cache exists, plus the continuation pre-warm."""

    ENGINE_PACKAGE = "resident_m5_11"

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        ensure_resident_cache()
        return super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cont_prewarm = None

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        first = self.cont_prewarm is None
        super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)
        if not first:
            return
        if self.sampling_trace is not None:
            raise RuntimeError("The continuation pre-warm must run before any trace capture")
        self.cont_prewarm = contwarm.run("batch", self.ENGINE_PACKAGE, lambda: self._cont_warm(kv_cache),
                                         lambda: ttnn.synchronize_device(self.mesh), log=logger.info)

    def _cont_warm(self, kv_cache):
        gen = self.generator
        passes = 0
        for a in continuation_starts():
            for b in first_buckets(CHUNK_LIMIT):
                end = a + b
                source = warm_tables(self, -(-end // PAGE))
                tokens = torch.zeros(1, end, dtype=torch.int32)
                key = ("cont-prewarm", a, b)
                for start, stop in ((0, a), (a, end)):
                    tables = self._tables(kv_cache, source, slots=[0], chunks=[(start, stop)])
                    out = gen.prefill_forward(tokens, page_table=tables, kv_cache=kv_cache, prompt_lens=[stop],
                                              slots=[0], return_device=True, start_positions=[start],
                                              history_keys=[key])
                    del out
                kv_cache.prefill_history.clear()
                self.prefill_slot_keys.clear()
                passes += 1
        self._l1_guard("continuation pre-warm")
        return passes

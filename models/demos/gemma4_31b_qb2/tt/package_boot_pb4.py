# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package's `single-user` class for update 4: package_boot_pb3.Gemma4ResidentSpecPB3Packaged (unchanged) plus the
continuation pre-warm (update 4 item B; contwarm.py; bench/packaging/update4/UPDATE4_PLAN.md §B, B_PREREG.md).

A `single-user` request is prefilled alone, so its continuations start at multiples of 8,192: start class 256, 1,152
history rows. The warm builds those 64 keys ("cont", b, 256, 1152, 0), b = 128 .. 8,192, from start 1,280 (the same
class and history; prefill_paged_keys.continuation_starts): a 1,280-row first chunk, then a b-row chunk from 1,280,
through the PB generator directly (not the adapter's prefill_forward, which would admit request state), with the tap
capture the serving path passes (taps / tap_windows), one history key per pass, the history and the request state
cleared after each pass. It runs at the end of the first warmup_model_prefill (after PB3's seed warm), before
warmup_model_decode captures any trace, and only when contwarm finds no valid marker for this kernel cache and code
(GEMMA4_CONT_PREWARM=auto). The engine package (resident_m3_2) is unchanged; so is every evidence-pinned file."""

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_31b_qb2.tt import contwarm
from models.demos.gemma4_31b_qb2.tt.package_boot_pb3 import Gemma4ResidentSpecPB3Packaged
from models.demos.gemma4_31b_qb2.tt.prefill_paged_keys import CHUNK_LIMIT, PAGE, first_buckets

START = 1280  # class 256, history 1,152: every single-user continuation's (class, history)


def warm_tables(adapter, pages):
    """One source table per KV group, `pages` blocks each in distinct nonzero blocks (a free pool before vLLM allocates
    any), laid out per layer as the adapter's own warmups lay them (generator_resident_spec_vllm_pb._pb_warm_prefill_
    sampling): group g at 1 + g * stride."""
    columns = adapter.max_seq_len // PAGE
    stride = pages + 1
    source = []
    for g in range(max(adapter.groups) + 1):
        table = torch.zeros(1, columns, dtype=torch.int32)
        table[0, :pages] = 1 + g * stride + torch.arange(pages, dtype=torch.int32)
        source.append(table)
    return [source[g] for g in adapter.groups]


class Gemma4ResidentSpecPB4Packaged(Gemma4ResidentSpecPB3Packaged):
    """PB3 (package_boot_pb3) plus the marker-gated continuation pre-warm."""

    ENGINE_PACKAGE = "resident_m3_2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cont_prewarm = None

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        first = self.cont_prewarm is None
        super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)
        if not first:
            return
        if self._pb3_live():
            raise RuntimeError("The continuation pre-warm must run before any trace capture")
        self.cont_prewarm = contwarm.run("single-user", self.ENGINE_PACKAGE, lambda: self._cont_warm(kv_cache),
                                         lambda: ttnn.synchronize_device(self.mesh), log=logger.info)

    def _cont_warm(self, kv_cache):
        gen = self.generator
        passes = 0
        for b in first_buckets(CHUNK_LIMIT):
            end = START + b
            tables = self._tables(kv_cache, warm_tables(self, -(-end // PAGE)), slots=[0])
            tokens = torch.zeros(1, end, dtype=torch.int32)
            key = ("cont-prewarm", b)
            for start, stop in ((0, START), (START, end)):
                out = gen.prefill_forward(tokens, page_table=tables, kv_cache=kv_cache, prompt_lens=[stop], slots=[0],
                                          return_device=True, start_positions=[start], history_keys=[key],
                                          taps=[dict.fromkeys(self.tap_layers)],
                                          tap_windows=[self.drafter.tap_window(stop)])
                del out
            kv_cache.prefill_history.clear()
            self.prefill_slot_keys.clear()
            self._reset_request()
            passes += 1
        self._l1_guard("continuation pre-warm")
        return passes

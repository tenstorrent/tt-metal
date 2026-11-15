# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`single-user` (resident engine + DFlash2, generator_resident_spec_vllm) with the paged-bucket prefill
(prefill_paged.py; bench/packaging/DRAM_PREFILL_FIX_PLAN_v2.md + v2 ADDENDUM_1). Differences from the shipped class:
- construction: Gemma4ModelPB (prefill-only, GEMMA4_RESIDENT_TILE_PAD as shipped) and the PB prefill generator,
  composed as Gemma4ResidentForCausalLM.initialize_vllm_model and the spec class's wrapper do;
- prefill_forward: a copy of the shipped one whose novelty test is the PB key (the one `# PB` hunk);
- the PB guard watches this adapter's sampling and draft traces (the ones the shipped novelty path releases);
- warmup_model_prefill: the shipped warmups (every first-chunk bucket through the PB path, the drafter's tap and seed
  programs) with the terminal slices, then every (bucket, tap window) capture (review D3: 65 x 4), the terminal
  slices of the rewind bucket, and the one-row prefill sampling (_sample_prefill's reshape and eager sampler step)
  standalone on a device prefill output, all before warmup_model_decode captures any trace; then the warmed keys.
The resident engine's own traces are not released on a new prefill shape by the shipped adapter either; the guard
follows the shipped retire set.
"""

import os
import time

import torch
from loguru import logger

from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm import Gemma4ResidentSpecForCausalLM
from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import _PrefillGenerator
import ttnn
from models.demos.gemma4_31b_qb2.tt.prefill_paged import (
    GeneratorPBMixin,
    Gemma4ModelPB,
    prewarm_tap_windows,
    prewarm_terminal,
)


class PrefillGeneratorPB(GeneratorPBMixin, _PrefillGenerator):
    """The resident adapters' prefill-only generator with the PB prefill."""


class Gemma4ResidentSpecPBForCausalLM(Gemma4ResidentSpecForCausalLM):
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, *, max_seq_len=262144, tt_data_parallel=1, optimizations=None
    ):
        # = Gemma4ResidentForCausalLM.initialize_vllm_model (generator_resident_vllm.py:106-119) with the PB model
        #   and generator, then Gemma4ResidentSpecForCausalLM.initialize_vllm_model's wrapper (tap layers).
        if tt_data_parallel != 1 or max_batch_size != 1:
            raise ValueError("Resident Gemma4 QB2 (M1) serves TP4, DP1, one request")
        if not 128 <= max_seq_len <= 262144 or max_seq_len % 128:
            raise ValueError("Serving context must be 128..262144, aligned to 128-token pages")
        if hf_config.text_config.hidden_size != 5376 or hf_config.text_config.num_hidden_layers != 60:
            raise ValueError("Expected the Gemma4 31B checkpoint")
        torch.set_num_threads(8)
        tile_pad = os.environ.get("GEMMA4_RESIDENT_TILE_PAD", "1") == "1"
        model = Gemma4ModelPB(mesh_device, decode_weights=False, prefill_tile_pad=tile_pad)
        generator = PrefillGeneratorPB(model, max_seq_len=max_seq_len)
        return cls(generator, max_batch_size, max_seq_len, tap_layers=cls.drafter_tap_layers())

    def __init__(self, generator, max_batch_size, max_seq_len, *, vllm_config=None, tap_layers=()):
        super().__init__(generator, max_batch_size, max_seq_len, vllm_config=vllm_config, tap_layers=tap_layers)
        generator.pb_trace_probe = lambda: self.sampling_trace is not None or self.draft_trace is not None
        generator.pb_on_taint = self._pb_release_traces

    def _pb_release_traces(self):
        self._release_sampling_trace()
        self._release_draft_trace()

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, *, sampling_params=None,
                        page_tables_per_layer=None, start_pos=None, empty_slots=None, **kwargs):
        """M1.1's prefill (generator_resident_vllm.prefill_forward, same steps) plus the drafter's tap window
        through the generator pass-through into the history. Kept here, not as a hook in the M1.1 adapter, so the
        two verify separately."""
        if len(prompt_lens) != 1:
            raise ValueError("Speculative serving prefills one request at a time")
        start = 0 if start_pos is None else int(start_pos[0])
        end = int(prompt_lens[0])
        slots = [0] if empty_slots is None else [int(s) for s in empty_slots]
        if start == 0:
            self._reset_request()
            self.spec_policy.new_request()
        tables = self._tables(kv_cache, page_tables_per_layer, slots=slots)
        full_layer = next(i for i, layer in enumerate(self.generator.model.layers) if not layer.sliding)
        keys = [int(tables[full_layer][slot, 0]) for slot in slots]
        self.prefill_slot_keys = {s: k for s, k in self.prefill_slot_keys.items() if s not in slots and k not in keys}
        self.prefill_slot_keys.update(zip(slots, keys))
        novel = self.generator.pb_novel([end], [start])  # PB: the row's bucket key, not the exact signature
        if novel:
            self._release_sampling_trace()
            self._release_draft_trace()  # F3
        window = self.drafter.tap_window(end)
        captured = dict.fromkeys(self.tap_layers)
        outputs = self.generator.prefill_forward(
            tokens,
            page_table=tables,
            kv_cache=kv_cache,
            prompt_lens=[end],
            slots=slots,
            return_device=sampling_params is not None,
            start_positions=[start],
            history_keys=keys,
            taps=[captured],
            tap_windows=[window],
        )
        if novel:
            self._l1_guard(f"prefill {(end,)} from {(start,)}")
        self._record_prefill_taps(captured, window, start, end)
        if sampling_params is None:
            return outputs
        return self._sample_prefill(outputs, sampling_params, [end])

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        first = self.generator.pb_warmed is None
        model = self.generator.model
        t0 = time.perf_counter()
        model.pb_warm_terminal = True
        try:
            super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)
        finally:
            model.pb_warm_terminal = False
        if not first:
            return
        t1 = time.perf_counter()
        windows = prewarm_tap_windows(model, self.drafter.SEED_ROWS) if self.drafter is not None else 0
        prewarm_terminal(model)
        self._pb_warm_prefill_sampling(kv_cache)
        self.generator.pb_seen.add(("rows", 1))
        self.generator.pb_mark_warmed()
        logger.info("PB prefill warmup: shipped warmups {:.1f} s, {} tap windows + terminal + sampling {:.1f} s",
                    t1 - t0, windows, time.perf_counter() - t1)

    def _pb_warm_prefill_sampling(self, kv_cache):
        """_sample_prefill's device ops for one row, standalone (no sampling params, no trace): the reshape of a real
        prefill output and the eager sampler step, with the sampler's seeds restored after, as _capture_sampling
        does. Its programs must exist before the draft and sampling traces do."""
        if self.sampling_trace is not None or self.draft_trace is not None:
            raise RuntimeError("The PB prefill-sampling warmup must run before any trace capture")
        gen = self.generator
        # One page per group in distinct nonzero blocks, laid out as the bucket warmup lays them
        # (generator_resident_vllm.py:586-600): the write lands in a free pool block before vLLM allocates any.
        columns = self.max_seq_len // 128
        source = []
        for g in range(max(self.groups) + 1):
            table = torch.zeros(1, columns, dtype=torch.int32)
            table[0, 0] = 1 + g * (min(self.MAX_PREFILL_TOKENS, self.max_seq_len) // 128)
            source.append(table)
        tables = self._tables(kv_cache, [source[g] for g in self.groups], slots=[0])
        out = gen.prefill_forward(torch.zeros(1, 1, dtype=torch.int32), page_table=tables, kv_cache=kv_cache,
                                  prompt_lens=[1], slots=[0], return_device=True, start_positions=[0])
        shape = ttnn.Shape([1, 1, 32, gen.model.config.vocab_size // 4])
        logits = ttnn.reshape(out[0], shape, shape)
        tokens = gen._upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
        gen.configure_sampling(batch=1, temperature=0)
        seeds = ttnn.clone(gen.sampler.seeds_tt_tensor)
        gen._sample_step({"logits": logits, "tokens": tokens})
        ttnn.copy(seeds, gen.sampler.seeds_tt_tensor)
        ttnn.synchronize_device(self.mesh)
        del out, logits, tokens, seeds
        kv_cache.prefill_history.clear()
        self.prefill_slot_keys.clear()

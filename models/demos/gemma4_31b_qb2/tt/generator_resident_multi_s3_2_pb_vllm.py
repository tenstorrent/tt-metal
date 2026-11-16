# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""vLLM adapter: S3-PB, the paged-bucket prefill (PB) on the resident multi-user `batch` adapter S3.2 for option A
(B on resident_m5_11 at max_model_len 262,144; bench/resident/s1/S3X_262K_A_ON_M511_PLAN.md, S3X_262K_PLAN_ADDENDUM_3
§4; bench/packaging/DRAM_PREFILL_FIX_PLAN_v2.md §1.4 "S3-PB on the final S3.x"; LEAD_DECISIONS_PKG P36 defaults).

Gemma4ResidentMultiS32ForCausalLM (generator_resident_multi_s3_2_vllm.py, unchanged) through its PB construction hooks:
- build_model / build_generator return the PB prefill model (prefill_paged.Gemma4ModelPB) and the PB prefill generator
  (generator_resident_spec_vllm_pb.PrefillGeneratorPB, the single-user PB's: GeneratorPBMixin + _PrefillGenerator);
- the generator's prefill_shapes is a PBShapes set: it still records every exact call signature (add, len: S1's
  "DRAM free per chip after prefill" line keeps firing on a new exact shape), but membership answers the PB question,
  so S0's novelty test (generator_resident_multi_vllm.py:421-424, `shape not in generator.prefill_shapes`: release the
  sampling trace before the call, the L1 guard after it) reads the PB key without a copy of S0's or S1's
  prefill_forward;
- the PB guard watches the adapter's sampling trace (the shipped novelty path's retire set: S0 releases only that trace;
  the resident engine's own traces are not released by the shipped adapter either), and its fire handler releases the
  sampling trace and runs S3's _l1_guard (DRAM_PREFILL_FIX_PLAN_v2_ADDENDUM_4: "S3 has its own _l1_guard");
- warmup_model_prefill: S1's (M1's embedding and 128-row bucket warmups, through this adapter's prefill_forward, so
  every first-chunk bucket is a PB first key) with the terminal slices, then the terminal slices of the rewind bucket,
  then _sample_prefill's device ops standalone (the merge for every row count 1..MAX_USERS and the eager sampler step),
  all before warmup_model_decode captures the sampling trace; then the warmed keys. Defaults per P36: bucket warmup on
  (GEMMA4_RESIDENT_WARM_BUCKETS, as gated), continuation pre-warm off (lazy). There is no drafter on this path
  (DraftSource None), so PB3's seed pre-warm does not apply; a drafter present is refused at warmup.
GEMMA4_PREFILL_WARM_CONT=1 is not implemented here and is refused (the B32-PB continuation warm is not ported).
"""

import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_vllm import Gemma4ResidentMultiS32ForCausalLM
from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_vllm import MAX_USERS
from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm_pb import PrefillGeneratorPB
from models.demos.gemma4_31b_qb2.tt.prefill_paged import Gemma4ModelPB, prewarm_merge, prewarm_terminal


class PBShapes(set):
    """generator.prefill_shapes for S3-PB: a set of exact call signatures (the PB generator still adds each one), whose
    membership test is the PB key's: (batch, capacity, lengths, starts, all_logits) is "seen" iff the generator has
    seen every PB key of that call (GeneratorPBMixin.pb_novel)."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def __contains__(self, shape_key):
        _, _, lengths, starts, all_logits = shape_key
        return not self.generator.pb_novel(list(lengths), list(starts), bool(all_logits))


class Gemma4ResidentMultiS32PBForCausalLM(Gemma4ResidentMultiS32ForCausalLM):
    # ------------------------------------------------------------------ construction (S3.2's hooks)

    @classmethod
    def build_model(cls, mesh_device, tile_pad):
        return Gemma4ModelPB(mesh_device, decode_weights=False, prefill_tile_pad=tile_pad)

    @classmethod
    def build_generator(cls, model, max_seq_len):
        generator = PrefillGeneratorPB(model, max_seq_len=max_seq_len)
        generator.prefill_shapes = PBShapes(generator)
        return generator

    def __init__(self, generator, *args, **kwargs):
        super().__init__(generator, *args, **kwargs)
        if not isinstance(generator.prefill_shapes, PBShapes):
            raise TypeError("S3-PB needs the PB generator from build_generator (PBShapes prefill_shapes)")
        generator.pb_trace_probe = lambda: self.sampling_trace is not None
        generator.pb_on_taint = self._pb_release_traces

    def _pb_release_traces(self):
        """A PB guard fire: a prefill built a program under the live sampling trace on a key the adapter did not treat
        as novel, so its own L1 guard after the call was skipped: release the trace and run S3's L1 guard."""
        self._release_sampling_trace()
        self._l1_guard("pb guard fire")

    # ------------------------------------------------------------------ warmup

    def warmup_model_prefill(self, kv_cache, enable_trace, **kwargs):
        gen = self.generator
        first = gen.pb_warmed is None
        if first and os.environ.get("GEMMA4_PREFILL_WARM_CONT", "0") == "1":
            raise ValueError("GEMMA4_PREFILL_WARM_CONT=1 is not implemented for S3-PB (lazy continuation keys, P36)")
        if first and self.drafter is not None:
            raise ValueError("S3-PB serves the plain launch only: a drafter needs PB3's seed pre-warm")
        model = gen.model
        t0 = time.perf_counter()
        model.pb_warm_terminal = True
        try:
            super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)
        finally:
            model.pb_warm_terminal = False
        if not first:
            return
        t1 = time.perf_counter()
        prewarm_terminal(model)
        self._pb_warm_prefill_sampling(kv_cache)
        gen.pb_seen |= {("rows", k) for k in range(1, MAX_USERS + 1)}
        gen.pb_mark_warmed()
        logger.info("S3-PB prefill warmup: S1/M1 warmups {:.1f} s, terminal + sampling merge 1..{} {:.1f} s; "
                    "continuations lazy", t1 - t0, MAX_USERS, time.perf_counter() - t1)

    def _pb_warm_prefill_sampling(self, kv_cache):
        """_sample_prefill's device ops (generator_resident_vllm.py:309-321) for every row count, standalone (no sampling
        params, no trace): from one real 1-row device prefill output, the merge for k = 1..MAX_USERS (concat, pad to 32,
        reshape; prefill_paged.prewarm_merge), then the eager sampler step on the merged logits with the sampler's
        seeds restored after, as _capture_sampling does (the single-user PB's _pb_warm_prefill_sampling, for k rows).
        Its programs must exist before the sampling trace does."""
        if self.sampling_trace is not None:
            raise RuntimeError("The S3-PB prefill-sampling warmup must run before any trace capture")
        gen = self.generator
        limit = min(self.MAX_PREFILL_TOKENS, self.max_seq_len)
        columns, groups = self.max_seq_len // 128, max(self.groups) + 1
        # One page per group in distinct nonzero blocks, laid out as M1's bucket warmup lays them
        # (generator_resident_vllm.py:586-596): the write lands in a free pool block before vLLM allocates any.
        source = []
        for g in range(groups):
            table = torch.zeros(1, columns, dtype=torch.int32)
            table[0, 0] = 1 + g * (limit // 128)
            source.append(table)
        tables = self._tables(kv_cache, [source[g] for g in self.groups], slots=[0])
        out = gen.prefill_forward(torch.zeros(1, 1, dtype=torch.int32), page_table=tables, kv_cache=kv_cache,
                                  prompt_lens=[1], slots=[0], return_device=True, start_positions=[0])[0]
        logits_like = gen.model.tensor(torch.zeros(1, 1, 32, gen.model.config.vocab_size), dim=3)
        prewarm_merge(gen, out, logits_like, MAX_USERS)
        shape = ttnn.Shape([1, 1, 32, gen.model.config.vocab_size // 4])
        tokens = gen._upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
        seeds = ttnn.clone(gen.sampler.seeds_tt_tensor)
        # The eager sampler step on each merged-logits form _sample_prefill makes: k = 1 (outputs[0] reshaped), 1 < k < 32
        # (concat, then pad to 32) and k = 32 (concat only), in case the step's program depends on its input's form.
        for k in (1, 2, MAX_USERS):
            merged = out if k == 1 else ttnn.concat([out] * k, dim=2)
            if 1 < k < 32:
                merged = ttnn.pad(merged, [(0, 0), (0, 0), (0, 32 - k), (0, 0)], value=0.0)
            gen.configure_sampling(batch=k, temperature=0)
            gen._sample_step({"logits": ttnn.reshape(merged, shape, shape), "tokens": tokens})
            ttnn.copy(seeds, gen.sampler.seeds_tt_tensor)
            del merged
        ttnn.synchronize_device(self.mesh)
        del out, logits_like, tokens, seeds
        kv_cache.prefill_history.clear()
        self.prefill_slot_keys.clear()

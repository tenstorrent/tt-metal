# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM serving adapter for the TTNN GLM-4.7-Flash full model (one Blackhole p150 chip).

Thin translation layer only: every actual forward/cache/sampling primitive is
``tt/generator.py``'s ``GLM47FlashGenerator``. This module owns nothing about
the model itself -- only vLLM's kwarg shapes, slot/row bookkeeping, and the
async decode split.

Row vs. slot, and why ``slot_remap`` is unconsumed
===================================================

This adapter never tracks "which request owns row N" across calls. Each call
addresses the model with whatever row order *that call's* tensors use:
prefill writes into ``empty_slots[i]`` (the physical row vLLM just assigned
request ``i``); decode writes into ``[0, sz)`` (vLLM's own front-packed active
rows for this step, which is not necessarily the same numbering a continuing
request had last step after a condense).

That is only safe because of the reset boundary discipline below, not because
row and slot are believed equal for a request's whole life:

* Persistent per-row *device* state (the split sampler's token/position
  tensors) is only ever trusted across consecutive calls where
  ``reset_batch=False`` -- and by construction, a scheduler-layout change
  (admission, removal, or condense reorder) always sets ``reset_batch=True``
  for that step (``vllm_tt_plugin/model_runner.py``:
  ``reset_batch = self._decode_layout_changed_since_last_decode``). So within
  any unbroken ``reset_batch=False`` stretch, row assignment provably did not
  change, and nothing needs remapping.
* On every ``reset_batch=True`` step this adapter fully rewrites every active
  row's token, position, and sampling-params state from vLLM's own current
  arrays (:meth:`GLM47FlashForCausalLM.decode_forward`), discarding any
  dependence on what those rows meant before. There is no stale device value
  left to have scrambled.
* KV cache content is addressed purely through page-table *values*, which are
  refreshed from vLLM's own current block table every call regardless of
  ``reset_batch`` -- request-owned, never row-identity-owned -- so cache
  correctness never depends on row/slot continuity in the first place.

Per-request seed reproducibility (added in VS-007, doc/vllm_integration/work_log.md)
does not need ``slot_remap`` for the row-reassignment case specifically:
``GLM47FlashGenerator.apply_decode_sampling_state`` anchors each seeded
request's RNG counter to its absolute decode position (``start_pos``), not to
a persisted per-slot counter, so a condense moving a request to a different
physical row does not by itself break that request's reproducibility -- see
that method's own docstring for the mechanism. This is a narrower, more
precise fix than a blanket "seed continuity across condense is dropped"
claim would suggest, and it does NOT close the separate, still-open,
upstream-tracked full-occupancy determinism defect described in "Known
limitations" (README.md / work_log.md VS-009): that defect reproduces even
without any condense, is not yet root-caused to this mechanism or to any
other, and remains a known limitation of the `--sampling-profile full`
evidence, not of this adapter's condense handling specifically.

Async decode
============

``supports_async_decode=True``. Steady state (``reset_batch=False``) never
writes host tokens/positions: it replays :meth:`GLM47FlashGenerator.decode_step_traced`,
which reads whatever the *previous* step's on-device split sampler already
wrote into the persistent token/position tensors, then samples the next one
the same way. ``decode_forward(read_from_device=False)`` returns that
persistent device tensor directly; :meth:`read_decode_output` issues the
non-blocking ``cpu(blocking=False)`` + event, and :meth:`process_decode_output_host`
does the final ``ttnn.to_torch`` after the caller has synchronized that event.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import GLM47FlashGenerator, build_generator
from models.autoports.zai_org_glm_4_7_flash.tt.model import DEFAULT_HF_MODEL_ID

_MODEL_DIR = Path(__file__).resolve().parent.parent
_CONTRACT_PATH = _MODEL_DIR / "doc" / "context_contract.json"

#: vLLM-serving-only override of the model's default prefill_chunk_size (2048)
#: and prefill_buckets (128, 256, 512, 1024, 2048). Not a precision/fidelity
#: knob (does not touch doc/datatype_sweep/selected_precision_config.json's
#: contract), just a compute/memory-layout one: FusedDecoder.prefill_forward
#: (tt/fused_decoder.py) splits any prompt into whole prefill_chunk_size chunks
#: plus a bucketed tail regardless of this value, so the full 202752-token
#: context is still reachable, just via more, smaller chunks -- spec-preserving.
#: The MoE gate_up prefill transpose (tt/fused_decoder.py's
#: ``_moe_prefill``, ``gu = ttnn.transpose(gu, 1, 3)``) materializes a
#: transient DRAM scratch buffer that scales linearly with chunk length
#: (measured: 402,653,184 B (384 MiB) at chunk=1024, 805,306,368 B (768 MiB)
#: at chunk=2048 -- exactly 2x for 2x the chunk). Under the readiness harness's
#: batch-1, single-request DRAM footprint this fits at chunk=2048 with room to
#: spare; under vLLM's 32-concurrency/202752-context KV pool (which the fixed
#: cache/weights/scratch budget leaves only ~0.75 GiB of headroom for
#: everything transient) it does not -- the 768 MiB peak OOM'd on real hardware
#: even after covering the smaller 384 MiB peak (doc/vllm_integration/work_log.md
#: VS-005, VS-006). Halving the chunk cap halves this specific transient peak;
#: 1024 is the largest value proven end-to-end on hardware in that same run
#: (every op for every warmup bucket up to and including 1024 completed; only
#: the next bucket, 2048, failed).
VLLM_PREFILL_CHUNK_SIZE = 1024
VLLM_PREFILL_BUCKETS = (128, 256, 512, 1024)


#: A/B knob for the optimized-vLLM stage's before/after measurement, and an
#: escape hatch if the compact decode path ever needs to be disabled in the
#: field without a code change. ``0``/``false``/``no`` rebuilds the generator on
#: the previous union decode MoE (one decode trace, no kc buckets); anything
#: else (including unset) keeps the shipped bucketed compact path. The two arms
#: differ only in this flag, which is what makes the recorded before/after a
#: same-harness comparison rather than a comparison against an older commit.
MOE_COMPACT_ENV = "GLM47_VLLM_MOE_COMPACT"

#: Second A/B knob for the same stage: compile every decode slot's prefill
#: program during warm-up. ``0`` reproduces the pre-stage behaviour where the
#: first request admitted into each slot compiles under a live decode trace and
#: forces a full decode-trace recapture. The two knobs are independent, which is
#: what lets the stage report attribute the single-user decode win and the
#: serving-burst TTFT win to the right change from one commit.
PREFILL_SLOT_WARM_ENV = "GLM47_VLLM_PREFILL_SLOT_WARM"

#: How many traced decode steps between one-line counter dumps into the server
#: log. This is the live-server evidence that the measured serving path really
#: is traced token-out decode with on-device sampling: a dump showing
#: ``eager_decode=0 eager_sampling=0 full_logits_readbacks=0 host_argmax=0
#: page_table_refreshes=0`` over a window of N steps is a fact from the running
#: server, not an inference from unit tests. The default lines up with both
#: readiness benchmark profiles (128-token single-user, 100-token burst) so each
#: produces exactly one window. Set to 0 to disable.
COUNTER_LOG_EVERY_ENV = "GLM47_VLLM_DECODE_COUNTER_LOG_EVERY"
DEFAULT_COUNTER_LOG_EVERY = 100


def _moe_decode_compact_enabled() -> bool:
    return os.environ.get(MOE_COMPACT_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _prefill_slot_warmup_enabled() -> bool:
    return os.environ.get(PREFILL_SLOT_WARM_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _load_context_contract() -> dict:
    return json.loads(_CONTRACT_PATH.read_text())


class GLM47FlashForCausalLM:
    """vLLM TT-plugin model wrapper. Registered as ``TTGlm4MoeLiteForCausalLM``."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_sample_on_device": True,
        "supports_async_decode": True,
    }

    def __init__(self, generator: GLM47FlashGenerator):
        self.generator = generator
        self.mesh_device = generator.mesh_device
        self.max_batch_size = generator.max_batch_size
        self.blocks_per_user = generator.model.blocks_per_user
        # Host mirror of the persistent, slot-indexed page table: vLLM hands
        # us a fresh torch tensor every call, row-ordered per its own request
        # order (prefill: admission order; decode: front-packed active rows).
        # This mirror is always the full [max_batch_size, blocks_per_user]
        # shape the generator's bound page table expects, so unaffected rows
        # (other live slots at prefill time, inactive rows at decode time)
        # survive an in-place row overwrite untouched.
        self._pt_mirror = torch.zeros((self.max_batch_size, self.blocks_per_user), dtype=torch.int32)
        self._cache_bound = False
        #: Page-table refresh accounting, so "the steady-state decode loop
        #: performs no page-table copies" is a counted fact rather than a claim.
        #: ``skipped`` is a decode/prefill call whose rows were already the ones
        #: on device; ``written`` is a call that actually re-uploaded the table.
        self.page_table_calls_skipped = 0
        self.page_table_calls_written = 0
        try:
            self._counter_log_every = int(os.environ.get(COUNTER_LOG_EVERY_ENV, DEFAULT_COUNTER_LOG_EVERY))
        except ValueError:
            self._counter_log_every = DEFAULT_COUNTER_LOG_EVERY
        self._decode_calls = 0
        self._counter_snapshot = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: Optional[int] = None,
        tt_data_parallel: int = 1,
        optimizations=None,
        **kwargs: Any,
    ) -> "GLM47FlashForCausalLM":
        if tt_data_parallel != 1:
            raise ValueError(
                f"GLM-4.7-Flash vLLM adapter targets a single Blackhole p150 chip (tt_data_parallel=1); "
                f"got tt_data_parallel={tt_data_parallel}"
            )
        if max_batch_size > 32:
            raise ValueError(
                f"GLM-4.7-Flash's decode batch is a single 32-row sampler tile (doc/context_contract.json "
                f"full_model.batch_contract.largest_tested_batch=32); requested max_batch_size={max_batch_size} "
                f"exceeds the hard physical/measured limit."
            )
        hf_model_id = (
            getattr(hf_config, "_name_or_path", None) or getattr(hf_config, "name_or_path", None) or DEFAULT_HF_MODEL_ID
        )
        # Always built with the full 32-row physical tile: the decoder pins its
        # per-slot shard grids at construction (see GLM47FlashGenerator.bind_decode_state),
        # so a narrower construction cannot later be widened. vLLM's own
        # max_batch_size (<=32, checked above) just bounds how many of those 32
        # rows the scheduler ever activates.
        generator = build_generator(
            _MODEL_DIR,
            mesh_device,
            hf_model_id=hf_model_id,
            hf_config=hf_config,
            max_batch_size=32,
            max_seq_len=max_seq_len,
            defer_cache_and_traces=True,  # vLLM owns the cache; see build_generator's docstring
            enable_sampling=True,
            host_sampling=False,
            # Capped for this serving config's DRAM headroom; see
            # VLLM_PREFILL_CHUNK_SIZE's module-level docstring (VS-006).
            prefill_chunk_size=VLLM_PREFILL_CHUNK_SIZE,
            prefill_buckets=VLLM_PREFILL_BUCKETS,
            moe_decode_compact=_moe_decode_compact_enabled(),
            prefill_slot_warmup=_prefill_slot_warmup_enabled(),
        )
        logger.info(
            "GLM-4.7-Flash vLLM adapter: compact decode MoE {} (buckets={}); per-slot prefill warm {}; {}={} {}={}",
            "on" if generator.moe_decode_compact else "off",
            generator._decode_kc_buckets,
            "on" if generator.prefill_slot_warmup else "off",
            MOE_COMPACT_ENV,
            os.environ.get(MOE_COMPACT_ENV, "<unset, default on>"),
            PREFILL_SLOT_WARM_ENV,
            os.environ.get(PREFILL_SLOT_WARM_ENV, "<unset, default on>"),
        )
        return cls(generator)

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Total-token KV-cache budget across all users, derived from the committed
        DRAM budget in ``doc/context_contract.json`` (measured, not modeled): the
        allocatable DRAM minus weights+persistent-scratch, sampler penalty buffers,
        and the reserved trace region, divided by the bf8 latent-cache bytes/token
        across all 47 layers -- *plus* one layer's worth of cache-reset zero buffer
        (``GLM47FlashModel._cache_zeros``/``prepare_cache_reset`` allocate exactly
        one paged cache tensor's worth to zero every layer via device-to-device
        copy). That buffer scales with vLLM's actual pool size, not the fixed
        202752-context-per-user figure baked into ``weights_plus_persistent_scratch``
        (measured there at 0.116 GiB against a 5.431 GiB cache -- a 47x smaller pool
        than the ~13.6 GiB multi-user pool this budget targets), so it must be
        reserved here as a function of the token count being solved for, not read
        as a constant: solving ``weights+scratch+sampler+trace + T*(bytes_all_layers
        + bytes_one_layer) <= allocatable`` for ``T``. Missing this term
        under-reserved by ~0.18 GiB and OOM'd during `allocate_kv_cache`'s
        `bind_decode_state -> prepare_cache_reset -> _cache_zeros` on real hardware
        (doc/vllm_integration/work_log.md VS-004). See ``full_model.dram_budget_gib``
        / ``full_model.kv_cache_bytes_per_token_all_47_layers_bf8`` /
        ``full_model.kv_cache_bytes_per_token_per_layer_bf8`` in that file.
        """
        contract = _load_context_contract()["full_model"]
        budget = contract["dram_budget_gib"]
        headroom_gib = (
            contract["measured_allocatable_dram_gib"]
            - budget["weights_plus_persistent_scratch"]
            - budget["sampler_penalty_buffers"]
            - budget["trace_region_reserved"]
        )
        bytes_per_token_all_layers = contract["kv_cache_bytes_per_token_all_47_layers_bf8"]
        bytes_per_token_one_layer = contract["kv_cache_bytes_per_token_per_layer_bf8"]
        # Safety margin covers two distinct, measured shortfalls -- not a guess:
        #
        # 1. Bank-rounding (~0.04 GiB): the allocator divides buffers across 8 DRAM
        #    banks and this budget is a whole-device average, not a per-bank one,
        #    so rounding/alignment can make a bank-local allocation fail a hair
        #    before the device-wide theoretical limit (first observed: needed
        #    ~37 MiB more on one bank at zero margin).
        # 2. MoE prefill-warmup activation scratch (~0.38 GiB): warmup_model_prefill
        #    (-> GLM47FlashGenerator.warmup_prefill -> the model's MoE routing path)
        #    transiently allocates a DRAM scratch buffer for the gate_up projection's
        #    post-sparse-matmul transpose (tt/fused_decoder.py's
        #    ``gu = ttnn.transpose(gu, 1, 3)``, one call site, ~384 MiB observed).
        #    This runs once per prefill-bucket shape regardless of the vLLM KV-cache
        #    size, so it is not itself part of the cache budget -- but it must fit
        #    in whatever headroom is left after weights+scratch+sampler+trace+cache,
        #    which this function's own token count controls. At the unmargined
        #    487,379-token budget this scratch buffer had nowhere to land: engine
        #    core died with TT_FATAL bank_manager.cpp:462 Out of Memory allocating a
        #    402,653,184 B (384 MiB) buffer at that exact transpose call
        #    (doc/vllm_integration/work_log.md VS-005).
        #
        # 0.75 GiB total covers both with margin, still measured against a real
        # observed failure rather than an arbitrarily large guess.
        safety_margin_gib = 0.75
        total_tokens = int(
            (headroom_gib - safety_margin_gib) * (1024**3) / (bytes_per_token_all_layers + bytes_per_token_one_layer)
        )
        logger.info(
            "GLM-4.7-Flash get_max_tokens_all_users: {} GiB headroom ({} GiB safety margin) / "
            "{} B/token (all-layers + one-layer zero-buffer) -> {} tokens "
            "(model_name={}, num_devices={}, max_model_len={}, max_num_seqs={})",
            round(headroom_gib, 3),
            safety_margin_gib,
            bytes_per_token_all_layers + bytes_per_token_one_layer,
            total_tokens,
            model_name,
            num_devices,
            max_model_len,
            max_num_seqs,
        )
        return max(total_tokens, 0)

    # ------------------------------------------------------------------ KV cache (vLLM-owned)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Build the 47 per-layer paged latent caches at vLLM's block geometry.

        ``dtype`` (a torch dtype from ``cache_config.cache_dtype``) is not a real
        ttnn block-float type and is intentionally ignored: the datatype-sweep
        selected policy (bfloat8_b, doc/datatype_sweep/selected_precision_config.json)
        is what serving must use, the same override tt_transformers'
        ``allocate_vllm_kv_cache_per_layer`` makes for its own optimizations-derived
        dtype. ``kv_cache_shape`` is ``(num_blocks, num_kv_heads, block_size, head_size)``;
        vLLM detects this model as MLA from HF's ``glm4_moe_lite`` model_type
        (vllm/transformers_utils/model_arch_config_convertor.py), giving
        num_kv_heads=1, head_size=kv_lora_rank+qk_rope_head_dim=576 -- exactly this
        model's paged latent cache entry.
        """
        model = self.generator.model
        num_blocks, num_kv_heads, block_size, head_size = (int(v) for v in kv_cache_shape)
        if num_layers != len(model.layers):
            raise ValueError(f"vLLM asked for {num_layers} KV-cache layers; model has {len(model.layers)}")
        if num_kv_heads != 1 or head_size != model.layers[0].kvpe_dim:
            raise ValueError(
                f"unexpected MLA cache shape from vLLM: num_kv_heads={num_kv_heads}, head_size={head_size}; "
                f"expected (1, {model.layers[0].kvpe_dim})"
            )
        if block_size != model.paged_config.block_size:
            raise ValueError(
                f"vLLM block_size={block_size} does not match this model's fixed paged-cache block_size="
                f"{model.paged_config.block_size}; pass --block-size {model.paged_config.block_size} to vllm serve"
            )
        logger.info(
            "GLM-4.7-Flash allocate_kv_cache: overriding vLLM's requested cache dtype {} with the selected "
            "policy {} (doc/datatype_sweep/selected_precision_config.json); num_blocks={} block_size={} "
            "head_size={}",
            dtype,
            model.cache_dtype,
            num_blocks,
            block_size,
            head_size,
        )
        # Same shared PagedCacheConfig object every decoder layer already holds
        # (assigned once, by reference, in GLM47FlashModel.from_pretrained) --
        # mutating it in place repoints every layer's own reference at once.
        model.paged_config.max_num_blocks = num_blocks
        # blocks_per_user is a PAGE-TABLE-WIDTH bound (the most blocks any one
        # request's own block list can need), not an equal-share quota of the
        # pool -- vLLM's paged allocator lets one request use far more than
        # num_blocks/max_batch_size blocks as long as the *sum* in flight fits
        # num_blocks; that admission accounting is vLLM's, not this adapter's.
        # The correct width is exactly what GLM47FlashModel.from_pretrained
        # already computed from the model's own max_seq_len (cdiv(max_seq_len,
        # block_size)) before this method ever ran. An earlier version instead
        # set it to ``num_blocks // max_batch_size`` (230 at this run's
        # num_blocks=7362), which is unrelated to how wide a single request's
        # block list can legitimately be: it silently truncated
        # _write_page_table_rows to 230 columns (14,720 tokens) while still
        # advertising max_model_len=202752, and max_seq_len_physical (used to
        # clamp prefill_physical_len) inherited the same wrong, smaller bound
        # -- a real capability reduction with no error, no evidence, and no
        # hard physical limit behind it (doc/vllm_integration/work_log.md
        # VS-011, caught by $stage-review). Recomputing it here from
        # max_seq_len (not num_blocks) fixes both call sites at once and holds
        # regardless of what num_blocks vLLM ends up choosing.
        blocks_per_user = -(-model.max_seq_len // block_size)
        if num_blocks < blocks_per_user:
            raise ValueError(
                f"vLLM's KV-cache pool ({num_blocks} blocks) is smaller than what a single request at "
                f"max_seq_len={model.max_seq_len} needs ({blocks_per_user} blocks); no request could ever "
                f"reach the advertised context. This is a hard physical limit, not something this adapter "
                f"can paper over -- get_max_tokens_all_users must report a smaller max_model_len-compatible "
                f"budget, or max_model_len must be reduced with evidence, not silently truncated per request."
            )
        model.blocks_per_user = blocks_per_user
        self.blocks_per_user = blocks_per_user
        self._pt_mirror = torch.zeros((self.max_batch_size, self.blocks_per_user), dtype=torch.int32)
        kv_cache = model.allocate_kv_cache(dtype=model.cache_dtype)

        # A clone, not self._pt_mirror itself: bind_decode_state stores whatever
        # object it is given as the "previous value" for refresh_page_table's
        # only_if_changed diff (torch.as_tensor does not copy an already-int32
        # tensor). Binding the live, mutable mirror directly would alias that
        # stored snapshot to it, so every later in-place mutation of the mirror
        # would be invisible to the diff from the very first call onward.
        self.generator.bind_decode_state(kv_cache=kv_cache, page_table=self._pt_mirror.clone())
        self.generator._ensure_owned_state()
        self._cache_bound = True
        # Compiling and trace-capture are the plugin's own dedicated warmup
        # entry points (warmup_model_prefill/warmup_model_decode below), called
        # from vllm_tt_plugin/model_runner.py's warmup_model() right after this
        # returns -- not done here, so this method stays what the contract says
        # it is ("Allocate or bind vLLM-owned attention KV cache"), and so nothing
        # here duplicates program compilation warmup_model_decode's own warm
        # pass already does before it captures.
        return kv_cache

    def warmup_model_prefill(self, *, kv_cache, can_sample_on_device: bool, enable_trace: bool, **kwargs: Any) -> None:
        """Compile every prefill-bucket program shape ahead of the first request.

        Called twice by the plugin (``enable_trace=False`` then ``True``); this
        model's prefill path is never traced (see the DeepSeek-V3 TT adapter's
        identical note), so both calls do the same idempotent compile sweep --
        cheap the second time since everything is already in the program cache.
        """
        self._check_cache(kv_cache)
        self.generator.warmup_prefill()

    def warmup_model_decode(
        self,
        *,
        kv_cache,
        max_batch_size: int,
        num_blocks: int,
        can_sample_on_device: bool,
        enable_trace: bool,
        **kwargs: Any,
    ) -> None:
        """Compile the decode program (phase 1) and capture the decode + split-
        sampling traces (phase 2, ``enable_trace=True``).

        ``capture_decode_trace`` already runs its own uncaptured warm pass
        immediately before capturing (Metal refuses to capture a program that
        is not already compiled), so phase 1 (``enable_trace=False``) is a
        no-op here: there is nothing this model's decode path needs compiled
        that phase 2's own warm pass will not compile anyway, and calling
        capture twice is a safe no-op (``capture_decode_trace`` returns early
        once ``_decode_trace_id`` is set).
        """
        self._check_cache(kv_cache)
        if not enable_trace:
            return
        self.generator.capture_decode_trace()
        self.generator.reset()

    def _check_cache(self, kv_cache) -> None:
        if kv_cache is None:
            return
        if any(entry is None for entry in kv_cache):
            return
        if not self._cache_bound:
            raise RuntimeError("allocate_kv_cache must run before prefill_forward/decode_forward")
        # vLLM allocates the serving cache exactly once at worker init and passes
        # the same list back every call; this is a same-object sanity check, not
        # a standalone re-allocation path (goal: "no hidden standalone-cache
        # assumptions").
        if kv_cache is not self.generator._bound_cache and list(kv_cache) != list(self.generator._bound_cache):
            raise RuntimeError(
                "GLM-4.7-Flash vLLM adapter does not support rebinding to a different KV cache after "
                "allocate_kv_cache; vLLM should allocate the serving cache once at worker init."
            )

    def _write_page_table_rows(self, rows: torch.Tensor, at: List[int]) -> None:
        """Scatter vLLM's row-ordered block table into the slot-ordered mirror."""
        rows = rows.to(torch.int32)
        if rows.shape[1] > self.blocks_per_user:
            # Must never silently drop columns: a wider table than
            # blocks_per_user (cdiv(max_seq_len, block_size), see
            # allocate_kv_cache) means some request needs more blocks than the
            # advertised max_seq_len allows for, or blocks_per_user was sized
            # wrong again -- either way a truncated write would address the
            # wrong physical blocks for the tail of that request's context
            # (doc/vllm_integration/work_log.md VS-011).
            raise ValueError(
                f"page table has {rows.shape[1]} block columns but this model's per-request page-table "
                f"width is {self.blocks_per_user} (cdiv(max_seq_len={self.generator.model.max_seq_len}, "
                f"block_size={self.generator.model.paged_config.block_size})); refusing to truncate."
            )
        width = rows.shape[1]
        # Diff per row against the mirror (which is exactly what is on device)
        # and do nothing at all when nothing moved. The steady-state decode
        # loop is the unchanged case: vLLM re-sends the same block list every
        # token and only extends it when a request crosses a 64-token block
        # boundary, so this makes the common step cost one comparison instead
        # of a full-width host copy plus a mirror clone plus the generator's
        # own diff.
        changed = False
        for i, slot in enumerate(at):
            row = rows[i, :width]
            if not torch.equal(self._pt_mirror[slot, :width], row):
                self._pt_mirror[slot, :width] = row
                changed = True
        if not changed:
            self.page_table_calls_skipped += 1
            return
        self.page_table_calls_written += 1
        # Pass a copy, not self._pt_mirror itself: refresh_page_table stores
        # whatever object it is handed as the "previous value" for its own
        # only_if_changed diff (torch.as_tensor does not copy an already-int32
        # tensor), so handing it the SAME mutable mirror object would alias
        # that snapshot to the live mirror and make every later in-place edit
        # invisible to it. The clone only happens on a real change now, so it
        # is off the per-token path entirely.
        self.generator.refresh_page_table(self._pt_mirror.clone(), only_if_changed=False)

    # ------------------------------------------------------------------ prefill

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,
        prompt_lens: List[int],
        page_table: torch.Tensor,
        kv_cache,
        start_pos=None,
        sampling_params=None,
        empty_slots: Optional[List[int]] = None,
        enable_trace: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._check_cache(kv_cache)
        if start_pos is not None and any(int(p) != 0 for p in start_pos):
            raise ValueError(
                "GLM-4.7-Flash's paged-latent prefill always starts a fresh request at position 0 "
                "(no resumed/chunked-prefill continuation), matching the DeepSeek-V3 TT adapter's "
                "identical MLA-family limitation; got non-zero start_pos in a prefill call. Launch "
                "the server without chunked prefill for this model."
            )
        num_rows = len(prompt_lens)
        slots = list(empty_slots) if empty_slots is not None else list(range(num_rows))
        if len(slots) != num_rows:
            raise ValueError(f"empty_slots has {len(slots)} entries for {num_rows} prefill rows")

        pt_rows = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        self._write_page_table_rows(pt_rows, slots)

        if sampling_params is None:
            # Host-sampling compatibility path (e.g. a logprobs request on this
            # single-chip mesh forces the whole batch's step to host-sample --
            # see vllm_tt_plugin/model_runner.py's check_perform_device_sampling).
            # Reuses the generator's own low-level multi-user prefill contract
            # (last-position logits only); no bespoke sampling here. Does not
            # touch the persistent per-slot decode state -- if this same slot
            # later decodes, that step is necessarily a reset_batch=True step
            # (a fresh admission), which fully (re)writes it from host state.
            # ``user_ids`` addresses absolute physical slots into a page table,
            # so this must pass the full [max_batch_size, blocks_per_user]
            # persistent mirror (just refreshed above), not vLLM's own
            # row-compacted table -- the low-level contract indexes page_table
            # rows by user_id directly, not by call-local row position.
            return self.generator.prefill_forward(
                tokens=tokens,
                page_table=self.generator._page_table_dev,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                user_ids=slots,
                return_all_logits=False,
            )

        out_tokens = torch.zeros(num_rows, dtype=torch.int64)
        for i, slot in enumerate(slots):
            plen = int(prompt_lens[i])
            if plen <= 0:
                continue
            prompt = tokens[i, :plen].tolist()
            row_params = _slice_sampling_params_row(sampling_params, i)
            self.generator.apply_prefill_sampling_state(row_params, empty_slots=[slot])
            token = self.generator.prefill_and_sample(prompt, user_id=slot, recapture=True)
            out_tokens[i] = int(token)
        return out_tokens

    # ------------------------------------------------------------------ decode

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params=None,
        reset_batch: bool = False,
        prompt_tokens: Optional[torch.Tensor] = None,
        output_tokens: Optional[torch.Tensor] = None,
        slot_remap=None,  # intentionally unconsumed; see module docstring
        **kwargs: Any,
    ):
        self._check_cache(kv_cache)
        toks = tokens.reshape(-1)
        pos = start_pos.reshape(-1)
        sz = toks.numel()
        if sz == 0:
            return torch.empty((0, 1), dtype=torch.int64)
        rows = list(range(sz))  # row == physical slot (see module docstring)

        pt_rows = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        self._write_page_table_rows(pt_rows, rows)

        if not enable_trace:
            # --enforce-eager / debugging compatibility path only: the low-level
            # eager decode_forward writes host tokens/positions every call by
            # design, which is fine off the measured/traced path. Pass the
            # already-bound device page-table tensor (just refreshed above via
            # _write_page_table_rows): the low-level API's torch-tensor branch
            # expects a full [max_batch_size, blocks_per_user] host tensor, not
            # a caller-sliced [sz, blocks_per_user] one.
            out = self.generator.decode_forward(
                toks,
                pos,
                page_table=self.generator._page_table_dev,
                kv_cache=self.generator._bound_cache,
                enable_trace=False,
                return_logits=(sampling_params is None),
            )
            if sampling_params is None:
                return out.unsqueeze(1)  # [sz, 1, vocab] host-sampling logits
            return out.to(torch.int64).unsqueeze(1)

        if reset_batch:
            if sampling_params is not None:
                # start_pos=pos (row-ordered live positions, all >= 0 and front-packed
                # into rows [0, sz)) lets apply_decode_sampling_state register and
                # position-align per-request seeds at the decode rows they will occupy.
                self.generator.apply_decode_sampling_state(
                    sampling_params,
                    reset_batch=True,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    start_pos=pos,
                )
            toks_full = [0] * self.max_batch_size
            pos_full = [-1] * self.max_batch_size
            for i in range(sz):
                toks_full[i] = int(toks[i].item())
                pos_full[i] = int(pos[i].item())
            self.generator.set_decode_tokens(toks_full)
            self.generator.set_decode_positions(pos_full)

        if sampling_params is None:
            # Host-sampling compatibility path (e.g. a logprobs request on this
            # single-chip mesh, where vLLM itself forces host sampling -- see
            # vllm_tt_plugin/model_runner.py's check_perform_device_sampling).
            # Still drives the traced model graph; only the on-device split
            # sampler is skipped, and vLLM's own host sampler consumes the
            # returned logits.
            out = self.generator.decode_forward(
                toks,
                pos,
                page_table=self.generator._page_table_dev,
                kv_cache=self.generator._bound_cache,
                enable_trace=True,
                return_logits=True,
            )
            return out.unsqueeze(1)  # [sz, 1, vocab]

        self._maybe_log_counters()
        # Steady-state async path: no host token/position write here when
        # reset_batch is False -- decode_step_traced() replays the model trace
        # (which advances its own device-resident position) and the split
        # sampler (which reads/writes the persistent token tensor), using
        # whatever the *previous* step already wrote on device.
        self.generator.decode_step_traced()
        if not read_from_device:
            return self.generator.decode_token_output  # raw ttnn.Tensor; deferred read below
        tokens_out = self.generator.read_decode_tokens(self.max_batch_size)
        return torch.tensor(tokens_out, dtype=torch.int64).unsqueeze(1)

    # ------------------------------------------------------------------ async decode split

    def _maybe_log_counters(self) -> None:
        """One line per ``_counter_log_every`` traced decode steps, as deltas."""
        self._decode_calls += 1
        if self._counter_log_every <= 0 or self._decode_calls % self._counter_log_every:
            return
        gen = self.generator
        now = {
            "model_trace_replays": gen.counters["model_trace_replays"],
            "sampling_trace_replays": gen.counters["sampling_trace_replays"],
            "eager_decode_steps": gen.counters["eager_decode_steps"],
            "eager_sampling_steps": gen.counters["eager_sampling_steps"],
            "full_logits_readbacks": gen.counters["full_logits_readbacks"],
            "host_argmax_calls": gen.counters["host_argmax_calls"],
            "token_input_refreshes": gen.counters["token_input_refreshes"],
            "position_refreshes": gen.counters["position_refreshes"],
            "page_table_refreshes": gen.counters["page_table_refreshes"],
            "trace_recaptures": gen.counters["trace_recaptures"],
            "decode_trace_bucket_switches": gen.counters["decode_trace_bucket_switches"],
            "token_readbacks": gen.counters["token_readbacks"],
            "page_table_calls_written": self.page_table_calls_written,
            "page_table_calls_skipped": self.page_table_calls_skipped,
        }
        prev = self._counter_snapshot or dict.fromkeys(now, 0)
        delta = {k: now[k] - prev.get(k, 0) for k in now}
        self._counter_snapshot = now
        logger.info(
            "GLM-4.7-Flash decode counters over the last {} decode calls (deltas): {} | kc_replays(total)={}",
            self._counter_log_every,
            " ".join(f"{k}={v}" for k, v in delta.items()),
            {str(k): v for k, v in gen.kc_replays.items()},
        )

    def read_decode_output(self, tt_out, async_read: bool = False):
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        if not isinstance(tt_out, ttnn.Tensor):
            raise TypeError(f"unsupported decode output type from GLM47FlashForCausalLM: {type(tt_out)}")
        if not async_read:
            return ttnn.to_torch(tt_out).reshape(-1).to(torch.int64).unsqueeze(1)
        host = tt_out.cpu(blocking=False)
        event = ttnn.record_event(self.mesh_device, 0)
        return host, [event]

    def process_decode_output_host(self, tt_out, is_tokens: bool = True):
        if isinstance(tt_out, torch.Tensor):
            return tt_out
        return ttnn.to_torch(tt_out).reshape(-1).to(torch.int64).unsqueeze(1)


def _slice_sampling_params_row(sampling_params, i: int):
    """Pull row ``i``'s scalar values out of vLLM's per-row ``TTSamplingParams``
    (or this package's ``SamplingParams``), so a per-user prefill sample call
    (one user at a time; see ``GLM47FlashForCausalLM.prefill_forward``) uses
    that user's own temperature/top_k/top_p/penalties rather than another
    concurrently-admitted request's.

    ``seed`` threads through unchanged. Prefill registers it via
    ``apply_prefill_sampling_state`` -> ``SamplingGenerator.apply_prefill_state``
    (which drives ``seed_manager.reset_seed`` + ``get_new_values``), and decode
    advances it every step: ``decode_forward`` passes ``start_pos`` into
    ``apply_decode_sampling_state`` (which re-registers and position-aligns the
    seed on a batch reset) and ``decode_step_traced`` calls
    ``seed_manager.get_new_values()`` before each ``sample``. A seeded request
    therefore reproduces regardless of its batch neighbours or physical row.
    """
    from dataclasses import fields, replace

    def _at(value):
        # ``TTSamplingParams`` types every per-user field as ``torch.Tensor |
        # list[...]`` (vllm_tt_plugin/model_input.py), and the plugin currently
        # hands us the list form. A tensor must still be indexed, not passed
        # through whole: a whole tensor would read as "one value per lane" and
        # silently give this request the batch's other rows' params.
        if value is None:
            return value
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return value[i].item() if i < value.shape[0] else value[0].item()
        if not isinstance(value, (list, tuple)):
            return value
        return value[i] if i < len(value) else value[0]

    updates = {f.name: _at(getattr(sampling_params, f.name)) for f in fields(sampling_params)}
    return replace(sampling_params, **updates)

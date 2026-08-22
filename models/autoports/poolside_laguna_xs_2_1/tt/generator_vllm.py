# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM serving adapter for poolside/Laguna-XS-2.1 on a 1×D Blackhole mesh.

This is the thin translation layer between the Tenstorrent vLLM plugin and the model-specific
``LagunaGenerator`` / ``LagunaModel`` (``tt/generator.py`` / ``tt/model.py``). It implements exactly
the method surface the plugin calls (``initialize_vllm_model``, ``allocate_kv_cache``,
``prefill_forward``, ``decode_forward``, ``read_decode_output``, ``process_decode_output_host``,
``get_max_tokens_all_users``, ``model_capabilities``) and delegates all real compute to the
generator's low-level pieces (``model.embed_*`` / ``prefill_layers`` / ``decode_layers`` /
``lm_head_shards_*`` and the canonical ``Sampling1D`` split-sampling path). It adds NO new sampling
strategy, NO host argmax on the perf path, NO full-logits readback on the perf path, and NO
Python readback/writeback token-feedback loop: the traced decode replays a single captured graph
that samples on device and feeds ``tt_out_tok`` back into the persistent decode token buffer.

Cache ownership: in vLLM mode the KV cache is **owned by vLLM** — ``allocate_kv_cache`` builds the
paged buffers in the exact layer-dict format ``LagunaModel`` consumes, and every prefill/decode
call receives that cache plus vLLM's per-step page table and positions. The generator's own
standalone cache/reset path (``tt/generator.py``) is untouched and used only by the readiness
checks.

Attention: Laguna is a hybrid model (10 full + 30 sliding layers, ``sliding_window=512``). The
qualified default remains a uniform full-context KV cache: sliding attention is enforced on the
READ side by the SDPA op. ``TT_LAGUNA_HYBRID_KV=1`` is a separate, fail-closed qualification path.
It exposes the exact 40-layer attention pattern to vLLM, which creates four independent block-table
groups and aliases equal group slots onto ten physical K/V tensor pairs. The TT adapter validates
that complete contract before allocating or executing and keeps prefix caching disabled until the
two ownership schemes have been qualified together.

Precision: construction goes through ``LagunaGenerator.from_pretrained`` →
``LagunaModel.from_pretrained``, which by default loads the datatype-sweep-selected precision
policy (``doc/datatype_sweep/selected_precision_config.json``): BFP8 attn/dense/shared weights,
BFP4 routed experts, BF16 router/norms/activations/CCL, BFP8 KV cache, BFP8 LM head, per-group
compute fidelities, fp32/HiFi4 SDPA. The serving path therefore uses the selected policy verbatim.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Optional

import torch

import ttnn

try:
    from .generator import LagunaGenerator, _replicate
    from .kv_grouping import HybridKVLayout, build_laguna_hybrid_kv_layout, validate_per_layer_tensor_aliases
    from .prefill_runtime import (
        PrefillRuntimeOffsets,
        PrefillStreamChunk,
        prefill_chunk_plan,
        prefill_stream_plan,
        streaming_prefill_capacity,
    )
except ImportError:  # loaded as a standalone module by some tooling
    from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator, _replicate
    from models.autoports.poolside_laguna_xs_2_1.tt.kv_grouping import (
        HybridKVLayout,
        build_laguna_hybrid_kv_layout,
        validate_per_layer_tensor_aliases,
    )
    from models.autoports.poolside_laguna_xs_2_1.tt.prefill_runtime import (
        PrefillRuntimeOffsets,
        PrefillStreamChunk,
        prefill_chunk_plan,
        prefill_stream_plan,
        streaming_prefill_capacity,
    )

# A selected profile's advertised context MUST equal its end-to-end qualified context. The HF config
# declares 262144 and the decoder addresses pos 262143 in isolation, but that does not establish
# serving capacity. The profile launcher supplies max_model_len=65536 for the D1 candidate and 131072
# for D2/D4; get_max_tokens_all_users bounds the KV pool by that request and this global ceiling. The
# historical 2026-07-31 D4 record served 131072 and OOMed at 262144, but its formerly cited raw sweep is
# absent from this checkout. D1/D2 full qualification is still incomplete; see doc/context_contract.json.
HF_CONFIG_MAX_CONTEXT = 262144  # what the HF config declares (not currently servable end-to-end)
# This is the global ceiling, not the per-profile qualification result. Env-overridable only for
# explicit context experiments; the launcher's smaller max_model_len still bounds D1. Raising the
# ceiling re-introduces the historical D4 OOM risk and must never be used to claim qualification.
ADVERTISED_MAX_CONTEXT = int(os.environ.get("TT_LAGUNA_ADVERTISED_CONTEXT", "131072"))


def _prefill_rope_capacity(max_model_len: int, *, streaming: bool = True) -> int:
    """RoPE horizon covering every padded prefill admitted by the adapter.

    Streaming executes at most one 8192-token chunk at a time and only pads its
    tail, so one rounding of the logical context is sufficient.  ``streaming=False``
    retains the former monolithic-bucket horizon as a rollback path.
    """
    requested = int(max_model_len)
    if streaming:
        capacity = streaming_prefill_capacity(requested, outer_chunk=min(8192, requested))
    else:
        largest_bucket = min(requested, ADVERTISED_MAX_CONTEXT)
        capacity = requested + largest_bucket
    if capacity > HF_CONFIG_MAX_CONTEXT:
        raise ValueError(
            f"bucketed prefill needs RoPE through {capacity}, beyond the HF context {HF_CONFIG_MAX_CONTEXT}"
        )
    return capacity


class LagunaForCausalLM:
    """vLLM bridge for TTLagunaForCausalLM.

    Registered as ``TTLagunaForCausalLM`` in the TT vLLM plugin (the plugin prepends ``TT`` to the
    HF architecture ``LagunaForCausalLM``).
    """

    # Marker consumed by the model-local worker lifecycle wrapper. It prevents
    # the global TT worker patch from touching any other model adapter.
    _LAGUNA_VLLM_ADAPTER = True

    # Capability flags read by the plugin platform hook. On-device sampling is REQUIRED here: the
    # readiness runner enforces ``sample_on_device_mode=all`` and the model serves its canonical
    # traced split-sampling path. Async decode is supported via the read/process split below.
    #
    # The qualified P150x2 launcher enables prefix caching. Resumed offsets stay runtime data: paged-fill
    # tables are host-rebased, RoPE uses indexed embedding into persistent outputs, and chunked SDPA
    # consumes ``chunk_start_idx_tensor``. Admission exposes only complete 8192-token cold outer chunks;
    # the model preserves that scheduler start and pads the fresh tail to canonical pipeline geometry,
    # which is warmed before trace capture. The capability remains fail-closed when the environment flag
    # is absent and is rejected on non-D2 topologies. Enabling it also freezes the program cache.
    _PREFIX_CACHE_ENABLED = os.environ.get("TT_LAGUNA_PREFIX_CACHE", "0") == "1"
    _PREFIX_CACHE_QUANTUM = 8192
    # Chunk-major prefill is the default D2 serving path after the device-free
    # planner/adapter contracts in tests/test_prefill_runtime.py and
    # tests/test_generator_vllm_prefix_resume.py and the retained D2 hardware
    # PCC/latency gate. Keep an acknowledged monolithic rollback for diagnostics.
    # D1 retains monolithic prefill because its sliding decoder carries K/V
    # locally within one call.
    _STREAMING_PREFILL_ENABLED = os.environ.get("TT_LAGUNA_STREAMING_PREFILL", "1") == "1"
    _PREFILL_STREAM_OUTER_CHUNK = 8192
    # vLLM 0.24 groups Laguna as four 10-layer block-table groups and aliases
    # equal slots onto ten physical K/V tensor pairs.  The feature remains
    # opt-in until its cache-off hardware gate completes.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = os.environ.get("TT_LAGUNA_HYBRID_KV", "0") == "1"
    # Published five-layer DFlash serving is a separate, default-off batch-one
    # cache-off mode.  It owns an eager verify/accept loop and therefore cannot
    # coexist with prefix, hybrid KV, or the ngram speculative controller.
    _DFLASH_SERVING_ENABLED = os.environ.get("TT_LAGUNA_DFLASH", "0") == "1"
    model_capabilities = {
        # Prefix + aliased hybrid KV is a separate ownership qualification.  Do
        # not advertise it while a cache-off hybrid or DFlash tranche is selected.
        "supports_prefix_caching": (
            _PREFIX_CACHE_ENABLED and not _HYBRID_KV_CACHE_GROUPS_ENABLED and not _DFLASH_SERVING_ENABLED
        ),
        "supports_prefix_caching_with_sliding_window": (
            _PREFIX_CACHE_ENABLED and not _HYBRID_KV_CACHE_GROUPS_ENABLED and not _DFLASH_SERVING_ENABLED
        ),
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    @staticmethod
    def _validate_prefix_cache_topology(device_count, enabled):
        if bool(enabled) and int(device_count) != 2:
            raise RuntimeError(
                "Laguna prefix caching is qualified only on the p150x2 two-chip topology; "
                f"got D={int(device_count)}. Set TT_LAGUNA_PREFIX_CACHE=0."
            )

    @staticmethod
    def _validate_kv_feature_combination(prefix_enabled, hybrid_enabled):
        if bool(prefix_enabled) and bool(hybrid_enabled):
            raise RuntimeError(
                "Laguna hybrid KV is qualified cache-off first and cannot be combined with "
                "prefix caching yet; set either TT_LAGUNA_HYBRID_KV=0 or "
                "TT_LAGUNA_PREFIX_CACHE=0."
            )

    @staticmethod
    def _validate_dflash_serving_envelope(
        *,
        enabled,
        device_count,
        max_batch_size,
        prefix_enabled,
        hybrid_enabled,
        spec_mode,
    ):
        if not bool(enabled):
            return
        if int(device_count) != 2:
            raise RuntimeError(
                "experimental Laguna DFlash serving is hardware-scoped only to p150x2, "
                f"got D={int(device_count)}; set TT_LAGUNA_DFLASH=0"
            )
        if int(max_batch_size) != 1:
            raise RuntimeError("Laguna DFlash serving requires --max-num-seqs 1, " f"got {int(max_batch_size)}")
        if bool(prefix_enabled):
            raise RuntimeError("Laguna DFlash serving requires TT_LAGUNA_PREFIX_CACHE=0")
        if bool(hybrid_enabled):
            raise RuntimeError("Laguna DFlash serving requires TT_LAGUNA_HYBRID_KV=0")
        if str(spec_mode):
            raise RuntimeError("Laguna DFlash serving cannot be combined with TT_LAGUNA_SPEC_DECODE")

    def _streaming_prefill_active(self):
        """D2-only until D1 sliding attention can cross adapter-call boundaries."""

        return bool(self._STREAMING_PREFILL_ENABLED) and int(self.D) == 2

    def __init__(self, generator: LagunaGenerator, mesh_device, max_batch_size: int, max_model_len: int):
        self._closed = False
        self.gen = generator
        self.model = generator.model
        self.mesh_device = mesh_device
        self.tokenizer = generator.tokenizer
        self.data_parallel = 1  # a single 1×D mesh; TP=D/EP=D are intra-mesh, not vLLM DP
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len
        self.vocab = generator.vocab
        self.hidden = generator.hidden
        self.D = mesh_device.get_num_devices()
        self._validate_prefix_cache_topology(self.D, self._PREFIX_CACHE_ENABLED)
        self._validate_kv_feature_combination(
            self._PREFIX_CACHE_ENABLED,
            self._HYBRID_KV_CACHE_GROUPS_ENABLED,
        )
        # Per-batch captured decode trace + persistent device buffers.
        self._decode: dict[int, dict] = {}
        # Per-K1 captured spec-decode VERIFY trace (batched decode over K+1 candidates, seq KV write).
        self._verify_dec: dict[int, dict] = {}
        # Persistent prefill buffers (sampling tensors + fixed [1,1,1,H] terminal + B=1 sampler),
        # allocated once BEFORE the decode trace is captured (see warmup_model_prefill).
        self._pf: Optional[dict] = None
        # Persistent prefill page-table buffers keyed by shape (allocate-once, then copy-in), kept
        # SEPARATE from the decode trace's page table so a prefill never overwrites the decode pt.
        # Attention and paged-fill use distinct buffers: attention padding maps to the valid scratch
        # block, while fill padding is -1 (the paged_fill_cache write-skip sentinel).
        self._pf_pt: dict = {}
        self._pf_fill_pt: dict = {}
        # Hybrid KV: persistent prefill page-table buffers per vLLM group. Laguna
        # has four groups, including three distinct sliding groups with different
        # block-id namespaces; attention kind alone is not a sufficient identity.
        self._pf_pt_groups: dict = {}
        self._pf_fill_pt_groups: dict = {}
        # Per built-layer kind and exact production alias layout, derived lazily.
        self._layer_kinds: Optional[list] = None
        self._hybrid_layout_cache: Optional[HybridKVLayout] = None
        # max_num_blocks_per_req, learned from warmup_model_decode; lets prefill warmup pre-allocate
        # the serving-shape page-table buffer before the decode trace is captured.
        self._max_blocks: Optional[int] = None
        self.already_warmed_up_prefill = False
        # The plugin resets ``already_warmed_up_prefill`` between its compile and trace phases even
        # though Laguna never captures a prefill trace. Keep a private completion latch so the full
        # bucket ladder is compiled once, not twice; unlike the plugin-owned flag this is not reset.
        self._prefill_programs_warmed = False
        self._in_prefill_warmup = False  # True only while warmup_model_prefill runs (suppresses the
        # _prefill_pt diagnostic for intentional warmup pre-allocs; a warning outside warmup = the W1 bug).
        # When prefix caching is enabled, freeze TTNN's program cache after the full prefill ladder and
        # decode trace have been built. Any missed runtime-offset shape then fails immediately instead of
        # compiling under a resident trace. The qualified prefix-cache profile always enables this guard.
        self._freeze_program_cache = (
            self._PREFIX_CACHE_ENABLED or os.environ.get("TT_LAGUNA_FREEZE_PROGRAM_CACHE", "0") == "1"
        )
        self._program_cache_entries_after_trace: Optional[int] = None
        # ---- eager spec-decode (opt-in) — served in-adapter, B==1 greedy. Phase 2. ----
        # TT_LAGUNA_SPEC_DECODE: "" off | "probe" = run the one-shot feasibility probe (does eager verify
        # run under the resident decode trace without an alloc-under-trace hang?) | "1" = full buffered loop.
        self._spec_mode = os.environ.get("TT_LAGUNA_SPEC_DECODE", "")
        if self._HYBRID_KV_CACHE_GROUPS_ENABLED and self._spec_mode == "1":
            raise RuntimeError(
                "Laguna hybrid KV does not support traced speculative decode yet; "
                "set TT_LAGUNA_SPEC_DECODE= or TT_LAGUNA_HYBRID_KV=0"
            )
        self._spec_probed = False
        self._spec_buf: list = []  # pending committed token ids, returned one per vLLM step
        self._spec_hist: list = []  # running token history for the single served request (ngram source)
        self._spec = None  # lazily-built SpeculativeDecoder (served mode); needs kv_cache/page_table per call
        self._spec_tok = None  # persistent [1,1,1,1] device token buffer the plugin reads back
        self._spec_next_pos = None  # position we expect on the next decode call; discontinuity = new request
        self._spec_prefill_seq: list = []  # prompt tokens stashed at prefill (greedy gives no history via kwargs)
        # Diagnostic sink: MPI-worker stdout isn't captured in the readiness log, so spec/probe verdicts go
        # to a file readable regardless of process. Only touched when spec mode is set (no normal-run noise).
        self._spec_log_path = str(Path(__file__).resolve().parents[1] / "doc/vllm_integration/_runs/spec_probe.txt")
        if self._spec_mode:
            self._spec_log(f"__init__ pid={os.getpid()} spec_mode={self._spec_mode!r}")
        self._validate_dflash_serving_envelope(
            enabled=self._DFLASH_SERVING_ENABLED,
            device_count=self.D,
            max_batch_size=self.max_batch_size,
            prefix_enabled=self._PREFIX_CACHE_ENABLED,
            hybrid_enabled=self._HYBRID_KV_CACHE_GROUPS_ENABLED,
            spec_mode=self._spec_mode,
        )
        self._dflash_core = None
        self._dflash_cache = None
        self._dflash_controller = None
        self._dflash_tok = None
        self._dflash_request_serial = 0
        self._dflash_request_id = None
        if self._DFLASH_SERVING_ENABLED:
            self._initialize_dflash_serving()
        # vLLM-owned cache dtype (from the selected precision policy), used for allocation.
        self._kv_dtype = self.model.precision_policy.kv_cache
        self._report_dram("weights")

    def _initialize_dflash_serving(self):
        """Allocate the published draft and its one-request cache after opt-in."""

        from .dflash_serving import DFlashServedController, DFlashServingEnvelope
        from .dflash_tt import DFlashTTCore

        if len(self.model.layers) != 40:
            raise RuntimeError(
                "Laguna DFlash serving requires the exact full 40-layer target; " f"got {len(self.model.layers)} layers"
            )
        # Proposal padding extends at most 64 rows beyond the admitted semantic
        # context.  Reject an unrepresentable horizon before loading any draft
        # tensor instead of truncating RoPE near the checkpoint limit.
        draft_horizon = int(self.max_model_len) + 64
        if draft_horizon > HF_CONFIG_MAX_CONTEXT:
            raise RuntimeError(
                "Laguna DFlash serving requires max_model_len + 64 <= "
                f"{HF_CONFIG_MAX_CONTEXT}, got {self.max_model_len}"
            )
        core = DFlashTTCore.from_checkpoint(
            self.mesh_device,
            max_seq_len=draft_horizon,
            enable_experimental=True,
        )
        cache = core.allocate_proposal_cache(enable_experimental=True)
        self._dflash_core = core
        self._dflash_cache = cache
        try:
            controller = DFlashServedController(
                core=core,
                proposal_cache=cache,
                target_model=self.model,
                verify_greedy=self.verify_greedy_decode_with_dflash_aux,
                draft_argmax=self._dflash_draft_argmax,
                envelope=DFlashServingEnvelope(
                    enabled=True,
                    batch_size=self.max_batch_size,
                    greedy=True,
                    prefix_caching=self._PREFIX_CACHE_ENABLED,
                    hybrid_kv=self._HYBRID_KV_CACHE_GROUPS_ENABLED,
                    cache_off=not self._PREFIX_CACHE_ENABLED,
                ),
            )
            self._dflash_controller = controller
            self._dflash_tok = self.gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)
        except Exception:
            self.close_dflash()
            raise

    def _dflash_draft_argmax(self, proposal):
        rows = self.model.logits_to_host(proposal.logits_shards).reshape(-1, int(self.vocab))
        expected = int(self._dflash_core.config.max_speculative_tokens)
        if int(rows.shape[0]) != expected:
            raise RuntimeError(f"DFlash proposal produced {rows.shape[0]} logit rows, expected {expected}")
        return torch.argmax(rows, dim=-1).to(torch.int32).tolist()

    def close_dflash(self):
        """Explicitly release request state and draft-owned KV allocations."""

        controller = getattr(self, "_dflash_controller", None)
        cache = getattr(self, "_dflash_cache", None)
        if controller is not None:
            controller.close()
        elif cache is not None:
            cache.close()
        self._dflash_controller = None
        self._dflash_cache = None
        self._dflash_core = None
        self._dflash_tok = None
        self._dflash_request_id = None

    def _release_decode_traces(self):
        """Release every captured TT trace before dropping its persistent state."""

        mappings = (getattr(self, "_decode", {}), getattr(self, "_verify_dec", {}))
        released = set()
        errors = []
        for mapping in mappings:
            for state in mapping.values():
                trace_id = state.get("tid") if isinstance(state, dict) else None
                if trace_id is None:
                    continue
                if trace_id in released:
                    state["tid"] = None
                    continue
                try:
                    ttnn.release_trace(self.mesh_device, trace_id)
                except Exception as error:
                    errors.append((trace_id, error))
                else:
                    state["tid"] = None
                    released.add(trace_id)
        if errors:
            failed = ", ".join(repr(trace_id) for trace_id, _ in errors)
            raise RuntimeError(f"failed to release Laguna TT trace(s): {failed}") from errors[0][1]
        for mapping in mappings:
            mapping.clear()

    def close(self):
        """Idempotently release adapter-owned traces and persistent request state."""

        if getattr(self, "_closed", False):
            return
        self._release_decode_traces()
        self.close_dflash()
        self._pf = None
        for name in ("_pf_pt", "_pf_fill_pt", "_pf_pt_groups", "_pf_fill_pt_groups"):
            mapping = getattr(self, name, None)
            if isinstance(mapping, dict):
                mapping.clear()
        self._spec = None
        self._spec_tok = None
        for name in ("_spec_buf", "_spec_hist", "_spec_prefill_seq"):
            values = getattr(self, name, None)
            if isinstance(values, list):
                values.clear()
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Interpreter teardown and hard-kill recovery cannot safely surface
            # Python exceptions; the TT worker still closes/resets the mesh.
            pass

    def _report_dram(self, stage: str, *, enforce: bool = False):
        """Log a synchronized DRAM snapshot and optionally enforce the serving safety margin.

        The ratio is topology-independent because TTNN reports uniform per-bank allocation. The
        contiguous-free guard catches fragmentation that a total-free check misses.
        """
        ttnn.synchronize_device(self.mesh_device)
        view = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
        total = int(view.total_bytes_per_bank) * int(view.num_banks)
        allocated = int(view.total_bytes_allocated_per_bank) * int(view.num_banks)
        free = int(view.total_bytes_free_per_bank) * int(view.num_banks)
        largest = int(view.largest_contiguous_bytes_free_per_bank)
        free_fraction = (free / total) if total else 0.0
        print(
            f"[laguna memory] stage={stage} used_mib={allocated / 2**20:.1f} "
            f"total_mib={total / 2**20:.1f} free_mib={free / 2**20:.1f} "
            f"free_fraction={free_fraction:.4f} largest_contiguous_mib_per_bank={largest / 2**20:.1f}",
            flush=True,
        )
        if enforce and os.environ.get("TT_LAGUNA_ENFORCE_MEMORY_MARGIN", "0") == "1":
            minimum_fraction = float(os.environ.get("TT_LAGUNA_MIN_DRAM_FREE_FRACTION", "0.10"))
            minimum_contiguous = float(os.environ.get("TT_LAGUNA_MIN_CONTIGUOUS_MIB", "128")) * 2**20
            if free_fraction < minimum_fraction or largest < minimum_contiguous:
                raise RuntimeError(
                    "Laguna DRAM safety margin failed after trace capture: "
                    f"free_fraction={free_fraction:.4f} (need >= {minimum_fraction:.4f}), "
                    f"largest_contiguous_mib_per_bank={largest / 2**20:.1f} "
                    f"(need >= {minimum_contiguous / 2**20:.1f})"
                )
        return {
            "stage": stage,
            "total_bytes": total,
            "allocated_bytes": allocated,
            "free_bytes": free,
            "free_fraction": free_fraction,
            "largest_contiguous_bytes_free_per_bank": largest,
        }

    def _freeze_program_cache_after_trace(self):
        """Reject any TTNN program-cache miss after the serving traces are resident.

        This is enabled automatically with prefix caching and can be forced for qualification with
        ``TT_LAGUNA_FREEZE_PROGRAM_CACHE=1``. It converts an unsafe serving-time specialization into
        an immediate failure and records the exact post-warmup cache cardinality for diagnostics.
        """
        if not self._freeze_program_cache or self._program_cache_entries_after_trace is not None:
            return
        ttnn.synchronize_device(self.mesh_device)
        count = int(self.mesh_device.num_program_cache_entries())
        self.mesh_device.set_program_cache_misses_allowed(False)
        self._program_cache_entries_after_trace = count
        print(f"[laguna] TTNN program cache frozen after trace: entries={count}", flush=True)

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=ADVERTISED_MAX_CONTEXT,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        """Plugin entry point (loader.py). ``optimizations`` (str|None) is accepted for interface
        parity but the precision policy comes from the datatype-sweep selection by default; a
        non-default policy is only used via ``TT_LAGUNA_PRECISION_CONFIG``. ``n_layers`` builds a
        reduced representative target for the minimum-surface bring-up loop."""
        assert tt_data_parallel == 1, (
            f"Laguna-XS-2.1 uses one 1×D mesh (intra-mesh TP=D/EP=D); tt_data_parallel must be 1, "
            f"got {tt_data_parallel}"
        )
        # Minimum-surface bring-up: TT_LAGUNA_VLLM_NUM_LAYERS builds a reduced representative target
        # (e.g. "0,1,4" = dense-full + sliding-MoE + full-MoE). vLLM still sees the full 40-layer HF
        # config (so it allocates 40 KV specs; the model's decode/prefill zip truncates to the built
        # layers). Reduced is an inner-loop debugging tool only — final evidence uses the full stack.
        if n_layers is None:
            import os as _os

            env_nl = _os.environ.get("TT_LAGUNA_VLLM_NUM_LAYERS")
            if env_nl:
                n_layers = [int(x) for x in env_nl.split(",")] if "," in env_nl else int(env_nl)
        requested_max_seq_len = int(max_seq_len)
        # D2 streams fixed 8192-token chunks and pads only the final tail, so its
        # shared RoPE tables need the logical context rounded once. D1 retains the
        # legacy monolithic horizon because its sliding decoder cannot carry a
        # local K/V tail across separate adapter calls. KV allocation and scheduler
        # admission remain bounded by the requested logical context on both.
        rope_capacity = _prefill_rope_capacity(
            requested_max_seq_len,
            streaming=(bool(cls._STREAMING_PREFILL_ENABLED) and int(mesh_device.get_num_devices()) == 2),
        )
        gen = LagunaGenerator.from_pretrained(
            mesh_device,
            max_seq_len=rope_capacity,
            num_layers=n_layers,
            hf_config=hf_config,
        )
        gen.max_seq_len = requested_max_seq_len
        return cls(gen, mesh_device, int(max_batch_size), requested_max_seq_len)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Return Laguna's exact opt-in hybrid KV-cache specification.

        When hybrid KV is disabled, return ``None`` so the TT plugin retains its
        qualified single-spec/full-context fallback.  When enabled, emit 30
        ``SlidingWindowSpec(sliding_window=512)`` and ten ``FullAttentionSpec``
        entries in logical-layer order. vLLM groups them as one full and three
        sliding groups, each containing ten layers. Equal slots across the four
        groups share one physical tensor while their block tables retain disjoint
        block-id namespaces. This reduces 40 physical K/V pairs to ten for the
        same global block pool; sliding-window eviction also reduces each sliding
        group's live per-request occupancy. Per-layer tables reach this adapter as
        ``page_tables_per_layer`` and are collapsed back to four persistent device
        buffers only after exact group-consistency validation.
        """
        cls._validate_kv_feature_combination(
            cls._PREFIX_CACHE_ENABLED,
            cls._HYBRID_KV_CACHE_GROUPS_ENABLED,
        )
        if not cls._HYBRID_KV_CACHE_GROUPS_ENABLED:
            return None

        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        hf_config = model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        num_layers = getattr(text_config, "num_hidden_layers", None)
        layer_types = getattr(text_config, "layer_types", None)
        if num_layers is None and layer_types is not None:
            num_layers = len(layer_types)
        num_layers = int(num_layers)
        sliding_window = int(getattr(text_config, "sliding_window", 0) or 0)
        raw_kinds = tuple(layer_types or ())
        unknown_kinds = sorted(set(raw_kinds) - {"full_attention", "sliding_attention"})
        if unknown_kinds:
            raise ValueError(f"Laguna hybrid KV received unknown layer_types {unknown_kinds}")
        normalized_kinds = tuple("sliding" if kind == "sliding_attention" else "full" for kind in raw_kinds)
        # Fails closed for a reduced stack, missing layer_types, unknown values,
        # or any checkpoint whose repeated attention pattern drifted.
        build_laguna_hybrid_kv_layout(normalized_kinds)
        if num_layers != len(raw_kinds):
            raise ValueError(
                f"Laguna hybrid KV num_hidden_layers={num_layers} but layer_types has {len(raw_kinds)} entries"
            )
        if sliding_window != 512:
            raise ValueError(f"Laguna hybrid KV requires sliding_window=512, got {sliding_window}")
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        try:  # vLLM moved this constant across versions (fork 0.16 vs stock 0.24) — tolerate both.
            from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        except ImportError:
            from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        common = dict(
            block_size=cache_config.block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )

        def _is_sliding(i):
            return bool(layer_types) and layer_types[i] == "sliding_attention" and sliding_window > 0

        spec = {}
        for i in range(num_layers):
            key = f"model.layers.{i}.self_attn"
            if _is_sliding(i):
                spec[key] = SlidingWindowSpec(sliding_window=sliding_window, **common)
            else:
                spec[key] = FullAttentionSpec(**common)
        # Optional audit trail: the pinned TT plugin resolves this class hook and
        # forwards its result to upstream's hybrid KV planner.
        if os.environ.get("TT_LAGUNA_KV_SPEC_LOG") != "1":
            return spec
        try:
            nsl = sum(1 for v in spec.values() if type(v).__name__ == "SlidingWindowSpec")
            nfa = len(spec) - nsl
            kinds = sorted({type(v).__name__ for v in spec.values()})
            _kv_log = Path(__file__).resolve().parents[1] / "doc/vllm_integration/_runs/kv_spec.txt"
            _kv_log.parent.mkdir(parents=True, exist_ok=True)
            with open(_kv_log, "a") as _f:
                _f.write(
                    f"[laguna kv_spec] get_kv_cache_spec CALLED pid={os.getpid()}: {len(spec)} layers, "
                    f"sliding={nsl} full={nfa}, kinds={kinds}, sliding_window={sliding_window}, "
                    f"hybrid_flag={cls._HYBRID_KV_CACHE_GROUPS_ENABLED}\n"
                )
        except Exception:
            pass
        return spec

    @classmethod
    def get_max_tokens_all_users(cls, model_name: str = "", num_devices: int = 1, tt_data_parallel: int = 1, **kwargs):
        """Total KV-cache token pool for the selected serving profile.

        ``ADVERTISED_MAX_CONTEXT`` is a global ceiling. The requested ``max_model_len`` supplies the
        profile cap (65536 for D1, 131072 for D2/D4), so a single request can use that whole window
        without treating the ceiling as proof that an unqualified profile is servable.  The opt-in
        two-sequence pool remains fail-closed unless every independently checked launcher invariant
        is also supplied to this hook.
        """
        max_model_len = kwargs.get("max_model_len")
        if os.environ.get("TT_LAGUNA_MULTI_SEQ_POOL", "0") == "1":
            max_num_seqs = kwargs.get("max_num_seqs")
            if int(num_devices) != 2:
                raise ValueError("TT_LAGUNA_MULTI_SEQ_POOL=1 requires num_devices=2, " f"got {num_devices}")
            if max_num_seqs is None or int(max_num_seqs) != 2:
                raise ValueError("TT_LAGUNA_MULTI_SEQ_POOL=1 requires max_num_seqs=2, " f"got {max_num_seqs}")
            if max_model_len is None or not (0 < int(max_model_len) <= 65536):
                raise ValueError(
                    "TT_LAGUNA_MULTI_SEQ_POOL=1 requires 0 < max_model_len <= 65536, " f"got {max_model_len}"
                )
            pool_tokens = int(max_model_len) * 2
            if pool_tokens > ADVERTISED_MAX_CONTEXT:
                raise ValueError(
                    "TT_LAGUNA_MULTI_SEQ_POOL=1 token pool exceeds advertised context: "
                    f"{pool_tokens} > {ADVERTISED_MAX_CONTEXT}"
                )
            return pool_tokens
        if max_model_len:
            return min(int(max_model_len), ADVERTISED_MAX_CONTEXT)
        return ADVERTISED_MAX_CONTEXT

    @property
    def cache_path(self):
        # Not used by this adapter's own allocator (weights are cached inside LagunaModel), but the
        # plugin may query it; return a harmless path.
        return Path("/tmp")

    # --------------------------------------------------------------------- #
    # KV cache (vLLM-owned)
    # --------------------------------------------------------------------- #
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Build the vLLM-owned paged KV cache. ``kv_cache_shape`` =
        ``(num_blocks, num_kv_heads_local, block_size, head_dim)`` — already folded to the per-device
        local KV heads (2 = 8/TP4) by the plugin. Each layer gets its own ``[k, v]`` buffer,
        replicated across the mesh (each device stores its own local-head slice; identical shape).
        Returns the list of per-layer dicts that ``LagunaModel.prefill_layers`` / ``decode_layers``
        consume. KV dtype is the selected-policy BFP8, independent of vLLM's torch ``dtype`` hint."""
        num_blocks, local_kv_heads, block_size, head_dim = kv_cache_shape
        # Traces close over the old KV tensors. Release them after validating the
        # replacement shape but before allocating its first buffer.
        self._release_decode_traces()
        # vLLM owns block ids [0, num_blocks). Bucketed prefill can compute beyond the last REAL
        # scheduler-allocated block, so reserve one adapter-private physical block for read-safe
        # attention padding. Paged-fill uses -1 to skip those columns; decode continues to use vLLM's
        # untouched table and can never address the scratch block.
        scratch_block_idx = int(num_blocks)
        physical_shape = (int(num_blocks) + 1, int(local_kv_heads), int(block_size), int(head_dim))
        kv_cache = []
        for _ in range(num_layers):
            k = ttnn.from_torch(
                torch.zeros(physical_shape, dtype=torch.float32),
                dtype=self._kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate(self.mesh_device),
            )
            v = ttnn.from_torch(
                torch.zeros(physical_shape, dtype=torch.float32),
                dtype=self._kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate(self.mesh_device),
            )
            kv_cache.append(
                {
                    "k": k,
                    "v": v,
                    "block_size": int(block_size),
                    "blocks_per_user": int(num_blocks),
                    "scratch_block_idx": scratch_block_idx,
                    "dtype": self._kv_dtype,
                }
            )
        self._report_dram("kv_cache")
        return kv_cache

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Allocate plugin-resolved per-layer caches, honoring hybrid tensor aliases.

        The plugin supplies ``(shape, dtype, tensor_idx)`` in logical layer
        order.  Uniform mode keeps one physical tensor pair per layer.  Hybrid
        mode validates the exact Laguna four-group layout, allocates each of the
        ten unique ``tensor_idx`` values once, and returns 40 logical cache
        dictionaries whose K/V objects alias by slot.  Group IDs remain separate
        metadata because they select different vLLM block tables.
        """

        hybrid = bool(self._HYBRID_KV_CACHE_GROUPS_ENABLED)
        if hybrid:
            self._validate_kv_feature_combination(self._PREFIX_CACHE_ENABLED, True)
            layout = self._hybrid_kv_layout()
            descriptors = validate_per_layer_tensor_aliases(per_layer_specs, layout)
            allocation_entries = [(*descriptors[tensor_idx], tensor_idx) for tensor_idx in range(layout.num_tensors)]
        else:
            layout = None
            allocation_entries = [
                (tuple(entry[0]), entry[1], layer_idx) for layer_idx, entry in enumerate(per_layer_specs)
            ]

        # Validate the complete alias plan before disturbing a live trace, then
        # release all captures before allocating the first replacement KV tensor.
        self._release_decode_traces()
        physical = {}
        for shape, _plugin_dtype, tensor_idx in allocation_entries:
            # ``num_blocks`` is the vLLM-visible pool (2460 in the qualified
            # hybrid envelope, including vLLM's null block). The extra row is
            # adapter-private prefill padding, so the physical tensor has 2461
            # rows without advertising another schedulable block ID.
            num_blocks, local_kv_heads, block_size, head_dim = tuple(shape)
            scratch_block_idx = int(num_blocks)
            physical_shape = (int(num_blocks) + 1, int(local_kv_heads), int(block_size), int(head_dim))
            k = ttnn.from_torch(
                torch.zeros(physical_shape, dtype=torch.float32),
                dtype=self._kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate(self.mesh_device),
            )
            v = ttnn.from_torch(
                torch.zeros(physical_shape, dtype=torch.float32),
                dtype=self._kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate(self.mesh_device),
            )
            physical[int(tensor_idx)] = {
                "k": k,
                "v": v,
                "block_size": int(block_size),
                "blocks_per_user": int(num_blocks),
                "scratch_block_idx": scratch_block_idx,
                "dtype": self._kv_dtype,
                "tensor_idx": int(tensor_idx),
            }

        kv_cache = []
        if hybrid:
            assert layout is not None
            for alias in layout.aliases:
                entry = dict(physical[alias.tensor_index])
                entry.update(
                    hybrid_group_id=alias.group_id,
                    hybrid_kind=alias.kind,
                    logical_layer_idx=alias.layer_index,
                )
                kv_cache.append(entry)
        else:
            for layer_idx in range(len(per_layer_specs)):
                kv_cache.append(dict(physical[layer_idx]))
        self._report_dram("kv_cache_per_layer_hybrid" if hybrid else "kv_cache_per_layer")
        return kv_cache

    # ---- hybrid KV: per-group page-table helpers ---- #
    def _group_kinds(self):
        """['full'|'sliding'] per BUILT layer, from each decoder's cfg.is_sliding."""
        if self._layer_kinds is None:
            self._layer_kinds = [
                "sliding" if bool(getattr(dec.cfg, "is_sliding", False)) else "full" for dec in self.model.layers
            ]
        return self._layer_kinds

    def _hybrid_kv_layout(self):
        """Return the exact full-model layout; reduced hybrid bring-up is unsafe."""

        if getattr(self, "_hybrid_layout_cache", None) is None:
            self._hybrid_layout_cache = build_laguna_hybrid_kv_layout(self._group_kinds())
        return self._hybrid_layout_cache

    def _group_reps(self):
        """Map each of Laguna's four vLLM group IDs to its representative layer."""

        layout = self._hybrid_kv_layout()
        return {group_id: layer_idx for group_id, layer_idx in enumerate(layout.representative_layers)}

    def _kv_cache_is_hybrid(self, kv_cache):
        """Validate logical metadata/tensor aliases and return whether it is hybrid."""

        if not kv_cache:
            return False
        marked = ["hybrid_group_id" in entry for entry in kv_cache]
        if any(marked) and not all(marked):
            raise ValueError("KV cache mixes uniform and hybrid logical-layer metadata")
        if not all(marked):
            return False
        layout = self._hybrid_kv_layout()
        if len(kv_cache) != len(layout.aliases):
            raise ValueError(f"hybrid KV cache has {len(kv_cache)} logical layers, expected {len(layout.aliases)}")
        physical = {}
        for alias, entry in zip(layout.aliases, kv_cache, strict=True):
            actual = (
                int(entry.get("logical_layer_idx", -1)),
                int(entry.get("hybrid_group_id", -1)),
                int(entry.get("tensor_idx", -1)),
                entry.get("hybrid_kind"),
            )
            expected = (alias.layer_index, alias.group_id, alias.tensor_index, alias.kind)
            if actual != expected:
                raise ValueError(
                    f"hybrid KV metadata drift at layer {alias.layer_index}: got {actual}, expected {expected}"
                )
            descriptor = (
                entry.get("k"),
                entry.get("v"),
                int(entry.get("block_size", -1)),
                int(entry.get("blocks_per_user", -1)),
                int(entry.get("scratch_block_idx", -1)),
            )
            prior = physical.setdefault(alias.tensor_index, descriptor)
            if prior[0] is not descriptor[0] or prior[1] is not descriptor[1] or prior[2:] != descriptor[2:]:
                raise ValueError(
                    f"hybrid KV tensor alias {alias.tensor_index} is not physically shared "
                    f"or has inconsistent cache metadata at layer {alias.layer_index}"
                )
        return True

    def _validated_group_page_tables(self, page_tables_per_layer, *, purpose):
        """Normalize and prove that every logical layer in a group has one table."""

        layout = self._hybrid_kv_layout()
        if len(page_tables_per_layer) != len(layout.aliases):
            raise ValueError(
                f"{purpose} page_tables_per_layer has {len(page_tables_per_layer)} entries "
                f"for {len(layout.aliases)} logical layers"
            )
        representatives = {}
        for group_id, group in enumerate(layout.groups):
            rep_index = group[0]
            rep = torch.as_tensor(page_tables_per_layer[rep_index], dtype=torch.int32)
            if rep.dim() == 1:
                rep = rep.reshape(1, -1)
            if rep.dim() != 2:
                raise ValueError(
                    f"{purpose} page table for group {group_id} must be rank 1 or 2, got {tuple(rep.shape)}"
                )
            for layer_index in group[1:]:
                candidate = torch.as_tensor(page_tables_per_layer[layer_index], dtype=torch.int32)
                if candidate.dim() == 1:
                    candidate = candidate.reshape(1, -1)
                if candidate.shape != rep.shape or not torch.equal(candidate, rep):
                    raise ValueError(
                        f"{purpose} page tables disagree within hybrid group {group_id}: "
                        f"layers {rep_index} and {layer_index}"
                    )
            representatives[group_id] = rep
        return representatives

    def _validate_page_table_mode(self, kv_cache, page_tables_per_layer, *, operation):
        """Require hybrid cache tensors and per-layer tables to travel together."""

        hybrid_cache = self._kv_cache_is_hybrid(kv_cache)
        hybrid_tables = page_tables_per_layer is not None
        if hybrid_cache != hybrid_tables:
            raise ValueError(
                f"{operation} received {'hybrid' if hybrid_cache else 'uniform'} KV cache but "
                f"{'per-layer' if hybrid_tables else 'uniform'} page tables"
            )
        return hybrid_cache

    def _prefill_pt_grouped_into(self, page_tables_per_layer, buffers, *, purpose):
        """Upload one persistent prefill page table per KV group and return a per-layer list."""
        layout = self._hybrid_kv_layout()
        hosts = self._validated_group_page_tables(page_tables_per_layer, purpose=purpose)
        bufs = {}
        for group_id, pt in hosts.items():
            key = (group_id, tuple(pt.shape))
            buf = buffers.get(key)
            if buf is None:
                buf = self.gen._rep(torch.zeros(pt.shape, dtype=torch.int32), ttnn.int32)
                buffers[key] = buf
            ttnn.copy_host_to_device_tensor(self.gen._host(pt, ttnn.int32), buf)
            bufs[group_id] = buf
        return layout.expand_group_values([bufs[group_id] for group_id in range(layout.num_groups)])

    def _prefill_pt_grouped(self, page_tables_per_layer):
        """Persistent attention page tables, one per KV group, expanded back to a per-layer list."""
        return self._prefill_pt_grouped_into(page_tables_per_layer, self._pf_pt_groups, purpose="attention")

    def _prefill_fill_pt_grouped(self, page_tables_per_layer):
        """Persistent paged-fill page tables, one per KV group, expanded to a per-layer list."""
        return self._prefill_pt_grouped_into(page_tables_per_layer, self._pf_fill_pt_groups, purpose="fill")

    def _decode_pt_grouped_alloc(self, page_tables_per_layer):
        """Allocate four persistent decode block tables before trace capture."""
        layout = self._hybrid_kv_layout()
        reps = self._group_reps()
        hosts = self._validated_group_page_tables(page_tables_per_layer, purpose="decode")
        groups = {}
        for group_id, pt in hosts.items():
            groups[group_id] = self.gen._rep(torch.zeros(pt.shape, dtype=torch.int32), ttnn.int32)
        per_layer = layout.expand_group_values([groups[group_id] for group_id in range(layout.num_groups)])
        return per_layer, groups, reps

    def _decode_pt_grouped_refresh(self, st, page_tables_per_layer):
        """Copy each group's host block table into its persistent decode buffer, only when changed."""
        hosts = self._validated_group_page_tables(page_tables_per_layer, purpose="decode")
        for group_id, pt_host in hosts.items():
            last = st["last_pt_host_groups"].get(group_id)
            if last is None or not torch.equal(pt_host, last):
                ttnn.copy_host_to_device_tensor(
                    self._page_table_to_device_host(pt_host),
                    st["pt_groups"][group_id],
                )
                st["last_pt_host_groups"][group_id] = pt_host.clone()
                self.gen.counters["page_table_refresh"] += 1

    # --------------------------------------------------------------------- #
    # Page-table / sampling helpers
    # --------------------------------------------------------------------- #
    def _page_table_to_device(self, page_table_torch):
        pt = torch.as_tensor(page_table_torch, dtype=torch.int32)
        return ttnn.from_torch(
            pt,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=_replicate(self.mesh_device),
        )

    def _prefill_pt_into(self, page_table, buffers, *, purpose):
        """Copy a host table into a persistent, shape-keyed prefill device buffer."""
        pt = torch.as_tensor(page_table, dtype=torch.int32)
        if pt.dim() == 1:
            pt = pt.reshape(1, -1)
        key = tuple(pt.shape)
        buf = buffers.get(key)
        if buf is None:
            if self.already_warmed_up_prefill and not self._in_prefill_warmup:
                # W1 diagnostic: any allocation AFTER warmup happens under the resident decode trace and
                # is the multi-minute stall. Warmup should have pre-touched every (N, prefill_w) shape.
                print(
                    f"[laguna] WARNING: prefill {purpose} page-table alloc for unwarmed shape {key} AT SERVING "
                    f"(under resident decode trace — this is the W1 stall). Widen warmup_model_prefill.",
                    flush=True,
                )
            buf = self.gen._rep(torch.zeros(pt.shape, dtype=torch.int32), ttnn.int32)
            buffers[key] = buf
        ttnn.copy_host_to_device_tensor(self.gen._host(pt, ttnn.int32), buf)
        return buf

    def _prefill_pt(self, page_table):
        """Persistent attention page table, separate from decode and paged-fill tables."""
        return self._prefill_pt_into(page_table, self._pf_pt, purpose="attention")

    def _prefill_fill_pt(self, page_table):
        """Persistent paged-fill table whose padding entries use the -1 write-skip sentinel."""
        return self._prefill_pt_into(page_table, self._pf_fill_pt, purpose="fill")

    @staticmethod
    def _protect_prefill_padding_blocks(
        page_table, ranges, bucket_lens, *, block_size, scratch_block_idx, target_width=None
    ):
        """Build separate, equal-shaped attention and fill tables for bucketed prefill.

        vLLM allocates logical page-table entries only through each request's absolute real end;
        unused columns are zero-filled. Laguna still computes a fixed bucket ``L``. Attention must
        receive a valid physical id for every column it may read, so its padding maps to the
        adapter-private, zero-initialized scratch block. ``paged_fill_cache`` must not write those
        logical padding blocks: mapping them all to scratch lets independently scheduled writer cores
        race on the same physical tiles. Its table therefore uses the kernel's explicit ``-1``
        write-skip sentinel. Padding inside the final real block remains mapped normally; future slots
        are overwritten before use. The scheduler/decode table is never modified.

        The fill table is host-rebased independently per request row: its column zero
        represents that row's ``start_pos``. This removes arbitrary absolute device-slice
        starts from the serving path. Returns ``(attention_page_table, fill_page_table)``
        with identical fixed shapes.
        """
        pt = torch.as_tensor(page_table, dtype=torch.int32)
        if pt.dim() == 1:
            pt = pt.reshape(1, -1)
        if pt.dim() != 2:
            raise ValueError(f"prefill page_table must be rank 1 or 2, got shape {tuple(pt.shape)}")
        if len(ranges) != len(bucket_lens):
            raise ValueError(f"prefill has {len(ranges)} ranges but {len(bucket_lens)} bucket lengths")
        if int(pt.shape[0]) != len(ranges):
            raise ValueError(f"prefill page_table has {pt.shape[0]} rows for batch {len(ranges)}")
        bs = int(block_size)
        scratch = int(scratch_block_idx)
        if bs <= 0:
            raise ValueError(f"prefill KV block_size must be positive, got {bs}")
        if scratch < 0:
            raise ValueError(f"prefill scratch block id must be non-negative, got {scratch}")

        source_width = int(pt.shape[1])
        width = source_width if target_width is None else int(target_width)
        if width < source_width:
            raise ValueError(f"prefill target page-table width {width} is smaller than scheduler width {source_width}")
        # Fixed, equal widths keep both persistent PT buffers and their TT programs trace-stable.
        attention = torch.full((int(pt.shape[0]), width), scratch, dtype=torch.int32)
        fill = torch.full((int(pt.shape[0]), width), -1, dtype=torch.int32)
        attention[:, :source_width] = pt
        for u, ((start, end, chunk_len), bucket_len) in enumerate(zip(ranges, bucket_lens)):
            start, end, chunk_len, bucket_len = int(start), int(end), int(chunk_len), int(bucket_len)
            if start % bs != 0:
                raise ValueError(
                    f"prefill request {u} start_pos {start} is not aligned to KV block_size {bs}; "
                    "paged_fill_cache and chunked SDPA require block-aligned resumed prefills"
                )
            if chunk_len != end - start or chunk_len <= 0:
                raise ValueError(
                    f"prefill request {u} has inconsistent range [{start}, {end}) and chunk_len {chunk_len}"
                )
            if bucket_len < chunk_len:
                raise ValueError(
                    f"prefill request {u} bucket length {bucket_len} is smaller than chunk length {chunk_len}"
                )
            real_block_end = (end + bs - 1) // bs
            padded_block_end = (start + bucket_len + bs - 1) // bs
            start_block = start // bs
            if real_block_end > source_width:
                raise ValueError(
                    f"prefill request {u} real end needs logical block {real_block_end - 1}, but "
                    f"the scheduler page table has only {source_width} columns"
                )
            if padded_block_end > width:
                raise ValueError(
                    f"prefill request {u} bucket ends at logical block {padded_block_end}, beyond "
                    f"page-table width {width}; cannot safely redirect bucket padding"
                )
            attention[u, real_block_end:] = scratch
            # Rebase the owned suffix blocks to column zero. paged_fill_cache consumes update token
            # zero at page-table column zero, so this maps it to the physical block containing the
            # absolute aligned start without an offset-dependent device slice. Whole bucket-padding
            # blocks remain -1; padding within the last real block is harmless and is overwritten
            # before decode can consume those future positions.
            owned_blocks = real_block_end - start_block
            fill[u, :owned_blocks] = pt[u, start_block:real_block_end]
        return attention, fill

    def _prefill_page_table_width(self, block_size):
        """Fixed prefill-only page-table width for the padded compute horizon.

        Streaming only pads a single tail chunk, so the horizon is the logical
        context rounded once to the 8192-token outer boundary.  The rollback path
        retains the historical ``max_model_len + largest_bucket`` width.  Round to
        eight int32 columns for a 32-byte row-major stick.
        """
        bs = int(block_size)
        if bs <= 0:
            raise ValueError(f"prefill KV block_size must be positive, got {bs}")
        if self._streaming_prefill_active():
            outer = min(int(self._PREFILL_STREAM_OUTER_CHUNK), int(self.max_model_len))
            horizon = streaming_prefill_capacity(
                int(self.max_model_len),
                outer_chunk=outer,
            )
        else:
            horizon = int(self.max_model_len) + max(self._prefill_bucket_lens())
        blocks = (horizon + bs - 1) // bs
        return ((blocks + 7) // 8) * 8

    def _prepare_prefill_page_tables(
        self,
        page_table,
        page_tables_per_layer,
        kv_cache,
        ranges,
        bucket_lens,
        *,
        operation,
    ):
        """Build and upload paired attention/fill page tables for one prefill operation."""
        if not kv_cache:
            raise ValueError(f"{operation} requires an allocated KV cache")

        if page_tables_per_layer is None:
            cache_meta = kv_cache[0]
            if "scratch_block_idx" not in cache_meta:
                raise ValueError(f"{operation} KV cache is missing its adapter-private scratch block")
            attention_host, fill_host = self._protect_prefill_padding_blocks(
                page_table,
                ranges,
                bucket_lens,
                block_size=cache_meta["block_size"],
                scratch_block_idx=cache_meta["scratch_block_idx"],
                target_width=self._prefill_page_table_width(cache_meta["block_size"]),
            )
            return self._prefill_pt(attention_host), self._prefill_fill_pt(fill_host)

        # Construct a protected pair from every logical layer's scheduler table,
        # then validate/collapse identical members onto one persistent attention
        # and fill buffer per group.
        layer_count = len(self._group_kinds())
        if len(page_tables_per_layer) < layer_count:
            raise ValueError(
                f"{operation} page_tables_per_layer has {len(page_tables_per_layer)} entries "
                f"for {layer_count} built layers"
            )
        if len(kv_cache) < layer_count:
            raise ValueError(f"{operation} KV cache has {len(kv_cache)} entries for {layer_count} built layers")
        attention_hosts = []
        fill_hosts = []
        for i in range(layer_count):
            cache_meta = kv_cache[i]
            if "scratch_block_idx" not in cache_meta:
                raise ValueError(f"{operation} KV cache layer {i} is missing its adapter-private scratch block")
            attention_host, fill_host = self._protect_prefill_padding_blocks(
                page_tables_per_layer[i],
                ranges,
                bucket_lens,
                block_size=cache_meta["block_size"],
                scratch_block_idx=cache_meta["scratch_block_idx"],
                target_width=self._prefill_page_table_width(cache_meta["block_size"]),
            )
            attention_hosts.append(attention_host)
            fill_hosts.append(fill_host)
        return self._prefill_pt_grouped(attention_hosts), self._prefill_fill_pt_grouped(fill_hosts)

    @staticmethod
    def _sampling_row_params(sp, row):
        """Map one row of a vLLM ``TTSamplingParams`` to (k, p, temp, seed). temperature==0 → greedy
        top-1. top_k<=0 (disabled) → the device candidate-set width (32).

        No explicit seed (``sp.seed[row] is None``) means "sample randomly" — so a FRESH random seed is
        drawn per call (via ``secrets``, independent of the torch/global RNG which vLLM pins to seed 0).
        Defaulting to a fixed 0 instead makes identical no-seed requests deterministic and collapses
        temperature/top-k variety (the plugin's no-seed / temperature-varied / top-k variety tests)."""
        temp = float(sp.temperature[row]) if sp.temperature is not None else 1.0
        top_k = int(sp.top_k[row]) if sp.top_k is not None else 0
        top_p = float(sp.top_p[row]) if sp.top_p is not None else 1.0
        seed = sp.seed[row] if sp.seed is not None else None
        if temp <= 0.0:  # greedy — seed irrelevant (top-k(k=1) is deterministic)
            return 1, 1.0, 1.0, 0
        k = top_k if 0 < top_k <= 32 else 32
        p = top_p if 0.0 < top_p <= 1.0 else 1.0
        s = int(seed) if seed is not None else secrets.randbelow(2_000_000_000)
        return k, p, temp, s

    def _sampling_buffers_from_params(self, sp, B):
        """Build host [B] arrays of k/p/temp/seed from a vLLM TTSamplingParams (lists), padding to B
        with greedy defaults for inactive rows."""
        k = torch.ones(B, dtype=torch.int32)
        p = torch.ones(B, dtype=torch.float32)
        t = torch.ones(B, dtype=torch.float32)
        s = torch.zeros(B, dtype=torch.int32)
        n = 0 if sp is None or sp.temperature is None else len(sp.temperature)
        for row in range(min(n, B)):
            kk, pp, tt_, ss = self._sampling_row_params(sp, row)
            k[row], p[row], t[row], s[row] = kk, pp, tt_, ss
        return k, p, t, s

    # --------------------------------------------------------------------- #
    # Prefill — trace-safe (bucketed length + fixed-shape terminal)
    # --------------------------------------------------------------------- #
    # Under vLLM continuous batching a NEW-request prefill is interleaved between decode-trace
    # replays, i.e. it runs while the decode trace is RESIDENT. ttnn forbids device-buffer allocation
    # while a trace is resident ("Allocating device buffers is unsafe due to the existence of an
    # active trace", allocator.cpp) — any such allocation can corrupt the captured trace (garbage
    # tokens, then a device wedge). So prefill must run ONLY already-compiled programs over
    # already-allocated buffers. Two things make that true:
    #   (1) The prompt is right-padded to a BUCKET length so `prefill_layers` sees a fixed shape per
    #       bucket (a bounded set of programs, all pre-compiled by warmup_model_prefill BEFORE the
    #       decode trace is captured). Right-padding is safe: causal attention means the last REAL
    #       token (plen-1) never attends to the pad positions, so its logits are exact. Padding within
    #       the final scheduler-allocated block lands in future slots. Whole padding blocks map to a
    #       valid scratch block for attention reads but to the -1 write-skip sentinel for paged fill,
    #       so they neither address invalid memory nor race on scratch or physical block 0.
    #   (2) The last-real-token hidden is selected without baking plen into a new program per distinct
    #       length, and WITHOUT a host round-trip. `_last_token_shards` builds a tiny
    #       [1,1,1,L] one-hot on host (1.0 at column plen-1 — the index is DATA, not program shape),
    #       copies it into a persistent per-bucket-L selector buffer, and runs the fixed-shape matmul
    #       sel[1,1,1,L] @ h[1,1,L,H] -> [1,1,1,H] to pick row plen-1 on device. In bf16 the one-hot ·
    #       hidden reproduces the selected row bit-exactly (1.0*x + 0-sum), so greedy output is
    #       identical. This removes the ~32 MB whole-hidden readback that used to run on EVERY prefill.
    #       Both the selector buffers (allocated in _prefill_state) and the matmul program (compiled by
    #       warmup_model_prefill's per-L prefill_forward calls) exist pre-trace, so serving only copies
    #       the one-hot in and runs a pre-compiled matmul — no alloc/compile under the resident decode
    #       trace. Sampling likewise reuses persistent B=1 buffers (copy-in, no alloc).

    def _prefill_bucket_lens(self):
        """Finite compute shapes warmed before the resident decode trace.

        Chunk-major streaming needs only powers of two from one tile through the
        8192-token outer chunk.  Short cold requests use that ladder; long D2
        requests reuse the canonical 8192 program for complete chunks and the
        padded tail, eliminating monolithic 16K..131K hidden/concat shapes while
        keeping one SDPA reduction geometry.  The environment warm cap remains a
        development-only way to reduce this finite set.

        On D1, or with ``TT_LAGUNA_STREAMING_PREFILL=0``, this returns the
        historical ladder through the whole servable context for a safe rollback.
        """
        import os as _os

        servable = min(int(self.max_model_len), ADVERTISED_MAX_CONTEXT)
        streaming = self._streaming_prefill_active()
        normal_cap = min(servable, int(self._PREFILL_STREAM_OUTER_CHUNK)) if streaming else servable
        cap = normal_cap
        env = _os.environ.get("TT_LAGUNA_PREFILL_WARM_CAP")
        if env:
            cap = min(normal_cap, int(env))
            if cap <= 0:
                raise ValueError(f"TT_LAGUNA_PREFILL_WARM_CAP must be positive, got {env}")
            if cap < normal_cap and not getattr(type(self), "_warned_warm_cap", False):
                consequence = (
                    f"stream chunks longer than {cap} tokens will be rejected before device execution"
                    if streaming
                    else f"prompts longer than {cap} tokens will compile prefill programs under the resident "
                    "decode trace (very slow + trace-unsafe)"
                )
                print(
                    f"[laguna] WARNING: TT_LAGUNA_PREFILL_WARM_CAP={cap} < required compute shape "
                    f"{normal_cap}; "
                    f"{consequence}. Dev-only knob — unset for serving.",
                    flush=True,
                )
                type(self)._warned_warm_cap = True
        buckets, b = [], 32  # floor 32 (one tile) to match small cached-suffix prefills
        while b < cap:
            buckets.append(b)
            b *= 2
        buckets.append(cap)
        return sorted(set(x for x in buckets if x >= 1))

    def _bucket_len(self, plen):
        """Smallest warmed bucket for one independently executed compute chunk."""
        buckets = self._prefill_bucket_lens()
        for b in buckets:
            if int(plen) <= b:
                return b
        top = buckets[-1]
        if self._streaming_prefill_active():
            raise ValueError(
                f"prefill compute chunk {int(plen)} exceeds streaming outer chunk {top}; "
                "build a prefill_stream_plan instead of a monolithic bucket"
            )
        # Rollback-only monolithic guard for an out-of-contract request.
        return ((plen + top - 1) // top) * top

    def _prefill_stream_outer_chunk(self, block_size):
        """Return and validate the adapter/model outer-chunk contract."""

        bs = int(block_size)
        if bs <= 0:
            raise ValueError(f"prefill stream block size must be positive, got {bs}")
        configured = int(self._PREFILL_STREAM_OUTER_CHUNK)
        outer = min(configured, int(self.max_model_len), ADVERTISED_MAX_CONTEXT)
        if outer % bs:
            raise ValueError(f"prefill streaming outer chunk {outer} is not aligned to KV block size {bs}")
        layers = getattr(self.model, "layers", None)
        if layers and outer == configured:
            model_chunk = (int(layers[0]._prefill_pipe_chunk) // bs) * bs
            if model_chunk != configured:
                raise RuntimeError(
                    f"adapter streaming outer chunk is {configured}, but the decoder is configured "
                    f"for {model_chunk}; keep adapter planning and chunked SDPA identical"
                )
        return outer

    def _prefill_plan_for_range(self, chunk_len, start_pos, block_size):
        """Plan one scheduler range with canonical D2 long-stream geometry."""

        length = int(chunk_len)
        start = int(start_pos)
        if not self._streaming_prefill_active():
            bucket = self._prefill_bucket_for_range(length, start, block_size)
            return (PrefillStreamChunk(0, length, bucket),)

        outer = self._prefill_stream_outer_chunk(block_size)
        buckets = tuple(int(bucket) for bucket in self._prefill_bucket_lens() if int(bucket) <= outer)
        plan = prefill_stream_plan(
            length,
            bucket_lens=buckets,
            outer_chunk=outer,
            # A short, unsegmented cold request retains the finite 32..8192
            # ladder. Every later scheduler range, or a first range longer than
            # one outer chunk, uses the canonical 8192 shape on D2. This matches
            # the established monolithic kernel family's tail reduction while
            # still removing the 32768 power-of-two cliff (16400 real rows
            # compute 24576 rows).
            canonical_tail=int(self.D) == 2 and (start > 0 or length > outer),
        )
        # Prefix hits additionally enforce scheduler alignment.  Their compute
        # plan is already canonical under the D2 long-stream rule above; retain
        # the explicit remap as a fail-closed assertion of that contract.
        if bool(self._PREFIX_CACHE_ENABLED) and int(self.D) == 2 and start > 0:
            quantum = self._prefix_resume_quantum(block_size)
            if start % quantum:
                raise ValueError(
                    f"prefix-cache resumed prefill start_pos {start} is not aligned to canonical "
                    f"outer-chunk quantum {quantum}; cache-hit admission must truncate to a complete chunk"
                )
            plan = tuple(PrefillStreamChunk(chunk.relative_start, chunk.real_len, quantum) for chunk in plan)
        return plan

    def _prefill_bucket_for_range(self, chunk_len, start_pos, block_size):
        """Select one bucket for legacy/verify callers.

        Streaming prefill uses :meth:`_prefill_plan_for_range`; this helper remains
        for the rollback path and small speculative-verify prefills.
        """
        length = int(chunk_len)
        start = int(start_pos)
        bucket = int(self._bucket_len(length))
        if not (bool(self._PREFIX_CACHE_ENABLED) and int(self.D) == 2 and start > 0):
            return bucket
        if not getattr(self.model, "layers", None):
            raise ValueError("prefix-cache resumed prefill requires at least one decoder layer")
        quantum = self._prefix_resume_quantum(block_size)
        if start % quantum:
            raise ValueError(
                f"prefix-cache resumed prefill start_pos {start} is not aligned to canonical "
                f"outer-chunk quantum {quantum}; cache-hit admission must truncate to a complete chunk"
            )
        bucket = max(bucket, quantum)
        if bucket % quantum:
            raise ValueError(
                f"prefix-cache resumed prefill bucket {bucket} is not a whole multiple of "
                f"canonical quantum {quantum}"
            )
        warmed = set(int(value) for value in self._prefill_bucket_lens())
        if bucket not in warmed:
            raise ValueError(
                f"prefix-cache resumed prefill bucket {bucket} is not in the warmed ladder; "
                f"canonical quantum is {quantum}"
            )
        return bucket

    def _prefix_resume_quantum(self, block_size):
        """Return the qualified outer chunk and reject model/admission contract drift."""
        bs = int(block_size)
        if bs <= 0:
            raise ValueError(f"prefix-cache resumed prefill block size must be positive, got {bs}")
        quantum = (int(self.model.layers[0]._prefill_pipe_chunk) // bs) * bs
        if quantum <= 0:
            raise ValueError(
                f"prefix-cache resumed prefill outer chunk must be at least block size {bs}, got {quantum}"
            )
        qualified = int(self._PREFIX_CACHE_QUANTUM)
        if quantum != qualified:
            raise RuntimeError(
                f"prefix caching requires prefill outer-chunk quantum {qualified}, but the model "
                f"is configured for {quantum}; keep cache-hit admission and model chunking identical"
            )
        return quantum

    def _allocate_prefill_runtime_offsets(self, st, block_size):
        """Allocate every runtime-offset/RoPE slot used by the warmed bucket ladder.

        Slots are shared across buckets by ``(single|pipe, chunk ordinal, chunk length)``. A long
        request needs every outer chunk's RoPE matrices live until all layers consume them, so
        different chunk ordinals have distinct outputs; smaller buckets reuse the leading slots.
        This keeps the fully hoisted 131K geometry bounded (~102 MiB/chip for both attention kinds)
        instead of allocating one complete output set per bucket (~192 MiB/chip).
        """
        bs = int(block_size)
        if bs <= 0:
            raise ValueError(f"prefill runtime KV block size must be positive, got {bs}")
        existing_bs = st.get("runtime_block_size")
        if existing_bs is not None:
            if int(existing_bs) != bs:
                raise ValueError(
                    f"prefill runtime was allocated for KV block size {existing_bs}, cannot reuse it for {bs}"
                )
            return
        if not getattr(self.model, "layers", None):
            raise ValueError("prefill runtime allocation requires at least one decoder layer")

        first = self.model.layers[0]
        pipe_threshold = int(first.PIPE_CHUNK)
        outer_chunk = (int(first._prefill_pipe_chunk) // bs) * bs
        buckets = tuple(self._prefill_bucket_lens())
        plans = {
            int(L): prefill_chunk_plan(
                int(L),
                pipe_threshold=pipe_threshold,
                outer_chunk=outer_chunk,
                block_size=bs,
            )
            for L in buckets
        }

        rotary_dims = {}
        for dec in self.model.layers:
            kind = str(dec.cfg.attention_type)
            rd = int(dec.cfg.rotary_dim)
            prior = rotary_dims.setdefault(kind, rd)
            if prior != rd:
                raise ValueError(f"attention kind {kind!r} has inconsistent rotary dims {prior} and {rd}")

        # Input and output tensors are persistent replicated DRAM allocations made before trace capture.
        # Values are refreshed with host-to-device copies; TTNN programs depend on shapes, not contents.
        input_slots = {}
        rope_slots = {kind: {} for kind in rotary_dims}
        bucket_slots = {}
        for L, plan in plans.items():
            pipelined = L > pipe_threshold
            keys = []
            for ordinal, (_offset, chunk_len) in enumerate(plan):
                key = ("pipe" if pipelined else "single", ordinal, int(chunk_len))
                keys.append(key)
                if key not in input_slots:
                    pos = self.gen._rep(torch.zeros([1, chunk_len], dtype=torch.int32), ttnn.uint32)
                    start = self.gen._rep(torch.zeros([1], dtype=torch.int32), ttnn.int32)
                    input_slots[key] = (pos, start)
                    for kind, rd in rotary_dims.items():
                        cos = self.gen._rep(
                            torch.zeros([1, 1, chunk_len, rd], dtype=torch.float32),
                            ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                        )
                        sin = self.gen._rep(
                            torch.zeros([1, 1, chunk_len, rd], dtype=torch.float32),
                            ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                        )
                        rope_slots[kind][key] = (cos, sin)
            bucket_slots[L] = tuple(keys)

        runtimes = {}
        for L, plan in plans.items():
            keys = bucket_slots[L]
            runtimes[L] = PrefillRuntimeOffsets(
                bucket_len=L,
                chunk_offsets=tuple(offset for offset, _length in plan),
                chunk_lengths=tuple(length for _offset, length in plan),
                position_ids=tuple(input_slots[key][0] for key in keys),
                chunk_start_idxs=tuple(input_slots[key][1] for key in keys),
                rope_outputs={kind: tuple(rope_slots[kind][key] for key in keys) for kind in rotary_dims},
            )

        st["runtime_block_size"] = bs
        st["runtime_offsets"] = runtimes
        st["runtime_slot_count"] = len(input_slots)

    def _refresh_prefill_runtime_offsets(self, bucket_len, start_pos, block_size):
        """Refresh persistent position IDs and absolute SDPA starts for one request."""
        L, start, bs = int(bucket_len), int(start_pos), int(block_size)
        st = self._prefill_state(bs)
        runtime = st["runtime_offsets"].get(L)
        if runtime is None:
            raise ValueError(
                f"prefill bucket {L} has no persistent runtime-offset buffers; warm the exact serving ladder"
            )
        if start < 0:
            raise ValueError(f"prefill runtime start_pos must be non-negative, got {start}")
        if start % bs:
            raise ValueError(f"prefill runtime start_pos {start} is not aligned to KV block size {bs}")
        rope_capacity = int(getattr(self.model, "meta", {}).get("max_seq_len", self.max_model_len))
        if start + L > rope_capacity:
            raise ValueError(f"prefill padded RoPE range [{start}, {start + L}) exceeds table capacity {rope_capacity}")
        for offset, length, pos_buf, start_buf in zip(
            runtime.chunk_offsets,
            runtime.chunk_lengths,
            runtime.position_ids,
            runtime.chunk_start_idxs,
        ):
            absolute = start + offset
            positions = torch.arange(absolute, absolute + length, dtype=torch.int32).reshape(1, length)
            ttnn.copy_host_to_device_tensor(self.gen._host(positions, ttnn.uint32), pos_buf)
            ttnn.copy_host_to_device_tensor(
                self.gen._host(torch.tensor([absolute], dtype=torch.int32), ttnn.int32), start_buf
            )
        return runtime

    def _runtime_offsets_for_prefill(self, bucket_len, start_pos, block_size):
        """Return qualified runtime inputs without perturbing legacy/cold single-shot paths."""
        if int(self.D) != 2:
            return None
        L, start = int(bucket_len), int(start_pos)
        if start == 0 and L <= int(self.model.layers[0].PIPE_CHUNK):
            # Preserve the established cold single-shot RoPE slice + local SDPA path exactly.
            return None
        return self._refresh_prefill_runtime_offsets(L, start, block_size)

    def _prefill_stream_warm_cases(self, block_size):
        """Return the canonical start>0 long-stream warm case.

        Running the returned cold prompt through :meth:`prefill_forward` executes
        two complete outer-query programs, the second at a nonzero absolute start.
        Short cold requests are covered separately by the finite bucket ladder.
        """

        if not self._streaming_prefill_active():
            return ()
        outer = self._prefill_stream_outer_chunk(block_size)
        max_tail = min(outer, max(0, int(self.max_model_len) - outer))
        if max_tail <= 0:
            return ()
        return ((outer, outer + max_tail),)

    def _prefill_state(self, block_size=None):
        """Allocate (once) the persistent prefill sampling buffers + B=1 sampler. Called from
        warmup_model_prefill BEFORE any decode trace is captured, so these allocations are safe."""
        if self._pf is not None:
            if block_size is not None and int(getattr(self, "D", 0)) == 2:
                self._allocate_prefill_runtime_offsets(self._pf, block_size)
            return self._pf
        z = torch.zeros([1], dtype=torch.int32)
        st = dict(
            tok=self.gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32),
            k=self.gen._rep(torch.ones([1], dtype=torch.int32), ttnn.uint32),
            p=self.gen._rep(torch.ones([1], dtype=torch.float32), ttnn.bfloat16),
            t=self.gen._rep(torch.ones([1], dtype=torch.float32), ttnn.bfloat16),
            seeds=self.gen._rep(z, ttnn.uint32),
            sampler=self.gen._sampler(1),
            # Persistent [1,1,1,H] buffer holding the selected last-real-token hidden. Fixed shape →
            # the terminal norm+LM-head+sample program is compiled ONCE (warmup) and reused, never
            # recompiled per prompt length under the resident decode trace.
            last_h=self.gen._rep(
                torch.zeros([1, 1, 1, self.hidden], dtype=torch.float32), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            ),
        )
        # persistent per-bucket-L on-device last-token SELECTOR buffers. Each is a
        # [1,1,1,L] bf16 (TILE, replicated) one-hot INPUT; the fixed-shape matmul
        # ``sel[1,1,1,L] @ h[1,1,L,H] -> [1,1,1,H]`` picks the last REAL row (the row index is DATA,
        # written per prompt via copy_host_to_device_tensor), replacing the ~32 MB host readback of
        # the whole bucketed hidden that ran on every prefill. Allocated HERE (pre-trace, once, keyed
        # by L exactly like the ``_pf_pt`` page tables) for EVERY warmed bucket — so at serve time the
        # selector matmul is a copy-in + pre-compiled matmul with no device allocation / compilation
        # under the resident decode trace. The matmul program itself is compiled during
        # ``warmup_model_prefill`` (its per-bucket ``prefill_forward`` calls run ``_last_token_shards``,
        # which executes this same ``sel @ h`` for each L). bf16 one-hot · bf16 hidden reproduces the
        # selected row bit-exactly (1.0*x + 0-sum), so greedy output is identical to the readback path.
        st["sel"] = {
            L: self.gen._rep(torch.zeros([1, 1, 1, L], dtype=torch.float32), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for L in self._prefill_bucket_lens()
        }
        if block_size is not None and int(getattr(self, "D", 0)) == 2:
            self._allocate_prefill_runtime_offsets(st, block_size)
        self._pf = st
        return st

    def _select_hidden_row(self, h, row, L, st):
        """Copy a host one-hot selector in and write the selected hidden row to ``st["last_h"]``.

        Both device tensors are persistent and allocated by ``_prefill_state`` before trace capture.
        Supplying ``last_h`` as the matmul output is essential: without it, TTNN allocates a fresh
        device output for every prefill while the decode trace is resident.
        """
        sel = st["sel"].get(L)
        if sel is None:
            return None
        if not 0 <= int(row) < int(L):
            raise ValueError(f"selector row {row} is outside bucket length {L}")

        onehot = torch.zeros([1, 1, 1, L], dtype=torch.float32)
        onehot[0, 0, 0, int(row)] = 1.0
        # No ``device=``: this is a host TT tensor and cannot allocate device memory under the
        # resident decode trace. The only device operation here copies into the persistent selector.
        src = ttnn.from_torch(
            onehot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=_replicate(self.mesh_device)
        )
        ttnn.copy_host_to_device_tensor(src, sel)
        h4 = ttnn.reshape(h, (1, 1, L, self.hidden))
        ttnn.matmul(sel, h4, optional_output_tensor=st["last_h"])
        return st["last_h"]

    def _last_token_shards(self, h, plen, L):
        """Select the last REAL token's logit shards with a fixed-shape ON-DEVICE one-hot selector.

        ``h`` is the bucketed prefill output ``[1, L, H]`` (L fixed per bucket) held on device and
        REPLICATED across the mesh. Instead of reading the whole ``[1,L,H]`` hidden back to
        host (~32 MB every prefill) to slice row ``plen-1``, build a tiny ``[1,1,1,L]`` one-hot on host
        (1.0 at column ``plen-1`` — the index is DATA, not program shape), copy it into the persistent
        per-L selector buffer (a COPY, no allocation), and run the fixed-shape matmul
        ``sel[1,1,1,L] @ h[1,1,L,H] -> [1,1,1,H]``. bf16 one-hot · bf16 hidden reproduces the selected
        row bit-exactly (the only nonzero term is ``1.0 * h[plen-1]`` and 1.0 is exact in bf16; every
        other product is 0), so the column-sharded LM head — and therefore greedy output — is identical
        to the old readback path. The selector matmul + the ``[1,L,H]->[1,1,L,H]`` reshape are compiled
        pre-trace by warmup (its ``prefill_forward`` calls run this for every bucket L). Decode never
        leaves the device; this is prefill-only. Falls back to the host readback if a bucket's selector
        is somehow missing (shouldn't happen post-warmup) so serving can never crash."""
        st = self._prefill_state()
        sel_row = self._select_hidden_row(h, plen - 1, L, st)
        if sel_row is not None:
            return self.model.lm_head_shards_decode(sel_row)
        # Fallback (bucket L not warmed): host readback of the replicated hidden, slice row on host.
        hh = ttnn.to_torch(h, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).reshape(
            -1, L, self.hidden
        )
        hrow = hh[0, plen - 1].to(torch.float32).reshape(1, 1, 1, self.hidden)
        hsrc = ttnn.from_torch(
            hrow, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=_replicate(self.mesh_device)
        )
        ttnn.copy_host_to_device_tensor(hsrc, st["last_h"])
        return self.model.lm_head_shards_decode(st["last_h"])

    def _refresh_prefill_sampling(self, st, sp, u):
        """Copy per-request k/p/temp/seed into the persistent B=1 sampling buffers (no allocation)."""
        k, p, t, s = self._sampling_row_params(sp, u)
        ttnn.copy_host_to_device_tensor(self.gen._host(torch.tensor([k], dtype=torch.int32), ttnn.uint32), st["k"])
        ttnn.copy_host_to_device_tensor(self.gen._host(torch.tensor([p], dtype=torch.float32), ttnn.bfloat16), st["p"])
        ttnn.copy_host_to_device_tensor(self.gen._host(torch.tensor([t], dtype=torch.float32), ttnn.bfloat16), st["t"])
        ttnn.copy_host_to_device_tensor(self.gen._host(torch.tensor([s], dtype=torch.int32), ttnn.uint32), st["seeds"])

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        start_pos=None,
        enable_trace=False,
        sampling_params=None,
        empty_slots=None,
        page_tables_per_layer=None,
        **kwargs,
    ):
        """One prefill step. ``tokens`` [num_reqs, padded_seq] int32, ``page_table`` [num_reqs, nb].
        Host-sampling (``sampling_params is None``): returns logits ``[num_reqs, 1, vocab]``.
        Device-sampling: samples the last position on device and returns ``(tokens[num_reqs,1], None)``.
        The plugin supplies the full token row and absolute half-open bounds for the work scheduled
        in this step: ``tokens[u, start_pos[u]:prompt_lens[u]]``. ``start_pos`` is retained as the
        absolute KV/RoPE position, while bucketing and last-row selection use the relative chunk
        length ``prompt_lens[u] - start_pos[u]``. On D2, complete 8192-token chunks run through the
        full layer stack before one canonical 8192-query padded tail for a long stream; only that
        final real chunk reaches the LM head and sampler. Short cold requests retain the finite bucket
        ladder, and D1 retains one monolithic bucket. In all cases the scheduled length may be any
        positive value up to context and only precompiled, trace-safe compute shapes are used."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() != 2:
            raise ValueError(f"prefill tokens must be rank 1 or 2, got shape {tuple(tokens.shape)}")
        batch = tokens.shape[0]
        if batch == 0:
            raise ValueError("prefill tokens must contain at least one request")
        token_width = int(tokens.shape[1])
        ends = [token_width] * batch if prompt_lens is None else [int(x) for x in prompt_lens]
        starts = [0] * batch if start_pos is None else [int(x) for x in start_pos]
        if len(ends) != batch:
            raise ValueError(f"prompt_lens has {len(ends)} entries for prefill batch {batch}")
        if len(starts) != batch:
            raise ValueError(f"start_pos has {len(starts)} entries for prefill batch {batch}")
        ranges = []
        for u, (start, end) in enumerate(zip(starts, ends)):
            if start < 0:
                raise ValueError(f"prefill request {u} has negative start_pos {start}")
            if end <= start:
                raise ValueError(
                    f"prefill request {u} has empty or reversed range [{start}, {end}); "
                    "a prefill step must schedule at least one token"
                )
            if end > token_width:
                raise ValueError(f"prefill request {u} ends at {end}, beyond supplied token width {token_width}")
            if end > int(self.max_model_len):
                raise ValueError(f"prefill request {u} ends at {end}, beyond max_model_len {self.max_model_len}")
            ranges.append((start, end, end - start))

        if not kv_cache:
            raise ValueError("prefill requires an allocated KV cache")
        self._validate_page_table_mode(
            kv_cache,
            page_tables_per_layer,
            operation="prefill",
        )
        runtime_bs = int(kv_cache[0]["block_size"])
        if any(int(entry["block_size"]) != runtime_bs for entry in kv_cache):
            raise ValueError("resumed prefill runtime offsets require a uniform KV block size across layers")
        plans = [self._prefill_plan_for_range(chunk_len, start, runtime_bs) for start, _, chunk_len in ranges]
        compute_spans = [plan[-1].relative_start + plan[-1].bucket_len for plan in plans]
        if bool(self._PREFIX_CACHE_ENABLED) and int(self.D) == 2:
            for u, ((start, end, _), plan) in enumerate(zip(ranges, plans)):
                if start > 0:
                    quantum = self._prefix_resume_quantum(runtime_bs)
                    buckets = [chunk.bucket_len for chunk in plan]
                    bucket_summary = (
                        f"compute_bucket={buckets[0]}" if len(buckets) == 1 else f"compute_buckets={buckets}"
                    )
                    print(
                        "[laguna prefix] resume "
                        f"request={u} scheduled_start={start} effective_start={start} "
                        f"real_end={end} {bucket_summary} canonical_chunk={quantum}",
                        flush=True,
                    )
        if self._spec_mode == "1" and batch == 1:
            # Stash the prompt token sequence for ngram seeding — on the greedy served path the plugin
            # passes NO prompt_tokens/output_tokens (those are penalty-gated, model_runner.py:1051), so
            # this prefill is the only place the running request's prompt is visible. Offset-write handles
            # chunked prefill (multiple calls with increasing start_pos); start_pos 0 begins a new request.
            try:
                s, end, _ = ranges[0]
                row = [int(v) for v in tokens[0, s:end].tolist()]
                if s == 0:
                    self._spec_prefill_seq = list(row)
                    self._spec_next_pos = None  # force reseed on the first decode of this request
                else:
                    need = end
                    if len(self._spec_prefill_seq) < need:
                        self._spec_prefill_seq += [0] * (need - len(self._spec_prefill_seq))
                    self._spec_prefill_seq[s:end] = row
            except Exception:  # noqa: BLE001 - diagnostic stash; never break prefill
                pass

        device_sampling = sampling_params is not None
        st = self._prefill_state() if device_sampling else None
        last_logits = []
        sampled = []
        for u, ((request_start, _request_end, _request_len), plan) in enumerate(zip(ranges, plans)):
            final_hidden = None
            final_real_len = None
            final_bucket_len = None
            for chunk_idx, chunk in enumerate(plan):
                absolute_start = request_start + chunk.relative_start
                absolute_end = absolute_start + chunk.real_len
                L = int(chunk.bucket_len)

                # Protect and rebase the active row for this absolute subchunk.
                # Other rows retain whole-request guards; they are not executed in
                # this iteration but keep the shared [batch,width] buffer valid.
                protected_ranges = list(ranges)
                protected_buckets = list(compute_spans)
                protected_ranges[u] = (absolute_start, absolute_end, int(chunk.real_len))
                protected_buckets[u] = L
                pt, fill_pt = self._prepare_prefill_page_tables(
                    page_table,
                    page_tables_per_layer,
                    kv_cache,
                    protected_ranges,
                    protected_buckets,
                    operation=f"prefill request {u} stream chunk {chunk_idx}",
                )

                runtime_offsets = self._runtime_offsets_for_prefill(L, absolute_start, runtime_bs)
                padded = torch.zeros(L, dtype=torch.int64)
                padded[: chunk.real_len] = tokens[u, absolute_start:absolute_end]
                tok_tt = self.gen._tokens_to_device(padded)
                x = self.model.embed_prefill(tok_tt)
                if self._DFLASH_SERVING_ENABLED:
                    h, capture = self.model.prefill_layers_with_dflash_aux(
                        x,
                        kv_cache,
                        pt,
                        fill_page_table=fill_pt,
                        fill_page_table_base_pos=absolute_start,
                        user_id=u,
                        start_pos=absolute_start,
                        runtime_offsets=runtime_offsets,
                        valid_seq_len=chunk.real_len,
                        enable_experimental=True,
                    )
                    new_request = absolute_start == 0
                    if new_request:
                        self._dflash_request_serial += 1
                        self._dflash_request_id = f"vllm-{self._dflash_request_serial}"
                    if self._dflash_request_id is None:
                        raise RuntimeError("DFlash cache-off prefill must begin at absolute position zero")
                    self._dflash_controller.ingest_prefill_capture(
                        self._dflash_request_id,
                        capture,
                        new_request=new_request,
                    )
                else:
                    h = self.model.prefill_layers(
                        x,
                        kv_cache,
                        pt,
                        fill_page_table=fill_pt,
                        fill_page_table_base_pos=absolute_start,
                        user_id=u,
                        start_pos=absolute_start,
                        runtime_offsets=runtime_offsets,
                    )
                if chunk_idx == len(plan) - 1:
                    final_hidden = h
                    final_real_len = int(chunk.real_len)
                    final_bucket_len = L

            if final_hidden is None or final_real_len is None or final_bucket_len is None:
                raise RuntimeError(f"prefill request {u} produced no terminal stream chunk")
            # Only the final real chunk reaches the norm/LM-head/sampler. Earlier
            # chunks exist solely to materialize every layer's causal K/V.
            shards = self._last_token_shards(final_hidden, final_real_len, final_bucket_len)
            if device_sampling:
                self._refresh_prefill_sampling(st, sampling_params, u)
                st["sampler"].decode_forward(
                    shards, k=st["k"], p=st["p"], temp=st["t"], seeds=st["seeds"], tt_out_tok=st["tok"]
                )
                sampled.append(self.gen._read_token(st["tok"], 1)[0])
            else:
                logits = self.model.logits_to_host(shards).reshape(1, self.vocab)
                last_logits.append(logits)
        if device_sampling:
            toks = torch.tensor(sampled, dtype=torch.int64).reshape(batch, 1)
            return toks, None
        return torch.stack(last_logits, dim=0)  # [num_reqs, 1, vocab]

    def _row_logits(self, h, row, L, st):
        """LM-head over a single row of the ON-DEVICE prefill hidden ``h`` ([1,L,H], replicated),
        selected by the same fixed-shape one-hot matmul as ``_last_token_shards``: a
        ``[1,1,1,L]`` one-hot with 1.0 at ``row`` picks that row bit-exactly. Reuses the persistent
        per-L selector buffer ``st["sel"][L]`` (same L bucket, so the matmul program is already warmed
        by ``_last_token_shards``). Avoids the ~32 MB whole-hidden readback the verify path used to do.
        Falls back to a per-row host readback if the bucket's selector is missing (shouldn't happen)."""
        sel_row = self._select_hidden_row(h, row, L, st)
        if sel_row is not None:
            return self.model.logits_to_host(self.model.lm_head_shards_decode(sel_row)).reshape(self.vocab)
        # Fallback: read the hidden back and slice this row on host.
        hh = ttnn.to_torch(h, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).reshape(
            -1, L, self.hidden
        )
        hrow = hh[0, row].to(torch.float32).reshape(1, 1, 1, self.hidden)
        hsrc = ttnn.from_torch(
            hrow, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=_replicate(self.mesh_device)
        )
        ttnn.copy_host_to_device_tensor(hsrc, st["last_h"])
        return self.model.logits_to_host(self.model.lm_head_shards_decode(st["last_h"])).reshape(self.vocab)

    def verify_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        page_tables_per_layer=None,
        logit_rows=None,
        **kwargs,
    ):
        """Speculative-decode VERIFY step for ONE request (batch-1 path).

        ``tokens`` is [1, S] processed through the (suffix-)prefill path against the paged KV starting
        at ``start_pos``. Returns host logits — row j is the next-token distribution given context
        through position ``start_pos+j`` (i.e. predicts the token at ``start_pos+j+1``), so
        argmax(row j) is the target-greedy token for that slot.

        **Alignment:** the prefill flash-attention requires ``chunk_start_idx (= start_pos) % 64 == 0``
        for suffix prefills (``start_pos>0``; q_chunk=32 ∧ k_chunk=64/128 → lcm 64 = the block size,
        see optimized_decoder.py:418-429). The caller (spec_decode.SpeculativeDecoder) therefore aligns
        ``start_pos`` down to a 64-boundary and prepends the already-known history tokens for that
        window; those real tokens rewrite identical KV (idempotent) and only the trailing rows are read.

        ``logit_rows``: optional list of row indices to run the LM head on (in order). Only those rows'
        logits are returned as ``[len(logit_rows), vocab]`` — the caller asks for just the K+1 trailing
        rows (anchor + drafts), skipping the vocab projection for the re-fed alignment prefix. ``None``
        returns all ``S`` rows.

        KV for all S positions is written; rejected-draft positions are overwritten by the next
        iteration's verify (implicit batch-1 rollback), and the pad/right-fill keeps stale future-KV
        harmless."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(1, -1)
        S = int(tokens.shape[1])
        start = int(start_pos)
        end = start + S
        if start < 0 or end > int(self.max_model_len):
            raise ValueError(f"verify prefill range [{start}, {end}) is outside max_model_len {self.max_model_len}")
        if not kv_cache:
            raise ValueError("verify prefill requires an allocated KV cache")
        self._validate_page_table_mode(
            kv_cache,
            page_tables_per_layer,
            operation="verify prefill",
        )
        st = self._prefill_state()
        L = self._bucket_len(S)
        pt, fill_pt = self._prepare_prefill_page_tables(
            page_table,
            page_tables_per_layer,
            kv_cache,
            ranges=[(start, end, S)],
            bucket_lens=[L],
            operation="verify prefill",
        )
        runtime_bs = int(kv_cache[0]["block_size"])
        if any(int(entry["block_size"]) != runtime_bs for entry in kv_cache):
            raise ValueError("verify prefill runtime offsets require a uniform KV block size across layers")
        runtime_offsets = self._runtime_offsets_for_prefill(L, start, runtime_bs)
        padded = torch.zeros(L, dtype=torch.int64)
        padded[:S] = tokens[0, :S]
        tok_tt = self.gen._tokens_to_device(padded)
        x = self.model.embed_prefill(tok_tt)
        h = self.model.prefill_layers(
            x,
            kv_cache,
            pt,
            fill_page_table=fill_pt,
            fill_page_table_base_pos=start,
            user_id=0,
            start_pos=start,
            runtime_offsets=runtime_offsets,
        )
        # select each requested row ON DEVICE (one-hot matmul over the still-resident hidden)
        # instead of reading the whole [1,L,H] hidden back to host first.
        rows = list(range(S)) if logit_rows is None else [int(r) for r in logit_rows]
        logits = torch.stack([self._row_logits(h, r, L, st) for r in rows], dim=0)  # [len(rows), vocab]
        return logits

    def _spec_log(self, msg):
        """Append a diagnostic line to the spec-probe file AND stdout. The model runs in an MPI worker
        whose stdout is not captured in the readiness log, so the file is the reliable sink."""
        line = f"[laguna spec] {msg}"
        try:
            print(line, flush=True)
        except Exception:
            pass
        try:
            import os as _os

            _os.makedirs(_os.path.dirname(self._spec_log_path), exist_ok=True)
            with open(self._spec_log_path, "a") as _f:
                _f.write(line + "\n")
        except Exception:
            pass

    def _spec_feasibility_probe(self, tokens, pos, page_table, kv_cache, page_tables_per_layer):
        """PHASE-2 FEASIBILITY PROBE (TT_LAGUNA_SPEC_DECODE=probe): run ONE eager batched-decode VERIFY
        (K1=2 = anchor + 1 dummy draft) under the RESIDENT decode trace and report whether it completes
        without an alloc-under-trace hang. This is the open question blocking eager in-adapter spec-decode
        (the eager verify allocates activation buffers; doing that under a resident trace may be the
        allocator.cpp:123 hazard). Writes throwaway KV at pos+1 -> probe boot only, never real serving."""
        import time as _t

        try:
            anchor = int(tokens[0, 0])
            p0 = int(pos[0])
            toks = [anchor, anchor]  # K1=2: anchor + one dummy draft at the next position
            positions = [p0, p0 + 1]
            pt_arg = None if page_tables_per_layer is not None else page_table
            self._spec_log(
                f"PROBE START pid={os.getpid()} anchor={anchor} pos={p0} hybrid={page_tables_per_layer is not None}"
            )
            # Run 3x: iter 0 = compile (slow), iters 1-2 = WARM. Warm eager-verify time vs a ~35ms traced
            # decode step is the go/no-go for eager spec-serving (decode is dispatch-bound; eager = full
            # host dispatch, which tracing eliminates). Compare warm ms to draft_len to judge break-even.
            for it in range(3):
                t0 = _t.perf_counter()
                g = self.verify_greedy_decode(
                    toks,
                    positions,
                    page_table=pt_arg,
                    kv_cache=kv_cache,
                    page_tables_per_layer=page_tables_per_layer,
                    traced=False,
                )
                dt = (_t.perf_counter() - t0) * 1000.0
                self._spec_log(
                    f"PROBE iter{it} ({'compile' if it == 0 else 'WARM'}): {dt:.0f}ms -> {[int(x) for x in g]}"
                )
            self._spec_log(
                "PROBE OK: eager verify under resident decode trace completed (no hang) -> FEASIBLE. "
                "Compare the WARM ms above to a ~35ms traced decode step to judge if eager spec can win."
            )
        except Exception as e:  # noqa: BLE001 - diagnostic probe, report any failure verbatim
            self._spec_log(f"PROBE FAILED under resident decode trace: {type(e).__name__}: {e}")

    def _spec_is_greedy(self, sampling_params):
        try:
            t = sampling_params.temperature
            return float(t[0] if hasattr(t, "__len__") else t) <= 0.0
        except Exception:  # noqa: BLE001
            return False

    def _spec_history(self, prompt_tokens, output_tokens, tokens):
        """Full current token sequence for the single served request (ngram source + position anchor)."""

        def _row0(x):
            if x is None:
                return []
            r = x[0] if hasattr(x, "__getitem__") else x
            return [int(v) for v in (r.tolist() if hasattr(r, "tolist") else r)]

        hist = _row0(prompt_tokens) + _row0(output_tokens)
        return hist or [int(torch.as_tensor(tokens).reshape(-1)[0])]

    def _dflash_serve(
        self,
        tokens,
        pos,
        page_table,
        kv_cache,
        page_tables_per_layer,
        sampling_params,
        read_from_device,
    ):
        """Return one buffered exact-greedy DFlash commit to vLLM."""

        if self._dflash_controller is None or self._dflash_tok is None:
            raise RuntimeError("DFlash serving was enabled but its request controller is unavailable")
        if int(tokens.shape[0]) != 1:
            raise RuntimeError(f"DFlash served decoding requires B=1, got B={tokens.shape[0]}")
        if page_tables_per_layer is not None:
            raise RuntimeError("DFlash served decoding does not support hybrid per-layer page tables")
        if sampling_params is None or not self._spec_is_greedy(sampling_params):
            raise RuntimeError("DFlash served decoding is exact-greedy only")
        try:
            block_sizes = {int(entry["block_size"]) for entry in kv_cache}
        except (KeyError, TypeError) as error:
            raise RuntimeError("DFlash served decoding requires explicit KV block-size metadata") from error
        if block_sizes != {64}:
            raise RuntimeError(f"DFlash served decoding requires uniform 64-token KV blocks, got {block_sizes}")
        position = int(pos.reshape(-1)[0])
        pending = bool(self._dflash_controller.pending_tokens)
        proposal_rows = int(self._dflash_core.config.block_size)
        # A full round can commit the target bonus at position P+16.  An exact
        # fit is valid; a new round beyond it must fail rather than emit outside
        # scheduler admission.  Already-verified buffered tokens remain safe.
        if not pending and position + proposal_rows > int(self.max_model_len):
            raise RuntimeError(
                f"DFlash full-round verify at position {position} would exceed " f"max_model_len {self.max_model_len}"
            )
        known_bonus = int(tokens.reshape(-1)[0])
        verify_kwargs = {
            "page_table": page_table,
            "kv_cache": kv_cache,
            "page_tables_per_layer": None,
        }
        # vLLM allocates the block containing the current input, but does not
        # reserve the next block for fixed speculative look-ahead.  At residues
        # 49..63, a 16-row verify would cross that ownership boundary.  Advance
        # with one exact target+aux row until the next block is allocated.
        target_only = not pending and position % 64 > 48
        serve = self._dflash_controller.serve_target_token if target_only else self._dflash_controller.serve_token
        token_id = serve(
            known_bonus=known_bonus,
            position=position,
            verify_kwargs=verify_kwargs,
        )
        ttnn.copy_host_to_device_tensor(
            self._host_rank4_tok_batch(torch.tensor([[token_id]], dtype=torch.int64), 1),
            self._dflash_tok,
        )
        if read_from_device:
            return self._read_tokens_host(self._dflash_tok, 1)
        return [self._dflash_tok]

    def _spec_serve(
        self, tokens, pos, page_table, kv_cache, page_tables_per_layer, reset_batch, kwargs, read_from_device
    ):
        """One served decode step via TRACED spec-decode. Runs one spec round when the commit buffer is
        empty, then returns the buffered committed tokens one per vLLM step (plugin reads self._spec_tok).
        Bounded-K keeps the round's look-ahead KV writes inside the current allocated block. Needs warmup to
        have captured the verify traces + omitted the normal decode trace (TT_LAGUNA_SPEC_DECODE=1)."""
        if self._spec is None:
            from models.autoports.poolside_laguna_xs_2_1.tt.spec_decode import SpeculativeDecoder

            k_max = int(os.environ.get("TT_LAGUNA_SPEC_K", "4"))
            traced = os.environ.get("TT_LAGUNA_SPEC_TRACED", "1") == "1"  # 0 = eager verify (bisection)
            single = os.environ.get("TT_LAGUNA_SPEC_SINGLE", "") == "1"  # 1 = fixed-K, one resident verify trace
            self._spec = SpeculativeDecoder(
                self,
                kv_cache=kv_cache,
                page_table=page_table,
                page_tables_per_layer=page_tables_per_layer,
                stop_tokens=None,
                draft_len=k_max,
                ngram_max_n=int(os.environ.get("TT_LAGUNA_SPEC_NGRAM_MAX", "10")),
                verify_mode="decode",
                traced=traced,
                adaptive=not single,  # single mode: fixed K=k_max so only the K1=k_max+1 trace is ever used
                k_min=1,
                k_max=k_max,
                guard=not single,  # single mode: never fall back to a K1=1 native step (would need a 2nd trace)
            )
            self._spec_log(f"serve INIT: traced={traced} k_max={k_max} single={single}")
            self._spec_tok = self.gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)
        # per-call context refresh (the request's block table grows as it advances)
        self._spec.kv_cache = kv_cache
        self._spec.page_table = page_table
        self._spec.page_tables_per_layer = page_tables_per_layer
        cur = int(torch.as_tensor(tokens).reshape(-1)[0])
        p0 = int(pos.reshape(-1)[0])
        # NEW-REQUEST detection by position CONTINUITY, not reset_batch. reset_batch fires every decode
        # step here (per-step full refresh, see laguna-batched-decode-corruption), so it cannot flag a new
        # request. Within a request the plugin advances pos by exactly 1 each call; a mismatch (or the very
        # first call) means a fresh request → reseed history from the stashed prompt + reset guard/adaptive.
        if self._spec_next_pos is None or p0 != self._spec_next_pos or not self._spec_hist:
            self._spec.serve_reset()
            self._spec_buf = []
            # Seed from the prompt stashed at prefill (greedy path gets no history via kwargs). History must
            # have len == p0+1 and end with cur (verify uses anchor_pos = len-1 as the absolute KV position).
            seed = list(self._spec_prefill_seq or [])
            if len(seed) >= p0:
                seed = seed[:p0]
            else:  # prompt stash short (unexpected) — front-pad so positions still line up
                seed = [cur] * (p0 - len(seed)) + seed
            seed.append(cur)
            self._spec_hist = seed
            self._spec_log(
                f"serve NEWREQ: seeded hist len={len(seed)} pos0={p0} cur={cur} "
                f"prompt_stash={len(self._spec_prefill_seq or [])}"
            )
        if not self._spec_buf:
            history = self._spec_hist  # SELF-TRACKED across the whole request (authoritative)
            # bounded-K: keep look-ahead writes (anchor_pos+1 .. anchor_pos+K) inside the anchor's own
            # block. anchor_pos = len-1, so the room left in the block is 63 - (anchor_pos % 64).
            k_cap = 63 - ((len(history) - 1) % 64)
            committed = list(self._spec.serve_round(history, k_cap=k_cap))
            self._spec_hist.extend(committed)  # grow history so the next round's anchor advances
            self._spec_buf = committed
            self._spec_log(
                f"serve ROUND: anchor_pos={p0} committed={len(committed)} toks={committed[:8]} "
                f"k_cur={getattr(self._spec,'_sv_k_cur',None)} spec_on={getattr(self._spec,'_sv_spec_on',None)}"
            )
        tok_id = int(self._spec_buf.pop(0))
        self._spec_next_pos = p0 + 1  # plugin appends the returned token → next decode is at p0+1
        ttnn.copy_host_to_device_tensor(
            self._host_rank4_tok_batch(torch.tensor([[tok_id]], dtype=torch.int64), 1), self._spec_tok
        )
        if read_from_device:
            return self._read_tokens_host(self._spec_tok, 1)
        return [self._spec_tok]

    def verify_forward_decode(
        self,
        tokens,
        positions,
        page_table=None,
        kv_cache=None,
        page_tables_per_layer=None,
        **kwargs,
    ):
        """Speculative-decode VERIFY via the batched-DECODE path (gemma4 `ttnn_verify_forward` pattern).

        The ``B = K+1`` candidate tokens ``[anchor, d0, …, d_{K-1}]`` occupy the BATCH dim at consecutive
        positions ``positions = [P-1, P, …, P-1+K]``, all pointing at the SAME user's KV blocks (the
        page-table row is replicated B times). One batched decode runs the fast paged-SDPA-**decode**
        (reads KV in O(1) w.r.t. context, unlike the prefill-path `verify_forward`), with
        ``sequential_kv_write=True`` so the B rows' shared-block cache writes serialize (no RMW race —
        see MultichipDecoder._seq_kv_write). Returns host logits ``[B, vocab]``; row j predicts the token
        at ``positions[j]+1``, so row j's argmax is the target-greedy token for that slot (greedy accept).

        Eager (untraced) — a first correctness/latency vehicle; the traced B=K+1 fast path is a follow-up.
        """
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)
        B = int(tokens.shape[0])
        pos = torch.as_tensor(positions, dtype=torch.int32).reshape(B)

        def _row_to_B(row):
            """One user's block row -> [B, w] device page table (replicated across the K+1 candidates)."""
            r = torch.as_tensor(row, dtype=torch.int32)
            if r.dim() == 1:
                r = r.unsqueeze(0)
            r = r[:1]  # the single served user's row
            if B > 1:
                r = r.repeat(B, 1)
            return self._page_table_to_device(r)

        if page_tables_per_layer is not None:
            # HYBRID serving: the 30 sliding layers read a different (smaller) KV pool than the 10 full
            # layers, so each layer needs ITS group's block row. decode_layers routes a per-layer list
            # (model.py: per_layer = isinstance(page_table,(list,tuple))). Build one device PT per built
            # layer, each = that layer's group row replicated to the B=K+1 candidate batch. Without this
            # the sliding layers would index the full pool -> silently wrong verify logits.
            pt = [_row_to_B(ptl_i) for ptl_i in page_tables_per_layer]
        else:
            pt = _row_to_B(page_table)  # uniform: one row shared by all layers
        tok_tt = self.gen._rep(tokens.reshape(1, B).to(torch.int32), ttnn.uint32)
        cur = self.gen._rep(pos, ttnn.int32)
        ridx = self.gen._rep(pos.reshape(1, B), ttnn.uint32)
        h = self.model.embed_decode(tok_tt)
        h = self.model.decode_layers(h, cur, ridx, pt, kv_cache, sequential_kv_write=True)
        shards = self.model.lm_head_shards_decode(h)
        logits = self.model.logits_to_host(shards).reshape(B, self.vocab)
        return logits

    def verify_greedy_decode_with_dflash_aux(
        self,
        tokens,
        positions,
        page_table=None,
        kv_cache=None,
        page_tables_per_layer=None,
        **kwargs,
    ):
        """Eager target verify plus exact auxiliary rows for one DFlash request."""

        if not self._DFLASH_SERVING_ENABLED:
            raise RuntimeError("DFlash target verify is default-off; set TT_LAGUNA_DFLASH=1")
        if page_tables_per_layer is not None:
            raise RuntimeError("DFlash target verify does not support hybrid per-layer page tables")
        if not kv_cache:
            raise ValueError("DFlash target verify requires an allocated KV cache")
        token_ids = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)
        B = int(token_ids.shape[0])
        if not 1 <= B <= 16:
            raise ValueError(f"DFlash target verify requires 1..16 rows, got {B}")
        if bool(((token_ids < 0) | (token_ids >= int(self.vocab))).any()):
            raise ValueError("DFlash target verify received a token outside the target vocabulary")
        pos = torch.as_tensor(positions, dtype=torch.int32).reshape(-1)
        if int(pos.numel()) != B:
            raise ValueError(f"DFlash target verify has {pos.numel()} positions for {B} tokens")
        expected = torch.arange(int(pos[0]), int(pos[0]) + B, dtype=torch.int32)
        if not torch.equal(pos.cpu(), expected):
            raise ValueError("DFlash target verify positions must be strictly contiguous")
        if int(pos[0]) < 0 or int(pos[-1]) >= int(self.max_model_len):
            raise ValueError(
                f"DFlash target verify interval [{int(pos[0])}, {int(pos[-1]) + 1}) "
                f"is outside max_model_len {self.max_model_len}"
            )

        pt_host = torch.as_tensor(page_table, dtype=torch.int32)
        if pt_host.dim() == 1:
            pt_host = pt_host.unsqueeze(0)
        if pt_host.dim() != 2 or int(pt_host.shape[0]) < 1:
            raise ValueError("DFlash target verify requires one uniform page-table row")
        pt_host = pt_host[:1].repeat(B, 1) if B > 1 else pt_host[:1]
        pt = self._page_table_to_device(pt_host)
        tok_tt = self.gen._rep(token_ids.reshape(1, B).to(torch.int32), ttnn.uint32)
        cur = self.gen._rep(pos, ttnn.int32)
        ridx = self.gen._rep(pos.reshape(1, B), ttnn.uint32)
        hidden = self.model.embed_decode(tok_tt)
        hidden, capture = self.model.decode_layers_with_dflash_aux(
            hidden,
            cur,
            ridx,
            pt,
            kv_cache,
            absolute_position=int(pos[0]),
            sequential_kv_write=True,
            enable_experimental=True,
        )
        shards = self.model.lm_head_shards_decode(hidden)
        logits = self.model.logits_to_host(shards).reshape(B, int(self.vocab))
        greedy = torch.argmax(logits, dim=-1).to(torch.int32).tolist()
        return greedy, capture

    def _alloc_verify_decode(self, K1, kv_cache, tokens, pos, pt_host):
        """Phase 1 of verify-trace warmup: allocate ALL persistent device buffers for one K1 and warm
        the program cache (compile), WITHOUT capturing a trace. Multi-K adaptive warmup MUST allocate
        every K1's buffers before ANY begin_trace_capture: allocating a device buffer while a captured
        trace is resident corrupts the trace (TTNN: 'Allocating device buffers is unsafe due to the
        existence of an active trace') and the subsequent replay hangs the mesh. Returns a state dict
        carrying the closured ``step`` for the later capture phase (tid filled in by _trace_verify_decode).

        Fully on-device — the greedy token per row is produced by the same Sampling1D (top-k=1) the
        model's serving decode uses, so the verify matches what the hardware actually greedy-decodes."""
        g = self.gen
        # LOGITS mode (default): the trace ends at the mesh-sharded logits; the greedy token is argmax'd ON
        # HOST after replay (verify_forward_decode's known-correct path). This AVOIDS the on-device Sampling1D,
        # which under trace at K1>=2 deterministically miscomputes some rows (float-garbage or valid-but-wrong
        # ids in the tok buffer -> corrupts the always-committed anchor -> garbage output). The forward
        # (decode_layers + lm_head) is still traced, so the dispatch-elimination speedup is preserved; the only
        # added cost is reading K1 x vocab logits to host + a host argmax per step. Set TT_LAGUNA_SPEC_LOGITS=0
        # to use the (buggy) on-device sampler path instead.
        # ROOT CAUSE of the traced-verify garbage: the on-device greedy argmax (inside Sampling1D) is the
        # multicore ttnn.argmax, which is ROW-PARALLEL and returns GARBAGE unless the batch (row) dim is
        # tile-aligned to 32 (gemma4 spec_decode._argmax_last: "1/5 rows -> wrong; padded to 32 -> exact").
        # The verify batch is K1=2..5 rows -> unaligned -> some rows (incl. the always-committed anchor) get
        # float-bit garbage / wrong ids -> cascades to garbage output. FIX (on-device, matches gemma4): run
        # the FORWARD at K1 rows (padding it would write KV at 32 positions -> OOB) but PAD THE LOGITS to 32
        # rows just before the argmax, then slice back to K1. Nearly free (tiled matmul already pays for a
        # 32-row tile). TT_LAGUNA_SPEC_LOGITS=1 keeps the host-argmax fallback (correct, but transfers logits).
        logits_mode = os.environ.get("TT_LAGUNA_SPEC_LOGITS", "0") == "1"
        R32 = 32
        tok = g._rep(torch.zeros([1, 1, 1, K1], dtype=torch.int32), ttnn.uint32)
        cur = g._rep(torch.zeros([K1], dtype=torch.int32), ttnn.int32)
        ridx = g._rep(torch.zeros([1, K1], dtype=torch.int32), ttnn.uint32)
        pt = self._page_table_to_device(pt_host)
        ttnn.copy_host_to_device_tensor(self._host_rank4_tok_batch(tokens.reshape(K1, 1), K1), tok)
        ttnn.copy_host_to_device_tensor(self._host_pos_batch(pos), cur)
        ttnn.copy_host_to_device_tensor(self._host_ridx_batch(pos), ridx)
        st = dict(tid=None, tok=tok, cur=cur, ridx=ridx, pt=pt, logits_mode=logits_mode, k1=K1)
        if not logits_mode:
            # FORCE-ARGMAX path (Sampling1D._sample_argmax = all-gather vocab + ttnn.argmax). Passing k/p/temp
            # all None selects it (allow_force_argmax=True); passing k=1/p=1/temp=1 instead selects the top-k
            # path (per-shard top-1 + gather) which returned WRONG rows in the batched verify. Sampler + output
            # run at 32 tile-aligned rows; the forward stays K1 (padding it would OOB the KV writes).
            sampler = g._sampler(R32)
            tok_out = g._rep(torch.zeros([1, 1, 1, R32], dtype=torch.int32), ttnn.uint32)  # 32-row argmax output
            st["tok_out"] = tok_out

        def step():
            hh = self.model.embed_decode(ttnn.reshape(tok, (1, K1)))
            hh = self.model.decode_layers(hh, cur, ridx, pt, kv_cache, sequential_kv_write=True)
            shards = self.model.lm_head_shards_decode(hh)  # [1,1,K1,V/D]
            if logits_mode:
                st["logits"] = shards  # persistent trace-output handle; ConcatMesh+argmax on host post-replay
            else:
                # pad logits rows K1->32 (tile-align) so the multicore argmax is row-correct, then force-argmax
                s32 = ttnn.pad(shards, [(0, 0), (0, 0), (0, R32 - K1), (0, 0)], value=0.0) if K1 < R32 else shards
                sampler.decode_forward(s32, tt_out_tok=tok_out)  # k/p/temp None -> force-argmax

        step()  # compile (warm program cache) — no trace resident yet, so allocations here are safe
        ttnn.synchronize_device(self.mesh_device)
        st["_step"] = step
        return st

    def _trace_verify_decode(self, K1, st):
        """Phase 2 of verify-trace warmup: capture the trace using the buffers _alloc_verify_decode
        already allocated. NO new persistent allocation happens here (step() only re-runs compiled
        programs into existing buffers), so this is safe to call repeatedly while earlier K1 traces are
        already resident."""
        ttnn.synchronize_device(self.mesh_device)
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        st["_step"]()  # capture
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        st["tid"] = tid
        st.pop("_step", None)
        self._verify_dec[K1] = st
        return st

    def _capture_verify_decode(self, K1, kv_cache, tokens, pos, pt_host):
        """Single-K capture (lazy fallback path): allocate then immediately capture. Safe only when no
        other verify trace is being captured in the same window — for multi-K adaptive warmup use
        warmup_verify_decode_multi, which allocates all buffers before capturing any trace."""
        st = self._alloc_verify_decode(K1, kv_cache, tokens, pos, pt_host)
        return self._trace_verify_decode(K1, st)

    def verify_greedy_decode(
        self, tokens, positions, page_table=None, kv_cache=None, page_tables_per_layer=None, traced=True
    ):
        """Greedy batched-DECODE verify → per-row target-greedy token ids ``[K+1]`` (torch int32).

        The fast path for spec-decode: K+1 candidates in the batch dim at consecutive positions run
        through ONE batched decode (fast paged-decode SDPA, O(1) in context) with race-safe
        sequential_kv_write and the on-device greedy sampler (top-k=1), so row j's id IS argmax of its
        logits (= g[j], the target-greedy token for slot j) — no host logit transfer. ``traced=True``
        replays a captured trace (host dispatch eliminated); the page-table row is constant across
        iterations (same user blocks), so only tokens+positions refresh per replay."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)
        K1 = int(tokens.shape[0])
        pos = torch.as_tensor(positions, dtype=torch.int32).reshape(K1)
        if not traced:
            # Eager verify: forward BOTH the uniform page_table and the hybrid page_tables_per_layer;
            # verify_forward_decode replicates the user's row to the K1 candidate batch and routes the
            # per-layer list to decode_layers (sliding layers read their own pool). page_table may be
            # None in pure-hybrid serving, so hand it through unchanged.
            logits = self.verify_forward_decode(
                tokens,
                pos,
                page_table=page_table,
                kv_cache=kv_cache,
                page_tables_per_layer=page_tables_per_layer,
            )
            return torch.argmax(logits, dim=-1).to(torch.int32)
        # HARD GUARD (audit item 4): traced spec-verify has NO hybrid grouped-PT path. The traced page-table
        # refresh below only fires when `page_tables_per_layer is None`; a hybrid per-layer PT would replay the
        # FROZEN warmup identity table and silently emit wrong greedy ids. Hybrid KV is dead at serving today
        # (the plugin never calls get_kv_cache_spec), so this is dormant — but enabling hybrid KV (the wanted
        # capacity win) MUST fail loudly here, not corrupt. Fix: eager verify (traced=False), or extend the
        # trace refresh to grouped PTs before combining hybrid KV with traced spec-decode.
        if page_tables_per_layer is not None:
            raise NotImplementedError(
                "traced spec-decode verify does not support hybrid per-layer page tables "
                "(page_tables_per_layer): the traced page-table refresh is uniform-only, so a hybrid PT would "
                "replay a stale identity table and produce silently-wrong tokens. Run eager verify "
                "(traced=False), or add a grouped-PT trace-refresh path before enabling hybrid KV + spec-decode."
            )
        pt_row = torch.as_tensor(page_table, dtype=torch.int32)
        if pt_row.dim() == 1:
            pt_row = pt_row.unsqueeze(0)
        pt_row = pt_row[:1]  # the single user's block row (replicated below to the batch size used)
        pt_host = pt_row.repeat(K1, 1) if K1 > 1 else pt_row
        st = self._verify_dec.get(K1)
        if st is None:
            # verify-decode trace for this K1 not pre-captured -> compile+capture in-request.
            print(
                f"[laguna] WARNING: lazy spec-decode VERIFY-trace capture for K1={K1} (not pre-warmed) — "
                f"this step includes compile+capture, not a warm replay.",
                flush=True,
            )
            st = self._capture_verify_decode(K1, kv_cache, tokens, pos, pt_host)  # lazy fallback
            if st.get("logits_mode"):
                return torch.argmax(self.model.logits_to_host(st["logits"]).reshape(K1, int(self.vocab)), dim=-1).to(
                    torch.int32
                )
            return ttnn.to_torch(ttnn.get_device_tensors(st["tok_out"])[0]).reshape(-1)[:K1].to(torch.int32)
        ttnn.copy_host_to_device_tensor(self._host_rank4_tok_batch(tokens.reshape(K1, 1), K1), st["tok"])
        ttnn.copy_host_to_device_tensor(self._host_pos_batch(pos), st["cur"])
        ttnn.copy_host_to_device_tensor(self._host_ridx_batch(pos), st["ridx"])
        # Refresh the page table into the persistent trace buffer that the replay reads. Without this,
        # st["pt"] stays frozen at the warmup identity table (arange) and the verify indexes the WRONG
        # physical KV blocks on any real (non-identity) served page table -> silently wrong greedy ids.
        # (Uniform serving path; hybrid grouped-PT verify is a separate follow-up — serving is uniform today.)
        if page_tables_per_layer is None and st.get("pt") is not None:
            if st.get("last_pt_host") is None or not torch.equal(pt_host, st["last_pt_host"]):
                ttnn.copy_host_to_device_tensor(self._page_table_to_device_host(pt_host), st["pt"])
                st["last_pt_host"] = pt_host.clone()
        ttnn.execute_trace(self.mesh_device, st["tid"], cq_id=0, blocking=True)
        if st.get("logits_mode"):
            # host argmax of the traced logit shards (same gather as eager verify_forward_decode) — bypasses
            # the buggy on-device sampler. traced_ids is bit-correct iff the traced FORWARD matches eager.
            logits = self.model.logits_to_host(st["logits"]).reshape(K1, int(self.vocab))
            traced_ids = torch.argmax(logits, dim=-1).to(torch.int32)
        else:
            # ON-DEVICE argmax (padded to 32 rows) — read the device-0 replica (gemma4 _ids_to_host pattern),
            # then slice off the row-padding back to K1.
            th = ttnn.to_torch(ttnn.get_device_tensors(st["tok_out"])[0])
            traced_ids = th.reshape(-1)[:K1].to(torch.int32)
        # CORRECTNESS GUARD: the traced on-device Sampling1D DETERMINISTICALLY fails to write some rows for
        # certain logit distributions, leaving stale FLOAT bit-patterns in the uint32 tok buffer (e.g.
        # 1096876032=0x41600000=14.0f, or a negative) — an out-of-range "token". When that lands on row 0
        # (the anchor, always committed) it corrupts output. Detect any out-of-vocab id and recompute the
        # WHOLE round via the eager host-argmax verify (verify_forward_decode — known bit-correct). Cheap:
        # only fires on the rare bad round. (Root cause is the sampler kernel under trace; this makes the
        # served path correct without it.)
        ids_t = torch.as_tensor(traced_ids).reshape(-1).to(torch.int64)
        if bool(((ids_t < 0) | (ids_t >= int(self.vocab))).any()):
            elog = self.verify_forward_decode(
                tokens, pos, page_table=page_table, kv_cache=kv_cache, page_tables_per_layer=page_tables_per_layer
            )
            eager_ids = torch.argmax(elog, dim=-1).to(torch.int32)
            self._spec_log(
                f"GUARD: out-of-vocab traced id at K1={K1} pos0={int(pos[0])} "
                f"traced={[int(x) for x in ids_t]} -> eager fallback={[int(x) for x in eager_ids]}"
            )
            traced_ids = eager_ids
        if os.environ.get("TT_LAGUNA_SPEC_DEBUG", "") == "1":
            # Shadow the traced verify with an EAGER verify (known-correct host-argmax path) on the SAME
            # tokens/positions/pt/kv. Both read the CURRENT (post-trace) KV, so agreement means the trace's
            # per-step COMPUTE matches eager (any full-run divergence is then accumulating KV-write drift);
            # disagreement means the trace's own read/compute is wrong THIS step.
            try:
                elog = self.verify_forward_decode(
                    tokens, pos, page_table=page_table, kv_cache=kv_cache, page_tables_per_layer=page_tables_per_layer
                )
                eager_ids = torch.argmax(elog, dim=-1).to(torch.int32).reshape(-1)
                t_ids = torch.as_tensor(traced_ids).reshape(-1)
                if not torch.equal(t_ids.to(torch.int64), eager_ids.to(torch.int64)):
                    self._spec_log(
                        f"TRACE-vs-EAGER MISMATCH K1={K1} pos={[int(x) for x in pos]} "
                        f"traced={[int(x) for x in t_ids]} eager={[int(x) for x in eager_ids]} "
                        f"pt0={[int(x) for x in pt_host[0][:4]]}"
                    )
                else:
                    self._spec_log(f"trace==eager K1={K1} pos0={int(pos[0])} pt0={[int(x) for x in pt_host[0][:4]]}")
            except Exception as e:  # noqa: BLE001
                self._spec_log(f"shadow-eager failed: {type(e).__name__}: {e}")
        return traced_ids

    def verify_sampler_eager(self, tokens, positions, page_table=None, kv_cache=None):
        """DIAGNOSTIC: run the batched decode-verify through the on-device SAMPLER but EAGERLY (no trace
        capture/replay). Isolates the sampler from the trace: the host-argmax eager path
        (verify_forward_decode) is known-correct, so if this sampler-eager path also matches it, the
        sampler is fine and the traced divergence is purely a trace-replay defect; if it diverges here,
        the on-device sampler is the culprit (model-layer fixable). Returns per-row greedy ids [B]."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1)
        B = int(tokens.shape[0])
        pos = torch.as_tensor(positions, dtype=torch.int32).reshape(B)
        pt_host = torch.as_tensor(page_table, dtype=torch.int32)
        if pt_host.dim() == 1:
            pt_host = pt_host.unsqueeze(0)
        if pt_host.shape[0] == 1 and B > 1:
            pt_host = pt_host.repeat(B, 1)
        g = self.gen
        tok = g._rep(torch.zeros([1, 1, 1, B], dtype=torch.int32), ttnn.uint32)
        cur = g._rep(pos, ttnn.int32)
        ridx = g._rep(pos.reshape(1, B), ttnn.uint32)
        k = g._rep(torch.ones([B], dtype=torch.int32), ttnn.uint32)
        p = g._rep(torch.ones([B], dtype=torch.float32), ttnn.bfloat16)
        t = g._rep(torch.ones([B], dtype=torch.float32), ttnn.bfloat16)
        seeds = g._rep(torch.zeros([B], dtype=torch.int32), ttnn.uint32)
        sampler = g._sampler(B)
        pt = self._page_table_to_device(pt_host)
        ttnn.copy_host_to_device_tensor(self._host_rank4_tok_batch(tokens.reshape(B, 1), B), tok)
        h = self.model.embed_decode(ttnn.reshape(tok, (1, B)))
        h = self.model.decode_layers(h, cur, ridx, pt, kv_cache, sequential_kv_write=True)
        shards = self.model.lm_head_shards_decode(h)
        sampler.decode_forward(shards, k=k, p=p, temp=t, seeds=seeds, tt_out_tok=tok)
        return self._read_tokens_host(tok, B)

    def warmup_verify_decode(self, draft_len, kv_cache, num_blocks, block_size=64):
        """Capture the spec-decode VERIFY trace in the SAFE warmup window (mirrors how the normal decode
        trace is captured by warmup_model_decode). Capturing lazily mid-serving hangs the mesh.

        Capture at a populated cache + block-internal positions to match serving structure; positions/tokens
        refresh per replay."""
        K1 = int(draft_len) + 1
        if kv_cache is None or K1 in self._verify_dec:
            return
        base = 2 * block_size
        dummy = torch.zeros(base, dtype=torch.int64)
        ptp = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        self.prefill_forward(
            dummy.reshape(1, base),
            page_table=ptp,
            kv_cache=kv_cache,
            prompt_lens=[base],
            start_pos=[0],
            sampling_params=None,
        )
        pos = torch.arange(base - 1, base - 1 + K1, dtype=torch.int32)
        tokens = torch.zeros(K1, dtype=torch.int64)
        pt_host = ptp.repeat(K1, 1)
        self._capture_verify_decode(K1, kv_cache, tokens, pos, pt_host)

    def warmup_verify_decode_multi(self, draft_lens, kv_cache, num_blocks, block_size=64):
        """Adaptive-K verify warmup: pre-capture a verify trace for EACH draft_len in one safe window.

        The naive loop `for k: warmup_verify_decode(k)` captures trace K1=2, then allocates K1=3's
        buffers while K1=2's trace is resident — TTNN flags 'Allocating device buffers is unsafe due to
        the existence of an active trace' and the first replay hangs the mesh. This stages ALL buffer
        allocation (phase 1) before ANY trace capture (phase 2), so no allocation ever races a resident
        trace."""
        K1s = sorted({int(d) + 1 for d in draft_lens if int(d) + 1 not in self._verify_dec})
        if kv_cache is None or not K1s:
            return
        base = 2 * block_size
        dummy = torch.zeros(base, dtype=torch.int64)
        ptp = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        self.prefill_forward(
            dummy.reshape(1, base),
            page_table=ptp,
            kv_cache=kv_cache,
            prompt_lens=[base],
            start_pos=[0],
            sampling_params=None,
        )
        staged = {}
        for K1 in K1s:  # phase 1: allocate + compile every K1 (no trace resident)
            pos = torch.arange(base - 1, base - 1 + K1, dtype=torch.int32)
            tokens = torch.zeros(K1, dtype=torch.int64)
            pt_host = ptp.repeat(K1, 1)
            staged[K1] = self._alloc_verify_decode(K1, kv_cache, tokens, pos, pt_host)
        for K1 in K1s:  # phase 2: capture every trace (buffers already allocated)
            self._trace_verify_decode(K1, staged[K1])

    # --------------------------------------------------------------------- #
    # Decode (traced split sampling + async split)
    # --------------------------------------------------------------------- #
    def _decode_state(self, B, kv_cache, pt_persist):
        """Capture (once per batch B) the decode trace over persistent device buffers:
        embed(tok) → 40-layer stack → norm → LM head → Sampling1D(k/p/temp/seed) → tt_out_tok, then
        plus_one(cur/ridx) on device. Nothing is rebuilt on host between replays except the page
        table (only when it changes) and positions/token (only on a batch-layout reset)."""
        st = self._decode.get(B)
        if st is not None:
            return st
        tok = self.gen._rep(torch.zeros([1, 1, 1, B], dtype=torch.int32), ttnn.uint32)
        cur = self.gen._rep(torch.zeros([B], dtype=torch.int32), ttnn.int32)
        ridx = self.gen._rep(torch.zeros([1, B], dtype=torch.int32), ttnn.uint32)
        k = self.gen._rep(torch.ones([B], dtype=torch.int32), ttnn.uint32)
        p = self.gen._rep(torch.ones([B], dtype=torch.float32), ttnn.bfloat16)
        t = self.gen._rep(torch.ones([B], dtype=torch.float32), ttnn.bfloat16)
        seeds = self.gen._rep(torch.zeros([B], dtype=torch.int32), ttnn.uint32)
        sampler = self.gen._sampler(B)

        def step():
            h = self.model.embed_decode(ttnn.reshape(tok, (1, B)))
            h = self.model.decode_layers(h, cur, ridx, pt_persist, kv_cache)
            shards = self.model.lm_head_shards_decode(h)
            sampler.decode_forward(shards, k=k, p=p, temp=t, seeds=seeds, tt_out_tok=tok)
            ttnn.plus_one(cur, skip_negative_entries=True)
            ttnn.plus_one(ridx)

        step()  # compile
        ttnn.synchronize_device(self.mesh_device)
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        step()  # capture
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        st = dict(
            tid=tid,
            tok=tok,
            cur=cur,
            ridx=ridx,
            k=k,
            p=p,
            t=t,
            seeds=seeds,
            pt=pt_persist,
            staged=False,
            last_pt_host=None,
            last_sp_key=None,
        )
        self._decode[B] = st
        return st

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
        page_tables_per_layer=None,
        **kwargs,
    ):
        """One decode step for the whole padded batch.

        Device sampling (``sampling_params`` given): traced split-sampling. Token/position refresh is
        done from host ONLY on a batch-layout change (``reset_batch``) or the first step after a
        (re)capture; otherwise the previous step's on-device sampled token (in ``tok``) and the
        device-advanced ``cur``/``ridx`` are reused (no host token/position work). The page table is
        copied only when its contents changed. Returns a per-DP list of device token tensors when
        ``read_from_device=False``, else host tokens.

        Host sampling (``sampling_params is None``, compat mode for min_p/logprobs/etc.): eager decode
        returning logits; never used for the measured perf path."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1, 1)
        B = tokens.shape[0]
        pos = torch.as_tensor(start_pos, dtype=torch.int32).reshape(B)
        if not kv_cache:
            raise ValueError("decode requires an allocated KV cache")
        hybrid_cache = self._validate_page_table_mode(
            kv_cache,
            page_tables_per_layer,
            operation="decode",
        )

        if self._DFLASH_SERVING_ENABLED:
            return self._dflash_serve(
                tokens,
                pos,
                page_table,
                kv_cache,
                page_tables_per_layer,
                sampling_params,
                read_from_device,
            )

        if self._spec_mode and not self._spec_probed:
            # One-time diagnostic + Phase-2 feasibility probe. Placed BEFORE the host-sampling return and
            # gated only on "once" (not B==1: served decode is padded to max_batch_size, so B is never 1).
            self._spec_probed = True
            self._spec_log(
                f"decode_forward#1 pid={os.getpid()} B={B} reset_batch={reset_batch} "
                f"host_sampling={sampling_params is None} hybrid={page_tables_per_layer is not None} "
                f"spec_mode={self._spec_mode!r}"
            )
            if self._spec_mode == "probe":
                # Run ONE eager verify under the resident decode trace on row 0; fall through to normal
                # decode (throwaway KV at pos+1 — probe boot only).
                self._spec_feasibility_probe(tokens, pos, page_table, kv_cache, page_tables_per_layer)

        # FULL served spec-decode: in this mode the normal decode trace was OMITTED at warmup, so ALL decode
        # goes through the traced verify path. Requires --max-num-seqs 1 (B==1, no padding) + greedy.
        if self._spec_mode == "1" and B == 1 and sampling_params is not None and self._spec_is_greedy(sampling_params):
            return self._spec_serve(
                tokens, pos, page_table, kv_cache, page_tables_per_layer, reset_batch, kwargs, read_from_device
            )

        if sampling_params is None:
            if hybrid_cache:
                raise RuntimeError("hybrid KV host-sampling decode is unsupported; use on-device sampling")
            return self._decode_host_sampling(tokens, pos, page_table, kv_cache, read_from_device)

        hybrid = hybrid_cache
        st = self._decode.get(B)
        if st is None:
            # the decode trace for this batch B was NOT pre-captured by warmup, so we compile +
            # capture it now — INSIDE a live request. This is orders of magnitude slower than a warm
            # replay and is invisible in the served latency unless flagged. Warn so a lazy capture is not
            # silently mistaken for warm serving (drive warmup_model_decode for every served B to avoid).
            print(
                f"[laguna] WARNING: lazy decode-trace capture for batch B={B} inside decode_forward "
                f"(warmup did not pre-capture this B) — first-token latency for this B includes "
                f"compile+capture, not a warm replay.",
                flush=True,
            )
            if hybrid:
                pt_persist, groups, reps = self._decode_pt_grouped_alloc(page_tables_per_layer)
                st = self._decode_state(B, kv_cache, pt_persist)
                st["pt_groups"], st["pt_reps"], st["last_pt_host_groups"] = groups, reps, {}
            else:
                pt_persist = self._page_table_to_device(page_table)
                st = self._decode_state(B, kv_cache, pt_persist)
        tok, cur, ridx, tid, pt = st["tok"], st["cur"], st["ridx"], st["tid"], st["pt"]

        # --- sampling params: refresh persistent buffers only when they change ---
        k_h, p_h, t_h, s_h = self._sampling_buffers_from_params(sampling_params, B)
        sp_key = (tuple(k_h.tolist()), tuple(p_h.tolist()), tuple(t_h.tolist()), tuple(s_h.tolist()))
        if sp_key != st["last_sp_key"]:
            ttnn.copy_host_to_device_tensor(self.gen._host(k_h, ttnn.uint32), st["k"])
            ttnn.copy_host_to_device_tensor(self.gen._host(p_h.to(torch.float32), ttnn.bfloat16), st["p"])
            ttnn.copy_host_to_device_tensor(self.gen._host(t_h.to(torch.float32), ttnn.bfloat16), st["t"])
            ttnn.copy_host_to_device_tensor(self.gen._host(s_h, ttnn.uint32), st["seeds"])
            st["last_sp_key"] = sp_key

        # --- token/position refresh: only on reset or first step (else device feedback) ---
        if reset_batch or not st["staged"]:
            ttnn.copy_host_to_device_tensor(self._host_rank4_tok_batch(tokens, B), tok)
            ttnn.copy_host_to_device_tensor(self._host_pos_batch(pos), cur)
            ttnn.copy_host_to_device_tensor(self._host_ridx_batch(pos), ridx)
            st["staged"] = True
            self.gen.counters["token_refresh"] += 1
            self.gen.counters["pos_refresh"] += 1

        # --- page table: copy only when contents changed ---
        if hybrid:
            # Refresh all four per-group persistent buffers that the trace closed over.
            self._decode_pt_grouped_refresh(st, page_tables_per_layer)
        else:
            pt_host = torch.as_tensor(page_table, dtype=torch.int32)
            if st["last_pt_host"] is None or not torch.equal(pt_host, st["last_pt_host"]):
                ttnn.copy_host_to_device_tensor(self._page_table_to_device_host(pt_host), pt)
                st["last_pt_host"] = pt_host.clone()
                self.gen.counters["page_table_refresh"] += 1

        ttnn.execute_trace(self.mesh_device, tid, cq_id=0, blocking=read_from_device)
        self.gen.counters["trace_replay"] += 1

        if read_from_device:
            host = self._read_tokens_host(tok, B)
            return host
        return [tok]  # device token buffer, per-DP list; read via read_decode_output/process...

    # ---- async split ---- #
    def read_decode_output(self, tt_out, async_read=False):
        """Non-blocking readback of the on-device sampled tokens. ``tt_out`` is the per-DP list
        returned by ``decode_forward(read_from_device=False)``."""
        if not async_read:
            return [t.cpu() for t in tt_out]
        host_outputs = [t.cpu(blocking=False) for t in tt_out]
        read_events = [ttnn.record_event(self.mesh_device, 0) for _ in tt_out]
        return host_outputs, read_events

    def process_decode_output_host(self, tt_out, is_tokens=False):
        """Convert the (host) ttnn tensors to torch. ``is_tokens`` True → sampled token ids [B];
        False → logits [B, vocab]. DP=1, so the single entry is returned directly."""
        out = tt_out[0] if isinstance(tt_out, list) else tt_out
        if isinstance(out, tuple):  # (tokens/logits, logprobs)
            out = out[0]
        if is_tokens:
            th = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
            B = th.shape[-1] if th.dim() >= 1 else 1
            return th.reshape(-1)[:B].to(torch.int32)
        # logits: gather vocab shards → [B, 1, vocab] (rank-3: the plugin's host-sampling path indexes
        # `tt_out[rows, -1, :]`, so decode logits must carry the seq axis, exactly like the prefill
        # host path's [num_reqs, 1, vocab]). Returning a rank-2 [B, vocab] triggers
        # `IndexError: too many indices for tensor of dimension 2` in model_runner._get_output_tokens.
        th = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        return th.reshape(-1, 1, self.vocab)

    # ---- host-sampling (compat) decode ---- #
    def _decode_host_sampling(self, tokens, pos, page_table, kv_cache, read_from_device):
        B = tokens.shape[0]
        pt = self._page_table_to_device(page_table)
        tok_tt = self.gen._rep(tokens.reshape(1, B).to(torch.int32), ttnn.uint32)
        cur = self.gen._rep(pos, ttnn.int32)
        ridx = self.gen._rep(pos.reshape(1, B), ttnn.uint32)
        h = self.model.embed_decode(tok_tt)
        h = self.model.decode_layers(h, cur, ridx, pt, kv_cache)
        shards = self.model.lm_head_shards_decode(h)
        if read_from_device:
            logits = self.model.logits_to_host(shards).reshape(B, self.vocab)
            return logits
        return [shards]

    def _read_tokens_host(self, tok_buf, B):
        th = ttnn.to_torch(tok_buf, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        return th.reshape(-1)[:B].to(torch.int32)

    # ---- host-tensor builders for persistent-buffer refresh ---- #
    def _host_rank4_tok_batch(self, tokens, B):
        return self.gen._host(tokens.reshape(1, 1, 1, B).to(torch.int32), ttnn.uint32)

    def _host_pos_batch(self, pos):
        return self.gen._host(pos.reshape(-1).to(torch.int32), ttnn.int32)

    def _host_ridx_batch(self, pos):
        return self.gen._host(pos.reshape(1, -1).to(torch.int32), ttnn.uint32)

    def _page_table_to_device_host(self, pt_host):
        return self.gen._host(pt_host.to(torch.int32), ttnn.int32)

    # --------------------------------------------------------------------- #
    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, can_sample_on_device=False, **kwargs):
        """Compile every supported prefill bucket length BEFORE the decode trace is captured.

        The plugin calls this in both its compile and trace phases, but Laguna does not capture a
        prefill trace: prefill must only have its programs compiled before the resident DECODE trace.
        The first call therefore compiles every bucket and allocates persistent buffers; the private
        ``_prefill_programs_warmed`` latch suppresses the plugin's redundant second full-model pass.
        This remains safe because the first call derives the exact serving page-table width directly
        from ``max_model_len`` and the logical KV pool, rather than waiting for decode warmup to set
        ``_max_blocks``. Any first-time program compilation / allocation during serving would corrupt
        the active decode trace, so every bucket and serving PT shape must be covered here."""
        if kv_cache is None:
            return None
        if getattr(self, "_prefill_programs_warmed", False):
            # The plugin reset its public compatibility flag between phases; restore the truthful state.
            self.already_warmed_up_prefill = True
            return None
        if self.already_warmed_up_prefill:
            return None
        self.already_warmed_up_prefill = True
        self._in_prefill_warmup = True  # suppress the _prefill_pt diagnostic for intentional warmup allocs
        bs = int(kv_cache[0]["block_size"])
        if bool(self._PREFIX_CACHE_ENABLED):
            self._prefix_resume_quantum(bs)
        # Allocate persistent sampling buffers, per-bucket-L one-hot selector buffers, runtime
        # position/start inputs, and hoisted indexed-RoPE outputs for the exact serving chunk geometry,
        # all pre-trace. The selector MATMUL program (sel[1,1,1,L] @ h[1,1,L,H]) + the [1,L,H]->[1,1,L,H]
        # reshape are compiled for every bucket L by the per-L ``prefill_forward`` calls below, which run
        # ``_last_token_shards`` (and thus the selector matmul) for each L — so no selector program
        # first-compiles under the resident decode trace at serving time.
        self._prefill_state(bs)
        total_blocks = int(kv_cache[0]["blocks_per_user"])
        if self._max_blocks is None:
            # Mirrors TTModelRunner.max_num_blocks_per_req exactly for the supported uniform-KV path:
            # min(cdiv(max_model_len, cache block size), kv_cache_config.num_blocks).
            self._max_blocks = min((int(self.max_model_len) + bs - 1) // bs, total_blocks)
        greedy = None
        if can_sample_on_device:
            from types import SimpleNamespace

            greedy = SimpleNamespace(temperature=[0.0], top_k=[0], top_p=[1.0], seed=[None])
        # Warm at the fixed PREFILL-only horizon, NOT a bucket-tight arange. D2's
        # streamed tail needs one outer-chunk rounding; D1's rollback path retains
        # max_model_len + largest_bucket. Attention uses valid scratch ids while
        # paged-fill uses -1 skip entries. Decode retains ``_max_blocks``.
        # The chunked prefill path (seq > PIPE_CHUNK) is keyed on this PT width. Warming at a narrower
        # width leaves those programs to RECOMPILE under the resident decode
        # trace on the first real >PIPE_CHUNK prefill — a ~200x slowdown (measured: chunked 4096 is
        # 3.2s standalone but 11+ min at serving until the wide-page-table programs recompile). Warming
        # at the serving width makes serving-time prefill compile-free. Single-shot buckets use the
        # table whole (width-agnostic) so this is a no-op for them.
        prefill_w = self._prefill_page_table_width(bs)
        hybrid = self._kv_cache_is_hybrid(kv_cache)
        for L in self._prefill_bucket_lens():
            nb = (L + bs - 1) // bs
            if nb > total_blocks:  # cache too small for this bucket (reduced bring-up); skip
                continue
            w = prefill_w
            dummy = torch.zeros((1, L), dtype=torch.int64)
            if hybrid:
                # Per-layer warmup page tables bounded to EACH layer's pool: block ids beyond a
                # (small) sliding pool would be OOB, so clamp each layer's real-block count to its
                # pool. Content is dummy (garbage prefill); only shape + in-bounds indices matter for
                # compiling the per-group programs and pre-allocating the per-group persistent buffers.
                ptl = []
                for kv in kv_cache:
                    pool = int(kv["blocks_per_user"])
                    m = min(nb, pool)
                    t = torch.zeros((1, w), dtype=torch.int32)
                    t[0, :m] = torch.arange(m, dtype=torch.int32)
                    ptl.append(t)
                self.prefill_forward(
                    dummy,
                    page_tables_per_layer=ptl,
                    kv_cache=kv_cache,
                    prompt_lens=[L],
                    start_pos=[0],
                    sampling_params=greedy,
                )
            else:
                pt = torch.zeros((1, w), dtype=torch.int32)
                pt[0, :nb] = torch.arange(nb, dtype=torch.int32)
                self.prefill_forward(
                    dummy, page_table=pt, kv_cache=kv_cache, prompt_lens=[L], start_pos=[0], sampling_params=greedy
                )
        # Streaming executes every subchunk through the full decoder stack. Warm
        # one canonical 8192-query tail at a nonzero absolute start before the
        # decode trace is resident; start is runtime data thereafter, so later
        # chunk ordinals reuse the same program. The cold ladder above covers
        # truly short requests. The rollback path retains the former cache-off
        # start=block-size probes.
        if self._streaming_prefill_active():
            start_gt_zero_cases = self._prefill_stream_warm_cases(bs)
            warm_ranges = [(0, end, bucket) for bucket, end in start_gt_zero_cases]
        else:
            single_shot = (
                [L for L in self._prefill_bucket_lens() if L <= int(self.model.layers[0].PIPE_CHUNK)]
                if int(self.D) == 2 and not bool(self._PREFIX_CACHE_ENABLED)
                else []
            )
            warm_ranges = [(bs, bs + int(L), int(L)) for L in single_shot]

        for start, end, _tail_bucket in warm_ranges:
            needed = (end + bs - 1) // bs
            if needed > total_blocks:
                continue
            dummy = torch.zeros((1, end), dtype=torch.int64)
            if hybrid:
                if any(int(kv["blocks_per_user"]) < needed for kv in kv_cache):
                    continue
                ptl = []
                for _kv in kv_cache:
                    t = torch.zeros((1, prefill_w), dtype=torch.int32)
                    t[0, :needed] = torch.arange(needed, dtype=torch.int32)
                    ptl.append(t)
                self.prefill_forward(
                    dummy,
                    page_tables_per_layer=ptl,
                    kv_cache=kv_cache,
                    prompt_lens=[end],
                    start_pos=[start],
                    sampling_params=greedy,
                )
            else:
                pt = torch.zeros((1, prefill_w), dtype=torch.int32)
                pt[0, :needed] = torch.arange(needed, dtype=torch.int32)
                self.prefill_forward(
                    dummy,
                    page_table=pt,
                    kv_cache=kv_cache,
                    prompt_lens=[end],
                    start_pos=[start],
                    sampling_params=greedy,
                )
        # W1 fix — warm the serving ROW-COUNT dimension of the prefill page table. Serving batches up to
        # ``max_num_seqs`` new requests into ONE prefill call (page_table ``[num_reqs, prefill_w]``), and
        # Both attention/fill persistent helpers are SHAPE-KEYED — an unseen
        # ``(num_reqs>1, prefill_w)`` shape would allocate a buffer under the resident decode trace, i.e. the
        # allocator.cpp:123 "unsafe alloc under active trace" -> the multi-minute recompile/alloc stall (W1).
        # The bucket loop above only warmed row-count 1. Pre-allocate every ``(N, prefill_w)`` buffer here
        # (pure allocation, before the decode trace exists — same as the (1,w) warmup), so serving-time
        # concurrent prefill is allocation-free for all batch sizes. Cheap: N buffer allocs, no compute.
        if prefill_w:
            for N in range(1, int(self.max_batch_size) + 1):
                if hybrid:
                    attention_tables = [torch.zeros((N, prefill_w), dtype=torch.int32) for _ in kv_cache]
                    fill_tables = [torch.full((N, prefill_w), -1, dtype=torch.int32) for _ in kv_cache]
                    self._prefill_pt_grouped(attention_tables)
                    self._prefill_fill_pt_grouped(fill_tables)
                else:
                    self._prefill_pt(torch.zeros((N, prefill_w), dtype=torch.int32))
                    self._prefill_fill_pt(torch.full((N, prefill_w), -1, dtype=torch.int32))
        self._in_prefill_warmup = False
        self._report_dram("prefill_warmup")
        self._prefill_programs_warmed = True
        return None

    def warmup_model_decode(
        self,
        kv_cache=None,
        enable_trace=False,
        max_batch_size=None,
        num_blocks=None,
        can_sample_on_device=False,
        **kwargs,
    ):
        """Decode warmup. Phase 2 (``enable_trace=True``) pre-captures the single decode trace for the
        full padded batch (``max_batch_size``) over the vLLM-owned cache, so the first real decode
        replays a ready trace instead of compiling+capturing under a live request. ``_decode_state``
        compiles then captures internally, so Phase 1 (``enable_trace=False``) is a no-op. A dummy
        all-zeros page table is used for capture (writes land in block 0 at position 0 and are
        overwritten by the first real prefill); every real decode refreshes the persistent page
        table / positions from the scheduler before replay."""
        # Remember the per-request block width in BOTH phases so prefill warmup (which the plugin runs
        # just before the decode trace is captured) can pre-allocate the serving-shape page table.
        if num_blocks:
            self._max_blocks = int(num_blocks)
        if not enable_trace or kv_cache is None or max_batch_size is None:
            return None
        B = int(max_batch_size)
        if self._DFLASH_SERVING_ENABLED:
            if B != 1:
                raise RuntimeError(f"DFlash decode warmup requires max_batch_size=1, got {B}")
            # The first serving tranche uses eager draft + target verification.
            # Keeping a normal CCL decode trace resident would make its dynamic
            # proposal allocations unsafe, so no trace is captured in this mode.
            print(
                "[laguna dflash] warmup: normal decode trace OMITTED; "
                "batch-1 eager five-layer proposal + target verify selected",
                flush=True,
            )
            self._report_dram("dflash_ready", enforce=True)
            return None
        # SPEC-DECODE served mode (TT_LAGUNA_SPEC_DECODE=1): capture the VERIFY traces (K1=1..k_max+1) and
        # OMIT the normal decode trace. Two resident CCL-bearing traces (normal decode + verify) deadlock
        # the mesh; routing ALL batch-1 greedy decode through the verify traces (K1=1 = a native single-token
        # step, K1=K+1 = a spec step) keeps only one trace family resident. Serve with --max-num-seqs 1 so
        # B==1 (no padding). Mirrors the standalone driver's capture_decode_trace=False.
        if self._spec_mode == "1":
            k_max = int(os.environ.get("TT_LAGUNA_SPEC_K", "4"))
            single = os.environ.get("TT_LAGUNA_SPEC_SINGLE", "") == "1"
            # SINGLE mode: capture ONLY the K1=k_max+1 verify trace (one resident trace, fixed-K, always-spec).
            # Tests whether multi-trace COEXISTENCE is the intermittent-corruption source: the standalone driver
            # (single fixed-K verify trace) is correct; serving uses adaptive-K -> up to 5 coexisting traces.
            draft_lens = [k_max] if single else list(range(0, k_max + 1))
            self.warmup_verify_decode_multi(draft_lens, kv_cache, int(num_blocks) if num_blocks else 1)
            print(
                f"[laguna spec] warmup: captured verify traces K1={[d + 1 for d in draft_lens]}; "
                f"normal decode trace OMITTED (deadlock-safe, batch-1 greedy spec-decode; single={single})",
                flush=True,
            )
            self._report_dram("trace", enforce=True)
            self._freeze_program_cache_after_trace()
            return None
        if B in self._decode:
            return None
        nb = int(num_blocks) if num_blocks else 1
        hybrid = self._kv_cache_is_hybrid(kv_cache)
        if hybrid:
            # Capture the decode trace over four full-width group page tables.
            ptl = [torch.zeros([B, nb], dtype=torch.int32) for _ in kv_cache]
            pt_persist, groups, reps = self._decode_pt_grouped_alloc(ptl)
            st = self._decode_state(B, kv_cache, pt_persist)
            st["pt_groups"], st["pt_reps"], st["last_pt_host_groups"] = groups, reps, {}
        else:
            pt_persist = self.gen._rep(torch.zeros([B, nb], dtype=torch.int32), ttnn.int32)
            self._decode_state(B, kv_cache, pt_persist)
        self._report_dram("trace", enforce=True)
        self._freeze_program_cache_after_trace()
        return None

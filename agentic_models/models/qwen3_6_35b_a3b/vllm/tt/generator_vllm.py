# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone TT-plugin translation for the selected Qwen full-model generator."""

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from agentic_models.models.qwen3_6_35b_a3b.vllm.tt.generator import QwenGenerator
from agentic_models.models.qwen3_6_35b_a3b.vllm.tt.model import QwenModel
from agentic_models.models.qwen3_6_35b_a3b.vllm.tt.serving_capabilities import QwenServingCapabilities


class QwenForCausalLM:
    model_capabilities = dict(QwenServingCapabilities.serving_capabilities)
    # vllm-tt-plugin patch P1 (agentic-models patches.json): a continuing chunked prefill keeps its GDN state slot.
    model_capabilities["preserve_prefill_state_slots"] = True
    if os.environ.get("QWEN_FAST", "0") == "1":
        # The fast generator runs presence/frequency/repetition penalties in its traced device sampler.
        model_capabilities["supports_device_penalties"] = True
        if int(os.environ.get("QWEN_SPEC", "0")) > 0:
            # MTP speculative decoding (tt/fast_spec.py, tt/fast_mtp.py): device drafter, argmax_ids verify whose ids
            # are the device sampler's draws (exact rejection sampling for sampled requests), narrow decode.
            model_capabilities.update(
                supports_spec_decode=True,
                spec_requirements=("device_propose",),
                spec_hidden_handoff=("on_device",),
                spec_ids_are_sampled=True,
                supports_async_spec_decode=True,
            )
        if os.environ.get("QWEN_PREFIX_CACHE", "0") == "1":
            # GDN state snapshots at 32-token block prefixes (tt/prefix_cache.py) make vLLM prefix hits usable.
            model_capabilities["supports_prefix_caching"] = True
    decode_input_update_contract = 1
    _HYBRID_KV_CACHE_GROUPS_ENABLED = False

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1:
            raise ValueError("This TP4/EP4 adapter requires one 1x4 mesh")
        if optimizations not in (None, "performance"):
            raise ValueError("Serving must use the datatype-sweep selection")
        if not 1 <= max_batch_size <= 32:
            raise ValueError("Supported serving capacity is 1 through 32")
        instance = cls()
        instance.max_batch_size = max_batch_size
        instance.fast = os.environ.get("QWEN_FAST", "0") == "1"
        if instance.fast:
            instance.tt_supported_decode_batch_sizes = sorted(
                {min(b, max_batch_size) for b in (8, 16, 24, 32)} | {max_batch_size}
            )
        else:
            instance.tt_supported_decode_batch_sizes = sorted(
                {min(2, max_batch_size), min(4, max_batch_size), max_batch_size}
            )
        instance.audit_enabled = os.environ.get("QWEN_VLLM_AUDIT", "0") == "1"
        instance.decode_route_counts = {"device": 0, "host_logits": 0}
        instance.last_decode_route = None
        instance.generator = None
        instance.phase_recorder = None
        phase_directory = os.environ.get("QWEN_BENCHMARK_PHASE_DIR")
        if phase_directory:
            from agentic_models.models.qwen3_6_35b_a3b.vllm.tt.benchmark_phases import PhaseRecorder

            instance.phase_recorder = PhaseRecorder(phase_directory, max_batch_size)
        instance.uses_mrope = getattr(hf_config, "model_type", "") == "qwen3_5_moe"
        checkpoint = getattr(hf_config, "_name_or_path", None)
        if checkpoint and not Path(checkpoint).is_dir():
            checkpoint = None
        # Diagnostic-only switch, read by this model, never by shared plugin code.
        reduced = os.environ.get("QWEN_VLLM_REDUCED") == "1"
        instance.model = QwenModel(
            mesh_device,
            checkpoint=checkpoint,
            layer_indices=[0, 3] if reduced else None,
            max_seq_len=max_seq_len,
            suppress_inactive_moe_routes=max_batch_size >= 2,
            fuse_moe_route_plan=max_batch_size >= 2,
            specialize_solo_slot0_attention=max_batch_size >= 2,
            fuse_recurrent_state_commit=max_batch_size >= 2,
            fuse_hidden_row_mask=max_batch_size >= 2,
            packed_postreduce_weights=max_batch_size >= 2,
            native_qk_reader=max_batch_size >= 2,
            prune_postreduce_padding=max_batch_size >= 2,
            fast=instance.fast,
        )
        instance._event(
            "initialize",
            adapter=f"{cls.__module__}:{cls.__name__}",
            module_file=__file__,
            worker_grid=str(mesh_device.compute_with_storage_grid_size()),
            device_ids=mesh_device.get_device_ids(),
            pid=os.getpid(),
            reduced=reduced,
            layers=list(instance.model.layer_indices),
            precision=instance.model.precision_config,
            max_model_len=max_seq_len,
            max_num_seqs=max_batch_size,
        )
        return instance

    @staticmethod
    def get_max_tokens_all_users(**kwargs):
        # Aggregate KV pool (tokens) shared by all slots. QWEN_KV_POOL_TOKENS raises it for many concurrent
        # long requests (each token costs ~10 KiB per chip across the 10 attention layers).
        return int(os.environ.get("QWEN_KV_POOL_TOKENS", "262144"))

    @staticmethod
    def get_kv_cache_spec(vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        config = vllm_config.model_config.hf_text_config
        return {
            f"model.layers.{index}.self_attn": FullAttentionSpec(
                block_size=32, num_kv_heads=config.num_key_value_heads, head_size=config.head_dim, dtype=torch.bfloat16
            )
            # The plugin orders cache layers by attention ordinal, while the
            # generator owns the mapping back to checkpoint indices 3,7,...39.
            for index in range(config.layer_types.count("full_attention"))
        }

    def _event(self, event, **fields):
        if event == "decode":
            route = fields["sampling"]
            self.decode_route_counts[route] += 1
            # Keep the compact route ledger even without detailed auditing:
            # vLLM may terminate its worker before the teardown totals print.
            print("QWEN_DECODE sampling_mode=" + ("device" if route == "device" else "host"), flush=True)
            self.last_decode_route = route
        if event in ("prefill", "decode", "decode_submitted") and not getattr(self, "audit_enabled", False):
            return
        generator = self.generator
        print(
            "QWEN_VLLM_EVENT "
            + json.dumps(
                dict(
                    event=event,
                    monotonic=time.monotonic(),
                    traces={} if generator is None else generator.traces,
                    counters={} if generator is None else generator.counters,
                    trace_replays={} if generator is None else generator.trace_replays,
                    sampling_trace_replays={} if generator is None else generator.sampling_trace_replays,
                    decode_route_counts=getattr(self, "decode_route_counts", {}),
                    **fields,
                ),
                default=str,
            ),
            flush=True,
        )

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        if self.generator is not None:
            raise RuntimeError("A serving cache cannot be rebound after allocation")
        specs = list(per_layer_specs)
        if os.environ.get("QWEN_VLLM_REDUCED") == "1":
            specs = specs[:1]
        cache = self.model.allocate_kv_cache_per_layer(specs, batch=self.max_batch_size)
        if getattr(self, "fast", False):
            from agentic_models.models.qwen3_6_35b_a3b.vllm.tt.fast_generator import FastQwenGenerator

            self.generator = FastQwenGenerator(self.model, max_batch_size=self.max_batch_size, kv_cache=cache)
        else:
            self.generator = QwenGenerator(
                self.model,
                max_batch_size=self.max_batch_size,
                kv_cache=cache,
                decode_buckets=self.tt_supported_decode_batch_sizes,
            )
        self._event("cache_allocated", specs=specs, owner="vllm")
        return cache

    def _prepare(self, kv_cache):
        if self.generator is None or kv_cache is not self.generator.cache:
            raise ValueError("Warmup must use the exact vLLM allocation")
        if not self.generator.prepared:
            started = time.monotonic()
            self.generator.prepare()
            self._event("prepared", seconds=time.monotonic() - started)

    def warmup_model_prefill(self, *, kv_cache, can_sample_on_device, enable_trace):
        if enable_trace:
            raise ValueError("Use trace_mode=decode_only; prefill uses prepared eager programs")
        self._prepare(kv_cache)

    def warmup_model_decode(self, *, kv_cache, max_batch_size, num_blocks, can_sample_on_device, enable_trace):
        self._prepare(kv_cache)

    def prefill_forward(self, tokens, **kwargs):
        # The checkpoint is multimodal, but this autoport serves text only.
        # The plugin still sends empty processor fields and expects mRoPE
        # deltas; ordinary text has the scalar positions and zero delta.
        for name in (
            "pixel_values",
            "image_grid_thw",
            "mm_identifiers",
            "mm_placeholder_lengths",
            "mm_placeholder_offsets",
        ):
            value = kwargs.pop(name, None)
            if value is not None and any(item is not None for item in value):
                raise ValueError(f"Text-only adapter cannot consume {name}")
        starts = torch.as_tensor(kwargs.get("start_pos")).tolist()
        ends = torch.as_tensor(kwargs["prompt_lens"]).tolist()
        recorder = getattr(self, "phase_recorder", None)
        phase_call = None
        if recorder is not None:
            slots = kwargs.get("empty_slots")
            slots = list(range(len(starts))) if slots is None else self._audit_cpu_list(slots)
            phase_call = recorder.prefill_start(starts, ends, slots)
        prompt_chunks = []
        if getattr(self, "audit_enabled", False):
            slots = kwargs.get("empty_slots")
            slots = range(len(starts)) if slots is None else slots
            for row, (start, end, slot) in enumerate(zip(starts, ends, slots)):
                ids = self._audit_cpu_list(tokens[row, start:end])
                prompt_chunks.append(
                    dict(
                        slot=int(slot),
                        start=start,
                        end=end,
                        token_digest=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                    )
                )
        self._event(
            "prefill",
            starts=starts,
            ends=ends,
            lengths=[end - start for start, end in zip(starts, ends)],
            slots=kwargs.get("empty_slots"),
            sampling="device" if kwargs.get("sampling_params") is not None else "host_logits",
            active_decode_slots=list(self.generator.active_slots),
            prompt_chunks=prompt_chunks,
        )
        self._translate_sampling(kwargs)
        result = self.generator.prefill_forward(tokens, read_from_device=True, **kwargs)
        if kwargs.get("sampling_params") is not None:
            self._audit_tokens(result)
        if recorder is not None:
            recorder.prefill_done(phase_call)
        return (result, torch.zeros(tokens.shape[0], dtype=torch.int64)) if self.uses_mrope else result

    # ------------------------------------------------------------------ speculative decoding (QWEN_SPEC=K)
    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        k = int(os.environ.get("QWEN_SPEC", "0"))
        if k <= 0:
            return SpecReject(reason="Qwen3.6 MTP speculation is built only with QWEN_SPEC=<K> set", supported_k=())
        if int(requested_k) < k:
            return SpecReject(
                reason=f"verify graph is captured at K={k} (QWEN_SPEC); request {k} drafts", supported_k=(k,)
            )
        return SpecPlan(
            effective_k=k,
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=True,
        )

    def _spec_verify(self, tokens, start_pos, kwargs, num_valid, accepted):
        from vllm_tt_plugin.spec_decode import VerifyOutput

        g = self.generator
        tokens = torch.as_tensor(tokens)
        start_pos = torch.as_tensor(start_pos)
        slots = kwargs.get("slots")
        rows = list(range(len(start_pos))) if slots is None else torch.as_tensor(slots).reshape(-1).tolist()
        if rows != list(range(len(rows))):
            raise ValueError("Qwen3.6 speculative verify expects the front-packed row = slot layout")
        accepted = torch.ones(len(rows), dtype=torch.int32) if accepted is None else torch.as_tensor(accepted)
        g._spec_pending_accepted = {r: int(accepted[r]) for r in range(len(rows))}
        if any(r >= g.spec.U for r in range(len(start_pos)) if int(start_pos[r, 0]) >= 0):
            # vllm-tt-plugin f6fe76b keeps proposals across a prefill step, so drafts made at <= QWEN_SPEC_MAX_ROWS
            # live rows can reach a step whose new peers sit in slots the verify graph does not serve. Drop the
            # drafts: decode column 0 the ordinary way (committing the pending accepted counts first) and answer
            # narrow, [B, 1], which the plugin's accept walk commits as one token per row.
            kwargs = dict(kwargs)
            kwargs["read_from_device"] = True
            try:
                host = self.generator.decode_forward_v1(
                    tokens[:, :1].contiguous(), start_pos[:, 0].contiguous(), **kwargs
                )
            finally:
                g._spec_pending_accepted = None
            host = torch.as_tensor(host)
            if host.is_floating_point() or host.numel() != len(rows):
                raise ValueError("Qwen3.6 verify fallback requires device-sampled token ids")
            self._spec_last = ("ordinary", None)
            return VerifyOutput(spec_mode="argmax_ids", argmax_ids=host.reshape(-1, 1).to(torch.int32), hidden=None)
        result = {}

        def forward():
            result["ids"] = g.spec_verify(tokens, start_pos, num_valid, accepted)

        g._spec_forward = forward
        try:
            # run the ordinary preamble (remap, sampling params, pages) with column-0 views of the block
            kwargs = dict(kwargs)
            kwargs["read_from_device"] = True
            self.generator.decode_forward_v1(tokens[:, :1].contiguous(), start_pos[:, 0].contiguous(), **kwargs)
        finally:
            g._spec_forward = None
            g._spec_pending_accepted = None
        self._spec_last = ("verify", start_pos.clone())
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=result["ids"], hidden=None)

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, DraftOutput

        g = self.generator
        B = len(committed_positions)
        k = int(num_drafts)
        out = torch.full((B, k), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        valid = torch.zeros(B, dtype=torch.int32)
        pos = torch.as_tensor(committed_positions)
        live = [r for r in range(B) if int(pos[r, 0]) >= 0]
        max_rows = int(os.environ.get("QWEN_SPEC_MAX_ROWS", "4"))
        sv = g.sampling_values
        penalized = any(
            sv["presence_penalty"][r] != 0.0 or sv["frequency_penalty"][r] != 0.0 or sv["repetition_penalty"][r] != 1.0
            for r in live
        )
        last = getattr(self, "_spec_last", None)
        if not live or len(live) > max_rows or any(r >= g.spec.U for r in live) or penalized or k != g.spec.K:
            return DraftOutput(draft_token_ids=out, num_valid=valid)
        if last is not None and last[0] == "verify":
            drafts, users = g.spec_propose(True, last[1], accepted=torch.as_tensor(accepted_counts))
        else:
            # after an ordinary decode: committed column 0 sits at (processed position + 1)
            processed = torch.tensor([int(pos[r, 0]) - 1 if r in live else -1 for r in range(B)])
            drafts, users = g.spec_propose(False, processed)
        for u in users:
            if u < B:
                out[u] = drafts[u]
                valid[u] = k
        return DraftOutput(draft_token_ids=out, num_valid=valid)

    def decode_forward(self, tokens, start_pos, **kwargs):
        spec_mode = kwargs.pop("spec_mode", None)
        num_valid = kwargs.pop("num_valid_drafts", None)
        accepted = kwargs.pop("accepted_counts", None)
        kwargs.pop("draft_token_ids", None)
        if spec_mode is not None:
            if spec_mode != "argmax_ids":
                raise ValueError(f"Qwen3.6 speculative verify serves argmax_ids, not {spec_mode}")
            kwargs.pop("rope_deltas_all_users", None)
            self._translate_sampling(kwargs)  # same plugin -> sampler normalization as the ordinary path
            return self._spec_verify(tokens, start_pos, kwargs, num_valid, accepted)
        if accepted is not None and getattr(self.generator, "spec", None) is not None:
            # ordinary decode right after a multi-token commit: leave the verify ring at the committed slot
            self.generator._spec_pending_accepted = {
                r: int(a) for r, a in enumerate(torch.as_tensor(accepted).tolist())
            }
        self._spec_last = ("ordinary", None)
        deltas = kwargs.pop("rope_deltas_all_users", None)
        if deltas is not None and torch.as_tensor(deltas).count_nonzero().item():
            raise ValueError("Text-only adapter requires zero mRoPE deltas")
        auditing = getattr(self, "audit_enabled", False)
        before = dict(self.generator.counters) if auditing else None
        page_changes = self._page_change_audit(kwargs) if auditing else None
        audit_fields = {}
        if auditing:
            self.audit_submission_id = getattr(self, "audit_submission_id", 0) + 1
            params = kwargs.get("sampling_params")
            audit_fields = dict(
                submission_id=self.audit_submission_id,
                submitted_slots=self._audit_cpu_list(kwargs.get("slots")),
                remembered_rows=list(getattr(self.generator, "serving_rows", range(self.max_batch_size))),
                # Diagnostic CPU metadata only. Async positions can lag the
                # device; never use this value to update resident state.
                advisory_host_positions=self._audit_cpu_list(start_pos),
                reload_sampling_params=kwargs.get("reload_sampling_params"),
                reset_sampling_state=kwargs.get("reset_sampling_state"),
                sampling_seeds=self._audit_cpu_list(getattr(params, "seed", None)),
            )
        self._event(
            "decode",
            sampling="device" if kwargs.get("sampling_params") is not None else "host_logits",
            reload_inputs=kwargs.get("reload_inputs"),
            reload_page_table=kwargs.get("reload_page_table"),
            page_changes=page_changes,
            slot_remap=self._audit_cpu_list(kwargs.get("slot_remap")) if auditing else None,
            pending_outputs=sum(item is not None for item in self.generator.output_pool.leases) if auditing else None,
            **audit_fields,
        )
        if kwargs.get("reload_sampling_params") or kwargs.get("reset_sampling_state"):
            self._translate_sampling(kwargs)
        recorder = getattr(self, "phase_recorder", None)
        if recorder is not None:
            reload_inputs = kwargs.get("reload_inputs", True)
            positions = self._audit_cpu_list(start_pos) if reload_inputs else None
            rows = self._audit_cpu_list(kwargs.get("slots"))
            rows = list(range(len(positions))) if rows is None and positions is not None else rows
            phase_call = recorder.decode_start(
                positions=positions,
                rows=rows,
                remap=self._audit_cpu_list(kwargs.get("slot_remap")),
                reload_inputs=reload_inputs,
            )
        result = self.generator.decode_forward_v1(tokens, start_pos, **kwargs)
        if recorder is not None:
            recorder.decode_accepted(
                phase_call,
                result,
                self.generator.active_slots,
                self.generator.model_trace_key(self.generator.active_slots),
            )
        if auditing:
            self._event(
                "decode_submitted",
                submission_id=self.audit_submission_id,
                active_slots=list(self.generator.active_slots),
                counter_delta={key: value - before[key] for key, value in self.generator.counters.items()},
                page_only=bool(kwargs.get("reload_page_table")) and not kwargs.get("reload_inputs"),
                pending_outputs=sum(item is not None for item in self.generator.output_pool.leases),
            )
        return result

    @staticmethod
    def _audit_cpu_list(value):
        """Serialize existing host metadata without initiating a device read."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("Optional routing audit accepts CPU metadata only")
            return value.tolist()
        if isinstance(value, (list, tuple, range)):
            return list(value)
        if isinstance(value, (int, float)):
            return value
        raise TypeError(f"Unsupported routing audit metadata: {type(value).__name__}")

    def _page_change_audit(self, kwargs):
        """Inspect existing CPU inputs/shadows only; no device observation."""
        if not kwargs.get("reload_page_table") or kwargs.get("reload_inputs"):
            return []
        table = kwargs.get("page_table")
        per_layer = kwargs.get("page_tables_per_layer")
        if per_layer is not None:
            table = per_layer[0]
        if table is None:
            return []
        table = torch.as_tensor(table)
        if table.device.type != "cpu":
            raise ValueError("Page audit requires the serving contract's CPU table")
        group = next(states for states in self.generator.cache if "page_table" in states[0])
        order = kwargs.get("slot_remap")
        order = list(range(self.max_batch_size)) if order is None else list(order)
        rows = kwargs.get("slots")
        rows = list(range(table.shape[0])) if rows is None else list(rows)
        changes = []
        for row, slot in enumerate(rows):
            old = group[int(order[slot])].get("_host_pages")
            if old is None:
                continue
            old = old.reshape(-1)[: table.shape[1]]
            new = table[row]
            for index in ((new >= 0) & (new != old)).nonzero().reshape(-1).tolist():
                changes.append(dict(slot=slot, logical_page=index, old=int(old[index]), new=int(new[index])))
        return changes

    @staticmethod
    def _translate_sampling(kwargs):
        params = kwargs.get("sampling_params")
        if params is None or params.num_logprobs is None:
            return
        # The pinned plugin uses -2 for disabled logprobs; the common model
        # sampler uses 0 plus enable_log_probs=False. Keep the enable flag and
        # every other feature intact so real unsupported requests still reject.
        counts = params.num_logprobs
        normalized = [0 if count == -2 else count for count in counts]
        kwargs["sampling_params"] = replace(params, num_logprobs=normalized)

    def read_decode_output(self, output, *, async_read=False, **kwargs):
        if isinstance(output, torch.Tensor):  # a verify's host ids (spec decode): nothing to read
            return output, []
        return self.generator.read_decode_output(output, async_read=async_read, **kwargs)

    def process_decode_output_host(self, output, *, is_tokens):
        # MODEL_EXECUTION.md: the plugin synchronizes returned read events
        # before this hook. Synchronous published outputs still perform their
        # blocking read in the pool. Standalone generator defaults keep waiting.
        result = self.generator.process_decode_output_host(output, is_tokens=is_tokens, events_synchronized=True)
        if is_tokens:
            self._audit_tokens(result)
        recorder = getattr(self, "phase_recorder", None)
        if recorder is not None:
            recorder.decode_done(output)
        return result

    def _audit_tokens(self, tokens):
        # Inspect only the token result already required by vLLM. Never read
        # logits or feed this host result back to the device decode inputs.
        invalid = int(((tokens < 0) | (tokens >= self.model.vocab_size)).sum())
        if getattr(self, "audit_enabled", False) or invalid:
            print(f"QWEN_TOKEN_AUDIT count={tokens.numel()} out_of_vocabulary={invalid}", flush=True)
        if invalid:
            raise RuntimeError("Device sampling emitted an out-of-vocabulary token")

    def release_request(self, slot):
        result = self.generator.release_request(slot)
        recorder = getattr(self, "phase_recorder", None)
        if recorder is not None:
            recorder.release(slot)
        return result

    def release_persistent_capture(self):
        if self.generator is not None:
            self._event("shutdown_before_release")
            self.generator.release_persistent_capture()
            self._event("shutdown_after_release")

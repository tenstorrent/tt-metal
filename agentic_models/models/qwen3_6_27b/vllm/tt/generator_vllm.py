# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Translation of the standalone TT plugin interface to QwenGenerator.

The scheduler owns attention pages. The generator owns fixed-size recurrent
state and all forward/sampling traces, including device token feedback.
"""

import json
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch

import ttnn

from .benchmark_timing import PhaseTiming, tensor_inventory
from .generator import QwenGenerator
from .model import Model


@dataclass
class DecodeOutput:
    tensor: object
    is_tokens: bool
    rows: int | None = None
    phase_event: object = None


class QwenVLLM:
    decode_input_update_contract = 1
    model_capabilities = {
        "supports_chunked_prefill": True,
        # GDN state cannot be reconstructed from cached attention blocks; with
        # QWEN_VLLM_PREFIX_CACHE=<N> the model keeps N recurrent-state snapshots
        # (perf/multiuser) and restores them on vLLM prefix-cache hits.
        "supports_prefix_caching": int(os.environ.get("QWEN_VLLM_PREFIX_CACHE", "0")) > 0,
        "supports_sample_on_device": True,
        "supports_prefill_sampling_mask": True,
        "supports_original_prompt_lens": True,
        # A continuing chunked prefill keeps its own recurrent-state slot (plugin P1).
        "preserve_prefill_state_slots": True,
        # APC with model-owned GDN snapshots: the plugin serves plain attention prefix caching
        # (mamba_cache_mode none, plugin P4). No prefix-hit clip is registered.
        "prefix_cache_recurrent_state_in_model": int(os.environ.get("QWEN_VLLM_PREFIX_CACHE", "0")) > 0,
        # ttnn.sampling at the tt-metal pin takes k <= 32; the plugin samples top_k > 32 on the host.
        "max_device_top_k": 32,
        "supports_device_penalties": True,
        "supports_device_grammar": False,
        "supports_async_decode": True,
    }
    # perf/spec: QWEN_VLLM_SPEC_K=<K> serves MTP speculative decoding (model-owned drafter,
    # launch with --speculative-config '{"method":"custom_class","model":"vllm_tt_plugin.model_owned_drafter",
    # "num_speculative_tokens":K}'). Verify returns device-sampled ids per candidate column.
    SPEC_K = int(os.environ.get("QWEN_VLLM_SPEC_K", "0"))
    if SPEC_K:
        model_capabilities.update(
            supports_spec_decode=True,
            spec_requirements=["device_propose"],
            spec_ids_are_sampled=True,
            supports_async_spec_decode=True,
        )

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        k = min(int(requested_k), cls.SPEC_K or 0, 3)
        if k < 1:
            return SpecReject(
                reason="QWEN_VLLM_SPEC_K must be 1..3 and match num_speculative_tokens", supported_k=(1, 2, 3)
            )
        # B live rows x (1+k) candidate columns must fit the 32-row decode tile: k<=3 up to 8 rows,
        # k=1 up to 16 rows; above 16 live rows the drafter declines (ordinary decode).
        table = ((8, k), (16, 1)) if k > 1 else ((16, 1),)
        if max_num_seqs > table[-1][0]:
            table += ((int(max_num_seqs), 1),)
        return SpecPlan(
            effective_k=k,
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=True,
            k_by_rows=table,
        )

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1:
            raise ValueError("Qwen autoport requires one TP4 replica")
        # This diagnostic is model-owned and never changes the plugin interface.
        layers = os.environ.get("QWEN_VLLM_DIAGNOSTIC_LAYERS")
        indices = None if layers is None else list(map(int, layers.split(",")))
        # Optional eval-only A/B: QWEN_VLLM_PRECISION_CONFIG=<policy json> (default: stage-8 selected policy).
        model = Model(mesh_device, layer_indices=indices, precision_config=os.environ.get("QWEN_VLLM_PRECISION_CONFIG"))
        if cls.SPEC_K:
            model.load_mtp()
        result = cls(model, max_batch_size, max_seq_len)
        print(
            "QWEN_VLLM_REGISTRATION "
            + json.dumps(dict(module=cls.__module__, adapter=cls.__name__, layers=model.layer_indices)),
            flush=True,
        )
        return result

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs):
        # Validated full-model pool, before plugin null/output reservation.
        # QWEN_VLLM_KV_POOL_TOKENS lets eval serving give concurrent long
        # reasoning requests their own pages (avoids vLLM preemption, which
        # was never validated with model-owned GDN state). Default unchanged.
        return int(os.environ.get("QWEN_VLLM_KV_POOL_TOKENS", "262144"))

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        config = vllm_config.model_config.hf_config
        config = getattr(config, "text_config", config)
        if vllm_config.cache_config.block_size != 32:
            raise ValueError("Selected KV layout requires block_size=32")
        return {
            f"model.layers.{i}.self_attn": FullAttentionSpec(
                block_size=32, num_kv_heads=config.num_key_value_heads, head_size=256, dtype=torch.bfloat16
            )
            for i, kind in enumerate(config.layer_types)
            if kind == "full_attention"
        }

    def __init__(self, model, max_batch_size, max_seq_len):
        self.model = model
        self.mesh = model.mesh
        self.batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.request_specific_rope = bool(getattr(model.config, "rope_parameters", {}).get("mrope_section"))
        # perf/multiuser: capture prefix decode buckets so N users pay for the
        # next bucket >= N rows instead of the full slot capacity.
        # QWEN_VLLM_DECODE_BUCKETS=1,32 restores the stage-10 bucket set.
        buckets = os.environ.get("QWEN_VLLM_DECODE_BUCKETS", "1,2,4,8,16,32")
        self.tt_supported_decode_batch_sizes = sorted(
            {b for b in map(int, buckets.split(",")) if 1 <= b <= max_batch_size} | {max_batch_size}
        )
        self.generator = None
        self.active_decode_rows = 0
        self._precision_reported = False
        self.telemetry = os.environ.get("QWEN_VLLM_TELEMETRY", "1") != "0"
        self.host_compatibility = os.environ.get("QWEN_VLLM_HOST_COMPATIBILITY") == "1"
        phase_path = os.environ.get("QWEN_BENCHMARK_PHASE_LOG")
        self.phase_timing = PhaseTiming(phase_path, max_batch_size, ttnn.event_query) if phase_path else None

    def allocate_kv_cache(self, shape, dtype, num_layers):
        if self.generator is not None:
            raise RuntimeError("Cannot replace a cache bound to live traces")
        if tuple(shape)[1:] != (1, 32, 256):
            raise ValueError(f"Unexpected scheduler KV shape: {shape}")
        pages = max(16, ((self.max_seq_len + 511) // 512) * 16)
        table = torch.zeros((self.batch_size, pages), dtype=torch.int32)
        # create_cache allocates max(page_table)+1 physical pages, independently
        # of virtual context width. This bootstrap map is replaced before use.
        table[0, -1] = shape[0] - 1
        cache = self.model.create_cache(table)
        self.generator = QwenGenerator(
            self.model,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            kv_cache=cache,
            page_table=table,
            decode_batch_sizes=self.tt_supported_decode_batch_sizes,
            batched_prefill=os.environ.get("QWEN_VLLM_BATCHED_PREFILL") == "1",
            spec_width=self.SPEC_K + 1 if self.SPEC_K else 0,
        )
        snapshots = int(os.environ.get("QWEN_VLLM_PREFIX_CACHE", "0"))
        if snapshots > 0:
            self.generator.enable_prefix_cache(snapshots)
        print("QWEN_VLLM_CACHE " + json.dumps(dict(shape=shape, attention_layers=num_layers)), flush=True)
        return cache

    def _check_cache(self, kv_cache):
        if self.generator is None or kv_cache is not self.generator.cache:
            raise ValueError("Serving must pass the exact scheduler-allocated cache")

    def warmup_model_prefill(self, *, kv_cache, **kwargs):
        self._check_cache(kv_cache)
        self.generator.prepare()
        if not self._precision_reported:
            print(
                "QWEN_VLLM_PRECISION "
                + json.dumps(
                    dict(
                        model=self.model.precision_summary(),
                        runtime=self.generator.precision_runtime_summary(),
                    )
                ),
                flush=True,
            )
            self._precision_reported = True
            if self.phase_timing:
                self.phase_timing.ready(
                    dict(
                        imported_module=__name__,
                        imported_file=os.path.realpath(__file__),
                        layer_count=len(self.model.layers),
                        layers=self.model.layer_indices,
                        precision=self.model.precision,
                        model_config=self.model.config.to_dict(),
                        weights=tensor_inventory(self.model, ttnn),
                        devices=list(self.mesh.get_device_ids()),
                        mesh_shape=list(self.mesh.shape),
                        compute_grid=[
                            self.mesh.compute_with_storage_grid_size().x,
                            self.mesh.compute_with_storage_grid_size().y,
                        ],
                        max_model_len=self.max_seq_len,
                    )
                )

    def warmup_model_decode(self, *, kv_cache, **kwargs):
        self._check_cache(kv_cache)
        self.generator.prepare()

    def _pages(self, table):
        table = torch.as_tensor(table, dtype=torch.int32)
        width = self.generator.page_host.shape[1]
        if table.shape[1] > width:
            raise ValueError("Scheduler table exceeds supported context")
        return torch.nn.functional.pad(table, (0, width - table.shape[1]))

    def _sampling_mode(self, operation, sampling_params):
        mode = "device" if sampling_params is not None else "host_compatibility"
        marker = " host_sampling" if sampling_params is None else ""
        print(f"QWEN_VLLM_SAMPLING operation={operation} mode={mode}{marker}", flush=True)
        if sampling_params is None and not self.host_compatibility:
            raise ValueError("Host sampling requires explicit QWEN_VLLM_HOST_COMPATIBILITY=1")

    def prefill_forward(self, tokens, *, page_table, kv_cache, sampling_params=None, **kwargs):
        phase_started = time.perf_counter() if self.phase_timing else None
        # Plugin P2 passes prefill_sampling_mask (True = final row) on device and host routes. The
        # generator's sampling_mask applies to device sampling only; the host route returns every row's
        # logits and the plugin discards intermediate ones, as before.
        prefill_sampling_mask = kwargs.pop("prefill_sampling_mask", None)
        if prefill_sampling_mask is not None and sampling_params is not None:
            kwargs["sampling_mask"] = list(prefill_sampling_mask)
        self._check_cache(kv_cache)
        if self.phase_timing:
            phase_event = self.phase_timing.prefill(
                list(map(int, kwargs["start_pos"])),
                list(map(int, kwargs["prompt_lens"])),
                list(map(int, kwargs.get("empty_slots", range(len(tokens))))),
                list(map(int, kwargs.get("original_prompt_lens", kwargs["prompt_lens"]))),
                kwargs.get("sampling_mask", [True] * len(tokens)),
                torch.as_tensor(tokens)[:, :64].tolist(),
                phase_started,
            )
        self._sampling_mode("prefill", sampling_params)
        print(
            "QWEN_VLLM_PREFILL "
            + json.dumps(
                dict(
                    starts=list(map(int, kwargs["start_pos"])),
                    ends=list(map(int, kwargs["prompt_lens"])),
                    slots=list(map(int, kwargs.get("empty_slots", range(len(tokens))))),
                    traces={key: int(trace) for key, trace in self.generator.traces.items()},
                    sampling_mask=kwargs.get("sampling_mask"),
                    original_prompt_lens=kwargs.get("original_prompt_lens"),
                    active_decode_rows=self.active_decode_rows,
                    counters=dict(self.generator.counters),
                    program_cache_entries=int(self.mesh.num_program_cache_entries()),
                )
            ),
            flush=True,
        )
        output = self.generator.prefill_forward(
            tokens,
            page_table=self._pages(page_table),
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            **kwargs,
        )
        if sampling_params is not None and bool(((output < 0) | (output >= self.model.config.vocab_size)).any()):
            raise RuntimeError(f"Device sampler emitted out-of-vocabulary IDs: {output.tolist()}")
        if self.phase_timing:
            self.phase_timing.complete_prefill(phase_event, ttnn.record_event(self.mesh, 0))
        # The checkpoint retains mRoPE metadata in text-only serving. Every
        # axis then has the ordinary text position, so its request delta is0.
        return (output, torch.zeros(len(tokens), dtype=torch.int64)) if self.request_specific_rope else output

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        page_table,
        kv_cache,
        sampling_params=None,
        read_from_device=True,
        **kwargs,
    ):
        phase_started = time.perf_counter() if self.phase_timing else None
        phase_event = None
        self._check_cache(kv_cache)
        self._sampling_mode("decode", sampling_params)
        if self.phase_timing:
            remap = kwargs.get("slot_remap")
            phase_event = self.phase_timing.decode(
                torch.as_tensor(start_pos).flatten().tolist() if kwargs.get("reload_inputs", True) else [],
                None if remap is None else list(map(int, remap)),
                kwargs.get("reload_inputs", True),
                len(tokens),
                phase_started,
            )
        submitted_batch = len(tokens)
        if submitted_batch not in self.tt_supported_decode_batch_sizes:
            raise ValueError("The plugin submitted an unprepared decode bucket")
        spec_mode = kwargs.pop("spec_mode", None)
        num_valid_drafts = kwargs.pop("num_valid_drafts", None)
        if spec_mode is not None:
            return self._verify(
                tokens, start_pos, page_table, kv_cache, sampling_params, num_valid_drafts, spec_mode, kwargs
            )
        if kwargs.get("reload_sampling_params", False) and sampling_params is not None:
            # vLLM pads parameters to its selected execution bucket. Merge the
            # submitted prefix into fixed-capacity controls, in the destination
            # slot order of the same remap the generator will apply below.
            params = SimpleNamespace(**vars(sampling_params))
            remap = kwargs.get("slot_remap")
            order = list(range(self.batch_size)) if remap is None else list(map(int, remap))
            for name in (
                "temperature",
                "top_k",
                "top_p",
                "seed",
                "presence_penalty",
                "frequency_penalty",
                "repetition_penalty",
            ):
                current = self.generator.request_seeds if name == "seed" else self.generator.sampling_values[name]
                merged = [current[index] for index in order]
                merged[:submitted_batch] = self.generator._sampling_rows(
                    getattr(params, name, 1.0 if name == "repetition_penalty" else 0.0), submitted_batch
                )
                setattr(params, name, merged)
            sampling_params = params
        if kwargs.get("reload_inputs", True) and submitted_batch < self.batch_size:
            tokens = torch.nn.functional.pad(torch.as_tensor(tokens), (0, 0, 0, self.batch_size - submitted_batch))
            start_pos = torch.nn.functional.pad(
                torch.as_tensor(start_pos).flatten(), (0, self.batch_size - submitted_batch), value=-1
            )
        deltas = kwargs.pop("rope_deltas_all_users", None)
        if deltas is not None and bool((torch.as_tensor(deltas) != 0).any()):
            raise ValueError("The text-only adapter requires zero mRoPE deltas")
        refresh_pages = kwargs.get("reload_inputs", True) or kwargs.get("reload_page_table", False)
        if refresh_pages:
            page_table = self._pages(page_table)
            if page_table.shape[0] < self.batch_size:
                full_table = self.generator.page_host.clone()
                full_table[: page_table.shape[0]] = page_table
                page_table = full_table
            if self.telemetry:
                changed = torch.nonzero(page_table != self.generator.page_host).tolist()
                print(
                    "QWEN_VLLM_PAGE_UPDATE "
                    + json.dumps(
                        dict(
                            scheduler_positions=torch.as_tensor(start_pos).flatten().tolist(),
                            reload_inputs=kwargs.get("reload_inputs", True),
                            changed=[
                                [
                                    row,
                                    column,
                                    int(self.generator.page_host[row, column]),
                                    int(page_table[row, column]),
                                ]
                                for row, column in changed
                            ],
                        )
                    ),
                    flush=True,
                )
        if kwargs.get("reload_inputs", True):
            self.active_decode_rows = int((torch.as_tensor(start_pos) >= 0).sum())
        if self.telemetry:
            print(
                "QWEN_VLLM_DECODE "
                + json.dumps(
                    dict(
                        active_rows=self.active_decode_rows,
                        program_cache_entries=int(self.mesh.num_program_cache_entries()),
                        logical_batch=submitted_batch,
                        reload_inputs=kwargs.get("reload_inputs", True),
                        reload_page_table=kwargs.get("reload_page_table", False),
                        slot_remap=kwargs.get("slot_remap"),
                        decode_replays=self.generator.counters["decode_replays"],
                    ),
                    default=lambda value: value.tolist(),
                ),
                flush=True,
            )
        output = self.generator.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            sample_on_device=sampling_params is not None,
            read_from_device=False,
            logical_batch=submitted_batch,
            **kwargs,
        )
        result = DecodeOutput(output, sampling_params is not None, submitted_batch, phase_event)
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(result), is_tokens=result.is_tokens)
        return result

    def _verify(self, tokens, start_pos, page_table, kv_cache, sampling_params, num_valid_drafts, spec_mode, kwargs):
        from vllm_tt_plugin.spec_decode import VerifyOutput

        if spec_mode != "argmax_ids":
            raise ValueError(f"Qwen MTP verify serves argmax_ids, not {spec_mode!r}")
        gen = self.generator
        tokens = torch.as_tensor(tokens, dtype=torch.int32)
        positions = torch.as_tensor(start_pos, dtype=torch.int32).reshape(tokens.shape)
        rows, block = tokens.shape
        valid = torch.as_tensor(num_valid_drafts, dtype=torch.int32).flatten()
        live = positions[:, 0] >= 0
        width = 1 + int(valid[live].max()) if bool(live.any()) else 1
        counts = kwargs.get("accepted_counts")
        if kwargs.get("reload_sampling_params", False) and sampling_params is not None:
            gen.configure_sampling(
                temperature=gen._sampling_rows(sampling_params.temperature, rows)
                + gen.sampling_values["temperature"][rows:],
                top_k=gen._sampling_rows(sampling_params.top_k, rows) + gen.sampling_values["top_k"][rows:],
                top_p=gen._sampling_rows(sampling_params.top_p, rows) + gen.sampling_values["top_p"][rows:],
                seed=gen._sampling_rows(getattr(sampling_params, "seed", None), rows) + gen.request_seeds[rows:],
                presence_penalty=gen._sampling_rows(getattr(sampling_params, "presence_penalty", 0.0), rows)
                + gen.sampling_values["presence_penalty"][rows:],
                frequency_penalty=gen._sampling_rows(getattr(sampling_params, "frequency_penalty", 0.0), rows)
                + gen.sampling_values["frequency_penalty"][rows:],
                repetition_penalty=gen._sampling_rows(getattr(sampling_params, "repetition_penalty", 1.0), rows)
                + gen.sampling_values["repetition_penalty"][rows:],
                reset_rng=False,
            )
        if kwargs.get("slot_remap") is not None:
            counts = gen.remap_slots(kwargs["slot_remap"], counts)
        table = self._pages(page_table)
        if width == 1:
            # Nothing drafted but an unresolved multi-token commit: the ordinary decode with slot selection.
            full_tokens = torch.nn.functional.pad(tokens[:, :1], (0, 0, 0, self.batch_size - rows))
            full_pos = torch.nn.functional.pad(positions[:, 0], (0, self.batch_size - rows), value=-1)
            full_table = gen.page_host.clone()
            full_table[:rows] = table[:rows]
            ids = gen.decode_forward(
                full_tokens,
                full_pos,
                page_table=full_table,
                kv_cache=kv_cache,
                sample_on_device=True,
                logical_batch=rows,
                reload_inputs=True,
                accepted_counts=counts,
            )[:rows]
        else:
            cols = torch.arange(width)[None]
            pos = torch.where((cols <= valid[:, None]) & live[:, None], positions[:, :width], -1)
            ids = gen.verify_forward(tokens[:, :width], pos, table, accepted_counts=counts, sample=True)
        out = torch.zeros((rows, block), dtype=torch.int64)
        out[:, : ids.shape[1]] = ids
        self.active_decode_rows = int(live.sum())
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=out, hidden=None)

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        gen = self.generator
        committed = torch.as_tensor(committed_tokens, dtype=torch.int32)
        positions = torch.as_tensor(committed_positions, dtype=torch.int32)
        counts = torch.as_tensor(accepted_counts, dtype=torch.int32).flatten()
        rows = committed.shape[0]
        live = positions[:, 0] >= 0
        kind, bucket, width = gen.last_forward
        steps = min(int(num_drafts), 32 // bucket - 1) if int(live.sum()) <= 16 else 0
        drafts = gen.propose(committed, positions, counts, steps)
        ids = torch.full((rows, int(num_drafts)), -1, dtype=torch.int32)
        ids[:, :steps] = drafts[:, :steps].to(torch.int32)
        num_valid = torch.where(live, steps, 0).to(torch.int32)
        return DraftOutput(draft_token_ids=ids, num_valid=num_valid)

    def read_decode_output(self, output, async_read=False):
        if isinstance(output, torch.Tensor):  # perf/spec: verify ids are already host tensors
            return (output, []) if async_read else output
        # Queue the transfer before another trace can overwrite the output.
        # Each cpu() owns separate host storage until its read event completes.
        tensor = ttnn.get_device_tensors(output.tensor)[0] if output.is_tokens else output.tensor
        host = DecodeOutput(tensor.cpu(blocking=not async_read), output.is_tokens, output.rows, output.phase_event)
        event = ttnn.record_event(self.mesh, 0) if async_read else None
        if self.phase_timing:
            self.phase_timing.read_decode(output.phase_event, event)
        return (host, [event]) if async_read else host

    def process_decode_output_host(self, output, is_tokens=False):
        if isinstance(output, torch.Tensor):
            return output
        phase_host_started = time.perf_counter() if self.phase_timing else None
        if output.is_tokens != is_tokens:
            raise ValueError("Submission/readback sampling mode mismatch")
        # The non-overlapped plugin finalizes device output directly. The
        # overlapped path has already queued and synchronized this same read.
        if ttnn.is_tensor_storage_on_device(output.tensor):
            output = self.read_decode_output(output)
        rows = self.batch_size if output.rows is None else output.rows
        if is_tokens:
            values = ttnn.to_torch(output.tensor).flatten()[:rows].to(torch.int64).reshape(-1, 1)
            if bool(((values < 0) | (values >= self.model.config.vocab_size)).any()):
                raise RuntimeError(f"Device sampler emitted out-of-vocabulary IDs: {values.tolist()}")
        else:
            values = ttnn.to_torch(output.tensor, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))[
                0, 0, :rows, None, :
            ].float()
        if self.phase_timing:
            self.phase_timing.complete_decode(output.phase_event, phase_host_started)
        return values

    def release_request(self, slot):
        self.generator.release_request(slot)
        if getattr(self.generator, "mtp_pending_pos", None):
            self.generator.mtp_pending_pos.pop(slot, None)
        if self.phase_timing:
            self.phase_timing.release(slot)

    def release_persistent_capture(self):
        if self.generator is not None:
            ttnn.synchronize_device(self.mesh)
            self.generator.close()

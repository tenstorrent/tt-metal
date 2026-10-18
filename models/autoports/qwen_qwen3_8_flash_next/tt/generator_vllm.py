# SPDX-License-Identifier: Apache-2.0
"""vLLM interface translation for the canonical Flash-Next generator.

The plugin owns attention pool allocation and scheduler reload decisions.
Sampling, trace preparation, token feedback and positions belong to generator.py.
"""

import json
import os
from pathlib import Path

import torch

import ttnn

from .generator import QwenFlashNextGenerator
from .model import QwenFlashNextModel


class QwenFlashNextForCausalLM:
    decode_input_update_contract = 1
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,  # Tracked adapter replay/control proof in readiness_vllm.
        "supports_sample_on_device": True,
        "supports_intermediate_prefill_device_sampling": True,
        "supports_device_penalties": False,
        "max_device_top_k": 32,
        "supports_chunked_prefill": True,
    }

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        if tt_data_parallel != 1 or mesh_device.get_num_devices() != 4:
            raise ValueError("Flash-Next requires the qualified TP4 mesh")
        if optimizations is not None:
            raise ValueError("Flash-Next uses its selected datatype-sweep policy")
        obj = cls()
        obj.mesh = mesh_device
        obj.batch_size = max_batch_size
        obj.context = max_seq_len
        obj.checkpoint = os.environ.get("QWEN_FLASH_CHECKPOINT", hf_config._name_or_path)
        if not Path(obj.checkpoint).is_dir():
            from huggingface_hub import snapshot_download

            obj.checkpoint = snapshot_download(
                obj.checkpoint,
                revision="de4b8e4d43b917e7706784d8bb445c9af86a3540",
                local_files_only=True,
            )
        obj.hf_config = hf_config
        obj.layer_indices = None
        reduced = os.environ.get("QWEN_FLASH_SERVING_LAYERS")
        if reduced:
            obj.layer_indices = [int(i) for i in reduced.split(",")]
        elif n_layers is not None:
            obj.layer_indices = list(range(n_layers))
        obj.allow_host_sampling = os.environ.get("QWEN_FLASH_ALLOW_HOST_SAMPLING", "0") == "1"
        obj.slot_map = list(range(max_batch_size))
        obj.generator = None
        print("QWEN_FLASH_ADAPTER", str(Path(__file__).resolve()), flush=True)
        return obj

    @classmethod
    def get_max_tokens_all_users(cls, *, max_model_len, **kwargs):
        # The plugin adds one output block per admitted request.
        return max_model_len

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        cfg = vllm_config.model_config.hf_text_config
        # QSA's opaque physical page owns V/K FP8 (512 B/token), raw index
        # BF16 (256 B/token), and pooled index BF16 (64 B/token). The common
        # full-attention allocator accounts for the opaque832-byte state. GDN/PLE
        # state and the fixed dense scoring workspace are not paged KV.
        return {
            f"model.layers.{i}.self_attn": FullAttentionSpec(
                block_size=32,
                num_kv_heads=1,
                head_size=256,
                dtype=torch.float8_e4m3fn,
                state_content_bytes=832,
            )
            for i in range(sum(kind != "linear_attention" for kind in cfg.layer_types))
        }

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        if self.generator is not None:
            raise RuntimeError("Cannot replace a live serving cache")
        blocks, _, block_size, _ = kv_cache_shape
        if block_size != 32:
            raise ValueError(f"Flash-Next requires block_size=32, got {block_size}")
        if blocks * block_size < self.context:
            raise ValueError("The vLLM pool must admit one full-context request")
        from transformers import AutoTokenizer

        model = QwenFlashNextModel(
            self.checkpoint,
            self.mesh,
            batch_size=self.batch_size,
            max_seq_len=self.context,
            layer_indices=self.layer_indices,
            cache_dir=os.environ.get(
                "QWEN_FLASH_WEIGHT_CACHE",
                "/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple",
            ),
            attention_pool_tokens=blocks * block_size,
        )
        self.generator = QwenFlashNextGenerator(
            model,
            AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True),
            prefill_trace_max=512 if self.batch_size == 1 else 8,
            trace_cache_reset=False,
            record_history=False,
        )
        self.kv_cache = model.cache
        self.phase_observer = None
        return self.kv_cache

    def begin_execution_phase(self, model_input, scheduler_output, start_ns):
        observer = getattr(self, "phase_observer", None)
        return observer.begin(model_input, scheduler_output, start_ns) if observer is not None else None

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device):
        self._prepare(kv_cache)

    def warmup_model_decode(self, kv_cache, enable_trace, **kwargs):
        self._prepare(kv_cache)

    def release_persistent_capture(self):
        """The plugin calls this while the mesh is still open at shutdown."""
        if self.generator is not None:
            ttnn.synchronize_device(self.mesh)
            self.generator.event("shutdown_counters", counters=dict(self.generator.counters))
            self.generator.close()
            destination = os.environ.get("QWEN_FLASH_COUNTER_REPORT")
            if destination:
                Path(destination).write_text(
                    json.dumps(
                        dict(
                            adapter=str(Path(__file__).resolve()),
                            batch=self.batch_size,
                            context=self.context,
                            counters=dict(self.generator.counters),
                        ),
                        indent=2,
                    )
                    + "\n"
                )

    def _prepare(self, kv_cache):
        g = self.generator
        if g.prepared:
            g.prepare(kv_cache=kv_cache)
            return
        report = {}
        if os.environ.get("QWEN_FLASH_AUDIT_POLICY") == "1":
            from ..tests.datatype_runtime_audit import RuntimePolicyAudit

            with RuntimePolicyAudit(g.model) as audit:
                g.prepare(kv_cache=kv_cache)
            report["captured_native_arguments"] = audit.report()
        else:
            g.prepare(kv_cache=kv_cache)
        if os.environ.get("QWEN_FLASH_PHASE_LOG"):
            from ..tools.benchmark_phase import PhaseObserver

            self.phase_observer = PhaseObserver(os.environ["QWEN_FLASH_PHASE_LOG"], self)
        destination = os.environ.get("QWEN_FLASH_RUNTIME_REPORT")
        if destination:
            from ttnn.tools import trace_allocation_tracker

            from ..tests.multichip_capacity_probe import view
            from .expert_weight_cache import runtime_cache_summary
            from .precision import runtime_summary

            report.update(
                adapter=str(Path(__file__).resolve()),
                pid=os.getpid(),
                expert_weight_cache=runtime_cache_summary(),
                trace_tracking=dict(
                    native_enabled=ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled(),
                    python_enabled=trace_allocation_tracker.TRACE_ALLOC_TRACKING,
                    skip_program_cache_environment=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
                    replay_wrapper_module=ttnn.execute_trace.__module__,
                    requested_validation=os.environ.get("QWEN_FLASH_BENCH_TRACE_VALIDATION", "0"),
                ),
                throw_exception_on_fallback=ttnn.CONFIG.throw_exception_on_fallback,
                runtime=runtime_summary(g.model),
                generator_runtime=dict(
                    logits_dtype=str(g.tt_logits.dtype),
                    token_ids_dtype=str(g.tt_token.dtype),
                    history_dtype=str(g.tt_history.dtype),
                    max_top_k=g.sampler.config.max_top_k,
                    prefill_view=g.prefill_view.runtime_summary() if getattr(g, "prefill_view", None) else None,
                    decode_view=g.decode_runtime_summary(),
                    parameter_dtypes={
                        name: [str(t.dtype) for t in tensors] for name, tensors in g.sample_params.items()
                    },
                ),
                batch=self.batch_size,
                context=self.context,
                public_pool_tokens=g.model.public_pool_tokens,
                scratch_page_base=g.model.scratch_page_base,
                cache_shapes=[
                    {name: list(tensor.shape) for name, tensor in row.items() if name != "ple_cache"}
                    for row in kv_cache.layers
                ],
                prepared_memory=view(self.mesh),
                trace_memory=view(self.mesh, ttnn.BufferType.TRACE),
                trace_ids=g.traces,
                counters=dict(g.counters),
                serving_buffer_sharing={
                    name: {
                        "references": len(tensors),
                        "unique_allocations": len({t.buffer_address() for t in tensors}),
                    }
                    for name, tensors in {
                        **{
                            name: [
                                getattr(layer.attention, name)
                                for layer in g.model.layers
                                if hasattr(layer.attention, name)
                            ]
                            for name in (
                                "zero_recurrent",
                                "zero_conv",
                                "cos",
                                "sin",
                                "recurrent_state",
                                "conv_state",
                                "recurrent_bank",
                                "conv_bank",
                                "packed_cache",
                                "raw_key_cache",
                                "pooled_cache",
                                "pooled_keys",
                            )
                        },
                        "moe_capacity": [layer.mlp.capacity for layer in g.model.layers],
                    }.items()
                },
            )
            Path(destination).write_text(json.dumps(report, indent=2, default=str) + "\n")

    def _pages(self, table):
        table = torch.as_tensor(table, dtype=torch.int32)
        shape = (self.batch_size, self.context // 32)
        if table.ndim != 2 or table.shape[0] > shape[0] or table.shape[1] > shape[1]:
            raise ValueError(f"Invalid serving page table {tuple(table.shape)}")
        if tuple(table.shape) == shape:
            # The generator owns a clone before retaining or scratch-remapping.
            return table
        target = torch.zeros(shape, dtype=torch.int32)
        target[: table.shape[0], : table.shape[1]] = table
        return target

    def _sampling(self, params):
        if params is None:
            if not self.allow_host_sampling:
                raise ValueError("Host sampling requires explicit QWEN_FLASH_ALLOW_HOST_SAMPLING=1")
            return "host"
        return self.generator.set_serving_sampling_params(params)

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        empty_slots=None,
        start_pos=None,
        sampling_params=None,
        enable_trace=True,
        **kwargs,
    ):
        g = self.generator
        rows = len(prompt_lens)
        starts = torch.zeros(self.batch_size, dtype=torch.int32)
        if start_pos is not None:
            starts[:rows] = torch.as_tensor(start_pos).flatten()[:rows]
        # The pinned plugin passes full-prefix tokens and absolute chunk ends.
        # The native API takes only the scheduled suffix and its logical length.
        ends = [int(n) for n in prompt_lens]
        if not 0 < rows <= self.batch_size or tokens.shape[0] < rows:
            raise ValueError("Invalid serving prefill row count")
        if any(not 0 <= int(starts[i]) < end <= tokens.shape[1] for i, end in enumerate(ends)):
            raise ValueError("Prefill requires 0 <= start < end <= supplied prefix length")
        lengths = [end - int(starts[i]) for i, end in enumerate(ends)]
        padded = torch.zeros(self.batch_size, max(lengths), dtype=torch.int32)
        for i, length in enumerate(lengths):
            padded[i, :length] = tokens[i, int(starts[i]) : ends[i]]
        lengths += [0] * (self.batch_size - rows)
        logical_slots = list(range(rows)) if empty_slots is None else list(empty_slots)
        physical = [self.slot_map[s] for s in logical_slots]
        physical += [s for s in range(self.batch_size) if s not in physical]
        mode = self._sampling(sampling_params)
        g.sampling_mode = "host" if mode == "host" else "device"
        if getattr(g, "event_path", None):
            g.event(
                "prefill_schedule",
                starts=starts[:rows].tolist(),
                ends=ends,
                suffix_lengths=lengths[:rows],
                physical_slots=physical[:rows],
                sampling=mode,
            )
        result = g.prefill_forward(
            padded,
            page_table=self._pages(page_table),
            kv_cache=kv_cache,
            prompt_lens=lengths,
            start_pos=starts,
            slots=physical,
            sampling="greedy" if mode == "host" else mode,
        )[:rows]
        observer = getattr(self, "phase_observer", None)
        if observer is not None and getattr(g, "prefill_view", None) is not None:
            observer.record_prefill_chunks(g.prefill_view.last_physical_chunks)
        if getattr(g, "event_path", None):
            from ..tests.multichip_capacity_probe import view

            g.event(
                "prefill_complete",
                sampling=mode,
                rows=rows,
                counters=dict(g.counters),
                memory=view(self.mesh),
                trace_memory=view(self.mesh, ttnn.BufferType.TRACE),
            )
        # Text-only Qwen's canonical M-RoPE has identical axes and zero delta.
        return result, torch.zeros(rows, dtype=torch.long)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        sampling_params=None,
        enable_trace=True,
        read_from_device=True,
        slot_remap=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        **kwargs,
    ):
        if "reset_batch" in kwargs:
            raise ValueError("Use decode_input_update_contract=1 commands")
        deltas = kwargs.get("rope_deltas_all_users")
        if deltas is not None and any(int(d) != 0 for d in deltas):
            raise ValueError("Flash-Next autoport serves text-only zero-delta RoPE")
        if not enable_trace:
            raise ValueError("Flash-Next serving requires decode tracing")
        if reload_inputs and reload_page_table:
            raise ValueError("Page-only and complete reload commands are exclusive")
        if reset_sampling_state and not reload_inputs:
            raise ValueError("Sampling reset requires authoritative inputs")
        g = self.generator
        slot_map = self.slot_map
        if slot_remap is not None:
            permutation = torch.as_tensor(slot_remap).flatten().tolist()
            if sorted(permutation) != list(range(self.batch_size)):
                raise ValueError("State slot remap must be a complete permutation")
            slot_map = [self.slot_map[i] for i in permutation]
        if sampling_params is None or reload_sampling_params or not hasattr(self, "sampling"):
            self.sampling = self._sampling(sampling_params)
        mode = self.sampling
        g.sampling_mode = "host" if mode == "host" else "device"
        positions = torch.as_tensor(start_pos, dtype=torch.int32).flatten() if reload_inputs else None
        active = (positions >= 0).tolist() if reload_inputs else None
        out = g.decode_forward(
            tokens if reload_inputs else None,
            positions.clamp_min(0) if reload_inputs else None,
            page_table=self._pages(page_table) if reload_inputs or reload_page_table else None,
            kv_cache=kv_cache,
            slots=slot_map if reload_inputs or slot_remap is not None else None,
            active_rows=active,
            sampling="greedy" if mode == "host" else mode,
            read_tokens=mode == "host",
        )
        observer = getattr(self, "phase_observer", None)
        if observer is not None:
            observer.record_decode_execution(g.last_decode_execution_batch, active_rows=g._active)
        if getattr(g, "event_path", None):
            g.event(
                "decode_submit",
                sampling=mode,
                reload_inputs=reload_inputs,
                reload_page_table=reload_page_table,
                reload_sampling_params=reload_sampling_params,
                slots=slot_map,
                active=g._active,
                execution_batch=g.last_decode_execution_batch,
                counters=dict(g.counters),
            )
        self.slot_map = slot_map
        if mode == "host":
            return out[:, None, :]
        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(out), is_tokens=True)
        return out

    def read_decode_output(self, tt_out, async_read=False):
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        host = ttnn.get_device_tensors(tt_out)[0].cpu(blocking=not async_read)
        self.generator.counters["token_readbacks"] += 1
        return (host, [ttnn.record_event(self.mesh, 0)]) if async_read else host

    def process_decode_output_host(self, output, is_tokens=False):
        if isinstance(output, torch.Tensor):
            return output
        if not is_tokens:
            raise ValueError("Device output is token-only; host compatibility returns explicit host logits")
        # Synchronous plugin submissions can arrive as replicated device
        # tensors; deferred submissions already hold a single host shard.
        shard = ttnn.get_device_tensors(output)[0]
        return ttnn.to_torch(shard).reshape(self.batch_size, 1).long()

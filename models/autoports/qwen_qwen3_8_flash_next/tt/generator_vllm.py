# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM translation for the canonical Qwen generator.

Register with TT_MODEL_CLASS_OVERRIDES; cache allocation and every device
operation remain owned by the full-model generator. QWEN_VLLM_LAYER_INDICES
selects a reduced real-layer debugging target, never acceptance evidence.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from .generator import QwenGenerator, build_generator


def _layer_indices(config):
    selected = os.environ.get("QWEN_VLLM_LAYER_INDICES")
    return tuple(map(int, selected.split(","))) if selected else tuple(range(config.num_hidden_layers))


class QwenFlashNextForCausalLM:
    decode_input_update_contract = 1
    # Queued adapter transitions preserve tokens and complete device state;
    # matched-width serving also agrees with synchronous execution.
    model_capabilities = dict(
        QwenGenerator.model_capabilities, supports_async_decode=True, supports_prefill_sample_mask=True
    )
    get_max_tokens_all_users = staticmethod(QwenGenerator.get_max_tokens_all_users)

    def __init__(self, generator):
        self.generator = generator
        self._trace_diagnostics = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
        generator.event_sink = self._log_event
        self._benchmark = None
        self._benchmark_tickets = {}
        self._benchmark_requests = {}
        self._benchmark_pages = {}
        self._benchmark_generation = 0
        self._benchmark_chunks = None
        phase_path = os.environ.get("QWEN_BENCHMARK_PHASE_LOG")
        if phase_path:
            from .benchmark_phases import PhaseRecorder

            self._benchmark = PhaseRecorder(generator.mesh, phase_path)
            Path(phase_path + ".identity.json").write_text(
                json.dumps(
                    dict(
                        module_file=str(Path(__file__).resolve()),
                        generator_file=str(Path(__file__).with_name("generator.py").resolve()),
                        pid=os.getpid(),
                        layers=list(generator.model.layer_indices),
                        precision=generator.model.precision_summary(),
                        devices=list(map(int, os.environ["TT_VISIBLE_DEVICES"].split(","))),
                        mesh_shape=list(generator.mesh.shape),
                        capacity=generator.capacity,
                        slots=generator.max_batch_size,
                    ),
                    indent=2,
                )
                + "\n"
            )
        self.tt_supported_decode_batch_sizes = tuple(n for n in (1, 2, 4, 8, 16, 32) if n <= generator.max_batch_size)
        if generator.max_batch_size not in self.tt_supported_decode_batch_sizes:
            self.tt_supported_decode_batch_sizes += (generator.max_batch_size,)

    def __getattr__(self, name):
        return getattr(self.generator, name)

    def _log_event(self, event):
        if self._benchmark_chunks is not None and event.get("kind") == "prefill_chunk":
            self._benchmark_chunks.append(dict(event))
        print("QWEN_VLLM_EVENT " + json.dumps({"time": time.monotonic(), **event}, separators=(",", ":")), flush=True)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1:
            raise ValueError("Qwen serving uses one TP4/EP4 mesh")
        config = getattr(hf_config, "text_config", hf_config)
        generator = build_generator(
            Path(__file__).resolve().parents[1],
            mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            layer_indices=_layer_indices(config),
            defer_kv_cache=True,
            hf_config=hf_config,
        )
        print(
            "QWEN_VLLM_CONSTRUCTION "
            + json.dumps(
                dict(
                    adapter=f"{cls.__module__}:{cls.__name__}",
                    pid=os.getpid(),
                    precision=generator.model.precision,
                    layers=generator.model.layer_indices,
                    max_seq_len=max_seq_len,
                    max_batch_size=max_batch_size,
                )
            ),
            flush=True,
        )
        return cls(generator)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        config = getattr(vllm_config.model_config.hf_config, "text_config", vllm_config.model_config.hf_config)
        if vllm_config.cache_config.block_size != 32:
            raise ValueError("The selected Qwen KV layout requires page32")
        return {
            f"model.layers.{index}.self_attn": FullAttentionSpec(
                block_size=32,
                num_kv_heads=config.num_key_value_heads,
                head_size=config.head_dim,
                dtype=torch.bfloat16,
            )
            for index in range(config.num_hidden_layers)
            if config.layer_types[index] in ("full_attention", "indexed_attention")
        }

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        attention_layers = [
            i
            for i, kind in enumerate(self.generator.model.config.layer_types)
            if kind in ("full_attention", "indexed_attention")
        ]
        if len(per_layer_specs) != len(attention_layers):
            raise ValueError("vLLM must supply one pool spec per QSA attention layer")
        selected = [
            spec
            for index, spec in zip(attention_layers, per_layer_specs, strict=True)
            if index in self.generator.model.layer_indices
        ]
        return self.generator.allocate_kv_cache_per_layer(selected)

    def prefill_forward(self, *args, **kwargs):
        slots = tuple(kwargs.get("empty_slots") or range(len(kwargs["prompt_lens"])))
        sample_mask = list(map(bool, kwargs.get("prefill_sample_mask", [True] * len(slots))))
        ticket = None
        if self._benchmark:
            starts = list(map(int, kwargs.get("start_pos", [0] * len(slots))))
            ends = list(map(int, kwargs["prompt_lens"]))
            tokens = kwargs.get("tokens", args[0] if args else None)
            requests = []
            for row, (slot, start, end) in enumerate(zip(slots, starts, ends, strict=True)):
                page_key = int(kwargs["page_table"][row, 0])
                if start == 0:
                    self._benchmark_generation += 1
                    self._benchmark_requests[slot] = dict(
                        request=self._benchmark_generation,
                        first_submission_ns=time.monotonic_ns(),
                        prompt_sha256=hashlib.sha256(tokens[row].numpy().tobytes()).hexdigest(),
                        first_page=page_key,
                    )
                    self._benchmark_pages[page_key] = self._benchmark_requests[slot]
                elif page_key not in self._benchmark_pages:
                    raise RuntimeError("Benchmark prefill has no request identity")
                else:
                    self._benchmark_requests[slot] = self._benchmark_pages[page_key]
                requests.append(dict(self._benchmark_requests[slot], slot=slot, start=start, end=end))
            self._benchmark_chunks = []
            ticket = self._benchmark.begin(
                "prefill", dict(requests=requests, sample_mask=sample_mask, chunks=self._benchmark_chunks)
            )
        self._log_event(
            dict(
                kind="prefill_submission",
                starts=list(map(int, kwargs.get("start_pos", [0] * len(slots)))),
                ends=list(map(int, kwargs["prompt_lens"])),
                slots=slots,
                sample_mask=sample_mask,
                active_decode_slots=[
                    i for i, cache in enumerate(self.generator.caches) if cache.position and i not in slots
                ],
                sampling_mode=(
                    "none"
                    if not any(sample_mask)
                    else "device"
                    if kwargs.get("sampling_params") is not None
                    else "host"
                ),
            )
        )
        before = dict(self.generator.counters)
        output = self.generator.prefill_forward(*args, **kwargs)
        self._log_event(
            dict(
                kind="prefill_result",
                sample_mask=sample_mask,
                output_present=output is not None,
                counters={key: value - before[key] for key, value in self.generator.counters.items()},
            )
        )
        if output is not None and kwargs.get("sampling_params") is not None:
            self._check_tokens(output)
        if ticket is not None:
            self._benchmark.annotate(ticket, chunks=self._benchmark_chunks)
            self._benchmark.end(ticket)
            if output is not None:
                self._benchmark.host_complete(ticket)
            self._benchmark_chunks = None
        # Text-only MRoPE advances all axes together; no image/video offset.
        return output, torch.zeros(len(slots), dtype=torch.int64)

    def decode_forward(self, *args, **kwargs):
        # The plugin expresses sampling choice by presence of sampling_params.
        # Explicit host compatibility uses the generator's logits boundary.
        kwargs["sample_on_device"] = kwargs.get("sampling_params") is not None
        event = dict(
            kind="decode_submission",
            sampling_mode="device" if kwargs["sample_on_device"] else "host",
            reload_inputs=kwargs.get("reload_inputs", True),
            reload_page_table=kwargs.get("reload_page_table", False),
        )
        if self._trace_diagnostics:
            submitted_positions = list(map(int, kwargs.get("start_pos", args[1] if len(args) > 1 else None)))
            submitted_slots = kwargs.get("slots")
            if submitted_slots is None:
                submitted_slots = tuple(range(len(submitted_positions)))
            event.update(
                positions=[cache.position for cache in self.generator.caches],
                submitted_positions=submitted_positions,
                active_slots=[slot for slot, position in zip(submitted_slots, submitted_positions) if position >= 0],
                submitted_page_table=(
                    kwargs["page_table"].tolist()
                    if (kwargs.get("reload_page_table", False) or kwargs.get("reload_inputs", True))
                    and isinstance(kwargs.get("page_table"), torch.Tensor)
                    else None
                ),
            )
        self._log_event(event)
        ticket = None
        if self._benchmark:
            remap = kwargs.get("slot_remap")
            if remap is not None:
                remap = tuple(map(int, remap))
                previous = self._benchmark_requests
                self._benchmark_requests = {dst: previous[src] for dst, src in enumerate(remap) if src in previous}
            positions = list(map(int, kwargs.get("start_pos", args[1] if len(args) > 1 else [])))
            slots = kwargs.get("slots")
            slots = tuple(range(len(positions))) if slots is None else tuple(slots)
            # Resident feedback advances positions without authoritative reloads.
            if not kwargs.get("reload_inputs", True):
                positions = [
                    self.generator.caches[remap[s] if remap is not None else s].position if p >= 0 else p
                    for s, p in zip(slots, positions)
                ]
            requests = [
                dict(self._benchmark_requests.get(s, {}), slot=s, position=p)
                for s, p in zip(slots, positions)
                if p >= 0
            ]
            ticket = self._benchmark.begin("decode", dict(requests=requests))
        output = self.generator.decode_forward(*args, **kwargs)
        if ticket is not None:
            if isinstance(output, torch.Tensor):
                self._benchmark.end(ticket)
                self._benchmark.host_complete(ticket)
            else:
                self._benchmark_tickets[id(output)] = ticket
        return output

    def read_decode_output(self, output, async_read=False, **kwargs):
        result = self.generator.read_decode_output(output, async_read=async_read, **kwargs)
        if self._benchmark:
            ticket = self._benchmark_tickets.pop(id(output), None)
            if ticket is None:
                return result
            host, events = result if async_read else (result, [])
            self._benchmark.end(ticket, event=events[-1] if events else None)
            self._benchmark_tickets[id(host)] = ticket
        return result

    def _check_tokens(self, tokens):
        invalid = int(((tokens < 0) & (tokens != -1) | (tokens >= self.generator.model.config.vocab_size)).sum())
        self._log_event(
            dict(
                kind="sampled_tokens",
                out_of_vocabulary=invalid,
                count=int((tokens >= 0).sum()),
                **({"token_ids": tokens.reshape(-1).tolist()} if self._trace_diagnostics else {}),
            )
        )
        if invalid:
            raise RuntimeError("Device sampler returned an out-of-vocabulary token")

    def process_decode_output_host(self, output, **kwargs):
        tokens = self.generator.process_decode_output_host(output, **kwargs)
        if kwargs.get("is_tokens", getattr(output, "is_tokens", False)):
            self._check_tokens(tokens)
        if self._benchmark:
            ticket = self._benchmark_tickets.pop(id(output))
            self._benchmark.host_complete(ticket)
        return tokens

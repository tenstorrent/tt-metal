# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM translation over Ornith's canonical generator and external KV state."""
import json
import logging
import os
import time

import torch

from .generator import OrnithGenerator
from .model import OrnithModel
from .precision import load_precision

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


class OrnithForCausalLM:
    model_capabilities = dict(OrnithGenerator.model_capabilities, preserve_prefill_state_slots=True)
    decode_input_update_contract = 1

    @classmethod
    def get_max_tokens_all_users(cls, model_name, num_devices, tt_data_parallel, max_model_len, max_num_seqs):
        return OrnithGenerator.get_max_tokens_all_users(
            model_name, num_devices, tt_data_parallel, max_model_len, max_num_seqs
        )

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        config = vllm_config.model_config.hf_config
        config = getattr(config, "text_config", config)
        if vllm_config.cache_config.block_size != 32:
            raise ValueError("Ornith's selected cache requires 32-token blocks")
        if load_precision()["kv_cache_dtype"] != "bfloat16":
            raise ValueError("Serving spec must match the selected BF16 KV policy")
        return {
            f"model.layers.{i}.self_attn": FullAttentionSpec(
                block_size=32, num_kv_heads=2, head_size=256, dtype=torch.bfloat16
            )
            for i, kind in enumerate(config.layer_types)
            if kind == "full_attention"
        }

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=262144, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1:
            raise ValueError("Ornith uses one TP4/EP4 mesh")
        self = cls()
        self.model = OrnithModel.from_pretrained(mesh_device, rope_context=max_seq_len)
        self.batch_size, self.context = int(max_batch_size), int(max_seq_len)
        self.generator = None
        self.tt_supported_decode_batch_sizes = [self.batch_size]
        logger.info(
            "ORNITH_ADAPTER %s:%s pid=%d layers=%s precision=%s",
            cls.__module__,
            cls.__name__,
            os.getpid(),
            self.model.layer_indices,
            json.dumps(self.model.precision),
        )
        return self

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        if self.generator is not None:
            raise RuntimeError("Serving KV storage may only be allocated once")
        full_indices = [i for i, kind in enumerate(self.model.config.layer_types) if kind == "full_attention"]
        if len(per_layer_specs) != len(full_indices):
            raise ValueError("vLLM must allocate each full-attention layer")
        shape = tuple(per_layer_specs[0][0])
        if shape[1:] != (1, 32, 256):
            raise ValueError(f"Unexpected selected KV geometry: {shape}")
        if any(tuple(s) != shape or d != torch.bfloat16 for s, d, _ in per_layer_specs):
            raise ValueError("All full-attention pools must use the selected uniform BF16 geometry")
        if len({i for _, _, i in per_layer_specs}) != len(per_layer_specs):
            raise ValueError("Independent attention layers cannot alias their cache")
        state = self.model.create_state(
            context=self.context, batch_size=self.batch_size, shared_cache=True, num_blocks=shape[0]
        )
        self.generator = OrnithGenerator(
            self.model, batch_size=self.batch_size, max_seq_len=self.context, cache_state=state, num_blocks=shape[0]
        )
        logger.info("ORNITH_KV external=true shape=%s layers=%s", shape, self.model.layer_indices)
        return state

    def __getattr__(self, name):
        # Lifecycle/readback hooks retain their generator implementation.
        generator = self.__dict__.get("generator")
        if generator is None:
            raise AttributeError(name)
        return getattr(generator, name)

    def _pages(self, pages, slots=None):
        g = self.generator
        pages = torch.as_tensor(pages, dtype=torch.int32)
        if pages.ndim != 2 or pages.shape[1] > g.pages_host.shape[1]:
            raise ValueError("Invalid scheduler block table")
        result = g.pages_host.clone() if slots is not None else torch.zeros_like(g.pages_host)
        destinations = range(pages.shape[0]) if slots is None else slots
        for row, slot in enumerate(destinations):
            result[slot].zero_()
            result[slot, : pages.shape[1]] = pages[row]
        return result

    def prefill_forward(self, tokens, *, page_table, kv_cache, prompt_lens, empty_slots=None, **kwargs):
        # vLLM forwards empty multimodal metadata even for text-only requests.
        for name in (
            "pixel_values",
            "image_grid_thw",
            "mm_identifiers",
            "mm_placeholder_lengths",
            "mm_placeholder_offsets",
        ):
            value = kwargs.pop(name, None)
            if value is not None and (not isinstance(value, (list, tuple)) or any(x is not None for x in value)):
                raise ValueError(f"Ornith text serving does not accept {name}")
        slots = list(range(len(prompt_lens))) if empty_slots is None else list(empty_slots)
        pages = self._pages(page_table, slots)
        emit = kwargs.get("prefill_sampling_mask")
        sampling_mode = (
            "none"
            if emit is not None and not any(emit)
            else "device"
            if kwargs.get("sampling_params") is not None
            else "host"
        )
        logger.info("sampling_mode=%s phase=prefill", sampling_mode)
        output = self.generator.prefill_forward(
            tokens, page_table=pages, kv_cache=kv_cache, prompt_lens=prompt_lens, empty_slots=slots, **kwargs
        )

        # Text-only Qwen3.5 mRoPE has zero delta, as in the HF implementation.
        return output, torch.zeros(len(prompt_lens), dtype=torch.long)

    def decode_forward(
        self, tokens, start_pos, *, page_table, kv_cache, sampling_params=None, rope_deltas_all_users=None, **kwargs
    ):
        if rope_deltas_all_users is not None and any(int(x) != 0 for x in rope_deltas_all_users):
            raise ValueError("Text-only Ornith requires zero RoPE deltas")
        # Retain the required fallback audit without detailed trace telemetry.
        logger.info("sampling_mode=%s phase=decode", "device" if sampling_params is not None else "host")
        # No table reconstruction or token/position access on resident steps.
        if kwargs.get("reload_inputs", True) or kwargs.get("reload_page_table", False):
            page_table = self._pages(page_table)
        output = self.generator.decode_forward(
            tokens, start_pos, page_table=page_table, kv_cache=kv_cache, sampling_params=sampling_params, **kwargs
        )
        return output

    def process_decode_output_host(self, output, *, is_tokens=False, **kwargs):
        # Synchronous plugin finalization may hand us an unread output handle.
        # Async output already owns a queued host snapshot; never read it twice.
        if output.host is None:
            self.generator.read_decode_output(output, async_read=False)
        result = self.generator.process_decode_output_host(output, is_tokens=is_tokens, **kwargs)
        if is_tokens and ((result < 0).any() or (result >= self.model.config.vocab_size).any()):
            raise RuntimeError("Device sampler returned an out-of-vocabulary token")
        return result

    def warmup_model_prefill(self, **kwargs):
        started = time.monotonic()
        self.generator.warmup_model_prefill(**kwargs)
        logger.info(
            "ORNITH_PREPARE seconds=%f traces=%s counters=%s",
            time.monotonic() - started,
            self.generator.traces,
            dict(self.generator.counters),
        )

    def warmup_model_decode(self, **kwargs):
        self.generator.warmup_model_decode(**kwargs)
        logger.info("ORNITH_PREPARED traces=%s counters=%s", self.generator.traces, dict(self.generator.counters))

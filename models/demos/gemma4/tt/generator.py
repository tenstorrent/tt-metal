# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

import ttnn
from models.demos.gemma4.tt.common import create_tt_model

if not hasattr(transformers, "AutoModelForVision2Seq"):

    class _AutoModelForVision2SeqUnavailable:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("AutoModelForVision2Seq is unavailable in this transformers build")

    transformers.AutoModelForVision2Seq = _AutoModelForVision2SeqUnavailable

from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import determine_device_name


def _replicate_to_mesh(mesh_device):
    if hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1:
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return None


def _prepare_inputs_prefill(
    self,
    tokens,
    start_pos=0,
    page_table=None,
    chunk_page_table=None,
    trace_enabled=False,
    last_token_idx=None,
    global_user_id=None,
    batch_size=1,
    user_id=0,
    batched_prefill=False,
    **kwargs,
):
    del start_pos, last_token_idx, global_user_id, batch_size, user_id, batched_prefill, kwargs

    device = None if trace_enabled else self.mesh_device
    mesh_mapper = _replicate_to_mesh(self.mesh_device)

    tokens_torch = tokens.to(torch.long)
    tt_tokens = ttnn.from_torch(
        tokens,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
    )

    tt_page_table = None
    if page_table is not None:
        tt_page_table = ttnn.from_torch(
            page_table,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )

    tt_chunk_page_table = None
    if chunk_page_table is not None:
        tt_chunk_page_table = ttnn.from_torch(
            chunk_page_table,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mesh_mapper,
        )

    self._gemma_prefill_input_ids_torch = tokens_torch
    if self._embed_weight_cpu is not None:
        self._gemma_prefill_embeds_torch = F.embedding(tokens_torch, self._embed_weight_cpu).float() * self.embed_scale
    else:
        self._gemma_prefill_embeds_torch = None

    if trace_enabled:
        return tt_tokens, None, None, tt_page_table, tt_chunk_page_table

    tt_embeds = self.embed_tokens(tt_tokens)
    if len(tt_embeds.shape) == 3:
        tt_embeds = ttnn.unsqueeze_to_4D(tt_embeds)
    tt_embeds = ttnn.to_layout(tt_embeds, ttnn.TILE_LAYOUT)

    return tt_embeds, None, None, tt_page_table, tt_chunk_page_table


def _prepare_prefill_inputs_trace(self, tokens, **kwargs):
    return self.prepare_inputs_prefill(tokens, trace_enabled=True, **kwargs)


def _transform_and_embed_prefill_inputs_device(self, tokens, tt_page_table, tt_chunk_page_table):
    tt_embeds = self.embed_tokens(tokens)
    if len(tt_embeds.shape) == 3:
        tt_embeds = ttnn.unsqueeze_to_4D(tt_embeds)
    tt_embeds = ttnn.to_layout(tt_embeds, ttnn.TILE_LAYOUT)
    return tt_embeds, tt_page_table, tt_chunk_page_table


def _ttnn_prefill_forward(
    self,
    x,
    rot_mats_global=None,
    rot_mats_local=None,
    user_id=0,
    page_table=None,
    chunk_page_table=None,
    chunk_start_idx=None,
    get_last_token=-1,
    kv_cache=None,
    batch_size=1,
    **kwargs,
):
    del rot_mats_global, rot_mats_local, chunk_page_table, chunk_start_idx, kwargs
    return self._gemma_generator_prefill_impl(
        x=x,
        user_id=user_id,
        page_table=page_table,
        get_last_token=get_last_token,
        kv_cache=kv_cache,
        batch_size=batch_size,
        input_ids_torch=self._gemma_prefill_input_ids_torch,
        embeds_torch=self._gemma_prefill_embeds_torch,
    )


def _process_output_prefill(self, tt_out, last_token_idx):
    if self.mesh_config is not None and self.mesh_config.tp > 1:
        torch_output = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
    else:
        torch_output = ttnn.to_torch(tt_out)
    return torch_output[..., last_token_idx, : self.vocab_size]


def _process_logits_after_prefill_trace(self, logits, last_token_idx):
    get_last_token = (last_token_idx // 32) * 32
    return ttnn.slice(
        logits,
        (0, 0, get_last_token, 0),
        (1, 1, get_last_token + 32, logits.shape[-1]),
    )


def _process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
    if is_tokens or is_log_probs:
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            torch_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
        else:
            torch_out = ttnn.to_torch(tt_out)
        return torch_out.reshape(-1)[:B]

    if self.mesh_config is not None and self.mesh_config.tp > 1:
        torch_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
    else:
        torch_out = ttnn.to_torch(tt_out)

    return torch_out[:, :, :B, : self.vocab_size].view(B, S, -1)


def _patch_model_args(model_args, mesh_device, max_batch_size, max_seq_len, model_path, tokenizer):
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len
    model_args.max_context_len = max_seq_len
    model_args.max_prefill_chunk_size = max_seq_len
    model_args.trace_prefill_supported_seq_lens = [128, 512]
    model_args.mesh_device = mesh_device
    model_args.device_name = determine_device_name(mesh_device)
    model_args.model_name = model_path
    model_args.base_model_name = Path(model_path).name
    model_args.tokenizer = tokenizer
    model_args.processor = None
    model_args.can_enable_trace = (
        lambda prefill_seq_len, num_cached_tokens=0: num_cached_tokens == 0
        and prefill_seq_len in model_args.trace_prefill_supported_seq_lens
    )
    model_args.is_llama_vision = lambda: False
    model_args.encode_prompt = lambda prompt, instruct=False: (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if instruct and getattr(tokenizer, "chat_template", None)
        else tokenizer.encode(prompt, add_special_tokens=True)
    )


def _patch_model_instance(model):
    model._gemma_generator_prefill_impl = model.ttnn_prefill_forward
    model.prepare_inputs_prefill = MethodType(_prepare_inputs_prefill, model)
    model.prepare_prefill_inputs_trace = MethodType(_prepare_prefill_inputs_trace, model)
    model.transform_and_embed_prefill_inputs_device = MethodType(_transform_and_embed_prefill_inputs_device, model)
    model.ttnn_prefill_forward = MethodType(_ttnn_prefill_forward, model)
    model.process_logits_after_prefill_trace = MethodType(_process_logits_after_prefill_trace, model)
    model.process_output_prefill = MethodType(_process_output_prefill, model)
    model.process_output_decode = MethodType(_process_output_decode, model)
    # Gemma decode inputs are rebuilt on host every token, so traced decode
    # must refresh the captured input buffers on every replay.
    model._tt_vllm_always_refresh_decode_trace_inputs = True
    # Keep the dedicated generator bringup on the simpler host-sampling path first.
    model._supports_on_device_sampling = False
    model._gemma_prefill_input_ids_torch = None
    model._gemma_prefill_embeds_torch = None


class Gemma4Generator(Generator):
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Gemma4 decode already returns sampled tokens when on-device sampling is enabled.
        self.enable_split_sampling = False

    def _clear_prefill_traces(self):
        for trace_key, trace_id in list(self.trace_id_prefill.items()):
            if trace_id is not None:
                parts = trace_key.split("_")
                model_id = int(parts[1]) if len(parts) >= 2 else 0
                ttnn.release_trace(self.model_args[model_id].mesh_device, trace_id)
            self.trace_id_prefill[trace_key] = None
            self.trace_inputs_prefill[trace_key] = None
            self.trace_output_prefill[trace_key] = None

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, non_greedy_decoding_on_device):
        super().warmup_model_prefill(
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            non_greedy_decoding_on_device=non_greedy_decoding_on_device,
        )
        if enable_trace:
            # Gemma4 prefill depends on prompt-specific per-layer inputs.
            # Warmup traces are only for compile coverage and must not be reused
            # for a different prompt at runtime.
            self._clear_prefill_traces()

    @classmethod
    def from_pretrained(
        cls,
        mesh_device,
        model_path,
        max_batch_size=1,
        max_seq_len=4096,
        num_layers=None,
        paged_attention_config=None,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if not hasattr(tokenizer, "stop_tokens"):
            tokenizer.stop_tokens = [tokenizer.eos_token_id]

        model_args, model, tt_kv_cache, _ = create_tt_model(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            model_path=model_path,
            create_kv_cache=True,
            paged_attention_config=paged_attention_config,
        )
        _patch_model_args(
            model_args,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            model_path=model_path,
            tokenizer=tokenizer,
        )
        _patch_model_instance(model)
        generator = cls([model], [model_args], mesh_device, processor=None, tokenizer=tokenizer)
        return generator, [tt_kv_cache], tokenizer

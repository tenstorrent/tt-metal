# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType

import torch
import torch.nn.functional as F

import ttnn
from models.demos.gemma4.tt.common import create_tt_model
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.generator_vllm import allocate_vllm_kv_cache


class _Gemma4VllmOptimizations:
    @staticmethod
    def get_tensor_dtype(decoder_id, tensor, prefetcher=False):
        del decoder_id, tensor, prefetcher
        return ttnn.bfloat16


def _replicate_to_mesh(mesh_device):
    is_mesh = hasattr(mesh_device, "shape")
    if is_mesh and mesh_device.get_num_devices() > 1:
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

    self._vllm_prefill_input_ids_torch = tokens_torch
    if self._embed_weight_cpu is not None:
        self._vllm_prefill_embeds_torch = F.embedding(tokens_torch, self._embed_weight_cpu).float() * self.embed_scale
    else:
        self._vllm_prefill_embeds_torch = None

    if trace_enabled:
        return tt_tokens, None, None, tt_page_table, tt_chunk_page_table

    tt_embeds = self.embed_tokens(tt_tokens)
    if len(tt_embeds.shape) == 3:
        tt_embeds = ttnn.unsqueeze_to_4D(tt_embeds)

    tt_embeds = ttnn.to_layout(tt_embeds, ttnn.TILE_LAYOUT)

    return tt_embeds, None, None, tt_page_table, tt_chunk_page_table


def _prepare_prefill_inputs_trace(self, tokens, **kwargs):
    return self.prepare_inputs_prefill(tokens, trace_enabled=True, **kwargs)


def _prepare_inputs_decode_compat(self, tokens, current_pos, page_table=None):
    decode_inputs = self._tt_vllm_prepare_inputs_decode_impl(tokens, current_pos, page_table)
    # Compatibility with the shared TT generator: Gemma4 decode input prep can
    # return an extra auxiliary tensor, but the common VLLM decode path still
    # unpacks only the first four values.
    if isinstance(decode_inputs, tuple) and len(decode_inputs) > 4:
        return decode_inputs[:4]
    return decode_inputs


def _transform_and_embed_prefill_inputs_device(
    self,
    tokens,
    tt_page_table,
    tt_chunk_page_table,
):
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
    return self._tt_vllm_prefill_impl(
        x=x,
        user_id=user_id,
        page_table=page_table,
        get_last_token=get_last_token,
        kv_cache=kv_cache,
        batch_size=batch_size,
        input_ids_torch=self._vllm_prefill_input_ids_torch,
        embeds_torch=self._vllm_prefill_embeds_torch,
    )


def _process_output_prefill(self, tt_out, last_token_idx):
    if self.mesh_config is not None and self.mesh_config.tp > 1:
        # Gemma4 all-gathers logits internally across TP.
        # Read back a single device tensor.
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
            concat_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
        else:
            concat_out = ttnn.to_torch(tt_out)
        return concat_out.reshape(-1)[:B]

    if self.mesh_config is not None and self.mesh_config.tp > 1:
        # Gemma4 decode logits are already all-gathered across TP inside the
        # model forward, so a single device tensor contains the full vocab.
        torch_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
    else:
        torch_out = ttnn.to_torch(tt_out)

    torch_out = torch_out[:, :, :B, : self.vocab_size].view(B, S, -1)
    return torch_out


def _patch_model_instance(model, model_args):
    # The shared TT vLLM cache allocator expects each model instance to expose
    # its config through `model.args`, matching the text transformer wrappers.
    model.args = model_args
    model._tt_vllm_prefill_impl = model.ttnn_prefill_forward
    model._tt_vllm_prepare_inputs_decode_impl = model.prepare_inputs_decode
    model.prepare_inputs_prefill = MethodType(_prepare_inputs_prefill, model)
    model.prepare_prefill_inputs_trace = MethodType(_prepare_prefill_inputs_trace, model)
    model.prepare_inputs_decode = MethodType(_prepare_inputs_decode_compat, model)
    model.transform_and_embed_prefill_inputs_device = MethodType(_transform_and_embed_prefill_inputs_device, model)
    model.ttnn_prefill_forward = MethodType(_ttnn_prefill_forward, model)
    model.process_logits_after_prefill_trace = MethodType(_process_logits_after_prefill_trace, model)
    model.process_output_prefill = MethodType(_process_output_prefill, model)
    model.process_output_decode = MethodType(_process_output_decode, model)
    # Gemma4 decode traces use host-computed embeddings and positions rather
    # than a mutable token buffer, so replay must refresh inputs every step.
    model._tt_vllm_always_refresh_decode_trace_inputs = True
    # Keep the vLLM bringup on the same host-sampling path as the working
    # dedicated Gemma4 generator until on-device sampling is proven correct.
    model._supports_on_device_sampling = False
    model._vllm_prefill_input_ids_torch = None
    model._vllm_prefill_embeds_torch = None


def _patch_model_args(model_args, mesh_device, max_batch_size, max_seq_len, model_path):
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len
    model_args.max_prefill_chunk_size = max_seq_len
    model_args.trace_prefill_supported_seq_lens = []
    model_args.optimizations = _Gemma4VllmOptimizations()
    model_args.mesh_device = mesh_device
    model_args._gemma4_model_path = model_path
    model_args.can_enable_trace = lambda prefill_seq_len, num_cached_tokens=0: False
    model_args.is_llama_vision = lambda: False


class Gemma4ForCausalLM(Generator):
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Gemma4 decode traces return sampled tokens directly, like the working
        # standalone demo. Do not split sampling into a second trace that
        # assumes the first trace input tensor is a token buffer.
        self.enable_split_sampling = False

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        if optimizations not in (None, "performance"):
            raise ValueError("Gemma4 TT does not support custom optimization profiles")

        model_path = hf_config._name_or_path
        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        model_args = []
        model = []
        state_dict = None
        for submesh in submesh_devices:
            model_args_i, model_i, _, state_dict = create_tt_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                dtype=ttnn.bfloat16,
                state_dict=state_dict,
                num_layers=n_layers,
                mesh_config=None,
                paged_attention_config=None,
                create_kv_cache=False,
                model_path=model_path,
            )
            _patch_model_args(
                model_args_i,
                submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                model_path=model_path,
            )
            _patch_model_instance(model_i, model_args_i)
            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].weight_cache_path(ttnn.bfloat16)

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(
            *args,
            **kwargs,
            dp_model=self.model,
            tt_cache_path=self.cache_path,
        )

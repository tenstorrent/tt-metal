# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B TTNN model.

    embedding -> [DecoderLayer] * 32 -> final RMSNorm -> lm_head

Key differences from GPT-OSS:
  * Dense SwiGLU MLP instead of MoE.
  * Llama3 RoPE frequency scaling instead of YaRN.
  * No attention sinks, no sliding window.
  * No projection biases.
  * 128 head_dim (vs 64), 4096 hidden_size (vs 2880).
"""

import torch

import ttnn
from models.demos.llama_3_1_8b.tt.config import MeshConfig
from models.demos.llama_3_1_8b.tt.rope import build_transformation_mat
from models.demos.llama_3_1_8b.utils.general_utils import get_cache_file_name
from models.demos.llama_3_1_8b.utils.substate import substate

from .layer import DecoderLayer
from .rms_norm import RMSNorm


def compute_per_device_vocab(vocab_size, num_tp):
    """Per-device vocab width: tile-aligned then rounded to next power of 2.

    Power-of-2 rounding enables ttnn.topk's multi-core path (bitonic sort requires power-of-2 width).
    Must match lm_head weight padding and TTSampling device-offset strides.
    """
    per_device = (((vocab_size + num_tp - 1) // num_tp + 31) // 32) * 32
    return 1 << (per_device - 1).bit_length()  # next power of 2


class Model:
    """Llama 3.1 8B TTNN prefill model (GQA + Llama3 RoPE + dense SwiGLU MLP)."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        max_local_batch_size=1,
        max_seq_len=128 * 1024,
        sequence_parallel=False,
    ):
        self.mesh_device = mesh_device
        self.vocab_size = hf_config.vocab_size
        self.hf_config = hf_config
        self.head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        self.max_local_batch_size = max_local_batch_size
        self.sequence_parallel = sequence_parallel
        self.ccl_manager = ccl_manager

        # Prefill mesh parallelization: TP over the cols (4); SP over the rows (8).
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])

        # Replicated RoPE transformation matrix for rotary_embedding_llama / rotary_embedding_indexed.
        self.transformation_mats = {"prefill": build_transformation_mat(mesh_device)}

        if state_dict:
            embedding_weight = substate(state_dict, "model.embed_tokens")["weight"]
            embedding_weight = embedding_weight.unsqueeze(0).unsqueeze(0)
        else:
            embedding_weight = None

        # Token embedding is replicated on every device (DRAM).
        self.embedding_weight = ttnn.as_tensor(
            embedding_weight,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "model.embed_tokens.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.layers = [
            DecoderLayer(
                mesh_device,
                hf_config,
                substate(state_dict, f"model.layers.{layer_idx}"),
                layer_idx,
                ccl_manager,
                dtype=dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, f"model.layers.{layer_idx}"),
                mesh_config=self.mesh_config,
                transformation_mats=self.transformation_mats,
                max_seq_len=max_seq_len,
                sequence_parallel=sequence_parallel,
            )
            for layer_idx in range(hf_config.num_hidden_layers)
        ]
        self.norm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "model.norm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "norm"),
            mesh_config=self.mesh_config,
        )

        # Pad lm_head vocab to padded_vocab_size BEFORE column-parallel sharding
        sampling_splits = mesh_device.shape[1]
        per_device_padded = compute_per_device_vocab(self.vocab_size, sampling_splits)
        padded_vocab_size = per_device_padded * sampling_splits
        if state_dict:
            lm_head_weight = substate(state_dict, "lm_head")["weight"].transpose(0, 1)  # [hidden, vocab]
            if lm_head_weight.shape[1] < padded_vocab_size:
                lm_head_weight = torch.nn.functional.pad(
                    lm_head_weight, (0, padded_vocab_size - lm_head_weight.shape[1]), "constant", 0
                )
        else:
            lm_head_weight = None
        self.lm_head_weight = ttnn.as_tensor(
            lm_head_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head_padded_pow2.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_config.column_parallel(mesh_device),
        )

    def _forward_layers_and_head(
        self,
        hidden_states,
        rope_mats,
        current_pos,
        get_last_token=-1,
        user_id=0,
        batch_size=1,
        skip_lm_head=False,
        on_layer_complete=None,
        kv_cache=None,
        cached_len=0,
        indexed_rope=False,
    ):
        """Prefill forward through decoder layers + final projection."""
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=rope_mats,
                position_idx=current_pos,
                kv_cache=kv_cache,
                user_id=user_id,
                batch_size=batch_size,
                cached_len=cached_len,
                indexed_rope=indexed_rope,
            )
            if on_layer_complete is not None:
                on_layer_complete(i)
        logits = hidden_states

        if get_last_token != -1:
            if len(logits.shape) == 3:
                logits = ttnn.unsqueeze(logits, dim=1)
            if batch_size > 1:
                per_user_seq = logits.shape[2] // batch_size
                tiles = []
                for b in range(batch_size):
                    start = b * per_user_seq + get_last_token
                    tile = ttnn.slice(logits, (0, 0, start, 0), (1, 1, start + 32, logits.shape[-1]))
                    tiles.append(tile)
                logits.deallocate(True)
                logits = ttnn.concat(tiles, dim=2)
                for t in tiles:
                    t.deallocate(True)
            else:
                logits_sliced = ttnn.slice(
                    logits, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, logits.shape[-1])
                )
                logits.deallocate(True)
                logits = logits_sliced
            hidden_states = logits

        if skip_lm_head:
            return hidden_states

        # Final norm and lm_head
        hidden_states = self.norm(hidden_states)
        logits = ttnn.matmul(hidden_states, self.lm_head_weight, dtype=ttnn.bfloat8_b)
        hidden_states.deallocate(True)
        return logits

    def prefill_forward(
        self,
        x,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        skip_lm_head=False,
        on_layer_complete=None,
        cached_len=0,
        indexed_rope=False,
    ):
        """Prefill forward pass. ``rot_mats_global`` is the whole-cache indexed rope (from the runtime)
        when ``indexed_rope`` is set."""
        assert rot_mats_global is not None, "Llama prefill uses the on-device indexed rope; pass rot_mats_global"
        rope_mats = rot_mats_global

        return self._forward_layers_and_head(
            hidden_states=x,
            rope_mats=rope_mats,
            current_pos=None,
            kv_cache=kv_cache,
            get_last_token=get_last_token,
            user_id=user_id,
            batch_size=batch_size,
            skip_lm_head=skip_lm_head,
            on_layer_complete=on_layer_complete,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )

    def prepare_inputs_prefill(self, tokens, start_pos=0, trace_enabled=False, batch_size=1, user_id=0, **kwargs):
        """Embed + SP-shard one chunk's token ids into the model input.

        RoPE is NOT built here — Llama prefill uses the on-device indexed rope, which the runtime
        builds once and passes into ``prefill_forward`` as ``rot_mats_global``. Returns
        ``(tokens_embd, None, None)`` to keep the 3-tuple interface.
        """
        device = None if trace_enabled else self.mesh_device

        if self.sequence_parallel:
            if tokens.dim() == 1:
                tokens = tokens.reshape(1, -1)
            seq_total = tokens.shape[-1]
            sp = self.mesh_device.shape[self.mesh_config.sp_axis]
            assert seq_total % sp == 0, f"SP prefill needs seq_len ({seq_total}) divisible by sp ({sp})"
            tokens = tokens.reshape(1, 1, 1, seq_total)
            tdims = [None, None]
            tdims[self.mesh_config.sp_axis] = 3  # seq dim across SP rows
            tokens = ttnn.from_torch(
                tokens,
                device=device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=tuple(tdims), mesh_shape=self.mesh_device.shape
                ),
            )
        else:
            if tokens.dim() == 2:
                tokens = tokens.reshape(1, 1, 1, -1)
            tokens = ttnn.from_torch(tokens, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        tokens_embd = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tokens.deallocate(True)
        if len(tokens_embd.shape) == 3:
            tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        return tokens_embd, None, None

    def process_output_prefill(self, tt_out, last_token_idx):
        """Host-side TP gather of the last-token logits."""
        tp = self.mesh_config.tp
        if tp > 1:
            device_tensors = ttnn.get_device_tensors(tt_out)
            torch_output = torch.cat([ttnn.to_torch(device_tensors[i]) for i in range(tp)], dim=-1)
        else:
            torch_output = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
        result = torch_output[..., last_token_idx, : self.vocab_size]
        return result

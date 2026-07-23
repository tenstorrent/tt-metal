# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS TTNN model. Mirrors ``minimax_m3/tt/model.py``.

    embedding -> [DecoderLayer] * num_hidden_layers -> final RMSNorm -> lm_head

Key differences from M3:
  * RoPE is the on-device INDEXED rope (whole-cache block-cyclic SP cos/sin built once by the
    runtime via ``tt/rope.build_indexed_rope`` and passed into ``prefill_forward`` as
    ``rot_mats_global``). This model therefore does NOT build a ``RotarySetup``; it only owns the
    replicated RoPE transformation matrix (``tt/rope.build_transformation_mat``).
  * Every layer is MoE (no dense schedule).
  * On-device sampling is OPTIONAL and OFF by default (prefill runs ``skip_lm_head=True``); the
    hooks are kept structurally identical to M3 for a later decode bring-up.
"""

import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss_d_p.tt.config import MeshConfig
from models.demos.gpt_oss_d_p.tt.rope import build_transformation_mat
from models.demos.gpt_oss_d_p.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss_d_p.utils.substate import substate

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
    """GPT-OSS-120B TTNN prefill model (GQA + attention sinks + sliding/full alternation + EP MoE)."""

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
        use_ep_moe=True,
        ep_seq_len_per_chip=1024,
        expert_weight_dtype=ttnn.bfloat4_b,
        sequence_parallel=False,
        enable_sampling=False,
    ):
        self.mesh_device = mesh_device
        self.vocab_size = hf_config.vocab_size
        self.hf_config = hf_config
        self.head_dim = hf_config.head_dim
        self.max_local_batch_size = max_local_batch_size
        self.sequence_parallel = sequence_parallel
        self.ccl_manager = ccl_manager

        # Prefill mesh parallelization: TP over the cols; SP and the EP MoE follow from the rows.
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])

        # Replicated RoPE transformation matrix for rotary_embedding_llama / rotary_embedding_indexed.
        # (The cos/sin themselves are the whole-cache indexed rope built by the runtime and passed in.)
        self.transformation_mats = {"prefill": build_transformation_mat(mesh_device)}

        if state_dict:
            embedding_weight = substate(state_dict, "model.embed_tokens")["weight"]
            embedding_weight = embedding_weight.unsqueeze(0).unsqueeze(0)
        else:
            embedding_weight = None

        # TODO: the token embedding is currently REPLICATED on every device (DRAM). Shard it across
        # the mesh (reuse deepseek_v3_d_p's TtParallelEmbedding) to save DRAM. Deferred.
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
                max_local_batch_size=max_local_batch_size,
                use_ep_moe=use_ep_moe,
                ep_seq_len_per_chip=ep_seq_len_per_chip,
                expert_weight_dtype=expert_weight_dtype,
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

        # Pad lm_head vocab to padded_vocab_size BEFORE column-parallel sharding so device shard
        # boundaries match the sampling device-offset strides (see M3 model.py rationale).
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

        # Optional on-device sampling (decode bring-up). OFF for prefill (skip_lm_head=True).
        self._supports_on_device_sampling = per_device_padded <= 64 * 1024
        self._prefill_sampling_active = False
        self.sampling_dp = 1
        self.sampling = None
        if enable_sampling and self._supports_on_device_sampling:
            from models.common.sampling.generator import SamplingGenerator

            self.sampling = SamplingGenerator(
                args=self._make_sampling_args(hf_config, mesh_device),
                mesh_device=mesh_device,
                tt_ccl=None,
            )
            logger.info(f"On-device sampling initialized (vocab_size={self.vocab_size}, splits={sampling_splits})")

    def _make_sampling_args(self, hf_config, mesh_device):
        """Minimal args object for SamplingGenerator/TTSampling."""

        class _SamplingArgs:
            pass

        args = _SamplingArgs()
        args.vocab_size = hf_config.vocab_size
        num_tp = mesh_device.shape[1]
        per_device_vocab = compute_per_device_vocab(args.vocab_size, num_tp)
        args.padded_vocab_size = per_device_vocab * num_tp
        args.cluster_shape = tuple(mesh_device.shape)
        args.sampling_all_gather_axis = 1
        args.num_devices = mesh_device.get_num_devices()
        args.is_galaxy = mesh_device.shape[0] > 1
        args.model_config = {}
        args.sampling_dp = self.sampling_dp
        args.use_topk_logprobs = True
        return args

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
        """Prefill forward through decoder layers + final projection.

        on_layer_complete: optional ``fn(layer_idx)`` invoked after each decoder layer — the SEAM for
        per-layer KV migration / validation in the disaggregated prefill pipeline. Default None = no-op.
        cached_len: valid prefix already in the cache before this chunk (0 = first/only chunk).
        """
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
        self._prefill_sampling_active = False
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
        assert rot_mats_global is not None, "GPT-OSS prefill uses the on-device indexed rope; pass rot_mats_global"
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

        RoPE is NOT built here — GPT-OSS prefill uses the on-device indexed rope, which the runtime
        builds once and passes into ``prefill_forward`` as ``rot_mats_global``. Returns
        ``(tokens_embd, None, None)`` to keep the M3 3-tuple interface.
        """
        device = None if trace_enabled else self.mesh_device

        if self.sequence_parallel:
            # SP prefill: ONE prompt of seq_len tokens, sharded by SEQUENCE across the SP rows and
            # replicated across the TP cols. Each device embeds its 1/sp seq-shard.
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

        # bf16 (not bf8) so the residual stream keeps full dynamic range (bf8's per-tile shared
        # exponent crushes small channels once massive activations appear; see layer.py adds).
        tokens_embd = ttnn.embedding(tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tokens.deallocate(True)
        if len(tokens_embd.shape) == 3:
            tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        return tokens_embd, None, None

    def process_output_prefill(self, tt_out, last_token_idx):
        """Host-side TP gather of the last-token logits (generator moves logits to CPU first)."""
        tp = self.mesh_config.tp
        if tp > 1:
            device_tensors = ttnn.get_device_tensors(tt_out)
            torch_output = torch.cat([ttnn.to_torch(device_tensors[i]) for i in range(tp)], dim=-1)
        else:
            torch_output = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
        result = torch_output[..., last_token_idx, : self.vocab_size]
        return result

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from .attention import Attention, AttentionConfig
from .attention_configs import GPTOSSAttentionProgramConfig
from .mlp import MLP
from .rms_norm import RMSNorm


class DecoderLayer:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        paged_attention_config=None,
        mesh_config=None,
        create_kv_cache=True,
        transformation_mats=None,
        max_seq_len=1024,
        max_local_batch_size=1,
        users_row_sharded=False,
        use_throughput_experts=False,
    ):
        self.input_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "input_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "input_layernorm"),
            mesh_config=mesh_config,
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "post_attention_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "post_attention_layernorm"),
            mesh_config=mesh_config,
        )
        self.mlp = MLP(
            mesh_device,
            hf_config,
            substate(state_dict, "mlp"),
            ccl_manager,
            dtype=dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
            mesh_config=mesh_config,
            use_throughput_experts=use_throughput_experts,
        )

        self.attention_type = hf_config.layer_types[layer_idx]

        # Create attention configuration
        attention_config = AttentionConfig(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            sliding_window=hf_config.sliding_window,
            max_seq_len=max_seq_len,
            max_local_batch_size=max_local_batch_size,
            users_row_sharded=users_row_sharded,
        )

        # Create attention program config
        attention_program_config = GPTOSSAttentionProgramConfig()

        self.self_attn = Attention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=substate(state_dict, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=attention_program_config,
            layer_idx=layer_idx,
            paged_attention_config=paged_attention_config,
            transformation_mats=transformation_mats,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "self_attn"),
            create_kv_cache=create_kv_cache,
        )
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx
        self.users_to_save = [0, 2, 96]
        # self.users_to_save = [i for i in range(128)]
        self.decode_iter = 0

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        user_id=0,
        debug_user_id=None,
    ):
        # hidden_states: [1, 1, tokens/num_rows, hidden_size/num_columns]
        # residual: [1, 1, tokens/num_rows, hidden_size/num_columns]
        if self.layer_idx == 0:
            if not is_decode and (debug_user_id in self.users_to_save):
                suffix = f"user_id{debug_user_id}"
                torch.save(
                    ttnn.to_torch(
                        hidden_states,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            self.mesh_device, dims=(0, 1), mesh_shape=self.mesh_device.shape
                        ),
                    )[0, 0, :, :],
                    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_pre_rmsnorm_{suffix}.pt",
                )
            elif is_decode and self.decode_iter < 10:
                suffix = f"iter{self.decode_iter}"
                torch.save(
                    ttnn.to_torch(
                        hidden_states,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            self.mesh_device, dims=(-2, 0), mesh_shape=self.mesh_device.shape
                        ),
                    )[0, 0, :, :],
                    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_pre_rmsnorm_{suffix}.pt",
                )

        residual = hidden_states
        hidden_states_post_norm = self.input_layernorm(hidden_states)

        # additional all_gather (cluster_axis=1) to get [1, 1, global_batch//num_rows, hidden_size]
        # hidden_states_post_norm: [1, 1, tokens/num_rows, hidden_size]
        if self.layer_idx == 0:
            if not is_decode and (debug_user_id in self.users_to_save):
                suffix = f"user_id{debug_user_id}"
                torch.save(
                    ttnn.to_torch(
                        hidden_states,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            self.mesh_device, dims=(0, 1), mesh_shape=self.mesh_device.shape
                        ),
                    )[0, 0, :, :],
                    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_pre_attn_{suffix}.pt",
                )
            elif is_decode and self.decode_iter < 10:
                suffix = f"iter{self.decode_iter}"
                torch.save(
                    ttnn.to_torch(
                        hidden_states,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            self.mesh_device, dims=(-2, 0), mesh_shape=self.mesh_device.shape
                        ),
                    )[0, 0, :, :],
                    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_pre_attn_{suffix}.pt",
                )
        hidden_states = self.self_attn(
            hidden_states_post_norm,
            rope_mats=position_embeddings,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            user_id=user_id,
            debug_layer_id=self.layer_idx,
            debug_decode_iter=self.decode_iter,
            debug_user_id=debug_user_id if not is_decode else None,
        )
        if self.layer_idx == 0:
            if not is_decode and (debug_user_id in self.users_to_save):
                suffix = f"user_id{debug_user_id}"
                torch.save(
                    ttnn.to_torch(
                        hidden_states,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            self.mesh_device, dims=(0, 1), mesh_shape=self.mesh_device.shape
                        ),
                    )[0, 0, :, :],
                    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_attn_{suffix}.pt",
                )
            elif is_decode and self.decode_iter < 10:
                suffix = f"iter{self.decode_iter}"
                torch.save(
                    ttnn.to_torch(
                        hidden_states,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            self.mesh_device, dims=(-2, 0), mesh_shape=self.mesh_device.shape
                        ),
                    )[0, 0, :, :],
                    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_attn_{suffix}.pt",
                )
        hidden_states_post_norm.deallocate(True)

        # after reduce scatter at end of attn: [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)
        residual = hidden_states
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)
        # another all_gather (cluster_axis=1) to get [1, 1, global_batch//num_rows, hidden_size]

        hidden_states = self.mlp(hidden_states_post_norm, is_decode=is_decode)  # diff with llama: router scores
        if self.layer_idx == 0:
            if not is_decode and (debug_user_id in self.users_to_save):
                suffix = f"user_id{debug_user_id}"
                torch.save(
                    ttnn.to_torch(
                        hidden_states,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            self.mesh_device, dims=(0, 1), mesh_shape=self.mesh_device.shape
                        ),
                    )[0, 0, :, :],
                    f"gpt-oss-bad-outputs-debug/intermediate_tensors/prefill_post_mlp_{suffix}.pt",
                )
            elif is_decode and self.decode_iter < 10:
                suffix = f"iter{self.decode_iter}"
                torch.save(
                    ttnn.to_torch(
                        hidden_states,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            self.mesh_device, dims=(-2, 0), mesh_shape=self.mesh_device.shape
                        ),
                    )[0, 0, :, :],
                    f"gpt-oss-bad-outputs-debug/intermediate_tensors/decode_post_mlp_{suffix}.pt",
                )
        hidden_states_post_norm.deallocate(True)

        # TODO: replace all_reduce at end of MLP with reduce_scatter so we get [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)

        if is_decode:
            self.decode_iter += 1

        return hidden_states

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional, Tuple

import ttnn
from ttnn import ReplicateTensorToMesh

from models.demos.t3000.falcon40b.tt.falcon_attention import TtFalconAttention
from models.demos.t3000.falcon40b.tt.falcon_mlp import TtFalconMLP
from models.utility_functions import torch2tt_tensor

from models.demos.t3000.falcon40b.tt.model_utils import fused_partial_layernorm


class TtFalconDecoderLayer:
    def __init__(
        self,
        device_mesh,
        state_dict,
        base_url,
        layer_num,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        global_cos_sin_cache,
        ln_output_tensors_dict,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.state_dict = state_dict
        self.base_url = base_url
        self.device_mesh = device_mesh
        self.layer_num = layer_num
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config
        self.num_devices = len(device_mesh.get_device_ids())
        self.ln_output_tensors_dict = ln_output_tensors_dict

        assert config.parallel_attn, "Path for config.parallel_attn=False is not implemented in TtFalconDecoderLayer!"

        self.self_attn = TtFalconAttention(
            device_mesh=device_mesh,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
            global_cos_sin_cache=global_cos_sin_cache,
        )

        self.mlp = TtFalconMLP(
            device_mesh=device_mesh,
            state_dict=state_dict,
            base_url=base_url,
            layer_num=layer_num,
            hidden_size=config.hidden_size,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

        layer_name = f"{base_url}.{layer_num}"

        ln_mlp_weights_str = f"{layer_name}.ln_mlp.weight"
        ln_mlp_bias_str = f"{layer_name}.ln_mlp.bias"

        def pad_ln_params(x):
            return torch.cat([x.reshape([1, 1, 1, -1]), torch.zeros(1, 1, 31, x.shape[-1])], dim=2)

        ln_mlp_weights_path = tt_cache_path / f"{ln_mlp_weights_str}_{self.model_config['LN_MLP_WEIGHTS_DTYPE'].name}"

        self.ln_mlp_gamma = ttnn.as_tensor(
            tensor=self.state_dict[ln_mlp_weights_str],
            dtype=self.model_config["LN_MLP_WEIGHTS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["LN_MLP_WEIGHTS_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            cache_file_name=ln_mlp_weights_path,
            preprocess=pad_ln_params,
        )

        ln_mlp_bias_path = tt_cache_path / f"{ln_mlp_bias_str}_{self.model_config['LN_MLP_BIAS_DTYPE'].name}"

        self.ln_mlp_beta = ttnn.as_tensor(
            tensor=self.state_dict[ln_mlp_bias_str],
            dtype=self.model_config["LN_MLP_BIAS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["LN_MLP_BIAS_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            cache_file_name=ln_mlp_bias_path,
            preprocess=pad_ln_params,
        )

        ln_attn_weights_str = f"{layer_name}.ln_attn.weight"
        ln_attn_bias_str = f"{layer_name}.ln_attn.bias"

        ln_attn_weights_path = (
            tt_cache_path / f"{ln_attn_weights_str}_{self.model_config['LN_ATTN_WEIGHTS_DTYPE'].name}"
        )

        self.ln_attn_gamma = ttnn.as_tensor(
            tensor=self.state_dict[ln_attn_weights_str],
            dtype=self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["LN_ATTN_WEIGHTS_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            cache_file_name=ln_attn_weights_path,
            preprocess=pad_ln_params,
        )

        ln_attn_bias_path = tt_cache_path / f"{ln_attn_bias_str}_{self.model_config['LN_ATTN_BIAS_DTYPE'].name}"

        self.ln_attn_beta = ttnn.as_tensor(
            tensor=self.state_dict[ln_attn_bias_str],
            dtype=self.model_config["LN_ATTN_BIAS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["LN_ATTN_BIAS_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            cache_file_name=ln_attn_bias_path,
            preprocess=pad_ln_params,
        )

        self.layernorm_eps = config.layer_norm_epsilon

    def set_model_config(self, model_config):
        self.model_config = model_config
        self.self_attn.set_model_config(model_config)
        self.mlp.set_model_config(model_config)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        alibi: torch.Tensor,
        attention_mask: ttnn.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Input shape: [batch, 1, seq_len, hidden_size]"""

        if llm_mode == "prefill":
            return self.fwd_prefill(
                hidden_states=hidden_states,
                alibi=alibi,
                attention_mask=attention_mask,
                llm_mode=llm_mode,
                user_id=user_id,
                layer_past=layer_past,
                layer_past_len=layer_past_len,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        elif llm_mode == "decode":
            return self.fwd_decode(
                hidden_states=hidden_states,
                alibi=alibi,
                attention_mask=attention_mask,
                llm_mode=llm_mode,
                user_id=user_id,
                layer_past=layer_past,
                layer_past_len=layer_past_len,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        else:
            assert False

    def fwd_prefill(
        self,
        hidden_states: ttnn.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Input shape: [batch, 1, seq_len, hidden_size]"""

        assert not output_attentions

        if self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"].is_sharded():
            replicated_hidden_states = ttnn.sharded_to_interleaved(
                hidden_states,
                memory_config=self.model_config["DEFAULT_MEMCFG"],
            )
        else:
            replicated_hidden_states = ttnn.clone(
                hidden_states,
                memory_config=self.model_config["DEFAULT_MEMCFG"],
            )

        if replicated_hidden_states.dtype != self.model_config["BFP8_DTYPE"]:
            replicated_hidden_states = ttnn.experimental.typecast(
                replicated_hidden_states, self.model_config["BFP8_DTYPE"], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        replicated_hidden_states = ttnn.all_gather(
            replicated_hidden_states,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )

        if self.model_config["LN_INPUT_DTYPE"] != self.model_config["BFP8_DTYPE"]:
            replicated_hidden_states = ttnn.experimental.typecast(
                replicated_hidden_states, self.model_config["LN_INPUT_DTYPE"], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        attn_ln_output, mlp_ln_output = fused_partial_layernorm(
            replicated_hidden_states,
            self.ln_attn_gamma,
            self.ln_attn_beta,
            self.ln_mlp_gamma,
            self.ln_mlp_beta,
            self.layernorm_eps,
            self.model_config["layernorm_params"],
            self.model_config["PARTIAL_LN_MEMCFG"],
            # self.model_config["PARTIAL_LN_INPLACE_PROGCFG"],
            self.model_config["PARTIAL_LN_PROGCFG"],
            self.ln_output_tensors_dict["attn_layernorm"],
            self.ln_output_tensors_dict["mlp_layernorm"],
        )

        output = hidden_states

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=attn_ln_output,
            alibi=alibi,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attention_output, outputs = attn_outputs[0], attn_outputs[1:]

        # Add attn output to residiual first in place to save memory
        # Note that this is only correct in inference when dropout is disabled
        output = ttnn.add(
            output,
            attention_output,
            memory_config=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"],
            # output_tensor=output,
        )
        attention_output.deallocate(True)

        # MLP
        # mlp will deallocate layernorm_output
        mlp_output = self.mlp(mlp_ln_output, llm_mode=llm_mode)

        # dropout_add
        # For inference, this is just add
        output = ttnn.add(
            output,
            mlp_output,
            memory_config=self.model_config["DROPOUT_ADD_OUTPUT_MEMCFG"],
            # output_tensor=output,
        )
        mlp_output.deallocate(True)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (
                output,
                (),
            )  # Outputs should be empty if we ignore layer_past as well

        return outputs

    def fwd_decode(
        self,
        hidden_states: ttnn.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Input shape: [batch, 1, seq_len, hidden_size]"""

        assert not output_attentions

        replicated_hidden_states = ttnn.sharded_to_interleaved(
            hidden_states,
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )

        replicated_hidden_states = ttnn.all_gather(
            replicated_hidden_states,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DEFAULT_MEMCFG"],
        )
        replicated_hidden_states = ttnn.interleaved_to_sharded(
            replicated_hidden_states,
            self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        attn_ln_output = ttnn.layer_norm(
            replicated_hidden_states,
            epsilon=self.layernorm_eps,
            weight=self.ln_attn_gamma,
            bias=self.ln_attn_beta,
            memory_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
            program_config=self.model_config["LN_ATTN_PROGCFG"],
        )
        mlp_ln_output = ttnn.layer_norm(
            replicated_hidden_states,
            epsilon=self.layernorm_eps,
            weight=self.ln_mlp_gamma,
            bias=self.ln_mlp_beta,
            memory_config=self.model_config["LN_MLP_OUTPUT_MEMCFG"],
            program_config=self.model_config["LN_MLP_PROGCFG"],
        )

        output = hidden_states

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=attn_ln_output,
            alibi=alibi,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attention_output, outputs = attn_outputs[0], attn_outputs[1:]

        # Add attn output to residiual first in place to save memory
        # Note that this is only correct in inference when dropout is disabled
        output = ttnn.add(
            output,
            attention_output,
            memory_config=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"],
            # output_tensor=output,
        )
        attention_output.deallocate(True)

        # MLP
        # mlp will deallocate layernorm_output
        mlp_output = self.mlp(mlp_ln_output, llm_mode=llm_mode)

        # dropout_add
        # For inference, this is just add
        output = ttnn.add(
            output,
            mlp_output,
            memory_config=self.model_config["DROPOUT_ADD_OUTPUT_MEMCFG"],
            # output_tensor=output,
        )
        mlp_output.deallocate(True)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (
                output,
                (),
            )  # Outputs should be empty if we ignore layer_past as well

        return outputs

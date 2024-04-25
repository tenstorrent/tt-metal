# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional, Tuple

import tt_lib

from models.demos.falcon40b.tt.falcon_attention import TtFalconAttention
from models.demos.falcon40b.tt.falcon_mlp import TtFalconMLP
from models.utility_functions import torch2tt_tensor, pad_by_zero

from models.demos.falcon40b.tt.model_utils import convert_to_layout, partial_layernorm


class TtFalconDecoderLayer:
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        global_cos_sin_cache,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.state_dict = state_dict
        self.base_url = base_url
        self.devices = devices
        self.layer_num = layer_num
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config
        self.num_devices = len(devices)

        assert config.parallel_attn, "Path for config.parallel_attn=False is not implemented in TtFalconDecoderLayer!"

        self.self_attn = TtFalconAttention(
            devices=devices,
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
            devices=devices,
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

        ln_mlp_weights_path = (
            tt_cache_path / f"{ln_mlp_weights_str}_rm_{self.model_config['LN_MLP_WEIGHTS_DTYPE'].name}.bin"
        )
        if (ln_mlp_weights_path).exists():
            ln_mlp_gamma_host = tt_lib.tensor.load_tensor(str(ln_mlp_weights_path))
            self.ln_mlp_gamma = [
                ln_mlp_gamma_host.to(device, self.model_config["LN_MLP_WEIGHTS_MEMCFG"]) for device in devices
            ]
        else:
            ln_mlp_gamma_host = tt_lib.tensor.Tensor(
                self.state_dict[ln_mlp_weights_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_MLP_WEIGHTS_DTYPE"],
            )
            self.ln_mlp_gamma = [
                ln_mlp_gamma_host.to(device, self.model_config["LN_MLP_WEIGHTS_MEMCFG"]) for device in devices
            ]
            tt_lib.tensor.dump_tensor(
                str(ln_mlp_weights_path),
                ln_mlp_gamma_host,
            )

        ln_mlp_bias_path = tt_cache_path / f"{ln_mlp_bias_str}_rm_{self.model_config['LN_MLP_BIAS_DTYPE'].name}.bin"
        if (ln_mlp_bias_path).exists():
            ln_mlp_beta_host = tt_lib.tensor.load_tensor(str(ln_mlp_bias_path))
            self.ln_mlp_beta = [
                ln_mlp_beta_host.to(device, self.model_config["LN_MLP_BIAS_MEMCFG"]) for device in devices
            ]
        else:
            ln_mlp_beta_host = tt_lib.tensor.Tensor(
                self.state_dict[ln_mlp_bias_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_MLP_BIAS_DTYPE"],
            )
            self.ln_mlp_beta = [
                ln_mlp_beta_host.to(device, self.model_config["LN_MLP_BIAS_MEMCFG"]) for device in devices
            ]
            tt_lib.tensor.dump_tensor(
                str(ln_mlp_bias_path),
                ln_mlp_beta_host,
            )

        ln_attn_weights_str = f"{layer_name}.ln_attn.weight"
        ln_attn_bias_str = f"{layer_name}.ln_attn.bias"

        ln_attn_weights_path = (
            tt_cache_path / f"{ln_attn_weights_str}_rm_{self.model_config['LN_ATTN_WEIGHTS_DTYPE'].name}.bin"
        )
        if (ln_attn_weights_path).exists():
            ln_attn_gamma_host = tt_lib.tensor.load_tensor(str(ln_attn_weights_path))
            self.ln_attn_gamma = [
                ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"]) for device in devices
            ]
        else:
            ln_attn_gamma_host = tt_lib.tensor.Tensor(
                self.state_dict[ln_attn_weights_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
            )
            self.ln_attn_gamma = [
                ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"]) for device in devices
            ]
            tt_lib.tensor.dump_tensor(
                str(ln_attn_weights_path),
                ln_attn_gamma_host,
            )

        ln_attn_bias_path = tt_cache_path / f"{ln_attn_bias_str}_rm_{self.model_config['LN_ATTN_BIAS_DTYPE'].name}.bin"
        if (ln_attn_bias_path).exists():
            ln_attn_beta_host = tt_lib.tensor.load_tensor(str(ln_attn_bias_path))
            self.ln_attn_beta = [
                ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"]) for device in devices
            ]
        else:
            ln_attn_beta_host = tt_lib.tensor.Tensor(
                self.state_dict[ln_attn_bias_str].reshape([1, 1, -1, 32]),
                self.model_config["LN_ATTN_BIAS_DTYPE"],
            )
            self.ln_attn_beta = [
                ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"]) for device in devices
            ]
            tt_lib.tensor.dump_tensor(
                str(ln_attn_bias_path),
                ln_attn_beta_host,
            )

        self.layernorm_eps = config.layer_norm_epsilon

    def set_model_config(self, model_config):
        self.model_config = model_config
        self.self_attn.set_model_config(model_config)
        self.mlp.set_model_config(model_config)

    def preprocessing(self, llm_mode, batch_size, sequence_size):
        self.self_attn.preprocessing(llm_mode, batch_size, sequence_size)

    def __call__(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[tt_lib.tensor.Tensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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
        hidden_states: tt_lib.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[tt_lib.tensor.Tensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Input shape: [batch, 1, seq_len, hidden_size]"""

        assert not output_attentions

        if self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"].is_sharded():
            replicated_hidden_states = []
            for i in range(len(hidden_states)):
                replicated_hidden_states.append(
                    tt_lib.tensor.sharded_to_interleaved(
                        hidden_states[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
                    )
                )
        else:
            replicated_hidden_states = []
            for i in range(len(hidden_states)):
                replicated_hidden_states.append(
                    tt_lib.tensor.clone(hidden_states[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"])
                )

        replicated_hidden_states = tt_lib.tensor.all_gather(
            replicated_hidden_states,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            dim=3,
            output_mem_config=self.model_config["DEFAULT_MEMCFG"],
        )

        attn_ln_output = partial_layernorm(
            replicated_hidden_states,
            self.ln_attn_gamma,
            self.ln_attn_beta,
            self.layernorm_eps,
            self.model_config["layernorm_params"],
            self.model_config["PARTIAL_LN_MEMCFG"],
            self.model_config["PARTIAL_LN_PROGCFG"],
            self.model_config["LN_MLP_OUTPUT_DTYPE"],
            self.hidden_size,
            self.devices,
        )

        mlp_ln_output = partial_layernorm(
            replicated_hidden_states,
            self.ln_mlp_gamma,
            self.ln_mlp_beta,
            self.layernorm_eps,
            self.model_config["layernorm_params"],
            self.model_config["PARTIAL_LN_MEMCFG"],
            self.model_config["PARTIAL_LN_INPLACE_PROGCFG"],
            self.model_config["LN_MLP_OUTPUT_DTYPE"],
            self.hidden_size,
            self.devices,
        )

        residual = hidden_states

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

        output = []

        # Add attn output to residiual first in place to save memory
        # Note that this is only correct in inference when dropout is disabled
        for i in range(len(residual)):
            output.append(
                tt_lib.operations.primary.add(
                    residual[i],
                    attention_output[i],
                    output_mem_config=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"],
                    in_place=True,
                )
            )
            attention_output[i].deallocate(True)

        # MLP
        # mlp will deallocate layernorm_output
        mlp_output = self.mlp(mlp_ln_output, llm_mode=llm_mode)

        # dropout_add
        # For inference, this is just add
        for i in range(len(output)):
            output[i] = tt_lib.operations.primary.add(
                output[i],
                mlp_output[i],
                output_mem_config=self.model_config["DROPOUT_ADD_OUTPUT_MEMCFG"],
                in_place=True,
            )

            mlp_output[i].deallocate(True)

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
        hidden_states: tt_lib.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[tt_lib.tensor.Tensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Input shape: [batch, 1, seq_len, hidden_size]"""

        assert not output_attentions

        replicated_hidden_states = []
        for i in range(len(hidden_states)):
            replicated_hidden_states.append(
                tt_lib.tensor.sharded_to_interleaved(
                    hidden_states[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
                )
            )
        replicated_hidden_states = tt_lib.tensor.all_gather(
            replicated_hidden_states,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            dim=3,
            output_mem_config=self.model_config["DEFAULT_MEMCFG"],
        )
        for i in range(len(replicated_hidden_states)):
            replicated_hidden_states[i] = tt_lib.tensor.interleaved_to_sharded(
                replicated_hidden_states[i], sharded_mem_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
            )

        attn_ln_output = []
        mlp_ln_output = []
        for i in range(len(replicated_hidden_states)):
            attn_ln_output.append(
                tt_lib.operations.primary.layernorm(
                    replicated_hidden_states[i],
                    self.layernorm_eps,
                    self.ln_attn_gamma[i],
                    self.ln_attn_beta[i],
                    self.model_config["LN_ATTN_OUTPUT_MEMCFG"],
                    self.model_config["LN_ATTN_PROGCFG"],
                )
            )
        # mlp_ln is in place, no need to deallocate original
        for i in range(len(replicated_hidden_states)):
            mlp_ln_output.append(
                tt_lib.operations.primary.layernorm(
                    replicated_hidden_states[i],
                    self.layernorm_eps,
                    self.ln_mlp_gamma[i],
                    self.ln_mlp_beta[i],
                    self.model_config["LN_MLP_OUTPUT_MEMCFG"],
                    self.model_config["LN_MLP_PROGCFG"],
                )
            )

        residual = hidden_states

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

        output = []

        # Add attn output to residiual first in place to save memory
        # Note that this is only correct in inference when dropout is disabled
        for i in range(len(residual)):
            output.append(
                tt_lib.operations.primary.add(
                    residual[i],
                    attention_output[i],
                    output_mem_config=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"],
                    in_place=True,
                )
            )
            attention_output[i].deallocate(True)

        # MLP
        # mlp will deallocate layernorm_output
        mlp_output = self.mlp(mlp_ln_output, llm_mode=llm_mode)

        # dropout_add
        # For inference, this is just add
        for i in range(len(output)):
            output[i] = tt_lib.operations.primary.add(
                output[i],
                mlp_output[i],
                output_mem_config=self.model_config["DROPOUT_ADD_OUTPUT_MEMCFG"],
                in_place=True,
            )

            mlp_output[i].deallocate(True)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (
                output,
                (),
            )  # Outputs should be empty if we ignore layer_past as well

        return outputs

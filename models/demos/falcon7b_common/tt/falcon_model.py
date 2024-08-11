# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import List, Optional, Tuple

import torch
import ttnn

import tt_lib

from models.demos.falcon7b_common.tt.falcon_decoder import TtFalconDecoderLayer
from models.demos.falcon7b_common.tt.model_utils import get_weights_cached, layernorm
from models.utility_functions import nearest_32, torch_tensors_to_tt_tensors
from models.demos.falcon7b_common.tests.test_utils import create_prefill_attn_mask_for_sharded_softmax
from tqdm import tqdm


class TtFalconModelShared(torch.nn.Module):
    @abstractmethod
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.devices = devices
        self.num_devices = len(devices)
        self.state_dict = state_dict
        self.base_url = base_url
        self.config = config
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config

        layer_name = f"{base_url}"
        embedding_weights_str = f"{layer_name}.word_embeddings.weight"
        self.embedding_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            embedding_weights_str,
            weight_config_str="WORD_EMBEDDING_WEIGHTS",
            weights_to_cache=(state_dict[embedding_weights_str] if state_dict else None),
            tt_layout=ttnn.experimental.tensor.Layout.ROW_MAJOR,
        )

        # stack all decoders
        self.layers = torch.nn.ModuleList(
            [
                TtFalconDecoderLayer(
                    devices=devices,
                    state_dict=state_dict,
                    base_url=f"{base_url}.h",
                    layer_num=layer_num,
                    config=config,
                    max_position_embeddings=max_position_embeddings,
                    model_config=model_config,
                    tt_cache_path=tt_cache_path,
                )
                for layer_num in tqdm(range(num_layers), desc="Loading decoder layers")
            ]
        )

        layernorm_weights_str = f"{layer_name}.ln_f.weight"
        layernorm_bias_str = f"{layer_name}.ln_f.bias"

        self.layernorm_gamma = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            layernorm_weights_str,
            weight_config_str="LN_F_WEIGHTS",
            weights_to_cache=(self.state_dict[layernorm_weights_str] if self.state_dict else None),
            padzero=True,
        )
        self.layernorm_beta = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            layernorm_bias_str,
            weight_config_str="LN_F_BIAS",
            weights_to_cache=(self.state_dict[layernorm_bias_str] if self.state_dict else None),
            padzero=True,
        )

        self.layernorm_eps = config.layer_norm_epsilon

    def model_preprocessing(self, llm_mode, input_ids, kv_cache_len, num_input_tokens):
        assert input_ids.dim() == 2
        global_batch_size, sequence_size = input_ids.shape
        batch_size = global_batch_size // self.num_devices

        # Generate input and attention_mask ---------------------------------------------
        if llm_mode == "prefill":
            assert batch_size == 1, "For prefill, batch_size must be 1!"
            assert sequence_size % 32 == 0, "For prefill, sequence_size must be multiple of 32!"
            assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

            # Prepare attention mask
            attention_mask_bool = torch.ones(batch_size, 1, sequence_size, num_input_tokens, dtype=bool)
            attention_mask_bool = attention_mask_bool.triu(diagonal=1)

            attention_mask_bool_padded = torch.cat(
                (
                    attention_mask_bool,
                    torch.ones(batch_size, 1, sequence_size, sequence_size - num_input_tokens, dtype=bool),
                ),
                dim=-1,
            )

            if self.model_config["PREFILL_OPTIMIZED_MODE"] and num_input_tokens in [128, 1024, 2048]:
                attention_mask_ = create_prefill_attn_mask_for_sharded_softmax(
                    attention_mask_bool_padded * -1e5,
                    self.config.num_attention_heads,
                    num_input_tokens,
                )
                # Send attn masks to device
                attn_masks_unordered = [
                    torch_tensors_to_tt_tensors(
                        [attention_mask_slice for _ in self.devices],
                        ttnn.experimental.tensor.Layout.ROW_MAJOR,
                        ttnn.experimental.tensor.DataType.BFLOAT16,  # subsequent tilize op excepts bfloat16 inputs
                        self.model_config["ATTN_MASK_MEMCFG"],
                        self.devices,
                    )
                    for attention_mask_slice in attention_mask_
                ]
                # Tilize attn masks
                for tt_attention_mask_slice in attn_masks_unordered:
                    for i in range(self.num_devices):
                        tt_attention_mask_slice[i] = ttnn.tilize(
                            tt_attention_mask_slice[i],
                            memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                            dtype=self.model_config["ATTN_MASK_OPTIMIZED_PREFILL_DTYPE"],
                        )
                # Expected output attention_masks
                # [dev0: [slice0, slice1, ...], dev1: [slice0, slice1, ...], ...]
                tt_attention_mask = [list(x) for x in zip(*attn_masks_unordered)]
            else:
                attention_mask_ = attention_mask_bool_padded * -1e3
                attention_masks = [attention_mask_.clone() for _ in self.devices]
                # Send attn masks to device
                tt_attention_mask = torch_tensors_to_tt_tensors(
                    attention_masks,
                    ttnn.experimental.tensor.Layout.ROW_MAJOR,
                    ttnn.experimental.tensor.DataType.BFLOAT16,  # subsequent tilize op excepts bfloat16 inputs
                    self.model_config["ATTN_MASK_MEMCFG"],
                    self.devices,
                )
                # Repeat attn masks for all heads
                for i in range(self.num_devices):
                    tt_attention_mask[i] = ttnn.repeat(
                        tt_attention_mask[i],
                        ttnn.Shape([1, self.config.num_attention_heads, 1, 1]),
                        memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                    )
                # Tilize attn masks
                for i in range(self.num_devices):
                    tt_attention_mask[i] = ttnn.tilize(
                        tt_attention_mask[i],
                        memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                        dtype=self.model_config["ATTN_MASK_DTYPE"],
                    )

            tt_input_ids = []
            for i, device in enumerate(self.devices):
                tt_input_ids.append(
                    ttnn.as_tensor(
                        input_ids[i : i + 1],
                        dtype=self.model_config["INPUT_DTYPE"],
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=self.model_config["INPUT_MEMCFG"],
                    )
                )
        elif llm_mode == "decode":
            tt_input_ids, attention_masks = [], []
            assert batch_size % 32 == 0, "For decode, batch_size must be multiple of 32!"
            assert sequence_size == 1, "For decode, q_len must be 1!"
            for i, device in enumerate(self.devices):
                attention_mask_bool = torch.zeros(batch_size, 1, sequence_size, num_input_tokens, dtype=bool)

                num_max_tokens = nearest_32(
                    kv_cache_len + 1
                )  # Potentially, num_max_tokens must be provided as a separate argument
                attention_mask_bool_padded = torch.cat(
                    (
                        attention_mask_bool,
                        torch.ones(batch_size, 1, sequence_size, num_max_tokens - num_input_tokens, dtype=bool),
                    ),
                    dim=-1,
                )
                if self.model_config["l1_sharded"] == False:
                    attention_masks.append(
                        (attention_mask_bool_padded.transpose(0, 2) * -1e3).expand(
                            -1, self.config.num_attention_heads, -1, -1
                        )
                    )
                else:
                    # Reshape width to tile-size since that is required by scale_mask_softmax_in_place with causal_mask=False (in falcon_attention.py)
                    attention_masks.append(attention_mask_bool_padded.reshape(batch_size, 1, -1, 32) * -1e3)
            # Send attn masks to device
            tt_attention_mask = torch_tensors_to_tt_tensors(
                attention_masks,
                ttnn.experimental.tensor.Layout.ROW_MAJOR,
                ttnn.experimental.tensor.DataType.BFLOAT16,  # subsequent tilize op excepts bfloat16 inputs
                self.model_config["ATTN_MASK_MEMCFG"],
                self.devices,
            )
            if not self.model_config["l1_sharded"]:
                # Tilize attn masks
                for i in range(self.num_devices):
                    tt_attention_mask[i] = ttnn.tilize(
                        tt_attention_mask[i],
                        memory_config=self.model_config["ATTN_MASK_MEMCFG"],
                        dtype=self.model_config["ATTN_MASK_DTYPE"],
                    )

            for i, device in enumerate(self.devices):
                tt_input_ids.append(
                    ttnn.as_tensor(
                        input_ids[batch_size * i : batch_size * (i + 1)].transpose(0, 1),
                        dtype=self.model_config["INPUT_DTYPE"],
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=self.model_config["INPUT_MEMCFG"],
                    )
                )

        else:
            raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

        return tt_input_ids, tt_attention_mask

    @abstractmethod
    def forward(
        self,
        input_ids: ttnn.experimental.tensor.Tensor,
        llm_mode: str,
        attention_mask: ttnn.experimental.tensor.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.experimental.tensor.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
        device_perf_run: bool = False,
    ) -> ttnn.experimental.tensor.Tensor:
        # Convert input tokens to embeddings
        input_embeddings = [
            ttnn.embedding(
                input_ids[i],
                self.embedding_weights[i],
                memory_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"],
            )
            for i in range(self.num_devices)
        ]
        for i in range(self.num_devices):
            input_embeddings[i] = ttnn.unsqueeze_to_4D(input_embeddings[i])
        for i in range(self.num_devices):
            input_embeddings[i] = ttnn.to_layout(input_embeddings[i], ttnn.experimental.tensor.Layout.TILE)

        layer_output = input_embeddings
        presents = ()
        for idx, layer in enumerate(self.layers):
            layer_output = layer(
                hidden_states=layer_output,
                alibi=None,
                attention_mask=attention_mask,
                llm_mode=llm_mode,
                user_id=user_id,
                layer_past=layer_past[idx],
                layer_past_len=layer_past_len,
                use_cache=use_cache,
            )
            presents += (layer_output[1],)
            layer_output = layer_output[0]

            if device_perf_run and idx % 8 == 0:
                for i in range(self.num_devices):
                    tt_lib.device.DumpDeviceProfiler(layer_output[i].device())

        # apply final norm layer
        layer_output = layernorm(
            layer_output,
            self.layernorm_eps,
            self.layernorm_gamma,
            self.layernorm_beta,
            self.num_devices,
            self.model_config,
        )

        return layer_output, presents


class TtFalconModel(TtFalconModelShared):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    ):
        super().__init__(
            devices=devices,
            state_dict=state_dict,
            base_url=base_url,
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            tt_cache_path=tt_cache_path,
        )

    def forward(
        self,
        input_ids: ttnn.experimental.tensor.Tensor,
        llm_mode: str,
        attention_mask: ttnn.experimental.tensor.Tensor = None,
        user_id: int = 0,
        layer_past: Optional[Tuple[Tuple[ttnn.experimental.tensor.Tensor]]] = None,
        layer_past_len: int = 0,
        use_cache: bool = False,
    ) -> ttnn.experimental.tensor.Tensor:
        hidden_states, presents = super().forward(
            input_ids=input_ids,
            llm_mode=llm_mode,
            attention_mask=attention_mask,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            use_cache=use_cache,
        )
        return hidden_states, presents

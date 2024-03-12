# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib as ttl

from typing import List
from models.utility_functions import torch2tt_tensor


class TtFalconCreateQKVHeads:
    def __init__(
        self,
        device,
        model_config,
        num_heads: int = 32,
        num_kv_heads: int = 2,
        head_dim: int = 64,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.model_config = model_config

    def __call__(self, x: ttl.tensor.Tensor) -> ttl.tensor.Tensor:
        # x = ttl.tensor.interleaved_to_sharded(
        #     x, sharded_mem_config=self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"]
        # )

        q_layer, k_layer, v_layer = ttl.tensor.nlp_create_qkv_heads(
            x,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            # output_mem_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
            # output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            output_mem_config=self.model_config["DRAM_MEMCFG"],
        )

        # q_layer = ttl.tensor.sharded_to_interleaved(
        #     q_layer, output_mem_config=self.model_config["DEFAULT_MEMCFG"]
        # )

        # k_layer = ttl.tensor.sharded_to_interleaved(
        #     k_layer, output_mem_config=self.model_config["DEFAULT_MEMCFG"]
        # )

        # v_layer = ttl.tensor.sharded_to_interleaved(
        #     v_layer, output_mem_config=self.model_config["DEFAULT_MEMCFG"]
        # )

        return q_layer, k_layer, v_layer

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor


class TtLlamaMLP_optimized(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = hidden_size
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        w1_str = f"{layer_name}.feed_forward.w1.weight"
        w2_str = f"{layer_name}.feed_forward.w2.weight"
        w3_str = f"{layer_name}.feed_forward.w3.weight"

        # if (
        #     tt_cache_path / f"{dense_h_to_4h_str}_{self.model_config['DENSE_H_TO_4H_MM_WEIGHTS_DTYPE'].name}.bin"
        # ).exists():
        #     self.dense_h_to_4h_weights = tt_lib.tensor.load_tensor(
        #         str(
        #             tt_cache_path
        #             / f"{dense_h_to_4h_str}_{self.model_config['DENSE_H_TO_4H_MM_WEIGHTS_DTYPE'].name}.bin"
        #         )
        #     ).to(device, self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"])
        # else:
        #     self.dense_h_to_4h_weights = torch2tt_tensor(
        #         torch.transpose(
        #             self.state_dict[dense_h_to_4h_str],
        #             -2,
        #             -1,
        #         ),
        #         self.device,
        #         tt_memory_config=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"],
        #         tt_dtype=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_DTYPE"],
        #     )
        #     tt_lib.tensor.dump_tensor(
        #         str(
        #             tt_cache_path
        #             / f"{dense_h_to_4h_str}_{self.model_config['DENSE_H_TO_4H_MM_WEIGHTS_DTYPE'].name}.bin"
        #         ),
        #         self.dense_h_to_4h_weights.cpu(),
        #     )

        # if (
        #     tt_cache_path / f"{dense_4h_to_h_str}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
        # ).exists():
        #     self.dense_4h_to_h_weights = tt_lib.tensor.load_tensor(
        #         str(
        #             tt_cache_path
        #             / f"{dense_4h_to_h_str}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
        #         )
        #     ).to(device, self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"])
        # else:
        #     self.dense_4h_to_h_weights = torch2tt_tensor(
        #         torch.transpose(
        #             self.state_dict[dense_4h_to_h_str],
        #             -2,
        #             -1,
        #         ),
        #         self.device,
        #         tt_memory_config=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"],
        #         tt_dtype=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_DTYPE"],
        #     )
        #     tt_lib.tensor.dump_tensor(
        #         str(
        #             tt_cache_path
        #             / f"{dense_4h_to_h_str}_{self.model_config['DENSE_4H_TO_H_MM_WEIGHTS_DTYPE'].name}.bin"
        #         ),
        #         self.dense_4h_to_h_weights.cpu(),
        #     )

        """
            self.w1 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,  # gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,  # input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,  # gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        """
        self.w1 = torch2tt_tensor(
            torch.transpose(
                self.state_dict[w1_str],
                -2,
                -1,
            ),
            self.device,
            tt_memory_config=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"],
            tt_dtype=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_DTYPE"],
        )
        self.w3 = torch2tt_tensor(
            torch.transpose(
                self.state_dict[w3_str],
                -2,
                -1,
            ),
            self.device,
            tt_memory_config=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_MEMCFG"],
            tt_dtype=self.model_config["DENSE_H_TO_4H_MM_WEIGHTS_DTYPE"],
        )
        self.w2 = torch2tt_tensor(
            torch.transpose(
                self.state_dict[w2_str],
                -2,
                -1,
            ),
            self.device,
            tt_memory_config=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_MEMCFG"],
            tt_dtype=self.model_config["DENSE_4H_TO_H_MM_WEIGHTS_DTYPE"],
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        print("w1 shape:", self.w1.shape())

        # w1_out = tt_lib.tensor.falcon_dense_h_to_4h_matmul(
        #     x,
        #     self.w1,
        #     fused_activation=tt_lib.tensor.FusibleActivation.SILU,
        #     output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
        #     output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
        # )

        # w3_out = tt_lib.tensor.falcon_dense_h_to_4h_matmul(
        #     x,
        #     self.w3,
        #     fused_activation=None,
        #     output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
        #     output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
        # )

        # x.deallocate()

        # w2_in = tt_lib.tensor.mul(w1_out, w3_out, output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"])

        # w2_out = tt_lib.tensor.falcon_dense_4h_to_h_matmul(
        #     w2_in,
        #     self.w2,
        #     output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
        #     output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
        # )

        w1_out = tt_lib.operations.primary.matmul_1d(
            x,
            self.w1,
            program_config=self.model_config["FF1_MM_PROGCFG"],
            output_mem_config=self.model_config["FF1_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
        )

        w3_out = tt_lib.operations.primary.matmul_1d(
            x,
            self.w3,
            program_config=self.model_config["FF3_MM_PROGCFG"],
            output_mem_config=self.model_config["FF3_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
        )

        x.deallocate(True)

        mul_mem_config = tt_lib.tensor.MemoryConfig(
            memory_layout=tt_lib.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=tt_lib.tensor.BufferType.L1,
        )

        w2_in = tt_lib.tensor.mul(w1_out, w3_out, output_mem_config=mul_mem_config)

        w2_out = tt_lib.operations.primary.matmul_1d(
            w2_in,
            self.w2,
            program_config=self.model_config["FF2_MM_PROGCFG"],
            output_mem_config=self.model_config["FF2_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
        )
        return w2_out

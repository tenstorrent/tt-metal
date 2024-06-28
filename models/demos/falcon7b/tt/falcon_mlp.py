# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.demos.falcon7b.tt.model_utils import get_falcon_default_core_grid, get_weights_cached
from torch import nn
from models.utility_functions import (
    is_grayskull,
    is_wormhole_b0,
)


def falcon_dense_4h_to_h_matmul(
    input_tensor_a,
    input_tensor_b,
    core_grid,
    output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
    output_dtype=None,
):
    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    elif is_wormhole_b0():
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    else:
        compute_kernel_config = None

    return ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        memory_config=output_mem_config,
        dtype=output_dtype,
        core_grid=core_grid,
        use_1d_systolic_array=True,
        compute_kernel_config=compute_kernel_config,
    )


def falcon_dense_h_to_4h_matmul(
    input_tensor_a,
    input_tensor_b,
    core_grid,
    fused_activation=None,
    output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
    output_dtype=None,
):
    seq_len = input_tensor_a.get_legacy_shape()[2]
    if seq_len > 1024:
        # TODO: Review if this path is used? If not, we can delete
        assert fused_activation == None
        return ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=output_mem_config, dtype=output_dtype)

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        )
    elif is_wormhole_b0():
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    else:
        compute_kernel_config = None

    return ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        memory_config=output_mem_config,
        dtype=output_dtype,
        core_grid=core_grid,
        activation=fused_activation,
        compute_kernel_config=compute_kernel_config,
    )


class TtFalconMLPPrefill(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        weights_dict=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.hidden_size = hidden_size
        self.model_config = model_config
        self.max_position_embeddings = max_position_embeddings
        self.padding_value = model_config["MLP_PADDING_VALUE"]
        self.seq_len = model_config["MLP_SEQ_LEN"]

        # Padding tensor for 1024 and 2048 sequence lengths

        layer_name = f"{base_url}.{layer_num}"
        dense_h_to_4h_str = f"{layer_name}.mlp.dense_h_to_4h.weight"
        dense_4h_to_h_str = f"{layer_name}.mlp.dense_4h_to_h.weight"

        custom_output_shape_h_to_4h = (
            (1, 1, self.padding_value, 4 * self.padding_value)
            if self.model_config["PREFILL_OPTIMIZED_MODE"] and self.seq_len in [1024, 2048]
            else None
        )
        custom_output_shape_4h_to_h = (
            (1, 1, 4 * self.padding_value, self.padding_value)
            if self.model_config["PREFILL_OPTIMIZED_MODE"] and self.seq_len in [1024, 2048]
            else None
        )

        self.dense_h_to_4h_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            dense_h_to_4h_str,
            weight_config_str="DENSE_H_TO_4H_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[dense_h_to_4h_str], -2, -1) if state_dict else None),
            weights_dict=weights_dict,
            custom_output_shape=custom_output_shape_h_to_4h,
        )
        self.dense_4h_to_h_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            dense_4h_to_h_str,
            weight_config_str="DENSE_4H_TO_H_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[dense_4h_to_h_str], -2, -1) if state_dict else None),
            weights_dict=weights_dict,
            custom_output_shape=custom_output_shape_4h_to_h,
        )

        if "MLP_PREFILL_PADDING_TENSORS" not in self.model_config and self.model_config["PREFILL_OPTIMIZED_MODE"]:
            self._load_mlp_padded_tensors()
        if "MLP_OUTPUT_TENSORS" not in self.model_config and self.model_config["PREFILL_OPTIMIZED_MODE"]:
            self._allocate_output_mlp_tensors()

    def _load_mlp_padded_tensors(self):
        # Load MLP padded tensors for 1024 and 2048 if they are smaller than max_position_embeddings or equal
        mlp_padding_tensors = []
        for device_id in range(self.num_devices):
            mlp_padding_tensor = dict()
            for seq_len in [1024, 2048]:
                # If explicitly set in model config, skip padding for larger sequence lengths, used for demo
                if seq_len > self.max_position_embeddings:
                    continue
                tt_padding = torch.zeros((1, 1, seq_len, 64)).bfloat16().float()  # 4608 - 4544 = 64
                tt_padding = ttnn.from_torch(
                    tt_padding,
                    ttnn.bfloat16,
                    device=self.devices[device_id],
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                mlp_padding_tensor[seq_len] = tt_padding
            mlp_padding_tensors.append(mlp_padding_tensor)

        self.model_config["MLP_PREFILL_PADDING_TENSORS"] = mlp_padding_tensors

    def _allocate_output_mlp_tensors(self):
        # prepare output tensor on device
        out_shape = [(1, 1, self.seq_len, self.dense_4h_to_h_weights[i].shape[-1]) for i in range(len(self.devices))]
        out_tensors = [torch.zeros(out_shape[i]).bfloat16().float() for i in range(len(self.devices))]

        out_tt = [
            ttnn.from_torch(
                out_tensors[i],
                ttnn.bfloat16,
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for i in range(len(self.devices))
        ]
        self.model_config["MLP_OUTPUT_TENSORS"] = out_tt

    def forward(self, x: ttnn.experimental.tensor.Tensor) -> ttnn.experimental.tensor.Tensor:
        if self.model_config["PREFILL_OPTIMIZED_MODE"] and self.seq_len in [1024, 2048]:
            for device_id in range(self.num_devices):
                tt_padding = self.model_config["MLP_PREFILL_PADDING_TENSORS"][device_id][self.seq_len]

                x[device_id] = ttnn.concat([x[device_id], tt_padding], dim=3)

            out_tt = self.model_config["MLP_OUTPUT_TENSORS"]

            num_slices = 2 if self.seq_len == 2048 else 1  # seq_len = 1024 num_slices = 1
            padded_hidden_size = self.model_config["MLP_PADDING_VALUE"]
            grid_size = self.model_config["MLP_GRID_SIZE"]

            for slice_idx in range(num_slices):
                slices = [
                    ttnn.experimental.tensor.interleaved_to_sharded_partial(
                        x[device_id],
                        grid_size,
                        [self.seq_len // num_slices // grid_size[1], padded_hidden_size // grid_size[0]],
                        num_slices,
                        slice_idx,
                        ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                        ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                    )
                    for device_id in range(len(self.devices))
                ]

                hidden_states = [
                    ttnn.matmul(
                        slices[device_id],
                        self.dense_h_to_4h_weights[device_id],
                        program_config=self.model_config["DENSE_H_TO_4H_MM_PROGCFG"],
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                        compute_kernel_config=self.model_config["MLP_KERNEL_CONFIG"],
                    )
                    for device_id in range(len(self.devices))
                ]  # 4544 -> 4608
                for i in range(len(self.devices)):
                    slices[i].deallocate()

                out_data = [
                    ttnn.matmul(
                        hidden_states[device_id],
                        self.dense_4h_to_h_weights[device_id],
                        program_config=self.model_config["DENSE_4H_TO_H_MM_PROGCFG"],
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                        compute_kernel_config=self.model_config["MLP_KERNEL_CONFIG"],
                    )
                    for device_id in range(len(self.devices))
                ]
                for i in range(len(self.devices)):
                    hidden_states[i].deallocate()

                for device_id in range(len(self.devices)):
                    ttnn.experimental.tensor.sharded_to_interleaved_partial(
                        out_data[device_id],
                        out_tt[device_id],
                        num_slices,
                        slice_idx,
                        self.model_config["MLP_INTERLEAVED_TO_SHARDED_MEM_CFG"],
                    )
                    out_data[device_id].deallocate()

            for device_id in range(len(self.devices)):
                x[device_id].deallocate()
                hidden_states[device_id].deallocate()

            # remove padding from output
            hidden_states = [out_tensor[:, :, :, : self.hidden_size] for out_tensor in out_tt]
        else:
            hidden_states = []
            for device_id in range(len(x)):
                hidden_states.append(
                    ttnn.linear(
                        x[device_id],
                        self.dense_h_to_4h_weights[device_id],
                        memory_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
                        dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
                        core_grid=get_falcon_default_core_grid(x[device_id].device()),
                        use_1d_systolic_array=True,
                        compute_kernel_config=self.model_config["MLP_KERNEL_CONFIG"],
                        activation="gelu",
                    )
                )
                x[device_id].deallocate()
            for device_id in range(len(x)):
                hidden_states[device_id] = ttnn.matmul(
                    hidden_states[device_id],
                    self.dense_4h_to_h_weights[device_id],
                    memory_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                    dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                    core_grid=get_falcon_default_core_grid(hidden_states[device_id].device()),
                    use_1d_systolic_array=True,
                    compute_kernel_config=self.model_config["MLP_KERNEL_CONFIG"],
                )

        # return TT Tensor
        return hidden_states


class TtFalconMLPDecode(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        weights_dict=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.hidden_size = hidden_size
        self.model_config = model_config
        self.padding_value = model_config["MLP_PADDING_VALUE"]
        self.prefill_seq_len = model_config["MLP_SEQ_LEN"]
        self.max_position_embeddings = max_position_embeddings
        layer_name = f"{base_url}.{layer_num}"
        dense_h_to_4h_str = f"{layer_name}.mlp.dense_h_to_4h.weight"
        dense_4h_to_h_str = f"{layer_name}.mlp.dense_4h_to_h.weight"

        custom_output_shape_h_to_4h = (
            (1, 1, self.padding_value, 4 * self.padding_value)
            if self.model_config["PREFILL_OPTIMIZED_MODE"] and self.prefill_seq_len in [1024, 2048]
            else None
        )
        custom_output_shape_4h_to_h = (
            (1, 1, 4 * self.padding_value, self.padding_value)
            if self.model_config["PREFILL_OPTIMIZED_MODE"] and self.prefill_seq_len in [1024, 2048]
            else None
        )

        self.dense_h_to_4h_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            dense_h_to_4h_str,
            weight_config_str="DENSE_H_TO_4H_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[dense_h_to_4h_str], -2, -1) if state_dict else None),
            weights_dict=weights_dict,
            custom_output_shape=custom_output_shape_h_to_4h,
        )
        self.dense_4h_to_h_weights = get_weights_cached(
            devices,
            model_config,
            tt_cache_path,
            dense_4h_to_h_str,
            weight_config_str="DENSE_4H_TO_H_MM_WEIGHTS",
            weights_to_cache=(torch.transpose(state_dict[dense_4h_to_h_str], -2, -1) if state_dict else None),
            weights_dict=weights_dict,
            custom_output_shape=custom_output_shape_4h_to_h,
        )
        if "MLP_DECODE_PADDING_TENSORS" not in self.model_config and self.model_config["PREFILL_OPTIMIZED_MODE"]:
            self._load_mlp_padded_tensors()

    def _load_mlp_padded_tensors(self):
        tt_paddings = []
        for device_id in range(self.num_devices):
            tt_padding = torch.zeros((1, 1, 32, 64)).bfloat16().float()  # 4608 - 4544 = 64, batch=32
            tt_padding = ttnn.from_torch(
                tt_padding,
                ttnn.bfloat16,
                device=self.devices[device_id],
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_paddings.append(tt_padding)

        self.model_config["MLP_DECODE_PADDING_TENSORS"] = tt_paddings

    def forward(self, x: ttnn.experimental.tensor.Tensor) -> ttnn.experimental.tensor.Tensor:
        batch_size = x[0].shape[-2]  # assume all devices have same shape
        hidden_states = []
        # pad inputs with padding tensor if not already padded
        if (
            self.model_config["PREFILL_OPTIMIZED_MODE"]
            and self.hidden_size < self.padding_value
            and self.prefill_seq_len in [1024, 2048]
        ):
            for device_id in range(self.num_devices):
                x[device_id] = ttnn.experimental.tensor.concat(
                    [x[device_id], self.model_config["MLP_DECODE_PADDING_TENSORS"][device_id]], dim=3
                )
        for device_id in range(self.num_devices):
            hidden_states.append(
                falcon_dense_h_to_4h_matmul(
                    x[device_id],
                    self.dense_h_to_4h_weights[device_id],
                    fused_activation="gelu",
                    output_mem_config=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["DENSE_H_TO_4H_MM_OUTPUT_DTYPE"],
                    core_grid=get_falcon_default_core_grid(x[device_id].device()),
                )
            )
            x[device_id].deallocate()
        for device_id in range(self.num_devices):
            hidden_states[device_id] = falcon_dense_4h_to_h_matmul(
                hidden_states[device_id],
                self.dense_4h_to_h_weights[device_id],
                output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_DTYPE"],
                core_grid=get_falcon_default_core_grid(hidden_states[device_id].device()),
            )
        # remove padding from output
        if self.model_config["PREFILL_OPTIMIZED_MODE"] and self.prefill_seq_len in [1024, 2048]:
            for i in range(self.num_devices):
                hidden_states[i] = ttnn.experimental.tensor.unpad(
                    hidden_states[i],
                    [0, 0, 0, 0],
                    [0, 0, batch_size - 1, self.hidden_size - 1],
                    output_mem_config=self.model_config["DENSE_4H_TO_H_MM_OUTPUT_MEMCFG"],
                )

        # return TT Tensor
        return hidden_states

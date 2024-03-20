# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.demos.llama2_70b.tt.llama_common import tt_all_gather_torch, get_weight_cache_path


class TtLlamaMLP_optimized(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        emulated=False,
        load_weights=True,
        cache_path=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.hidden_size = hidden_size
        self.model_config = model_config
        self.emulated = emulated
        self.cache_path = cache_path
        self.llm_mode = model_config["LLM_MODE"]

        self.layer_name = f"{base_url}.{layer_num}"

        if load_weights:
            self.load_weights()

    def free_weights(self):
        # Free weights
        for i in range(self.num_devices):
            self.w1_list[i].deallocate(True)
            self.w3_list[i].deallocate(True)
            self.w2_list[i].deallocate(True)
        del self.w1_list
        del self.w3_list
        del self.w2_list

    def load_weights(self):
        assert not hasattr(self, "w1_list"), "w1_list is already an attribute of this object"
        assert not hasattr(self, "w3_list"), "w3_list is already an attribute of this object"
        assert not hasattr(self, "w2_list"), "w2_list is already an attribute of this object"

        w1_str = f"{self.layer_name}.feed_forward.w1.weight"
        w2_str = f"{self.layer_name}.feed_forward.w2.weight"
        w3_str = f"{self.layer_name}.feed_forward.w3.weight"

        self.w1_list = []
        self.w3_list = []
        self.w2_list = []

        # Test if the all weights have been cached
        test_cache_path = get_weight_cache_path(self.cache_path, w3_str, self.num_devices - 1, self.num_devices)
        if test_cache_path.exists():
            for i in range(self.num_devices):
                tensor_cache_path = get_weight_cache_path(self.cache_path, w1_str, i, self.num_devices)
                self.w1_list.append(
                    tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["DRAM_MEMCFG"]
                    )
                )

                tensor_cache_path = get_weight_cache_path(self.cache_path, w2_str, i, self.num_devices)
                self.w2_list.append(
                    tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["DRAM_MEMCFG"]
                    )
                )

                tensor_cache_path = get_weight_cache_path(self.cache_path, w3_str, i, self.num_devices)
                self.w3_list.append(
                    tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["DRAM_MEMCFG"]
                    )
                )
        else:
            # Do padding
            H = 8 * 1024
            PADDED_H4 = 32 * 1024
            H4 = 28 * 1024
            padded_w1 = torch.zeros(H, PADDED_H4)
            padded_w2 = torch.zeros(PADDED_H4, H)
            padded_w3 = torch.zeros(H, PADDED_H4)
            padded_w1[:, :H4] = self.state_dict[w1_str].transpose(-2, -1)
            padded_w2[:H4, :] = self.state_dict[w2_str].transpose(-2, -1)
            padded_w3[:, :H4] = self.state_dict[w3_str].transpose(-2, -1)

            padded_w1_chunks = torch.chunk(padded_w1, self.num_devices, dim=-1)
            padded_w2_chunks = torch.chunk(padded_w2, self.num_devices, dim=-1)
            padded_w3_chunks = torch.chunk(padded_w3, self.num_devices, dim=-1)

            for i in range(self.num_devices):
                w1_host = torch2tt_tensor(
                    padded_w1_chunks[i],
                    None,
                    tt_memory_config=self.model_config["DRAM_MEMCFG"],
                    tt_dtype=self.model_config["BFP8_DTYPE"],
                )
                self.w1_list.append(w1_host.to(self.devices[i], self.model_config["DRAM_MEMCFG"]))
                tt_lib.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, w1_str, i, self.num_devices)),
                    w1_host,
                )

                w2_host = torch2tt_tensor(
                    padded_w2_chunks[i],
                    None,
                    tt_memory_config=self.model_config["DRAM_MEMCFG"],
                    tt_dtype=self.model_config["BFP8_DTYPE"],
                )
                self.w2_list.append(w2_host.to(self.devices[i], self.model_config["DRAM_MEMCFG"]))
                tt_lib.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, w2_str, i, self.num_devices)),
                    w2_host,
                )

                w3_host = torch2tt_tensor(
                    padded_w3_chunks[i],
                    None,
                    tt_memory_config=self.model_config["DRAM_MEMCFG"],
                    tt_dtype=self.model_config["BFP8_DTYPE"],
                )
                self.w3_list.append(w3_host.to(self.devices[i], self.model_config["DRAM_MEMCFG"]))
                tt_lib.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, w3_str, i, self.num_devices)),
                    w3_host,
                )

    def prepare_inputs(self, x):
        if self.llm_mode == "decode":
            batch, seq_len = 32, 1
            assert x.size() == (seq_len, 1, batch, self.hidden_size)
            x_multichip = []
            for i in range(self.num_devices):
                x_multichip.append(
                    torch2tt_tensor(
                        x.clone(),
                        self.devices[i],
                        tt_dtype=self.model_config["LN_MLP_OUTPUT_DTYPE"],
                        tt_memory_config=self.model_config["L1_MEMCFG"],
                    )
                )
            for i in range(self.num_devices):
                x_multichip[i] = tt_lib.tensor.interleaved_to_sharded(
                    x_multichip[i], sharded_mem_config=self.model_config["LN_MLP_OUTPUT_MEMCFG"]
                )
            return x_multichip
        elif self.llm_mode == "prefill":
            x_multichip = []
            for i in range(self.num_devices):
                x_multichip.append(
                    torch2tt_tensor(
                        x.clone(),
                        self.devices[i],
                        tt_dtype=self.model_config["LN_MLP_OUTPUT_DTYPE"],
                    )
                )
            return x_multichip

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        # Decode should have input tensor of shape (seqlen=1, 1, batch, hidden_size)
        # Prefill should have input tensor of shape (1, batch, seqlen, hidden_size)
        if self.llm_mode == "decode":
            return self.decode_forward(x)
        elif self.llm_mode == "prefill":
            return self.prefill_forward(x)
        else:
            raise ValueError(f"Invalid mode: {self.llm_mode}")

    def prefill_forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        hidden_states = []
        w1_outs = []
        w3_outs = []
        # TODO: Use FP32 accumulate after the issue with primary.matmul with FP32 accumulate is fixed
        for i in range(len(x)):
            """
            x[i] is shape [1,32,128,8192]
            self.w1_list[i] is shape [1,1,8192,4096]
            """
            w1_outs.append(
                tt_lib.operations.primary.matmul(
                    x[i],
                    self.w1_list[i],
                    program_config=self.model_config["PADDED_FF1_MM_PROGCFG"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"],
                )
            )

        for i in range(len(x)):
            w3_outs.append(
                tt_lib.operations.primary.matmul(
                    x[i],
                    self.w3_list[i],
                    program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"],
                )
            )
            x[i].deallocate(True)

        for i in range(len(w1_outs)):
            hidden_states.append(ttnn.mul(w1_outs[i], w3_outs[i]))
            w1_outs[i].deallocate(True)
            w3_outs[i].deallocate(True)

        if self.emulated:
            hidden_states = tt_all_gather_torch(hidden_states, dim=-1)
        else:
            hidden_states = tt_lib.tensor.all_gather(
                hidden_states,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            )

        for i in range(len(hidden_states)):
            hidden_states[i] = tt_lib.operations.primary.matmul(
                hidden_states[i],
                self.w2_list[i],
                program_config=self.model_config["PADDED_FF2_MM_PROGCFG"],
                compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"],
            )

        return hidden_states

    def decode_forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        hidden_states = []
        w1_outs = []
        w3_outs = []
        for i in range(len(x)):
            w1_outs.append(
                tt_lib.operations.primary.matmul_1d(
                    x[i],
                    self.w1_list[i],
                    program_config=self.model_config["PADDED_FF1_MM_PROGCFG"],
                    output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
                    output_dtype=self.model_config["PADDED_FF1_MM_OUTPUT_DTYPE"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )
            )
        for i in range(len(x)):
            w3_outs.append(
                tt_lib.operations.primary.matmul_1d(
                    x[i],
                    self.w3_list[i],
                    program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
                    output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
                    output_dtype=self.model_config["PADDED_FF3_MM_OUTPUT_DTYPE"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )
            )
            x[i].deallocate(True)

        for i in range(len(w1_outs)):
            hidden_states.append(
                tt_lib.tensor.mul(w1_outs[i], w3_outs[i], output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"])
            )
            w1_outs[i].deallocate(True)
            w3_outs[i].deallocate(True)

        for i in range(len(hidden_states)):
            # Put w2_inputs in DRAM
            hidden_states[i] = tt_lib.tensor.sharded_to_interleaved(
                hidden_states[i], output_mem_config=self.model_config["L1_MEMCFG"]
            )

        if self.emulated:
            hidden_states = tt_all_gather_torch(hidden_states, dim=-1)
        else:
            hidden_states = tt_lib.tensor.all_gather(
                hidden_states,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
                output_mem_config=self.model_config["L1_MEMCFG"],
            )

        # Put AllGather results in L1 Sharded
        for i in range(len(hidden_states)):
            hidden_states[i] = tt_lib.tensor.interleaved_to_sharded(
                hidden_states[i], sharded_mem_config=self.model_config["PADDED_MLP_ALL_GATHER_OUTPUT_MEMCFG"]
            )

        for i in range(len(hidden_states)):
            hidden_states[i] = tt_lib.operations.primary.matmul_1d(
                hidden_states[i],
                self.w2_list[i],
                program_config=self.model_config["PADDED_FF2_MM_PROGCFG"],
                output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
                output_dtype=self.model_config["PADDED_FF2_MM_OUTPUT_DTYPE"],
                compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
            )

        return hidden_states

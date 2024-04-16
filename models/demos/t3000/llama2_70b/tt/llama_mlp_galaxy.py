# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.demos.llama2_70b.tt.llama_common import tt_all_gather_torch, get_weight_cache_path_galaxy, tt_all_reduce


class TtLlamaMLP_galaxy(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        emulated=True,
        load_weights=True,
        cache_path=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        assert self.num_devices == 32

        self.frac_grid = [4, 8]  # [8,4]

        self.hidden_size = hidden_size
        self.model_config = model_config
        self.emulated = emulated
        self.cache_path = cache_path

        self.layer_name = f"{base_url}.{layer_num}"

        self.FF1_groups = [list(range(i, i + self.frac_grid[0])) for i in range(0, self.num_devices, self.frac_grid[0])]
        # [[0, 1, 2, 3],
        # [4, 5, 6, 7],
        # [8, 9, 10, 11],
        # [12, 13, 14, 15],
        # [16, 17, 18, 19],
        # [20, 21, 22, 23],
        # [24, 25, 26, 27],
        # [28, 29, 30, 31]]
        self.FF2_groups = [
            [i + j for j in range(0, self.num_devices, self.frac_grid[0])] for i in range(self.frac_grid[0])
        ]
        # [[0, 4, 8, 12, 16, 20, 24, 28],
        # [1, 5, 9, 13, 17, 21, 25, 29],
        # [2, 6, 10, 14, 18, 22, 26, 30],
        # [3, 7, 11, 15, 19, 23, 27, 31]]

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
        test_cache_path = get_weight_cache_path_galaxy(
            self.cache_path, w2_str, self.num_devices - 1, self.num_devices, x=3, y=7
        )
        if test_cache_path.exists():
            for x in range(self.frac_grid[1]):
                for y in range(self.frac_grid[0]):
                    device_id = self.FF1_groups[x][y]
                    # logger.info(f"Loading weights FF1 for weight chunk ({x},{y}) on device {device_id}")
                    tensor_cache_path = get_weight_cache_path_galaxy(
                        self.cache_path, w1_str, device_id, self.num_devices, x, y
                    )
                    self.w1_list.append(
                        tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                            self.devices[device_id], self.model_config["DRAM_MEMCFG"]
                        )
                    )
                    tensor_cache_path = get_weight_cache_path_galaxy(
                        self.cache_path, w3_str, device_id, self.num_devices, x, y
                    )
                    self.w3_list.append(
                        tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                            self.devices[device_id], self.model_config["DRAM_MEMCFG"]
                        )
                    )
            for y in range(self.frac_grid[1]):
                for x in range(self.frac_grid[0]):
                    device_id = self.FF2_groups[x][y]
                    # logger.info(f"Loading weights FF2 for weight chunk ({x},{y}) on device {device_id}")
                    tensor_cache_path = get_weight_cache_path_galaxy(
                        self.cache_path, w2_str, device_id, self.num_devices, x, y
                    )
                    self.w2_list.append(
                        tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                            self.devices[device_id], self.model_config["DRAM_MEMCFG"]
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

            # Chunk by 8 in the columns
            col_w1_chunks = torch.chunk(padded_w1, self.frac_grid[1], dim=-1)
            col_w3_chunks = torch.chunk(padded_w3, self.frac_grid[1], dim=-1)
            # Chunk by 4 in the columns
            col_w2_chunks = list(torch.chunk(padded_w2, self.frac_grid[0], dim=-1))

            block_w1_chunks = [torch.chunk(chunk, self.frac_grid[0], dim=0) for chunk in col_w1_chunks]
            block_w3_chunks = [torch.chunk(chunk, self.frac_grid[0], dim=0) for chunk in col_w3_chunks]
            block_w2_chunks = [torch.chunk(chunk, self.frac_grid[1], dim=0) for chunk in col_w2_chunks]

            # Loop down and then right
            for x in range(len(block_w1_chunks)):  # 0-7
                for y in range(len(block_w1_chunks[x])):  # 0-3
                    device_id = self.FF1_groups[x][y]
                    # logger.info(f"Saving weights FF1 for weight chunk ({x},{y}) on device {device_id}")
                    w1_host = torch2tt_tensor(
                        block_w1_chunks[x][y],
                        None,
                        tt_memory_config=self.model_config["DRAM_MEMCFG"],
                        tt_dtype=self.model_config["BFP8_DTYPE"],
                    )
                    self.w1_list.append(w1_host.to(self.devices[device_id], self.model_config["DRAM_MEMCFG"]))
                    tt_lib.tensor.dump_tensor(
                        str(get_weight_cache_path_galaxy(self.cache_path, w1_str, device_id, self.num_devices, x, y)),
                        w1_host,
                    )
                    w3_host = torch2tt_tensor(
                        block_w3_chunks[x][y],
                        None,
                        tt_memory_config=self.model_config["DRAM_MEMCFG"],
                        tt_dtype=self.model_config["BFP8_DTYPE"],
                    )
                    self.w3_list.append(w3_host.to(self.devices[device_id], self.model_config["DRAM_MEMCFG"]))
                    tt_lib.tensor.dump_tensor(
                        str(get_weight_cache_path_galaxy(self.cache_path, w3_str, device_id, self.num_devices, x, y)),
                        w3_host,
                    )

            # Loop right then down
            for y in range(len(block_w2_chunks[0])):  # 0-7
                for x in range(len(block_w2_chunks)):  # 0-3
                    device_id = self.FF2_groups[x][y]
                    # logger.info(f"Saving weights FF2 for weight chunk ({x},{y}) on device {device_id}")
                    w2_host = torch2tt_tensor(
                        block_w2_chunks[x][y],
                        None,
                        tt_memory_config=self.model_config["DRAM_MEMCFG"],
                        tt_dtype=self.model_config["BFP8_DTYPE"],
                    )
                    self.w2_list.append(w2_host.to(self.devices[device_id], self.model_config["DRAM_MEMCFG"]))
                    tt_lib.tensor.dump_tensor(
                        str(get_weight_cache_path_galaxy(self.cache_path, w2_str, device_id, self.num_devices, x, y)),
                        w2_host,
                    )

    def prepare_inputs(self, x):
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

    def prepare_inputs_mlp(self, x_multichip):
        # len(x) = 32, each is 1 x 1 x 32 x 8k sharded on a single chip
        assert len(x_multichip) == 32
        batch, seq_len = 32, 1
        for i in range(len(x_multichip)):
            assert x_multichip[i].shape == (seq_len, 1, batch, self.hidden_size)
            x_multichip[i] = tt_lib.tensor.sharded_to_interleaved(
                x_multichip[i], output_mem_config=self.model_config["L1_MEMCFG"]
            )
        for FF1_group in self.FF1_groups:
            for chunk_id, device_id in enumerate(FF1_group):
                # logger.info(f"Preparing input for FF1 on device {device_id} with chunk {chunk_id}")
                start = chunk_id * self.hidden_size // 4
                end = start + self.hidden_size // 4
                x_multichip[device_id] = tt_lib.tensor.unpad(
                    x_multichip[device_id],
                    [0, 0, 0, start],
                    [
                        seq_len - 1,
                        0,
                        batch - 1,
                        end - 1,
                    ],
                    output_mem_config=self.model_config["L1_MEMCFG"],
                )

        return x_multichip

    def forward(self, x: list) -> list:
        x = self.prepare_inputs_mlp(x)
        hidden_states_32chips = []
        w1_32chips = []
        w3_32chips = []

        for FF1_group in self.FF1_groups:
            w1_4chips = []
            for device_id in FF1_group:
                # logger.info(f"FF1 matmul on device {device_id} for chips {FF1_group}")
                w1_4chips.append(
                    tt_lib.operations.primary.matmul(
                        x[device_id],
                        self.w1_list[device_id],
                        program_config=self.model_config["PADDED_FF1_MM_PROGCFG"],
                        output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
                        output_dtype=self.model_config["PADDED_FF1_MM_OUTPUT_DTYPE"],
                        compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                    )
                )
            w1_32chips.append(w1_4chips)

        for i in range(len(w1_32chips)):
            for j in range(len(w1_32chips[i])):
                w1_32chips[i][j] = tt_lib.tensor.sharded_to_interleaved(
                    w1_32chips[i][j], output_mem_config=self.model_config["L1_MEMCFG"]
                )

        if self.emulated:
            for i in range(len(w1_32chips)):
                # logger.info(f"FF1 All-Reduce for chips {self.FF1_groups[i]}")
                w1_32chips[i] = tt_all_reduce(
                    w1_32chips[i],
                )

        for i in range(len(w1_32chips)):
            for j in range(len(w1_32chips[i])):
                w1_32chips[i][j] = tt_lib.tensor.silu(w1_32chips[i][j])

        for FF3_group in self.FF1_groups:
            w3_4chips = []
            for device_id in FF3_group:
                w3_4chips.append(
                    tt_lib.operations.primary.matmul(
                        x[device_id],
                        self.w3_list[device_id],
                        program_config=self.model_config["PADDED_FF3_MM_PROGCFG"],
                        output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
                        output_dtype=self.model_config["PADDED_FF3_MM_OUTPUT_DTYPE"],
                        compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                    )
                )
                x[device_id].deallocate(True)
            w3_32chips.append(w3_4chips)

        for i in range(len(w1_32chips)):
            for j in range(len(w1_32chips[i])):
                w3_32chips[i][j] = tt_lib.tensor.sharded_to_interleaved(
                    w3_32chips[i][j], output_mem_config=self.model_config["L1_MEMCFG"]
                )

        if self.emulated:
            for i in range(len(w3_32chips)):
                # logger.info(f"FF3 All-Reduce for chips {self.FF1_groups[i]}")
                w3_32chips[i] = tt_all_reduce(
                    w3_32chips[i],
                )

        for i in range(len(w1_32chips)):
            for j in range(len(w1_32chips[i])):
                w1_32chips[i][j] = tt_lib.tensor.interleaved_to_sharded(
                    w1_32chips[i][j],
                    sharded_mem_config=self.model_config["PADDED_MLP_ALL_GATHER_OUTPUT_MEMCFG"],
                )
                w3_32chips[i][j] = tt_lib.tensor.interleaved_to_sharded(
                    w3_32chips[i][j],
                    sharded_mem_config=self.model_config["PADDED_MLP_ALL_GATHER_OUTPUT_MEMCFG"],
                )

        # w1_4chips = [ff1_out, ff1_out, ff1_out, ff1_out]
        # w1_32chips = [[ff1_out, ff1_out, ff1_out, ff1_out], [ff1_out, ff1_out, ff1_out, ff1_out], ...]
        for i in range(len(w1_32chips)):
            hidden_states_4chips = []
            for j in range(len(w1_32chips[i])):
                hidden_states_4chips.append(
                    tt_lib.tensor.mul(
                        w1_32chips[i][j],
                        w3_32chips[i][j],
                        output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
                    )
                )
                w1_32chips[i][j].deallocate(True)
                w3_32chips[i][j].deallocate(True)
            hidden_states_32chips.append(hidden_states_4chips)

        # Flatten the original 8x4 2D list into a 1D list with chip 0-31
        hidden_states_32chips = [chip for column_chips in hidden_states_32chips for chip in column_chips]

        # Transform the flattened list into the 4x8 2D list for FF2 matmuls
        hidden_states_32chips = [
            [hidden_states_32chips[i + j * self.frac_grid[0]] for j in range(self.frac_grid[1])]
            for i in range(self.frac_grid[0])
        ]

        for i in range(len(hidden_states_32chips)):
            for j in range(len(hidden_states_32chips[i])):
                device_id = self.FF2_groups[i][j]
                # logger.info(f"FF2 matmul on device {device_id} for chips {self.FF2_groups[i]}")
                hidden_states_32chips[i][j] = tt_lib.operations.primary.matmul(
                    hidden_states_32chips[i][j],
                    self.w2_list[device_id],
                    program_config=self.model_config["PADDED_FF2_MM_PROGCFG"],
                    output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
                    output_dtype=self.model_config["PADDED_FF2_MM_OUTPUT_DTYPE"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )

        for i in range(len(hidden_states_32chips)):
            for j in range(len(hidden_states_32chips[i])):
                hidden_states_32chips[i][j] = tt_lib.tensor.sharded_to_interleaved(
                    hidden_states_32chips[i][j], output_mem_config=self.model_config["L1_MEMCFG"]
                )

        if self.emulated:
            for i in range(len(hidden_states_32chips)):
                # logger.info(f"Final All-Reduce for chips {self.FF2_groups[i]}")
                hidden_states_32chips[i] = tt_all_reduce(
                    hidden_states_32chips[i],
                )

        # Select the first chip of each column to get the full output
        # hidden_states_width_sharded_by_4 = [chip_column[0] for chip_column in hidden_states_32chips]
        # return hidden_states_width_sharded_by_4
        # TODO: Do a all_gather along x axis to let every chip have a full activation?
        hidden_states_32chips = [chip for column_chips in hidden_states_32chips for chip in column_chips]

        # Transform back to the original pattern
        hidden_states_32chips = [hidden_states_32chips[i :: self.frac_grid[1]] for i in range(self.frac_grid[1])]

        if self.emulated:
            for i in range(len(hidden_states_32chips)):
                hidden_states_32chips[i] = tt_all_gather_torch(hidden_states_32chips[i], dim=-1)

        hidden_states_32chips = [chip for column_chips in hidden_states_32chips for chip in column_chips]

        if self.emulated:
            # FOR BRINGUP! Outputs are Interaved, Shard them
            for i in range(len(hidden_states_32chips)):
                hidden_states_32chips[i] = tt_lib.tensor.interleaved_to_sharded(
                    hidden_states_32chips[i], sharded_mem_config=self.model_config["LN_MLP_OUTPUT_MEMCFG"]
                )

        return hidden_states_32chips

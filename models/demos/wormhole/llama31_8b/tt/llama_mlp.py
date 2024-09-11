# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtLlamaMLP(torch.nn.Module):
    def __init__(
        self,
        device,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.args = args
        self.model_config = model_config

        base_name = f"layers.{layer_num}.feed_forward"
        torch_weight = lambda name: torch.transpose(self.state_dict[f"{base_name}.{name}.weight"], -2, -1)
        cache_name = lambda name: weight_cache_path / (base_name + f".{name}")
        as_tensor = lambda name, type: ttnn.as_tensor(
            torch_weight(name),
            dtype=type,
            device=self.device,
            layout=self.model_config["MLP_W_LAYOUT_TILE"],
            memory_config=self.model_config["MLP_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name(name),
        )

        self.w1 = as_tensor("w1", ttnn.bfloat8_b)  # bfp4 normally ok here but sub .99 pcc for llama 3.1 weights
        self.w2 = as_tensor("w2", ttnn.bfloat8_b)
        self.w3 = as_tensor("w3", ttnn.bfloat8_b)  # same here

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        seq_len = x.shape[-2]
        compute_kernel_config = self.model_config["MLP_KERNEL_CONFIG"]
        if seq_len >= 1024:  # Too big to compute. Set different program configs based on seqlen
            # Reshape input to to fit on device and parallelize computation
            x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])
            pc_1 = self.model_config["PREFILL_MLP_W1_PRG_CONFIG"]
            pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG"]
            pc_3 = self.model_config["PREFILL_MLP_W3_PRG_CONFIG"]
        else:
            pc_1 = self.model_config["PREFILL_MLP_W1_PRG_CONFIG_128"](seq_len)
            pc_2 = self.model_config["PREFILL_MLP_W2_PRG_CONFIG_128"](seq_len)
            pc_3 = self.model_config["PREFILL_MLP_W3_PRG_CONFIG_128"](seq_len)

        w1_out = ttnn.linear(
            x,
            self.w1,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            dtype=ttnn.bfloat16,
            activation="silu" if not pc_1 else None,
            program_config=pc_1,
            memory_config=ttnn.L1_MEMORY_CONFIG if seq_len <= 32 else ttnn.DRAM_MEMORY_CONFIG,
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            dtype=ttnn.bfloat16,
            program_config=pc_3,
            memory_config=ttnn.L1_MEMORY_CONFIG if seq_len <= 32 else ttnn.DRAM_MEMORY_CONFIG,
        )

        # x.deallocate(True)
        w2_in = ttnn.multiply(w1_out, w3_out)

        w3_out.deallocate(True)
        w1_out.deallocate(True)

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
            dtype=ttnn.bfloat8_b,
            program_config=pc_2,
            memory_config=ttnn.L1_MEMORY_CONFIG if seq_len <= 32 else ttnn.DRAM_MEMORY_CONFIG,
        )

        w2_in.deallocate(True)

        if seq_len >= 2048:  # Reshape back to intended shape
            w2_out = ttnn.reshape(w2_out, [1, 1, seq_len, -1])

        return w2_out

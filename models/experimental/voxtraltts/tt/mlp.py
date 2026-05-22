# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class VoxtralTTMLP:
    """Voxtral SwiGLU MLP: ``ttnn.linear`` + SiLU-gated path (shared by text, acoustic FM, audio tokenizer blocks).

    Instantiate with the checkpoint key prefix for ``w1``/``w2``/``w3``; hidden/intermediate sizes follow weights.
    """

    def __init__(
        self,
        device,
        state_dict: dict[str, torch.Tensor],
        w1_key: str,
        w2_key: str,
        w3_key: str,
        weight_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        exact_silu: bool = False,
        compute_kernel_config=None,
        activation_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> None:
        self.device = device
        self.output_dtype = output_dtype
        self.exact_silu = exact_silu
        self.compute_kernel_config = compute_kernel_config
        self.activation_memory_config = activation_memory_config

        def get_weight(key: str) -> torch.Tensor:
            if key in state_dict:
                return state_dict[key]
            if f"{key}.weight" in state_dict:
                return state_dict[f"{key}.weight"]
            raise KeyError(f"Missing MLP weight for key '{key}'")

        w1 = get_weight(w1_key).transpose(-2, -1).contiguous()
        w2 = get_weight(w2_key).transpose(-2, -1).contiguous()
        w3 = get_weight(w3_key).transpose(-2, -1).contiguous()

        self.w1 = ttnn.from_torch(
            w1,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.w2 = ttnn.from_torch(
            w2,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.w3 = ttnn.from_torch(
            w3,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor, *, activation_memory_config=None) -> ttnn.Tensor:
        act_mem = activation_memory_config or self.activation_memory_config
        _lin_kw = {"dtype": self.output_dtype, "memory_config": act_mem}
        if self.compute_kernel_config is not None:
            _lin_kw["compute_kernel_config"] = self.compute_kernel_config
        w1_out = ttnn.linear(x, self.w1, **_lin_kw)
        w3_out = ttnn.linear(x, self.w3, **_lin_kw)

        if self.exact_silu:
            w1_act = ttnn.silu(w1_out, memory_config=act_mem)
            ttnn.deallocate(w1_out)
            w2_in = ttnn.mul(
                w1_act,
                w3_out,
                memory_config=act_mem,
                dtype=self.output_dtype,
            )
            ttnn.deallocate(w1_act)
        else:
            w2_in = ttnn.mul(
                w1_out,
                w3_out,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=act_mem,
                dtype=self.output_dtype,
            )
            ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        out = ttnn.linear(w2_in, self.w2, **_lin_kw)
        ttnn.deallocate(w2_in)
        return out

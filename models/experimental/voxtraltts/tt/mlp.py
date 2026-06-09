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
        ff1_3_program_config=None,
        ff2_program_config=None,
        w1_mem_config=None,
        w2_mem_config=None,
        w3_mem_config=None,
        ff1_3_in0_shard_mem_config=None,
    ) -> None:
        self.device = device
        self.output_dtype = output_dtype
        self.exact_silu = exact_silu
        self.compute_kernel_config = compute_kernel_config
        self.activation_memory_config = activation_memory_config
        self.ff1_3_program_config = ff1_3_program_config
        self.ff2_program_config = ff2_program_config
        self.ff1_3_in0_shard_mem_config = ff1_3_in0_shard_mem_config

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
            memory_config=w1_mem_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        self.w2 = ttnn.from_torch(
            w2,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        self.w3 = ttnn.from_torch(
            w3,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w3_mem_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x: ttnn.Tensor, *, activation_memory_config=None) -> ttnn.Tensor:
        act_mem = activation_memory_config or self.activation_memory_config

        # w1/w3: use width-sharded output when a 1D-mcast program config is provided so
        # per-core work increases (fewer cores, better arithmetic intensity for M=1 tile).
        has_ff1_3_prg = self.ff1_3_program_config is not None
        ff1_3_out_mem = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if has_ff1_3_prg else act_mem

        _ff1_3_kw = {"dtype": self.output_dtype, "memory_config": ff1_3_out_mem}
        if self.compute_kernel_config is not None:
            _ff1_3_kw["compute_kernel_config"] = self.compute_kernel_config
        if has_ff1_3_prg:
            _ff1_3_kw["program_config"] = self.ff1_3_program_config

        _ff2_kw = {"dtype": self.output_dtype, "memory_config": act_mem}
        if self.compute_kernel_config is not None:
            _ff2_kw["compute_kernel_config"] = self.compute_kernel_config
        if self.ff2_program_config is not None:
            _ff2_kw["program_config"] = self.ff2_program_config

        # For DS matmuls, in0 must be L1 K-width-sharded; shard x then discard the copy.
        if self.ff1_3_in0_shard_mem_config is not None:
            x_in = ttnn.to_memory_config(x, self.ff1_3_in0_shard_mem_config)
        else:
            x_in = x

        w1_out = ttnn.linear(x_in, self.w1, **_ff1_3_kw)
        w3_out = ttnn.linear(x_in, self.w3, **_ff1_3_kw)

        if x_in is not x:
            ttnn.deallocate(x_in)

        if self.exact_silu:
            w1_act = ttnn.silu(w1_out, memory_config=ff1_3_out_mem)
            ttnn.deallocate(w1_out)
            w2_in = ttnn.mul(
                w1_act,
                w3_out,
                memory_config=ff1_3_out_mem,
                dtype=self.output_dtype,
            )
            ttnn.deallocate(w1_act)
        else:
            w2_in = ttnn.mul(
                w1_out,
                w3_out,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=ff1_3_out_mem,
                dtype=self.output_dtype,
            )
            ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # De-shard to L1-interleaved before w2: mcast_in0 requires interleaved in0.
        if has_ff1_3_prg:
            w2_in_l1 = ttnn.to_memory_config(w2_in, act_mem)
            ttnn.deallocate(w2_in)
            w2_in = w2_in_l1

        out = ttnn.linear(w2_in, self.w2, **_ff2_kw)
        ttnn.deallocate(w2_in)
        return out

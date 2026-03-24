# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SwiGLU MLP for Molmo2 Text Model.

Implements SwiGLU activation:
    output = ff_out(silu(gate) * up)

Where gate and up are computed from a fused ff_proj:
    ff_proj = [gate; up]  (concatenated along output dim)

Dimensions:
    - hidden_dim: 4096
    - intermediate_dim: 11008 (per gate/up branch)

Supports tensor parallelism across multiple devices:
    - gate_proj, up_proj: Column parallel (shard output dim)
    - down_proj: Row parallel (shard input dim, all-reduce output)
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TextMLP(LightweightModule):
    """
    SwiGLU MLP for Molmo2 text model.

    Architecture:
        - ff_proj: fused gate+up projection [hidden_dim -> 2*intermediate_dim]
        - ff_out: down projection [intermediate_dim -> hidden_dim]
        - Output: ff_out(silu(gate) * up)

    Tensor Parallelism:
        - gate_proj, up_proj: Each device computes a slice of intermediate dim
        - down_proj: Each device has a row slice, outputs are all-reduced
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        layer_num: int,
        hidden_dim: int = 4096,
        intermediate_dim: int = 12288,
        weight_cache_path=None,
        state_dict_prefix: str = "model.transformer.blocks",
        dtype=ttnn.bfloat8_b,
    ):
        """
        Initialize TextMLP.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            layer_num: Layer number (0-35)
            hidden_dim: Hidden dimension (4096)
            intermediate_dim: Intermediate dimension per branch (11008)
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.layer_num = layer_num

        # Layer prefix
        prefix = f"{state_dict_prefix}.{layer_num}.mlp"

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{prefix}.{name}"

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.is_mesh_device = is_mesh_device

        if is_mesh_device:
            self.num_devices = mesh_device.get_num_devices()
            # Column parallel for gate/up: shard output dimension
            col_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
            # Row parallel for down: shard input dimension
            row_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
        else:
            self.num_devices = 1
            col_mesh_mapper = None
            row_mesh_mapper = None

        # For multi-device sharding: store bfloat16 directly (fits when sharded)
        # For single device: store bfloat8_b + CPU copies for on-demand conversion (bfloat16 doesn't fit)
        use_sharded_bf16 = is_mesh_device and self.num_devices > 1

        # Load fused ff_proj: [hidden_dim -> 2*intermediate_dim]
        ff_proj = state_dict[f"{prefix}.ff_proj.weight"]

        # Split into gate and up projections
        # ff_proj shape: [2*intermediate_dim, hidden_dim]
        # HuggingFace order: first half is GATE, second half is UP
        # Output: silu(gate) * up
        up_proj = ff_proj[:intermediate_dim, :]
        gate_proj = ff_proj[intermediate_dim:, :]

        # Transpose for TTNN linear: [1, 1, hidden_dim, intermediate_dim]
        gate_proj_t = torch.transpose(gate_proj, -2, -1).unsqueeze(0).unsqueeze(0)
        up_proj_t = torch.transpose(up_proj, -2, -1).unsqueeze(0).unsqueeze(0)

        # For multi-device: store weights as bfloat16 (sharded across devices, fits in memory)
        # For single device: store as bfloat8_b + CPU copies for on-demand conversion
        # bfloat8_b causes numerical overflow during decode, but bfloat16 doesn't fit in single device
        if use_sharded_bf16:
            # Store directly as bfloat16, sharded across devices
            self.gate_proj = ttnn.from_torch(
                gate_proj_t.to(torch.bfloat16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=col_mesh_mapper,
            )
            self.up_proj = ttnn.from_torch(
                up_proj_t.to(torch.bfloat16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=col_mesh_mapper,
            )
            self._gate_proj_cpu = None
            self._up_proj_cpu = None
        else:
            # Single device: store as bfloat8_b + keep CPU copies for decode conversion
            self._gate_proj_cpu = gate_proj_t.clone()
            self._up_proj_cpu = up_proj_t.clone()
            self._col_mesh_mapper = col_mesh_mapper

            self.gate_proj = ttnn.as_tensor(
                gate_proj_t,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=col_mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name("gate_proj.weight"),
            )
            self.up_proj = ttnn.as_tensor(
                up_proj_t,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=col_mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name("up_proj.weight"),
            )

        # Load ff_out (down projection): [intermediate_dim, hidden_dim]
        # Transpose: [1, 1, intermediate_dim, hidden_dim]
        ff_out = state_dict[f"{prefix}.ff_out.weight"]
        ff_out_t = torch.transpose(ff_out, -2, -1).unsqueeze(0).unsqueeze(0)

        if use_sharded_bf16:
            self.down_proj = ttnn.from_torch(
                ff_out_t.to(torch.bfloat16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=row_mesh_mapper,
            )
            self._down_proj_cpu = None
        else:
            self._down_proj_cpu = ff_out_t.clone()
            self._row_mesh_mapper = row_mesh_mapper

            self.down_proj = ttnn.as_tensor(
                ff_out_t,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=row_mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name("down_proj.weight"),
            )

        # Compute kernel config - use HiFi4 with fp32 accumulation for numerical stability
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through SwiGLU MLP.

        Args:
            x: Input tensor of shape [1, 1, seq_len, hidden_dim]

        Returns:
            Output tensor of shape [1, 1, seq_len, hidden_dim]

        Note:
            With tensor parallelism:
            - Input x is replicated across devices
            - gate_proj/up_proj compute partial intermediate activations
            - down_proj computes partial outputs
            - All-reduce combines partial outputs
        """
        # For single device, convert weights to bfloat16 on-demand to prevent overflow
        if self._gate_proj_cpu is not None:
            gate_proj = ttnn.from_torch(
                self._gate_proj_cpu.to(torch.bfloat16),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._col_mesh_mapper,
            )
            up_proj = ttnn.from_torch(
                self._up_proj_cpu.to(torch.bfloat16),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._col_mesh_mapper,
            )
            down_proj = ttnn.from_torch(
                self._down_proj_cpu.to(torch.bfloat16),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._row_mesh_mapper,
            )
        else:
            gate_proj = self.gate_proj
            up_proj = self.up_proj
            down_proj = self.down_proj

        # Gate projection with SiLU activation
        gate = ttnn.linear(
            x,
            gate_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self._gate_proj_cpu is not None:
            ttnn.deallocate(gate_proj)
        gate = ttnn.silu(gate)

        # Up projection
        up = ttnn.linear(
            x,
            up_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self._up_proj_cpu is not None:
            ttnn.deallocate(up_proj)

        # Element-wise multiply: silu(gate) * up
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Down projection
        output = ttnn.linear(
            hidden,
            down_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self._down_proj_cpu is not None:
            ttnn.deallocate(down_proj)
        ttnn.deallocate(hidden)

        # All-reduce for tensor parallelism (sum partial outputs from all devices)
        if self.is_mesh_device and self.num_devices > 1:
            output = ttnn.all_reduce(
                output,
                cluster_axis=1,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return output

    def forward_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Optimized decode forward pass using L1 memory.

        For decode mode, tensors are small enough to fit in L1, which is faster than DRAM.

        Args:
            x: Input tensor of shape [1, 1, 1, hidden_dim]

        Returns:
            Output tensor of shape [1, 1, 1, hidden_dim]
        """
        # Use pre-loaded weights (multi-device case with bfloat16)
        gate_proj = self.gate_proj
        up_proj = self.up_proj
        down_proj = self.down_proj

        # Gate projection with SiLU activation - L1 output
        gate = ttnn.linear(
            x,
            gate_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        gate = ttnn.silu(gate)

        # Up projection - L1 output
        up = ttnn.linear(
            x,
            up_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        # Element-wise multiply: silu(gate) * up - L1 output
        hidden = ttnn.mul(gate, up, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Down projection - L1 output
        output = ttnn.linear(
            hidden,
            down_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)

        # All-reduce for tensor parallelism (sum partial outputs from all devices)
        # Note: all_reduce needs DRAM input, convert from L1 first
        if self.is_mesh_device and self.num_devices > 1:
            output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
            output = ttnn.all_reduce(
                output,
                cluster_axis=1,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return output

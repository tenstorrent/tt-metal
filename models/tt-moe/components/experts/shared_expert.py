# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared expert implementation for DeepSeek-V3 style MoE.

This implementation is based on models/demos/deepseek_v3/tt/mlp/shared_expert.py
which inherits from MLPDequant (which inherits from MLP).
"""

from itertools import takewhile
from typing import Any

import torch

import ttnn

try:
    from ...utils.debug_logger import log_tensor_props
except ImportError:
    try:
        from models.tt_moe.utils.debug_logger import log_tensor_props
    except ImportError:
        from utils.debug_logger import log_tensor_props

try:
    from .base_expert import BaseExpert
except ImportError:
    from components.experts.base_expert import BaseExpert


# Helper functions from reference implementation for exact compatibility
def even_int_div(a: int, b: int) -> int:
    """Integer division that raises an error if b does not divide a without a remainder."""
    assert a % b == 0, f"{a} must be divisible by {b}"
    return a // b


def find_largest_divisor(n: int, max_val: int = 8) -> int:
    """Find the largest divisor of n that is <= max_val."""
    for divisor in range(min(n, max_val), 0, -1):
        if n % divisor == 0:
            return divisor
    return 1


def find_all_divisors(n):
    """Find all divisors of n."""
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def get_activation_sharding_core_counts_for_dram_matmul(activation_width: int, max_num_cores: int) -> set[int]:
    """
    Get the set of valid core counts for sharding activations in DRAM matmul.
    The activation_width is the width of the tensor in tiles.
    """
    return set(
        takewhile(lambda x: x <= max_num_cores, find_all_divisors(ttnn.core.divup(activation_width, ttnn.TILE_SIZE)))
    )


def get_dram_sharded_matmul_config(m: int, k: int, n: int, input_num_shards: int, output_num_shards: int):
    """Generate DRAM sharded matmul config matching reference implementation."""
    m_tiles = ttnn.core.divup(m, ttnn.TILE_SIZE)
    k_tiles = ttnn.core.divup(k, ttnn.TILE_SIZE)
    n_tiles = ttnn.core.divup(n, ttnn.TILE_SIZE)

    assert (
        k_tiles % input_num_shards == 0
    ), "The input tensor must evenly shard across input_num_shards (without padding)"
    assert (
        n_tiles % output_num_shards == 0
    ), "The output tensor must evenly shard across output_num_shards (without padding)"

    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=find_largest_divisor(
            even_int_div(k_tiles, input_num_shards)
        ),  # in0_block_w has to divide k_tiles evenly
        per_core_M=m_tiles,
        per_core_N=even_int_div(n_tiles, output_num_shards),
        fused_activation=None,
    )


class SharedExpert(BaseExpert):
    """
    Shared expert that processes all tokens in parallel with MoE experts.

    This expert type is used in DeepSeek-V3 where a dense FFN runs in
    parallel with the sparse MoE experts, and outputs are added together.

    This implementation is based on the DeepSeek demo's SharedExpert which:
    - Inherits from MLPDequant for weight quantization support
    - Uses moe_intermediate_size instead of intermediate_size
    - Includes all optimizations from the base MLP class
    """

    # Additional weight configuration (extends base)
    WEIGHT_TORCH_DTYPE = torch.float8_e4m3fn
    WEIGHT_DTYPE_SHARED = ttnn.bfloat4_b  # For quantized weights in shared expert
    SEQ_LEN_CHUNK_SIZE = 16384  # Maximum chunk size for prefill
    USERS_PER_ROW = 32  # Batch size per device in decode mode (from reference)

    def __init__(self, config: dict, mesh_device: ttnn.MeshDevice):
        """
        Initialize shared expert.

        Args:
            config: Expert configuration containing:
                - intermediate_size: FFN intermediate dimension (moe_intermediate_size in HF config)
                - hidden_size: Model hidden dimension
                - memory_config: Memory configuration string
                - use_quantized_weights: Whether to use quantized weights
                - weight_block_size: Quantization block size [height, width]
            mesh_device: TTNN mesh device
        """
        # Initialize base class - handles common config
        super().__init__(config, mesh_device)

        # Calculate device metrics for shared expert specific operations
        self.mesh_height, self.mesh_width = mesh_device.shape
        self.max_num_cores = mesh_device.core_grid.x * mesh_device.core_grid.y
        self.matmul_core_grid_size = ttnn.CoreCoord(
            mesh_device.core_grid.x,
            mesh_device.core_grid.y,
        )

        # Shared expert weights (will be loaded later)
        self.gate_proj = None  # w1
        self.up_proj = None  # w3
        self.down_proj = None  # w2

        # Scale tensors for quantized weights
        self.gate_proj_scale_inv = None
        self.up_proj_scale_inv = None
        self.down_proj_scale_inv = None

    def _get_prefill_pc(self, seq_len: int, is_down_proj: bool) -> Any:
        """Get the program config for linear layers in prefill mode based on sequence length."""
        if is_down_proj:
            # Down projection: [intermediate_size/tp] -> [hidden_size]
            per_device_in_features = self.intermediate_size // self.mesh_width
            per_device_out_features = self.hidden_size
        else:
            # Gate/Up projections: [hidden_size] -> [intermediate_size/tp]
            per_device_in_features = self.hidden_size
            per_device_out_features = self.intermediate_size // self.mesh_width

        per_core_M_tiles = ttnn.core.divup(seq_len, ttnn.TILE_SIZE * self.matmul_core_grid_size.y)
        K_tiles = ttnn.core.divup(per_device_in_features, ttnn.TILE_SIZE)
        per_core_N_tiles = ttnn.core.divup(per_device_out_features, ttnn.TILE_SIZE * self.matmul_core_grid_size.x)

        # Find largest divisor for blocking
        def find_largest_divisor(n, max_val=8):
            for divisor in range(min(n, max_val), 0, -1):
                if n % divisor == 0:
                    return divisor
            return 1

        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=self.matmul_core_grid_size,
            in0_block_w=find_largest_divisor(K_tiles),
            out_subblock_h=1,
            out_subblock_w=find_largest_divisor(per_core_N_tiles, 4),
            per_core_M=per_core_M_tiles,
            per_core_N=per_core_N_tiles,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    def _get_decode_memory_config(self, per_device_width: int, num_cores: int) -> ttnn.MemoryConfig:
        """Get memory config for decode mode activations."""
        return ttnn.create_sharded_memory_config_(
            shape=(
                ttnn.core.roundup(self.USERS_PER_ROW, ttnn.TILE_SIZE),
                ttnn.core.roundup(per_device_width // num_cores, ttnn.TILE_SIZE),
            ),
            core_grid=ttnn.num_cores_to_corerangeset(
                num_cores,
                ttnn.CoreCoord(self.mesh_device.core_grid.x, self.mesh_device.core_grid.y),
                row_wise=True,
            ),
            strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )

    def _silu_workaround(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Workaround for the silu PCC issue in ttnn.
        Implements silu(x) = x * sigmoid(x) using explicit operations.
        """
        # -x
        x1 = ttnn.neg(x)

        # 1
        x2 = ttnn.ones_like(x)

        # exp(-x)
        x3 = ttnn.exp(x1)
        ttnn.deallocate(x1)

        # 1 + exp(-x)
        x4 = ttnn.add(x3, 1)
        ttnn.deallocate(x3)

        # 1 / (1 + exp(-x))  [this is sigmoid(x)]
        x5 = ttnn.div(x2, x4)
        ttnn.deallocate(x2)
        ttnn.deallocate(x4)

        # x * sigmoid(x)
        x6 = ttnn.mul(x, x5)
        ttnn.deallocate(x5)

        return x6

    def load_weights(self, state_dict: dict, weight_path: str = None):
        """
        Load shared expert FFN weights.

        Args:
            state_dict: Dictionary containing:
                - gate_proj.weight: Gate projection [intermediate_size, hidden_size]
                - up_proj.weight: Up projection [intermediate_size, hidden_size]
                - down_proj.weight: Down projection [hidden_size, intermediate_size]
                - Optional scale_inv tensors for quantization
            weight_path: Optional path for cached weights
        """

        # Helper function to handle quantization if needed
        def load_weight_with_optional_quantization(weight_name: str, transpose: bool = True):
            """Load weight with optional dequantization."""
            weight = state_dict[f"{weight_name}.weight"]
            was_dequantized = False

            # Check if we have quantization scale
            scale_name = f"{weight_name}.weight_scale_inv"
            if self.use_quantized_weights and scale_name in state_dict:
                scale_inv = state_dict[scale_name]
                # Dequantize the weight
                weight = self._dequantize(weight, scale_inv, self.weight_block_size)
                # Store scale for potential future use
                setattr(self, f"{weight_name.replace('.', '_')}_scale_inv", scale_inv)
                was_dequantized = True

            # Transpose for matmul if needed
            if transpose:
                weight = weight.T

            # Convert to TTNN tensor
            # Choose dtype based on weight state and quantization
            if was_dequantized:
                # If we dequantized, weights are now float32, use bfloat16
                dtype = ttnn.bfloat16
                weight = weight.to(torch.bfloat16)
            elif weight.dtype == torch.float8_e4m3fn:
                # If weight is already quantized but we couldn't dequantize (no scale),
                # convert to bfloat16 as a fallback (TTNN doesn't support direct float8 conversion)
                dtype = ttnn.bfloat16
                weight = weight.to(torch.bfloat16)
            elif self.use_quantized_weights:
                # If quantization is enabled but weight is not quantized, use quantized dtype
                dtype = self.WEIGHT_DTYPE_SHARED
            else:
                # Default to bfloat16
                dtype = ttnn.bfloat16
                if weight.dtype != torch.bfloat16:
                    weight = weight.to(torch.bfloat16)

            return ttnn.from_torch(
                weight.unsqueeze(0).unsqueeze(0),
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Load gate projection (w1 in MLP terminology)
        if "gate_proj.weight" in state_dict:
            self.gate_proj = load_weight_with_optional_quantization("gate_proj")

        # Load up projection (w3 in MLP terminology)
        if "up_proj.weight" in state_dict:
            self.up_proj = load_weight_with_optional_quantization("up_proj")

        # Load down projection (w2 in MLP terminology)
        if "down_proj.weight" in state_dict:
            self.down_proj = load_weight_with_optional_quantization("down_proj")

    def forward_prefill(self, x: ttnn.Tensor, cfg: dict = None) -> ttnn.Tensor:
        """
        Forward pass for prefill mode with optimizations.

        This follows the MLP.forward_prefill implementation with:
        - Large sequences via chunking
        - Dynamic program configs for matmuls
        - Optimized memory configurations

        Args:
            x: Input tensor [num_layers, 1, seq_len, hidden_dim]
            cfg: Optional runtime configuration

        Returns:
            Output tensor [num_layers, 1, seq_len, hidden_dim]
        """
        if cfg is None:
            cfg = {}

        log_tensor_props("SharedExpert prefill input", x)

        num_layers, _, seq_len, _ = x.shape
        original_seq_len = seq_len

        # Chunk the input if sequence is too long
        max_rows = cfg.get("max_rows", self.SEQ_LEN_CHUNK_SIZE)
        pad_rows = 0

        if seq_len > max_rows:
            # Pad to multiple of chunk size if needed
            if seq_len % max_rows != 0:
                pad_rows = max_rows - (seq_len % max_rows)
                x_padded = ttnn.pad(x, padding=((0, 0), (0, 0), (0, pad_rows), (0, 0)), value=0.0)
                ttnn.deallocate(x)
                x = x_padded
                seq_len += pad_rows

            # Reshape into chunks
            x = ttnn.reshape(x, [num_layers, seq_len // max_rows, max_rows, -1])
            seq_len = max_rows

        # Gate and up projections with dynamic program configs
        gate_pc = self._get_prefill_pc(seq_len=seq_len, is_down_proj=False)

        # w1 (gate) projection
        w1_out = ttnn.linear(
            x,
            self.gate_proj,
            program_config=gate_pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )

        # w3 (up) projection
        w3_out = ttnn.linear(
            x,
            self.up_proj,
            program_config=gate_pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        ttnn.deallocate(x)

        # Apply SwiGLU activation
        # Note: Using SILU activation on the gate output
        if cfg.get("use_silu_workaround", True):
            w1_out_activated = self._silu_workaround(w1_out)
            ttnn.deallocate(w1_out)
        else:
            # Use ttnn.mul with SILU activation fused
            w1_out_activated = w1_out

        # Multiply gate and up outputs
        activated = ttnn.mul(
            w1_out_activated,
            w3_out,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU] if not cfg.get("use_silu_workaround", True) else None,
        )
        if cfg.get("use_silu_workaround", True):
            ttnn.deallocate(w1_out_activated)
        else:
            ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Down projection (w2) with dynamic program config
        down_pc = self._get_prefill_pc(seq_len=seq_len, is_down_proj=True)
        output = ttnn.linear(
            activated,
            self.down_proj,
            program_config=down_pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        ttnn.deallocate(activated)

        # De-chunk the output if the input was chunked
        _, num_chunks, _, output_dim = output.shape
        if num_chunks > 1:
            output = ttnn.reshape(output, [num_layers, 1, -1, output_dim])
            if pad_rows > 0:
                output = ttnn.slice(output, [0, 0, 0, 0], [num_layers, 1, original_seq_len, output_dim])

        return output

    def forward_decode(self, x: ttnn.Tensor, cfg: dict = None) -> ttnn.Tensor:
        """
        Forward pass for decode mode with optimizations.

        Uses L1 width-sharded memory configurations for better performance.

        Args:
            x: Input tensor [batch, 1, hidden_dim]
            cfg: Optional runtime configuration

        Returns:
            Output tensor [batch, 1, hidden_dim]
        """
        if cfg is None:
            cfg = {}

        log_tensor_props("SharedExpert decode input", x)

        # Calculate sharding configuration exactly like reference MLP
        # Get dimensions
        dim = self.hidden_size
        hidden_dim = self.intermediate_size

        # Calculate device metrics
        max_num_cores = self.max_num_cores

        # Calculate core counts using reference method
        input_num_cores = max(get_activation_sharding_core_counts_for_dram_matmul(dim, max_num_cores))
        inner_num_cores = max(
            get_activation_sharding_core_counts_for_dram_matmul(
                even_int_div(hidden_dim, self.mesh_width), max_num_cores
            )
        )
        output_num_cores = max(
            get_activation_sharding_core_counts_for_dram_matmul(even_int_div(dim, self.mesh_width), max_num_cores)
        )

        # Gate projection (w1) - use simple config like distributed experts
        w1_out = ttnn.linear(
            x,
            self.gate_proj,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # Let TTNN choose program config automatically
            compute_kernel_config=self.compute_config,
        )

        # Up projection (w3) - use simple config like distributed experts
        w3_out = ttnn.linear(
            x,
            self.up_proj,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # Let TTNN choose program config automatically
            compute_kernel_config=self.compute_config,
        )

        # Apply SwiGLU activation with workaround
        w1_out_activated = self._silu_workaround(w1_out)
        ttnn.deallocate(w1_out)

        activated = ttnn.mul(
            w1_out_activated,
            w3_out,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(w1_out_activated)
        ttnn.deallocate(w3_out)

        # Down projection (w2) - use simple config like distributed experts
        output = ttnn.linear(
            activated,
            self.down_proj,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # Let TTNN choose program config automatically
            compute_kernel_config=self.compute_config,
        )
        ttnn.deallocate(activated)

        return output

    def forward(self, x: ttnn.Tensor, mode: str = "decode") -> ttnn.Tensor:
        """
        Forward pass through shared expert (dense FFN).

        This processes all tokens, unlike the sparse experts which only
        process routed tokens. This follows the MLP forward pattern.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim] or TP-sharded
            mode: "decode" or "prefill" mode

        Returns:
            Shared expert output [batch, seq_len, hidden_dim]
        """
        if mode == "prefill":
            return self.forward_prefill(x)
        else:
            return self.forward_decode(x)

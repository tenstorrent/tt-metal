from typing import Dict, Optional

import ttnn
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight
from models.experimental.tt_symbiote.core.module import deallocate_weights_after


class SmartTTNNLinear(TTNNLinear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)

        self.compute_kernel_config = compute_kernel_config()
        self.grid_size = self.device.compute_with_storage_grid_size() if self.device else None
        self._prefill_pc_cache: Dict[int, Optional[ttnn.MatmulMultiCoreReuseMultiCastProgramConfig]] = {}

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Dispatch to prefill or decode path based on input sequence length."""

        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        if len(input_shape) == 2:
            input_shape.insert(0, 1)
        if len(input_shape) == 3:
            input_shape.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        seq_len = int(input_shape[-2])
        mode = "decode" if seq_len <= 32 else "prefill"

        if self.grid_size is None:
            self.grid_size = self.device.compute_with_storage_grid_size()

        if mode == "decode":
            tt_output = self.decode_forward(input_tensor)
        else:
            tt_output = self.prefill_forward(input_tensor, seq_len)

        return ttnn.reshape(tt_output, input_tensor_shape[:-1] + [self.out_features])

    def _get_prefill_pc(self, seq_len: int) -> Optional[ttnn.MatmulMultiCoreReuseMultiCastProgramConfig]:
        if seq_len in self._prefill_pc_cache:
            return self._prefill_pc_cache[seq_len]

        if self.grid_size is None:
            self.grid_size = self.device.compute_with_storage_grid_size()

        if self.module_name == "lm_head":
            self._prefill_pc_cache[seq_len] = None
            return None

        program_config = get_prefill_pc(seq_len, self.in_features, self.out_features, self.grid_size)
        self._prefill_pc_cache[seq_len] = program_config
        return program_config

    def prefill_forward(self, input_tensor: ttnn.Tensor, seq_len: int) -> ttnn.Tensor:
        """Forward pass for prefill mode."""

        return ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=self.tt_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self._get_prefill_pc(seq_len),
        )

    def decode_forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass for decode mode."""
        # TODO: add dram program configuration and memory configuration for decode mode
        return ttnn.linear(
            input_tensor,
            self.tt_weight,
            bias=self.tt_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


class SmartTTNNLinearLLama(SmartTTNNLinear):
    """SmartTTNN Linear layer optimized for LLaMA models using bfloat8."""

    def preprocess_weights_impl(self):
        """Preprocess linear weights with bfloat8 precision."""
        self.tt_weight_host = preprocess_linear_weight(self.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(self.bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    @deallocate_weights_after
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with automatic weight deallocation."""
        return super().forward(input_tensor)


class SmartTTNNLinearLLamaBFloat16(SmartTTNNLinear):
    """TTNN Linear layer optimized for LLaMA models using bfloat16."""

    @deallocate_weights_after
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with automatic weight deallocation."""
        return super().forward(input_tensor)


########################################################################################
## Helper Functions
########################################################################################


def get_prefill_pc(
    M: int, K: int, N: int, core_grid_size: ttnn.CoreCoord
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """Get the program config for linear layers in prefill mode based on sequence length."""

    per_core_M_tiles = ttnn.core.divup(M, ttnn.TILE_SIZE * core_grid_size.y)
    K_tiles = ttnn.core.divup(K, ttnn.TILE_SIZE)
    per_core_N_tiles = ttnn.core.divup(N, ttnn.TILE_SIZE * core_grid_size.x)
    budget = 32
    max_divisor = max(1, min(8, budget // max(1, per_core_N_tiles)))
    out_subblock_h = 1

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=core_grid_size,
        in0_block_w=find_largest_divisor(K_tiles, max_divisor=max_divisor),
        out_subblock_h=out_subblock_h,
        out_subblock_w=get_out_subblock_w(
            per_core_N_tiles,
            out_subblock_h,
        ),
        per_core_M=per_core_M_tiles,
        per_core_N=per_core_N_tiles,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def find_largest_divisor(n, max_divisor=8):
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1  # Fallback to 1 if no divisor found


def get_out_subblock_w(per_core_N, out_subblock_h):
    """Calculate output subblock width for matmul config."""
    # Find the largest value <= 4 that evenly divides per_core_N
    max_val = 4 // out_subblock_h
    for i in range(max_val, 0, -1):
        if per_core_N % i == 0:
            return i
    return 1


def compute_kernel_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

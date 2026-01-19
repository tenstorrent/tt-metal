# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Flash Multi-Latent Attention (MLA) Decode operation.

This implements flash attention decode for MLA where K and V share the same tensor,
with V being the first head_dim_v elements of the KV cache.

This Python implementation matches the C++ sdpa_decode_program_factory.cpp exactly.
"""

import math
from dataclasses import dataclass

import torch

import ttnn


def pack_two_bfloat16_into_uint32(val: float) -> int:
    """Pack a float value into two bfloat16 values in a uint32."""
    import struct

    # Convert float to bfloat16 (truncate lower 16 bits of float32)
    float_bytes = struct.pack("f", val)
    float_int = struct.unpack("I", float_bytes)[0]
    bfloat16_val = float_int >> 16
    # Pack two bfloat16 values into uint32
    return (bfloat16_val << 16) | bfloat16_val


def float_to_uint32(val: float) -> int:
    """Convert float to uint32 representation."""
    import struct

    return struct.unpack("I", struct.pack("f", val))[0]


def get_noc_max_page_size() -> int:
    """Get NOC max page size for Blackhole architecture."""
    return 16384


def get_max_page_size_and_num_pages(noc_max_page_size: int, num_tiles: int, tile_size: int) -> tuple:
    """
    Calculate optimal page size and number of pages for NOC transfers.

    Returns (page_size, num_pages) where:
    - page_size is the largest multiple of tile_size that fits in NOC max
    - num_pages is total_size / page_size
    """
    total_size = num_tiles * tile_size

    # Calculate page size as largest multiple of tile_size that fits
    page_size = (noc_max_page_size // tile_size) * tile_size

    # Ensure total_size is divisible by page_size
    while total_size % page_size != 0 and page_size >= tile_size:
        page_size -= tile_size

    num_pages = total_size // page_size
    return page_size, num_pages


# =============================================================================
# Hard-coded S block definitions for SDPA compute grid
# NOTE: These layouts are optimal ONLY for NOC0. NOC1 has different optimal
# coordinates due to the mirrored coordinate system.
# =============================================================================


class FlashMLAOptimalGridNOC0:
    """
    S block grid layout for SDPA compute.
    IMPORTANT: These layouts are optimized for NOC0 only.

    S blocks layout:
    - S1, S2, S3, S4: Left side (columns 0-3)
    - S5, S6, S7, S8: Right side (columns 7-10)
    Each S block: 8 cores for seq len parallelism per Q head group

    DRAM bank mapping based on proximity:
    S1→1, S2→3, S3→2, S4→0, S5→5, S6→7, S7→6, S8→4
    """

    # S block definitions: (cores, optimal_dram_bank)
    # S1-S4: left side (cols 0-3), S5-S8: right side (cols 7-10)
    BLOCKS = (
        (((0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2)), 1),  # S1
        (((0, 3), (1, 3), (2, 3), (3, 3), (0, 4), (1, 4), (2, 4), (3, 4)), 3),  # S2
        (((0, 7), (1, 7), (2, 7), (3, 7), (0, 8), (1, 8), (2, 8), (3, 8)), 2),  # S3
        (((0, 9), (1, 9), (2, 9), (3, 9), (0, 0), (1, 0), (2, 0), (3, 0)), 0),  # S4
        (((7, 1), (8, 1), (9, 1), (10, 1), (7, 2), (8, 2), (9, 2), (10, 2)), 5),  # S5
        (((7, 4), (8, 4), (9, 4), (10, 4), (7, 5), (8, 5), (9, 5), (10, 5)), 7),  # S6
        (((7, 6), (8, 6), (9, 6), (10, 6), (7, 7), (8, 7), (9, 7), (10, 7)), 6),  # S7
        (((7, 9), (8, 9), (9, 9), (10, 9), (7, 0), (8, 0), (9, 0), (10, 0)), 4),  # S8
    )

    NUM_BLOCKS = len(BLOCKS)
    CORES_PER_BLOCK = len(BLOCKS[0][0])

    # Optimal DRAM bank order for KV cache sharding (matches S block work assignment)
    OPTIMAL_DRAM_BANK_ORDER = tuple(block[1] for block in BLOCKS)  # (1, 3, 2, 0, 5, 7, 6, 4)

    # Tree reduction order for final tail reduction across S blocks
    # Each step contains (dst_s_block_idx, src_s_block_idx) pairs that can run in parallel
    # Using 0-indexed S block indices: S1=0, S2=1, ..., S8=7
    # This reduces 8 S blocks to 1 in 3 steps (log2(8) = 3) instead of 7 sequential steps
    TREE_REDUCTION_ORDER = (
        # Step 1: Adjacent pairs reduce (4 parallel reductions)
        ((0, 1), (2, 3), (4, 5), (6, 7)),  # S2→S1, S4→S3, S6→S5, S8→S7
        # Step 2: Merge left pairs, merge right pairs (2 parallel reductions)
        ((0, 2), (4, 6)),  # S3→S1, S7→S5
        # Step 3: Final merge left and right (1 reduction)
        ((0, 4),),  # S5→S1
    )

    NUM_TREE_REDUCTION_STEPS = len(TREE_REDUCTION_ORDER)  # 3

    @classmethod
    def get_tree_reduction_role(cls, s_block_idx: int) -> list:
        """
        Get the tree reduction role for a given S block across all steps.

        Returns a list of (role, partner_s_block_idx) tuples for each step:
        - role: 'sender' if this S block sends to partner
                'receiver' if this S block receives from partner
                'idle' if this S block doesn't participate in this step
        - partner_s_block_idx: the S block index of the partner (-1 if idle)

        Example for S block 0 (S1):
        - Step 1: ('receiver', 1) - receives from S2
        - Step 2: ('receiver', 2) - receives from S3
        - Step 3: ('receiver', 4) - receives from S5
        """
        roles = []
        for step in cls.TREE_REDUCTION_ORDER:
            role = "idle"
            partner = -1
            for dst, src in step:
                if s_block_idx == dst:
                    role = "receiver"
                    partner = src
                    break
                elif s_block_idx == src:
                    role = "sender"
                    partner = dst
                    break
            roles.append((role, partner))
        return roles

    @classmethod
    def is_tree_reduction_receiver(cls, s_block_idx: int) -> bool:
        """Check if this S block receives from any partner in tree reduction."""
        for role, _ in cls.get_tree_reduction_role(s_block_idx):
            if role == "receiver":
                return True
        return False

    @classmethod
    def is_tree_reduction_sender(cls, s_block_idx: int) -> bool:
        """Check if this S block sends to any partner in tree reduction."""
        for role, _ in cls.get_tree_reduction_role(s_block_idx):
            if role == "sender":
                return True
        return False

    @classmethod
    def get_tree_reduction_partner_coords(cls, device, s_block_idx: int, batch_idx: int) -> list:
        """
        Get the physical NOC coordinates of tree reduction partners for a given S block and batch.

        Returns a list of (role_code, partner_s_block_idx, partner_x, partner_y) tuples for each step:
        - role_code: 0=idle, 1=sender, 2=receiver
        - partner_s_block_idx: S block index of partner (for checking if partner is active)
        - partner_x, partner_y: physical NOC coords of partner (0,0 if idle)
        """
        roles = cls.get_tree_reduction_role(s_block_idx)
        result = []
        for role, partner_s_block_idx in roles:
            if role == "idle" or partner_s_block_idx < 0:
                result.append((0, 0, 0, 0))  # idle: role=0, partner_idx=0, x=0, y=0
            else:
                # Get the partner core for this batch
                partner_cores = cls.get_cores(partner_s_block_idx)
                partner_x, partner_y = partner_cores[batch_idx]
                partner_physical = device.worker_core_from_logical_core(ttnn.CoreCoord(partner_x, partner_y))
                role_code = 1 if role == "sender" else 2
                result.append((role_code, partner_s_block_idx, partner_physical.x, partner_physical.y))
        return result

    @classmethod
    def optimal_dram_grid(cls) -> "ttnn.CoreRangeSet":
        """
        Get optimal DRAM CoreRangeSet for KV cache sharding.
        The order matches S block work assignment for optimal locality.
        """
        # DRAM banks map to coords (bank_id, 0) for 1D DRAM grid
        core_ranges = [
            ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0))
            for bank_id in cls.OPTIMAL_DRAM_BANK_ORDER
        ]
        return ttnn.CoreRangeSet(core_ranges)

    @classmethod
    def get_cores(cls, s_block_idx: int) -> tuple:
        """Get cores for S block at index."""
        return cls.BLOCKS[s_block_idx][0]

    @classmethod
    def output_cores(cls, s_block_idx: int, num_cores: int) -> tuple:
        """Get the first N cores from an S block (output/Q shard cores)."""
        return cls.BLOCKS[s_block_idx][0][:num_cores]

    @classmethod
    def physical_multicast_coords(cls, device, s_block_idx: int) -> tuple:
        """
        Get multicast NOC coordinates for an S block in PHYSICAL coordinates.
        Returns (start_x, start_y, end_x, end_y, num_mcast_dests).

        Uses first core as start and last core as end. NOC is a torus architecture
        so wraparound multicast (e.g., S4, S8) works correctly.
        """
        cores = cls.BLOCKS[s_block_idx][0]
        first_x, first_y = cores[0]
        last_x, last_y = cores[-1]

        first_logical = ttnn.CoreCoord(first_x, first_y)
        last_logical = ttnn.CoreCoord(last_x, last_y)
        first_physical = device.worker_core_from_logical_core(first_logical)
        last_physical = device.worker_core_from_logical_core(last_logical)

        num_mcast_dests = len(cores) - 1
        return (first_physical.x, first_physical.y, last_physical.x, last_physical.y, num_mcast_dests)


def get_interleaved_tensor_accessor_args(tensor):
    """
    Construct tensor accessor compile-time args for interleaved tensors (DRAM or L1).

    Returns [args_config] where args_config = IsDram (2) for DRAM, 0 for L1.
    """
    is_dram = tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    return [2 if is_dram else 0]


def get_tensor_accessor_args(tensor):
    """
    Construct tensor accessor compile-time args for any tensor (interleaved or sharded).

    Uses ttnn.TensorAccessorArgs to get proper args for the tensor's memory config.
    Returns list of compile-time args.
    """
    accessor_args = ttnn.TensorAccessorArgs(tensor)
    return accessor_args.get_compile_time_args()


@dataclass
class FlashMLAProgramConfig:
    """Program config for FlashMLADecode operation."""

    k_chunk_size: int = 128
    exp_approx_mode: bool = True
    grid: type = FlashMLAOptimalGridNOC0  # Grid layout class (NOC0 optimized by default)


class FlashMLADecode:
    """
    Flash Multi-Latent Attention Decode.

    This class implements flash attention decode operation optimized for
    Multi-Latent Attention (MLA) where K and V are stored together in
    a single KV cache tensor.

    Q: [1, batch, num_heads, kvpe_dim] - Query tensor
    KV Cache: [batch, 1, max_seq_len, kvpe_dim] - Combined KV cache
    V is the first head_dim_v elements of the kvpe_dim dimension.
    """

    # Program config class for this op
    ProgramConfig = FlashMLAProgramConfig

    @staticmethod
    def golden(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        position_ids: torch.Tensor,
        head_dim_v: int,
        scale: float,
    ) -> torch.Tensor:
        """
        PyTorch reference implementation for flash MLA decode.

        Args:
            q: Query tensor [1, batch_size, num_heads, kvpe_dim]
            kv_cache: KV cache tensor [batch_size, 1, max_seq_len, kvpe_dim]
            position_ids: Position indices [batch_size]
            head_dim_v: The value head dimension (first head_dim_v elements of kvpe_dim)
            scale: Attention scale factor

        Returns:
            Attention output [1, batch_size, num_heads, head_dim_v]
        """
        batch_size = q.shape[1]
        num_heads = q.shape[2]
        kvpe_dim = q.shape[3]

        # Reshape Q: [1, batch, num_heads, kvpe_dim] -> [batch, num_heads, 1, kvpe_dim]
        q = q.permute(1, 2, 0, 3)  # [batch, num_heads, 1, kvpe_dim]

        # Process each batch element with its own sequence length
        outputs = []
        for b in range(batch_size):
            seq_len = position_ids[b].item() + 1  # Attend to positions 0 to position_ids[b]

            # Get KV for this batch: [1, seq_len, kvpe_dim]
            kv = kv_cache[b, :, :seq_len, :]  # [1, seq_len, kvpe_dim]

            # Expand KV to num_heads: [num_heads, seq_len, kvpe_dim]
            kv_expanded = kv.expand(num_heads, seq_len, kvpe_dim)

            # Q for this batch: [num_heads, 1, kvpe_dim]
            q_b = q[b]  # [num_heads, 1, kvpe_dim]

            # Attention scores: Q @ K^T -> [num_heads, 1, seq_len]
            attn_scores = torch.matmul(q_b, kv_expanded.transpose(-2, -1)) * scale

            # Softmax
            attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(q.dtype)

            # V is the first head_dim_v elements of KV
            v = kv_expanded[:, :, :head_dim_v]  # [num_heads, seq_len, head_dim_v]

            # Output: attn_probs @ V -> [num_heads, 1, head_dim_v]
            out_b = torch.matmul(attn_probs, v)
            outputs.append(out_b)

        # Stack outputs: [batch, num_heads, 1, head_dim_v]
        output = torch.stack(outputs, dim=0)

        # Reshape to [1, batch, num_heads, head_dim_v]
        output = output.squeeze(2)  # [batch, num_heads, head_dim_v]
        output = output.unsqueeze(0)  # [1, batch, num_heads, head_dim_v]

        return output

    @staticmethod
    def op(
        q_tensor: ttnn.Tensor,
        kv_cache_tensor: ttnn.Tensor,
        head_dim_v: int,
        cur_pos_tensor: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        scale: float,
        program_config: "FlashMLAProgramConfig",
        compute_kernel_config: "ttnn.DeviceComputeKernelConfig",
    ) -> ttnn.Tensor:
        """
        Execute flash MLA decode operation using ttnn.generic_op.

        This implementation matches the C++ sdpa_decode_program_factory.cpp exactly.

        Args:
            q_tensor: Query tensor [1, batch, num_heads, kvpe_dim] (must be height sharded)
            kv_cache_tensor: KV cache tensor [batch, 1, max_seq_len, kvpe_dim]
                            MUST be ND sharded in DRAM with shard_height = k_chunk_size
            head_dim_v: The value head dimension (first head_dim_v elements of kvpe_dim)
            cur_pos_tensor: Current position tensor [batch] (int32)
            output_tensor: Pre-allocated output tensor [1, batch, num_heads, head_dim_v]
            scale: Attention scale factor
            program_config: FlashMLAProgramConfig with k_chunk_size and exp_approx_mode
            compute_kernel_config: ttnn.DeviceComputeKernelConfig

        Returns:
            Output tensor with attention result [1, batch, num_heads, head_dim_v]
        """
        # =========================================================================
        # Extract parameters (matching C++ lines 40-55)
        # =========================================================================
        input_tensor_q = q_tensor
        input_tensor_k = kv_cache_tensor  # MLA: V is first vDHt columns of K

        device = input_tensor_q.device()

        # Program config parameters
        k_chunk_size = program_config.k_chunk_size
        exp_approx_mode = program_config.exp_approx_mode
        grid = program_config.grid

        # Validate device has sufficient grid size for hard-coded S block layout
        # S blocks use columns 0-3 and 7-10 (11 cols), rows 0-9 (10 rows)
        device_grid = device.compute_with_storage_grid_size()
        assert device_grid.x >= 11, f"Device must have at least 11 columns, got {device_grid.x}"
        assert device_grid.y == 10, f"Device must have exactly 10 rows, got {device_grid.y}"

        # Compute kernel config
        math_fidelity = compute_kernel_config.math_fidelity
        math_approx_mode = compute_kernel_config.math_approx_mode
        fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en
        packer_l1_acc = compute_kernel_config.packer_l1_acc

        # =========================================================================
        # Shape extraction (matching C++ lines 70-114)
        # =========================================================================
        # Get tile dimensions from tensors (Q may use tiny tiles, K uses standard tiles)
        q_tile = input_tensor_q.get_tile()
        k_tile = input_tensor_k.get_tile()
        Q_TILE_HEIGHT = q_tile.tile_shape[0]
        K_TILE_HEIGHT = k_tile.tile_shape[0]
        TILE_WIDTH = 32  # Width is always 32

        q_shape = input_tensor_q.padded_shape
        k_shape = input_tensor_k.padded_shape

        # Q: [1, 1, total_q_heads, DH] - Q is height-sharded across output cores
        # K: [Bkv, NKV, S, DH]
        num_q_heads_per_core = input_tensor_q.memory_config().shard_spec.shape[0]  # Q heads per core from shard height
        total_q_heads = q_shape[2]
        S = k_shape[2]
        DH = k_shape[3]

        # Validate Q tensor assumptions
        assert q_shape[0] == 1, f"Q dim 0 must be 1, got {q_shape[0]}"
        assert q_shape[1] == 1, f"Q batch dim must be 1, got {q_shape[1]}"
        assert num_q_heads_per_core < 32, f"num_q_heads_per_core must be < 32, got {num_q_heads_per_core}"
        assert (
            total_q_heads % num_q_heads_per_core == 0
        ), f"total_q_heads ({total_q_heads}) must be divisible by num_q_heads_per_core ({num_q_heads_per_core})"

        # Validate KV cache is ND sharded (required for Deepseek V3 B1)
        kv_mem_config = kv_cache_tensor.memory_config()
        assert kv_mem_config.is_sharded(), "KV cache must be ND sharded (not interleaved)"
        assert (
            hasattr(kv_mem_config, "nd_shard_spec") and kv_mem_config.nd_shard_spec is not None
        ), "KV cache must use ND sharding with nd_shard_spec"

        # Validate k_chunk_size matches KV cache shard height
        kv_shard_shape = kv_mem_config.nd_shard_spec.shard_shape
        kv_shard_height = kv_shard_shape[2]  # Shape is [batch, nkv, seq_len, head_dim]
        assert kv_shard_height == k_chunk_size, (
            f"k_chunk_size ({k_chunk_size}) must match KV cache shard height ({kv_shard_height}). "
            f"Each KV shard should contain exactly one k_chunk."
        )

        # B = number of Q shards (output cores), each processing num_q_heads_per_core heads
        B = total_q_heads // num_q_heads_per_core

        num_kv_heads = k_shape[1]
        Bkv = k_shape[0]
        St = S // K_TILE_HEIGHT  # K/V use standard tile height
        DHt = DH // TILE_WIDTH
        vDHt = head_dim_v // TILE_WIDTH
        PNHt = num_q_heads_per_core // Q_TILE_HEIGHT  # Q uses its own tile height

        Sk_chunk_t = k_chunk_size // K_TILE_HEIGHT  # K chunks use K tile height

        # =========================================================================
        # Parallelization scheme - All S blocks for seq len parallelism
        # =========================================================================
        # S1 holds Q output cores (8 cores for 8 Q shards)
        # S2-S8 provide worker cores for sequence length parallelism
        # Each Q shard (batch) gets: 1 output core from S1 + 7 worker cores from S2-S8
        # Layout: Q1 uses S1[0], S2[0], S3[0], ..., S8[0]
        #         Q2 uses S1[1], S2[1], S3[1], ..., S8[1], etc.
        num_s_blocks = grid.NUM_BLOCKS  # 8 S blocks
        cores_per_s_block = grid.CORES_PER_BLOCK  # 8 cores per S block

        # Validate Q shards fit in one S block (max 8 Q shards)
        assert B <= cores_per_s_block, f"Too many Q shards ({B}), max is {cores_per_s_block}"

        # Calculate parallelization parameters
        # Each batch (Q shard) gets 8 cores: 1 from each S block
        num_cores_per_batch = num_s_blocks  # 8 cores per Q shard (seq len parallelism)
        num_active_cores = B * num_cores_per_batch  # Total active cores
        num_cores_per_head = num_cores_per_batch // num_kv_heads  # Cores per KV head
        num_heads_per_core = max(1, math.ceil(num_kv_heads / num_cores_per_batch))

        # Build all_cores list: for each batch, collect cores across all S blocks
        # This gives the interleaved layout needed for parallelization
        all_cores = []
        for batch_idx in range(B):
            for s_block_idx in range(num_s_blocks):
                x, y = grid.get_cores(s_block_idx)[batch_idx]
                all_cores.append((x, y))

        # Build core grid from all active cores
        core_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in all_cores]
        )

        # Create core group with the same layout
        core_group = [ttnn.CoreCoord(x, y) for x, y in all_cores]

        # Multicast: each S block's first core (Q1) reads KV and multicasts to others in that S block
        # Since all S blocks have 8 cores, num_mcast_dests is the same for all (7 = 8-1)
        num_mcast_dests = cores_per_s_block - 1  # 7 receivers per S block

        # Pre-compute physical multicast coordinates for each S block (used in runtime args)
        s_block_mcast_coords = []
        for s_idx in range(num_s_blocks):
            coords = grid.physical_multicast_coords(device, s_idx)
            s_block_mcast_coords.append(coords)

        # =========================================================================
        # CB tile counts (matching C++ lines 285-299)
        # =========================================================================
        dst_size = 4 if fp32_dest_acc_en else 8

        q_tiles = PNHt * DHt
        # Double buffer K for overlap between DRAM reads and compute.
        # Receivers signal sender when ready (CB reserved) to ensure consistent addresses.
        k_tiles = Sk_chunk_t * DHt * 2
        qk_tiles = PNHt * Sk_chunk_t
        out_im_tiles = PNHt * vDHt
        out0_t = PNHt * vDHt
        scale_tiles = 1
        statistics_tiles = PNHt

        # =========================================================================
        # Matmul configs (matching C++ lines 310-371)
        # =========================================================================
        qk_in0_block_w = DHt
        qk_num_blocks = DHt // qk_in0_block_w

        qk_out_subblock_w = min(Sk_chunk_t, dst_size)
        qk_out_subblock_h = min(PNHt, dst_size // qk_out_subblock_w) if qk_out_subblock_w == Sk_chunk_t else 1
        qk_in0_num_subblocks = PNHt // qk_out_subblock_h
        qk_in1_num_subblocks = Sk_chunk_t // qk_out_subblock_w

        out_in0_block_w = Sk_chunk_t
        out_num_blocks = Sk_chunk_t // out_in0_block_w

        out_out_subblock_w = min(vDHt, dst_size)
        out_out_subblock_h = min(PNHt, dst_size // out_out_subblock_w) if out_out_subblock_w == vDHt else 1
        out_in0_num_subblocks = PNHt // out_out_subblock_h
        out_in1_num_subblocks = vDHt // out_out_subblock_w

        dht_granularity = min(DHt, dst_size)
        log2_dht_granularity = int(math.log2(dht_granularity)) if dht_granularity > 0 else 0
        if dht_granularity != (1 << log2_dht_granularity):
            dht_granularity = 1
            log2_dht_granularity = 0

        # =========================================================================
        # Data formats (matching C++ lines 374-426)
        # =========================================================================
        q_df = input_tensor_q.dtype
        k_df = input_tensor_k.dtype
        out_df = output_tensor.dtype
        scalar_df = ttnn.bfloat16
        im_df = ttnn.bfloat16
        stats_df = ttnn.bfloat16

        # Create tile objects - Q uses tiny tiles, K/V use full tiles
        # Tiny tile height matches Q tile height (e.g., 8 for 8 heads per core)
        q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))
        full_tile = ttnn.Tile((K_TILE_HEIGHT, TILE_WIDTH))

        # All intermediate/stats/scalar tiles use tiny tile dimensions (same as Q)
        im_tile = q_tiny_tile
        stats_tile = q_tiny_tile
        scalar_tile = q_tiny_tile
        mask_tile = q_tiny_tile
        # K, col_identity use full tiles (V read from K directly)
        k_tile_obj = full_tile

        # Create tile descriptors for CB setup
        q_tile_descriptor = ttnn.TileDescriptor(q_tiny_tile)
        im_tile_descriptor = ttnn.TileDescriptor(im_tile)
        stats_tile_descriptor = ttnn.TileDescriptor(stats_tile)
        scalar_tile_descriptor = ttnn.TileDescriptor(scalar_tile)
        mask_tile_descriptor = ttnn.TileDescriptor(mask_tile)
        full_tile_descriptor = ttnn.TileDescriptor(full_tile)

        # Tile sizes - use tile.get_tile_size(dtype) for proper sizing
        q_tile_size = q_tiny_tile.get_tile_size(q_df)
        k_tile_size = k_tile_obj.get_tile_size(k_df)
        scalar_tile_size = scalar_tile.get_tile_size(scalar_df)
        im_tile_size = im_tile.get_tile_size(im_df)
        stats_tile_size = stats_tile.get_tile_size(stats_df)
        mask_tile_size = mask_tile.get_tile_size(im_df)
        col_identity_tile_size = full_tile.get_tile_size(scalar_df)

        # Intermediate output tiles for tree reduction
        # With tree reduction, senders can complete their steps out of order (e.g., S5 may send
        # in step 3 before S3 sends in step 2). To prevent data corruption, each tree reduction
        # step uses a separate buffer slot. This requires num_tree_reduction_steps * per_step_tiles.
        # Each transfer contains: output tiles (out0_t) + max stats (PNHt) + sum stats (PNHt)
        per_step_tiles = out0_t + 2 * PNHt
        intermed_output_tiles = per_step_tiles * grid.NUM_TREE_REDUCTION_STEPS

        # =========================================================================
        # DRAM Streaming Optimization: Calculate page size for K chunk reads
        # =========================================================================
        # K chunk tiles: Sk_chunk_t * DHt tiles per chunk
        k_chunk_tiles = Sk_chunk_t * DHt
        noc_max_page_size = get_noc_max_page_size()
        k_page_size, k_num_pages = get_max_page_size_and_num_pages(noc_max_page_size, k_chunk_tiles, k_tile_size)

        # KV cache is always ND sharded (validated above) - page-level pipelining enabled
        # Index stick size for position tensor (C++ lines 429-445)
        index_stick_size = B * 4  # int32 = 4 bytes per element
        index_stick_size = ((index_stick_size + 31) // 32) * 32  # Align to 32 bytes

        # =========================================================================
        # Packed constants (matching C++ lines 658-670)
        # =========================================================================
        packed_identity_scalar = pack_two_bfloat16_into_uint32(1.0)
        packed_zero_scalar = pack_two_bfloat16_into_uint32(0.0)
        scale_uint32 = float_to_uint32(scale)

        # =========================================================================
        # Create output core lists (matching C++ lines 671-721)
        # =========================================================================
        output_core_physical_xs = []
        output_core_physical_ys = []

        for i in range(num_active_cores):
            if i % num_cores_per_batch == 0:  # First core in batch group is output (S1)
                core_physical = device.worker_core_from_logical_core(core_group[i])
                output_core_physical_xs.append(core_physical.x)
                output_core_physical_ys.append(core_physical.y)

        # =========================================================================
        # Compile time args (simplified for Deepseek V3 B1)
        # Assumptions: PNHt=1, num_kv_heads=1, num_heads_per_core=1, Bkv=1
        # =========================================================================
        # Q is always tilized (TILE_LAYOUT)
        assert input_tensor_q.layout == ttnn.TILE_LAYOUT, "Q tensor must be TILE_LAYOUT (tilized)"
        assert PNHt == 1, f"PNHt must be 1, got {PNHt}"
        assert num_kv_heads == 1, f"num_kv_heads must be 1, got {num_kv_heads}"
        assert num_heads_per_core == 1, f"num_heads_per_core must be 1, got {num_heads_per_core}"
        assert Bkv == 1, f"Bkv must be 1, got {Bkv}"

        # Q chunk size bytes (PNHt=1, so q_tiles = DHt)
        q_chunk_size_bytes = q_tiles * q_tile_size

        # Reader compile time args (simplified - V is read from K by compute kernel)
        reader_compile_time_args = [
            St,  # 0: full sequence length in tiles
            DHt,  # 1: head dim in tiles (K width, V read as first vDHt columns)
            Sk_chunk_t,  # 2: tiles per K chunk
            num_cores_per_head,  # 3: cores for seq len parallelism (8)
            k_chunk_size,  # 4: K chunk size
            q_chunk_size_bytes,  # 5: Q chunk size in bytes
            num_mcast_dests,  # 6: multicast destinations (7)
            2,  # 7: mcast_semaphore_id
            k_page_size,  # 8: page size for DRAM streaming
            k_num_pages,  # 9: pages per K chunk
            3,  # 10: ncrisc_brisc_sync_semaphore_id
            4,  # 11: receiver_ready_semaphore_id (for double-buffer sync)
        ]
        # TensorAccessorArgs for K (KV cache) and pos tensor
        reader_compile_time_args.extend(get_tensor_accessor_args(kv_cache_tensor))  # K
        reader_compile_time_args.extend(
            get_interleaved_tensor_accessor_args(cur_pos_tensor)
        )  # pos (always DRAM interleaved)

        # Writer compile time args (simplified)
        writer_compile_time_args = [
            vDHt,  # 0: V head dim in tiles
            Sk_chunk_t,  # 1: tiles per K chunk
            packed_identity_scalar,  # 2
            packed_zero_scalar,  # 3
            num_cores_per_head,  # 4: cores for seq len parallelism (8)
            0,  # 5: reducer_semaphore_id
            k_chunk_size,  # 6: K chunk size
            Q_TILE_HEIGHT,  # 7: Q tile height
            DHt,  # 8: head dim in tiles
            num_mcast_dests,  # 9: multicast destinations (7)
            2,  # 10: mcast_semaphore_id
            3,  # 11: ncrisc_brisc_sync_semaphore_id
            k_page_size,  # 12: page size for pipelining
            k_num_pages,  # 13: pages per K chunk
            grid.NUM_TREE_REDUCTION_STEPS,  # 14: tree reduction steps (3)
            4,  # 15: receiver_ready_semaphore_id (for double-buffer sync)
        ]

        # Compute compile time args (keep existing interface for compute kernel)
        compute_compile_time_args = [
            St,  # 0
            DHt,  # 1
            vDHt,  # 2
            PNHt,  # 3
            Sk_chunk_t,  # 4
            qk_in0_block_w,  # 5
            qk_out_subblock_w,  # 6
            qk_out_subblock_h,  # 7
            qk_in0_num_subblocks,  # 8
            qk_in1_num_subblocks,  # 9
            qk_num_blocks,  # 10
            out_in0_block_w,  # 11
            out_out_subblock_w,  # 12
            out_out_subblock_h,  # 13
            out_in0_num_subblocks,  # 14
            out_in1_num_subblocks,  # 15
            out_num_blocks,  # 16
            num_cores_per_batch,  # 17
            k_chunk_size,  # 18
            num_cores_per_head,  # 19
            num_heads_per_core,  # 20 (always 1)
            B,  # 21: q_heads_parallel_factor
            Q_TILE_HEIGHT,  # 22: Q tile height
            scale_uint32,  # 23
            grid.NUM_TREE_REDUCTION_STEPS,  # 24: tree reduction steps (3)
        ]

        # Compute defines (C++ lines 844-892)
        sub_exp_granularity = min(Sk_chunk_t, dst_size)
        log2_sub_exp_granularity = int(math.log2(sub_exp_granularity))
        mul_bcast_granularity = min(PNHt * Sk_chunk_t, dst_size)
        log2_mul_bcast_granularity = int(math.log2(mul_bcast_granularity))
        stats_granularity = min(Sk_chunk_t, dst_size)
        log2_stats_granularity = int(math.log2(stats_granularity))

        compute_defines = [
            ("SUB_EXP_GRANULARITY", str(sub_exp_granularity)),
            ("LOG2_SUB_EXP_GRANULARITY", str(log2_sub_exp_granularity)),
            ("MUL_BCAST_GRANULARITY", str(mul_bcast_granularity)),
            ("LOG2_MUL_BCAST_GRANULARITY", str(log2_mul_bcast_granularity)),
            ("STATS_GRANULARITY", str(stats_granularity)),
            ("LOG2_STATS_GRANULARITY", str(log2_stats_granularity)),
            ("EXP_APPROX_MODE", "1" if exp_approx_mode else "0"),
            ("DHT_GRANULARITY", str(dht_granularity)),
            ("LOG2_DHT_GRANULARITY", str(log2_dht_granularity)),
        ]

        # =========================================================================
        # Create CB descriptors (matching C++ lines 475-655)
        # =========================================================================
        cb_descriptors = []

        # c_0: Q input (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=q_tiles * q_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(0, q_df, q_tile_size, q_tile_descriptor)],
            )
        )

        # c_1: K input (full tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=k_tiles * k_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(1, k_df, k_tile_size)],
            )
        )

        # V is read directly from K buffer (strided matmul) - no separate V CB needed

        # c_3: attn_mask input (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=qk_tiles * mask_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(3, im_df, mask_tile_size, mask_tile_descriptor)],
            )
        )

        # c_5: identity scale input (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=scale_tiles * scalar_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(5, scalar_df, scalar_tile_size, scalar_tile_descriptor)],
            )
        )

        # c_6: cb_m_in (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(6, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_7: cb_l_in (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(7, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_8: cur_pos input
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=index_stick_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(8, ttnn.int32, index_stick_size)],
            )
        )

        # c_11: cb_col_identity (full tile - always 32x32)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=scale_tiles * col_identity_tile_size,
                core_ranges=core_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(11, scalar_df, col_identity_tile_size, full_tile_descriptor)
                ],
            )
        )

        # c_12: cb zero config (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=scale_tiles * scalar_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(12, scalar_df, scalar_tile_size, scalar_tile_descriptor)],
            )
        )

        # c_24: cb_qk_im (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=qk_tiles * im_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(24, im_df, im_tile_size, im_tile_descriptor)],
            )
        )

        # c_25: cb_out_im (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=out_im_tiles * im_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(25, im_df, im_tile_size, im_tile_descriptor)],
            )
        )

        # c_26: cb_out_accumulate_im (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=out_im_tiles * im_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(26, im_df, im_tile_size, im_tile_descriptor)],
            )
        )

        # c_27: cb_cur_max (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(27, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_28: cb_prev_max (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(28, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_29: cb_cur_sum (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(29, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_30: cb_prev_sum (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(30, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_31: cb_exp_max_diff (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(31, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_21: cb_prev_sum_2 (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(21, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_22: cb_exp_max_diff_2 (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(22, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_23: cb_out_accumulate_im_2 (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=out_im_tiles * im_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(23, im_df, im_tile_size, im_tile_descriptor)],
            )
        )

        # c_16: cb_out_o (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=out0_t * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(16, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_17: cb_out_m (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(17, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_18: cb_out_l (tiny tile)
        cb_descriptors.append(
            ttnn.CBDescriptor(
                total_size=statistics_tiles * stats_tile_size,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(18, stats_df, stats_tile_size, stats_tile_descriptor)],
            )
        )

        # c_19: cb_intermed_out (tiny tile, only if intermed_output_tiles > 0)
        if intermed_output_tiles > 0:
            cb_descriptors.append(
                ttnn.CBDescriptor(
                    total_size=intermed_output_tiles * stats_tile_size,
                    core_ranges=core_grid,
                    format_descriptors=[ttnn.CBFormatDescriptor(19, stats_df, stats_tile_size, stats_tile_descriptor)],
                )
            )

        # c_20: cb_out_final (always sharded output)
        cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(20, output_tensor)
        cb_descriptors.append(cb_out_descriptor)

        # =========================================================================
        # Create semaphore descriptors (matching C++ lines 724-725)
        # =========================================================================
        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, core_grid, 0),  # reducer_semaphore
            ttnn.SemaphoreDescriptor(1, ttnn.CoreType.WORKER, core_grid, 0),  # output_semaphore
            ttnn.SemaphoreDescriptor(2, ttnn.CoreType.WORKER, core_grid, 0),  # mcast_semaphore for KV cache
            ttnn.SemaphoreDescriptor(3, ttnn.CoreType.WORKER, core_grid, 0),  # brisc_ncrisc_sync for DRAM/mcast overlap
            ttnn.SemaphoreDescriptor(4, ttnn.CoreType.WORKER, core_grid, 0),  # receiver_ready for double-buffer sync
        ]

        # =========================================================================
        # Create kernel descriptors (matching C++ lines 894-918)
        # =========================================================================
        # Addresses for runtime args
        q_addr = q_tensor.buffer_address()
        k_addr = kv_cache_tensor.buffer_address()
        pos_addr = cur_pos_tensor.buffer_address()
        out_addr = output_tensor.buffer_address()

        # Create single RuntimeArgs objects for all cores (matching C++ pattern)
        reader_rtargs = ttnn.RuntimeArgs()
        writer_rtargs = ttnn.RuntimeArgs()
        compute_rtargs = ttnn.RuntimeArgs()

        # Add runtime args for active cores (simplified for Deepseek V3 B1)
        for i in range(num_active_cores):
            core = core_group[i]

            # Calculate per-core values
            # Core layout: [S1[0], S2[0], ..., S8[0], S1[1], S2[1], ..., S8[1], ...]
            s_block_idx = i % num_s_blocks
            cur_batch = i // num_cores_per_batch
            core_num_in_reduce = i % num_cores_per_head
            core_num_in_output = i % num_cores_per_batch

            # Tree reduction: do_reduce is true for any S block that receives from others
            do_reduce = 1 if grid.is_tree_reduction_receiver(s_block_idx) else 0
            # S1 (s_block_idx == 0) is both the output core and the final reducer
            is_output_core = 1 if s_block_idx == 0 else 0
            is_mcast_sender = 1 if i < num_s_blocks else 0

            # Get multicast coordinates for this core's S block
            mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, _ = s_block_mcast_coords[s_block_idx]

            # Virtual channel for DRAM streaming
            if s_block_idx < 4:
                vc = s_block_idx & 0x1
            else:
                vc = 2 + ((s_block_idx - 4) & 0x1)

            # Output core NOC coordinates (S1 core for this batch)
            output_core_noc_x = output_core_physical_xs[cur_batch] if cur_batch < len(output_core_physical_xs) else 0
            output_core_noc_y = output_core_physical_ys[cur_batch] if cur_batch < len(output_core_physical_ys) else 0

            # Reader runtime args (simplified)
            reader_runtime_args = [
                q_addr,
                k_addr,
                pos_addr,
                is_output_core,
                cur_batch,
                core_num_in_reduce,
                is_mcast_sender,
                mcast_start_x,
                mcast_start_y,
                mcast_end_x,
                mcast_end_y,
                vc,
                output_core_noc_x,
                output_core_noc_y,
            ]

            # Tree reduction partner coordinates
            tree_reduction_info = grid.get_tree_reduction_partner_coords(device, s_block_idx, cur_batch)

            # Writer runtime args (simplified)
            writer_runtime_args = [
                out_addr,
                cur_batch,
                core_num_in_reduce,
                is_mcast_sender,
                mcast_start_x,
                mcast_start_y,
                mcast_end_x,
                mcast_end_y,
            ]
            # Tree reduction info: 3 steps × 4 values
            for role_code, partner_s_block_idx, partner_x, partner_y in tree_reduction_info:
                writer_runtime_args.extend([role_code, partner_s_block_idx, partner_x, partner_y])

            # Is this core a sender after receiving? (intermediate node)
            is_sender_after_reduce = 1 if (do_reduce and grid.is_tree_reduction_sender(s_block_idx)) else 0

            # Compute runtime args (unchanged interface)
            compute_runtime_args = [
                do_reduce,
                is_output_core,
                0,  # cur_head (always 0, num_heads_per_core=1)
                cur_batch,
                core_num_in_reduce,
                core_num_in_output,
                is_sender_after_reduce,
            ]
            for role_code, partner_s_block_idx, partner_x, partner_y in tree_reduction_info:
                compute_runtime_args.extend([role_code, partner_s_block_idx])

            reader_rtargs.append(core, reader_runtime_args)
            writer_rtargs.append(core, writer_runtime_args)
            compute_rtargs.append(core, compute_runtime_args)

        # Create 3 kernel descriptors covering ALL cores (not per-core)
        kernel_descriptors = [
            # Reader kernel (use NOC_0 explicitly for DRAM reads)
            ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/dataflow/reader_decode_all.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=reader_compile_time_args,
                runtime_args=reader_rtargs,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.NOC_0,
                ),
            ),
            # Writer kernel (use NOC_1 to avoid conflict with reader on NOC_0)
            ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/dataflow/writer_decode_all.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=writer_compile_time_args,
                runtime_args=writer_rtargs,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.NOC_1,
                ),
            ),
            # Compute kernel
            ttnn.KernelDescriptor(
                kernel_source="models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/compute/sdpa_flash_decode.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_grid,
                compile_time_args=compute_compile_time_args,
                runtime_args=compute_rtargs,
                defines=compute_defines,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=math_fidelity,
                    fp32_dest_acc_en=fp32_dest_acc_en,
                    math_approx_mode=math_approx_mode,
                ),
            ),
        ]

        # =========================================================================
        # Create and execute program
        # =========================================================================
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors,
            cbs=cb_descriptors,
            semaphores=semaphore_descriptors,
        )

        io_tensors = [q_tensor, kv_cache_tensor, cur_pos_tensor, output_tensor]
        ttnn.generic_op(io_tensors, program_descriptor)

        return output_tensor

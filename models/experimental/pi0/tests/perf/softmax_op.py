"""SigLIP row-wise softmax micro-op — Python wrapper.

Shape: M=256, K=1152, bf16. 8 cores × 1 M-tile (32 rows) each. Softmax along K.
Used as a sandboxed validation of the softmax primitive before integration
into the SDPA / attention block.

Kernel: siglip_softmax_kernel.cpp.
"""
import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class SigLIPSoftmaxOp:
    KERNEL_SOURCE = "models/experimental/pi0/tests/perf/siglip_softmax_kernel.cpp"

    # Default shape (sandbox / LN-style); override via .op(..., M=, K=) for SDPA.
    M = 256
    K = 1152
    TILE = 32

    @staticmethod
    def op(
        activation_tensor,
        scaler_tensor,
        output_tensor,
        max_tensor,
        exp_tensor,
        sum_tensor,
        isum_tensor,
        M: int = 256,
        K: int = 1152,
        num_cores: int = 8,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    ):
        TILE = SigLIPSoftmaxOp.TILE
        M_TILES_TOTAL = M // TILE
        K_TILES = K // TILE
        assert M_TILES_TOTAL % num_cores == 0
        M_TILES_PER_CORE = M_TILES_TOTAL // num_cores

        in_cb, scaler_cb = 0, 1
        max_cb, exp_cb = 2, 3
        sum_cb, isum_cb = 4, 5
        out_cb = 6

        in_tiles = M_TILES_PER_CORE * K_TILES

        ct_args = [
            ("in_cb", in_cb),
            ("scaler_cb", scaler_cb),
            ("max_cb", max_cb),
            ("exp_cb", exp_cb),
            ("sum_cb", sum_cb),
            ("isum_cb", isum_cb),
            ("out_cb", out_cb),
            ("k_tiles", K_TILES),
            ("m_tiles", M_TILES_PER_CORE),
            ("in_tiles", in_tiles),
        ]

        core_grid = output_tensor.memory_config().shard_spec.grid

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=SigLIPSoftmaxOp.KERNEL_SOURCE,
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=ct_args,
            brisc_named_compile_time_args=[],
            trisc_named_compile_time_args=ct_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=math_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=True,
            ),
        )

        full_tile = ttnn.Tile((TILE, TILE))
        tile_descriptor = ttnn.TileDescriptor(full_tile)
        bf16_page = full_tile.get_tile_size(ttnn.bfloat16)

        def _cb(cb_id, tensor):
            d = ttnn.cb_descriptor_from_sharded_tensor(cb_id, tensor)
            d.format_descriptors[0].tile = tile_descriptor
            d.format_descriptors[0].page_size = bf16_page
            return d

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[
                _cb(in_cb, activation_tensor),
                _cb(scaler_cb, scaler_tensor),
                _cb(max_cb, max_tensor),
                _cb(exp_cb, exp_tensor),
                _cb(sum_cb, sum_tensor),
                _cb(isum_cb, isum_tensor),
                _cb(out_cb, output_tensor),
            ],
            semaphores=[],
        )

        ttnn.generic_op(
            [activation_tensor, scaler_tensor, max_tensor, exp_tensor, sum_tensor, isum_tensor, output_tensor],
            program_descriptor,
        )
        return output_tensor


def build_tensors_for_softmax_test(device, x_torch, num_cores: int = 8):
    """Build 7 sharded tensors for the softmax op. Shape inferred from x_torch."""
    Cls = SigLIPSoftmaxOp
    M, K = x_torch.shape
    TILE = Cls.TILE
    m_per_core = M // num_cores
    assert M % num_cores == 0 and K % TILE == 0

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})

    # Input HEIGHT_SHARDED.
    in_shard = ttnn.ShardSpec(core_grid, (m_per_core, K), ttnn.ShardOrientation.ROW_MAJOR)
    in_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in_shard)
    activation_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem,
    )

    # Scaler tile: (TILE, TILE) all-ones, replicated per core.
    tile_shard = ttnn.ShardSpec(core_grid, (TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR)
    tile_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, tile_shard)
    scaler_tile = torch.full((TILE, TILE), 1.0, dtype=torch.bfloat16)
    scaler_stacked = scaler_tile.unsqueeze(0).repeat(num_cores, 1, 1).reshape(num_cores * TILE, TILE)
    scaler_tt = ttnn.from_torch(
        scaler_stacked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=tile_mem
    )

    # Intermediate single-tile-per-core CBs: max, sum, isum.
    def _make_1tile():
        return ttnn.from_torch(
            torch.zeros(num_cores * TILE, TILE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=tile_mem,
        )

    max_tt = _make_1tile()
    sum_tt = _make_1tile()
    isum_tt = _make_1tile()

    # Intermediate per-core K-tile CB: exp.
    exp_tt = ttnn.from_torch(
        torch.zeros(M, K, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem,
    )

    # Output, same shape as input.
    output_tt = ttnn.from_torch(
        torch.zeros(M, K, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem,
    )

    return activation_tt, scaler_tt, max_tt, exp_tt, sum_tt, isum_tt, output_tt

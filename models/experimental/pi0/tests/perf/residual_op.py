"""SigLIP residual-add micro-op — Python wrapper.

Elementwise out[m, k] = a[m, k] + b[m, k] on (M=256, D=1152) bf16 tensors.
8 cores × 1 M-tile (32 rows) each. Same sharding layout as LN / softmax.

Kernel: siglip_residual_kernel.cpp.
"""
import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class SigLIPResidualAddOp:
    KERNEL_SOURCE = "models/experimental/pi0/tests/perf/siglip_residual_kernel.cpp"

    M = 256
    D = 1152
    TILE = 32

    @staticmethod
    def op(
        a_tensor,
        b_tensor,
        output_tensor,
        M: int = 256,
        D: int = 1152,
        num_cores: int = 8,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    ):
        TILE = SigLIPResidualAddOp.TILE
        M_TILES_TOTAL = M // TILE
        K_TILES = D // TILE
        assert M_TILES_TOTAL % num_cores == 0
        M_TILES_PER_CORE = M_TILES_TOTAL // num_cores

        a_cb, b_cb, out_cb = 0, 1, 2
        in_tiles = M_TILES_PER_CORE * K_TILES

        ct_args = [
            ("a_cb", a_cb),
            ("b_cb", b_cb),
            ("out_cb", out_cb),
            ("in_tiles", in_tiles),
        ]
        core_grid = output_tensor.memory_config().shard_spec.grid

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=SigLIPResidualAddOp.KERNEL_SOURCE,
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
                _cb(a_cb, a_tensor),
                _cb(b_cb, b_tensor),
                _cb(out_cb, output_tensor),
            ],
            semaphores=[],
        )

        ttnn.generic_op([a_tensor, b_tensor, output_tensor], program_descriptor)
        return output_tensor


def build_tensors_for_residual_test(device, a_torch, b_torch, num_cores: int = 8):
    """Build 3 HEIGHT_SHARDED bf16 tensors for the residual-add op."""
    M, D = a_torch.shape
    assert b_torch.shape == (M, D)
    assert M % num_cores == 0
    m_per_core = M // num_cores

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard = ttnn.ShardSpec(core_grid, (m_per_core, D), ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)

    a_tt = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    b_tt = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    out_tt = ttnn.from_torch(
        torch.zeros(M, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem,
    )
    return a_tt, b_tt, out_tt

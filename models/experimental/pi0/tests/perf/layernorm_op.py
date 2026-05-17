"""SigLIP LayerNorm micro-op — Python wrapper (v3, Stage-A/Stage-B reduce).

Shape: M=256, D=1152, bf16. 8 cores × 1 M-tile (32 rows) each. Real π0.5 weights.
Kernel: siglip_layernorm_kernel.cpp.

v3 changes from v2 (which hung in phase 1):
- Added ones_cb (1 tile filled with 1.0) for the Stage-A mul-accumulate pattern.
- Added accum_cb (1 tile) as intermediate before the single reduce_tile call.
- Phases 1 and 4 (mean/variance reductions) restructured per the groupnorm
  pattern. See [[pi05-ln-kernel-multi-reduce-pattern]].
"""
import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class SigLIPLayerNormOp:
    KERNEL_SOURCE = "models/experimental/pi0/tests/perf/siglip_layernorm_kernel.cpp"

    M = 256
    D = 1152
    TILE = 32
    M_TILES = M // TILE
    D_TILES = D // TILE
    EPS = 1e-6

    @staticmethod
    def op(
        activation_tensor,
        gamma_tensor,
        beta_tensor,
        scaler_tensor,
        ones_tensor,
        accum_tensor,
        xmm_tensor,
        xmm2_tensor,
        mean_tensor,
        var_tensor,
        ivar_tensor,
        output_tensor,
        num_cores: int = 8,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        eps: float = 1e-6,
    ):
        m_per_core = SigLIPLayerNormOp.M_TILES // num_cores
        assert m_per_core * num_cores == SigLIPLayerNormOp.M_TILES

        in_cb, gamma_cb, beta_cb, scaler_cb = 0, 1, 2, 3
        ones_cb, accum_cb = 4, 5
        xmm_cb, xmm2_cb = 6, 7
        mean_cb, var_cb, ivar_cb = 8, 9, 10
        out_cb = 11

        in_tiles = m_per_core * SigLIPLayerNormOp.D_TILES
        gamma_tiles = SigLIPLayerNormOp.D_TILES

        eps_bf16 = torch.tensor(eps, dtype=torch.bfloat16).view(torch.uint16).item()
        eps_bits = eps_bf16 << 16

        ct_args = [
            ("in_cb", in_cb),
            ("gamma_cb", gamma_cb),
            ("beta_cb", beta_cb),
            ("scaler_cb", scaler_cb),
            ("ones_cb", ones_cb),
            ("accum_cb", accum_cb),
            ("xmm_cb", xmm_cb),
            ("xmm2_cb", xmm2_cb),
            ("mean_cb", mean_cb),
            ("var_cb", var_cb),
            ("ivar_cb", ivar_cb),
            ("out_cb", out_cb),
            ("d_tiles", SigLIPLayerNormOp.D_TILES),
            ("in_tiles", in_tiles),
            ("gamma_tiles", gamma_tiles),
            ("eps_bits", eps_bits),
        ]

        core_grid = output_tensor.memory_config().shard_spec.grid

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=SigLIPLayerNormOp.KERNEL_SOURCE,
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

        # CB descriptor metadata: explicit tile + page_size on every CB so the
        # PACK / UNPACK CB-pointer increments align. cb_descriptor_from_sharded_tensor
        # alone leaves these fields unset for some shard layouts, which is the most
        # likely root cause of the Phase 1 Stage-B reduce_tile hang
        # (see RESUME_PROMPT.md and models/demos/deepseek_v3_b1/micro_ops/rmsnorm/op.py:110-129).
        full_tile = ttnn.Tile((SigLIPLayerNormOp.TILE, SigLIPLayerNormOp.TILE))
        tile_descriptor = ttnn.TileDescriptor(full_tile)
        cb_page_size = full_tile.get_tile_size(ttnn.bfloat16)  # 2048 bytes for 32×32 bf16

        def _cb(cb_id, tensor):
            d = ttnn.cb_descriptor_from_sharded_tensor(cb_id, tensor)
            d.format_descriptors[0].tile = tile_descriptor
            d.format_descriptors[0].page_size = cb_page_size
            return d

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[
                _cb(in_cb, activation_tensor),
                _cb(gamma_cb, gamma_tensor),
                _cb(beta_cb, beta_tensor),
                _cb(scaler_cb, scaler_tensor),
                _cb(ones_cb, ones_tensor),
                _cb(accum_cb, accum_tensor),
                _cb(xmm_cb, xmm_tensor),
                _cb(xmm2_cb, xmm2_tensor),
                _cb(mean_cb, mean_tensor),
                _cb(var_cb, var_tensor),
                _cb(ivar_cb, ivar_tensor),
                _cb(out_cb, output_tensor),
            ],
            semaphores=[],
        )

        ttnn.generic_op(
            [
                activation_tensor,
                gamma_tensor,
                beta_tensor,
                scaler_tensor,
                ones_tensor,
                accum_tensor,
                xmm_tensor,
                xmm2_tensor,
                mean_tensor,
                var_tensor,
                ivar_tensor,
                output_tensor,
            ],
            program_descriptor,
        )
        return output_tensor


def build_tensors_for_ln_test(device, gamma_torch, beta_torch, x_torch, num_cores=8):
    """Build the 12 sharded tensors required for SigLIPLayerNormOp v3."""
    M, D = SigLIPLayerNormOp.M, SigLIPLayerNormOp.D
    TILE = SigLIPLayerNormOp.TILE
    assert x_torch.shape == (M, D)
    assert gamma_torch.shape == (D,)
    assert beta_torch.shape == (D,)

    m_per_core = M // num_cores

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})

    # Input HEIGHT_SHARDED.
    in_shard = ttnn.ShardSpec(core_grid, (m_per_core, D), ttnn.ShardOrientation.ROW_MAJOR)
    in_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in_shard)
    activation_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem,
    )

    # Gamma / Beta replicated.
    gamma_full = gamma_torch.unsqueeze(0).expand(TILE, -1).contiguous()
    beta_full = beta_torch.unsqueeze(0).expand(TILE, -1).contiguous()
    gamma_stacked = gamma_full.unsqueeze(0).repeat(num_cores, 1, 1).reshape(num_cores * TILE, D)
    beta_stacked = beta_full.unsqueeze(0).repeat(num_cores, 1, 1).reshape(num_cores * TILE, D)
    gb_shard = ttnn.ShardSpec(core_grid, (TILE, D), ttnn.ShardOrientation.ROW_MAJOR)
    gb_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gb_shard)
    gamma_tt = ttnn.from_torch(
        gamma_stacked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=gb_mem
    )
    beta_tt = ttnn.from_torch(
        beta_stacked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=gb_mem
    )

    # Scaler tile (1/D) and ones tile (1.0). Shard (TILE, TILE) replicated × cores.
    inv_d = 1.0 / D
    tile_shard = ttnn.ShardSpec(core_grid, (TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR)
    tile_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, tile_shard)

    scaler_tile = torch.full((TILE, TILE), inv_d, dtype=torch.bfloat16)
    ones_tile = torch.full((TILE, TILE), 1.0, dtype=torch.bfloat16)
    scaler_stacked = scaler_tile.unsqueeze(0).repeat(num_cores, 1, 1).reshape(num_cores * TILE, TILE)
    ones_stacked = ones_tile.unsqueeze(0).repeat(num_cores, 1, 1).reshape(num_cores * TILE, TILE)
    scaler_tt = ttnn.from_torch(
        scaler_stacked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=tile_mem
    )
    ones_tt = ttnn.from_torch(
        ones_stacked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=tile_mem
    )

    # accum / mean / var / ivar: 1 tile per core each.
    def _make_1tile():
        return ttnn.from_torch(
            torch.zeros(num_cores * TILE, TILE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=tile_mem,
        )

    accum_tt = _make_1tile()
    mean_tt = _make_1tile()
    var_tt = _make_1tile()
    ivar_tt = _make_1tile()

    # xmm / xmm2 / out: D-tile per core (same shape as input).
    def _make_dtile():
        return ttnn.from_torch(
            torch.zeros(M, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in_mem,
        )

    xmm_tt = _make_dtile()
    xmm2_tt = _make_dtile()
    out_tt = _make_dtile()

    return (
        activation_tt,
        gamma_tt,
        beta_tt,
        scaler_tt,
        ones_tt,
        accum_tt,
        xmm_tt,
        xmm2_tt,
        mean_tt,
        var_tt,
        ivar_tt,
        out_tt,
    )

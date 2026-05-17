"""Fused SigLIP attention sub-block — Python op wrapper.

First increment: LN1 + residual fused into one dispatch. Math:
    out = LN1(x; gamma, beta) + x

This is the validation skeleton for the multi-Op chaining pattern. Future
increments will add QKV, 16-head SDPA, O-proj.

CB plan (this increment):
    LN1 inputs:      ln_in_cb=0, gamma_cb=1, beta_cb=2, scaler_cb=3, ones_cb=4
    LN1 intermediates: accum_cb=5, xmm_cb=6, xmm2_cb=7, mean_cb=8, var_cb=9, ivar_cb=10
    Chaining:        ln_out_cb=11 (LN writes, residual reads as its a-input)
    Residual second: x_residual_cb=12 (separate L1 copy of x for b-input)
    Final output:    final_out_cb=13
"""
import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class SigLIPAttentionBlockFused:
    """Fused LN1 + residual on a single 8-core row (first increment)."""

    KERNEL_SOURCE = "models/experimental/pi0/fused_ops/attention_block/kernels/attention_block_kernel.cpp"

    M = 256
    D = 1152
    TILE = 32
    M_TILES = M // TILE
    D_TILES = D // TILE
    EPS = 1e-6

    # CB IDs (must match the kernel's get_named_compile_time_arg_val calls)
    CB = {
        "ln_in_cb": 0,
        "gamma_cb": 1,
        "beta_cb": 2,
        "scaler_cb": 3,
        "ones_cb": 4,
        "accum_cb": 5,
        "xmm_cb": 6,
        "xmm2_cb": 7,
        "mean_cb": 8,
        "var_cb": 9,
        "ivar_cb": 10,
        "ln_out_cb": 11,
        "x_residual_cb": 12,
        "final_out_cb": 13,
    }

    @staticmethod
    def op(
        ln_in_tt,
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
        ln_out_tt,
        x_residual_tt,
        final_out_tt,
        num_cores: int = 8,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        eps: float = 1e-6,
    ):
        m_per_core = SigLIPAttentionBlockFused.M_TILES // num_cores
        assert m_per_core * num_cores == SigLIPAttentionBlockFused.M_TILES

        in_tiles = m_per_core * SigLIPAttentionBlockFused.D_TILES
        gamma_tiles = SigLIPAttentionBlockFused.D_TILES

        eps_bf16 = torch.tensor(eps, dtype=torch.bfloat16).view(torch.uint16).item()
        eps_bits = eps_bf16 << 16

        # NCRISC needs only the CBs it sets up sharded buffers for.
        ncrisc_ct = [
            ("ln_in_cb", SigLIPAttentionBlockFused.CB["ln_in_cb"]),
            ("gamma_cb", SigLIPAttentionBlockFused.CB["gamma_cb"]),
            ("beta_cb", SigLIPAttentionBlockFused.CB["beta_cb"]),
            ("scaler_cb", SigLIPAttentionBlockFused.CB["scaler_cb"]),
            ("ones_cb", SigLIPAttentionBlockFused.CB["ones_cb"]),
            ("x_residual_cb", SigLIPAttentionBlockFused.CB["x_residual_cb"]),
            ("final_out_cb", SigLIPAttentionBlockFused.CB["final_out_cb"]),
            ("in_tiles", in_tiles),
            ("gamma_tiles", gamma_tiles),
        ]
        # TRISC needs all CBs + tile counts + eps.
        trisc_ct = [
            *[(k, v) for k, v in SigLIPAttentionBlockFused.CB.items()],
            ("d_tiles", SigLIPAttentionBlockFused.D_TILES),
            ("in_tiles", in_tiles),
            ("eps_bits", eps_bits),
        ]

        core_grid = final_out_tt.memory_config().shard_spec.grid

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=SigLIPAttentionBlockFused.KERNEL_SOURCE,
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=ncrisc_ct,
            brisc_named_compile_time_args=[],
            trisc_named_compile_time_args=trisc_ct,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=math_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=True,
            ),
        )

        full_tile = ttnn.Tile((SigLIPAttentionBlockFused.TILE, SigLIPAttentionBlockFused.TILE))
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
                _cb(SigLIPAttentionBlockFused.CB["ln_in_cb"], ln_in_tt),
                _cb(SigLIPAttentionBlockFused.CB["gamma_cb"], gamma_tt),
                _cb(SigLIPAttentionBlockFused.CB["beta_cb"], beta_tt),
                _cb(SigLIPAttentionBlockFused.CB["scaler_cb"], scaler_tt),
                _cb(SigLIPAttentionBlockFused.CB["ones_cb"], ones_tt),
                _cb(SigLIPAttentionBlockFused.CB["accum_cb"], accum_tt),
                _cb(SigLIPAttentionBlockFused.CB["xmm_cb"], xmm_tt),
                _cb(SigLIPAttentionBlockFused.CB["xmm2_cb"], xmm2_tt),
                _cb(SigLIPAttentionBlockFused.CB["mean_cb"], mean_tt),
                _cb(SigLIPAttentionBlockFused.CB["var_cb"], var_tt),
                _cb(SigLIPAttentionBlockFused.CB["ivar_cb"], ivar_tt),
                _cb(SigLIPAttentionBlockFused.CB["ln_out_cb"], ln_out_tt),
                _cb(SigLIPAttentionBlockFused.CB["x_residual_cb"], x_residual_tt),
                _cb(SigLIPAttentionBlockFused.CB["final_out_cb"], final_out_tt),
            ],
            semaphores=[],
        )

        ttnn.generic_op(
            [
                ln_in_tt,
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
                ln_out_tt,
                x_residual_tt,
                final_out_tt,
            ],
            program_descriptor,
        )
        return final_out_tt


def build_tensors_for_fused_attention_block(device, x_torch, gamma_torch, beta_torch, num_cores: int = 8):
    """Build all 14 sharded tensors needed by the fused LN1 + residual op.

    Extends build_tensors_for_ln_test with the additional `x_residual` and
    `final_out` tensors needed for the residual phase.
    """
    M = SigLIPAttentionBlockFused.M
    D = SigLIPAttentionBlockFused.D
    TILE = SigLIPAttentionBlockFused.TILE
    assert x_torch.shape == (M, D)
    assert gamma_torch.shape == (D,)
    assert beta_torch.shape == (D,)
    assert M % num_cores == 0
    m_per_core = M // num_cores

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})

    # Input x (LN1 in_cb).
    in_shard = ttnn.ShardSpec(core_grid, (m_per_core, D), ttnn.ShardOrientation.ROW_MAJOR)
    in_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in_shard)
    ln_in_tt = ttnn.from_torch(
        x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem
    )

    # Separate L1 copy of x for the residual b-input.
    x_residual_tt = ttnn.from_torch(
        x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem
    )

    # Gamma / Beta replicated × cores.
    gamma_stacked = (
        gamma_torch.unsqueeze(0)
        .expand(TILE, -1)
        .contiguous()
        .unsqueeze(0)
        .repeat(num_cores, 1, 1)
        .reshape(num_cores * TILE, D)
    )
    beta_stacked = (
        beta_torch.unsqueeze(0)
        .expand(TILE, -1)
        .contiguous()
        .unsqueeze(0)
        .repeat(num_cores, 1, 1)
        .reshape(num_cores * TILE, D)
    )
    gb_shard = ttnn.ShardSpec(core_grid, (TILE, D), ttnn.ShardOrientation.ROW_MAJOR)
    gb_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gb_shard)
    gamma_tt = ttnn.from_torch(
        gamma_stacked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=gb_mem
    )
    beta_tt = ttnn.from_torch(
        beta_stacked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=gb_mem
    )

    # Single-tile per-core helpers (scaler 1/D, ones).
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

    # LN intermediates (1-tile-per-core).
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

    # D-tile per core intermediates and outputs.
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
    ln_out_tt = _make_dtile()
    final_out_tt = _make_dtile()

    return (
        ln_in_tt,
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
        ln_out_tt,
        x_residual_tt,
        final_out_tt,
    )

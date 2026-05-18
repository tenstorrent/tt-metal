"""Fused SigLIP attention sub-block — Python op wrapper.

Increment status (task #10):
    Commit 1: LN1 + residual fused into one dispatch on the 8-core LN1 row.
        Math: residual_out = LN1(x; gamma, beta) + x_residual.
    Commit 2 (current): expand the kernel grid to LN1 ∪ QKV = 38 distinct cores.
        After LN1's TRISC pushes ln_out_cb on the 8-core row (y=0, x=0..7), the
        36 QKV receivers (y=0..5, x=0..5) pull 8 shards each via noc_async_read
        and assemble the full (256, 1152) activation in qkv_act_cb. Residual is
        still added on the LN1 row only — qkv_act_cb is produced for the probe
        test, not consumed downstream yet (QKV matmul lands in Commit 3).
    Commit 3 (next): QKV EncoderMatmul::Op on the 36-core grid.

CB plan (this increment):
    LN1 inputs (LN1 grid):       ln_in_cb=0, gamma_cb=1, beta_cb=2, scaler_cb=3, ones_cb=4
    LN1 intermediates (LN1 grid): accum_cb=5, xmm_cb=6, xmm2_cb=7, mean_cb=8, var_cb=9, ivar_cb=10
    Chaining (LN1 grid):         ln_out_cb=11
    Residual extras (LN1 grid):  x_residual_cb=12, final_out_cb=13
    Mcast destination (QKV grid): qkv_act_cb=14 — holds the full (256, 1152)
                                  activation re-assembled from 8 LN1 shards.

Semaphores:
    counter_sem (QKV grid, id=2): each QKV receiver waits for it to reach 8
        atomic +1s (one per LN1 sender), then resets to 0.

NCRISC common RT args (positional, named on host side for clarity):
    [0] ln_out_l1_addr: deterministic L1 address of every LN1 core's ln_out_cb
        shard (sharded buffers share an address across all owning cores).

Receiver-pull mechanics (vs an Mcast::Op sender push):
    Avoids 8 parallel multicasts contending for fabric. Senders are 8, each
    does 36 tiny atomic NoC writes (288 total). Receivers each issue 8
    noc_async_reads (288 total). Both numbers are well below fabric limits
    and decouple cleanly from any matmul work that follows.
"""
import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class SigLIPAttentionBlockFused:
    """Fused LN1 + residual on the LN1 row, with receiver-pull mcast of LN1's
    output to a 36-core QKV grid (no QKV matmul yet)."""

    KERNEL_SOURCE = "models/experimental/pi0/fused_ops/attention_block/kernels/attention_block_kernel.cpp"

    M = 256
    D = 1152
    TILE = 32
    M_TILES = M // TILE
    D_TILES = D // TILE
    EPS = 1e-6

    # Grid layout (task #10 Commit 2):
    #   LN1 row: y=0, x=0..7  → 8 cores. Produce ln_out_cb shards (32, 1152).
    #   QKV grid: y=0..5, x=0..5  → 36 cores. Pull 8 shards each into
    #     qkv_act_cb of shape (256, 1152) (full LN1 output, replicated per core).
    #   Union (38 distinct cores) is the kernel core_ranges; CBs and semaphores
    #     live on the sub-grids they need to.
    LN1_NUM_CORES = 8
    QKV_GRID_X = 6
    QKV_GRID_Y = 6
    QKV_NUM_CORES = QKV_GRID_X * QKV_GRID_Y  # 36

    # QKV matmul (encoder-shape) — fused Q/K/V projection.
    #   Activation per core: (M=256, K=1152) bf16, replicated across 36 cores.
    #   Weight per core:     (K=1152, N=96)  bfp8, width-sharded across 36 cores.
    #   Output per core:     (M=256, N=96)   bf16, width-sharded across 36 cores.
    # Total weight + output cover N=3456 = 3·D — concatenated Q, K, V projections.
    QKV_N = 3 * D  # 3456
    QKV_N_TILES = QKV_N // TILE  # 108
    QKV_N_TILES_PER_CORE = QKV_N_TILES // QKV_NUM_CORES  # 3

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
        "qkv_act_cb": 14,
        # 1-tile sync CB (LN1 row): TRISC pushes after LN1 completes, NCRISC
        # sender waits/pops before issuing 36 counter-sem increments. Decouples
        # the NCRISC sender from TRISC residual on ln_out_cb (multi-consumer
        # races: residual's cb_pop_front would otherwise starve the sender's
        # cb_wait_front, since both read pages_received - pages_acked).
        "ln_done_trigger_cb": 15,
        # Commit 3: QKV matmul on the 36-core receiver grid.
        # qkv_w_cb: bfp8 weight, width-sharded (K=1152, N/36=96) per core, all
        #   K_TILES × N_TILES_PER_CORE = 36×3 = 108 tiles per core; L1-resident,
        #   pre-loaded via from_torch + setup_sharded_buffer.
        # qkv_out_cb: bf16 output, width-sharded (M=256, 96) per core, M_TILES ×
        #   N_TILES_PER_CORE = 8×3 = 24 tiles per core. Produced by TRISC matmul;
        #   left for downstream (#11 SDPA) to consume.
        "qkv_w_cb": 16,
        "qkv_out_cb": 17,
    }

    # Counter semaphore for the LN1→QKV receiver-pull. Each QKV receiver
    # increments-and-waits on its own copy at this ID; the host allocates the
    # actual L1 backing on the QKV grid.
    COUNTER_SEM_ID = 2

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
        qkv_act_tt,
        ln_done_trigger_tt,
        qkv_w_tt,
        qkv_out_tt,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        eps: float = 1e-6,
    ):
        num_cores = SigLIPAttentionBlockFused.LN1_NUM_CORES
        m_per_core = SigLIPAttentionBlockFused.M_TILES // num_cores
        assert m_per_core * num_cores == SigLIPAttentionBlockFused.M_TILES

        in_tiles = m_per_core * SigLIPAttentionBlockFused.D_TILES  # per-LN1-core: 1 * 36 = 36
        gamma_tiles = SigLIPAttentionBlockFused.D_TILES

        # Per-receiver qkv_act_cb holds the full re-assembled (256, 1152) = 8 LN1
        # shards = 8 * 36 = 288 tiles.
        qkv_act_tiles_per_core = SigLIPAttentionBlockFused.LN1_NUM_CORES * in_tiles

        # QKV matmul per-core tile counts (encoder-shape Op-struct convention):
        #   M_TILES=8, K_TILES=36, N_TILES_PER_CORE=3.
        # ACT_TOTAL_TILES = 288 (matches qkv_act_tiles_per_core above).
        # WEIGHT_TILES = K_TILES * N_TILES_PER_CORE = 36*3 = 108.
        # OUT_TOTAL_TILES = M_TILES * N_TILES_PER_CORE = 8*3 = 24.
        # K_TILES for the matmul equals D_TILES (per-receiver activation is the
        # full (M, D) = (256, 1152) replicated).
        qkv_weight_tiles = SigLIPAttentionBlockFused.D_TILES * SigLIPAttentionBlockFused.QKV_N_TILES_PER_CORE
        qkv_out_tiles_per_core = SigLIPAttentionBlockFused.M_TILES * SigLIPAttentionBlockFused.QKV_N_TILES_PER_CORE

        eps_bf16 = torch.tensor(eps, dtype=torch.bfloat16).view(torch.uint16).item()
        eps_bits = eps_bf16 << 16

        # NCRISC CT args: existing setup-sharded-buffer plumbing for the LN1
        # row, plus the new qkv_act_cb / shard sizing / semaphore-id needed by
        # the receiver-pull added in Commit 2.
        ncrisc_ct = [
            ("ln_in_cb", SigLIPAttentionBlockFused.CB["ln_in_cb"]),
            ("gamma_cb", SigLIPAttentionBlockFused.CB["gamma_cb"]),
            ("beta_cb", SigLIPAttentionBlockFused.CB["beta_cb"]),
            ("scaler_cb", SigLIPAttentionBlockFused.CB["scaler_cb"]),
            ("ones_cb", SigLIPAttentionBlockFused.CB["ones_cb"]),
            ("x_residual_cb", SigLIPAttentionBlockFused.CB["x_residual_cb"]),
            ("final_out_cb", SigLIPAttentionBlockFused.CB["final_out_cb"]),
            ("ln_out_cb", SigLIPAttentionBlockFused.CB["ln_out_cb"]),
            ("qkv_act_cb", SigLIPAttentionBlockFused.CB["qkv_act_cb"]),
            ("ln_done_trigger_cb", SigLIPAttentionBlockFused.CB["ln_done_trigger_cb"]),
            ("in_tiles", in_tiles),
            ("gamma_tiles", gamma_tiles),
            ("ln1_num_cores", SigLIPAttentionBlockFused.LN1_NUM_CORES),
            ("qkv_grid_x", SigLIPAttentionBlockFused.QKV_GRID_X),
            ("qkv_grid_y", SigLIPAttentionBlockFused.QKV_GRID_Y),
            ("qkv_act_tiles_per_core", qkv_act_tiles_per_core),
            ("counter_sem_id", SigLIPAttentionBlockFused.COUNTER_SEM_ID),
            # QKV weight is pre-loaded; NCRISC marks all 108 tiles pushed so
            # TRISC matmul's cb_wait_front returns immediately.
            ("qkv_w_cb", SigLIPAttentionBlockFused.CB["qkv_w_cb"]),
            ("qkv_weight_tiles", qkv_weight_tiles),
        ]
        # TRISC needs all CBs + tile counts + eps + QKV matmul shape params.
        trisc_ct = [
            *[(k, v) for k, v in SigLIPAttentionBlockFused.CB.items()],
            ("d_tiles", SigLIPAttentionBlockFused.D_TILES),
            ("in_tiles", in_tiles),
            ("eps_bits", eps_bits),
            # QKV matmul shape (encoder-shape Op-struct):
            ("qkv_m_tiles", SigLIPAttentionBlockFused.M_TILES),
            ("qkv_k_tiles", SigLIPAttentionBlockFused.D_TILES),
            ("qkv_n_tiles_per_core", SigLIPAttentionBlockFused.QKV_N_TILES_PER_CORE),
            ("qkv_act_total_tiles", qkv_act_tiles_per_core),
            ("qkv_weight_tiles", qkv_weight_tiles),
        ]

        # Physical NoC coords for the worker grid. BH's logical-to-physical x
        # mapping is contiguous for the first 6..7 columns (1..7) but then
        # jumps over the eth/PCIe rows: logical x=7 → physical x=10, NOT 8.
        # The 6×6 QKV receiver grid stays inside the contiguous block, so we
        # can pass an origin and add logical offsets. The 8-core LN1 row spans
        # the gap (logical x=0..7), so we explicitly enumerate the 8 LN1
        # physical x coords and pass them as a runtime array.
        _device = ln_in_tt.device()
        _phys_origin = _device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        _ln1_phys_xs = [
            _device.worker_core_from_logical_core(ttnn.CoreCoord(lx, 0)).x
            for lx in range(SigLIPAttentionBlockFused.LN1_NUM_CORES)
        ]

        # Kernel runs on every core in LN1 ∪ QKV (38 distinct). Runtime role
        # flags inside the kernel gate the LN1 / sender / receiver bodies.
        ln1_range = ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(SigLIPAttentionBlockFused.LN1_NUM_CORES - 1, 0),
        )
        # QKV grid is y=0..5, x=0..5; the y=0 row overlaps LN1's x=0..5. Express
        # the union as the LN1 row plus the strictly-below-y=0 QKV rows so the
        # two ranges don't overlap (CoreRangeSet requires disjoint ranges).
        qkv_rows_below = ttnn.CoreRange(
            ttnn.CoreCoord(0, 1),
            ttnn.CoreCoord(SigLIPAttentionBlockFused.QKV_GRID_X - 1, SigLIPAttentionBlockFused.QKV_GRID_Y - 1),
        )
        union_core_grid = ttnn.CoreRangeSet({ln1_range, qkv_rows_below})

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=SigLIPAttentionBlockFused.KERNEL_SOURCE,
            core_ranges=union_core_grid,
            ncrisc_named_compile_time_args=ncrisc_ct,
            brisc_named_compile_time_args=[],
            trisc_named_compile_time_args=trisc_ct,
            # NCRISC: senders read get_read_ptr(ln_out_cb), receivers read the
            # remote LN1 cores' ln_out_cb shard at the same L1 address (sharded
            # persistent buffer ⇒ deterministic per-core L1 location). The two
            # phys_origin args carry the worker-grid logical-(0,0)→NoC-physical
            # mapping: BH places logical (0,0) at NoC physical (1,2), and the
            # worker grid is contiguous from there, so kernel derives a target
            # core's physical NoC coord as (phys_origin + logical). Atomic-inc
            # responses don't appear to route through the HW translation table
            # on BH, so passing physical coords directly is the safe pattern
            # (matches deepseek's gather op host-side compute).
            ncrisc_named_common_runtime_args=[
                ("ln_out_l1_addr", ln_out_tt.buffer_address()),
                ("worker_phys_origin_x", _phys_origin.x),
                ("worker_phys_origin_y", _phys_origin.y),
            ],
            ncrisc_named_common_runtime_arg_arrays=[
                # 8 physical NoC x coords, one per LN1 sender (logical x=0..7).
                # Kernel reads via positional get_common_arg_val starting at
                # the slot after the 3 scalar named common args above.
                ("ln1_phys_x", _ln1_phys_xs),
            ],
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
            # The Op-struct ports were validated with an explicit bf16 page-size
            # override (2048 B per 32×32 tile). bfp8 tensors have a different
            # tile size (1088 B = 32×32 mantissa + per-row exponents), so the
            # override would yield a non-divisible CB size and fail validation.
            # Skip the override for non-bf16 dtypes and let the descriptor's
            # auto-derived page_size carry the layout (matches qkv_op.py's
            # standalone QKV matmul, which uses the unmodified helper).
            if tensor.dtype == ttnn.bfloat16:
                d.format_descriptors[0].tile = tile_descriptor
                d.format_descriptors[0].page_size = bf16_page
            return d

        # Counter semaphore lives on the 36 QKV receivers. Senders unicast +1
        # via NoC; receivers wait for value ≥ ln1_num_cores then reset.
        qkv_sem_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(SigLIPAttentionBlockFused.QKV_GRID_X - 1, SigLIPAttentionBlockFused.QKV_GRID_Y - 1),
                )
            }
        )
        counter_sem = ttnn.SemaphoreDescriptor(
            id=SigLIPAttentionBlockFused.COUNTER_SEM_ID,
            core_ranges=qkv_sem_grid,
            initial_value=0,
        )

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
                _cb(SigLIPAttentionBlockFused.CB["qkv_act_cb"], qkv_act_tt),
                _cb(SigLIPAttentionBlockFused.CB["ln_done_trigger_cb"], ln_done_trigger_tt),
                _cb(SigLIPAttentionBlockFused.CB["qkv_w_cb"], qkv_w_tt),
                _cb(SigLIPAttentionBlockFused.CB["qkv_out_cb"], qkv_out_tt),
            ],
            semaphores=[counter_sem],
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
                qkv_act_tt,
                ln_done_trigger_tt,
                qkv_w_tt,
                qkv_out_tt,
            ],
            program_descriptor,
        )
        return final_out_tt, qkv_act_tt, qkv_out_tt


def build_tensors_for_fused_attention_block(device, x_torch, gamma_torch, beta_torch, w_qkv_torch=None):
    """Build all sharded tensors needed by the fused attention-block op.

    Returns a 17-tuple matching SigLIPAttentionBlockFused.op's positional inputs.
    Entries 0..13: LN1 + residual on the 8-core row (Commit 1).
    Entry 14: qkv_act_tt — (36×256, 1152) HEIGHT_SHARDED, written by NCRISC
        receiver-pull mcast (Commit 2).
    Entry 15: ln_done_trigger_tt — 1-tile sync CB on the LN1 row (Commit 2).
    Entries 16..17: qkv_w_tt + qkv_out_tt — QKV matmul weight (bfp8) and output
        (bf16), both WIDTH_SHARDED on the 36-core QKV grid (Commit 3).

    If `w_qkv_torch` is None, a deterministic random `(1152, 3456)` weight is
    generated — useful for math-only tests where the caller passes the same
    seed to its torch golden.
    """
    M = SigLIPAttentionBlockFused.M
    D = SigLIPAttentionBlockFused.D
    TILE = SigLIPAttentionBlockFused.TILE
    num_cores = SigLIPAttentionBlockFused.LN1_NUM_CORES
    assert x_torch.shape == (M, D)
    assert gamma_torch.shape == (D,)
    assert beta_torch.shape == (D,)
    assert M % num_cores == 0
    m_per_core = M // num_cores

    ln1_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})

    # Input x (LN1 in_cb).
    in_shard = ttnn.ShardSpec(ln1_core_grid, (m_per_core, D), ttnn.ShardOrientation.ROW_MAJOR)
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
    gb_shard = ttnn.ShardSpec(ln1_core_grid, (TILE, D), ttnn.ShardOrientation.ROW_MAJOR)
    gb_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gb_shard)
    gamma_tt = ttnn.from_torch(
        gamma_stacked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=gb_mem
    )
    beta_tt = ttnn.from_torch(
        beta_stacked, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=gb_mem
    )

    # Single-tile per-core helpers (scaler 1/D, ones).
    inv_d = 1.0 / D
    tile_shard = ttnn.ShardSpec(ln1_core_grid, (TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR)
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

    # D-tile per LN1 core intermediates and outputs.
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

    # qkv_act_tt — 36 receivers each hold the full (256, 1152) re-assembled
    # activation. HEIGHT_SHARDED across the 6×6 QKV grid at (M, D) per shard.
    qkv_num_cores = SigLIPAttentionBlockFused.QKV_NUM_CORES
    qkv_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(SigLIPAttentionBlockFused.QKV_GRID_X - 1, SigLIPAttentionBlockFused.QKV_GRID_Y - 1),
            )
        }
    )
    qkv_shard = ttnn.ShardSpec(qkv_core_grid, (M, D), ttnn.ShardOrientation.ROW_MAJOR)
    qkv_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, qkv_shard)
    qkv_act_tt = ttnn.from_torch(
        torch.zeros(qkv_num_cores * M, D, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=qkv_mem,
    )

    # 1-tile sync CB on the LN1 row. TRISC pushes after LN1 completes; NCRISC
    # sender waits/pops before issuing the receiver-pull NoC increments.
    # Single-tile-per-core; payload is unused — only the push/pop counters
    # carry the signal.
    ln_done_trigger_tt = ttnn.from_torch(
        torch.zeros(num_cores * TILE, TILE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tile_mem,
    )

    # QKV matmul weight + output, WIDTH_SHARDED across the 36-core QKV grid.
    # Weight is bfp8 (matches the qkv_op.py pattern), output is bf16.
    N = SigLIPAttentionBlockFused.QKV_N  # 3456
    n_per_core = N // qkv_num_cores  # 96
    if w_qkv_torch is None:
        g = torch.Generator().manual_seed(7)
        w_qkv_torch = torch.randn(D, N, generator=g, dtype=torch.bfloat16) * 0.05
    assert w_qkv_torch.shape == (D, N), f"w_qkv shape {tuple(w_qkv_torch.shape)} != {(D, N)}"

    qkv_w_shard = ttnn.ShardSpec(qkv_core_grid, (D, n_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    qkv_w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, qkv_w_shard)
    qkv_w_tt = ttnn.from_torch(
        w_qkv_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=qkv_w_mem,
    )

    qkv_out_shard = ttnn.ShardSpec(qkv_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    qkv_out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, qkv_out_shard)
    qkv_out_tt = ttnn.from_torch(
        torch.zeros(M, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=qkv_out_mem,
    )

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
        qkv_act_tt,
        ln_done_trigger_tt,
        qkv_w_tt,
        qkv_out_tt,
    )

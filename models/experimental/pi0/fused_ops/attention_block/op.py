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

    # Grid layout (task #11 Commit 1):
    #   LN1 row: y=0, x=0..7  → 8 cores. Produce ln_out_cb shards (32, 1152).
    #   QKV grid: y=0..7, x=0..5  → 48 cores. Pull 8 shards each into qkv_act_cb
    #     of shape (256, 1152) (full LN1 output, replicated per core). Each core
    #     also owns exactly ONE head's padded slice of Q, K, or V (see below).
    #   Union (50 distinct cores) is the kernel core_ranges.
    #
    # Why 48-core QKV instead of 36 (Commit 3): SDPA in task #11 needs each
    # core's slice to be exactly one head's worth, so head_dim addressing is
    # clean across cores. SigLIP-So400m has head_dim=72, padded to 96 for tile
    # alignment, × 16 heads × 3 Q/K/V = 4608 N-cols total. 4608 / 96 = 48 cores.
    LN1_NUM_CORES = 8
    QKV_GRID_X = 6
    QKV_GRID_Y = 8
    QKV_NUM_CORES = QKV_GRID_X * QKV_GRID_Y  # 48

    # Multi-head shape (SigLIP-So400m).
    NUM_HEADS = 16
    HEAD_DIM_TRUE = 72  # NUM_HEADS * HEAD_DIM_TRUE = D (=1152)
    HEAD_DIM_PADDED = 96  # 3 tiles; last 24 cols of each per-head slice are 0.

    # QKV matmul (encoder-shape) — fused Q/K/V projection, head-padded.
    #   Activation per core: (M=256, K=1152) bf16, replicated across 48 cores.
    #   Weight per core:     (K=1152, N=96)  bfp8, width-sharded across 48 cores.
    #   Output per core:     (M=256, N=96)   bf16, width-sharded across 48 cores.
    # N=96 per core is exactly ONE head's padded slice; the 48-core grid covers
    #   3 (Q,K,V) × 16 heads = 48 head-positions. Total padded N=4608.
    QKV_N = 3 * NUM_HEADS * HEAD_DIM_PADDED  # 4608
    QKV_N_TILES = QKV_N // TILE  # 144
    QKV_N_TILES_PER_CORE = QKV_N_TILES // QKV_NUM_CORES  # 3 (= 1 padded head)

    # SDPA grid (task #11 Commit 4 — relocated to a region disjoint from LN1
    # and QKV). Logical layout: 4 cols × 8 rows = 32 cores at
    # x=SDPA_X_OFFSET..SDPA_X_OFFSET+SDPA_GRID_X-1, y=0..SDPA_GRID_Y-1
    # → x=8..11, y=0..7.
    #
    # Why DISJOINT from QKV instead of overlapping (the earlier 64-core 8×8
    # plan)? Even with the LN1 row dodged, the QKV+SDPA overlap cores at
    # y=1..7, x=0..5 have ~13 KB of L1 free after qkv_act_cb (576 KB),
    # qkv_w_cb (117 KB), qkv_out_cb, scratches, and system overhead. K+V CBs
    # need ~96 KB per worker (24 tiles bf16 each), and even the (a)+(b)
    # bfp8-shared-KV mitigation (~26 KB) doesn't fit. Moving SDPA to the
    # right of QKV (x=8..11) gives each SDPA core the full 1.4 MB L1 to
    # itself — Q + K + V allocations land in ~120 KB with massive headroom.
    #
    # The cost: SDPA_NUM_CORES drops from 64 → 32 (16 heads × 2 workers/head
    # instead of 4 workers/head). M-parallel splits 256 rows two ways =
    # 128 rows per worker. Kernel head mapping:
    #   head h ∈ [0..15]:
    #     head_col = h % SDPA_GRID_X (= h % 4)
    #     head_row = h / SDPA_GRID_X (= h / 4)
    #     workers at (SDPA_X_OFFSET + head_col, head_row*2 + worker_idx)
    #     for worker_idx ∈ [0..1].
    # SDPA workers/head can grow back to 4 in a later perf pass if needed.
    SDPA_GRID_X = 4
    SDPA_GRID_Y = 8
    SDPA_X_OFFSET = 8  # right of LN1/QKV; logical x ∈ [8..11], physical ∈ [11..14]
    SDPA_Y_OFFSET = 0  # top-aligned; SDPA fully disjoint from LN1∪QKV in x
    SDPA_NUM_CORES = SDPA_GRID_X * SDPA_GRID_Y  # 32
    NUM_SDPA_WORKERS_PER_HEAD = SDPA_NUM_CORES // NUM_HEADS  # 2

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
        # #11 Commit 3: QKV → SDPA per-head delivery.
        # qkv_done_trigger_cb: 1-tile sync CB on the QKV grid. TRISC pushes
        #   after EncoderMatmul finishes; NCRISC waits then atomic-incs each of
        #   the 4 SDPA workers for this core's head.
        # sdpa_q_cb: per-SDPA-worker Q slice, (64, 96) bf16 = 6 tiles. The 4
        #   workers of head h together cover Q head h's full (256, 96) by
        #   pulling 64-row M-slices. K and V land in subsequent commits.
        "qkv_done_trigger_cb": 18,
        "sdpa_q_cb": 19,
        # #11 Commit 6 — streaming K row 0 + first QK^T tile probe.
        # sdpa_k_partial_cb: 3 tiles bf16 = (32, 96) per worker. Holds K
        #   row 0 (the 3 head_dim tiles needed for QK^T's (0, 0) output
        #   tile). Streamed in (3-tile burst) by NCRISC from the head's K
        #   source core's qkv_out_cb at L1 offset 0. The streaming reader
        #   shape extends naturally to all 8 K rows in future commits.
        # sdpa_qk_probe_cb: 1 tile bf16 = (32, 32) per worker. TRISC matmul
        #   output: QK^T[worker_M_slice[0:32], 0:32] for the worker's head.
        # V probe dropped (V isn't used until Attn@V; V signaling stays).
        "sdpa_k_partial_cb": 20,
        "sdpa_qk_probe_cb": 21,
    }

    # SDPA QKV-ready semaphore (id=3). All three QKV producers (Q, K, V)
    # atomic-inc each head's SDPA workers; workers wait for sem ≥ 3 before
    # reading any of the three.
    SDPA_QKV_READY_SEM_ID = 3
    NUM_QKV_SIGNALS_PER_WORKER = 3  # Q + K + V

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
        qkv_done_trigger_tt,
        sdpa_q_tt,
        sdpa_k_partial_tt,
        sdpa_qk_probe_tt,
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

        # #11 Commit 3 — per-SDPA-worker Q tile count.
        # Each head's (256, 96) padded Q is split M-parallel 4 ways: each
        # worker gets (64, 96) = 2 row-tiles × 3 col-tiles = 6 tiles.
        m_tiles_per_sdpa_worker = (
            SigLIPAttentionBlockFused.M_TILES // SigLIPAttentionBlockFused.NUM_SDPA_WORKERS_PER_HEAD
        )
        assert (
            m_tiles_per_sdpa_worker * SigLIPAttentionBlockFused.NUM_SDPA_WORKERS_PER_HEAD
            == SigLIPAttentionBlockFused.M_TILES
        )
        sdpa_q_tiles_per_worker = m_tiles_per_sdpa_worker * SigLIPAttentionBlockFused.QKV_N_TILES_PER_CORE

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
            # SDPA grid bounds (task #11 Commit 2 — role-flag plumbing only).
            ("sdpa_grid_x", SigLIPAttentionBlockFused.SDPA_GRID_X),
            ("sdpa_grid_y", SigLIPAttentionBlockFused.SDPA_GRID_Y),
            # #11 Commit 3: per-head Q delivery from QKV → SDPA workers.
            ("qkv_done_trigger_cb", SigLIPAttentionBlockFused.CB["qkv_done_trigger_cb"]),
            ("sdpa_q_cb", SigLIPAttentionBlockFused.CB["sdpa_q_cb"]),
            ("sdpa_qkv_ready_sem_id", SigLIPAttentionBlockFused.SDPA_QKV_READY_SEM_ID),
            ("num_qkv_signals_per_worker", SigLIPAttentionBlockFused.NUM_QKV_SIGNALS_PER_WORKER),
            ("num_heads", SigLIPAttentionBlockFused.NUM_HEADS),
            ("num_sdpa_workers_per_head", SigLIPAttentionBlockFused.NUM_SDPA_WORKERS_PER_HEAD),
            ("m_tiles_per_sdpa_worker", m_tiles_per_sdpa_worker),
            ("sdpa_q_tiles_per_worker", sdpa_q_tiles_per_worker),
            # #11 Commit 4: SDPA grid relocation (x_offset=8, y_offset=0).
            ("sdpa_y_offset", SigLIPAttentionBlockFused.SDPA_Y_OFFSET),
            ("sdpa_x_offset", SigLIPAttentionBlockFused.SDPA_X_OFFSET),
            # #11 Commit 6: streaming K partial (3 tiles) + QK^T probe.
            ("sdpa_k_partial_cb", SigLIPAttentionBlockFused.CB["sdpa_k_partial_cb"]),
            ("sdpa_qk_probe_cb", SigLIPAttentionBlockFused.CB["sdpa_qk_probe_cb"]),
            ("sdpa_k_partial_tiles", 3),  # K row 0: 3 head_dim tiles
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
            # Role-flag bounds duplicated for TRISC (NCRISC has them above).
            ("ln1_num_cores", SigLIPAttentionBlockFused.LN1_NUM_CORES),
            ("qkv_grid_x", SigLIPAttentionBlockFused.QKV_GRID_X),
            ("qkv_grid_y", SigLIPAttentionBlockFused.QKV_GRID_Y),
            ("sdpa_grid_x", SigLIPAttentionBlockFused.SDPA_GRID_X),
            ("sdpa_grid_y", SigLIPAttentionBlockFused.SDPA_GRID_Y),
            ("sdpa_y_offset", SigLIPAttentionBlockFused.SDPA_Y_OFFSET),
            ("sdpa_x_offset", SigLIPAttentionBlockFused.SDPA_X_OFFSET),
            # #11 Commit 6: TRISC QK^T first-tile matmul tile counts.
            ("sdpa_q_tiles_per_worker", sdpa_q_tiles_per_worker),
            ("sdpa_k_partial_tiles", 3),  # head_dim_padded / TILE = 96/32 = 3
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
        # SDPA grid logical x ∈ [SDPA_X_OFFSET .. SDPA_X_OFFSET + SDPA_GRID_X - 1]
        # = [8..11], which is fully past the BH eth/PCIe gap at logical x=7.
        # Pass the 4 physical x coords so QKV NCRISC can atomic-inc each head's
        # SDPA workers without assuming contiguous physical addressing.
        _sdpa_phys_xs = [
            _device.worker_core_from_logical_core(ttnn.CoreCoord(SigLIPAttentionBlockFused.SDPA_X_OFFSET + lx, 0)).x
            for lx in range(SigLIPAttentionBlockFused.SDPA_GRID_X)
        ]

        # Kernel runs on every core in LN1 ∪ QKV ∪ SDPA. With SDPA relocated
        # to (x=SDPA_X_OFFSET..+SDPA_GRID_X-1, y=0..SDPA_GRID_Y-1) =
        # (x=8..11, y=0..7), it's disjoint from LN1+QKV (which live in
        # x=0..7). The union is two non-overlapping rectangles:
        #   Range A: LN1 ∪ QKV = (y=0..7, x=0..7)   = 64 cores
        #   Range B: SDPA       = (y=0..7, x=8..11) = 32 cores
        # Total kernel core_ranges: 96 distinct cores.
        ln1_qkv_range = ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(SigLIPAttentionBlockFused.LN1_NUM_CORES - 1, SigLIPAttentionBlockFused.QKV_GRID_Y - 1),
        )
        sdpa_x_lo = SigLIPAttentionBlockFused.SDPA_X_OFFSET
        sdpa_x_hi = sdpa_x_lo + SigLIPAttentionBlockFused.SDPA_GRID_X - 1
        sdpa_y_hi = SigLIPAttentionBlockFused.SDPA_Y_OFFSET + SigLIPAttentionBlockFused.SDPA_GRID_Y - 1
        sdpa_range = ttnn.CoreRange(
            ttnn.CoreCoord(sdpa_x_lo, SigLIPAttentionBlockFused.SDPA_Y_OFFSET),
            ttnn.CoreCoord(sdpa_x_hi, sdpa_y_hi),
        )
        union_core_grid = ttnn.CoreRangeSet({ln1_qkv_range, sdpa_range})

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
                # #11 Commit 3: SDPA workers read each head's Q from its QKV
                # source core's qkv_out_cb. Sharded persistent buffer ⇒ the
                # L1 address is the same on every QKV core, so one runtime
                # arg suffices.
                ("qkv_out_l1_addr", qkv_out_tt.buffer_address()),
            ],
            ncrisc_named_common_runtime_arg_arrays=[
                # 8 physical NoC x coords, one per LN1 sender (logical x=0..7).
                # Slots after the 4 scalar named common args above.
                ("ln1_phys_x", _ln1_phys_xs),
                # 8 physical NoC x coords for the SDPA grid (logical x=0..7).
                # Used by QKV NCRISC for the head's atomic-inc fan-out and by
                # SDPA NCRISC for self-coord lookup if needed.
                ("sdpa_phys_x", _sdpa_phys_xs),
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

        # #11 Commit 3/4: SDPA-Q-ready semaphore on the 64-core SDPA grid
        # at y=SDPA_Y_OFFSET..SDPA_Y_OFFSET+SDPA_GRID_Y-1. QKV NCRISC
        # atomic-incs each head's 4 SDPA workers; workers wait for value ≥
        # NUM_QKV_SIGNALS_PER_WORKER (= 3 once K+V lands in #11 Commit 4).
        sdpa_x_lo_sem = SigLIPAttentionBlockFused.SDPA_X_OFFSET
        sdpa_x_hi_sem = sdpa_x_lo_sem + SigLIPAttentionBlockFused.SDPA_GRID_X - 1
        sdpa_y_lo_sem = SigLIPAttentionBlockFused.SDPA_Y_OFFSET
        sdpa_y_hi_sem = sdpa_y_lo_sem + SigLIPAttentionBlockFused.SDPA_GRID_Y - 1
        sdpa_sem_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(sdpa_x_lo_sem, sdpa_y_lo_sem),
                    ttnn.CoreCoord(sdpa_x_hi_sem, sdpa_y_hi_sem),
                )
            }
        )
        sdpa_qkv_ready_sem = ttnn.SemaphoreDescriptor(
            id=SigLIPAttentionBlockFused.SDPA_QKV_READY_SEM_ID,
            core_ranges=sdpa_sem_grid,
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
                _cb(SigLIPAttentionBlockFused.CB["qkv_done_trigger_cb"], qkv_done_trigger_tt),
                _cb(SigLIPAttentionBlockFused.CB["sdpa_q_cb"], sdpa_q_tt),
                _cb(SigLIPAttentionBlockFused.CB["sdpa_k_partial_cb"], sdpa_k_partial_tt),
                _cb(SigLIPAttentionBlockFused.CB["sdpa_qk_probe_cb"], sdpa_qk_probe_tt),
            ],
            semaphores=[counter_sem, sdpa_qkv_ready_sem],
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
                qkv_done_trigger_tt,
                sdpa_q_tt,
                sdpa_k_partial_tt,
                sdpa_qk_probe_tt,
            ],
            program_descriptor,
        )
        return final_out_tt, qkv_act_tt, qkv_out_tt, sdpa_q_tt, sdpa_k_partial_tt, sdpa_qk_probe_tt


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

    # QKV matmul weight + output, WIDTH_SHARDED across the 48-core QKV grid.
    # Weight is bfp8, output is bf16. The weight is supplied as an unpadded
    # (D, 3·D) = (1152, 3456) tensor (Q,K,V concatenated, each (D, D); within
    # each, heads laid out as (D, num_heads*head_dim_true)). We pad each head's
    # head_dim from 72 → 96 by zero-extending columns, producing (D, 3·16·96)
    # = (D, 4608). The padded zero columns produce zero output columns; SDPA
    # later slices the first head_dim_true cols of each head's output.
    N_padded = SigLIPAttentionBlockFused.QKV_N  # 4608
    N_unpadded = 3 * D  # 3456
    num_heads = SigLIPAttentionBlockFused.NUM_HEADS  # 16
    head_dim_true = SigLIPAttentionBlockFused.HEAD_DIM_TRUE  # 72
    head_dim_padded = SigLIPAttentionBlockFused.HEAD_DIM_PADDED  # 96
    n_per_core = N_padded // qkv_num_cores  # 96 = exactly 1 head's padded slice

    if w_qkv_torch is None:
        g = torch.Generator().manual_seed(7)
        w_qkv_torch = torch.randn(D, N_unpadded, generator=g, dtype=torch.bfloat16) * 0.05
    assert w_qkv_torch.shape == (
        D,
        N_unpadded,
    ), f"w_qkv shape {tuple(w_qkv_torch.shape)} != expected unpadded {(D, N_unpadded)}"

    # Pad each head's head_dim from 72 → 96. Per Q/K/V (each (D, D) = (D, 1152)
    # of 16 heads of 72), the padded form is (D, 1536) of 16 heads of 96 with
    # zeros in cols [72:96] of each head block.
    w_qkv_padded = torch.zeros(D, N_padded, dtype=w_qkv_torch.dtype)
    for qkv_idx in range(3):  # Q, K, V
        for h in range(num_heads):
            src_start = qkv_idx * D + h * head_dim_true
            dst_start = qkv_idx * num_heads * head_dim_padded + h * head_dim_padded
            w_qkv_padded[:, dst_start : dst_start + head_dim_true] = w_qkv_torch[
                :, src_start : src_start + head_dim_true
            ]

    qkv_w_shard = ttnn.ShardSpec(qkv_core_grid, (D, n_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    qkv_w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, qkv_w_shard)
    qkv_w_tt = ttnn.from_torch(
        w_qkv_padded,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=qkv_w_mem,
    )

    qkv_out_shard = ttnn.ShardSpec(qkv_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    qkv_out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, qkv_out_shard)
    qkv_out_tt = ttnn.from_torch(
        torch.zeros(M, N_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=qkv_out_mem,
    )

    # #11 Commit 3: 1-tile sync CB on the 48-core QKV grid — TRISC pushes
    # after the QKV matmul; NCRISC waits then atomic-incs SDPA workers.
    qkv_done_trigger_tt = ttnn.from_torch(
        torch.zeros(qkv_num_cores * TILE, TILE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(qkv_core_grid, (TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )

    # #11 Commit 3/4: per-SDPA-worker Q / K / V CBs on the 64-core SDPA grid.
    # Grid shifted down by SDPA_Y_OFFSET=1 so SDPA workers don't double up
    # with LN1 cores at y=0 (triple-role L1 overflow). Each worker holds:
    #   sdpa_q_cb:  (rows_per_worker, head_dim_padded) = (64, 96) bf16, 6 tiles
    #   sdpa_k_cb:  (M, head_dim_padded)              = (256, 96) bf16, 24 tiles
    #   sdpa_v_cb:  (M, head_dim_padded)              = (256, 96) bf16, 24 tiles
    # K and V are NOT M-parallel split (every Q row needs the full K/V dim).
    sdpa_grid_x = SigLIPAttentionBlockFused.SDPA_GRID_X
    sdpa_grid_y = SigLIPAttentionBlockFused.SDPA_GRID_Y
    sdpa_x_offset = SigLIPAttentionBlockFused.SDPA_X_OFFSET
    sdpa_y_offset = SigLIPAttentionBlockFused.SDPA_Y_OFFSET
    sdpa_num_workers = SigLIPAttentionBlockFused.SDPA_NUM_CORES
    sdpa_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(sdpa_x_offset, sdpa_y_offset),
                ttnn.CoreCoord(sdpa_x_offset + sdpa_grid_x - 1, sdpa_y_offset + sdpa_grid_y - 1),
            )
        }
    )
    rows_per_worker = M // SigLIPAttentionBlockFused.NUM_SDPA_WORKERS_PER_HEAD  # 64
    sdpa_q_shard = ttnn.ShardSpec(sdpa_core_grid, (rows_per_worker, head_dim_padded), ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_q_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_q_shard)
    sdpa_q_tt = ttnn.from_torch(
        torch.zeros(sdpa_num_workers * rows_per_worker, head_dim_padded, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sdpa_q_mem,
    )

    # #11 Commit 6: streaming K partial (3 tiles per worker = (TILE, 3*TILE)
    # = (32, 96)) + QK^T first-tile probe (1 tile per worker = (32, 32)).
    # K-partial holds the K source's row 0 (all 3 head_dim tiles) — enough
    # for the first QK^T output tile. The streaming-read shape is the same
    # the future Commits will iterate across other K rows tile-by-tile.
    sdpa_k_partial_shard = ttnn.ShardSpec(sdpa_core_grid, (TILE, 3 * TILE), ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_k_partial_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_k_partial_shard
    )
    sdpa_k_partial_tt = ttnn.from_torch(
        torch.zeros(sdpa_num_workers * TILE, 3 * TILE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sdpa_k_partial_mem,
    )

    sdpa_qk_probe_shard = ttnn.ShardSpec(sdpa_core_grid, (TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_qk_probe_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_qk_probe_shard
    )
    sdpa_qk_probe_tt = ttnn.from_torch(
        torch.zeros(sdpa_num_workers * TILE, TILE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sdpa_qk_probe_mem,
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
        qkv_done_trigger_tt,
        sdpa_q_tt,
        sdpa_k_partial_tt,
        sdpa_qk_probe_tt,
    )

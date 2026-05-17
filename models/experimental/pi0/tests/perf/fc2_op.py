"""SigLIP MLP FC2 matmul (intermediate→hidden) — 2D 3x9 K x N parallel.

Shape: M=256, K=4304→pad 4320, N=1152.
Layout: 27 cores on 9 cols × 3 rows.
  - col index = ng ∈ [0, 9) — N-group (each owns 4 N-tiles of output)
  - row index = kg ∈ [0, 3) — K-group (each spans 45 of the 135 total K-tiles)

Activation is WIDTH_SHARDED across all 27 cores in row-major order, with shard
shape (M=256, K_per_core=160 elts = 5 K-tiles). This is the natural output
layout of FC1 (see fc1_op.py) — FC2 inherits it 1:1 with no resharding.
The shard at logical core (col=c, row=r) holds K-elements [(r*9+c)*160 ..
(r*9+c)*160+160). The kernel's BRISC then gathers the 9 K-slices that belong to
its K-group via noc_async_read.

Weight is also BLOCK_SHARDED 9×3: each core has (K_per_core=45 tiles, N_per_core=4
tiles) bfp8. Total weight = 27 cores × 45 × 4 tiles = 4860 tiles = full
(K=135, N=36) matrix.

Output is BLOCK_SHARDED 9×3: each core writes its (M=256, N_per_core=128)
slice. The 3 rows have identical content (replicated). The Python-side
to_torch then takes the first M rows for the result.

Kernel: fc2_matmul_kernel.cpp (intra-K-group activation gather + 3-way N-col reduce).
"""
import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)


class SigLIPFC2MatmulOp:
    KERNEL_SOURCE = "models/experimental/pi0/tests/perf/fc2_matmul_kernel.cpp"

    M = 256
    K_LOGICAL = 4304
    K_PADDED = 4320
    N = 1152
    TILE = 32

    M_TILES = M // TILE  # 8
    K_TILES_TOTAL = K_PADDED // TILE  # 135
    N_TILES_TOTAL = N // TILE  # 36

    # 2D grid: 9 cols × 3 rows = 27 cores
    GRID_COLS = 9
    GRID_ROWS = 3
    NUM_CORES = GRID_COLS * GRID_ROWS  # 27

    # Per-core slicing
    K_PER_CHUNK = 5  # one peer's K-slice within a K-row (=K_TILES_TOTAL / NUM_CORES)
    K_TILES_GROUP = 45  # K-tiles per K-group (=K_TILES_TOTAL / GRID_ROWS)
    N_PER_CORE = 4  # N-tiles per N-group (=N_TILES_TOTAL / GRID_COLS)

    TILE_BYTES_BF16 = TILE * TILE * 2  # 2048

    # Derived tile counts
    ACT_LOCAL_TILES = M_TILES * K_PER_CHUNK  # 8 * 5 = 40
    ACT_GATHER_TILES = M_TILES * K_TILES_GROUP  # 8 * 45 = 360
    WEIGHT_TILES = K_TILES_GROUP * N_PER_CORE  # 45 * 4 = 180
    PARTIAL_TILES = M_TILES * N_PER_CORE  # 8 * 4 = 32
    OUT_TILES = PARTIAL_TILES  # 32

    # Byte sizes
    ACT_CHUNK_BYTES = ACT_LOCAL_TILES * TILE_BYTES_BF16  # 80 KB
    ACT_GATHER_BYTES = ACT_GATHER_TILES * TILE_BYTES_BF16  # 720 KB
    PARTIAL_BYTES = PARTIAL_TILES * TILE_BYTES_BF16  # 64 KB
    RECV_BYTES = 2 * PARTIAL_BYTES  # 128 KB (2 peer slots)
    OUT_BYTES = OUT_TILES * TILE_BYTES_BF16  # 64 KB

    @staticmethod
    def op(activation_tensor, weight_tensor, output_tensor, device, math_fidelity=ttnn.MathFidelity.HiFi2):
        Cls = SigLIPFC2MatmulOp

        # CB indices
        act_local_cb = 0
        weight_cb = 1
        act_gather_cb = 2  # single-page 720 KB gather buffer
        partial_cb = 3
        recv_cb = 4
        out_cb = 5

        core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(Cls.GRID_COLS - 1, Cls.GRID_ROWS - 1))}
        )

        # ----- Global semaphore for N-col recv signaling -----
        recv_sem = ttnn.create_global_semaphore(device, core_grid, 0)

        # ----- Build peer-mapping for all 27 cores -----
        # Logical core (col, row) → physical NoC coord
        phys = {}
        for col in range(Cls.GRID_COLS):
            for row in range(Cls.GRID_ROWS):
                p = device.worker_core_from_logical_core(ttnn.CoreCoord(col, row))
                phys[(col, row)] = (p.x, p.y)

        # Per-core runtime args (BRISC) and per-core "my_kg" compile-time values.
        brisc_per_core_args = []
        my_kg_core_values = []

        for col in range(Cls.GRID_COLS):
            for row in range(Cls.GRID_ROWS):
                my_col, my_kg = col, row

                # 8 K-row peers (same kg, different col)
                kg_peers = []
                for c in range(Cls.GRID_COLS):
                    if c == my_col:
                        continue
                    px, py = phys[(c, my_kg)]
                    kg_peers.append((px, py, c))  # k_slot == c

                # 2 N-col peers (same col, different kg)
                nc_peers = []
                for r in range(Cls.GRID_ROWS):
                    if r == my_kg:
                        continue
                    px, py = phys[(my_col, r)]
                    nc_peers.append((px, py))

                rt = [my_col, my_kg]
                for px, py, ks in kg_peers:
                    rt += [px, py, ks]
                for px, py in nc_peers:
                    rt += [px, py]
                rt += [ttnn.get_global_semaphore_address(recv_sem)]

                brisc_per_core_args.append((ttnn.CoreCoord(col, row), rt))
                my_kg_core_values.append((ttnn.CoreCoord(col, row), my_kg))

        per_core_ct_descriptors = [
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="my_kg",
                core_values=my_kg_core_values,
                other_value=0,
            ),
        ]

        # ----- Compile-time args -----
        ncrisc_ct = [
            ("act_local_cb", act_local_cb),
            ("weight_cb", weight_cb),
            ("out_cb", out_cb),
            ("act_local_tiles", Cls.ACT_LOCAL_TILES),
            ("weight_tiles", Cls.WEIGHT_TILES),
            ("out_tiles", Cls.OUT_TILES),
        ]
        brisc_ct = [
            ("act_local_cb", act_local_cb),
            ("act_gather_cb", act_gather_cb),
            ("partial_cb", partial_cb),
            ("recv_cb", recv_cb),
            ("out_cb", out_cb),
            ("act_chunk_bytes", Cls.ACT_CHUNK_BYTES),
            ("partial_bytes", Cls.PARTIAL_BYTES),
            ("act_local_tiles", Cls.ACT_LOCAL_TILES),
            ("act_gather_tiles", Cls.ACT_GATHER_TILES),
            ("partial_tiles", Cls.PARTIAL_TILES),
        ]
        trisc_ct = [
            ("act_gather_cb", act_gather_cb),
            ("weight_cb", weight_cb),
            ("partial_cb", partial_cb),
            ("recv_cb", recv_cb),
            ("out_cb", out_cb),
            ("m_tiles", Cls.M_TILES),
            ("k_tiles", Cls.K_TILES_GROUP),
            ("k_per_chunk", Cls.K_PER_CHUNK),
            ("n_per_core", Cls.N_PER_CORE),
            ("act_gather_tiles", Cls.ACT_GATHER_TILES),
            ("weight_tiles", Cls.WEIGHT_TILES),
            ("partial_tiles", Cls.PARTIAL_TILES),
            # "my_kg" is supplied per-core via PerCoreCompileTimeDescriptor.
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=Cls.KERNEL_SOURCE,
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=ncrisc_ct,
            brisc_named_compile_time_args=brisc_ct,
            trisc_named_compile_time_args=trisc_ct,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=math_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=True,
            ),
            per_core_compile_time_descriptors=per_core_ct_descriptors,
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                brisc_args=brisc_per_core_args,
            ),
        )

        # ----- CB descriptors -----
        full_tile = ttnn.Tile((Cls.TILE, Cls.TILE))
        tile_descriptor = ttnn.TileDescriptor(full_tile)
        bf16_page_size = full_tile.get_tile_size(ttnn.bfloat16)
        bfp8_page_size = full_tile.get_tile_size(ttnn.bfloat8_b)

        def _sharded_cb(cb_id, tensor, page_size):
            d = ttnn.cb_descriptor_from_sharded_tensor(cb_id, tensor)
            d.format_descriptors[0].tile = tile_descriptor
            d.format_descriptors[0].page_size = page_size
            return d

        # Intermediate (non-tensor-backed) CBs
        def _intermed_cb(cb_id, total_size, data_format=ttnn.bfloat16):
            fmt = ttnn.CBFormatDescriptor(
                buffer_index=cb_id,
                data_format=data_format,
                page_size=full_tile.get_tile_size(data_format),
                tile=tile_descriptor,
            )
            return ttnn.CBDescriptor(
                total_size=total_size,
                core_ranges=core_grid,
                format_descriptors=[fmt],
            )

        cb_list = [
            _sharded_cb(act_local_cb, activation_tensor, bf16_page_size),
            _sharded_cb(weight_cb, weight_tensor, bfp8_page_size),
            _intermed_cb(act_gather_cb, Cls.ACT_GATHER_BYTES, ttnn.bfloat16),
            _intermed_cb(partial_cb, Cls.PARTIAL_BYTES, ttnn.bfloat16),
            _intermed_cb(recv_cb, Cls.RECV_BYTES, ttnn.bfloat16),
            _sharded_cb(out_cb, output_tensor, bf16_page_size),
        ]

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=cb_list,
            semaphores=[],
        )

        ttnn.generic_op(
            [activation_tensor, weight_tensor, output_tensor],
            program_descriptor,
        )
        return output_tensor


def build_tensors_for_fc2_test(device, w_torch_padded, x_torch_padded):
    """Args:
        w_torch_padded: (K=4320, N=1152) bf16/fp32, K-padded weight (W^T of HF original)
        x_torch_padded: (M=256, K=4320) bf16, K-padded activation (FC1 output shape)

    Layout:
        activation_tt: WIDTH_SHARDED, 27 cores in 9×3 row-major. Shard (M, 160 elts).
                       Matches FC1's output exactly.
        weight_tt:     BLOCK_SHARDED 9×3. Each core's shard is (K_per_core=45 tiles
                       = 1440 elts, N_per_core=4 tiles = 128 elts).
        output_tt:     BLOCK_SHARDED 9×3 — N dim split 9-way, K-row replicated.
                       Final result is in row 0 (all rows identical after reduce).
    """
    Cls = SigLIPFC2MatmulOp
    M, K, N = Cls.M, Cls.K_PADDED, Cls.N
    assert x_torch_padded.shape == (M, K), f"x shape {x_torch_padded.shape}"
    assert w_torch_padded.shape == (K, N), f"w shape {w_torch_padded.shape}"

    core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(Cls.GRID_COLS - 1, Cls.GRID_ROWS - 1))}
    )

    # --- Activation: WIDTH_SHARDED across 27 cores, shard (M, K/27=160) ---
    k_per_core = K // Cls.NUM_CORES  # 160
    act_shard = ttnn.ShardSpec(core_grid, (M, k_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    act_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, act_shard)
    # Reshape to flat (M, K) — width-sharding will slice K across cores.
    activation_tt = ttnn.from_torch(
        x_torch_padded,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=act_mem,
    )

    # --- Weight: BLOCK_SHARDED 9×3 grid, shard (K_per_core_elts=1440, N_per_core_elts=128) bfp8 ---
    k_per_core_w = Cls.K_TILES_GROUP * Cls.TILE  # 1440
    n_per_core_w = Cls.N_PER_CORE * Cls.TILE  # 128
    w_shard = ttnn.ShardSpec(core_grid, (k_per_core_w, n_per_core_w), ttnn.ShardOrientation.ROW_MAJOR)
    w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, w_shard)
    weight_tt = ttnn.from_torch(
        w_torch_padded,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=w_mem,
    )

    # --- Output: BLOCK_SHARDED 9×3, shard (M, N_per_core=128), 3 K-rows replicated ---
    # Build a (M*3, N) "stacked" tensor of zeros; the kernel writes the same content
    # into each of the 3 row-replicas after the all-reduce.
    out_shard = ttnn.ShardSpec(core_grid, (M, n_per_core_w), ttnn.ShardOrientation.ROW_MAJOR)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, out_shard)
    output_tt = ttnn.from_torch(
        torch.zeros(M * Cls.GRID_ROWS, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem,
    )

    return activation_tt, weight_tt, output_tt

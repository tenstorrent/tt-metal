"""Iteration 31: skinny_matmul's x stream reads ahead: up to 8 blocks (32 tiles) of a row's K slice in flight before one
barrier instead of a barrier per 4-tile block (the partial kernel was latency-bound: 36 us per call for a 10 MB read
over 104 cores). CB x holds two such chunks; chunks are always full (a shorter tail is padded, the compute pops it)."""
import sys

M = "tt-train/sources/ttml/metal/ops/skinny_matmul/device"
RD = f"{M}/kernels/dataflow/reader_skinny_matmul.cpp"
CK = f"{M}/kernels/compute/skinny_matmul_partial_kernel.cpp"
F = f"{M}/skinny_matmul_program_factory.cpp"


def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        print(f"FAILED {path}: {old[:90]!r} found {n}, expected {count}")
        sys.exit(2)
    open(path, "w").write(s.replace(old, new))


sub(
    RD,
    "constexpr uint32_t Kx = get_compile_time_arg_val(7);  // x row stride in tiles (>= Kt)\n",
    "constexpr uint32_t Kx = get_compile_time_arg_val(7);  // x row stride in tiles (>= Kt)\n"
    "// x reads in flight before a barrier (iteration 31): up to kAheadBlocks blocks of a row's slice; the CB holds two such\n"
    "// chunks and every chunk is pushed whole (a shorter tail is padded with unused tiles, the compute pops them).\n"
    "constexpr uint32_t kAheadBlocks = ((St + block - 1U) / block < 8U) ? ((St + block - 1U) / block) : 8U;\n"
    "constexpr uint32_t chunk_tiles = kAheadBlocks * block;\n",
)
sub(
    RD,
    """        // x tiles in full `block` pushes (the tail of a slice reads fewer tiles into a full block) for the same reason.
        for (uint32_t r = r0; r < r0 + nr; ++r) {
            for (uint32_t k = 0; k < nk; k += block) {
                const uint32_t n = (k + block <= nk) ? block : (nk - k);
                cb_reserve_back(cb::x, block);
                uint32_t l1 = get_write_ptr(cb::x);
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_read_page(r * Kx + k0 + k + j, in0_acc, l1);
                    l1 += tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb::x, block);
            }
        }
""",
    """        // x tiles in whole chunks of kAheadBlocks blocks (the tail of a slice reads fewer tiles into a full chunk) for
        // the same reason; one barrier per chunk.
        for (uint32_t r = r0; r < r0 + nr; ++r) {
            for (uint32_t k = 0; k < nk; k += chunk_tiles) {
                const uint32_t n = (k + chunk_tiles <= nk) ? chunk_tiles : (nk - k);
                cb_reserve_back(cb::x, chunk_tiles);
                uint32_t l1 = get_write_ptr(cb::x);
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_read_page(r * Kx + k0 + k + j, in0_acc, l1);
                    l1 += tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb::x, chunk_tiles);
            }
        }
""",
)
sub(
    CK,
    "constexpr uint32_t block = get_compile_time_arg_val(7);\n",
    "constexpr uint32_t block = get_compile_time_arg_val(7);\n"
    "// The reader pushes x in whole chunks of kAheadBlocks blocks (iteration 31): the padding blocks of a slice's last\n"
    "// chunk are popped unread.\n"
    "constexpr uint32_t kAheadBlocks = ((St + block - 1U) / block < 8U) ? ((St + block - 1U) / block) : 8U;\n",
)
sub(
    CK,
    """            for (uint32_t k = 0; k < nk; k += block) {
                const uint32_t n = (k + block <= nk) ? block : (nk - k);
                cb_wait_front(cb::x, block);
                for (uint32_t j = 0; j < n; ++j) {
                    for (uint32_t col = 0; col < Nt; ++col) {
                        matmul_tiles(cb::x, cb::w, j, (k + j) * Nt + col, col);
                    }
                }
                cb_pop_front(cb::x, block);
            }
            tile_regs_commit();""",
    """            uint32_t blocks_popped = 0;
            for (uint32_t k = 0; k < nk; k += block) {
                const uint32_t n = (k + block <= nk) ? block : (nk - k);
                cb_wait_front(cb::x, block);
                for (uint32_t j = 0; j < n; ++j) {
                    for (uint32_t col = 0; col < Nt; ++col) {
                        matmul_tiles(cb::x, cb::w, j, (k + j) * Nt + col, col);
                    }
                }
                cb_pop_front(cb::x, block);
                ++blocks_popped;
            }
            // The chunk padding of the slice's tail (whole chunks of kAheadBlocks blocks are pushed).
            const uint32_t chunks = (blocks_popped + kAheadBlocks - 1U) / kAheadBlocks;
            for (uint32_t b = blocks_popped; b < chunks * kAheadBlocks; ++b) {
                cb_wait_front(cb::x, block);
                cb_pop_front(cb::x, block);
            }
            tile_regs_commit();""",
)
sub(
    F,
    "    create_circular_buffer(program, split.all_cores, cb::x, bf16, bf16_tile, 2U * block);\n",
    "    // Two chunks of up to 8 blocks (iteration 31: the reader's x reads in flight before a barrier).\n"
    "    const uint32_t ahead_blocks = std::min(8U, (St + block - 1U) / block);\n"
    "    create_circular_buffer(program, split.all_cores, cb::x, bf16, bf16_tile, 2U * ahead_blocks * block);\n",
)
print("iteration 31 patch applied")

# SPDX-License-Identifier: Apache-2.0
"""Serial-prefill resident FP32 GDN recurrence; owner/decode paths are inherited.

Embedded kernel bytes match the isolated scalar-cache plus tail candidate.
No runtime source rewriting, test imports, or alternate weights are used.
"""

import hashlib
from pathlib import Path

import ttnn

from .gdn_prefill_qk import PrefillQKRowMajorGDN

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

uint32_t off(uint32_t r, uint32_t c) {
    return (r/16)*512 + (c/16)*256 + (r%16)*16 + c%16;
}
void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto fa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const auto decay = TensorAccessor(da, get_arg_val<uint32_t>(3));
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(4));
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(5));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(6));
    const auto final = TensorAccessor(fa, get_arg_val<uint32_t>(7));
    const uint32_t first = get_arg_val<uint32_t>(8), count = get_arg_val<uint32_t>(9);
    const uint32_t length=get_arg_val<uint32_t>(10), st=(length+31)/32;
    const uint32_t poison=get_arg_val<uint32_t>(11);
    cb_reserve_back(15,1);
    auto assembled=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(15));
    Noc noc;
    cb_reserve_back(14, 1);
    const uint32_t scratch = get_write_ptr(14);
    cb_reserve_back(16,2);
    const uint32_t scalar_cache = get_write_ptr(16);
    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<6, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL>();
    for (uint32_t job = first; job < first + count; ++job) {
        const uint32_t head=job/8, column=(job%8)/2, half=job%2;
        const uint32_t b = head/12, h = head%12;
        for(uint32_t token=0;token<length;++token) {
        if(token==0) {
        cb_reserve_back(0,4);
        for (uint32_t i=0;i<4;++i) {
            if(token==0) {
                auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+i*4096);
                for(uint32_t f=0;f<2;++f) {
                    for(uint32_t j=0;j<256;++j) dst[f*512+256+j]=poison;
                    noc.async_read(state,CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096+f*2048),1024,
                        {.page_id=head*16+i*4+column,.offset_bytes=f*2048+half*1024},{});
                }
            }
            // Later tokens reuse the FP32 bytes through aliased CB0/CB12.

        }
        noc.async_read_barrier();
        cb_push_back(0,4);
        }
        // Raw 32-bit replication: no floating point conversion on dataflow RISCs.
        for (uint32_t which=1;which<=3;++which) {
            cb_reserve_back(which,which==3?1:4);
            auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            if(which!=3) {
                const uint32_t row=(b*length+token)*4+h/3;
                if(which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),512,{.page_id=row},{});
                else noc.async_read(k,CoreLocalMem<uint32_t>(scratch),512,{.page_id=row},{});
                noc.async_read_barrier();
            }
            for (uint32_t tile=0;tile<(which==3?1:4);++tile) {
                uint32_t page = which==3 ? ((b*length+token)+h/32)*4+column : (b*length+token)*4+tile;
                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),64,{.page_id=page,.offset_bytes=4*off(h%32,half*16)},{});
                if(which==3) noc.async_read_barrier();
                if (which==3) {
                    for(uint32_t c=0;c<16;++c) {
                        dst[tile*1024+off(0,c)]=src[c];
                        dst[tile*1024+off(0,c+16)]=poison;
                    }
                } else {
                    for(uint32_t r=0;r<32;++r) {
                        const uint32_t value=src[tile*32+r];
                        dst[tile*1024+off(r,0)]=value;
                    }
                }
            }
            cb_push_back(which,which==3?1:4);
        }
        if(token%32==0) {
            noc.async_read(decay,CoreLocalMem<uint32_t>(scalar_cache),4096,{.page_id=(b*st+token/32)+h/32},{});
            noc.async_read(beta,CoreLocalMem<uint32_t>(scalar_cache+4096),4096,{.page_id=(b*st+token/32)+h/32},{});
            noc.async_read_barrier();
        }
        for(uint32_t which=4;which<=5;++which) {
            cb_reserve_back(which,1);
            auto scalar_src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scalar_cache+(which-4)*4096);
            const uint32_t value=scalar_src[off(token%32,h%32)];
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            dst[0]=value;
            cb_push_back(which,1);
        }
        if(token+1==length) cb_wait_front(12,4);
        if(token+1==length) for(uint32_t i=0;i<4;++i) for(uint32_t f=0;f<2;++f)
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096+f*2048),final,1024,{},
                {.page_id=head*16+i*4+column,.offset_bytes=f*2048+half*1024});
        noc.async_write_barrier();
        // Final state is still read by compute until output has been published.
        cb_wait_front(13,1);
        auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(13));
        if(token%32==0) for(uint32_t i=0;i<1024;++i) assembled[i]=0;
        for(uint32_t t=0;t<1;++t) for(uint32_t c=0;c<16;++c)
            assembled[t*1024+off(token%32,c)]=out[t*1024+off(0,c)];
        if(token%32==31 || token+1==length) {
            for(uint32_t f=0;f<2;++f)
                noc.async_write(CoreLocalMem<uint32_t>(get_write_ptr(15)+f*2048),output,1024,{},
                    {.page_id=(head*st+token/32)*4+column,.offset_bytes=f*2048+half*1024});
            noc.async_write_barrier();
        }
        cb_pop_front(13,1);
        if(token+1==length) cb_pop_front(12,4);
        }
    }
}
"""

COMPUTE = r"""#include "api/compute/bcast.h"

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

// SPDX-License-Identifier: Apache-2.0
// TEST ONLY: exact native FP32 column SUM body, pruning independent right-column groups.
// Full 32x32 CB/DST storage is retained. Native initialization remains unchanged.
#if defined(TRISC_MATH)
namespace ckernel {
namespace sfpu {
template <
    PoolType pool_type,
    InstrModLoadStore INSTRUCTION_MODE,
    bool clear_high_bits,
    bool pack_low16,
    bool is_signed_int>
inline void half_column_reduce_sum_avg() {
    // Determine if integer or float mode at compile time
    constexpr bool is_integer_mode =
        (INSTRUCTION_MODE == InstrModLoadStore::INT32 || INSTRUCTION_MODE == InstrModLoadStore::LO16);

    constexpr std::uint32_t UPPER_FACE_ADDRS[NUM_FACES] = {0, 0, 16, 16};    // Face 0, 0, 1, 1
    constexpr std::uint32_t LOWER_FACE_ADDRS[NUM_FACES] = {32, 32, 48, 48};  // Face 2, 2, 3, 3
    constexpr std::uint32_t COLUMN_OFFSETS[NUM_FACES] = {0, 2, 0, 2};        // even, odd, even, odd

    // Optimized column reduction: Reduce → Add → Transpose → HalfReduce
    // Instead of the naive Transpose → Reduce → Transpose → Reduce → Add approach, we first reduce
    // across registers, then add upper+lower faces (all 4 positions carry meaningful partial sums),
    // then transpose, then do a final half-reduce on LREG0-3 only. This eliminates one transpose
    // and halves the second reduction pass, saving 4 instructions per iteration.
    for (std::uint32_t i = 0; i < 2; i++) {
        const std::uint32_t upper_face_addr = UPPER_FACE_ADDRS[i];
        const std::uint32_t lower_face_addr = LOWER_FACE_ADDRS[i];
        const std::uint32_t column_offset = COLUMN_OFFSETS[i];

        // Step 1: Tree-reduce across registers (LREG0-3→LREG0, LREG4-7→LREG4) without transpose.
        // After this, each of the 4 positions in LREG0 holds the sum of rows at that position
        // across all 4 loaded LREGs (e.g., LREG0[i] = sum of row[i], row[i+4], row[i+8], row[i+12]).
        load_face_data<INSTRUCTION_MODE, clear_high_bits>(upper_face_addr, lower_face_addr, column_offset);
        lltt::replay(0, 6);

        // Step 2: Cross-face addition. Unlike the old approach where only position 0 of the
        // cross-face sum was meaningful, here ALL 4 positions carry useful partial sums.
        if constexpr (is_integer_mode) {
            TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);  // LREG0 = upper + lower (int)
        } else {
            TTI_SFPADD(
                p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);  // LREG0 = upper + lower (float)
        }

        // Result of column reduction now stored in LREG0 as 4 partial sums
        // Step 3: Transpose to rearrange the 4 partial sums for final reduction
        TTI_SFPTRANSP(0, 0, 0, 0);

        // Step 4: Final tree-reduce across LREG0-3 only (LREG4-7 no longer needed).
        // This sums the 4 partial sums into LREG0[0] = total column sum.
        lltt::replay(6, 3);

        // Perform averaging if requested (different for int vs float)
        if constexpr (pool_type == PoolType::AVG) {
            if constexpr (is_integer_mode) {
                perform_int_average<INSTRUCTION_MODE, is_signed_int>();
            } else {
                perform_float_average();
            }
        }
        // Store the final column sum/average to the first row.
        // Mode 9 (SFPSTORE_MOD0_FMT_LO16) is only needed when the packer-visible OUTPUT is UInt16 in a
        // 32-bit dest: there the reduced value sits in the low 16 bits but the packer reads the high 16,
        // so we move low->high. When the output is a full 32-bit format (e.g. UInt32) the packer reads
        // the whole dest word, so we use the plain INSTRUCTION_MODE store even for UInt16 input.
        constexpr std::uint32_t STORE_MODE =
            pack_low16 ? 9u /* SFPSTORE_MOD0_FMT_LO16 */ : static_cast<std::uint32_t>(INSTRUCTION_MODE);
        // Runtime-address store (TT_ not TTI_): the face address comes from the loop index and is not
        // guaranteed to fold to a compile-time constant.
        TT_SFPSTORE(p_sfpu::LREG0, STORE_MODE, ADDR_MOD_7, upper_face_addr + column_offset);
    }
}

inline void half_column_reduce_sum() {
    half_column_reduce_sum_avg<
        PoolType::SUM,
        GetSfpLoadStoreInstrMod<DataFormat::Float32, true>(),
        false,
        false,
        false>();
}
}  // namespace sfpu
}  // namespace ckernel
#endif

ALWI void half_mul(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
        a,
        b,
        out,
        VectorMode::C)));
}
ALWI void half_add(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        VectorMode::C)));
}
ALWI void half_sub(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        out,
        VectorMode::C)));
}

template <uint32_t Out>
void half_reduce_product() {
    cb_wait_front(8, 4);
    reconfig_data_format(8, 6);
    pack_reconfig_data_format(Out);
    copy_init(8);
    cb_wait_front(6, 1);
    PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL, PackMode::Default>(Out)));
    tile_regs_acquire();
    add_binary_tile_init();
    copy_tile(8, 0, 0);
    // Preserve native Accurate reducer's K-tile fold order and work-DST index.
    constexpr uint32_t work_dst = compute_kernel_lib::DEST_AUTO_LIMIT - 1;
    for (uint32_t ht = 1; ht < 4; ++ht) {
        copy_tile(8, ht, work_dst);
        half_add(0, work_dst, 0);
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, half_column_reduce_sum, 0, VectorMode::RC_custom)));
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(Out, 1);
    pack_tile(0, Out);
    cb_push_back(Out, 1);
    tile_regs_release();
    PACK((llk_pack_reduce_mask_clear()));
    cb_pop_front(8, 4);
}

void load(uint32_t cb,uint32_t tile,uint32_t dst) {
    reconfig_data_format_srca(cb);
    if(cb==1 || cb==2) {
        unary_bcast_init<BroadcastType::COL>(cb);
        unary_bcast<BroadcastType::COL>(cb,tile,dst);
        unary_bcast_uninit<BroadcastType::COL>(cb);
    } else if(cb==4 || cb==5) {
        unary_bcast_init<BroadcastType::SCALAR>(cb);
        unary_bcast<BroadcastType::SCALAR>(cb,tile,dst);
        unary_bcast_uninit<BroadcastType::SCALAR>(cb);
    } else {
        copy_init(cb);
        copy_tile(cb,tile,dst);
    }
}
void finish(uint32_t cb,uint32_t tile) {
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,cb,tile); tile_regs_release();
}
// SPDX-License-Identifier: Apache-2.0
// TEST ONLY. Prepended helpers are checksum-pinned from the validated half-column probe.
// Full-sync FP32 DST0..3 retain state, DST4 accumulator/delta, DST5/6 scratch.
// Every multiply/add is separate and retains the baseline operand and K-tile order.
// SPDX-License-Identifier: Apache-2.0
// TEST ONLY. Native row-broadcast instructions with right-face data loads/stores omitted.
// All four transposed-register multiplies and both transpose operations remain unchanged.
#if defined(TRISC_MATH)
namespace ckernel { namespace sfpu {
template <BinaryOp BINOP>
inline void _resident_left_row_band_(std::uint32_t data_tile_base, std::uint32_t out_tile_base, std::uint32_t face_pair_base, std::uint32_t band_off)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    const std::uint32_t data_slot_base = data_tile_base + face_pair_base + band_off;
    const std::uint32_t out_slot_base  = out_tile_base + face_pair_base + band_off;

    // (1) Load data's 4 col-groups into LREG0..3. Bcast (LREG4..7) was
    //     pre-loaded once by the caller and survives across iterations.
    TT_SFPLOAD(p_sfpu::LREG0, IM, ADDR_MOD_7, data_slot_base + COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG1, IM, ADDR_MOD_7, data_slot_base + COL_GROUP_OFFSETS[1]);

    // (2) Transpose: LREG0..3 become the 4 data rows; LREG4..7 become the
    //     transposed bcast rows (of which only LREG4 = bcast row 0 is useful).
    TTI_SFPTRANSP(0, 0, 0, 0);

    // (3) In-register eltwise: LREG_k = LREG_k OP LREG4 for k in 0..3.
    //
    // Pipelining: the 4 binops all read LREG4 and a distinct LREG_k, and each
    // writes a distinct LREG_k. They are pairwise independent, so we can
    // issue them back-to-back with no NOPs between. Only a single trailing
    // NOP is needed before SFPTRANSP reads LREG0..3 (2-cycle SFPADD/SFPMUL
    // latency).
    //
    // IMPORTANT: this block must not write LREG4..7. The persistent bcast
    // values rely on the second SFPTRANSP being a pure inverse of the first
    // for that group. The SUB path uses SFPMAD with LCONST_neg1
    // (data = bcast * -1 + data) to fuse the negation into the FMA instead
    // of the more obvious SFPMOV-NEGATE-into-LREG5 + SFPADD pattern, which
    // would corrupt LREG5 and force a reload of the bcast every band.
    if constexpr (BINOP == BinaryOp::ADD)
    {
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG1, 0);
        TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG2, 0);
        TTI_SFPADD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG4, p_sfpu::LREG3, 0);
    }
    else if constexpr (BINOP == BinaryOp::SUB)
    {
        // data - bcast = bcast * (-1) + data, fused into a single SFPMAD per
        // element so we never write LREG4..7 and the bcast survives the
        // second SFPTRANSP.
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG2, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LCONST_neg1, p_sfpu::LREG3, p_sfpu::LREG3, 0);
    }
    else if constexpr (BINOP == BinaryOp::MUL)
    {
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    }
    else
    {
        static_assert(BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::MUL, "SFPU binary-bcast kernel only supports ADD, SUB, MUL");
    }
    TTI_SFPNOP; // single drain before SFPTRANSP reads LREG0..3.

    // (4) Transpose back. Since step (3) left LREG4..7 untouched, this also
    //     restores the bcast registers to their pre-transpose face-major
    //     layout, ready for the next band's first SFPTRANSP.
    TTI_SFPTRANSP(0, 0, 0, 0);

    // (5) Store LREG0..3 to the 4 col-groups of the output band.
    TT_SFPSTORE(p_sfpu::LREG0, IM, ADDR_MOD_7, out_slot_base + COL_GROUP_OFFSETS[0]);
    TT_SFPSTORE(p_sfpu::LREG1, IM, ADDR_MOD_7, out_slot_base + COL_GROUP_OFFSETS[1]);
}

template <BinaryOp BINOP>
inline void _resident_left_row_tile_(std::uint32_t dst_index_data, std::uint32_t dst_index_bcast, std::uint32_t dst_index_out)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;

    const std::uint32_t data_base  = dst_index_data * DEST_TILE_SIZE_RAW;
    const std::uint32_t bcast_base = dst_index_bcast * DEST_TILE_SIZE_RAW;
    const std::uint32_t out_base   = dst_index_out * DEST_TILE_SIZE_RAW;

    // Hoisted bcast load: bcast row 0 lives in face 0 row band 0 and is
    // identical for every output band, so we load its 4 col-groups exactly
    // once. The per-band binop is structured to leave LREG4..7 unchanged,
    // so the second SFPTRANSP per band restores them to this face-major
    // layout.
    const std::uint32_t bcast_row0_base = bcast_base + FACE0_BASE;
    TT_SFPLOAD(p_sfpu::LREG4, IM, ADDR_MOD_7, bcast_row0_base + COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG5, IM, ADDR_MOD_7, bcast_row0_base + COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG6, IM, ADDR_MOD_7, bcast_row0_base + COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG7, IM, ADDR_MOD_7, bcast_row0_base + COL_GROUP_OFFSETS[3]);

    // Upper half (face 0/1): bands 0..3.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++)
    {
        _resident_left_row_band_<BINOP>(data_base, out_base, FACE0_BASE, band * ROW_BAND_STRIDE);
    }

    // Lower half (face 2/3): bands 0..3.
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; band++)
    {
        _resident_left_row_band_<BINOP>(data_base, out_base, FACE2_BASE, band * ROW_BAND_STRIDE);
    }
}

} }
#endif
ALWI void resident_row_delta_sub(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, 2, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a, b, out, VectorMode::RC_custom)));
}
ALWI void resident_row_delta_mul(uint32_t a, uint32_t b, uint32_t out) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 2, DST_ACCUM_MODE), a, b, out, VectorMode::RC_custom)));
}
ALWI void resident_left_mul_bcast_row(uint32_t data, uint32_t row) {
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _resident_left_row_tile_,
        (ckernel::BinaryOp::MUL), data, row, data, VectorMode::None)));
}

ALWI void reduce_dst4() {
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, half_column_reduce_sum, 4, VectorMode::RC_custom)));
}

void kernel_main() {
    static_assert(DST_SYNC_MODE == DstSync::SyncFull);
    static_assert(DST_ACCUM_MODE);
    static_assert(compute_kernel_lib::DEST_AUTO_LIMIT == 8);
    compute_kernel_hw_startup(0, 4, 7);
    const uint32_t jobs = get_arg_val<uint32_t>(0);
    const uint32_t length = get_arg_val<uint32_t>(1);
    for (uint32_t job = 0; job < jobs; ++job) {
        cb_wait_front(0, 4);
        tile_regs_acquire();
        for (uint32_t i = 0; i < 4; ++i) load(0, i, i);
        cb_pop_front(0, 4);
        for (uint32_t token = 0; token < length; ++token) {
            if (token != 0) tile_regs_acquire();
            cb_wait_front(1, 4);
            cb_wait_front(2, 4);
            cb_wait_front(3, 1);
            cb_wait_front(4, 1);
            cb_wait_front(5, 1);
            // Decay: state * decay, unchanged operand order, FP32 rounding after MUL.
            load(4, 0, 4);
            for (uint32_t i = 0; i < 4; ++i) {
                mul_binary_tile_init();
                half_mul(i, 4, i);
            }
            // Prediction: ((state0*K0 + state1*K1) + state2*K2) + state3*K3.
            for (uint32_t i = 0; i < 4; ++i) {
                load(2, i, 6);
                mul_binary_tile_init();
                half_mul(i, 6, i == 0 ? 4 : 5);
                if (i != 0) {
                    add_binary_tile_init();
                    half_add(4, 5, 4);
                }
            }
            reduce_dst4();
            // Delta: (V - prediction) * beta. No fused multiply-add.
            load(3, 0, 5);
            load(5, 0, 6);
            sub_binary_tile_init();
            resident_row_delta_sub(5, 4, 4);
            mul_binary_tile_init();
            resident_row_delta_mul(4, 6, 4);
            // Update: (K * delta_row) + decayed_state, preserving add operand order.
            // The validated SFPU row broadcast reads only row0 of DST4; no CB11 roundtrip.
            for (uint32_t i = 0; i < 4; ++i) {
                load(2, i, 6);
                sfpu_mul_bcast_row_init();
                resident_left_mul_bcast_row(6, 4);
                add_binary_tile_init();
                half_add(6, i, i);
            }
            // Output uses the updated state and the identical four-K-tile fold/tree.
            for (uint32_t i = 0; i < 4; ++i) {
                load(1, i, 6);
                mul_binary_tile_init();
                half_mul(i, 6, i == 0 ? 4 : 5);
                if (i != 0) {
                    add_binary_tile_init();
                    half_add(4, 5, 4);
                }
            }
            reduce_dst4();
            cb_reserve_back(13, 1);
            if (token + 1 == length) cb_reserve_back(12, 4);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(13);
            PACK((llk_pack_reduce_mask_config<ReduceDim::REDUCE_COL, PackMode::Default>(13)));
            pack_tile(4, 13);
            PACK((llk_pack_reduce_mask_clear()));
            if (token + 1 == length) {
                pack_reconfig_data_format(12);
                for (uint32_t i = 0; i < 4; ++i) pack_tile(i, 12, i);
            }
            // Native no-clear release waits for PACK before releasing MATH; full-sync
            // never flips DST banks. Standard tile_regs_release would zero all state.
            PACK((_llk_pack_dest_section_done_<DstSync::SyncFull, true, false>()));
            cb_push_back(13, 1);
            if (token + 1 == length) cb_push_back(12, 4);
            cb_pop_front(1, 4);
            cb_pop_front(2, 4);
            cb_pop_front(3, 1);
            cb_pop_front(4, 1);
            cb_pop_front(5, 1);
        }
    }
    cb_pop_front(6, 1);
}
"""

NATIVE_CONTRACT_SHA256 = {
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h": "933caf779e7deb9476450deba04553ab900b35222357a5f52294b3bc395b18b4",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h": "343195b140e5d1eabcba970b5e8f9dc0c9107a51b0dfaa53920ca4e4efa1283e",
    "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp": "581b21aa6830f1d2d0429a33aa9af81639d95f005e4d9ee38912363d6dcc72c8",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h": "b8f85a9151d88409aefb0643d37291f7d330c9f1385ac0e4a8fb4334c4dfcd39",
    "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h": "47fbf6f1be529f34f261d6c3567f4e8c83763fc0b50b5ebf9678f12973705829",
    "tt_metal/hw/inc/internal/debug/sanitize.h": "f71d92dba723b34101966eda13fc784f47b83af9d3606426d47dc6a0171773e9",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary_bcast.h": "28bf9237ad07c5e71ad587a488c0c5927085474d9186c08f441062bfc0f85584",
}

_REPO = Path(__file__).resolve().parents[4]
for _name, _expected in NATIVE_CONTRACT_SHA256.items():
    if hashlib.sha256((_REPO / _name).read_bytes()).hexdigest() != _expected:
        raise ValueError(f"Resident GDN native contract changed: {_name}")
_PROVENANCE = dict(
    helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    reader_sha256=hashlib.sha256(READER.encode()).hexdigest(),
    compute_sha256=hashlib.sha256(COMPUTE.encode()).hexdigest(),
    native_contract_sha256=NATIVE_CONTRACT_SHA256,
)


def _record_signature(self, batch, length, workers, config, cbs):
    scalar = next(cb for cb in cbs if any(fmt.buffer_index == 16 for fmt in cb.format_descriptors))
    state = next(cb for cb in cbs if {fmt.buffer_index for fmt in cb.format_descriptors} == {0, 12})
    signature = dict(
        sequence=length,
        batch=batch,
        workers=workers,
        output_columns=16,
        local_tile_shape=[32, 32],
        dst_tiles=8,
        dst_sync="full" if config.dst_full_sync_en else "half",
        state_tiles=state.total_size // state.format_descriptors[0].page_size,
        state_resident=True,
        compute_kernel_config=dict(
            math_fidelity=str(config.math_fidelity),
            fp32_dest_acc_en=bool(config.fp32_dest_acc_en),
            math_approx_mode=bool(config.math_approx_mode),
            dst_full_sync_en=bool(config.dst_full_sync_en),
        ),
        reader_variant="scalar_cache_narrow_rm",
        scalar_cache_cb=scalar.format_descriptors[0].buffer_index,
        scalar_cache_bytes_per_worker=scalar.total_size,
        qk_read_bytes_per_token_per_job=[512, 512],
        v_read_bytes_per_token_per_job=64,
        scalar_read_bytes_per_32_tokens_per_job=8192,
        initial_state_reads_per_job=1,
        final_state_writes_per_job=1,
        provenance=_PROVENANCE,
    )
    self.prefill_resident_signatures[str(length)] = signature


def _resident_recurrence(self, q, k, v, g, beta):
    """Exact token recurrence; prefill keeps FP32 state in aliased local buffers."""
    batch, length = q.shape[0], q.shape[1]
    if not (self.recurrence_broadcast and batch == 1 and length in (128, 256, 512)):
        raise ValueError("ROW_MAJOR Q/K requires the qualified serial prefill geometry")
    signature = dict(sequence=length)
    values = []
    for name, value in (("q", q), ("k", k)):
        converted = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT, memory_config=value.memory_config())
        if value.dtype != ttnn.float32 or converted.dtype != ttnn.float32:
            raise ValueError("ROW_MAJOR Q/K must preserve normalized FP32 operands")
        signature[name] = dict(
            logical_shape=list(value.shape),
            tile_padded_shape=list(value.padded_shape),
            rm_padded_shape=list(converted.padded_shape),
            tile_dtype=str(value.dtype),
            rm_dtype=str(converted.dtype),
            tile_memory_config=str(value.memory_config()),
            rm_memory_config=str(converted.memory_config()),
            tile_layout=str(value.layout),
            rm_layout=str(converted.layout),
        )
        values.append(converted)
    self.prefill_qk_rm_signatures[str(length)] = signature
    q, k = values
    del value, converted, values
    # Query normalization already applied the scale as its output activation.
    decay = g  # Already exponentiated by the multiply output activation.
    output = ttnn.empty(
        (batch, 12, length, 128),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=self.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    final = self.recurrent_state
    tensors = [q, k, v, decay, beta, self.recurrent_state, output, final]
    grid = self.device.compute_with_storage_grid_size()
    jobs = batch * 96
    workers = min(jobs, grid.x * grid.y)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = []
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    assignments = [(i * jobs // workers, (i + 1) * jobs // workers - i * jobs // workers) for i in range(workers)]
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=True
    )
    config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    reader = READER
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=reader,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            config=ttnn.ReaderConfigDescriptor(),
            runtime_args=[
                ((c.x, c.y), [*[t.buffer_address() for t in tensors], *job, length, 0])
                for c, job in zip(coords, assignments)
            ],
        ),
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            config=config,
            runtime_args=[((c.x, c.y), [job[1], length]) for c, job in zip(coords, assignments)],
        ),
    ]
    sizes = [4, 4, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4, 1, 1]
    if length > 1:
        sizes.append(1)
    if len(sizes) != 16:
        raise ValueError("Scalar-cache CB16 requires qualified prefill geometry")
    sizes.append(2)  # CB16 owns8192 bytes, reserved by reader only.
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * 4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
        )
        for i, count in enumerate(sizes)
        if not (length > 1 and i in (0, 12))
    ]
    if length > 1:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=4 * 4096,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096) for i in (0, 12)
                ],
            )
        )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs))
    _record_signature(self, batch, length, workers, config, cbs)
    return output, final


class PrefillResidentGDN(PrefillQKRowMajorGDN):
    prefill_resident_enabled = False

    def _recurrence(self, q, k, v, g, beta):
        b, s, _ = q.shape
        if not (
            self.prefill_resident_enabled
            and self.prefill_qk_rm_enabled
            and self.recurrence_broadcast
            and b == 1
            and s in (128, 256, 512)
        ):
            return super()._recurrence(q, k, v, g, beta)
        q = self._heads(q, b, s, 4)
        k = self._heads(k, b, s, 4)
        q = ttnn.rms_norm(
            ttnn.typecast(q, ttnn.float32), weight=self.qscale, epsilon=1e-06 / 128, compute_kernel_config=self.compute
        )
        k = ttnn.rms_norm(
            ttnn.typecast(k, ttnn.float32), weight=self.kscale, epsilon=1e-06 / 128, compute_kernel_config=self.compute
        )
        v = ttnn.typecast(self._heads(v, b, s, 12), ttnn.float32)
        return _resident_recurrence(self, q, k, v, g, beta)

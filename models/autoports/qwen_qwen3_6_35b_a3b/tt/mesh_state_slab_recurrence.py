# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact ordered FP32 recurrence with local state slabs on 32 workers.

BF16 consumption rounding is optional; persistent state remains FP32. Native
Q/K layout conversions are preserved by default; the optional native reader
reconstructs their consumed columns directly. Replay-time activity gates only
the persistent state writes; the compute and output protocols are unchanged.
"""

from .mesh_direct_recurrent_sum import DirectRecurrentSum

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
 const auto state=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(0)>(),get_arg_val<uint32_t>(0));
 const auto key=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(1)>(),get_arg_val<uint32_t>(1));
 const auto query=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(2)>(),get_arg_val<uint32_t>(2));
 const auto value=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(3)>(),get_arg_val<uint32_t>(3));
 const auto beta=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(4)>(),get_arg_val<uint32_t>(4));
 const auto decay=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(5)>(),get_arg_val<uint32_t>(5));
 const uint32_t head=get_arg_val<uint32_t>(6),col=get_arg_val<uint32_t>(7);
 DataflowBuffer s(0),k(1),q(2),v(3),b(4),d(5);Noc noc;
 s.reserve_back(4);k.reserve_back(4);q.reserve_back(4);v.reserve_back(1);b.reserve_back(1);d.reserve_back(1);
 ASSERT((get_write_ptr(0)&63)==0);ASSERT((get_write_ptr(1)&63)==0);ASSERT((get_write_ptr(2)&63)==0);
 ASSERT((get_write_ptr(3)&63)==0);ASSERT((get_write_ptr(4)&63)==0);ASSERT((get_write_ptr(5)&63)==0);
 for(uint32_t row=0;row<4;++row) {
  noc.async_read(state,s,4096,{.page_id=head*16+row*4+col},{.offset_bytes=row*4096});
  noc.async_read(key,k,4096,{.page_id=(head/2)*4+row},{.offset_bytes=row*4096});
  noc.async_read(query,q,4096,{.page_id=(head/2)*4+row},{.offset_bytes=row*4096});
 }
 // Whole value tile then two 16-word copies, avoiding narrow DRAM alignment
 // ambiguity for odd head rows. Only row0 is consumed by SFPU row broadcast.
 noc.async_read(value,v,2048,{.page_id=col},{});
 noc.async_read(beta,b,64,{.page_id=0},{});
 noc.async_read(decay,d,64,{.page_id=0},{});
 noc.async_read_barrier();
 auto* vp=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(3));
 for(uint32_t c=0;c<16;++c) {vp[c]=vp[head*16+c];vp[256+c]=vp[256+head*16+c];}
 auto* bp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4));
 auto* dp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(5));
 bp[0]=bp[head];dp[0]=dp[head];
 s.push_back(4);k.push_back(4);q.push_back(4);v.push_back(1);b.push_back(1);d.push_back(1);
}
"""

# Default-off native-row variant: only Q/K input transport changes. Every
# consumed column0 word is reconstructed before the existing compute starts.
NATIVE_QK_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 const auto state=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(0)>(),get_arg_val<uint32_t>(0));
 const auto key=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(1)>(),get_arg_val<uint32_t>(1));
 const auto query=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(2)>(),get_arg_val<uint32_t>(2));
 const auto value=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(3)>(),get_arg_val<uint32_t>(3));
 const auto beta=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(4)>(),get_arg_val<uint32_t>(4));
 const auto decay=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(5)>(),get_arg_val<uint32_t>(5));
 const uint32_t head=get_arg_val<uint32_t>(6),col=get_arg_val<uint32_t>(7);
 DataflowBuffer s(0),k(1),q(2),v(3),b(4),d(5),scratch(28);Noc noc;
 s.reserve_back(4);k.reserve_back(4);q.reserve_back(4);v.reserve_back(1);b.reserve_back(1);d.reserve_back(1);
 scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(28)&63))&63;
 ASSERT((get_write_ptr(0)&63)==0);ASSERT((get_write_ptr(1)&63)==0);ASSERT((get_write_ptr(2)&63)==0);
 ASSERT((get_write_ptr(3)&63)==0);ASSERT((get_write_ptr(4)&63)==0);ASSERT((get_write_ptr(5)&63)==0);
 for(uint32_t row=0;row<4;++row) {
  noc.async_read(state,s,4096,{.page_id=head*16+row*4+col},{.offset_bytes=row*4096});
  const uint32_t source_row=(head/2)*64;
  noc.async_read(key,scratch,64,{.page_id=row,.offset_bytes=source_row},{.offset_bytes=bias+row*256});
  noc.async_read(key,scratch,64,{.page_id=row,.offset_bytes=1024+source_row},{.offset_bytes=bias+row*256+64});
  noc.async_read(query,scratch,64,{.page_id=row,.offset_bytes=source_row},{.offset_bytes=bias+row*256+128});
  noc.async_read(query,scratch,64,{.page_id=row,.offset_bytes=1024+source_row},{.offset_bytes=bias+row*256+192});
 }
 noc.async_read(value,v,2048,{.page_id=col},{});
 noc.async_read(beta,b,64,{.page_id=0},{});
 noc.async_read(decay,d,64,{.page_id=0},{});
 noc.async_read_barrier();
 const auto* source=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(28)+bias);
 auto* kp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
 auto* qp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
 // All padding is deterministic; only column0 participates in the existing
 // broadcast operations. No arithmetic, narrowing or host conversion here.
 noc.async_write_zeros(k,4*4096);
 noc.async_write_zeros(q,4*4096);
 noc.write_zeros_l1_barrier();
 for(uint32_t tile=0;tile<4;++tile) {
  for(uint32_t row=0;row<32;++row) {
   const uint32_t dst=tile*1024+(row/16)*512+(row%16)*16;
   const uint32_t src=tile*64+row;
   kp[dst]=source[src];qp[dst]=source[src+32];
  }
 }
 auto* vp=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(3));
 for(uint32_t c=0;c<16;++c) {vp[c]=vp[head*16+c];vp[256+c]=vp[256+head*16+c];}
 auto* bp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4));
 auto* dp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(5));
 bp[0]=bp[head];dp[0]=dp[head];
 s.push_back(4);k.push_back(4);q.push_back(4);v.push_back(1);b.push_back(1);d.push_back(1);
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
 const auto state=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(0)>(),get_arg_val<uint32_t>(0));
 const auto output=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(1)>(),get_arg_val<uint32_t>(1));
 const uint32_t head=get_arg_val<uint32_t>(2),col=get_arg_val<uint32_t>(3);
 DataflowBuffer ns(17),out(16);Noc noc;
 bool commit=true;
#ifdef CONDITIONAL_STATE_COMMIT
 const auto activity=TensorAccessor(TensorAccessorArgs<get_compile_time_arg_val(2)>(),get_arg_val<uint32_t>(4));
 // Writer-private aligned scratch; no compute or reader CB protocol.
 DataflowBuffer predicate(27);predicate.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(27)&63))&63;
 noc.async_read(activity,predicate,4,{.page_id=0},{.offset_bytes=bias});
 noc.async_read_barrier();
 commit=*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(27)+bias)!=0;
#endif
 ns.wait_front(4);
 if(commit) {
  for(uint32_t row=0;row<4;++row)
   noc.async_write(ns,state,4096,{.offset_bytes=row*4096},{.page_id=head*16+row*4+col});
 }
 // Always drain state CBs, even when the persistent state must stay untouched.
 noc.async_write_barrier();ns.pop_front(4);
 out.wait_front(1);
 noc.async_write(out,output,64,{.offset_bytes=0},{.page_id=col,.offset_bytes=head*64});
 noc.async_write(out,output,64,{.offset_bytes=1024},{.page_id=col,.offset_bytes=1024+head*64});
 noc.async_write_barrier();out.pop_front(1);
}
"""

COMPUTE = r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/dataflow_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu.h"
#include "ckernel_sfpu_cumsum.h"
// The SFPU transpose produces four successive rows in LREG0..3. A
// second transpose preserves the running carry in the same representation
// used by the selected compensated scan. Unlike cumsum, each group starts
// with its first operand, with no initial addition of zero.
template<uint32_t GROUP,uint32_t PASS> inline void rows() {
 constexpr uint32_t tile=GROUP/2,half=GROUP%2;
 constexpr uint32_t row=tile*64+half*32+PASS*4;
 TTI_SFPLOAD(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,row);
 TTI_SFPLOAD(1,InstrModLoadStore::DEFAULT,ADDR_MOD_7,row+2);
 TTI_SFPLOAD(2,InstrModLoadStore::DEFAULT,ADDR_MOD_7,row+16);
 TTI_SFPLOAD(3,InstrModLoadStore::DEFAULT,ADDR_MOD_7,row+18);
 TTI_SFPTRANSP(0,0,0,0);
 if constexpr(PASS!=0) {TTI_SFPADD(10,7,0,0,0);TTI_SFPNOP;}
 TTI_SFPADD(10,0,1,1,0);TTI_SFPNOP;
 TTI_SFPADD(10,1,2,2,0);TTI_SFPNOP;
 TTI_SFPADD(10,2,3,3,0);TTI_SFPNOP;
 if constexpr(PASS==3) {
  if constexpr(GROUP!=0) {
   TTI_SFPLOAD(4,InstrModLoadStore::DEFAULT,ADDR_MOD_7,256);
   TTI_SFPADD(10,4,3,3,0);TTI_SFPNOP;
  }
  // Raw transposed scratch, not a packed tensor boundary.
  TTI_SFPSTORE(3,InstrModLoadStore::DEFAULT,ADDR_MOD_7,256);
 }
 TTI_SFPMOV(0,3,7,0);
 TTI_SFPTRANSP(0,0,0,0);
}
template<uint32_t GROUP> inline void group() {
 rows<GROUP,0>();rows<GROUP,1>();rows<GROUP,2>();rows<GROUP,3>();
}
inline void exact_sum() {
 sfpu::_init_sfpu_config_reg();
 // Explicit SFPU addressing must not inherit datacopy address modifiers.
 addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_7);
 math::reset_counters(p_setrwc::SET_ABD_F);
 _llk_math_eltwise_sfpu_start_(0);
 group<0>();group<1>();group<2>();group<3>();
 group<4>();group<5>();group<6>();group<7>();
 TTI_SFPLOAD(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,256);
 TTI_SFPMOV(0,0,1,0);TTI_SFPMOV(0,0,2,0);TTI_SFPMOV(0,0,3,0);
 TTI_SFPTRANSP(0,0,0,0);
 TTI_SFPSTORE(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,256);
 TTI_SFPSTORE(1,InstrModLoadStore::DEFAULT,ADDR_MOD_7,258);
 TTI_SFPSTORE(2,InstrModLoadStore::DEFAULT,ADDR_MOD_7,272);
 TTI_SFPSTORE(3,InstrModLoadStore::DEFAULT,ADDR_MOD_7,274);
 _llk_math_eltwise_sfpu_done_();
}
#endif

#ifdef TRISC_MATH
inline void slab_sfpu_begin() {
 sfpu::_init_sfpu_config_reg();
 addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_7);
 math::reset_counters(p_setrwc::SET_ABD_F);_llk_math_eltwise_sfpu_start_(0);
}
// Bitwise horizontal fan-out: preserve negative zero/subnormal/payload bits;
// no multiply-by-one or floating add-to-zero is used to construct an operand.
inline void spread_bits() {
 TTI_SFPSHFT2(0,0,2,3);TTI_SFPNOP;TTI_SFPOR(0,2,0,0);
 TTI_SFPSHFT2(0,0,2,3);TTI_SFPNOP;TTI_SFPSHFT2(0,2,2,3);TTI_SFPNOP;TTI_SFPOR(0,2,0,0);
 TTI_SFPSHFT2(0,0,2,3);TTI_SFPNOP;TTI_SFPSHFT2(0,2,2,3);TTI_SFPNOP;
 TTI_SFPSHFT2(0,2,2,3);TTI_SFPNOP;TTI_SFPSHFT2(0,2,2,3);TTI_SFPNOP;TTI_SFPOR(0,2,0,0);
}
inline void broadcast_column_bits(uint32_t src,uint32_t dst) {
 slab_sfpu_begin();sfpu::_build_lane_mask_col0_();
 for(uint32_t half=0;half<2;++half)for(uint32_t band=0;band<4;++band) {
  const uint32_t row=half*32+band*4;
  TT_SFPLOAD(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,src*64+row);
  TTI_SFPAND(0,6,0,0);spread_bits();
  TT_SFPSTORE(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,dst*64+row);
  TT_SFPSTORE(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,dst*64+row+2);
  TT_SFPSTORE(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,dst*64+row+16);
  TT_SFPSTORE(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,dst*64+row+18);
 }
 _llk_math_eltwise_sfpu_done_();
}
inline void multiply_scalar(uint32_t tile,uint32_t scalar) {
 slab_sfpu_begin();sfpu::_build_lane_mask_col0_();
 TT_SFPLOAD(0,InstrModLoadStore::DEFAULT,ADDR_MOD_7,scalar*64);
 TTI_SFPAND(0,6,0,0);
 TTI_SFPCONFIG(0,p_sfpu::LREG14,0);TTI_SFPMOV(0,p_sfpu::LREG14,0,0);
 spread_bits();
 for(uint32_t lane=0;lane<32;++lane) {
  TT_SFPLOAD(1,InstrModLoadStore::DEFAULT,ADDR_MOD_7,tile*64+lane*2);
  TTI_SFPMUL(1,0,p_sfpu::LCONST_0,1,0);TTI_SFPNOP;
  TT_SFPSTORE(1,InstrModLoadStore::DEFAULT,ADDR_MOD_7,tile*64+lane*2);
 }
 _llk_math_eltwise_sfpu_done_();
}
#endif
void kernel_main() {
 constexpr bool rounded=get_compile_time_arg_val(0)!=0;
 compute_kernel_hw_startup(0,16);
 DataflowBuffer s(0),k(1),q(2),v(3),beta(4),decay(5),local_d(24),delta(25),local_new(26),ns(17),out(16);
 s.wait_front(4);k.wait_front(4);q.wait_front(4);v.wait_front(1);beta.wait_front(1);decay.wait_front(1);
 // Phase1: exact BF16-on-consumption then decay. Save D only to local CB.
 tile_regs_acquire();copy_init(5);copy_tile(5,0,6);
 for(uint32_t i=0;i<4;++i) {
  copy_init(0);copy_tile(0,i,i);
  if constexpr(rounded) {
   typecast_tile_init<(uint32_t)DataFormat::Float32,(uint32_t)DataFormat::Float16_b>();
   typecast_tile<(uint32_t)DataFormat::Float32,(uint32_t)DataFormat::Float16_b,true>(i);
  }
#ifdef TRISC_MATH
  multiply_scalar(i,6);
#endif
 }
 tile_regs_commit();tile_regs_wait();local_d.reserve_back(4);
 for(uint32_t i=0;i<4;++i)pack_tile(i,24);
 local_d.push_back(4);tile_regs_release();s.pop_front(4);decay.pop_front(1);
 // Phase2: exact D*k contraction and (v-memory)*beta. Save delta locally.
 local_d.wait_front(4);tile_regs_acquire();
 for(uint32_t i=0;i<4;++i) {
  copy_init(24);copy_tile(24,i,i);copy_init(1);copy_tile(1,i,4);
  sfpu_mul_bcast_col_init();sfpu_mul_bcast_col<true>(i,4);
 }
#ifdef TRISC_MATH
 exact_sum();
#endif
 reconfig_data_format_srca(3);copy_init(3);copy_tile(3,0,5);
 sub_binary_tile_init();sub_binary_tile(5,4,4);
 reconfig_data_format_srca(4);copy_init(4);copy_tile(4,0,6);
#ifdef TRISC_MATH
 multiply_scalar(4,6);
#endif
 tile_regs_commit();tile_regs_wait();delta.reserve_back(1);pack_tile(4,25);delta.push_back(1);
 tile_regs_release();v.pop_front(1);beta.pop_front(1);
 // Phase3: exact k*delta then D+update. Save live FP32 state, never BF16.
 delta.wait_front(1);tile_regs_acquire();copy_init(25);copy_tile(25,0,4);
 for(uint32_t i=0;i<4;++i) {
  copy_init(24);copy_tile(24,i,i);copy_init(1);copy_tile(1,i,5);
#ifdef TRISC_MATH
  broadcast_column_bits(5,6);
#endif
  sfpu_mul_bcast_row_init();sfpu_mul_bcast_row<true>(6,4);
  add_binary_tile_init();add_binary_tile(i,6,i);
 }
 tile_regs_commit();tile_regs_wait();ns.reserve_back(4);local_new.reserve_back(4);
 for(uint32_t i=0;i<4;++i) {pack_tile(i,17);pack_tile(i,26);}
 ns.push_back(4);local_new.push_back(4);tile_regs_release();local_d.pop_front(4);delta.pop_front(1);k.pop_front(4);
 // Phase4 consumes the local newly-written state; global writer can commit
 // independently. Dedicated local_new has one producer and one consumer.
 local_new.wait_front(4);tile_regs_acquire();
 for(uint32_t i=0;i<4;++i) {
  copy_init(26);copy_tile(26,i,i);copy_init(2);copy_tile(2,i,4);
  binop_with_scalar_tile_init();mul_unary_tile<true>(4,0x3db504f3);
  sfpu_mul_bcast_col_init();sfpu_mul_bcast_col<true>(i,4);
 }
#ifdef TRISC_MATH
 exact_sum();
#endif
 tile_regs_commit();tile_regs_wait();out.reserve_back(1);pack_tile(4,16);out.push_back(1);
 tile_regs_release();local_new.pop_front(4);q.pop_front(4);
}
"""


class StateSlabRecurrence(DirectRecurrentSum):
    def __init__(self, t, mesh):
        super().__init__(t, mesh)
        self.execution_geometry = {
            "workers": 32,
            "slab": [128, 32],
            "heads_per_rank": 8,
            "qk_heads_per_rank": 4,
            "state_dtype": "float32",
            "state_consumption": "native BF16 round/widen only when round_state=True",
            "state_commit": "FP32, persistent DRAM or temporary L1",
            "contraction": "ordered 16 then 8 FP32 additions, starts first operand",
            "products": "separate FP32 SFPU multiply; no FMA or TF32",
            "q_scale": "native MUL_UNARY_SFPU 128**-0.5",
            "scalar_fanout": "bitwise shift/OR",
            "qk_layout": "native rows when native_qk_reader=True; original head-major permutation otherwise",
            "output": "FP32 L1 terminal head rows",
            "inactive_commit": "optional persistent activity gates writer stores; otherwise caller-owned masked commit",
            "fallback": "selected product/direct-sum recurrence when unsupported",
        }

    def _supports_activity(self, activity, output_state):
        t = self.t
        return activity is None or (
            output_state is not None
            and output_state.memory_config() == t.DRAM_MEMORY_CONFIG
            and tuple(activity.shape) == (1,)
            and tuple(activity.padded_shape) == (1,)
            and activity.dtype == t.int32
            and activity.layout == t.ROW_MAJOR_LAYOUT
            and activity.memory_config() == t.DRAM_MEMORY_CONFIG
        )

    def supports_native(self, q, k, v, beta, state, decay, *, output_state=None, activity=None):
        t = self.t

        def valid(value, shape, dtype):
            if value is None:
                return False
            padded = tuple(shape[:-2]) + tuple((n + 31) // 32 * 32 for n in shape[-2:])
            return (
                tuple(value.shape) == shape
                and tuple(value.padded_shape) == padded
                and value.dtype == dtype
                and value.layout == t.TILE_LAYOUT
                and value.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
            )

        return (
            valid(q, (1, 1, 4, 128), t.float32)
            and valid(k, (1, 1, 4, 128), t.float32)
            and valid(v, (1, 1, 8, 128), t.bfloat16)
            and valid(beta, (1, 1, 8), t.float32)
            and valid(decay, (1, 1, 8), t.float32)
            and valid(state, (1, 8, 128, 128), t.float32)
            and (output_state is None or valid(output_state, (1, 8, 128, 128), t.float32))
            and self._supports_activity(activity, output_state)
        )

    def from_native(
        self, q, k, v, beta, state, decay, *, round_state, output_state=None, activity=None, native_qk_reader=False
    ):
        assert self.supports_native(q, k, v, beta, state, decay, output_state=output_state, activity=activity)
        if type(native_qk_reader) is not bool:
            raise TypeError("native_qk_reader must be a construction-time boolean")
        if native_qk_reader:
            return self(
                state,
                k,
                q,
                v,
                beta,
                decay,
                rounded=round_state,
                output_state=output_state,
                activity=activity,
                native_qk_reader=True,
            )
        t = self.t
        q = t.reshape(t.permute(q, (0, 2, 3, 1)), (1, 4, 1, 128, 1))
        k = t.reshape(t.permute(k, (0, 2, 3, 1)), (1, 4, 1, 128, 1))
        return self(state, k, q, v, beta, decay, rounded=round_state, output_state=output_state, activity=activity)

    def __call__(
        self, state, k, q, v, beta, decay, *, rounded=False, output_state=None, activity=None, native_qk_reader=False
    ):
        t = self.t
        assert self._supports_activity(activity, output_state)
        inputs = [state, k, q, v, beta, decay]
        assert tuple(state.shape) == (1, 8, 128, 128)
        if native_qk_reader:
            assert self.supports_native(q, k, v, beta, state, decay, output_state=output_state, activity=activity)
        else:
            assert tuple(k.shape) == tuple(q.shape) == (1, 4, 1, 128, 1)
        assert tuple(v.shape) == (1, 1, 8, 128)
        assert tuple(beta.shape) == tuple(decay.shape) == (1, 1, 8)
        for i, value in enumerate(inputs):
            assert value.dtype == (t.bfloat16 if i == 3 else t.float32)
            assert value.layout == t.TILE_LAYOUT
            assert value.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
        if output_state is None:
            output_state = t.empty(
                (1, 8, 128, 128),
                device=self.mesh,
                dtype=t.float32,
                layout=t.TILE_LAYOUT,
                memory_config=t.L1_MEMORY_CONFIG,
            )
        assert tuple(output_state.shape) == (1, 8, 128, 128) and output_state.dtype == t.float32
        assert output_state.layout == t.TILE_LAYOUT and output_state.memory_config() in (
            t.L1_MEMORY_CONFIG,
            t.DRAM_MEMORY_CONFIG,
        )
        output = t.empty(
            (1, 1, 8, 128), device=self.mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=t.L1_MEMORY_CONFIG
        )
        rd, wr = t.RuntimeArgs(), t.RuntimeArgs()
        for i, (x, y) in enumerate(self.coords):
            rd[x][y] = [z.buffer_address() for z in inputs] + [i // 4, i % 4]
            wr[x][y] = [output_state.buffer_address(), output.buffer_address(), i // 4, i % 4] + (
                [] if activity is None else [activity.buffer_address()]
            )
        accessors = [list(t.TensorAccessorArgs(z).get_compile_time_args()) for z in inputs]
        offset = len(inputs)
        offsets = []
        for args in accessors:
            offsets.append(offset)
            offset += len(args)
        reader_args = offsets + [x for args in accessors for x in args]
        sa = list(t.TensorAccessorArgs(output_state).get_compile_time_args())
        oa = list(t.TensorAccessorArgs(output).get_compile_time_args())
        config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 2, 3, 4, 5, 24, 25, 26):
            modes[i] = t.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes

        def kernel(source, cfg, args, runtime=None, defines=None):
            kw = dict(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=self.cores,
                config=cfg,
                compile_time_args=args,
            )
            if runtime is not None:
                kw["runtime_args"] = runtime
            if defines is not None:
                kw["defines"] = defines
            return t.KernelDescriptor(**kw)

        writer_args = [2, 2 + len(sa), *sa, *oa]
        writer_defines = None
        activity_cb = []
        activity_inputs = []
        if activity is not None:
            aa = list(t.TensorAccessorArgs(activity).get_compile_time_args())
            writer_args = [3, 3 + len(sa), 3 + len(sa) + len(oa), *sa, *oa, *aa]
            writer_defines = [("CONDITIONAL_STATE_COMMIT", "1")]
            activity_cb = [(27, 1, 128, t.uint32)]
            activity_inputs = [activity]

        program = t.ProgramDescriptor(
            kernels=[
                kernel(NATIVE_QK_READER if native_qk_reader else READER, t.ReaderConfigDescriptor(), reader_args, rd),
                kernel(COMPUTE, config, [int(rounded)]),
                kernel(WRITER, t.WriterConfigDescriptor(), writer_args, wr, writer_defines),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=count * page,
                    core_ranges=self.cores,
                    format_descriptors=[t.CBFormatDescriptor(i, dtype, page)],
                )
                for i, count, page, dtype in (
                    (0, 4, 4096, t.float32),
                    (1, 4, 4096, t.float32),
                    (2, 4, 4096, t.float32),
                    (3, 1, 2048, t.bfloat16),
                    (4, 1, 4096, t.float32),
                    (5, 1, 4096, t.float32),
                    (24, 4, 4096, t.float32),
                    (25, 1, 4096, t.float32),
                    (26, 4, 4096, t.float32),
                    (17, 4, 4096, t.float32),
                    (16, 1, 4096, t.float32),
                )
                + tuple(activity_cb)
                + (((28, 1, 2048, t.uint32),) if native_qk_reader else ())
            ],
        )
        t.generic_op([*inputs, *activity_inputs, output_state, output], program)
        return output, output_state

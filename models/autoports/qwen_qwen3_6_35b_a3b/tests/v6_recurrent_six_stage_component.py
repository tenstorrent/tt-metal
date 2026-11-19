# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Two exact-order recurrent producers around unchanged native reductions.

Experimental isolated component; no host tensor operations or persistent buffers.
"""

PHASE1_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
constexpr uint32_t face(uint32_t r,uint32_t c){return (r/16)*512+(c/16)*256+(r%16)*16+c%16;}
void kernel_main(){
 constexpr uint32_t ka=get_compile_time_arg_val(0),da=get_compile_time_arg_val(1);
 const auto state=TensorAccessor(TensorAccessorArgs<2>(),get_arg_val<uint32_t>(0));
 const auto key=TensorAccessor(TensorAccessorArgs<ka>(),get_arg_val<uint32_t>(1));
 const auto decay=TensorAccessor(TensorAccessorArgs<da>(),get_arg_val<uint32_t>(2));
 const uint32_t group=get_arg_val<uint32_t>(3),vb=get_arg_val<uint32_t>(4);
 DataflowBuffer inputs(0),keys(1),decays(2),scratch(3);Noc noc;scratch.reserve_back(2);
 const uint32_t bias=(64-(get_write_ptr(3)&63))&63;
 auto* raw=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3)+bias);
 for(uint32_t h=0;h<16;++h)
  noc.async_read(key,scratch,64,{.page_id=group/2,.offset_bytes=(group%2)*1024+h*64},{.offset_bytes=bias+4096+h*64});
 noc.async_read(decay,scratch,64,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias+5120});
 noc.async_read(decay,scratch,64,{.page_id=0,.offset_bytes=1024},{.offset_bytes=bias+5184});
 noc.async_read_barrier();
 decays.reserve_back(1);auto* dt=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
 for(uint32_t h=0;h<32;++h)for(uint32_t c=0;c<32;++c)dt[face(h,c)]=raw[1280+h];
 decays.push_back(1);
 for(uint32_t i=0;i<16;++i){
  const uint32_t k=group*16+i;
  for(uint32_t h=0;h<32;++h){
   const uint32_t page=h*16+(k/32)*4+vb,offset=(k%32/16)*2048+(k%16)*64;
   noc.async_read(state,scratch,64,{.page_id=page,.offset_bytes=offset},{.offset_bytes=bias+h*128});
   noc.async_read(state,scratch,64,{.page_id=page,.offset_bytes=offset+1024},{.offset_bytes=bias+h*128+64});
  }
  noc.async_read_barrier();inputs.reserve_back(1);keys.reserve_back(1);
  auto* st=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
  auto* kt=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
  for(uint32_t h=0;h<32;++h)for(uint32_t c=0;c<32;++c){
   st[face(h,c)]=raw[h*32+c];kt[face(h,c)]=raw[1024+(h/2)*16+i];
  }
  inputs.push_back(1);keys.push_back(1);
 }
}
"""

PHASE1_COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main(){
 constexpr bool round_state=get_compile_time_arg_val(0)!=0,debug=get_compile_time_arg_val(1)!=0;
 CircularBuffer input(0),key(1),decay(2),feedback(24),decayed(16),product(17),rounded(18);
 compute_kernel_hw_startup(0,24);copy_init(0);decay.wait_front(1);
 for(uint32_t i=0;i<16;++i){
  input.wait_front(1);key.wait_front(1);
  if constexpr(round_state || debug){
   tile_regs_acquire_math_clear();copy_tile(0,0,0);
   if constexpr(round_state){
    typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
    typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b),true>(0);
    typecast_tile<static_cast<uint32_t>(DataFormat::Float16_b),static_cast<uint32_t>(DataFormat::Float32),true>(0);
   }
   tile_regs_commit();tile_regs_wait();
   if constexpr(round_state){feedback.reserve_back(1);pack_tile(0,24);feedback.push_back(1);}
   if constexpr(debug){rounded.reserve_back(1);pack_tile(0,18);rounded.push_back(1);}
   tile_regs_release_math_clear();
  }
  tile_regs_acquire_math_clear();
  if constexpr(round_state){feedback.wait_front(1);copy_tile(24,0,0);feedback.pop_front(1);}
  else copy_tile(0,0,0);
  copy_tile(2,0,1);mul_binary_tile_init();mul_binary_tile<true>(0,1,0);
  input.pop_front(1);tile_regs_commit();tile_regs_wait();
  feedback.reserve_back(1);pack_tile(0,24);feedback.push_back(1);
  decayed.reserve_back(1);pack_tile(0,16);decayed.push_back(1);tile_regs_release_math_clear();
  tile_regs_acquire_math_clear();feedback.wait_front(1);copy_tile(24,0,0);copy_tile(1,0,1);
  feedback.pop_front(1);key.pop_front(1);mul_binary_tile_init();mul_binary_tile<true>(0,1,0);
  tile_regs_commit();tile_regs_wait();product.reserve_back(1);pack_tile(0,17);product.push_back(1);tile_regs_release_math_clear();
 }
 decay.pop_front(1);
}
"""

PHASE1_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main(){
 constexpr uint32_t pa=get_compile_time_arg_val(0),ra=get_compile_time_arg_val(1);
 constexpr bool debug=get_compile_time_arg_val(2)!=0;
 const auto decayed=TensorAccessor(TensorAccessorArgs<3>(),get_arg_val<uint32_t>(0));
 const auto product=TensorAccessor(TensorAccessorArgs<pa>(),get_arg_val<uint32_t>(1));
 const auto rounded=TensorAccessor(TensorAccessorArgs<ra>(),get_arg_val<uint32_t>(2));
 const uint32_t group=get_arg_val<uint32_t>(3),vb=get_arg_val<uint32_t>(4);
 DataflowBuffer d(16),p(17),r(18);Noc noc;
 for(uint32_t i=0;i<16;++i){
  const uint32_t page=(group*16+i)*4+vb;
  if constexpr(debug){r.wait_front(1);noc.async_write(r,rounded,4096,{}, {.page_id=page});noc.async_write_barrier();r.pop_front(1);}
  d.wait_front(1);noc.async_write(d,decayed,4096,{}, {.page_id=page});noc.async_write_barrier();d.pop_front(1);
  p.wait_front(1);noc.async_write(p,product,4096,{}, {.page_id=page});noc.async_write_barrier();p.pop_front(1);
 }
}
"""

PHASE2_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
constexpr uint32_t face(uint32_t r,uint32_t c){return (r/16)*512+(c/16)*256+(r%16)*16+c%16;}
void kernel_main(){
 constexpr uint32_t ma=get_compile_time_arg_val(0),va=get_compile_time_arg_val(1),ka=get_compile_time_arg_val(2);
 constexpr uint32_t qa=get_compile_time_arg_val(3),ba=get_compile_time_arg_val(4);
 const auto decayed=TensorAccessor(TensorAccessorArgs<5>(),get_arg_val<uint32_t>(0));
 const auto memory=TensorAccessor(TensorAccessorArgs<ma>(),get_arg_val<uint32_t>(1));
 const auto value=TensorAccessor(TensorAccessorArgs<va>(),get_arg_val<uint32_t>(2));
 const auto key=TensorAccessor(TensorAccessorArgs<ka>(),get_arg_val<uint32_t>(3));
 const auto query=TensorAccessor(TensorAccessorArgs<qa>(),get_arg_val<uint32_t>(4));
 const auto beta=TensorAccessor(TensorAccessorArgs<ba>(),get_arg_val<uint32_t>(5));
 const uint32_t group=get_arg_val<uint32_t>(6),vb=get_arg_val<uint32_t>(7);
 DataflowBuffer inputs(0),keys(1),queries(2),scratch(3),memories(4),values(5),betas(6);Noc noc;
 scratch.reserve_back(1);const uint32_t bias=(64-(get_write_ptr(3)&63))&63;
 auto* raw=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3)+bias);
 for(uint32_t h=0;h<16;++h){
  noc.async_read(key,scratch,64,{.page_id=group/2,.offset_bytes=(group%2)*1024+h*64},{.offset_bytes=bias+h*64});
  noc.async_read(query,scratch,64,{.page_id=group/2,.offset_bytes=(group%2)*1024+h*64},{.offset_bytes=bias+1024+h*64});
 }
 noc.async_read(beta,scratch,64,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias+2048});
 noc.async_read(beta,scratch,64,{.page_id=0,.offset_bytes=1024},{.offset_bytes=bias+2112});
 memories.reserve_back(1);values.reserve_back(1);
 noc.async_read(memory,memories,4096,{.page_id=vb},{});
 noc.async_read(value,values,2048,{.page_id=vb},{});noc.async_read_barrier();
 memories.push_back(1);values.push_back(1);betas.reserve_back(1);
 auto* bt=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(6));
 for(uint32_t h=0;h<32;++h)for(uint32_t c=0;c<32;++c)bt[face(h,c)]=raw[512+h];betas.push_back(1);
 for(uint32_t i=0;i<16;++i){
  inputs.reserve_back(1);keys.reserve_back(1);queries.reserve_back(1);
  noc.async_read(decayed,inputs,4096,{.page_id=(group*16+i)*4+vb},{});
  auto* kt=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
  auto* qt=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
  for(uint32_t h=0;h<32;++h)for(uint32_t c=0;c<32;++c){
   kt[face(h,c)]=raw[(h/2)*16+i];qt[face(h,c)]=raw[256+(h/2)*16+i];
  }
  noc.async_read_barrier();inputs.push_back(1);keys.push_back(1);queries.push_back(1);
 }
}
"""

PHASE2_COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main(){
 constexpr bool debug=get_compile_time_arg_val(0)!=0;
 constexpr uint32_t scale=get_compile_time_arg_val(1);
 CircularBuffer input(0),key(1),query(2),memory(4),value(5),beta(6),feedback(24),delta(25),scaled(26);
 CircularBuffer state(16),product(17),sub_debug(18),delta_debug(19),update_debug(20),scale_debug(21),wide_debug(22);
 compute_kernel_hw_startup(0,24);copy_init(0);
 value.wait_front(1);memory.wait_front(1);beta.wait_front(1);
 tile_regs_acquire_math_clear();reconfig_data_format_srca(0,5);copy_init(5);copy_tile(5,0,0);
 typecast_tile<static_cast<uint32_t>(DataFormat::Float16_b),static_cast<uint32_t>(DataFormat::Float32),true>(0);
 reconfig_data_format_srca(5,0);copy_init(0);value.pop_front(1);
 tile_regs_commit();tile_regs_wait();feedback.reserve_back(1);pack_tile(0,24);feedback.push_back(1);
 if constexpr(debug){wide_debug.reserve_back(1);pack_tile(0,22);wide_debug.push_back(1);}tile_regs_release_math_clear();
 tile_regs_acquire_math_clear();feedback.wait_front(1);copy_tile(24,0,0);copy_tile(4,0,1);feedback.pop_front(1);memory.pop_front(1);
 sub_binary_tile_init();sub_binary_tile(0,1,0);tile_regs_commit();tile_regs_wait();
 feedback.reserve_back(1);pack_tile(0,24);feedback.push_back(1);
 if constexpr(debug){sub_debug.reserve_back(1);pack_tile(0,18);sub_debug.push_back(1);}tile_regs_release_math_clear();
 tile_regs_acquire_math_clear();feedback.wait_front(1);copy_tile(24,0,0);copy_tile(6,0,1);feedback.pop_front(1);beta.pop_front(1);
 mul_binary_tile_init();mul_binary_tile<true>(0,1,0);tile_regs_commit();tile_regs_wait();
 delta.reserve_back(1);pack_tile(0,25);delta.push_back(1);
 if constexpr(debug){delta_debug.reserve_back(1);pack_tile(0,19);delta_debug.push_back(1);}tile_regs_release_math_clear();delta.wait_front(1);
 for(uint32_t i=0;i<16;++i){
  input.wait_front(1);key.wait_front(1);query.wait_front(1);
  tile_regs_acquire_math_clear();copy_tile(1,0,0);copy_tile(25,0,1);key.pop_front(1);
  mul_binary_tile_init();mul_binary_tile<true>(0,1,0);tile_regs_commit();tile_regs_wait();
  feedback.reserve_back(1);pack_tile(0,24);feedback.push_back(1);
  if constexpr(debug){update_debug.reserve_back(1);pack_tile(0,20);update_debug.push_back(1);}tile_regs_release_math_clear();
  tile_regs_acquire_math_clear();feedback.wait_front(1);copy_tile(0,0,0);copy_tile(24,0,1);input.pop_front(1);feedback.pop_front(1);
  add_binary_tile_init();add_binary_tile(0,1,0);tile_regs_commit();tile_regs_wait();
  feedback.reserve_back(1);pack_tile(0,24);feedback.push_back(1);
  state.reserve_back(1);pack_tile(0,16);state.push_back(1);tile_regs_release_math_clear();
  tile_regs_acquire_math_clear();copy_tile(2,0,0);query.pop_front(1);binop_with_scalar_tile_init();mul_unary_tile(0,scale);
  tile_regs_commit();tile_regs_wait();scaled.reserve_back(1);pack_tile(0,26);scaled.push_back(1);
  if constexpr(debug){scale_debug.reserve_back(1);pack_tile(0,21);scale_debug.push_back(1);}tile_regs_release_math_clear();
  tile_regs_acquire_math_clear();feedback.wait_front(1);scaled.wait_front(1);copy_tile(24,0,0);copy_tile(26,0,1);
  feedback.pop_front(1);scaled.pop_front(1);mul_binary_tile_init();mul_binary_tile<true>(0,1,0);
  tile_regs_commit();tile_regs_wait();product.reserve_back(1);pack_tile(0,17);product.push_back(1);tile_regs_release_math_clear();
 }
 delta.pop_front(1);
}
"""

PHASE2_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main(){
 constexpr uint32_t pa=get_compile_time_arg_val(0),sua=get_compile_time_arg_val(1),da=get_compile_time_arg_val(2);
 constexpr uint32_t ua=get_compile_time_arg_val(3),qa=get_compile_time_arg_val(4),wa=get_compile_time_arg_val(5);
 constexpr bool debug=get_compile_time_arg_val(6)!=0;
 const auto state=TensorAccessor(TensorAccessorArgs<7>(),get_arg_val<uint32_t>(0));
 const auto product=TensorAccessor(TensorAccessorArgs<pa>(),get_arg_val<uint32_t>(1));
 const auto sub=TensorAccessor(TensorAccessorArgs<sua>(),get_arg_val<uint32_t>(2));
 const auto delta=TensorAccessor(TensorAccessorArgs<da>(),get_arg_val<uint32_t>(3));
 const auto update=TensorAccessor(TensorAccessorArgs<ua>(),get_arg_val<uint32_t>(4));
 const auto scaled=TensorAccessor(TensorAccessorArgs<qa>(),get_arg_val<uint32_t>(5));
 const auto wide=TensorAccessor(TensorAccessorArgs<wa>(),get_arg_val<uint32_t>(6));
 const uint32_t group=get_arg_val<uint32_t>(7),vb=get_arg_val<uint32_t>(8);
 DataflowBuffer st(16),pr(17),su(18),de(19),up(20),sc(21),wi(22);Noc noc;
 if constexpr(debug){
  wi.wait_front(1);noc.async_write(wi,wide,4096,{}, {.page_id=group*4+vb});noc.async_write_barrier();wi.pop_front(1);
  su.wait_front(1);noc.async_write(su,sub,4096,{}, {.page_id=group*4+vb});noc.async_write_barrier();su.pop_front(1);
  de.wait_front(1);noc.async_write(de,delta,4096,{}, {.page_id=group*4+vb});noc.async_write_barrier();de.pop_front(1);
 }
 for(uint32_t i=0;i<16;++i){
  const uint32_t k=group*16+i,gpage=k*4+vb;
  if constexpr(debug){up.wait_front(1);noc.async_write(up,update,4096,{}, {.page_id=gpage});noc.async_write_barrier();up.pop_front(1);}
  st.wait_front(1);
  for(uint32_t h=0;h<32;++h){
   const uint32_t page=h*16+(k/32)*4+vb,offset=(k%32/16)*2048+(k%16)*64;
   const uint32_t source=(h/16)*2048+(h%16)*64;
   noc.async_write(st,state,64,{.offset_bytes=source},{.page_id=page,.offset_bytes=offset});
   noc.async_write(st,state,64,{.offset_bytes=source+1024},{.page_id=page,.offset_bytes=offset+1024});
  }
  noc.async_write_barrier();st.pop_front(1);
  if constexpr(debug){sc.wait_front(1);noc.async_write(sc,scaled,4096,{}, {.page_id=gpage});noc.async_write_barrier();sc.pop_front(1);}
  pr.wait_front(1);noc.async_write(pr,product,4096,{}, {.page_id=gpage});noc.async_write_barrier();pr.pop_front(1);
 }
}
"""


class RecurrentSixStage:
    def __init__(self, t, mesh, compute):
        self.t, self.mesh, self.compute = t, mesh, compute

    def _empty(self, shape, memory=None):
        t = self.t
        return t.empty(
            shape, device=self.mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=memory or t.L1_MEMORY_CONFIG
        )

    def _program(self, sources, bindings, runtime_values, compute_args, specs):
        t = self.t
        grid = self.mesh.compute_with_storage_grid_size()
        assert grid.x * grid.y >= 32
        partitions = [(i % grid.x, i // grid.x, i // 4, i % 4) for i in range(32)]
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partitions])
        cfg = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for index, _, _, _ in specs:
            if index not in (3, 16, 17, 18, 19, 20, 21, 22):
                modes[index] = t.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
        kernels = []
        for which, source, config in (
            (0, sources[0], t.ReaderConfigDescriptor()),
            (1, sources[1], cfg),
            (2, sources[2], t.WriterConfigDescriptor()),
        ):
            runtime = t.RuntimeArgs()
            for x, y, group, vb in partitions:
                runtime[x][y] = runtime_values[which](group, vb)
            if which == 1:
                args = compute_args
            else:
                tensors, tail = bindings[which]
                accessors = [list(t.TensorAccessorArgs(tensor).get_compile_time_args()) for tensor in tensors]
                # Offset words for every accessor after the first, followed by
                # optional flags. Every constexpr-false accessor is still valid.
                header = len(tensors) - 1 + len(tail)
                offsets, cursor = [], header
                for accessor in accessors:
                    offsets.append(cursor)
                    cursor += len(accessor)
                args = offsets[1:] + tail + sum(accessors, [])
            kernels.append(
                t.KernelDescriptor(
                    kernel_source=source,
                    source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    config=config,
                    compile_time_args=args,
                    runtime_args=runtime,
                )
            )
        return t.ProgramDescriptor(
            kernels=kernels,
            cbs=[
                t.CBDescriptor(
                    total_size=pages * size,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(index, dtype, size)],
                )
                for index, pages, size, dtype in specs
            ],
        )

    def __call__(self, q, k, v, beta, decay, state, *, output_state=None, round_state=True, debug=False):
        t = self.t
        assert tuple(q.shape) == tuple(k.shape) == (1, 1, 16, 128)
        assert q.dtype == k.dtype == t.float32
        assert tuple(v.shape) == (1, 1, 32, 128) and v.dtype == t.bfloat16
        assert tuple(beta.shape) == tuple(decay.shape) == (1, 1, 32) and beta.dtype == decay.dtype == t.float32
        assert tuple(state.shape) == (1, 32, 128, 128) and state.dtype == t.float32
        assert all(x.layout == t.TILE_LAYOUT for x in (q, k, v, beta, decay, state))
        assert all(
            x.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG) for x in (q, k, v, beta, decay, state)
        )
        output_state = self._empty(tuple(state.shape), state.memory_config()) if output_state is None else output_state
        assert tuple(output_state.shape) == tuple(state.shape) and output_state.dtype == t.float32
        assert output_state.layout == t.TILE_LAYOUT and output_state.memory_config() == state.memory_config()
        grouped = (8, 16, 32, 128)
        decayed, products = self._empty(grouped), self._empty(grouped)
        rounded = self._empty(grouped) if debug else decayed
        phase1_specs = [
            (0, 2, 4096, t.float32),
            (1, 2, 4096, t.float32),
            (2, 1, 4096, t.float32),
            (3, 2, 4096, t.float32),
            (24, 2, 4096, t.float32),
            (16, 1, 4096, t.float32),
            (17, 1, 4096, t.float32),
        ]
        if debug:
            phase1_specs.append((18, 1, 4096, t.float32))
        program = self._program(
            (PHASE1_READER, PHASE1_COMPUTE, PHASE1_WRITER),
            {0: ([state, k, decay], []), 2: ([decayed, products, rounded], [int(debug)])},
            {
                0: lambda group, vb: [state.buffer_address(), k.buffer_address(), decay.buffer_address(), group, vb],
                1: lambda group, vb: [],
                2: lambda group, vb: [
                    decayed.buffer_address(),
                    products.buffer_address(),
                    rounded.buffer_address(),
                    group,
                    vb,
                ],
            },
            [int(round_state), int(debug)],
            phase1_specs,
        )
        t.generic_op([state, k, decay, decayed, products] + ([rounded] if debug else []), program)
        memory = t.experimental.fast_reduce_nc(
            products, dims=[0, 1], compute_kernel_config=self.compute, memory_config=t.L1_MEMORY_CONFIG
        )
        observed = dict(rounded=rounded, decayed=decayed, memory_products=products, memory=memory) if debug else {}
        if not debug:
            del products
        output_products = self._empty(grouped)
        small = (8, 1, 32, 128)
        sub, delta, update, scaled, wide = (
            [self._empty(s) for s in (small, small, grouped, grouped, small)] if debug else [output_products] * 5
        )
        phase2_specs = [
            (0, 2, 4096, t.float32),
            (1, 2, 4096, t.float32),
            (2, 2, 4096, t.float32),
            (3, 1, 4096, t.float32),
            (4, 1, 4096, t.float32),
            (5, 1, 2048, t.bfloat16),
            (6, 1, 4096, t.float32),
            (24, 2, 4096, t.float32),
            (25, 1, 4096, t.float32),
            (26, 1, 4096, t.float32),
            (16, 1, 4096, t.float32),
            (17, 1, 4096, t.float32),
        ]
        if debug:
            phase2_specs.extend((i, 1, 4096, t.float32) for i in (18, 19, 20, 21, 22))
        program = self._program(
            (PHASE2_READER, PHASE2_COMPUTE, PHASE2_WRITER),
            {
                0: ([decayed, memory, v, k, q, beta], []),
                2: ([output_state, output_products, sub, delta, update, scaled, wide], [int(debug)]),
            },
            {
                0: lambda group, vb: [
                    decayed.buffer_address(),
                    memory.buffer_address(),
                    v.buffer_address(),
                    k.buffer_address(),
                    q.buffer_address(),
                    beta.buffer_address(),
                    group,
                    vb,
                ],
                1: lambda group, vb: [],
                2: lambda group, vb: [
                    x.buffer_address() for x in (output_state, output_products, sub, delta, update, scaled, wide)
                ]
                + [group, vb],
            },
            [int(debug), 0x3DB504F3],
            phase2_specs,
        )
        t.generic_op(
            [decayed, memory, v, k, q, beta, output_state, output_products]
            + ([sub, delta, update, scaled, wide] if debug else []),
            program,
        )
        if not debug:
            del decayed, memory
        core = t.experimental.fast_reduce_nc(
            output_products, dims=[0, 1], compute_kernel_config=self.compute, memory_config=t.L1_MEMORY_CONFIG
        )
        if debug:
            observed.update(
                value_wide=wide,
                subtract=sub,
                delta=delta,
                update=update,
                scaled_q=scaled,
                output_products=output_products,
            )
        return core, output_state, observed

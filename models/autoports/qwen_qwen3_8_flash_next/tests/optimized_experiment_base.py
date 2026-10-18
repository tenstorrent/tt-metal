# SPDX-License-Identifier: Apache-2.0
"""Single-device optimized decoder. Inherits the qualified fused state semantics."""
import ttnn
from ..tt.fused_decoder import (
    FusedDecoder, FusedGatedDeltaNet, FusedQSA, FusedPLE,
    FusedMoE, FusedHyperConnection, _PrefixView,
)


class OptimizedDecoder(FusedDecoder):
    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch_size=1,
        max_requests=None,
        chunk_size=128,
        cache_dir=None,
    ):
        import torch  # Setup only.

        obj = cls()
        cfg = getattr(hf_config, "text_config", hf_config)
        if (cfg.hidden_size, cfg.hc_count) != (2560, 4):
            raise ValueError("Expected the target Qwen3.8-Flash-Next configuration")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        prefix = f"model.language_model.layers.{layer_idx}."
        state = _PrefixView(state_dict, prefix) if prefix + "mlp.gate.weight" in state_dict else state_dict
        obj.batch_size, obj.chunk_size = batch_size, chunk_size
        obj.context = cfg.max_position_embeddings
        obj.mesh = mesh_device
        obj.kind = cfg.layer_types[layer_idx]
        obj.attn_hyper = OptimizedHyperConnection.from_state_dict(
            state, prefix="attn_hyper_connection", hf_config=cfg, mesh_device=mesh_device
        )
        obj.mlp_hyper = OptimizedHyperConnection.from_state_dict(
            state, prefix="mlp_hyper_connection", hf_config=cfg, mesh_device=mesh_device
        )
        obj.mlp = OptimizedMoE.from_state_dict(state, hf_config=cfg, mesh_device=mesh_device)
        requests = max_requests or batch_size
        if obj.kind == "linear_attention":
            obj.attention = OptimizedGatedDeltaNet.from_state_dict(
                _PrefixView(state, "linear_attn."),
                hf_config=cfg,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                batch_size=batch_size,
                max_requests=requests,
                chunk_size=chunk_size,
            )
        else:
            obj.attention = OptimizedQSA.from_state_dict(
                _PrefixView(state, "self_attn."),
                hf_config=cfg,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                batch_size=batch_size,
                max_requests=requests,
            )
        obj.ple = None
        if layer_idx + 1 in cfg.ple_layer_ids:
            obj.ple = OptimizedPLE.from_state_dict(
                state, hf_config=cfg, layer_idx=layer_idx, prefix="ple.", mesh_device=mesh_device, cache_dir=cache_dir
            )
            obj.ple_cache = obj.ple.make_cache(max_requests=requests)
        obj.lengths = {}
        obj.prepare([1, chunk_size])
        return obj



class _ProjectionPolicy:
    projection_dtype = ttnn.bfloat16
    projection_fidelity = ttnn.MathFidelity.HiFi4
    output_dtype = None
    output_fidelity = None

    def _prepare_projections(self, names):
        self.projection_compute = ttnn.init_device_compute_kernel_config(
            self.device.arch(), math_fidelity=self.projection_fidelity,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        if self.projection_dtype == ttnn.bfloat16 and self.projection_fidelity == ttnn.MathFidelity.HiFi4:
            self.projection_compute = self.compute
        self.projection_computes = {}
        for name in names:
            output = name in ("out_weight", "wo")
            dtype = self.output_dtype if output and self.output_dtype is not None else self.projection_dtype
            value = getattr(self, name)
            if value.dtype != dtype:
                value = ttnn.typecast(value, dtype)
                setattr(self, name, value)
            compute = self.projection_compute
            if output and self.output_fidelity is not None:
                compute = ttnn.init_device_compute_kernel_config(self.device.arch(), math_fidelity=self.output_fidelity,
                    math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
            self.projection_computes[value.buffer_address()] = compute

    def _linear(self, x, weight):
        return ttnn.linear(x, weight, compute_kernel_config=self.projection_computes.get(weight.buffer_address(),self.projection_compute),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


class OptimizedGatedDeltaNet(_ProjectionPolicy, FusedGatedDeltaNet):
    state_workers = 32

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        obj._prepare_projections(("packed_weight", "out_weight"))
        return obj

    def _route_state(self, slots, positions, *, store):
        """Gather/scatter complete raw tiles using device request IDs.

        Unique slot IDs within each batch are a caller precondition. Position
        zero initializes the selected lane on gather; scatter updates only the
        selected slot. Banks and work buffers are persistent across replay.
        """
        for bank, work, tiles, tile_bytes in (
            (self.recurrent_bank, self.recurrent_state, 48 * 4 * 4, 4096),
            (self.conv_bank, self.conv_state, 320, 2048),
        ):
            grid = self.device.compute_with_storage_grid_size()
            coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(min(grid.x * grid.y, self.batch_size * self.state_workers))]
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
            tensors = [bank, work, slots, positions]
            workers = max(1, len(coords) // self.batch_size)
            coords = coords[:workers * self.batch_size]
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
            args = [tiles, tile_bytes, self.max_requests, int(store), workers]
            for tensor in tensors:
                args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
            cb = ttnn.CBDescriptor(
                total_size=tile_bytes + 256,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=tile_bytes + 256)
                ],
            )
            kernel = ttnn.KernelDescriptor(
                kernel_source=_PARALLEL_STATE_SOURCE,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=args,
                runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i // workers, i % workers]) for i, c in enumerate(coords)],
                config=ttnn.ReaderConfigDescriptor(),
            )
            ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))



class OptimizedQSA(_ProjectionPolicy, FusedQSA):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        obj._prepare_projections(("packed_projection", "wo"))
        return obj


class OptimizedPLE(FusedPLE):
    pass


class OptimizedMoE(FusedMoE):
    down_dtype = ttnn.bfloat8_b
    logical_router = True

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        if cls.down_dtype != cls.dedicated_weight_dtype:
            state = args[0] if args else kwargs["state"]
            raw = state["mlp.experts.down_proj"]
            obj.expert_downs = [ttnn.from_torch(raw[e].T.contiguous(), dtype=cls.down_dtype,
                layout=ttnn.TILE_LAYOUT, device=kwargs["mesh_device"]) for e in range(obj.experts)]
        import torch
        obj.mesh = kwargs['mesh_device']
        obj.compact_indices = ttnn.from_torch(torch.arange(10,dtype=torch.int32).reshape(1,1,1,10),
            dtype=ttnn.uint16,layout=ttnn.ROW_MAJOR_LAYOUT,device=obj.mesh)
        return obj

    def _group(self, x):
        # The unified kernel has independently described gate/up/down formats;
        # the older prefill kernel requires a single shared weight format.
        if self.down_dtype != self.dedicated_weight_dtype:
            return FusedMoE._group_unified(self, x)
        return super()._group(x)

    def _compact(self, down, indices, selected):
        indices = ttnn.to_layout(ttnn.typecast(indices[:, :, :1, :], ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(selected[:, :, :1, :], ttnn.ROW_MAJOR_LAYOUT)
        compact = ttnn.empty((1,10,32,self.hidden), dtype=down.dtype, layout=ttnn.TILE_LAYOUT,
                             device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        coords=[ttnn.CoreCoord(i%10,i//10) for i in range(80)]
        cores=ttnn.CoreRangeSet([ttnn.CoreRange(c,c) for c in coords])
        tensors=[down,indices,scores,compact]
        args=[]
        for t in tensors: args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        kernel=ttnn.KernelDescriptor(kernel_source=_COMPACT_EXPERT_SOURCE,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,compile_time_args=args,
            runtime_args=[((c.x,c.y),[*[t.buffer_address() for t in tensors],i]) for i,c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor())
        cb=ttnn.CBDescriptor(total_size=1280,core_ranges=cores,format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0,data_format=ttnn.uint32,page_size=1280)])
        ttnn.generic_op(tensors,ttnn.ProgramDescriptor(kernels=[kernel],cbs=[cb]))
        compact=ttnn.typecast(compact,ttnn.bfloat16)
        scores=scores[:,:,:,:10]
        return ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(compact,self.compact_indices,self._reduce_mapping,
            reduce_dim=1,split_size=self.hidden,cluster_axis=0,scores_tensor=scores,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,compute_kernel_config=self.compute)[0]

    def _group_unified(self,x):
        router_input = x[:, :, :1, :] if self.logical_router else x
        router_shared=ttnn.linear(router_input,self.router_shared_packed,compute_kernel_config=self.compute)
        logits=router_shared[...,:512]
        values,indices=ttnn.topk(logits,k=32,dim=-1,sorted=True)
        selected=ttnn.softmax(ttnn.add(values,self.top_mask),dim=-1,numeric_stable=True,compute_kernel_config=self.compute)
        routing=ttnn.scatter(ttnn.zeros_like(logits),dim=3,index=indices,src=selected)
        active=ttnn.max(routing,dim=2,keepdim=True)>0
        counts=ttnn.typecast(ttnn.where(active,32,0),ttnn.uint32)
        counts=ttnn.reshape(ttnn.to_layout(counts,ttnn.ROW_MAJOR_LAYOUT),(self.experts,))
        dispatched=self._dispatch_input(x,counts)
        down=ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(dispatched,self.expert_offsets,counts,
            self.expert_ids,self.expert_gates,self.expert_ups,self.expert_downs,32)
        result=self._compact(down,indices,selected)
        sg,su=router_shared[...,512:1152],router_shared[...,1152:1792]
        scale=router_shared[...,1792:1793]
        shared=ttnn.linear(ttnn.multiply(sg,su,input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,compute_kernel_config=self.compute)
        return ttnn.add(result,ttnn.multiply(shared,scale,input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))

class OptimizedHyperConnection(FusedHyperConnection):
    pass


_PARALLEL_STATE_SOURCE = r"""
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Request-slot gather/scatter of complete interleaved state tiles. No numeric
// conversion: FP32 recurrence and BF16 convolution state retain their bits.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t pages = get_compile_time_arg_val(0);
    constexpr uint32_t bytes = get_compile_time_arg_val(1);
    constexpr uint32_t slots = get_compile_time_arg_val(2);
    constexpr bool store = get_compile_time_arg_val(3);
    constexpr uint32_t workers = get_compile_time_arg_val(4);
    constexpr auto ba = TensorAccessorArgs<5>();
    constexpr auto wa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto bank = TensorAccessor(ba, get_arg_val<uint32_t>(0));
    const auto work = TensorAccessor(wa, get_arg_val<uint32_t>(1));
    const auto slot = TensorAccessor(sa, get_arg_val<uint32_t>(2));
    const auto pos = TensorAccessor(pa, get_arg_val<uint32_t>(3));
    const uint32_t lane = get_arg_val<uint32_t>(4);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    Noc noc;
    noc.async_read(slot, CoreLocalMem<uint32_t>(base + bytes), 128, {.page_id = lane}, {});
    noc.async_read(pos, CoreLocalMem<uint32_t>(base + bytes + 128), 128, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t request = words[bytes / 4];
    const uint32_t position = words[bytes / 4 + 32];
    ASSERT(request < slots);
    if (!store && position == 0) {
        for (uint32_t j = 0; j < bytes / 4; ++j) {
            words[j] = 0;
        }
    }
    for (uint32_t tile = get_arg_val<uint32_t>(5); tile < pages; tile += workers) {
        if constexpr (store) {
            noc.async_read(work, CoreLocalMem<uint32_t>(base), bytes, {.page_id = lane * pages + tile}, {});
            noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(base), bank, bytes, {}, {.page_id = request * pages + tile});
        } else {
            if (position != 0) {
                noc.async_read(bank, CoreLocalMem<uint32_t>(base), bytes, {.page_id = request * pages + tile}, {});
                noc.async_read_barrier();
            }
            noc.async_write(CoreLocalMem<uint32_t>(base), work, bytes, {}, {.page_id = lane * pages + tile});
        }
        noc.async_write_barrier();
    }
}
"""


_COMPACT_EXPERT_SOURCE = r'''
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto da=TensorAccessorArgs<0>();
    constexpr auto ia=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto down=TensorAccessor(da,get_arg_val<uint32_t>(0));
    const auto indices=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto out=TensorAccessor(oa,get_arg_val<uint32_t>(3));
    const uint32_t col=get_arg_val<uint32_t>(4);
    CircularBuffer cb(0); cb.reserve_back(1);
    const uint32_t base=cb.get_write_ptr();
    Noc noc;
    noc.async_read(indices,CoreLocalMem<uint32_t>(base+1088),128,{.page_id=0},{});
    noc.async_read(scores,CoreLocalMem<uint32_t>(base+1216),64,{.page_id=0},{});
    noc.async_read_barrier();
    auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+1088);
    auto weights=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+1216);
    for(uint32_t e=0;e<10;++e) {
        ASSERT(ids[e]<512);
        if (weights[e]!=0) {
            noc.async_read(down,CoreLocalMem<uint32_t>(base),1088,{.page_id=ids[e]*80+col},{});
            noc.async_read_barrier();
        } else {
            auto words=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
            for(uint32_t j=0;j<272;++j) words[j]=0;
        }
        noc.async_write(CoreLocalMem<uint32_t>(base),out,1088,{}, {.page_id=e*80+col});
        noc.async_write_barrier();
    }
}
'''

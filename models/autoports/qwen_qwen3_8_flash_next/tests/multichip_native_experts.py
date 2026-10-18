"""Native selected-expert EP experiment, using one representation for both modes.

Native fused expert FFN receives only locally selected experts (counts>0).
It computes a token tile per active expert; a device compactor reads only each
row's selected experts. Weight payload equals TP4/sparse EP4, without duplicates.
"""

import torch, ttnn


class NativeExpertParallel:
    def __init__(self, state, cfg, ops):
        self.ops = ops
        self.router = ops.upload(
            torch.cat(
                (
                    state["mlp.gate.weight"].T,
                    torch.nn.functional.pad(state["mlp.shared_expert_gate.weight"].T, (0, 31)),
                ),
                -1,
            )
        )
        self.shared_gu = ops.shards(
            [
                torch.cat(
                    (
                        state["mlp.shared_expert.gate_proj.weight"][r * 160 : (r + 1) * 160].T,
                        state["mlp.shared_expert.up_proj.weight"][r * 160 : (r + 1) * 160].T,
                    ),
                    -1,
                )
                for r in range(4)
            ]
        )
        self.shared_down = ops.shards(
            [state["mlp.shared_expert.down_proj.weight"][:, r * 160 : (r + 1) * 160].T for r in range(4)]
        )
        mask = torch.full((1, 1, 1, 32), float("-inf"))
        mask[..., :10] = 0
        self.mask = ops.upload(mask)
        self.shared_compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        gu = state["mlp.experts.gate_up_proj"]
        down = state["mlp.experts.down_proj"]
        self.gates = []
        self.ups = []
        self.downs = []
        for e in range(128):
            self.gates.append(ops.shards([gu[r * 128 + e, :640].T for r in range(4)], ttnn.bfloat4_b))
            self.ups.append(ops.shards([gu[r * 128 + e, 640:].T for r in range(4)], ttnn.bfloat4_b))
            self.downs.append(ops.shards([down[r * 128 + e].T for r in range(4)], ttnn.bfloat4_b))
        self.ids = ops.upload(torch.arange(128, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.offsets = ops.upload(torch.arange(128, dtype=torch.int32) * 32, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.rank = ops.shards(
            [torch.tensor([r], dtype=torch.int32) for r in range(4)], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.capacity = ttnn.empty(
            (1, 1, 128 * 32, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=ops.mesh
        )
        self.columns_by_n = {
            n: ops.shards(
                [
                    torch.arange(r * 128, (r + 1) * 128, dtype=torch.int32).reshape(1, 1, 1, 128).expand(1, 1, n, 128)
                    for r in range(4)
                ],
                ttnn.uint32,
            )
            for n in range(1, 33)
        }

    def group_native(self, x):
        n = x.shape[2]
        router = ttnn.linear(x, self.router, compute_kernel_config=self.shared_compute)
        logits = router[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
        )
        routing = ttnn.gather(
            ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected), 3, index=self.columns_by_n[n]
        )
        counts = ttnn.to_layout(
            ttnn.typecast(ttnn.multiply(ttnn.gt(ttnn.max(routing, dim=2, keepdim=True), 0), n), ttnn.uint32),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        counts = ttnn.to_memory_config(counts, ttnn.DRAM_MEMORY_CONFIG)
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        _kernel(self.ops.mesh, [source, counts, self.capacity], _DISPATCH, [n], workers=32, cb_size=5632)
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            self.capacity, self.offsets, counts, self.ids, self.gates, self.ups, self.downs, 32
        )
        ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
        compact = ttnn.empty(
            (1, 10, 32, 2560),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.ops.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _kernel(self.ops.mesh, [down, ids, scores, self.rank, compact], _COMPACT, [n], workers=80, cb_size=8448)
        padded_scores = (
            ttnn.pad(selected[..., :10], [(0, 0), (0, 0), (0, 32 - n), (0, 0)], 0) if n < 32 else selected[..., :10]
        )
        weighted = ttnn.multiply(compact, ttnn.permute(padded_scores, (0, 3, 2, 1)), output_tensor=compact)
        result = ttnn.sum(weighted, dim=1, keepdim=True)[:, :, :n, :]
        shared = ttnn.linear(x, self.shared_gu, compute_kernel_config=self.shared_compute)
        shared = ttnn.linear(
            ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.shared_compute,
        )
        return ttnn.add(
            result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        )

    def __call__(self, x):
        b, s = tuple(x.shape)[1:3]
        lanes = []
        for lane in range(b):
            chunks = [
                self.group_native(x[:, lane : lane + 1, start : min(start + 32, s), :]) for start in range(0, s, 32)
            ]
            lanes.append(chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2))
        result = lanes[0] if len(lanes) == 1 else ttnn.concat(lanes, dim=1)
        return self.ops.scatter_reduce(result)


def _kernel(mesh, tensors, source, extra, workers, cb_size):
    coords = [ttnn.CoreCoord(i % 10, i // 10) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = [*extra, workers]
    for t in tensors:
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernel = ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=cb_size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=cb_size)],
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))


_HEADER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
"""
_DISPATCH = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0),W=get_compile_time_arg_val(1);
    constexpr auto xa=TensorAccessorArgs<2>();
    constexpr auto ca=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(1));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(2));
    const uint32_t worker=get_arg_val<uint32_t>(3);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    noc.async_read(counts,CoreLocalMem<uint32_t>(base+5120),512,{.page_id=0},{});noc.async_read_barrier();
    auto active=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+5120);
    for(uint32_t e=worker;e<128;e+=W) if(active[e]) {
        ASSERT(active[e]==N);
        for(uint32_t row=0;row<N;++row) {
            noc.async_read(x,CoreLocalMem<uint32_t>(base),5120,{.page_id=row},{});noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(base),output,5120,{}, {.page_id=e*32+row});noc.async_write_barrier();
        }
    }
}
"""
_COMPACT = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0);
    constexpr auto da=TensorAccessorArgs<2>();
    constexpr auto ia=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    const auto down=TensorAccessor(da,get_arg_val<uint32_t>(0));
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(4));
    const uint32_t col=get_arg_val<uint32_t>(5);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    for(uint32_t row=0;row<N;++row) {
        noc.async_read(ids,CoreLocalMem<uint32_t>(base+2176+row*128),128,{.page_id=row},{});
        noc.async_read(scores,CoreLocalMem<uint32_t>(base+6272+row*64),64,{.page_id=row},{});
    }
    noc.async_read(rank_tensor,CoreLocalMem<uint32_t>(base+8320),4,{.page_id=0},{});noc.async_read_barrier();
    const uint32_t rank=*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+8320);
    auto id=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+2176);
    auto score=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+6272);
    auto src=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base);
    auto out=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base+1088);
    for(uint32_t route=0;route<10;++route) {
        for(uint32_t i=0;i<1088;++i)out[i]=0;
        for(uint32_t row=0;row<N;++row) {
            const uint32_t e=id[row*32+route];ASSERT(e<512);
            if(e/128!=rank || score[row*32+route]==0)continue;
            noc.async_read(down,CoreLocalMem<uint32_t>(base),1088,{.page_id=(e%128)*80+col},{});noc.async_read_barrier();
            for(uint32_t side=0;side<2;++side) {
                const uint32_t face=(row/16)*2+side,exp=face*16+row%16,data=64+face*256+(row%16)*16;
                out[exp]=src[exp];for(uint32_t j=0;j<16;++j)out[data+j]=src[data+j];
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(base+1088),output,1088,{}, {.page_id=route*80+col});noc.async_write_barrier();
    }
}
"""


def main():
    from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench
    from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target

    target.ParallelMoE = NativeExpertParallel
    bench.CANDIDATE = dict(expert_placement="native_ep_single_representation")
    bench.main()


if __name__ == "__main__":
    main()

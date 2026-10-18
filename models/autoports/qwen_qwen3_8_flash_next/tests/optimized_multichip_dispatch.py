"""Selected-route dispatch candidate, with identical active-expert math."""
import sys
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import _expert_kernel, _EXPERT_COMPACT

class DirectDispatch(target.ParallelMoE):
    def group_native(self, x):
        n = x.shape[2]
        router = self.ops.linear(x, self.router, compute_kernel_config=self.shared_compute)
        logits = router[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
        )
        ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
        counts = ttnn.empty((128,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        _expert_kernel(self.ops.mesh, [source, ids, scores, self.rank, counts, self.capacity],
                       DISPATCH, [n], workers=32, cb_size=5888)
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            self.capacity, self.offsets, counts, self.ids, self.gates, self.ups, self.downs, 32
        )
        compact = ttnn.empty(
            (1, 10, 32, 2560),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.ops.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _expert_kernel(
            self.ops.mesh, [down, ids, scores, self.rank, compact], _EXPERT_COMPACT, [n], workers=80, cb_size=8448
        )
        padded_scores = (
            ttnn.pad(selected[..., :10], [(0, 0), (0, 0), (0, 32 - n), (0, 0)], 0) if n < 32 else selected[..., :10]
        )
        weighted = ttnn.multiply(compact, ttnn.permute(padded_scores, (0, 3, 2, 1)), output_tensor=compact)
        result = ttnn.sum(weighted, dim=1, keepdim=True)[:, :, :n, :]
        shared = self.ops.linear(x, self.shared_gu, compute_kernel_config=self.shared_compute)
        shared = self.ops.linear(
            ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.shared_compute,
        )
        return ttnn.add(
            result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        )


DISPATCH = target._EXPERT_HEADER + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0),W=get_compile_time_arg_val(1);
    static_assert(N==1);
    constexpr auto xa=TensorAccessorArgs<2>();
    constexpr auto ia=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0));
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(4));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(5));
    const uint32_t worker=get_arg_val<uint32_t>(6);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    noc.async_read(ids,CoreLocalMem<uint32_t>(base+5120),128,{.page_id=0},{});
    noc.async_read(scores,CoreLocalMem<uint32_t>(base+5248),64,{.page_id=0},{});
    noc.async_read(rank_tensor,CoreLocalMem<uint32_t>(base+5312),4,{.page_id=0},{});
    noc.async_read_barrier();
    const uint32_t rank=*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+5312);
    auto id=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+5120);
    auto score=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+5248);
    auto tmp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+5328);
    for(uint32_t e=worker;e<128;e+=W) {
        uint32_t active=0;
        for(uint32_t route=0;route<10;++route) {
            ASSERT(id[route]<512);
            if(id[route]==rank*128+e && score[route]!=0)active=1;
        }
        if(active) {
            noc.async_read(x,CoreLocalMem<uint32_t>(base),5120,{.page_id=0},{});noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(base),output,5120,{}, {.page_id=e*32});
        }
        const uint32_t lane=(e*4)&15;tmp[lane/4]=active;
        noc.async_write(CoreLocalMem<uint32_t>(base+5328+lane),counts,4,{}, {.page_id=0,.offset_bytes=e*4});
        noc.async_write_barrier();
    }
}
"""

def main():
    target.ParallelMoE = DirectDispatch
    from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench
    bench.CANDIDATE = {"name": "direct_selected_route_dispatch"}
    bench.main()

if __name__ == "__main__":
    main()

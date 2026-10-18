"""EP prefill packing: each selected row is dispatched once to its local owner."""

import math
import torch, ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_native_experts import (
    NativeExpertParallel,
    _kernel,
    _HEADER,
)


class PackedExpertParallel(NativeExpertParallel):
    def __init__(self, state, cfg, ops):
        super().__init__(state, cfg, ops)
        self.offsets128 = ops.upload(torch.arange(128, dtype=torch.int32) * 128, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def group_packed(self, x):
        n = x.shape[2]
        padded = math.ceil(n / 32) * 32
        router = ttnn.linear(x, self.router, compute_kernel_config=self.shared_compute)
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
        mapping = ttnn.empty((1, 1, n, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
        capacity = ttnn.empty(
            (1, 1, 128 * 128, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh
        )
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        _kernel(
            self.ops.mesh,
            [source, ids, scores, self.rank, counts, mapping, capacity],
            _DISPATCH_PACKED,
            [n],
            workers=32,
            cb_size=5120 + n * 192 + 64,
        )
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            capacity, self.offsets128, counts, self.ids, self.gates, self.ups, self.downs, 128
        )
        compact = ttnn.empty(
            (1, 10, padded, 2560),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.ops.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _kernel(
            self.ops.mesh,
            [down, ids, scores, self.rank, mapping, compact],
            _COMPACT_PACKED,
            [n],
            workers=80,
            cb_size=2176 + n * 320 + 64,
        )
        scores = (
            ttnn.pad(selected[..., :10], [(0, 0), (0, 0), (0, padded - n), (0, 0)], 0)
            if n < padded
            else selected[..., :10]
        )
        weighted = ttnn.multiply(compact, ttnn.permute(scores, (0, 3, 2, 1)), output_tensor=compact)
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
        if s == 1:
            return super().__call__(x)
        lanes = []
        for lane in range(b):
            chunks = [
                self.group_packed(x[:, lane : lane + 1, start : min(start + 128, s), :]) for start in range(0, s, 128)
            ]
            lanes.append(chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2))
        return self.ops.scatter_reduce(lanes[0] if len(lanes) == 1 else ttnn.concat(lanes, dim=1))


_DISPATCH_PACKED = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0),W=get_compile_time_arg_val(1);
    constexpr auto xa=TensorAccessorArgs<2>();
    constexpr auto ia=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto ma=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0));
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(4));
    const auto mapping=TensorAccessor(ma,get_arg_val<uint32_t>(5));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(6));
    const uint32_t worker=get_arg_val<uint32_t>(7);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    constexpr uint32_t scratch=5120+N*192;
    for(uint32_t row=0;row<N;++row) {
        noc.async_read(ids,CoreLocalMem<uint32_t>(base+5120+row*128),128,{.page_id=row},{});
        noc.async_read(scores,CoreLocalMem<uint32_t>(base+5120+N*128+row*64),64,{.page_id=row},{});
    }
    noc.async_read(rank_tensor,CoreLocalMem<uint32_t>(base+scratch),4,{.page_id=0},{});noc.async_read_barrier();
    const uint32_t rank=*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+scratch);
    auto id=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+5120);
    auto score=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+5120+N*128);
    auto tmp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+scratch);
    for(uint32_t e=worker;e<128;e+=W) {
        uint32_t count=0;
        for(uint32_t row=0;row<N;++row)for(uint32_t route=0;route<10;++route) {
            if(id[row*32+route]!=rank*128+e || score[row*32+route]==0)continue;
            noc.async_read(x,CoreLocalMem<uint32_t>(base),5120,{.page_id=row},{});noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(base),output,5120,{}, {.page_id=e*128+count});
            const uint32_t map_lane=(route*4)&15;
            tmp[map_lane/4]=count;
            noc.async_write(CoreLocalMem<uint32_t>(base+scratch+map_lane),mapping,4,{}, {.page_id=row,.offset_bytes=route*4});noc.async_write_barrier();
            ++count;
        }
        const uint32_t count_lane=(e*4)&15;
        tmp[count_lane/4]=count;noc.async_write(CoreLocalMem<uint32_t>(base+scratch+count_lane),counts,4,{}, {.page_id=0,.offset_bytes=e*4});noc.async_write_barrier();
    }
}
"""
_COMPACT_PACKED = _HEADER + r"""
void kernel_main() {
    constexpr uint32_t N=get_compile_time_arg_val(0),T=(N+31)/32;
    constexpr auto da=TensorAccessorArgs<2>();
    constexpr auto ia=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ma=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    const auto down=TensorAccessor(da,get_arg_val<uint32_t>(0));
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto rank_tensor=TensorAccessor(ra,get_arg_val<uint32_t>(3));
    const auto mapping=TensorAccessor(ma,get_arg_val<uint32_t>(4));
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(5));
    const uint32_t col=get_arg_val<uint32_t>(6);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    constexpr uint32_t ib=2176,sb=ib+N*128,mb=sb+N*64,rb=mb+N*128;
    for(uint32_t row=0;row<N;++row) {
        noc.async_read(ids,CoreLocalMem<uint32_t>(base+ib+row*128),128,{.page_id=row},{});
        noc.async_read(scores,CoreLocalMem<uint32_t>(base+sb+row*64),64,{.page_id=row},{});
        noc.async_read(mapping,CoreLocalMem<uint32_t>(base+mb+row*128),128,{.page_id=row},{});
    }
    noc.async_read(rank_tensor,CoreLocalMem<uint32_t>(base+rb),4,{.page_id=0},{});noc.async_read_barrier();
    const uint32_t rank=*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+rb);
    auto id=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+ib);
    auto score=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+sb);
    auto map=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+mb);
    auto src=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base);
    auto out=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base+1088);
    for(uint32_t route=0;route<10;++route)for(uint32_t tile=0;tile<T;++tile) {
        for(uint32_t i=0;i<1088;++i)out[i]=0;
        for(uint32_t row=tile*32;row<N && row<(tile+1)*32;++row) {
            const uint32_t e=id[row*32+route];ASSERT(e<512);
            if(e/128!=rank || score[row*32+route]==0)continue;
            const uint32_t packed=map[row*32+route];ASSERT(packed<128);
            noc.async_read(down,CoreLocalMem<uint32_t>(base),1088,{.page_id=((e%128)*4+packed/32)*80+col},{});noc.async_read_barrier();
            for(uint32_t side=0;side<2;++side) {
                const uint32_t sf=(packed%32/16)*2+side,df=(row%32/16)*2+side;
                out[df*16+row%16]=src[sf*16+packed%16];
                for(uint32_t j=0;j<16;++j)out[64+df*256+(row%16)*16+j]=src[64+sf*256+(packed%16)*16+j];
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(base+1088),output,1088,{}, {.page_id=(route*T+tile)*80+col});noc.async_write_barrier();
    }
}
"""


def main():
    from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench
    from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target

    target.ParallelMoE = PackedExpertParallel
    bench.CANDIDATE = dict(expert_placement="native_ep_packed_prefill")
    bench.main()


if __name__ == "__main__":
    main()

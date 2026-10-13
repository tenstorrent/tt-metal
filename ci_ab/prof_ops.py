#!/usr/bin/env python3
"""Scratch, not for merge. Device time of the production ops that run the bitonic TopK network (ttnn.topk single-core and
tree merge, sampling, sort, moe, the DeepSeek grouped gates, the GPT-OSS router) under the device profiler: per case a
signpost, N_MEAS measured calls after N_WARM warm-up calls, then an "end" signpost. Cases from $OPS_CASES (all if empty);
OPS_ROW_DISPATCH=1 opens the device with row dispatch (the GPT-OSS router needs eight DRAM-aligned workers)."""
import gc
import os

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("OPS_NMEAS", "8"))


def tile(t, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)


def case_topk(shape, k, dtype=ttnn.bfloat16, tdtype=torch.bfloat16, largest=True, stable=False, grid=None, sub=None):
    def setup(dev):
        torch.manual_seed(2007)
        x = tile(torch.randn(shape, dtype=torch.float32).to(tdtype), dev, dtype)
        kw = {}
        if grid is not None:
            kw["sub_core_grids"] = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0], grid[1] + 1))])
        if sub is not None:
            kw["sub_core_grids"] = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*sub[0]), ttnn.CoreCoord(*sub[1]))])
        return (x, kw)

    def run(x, kw):
        return ttnn.topk(x, k, dim=-1, largest=largest, sorted=True, stable=stable, **kw)

    return setup, run


def case_moe(shape, k, E, e):
    def setup(dev):
        torch.manual_seed(2005)
        x = torch.randn(shape, dtype=torch.bfloat16)
        x[..., E:] = 0
        em = torch.zeros(list(shape[:2]) + [1, shape[-1]], dtype=torch.bfloat16)
        em[..., E:] = float("-inf")
        tm = torch.zeros(list(shape[:2]) + [1, k], dtype=torch.bfloat16)
        tm[..., e:] = float("-inf")
        return (tile(x, dev), tile(em, dev), tile(tm, dev))

    def run(x, em, tm):
        return ttnn.moe(x, em, tm, k)

    return setup, run


def case_router(tokens, experts, k):
    def setup(dev):
        torch.manual_seed(0)
        t = torch.softmax(torch.randn((1, 1, tokens, experts), dtype=torch.float32), dim=-1).to(torch.bfloat16)
        return (tile(t, dev),)

    def run(x):
        return ttnn.topk(x, k=k, dim=-1)

    return setup, run


def case_sampling(shape):
    def setup(dev):
        torch.manual_seed(2024)
        v = tile(torch.randn(shape), dev)
        i = ttnn.from_torch(torch.arange(0, shape[-1], dtype=torch.int32).expand(shape), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        k = ttnn.from_torch(torch.tensor([10, 15, 20, 25, 30] * 6 + [10, 20]), device=dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        p = ttnn.from_torch(torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        temp = ttnn.from_torch(torch.ones(32), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        return (v, i, k, p, temp)

    def run(v, i, k, p, temp):
        return ttnn.sampling(v, i, k=k, p=p, temp=temp, seed=2024)

    return setup, run


def case_sort(shape, descending=True, stable=False):
    def setup(dev):
        torch.manual_seed(0)
        return (tile(torch.randn(shape, dtype=torch.float32).to(torch.bfloat16), dev),)

    def run(x):
        return ttnn.sort(x, dim=-1, descending=descending, stable=stable)

    return setup, run


def case_moe_grouped(tokens, stable_sort=True):
    def setup(dev):
        torch.manual_seed(7)
        sc = torch.randn((1, 1, tokens, 256), dtype=torch.float32).to(torch.bfloat16)
        bias = (0.1 * torch.randn((1, 1, tokens, 256), dtype=torch.float32)).to(torch.bfloat16)
        return (tile(sc, dev), tile(bias, dev))

    def run(sc, bias):
        return ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            sc, bias, n_groups=8, summed_experts_per_group=2, topk_groups=4, n_activated_experts=8,
            route_scale=2.5, stable_sort=stable_sort, epsilon=1e-20, score_func="sigmoid",
            weights_layout=ttnn.ROW_MAJOR_LAYOUT)

    return setup, run


def case_grouped_gate(nb, bs, sl):
    def setup(dev):
        torch.manual_seed(11)
        sc = torch.sigmoid(torch.randn((nb, bs, sl, 256), dtype=torch.float32)).to(torch.bfloat16)
        bias = torch.randn((nb, bs, sl, 256), dtype=torch.float32).to(torch.bfloat16)
        return (tile(sc, dev), tile(bias, dev))

    def run(sc, bias):
        return ttnn.experimental.deepseek_grouped_gate(
            sc, bias, n_groups=8, summed_experts_per_group=2, topk_groups=4, n_activated_experts=8,
            route_scale=2.5, epsilon=1e-20)

    return setup, run


def case_router_gpt(B, K, N, k):
    def setup(dev):
        torch.manual_seed(42)
        x = tile(torch.randn((B, K), dtype=torch.bfloat16), dev)
        w = tile(torch.randn((K, N), dtype=torch.bfloat16) * 0.02, dev)
        b = tile(torch.randn((1, N), dtype=torch.bfloat16).expand(B, N).contiguous(), dev)
        return (x, w, b)

    def run(x, w, b):
        return ttnn.experimental.topk_router_gpt(x, weight_tensor=w, bias_tensor=b, k=k, num_experts=N)

    return setup, run


CASES = {
    # ttnn.topk single core: MoE routers, wide rows, uint32 indices, stable forms
    "router_e128_t32_k4": case_router(32, 128, 4),
    "router_e128_t1024_k4": case_router(1024, 128, 4),
    "router_e256_t32_k8": case_router(32, 256, 8),
    "router_e256_t2048_k8": case_router(2048, 256, 8),
    "router_e512_t128_k10": case_router(128, 512, 10),
    "router_e256_t1024_k8_stable": case_topk((1, 1, 1024, 256), 8, stable=True),
    "sampling_chunk_16032_k32_subgrid": case_topk((1, 1, 32, 16032), 32, sub=((0, 0), (0, 0))),
    "sampling_chunk_16032_k32_subgrid_stable": case_topk((1, 1, 32, 16032), 32, stable=True, sub=((0, 0), (0, 0))),
    "w2048_h256_k32": case_topk((1, 1, 256, 2048), 32),
    "w2048_h256_k64": case_topk((1, 1, 256, 2048), 64),
    "w2048_h256_k32_smallest": case_topk((1, 1, 256, 2048), 32, largest=False),
    "w2048_h256_k32_fp32": case_topk((1, 1, 256, 2048), 32, dtype=ttnn.float32, tdtype=torch.float32),
    "w2048_h256_k32_fp32_stable": case_topk((1, 1, 256, 2048), 32, dtype=ttnn.float32, tdtype=torch.float32, stable=True),
    "w2048_h256_k32_stable": case_topk((1, 1, 256, 2048), 32, stable=True),
    "w2048_h256_k64_stable": case_topk((1, 1, 256, 2048), 64, stable=True),
    "w65600_h32_k32_smallest_u32idx": case_topk((1, 1, 32, 65600), 32, largest=False),
    "w65600_h32_k32_smallest_u32idx_stable": case_topk((1, 1, 32, 65600), 32, largest=False, stable=True),
    "uniad_topk_1x900_k100_stable": case_topk((1, 900), 100, stable=True),
    # ttnn.topk tree merge (#57396): sampling_1d shard, low-tile-row rows, the multicore tree on 2 to 32 local cores
    "w16384_h32_k32": case_topk((1, 1, 32, 16384), 32),
    "w16384_h32_k64": case_topk((1, 1, 32, 16384), 64),
    "w16384_h32_k32_stable": case_topk((1, 1, 32, 16384), 32, stable=True),
    "w8192_h64_k64": case_topk((1, 1, 64, 8192), 64),
    "w1024_h32_k32": case_topk((1, 1, 32, 1024), 32),
    "w1024_h32_k64": case_topk((1, 1, 32, 1024), 64),
    "w2048_h32_k8": case_topk((1, 1, 32, 2048), 8),
    "w2048_h32_k32": case_topk((1, 1, 32, 2048), 32),
    "w2048_h32_k64": case_topk((1, 1, 32, 2048), 64),
    # sampling, sort, moe, DeepSeek prefill gate, DeepSeek grouped gate
    "sampling_llama_tg_w256": case_sampling([1, 1, 32, 256]),
    "sampling_llama_n300_w64": case_sampling([1, 1, 32, 64]),
    "sort_32x2048_desc": case_sort((1, 1, 32, 2048)),
    "sort_32x2048_desc_stable": case_sort((1, 1, 32, 2048), stable=True),
    "sort_64x64_asc": case_sort((1, 1, 64, 64), descending=False),
    "uniad_sort_1x101_desc_stable": case_sort((1, 101), stable=True),
    "moe_mixtral_decode_w64_k32": case_moe((1, 1, 32, 64), 32, 8, 2),
    "gate_t640_stable": case_moe_grouped(640),
    "gate_t4096_stable": case_moe_grouped(4096),
    "gate_t640": case_moe_grouped(640, stable_sort=False),
    "grouped_gate_1x1x8192": case_grouped_gate(1, 1, 8192),
    "grouped_gate_1x8x512": case_grouped_gate(1, 8, 512),
}
for name, g in (("n2", (2, 1)), ("n4", (4, 1)), ("n8", (4, 2)), ("n32", (8, 4))):
    for k in (32, 64):
        for st in (False, True):
            CASES[f"tree_{name}_k{k}" + ("_stable" if st else "")] = case_topk((1, 1, 64, 16384), k, stable=st, grid=g)
ROW_CASES = {"router_gpt_32x2880x128_k4": case_router_gpt(32, 2880, 128, 4)}


def main():
    row = os.environ.get("OPS_ROW_DISPATCH") == "1"
    table = ROW_CASES if row else CASES
    names = os.environ.get("OPS_CASES", "").split() or list(table)
    kw = {"dispatch_core_config": ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.ROW)} if row else {}
    dev = ttnn.open_device(device_id=0, **kw)
    try:
        for name in names:
            setup, run = table[name]
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(*args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    run(*args)
                ttnn.synchronize_device(dev)
                signpost(header="end")
                ttnn.ReadDeviceProfiler(dev)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                ttnn.synchronize_device(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()

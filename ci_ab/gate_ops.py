#!/usr/bin/env python3
"""Scratch, not for merge. The prefill MoE gate (moe_grouped_topk) and the hash gate at the model configurations.
Per case GATE_WARM calls, then GATE_NMEAS calls, each after a signpost "<case>#<k>"; the device profiler is read after
every case. GATE_MOCK=1 (mock cluster): one call per case, no profiler, only the JIT builds matter.
usage: gate_ops.py [case...]"""
import gc
import os
import sys

import torch
import ttnn

MOCK = os.environ.get("GATE_MOCK") == "1"
N_WARM = 0 if MOCK else int(os.environ.get("GATE_WARM", "2"))
N_MEAS = 1 if MOCK else int(os.environ.get("GATE_NMEAS", "12"))
try:
    from tracy import signpost
except ImportError:  # the mock runs need no signposts
    def signpost(header):
        pass


def moe(tokens, experts, n_groups, topk_groups, n_act, scale, score="sigmoid", stable=True, tile_w=False):
    def setup(dev):
        torch.manual_seed(7)
        sc = torch.randn((1, 1, tokens, experts), dtype=torch.float32).to(torch.bfloat16)
        bias = (0.1 * torch.randn((1, 1, tokens, experts), dtype=torch.float32)).to(torch.bfloat16)
        t = lambda a: ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        return (t(sc), t(bias))

    def run(dev, sc, bias):
        return ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            sc, bias, n_groups=n_groups, summed_experts_per_group=n_groups // topk_groups if n_groups > 1 else 1,
            topk_groups=topk_groups, n_activated_experts=n_act, route_scale=scale, stable_sort=stable,
            epsilon=1e-20, score_func=score,
            weights_layout=ttnn.ROW_MAJOR_LAYOUT if n_act <= 32 and not tile_w else ttnn.TILE_LAYOUT)

    return setup, run


def hashg(tokens, experts, k, scale, vocab=2048, score="sqrtsoftplus"):
    def setup(dev):
        torch.manual_seed(7)
        logits = torch.randn((1, 1, tokens, experts), dtype=torch.float32)
        ids = torch.randint(0, vocab, (tokens,), dtype=torch.int64)
        t2 = torch.zeros(vocab, 16, dtype=torch.int16)
        t2[:, :k] = torch.stack([torch.randperm(experts)[:k] for _ in range(vocab)]).to(torch.int16)
        return (ttnn.from_torch(logits, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev),
                ttnn.from_torch(ids.reshape(tokens // 32, 32).to(torch.int32), dtype=ttnn.uint32,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=dev),
                ttnn.from_torch(t2, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev))

    def run(dev, logits, ids, tid2eid):
        return ttnn.experimental.deepseek_prefill.moe_hash_gate(
            logits, ids, tid2eid, n_activated_experts=k, route_scale=scale, epsilon=1e-20, score_func=score)

    return setup, run


CASES = {
    # DeepSeek V3: 8 groups of 32 experts, 4 kept, 8 activated; stable as the model calls it, and the unstable network
    "dsv3_t640": moe(640, 256, 8, 4, 8, 2.5),
    "dsv3_t640_u": moe(640, 256, 8, 4, 8, 2.5, stable=False),
    "dsv3_t4096": moe(4096, 256, 8, 4, 8, 2.5),
    "dsv3_t4096_u": moe(4096, 256, 8, 4, 8, 2.5, stable=False),
    "kimik3_t640": moe(640, 896, 1, 1, 16, 1.0),
    "dsv4f_t640": moe(640, 256, 1, 1, 6, 1.5, "sqrtsoftplus"),
    "dsv4p_t640": moe(640, 384, 1, 1, 6, 1.5, "sqrtsoftplus"),
    "glm53_t640": moe(640, 256, 1, 1, 8, 2.5),
    "kimik27_t640": moe(640, 384, 1, 1, 8, 2.5),
    "mm3_t640": moe(640, 128, 1, 1, 4, 2.0, "sigmoid", stable=False, tile_w=True),
    "hash_v4f_t640": hashg(640, 256, 6, 1.5),
    "hash_v4p_t640": hashg(640, 384, 6, 2.5),
}


def main():
    names = sys.argv[1:] or list(CASES)
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        for name in names:
            setup, run = CASES[name]
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                for k in range(N_MEAS):
                    signpost(header=f"{name}#{k}")
                    run(dev, *args)
                    ttnn.synchronize_device(dev)
                if not MOCK:
                    ttnn.ReadDeviceProfiler(dev)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            args = None
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Scratch, not for merge. ttnn.experimental.deepseek.moe.generalized_moe_gate, one 256-expert token per core.
GATE_MODE=prof: device time under the device profiler (signpost per case, N_MEAS calls after N_WARM).
GATE_MODE=dump <outdir>: outputs of many tokens per configuration saved to <outdir>/results.pt for an offline compare."""
import os
import sys

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("GATE_NMEAS", "8"))
MODE = os.environ.get("GATE_MODE", "prof")


def sharded(dev, t, dtype, cores):
    grid = dev.compute_with_storage_grid_size()
    cg = ttnn.num_cores_to_corerangeset(cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    spec = ttnn.ShardSpec(cg, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc, tile=ttnn.Tile((32, 32)))


def make(dev, batch, logits, bias):
    shape = (batch, 16, 16)
    x = sharded(dev, logits.reshape(shape).to(torch.bfloat16), ttnn.bfloat16, batch)
    b = sharded(dev, torch.transpose(bias.reshape(shape), -2, -1).to(torch.bfloat16), ttnn.bfloat16, batch)
    ids = torch.arange(256, dtype=torch.int32).unsqueeze(0).expand(batch, -1).reshape(shape)
    i = sharded(dev, torch.transpose(ids, -2, -1).to(torch.uint16), ttnn.uint16, batch)
    o = sharded(dev, torch.zeros((batch, 1, 16), dtype=torch.bfloat16), ttnn.bfloat16, batch)
    oi = sharded(dev, torch.zeros((batch, 1, 16), dtype=torch.uint16), ttnn.uint16, batch)
    return x, b, i, o, oi


def run(x, b, i, o, oi, sig, topk, smax):
    return ttnn.experimental.deepseek.moe.generalized_moe_gate(
        x, bias_tensor=b, input_indices_tensor=i, output_tensor=o, output_indices_tensor=oi, eps=1e-20,
        scaling_factor=2.5, enable_sigmoid=sig, topk=topk, output_softmax=smax)


def stimuli(batch, kind, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":  # the op test's stimulus
        logits = 2 * torch.rand((batch, 256), generator=g) - 1
        bias = 2 * torch.rand((batch, 256), generator=g) - 1
    elif kind == "router":  # router logits and a small score-correction bias, as DeepSeek / Kimi gates see them
        logits = torch.randn((batch, 256), generator=g) * 2.0
        bias = torch.randn((batch, 256), generator=g) * 0.02
    else:  # "nobias": softmax-style models without a correction bias
        logits = torch.randn((batch, 256), generator=g) * 3.0
        bias = torch.zeros((batch, 256))
    return logits.to(torch.bfloat16), bias.to(torch.bfloat16)


CONFIGS = [(sig, topk, smax) for sig in (True, False) for topk in (8, 6, 4) for smax in (False, True)]


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        if MODE == "prof":
            for batch in (1, 32, 64):
                for sig, topk, smax in ((True, 8, False), (False, 8, True), (True, 6, False), (False, 4, True)):
                    name = f"gate_b{batch}_{'sig' if sig else 'raw'}_k{topk}_{'smax' if smax else 'lin'}"
                    logits, bias = stimuli(batch, "router", 7)
                    if not sig:
                        logits = torch.sigmoid(logits.float()).to(torch.bfloat16) if not smax else logits
                    args = make(dev, batch, logits, bias)
                    for _ in range(N_WARM):
                        run(*args, sig, topk, smax)
                    ttnn.synchronize_device(dev)
                    signpost(header=name)
                    for _ in range(N_MEAS):
                        run(*args, sig, topk, smax)
                    ttnn.synchronize_device(dev)
                    signpost(header="end")
                    ttnn.ReadDeviceProfiler(dev)
                    print(f"CASE {name} ok", flush=True)
        else:
            out = sys.argv[1]
            os.makedirs(out, exist_ok=True)
            res = {}
            for kind in ("uniform", "router", "nobias"):
                for seed in range(int(os.environ.get("GATE_SEEDS", "8"))):
                    logits, bias = stimuli(64, kind, 1000 + seed)
                    for sig, topk, smax in CONFIGS:
                        lg = logits
                        if not sig and not smax:
                            lg = torch.sigmoid(logits.float()).to(torch.bfloat16)
                        args = make(dev, 64, lg, bias)
                        w, idx = run(*args, sig, topk, smax)
                        res[f"{kind}_s{seed}_{int(sig)}_{topk}_{int(smax)}"] = [lg, bias, ttnn.to_torch(w)[:, 0, :8].clone(), ttnn.to_torch(idx)[:, 0, :8].clone()]
                print(f"CASE {kind} ok", flush=True)
            torch.save(res, os.path.join(out, "results.pt"))
            print("DUMP DONE", len(res), flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()

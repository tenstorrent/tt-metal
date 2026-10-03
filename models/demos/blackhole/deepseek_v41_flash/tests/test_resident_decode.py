# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""All selected layers resident on the 4x8 mesh, one decode token (batch 16) as ONE captured trace; reports the
replay latency (ms/token for the device part: layers only, no host embedding / Engram / LM head).
Env: DSV41_LAYERS (default "0-39"), DSV41_CHAIN (reference chain dir for cache state and the input)."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain

CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-e")
LAYERS = os.environ.get("DSV41_LAYERS", "0-39")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 600_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_resident_decode(mesh_device):
    md = mesh_device
    a, _, b = LAYERS.partition("-")
    layers = list(range(int(a), int(b or a) + 1))
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S = toks["prefill_tokens"].shape[1]
    chain = DSV41DecodeChain(md, log=lambda m: print(m, flush=True))
    pool = ThreadPoolExecutor(max_workers=2)
    futs = {}
    submit = lambda L: futs.setdefault(L, pool.submit(load_layer, L)) if L in layers else None
    for L in layers[:2]:
        submit(L)

    built = []  # (layer, step_inputs)
    t0 = time.time()
    for L in layers:
        ref = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
        ref["S"] = S
        submit(L + 1), submit(L + 2)
        w = futs.pop(L).result()
        layer, attn = chain.build_layer(L, ref, w)
        del w
        built.append((layer, attn.step_inputs(torch.full((chain.B,), S))))
        print(f"built layer {L} ({time.time() - t0:.0f}s)", flush=True)
        mv = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
        print(
            f"DRAM allocated/bank {mv.total_bytes_allocated_per_bank / 2**20:.0f} MiB, free {mv.total_bytes_free_per_bank / 2**20:.0f} MiB",
            flush=True,
        )

    first = torch.load(os.path.join(CHAIN, f"layer_{layers[0]}.pt"))
    x0, pre0 = chain.to_dev(first["dec_in"].reshape(chain.B, -1), 4 * 5120), chain.to_dev(
        first["pre_in"].reshape(chain.B, -1), 4
    )

    def forward():
        x, pre = x0, pre0
        for layer, st in built:
            x, pre = layer.forward(x, pre, st)
        return x, pre

    t = time.time()
    forward()  # compile everything
    ttnn.synchronize_device(md)
    print(f"eager first pass {time.time() - t:.1f}s", flush=True)
    t = time.time()
    forward()
    ttnn.synchronize_device(md)
    print(f"eager second pass {(time.time() - t) * 1e3:.1f} ms", flush=True)

    tid = ttnn.begin_trace_capture(md, cq_id=0)
    out = forward()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    n = 20
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    ms = (time.perf_counter() - t) / n * 1e3
    nl = len(layers)
    print(
        f"RESIDENT_TRACED {nl} layers: {ms:.2f} ms/token = {ms / nl:.3f} ms/layer -> {1e3 / ms * nl / 40:.1f} tok/s/user (scaled to 40 layers)",
        flush=True,
    )

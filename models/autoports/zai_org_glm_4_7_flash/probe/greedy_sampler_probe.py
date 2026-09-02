# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the on-device greedy strategies available to this model.

`$tt-enable-tracing` requires the greedy token-out path to be the fastest
*correct* on-device strategy for the target mesh, and says force-argmax is only
a candidate, never the default. This measures both `TTSampling` greedy paths on
a real `[1, 1, 32, 154880]` bf16 logits tensor on one Blackhole chip:

* **split top-k greedy** (the shipped path): 4 same-device vocab chunks ->
  `ttnn.topk(k=32)` each -> concat -> global index offsets -> tie-break ->
  `ttnn.sampling(k=1, p=0, temp=1)`. Emits `[1, 1, 1, 32]` uint32.
* **force-argmax**: untilize the full vocab -> `ttnn.argmax(dim=-1)`. Emits
  `[1, 1, 32]` uint32 - the wrong rank for the `[1, 1, 1, 32]` decode token
  buffer this model feeds back through `tt_out_tok`.

    python models/autoports/zai_org_glm_4_7_flash/probe/greedy_sampler_probe.py
"""

import json
import time
from pathlib import Path

import torch

import ttnn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params

VOCAB = 154880
ROWS = 32
ITERS = 30
OUT = Path(__file__).resolve().parents[1] / "doc" / "full_model" / "greedy_sampler_benchmark.json"


class _Args:
    def __init__(self, mesh_device, allow_force_argmax):
        self.vocab_size = VOCAB
        self.padded_vocab_size = VOCAB
        self.cluster_shape = tuple(int(d) for d in mesh_device.shape)
        self.max_batch_size = ROWS
        self.max_top_k = 32
        self.sampling_dp = 1
        if allow_force_argmax:
            self.model_config = {
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": True,
                    "num_links": 1,
                    "topology": ttnn.Topology.Linear,
                }
            }


def bench(dev, fn, iters=ITERS):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=90_000_000)
    torch.manual_seed(0)
    logits_host = torch.randn(1, 1, ROWS, VOCAB) * 4.0
    expected = logits_host[0, 0].argmax(-1).tolist()
    results = {"vocab_size": VOCAB, "rows": ROWS, "iterations": ITERS, "arms": {}}
    try:
        for name, force in (("split_topk_greedy", False), ("force_argmax", True)):
            logits = ttnn.from_torch(
                logits_host,
                device=dev,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gen = SamplingGenerator(args=_Args(dev, force), mesh_device=dev, tt_ccl=None)
            gen.reset_sampling_params(format_sampling_params(SamplingParams(temperature=1.0, top_k=1, top_p=0.0), ROWS))
            active = gen.tt_sampling.force_argmax_sampling
            out_buf = ttnn.from_torch(
                torch.zeros(1, 1, 1, ROWS, dtype=torch.int32),
                device=dev,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            eager = gen.sample(logits=logits, tt_out_tok=out_buf, enable_trace=False)
            tok = eager[0] if isinstance(eager, tuple) else eager
            got = [int(v) for v in ttnn.to_torch(tok).reshape(-1).tolist()[:ROWS]]
            correct = sum(int(a == b) for a, b in zip(got, expected))
            eager_ms = bench(dev, lambda: gen.sample(logits=logits, tt_out_tok=out_buf, enable_trace=False))
            gen.capture_trace(logits=logits, tt_out_tok=out_buf, skip_precompile=True)
            traced_ms = bench(dev, lambda: gen.sample(logits=logits, tt_out_tok=out_buf, enable_trace=True))
            results["arms"][name] = {
                "force_argmax_active": bool(active),
                "output_shape": list(tok.shape),
                "matches_torch_argmax": f"{correct}/{ROWS}",
                "eager_ms": round(eager_ms, 3),
                "traced_ms": round(traced_ms, 3),
            }
            print(name, json.dumps(results["arms"][name]))
            gen.reset_trace()
            ttnn.deallocate(out_buf)
            ttnn.deallocate(logits)
    finally:
        ttnn.close_mesh_device(dev)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

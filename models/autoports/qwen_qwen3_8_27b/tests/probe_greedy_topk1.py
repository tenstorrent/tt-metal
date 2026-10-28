# SPDX-License-Identifier: Apache-2.0
"""Compare common split greedy samplers on real selected-policy logits.

This diagnostic does not replace full64 generation/quality/performance gates.
"""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator
from models.common.modules.sampling.sampling_1d import Sampling1D

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--batch", type=int, default=1)
p.add_argument("--replays", type=int, default=200)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(8)
ttnn.CONFIG.throw_exception_on_fallback = True
root = Path(__file__).resolve().parents[1]
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024**2)
result = dict(
    status="running",
    scope="Greedy split sampler only; real layers0/3 logits plus tie/boundary controls; not full64 TSU or quality",
    batch=a.batch,
    tracking=os.getenv("TT_METAL_TRACE_ALLOC_TRACKING"),
    skip_program_cache=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
)
traces = {}
original_sampling = ttnn.sampling


def sample_padded(values, indices, **kw):
    # The pinned backend admits W32 but consumes tile pairs in its top-k
    # kernel. Match the known-working baseline final-sampling width instead.
    # This also includes the preparation cost in the candidate trace.
    if values.shape[-1] < 128:
        padding = [(0, 0), (0, 0), (0, 0), (0, 128 - values.shape[-1])]
        values = ttnn.pad(values, padding, value=float("-inf"))
        indices = ttnn.pad(indices, padding, value=0)
    return original_sampling(values, indices, **kw)


ttnn.sampling = sample_padded
try:
    gen = build_generator(
        root,
        mesh,
        checkpoint=a.checkpoint,
        layer_indices=[0, 3],
        batch_size=a.batch,
        context_length=512,
        prefill_chunk_size=128,
        prepare=False,
    )
    real = gen.prefill_forward(
        torch.full((a.batch, 33), 100, dtype=torch.int64),
        page_table=gen.page_table,
        kv_cache=gen.kv_cache,
        prompt_lens=[33] * a.batch,
        return_device=True,
    )
    logits = ttnn.clone(ttnn.reshape(real, [1, 1, a.batch, 248320 // 4]))
    del real
    k = gen.k
    pval = gen.p
    temp = gen.temp
    samplers = {}
    outputs = {}
    seeds = {}
    for maximum in (32, 1):
        name = f"topk{maximum}"
        sampler = Sampling1D(
            vocab_size=248320,
            mesh_device=mesh,
            tt_ccl=gen.model.layers[0].ccl,
            max_batch_size=32,
            max_top_k=maximum,
            allow_force_argmax=False,
            pad_to_power_of_2=True,
        )
        sampler.load_device_buffers()
        samplers[name] = sampler
        outputs[name] = gen.put(torch.zeros(1, 1, 1, a.batch, dtype=torch.int32), ttnn.uint32)
        seeds[name] = gen.put(torch.arange(32, dtype=torch.int32) + 1234, ttnn.uint32)

    def run(name):
        samplers[name](logits, k=k, p=pval, temp=temp, seeds=seeds[name], tt_out_tok=outputs[name])
        ttnn.plus_one(seeds[name])

    for name in samplers:
        print(f"WARMUP_START {name}", flush=True)
        run(name)
        ttnn.synchronize_device(mesh)
        print(f"WARMUP_DONE {name}", flush=True)

    def read(name):
        return [ttnn.to_torch(shard).reshape(-1).tolist() for shard in ttnn.get_device_tensors(outputs[name])]

    result["real_tokens"] = {name: read(name) for name in samplers}
    assert result["real_tokens"]["topk32"] == result["real_tokens"]["topk1"]
    real_host = torch.cat([ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(logits)], dim=-1)
    real_argmax = real_host.reshape(a.batch, -1).argmax(dim=-1).tolist()
    result["real_cpu_argmax"] = real_argmax
    assert all(replica == real_argmax for replica in result["real_tokens"]["topk1"])
    for name in samplers:
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        run(name)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        traces[name] = trace
    timings = {name: [] for name in samplers}
    for order in [("topk32", "topk1"), ("topk1", "topk32"), ("topk32", "topk1")]:
        for name in order:
            ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=True)
            started = time.monotonic()
            for _ in range(a.replays):
                ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            timings[name].append((time.monotonic() - started) * 1e6 / a.replays)
    result["trace_us"] = timings
    result[
        "sampling_contract"
    ] = "Pad four gathered local winners to128: negative-infinity values/zero indices before common sampling; exact token parity required, including ties"
    result["speed_ratio"] = statistics.median(timings["topk32"]) / statistics.median(timings["topk1"])
    # Refill the exact captured logits without allocating a device buffer.
    controls = []
    for tokens in [
        [0],
        [7],
        [62079],
        [62080],
        [124159],
        [124160],
        [186239],
        [186240],
        [248319],
        [7, 8],
        [62078, 62079],
        [7, 62080],
        [62079, 248319],
    ]:
        host = torch.full((1, 1, a.batch, 248320), -100.0, dtype=torch.bfloat16)
        # Distinct rows expose batch mapping errors when this probe is used
        # beyond B1. Shift within the vocabulary, retaining cross-shard cases.
        expected = []
        for row in range(a.batch):
            row_tokens = [(token + row * 31) % 248320 for token in tokens]
            host[:, :, row, row_tokens] = 100.0
            expected.append(row_tokens)
        source = ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1)
        )
        ttnn.copy_host_to_device_tensor(source, logits)
        actual = {}
        for name in samplers:
            ttnn.execute_trace(mesh, traces[name], cq_id=0, blocking=True)
            actual[name] = read(name)
        controls.append(dict(max_tokens_by_row=expected, outputs=actual))
        assert actual["topk32"] == actual["topk1"], controls[-1]
        assert all(
            len(replica) == a.batch and all(token in expected[row] for row, token in enumerate(replica))
            for replica in actual["topk1"]
        ), controls[-1]
    result.update(
        status="pass", controls=controls, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
except BaseException as exc:
    result.update(status="fail", error=repr(exc))
    raise
finally:
    ttnn.sampling = original_sampling
    for trace in traces.values():
        ttnn.release_trace(mesh, trace)
    ttnn.close_mesh_device(mesh)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)

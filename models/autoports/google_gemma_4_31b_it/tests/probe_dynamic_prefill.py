# SPDX-License-Identifier: Apache-2.0
"""Synthetic target-shape SDPA equivalence; not full-model accuracy evidence.

Run in the serialized device queue with TT_METAL_TRACE_ALLOC_TRACKING=1 and
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0. Checks all TP4 local outputs, dynamic
program reuse, refreshed offsets/table permutations, and repeated trace replay.
"""

import argparse
import json

import torch
from transformers import AutoConfig
from ttnn.types import BlackholeComputeKernelConfig

import ttnn
from models.autoports.google_gemma_4_31b_it.tt.functional_decoder import HF_MODEL, HF_REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5])
    parser.add_argument("--starts", type=int, nargs="+", default=None)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--cache-dtype", choices=["bfloat8_b", "bfloat16"], default="bfloat8_b")
    args = parser.parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(17)
    config = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    starts = args.starts or [0, 1024, 2048, config.max_position_embeddings - args.chunk]
    assert args.chunk > 0 and args.chunk <= 1024 and args.chunk % 32 == 0
    assert all(s >= 0 and s % 1024 == 0 and s + args.chunk <= config.max_position_embeddings for s in starts)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16 * 1024 * 1024)
    try:
        for index in args.layers:
            probe(mesh, config, index, starts, args)
    finally:
        ttnn.close_mesh_device(mesh)


def probe(mesh, config, index, starts, args):
    sliding = config.layer_types[index] == "sliding_attention"
    window = config.sliding_window if sliding else None
    dim = config.head_dim if sliding else config.global_head_dim
    nkv = (config.num_key_value_heads if sliding else config.num_global_key_value_heads) // 4
    nq = config.num_attention_heads // 4
    page_size = 32
    blocks = 64 if sliding else config.max_position_embeddings // page_size

    def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value, device=mesh, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )

    def refresh(value, target):
        host = ttnn.from_torch(
            value, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )
        ttnn.copy_host_to_device_tensor(host, target)

    def read(value):
        return [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(value)]

    # Normalized-scale Q keeps this synthetic check away from saturated softmax.
    q = upload(torch.randn(1, nq, args.chunk, dim, dtype=torch.bfloat16) / dim**0.5)
    cache = tuple(
        upload(torch.randn(blocks, nkv, page_size, dim, dtype=torch.bfloat16), getattr(ttnn, args.cache_dtype))
        for _ in range(2)
    )
    page = upload(torch.arange(blocks, dtype=torch.int32)[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    offset = upload(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    kernel = BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    program = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=128, k_chunk_size=128)

    def setup(start):
        local = min(start, window) if sliding else start
        base = max(0, start - window) // page_size if sliding else 0
        mapping = (torch.arange(blocks, dtype=torch.int32) + base) % blocks
        refresh(mapping[None], page)
        refresh(torch.tensor([local], dtype=torch.int32), offset)
        return local

    def forward(local=None):
        kwargs = {"chunk_start_idx_tensor": offset} if local is None else {"chunk_start_idx": local}
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            *cache,
            page,
            **kwargs,
            scale=1.0,
            sliding_window_size=window,
            program_config=program,
            compute_kernel_config=kernel,
        )

    # Compile all scalar controls before tracking dynamic reuse or retaining a trace.
    expected = {}
    for start in starts:
        expected[start] = read(forward(setup(start)))
    counts = []
    for start in starts:
        setup(start)
        actual = forward()
        check(read(actual), expected[start], index, start, "eager")
        del actual
        counts.append(mesh.num_program_cache_entries())
    assert all(count == counts[0] for count in counts), counts
    setup(starts[0])
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    output = forward()
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        for start in starts + list(reversed(starts)):
            setup(start)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            check(read(output), expected[start], index, start, "trace")
    finally:
        ttnn.release_trace(mesh, trace)
    print(
        json.dumps(
            {
                "layer": index,
                "kind": config.layer_types[index],
                "q_shape": list(q.shape),
                "cache_shape": list(cache[0].shape),
                "page_width": blocks,
                "cache_dtype": args.cache_dtype,
                "dynamic_cache_counts": counts,
                "scope": "synthetic SDPA boundary only",
            }
        ),
        flush=True,
    )


def check(actual, expected, layer, start, mode):
    pccs = [float(torch.corrcoef(torch.stack((a.flatten(), b.flatten())))[0, 1]) for a, b in zip(actual, expected)]
    max_errors = [float((a - b).abs().max()) for a, b in zip(actual, expected)]
    print(
        json.dumps(
            {"layer": layer, "start": start, "mode": mode, "pcc_per_device": pccs, "max_abs_per_device": max_errors}
        ),
        flush=True,
    )
    assert min(pccs) >= 0.999, (layer, start, mode, pccs)


if __name__ == "__main__":
    main()

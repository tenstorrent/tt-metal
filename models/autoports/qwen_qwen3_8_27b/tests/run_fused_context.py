# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-capacity fused/functional comparison using the validated baseline inputs.

The prior-stage HF correlation and measured fused/functional correlation also
provide a conservative HF correlation bound by the angle triangle inequality.
No HF computation is replaced inside either decoder's measured forward.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.reference import load_config, load_layer_weights
from models.autoports.qwen_qwen3_8_27b.tests.run_decoder import device_only, pcc
from models.autoports.qwen_qwen3_8_27b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_27b.tt.fused_decoder import FusedDecoder


def lower_bound(a, b):
    a, b = min(a, 1.0), min(b, 1.0)
    return a * b - math.sqrt(max(0.0, (1 - a * a) * (1 - b * b)))


def case(length, config, mesh, decoders, baseline):
    torch.manual_seed(123 + length)
    x = (torch.randn(1, length + 1, config.hidden_size) * 0.1).bfloat16()
    cos, sin = Qwen3_5TextRotaryEmbedding(config)(x, torch.arange(length + 1).unsqueeze(0))

    def upload(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    # Match run_decoder.py's random page permutation exactly after creating x.
    pages = (length + 1 + 31) // 32
    permutation = torch.randperm(pages + 3)
    table = upload(permutation[:pages].reshape(1, pages).int(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    dx, cc, ss = upload(x[:, :-1]), upload(cos[:, :-1]), upload(sin[:, :-1])
    states, outputs = [], []
    for decoder in decoders:
        state = decoder.allocate_state(batch_size=1, num_pages=pages + 3)
        states.append(state)
        print(type(decoder).__name__, "PREFILL_BEGIN", length, flush=True)
        with device_only():
            out = decoder.prefill_forward(dx, state=state, page_table=table, cos=cc, sin=ss)
        outputs.append(ttnn.to_torch(out))
        ttnn.deallocate(out)
        print(type(decoder).__name__, "PREFILL_END", length, flush=True)
    correlation = pcc(*outputs)
    metrics = {
        "length": length,
        "batch": 1,
        "kind": decoders[0].kind,
        "prefill_equivalence_pcc": correlation,
        "functional_hf_prefill_pcc": baseline["prefill_pcc"],
        "fused_hf_prefill_pcc_lower_bound": lower_bound(correlation, baseline["prefill_pcc"]),
    }
    assert correlation >= 0.995
    assert metrics["fused_hf_prefill_pcc_lower_bound"] >= 0.995, metrics
    del outputs
    if length < config.max_position_embeddings:
        nx, nc, ns = upload(x[:, -1:]), upload(cos[:, -1:]), upload(sin[:, -1:])
        pos = upload(torch.tensor([length], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        decoded = []
        for decoder, state in zip(decoders, states):
            snapshots = {k: ttnn.clone(v) for k, v in vars(state).items() if v is not None}

            def restore():
                for key, value in snapshots.items():
                    ttnn.copy(value, getattr(state, key))

            def decode():
                with device_only():
                    return decoder.decode_forward(nx, state=state, page_table=table, current_pos=pos, cos=nc, sin=ns)

            warm = decode()
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(warm)
            restore()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = decode()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            try:
                restore()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                result = ttnn.to_torch(out)
                for _ in range(3):
                    restore()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    assert torch.equal(result, ttnn.to_torch(out))
                decoded.append(result)
            finally:
                ttnn.release_trace(mesh, trace)
        correlation = pcc(*decoded)
        metrics.update(
            traced_decode_equivalence_pcc=correlation,
            functional_hf_decode_pcc=baseline["traced_decode_pcc"],
            fused_hf_decode_pcc_lower_bound=lower_bound(correlation, baseline["traced_decode_pcc"]),
            repeat_bitwise_equal=True,
        )
        assert correlation >= 0.995
        assert metrics["fused_hf_decode_pcc_lower_bound"] >= 0.995, metrics
    print(json.dumps(metrics), flush=True)
    return metrics


def run(args):
    torch.set_num_threads(4)
    config = load_config(args.snapshot)
    weights = load_layer_weights(args.snapshot, args.layer)
    baseline = {row["length"]: row for row in json.loads(args.baseline.read_text())}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoders = [
            cls.from_state_dict(weights, hf_config=config, layer_idx=args.layer, mesh_device=mesh)
            for cls in (FunctionalDecoder, FusedDecoder)
        ]
        results = []
        for length in map(int, args.lengths.split(",")):
            result = case(length, config, mesh, decoders, baseline[length])
            result["functional_hf_evidence"] = str(args.baseline)
            result["functional_hf_evidence_sha256"] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
            result["source_sha256"] = {
                name: hashlib.sha256((Path(__file__).parents[1] / "tt" / name).read_bytes()).hexdigest()
                for name in ("functional_decoder.py", "fused_decoder.py")
            }
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(0, 3), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--lengths", default="4097,262143,262144,31")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())

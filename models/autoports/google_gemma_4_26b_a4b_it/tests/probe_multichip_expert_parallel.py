# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""EP4 sparse-mask correctness, zero-rank, and changing-route trace probe."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import _ExpertParallelExperts
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.ccl import CCLManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path(__file__).parents[1]
    config = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    hf = load_layer(config, args.layer, True)
    fixture = torch.load(root / f"doc/optimized_decoder/actual_text_layer{args.layer}_4096_128.pt", weights_only=True)
    sliding = config.layer_types[args.layer] == "sliding_attention"
    with torch.no_grad():
        decode_x = hf.pre_feedforward_layernorm_2(fixture["decode"][:, :1]).unsqueeze(0).bfloat16()
        prefill_x = hf.pre_feedforward_layernorm_2(fixture["prefill"][:, :32]).unsqueeze(0).bfloat16()

    def routes(ids):
        ids = torch.tensor(ids).reshape(-1, 8)
        weights = torch.arange(1, 9).float() / 36
        result = torch.zeros(1, 1, len(ids), 128, dtype=torch.bfloat16)
        result[0, 0].scatter_(1, ids, weights.bfloat16().expand(len(ids), -1))
        return result

    decode_cases = [(f"rank{rank}_eight", routes([list(range(rank * 32, rank * 32 + 8))])) for rank in range(4)]
    decode_cases += [
        ("balanced", routes([[0, 1, 32, 33, 64, 65, 96, 97]])),
        ("mixed", routes([[0, 32, 64, 96, 97, 98, 99, 100]])),
        ("rank0_eight_again", routes([list(range(8))])),
    ]
    prefill_cases = [
        ("prefill_empty_ranks", routes([[(row + i) % 32 for i in range(8)] for row in range(32)])),
        ("prefill_union", routes([[(row * 4 + i) % 128 for i in range(8)] for row in range(32)])),
    ]
    print("CPU_FIXTURES_READY", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16777216)
    records = []
    try:
        experts = _ExpertParallelExperts(hf.state_dict(), config, mesh, sliding)
        mesh_config = MeshConfig(mesh.shape, decode=ModeConfig(tp=4))
        ccl = CCLManager(mesh, 1, ttnn.Topology.Linear)
        mapper = ttnn.ReplicateTensorToMesh(mesh)

        def upload(value):
            return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

        def read_parts(value):
            return [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(value)]

        # Dequantized selected weight policy is the CPU oracle; no CPU work is
        # used by the device forward. Expert matmuls below remain mask-driven.
        gates = [ttnn.to_torch(part).squeeze(0) for part in ttnn.get_device_tensors(experts.gate_up)]
        downs = [ttnn.to_torch(part).squeeze(0) for part in ttnn.get_device_tensors(experts.down)]
        print("EP_WEIGHTS_READY", [tuple(part.shape) for part in gates], flush=True)

        def reference(x, routing, decode):
            if decode and experts.decode_activation_dtype is not None:
                dx = ttnn.typecast(upload(x), experts.decode_activation_dtype)
                x = read_parts(dx)[0]
            x = x.squeeze(0).squeeze(0).float()
            refs = []
            for rank in range(4):
                weights = routing[0, 0, :, rank * 32 : (rank + 1) * 32].float()
                result = torch.zeros(x.shape[0], config.hidden_size)
                for expert in torch.nonzero(weights.abs().sum(0), as_tuple=False).flatten().tolist():
                    gu = (x @ gates[rank][expert].float()).bfloat16().float()
                    gate, up = gu.split(704, dim=-1)
                    hidden = (torch.nn.functional.gelu(gate, approximate="none") * up).bfloat16().float()
                    out = (hidden @ downs[rank][expert].float()).bfloat16().float()
                    result += out * weights[:, expert, None]
                refs.append(result.reshape(1, 1, -1, config.hidden_size))
            return refs

        def verify(name, routing, local, reduced, expected, trace=False):
            actual = read_parts(local)
            summed = read_parts(reduced)
            assert all(torch.equal(summed[0], part) for part in summed[1:]), "collective replicas differ"
            counts = [int((routing[0, 0, :, r * 32 : (r + 1) * 32].abs().sum(0) != 0).sum()) for r in range(4)]
            pccs = []
            for count, value, ref in zip(counts, actual, expected):
                assert torch.isfinite(value).all()
                if count == 0:
                    assert torch.count_nonzero(value) == 0, "empty partition contains stale output"
                else:
                    pcc = float(torch.corrcoef(torch.stack((value.flatten().double(), ref.flatten().double())))[0, 1])
                    assert pcc >= 0.995, (name, count, pcc)
                    pccs.append(pcc)
            total = sum(expected)
            total_pcc = float(
                torch.corrcoef(torch.stack((summed[0].flatten().double(), total.flatten().double())))[0, 1]
            )
            assert total_pcc >= 0.995, (name, total_pcc)
            record = dict(
                name=name,
                local_union_counts=counts,
                local_pcc=pccs,
                total_pcc=total_pcc,
                zero_partitions_exact=True,
                trace=trace,
            )
            records.append(record)
            print("PASS", record, flush=True)

        token = upload(decode_x)
        routing = upload(decode_cases[0][1])

        def forward(x=token, route=routing):
            with device_only():
                local = experts(x, route)
                reduced = mesh_config.allreduce(ttnn.to_memory_config(local, ttnn.DRAM_MEMORY_CONFIG), ccl, axis=1)
            return local, reduced

        eager = {}
        expected = {}
        for name, value in decode_cases:
            host = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(host, routing)
            local, reduced = forward()
            expected[name] = reference(decode_x, value, True)
            verify(name, value, local, reduced, expected[name])
            eager[name] = read_parts(reduced)[0]
        # Warm twice so the CCL manager's ping-pong semaphore signatures exist.
        forward()
        forward()
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        local, reduced = forward()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        try:
            for name, value in decode_cases:
                host = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(host, routing)
                ttnn.synchronize_device(mesh)
                samples = []
                for _ in range(args.samples):
                    before = time.perf_counter()
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    samples.append((time.perf_counter() - before) * 1e6)
                verify(name, value, local, reduced, expected[name], trace=True)
                assert torch.equal(eager[name], read_parts(reduced)[0]), "trace differs from eager"
                records[-1]["host_us"] = samples
        finally:
            ttnn.release_trace(mesh, trace_id)
        for name, value in prefill_cases:
            local, reduced = forward(upload(prefill_x), upload(value))
            verify(name, value, local, reduced, reference(prefill_x, value, False))
    finally:
        ttnn.close_mesh_device(mesh)
    args.output.write_text(json.dumps(dict(layer=args.layer, passed=True, records=records), indent=2) + "\n")
    print("EP_PROBE_PASS", flush=True)


if __name__ == "__main__":
    main()

"""Alternating reader-count microprofiles on real decoder operands; not layer telemetry."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from tracy import signpost
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.optimized_trace import capture
from models.autoports.google_gemma_4_31b_it.tests.run_optimized_decoder import pcc, weights
from models.autoports.google_gemma_4_31b_it.tt.multichip_decoder import MultichipDecoder
from models.autoports.google_gemma_4_31b_it.tt.optimized_decoder import HF_MODEL, HF_REVISION

p = argparse.ArgumentParser()
p.add_argument("--layer", type=int, required=True)
p.add_argument("--result", required=True)
a = p.parse_args()
torch.set_num_threads(16)
c = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
fixture = Path(f"bringup/artifacts/optimized_decoder_layer{a.layer}_inputs2050.pt")
metadata = json.loads(fixture.with_suffix(".pt.json").read_text())
assert (metadata["model"], metadata["revision"], metadata["layer"]) == (HF_MODEL, HF_REVISION, a.layer)
x = torch.load(f"bringup/artifacts/optimized_decoder_layer{a.layer}_inputs2050.pt", weights_only=True, mmap=True)[
    :, :34
]
cos, sin = Gemma4TextRotaryEmbedding(c)(x, torch.arange(34)[None], layer_type=c.layer_types[a.layer])
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
traces = []
try:

    def tt(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        h = ttnn.from_torch(x.contiguous(), dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        return ttnn.to_device(h, mesh)

    model = MultichipDecoder.from_state_dict(weights(a.layer), hf_config=c, layer_idx=a.layer, mesh_device=mesh)
    cache = model.allocate_cache(num_blocks=3)
    page = tt(torch.tensor([[0, 1]], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    warm = model.prefill_forward(
        tt(x[:, :33].unsqueeze(1)),
        cos=tt(cos[:, :33].unsqueeze(1)),
        sin=tt(sin[:, :33].unsqueeze(1)),
        kv_cache=cache,
        page_table=page,
    )
    del warm
    actual = {}
    linear = model._linear

    def record(value, name):
        if model.decoding:
            actual[name] = value
        return linear(value, name)

    model._linear = record
    warm = model.decode_forward(
        tt(x[:, 33:].reshape(1, 1, 1, -1)),
        cos=tt(cos[:, 33:].reshape(1, 1, 1, -1)),
        sin=tt(sin[:, 33:].reshape(1, 1, 1, -1)),
        kv_cache=cache,
        page_table=page,
        current_pos=tt(torch.tensor([33], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
    )
    ttnn.synchronize_device(mesh)
    del warm
    cases = {}
    for role, source in actual.items():
        cfg = model.geometry[role]
        inp = ttnn.to_memory_config(source, model._working_memory(source, cfg["cores"]))
        w = model.weights[role + ".weight"]
        bank_tiles = w.memory_config().shard_spec.shape[1] // 32
        for readers in (1, 2, 3):
            if bank_tiles % readers:
                continue
            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=cfg["block"],
                per_core_M=1,
                per_core_N=math.ceil(w.shape[-1] / 32 / cfg["cores"]),
                num_workers_per_dram_bank=readers,
            )
            kernel = model.linear_kernels[
                "attention" if role.startswith("self_attn.") else "down" if role == "mlp.down_proj" else "mlp"
            ]
            cases[(role, readers)] = {
                "input": inp,
                "weight": w,
                "pc": pc,
                "kernel": kernel,
                "bank_tiles": bank_tiles,
                "geometry": cfg,
            }

    def forward(case):
        return ttnn.linear(
            case["input"],
            case["weight"],
            program_config=case["pc"],
            compute_kernel_config=case["kernel"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    references = {}
    for (role, readers), case in cases.items():
        out = forward(case)
        values = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
        if readers == 1:
            references[role] = values
        case["pcc"] = min(pcc(v, r) for v, r in zip(values, references[role]))
        assert case["pcc"] >= 0.999
        del out
    for key, case in cases.items():
        trace, out = capture(mesh, lambda case=case: forward(case))
        traces.append(trace)
        case.update(trace=trace, output=out, host_us=[])
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    for repeat in range(4):
        order = list(cases)
        if repeat % 2:
            order.reverse()
        for role, readers in order:
            case = cases[(role, readers)]
            label = f"READER_{role}_{readers}_{repeat}"
            signpost(label)
            start = time.perf_counter()
            for _ in range(8):
                ttnn.execute_trace(mesh, case["trace"], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            case["host_us"].append((time.perf_counter() - start) * 1e6 / 8)
            signpost(label + "_END")
            ttnn.ReadDeviceProfiler(mesh)
            values = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(case["output"])]
            traced_pcc = min(pcc(v, r) for v, r in zip(values, references[role]))
            assert traced_pcc >= 0.999, (role, readers, traced_pcc)
            case["pcc"] = min(case["pcc"], traced_pcc)
    records = []
    for (role, readers), case in cases.items():
        k, n = tuple(case["weight"].shape)[-2:]
        records.append(
            dict(
                role=role,
                readers=readers,
                shape=[32, k, n],
                logical_batch=1,
                stored_weight_dtype=str(case["weight"].dtype),
                fidelity="LoFi",
                pcc=case["pcc"],
                host_us=case["host_us"],
                geometry=case["geometry"],
                input_memory=str(case["input"].memory_config()),
                weight_memory=str(case["weight"].memory_config()),
                per_bank_tiles=case["bank_tiles"],
                per_reader_physical_row_bytes=case["bank_tiles"] // readers * 576,
                physical_weight_bytes=k // 32 * case["bank_tiles"] * 8 * 576,
                logical_weight_bytes=k * n * 0.5,
                peak_dram_bytes_per_s=512e9,
            )
        )
    Path(a.result).write_text(
        json.dumps(
            {
                "layer": a.layer,
                "source": "real decoder projection operands, isolated matmul; not headline layer performance",
                "alternating_order": True,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
finally:
    for trace in reversed(traces):
        ttnn.release_trace(mesh, trace)
    ttnn.close_mesh_device(mesh)

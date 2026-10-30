"""Isolated alternating reader-count experiment using real decoder inputs."""

import argparse
import json
import math
from pathlib import Path

import torch
from tracy import signpost
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn

from ..tt.optimized_decoder import HF_MODEL, HF_REVISION, OptimizedDecoder
from .optimized_trace import capture
from .run_optimized_decoder import pcc, weights


def run(layer, result):
    torch.set_num_threads(16)
    cfg = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    x = torch.load(f"bringup/artifacts/optimized_decoder_layer{layer}_inputs.pt", weights_only=True)
    cos, sin = Gemma4TextRotaryEmbedding(cfg)(x, torch.arange(34)[None], layer_type=cfg.layer_types[layer])
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    traces = []

    def tt(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)

    try:
        model = OptimizedDecoder.from_state_dict(weights(layer), hf_config=cfg, layer_idx=layer, mesh_device=mesh)
        page = tt(torch.tensor([[0, 1]], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cache = model.allocate_cache(num_blocks=2)
        pre = model.prefill_forward(
            tt(x[:, :33, None].transpose(1, 2)),
            cos=tt(cos[:, :33, None].transpose(1, 2)),
            sin=tt(sin[:, :33, None].transpose(1, 2)),
            kv_cache=cache,
            page_table=page,
        )
        del pre
        snapshots = {}
        original = model._linear

        def snapshot(value, name):
            snapshots[name] = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
            return original(value, name)

        model._linear = snapshot
        out = model.decode_forward(
            tt(x[:, 33].reshape(1, 1, 1, -1)),
            cos=tt(cos[:, 33].reshape(1, 1, 1, -1)),
            sin=tt(sin[:, 33].reshape(1, 1, 1, -1)),
            kv_cache=cache,
            page_table=page,
            current_pos=tt(torch.tensor([33], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        )
        del out, cache
        model._linear = original
        records = []
        for role_id, (name, source) in enumerate(snapshots.items()):
            geometry = model.geometry[name]
            weight = model.weights[name + ".weight"]
            k, n = tuple(weight.shape)[-2:]
            activation = ttnn.to_memory_config(source, model._working_memory(source, geometry["cores"]))
            group = "attention" if name.startswith("self_attn.") else "down" if name == "mlp.down_proj" else "mlp"
            legal = [r for r in (1, 2, 3) if (n // 32 // model.banks) % r == 0]
            variants = {}
            gold = None
            for readers in legal:
                config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=geometry["block"],
                    per_core_M=1,
                    per_core_N=math.ceil(n / 32 / geometry["cores"]),
                    num_workers_per_dram_bank=readers,
                )

                def forward(config=config):
                    return ttnn.linear(
                        activation,
                        weight,
                        program_config=config,
                        compute_kernel_config=model.linear_kernels[group],
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16,
                    )

                warm = forward()
                actual = ttnn.to_torch(warm)
                del warm
                if gold is None:
                    gold = actual
                score = pcc(actual, gold)
                assert score >= 0.9999
                tid, output = capture(mesh, forward)
                traces.append(tid)
                ttnn.execute_trace(mesh, tid, blocking=True)
                variants[readers] = (tid, output, actual)
                records.append(
                    dict(
                        role=name,
                        role_id=role_id,
                        readers=readers,
                        k=k,
                        n=n,
                        logical_n=model.hidden if name in model.output_padding else n,
                        weight_dtype="BFLOAT4_B",
                        fidelity="LoFi",
                        input_cores=geometry["cores"],
                        in0_block_w=geometry["block"],
                        bank_tiles=n // 32 // model.banks,
                        per_reader_row_bytes=n // 32 // model.banks // readers * 576,
                        pcc_vs_one_reader=score,
                        illegal_readers=[r for r in (1, 2, 3) if r not in legal],
                        illegal_reason="bank shard tile width not divisible by reader count",
                    )
                )
            ttnn.ReadDeviceProfiler(mesh)
            for trial in range(6):
                for readers in legal if trial % 2 == 0 else list(reversed(legal)):
                    tid, output, reference = variants[readers]
                    label = f"READER_{role_id}_{readers}_{trial}"
                    signpost(label)
                    for _ in range(16):
                        ttnn.execute_trace(mesh, tid, blocking=False)
                    ttnn.synchronize_device(mesh)
                    signpost(label + "_END")
                    assert torch.equal(ttnn.to_torch(output), reference), "Repeated reader output changed"
                    ttnn.ReadDeviceProfiler(mesh)
            for tid, _, _ in variants.values():
                ttnn.release_trace(mesh, tid)
                traces.remove(tid)
            del variants, activation
        Path(result).write_text(
            json.dumps(
                dict(
                    layer_type=model.layer_type,
                    records=records,
                    alternating_trials=6,
                    replays_per_trial=16,
                    deterministic=True,
                    source="real pinned HF decoder activation snapshots",
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        for tid in traces:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--result", required=True)
    a = p.parse_args()
    run(a.layer, a.result)

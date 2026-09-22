# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision-locked per-role DRAM geometry sweep using recorded HF activations."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.reference import load_config, load_layer_weights, make_reference
from models.autoports.qwen_qwen3_8_27b.tests.run_optimized_decoder import device_only, pcc
from models.autoports.qwen_qwen3_8_27b.tests.sweep_optimized_decoder import BASE
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import OptimizedDecoder


def run(a):
    torch.set_num_threads(4)
    config = load_config(a.snapshot)
    weights = load_layer_weights(a.snapshot, a.layer)
    ref = make_reference(config, a.layer, weights)
    x = torch.load(a.activations / f"layer{a.layer}.pt", weights_only=True)[:1, :128]
    captured = {}
    handles = []
    for name, module in ref.named_modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(
                module.register_forward_pre_hook(
                    lambda module, inputs, name=name: captured.__setitem__(name, inputs[0][:, -1:].clone())
                )
            )
    rope = Qwen3_5TextRotaryEmbedding(config)
    cos, sin = rope(x, torch.arange(128)[None])
    mask = torch.full((128, 128), float("-inf")).triu(1).bfloat16()[None, None] if a.layer == 3 else None
    with torch.no_grad():
        ref(x, position_embeddings=(cos, sin), attention_mask=mask, past_key_values=DynamicCache(config=config))
    for h in handles:
        h.remove()
    del ref
    source = "self_attn.q_proj" if a.layer == 3 else "linear_attn.in_proj_qkv"
    packed = "self_attn.qkvg" if a.layer == 3 else "linear_attn.packed"
    captured[packed] = captured[source]
    names = [
        packed,
        "self_attn.o_proj" if a.layer == 3 else "linear_attn.out_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    results = []
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for readers in [1, 2, 3]:
            policy = (
                BASE | {"packed_mlp": True} | {r + "_readers": readers for r in ("attention", "gate", "up", "down")}
            )
            decoder = OptimizedDecoder.from_state_dict(
                weights, hf_config=config, layer_idx=a.layer, mesh_device=mesh, policy=policy
            )
            captured["mlp.gate_up"] = captured["mlp.gate_proj"]
            for name in names + ["mlp.gate_up"]:
                if a.roles and name not in a.roles.split(","):
                    continue
                role = decoder._role(name)
                inp = captured[name].contiguous()
                device_x = ttnn.from_torch(
                    inp,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                quantized_weight = ttnn.to_torch(decoder.weights[name + ".weight"])
                expected = inp.float() @ quantized_weight.float()
                activation = "silu" if name == "mlp.gate_proj" else None
                if activation:
                    expected = torch.nn.functional.silu(expected)
                k, n = quantized_weight.shape
                geometries = {5120: [5, 10, 20, 40, 80], 6144: [6, 12, 24, 48, 96], 17408: [8, 16, 32, 64]}[k]
                for cores in geometries:
                    shard_tiles = (k // 32 + cores - 1) // cores
                    blocks = [
                        v
                        for v in [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 17, 24, 32, 34, 48, 68]
                        if shard_tiles % v == 0 and (k // 32) % v == 0
                    ]
                    if not blocks:
                        blocks = [1]
                    for block in blocks:
                        decoder.policy[role + "_cores"] = cores
                        decoder.policy[role + "_block"] = block
                        row = dict(
                            name=name,
                            readers=readers,
                            cores=cores,
                            block=block,
                            input_shard_tiles=shard_tiles,
                            k=k,
                            n=n,
                            weight_dtype=str(decoder.dram_weights[name + ".weight"].dtype),
                            fidelity="LoFi",
                            layer=a.layer,
                        )
                        trace = None
                        try:
                            with device_only():
                                out = decoder._linear(device_x, name, activation=activation)
                            ttnn.synchronize_device(mesh)
                            row["pcc_quantized_control"] = pcc(expected, ttnn.to_torch(out))
                            ttnn.deallocate(out)
                            with device_only():
                                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                                out = decoder._linear(device_x, name, activation=activation)
                                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                            samples = []
                            for _ in range(9):
                                ttnn.synchronize_device(mesh)
                                start = time.perf_counter()
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                                ttnn.synchronize_device(mesh)
                                samples.append((time.perf_counter() - start) * 1e6)
                            row["traced_us"] = statistics.median(samples)
                            row["samples_us"] = samples
                            row["status"] = "pass" if row["pcc_quantized_control"] >= 0.995 else "numeric_diagnostic"
                            row["weight_memory"] = str(decoder.dram_weights[name + ".weight"].memory_config())
                        except RuntimeError as e:
                            row["status"] = "runtime_error"
                            row["error"] = str(e).split("backtrace:")[0]
                        finally:
                            if trace is not None:
                                ttnn.release_trace(mesh, trace)
                            if "out" in locals() and out.is_allocated():
                                ttnn.deallocate(out)
                            gc.collect()
                        results.append(row)
                        a.output.write_text(json.dumps(results, indent=2) + "\n")
                        print(
                            json.dumps({k: v for k, v in row.items() if k not in ("samples_us", "weight_memory")}),
                            flush=True,
                        )
                ttnn.deallocate(device_x)
            del decoder
            gc.collect()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--activations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--roles")
    run(p.parse_args())

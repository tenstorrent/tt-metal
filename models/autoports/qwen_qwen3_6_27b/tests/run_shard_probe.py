# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""First-divergence diagnostic: DRAM weight storage round trip and prefill projection."""
import json
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import MODEL, REVISION, load_layer, pcc
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder


def main():
    torch.set_num_threads(8)
    torch.manual_seed(17)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    results = []
    try:
        layer = OptimizedDecoder.from_state_dict(
            load_layer(0),
            hf_config=config,
            layer_idx=0,
            mesh_device=mesh,
            policy={"dram_sharded": True, "readers": 2, "prefill_grid": [11, 10]},
        )
        read = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        for key in layer.projection_info:
            weight = layer.weights[key]
            k, n = weight.shape
            x = torch.randn(1, 512, k, dtype=torch.bfloat16) * 0.1
            w = read(weight)
            expected = x.float() @ w.float()
            actual = read(layer._linear(layer._upload(x), key))
            entry = {"key": key, "shape": [512, k, n], "pcc": pcc(expected, actual)}
            # Explicit physical padding makes the logical N agree with each bank's stored extent.
            physical_n = weight.memory_config().shard_spec.shape[1] * mesh.dram_grid_size().x
            padded = torch.nn.functional.pad(w, (0, physical_n - n))
            layer.weights[key] = ttnn.from_torch(
                padded, device=mesh, dtype=weight.dtype, layout=ttnn.TILE_LAYOUT, memory_config=weight.memory_config()
            )
            fixed = read(layer._linear(layer._upload(x), key))[..., :n]
            entry["physical_n"] = physical_n
            entry["explicit_pad_pcc"] = pcc(expected, fixed)
            results.append(entry)
            print(entry, flush=True)
        Path("models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/shard_probe.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

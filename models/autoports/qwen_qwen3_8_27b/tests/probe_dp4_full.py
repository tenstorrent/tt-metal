# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full autoport DP4 prefill feasibility probe; not a serving implementation.

Replicated weights, four disjoint request shards, no inter-chip model collectives.
Decode-only DRAM weight copies are omitted to fit full replicas. This intentionally
measures prefill, not a claim about end-to-end or decode performance.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.model import Checkpoint, checkpoint_path
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import DEFAULT_POLICY, OptimizedDecoder
from models.autoports.qwen_qwen3_8_27b.tt.precision import decoder_policy, load_precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lengths", default="4096,32768")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch % 4:
        parser.error("Batch must be divisible by four")
    lengths = list(map(int, args.lengths.split(",")))
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(checkpoint_path(), local_files_only=True).text_config
    weights = Checkpoint(checkpoint_path())
    precision = load_precision()
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    report = dict(batch=args.batch, local_batch=args.batch // 4, layout="dp4", rows=[])
    try:

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=False):
            return ttnn.from_torch(
                x.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0) if shard else ttnn.ReplicateTensorToMesh(mesh),
            )

        layers = []
        for index in range(config.num_hidden_layers):
            layer = OptimizedDecoder.from_state_dict(
                weights.layer(index),
                hf_config=config,
                layer_idx=index,
                mesh_device=mesh,
                policy={
                    **DEFAULT_POLICY,
                    **decoder_policy(precision, index),
                    "chunk_size": 4096,
                    "packed_mlp": True,
                    "dram": False,
                },
                replicated_mesh=True,
            )
            # Packed gate/up is the only selected MLP projection path.
            for name in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                ttnn.deallocate(layer.weights.pop(name))
            layers.append(layer)
            print("DP4_LOAD_LAYER", index, flush=True)
        embedding = upload(weights.tensor("model.language_model.embed_tokens.weight"), layout=ttnn.ROW_MAJOR_LAYOUT)
        norm = upload((weights.tensor("model.language_model.norm.weight").float() + 1).reshape(1, 1, -1))
        head = upload(weights.tensor("lm_head.weight").T, dtype=getattr(ttnn, precision["weight_groups"]["head"]))
        compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        norm_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        batch = args.batch // 4
        pages = (max(lengths) + 31) // 32
        table = upload(
            torch.arange(batch * pages, dtype=torch.int32).reshape(batch, pages), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        )
        states = [layer.allocate_state(batch_size=batch, num_pages=batch * pages) for layer in layers]
        rotary = Qwen3_5TextRotaryEmbedding(config)
        print("DP4_MODEL_READY", flush=True)
        for length in lengths:
            tokens = (torch.arange(length) % 256 + 100).repeat(args.batch, 1) + torch.arange(args.batch)[:, None] * 13
            cos, sin = rotary(
                torch.empty(batch, length, 1, dtype=torch.bfloat16), torch.arange(length)[None].expand(batch, -1)
            )
            for repeat in range(2):
                for state in states:
                    for tensor in vars(state).values():
                        if tensor is not None:
                            ttnn.copy(ttnn.zeros_like(tensor), tensor)
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                for start in range(0, length, 4096):
                    end = min(start + 4096, length)
                    ids = upload(tokens[:, start:end].int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, shard=True)
                    cc, ss = [upload(t[:, start:end]) for t in (cos, sin)]
                    x = ttnn.embedding(ids, embedding, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    x = ttnn.reshape(x, [batch, end - start, config.hidden_size])
                    for layer, state in zip(layers, states):
                        x = layer.prefill_forward(x, state=state, start_pos=start, page_table=table, cos=cc, sin=ss)
                    if end == length:
                        hidden = ttnn.rms_norm(
                            x[:, -1:, :],
                            weight=norm,
                            epsilon=config.rms_norm_eps,
                            compute_kernel_config=norm_compute,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        logits = ttnn.linear(
                            hidden,
                            head,
                            compute_kernel_config=compute,
                            core_grid=ttnn.CoreGrid(y=8, x=8),
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    del x, ids, cc, ss
                ttnn.synchronize_device(mesh)
                seconds = time.perf_counter() - begin
                host = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
                row = dict(
                    length=length,
                    repeat=repeat,
                    prefill_s=seconds,
                    first_tokens=host.argmax(-1).reshape(-1).tolist(),
                    finite=bool(torch.isfinite(host).all()),
                )
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("DP4_PREFILL", json.dumps(row), flush=True)
                del logits, hidden, host
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

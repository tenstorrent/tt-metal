# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real four-chip TP4 versus DP4 layer experiment, not an HTTP/full-model result.

DP4 replicates weights on all four chips and shards distinct requests across them.
All chips execute concurrently. Scope: layers 0 and 3, with isolated fresh state.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.model import Checkpoint, checkpoint_path
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import DEFAULT_POLICY, OptimizedDecoder
from models.autoports.qwen_qwen3_8_27b.tt.precision import decoder_policy, load_precision
from models.common.modules.tt_ccl import TT_CCL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--modes", default="tp4,tp4_sharded,dp4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch % 4:
        parser.error("Global batch must be divisible by four")
    torch.set_num_threads(8)
    torch.manual_seed(23)
    config = AutoConfig.from_pretrained(checkpoint_path(), local_files_only=True).text_config
    weights = Checkpoint(checkpoint_path())
    precision = load_precision()
    host_x = (torch.randn(args.batch, args.length, config.hidden_size) * 0.1).bfloat16()
    cos, sin = Qwen3_5TextRotaryEmbedding(config)(host_x, torch.arange(args.length)[None].expand(args.batch, -1))
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    ccl = TT_CCL(mesh)
    report = dict(batch=args.batch, length=args.length, layers=[0, 3], rows=[])
    reference = None
    originals = []
    try:
        for mode in args.modes.split(","):
            if mode not in ("tp4", "tp4_sharded", "dp4"):
                raise ValueError(mode)
            local_batch = args.batch // 4 if mode == "dp4" else args.batch

            def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard_dim=None):
                return ttnn.from_torch(
                    x.contiguous(),
                    dtype=dtype,
                    layout=layout,
                    device=mesh,
                    mesh_mapper=(
                        ttnn.ShardTensorToMesh(mesh, dim=shard_dim)
                        if shard_dim is not None
                        else ttnn.ReplicateTensorToMesh(mesh)
                    ),
                )

            layer_list = []
            for index in (0, 3):
                policy = decoder_policy(precision, index)
                if mode.startswith("tp4"):
                    if mode == "tp4_sharded":
                        policy["residual_layout"] = "sharded"
                    layer = MultichipDecoder.from_state_dict(
                        weights.layer(index),
                        hf_config=config,
                        layer_idx=index,
                        mesh_device=mesh,
                        policy=policy,
                        ccl=ccl,
                    )
                else:
                    layer = OptimizedDecoder.from_state_dict(
                        weights.layer(index),
                        hf_config=config,
                        layer_idx=index,
                        mesh_device=mesh,
                        policy={**DEFAULT_POLICY, **policy, "chunk_size": 4096, "packed_mlp": True},
                        replicated_mesh=True,
                    )
                layer_list.append(layer)
            pages = (args.length + 31) // 32
            table = upload(
                torch.arange(local_batch * pages, dtype=torch.int32).reshape(local_batch, pages),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            states = [x.allocate_state(batch_size=local_batch, num_pages=local_batch * pages) for x in layer_list]
            x = upload(host_x, shard_dim=0 if mode == "dp4" else -1 if mode == "tp4_sharded" else None)
            cc, ss = [upload(v, shard_dim=0 if mode == "dp4" else None) for v in (cos, sin)]
            zero_states = [{k: ttnn.zeros_like(v) for k, v in vars(s).items() if v is not None} for s in states]

            def reset():
                for state, zero in zip(states, zero_states):
                    for key, value in zero.items():
                        ttnn.copy(value, getattr(state, key))

            def forward():
                output = x
                for layer, state in zip(layer_list, states):
                    output = layer.prefill_forward(output, state=state, page_table=table, cos=cc, sin=ss)
                return output

            for repeat in range(3):
                reset()
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                output = forward()
                ttnn.synchronize_device(mesh)
                row = dict(mode=mode, local_batch=local_batch, repeat=repeat, seconds=time.perf_counter() - begin)
                if repeat == 2:
                    actual = (
                        ttnn.to_torch(ttnn.get_device_tensors(output)[0])
                        if mode == "tp4"
                        else ttnn.to_torch(
                            output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0 if mode == "dp4" else -1)
                        )
                    )
                    # Bounded host memory: compare a deterministic feature subsample.
                    actual = actual[:, ::32, ::8].float().clone()
                    if reference is None:
                        reference = actual
                    row["sampled_pcc"] = torch.corrcoef(torch.stack([reference.flatten(), actual.flatten()]))[
                        0, 1
                    ].item()
                    row["sampled_max_error"] = (actual - reference).abs().max().item()
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("PARALLEL_LAYOUT", json.dumps(row), flush=True)
                del output

            costs, calls = defaultdict(float), defaultdict(int)

            def wrap(owner, name):
                original = getattr(owner, name)
                originals.append((owner, name, original))

                def measured(*a, **kw):
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    result = original(*a, **kw)
                    ttnn.synchronize_device(mesh)
                    costs[name] += time.perf_counter() - begin
                    calls[name] += 1
                    return result

                setattr(owner, name, measured)

            for owner, names in (
                (ttnn.experimental, ("minimal_matmul", "reduce_scatter_minimal_async", "all_gather_async")),
                (ttnn.transformer, ("chunk_gated_delta_rule", "chunked_scaled_dot_product_attention")),
                (ttnn.experimental.kda, ("qkv_causal_conv1d_silu",)),
            ):
                for name in names:
                    wrap(owner, name)
            reset()
            output = forward()
            ttnn.synchronize_device(mesh)
            report.setdefault("instrumented", []).append(dict(mode=mode, seconds=dict(costs), calls=dict(calls)))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("PARALLEL_PHASES", json.dumps(report["instrumented"][-1]), flush=True)
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)
            originals.clear()
            del output, x, cc, ss, states, zero_states, layer_list, layer, table
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

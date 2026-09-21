# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two concurrent TP2 submeshes versus TP4; representative layers, not serving."""

import argparse
import faulthandler
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.model import Checkpoint, checkpoint_path
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_27b.tt.precision import decoder_policy, load_precision
from models.common.modules.tt_ccl import TT_CCL


class TP2Decoder(MultichipDecoder):
    TP = 2


def main():
    faulthandler.dump_traceback_later(90, repeat=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--debug-sync", action="store_true")
    parser.add_argument("--tp2-ring", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch % 2 or args.length % 32:
        parser.error("Batch must be even and length tile-aligned")
    torch.set_num_threads(8)
    torch.manual_seed(23)
    config = AutoConfig.from_pretrained(checkpoint_path(), local_files_only=True).text_config
    weights = Checkpoint(checkpoint_path())
    precision = load_precision()
    host = (torch.randn(args.batch, args.length, config.hidden_size) * 0.1).bfloat16()
    rotary = Qwen3_5TextRotaryEmbedding(config)
    cos, sin = rotary(host, torch.arange(args.length)[None].expand(args.batch, -1))
    host_decode = (torch.randn(args.batch, 1, config.hidden_size) * 0.1).bfloat16()
    configure_fabric()
    root = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    report = dict(
        batch=args.batch, length=args.length, layers=[0, 3], debug_sync=args.debug_sync, tp2_ring=args.tp2_ring, rows=[]
    )
    references = {}
    try:
        for mode in ("tp4", "tp2dp2"):
            print("TP2_STAGE", mode, "create_meshes", flush=True)
            meshes = [root] if mode == "tp4" else root.create_submeshes(ttnn.MeshShape(1, 2))
            groups = []
            local_batch = args.batch // len(meshes)
            for group_id, mesh in enumerate(meshes):
                print("TP2_STAGE", mode, group_id, "load", flush=True)
                cls = MultichipDecoder if mode == "tp4" else TP2Decoder
                ccl = TT_CCL(mesh)
                layers = []
                for index in (0, 3):
                    policy = {
                        **decoder_policy(precision, index),
                        "prefill_sharded_residual": True,
                        "prefill_replicated_norm": True,
                        "direct_allreduce": False,
                        "persistent_ccl": False,
                        "ring": mode == "tp4" or args.tp2_ring,
                    }
                    layers.append(
                        cls.from_state_dict(
                            weights.layer(index),
                            hf_config=config,
                            layer_idx=index,
                            mesh_device=mesh,
                            policy=policy,
                            ccl=ccl,
                        )
                    )
                if mode == "tp2dp2" and args.debug_sync:
                    for layer in layers:
                        for name in ("_norm", "_gather", "_linear", "_delta", "_full_prefill"):
                            original = getattr(layer, name)
                            label = f"group{group_id}.layer{layer.layer_idx}.{name}"

                            def measured(*a, _original=original, _label=label, _mesh=mesh, _name=name, **kw):
                                print("TP2_OP_BEGIN", _label, flush=True)
                                with ExitStack() as stack:
                                    if _name == "_full_prefill":
                                        for owner, op in (
                                            (ttnn.transformer, "chunked_scaled_dot_product_attention"),
                                            (ttnn.experimental, "paged_fill_cache"),
                                        ):
                                            native = getattr(owner, op)

                                            def checked(*a, _native=native, _op=op, **kw):
                                                print("TP2_NATIVE_BEGIN", _label, _op, flush=True)
                                                result = _native(*a, **kw)
                                                ttnn.synchronize_device(_mesh)
                                                print("TP2_NATIVE_END", _label, _op, flush=True)
                                                return result

                                            stack.enter_context(patch.object(owner, op, checked))
                                    result = _original(*a, **kw)
                                ttnn.synchronize_device(_mesh)
                                print("TP2_OP_END", _label, flush=True)
                                return result

                            setattr(layer, name, measured)

                def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=False):
                    return ttnn.from_torch(
                        x.contiguous(),
                        dtype=dtype,
                        layout=layout,
                        device=mesh,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1) if shard else ttnn.ReplicateTensorToMesh(mesh),
                    )

                rows = slice(group_id * local_batch, (group_id + 1) * local_batch)
                pages = (args.length + 32) // 32
                table = upload(
                    torch.arange(local_batch * pages, dtype=torch.int32).reshape(local_batch, pages),
                    ttnn.int32,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                states = [
                    layer.allocate_state(batch_size=local_batch, num_pages=local_batch * pages) for layer in layers
                ]
                zeros = [
                    {key: ttnn.zeros_like(value) for key, value in vars(state).items() if value is not None}
                    for state in states
                ]
                positions = upload(
                    torch.full((local_batch,), args.length, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                )
                dc, ds = rotary(host_decode[rows], torch.full((local_batch, 1), args.length))
                groups.append(
                    dict(
                        mesh=mesh,
                        layers=layers,
                        states=states,
                        zeros=zeros,
                        table=table,
                        x=upload(host[rows], shard=True),
                        cos=upload(cos[rows]),
                        sin=upload(sin[rows]),
                        dx=upload(host_decode[rows]),
                        dc=upload(dc),
                        ds=upload(ds),
                        positions=positions,
                    )
                )

            def sync():
                for group in groups:
                    ttnn.synchronize_device(group["mesh"])

            def reset():
                for group in groups:
                    for state, zero in zip(group["states"], group["zeros"]):
                        for key, value in zero.items():
                            ttnn.copy(value, getattr(state, key))

            for repeat in range(3):
                print("TP2_STAGE", mode, repeat, "execute", flush=True)
                reset()
                sync()
                begin = time.perf_counter()
                outputs = []
                for group in groups:
                    output = group["x"]
                    for layer, state in zip(group["layers"], group["states"]):
                        output = layer.prefill_sharded_forward(
                            output, state=state, page_table=group["table"], cos=group["cos"], sin=group["sin"]
                        )
                    outputs.append(output)
                sync()
                prefill = time.perf_counter() - begin
                samples = [
                    ttnn.to_torch(output[:, ::32, ::8], mesh_composer=ttnn.ConcatMeshToTensor(group["mesh"], dim=-1))
                    for output, group in zip(outputs, groups)
                ]
                actual = torch.cat(samples, dim=0).double()
                if mode == "tp4":
                    references[repeat] = actual.clone()
                reference = references[repeat]
                row = dict(
                    mode=mode,
                    repeat=repeat,
                    prefill_seconds=prefill,
                    sampled_exact=torch.equal(actual, reference),
                    sampled_pcc=torch.corrcoef(torch.stack([actual.flatten(), reference.flatten()]))[0, 1].item(),
                    sampled_relative_l2=((actual - reference).norm() / reference.norm()).item(),
                    sampled_max_abs=(actual - reference).abs().max().item(),
                )
                # Eager one-step decode includes submission to both groups, but
                # is not a traced full-model decode or a token-quality check.
                sync()
                begin = time.perf_counter()
                decoded = []
                for group in groups:
                    output = group["dx"]
                    for layer, state in zip(group["layers"], group["states"]):
                        output = layer.decode_forward(
                            output,
                            state=state,
                            page_table=group["table"],
                            current_pos=group["positions"],
                            cos=group["dc"],
                            sin=group["ds"],
                        )
                    decoded.append(output)
                sync()
                row["eager_decode_seconds"] = time.perf_counter() - begin
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("TP2_DP2", json.dumps(row), flush=True)
                del outputs, decoded, output, samples, actual
            # Keep submesh lifetimes within the root mesh; release tensors first.
            del groups, layers, states, zeros, table, positions, group, layer, state
    finally:
        ttnn.close_mesh_device(root)


if __name__ == "__main__":
    main()

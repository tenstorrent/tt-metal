"""Real-weight TP4 regression for shared decode gather storage across layers/traces.

Run with TT_METAL_TRACE_ALLOC_TRACKING=1 and
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0. Device use must be serialized by the caller.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.optimized_trace import capture
from models.autoports.google_gemma_4_31b_it.tests.run_optimized_decoder import weights
from models.autoports.google_gemma_4_31b_it.tt.multichip_decoder import MultichipDecoder
from models.autoports.google_gemma_4_31b_it.tt.optimized_decoder import HF_MODEL, HF_REVISION


def addresses(tensor):
    return tuple(int(local.buffer_address()) for local in ttnn.get_device_tensors(tensor))


def check_workspace(models, shared):
    assert set(shared) == {"self_attn.o_proj", "mlp.down_proj"}, set(shared)
    identity = {}
    for role, backing in shared.items():
        assert tuple(backing.shape) == (1, 1, 32, 5376)
        memory = backing.memory_config()
        assert memory.buffer_type == ttnn.BufferType.L1
        assert memory.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        identity[role] = addresses(backing)
        assert len(identity[role]) == 4
        for model in models.values():
            for batch in (1, 8, 16, 32):
                view = model.persistent_ccl[(batch, role)][1]
                assert tuple(view.shape) == (1, 1, batch, 5376)
                assert addresses(view) == identity[role], (role, batch)
                assert view.memory_config() == memory
    assert all(a != b for a, b in zip(*identity.values())), identity
    return identity


def run(args):
    for name, expected in (
        ("TT_METAL_TRACE_ALLOC_TRACKING", "1"),
        ("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
    ):
        assert os.getenv(name) == expected, f"Set {name}={expected} before starting Python"
    assert args.seq >= 32 and args.rounds >= 2
    torch.set_num_threads(16)
    torch.manual_seed(123)
    config = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    layers = (0, 5)
    assert {config.layer_types[layer] for layer in layers} == {"sliding_attention", "full_attention"}
    fixtures = {}
    for layer in layers:
        path = Path(args.inputs_dir) / f"optimized_decoder_layer{layer}_inputs2050.pt"
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
        assert (metadata["model"], metadata["revision"], metadata["layer"]) == (HF_MODEL, HF_REVISION, layer)
        source = torch.load(path, weights_only=True, mmap=True)
        assert source.shape[0] == 1 and source.shape[1] >= args.seq + 1 + 17 * 31
        fixtures[layer] = source

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    traces = []
    try:
        shared, models, cases = {}, {}, {}

        def upload(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            host_tensor = ttnn.from_torch(
                tensor.contiguous(),
                dtype=dtype,
                layout=layout,
                device=None,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            return ttnn.to_device(host_tensor, mesh)

        def host(tensor):
            return [ttnn.to_torch(local).clone() for local in ttnn.get_device_tensors(tensor)]

        for layer in layers:
            models[layer] = MultichipDecoder.from_state_dict(
                weights(layer),
                hf_config=config,
                layer_idx=layer,
                mesh_device=mesh,
                residual_sharded=False,
                decode_gather_buffers=shared,
            )
            print("MODEL_READY", layer, flush=True)
        identity = check_workspace(models, shared)
        for layer, model in models.items():
            for batch in (1, 32):
                source = fixtures[layer]
                x = torch.stack([source[0, 17 * i : 17 * i + args.seq + 1] for i in range(batch)])
                cos, sin = Gemma4TextRotaryEmbedding(config)(
                    x,
                    torch.arange(args.seq + 1).expand(batch, -1),
                    layer_type=config.layer_types[layer],
                )
                blocks = (args.seq + 1 + 31) // 32
                pages = (torch.randperm(batch * blocks) + 3).reshape(batch, blocks).to(torch.int32)
                cache = model.allocate_cache(num_blocks=batch * blocks + 3)
                page = upload(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                px = upload(x[:, : args.seq].unsqueeze(1))
                pc, ps = upload(cos[:, : args.seq].unsqueeze(1)), upload(sin[:, : args.seq].unsqueeze(1))
                pref = model.prefill_forward(px, cos=pc, sin=ps, kv_cache=cache, page_table=page)
                ttnn.synchronize_device(mesh)
                del pref, px, pc, ps
                cases[(layer, batch)] = {
                    "x": upload(x[:, args.seq :].reshape(1, 1, batch, -1)),
                    "kwargs": dict(
                        cos=upload(cos[:, args.seq :].reshape(1, batch, 1, -1)),
                        sin=upload(sin[:, args.seq :].reshape(1, batch, 1, -1)),
                        current_pos=upload(
                            torch.full((batch,), args.seq, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                        ),
                        kv_cache=cache,
                        page_table=page,
                    ),
                }

        def forward(key):
            case = cases[key]
            with (
                patch("ttnn.from_torch", side_effect=AssertionError("runtime host conversion")),
                patch("ttnn.to_torch", side_effect=AssertionError("runtime host read")),
            ):
                return models[key[0]].decode_forward(case["x"], **case["kwargs"])

        # All persistent inputs, caches, program signatures and semaphores are
        # exercised before the first capture. Rewriting the same KV position is
        # idempotent, so every replay has an identical eager reference.
        for _ in range(2):
            for key, case in cases.items():
                output = forward(key)
                case["expected"] = host(output)
                assert all(torch.isfinite(value).all() for value in case["expected"])
                del output
        ttnn.synchronize_device(mesh)
        for key, case in cases.items():
            tid, output = capture(mesh, lambda key=key: forward(key))
            traces.append(tid)
            case.update(trace=tid, output=output)
            for backing in shared.values():
                assert all(a != b for a, b in zip(addresses(output), addresses(backing)))
            # A captured output may alias another trace's scratch. Its contents
            # are valid only after its own replay, and are read before any other
            # replay. Exempt exactly this output, never inputs or workspace.
            trace_allocation_tracker.acknowledge_corruptible(output)
        assert check_workspace(models, shared) == identity

        def check_output(key):
            case = cases[key]
            actual = host(case["output"])
            for device, (observed, expected) in enumerate(zip(actual, case["expected"])):
                assert torch.equal(observed, expected), (key, device, "trace differs from eager")

        order = ((0, 1), (5, 32), (5, 1), (0, 32))
        for iteration in range(args.rounds):
            for key in order:
                ttnn.execute_trace(mesh, cases[key]["trace"], cq_id=0, blocking=True)
                check_output(key)
            assert check_workspace(models, shared) == identity
            print("REPLAY_ROUND_PASS", iteration, flush=True)

        source_paths = [Path(__file__), Path(__file__).parents[1] / "tt/multichip_decoder.py"]
        result = dict(
            passed=True,
            model=HF_MODEL,
            revision=HF_REVISION,
            layers=list(layers),
            batches=[1, 32],
            seq=args.seq,
            rounds=args.rounds,
            captures=len(traces),
            replays=args.rounds * len(order),
            all_devices_bitwise_equal_to_eager=True,
            outputs_consumed_immediately_after_own_replay=True,
            output_addresses={
                f"layer{layer}_batch{batch}": addresses(case["output"]) for (layer, batch), case in cases.items()
            },
            acknowledged_corruptible_buffers="captured outputs only",
            shared_addresses=identity,
            identity_checked_batches=[1, 8, 16, 32],
            allocation_tracking=True,
            skip_program_cache=False,
            source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        )
        Path(args.result).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        for tid in reversed(traces):
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=33)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--inputs-dir", default="bringup/artifacts")
    parser.add_argument("--result", required=True)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        run(parser.parse_args())

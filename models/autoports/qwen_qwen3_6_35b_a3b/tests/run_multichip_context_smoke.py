# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-pass long-context optimized-vs-mesh decoder correctness smoke.

Runs one prefill and a short eager decode stream on each side. Does not capture
traces or claim warmed latency. CPU boundaries preserve full output tensors and
logical written cache cells; PCC uses bounded float64 chunks to avoid duplicating
multi-gigabyte context outputs in float64 simultaneously.
"""

import argparse
import json
import math
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import REVISION, config, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder import _write
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_mixed_stack import _dram_census
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only


def whole_pcc(expected, actual, chunk_elements=1048576):
    """Whole-tensor PCC via float64 sufficient statistics in bounded chunks."""
    if expected.shape != actual.shape:
        raise ValueError(f"PCC shapes differ: {expected.shape} != {actual.shape}")
    a, b = expected.reshape(-1), actual.reshape(-1)
    count = a.numel()
    if not count:
        raise ValueError("PCC needs nonempty tensors")
    sx = sy = sxx = syy = sxy = 0.0
    exact = True
    for start in range(0, count, chunk_elements):
        x, y = a[start : start + chunk_elements].double(), b[start : start + chunk_elements].double()
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError("PCC input contains nonfinite values")
        exact = exact and torch.equal(x, y)
        sx += x.sum().item()
        sy += y.sum().item()
        sxx += torch.dot(x, x).item()
        syy += torch.dot(y, y).item()
        sxy += torch.dot(x, y).item()
    if exact:
        return 1.0
    vx, vy = sxx - sx * sx / count, syy - sy * sy / count
    if vx <= 0 or vy <= 0:
        return 0.0
    return min(1.0, max(-1.0, (sxy - sx * sy / count) / math.sqrt(vx * vy)))


def make_inputs(args, cfg):
    torch.manual_seed(args.seed)
    x = torch.randn(1, args.length, cfg.hidden_size, dtype=torch.bfloat16) * 0.05
    xd = torch.randn(args.decode_steps, 1, 1, cfg.hidden_size, dtype=torch.bfloat16) * 0.05
    if args.activation_file:
        payload = torch.load(args.activation_file, weights_only=True)
        assert payload["metadata"]["checkpoint_revision"] == REVISION
        values = payload["layers"][str(args.layer)]
        assert values["x"].shape == x.shape
        x = values["x"]
        if values["xd"].ndim == 4:
            assert values["xd"].shape[0] >= args.decode_steps
            assert values["xd"].shape[1:] == xd.shape[1:]
            xd = values["xd"][: args.decode_steps].clone()
        else:
            assert args.decode_steps == 1, "An activation payload with one held-out token cannot supply multiple steps"
            assert values["xd"].shape == xd[0].shape
            xd[0] = values["xd"]
    return dict(x=x, xd=xd, pages=torch.randperm((args.context + 31) // 32))


def run_side(args, cfg, sd, inputs, *, multichip):
    if multichip:
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder as Decoder

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    else:
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import OptimizedDecoder as Decoder

        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    memory = []
    label = "multichip" if multichip else "baseline"
    memory_path = str(Path(args.output).with_suffix("")) + f".{label}.dram.json"

    def census(stage):
        ttnn.synchronize_device(mesh)
        memory.append(_dram_census(mesh, stage, memory[-1] if memory else None))
        _write(memory_path, memory)
        print("CONTEXT_DRAM", label, stage, memory[-1]["allocated_bytes_per_device"], flush=True)

    try:
        census("before_loading")
        dec = Decoder.from_state_dict(
            sd,
            hf_config=cfg,
            layer_idx=args.layer,
            mesh_device=mesh,
            **json.loads(args.mesh_build_kwargs if multichip else args.baseline_build_kwargs),
        )
        census("decoder_loaded")
        sharded = multichip and getattr(dec, "residual_layout", "sharded") != "replicated"
        mapper = ttnn.ShardTensorToMesh(mesh, dim=-1) if sharded else ttnn.ReplicateTensorToMesh(mesh)

        def upload(value):
            return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

        def host(value):
            if sharded:
                return ttnn.to_torch(value, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).clone()
            return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).clone()

        state = (
            dec.new_recurrent_state()
            if dec.kind == "linear_attention"
            else dec.new_kv_state(context_length=args.context, page_order=inputs["pages"])
        )
        plan = dec.prepare_prefill(
            seq_len=args.length, **({"chunk_size": args.chunk_size} if args.chunk_size is not None else {})
        )
        tx = upload(inputs["x"])
        decode_inputs = [upload(row) for row in inputs["xd"]]
        positions = [
            dec.prepare_positions(start_pos=args.length + step, seq_len=1) if dec.kind == "full_attention" else None
            for step in range(args.decode_steps)
        ]
        census("cache_inputs_positions_and_workspace_prepared")
        with device_only():
            output = dec.prefill_forward(tx, state=state, plan=plan)
        prefill = host(output).reshape(inputs["x"].shape)
        del output
        census("prefill_complete")
        print("CONTEXT_PREFILL_DONE", label, args.layer, args.length, flush=True)
        decoded = []
        for step, (row, position) in enumerate(zip(decode_inputs, positions)):
            with device_only():
                output = dec.decode_forward(row, state=state, positions=position)
            decoded.append(host(output).reshape(1, 1, cfg.hidden_size))
            del output
            print("CONTEXT_DECODE_DONE", label, step, args.length + step, flush=True)
        census("eager_decode_complete")
        host_state, ownership = {}, {}
        keys = ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v")
        shapes = {key: list(state[key].shape) for key in keys}
        if dec.kind == "linear_attention":
            for key in keys:
                parts = [ttnn.to_torch(part) for part in ttnn.get_device_tensors(state[key])]
                if not multichip:
                    host_state[key] = parts[0].clone()
                elif key == "conv":
                    split = [torch.split(part, (512, 512, 1024), dim=-1) for part in parts]
                    host_state[key] = torch.cat(
                        [torch.cat([part[group] for part in split], dim=-1) for group in range(3)], dim=-1
                    )
                else:
                    host_state[key] = torch.cat(parts, dim=1)
                del parts
            ownership["layout"] = (
                "local packed Q512/K512/V1024 conv; local8 recurrent heads" if multichip else "full state"
            )
        else:
            written = args.length + args.decode_steps
            page_order = inputs["pages"].to(torch.long)
            tables = [ttnn.to_torch(part).reshape(-1) for part in ttnn.get_device_tensors(state["page_table"])]
            expected_table = torch.cat((page_order, page_order[-1:].expand((-len(page_order)) % 8)))
            ownership["page_tables_exact"] = all(torch.equal(table.to(torch.long), expected_table) for table in tables)
            ownership["rank_to_global_kv_head"] = [0, 0, 1, 1] if multichip else [0, 1]
            ownership["written_positions"] = written
            for key in keys:
                devices = ttnn.get_device_tensors(state[key])
                owned = []
                replicas_equal = True
                for rank in (0, 2) if multichip else (0,):
                    physical = ttnn.to_torch(devices[rank])
                    if multichip:
                        replica = ttnn.to_torch(devices[rank + 1])
                        replicas_equal = replicas_equal and torch.equal(physical, replica)
                        del replica
                    logical = physical[page_order].permute(1, 0, 2, 3).flatten(1, 2)
                    ownership[f"{key}_unwritten_nonzero_rank{rank}"] = int(torch.count_nonzero(logical[:, written:]))
                    owned.append(logical[:, :written].contiguous())
                    del physical, logical
                host_state[key] = torch.cat(owned, dim=0) if multichip else owned[0]
                ownership[key + "_replicas_equal"] = replicas_equal
                del owned
        return dict(
            prefill=prefill,
            decode=torch.stack(decoded),
            state=host_state,
            summary=dict(
                decoder=Decoder.__name__,
                kind=dec.kind,
                layer=args.layer,
                logical_length=args.length,
                context=args.context,
                decode_steps=args.decode_steps,
                decode_positions=list(range(args.length, args.length + args.decode_steps)),
                residual_layout="hidden-sharded" if sharded else "replicated",
                prefill_call_count=1,
                prefill_plan_chunks=len(plan),
                prefill_physical_sizes=sorted({part[1] for part in plan}),
                runtime_fallback_audit="pass",
                local_state_shapes=shapes,
                ownership=ownership,
                dram_census=memory,
                dram_census_artifact=memory_path,
                timing_claim=False,
                trace_claim=False,
                regime="serial; mesh1x4" if multichip else "serial; chip0",
            ),
        )
    finally:
        ttnn.close_mesh_device(mesh)


def run(args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    if args.length < 1 or args.decode_steps < 1 or args.length + args.decode_steps > args.context:
        raise ValueError("Positive prefill/decode lengths must fit the requested context")
    cfg = config()
    if args.context > cfg.max_position_embeddings:
        raise ValueError("Context exceeds the HF capability contract")
    torch.set_num_threads(8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(str(output))
    provenance_path = str(output) + ".provenance.json"
    provenance = json.loads(Path(provenance_path).read_text())
    provenance.update(device="serial optimized chip0 then multichip1x4", decoder="context-smoke", trace_capture=False)
    _write(provenance_path, provenance)
    signature = {
        name: getattr(args, name)
        for name in (
            "layer",
            "length",
            "context",
            "decode_steps",
            "seed",
            "activation_file",
            "baseline_build_kwargs",
            "chunk_size",
        )
    }
    signature["checkpoint_revision"] = REVISION
    sd = weights(args.layer)
    if args.baseline_load:
        baseline_path = Path(args.baseline_load)
        saved = torch.load(baseline_path, weights_only=True)
        assert saved["signature"] == signature, "Saved baseline does not match requested inputs/configuration"
        inputs, baseline = saved["inputs"], saved["result"]
    else:
        inputs = make_inputs(args, cfg)
        baseline = run_side(args, cfg, sd, inputs, multichip=False)
        baseline_path = Path(args.baseline_save) if args.baseline_save else output.with_suffix(".baseline.pt")
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(signature=signature, inputs=inputs, result=baseline), baseline_path)
    report = dict(
        signature=signature,
        baseline=baseline["summary"],
        baseline_artifact=str(baseline_path),
        activation_source=args.activation_file or "seeded random BF16 std0.05; identical saved inputs on both sides",
        pcc_method="whole-output float64 sums/squares/cross-products, 1048576-element chunks",
        timing_claim=False,
        trace_claim=False,
    )
    if args.baseline_only:
        _write(str(output), report)
        return
    candidate = run_side(args, cfg, sd, inputs, multichip=True)
    mesh_path = output.with_suffix(".multichip.pt")
    torch.save(dict(signature=signature, result=candidate), mesh_path)
    report.update(
        multichip=candidate["summary"],
        multichip_artifact=str(mesh_path),
        prefill_pcc=whole_pcc(baseline["prefill"], candidate["prefill"]),
        decode_pcc=[whole_pcc(a, b) for a, b in zip(baseline["decode"], candidate["decode"])],
        state_pcc={key: whole_pcc(value, candidate["state"][key]) for key, value in baseline["state"].items()},
    )
    ownership_ok = True
    for result in (baseline, candidate):
        owned = result["summary"]["ownership"]
        ownership_ok = ownership_ok and owned.get("page_tables_exact", True)
        ownership_ok = ownership_ok and all(value for key, value in owned.items() if key.endswith("_replicas_equal"))
        ownership_ok = ownership_ok and all(value == 0 for key, value in owned.items() if "_unwritten_nonzero_" in key)
    report["ownership_passed"] = ownership_ok
    report["passed"] = (
        report["prefill_pcc"] >= 0.995
        and min(report["decode_pcc"]) >= 0.995
        and min(report["state_pcc"].values()) >= 0.995
        and ownership_ok
    )
    _write(str(output), report)
    print(json.dumps(report, indent=2), flush=True)
    assert report["passed"], "Long-context whole-output, written-cache and ownership gates must pass"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=32769)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--activation-file")
    parser.add_argument("--baseline-save")
    parser.add_argument("--baseline-load")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--baseline-build-kwargs", default="{}")
    parser.add_argument("--mesh-build-kwargs", default="{}")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())

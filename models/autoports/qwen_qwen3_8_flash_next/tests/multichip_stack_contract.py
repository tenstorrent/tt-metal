"""Representative or recorded adjacent decoders pass residuals directly.

The single-device control loads one layer at a time to fit its replicated PLE
weights. The mesh loads the selected layers and traces the actual direct chain.
This is a decoder interface test, without embeddings, final norm or generation.
"""

import argparse, gc, hashlib, json, os
from pathlib import Path
import torch, ttnn
from transformers import AutoConfig
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_components import CHECKPOINT
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_decoder import no_runtime_host_conversion
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder, _hidden_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["single", "mesh"], required=True)
    ap.add_argument("--length", type=int, default=129)
    ap.add_argument("--recorded-tag", help="Use recorded layer-0 residuals and adjacent layers 0–3")
    ap.add_argument("--tag", help="Isolate reference tensors and reports under this directory")
    ap.add_argument("--reference-tag")
    a = ap.parse_args()
    torch.set_num_threads(8)
    rng = torch.Generator().manual_seed(121)
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config

    def activation(n):
        return (
            torch.randn(1, 1, n, 2560, generator=rng).repeat(1, 1, 1, 4) * 0.05
            + torch.randn(1, 1, n, 10240, generator=rng) * 0.005
        ).bfloat16()

    inputs = [activation(a.length), activation(1), activation(1)]
    tokens = [torch.randint(cfg.vocab_size, (1, n), generator=rng, dtype=torch.int32) for n in [a.length, 1, 1]]
    page = torch.randperm(cfg.max_position_embeddings // 32, generator=rng).reshape(1, -1).int()
    cache = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/multichip")
    dest = Path("models/autoports/qwen_qwen3_8_flash_next") / os.environ.get("QWEN38_STAGE_DIR", "doc/multichip_decoder")
    source = dest.parents[1] / "tt" / ("multichip_decoder.py" if a.impl == "mesh" else "optimized_decoder.py")
    fixture_info = None
    if a.recorded_tag:
        fixture_path = cache / a.recorded_tag / "layer0.pt"
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
        residuals = torch.cat(fixture["inputs"], dim=1).unsqueeze(0)
        ids = torch.cat(fixture["tokens"], dim=1)
        if residuals.shape[2] < a.length + 2 or ids.shape[1] < a.length + 2:
            raise ValueError("Recorded stack requires the prefix plus two adjacent decode tokens")
        spans = [(0, a.length), (a.length, a.length + 1), (a.length + 1, a.length + 2)]
        inputs = [residuals[:, :, start:stop, :].contiguous() for start, stop in spans]
        tokens = [ids[:, start:stop].contiguous() for start, stop in spans]
        fixture_info = dict(path=str(fixture_path), sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest())
    namespace = a.tag or (f"stack_{a.recorded_tag}" if a.recorded_tag else None)
    if namespace:
        cache, dest = cache / namespace, dest / namespace
    reference_cache = cache.parent / a.reference_tag if a.reference_tag else cache
    reference_dest = dest.parent / a.reference_tag if a.reference_tag else dest
    if a.reference_tag and not reference_dest.exists():
        reference_dest = Path(__file__).resolve().parents[1] / "doc/multichip_decoder" / a.reference_tag
    cache.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    layers = [0, 1, 2, 3] if a.recorded_tag else [0, 3, 1]
    metadata = dict(
        args=vars(a),
        layers=layers,
        fixture=fixture_info,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        status="running",
    )
    if namespace and a.impl == "mesh":
        reference = json.loads((reference_dest / "stack_single_manifest.json").read_text())
        if (
            reference.get("status") != "passed"
            or reference.get("layers") != layers
            or reference.get("fixture") != fixture_info
            or reference["args"]["length"] != a.length
        ):
            raise ValueError("Stack reference provenance differs; rerun the single control with the same arguments")
    (dest / f"stack_{a.impl}_manifest.json").write_text(json.dumps(metadata, indent=2))
    (dest / "stack_runner.py.txt").write_bytes(Path(__file__).read_bytes())
    results = []

    def construct(mesh, layer):
        return (MultichipDecoder if a.impl == "mesh" else OptimizedDecoder).from_state_dict(
            CheckpointState(CHECKPOINT, f"model.language_model.layers.{layer}."),
            hf_config=cfg,
            layer_idx=layer,
            mesh_device=mesh,
            batch_size=1,
            max_requests=2,
            cache_dir=Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"),
        )

    if a.impl == "single":
        for layer in layers:
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
            try:
                model = construct(mesh, layer)
                model.prepare([a.length, 1])
                next_inputs = []

                def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
                    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)

                slots = torch.zeros(1, 32, dtype=torch.int32)
                slots[0, 0] = 1
                sd = upload(slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                pg = upload(page, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                for step in range(3):
                    pos = torch.zeros(1, 32, dtype=torch.int32)
                    pos[0, 0] = 0 if step == 0 else a.length + step - 1
                    pd = upload(pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                    xd = upload(inputs[step])
                    td = upload(tokens[step], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                    fn = model.prefill_forward if step == 0 else model.decode_forward
                    with no_runtime_host_conversion():
                        out = fn(
                            xd,
                            current_positions=pd,
                            slots=sd,
                            page_table=pg if layer == 3 else None,
                            tokens=td if layer == 1 else None,
                        )
                    next_inputs.append(ttnn.to_torch(out))
                    del out, xd, td, pd
                inputs = next_inputs
                torch.save(inputs, cache / f"stack_single_after_l{layer}_s{a.length}.pt")
            finally:
                ttnn.close_mesh_device(mesh)
            del model
            gc.collect()
            print("STACK_CONTROL", layer, flush=True)
    else:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        trace = None
        try:
            models = [construct(mesh, l) for l in layers]
            ops = models[0].ops
            for model in models:
                model.prepare([a.length, 1])
            sdhost = torch.zeros(1, 32, dtype=torch.int32)
            sdhost[0, 0] = 1
            sd = ops.upload(sdhost, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            pd = ops.upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            dp = ops.upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            pg = ops.upload(page, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            pref = ops.shards(_hidden_parts(inputs[0]))
            xd = ops.shards(_hidden_parts(inputs[1]))
            pt = ops.upload(tokens[0], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            dt = ops.upload(tokens[1], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

            def host(t):
                return torch.cat(
                    [ttnn.to_torch(v).reshape(1, 1, t.shape[2], 4, 640) for v in ttnn.get_device_tensors(t)], -1
                ).reshape(1, 1, t.shape[2], 10240)

            def refresh(t, x, dtype=ttnn.uint32, residual=False):
                mapper = ttnn.ReplicateTensorToMesh(mesh)
                if residual:
                    x = torch.cat(_hidden_parts(x), 0)
                    mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
                h = ttnn.from_torch(
                    x, dtype=dtype, layout=ttnn.TILE_LAYOUT if residual else ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
                )
                ttnn.copy_host_to_device_tensor(h, t)

            def chain(value, decode):
                outputs = []
                with no_runtime_host_conversion():
                    for layer, model in zip(layers, models):
                        fn = model.decode_forward if decode else model.prefill_forward
                        value = fn(
                            value,
                            current_positions=dp if decode else pd,
                            slots=sd,
                            page_table=pg if layer == 3 else None,
                            tokens=(dt if decode else pt) if layer == 1 else None,
                        )
                        assert tuple(value.shape) == (1, 1, 1 if decode else a.length, 2560)
                        outputs.append(value)
                    if decode:
                        ttnn.plus_one(dp)
                return outputs

            position = torch.zeros(1, 32, dtype=torch.int32)
            position[0, 0] = a.length
            eager = chain(pref, False)
            for l, out in zip(layers, eager):
                metric = pcc_metrics(
                    torch.load(reference_cache / f"stack_single_after_l{l}_s{a.length}.pt", weights_only=True)[0],
                    host(out),
                )
                assert metric["pcc"] >= 0.995, (l, "prefill", metric)
                results.append(dict(layer=l, case="prefill", **metric))
            del eager
            refresh(dp, position)
            warm = chain(xd, True)
            del warm
            eager = chain(pref, False)
            del eager
            refresh(dp, position)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            outputs = chain(xd, True)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            eager = chain(pref, False)
            del eager
            refresh(dp, position)
            for step in [1, 2]:
                refresh(xd, inputs[step], ttnn.bfloat16, True)
                refresh(dt, tokens[step])
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                for l, out in zip(layers, outputs):
                    metric = pcc_metrics(
                        torch.load(reference_cache / f"stack_single_after_l{l}_s{a.length}.pt", weights_only=True)[
                            step
                        ],
                        host(out),
                    )
                    assert metric["pcc"] >= 0.995, (l, step, metric)
                    results.append(dict(layer=l, case=f"decode{step}", **metric))
            (dest / "stack_contract.json").write_text(
                json.dumps(
                    dict(
                        status="passed",
                        source_sha256=metadata["source_sha256"],
                        layers=layers,
                        args=vars(a),
                        metrics=results,
                        captures=1,
                        replays=2,
                        trace_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
                        skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
                    ),
                    indent=2,
                )
            )
        finally:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)
    metadata["status"] = "passed"
    (dest / f"stack_{a.impl}_manifest.json").write_text(json.dumps(metadata, indent=2))
    print("STACK_CONTRACT_PASS", a.impl, flush=True)


if __name__ == "__main__":
    main()

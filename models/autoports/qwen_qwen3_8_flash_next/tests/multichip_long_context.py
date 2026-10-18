"""Stream every advertised-context token, then compare final prefill and trace.

Single-chip artifacts are saved per chunk outside the repository. The mesh run
compares every chunk, not only selected positions. No operation goes beyond the
last valid context position. Host refresh/readback occurs only at test boundaries.
"""

import argparse, hashlib, json, os, time
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
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["single", "mesh"], required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--context", type=int)
    p.add_argument("--tag", default="full_context")
    p.add_argument("--reference-tag")
    p.add_argument("--qsa-fidelity", choices=["LoFi", "HiFi2", "HiFi4"])
    p.add_argument("--qsa-fidelity-role", choices=["all", "input", "output", "prefill", "prefill_input"], default="all")
    p.add_argument("--recorded", action="store_true")
    p.add_argument("--recorded-tag", default="recorded_inputs")
    a = p.parse_args()
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    context = a.context or cfg.max_position_embeddings
    assert 1 < context <= cfg.max_position_embeddings
    root = Path("models/autoports/qwen_qwen3_8_flash_next")
    dest = root / os.environ.get("QWEN38_STAGE_DIR", "doc/multichip_decoder") / a.tag
    dest.mkdir(parents=True, exist_ok=True)
    cache = (
        Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/multichip")
        / (a.reference_tag or a.tag)
        / f"layer{a.layer}"
    )
    cache.mkdir(parents=True, exist_ok=True)
    state = CheckpointState(CHECKPOINT, f"model.language_model.layers.{a.layer}.")
    if a.impl == "mesh":
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4 if a.impl == "mesh" else 1), trace_region_size=0)
    trace = None
    records = []
    start = time.monotonic()
    manifest = dict(
        args=vars(a),
        context=context,
        status="running",
        records=records,
        trace_captures=0,
        trace_replays=0,
        env={k: v for k, v in os.environ.items() if k.startswith("TT_METAL_TRACE")},
    )
    artifact = dest / f"{a.impl}_l{a.layer}.json"
    source = root / "tt" / ("multichip_decoder.py" if a.impl == "mesh" else "optimized_decoder.py")
    manifest["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (dest / f"{a.impl}_source.py.txt").write_bytes(source.read_bytes())
    (dest / "runner.py.txt").write_bytes(Path(__file__).read_bytes())
    try:
        model = (MultichipDecoder if a.impl == "mesh" else OptimizedDecoder).from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=a.layer,
            mesh_device=mesh,
            batch_size=1,
            max_requests=2,
            chunk_size=a.chunk,
            cache_dir=Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"),
        )
        if a.qsa_fidelity:
            assert a.layer == 3 and a.impl == "mesh"
            selected_compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, a.qsa_fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            original_linear = model.attention._linear

            def role_linear(x, weight):
                selected = (
                    a.qsa_fidelity_role == "all"
                    or (a.qsa_fidelity_role == "input" and weight is model.attention.packed_projection)
                    or (a.qsa_fidelity_role == "output" and weight is model.attention.wo)
                    or (a.qsa_fidelity_role == "prefill" and x.shape[-2] > 1)
                    or (
                        a.qsa_fidelity_role == "prefill_input"
                        and x.shape[-2] > 1
                        and weight is model.attention.packed_projection
                    )
                )
                return (
                    model.ops.linear(x, weight, compute_kernel_config=selected_compute)
                    if selected
                    else original_linear(x, weight)
                )

            model.attention._linear = role_linear
        fixture = None
        if a.recorded:
            fixture = torch.load(cache.parent.parent / a.recorded_tag / f"layer{a.layer}.pt", weights_only=True)
            recorded_x = torch.cat(fixture["inputs"], dim=1).unsqueeze(0)
            recorded_tokens = torch.cat(fixture["tokens"], dim=1)
        model.prepare([1, a.chunk, (context - 1) % a.chunk or a.chunk])

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, residual=False):
            if a.impl == "mesh":
                return (
                    model.ops.shards(_hidden_parts(x), dtype, layout)
                    if residual
                    else model.ops.upload(x, dtype, layout)
                )
            return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)

        def refresh(t, x, dtype=ttnn.uint32, residual=False):
            mapper = ttnn.ReplicateTensorToMesh(mesh) if a.impl == "mesh" else None
            if residual and a.impl == "mesh":
                x = torch.cat(_hidden_parts(x), 0)
                mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
            h = ttnn.from_torch(
                x, dtype=dtype, layout=ttnn.TILE_LAYOUT if residual else ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
            )
            ttnn.copy_host_to_device_tensor(h, t)

        def host(t):
            if a.impl == "single":
                return ttnn.to_torch(t)
            return torch.cat(
                [ttnn.to_torch(v).reshape(1, 1, t.shape[2], 4, 640) for v in ttnn.get_device_tensors(t)], -1
            ).reshape(1, 1, t.shape[2], 10240)

        def data(position, length, variant=0):
            if fixture is not None:
                indices = (torch.arange(position, position + length) + variant) % recorded_x.shape[2]
                return recorded_x[:, :, indices, :], recorded_tokens[:, indices]

            gen = torch.Generator().manual_seed(2026 + position + variant * 1000000)
            x = (
                torch.randn(1, 1, length, 2560, generator=gen).repeat(1, 1, 1, 4) * 0.05
                + torch.randn(1, 1, length, 10240, generator=gen) * 0.005
            ).bfloat16()
            tok = torch.randint(cfg.vocab_size, (1, length), generator=gen, dtype=torch.int32)
            return x, tok

        zero = torch.zeros(1, 32, dtype=torch.int32)
        slot = zero.clone()
        slot[0, 0] = 1
        pd = upload(zero, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        sd = upload(slot, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        pg = (
            upload(
                torch.randperm(cfg.max_position_embeddings // 32, generator=torch.Generator().manual_seed(43))
                .reshape(1, -1)
                .int(),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            if a.layer == 3
            else None
        )

        def position(pos):
            v = zero.clone()
            v[0, 0] = pos
            refresh(pd, v)

        def validate(out, name, pos, length):
            actual = host(out)
            path = cache / f"{name}.pt"
            rec = dict(case=name, position=pos, length=length)
            if a.impl == "single":
                torch.save(actual, path)
            else:
                rec.update(pcc_metrics(torch.load(path, weights_only=True), actual))
                if rec["pcc"] < 0.995:
                    failure = cache.parent.parent / a.tag / f"layer{a.layer}" / f"failed_{name}.pt"
                    failure.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(actual, failure)
                    manifest.update(status="failed_pcc", failure=rec, failure_output=str(failure))
                    raise AssertionError(rec)
            records.append(rec)
            print(json.dumps(rec), flush=True)
            artifact.write_text(json.dumps(manifest, indent=2))
            return actual

        for pos in range(0, context - 1, a.chunk):
            n = min(a.chunk, context - 1 - pos)
            x, tok = data(pos, n)
            xd = upload(x, residual=True)
            td = upload(tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if a.layer == 1 else None
            position(pos)
            with no_runtime_host_conversion():
                out = model.prefill_forward(xd, current_positions=pd, slots=sd, tokens=td, page_table=pg)
            validate(out, f"prefix_{pos}", pos, n)
            del out, xd, td
        # State banks suffice: attention writes the last KV entry before reading it;
        # no earlier packed KV/raw-index entry changes at the final position.
        banks = []
        for name in ("recurrent_bank", "conv_bank", "pooled_bank"):
            if hasattr(model.attention, name):
                banks.append(getattr(model.attention, name))
        if model.ple is not None:
            banks += [model.ple_cache.token_history, model.ple_cache.conv_history]
        snapshots = [ttnn.clone(t) for t in banks]

        def restore():
            for saved, t in zip(snapshots, banks):
                ttnn.copy(saved, t)

        x, tok = data(context - 1, 1)
        xd = upload(x, residual=True)
        td = upload(tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if a.layer == 1 else None
        position(context - 1)
        with no_runtime_host_conversion():
            out = model.prefill_forward(xd, current_positions=pd, slots=sd, tokens=td, page_table=pg)
        validate(out, "final_prefill", context - 1, 1)
        del out
        restore()

        def decode():
            with no_runtime_host_conversion():
                return model.decode_forward(xd, current_positions=pd, slots=sd, tokens=td, page_table=pg)

        out = decode()
        del out
        restore()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        manifest["trace_captures"] = 1
        seen = []
        for rep in range(4):
            x, tok = data(context - 1, 1, rep % 3)
            refresh(xd, x, ttnn.bfloat16, residual=True)
            if td is not None:
                refresh(td, tok)
            restore()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            manifest["trace_replays"] += 1
            actual = validate(out, f"final_decode_{rep}", context - 1, 1)
            if rep == 3:
                assert torch.equal(actual, seen[0])
            seen.append(actual)
        manifest.update(
            status="passed",
            wall_seconds=time.monotonic() - start,
            validated_tokens=context,
            minimum_pcc=min((r["pcc"] for r in records if "pcc" in r), default=None),
        )
        print("FULL_CONTEXT_PASS", a.impl, a.layer, context, flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        artifact.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

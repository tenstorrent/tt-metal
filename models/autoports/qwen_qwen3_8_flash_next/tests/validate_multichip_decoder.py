"""Identical real-weight inputs, logical tails and tracked changed-input replay."""

import argparse, hashlib, json, os, time
from pathlib import Path
import torch, ttnn
from transformers import AutoConfig
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder, _hidden_parts
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_decoder import no_runtime_host_conversion
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_components import CHECKPOINT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["single", "mesh"], required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--lengths", default="1,33,129")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--recorded", action="store_true")
    ap.add_argument("--recorded-tag", default="recorded_inputs")
    ap.add_argument("--mutate", action="store_true")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--tag", default="initial")
    ap.add_argument("--reference-tag")
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()
    lengths = [int(v) for v in a.lengths.split(",")]
    torch.set_num_threads(8)
    root = Path("models/autoports/qwen_qwen3_8_flash_next")
    dest = root / os.environ.get("QWEN38_STAGE_DIR", "doc/multichip_decoder") / a.tag
    dest.mkdir(parents=True, exist_ok=True)
    cache = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/multichip")
    cache.mkdir(exist_ok=True)
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    state = CheckpointState(CHECKPOINT, f"model.language_model.layers.{a.layer}.")
    source = root / "tt" / ("optimized_decoder.py" if a.impl == "single" else "multichip_decoder.py")
    manifest = dict(
        args=vars(a),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        env={k: v for k, v in os.environ.items() if k.startswith("TT_METAL_TRACE") or k.startswith("TT_METAL_WATCHER")},
        status="running",
    )
    (dest / f"{a.impl}_l{a.layer}_b{a.batch}_manifest.json").write_text(json.dumps(manifest, indent=2))
    (dest / f"{a.impl}_l{a.layer}_b{a.batch}_source.py.txt").write_bytes(source.read_bytes())
    (dest / "runner.py.txt").write_bytes(Path(__file__).read_bytes())
    if a.impl == "mesh":
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4 if a.impl == "mesh" else 1), trace_region_size=0)
    trace = None
    try:
        model = (MultichipDecoder if a.impl == "mesh" else OptimizedDecoder).from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=a.layer,
            mesh_device=mesh,
            batch_size=a.batch,
            max_requests=a.batch + 1,
            chunk_size=a.chunk,
            cache_dir=Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"),
        )
        model.prepare(lengths)

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, residual=False):
            if a.impl == "mesh":
                return (
                    model.ops.shards(_hidden_parts(x), dtype, layout)
                    if residual
                    else model.ops.upload(x, dtype, layout)
                )
            return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)

        def host(x):
            if a.impl == "single":
                return ttnn.to_torch(x)
            pieces = [ttnn.to_torch(t).reshape(1, a.batch, x.shape[2], 4, 640) for t in ttnn.get_device_tensors(x)]
            return torch.cat(pieces, -1).reshape(1, a.batch, x.shape[2], 10240)

        def refresh(x, data, dtype=ttnn.uint32, residual=False):
            mapper = ttnn.ReplicateTensorToMesh(mesh) if a.impl == "mesh" else None
            if residual and a.impl == "mesh":
                data = torch.cat(_hidden_parts(data), 0)
                mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
            h = ttnn.from_torch(
                data, dtype=dtype, layout=ttnn.TILE_LAYOUT if residual else ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper
            )
            ttnn.copy_host_to_device_tensor(h, x)

        zero = torch.zeros(a.batch, 32, dtype=torch.int32)
        sl = zero.clone()
        sl[:, 0] = torch.arange(1, a.batch + 1)
        pd = upload(zero, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        dp = upload(zero, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        sd = upload(sl, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        rng = torch.Generator().manual_seed(2026)
        page = None
        if a.layer == 3:
            base_page = (
                torch.randperm(a.batch * cfg.max_position_embeddings // 32, generator=rng).reshape(a.batch, -1).int()
            )
            page = upload(base_page, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        fixture = None
        if a.recorded:
            fixture = torch.load(cache / a.recorded_tag / f"layer{a.layer}.pt", weights_only=True)
            recorded_x = torch.cat(fixture["inputs"], dim=1).unsqueeze(0).repeat(1, a.batch, 1, 1)
            recorded_tokens = torch.cat(fixture["tokens"], dim=1).repeat(a.batch, 1)
            if max(lengths) > recorded_x.shape[2]:
                raise ValueError(f"Recorded fixture provides {recorded_x.shape[2]} tokens")
        cases = []
        for length in lengths:
            x = (
                torch.randn(1, a.batch, length, 2560, generator=rng).repeat(1, 1, 1, 4) * 0.05
                + torch.randn(1, a.batch, length, 10240, generator=rng) * 0.005
            ).bfloat16()
            tok = torch.randint(cfg.vocab_size, (a.batch, length), generator=rng, dtype=torch.int32)
            if fixture is not None:
                x, tok = recorded_x[:, :, :length, :], recorded_tokens[:, :length]
            cases.append(
                (
                    length,
                    upload(x, residual=True),
                    upload(tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if a.layer == 1 else None,
                )
            )
        dx = (
            torch.randn(1, a.batch, 1, 2560, generator=rng).repeat(1, 1, 1, 4) * 0.05
            + torch.randn(1, a.batch, 1, 10240, generator=rng) * 0.005
        ).bfloat16()
        if fixture is not None:
            dx = fixture["inputs"][1].unsqueeze(0).repeat(1, a.batch, 1, 1)
        dd = upload(dx, residual=True)
        base_tokens = (
            torch.randint(cfg.vocab_size, (a.batch, 1), generator=rng, dtype=torch.int32) if a.layer == 1 else None
        )
        if fixture is not None and a.layer == 1:
            base_tokens = fixture["tokens"][1].repeat(a.batch, 1)
        dt = upload(base_tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if a.layer == 1 else None

        def prefill(case):
            with no_runtime_host_conversion():
                return model.prefill_forward(case[1], current_positions=pd, slots=sd, tokens=case[2], page_table=page)

        def decode():
            with no_runtime_host_conversion():
                out = model.decode_forward(dd, current_positions=dp, slots=sd, tokens=dt, page_table=page)
                ttnn.plus_one(dp)
                return out

        # Warm every prefill and decode variant before capturing. Temporary
        # outputs are consumed/dropped before a trace can overwrite their storage.
        for case in cases:
            out = prefill(case)
            del out
            pos = zero.clone()
            pos[:, 0] = case[0]
            refresh(dp, pos)
            out = decode()
            del out
            ttnn.synchronize_device(mesh)
            print("WARM_OK", case[0], flush=True)
        out = prefill(cases[0])
        del out
        pos = zero.clone()
        pos[:, 0] = cases[0][0]
        refresh(dp, pos)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        outputs = {}
        rows = []
        output_placements = {}
        for case in cases:
            length = case[0]
            py = prefill(case)
            output_placements[str(length)] = str(py.memory_config())
            outputs[f"prefill_{length}"] = host(py)
            del py
            pos = zero.clone()
            pos[:, 0] = length
            refresh(dp, pos)
            for step in range(a.steps):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                outputs[f"decode_{length}_{step}"] = host(out)
            # Repeat the same prefix and decode sequence to prove exact state reset.
            py = prefill(case)
            assert torch.equal(host(py), outputs[f"prefill_{length}"])
            del py
            refresh(dp, pos)
            for step in range(a.steps):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                assert torch.equal(host(out), outputs[f"decode_{length}_{step}"])
            if a.mutate:
                # Change every captured public input in place; recapture is forbidden.
                altered = (dx.roll(137, -1) * 0.8 + 0.003).bfloat16()
                if fixture is not None:
                    altered = fixture["inputs"][2].unsqueeze(0).repeat(1, a.batch, 1, 1)
                refresh(dd, altered, ttnn.bfloat16, residual=True)
                changed_slots = sl.clone()
                changed_slots[:, 0] = (sl[:, 0] + 1) % (a.batch + 1)
                refresh(sd, changed_slots)
                if dt is not None:
                    refresh(
                        dt,
                        (
                            fixture["tokens"][2].repeat(a.batch, 1)
                            if fixture is not None
                            else torch.full((a.batch, 1), 42 + length, dtype=torch.int32)
                        ),
                    )
                if page is not None:
                    refreshed_page = (
                        torch.randperm(a.batch * cfg.max_position_embeddings // 32, generator=rng)
                        .reshape(a.batch, -1)
                        .int()
                    )
                    refresh(page, refreshed_page, ttnn.int32)
                py = prefill(case)
                del py
                refresh(dp, pos)
                for step in range(a.steps):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    outputs[f"mutated_{length}_{step}"] = host(out)
                py = prefill(case)
                del py
                refresh(dp, pos)
                for step in range(a.steps):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    assert torch.equal(host(out), outputs[f"mutated_{length}_{step}"])
                assert not torch.equal(outputs[f"mutated_{length}_0"], outputs[f"decode_{length}_0"])
                refresh(dd, dx, ttnn.bfloat16, residual=True)
                refresh(sd, sl)
                if dt is not None:
                    refresh(dt, base_tokens)
                if page is not None:
                    refresh(page, base_page, ttnn.int32)
            print("CASE_OK", length, flush=True)
        ttnn.release_trace(mesh, trace)
        trace = None
        path = cache / f"{a.tag}_{a.impl}_l{a.layer}_b{a.batch}.pt"
        torch.save(outputs, path)
        if a.impl == "mesh":
            ref = torch.load(cache / f"{a.reference_tag or a.tag}_single_l{a.layer}_b{a.batch}.pt", weights_only=True)
            for name, value in outputs.items():
                metric = pcc_metrics(ref[name], value)
                rows.append(dict(case=name, **metric))
                print(json.dumps(rows[-1]), flush=True)
            manifest.update(metrics=rows, outputs=str(path), status="failed_pcc")
            assert min(r["pcc"] for r in rows) >= 0.995
        manifest.update(
            status="passed",
            metrics=rows,
            outputs=str(path),
            replays=len(cases) * a.steps * (4 if a.mutate else 2),
            captures=1,
            releases=1,
            prefill_output_placements=output_placements,
        )
        print("DECODER_PASS", a.impl, a.layer, a.batch, flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        (dest / f"{a.impl}_l{a.layer}_b{a.batch}_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

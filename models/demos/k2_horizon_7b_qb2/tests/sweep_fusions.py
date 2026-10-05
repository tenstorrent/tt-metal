"""Paired layer PCC and warmed timing. Host trace times are explicitly host time."""

import argparse
import hashlib
import inspect
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn

from .fusion_candidates import CANDIDATES
from .fusion_v1 import InitialFusion as FusedDecoder
from .run_functional import load_reference, pcc, to_device


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    p.add_argument("--output", required=True)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(16)
    config, state, hf, rope_fn = load_reference()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    records = []
    try:
        layer = FusedDecoder.from_state_dict(state, hf_config=config, layer_idx=0, mesh_device=mesh)
        layer.wgateup = to_device(
            torch.cat(
                [state["model.layers.0.mlp.gate_proj.weight"].T, state["model.layers.0.mlp.up_proj.weight"].T], -1
            ),
            mesh,
        )
        if any(name.startswith("fold_norm") for name in a.candidates):
            gamma1 = state["model.layers.0.input_layernorm.weight"].bfloat16().float()
            gamma2 = state["model.layers.0.post_attention_layernorm.weight"].bfloat16().float()

            def folded(key, gamma):
                return (state["model.layers.0." + key].bfloat16().float().T * gamma[:, None]).bfloat16()

            qkv = torch.cat([folded("self_attn." + p + "_proj.weight", gamma1) for p in "qkv"], -1)
            gate = folded("mlp.gate_proj.weight", gamma2)
            up = folded("mlp.up_proj.weight", gamma2)
            for name, tensor in [("wqkv", qkv), ("wgate", gate), ("wup", up), ("wgateup", torch.cat([gate, up], -1))]:
                setattr(layer, "folded_" + name, to_device(tensor, mesh))
        torch.manual_seed(600)
        seq = a.seq
        x = (torch.randn(a.batch, seq, 4096) * 0.03).bfloat16()
        rope = rope_fn(x, torch.arange(seq)[None].expand(a.batch, -1))
        tx = to_device(x[None], mesh)
        tr = tuple(to_device(r[:, None], mesh) for r in rope)
        pages = math.ceil((seq + 128) / 32)
        table = to_device(torch.randperm(a.batch * pages).reshape(a.batch, pages).int(), mesh, True)
        caches = tuple(
            ttnn.zeros((a.batch * pages, 8, 32, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            for _ in range(2)
        )
        dx = (torch.randn(a.batch, 1, 4096) * 0.03).bfloat16()
        dr = rope_fn(dx, torch.full((a.batch, 1), seq, dtype=torch.int64))
        td = to_device(dx.transpose(0, 1)[None], mesh)
        rr = tuple(to_device(r[None].repeat(1, 1, 32, 1), mesh) for r in dr)
        pos = to_device(torch.full((a.batch,), seq, dtype=torch.int32), mesh, True)
        layer.expert_ids = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32), device=mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        layer.expert_counts = {
            n: ttnn.from_torch(
                torch.tensor([n], dtype=torch.int32), device=mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            for n in (1, 32, 256)
        }
        layer.expert_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        baseline = {}
        for name in a.candidates:
            if name in ("delivered", "before_v"):
                candidate = CANDIDATES[name].from_state_dict(state, hf_config=config, layer_idx=0, mesh_device=mesh)
            else:
                candidate = CANDIDATES[name]()
                candidate.__dict__.update(layer.__dict__)
            if hasattr(candidate, "configure"):
                candidate.configure()
            rec = {
                "candidate": name,
                "seq": seq,
                "batch": a.batch,
                "implementation": type(candidate).__module__ + "." + type(candidate).__name__,
                "source_sha256": hashlib.sha256(Path(inspect.getfile(type(candidate))).read_bytes()).hexdigest(),
            }
            print("START", name, flush=True)
            try:
                plan = candidate.prepare_prefill(seq_len=seq)
                kw = dict(rope=tr, kv_cache=caches, page_table=table, plan=plan)
                out = candidate.prefill_forward(tx, **kw)
                host = ttnn.to_torch(out)
                out.deallocate(True)
                if "prefill" not in baseline:
                    baseline["prefill"] = host
                rec["prefill_pcc"] = pcc(host, baseline["prefill"])
                assert rec["prefill_pcc"] >= 0.995
                times = []
                for _ in range(3):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    out = candidate.prefill_forward(tx, **kw)
                    ttnn.synchronize_device(mesh)
                    times.append((time.perf_counter() - start) * 1e6)
                    out.deallocate(True)
                rec["prefill_host_us"] = statistics.median(times)
                dkw = dict(rope=rr, kv_cache=caches, page_table=table, current_pos=pos)
                out = candidate.decode_forward(td, **dkw)
                host = ttnn.to_torch(out)
                out.deallocate(True)
                if "decode" not in baseline:
                    baseline["decode"] = host
                rec["decode_pcc"] = pcc(host, baseline["decode"])
                assert rec["decode_pcc"] >= 0.995
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = candidate.decode_forward(td, **dkw)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                assert pcc(ttnn.to_torch(out), host) >= 0.99999
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    for _ in range(100):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times.append((time.perf_counter() - start) * 1e6 / 100)
                rec["decode_traced_host_us"] = statistics.median(times)
                rec["decode_traced_host_samples_us"] = times
                assert torch.equal(ttnn.to_torch(out), host)
                ttnn.release_trace(mesh, trace)
                out.deallocate(True)
                rec["passed"] = True
            except Exception as e:
                rec["error"] = str(e)
                rec["passed"] = False
            records.append(rec)
            print(json.dumps(rec), flush=True)
            Path(a.output).write_text(
                json.dumps(
                    {
                        "records": records,
                        "invocation": sys.argv,
                        "timing_basis": "host elapsed; 100 asynchronous trace replays then device sync; 5 repetitions",
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

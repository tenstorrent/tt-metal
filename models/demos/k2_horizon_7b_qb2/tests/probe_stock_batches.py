"""Numerical control for batch-dependent split-K at bounded stock-decode limit."""

import argparse
import json
from pathlib import Path

import torch
from transformers import DynamicCache

import ttnn

from ..tt.functional_decoder import FunctionalDecoder
from ..tt.fused_decoder import FusedDecoder
from .run_functional import load_reference, pcc, to_device


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    a = parser.parse_args()
    torch.set_num_threads(16)
    config, state, hf, rope_fn = load_reference()
    n = a.context
    if n < 32 or n % 32:
        raise ValueError("This diagnostic probes page-aligned capacities")
    torch.manual_seed(3702)
    keys = []
    values = []
    for start in range(0, n, 4096):
        count = min(4096, n - start)
        x = (torch.randn(1, count, 4096) * 0.03).bfloat16()
        rr = rope_fn(x, torch.arange(start, start + count)[None])
        norm = hf.input_layernorm(x)
        k = hf.self_attn.k_proj(norm).reshape(1, -1, 8, 128).transpose(1, 2)
        keys.append(k * rr[0][:, None] + torch.cat([-k[..., 64:], k[..., :64]], -1) * rr[1][:, None])
        values.append(hf.self_attn.v_proj(norm).reshape(1, -1, 8, 128).transpose(1, 2))
    k = torch.cat(keys, 2)
    v = torch.cat(values, 2)
    del keys, values, norm, x, rr
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    evidence = {
        "context": n,
        "prefix": "Real target weights and HF K/V; shared prefix across batch, independent last-token inputs.",
        "records": [],
    }
    try:
        functional = FunctionalDecoder.from_state_dict(state, hf_config=config, layer_idx=0, mesh_device=mesh)
        fused = FusedDecoder.from_state_dict(state, hf_config=config, layer_idx=0, mesh_device=mesh)
        for batch in a.batches:
            dx = (torch.randn(batch, 1, 4096) * 0.03).bfloat16()
            positions = torch.full((batch,), n - 1, dtype=torch.int32)
            rope = rope_fn(dx, positions[:, None].long())
            ref = DynamicCache(config=config)
            ref.update(k[:, :, : n - 1].repeat(batch, 1, 1, 1), v[:, :, : n - 1].repeat(batch, 1, 1, 1), 0)
            expected = hf(dx, position_embeddings=rope, past_key_values=ref)
            del ref
            table = torch.randperm(batch * n // 32).reshape(batch, n // 32).int()
            pt = to_device(table, mesh, True)
            caches = []
            for data in (k, v):
                physical = torch.empty(batch * n // 32, 8, 32, 128, dtype=torch.bfloat16)
                logical = data.reshape(8, n // 32, 32, 128).permute(1, 0, 2, 3)
                for b in range(batch):
                    physical[table[b].long()] = logical
                caches.append(to_device(physical, mesh))
                del physical
            td = to_device(dx.transpose(0, 1)[None], mesh)
            tr = tuple(to_device(r[None].repeat(1, 1, 32, 1), mesh) for r in rope)
            tp = to_device(positions, mesh, True)
            kw = dict(rope=tr, kv_cache=caches, page_table=pt, current_pos=tp)
            hosts = {}
            for name, layer in [("functional", functional), ("fused", fused)]:
                out = layer.decode_forward(td, **kw)
                actual = ttnn.to_torch(out).reshape_as(expected)
                out.deallocate(True)
                score = pcc(actual, expected)
                rel = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
                ratio = (actual.float().norm() / expected.float().norm()).item()
                row = {"batch": batch, "implementation": name, "pcc": score, "relative_l2": rel, "norm_ratio": ratio}
                hosts[name] = actual
                evidence["records"].append(row)
                print(json.dumps(row), flush=True)
                assert score >= 0.995
            evidence["records"][-1]["vs_functional_pcc"] = pcc(hosts["fused"], hosts["functional"])
            evidence["records"][-1]["relative_l2_vs_functional"] = (
                (hosts["fused"].float() - hosts["functional"].float()).norm() / hosts["functional"].float().norm()
            ).item()
            del kw, caches, td, tr, tp, pt, hosts
            Path(a.output).write_text(json.dumps(evidence, indent=2) + "\n")
        evidence["completed"] = True
    finally:
        ttnn.close_mesh_device(mesh)
        Path(a.output).write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()

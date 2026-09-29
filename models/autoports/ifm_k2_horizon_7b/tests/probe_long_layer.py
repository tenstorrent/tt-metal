"""Diagnosis only: real HF prefix K/V, final TT prefill/decode at full context.

This is not a substitute for streaming every TT prefix token in run_long_context.
"""

import argparse
import json

import torch
from transformers import DynamicCache

import ttnn

from ..tt.functional_decoder import FunctionalDecoder
from .run_functional import DOC, load_reference, pcc, to_device


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="debug_long/layer_probe.json")
    parser.add_argument("--tails", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(16)
    config, state, hf, hf_rope = load_reference()
    n = config.max_position_embeddings
    torch.manual_seed(948)
    keys = torch.empty(1, 8, n, 128, dtype=torch.bfloat16)
    values = torch.empty_like(keys)
    for start in range(0, n, 4096):
        x = (torch.randn(1, 4096, 4096) * 0.03).bfloat16()
        rope = hf_rope(x, torch.arange(start, start + 4096)[None])
        norm = hf.input_layernorm(x)
        k = hf.self_attn.k_proj(norm).reshape(1, -1, 8, 128).transpose(1, 2)
        keys[:, :, start : start + 4096] = k * rope[0].unsqueeze(1) + torch.cat([-k[..., 64:], k[..., :64]], -1) * rope[
            1
        ].unsqueeze(1)
        values[:, :, start : start + 4096] = hf.self_attn.v_proj(norm).reshape(1, -1, 8, 128).transpose(1, 2)
        if (start + 4096) % 65536 == 0:
            print("HF_PREFIX", start + 4096, flush=True)
    ref = DynamicCache(config=config)
    ref.update(keys[:, :, : n - 32], values[:, :, : n - 32], 0)
    mask = torch.where(
        torch.arange(n)[None, :] <= torch.arange(n - 32, n)[:, None], 0.0, torch.finfo(torch.bfloat16).min
    ).bfloat16()[None, None]
    expected = hf(
        x[:, -32:], position_embeddings=tuple(r[:, -32:] for r in rope), attention_mask=mask, past_key_values=ref
    )
    del ref, mask, norm, k
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    evidence = {"scope": "diagnostic: HF-initialized full prefix; final TT chunk only", "context": n}
    try:
        layer = FunctionalDecoder.from_state_dict(state, hf_config=config, layer_idx=0, mesh_device=mesh)
        del state
        table = torch.randperm(n // 32).int()[None]
        pt = to_device(table, mesh, True)
        caches = []
        for data in (keys, values):
            physical = torch.empty_like(data.reshape(n // 32, 8, 32, 128))
            physical[table[0].long()] = data.reshape(8, n // 32, 32, 128).permute(1, 0, 2, 3)
            caches.append(to_device(physical, mesh))
        del physical
        tx = to_device(x[:, -256:].unsqueeze(0), mesh)
        tr = tuple(to_device(r[:, -256:].unsqueeze(1), mesh) for r in rope)
        out = layer.prefill_forward(
            tx, rope=tr, kv_cache=caches, page_table=pt, plan=layer.prepare_prefill(seq_len=256, start_pos=n - 256)
        )
        actual = ttnn.to_torch(out)[:, :, -32:].reshape_as(expected)
        evidence["prefill_pcc"] = pcc(actual, expected)
        evidence["prefill_relative_l2"] = ((actual.float() - expected.float()).norm() / expected.float().norm()).item()
        print(json.dumps(evidence), flush=True)
        out.deallocate(True)
        if args.tails:
            evidence["tails"] = []
            for count in (1, 31, 32):
                tx = to_device(x[:, -32:][:, :count].unsqueeze(0), mesh)
                tr = tuple(to_device(r[:, -32:][:, :count].unsqueeze(1), mesh) for r in rope)
                out = layer.prefill_forward(
                    tx,
                    rope=tr,
                    kv_cache=caches,
                    page_table=pt,
                    plan=layer.prepare_prefill(seq_len=count, start_pos=n - 32),
                )
                actual = ttnn.to_torch(out).reshape_as(expected[:, :count])
                row = {
                    "start_pos": n - 32,
                    "seq_len": count,
                    "context": n - 32 + count,
                    "pcc": pcc(actual, expected[:, :count]),
                    "relative_l2": (
                        (actual.float() - expected[:, :count].float()).norm() / expected[:, :count].float().norm()
                    ).item(),
                }
                evidence["tails"].append(row)
                print("LONG_TAIL", json.dumps(row), flush=True)
                assert row["pcc"] >= 0.995
                out.deallocate(True)
        dx = to_device(x[:, -1:].unsqueeze(0), mesh)
        dr = tuple(to_device(r[:, -1:].unsqueeze(0).repeat(1, 1, 32, 1), mesh) for r in rope)
        pos = to_device(torch.tensor([n - 1], dtype=torch.int32), mesh, True)
        out = layer.decode_forward(dx, rope=dr, kv_cache=caches, page_table=pt, current_pos=pos)
        actual = ttnn.to_torch(out).reshape_as(expected[:, -1:])
        evidence["decode_pcc"] = pcc(actual, expected[:, -1:])
        evidence["decode_relative_l2"] = (
            (actual.float() - expected[:, -1:].float()).norm() / expected[:, -1:].float().norm()
        ).item()
        print(json.dumps(evidence), flush=True)
        if args.tails:
            assert min(evidence["prefill_pcc"], evidence["decode_pcc"]) >= 0.995
            evidence["completed"] = True
    finally:
        ttnn.close_mesh_device(mesh)
        (DOC / args.output).write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()

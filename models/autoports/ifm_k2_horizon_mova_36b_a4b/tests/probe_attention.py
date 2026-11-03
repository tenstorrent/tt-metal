# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: isolate long SDPA from projection/router/cache-fill errors."""

import argparse
import importlib
import json

import torch
from transformers.cache_utils import DynamicCache

import ttnn

from .run_decoder import ROOT, pcc, reference

ap = argparse.ArgumentParser()
ap.add_argument("--context", type=int, default=262144)
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--k-chunks", type=int, nargs="+", default=[32, 128, 256, 512])
ap.add_argument("--decoder", action="store_true")
args = ap.parse_args()
torch.set_num_threads(16)
torch.manual_seed(812)
c, hf, rope, sd = reference(args.layer)
pattern = (torch.randn(1, 256, 2560) * 0.1).bfloat16()
mod = importlib.import_module(type(hf).__module__)
length = args.context
start = length - 256
with torch.no_grad():
    norm = hf.input_layernorm(pattern)
    kbase = hf.self_attn.k_proj(norm).reshape(1, 256, 8, 128).transpose(1, 2)
    qbase = hf.self_attn.q_proj(norm).reshape(1, 256, 32, 128).transpose(1, 2)
    cache = DynamicCache(config=c)
    hf.self_attn(
        norm, position_embeddings=rope(pattern, torch.arange(256)[None]), attention_mask=None, past_key_values=cache
    )
    vbase = cache.layers[args.layer].values
    cs = rope(pattern, torch.arange(length)[None])
    k = kbase.repeat(1, 1, length // 256, 1)
    k = k * cs[0][:, None] + mod.rotate_half(k) * cs[1][:, None]
    v = vbase.repeat(1, 1, length // 256, 1).contiguous()
    q = qbase * cs[0][:, None, start:] + mod.rotate_half(qbase) * cs[1][:, None, start:]
    mask = torch.arange(length)[None, :] > torch.arange(start, length)[:, None]
    mask = torch.zeros(256, length).masked_fill(mask, float("-inf"))[None, None]
    golden = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
print("REFERENCE_READY", flush=True)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
try:

    def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(v, device=mesh, dtype=dtype, layout=layout)

    def cpu(v):
        return ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

    tq = tt(q)
    tk, tv = [tt(a[0].reshape(8, length // 32, 32, 128).permute(1, 0, 2, 3).contiguous()) for a in (k, v)]
    pt = tt(torch.arange(length // 32, dtype=torch.int32)[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    pos = tt(torch.tensor([start], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    rows = []
    for kchunk in args.k_chunks:
        compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            pt,
            chunk_start_idx=None,
            chunk_start_idx_tensor=pos,
            compute_kernel_config=compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=kchunk, exp_approx_mode=False
            ),
        )
        got = cpu(out)
        row = dict(
            k_chunk=kchunk,
            context=length,
            pcc=pcc(golden, got),
            mean_abs_error=(golden.float() - got.float()).abs().mean().item(),
            reference_rms=golden.float().square().mean().sqrt().item(),
            actual_rms=got.float().square().mean().sqrt().item(),
        )
        rows.append(row)
        print("ATTENTION_PROBE", json.dumps(row), flush=True)
        out.deallocate(True)
    (ROOT / f"doc/functional_decoder/attention_probe_{length}_{args.k_chunks[0]}.json").write_text(
        json.dumps(rows, indent=2)
    )
    if args.decoder:
        from unittest.mock import patch

        from ..tt.functional_decoder import FunctionalDecoder

        prefix = DynamicCache(config=c)
        prefix.update(k[:, :, :start].clone(), v[:, :, :start].clone(), args.layer)
        with torch.no_grad():
            expected = hf(
                pattern,
                position_embeddings=(cs[0][:, start:], cs[1][:, start:]),
                attention_mask=mask,
                past_key_values=prefix,
            )
        model = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        tx = tt(pattern.unsqueeze(0))
        tc, ts = [tt(a[:, None, start:]) for a in cs]
        original = ttnn.SDPAProgramConfig
        for kchunk in args.k_chunks:

            def config(**kw):
                kw["k_chunk_size"] = kchunk
                return original(**kw)

            with patch.object(ttnn, "SDPAProgramConfig", config):
                out = model._prefill_chunk(tx, cos=tc, sin=ts, page_table=pt, kv_cache=[tk, tv], start_pos=start)
            score = pcc(expected, cpu(out).reshape(1, 256, 2560))
            print("DECODER_PROBE", kchunk, length, score, flush=True)
            out.deallocate(True)
        dq = tt(q[:, :, -1:, :].transpose(1, 2).contiguous())
        dpos = tt(torch.tensor([length - 1], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        for kchunk in [32, 256, 1024]:
            out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                dq,
                tk,
                tv,
                pt,
                cur_pos_tensor=dpos,
                compute_kernel_config=compute,
                program_config=original(
                    compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=kchunk, exp_approx_mode=False
                ),
            )
            print(
                "DECODE_ATTENTION_PROBE",
                kchunk,
                length,
                pcc(golden[:, :, -1:, :].transpose(1, 2), cpu(out)),
                flush=True,
            )
            out.deallocate(True)
finally:
    ttnn.close_mesh_device(mesh)

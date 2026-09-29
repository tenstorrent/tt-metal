"""Numerical range of stock paged prefill attention on actual layer Q/K/V."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn

from ..tt.accurate_attention import accurate_attention
from .run_functional import load_reference, pcc, to_device
from .sweep_optimized import real_activations


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--contexts", nargs="+", type=int, default=[4096, 8192, 16384, 32768, 65536])
    a = p.parse_args()
    torch.set_num_threads(16)
    cfg, state, hf, rope = load_reference()
    del state
    acts = real_activations(8192)[None]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        for length in a.contexts:
            chunks = []
            for start in range(0, length, 4096):
                x = acts[:, start % 8192 : start % 8192 + min(4096, length - start)]
                cos, sin = rope(x, torch.arange(start, start + x.shape[1])[None])
                norm = hf.input_layernorm(x)
                k = hf.self_attn.k_proj(norm).reshape(1, -1, 8, 128).transpose(1, 2)
                k = k * cos[:, None] + torch.cat([-k[..., 64:], k[..., :64]], -1) * sin[:, None]
                v = hf.self_attn.v_proj(norm).reshape(1, -1, 8, 128).transpose(1, 2)
                chunks.append((k, v))
            keys = torch.cat([kv[0] for kv in chunks], 2)
            vals = torch.cat([kv[1] for kv in chunks], 2)
            pages = length // 32
            table = torch.arange(pages, dtype=torch.int32)[None]
            pt = to_device(table, mesh, True)
            caches = tuple(
                ttnn.from_torch(
                    t.reshape(8, pages, 32, 128).permute(1, 0, 2, 3).contiguous(),
                    device=mesh,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                )
                for t in (keys, vals)
            )
            for count in [32, 128, 256]:
                q = hf.self_attn.q_proj(norm[:, -count:]).reshape(1, count, 32, 128).transpose(1, 2)
                c, s = cos[:, -count:][:, None], sin[:, -count:][:, None]
                q = q * c + torch.cat([-q[..., 64:], q[..., :64]], -1) * s
                mask = torch.where(
                    torch.arange(length)[None, :] <= torch.arange(length - count, length)[:, None], 0.0, float("-inf")
                ).bfloat16()[None, None]
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q, keys.repeat_interleave(4, 1), vals.repeat_interleave(4, 1), attn_mask=mask
                )
                qt = to_device(q, mesh)
                for kind in ["accurate", "stock"]:

                    def forward():
                        if kind == "accurate":
                            return accurate_attention(
                                qt,
                                *caches,
                                pt,
                                chunk_start_idx=length - count,
                                q_chunk_size=min(count, 128),
                                k_chunk_size=min(count, 128),
                            )
                        return ttnn.transformer.chunked_scaled_dot_product_attention(
                            qt,
                            *caches,
                            pt,
                            chunk_start_idx=length - count,
                            program_config=ttnn.SDPAProgramConfig(
                                compute_with_storage_grid_size=(11, 10),
                                q_chunk_size=min(count, 128),
                                k_chunk_size=count,
                                exp_approx_mode=False,
                            ),
                            compute_kernel_config=compute,
                        )

                    y = forward()
                    actual = ttnn.to_torch(y)
                    y.deallocate(True)
                    times = []
                    for _ in range(3):
                        start = time.perf_counter()
                        y = forward()
                        ttnn.synchronize_device(mesh)
                        times.append((time.perf_counter() - start) * 1e6)
                        y.deallocate(True)
                    row = dict(
                        context=length,
                        count=count,
                        kind=kind,
                        pcc=pcc(actual, expected),
                        relative_l2=((actual.float() - expected.float()).norm() / expected.float().norm()).item(),
                        host_us=statistics.median(times),
                    )
                    rows.append(row)
                    print(row, flush=True)
                    Path(a.output).write_text(
                        json.dumps(
                            {
                                "records": rows,
                                "scope": "attention only on real-weight QKV from checkpoint embeddings, BFP8 KV",
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                qt.deallocate(True)
            for t in caches:
                t.deallocate(True)
            pt.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

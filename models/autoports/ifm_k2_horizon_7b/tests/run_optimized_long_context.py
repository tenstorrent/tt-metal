"""Stream the full advertised context through the actual layer on one device.

HF projects every input into the canonical K/V cache. At checkpoints the real
HF decoder computes the final 32 query outputs against the complete prefix.
This bounds host reference attention memory without reducing TT work/context.
"""

import argparse
import json

import torch
from transformers import DynamicCache

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder
from .run_optimized import DOC, load_reference, pcc, to_device
from .sweep_optimized import real_activations


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="long_context.json")
    args = parser.parse_args()
    torch.set_num_threads(16)
    config, state, hf, hf_rope = load_reference()
    limit = config.max_position_embeddings
    evidence = {
        "advertised_context": limit,
        "reference_scope": "final 32 query outputs at checkpoints; full HF K/V",
        "checkpoints": [],
        "complete": False,
    }
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        layer = OptimizedDecoder.from_state_dict(state, hf_config=config, layer_idx=0, mesh_device=mesh)
        del state
        torch.manual_seed(948)
        table = torch.randperm(limit // 32).reshape(1, -1).int()
        tt_table = to_device(table, mesh, True)
        caches = tuple(
            ttnn.zeros(
                (limit // 32, 8, 32, 128),
                dtype=layer.kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        )
        keys = torch.empty(1, 8, limit, 128, dtype=torch.bfloat16)
        values = torch.empty_like(keys)
        recorded = real_activations(8192)[None]
        start = 0
        recent_x = None
        recent_rope = None
        checkpoints = [4096, 32768, 65536, limit - 31, limit]
        for checkpoint in checkpoints:
            while start < checkpoint:
                end = min(start + 4096, checkpoint)
                x = recorded[:, start % 4096 : start % 4096 + end - start].contiguous()
                positions = torch.arange(start, end).unsqueeze(0)
                rope = hf_rope(x, positions)
                recent_x = x[:, -256:] if recent_x is None else torch.cat([recent_x, x], dim=1)[:, -256:]
                recent_rope = (
                    tuple(r[:, -256:] for r in rope)
                    if recent_rope is None
                    else tuple(torch.cat([old, new], dim=1)[:, -256:] for old, new in zip(recent_rope, rope))
                )
                normalized = hf.input_layernorm(x)
                k = hf.self_attn.k_proj(normalized).reshape(1, -1, 8, 128).transpose(1, 2)
                v = hf.self_attn.v_proj(normalized).reshape(1, -1, 8, 128).transpose(1, 2)
                rotated = torch.cat([-k[..., 64:], k[..., :64]], dim=-1)
                k = k * rope[0].unsqueeze(1) + rotated * rope[1].unsqueeze(1)
                keys[:, :, start:end, :] = k
                values[:, :, start:end, :] = v
                tx = to_device(x.unsqueeze(0), mesh)
                tr = tuple(to_device(r.unsqueeze(1), mesh) for r in rope)
                plan = layer.prepare_prefill(seq_len=end - start, start_pos=start)
                out = layer.prefill_forward(tx, rope=tr, kv_cache=caches, page_table=tt_table, plan=plan)
                ttnn.synchronize_device(mesh)
                if end == checkpoint:
                    count = min(32, end - start)
                    reference_cache = DynamicCache(config=config)
                    reference_cache.update(keys[:, :, : end - count, :], values[:, :, : end - count, :], 0)
                    mask = torch.where(
                        torch.arange(end)[None, :] <= torch.arange(end - count, end)[:, None],
                        0.0,
                        torch.finfo(torch.bfloat16).min,
                    ).bfloat16()[None, None]
                    expected = hf(
                        x[:, -count:, :],
                        position_embeddings=tuple(r[:, -count:, :] for r in rope),
                        attention_mask=mask,
                        past_key_values=reference_cache,
                    )
                    actual = ttnn.to_torch(out[:, :, -count:, :]).reshape_as(expected)
                    score = pcc(actual, expected)
                    print(f"LONG_PREFILL context={end} samples={count} pcc={score}", flush=True)
                    evidence["checkpoints"].append(
                        {
                            "context": end,
                            "prefill_pcc": score,
                            "query_samples": count,
                            "passed": score >= 0.995,
                            "relative_l2": (
                                (actual.float() - expected.float()).norm() / expected.float().norm()
                            ).item(),
                            "leading_decode_tokens": len(plan.leading_positions),
                        }
                    )
                    assert score >= 0.995, score
                    del reference_cache, mask, actual
                out.deallocate(True)
                del tx, tr, plan, out, normalized, k, v, rotated
                start = end
                print(f"PROGRESS {end}/{limit}", flush=True)
            (DOC / args.output).write_text(json.dumps(evidence, indent=2) + "\n")
        # Also exercise an aligned chunk ending at the advertised maximum. The
        # preceding arbitrary continuation used 31 leading decode tokens.
        tx = to_device(recent_x.unsqueeze(0), mesh)
        tr = tuple(to_device(r.unsqueeze(1), mesh) for r in recent_rope)
        out = layer.prefill_forward(
            tx,
            rope=tr,
            kv_cache=caches,
            page_table=tt_table,
            plan=layer.prepare_prefill(seq_len=256, start_pos=limit - 256),
        )
        actual = ttnn.to_torch(out[:, :, -31:, :]).reshape_as(expected)
        evidence["aligned_final_chunk_pcc"] = pcc(actual, expected)
        assert evidence["aligned_final_chunk_pcc"] >= 0.995
        out.deallocate(True)
        # Validate logical-to-physical mapping and numeric K/V at distant pages.
        evidence["sampled_cache_pages"] = []
        for logical_page in (0, limit // 64, limit // 32 - 1):
            physical_page = int(table[0, logical_page])
            scores = []
            for cache, reference in zip(caches, (keys, values)):
                sample = ttnn.to_torch(cache[physical_page : physical_page + 1])
                target = reference[:, :, logical_page * 32 : (logical_page + 1) * 32]
                scores.append(pcc(sample, target))
            evidence["sampled_cache_pages"].append(
                {
                    "logical_page": logical_page,
                    "physical_page": physical_page,
                    "key_pcc": scores[0],
                    "value_pcc": scores[1],
                }
            )
            assert min(scores) >= 0.995
        # Recompute the legal final token against the completely filled cache.
        dx = x[:, -1:, :]
        dr = tuple(r[:, -1:, :].unsqueeze(0).repeat(1, 1, 32, 1) for r in rope)
        td = to_device(dx.unsqueeze(0), mesh)
        tr = tuple(to_device(r, mesh) for r in dr)
        tp = to_device(torch.tensor([limit - 1], dtype=torch.int32), mesh, True)
        kwargs = dict(rope=tr, kv_cache=caches, page_table=tt_table, current_pos=tp)
        warm = layer.decode_forward(td, **kwargs)
        warm.deallocate(True)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = layer.decode_forward(td, **kwargs)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        actual = ttnn.to_torch(out).reshape(1, 1, 4096)
        score = pcc(actual, expected[:, -1:, :])
        assert score >= 0.995, score
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        assert torch.equal(actual, ttnn.to_torch(out).reshape_as(actual))
        ttnn.release_trace(mesh, trace)
        evidence.update(
            complete=True, decode_context=limit, decode_pcc=score, decode_traced=True, decode_deterministic=True
        )
        print(f"LONG_DECODE context={limit} pcc={score}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        (DOC / args.output).write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()

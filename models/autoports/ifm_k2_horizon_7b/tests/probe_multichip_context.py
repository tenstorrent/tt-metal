"""HF-initialized TP4 cache controls near decode dispatch/context boundaries.

Only short TT suffixes and two decode positions execute. This is NOT evidence
that TP4 computed a full prefix; run_multichip_long_context supplies that proof.
Imports of TTNN and model/reference modules occur only inside run().
"""

import argparse
import hashlib
import json
from pathlib import Path


def run(args):
    import gc

    import torch
    from transformers import DynamicCache

    import ttnn

    from ..tt.multichip_decoder import MultichipDecoder
    from .run_functional import load_reference, pcc
    from .run_multichip import read, refresh, upload
    from .sweep_optimized import real_activations

    torch.set_grad_enabled(False)
    torch.set_num_threads(16)
    config, state, hf, hf_rope = load_reference()
    if any(context < 32 or context % 32 or context > config.max_position_embeddings for context in args.contexts):
        raise ValueError("Contexts must be page aligned and within the advertised limit")
    span = args.suffix_span
    if any(context < span for context in args.contexts):
        raise ValueError("Context must contain the requested suffix span")
    maximum = max(args.contexts)
    evidence = {
        "scope": "diagnostic: HF-initialized prefix, short TP4 suffixes and decode only",
        "full_tt_prefix_computed": False,
        "complete": False,
        "advertised_context": config.max_position_embeddings,
        "command_args": {**vars(args), "output": str(args.output)},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "tt/multichip_decoder.py").read_bytes()
        ).hexdigest(),
        "contexts": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    recorded = real_activations(4096)[None]
    normalized = hf.input_layernorm(recorded)
    base_k = hf.self_attn.k_proj(normalized).reshape(1, -1, 8, 128).transpose(1, 2)
    base_v = hf.self_attn.v_proj(normalized).reshape(1, -1, 8, 128).transpose(1, 2)
    keys = torch.empty(1, 8, maximum, 128, dtype=torch.bfloat16)
    values = torch.empty_like(keys)
    for start in range(0, maximum, 4096):
        end = min(start + 4096, maximum)
        rope = hf_rope(recorded[:, : end - start], torch.arange(start, end)[None])
        k = base_k[:, :, : end - start]
        keys[:, :, start:end] = k * rope[0].unsqueeze(1) + torch.cat([-k[..., 64:], k[..., :64]], -1) * rope[
            1
        ].unsqueeze(1)
        values[:, :, start:end] = base_v[:, :, : end - start]
        if end % 65536 == 0 or end == maximum:
            print("HF_PREFIX_INITIALIZED", end, flush=True)
    del normalized, base_k, base_v, k, rope

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    trace = None
    original_stock = ttnn.transformer.chunked_scaled_dot_product_attention
    try:
        layer = MultichipDecoder.from_state_dict(state, hf_config=config, layer_idx=0, mesh_device=mesh)
        del state
        accurate_calls = []
        stock_calls = []

        def observed_stock(q, *args, **kwargs):
            program = kwargs["program_config"]
            stock_calls.append(
                {
                    "q_shape": list(q.shape),
                    "q_chunk_size": program.q_chunk_size,
                    "k_chunk_size": program.k_chunk_size,
                    "start_pos": kwargs["chunk_start_idx"],
                }
            )
            return original_stock(q, *args, **kwargs)

        ttnn.transformer.chunked_scaled_dot_product_attention = observed_stock
        accurate = layer._attention

        def observed_attention(q, k, v, table, **kwargs):
            accurate_calls.append(
                {
                    "q_shape": list(q.shape),
                    "kv_shape": list(k.shape),
                    "tensor_offset": "chunk_start_idx_tensor" in kwargs,
                    "q_chunk_size": kwargs.get("q_chunk_size"),
                    "k_chunk_size": kwargs.get("k_chunk_size"),
                }
            )
            return accurate(q, k, v, table, **kwargs)

        layer._attention = observed_attention

        def metrics(actual, expected):
            score = pcc(actual, expected)
            result = {
                "pcc": score,
                "relative_l2": ((actual.float() - expected.float()).norm() / expected.float().norm()).item(),
                "passed": score >= 0.995,
            }
            assert result["passed"], result
            return result

        for context in args.contexts:
            torch.manual_seed(948 + context)
            pages = context // 32
            table = torch.randperm(pages).int()[None]
            tt_table = upload(table, mesh, integer=True)
            caches = []
            for data in (keys, values):
                physical = torch.empty(pages, 8, 32, 128, dtype=torch.bfloat16)
                physical[table[0].long()] = data[:, :, :context].reshape(8, pages, 32, 128).permute(1, 0, 2, 3)
                caches.append(
                    ttnn.from_torch(
                        physical,
                        device=mesh,
                        dtype=layer.kv_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
                    )
                )
            del physical
            positions = torch.arange(context - span, context)[None]
            x = recorded[:, positions[0] % 4096].contiguous()
            rope = hf_rope(x, positions)
            reference_cache = DynamicCache(config=config)
            reference_cache.update(keys[:, :, : context - span], values[:, :, : context - span], 0)
            mask = torch.where(
                torch.arange(context)[None] <= positions[0, :, None], 0.0, torch.finfo(torch.bfloat16).min
            ).bfloat16()[None, None]
            expected = hf(x, position_embeddings=rope, attention_mask=mask, past_key_values=reference_cache)
            del reference_cache, mask
            row = {
                "cache_capacity": context,
                "reference": "HF decoder with complete canonical HF prefix K/V",
                "prefill": [],
                "decode": [],
                "complete": False,
            }
            evidence["contexts"].append(row)
            # Aligned starts exercise padded suffixes; ending-at-limit variants
            # additionally exercise decode-assisted unaligned continuation.
            counts = sorted({1, 31, 32, span - 1, span})
            cases = [(0, count) for count in counts] + [(span - 1, 1), (1, span - 1)]
            for offset, count in cases:
                before = len(accurate_calls)
                stock_before = len(stock_calls)
                tx = upload(x[:, offset : offset + count].unsqueeze(0), mesh, shard=-1)
                tr = tuple(upload(r[:, offset : offset + count].unsqueeze(1), mesh) for r in rope)
                start_pos = context - span + offset
                plan = layer.prepare_prefill(seq_len=count, start_pos=start_pos)
                out = layer.prefill_forward(tx, rope=tr, kv_cache=caches, page_table=tt_table, plan=plan)
                actual = read(out, mesh, -1).reshape(1, count, 4096)
                record = {
                    "start_pos": start_pos,
                    "logical_length": count,
                    "logical_context": start_pos + count,
                    "leading_decode_tokens": len(plan.leading_positions),
                    "accurate_attention_calls": accurate_calls[before:],
                    "stock_attention_calls": stock_calls[stock_before:],
                    **metrics(actual, expected[:, offset : offset + count]),
                }
                row["prefill"].append(record)
                print("INITIALIZED_SUFFIX", context, record, flush=True)
                out.deallocate(True)
                del tx, tr, plan, out, actual

            # Capture once, then change both the token and absolute position.
            dx = x[:, -2:-1]
            packed = tuple(r[:, -2:-1].unsqueeze(0).repeat(1, 1, 32, 1) for r in rope)
            td = upload(dx.unsqueeze(0), mesh, shard=-1)
            tr = tuple(upload(r, mesh) for r in packed)
            tp = upload(torch.tensor([context - 2], dtype=torch.int32), mesh, integer=True)
            kwargs = dict(rope=tr, kv_cache=caches, page_table=tt_table, current_pos=tp)
            before = len(accurate_calls)
            warm = layer.decode_forward(td, **kwargs)
            warm.deallocate(True)
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = layer.decode_forward(td, **kwargs)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            row["decode_capture_accurate_attention_calls"] = accurate_calls[before:]
            for index in (span - 2, span - 1):
                if index == span - 1:
                    refresh(td, x[:, index : index + 1].unsqueeze(0), mesh, shard=-1)
                    refresh(tp, torch.tensor([context - 1], dtype=torch.int32), mesh, integer=True)
                    for dest, source in zip(tr, rope):
                        refresh(dest, source[:, index : index + 1].unsqueeze(0).repeat(1, 1, 32, 1), mesh)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = read(out, mesh, -1).reshape(1, 1, 4096)
                record = {"position": context - span + index, **metrics(actual, expected[:, index : index + 1])}
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                record["same_input_replay_bitwise"] = torch.equal(actual, read(out, mesh, -1).reshape_as(actual))
                assert record["same_input_replay_bitwise"]
                row["decode"].append(record)
                print("INITIALIZED_DECODE", context, record, flush=True)
            ttnn.release_trace(mesh, trace)
            trace = None
            out.deallocate(True)
            row["complete"] = True
            args.output.write_text(json.dumps(evidence, indent=2) + "\n")
            del caches, tt_table, td, tr, tp, kwargs, out, expected, actual
            gc.collect()
        evidence["complete"] = True
    finally:
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.transformer.chunked_scaled_dot_product_attention = original_stock
        ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", nargs="+", type=int, default=[65536, 65568, 131072, 524288])
    parser.add_argument("--suffix-span", type=int, choices=[32, 128, 256], default=32)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

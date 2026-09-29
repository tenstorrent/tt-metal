"""Real-config, layer-only HF parity runner; all host work is an explicit boundary."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from tracy import signpost
from transformers import DynamicCache

import ttnn
from models.autoports.ifm_k2_horizon_7b.tt.fused_decoder import FusedDecoder

from .run_functional import load_reference as functional_reference

MODEL = "IFM/K2-Horizon-7B"
REVISION = "036114ce8d46c32b24c15423211069abb9c5d25e"
DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder"


def load_reference(synthetic=False):
    return functional_reference(synthetic)


def pcc(actual, expected):
    a, b = actual.float().reshape(-1), expected.float().reshape(-1)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def to_device(x, mesh, integer=False):
    return ttnn.from_torch(
        x.contiguous(),
        device=mesh,
        dtype=ttnn.int32 if integer else ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
    )


def refresh(dst, x, integer=False):
    src = ttnn.from_torch(
        x.contiguous(),
        dtype=ttnn.int32 if integer else ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
    )
    ttnn.copy_host_to_device_tensor(src, dst)


@torch.no_grad()
def run(args):
    torch.set_num_threads(16)
    config, state, hf, hf_rope = load_reference(args.synthetic)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    records = []
    audit_counts = None
    completed = False
    try:
        layer = FusedDecoder.from_state_dict(state, hf_config=config, layer_idx=0, mesh_device=mesh)
        if args.audit:
            from .runtime_audit import instrument

            audit_counts = instrument(layer)
        del state
        for seq in args.lengths:
            torch.manual_seed(seq + args.batch)
            batch = args.batch
            capacity = math.ceil((seq + args.decode_steps) / 32) * 32
            if capacity > config.max_position_embeddings:
                capacity = config.max_position_embeddings
            pages = capacity // 32
            table = torch.randperm(batch * pages).reshape(batch, pages).int()
            tt_table = to_device(table, mesh, True)
            cache_shape = (batch * pages, 8, 32, 128)
            caches = tuple(
                ttnn.zeros(
                    cache_shape,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            )
            x = (torch.randn(batch, seq, 4096) * 0.03).bfloat16()
            positions = torch.arange(seq).unsqueeze(0).expand(batch, -1)
            rope = hf_rope(x, positions)
            hf_cache = DynamicCache(config=config)
            expected = hf(x, position_embeddings=rope, past_key_values=hf_cache)
            tt_x = to_device(x.unsqueeze(0), mesh)
            tt_rope = tuple(to_device(r.unsqueeze(1), mesh) for r in rope)
            plan = layer.prepare_prefill(seq_len=seq)
            print(f"PREFILL_START seq={seq} batch={batch}", flush=True)
            if args.split:
                split = args.split
                assert 0 < split < seq
                first = layer.prefill_forward(
                    tt_x[:, :, :split, :],
                    rope=tuple(r[:, :, :split, :] for r in tt_rope),
                    kv_cache=caches,
                    page_table=tt_table,
                    plan=layer.prepare_prefill(seq_len=split),
                )
                second = layer.prefill_forward(
                    tt_x[:, :, split:, :],
                    rope=tuple(r[:, :, split:, :] for r in tt_rope),
                    kv_cache=caches,
                    page_table=tt_table,
                    plan=layer.prepare_prefill(seq_len=seq - split, start_pos=split),
                )
                result = ttnn.concat([first, second], dim=2)
                del first, second
            else:
                result = layer.prefill_forward(tt_x, rope=tt_rope, kv_cache=caches, page_table=tt_table, plan=plan)
            actual = ttnn.to_torch(result).reshape(batch, seq, 4096)
            score = pcc(actual, expected)
            print(f"PREFILL_PCC {score}", flush=True)
            assert score >= 0.995, score
            records.append({"phase": "prefill", "seq_len": seq, "batch": batch, "pcc": score})
            control_caches = None
            if args.unchunked_control:
                assert batch == 1 and seq <= 1024
                control_caches = tuple(
                    ttnn.zeros(
                        cache_shape,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(2)
                )
                control = layer._prefill_chunk(
                    tt_x, rope=tt_rope, kv_cache=control_caches, page_table=tt_table, start_pos=0
                )
                control_host = ttnn.to_torch(control).reshape_as(actual)
                control_pcc = pcc(control_host, expected)
                chunk_pcc = pcc(control_host, actual)
                assert control_pcc >= 0.995 and chunk_pcc >= 0.995
                records.append(
                    {
                        "phase": "unchunked_prefill_control",
                        "pcc": control_pcc,
                        "chunked_vs_unchunked_pcc": chunk_pcc,
                        "seq_len": seq,
                    }
                )
                control.deallocate(True)
            for cache_name, tt_cache, ref_cache in zip(
                ("key", "value"), caches, (hf_cache.layers[0].keys, hf_cache.layers[0].values)
            ):
                physical_cache = ttnn.to_torch(tt_cache)
                logical_cache = torch.stack(
                    [
                        physical_cache[table[b].long()].permute(1, 0, 2, 3).reshape(8, capacity, 128)[:, :seq]
                        for b in range(batch)
                    ]
                )
                cache_score = pcc(logical_cache, ref_cache)
                print(f"CACHE_PCC {cache_name}={cache_score}", flush=True)
                assert cache_score >= 0.995
                records.append(
                    {"phase": "cache", "name": cache_name, "seq_len": seq, "batch": batch, "pcc": cache_score}
                )
            result.deallocate(True)
            repeat = layer.prefill_forward(tt_x, rope=tt_rope, kv_cache=caches, page_table=tt_table, plan=plan)
            if args.split:
                control_pcc = pcc(actual, ttnn.to_torch(repeat).reshape_as(actual))
                assert control_pcc >= 0.995, "continuation differs from fresh prefill"
                records.append({"phase": "continuation_control", "pcc": control_pcc, "split": args.split})
                repeat.deallocate(True)
                first = layer.prefill_forward(
                    tt_x[:, :, :split, :],
                    rope=tuple(r[:, :, :split, :] for r in tt_rope),
                    kv_cache=caches,
                    page_table=tt_table,
                    plan=layer.prepare_prefill(seq_len=split),
                )
                second = layer.prefill_forward(
                    tt_x[:, :, split:, :],
                    rope=tuple(r[:, :, split:, :] for r in tt_rope),
                    kv_cache=caches,
                    page_table=tt_table,
                    plan=layer.prepare_prefill(seq_len=seq - split, start_pos=split),
                )
                repeat = ttnn.concat([first, second], dim=2)
                del first, second
            assert torch.equal(actual, ttnn.to_torch(repeat).reshape_as(actual)), "prefill nondeterminism"
            repeat.deallocate(True)
            if args.profile:
                ttnn.synchronize_device(mesh)
                signpost("PERF_PREFILL")
                measured = layer.prefill_forward(tt_x, rope=tt_rope, kv_cache=caches, page_table=tt_table, plan=plan)
                ttnn.synchronize_device(mesh)
                signpost("PERF_PREFILL_END")
                measured.deallocate(True)
            # A full context has no legal following token; decode last position below.
            decode_steps = min(args.decode_steps, config.max_position_embeddings - seq)
            if decode_steps:
                pos = torch.full((batch,), seq, dtype=torch.int32)
                dx = (torch.randn(batch, 1, 4096) * 0.03).bfloat16()
                dr = hf_rope(dx, pos[:, None].long())
                packed = tuple(r.unsqueeze(0).repeat(1, 1, 32, 1) for r in dr)
                td = to_device(dx.transpose(0, 1).unsqueeze(0), mesh)
                tr = tuple(to_device(r, mesh) for r in packed)
                tp = to_device(pos, mesh, True)
                kwargs = dict(rope=tr, kv_cache=caches, page_table=tt_table, current_pos=tp)
                warm = layer.decode_forward(td, **kwargs)
                warm.deallocate(True)
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                dout = layer.decode_forward(td, **kwargs)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                # Warm replay itself before any measured window. This overwrites
                # the same first-decode slot with the same K/V; prefix is intact.
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                for step in range(decode_steps):
                    if step:
                        if args.remap and batch > 1:
                            table = table.flip(0).contiguous()
                            refresh(tt_table, table, True)
                            hf_cache.layers[0].keys = hf_cache.layers[0].keys.flip(0).contiguous()
                            hf_cache.layers[0].values = hf_cache.layers[0].values.flip(0).contiguous()
                        pos += 1
                        dx = (torch.randn(batch, 1, 4096) * 0.03).bfloat16()
                        dr = hf_rope(dx, pos[:, None].long())
                        refresh(td, dx.transpose(0, 1).unsqueeze(0))
                        for dest, source in zip(tr, dr):
                            refresh(dest, source.unsqueeze(0).repeat(1, 1, 32, 1))
                        refresh(tp, pos, True)
                    expected_d = hf(dx, position_embeddings=dr, past_key_values=hf_cache)
                    if step == 0 and control_caches is not None:
                        control_first = (dx.clone(), tuple(r.clone() for r in dr), expected_d.clone())
                    if args.profile:
                        ttnn.synchronize_device(mesh)
                        signpost(f"PERF_DECODE_{step:03d}")
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    if args.profile:
                        signpost(f"PERF_DECODE_{step:03d}_END")
                    actual_d = ttnn.to_torch(dout).reshape(batch, 1, 4096)
                    dscore = pcc(actual_d, expected_d)
                    print(f"DECODE_PCC seq={seq} pos={pos.tolist()} pcc={dscore}", flush=True)
                    assert dscore >= 0.995, dscore
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    assert torch.equal(actual_d, ttnn.to_torch(dout).reshape_as(actual_d)), "decode nondeterminism"
                    records.append(
                        {
                            "phase": "decode",
                            "context": seq + step + 1,
                            "batch": batch,
                            "pcc": dscore,
                            "traced": True,
                            "deterministic": True,
                        }
                    )
                ttnn.release_trace(mesh, trace)
                if control_caches is not None:
                    first_x, first_rope, first_expected = control_first
                    refresh(td, first_x.transpose(0, 1).unsqueeze(0))
                    refresh(tp, torch.full((batch,), seq, dtype=torch.int32), True)
                    for dest, source in zip(tr, first_rope):
                        refresh(dest, source.unsqueeze(0).repeat(1, 1, 32, 1))
                    control_kwargs = dict(rope=tr, kv_cache=control_caches, page_table=tt_table, current_pos=tp)
                    control_warm = layer.decode_forward(td, **control_kwargs)
                    control_warm.deallocate(True)
                    ttnn.synchronize_device(mesh)
                    control_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    control_out = layer.decode_forward(td, **control_kwargs)
                    ttnn.end_trace_capture(mesh, control_trace, cq_id=0)
                    ttnn.execute_trace(mesh, control_trace, cq_id=0, blocking=True)
                    control_score = pcc(ttnn.to_torch(control_out).reshape_as(first_expected), first_expected)
                    assert control_score >= 0.995
                    records.append({"phase": "unchunked_cache_traced_decode", "pcc": control_score, "context": seq + 1})
                    ttnn.release_trace(mesh, control_trace)
                    del control_out, control_kwargs, control_caches
                del td, tr, tp, dout, kwargs
            del caches, tt_table, tt_x, tt_rope, plan, hf_cache, result, repeat
        completed = True
    finally:
        ttnn.close_mesh_device(mesh)
        Path(args.output).write_text(
            json.dumps(
                {
                    "implementation": type(layer).__module__ + "." + type(layer).__name__,
                    "weights": "synthetic" if args.synthetic else "real",
                    "revision": REVISION,
                    "layer_type": "dense",
                    "records": records,
                    "runtime_audit_completed_calls": audit_counts,
                    "completed": completed,
                    "invocation": sys.argv,
                    "parameters": vars(args),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[32])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--split", type=int)
    parser.add_argument("--remap", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--unchunked-control", action="store_true")
    parser.add_argument("--output", default=str(DOC / "smoke.json"))
    run(parser.parse_args())

# SPDX-License-Identifier: Apache-2.0
import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.aleph_alpha_kolibri_1_bf16.tests.reference import ReferenceDecoder, config, load_weights
from models.autoports.aleph_alpha_kolibri_1_bf16.tt.optimized_decoder import OptimizedDecoder

from .optimized_provenance import provenance

ROOT = Path(__file__).resolve().parents[1]


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--capacity", type=int, default=1024)
    ap.add_argument("--output", default="smoke.json")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") != "1"
    run_provenance = provenance()
    print("PROVENANCE", json.dumps(run_provenance), flush=True)
    torch.set_num_threads(16)
    torch.manual_seed(927)
    weights = load_weights(
        args.layer, stats_path=ROOT / f"doc/functional_decoder/weight_stats_{args.layer}.json", synthetic=args.synthetic
    )
    print("WEIGHTS_LOADED", flush=True)
    ref = ReferenceDecoder(weights, args.layer)
    x = (torch.randn(args.batch, args.length + 1, 2560) * 0.48495227098464966).bfloat16()
    recorded = None
    if os.environ.get("OPT_REAL_INPUT") == "1":
        recorded = torch.load(ROOT / f"doc/optimized_decoder/recorded_inputs/layer_{args.layer}.pt", weights_only=True)
        indices = torch.arange(args.batch * (args.length + 1)) % recorded.shape[1]
        x = recorded[:, indices].reshape(args.batch, args.length + 1, 2560).contiguous()
    t = time.monotonic()
    expected = ref(x[:, : args.length])
    expected_decode = ref(x[:, args.length :], start=args.length)
    print("REF_READY", time.monotonic() - t, flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)

    def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(v, device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def angles(start, s):
        phase = torch.arange(start, start + s)[:, None] / (10000.0 ** (torch.arange(0, 128, 2).float() / 128))
        phase = torch.cat([phase, phase], dim=-1)[None, None]
        return tt(phase.cos().bfloat16()), tt(phase.sin().bfloat16())

    try:
        model = OptimizedDecoder.from_state_dict(weights, hf_config=config(), layer_idx=args.layer, mesh_device=mesh)
        print("TT_WEIGHTS_READY", flush=True)
        blocks = (args.capacity + 31) // 32
        pages = torch.randperm(blocks * args.batch, dtype=torch.int32).reshape(args.batch, blocks)
        page_table = tt(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cache = tuple(
            tt(
                torch.zeros(blocks * args.batch, 4, 32, 128, dtype=torch.bfloat16),
                getattr(ttnn, os.environ.get("OPT_CACHE", "bfloat16")),
            )
            for _ in range(2)
        )
        if args.public or args.batch > 1:
            plan = model.prepare_prefill(page_table_host=pages, seq_len=args.length)
            prefill_input = tt(x[:, : args.length][None].contiguous())
            prefill_output = model.prefill_forward(prefill_input, kv_cache=cache, plan=plan)
            actual = ttnn.to_torch(prefill_output)[0]
            del prefill_output
        else:
            outputs = []
            for start in range(0, args.length, 128):
                n = min(128, args.length - start)
                padded = (n + 31) // 32 * 32
                value = torch.zeros(1, 1, padded, 2560, dtype=torch.bfloat16)
                value[0, 0, :n] = x[0, start : start + n]
                cos, sin = angles(start, padded)
                kw = dict(
                    kv_cache=cache,
                    page_table=page_table,
                    chunk_page_table=tt(
                        pages[:, start // 32 : (start + padded) // 32], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                    ),
                    chunk_start=tt(torch.tensor([start], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    cos=cos,
                    sin=sin,
                )
                inp = tt(value)
                out = model.prefill_chunk_forward(inp, **kw)
                outputs.append(ttnn.to_torch(out)[0, 0, :n])
                del out
            actual = torch.cat(outputs)[None]
        prefill_pcc = pcc(expected, actual)
        prefill_per_row = [pcc(expected[b], actual[b]) for b in range(args.batch)]
        print("PREFILL_PCC", prefill_pcc, flush=True)
        inp = tt(x[:, args.length :].reshape(1, 1, args.batch, 2560))
        cos, sin = angles(args.length, 1)
        if args.batch > 1:
            phase = torch.full((args.batch, 1), args.length) / (10000.0 ** (torch.arange(0, 128, 2).float() / 128))
            phase = torch.cat([phase, phase], dim=-1)[None, None]
            cos = tt(phase.cos().bfloat16())
            sin = tt(phase.sin().bfloat16())
        kw = dict(
            kv_cache=cache,
            page_table=page_table,
            current_pos=tt(
                torch.full((args.batch,), args.length, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
            ),
            cos=cos,
            sin=sin,
        )
        eager = model.decode_forward(inp, **kw)
        eager_cpu = ttnn.to_torch(eager).reshape(args.batch, 1, 2560)
        eager_per_row = [pcc(expected_decode[b], eager_cpu[b]) for b in range(args.batch)]
        eager_pcc = pcc(expected_decode, eager_cpu)
        print("EAGER_DECODE_PCC", eager_pcc, "MIN_ROW", min(eager_per_row), flush=True)
        assert min(eager_per_row) >= 0.995 and eager_pcc >= 0.995
        del eager
        gc.collect()
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            output = model.decode_forward(inp, **kw)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        repeats = []
        for _ in range(3):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            repeats.append(ttnn.to_torch(output))
        decode_pcc = pcc(expected_decode, repeats[-1])
        decode_cpu = repeats[-1].reshape(args.batch, 1, 2560)
        decode_per_row = [pcc(expected_decode[b], decode_cpu[b]) for b in range(args.batch)]
        deterministic = all(torch.equal(repeats[0], v) for v in repeats[1:])
        print("DECODE_PCC", decode_pcc, "DETERMINISTIC", deterministic, flush=True)
        # Refresh all scheduler-owned inputs on the same trace, with shuffled
        # request rows and distinct current positions. No device allocations.
        order = torch.randperm(args.batch)
        positions = torch.randint(0, args.length, (args.batch,), dtype=torch.int32)
        new_input = (torch.randn(args.batch, 1, 2560) * 0.48495227098464966).bfloat16()
        if recorded is not None:
            indices = (torch.arange(args.batch) + 511) % recorded.shape[1]
            new_input = recorded[:, indices].reshape(args.batch, 1, 2560).contiguous()
        expected_changed = []
        for row in range(args.batch):
            source = int(order[row])
            position = int(positions[row])
            item_ref = ReferenceDecoder(weights, args.layer)
            item_ref.cache = tuple(c[source : source + 1] for c in ref.cache)
            expected_changed.append(item_ref(new_input[row : row + 1], start=position))
        expected_changed = torch.cat(expected_changed)

        def refresh(value, destination):
            host = ttnn.from_torch(value.contiguous(), dtype=destination.dtype, layout=destination.layout)
            ttnn.copy_host_to_device_tensor(host, destination)

        refresh(new_input.reshape(1, 1, args.batch, 2560), inp)
        refresh(pages[order], page_table)
        refresh(positions, kw["current_pos"])
        phase = positions.float()[:, None] / (10000.0 ** (torch.arange(0, 128, 2).float() / 128))
        phase = torch.cat([phase, phase], dim=-1)[None, None]
        refresh(phase.cos().bfloat16(), kw["cos"])
        refresh(phase.sin().bfloat16(), kw["sin"])
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        changed = ttnn.to_torch(output).reshape(args.batch, 1, 2560)
        changed_pcc = pcc(expected_changed, changed)
        changed_per_row = [pcc(expected_changed[b], changed[b]) for b in range(args.batch)]
        print("CHANGED_BATCH_REPLAY", changed_pcc, "MIN_ROW", min(changed_per_row), flush=True)
        ttnn.release_trace(mesh, tid)
        result = dict(
            provenance=run_provenance,
            changed_batch_pcc=changed_pcc,
            changed_batch_per_row_pcc=changed_per_row,
            changed_positions=positions.tolist(),
            changed_request_order=order.tolist(),
            batch=args.batch,
            public_api=args.public,
            layer=args.layer,
            length=args.length,
            capacity=args.capacity,
            prefill_pcc=prefill_pcc,
            prefill_per_row_pcc=prefill_per_row,
            decode_pcc=decode_pcc,
            decode_per_row_pcc=decode_per_row,
            eager_decode_pcc=eager_pcc,
            eager_decode_per_row_pcc=eager_per_row,
            deterministic=deterministic,
            real_weights=not args.synthetic,
            allocation_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
            skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "1",
        )
        (ROOT / "doc/optimized_decoder" / args.output).write_text(json.dumps(result, indent=2) + "\n")
        assert (
            prefill_pcc >= 0.995
            and decode_pcc >= 0.995
            and changed_pcc >= 0.995
            and min(changed_per_row) >= 0.995
            and deterministic
            and min(prefill_per_row + decode_per_row) >= 0.995
        ), result
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

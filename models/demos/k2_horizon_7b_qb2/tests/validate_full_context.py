"""Public prompt boundaries, unaligned continuation and full-stack DRAM capacity.

Late-context work uses initialized zero prefix caches and is structural evidence;
it is explicitly not a full-context HF numerical comparison.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator

DOC = Path("models/demos/k2_horizon_7b_qb2/doc/full_model")


def memory(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return {
        key: getattr(view, key)
        for key in [
            "num_banks",
            "total_bytes_per_bank",
            "total_bytes_allocated_per_bank",
            "largest_contiguous_bytes_free_per_bank",
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--capacity-only", action="store_true")
    parser.add_argument("--optimized-buffers", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--precision-config", type=Path)
    parser.add_argument(
        "--normal-text",
        action="store_true",
        help="Encode repeated prose once; retain default repeated-BOS stress repro",
    )
    args = parser.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    result = {"layers": args.layers, "scope": "capacity" if args.capacity_only else "all", "boundaries": []}
    path = args.output or (
        DOC
        / f"context_layers{args.layers}{'_capacity' if args.capacity_only else ''}{'_normal' if args.normal_text else ''}.json"
    )
    try:
        gen = K2Generator(mesh, override_num_layers=args.layers, precision_config=args.precision_config)
        result["precision_config"] = gen.model.precision_config
        result["runtime_policies"] = [asdict(layer.policy) for layer in gen.model.layers]
        result["weight_allocations"] = [layer.weight_allocations for layer in gen.model.layers]
        base = gen.tokenizer.encode("A careful scientist checks the evidence before drawing a conclusion. ")
        if args.normal_text:
            base = gen.tokenizer.encode("A careful scientist checks the evidence before drawing a conclusion. " * 1000)
            assert gen.tokenizer.bos_token_id not in base[1:]
        result["prompt_construction"] = (
            "encode repeated prose once" if args.normal_text else "repeat BOS-containing encoded sentence"
        )
        if not args.capacity_only:
            for length in [1, 31, 32, 33, 255, 256, 257, 4095, 4096, 4097, 4353]:
                prompt = (base * ((length + len(base) - 1) // len(base)))[:length]
                first = gen.generate(prompt, 3)
                second = gen.generate(prompt, 3)
                assert first == second
                result["boundaries"].append(
                    {"logical_prompt_len": length, "tokens": first, "repeat_exact": True, "perf": gen.last_perf.copy()}
                )
                path.write_text(json.dumps(result, indent=2) + "\n")
                print("BOUNDARY", length, first, flush=True)
            prompt = torch.tensor([(base * 30)[:257]])
            gen.reset()
            full = gen.prefill_forward(
                prompt, page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[257], sampling_mode="host"
            )
            gen.reset()
            gen.prefill_forward(prompt[:, :31], page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[31])
            split = gen.prefill_forward(
                prompt[:, 31:],
                page_table=gen.page_table,
                kv_cache=gen.kv_cache,
                prompt_lens=[226],
                start_pos=[31],
                sampling_mode="host",
            )
            a, b = full.flatten().float(), split.flatten().float()
            pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            assert pcc >= 0.995
            assert a.argmax() == b.argmax()
            result["unaligned_continuation"] = {
                "logical_length": 257,
                "split": 31,
                "logits_pcc": pcc,
                "same_greedy_token": True,
            }
        gen._release_traces()
        gen.kv_cache = None
        result["dram_before_full_context"] = memory(mesh)
        start = time.perf_counter()
        gen._ensure_owned_cache(1, gen.model.context)
        if args.optimized_buffers:
            gen._ensure_token_history(gen.model.context)
            gen._prepare_prefill_inputs((base * ((4096 + len(base) - 1) // len(base)))[:4096])
            gen.reset()
            result["optimized_buffers"] = {
                "history_shape": list(gen.token_history.shape),
                "history_dtype": str(gen.token_history.dtype),
                "prepared_prefill_length": gen.prefill_state["plan"].seq_len,
                "prefill_logits_shape": list(gen.prefill_state["logits"].shape),
                "max_history_and_full_cache_coexist": True,
            }
        ttnn.synchronize_device(mesh)
        result["capacity"] = {
            "context": gen.model.context,
            "layers": gen.model.num_layers,
            "cache_pairs": len(gen.kv_cache),
            "local_cache_shape": list(gen.kv_cache[0][0].shape),
            "allocation_seconds": time.perf_counter() - start,
            "dram": memory(mesh),
        }
        # End-of-context embedding/RoPE, page fill, cache read and output slicing.
        # Prefix pages are allocated and initialized, but do not represent text.
        tail = torch.tensor([(base * 4)[:35]])
        start_pos = gen.model.context - 36
        logits = gen.prefill_forward(
            tail,
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=[35],
            start_pos=[start_pos],
            sampling_mode="host",
        )
        assert torch.isfinite(logits).all()
        token = logits[0, 0].argmax().reshape(1)
        output = gen.decode_forward(
            token, torch.tensor([gen.model.context - 1]), page_table=gen.page_table, kv_cache=gen.kv_cache
        )
        assert 0 <= int(output[0]) < gen.model.vocab_size
        result["late_context"] = {
            "prefix": "initialized zero KV, structural test only",
            "start_pos": start_pos,
            "logical_prompt_len": 35,
            "decode_position": gen.model.context - 1,
            "token": int(output[0]),
            "finite_logits": True,
        }
        result["pass"] = True
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

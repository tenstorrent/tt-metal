# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fixed-slot regression: remap, inactive GDN state, and trace replay."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_model_evidence import ccl_inventory, source_identity
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model


def first(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=range(2, 33), default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--prefill-length", type=int, default=0)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--physical-pages-per-slot", type=int)
    parser.add_argument("--full-prepare", action="store_true")
    parser.add_argument("--decode-steps", type=int, default=2)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    generator = None
    try:
        model = Model(mesh, layer_indices=None if args.all_layers else [0, 3])
        table = None
        if args.physical_pages_per_slot is not None:
            pages = max(16, (args.context + 511) // 512 * 16)
            width = args.physical_pages_per_slot
            assert width >= 16 and width <= pages
            physical = torch.randperm(args.batch * width, generator=torch.Generator().manual_seed(42))
            table = physical.reshape(args.batch, width)[:, torch.arange(pages) % width].to(torch.int32)
        generator = QwenGenerator(model, batch_size=args.batch, max_seq_len=args.context, page_table=table)
        lengths = sorted({1, args.prefill_length} - {0})
        if not args.full_prepare:
            generator.prefill_single.prepare(_diagnostic_lengths=lengths)
            generator.prefill_uniform.prepare(_diagnostic_lengths=lengths)
        generator.prepare()
        prefill = None
        if args.prefill_length:
            prompt = torch.arange(100, 100 + args.prefill_length, dtype=torch.int32)[None].repeat(args.batch, 1)
            logits = generator.prefill_forward(
                prompt,
                page_table=generator.page_host,
                kv_cache=generator.cache,
                prompt_lens=[args.prefill_length] * args.batch,
            )
            assert tuple(logits.shape) == (args.batch, 1, model.config.vocab_size)
            assert bool(torch.isfinite(logits).all())
            assert all(torch.equal(logits[0], row) for row in logits), "Identical prompts differ across batch slots"
            prefill = dict(
                length=args.prefill_length,
                finite=True,
                equal_logits_across_slots=True,
                argmax=logits.argmax(-1).flatten().tolist(),
            )
            generator.reset()
        tokens = torch.arange(70000, 70000 + args.batch).reshape(-1, 1)
        generator.configure_sampling(seed=list(range(100, 100 + args.batch)))
        position = torch.zeros(args.batch, dtype=torch.int32)
        generator.decode_forward(tokens, position, page_table=generator.page_host, kv_cache=generator.cache)
        state = generator.cache[0]
        before = [first(t) for t in (state.conv, state.recurrent)]
        position.fill_(1)
        position[1] = -1
        output = generator.decode_forward(
            tokens + 1, position, page_table=generator.page_host, kv_cache=generator.cache
        )
        after = [first(t) for t in (state.conv, state.recurrent)]
        assert all(torch.equal(a[1], b[1]) for a, b in zip(before, after)), "Inactive GDN state mutated"
        assert all(not torch.equal(a[0], b[0]) for a, b in zip(before, after)), "Active GDN state did not advance"
        controls = [generator.tokens, generator.rope_positions, generator.sampling.tt_sampling.seeds_tt_tensor]
        control_before = [first(t).flatten().clone() for t in controls]
        readbacks = generator.counters["token_readbacks"]
        ttnn.synchronize_device(mesh)
        begin = time.perf_counter()
        for _ in range(args.decode_steps):
            resident = generator.decode_forward(
                tokens,
                position,
                page_table=generator.page_host,
                kv_cache=generator.cache,
                reload_inputs=False,
                read_from_device=False,
            )
            assert resident is generator.tokens
        ttnn.synchronize_device(mesh)
        resident_seconds = time.perf_counter() - begin
        assert generator.counters["token_readbacks"] == readbacks
        control_after = [first(t).flatten().clone() for t in controls]
        assert all(a[1] == b[1] for a, b in zip(control_before, control_after)), "Inactive token/RoPE/RNG changed"
        assert first(generator.positions).flatten()[1] == -1
        after = [first(t) for t in (state.conv, state.recurrent)]
        assert all(torch.equal(a[1], b[1]) for a, b in zip(before, after)), "Resident inactive GDN state mutated"
        remap = list(reversed(range(args.batch)))
        generator.remap_slots(remap)
        moved = [first(t) for t in (state.conv, state.recurrent)]
        assert all(torch.equal(a[remap], b) for a, b in zip(after, moved)), "Slot remap lost state"
        generator.release_request(1)
        released = [first(t) for t in (state.conv, state.recurrent)]
        assert all(torch.count_nonzero(t[1]) == 0 for t in released)
        assert all(torch.equal(a[0], b[0]) for a, b in zip(moved, released))
        # End with replay so allocation tracking checks every preceding mutation.
        generator.decode_forward(
            tokens, torch.full_like(position, 2), page_table=generator.page_host, kv_cache=generator.cache
        )
        result = dict(
            passed=False,
            functional_assertions_passed=True,
            batch=args.batch,
            layers=model.layer_indices,
            source_sha256=source_identity(),
            context=args.context,
            virtual_table_shape=list(generator.page_host.shape),
            physical_pages_per_slot=args.physical_pages_per_slot,
            prepared_memory=model.memory_view(),
            preparation_scope="full1..512" if args.full_prepare else f"diagnostic lengths{lengths}",
            prefill=prefill,
            resident_decode=dict(
                steps=args.decode_steps,
                seconds=resident_seconds,
                per_token_readbacks=0,
                tokens_per_second_per_active_user=args.decode_steps / resident_seconds,
                active_slots=args.batch - 1,
            ),
            output=output.tolist(),
            counters=dict(generator.counters),
        )
        # Preserve completed assertions and measured memory even if a reporting
        # helper fails; only the complete report earns passed=True.
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        result["ccl_inventory"] = ccl_inventory(model)
        result["passed"] = True
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if generator is not None:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()

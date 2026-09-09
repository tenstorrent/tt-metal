# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The three slot-boundary paths vLLM drives, at the served batch, fused conv.

Each of these touches per-slot linear state, and each was written against the
composite conv layout, tiled ``[1, batch, channels, kernel]``:

1. ``reset_slots`` -- ``generator_vllm.prefill_forward`` clears a slot it is
   about to refill.
2. ``single_slot_prefill_view`` -- the active-row optimization, on by default,
   narrows the model to that one slot for the prefill.
3. ``remap_slots`` -- a scheduler ``slot_remap`` permutes decode rows.

The fused KDA decode path instead stores the state *as* its row-major user-major
window ``[batch, kernel, channels]``, which differs in rank, in layout, and in
which axis is the fixed slot. So each of the three did the wrong thing on it:
``reset_slots`` raised "Optional output tensor with Row Major input is not
supported right now for Elementwise operations", ``single_slot_prefill_view``
raised "Input rank 3 and begins 4 must have the same size", and ``remap_slots``
addressed the kernel axis instead of the slot axis. The first two killed the vLLM
EngineCore on the first request; both were found by CI, not here.

Why they were not found here: ``full_model_mixed_slots.py`` covers 1 and 3, and
``vllm_reduced_target.py`` covers the whole adapter lifecycle, but both at
batch 2 -- and the fused decode path needs ``kernel * batch`` tile aligned, so at
batch 2 it never engages. Every test that touched this state exercised the one
layout the served batch does not use.

The assertions are exact rather than PCC: reset must zero exactly the named slots
and leave every peer bit identical, remap must move each slot's state to its new
row bit identically, and the narrowed prefill must advance its own slot without
touching any other. The test also refuses to pass vacuously -- it fails if no
layer took the fused path, or if prefill left the state at zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator

BATCH = 32


def _linear_state(generator):
    """Every linear layer's conv and recurrent state, slot-axis first."""
    state = []
    for index, layer in enumerate(generator.model.layers):
        if layer.layer_kind != "linear_attention":
            continue
        for name in ("conv", "recurrent"):
            cache = layer.caches[name]
            shards = []
            for shard in ttnn.get_device_tensors(cache):
                host = ttnn.to_torch(shard).clone().float()
                # Composite conv is [1, batch, C, K]; the fused window is
                # [batch, K, C]; recurrent is [batch, heads, V, K].
                if name == "conv" and host.dim() == 4:
                    host = host[0]
                shards.append(host)
            state.append((index, name, shards))
    return state


def _slot(entry, slot):
    return [shard.select(0, slot) for shard in entry[2]]


def _same(left, right):
    return all(torch.equal(a, b) for a, b in zip(left, right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--reset-slots", type=int, nargs="+", default=[3, 17])
    parser.add_argument("--narrow-slot", type=int, default=9)
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    result = {"batch": BATCH, "num_layers": args.num_layers, "reset_slots": list(args.reset_slots)}
    try:
        generator = build_generator(
            model_dir=Path("models/autoports/qwen_qwen3_6_27b"),
            mesh_device=mesh,
            max_context=256,
            batch=BATCH,
            num_layers=args.num_layers,
        )
        kda = [
            bool(getattr(layer, "linear_kda_decode_ready", False))
            for layer in generator.model.layers
            if layer.layer_kind == "linear_attention"
        ]
        result["fused_kda_decode_layers"] = f"{sum(kda)}/{len(kda)}"
        if not any(kda):
            raise AssertionError("no layer took the fused conv path; this test would be vacuous")

        # Give every slot distinct non-zero state.
        generator.reset()
        prompt = list(range(11, 11 + args.prompt_tokens))
        tokens = torch.stack([torch.tensor(prompt, dtype=torch.long) + slot for slot in range(BATCH)])
        generator.prefill_forward(
            tokens,
            page_table=generator._page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=[args.prompt_tokens] * BATCH,
        )
        ttnn.synchronize_device(mesh)
        before = _linear_state(generator)
        nonzero = sum(1 for e in before for sh in e[2] if float(sh.abs().sum()) > 0)
        if not nonzero:
            raise AssertionError("prefill left the linear state at zero; the checks would be vacuous")

        # 1. reset_slots: named slots go to zero, every other slot is untouched.
        generator.reset_slots(args.reset_slots)
        ttnn.synchronize_device(mesh)
        after = _linear_state(generator)
        not_zeroed, disturbed = [], []
        for pre, post in zip(before, after):
            for slot in range(BATCH):
                got = _slot(post, slot)
                if slot in args.reset_slots:
                    if any(float(t.abs().sum()) != 0.0 for t in got):
                        not_zeroed.append((pre[0], pre[1], slot))
                elif not _same(_slot(pre, slot), got):
                    disturbed.append((pre[0], pre[1], slot))
        result["reset_not_zeroed"] = len(not_zeroed)
        result["reset_disturbed_peers"] = len(disturbed)
        if not_zeroed:
            raise AssertionError(f"reset_slots left state in {not_zeroed[:6]}")
        if disturbed:
            raise AssertionError(f"reset_slots disturbed peer slots {disturbed[:6]}")

        # 2. remap_slots: a rotation must move each slot's state, exactly.
        generator.reset()
        generator.prefill_forward(
            tokens,
            page_table=generator._page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=[args.prompt_tokens] * BATCH,
        )
        ttnn.synchronize_device(mesh)
        before = _linear_state(generator)
        remap = [(slot + 1) % BATCH for slot in range(BATCH)]  # remap[new] = old
        generator.remap_decode_slots(remap)
        ttnn.synchronize_device(mesh)
        after = _linear_state(generator)
        misplaced = []
        for pre, post in zip(before, after):
            for new, old in enumerate(remap):
                if not _same(_slot(pre, old), _slot(post, new)):
                    misplaced.append((pre[0], pre[1], new, old))
        result["remap_misplaced"] = len(misplaced)
        if misplaced:
            raise AssertionError(f"remap_slots put the wrong state in {misplaced[:6]}")

        # 3. The narrowed single-slot prefill.  This is what vLLM actually does
        # per scheduler step, and it slices the conv state with a rank-4 start,
        # so on the fused window it raised "Input rank 3 and begins 4 must have
        # the same size" and killed the engine.  Drive it through the same
        # public entry point vLLM uses, then check the narrowed slot advanced and
        # every other slot is untouched.
        generator.reset()
        before = _linear_state(generator)
        narrow = int(args.narrow_slot)
        one = torch.zeros((BATCH, args.prompt_tokens), dtype=torch.long)
        one[narrow] = torch.tensor(prompt, dtype=torch.long) + narrow
        lens = [0] * BATCH
        lens[narrow] = args.prompt_tokens
        # Count the narrowing rather than assuming it: the view only engages for
        # exactly one active row with batch > 1 and QWEN36_PREFILL_NARROW unset,
        # and a test that silently skipped it would pass while covering nothing.
        narrow_calls = []
        original_view = generator.model.single_slot_prefill_view

        def counting_view(slot):
            narrow_calls.append(int(slot))
            return original_view(slot)

        generator.model.single_slot_prefill_view = counting_view
        generator.prefill_forward(
            one,
            page_table=generator._page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=lens,
        )
        generator.model.single_slot_prefill_view = original_view
        ttnn.synchronize_device(mesh)
        if narrow_calls != [narrow]:
            raise AssertionError(f"narrowed prefill view did not engage for slot {narrow}: calls={narrow_calls}")
        after = _linear_state(generator)
        narrowed_moved, narrow_peers = 0, []
        for pre, post in zip(before, after):
            for slot in range(BATCH):
                same = _same(_slot(pre, slot), _slot(post, slot))
                if slot == narrow:
                    narrowed_moved += 0 if same else 1
                elif not same:
                    narrow_peers.append((pre[0], pre[1], slot))
        result["narrow_slot"] = narrow
        result["narrow_view_calls"] = narrow_calls
        result["narrowed_state_moved"] = narrowed_moved
        result["narrow_disturbed_peers"] = len(narrow_peers)
        if narrow_peers:
            raise AssertionError(f"narrowed prefill disturbed peer slots {narrow_peers[:6]}")
        if not narrowed_moved:
            raise AssertionError("narrowed prefill left the target slot's state unchanged")

        result["status"] = "SLOT_LIFECYCLE_EXACT"
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if generator is not None:
            generator.reset()
        ttnn.close_mesh_device(mesh)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

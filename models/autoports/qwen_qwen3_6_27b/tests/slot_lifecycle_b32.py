# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``reset_slots`` and ``remap_slots`` at the served batch, with the fused conv.

These are the two request-boundary methods vLLM calls that touch linear state:
``prefill_forward`` resets a slot it is about to refill
(``generator_vllm.py``), and a scheduler ``slot_remap`` permutes decode rows.
Both were written against the composite conv layout, tiled
``[1, batch, channels, kernel]``.  The fused KDA decode path stores the state as
its row-major user-major window ``[batch, kernel, channels]`` instead, which
differs in rank, layout *and* which axis is the fixed slot.

``full_model_mixed_slots.py`` covers both methods but at batch 2, where
``kernel * batch`` is not tile aligned so the fused path never engages -- so
neither was ever run against the layout the served batch uses.  Enabling the
fused conv without this test crashed a CI eval run inside ``reset_slots`` with
"Optional output tensor with Row Major input is not supported right now for
Elementwise operations", and would have silently permuted the kernel axis
instead of the slot axis in ``remap_slots``.

Both assertions are exact: reset must zero the named slots and leave every other
slot bit identical, and remap must move each slot's state to its new row bit
identically.
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

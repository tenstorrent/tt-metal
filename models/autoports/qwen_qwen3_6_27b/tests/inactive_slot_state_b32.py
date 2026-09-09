# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""An inactive fixed slot's linear-attention state must not move, at batch 32.

``full_model_mixed_slots.py`` already asserts this, but only at batch 2 -- and
batch 2 keeps the composite causal convolution, because the fused decode path
needs ``kernel * batch`` tile aligned.  The served batch is 32, where the fused
path *is* active and where ``_linear_attention_decode`` preserves an inactive
row by making the recurrence the identity (``decay = 1``, ``beta = 0``) rather
than by blending the new state against the old.  Both of those are only
exercised here.

The assertion is bit equality, not PCC: an inactive slot is a slot the scheduler
has not admitted, and its state is what a later request inherits.  Run with
``QWEN36_DECODE_STATE_MASK=blend`` to check the same contract on the previous
implementation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator
from models.common.sampling import SamplingParams

BATCH = 32
ACTIVE_SLOTS = (0, 1, 2, 3)


def _shards(tensor):
    return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(tensor)]


def _linear_state(generator):
    """Every linear-attention layer's conv window and recurrent state, per rank."""
    state = []
    for index, layer in enumerate(generator.model.layers):
        if layer.layer_kind != "linear_attention":
            continue
        for name in ("conv", "recurrent"):
            state.append((index, name, _shards(layer.caches[name])))
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=8)
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    result = {
        "batch": BATCH,
        "num_layers": args.num_layers,
        "active_slots": list(ACTIVE_SLOTS),
        "state_mask_mode": os.environ.get("QWEN36_DECODE_STATE_MASK", "gate"),
    }
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

        # Prefill only the active slots.  Every other row keeps the zero state
        # the reset left, which is exactly the state that must survive decode.
        generator.reset()
        prompt = list(range(10, 10 + args.prompt_tokens))
        tokens = torch.zeros((BATCH, args.prompt_tokens), dtype=torch.long)
        prompt_lens = [0] * BATCH
        for slot in ACTIVE_SLOTS:
            tokens[slot] = torch.tensor(prompt, dtype=torch.long)
            prompt_lens[slot] = args.prompt_tokens
        generator.prefill_forward(
            tokens,
            page_table=generator._page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=prompt_lens,
        )

        before = _linear_state(generator)

        positions = [args.prompt_tokens if length else 0 for length in prompt_lens]
        active = [1 if length else 0 for length in prompt_lens]
        generator.setup_token_out_decode(
            [11] * BATCH,
            positions,
            active_mask=active,
            sampling_params=SamplingParams(temperature=1.0, top_k=1, top_p=0.0),
        )
        for _ in range(args.decode_steps):
            generator.token_out_decode_step(readback=False)
        ttnn.synchronize_device(mesh)

        after = _linear_state(generator)
        assert len(before) == len(after)

        inactive = [slot for slot in range(BATCH) if slot not in ACTIVE_SLOTS]
        moved_inactive, moved_active = [], []
        for (layer, name, old), (_, _, new) in zip(before, after):
            for rank, (old_shard, new_shard) in enumerate(zip(old, new)):
                # Both cache layouts index the fixed slot on a single axis:
                # the fused conv window is [B, K, C] and the recurrent state is
                # [B, heads, V, K], while the composite window is [1, B, C, K].
                axis = 1 if (name == "conv" and old_shard.dim() == 4) else 0
                for slot in inactive:
                    if not torch.equal(old_shard.select(axis, slot), new_shard.select(axis, slot)):
                        moved_inactive.append((layer, name, rank, slot))
                for slot in ACTIVE_SLOTS:
                    if not torch.equal(old_shard.select(axis, slot), new_shard.select(axis, slot)):
                        moved_active.append((layer, name, rank, slot))

        result["inactive_slots_checked"] = len(inactive)
        result["inactive_state_moved"] = len(moved_inactive)
        result["active_state_moved"] = len(moved_active)
        result["first_inactive_violations"] = moved_inactive[:8]
        if moved_inactive:
            raise AssertionError(f"{len(moved_inactive)} inactive linear-state slices moved: {moved_inactive[:8]}")
        # An active slot whose state never moved would mean decode did nothing,
        # which would make the check above vacuous.
        if not moved_active:
            raise AssertionError("no active slot's state changed; the inactive check would be vacuous")
        result["status"] = "INACTIVE_LINEAR_STATE_EXACT"
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if generator is not None:
            generator.reset()
        ttnn.close_mesh_device(mesh)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

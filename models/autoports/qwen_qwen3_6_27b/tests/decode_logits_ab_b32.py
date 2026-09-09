# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full-model decode logits at the served shape, dumped for an A/B.

Layer PCC and per-layer harnesses cannot answer "did the served model's output
change", because they run one layer, at one policy, often at batch 1 and often
without an active mask.  This prefills all 32 fixed slots with distinct real
prompts, then runs greedy decode steps through the whole 64-layer stack and
writes every step's per-slot logits and argmax under a fixed, teacher-forced
token sequence.

Run it twice with different configurations and compare with ``--baseline``:
the comparison reports per-slot logit PCC and greedy-token agreement, which is
what a text-quality regression would actually show up as.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator

BATCH = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--baseline", type=Path, help="a previous --output to compare against")
    parser.add_argument(
        "--reverse-slots",
        action="store_true",
        help=(
            "Assign the same 32 prompts to slots in reverse order.  Comparing a normal and a "
            "reversed pair separates a slot-indexed defect (the anomaly stays on the same slot) "
            "from a data-dependent one (it follows the prompt to the mirrored slot)."
        ),
    )
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    try:
        generator = build_generator(
            model_dir=Path("models/autoports/qwen_qwen3_6_27b"),
            mesh_device=mesh,
            max_context=512,
            batch=BATCH,
            num_layers=args.num_layers,
        )
        rendered = generator.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt.read_text().strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        all_ids = generator.tokenizer.encode(rendered, add_special_tokens=False)
        if len(all_ids) < args.prompt_tokens + BATCH:
            raise SystemExit("prompt is too short to give every slot a distinct window")
        # Distinct prompts per slot: a shared prompt would make 32 identical
        # rows and hide any slot-indexed state bug.
        offsets = [(BATCH - 1 - slot) if args.reverse_slots else slot for slot in range(BATCH)]
        tokens = torch.stack(
            [torch.tensor(all_ids[off : off + args.prompt_tokens], dtype=torch.long) for off in offsets]
        )
        prompt_lens = [args.prompt_tokens] * BATCH

        generator.reset()
        logits = generator.prefill_forward(
            tokens,
            page_table=generator._page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=prompt_lens,
        )
        steps = [logits.reshape(BATCH, -1).float()]
        positions = torch.tensor(prompt_lens, dtype=torch.long)
        # Teacher-forced, not free-running.  Feeding each run its own argmax
        # makes the two arms diverge after the first slot that picks a different
        # token, and every later step is then computed from a different history
        # -- the comparison stops measuring the change under test.  A fixed
        # token sequence keeps both arms on identical inputs at every step.
        fed = []
        for step in range(args.steps):
            next_tokens = torch.tensor(
                [all_ids[(offsets[slot] * 7 + step * 13) % len(all_ids)] for slot in range(BATCH)], dtype=torch.long
            )
            fed.append(next_tokens.tolist())
            step_logits = generator.decode_forward(
                next_tokens,
                positions,
                page_table=generator._page_table,
                kv_cache=generator.kv_cache,
            )
            steps.append(step_logits.reshape(BATCH, -1).float())
            positions = positions + 1

        record = {
            "batch": BATCH,
            "prompt_tokens": args.prompt_tokens,
            "steps": args.steps,
            "num_layers": args.num_layers if args.num_layers is not None else len(generator.model.layers),
            "precision_config": generator.model.precision_config.config_id,
            "state_mask_mode": os.environ.get("QWEN36_DECODE_STATE_MASK", "gate"),
            "reverse_slots": args.reverse_slots,
            "prompt_offsets": offsets,
            "forced_tokens": fed,
            "argmax": [step.argmax(dim=-1).tolist() for step in steps],
        }
    finally:
        if generator is not None:
            generator.reset()
        ttnn.close_mesh_device(mesh)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"logits": [step for step in steps], "record": record}, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "argmax"}, indent=2), flush=True)
    print("argmax step 0, first 8 slots:", record["argmax"][0][:8], flush=True)

    if args.baseline:
        base = torch.load(args.baseline.with_suffix(".pt"), weights_only=False)
        base_logits, base_record = base["logits"], base["record"]
        if len(base_logits) != len(steps):
            raise SystemExit("baseline has a different number of steps")
        if base_record.get("forced_tokens") != record["forced_tokens"]:
            raise SystemExit("baseline was driven with different tokens; the comparison would be meaningless")
        worst_pcc, token_matches, token_total = 1.0, 0, 0
        per_step = []
        for index, (mine, theirs) in enumerate(zip(steps, base_logits)):
            step_worst, step_match = 1.0, 0
            for slot in range(BATCH):
                a, b = mine[slot], theirs[slot].float()
                pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
                step_worst = min(step_worst, pcc)
                step_match += int(a.argmax().item() == b.argmax().item())
            per_step.append({"step": index, "worst_slot_pcc": step_worst, "argmax_agree": f"{step_match}/{BATCH}"})
            worst_pcc = min(worst_pcc, step_worst)
            token_matches += step_match
            token_total += BATCH
        summary = {
            "baseline": str(args.baseline),
            "baseline_precision_config": base_record["precision_config"],
            "baseline_state_mask_mode": base_record.get("state_mask_mode"),
            "this_precision_config": record["precision_config"],
            "this_state_mask_mode": record["state_mask_mode"],
            "worst_per_slot_logit_pcc": worst_pcc,
            "argmax_agreement": f"{token_matches}/{token_total}",
            "per_step": per_step,
        }
        print(json.dumps(summary, indent=2), flush=True)
        args.output.with_name(args.output.stem + "_vs_baseline.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

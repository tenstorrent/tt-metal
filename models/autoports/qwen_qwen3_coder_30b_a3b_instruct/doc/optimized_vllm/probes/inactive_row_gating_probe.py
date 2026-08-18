# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Does skipping the inactive rows' expert work change what the live rows say?

``Qwen3CoderModel._decode_active_mask`` zeroes the routing vector of every decode
slot whose ``current_pos`` is the inactive sentinel ``-1``, which takes those
``(row, expert)`` pairs out of ``sparse_matmul``'s sparsity. The claim is that
this is *free* in the model-output sense: the rows it touches are rows whose
output vLLM discards, and every op in the expert path is per-row, so a live row
cannot see it.

That is a claim about a 48-layer MoE, so it is measured rather than argued. The
probe runs the **same** prompts through the **same** generator twice, once with
``QWEN3_DECODE_ACTIVE_ROW_GATING`` off and once on, and requires the live rows'
sampled token sequences to be **identical token for token**. Greedy decoding
makes that an exact comparison rather than a tolerance.

The prompts are **real text**, not synthetic id ranges, and
``outputs_are_varied`` requires each row to emit several distinct tokens and the
rows to differ from each other. Without that leg the equality check has no
teeth: a synthetic prompt makes this checkpoint emit the same newline token
forever, and two runs of "newline x16" agree whatever the expert path did.

Three further legs pin the mechanism rather than the outcome:

``mask_matches_positions``
    the mask read back off the device equals ``current_pos >= 0`` for the exact
    ``current_pos`` the live trace holds -- so the mask is derived from the
    trace input it claims to be derived from;
``mask_survives_replays``
    after N traced replays have advanced every live row's position, the mask is
    still exactly the original active set. This is the property that makes the
    mask safe without a host refresh: ``ttnn.plus_one(..., skip_negative_entries
    =True)`` leaves an inactive row at ``-1`` forever.

Writes ``inactive_row_gating_probe.json`` next to this file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import build_generator  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import DEFAULT_TRACE_REGION_SIZE  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[1]


def _serving_shaped_page_table(gen, slots: int, active: int, blocks_per_row: int) -> torch.Tensor:
    table = torch.zeros((slots, gen.pages_per_user), dtype=torch.int32)
    for row in range(active):
        start = 1 + row * blocks_per_row
        table[row, :blocks_per_row] = torch.arange(start, start + blocks_per_row, dtype=torch.int32)
    return table


def _mask_to_torch(mask) -> torch.Tensor:
    shards = ttnn.get_device_tensors(mask)
    return ttnn.to_torch(shards[0] if shards else mask).reshape(-1).to(torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--active", type=int, default=4)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--context", type=int, default=4096)
    args = parser.parse_args()

    checks: dict[str, dict] = {}
    result = {
        "what": "inactive-row expert gating: token equality against the ungated graph, plus mask mechanics",
        "slots": args.slots,
        "active": args.active,
        "layers": args.layers,
        "steps": args.steps,
        "context": args.context,
        "sampling": "device, greedy (top_k=1) -- so the comparison is exact token equality",
        "checks": checks,
    }

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    try:
        gen = build_generator(
            MODEL_DIR,
            mesh,
            override_num_layers=args.layers,
            max_context_len=args.context,
            max_batch_size=args.slots,
        )
        # Real text, so the greedy continuations are varied and the token
        # equality check has something to fail on.
        texts = [
            "The three laws of thermodynamics are:",
            'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n',
            "Paged attention needs a block table because",
            "Translate to French: Hello, how are you today?\nFrench:",
        ][: args.active]
        encoded = [gen.tokenizer.encode(text) for text in texts]
        prompt_len = max(len(ids) for ids in encoded)
        # Rows may be different lengths -- prefill takes each row's own length --
        # but a common decode start position keeps the batch shape simple, so the
        # shorter prompts are left-padded with the first token they already have.
        prompts = [ids[:1] * (prompt_len - len(ids)) + ids for ids in encoded]
        result["prompt_texts"] = texts
        result["prompt_len"] = prompt_len
        horizon = prompt_len + args.steps + 2
        blocks_per_row = gen._sdpa_rounded_page_count(horizon)
        page_table = _serving_shaped_page_table(gen, args.slots, args.active, blocks_per_row)

        runs: dict[str, dict] = {}
        for label, gating in (("ungated", False), ("gated", True)):
            gen.model.active_row_gating = gating
            gen.reset()
            gen.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0, active_batch=args.active)
            gen.prefill_forward(
                torch.tensor(prompts),
                page_table=page_table,
                kv_cache=gen._ensure_kv_cache(),
                prompt_lens=[prompt_len] * args.active,
                sampling_mode="device",
            )
            positions = torch.full((args.slots,), -1, dtype=torch.int64)
            positions[: args.active] = prompt_len
            sampled = gen.decode_forward(
                None,
                positions,
                page_table=page_table,
                kv_cache=gen._ensure_kv_cache(),
                sampling_mode="device",
                enable_trace=True,
                active_batch=args.slots,
                decode_horizon=horizon,
                validate_page_coverage=False,
            )
            tokens = [gen._sampled_to_torch(sampled)[: args.active].tolist()]
            latencies = []
            for _ in range(args.steps - 1):
                t0 = time.perf_counter()
                sampled = gen.decode_forward(
                    None,
                    None,
                    page_table=None,
                    kv_cache=gen._ensure_kv_cache(),
                    sampling_mode="device",
                    enable_trace=True,
                    active_batch=args.slots,
                )
                row_tokens = gen._sampled_to_torch(sampled)[: args.active].tolist()
                latencies.append((time.perf_counter() - t0) * 1e3)
                tokens.append(row_tokens)

            per_row = [[step[row] for step in tokens] for row in range(args.active)]
            runs[label] = {
                "tokens_per_row": per_row,
                "median_step_ms": statistics.median(latencies),
                "tps_user": 1e3 / statistics.median(latencies),
            }

            if gating:
                # -- mask mechanics, on the live trace's own current_pos -------
                current_pos = gen._trace_inputs[1]
                device_positions = ttnn.to_torch(ttnn.get_device_tensors(current_pos)[0]).reshape(-1)
                mask = gen.model._decode_active_mask(current_pos)
                mask_host = _mask_to_torch(mask)
                expected = (device_positions >= 0).to(torch.float32)
                checks["mask_matches_positions"] = {
                    "passed": bool(torch.equal(mask_host, expected)),
                    "device_positions": device_positions.tolist(),
                    "mask": mask_host.tolist(),
                }
                checks["mask_survives_replays"] = {
                    "passed": bool(
                        torch.equal(
                            mask_host,
                            torch.cat(
                                (
                                    torch.ones(args.active),
                                    torch.zeros(args.slots - args.active),
                                )
                            ),
                        )
                    ),
                    "replays_since_install": args.steps - 1,
                    "note": (
                        "every live row's current_pos advanced by the traced ttnn.plus_one; every "
                        "inactive row is still -1, so the mask is unchanged without any host refresh"
                    ),
                }
                ttnn.deallocate(mask, True)

        gated_rows = runs["gated"]["tokens_per_row"]
        distinct_per_row = [len(set(row)) for row in gated_rows]
        rows_all_distinct = len({tuple(row) for row in gated_rows}) == len(gated_rows)
        checks["outputs_are_varied"] = {
            # Without this the equality leg is vacuous: identical streams of one
            # repeated token agree no matter what the expert path did.
            "passed": bool(min(distinct_per_row) >= 5 and rows_all_distinct),
            "distinct_tokens_per_row": distinct_per_row,
            "rows_differ_from_each_other": rows_all_distinct,
            "decoded": [gen.tokenizer.decode(row) for row in gated_rows],
        }

        live_equal = runs["ungated"]["tokens_per_row"] == runs["gated"]["tokens_per_row"]
        checks["live_rows_token_identical"] = {
            "passed": bool(live_equal),
            "rows_compared": args.active,
            "tokens_per_row": args.steps,
            "ungated": runs["ungated"]["tokens_per_row"],
            "gated": runs["gated"]["tokens_per_row"],
        }
        result["runs"] = {label: {k: v for k, v in run.items() if k != "tokens_per_row"} for label, run in runs.items()}
        result["speedup"] = runs["ungated"]["median_step_ms"] / runs["gated"]["median_step_ms"]
        result["trace_stats"] = dict(gen.trace_stats)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    failed = [name for name, check in checks.items() if not check.get("passed")]
    result["failed"] = failed
    (HERE / "inactive_row_gating_probe.json").write_text(json.dumps(result, indent=2))
    for name, check in checks.items():
        print(f"{'PASS' if check.get('passed') else 'FAIL'}  {name}")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    if failed:
        raise SystemExit(f"failed checks: {failed}")


if __name__ == "__main__":
    main()

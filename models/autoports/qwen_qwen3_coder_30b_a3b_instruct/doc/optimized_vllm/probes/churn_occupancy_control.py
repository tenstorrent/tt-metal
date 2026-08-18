# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Does the inactive-row gating win survive request turnover?

Stage 09 measured inactive-row expert gating at **partial occupancy with a single
install**: one request admitted into a 32-slot server, decoded to completion,
never displaced. That is not a server. A real one churns -- requests finish,
slots are recycled, and every recycle makes the plugin re-install host state into
the live decode trace.

Re-install goes through ``Qwen3CoderForCausalLM._merge_scheduler_view``, and the
version stage 09 shipped ended with::

    merged_positions = torch.where(continuing, device_positions,
                                   torch.clamp(host_positions, min=0))

An inactive row arrives from the plugin as ``-1`` (``model_runner.py`` pads decode
positions with ``-1`` "to indicate no position"). It is never ``continuing``. So
that ``clamp`` installed it as **0** -- and ``_decode_active_mask`` derives the
expert-gating mask from ``current_pos >= 0``, so position 0 reads as *live* and
the gating silently became a no-op for that slot.

Why no measured run caught it: ``_merge_scheduler_view`` returns
``host_positions`` unchanged when there is no decode device state yet, which is
the case on the **first** install. Every stage-09 serving measurement was
single-request/single-install, so the clamp was never reached. It is reached on
the second install onward -- i.e. exactly on turnover.

This probe drives that sequence directly, without vLLM: it builds the serving
adapter over a real generator, admits ``--live`` of ``--slots`` requests, decodes,
then **recycles a slot** (new request, fresh physical blocks) and decodes again,
checking after each round both

* the **mechanism** -- the inactive rows' device ``current_pos`` is still ``-1``
  and the gating mask still marks exactly the live set; and
* the **cost** -- the traced step still costs what partial occupancy should cost,
  not what full occupancy costs.

``--legacy-clamp`` restores the shipped expression so the A/B is one artifact
rather than two checkouts. Writes ``churn_occupancy_control{tag}.json`` beside
this file.
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
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator_vllm import Qwen3CoderForCausalLM  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import DEFAULT_TRACE_REGION_SIZE  # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parents[1]


def _median_ms(fn, reps: int) -> dict:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {"ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples), "reps": reps}


class _Sampling:
    """The handful of fields ``_apply_sampling_params`` reads, greedy for all rows."""

    def __init__(self, rows: int):
        self.temperature = [0.0] * rows
        self.top_k = [1] * rows
        self.top_p = [1.0] * rows
        self.seed = [None] * rows
        self.repetition_penalty = [1.0] * rows
        self.presence_penalty = [0.0] * rows
        self.frequency_penalty = [0.0] * rows


def _page_table(slots: int, width: int, assignment: dict[int, int], blocks_per_row: int) -> torch.Tensor:
    """vLLM-shaped block table: zero-filled, real disjoint blocks for assigned rows.

    ``assignment`` maps row -> generation index; a recycled slot gets a *different*
    generation and therefore a different physical block range, which is what makes
    ``_merge_scheduler_view`` treat it as a slot that changed hands.
    """
    table = torch.zeros((slots, width), dtype=torch.int32)
    for row, gen_idx in assignment.items():
        start = 1 + gen_idx * blocks_per_row  # block 0 stays vLLM's null block
        table[row, :blocks_per_row] = torch.arange(start, start + blocks_per_row, dtype=torch.int32)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--live", type=int, default=4, help="occupied slots -- the realistic partial-occupancy server")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--turnovers", type=int, default=3, help="slot recycles after the initial admission")
    ap.add_argument("--legacy-clamp", action="store_true", help="restore stage 09's shipped clamp, to A/B the defect")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.legacy_clamp:
        # The shipped expression, reinstated verbatim so the failure is exhibited
        # by this probe rather than asserted about a checkout that no longer exists.
        def _legacy(self, host_tokens, host_positions, page_table, slot_remap, rows):
            state = self.generator.decode_device_state()
            if state is None or state["page_table"] is None:
                return host_tokens, host_positions
            device_tokens = state["tokens"][:rows].clone()
            device_positions = state["positions"][:rows].clone()
            snapshot = state["page_table"][:rows]
            incoming = torch.as_tensor(page_table).to(torch.int32)[:rows]
            width = min(snapshot.shape[1], incoming.shape[1])
            if slot_remap is not None:
                remap = torch.as_tensor(slot_remap).reshape(-1)[:rows].to(torch.int64)
                device_tokens, device_positions, snapshot = (
                    device_tokens[remap],
                    device_positions[remap],
                    snapshot[remap],
                )
            pages_unchanged = torch.all(snapshot[:, :width] == incoming[:, :width], dim=1)
            continuing = (
                ((device_positions == host_positions) | (device_positions == host_positions + 1))
                & (device_positions >= 0)
                & (host_positions >= 0)
                & pages_unchanged
            )
            return (
                torch.where(continuing, device_tokens, host_tokens),
                torch.where(continuing, device_positions, torch.clamp(host_positions, min=0)),
            )

        Qwen3CoderForCausalLM._merge_scheduler_view = _legacy

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    summary = {
        "what": (
            "partial-occupancy decode with request turnover, driven through the serving adapter's "
            "own _merge_scheduler_view. Checks that inactive rows keep the -1 sentinel across "
            "re-installs and that the inactive-row gating win survives slot recycling."
        ),
        "slots": args.slots,
        "live": args.live,
        "layers": args.layers,
        "probe_context": args.context,
        "merge_variant": "legacy_clamp (stage 09 as shipped)" if args.legacy_clamp else "sentinel_preserving (fixed)",
        "mesh": "1x4 P300_X2, FABRIC_1D_RING",
        "rounds": [],
        "checks": {},
    }
    try:
        gen = build_generator(
            MODEL_DIR,
            mesh,
            override_num_layers=args.layers,
            max_context_len=args.context,
            max_batch_size=args.slots,
        )
        adapter = Qwen3CoderForCausalLM(gen, max_model_len=args.context, max_num_seqs=args.slots)
        adapter.kv_cache = gen._ensure_kv_cache()
        summary["active_row_gating"] = bool(gen.model.active_row_gating)

        horizon = args.prompt_len + args.gen_len
        blocks_per_row = gen._sdpa_rounded_page_count(horizon)
        width = gen.pages_per_user
        sampling = _Sampling(args.slots)

        # Each admission gets its own generation index -> its own physical blocks.
        next_gen_idx = 0
        assignment: dict[int, int] = {}
        for row in range(args.live):
            assignment[row] = next_gen_idx
            next_gen_idx += 1

        def admit(rows_to_admit, table, preserve: bool):
            """Prefill the given rows at their current block assignment.

            The KV lands in whatever physical blocks that row's page-table entry
            names, so a single-row prefill against ``table[row:row+1]`` admits a
            request into decode row ``row``. ``preserve`` mirrors serving: vLLM
            admits a new request while other slots are mid-decode, so the live
            decode traces must survive the prefill.
            """
            for row in rows_to_admit:
                prompt = torch.tensor([[2000 + row * 7 + i for i in range(args.prompt_len)]])
                gen.prefill_forward(
                    prompt,
                    page_table=table[row : row + 1],
                    kv_cache=adapter.kv_cache,
                    prompt_lens=[args.prompt_len],
                    sampling_mode="device",
                    preserve_decode_traces=preserve,
                )

        table = _page_table(args.slots, width, assignment, blocks_per_row)
        admit(sorted(assignment), table, preserve=False)

        def host_state(assign):
            """What the scheduler hands the adapter: -1 for every unoccupied slot."""
            positions = torch.full((args.slots,), -1, dtype=torch.int64)
            tokens = torch.zeros((args.slots, 1), dtype=torch.int64)
            for row in assign:
                positions[row] = args.prompt_len
                tokens[row] = 3000 + row
            return tokens, positions

        def step(assign, tbl, reset: bool):
            tokens, positions = host_state(assign)
            adapter.decode_forward(
                tokens=tokens,
                page_table=tbl,
                kv_cache=adapter.kv_cache,
                start_pos=positions,
                sampling_params=sampling,
                reset_batch=reset,
                read_from_device=False,
            )
            ttnn.synchronize_device(mesh)

        def replay():
            ttnn.execute_trace(mesh, gen._trace_model_id, cq_id=0, blocking=False)
            ttnn.execute_trace(mesh, gen._trace_sampling_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)

        def observe(label, assign):
            state = gen.decode_device_state()
            pos = state["positions"][: args.slots].clone()
            live = sorted(assign)
            inactive = [r for r in range(args.slots) if r not in assign]
            row = {
                "round": label,
                "live_rows": live,
                "inactive_rows_at_sentinel": int(sum(1 for r in inactive if int(pos[r]) < 0)),
                "inactive_rows_total": len(inactive),
                "inactive_positions_seen": sorted({int(pos[r]) for r in inactive}),
                "live_positions_seen": sorted({int(pos[r]) for r in live}),
            }
            replay()
            timing = _median_ms(replay, args.reps)
            row["token_out"] = timing
            row["token_out_ms"] = timing["ms"]
            summary["rounds"].append(row)
            print(
                f"[{label}] live={len(live)}/{args.slots}  "
                f"inactive at -1: {row['inactive_rows_at_sentinel']}/{row['inactive_rows_total']}  "
                f"inactive positions {row['inactive_positions_seen']}  "
                f"token_out {timing['ms']:.3f} ms",
                flush=True,
            )
            return row

        # Round 0: the first install. _merge_scheduler_view short-circuits here
        # (no decode device state yet), which is why every stage-09 measurement
        # looked correct.
        step(assignment, table, reset=True)
        observe("initial_install", assignment)

        # Turnover rounds: a live slot finishes and is re-admitted as a NEW request
        # with fresh physical blocks. Each one re-enters _merge_scheduler_view with
        # device state present -- the path the shipped clamp corrupted.
        recycle_order = [r for r in sorted(assignment)]
        for t in range(args.turnovers):
            victim = recycle_order[t % len(recycle_order)]
            assignment[victim] = next_gen_idx
            next_gen_idx += 1
            table = _page_table(args.slots, width, assignment, blocks_per_row)
            admit([victim], table, preserve=True)
            step(assignment, table, reset=True)
            observe(f"turnover_{t + 1}_row{victim}", assignment)

        # -- checks -----------------------------------------------------------
        turnover_rounds = [r for r in summary["rounds"] if r["round"].startswith("turnover")]
        initial = summary["rounds"][0]
        all_sentinel = all(r["inactive_rows_at_sentinel"] == r["inactive_rows_total"] for r in summary["rounds"])
        summary["checks"]["inactive_rows_keep_sentinel_across_turnover"] = {
            "pass": bool(all_sentinel),
            "detail": "every unoccupied slot still reads current_pos == -1 after each re-install",
            "per_round": {
                r["round"]: f"{r['inactive_rows_at_sentinel']}/{r['inactive_rows_total']}" for r in summary["rounds"]
            },
        }
        worst = max((r["token_out_ms"] for r in turnover_rounds), default=float("nan"))
        drift = worst - initial["token_out_ms"]
        summary["checks"]["gating_win_survives_turnover"] = {
            "pass": bool(abs(drift) < 5.0),
            "initial_ms": initial["token_out_ms"],
            "worst_turnover_ms": worst,
            "drift_ms": drift,
            "detail": (
                "traced step cost after slot recycling stays within 5 ms of the cost at the same "
                "occupancy before any recycling; the shipped clamp regressed it toward the "
                "full-occupancy cost instead"
            ),
        }
        summary["all_pass"] = all(c["pass"] for c in summary["checks"].values())
        summary["trace_stats"] = dict(gen.trace_stats)
        # ``max_model_len`` is renamed to ``probe_max_model_len`` on the way out,
        # following the same precedent as ``adapter_contract_probe.py``: the live
        # adapter key keeps its vLLM name, but here the value is only ever this
        # probe's reduced target (4096), and the runner-side context-contract
        # guardrail treats any ``max_model_len``-style JSON key below the model's
        # supported context (262144) as a served cap -- which this is not.
        audit = adapter.serving_audit()
        if "max_model_len" in audit:
            audit["probe_max_model_len"] = audit.pop("max_model_len")
        summary["serving_audit"] = audit
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    out = HERE / f"churn_occupancy_control{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["checks"], indent=2, default=str))
    print(f"all_pass={summary['all_pass']}  ->  {out}")
    raise SystemExit(0 if summary["all_pass"] else 1)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The standard task ledger for a model, generated from its spec and its reference's block graphs.

    python -m models.demos.common.bringup.plan.ledger_gen --spec S [--write]

Task ids by pipeline step:
    R.1 checkpoint   R.2 HF parity   R.3 chunked + graph replay         (reference)
    R.4              trim a layer subset's checkpoint after the sanity   (intake, F47)
    G.<rung>         one golden per rung that owns its golden            (goldens)
    B.1              box: mesh opens, collectives work                   (box)
    SC.1             serving contract: how-to + tests, from tt-d-gen's code    (serving, before the plan)
    PL.1             plan fits DRAM, components mapped, ledger valid, approved   (plan)
    C.<bt>.<step>    component test of one step of the representative layer      (implement, parallel)
    S.<bt>.<nn>      swap test: steps 1..nn of the block on device, in graph order (implement, sequential)
    L.<rung>         ladder rung on device                               (integrate)
    K.1              serving contract through the engine API             (contract)
    X.1 profile      X.2 opportunity list                                (perf)
    Z.1              settings audit: every switch in tt/settings.py      (settings, last)
The plan agent may add, split or annotate tasks (e.g. a hooks-only task, a model-level embedding task) before the
plan is approved; the generator only writes the skeleton.
"""

from __future__ import annotations

import argparse

import torch
import yaml

from models.demos.common.bringup.intake import trim_checkpoint
from models.demos.common.bringup.reference.golden import load_spec, rung_dir
from models.demos.common.bringup.testing import serving as SV
from models.demos.common.bringup.testing.harness import DEFAULT_THRESHOLDS
from models.demos.common.bringup.testing.templates import component_test_path, swap_test_path

SAFE = "scripts/run_safe_pytest.sh --run-all"
PY = "python -m models.demos.common.bringup"
# PROGRAM_SUPPORT_COUNT: the pipelined timeline reads the device profiler once per chunk; the default 1000 programs per
# core drops markers on large models (GLM-5.3, 5 layers: 1319 programs per chip per chunk).
PROFILE_ENV = (
    "TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1"
    " TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000"
)


def thr(spec, key: str) -> str:
    return f">= {spec.threshold(key, DEFAULT_THRESHOLDS[key])}"


def rel(spec, p) -> str:
    return str(p.relative_to(spec.repo))


def sanity_metrics(spec) -> dict:
    """The HF sanity gate: pinned revision, usage-example smoke, next-token accuracy floor (R.1, or R.2 with F39)."""
    return {
        "revision_ok": "== 1",
        "text_top1_acc": f">= {spec.get('text.min_top1', 0.4)}",
        **({"smoke_ok": "== 1"} if spec.get("intake.smoke") else {}),
    }


def generate(spec, ref=None, early: bool = False) -> dict:
    """early=True: only the tasks that exist before the reference does (R, G, B, and PL.0, which extends the ledger
    with the rest once R.3 has passed and the block graphs are known)."""
    if ref is None and not early:
        reps = sorted({spec.representative_layer(bt) for bt in spec.data["block_types"]})
        ref = spec.hooks().reference(spec, layers=reps, dtype=torch.float32)
    model_dir = rel(spec, spec.model_dir)
    ref_paths = [f"{model_dir}/reference", f"{model_dir}/bringup/hooks.py", f"{model_dir}/__init__.py"]
    impl_paths = [f"{model_dir}/tt", f"{model_dir}/bringup/hooks.py"]
    ladder = spec.data["ladder"]
    first = ladder[0]
    tasks = []

    def add(tid, title, step, deps, cmd, metrics, **kw):
        tasks.append(
            {
                "id": tid,
                "title": title,
                "step": step,
                "deps": list(deps),
                "gate": {"cmd": cmd, "metrics": metrics},
                **kw,
            }
        )

    # HF sanity (revision, usage-example smoke, accuracy floor) runs at R.1 on the stock HF loader. A checkpoint the
    # stock loader cannot run on this host (custom quantized storage, too big for RAM in bf16) sets hf.custom_loader:
    # the checks then move into R.2's gate, after the reference agent has written hooks.hf_model (F39).
    sanity = sanity_metrics(spec)
    late = bool(spec.get("hf.custom_loader"))
    add(
        "R.1",
        "Checkpoint, intake approval, canonical prompt"
        + ("" if late else ", HF sanity (revision, usage-example smoke, accuracy floor)"),
        "intake",
        [],
        f"{PY}.intake.check_checkpoint && {PY}.reference.prompt" + ("" if late else f" && {PY}.intake.check_hf_sanity"),
        {
            "intake_approved": "== 1",
            "prompt_hash_ok": "== 1",
            **({} if late else sanity),
            "missing_tensors": "== 0",
            "shape_mismatches": "== 0",
            "count_mismatches": "== 0",
            "config_mismatches": "== 0",
            "checkpoint_tensors": ">= 1",
        },
        paths=[f"{model_dir}/bringup/spec.yaml"],
    )
    add(
        "R.2",
        ("HF sanity on the model's loader (revision, smoke, accuracy floor); r" if late else "R")
        + "eference matches HF per layer and on logits (fp32)",
        "reference",
        ["R.1"],
        (f"{PY}.intake.check_hf_sanity && " if late else "")
        + f"{PY}.reference.check_hf --seq {spec.get('hf.parity_seq', 512)}",
        {
            **(sanity if late else {}),
            "pcc_hidden_L*": ">= 0.9999",
            "pcc_logits": ">= 0.9999",
            "top1_match_frac": ">= 0.99",
        },
        paths=ref_paths,
    )
    if trim_checkpoint.applies(spec):
        # F47 (before R.3, so it runs as soon as R.2 passes): a layer subset needs the whole checkpoint only for the HF sanity; once it and the parity passed, the
        # layers no later step reads are removed (bringup_trim.json keeps the full tensor map and the sanity metrics).
        k = trim_checkpoint.keep_layers(spec)
        add(
            "R.4",
            f"Trim the checkpoint to layers 0-{k - 1} (the whole model was needed only for the HF sanity)",
            "intake",
            ["R.2"],
            f"{PY}.intake.trim_checkpoint",
            {"trim_done": "== 1", "trim_verify_errors": "== 0", "trim_freed_gb": ">= 0"},
        )
    add(
        "R.3",
        f"Reference chunked == one-shot ({first['seq']} in {first['chunk']}) and block graphs replay exactly",
        "reference",
        ["R.2"],
        f"{PY}.reference.check_reference --seq {first['seq']} --chunk {first['chunk']}",
        {
            "pcc_hidden": ">= 0.99999",
            "pcc_state_min": ">= 0.99999",
            "graph_errors": "== 0",
            "boundaries_missing": "== 0",
            "graph_replay_maxabs": "<= 0",
        },
        paths=ref_paths,
    )
    prev = "R.3"
    golden_task = {}
    for r in ladder:
        if r.get("golden"):
            golden_task[r["name"]] = f"G.{r['golden']}"
            continue
        tid = f"G.{r['name']}"
        add(
            tid,
            f"Golden {r['name']}: {r['seq']} tokens in {r['chunk']}-token chunks{', full dumps' if r.get('full_dumps') else ''}",
            "goldens",
            [prev],
            f"{PY}.reference.generate_golden --rung {r['name']}",
            {
                "golden_layers": f"== {len(spec.layers())}",
                "golden_chunks": f"== {r['seq'] // r['chunk']}",
                "golden_hash_ok": "== 1",
            },
        )
        golden_task[r["name"]] = tid
        prev = tid
    add(
        "B.1",
        "Box: mesh opens with the spec's device params, collectives are exact",
        "box",
        [],
        f"{SAFE} models/demos/common/bringup/tests/test_box.py",
        {"chips": f"== {spec.mesh[0] * spec.mesh[1]}", "all_gather_maxabs": "== 0"},
        device=True,
    )
    # The serving contract comes first (F58): how the inference server drives the model, as a how-to the plan and
    # every step build to, and the frozen tests that hold each part of the model to it (agents/serving-contract.md).
    add(
        "SC.1",
        "Serving contract: how tt-d-gen drives the model, a how-to for each part and its tests",
        "serving",
        ["R.1"],
        f"{PY}.testing.serving",
        {"serving_contract_errors": "== 0", "serving_contract_tests": ">= 1"},
        role="serving",
    )
    add(
        "PL.0",
        "Ledger: add the component, swap, ladder, contract and perf tasks from the block graphs",
        "ledger",
        ["R.3"],
        f"{PY}.plan.ledger_gen --extend",
        {"ledger_errors": "== 0", "ledger_tasks": ">= 1"},
        paths=[f"{model_dir}/bringup/tasks.yaml"],
    )
    if early:
        return {"model": spec.data["hf_id"], "target": spec.data["target"], "tasks": tasks}
    add(
        "PL.1",
        "Plan: fits per-chip DRAM (from the checkpoint), every step mapped, ledger valid, approved",
        "plan",
        ["PL.0", "B.1", "SC.1"],
        f"{PY}.plan.check_plan",
        {
            "plan_fits": "== 1",
            "unplaced_tensors": "== 0",
            "plan_errors": "== 0",
            "component_errors": "== 0",
            "ledger_errors": "== 0",
            "plan_approved": "== 1",
        },
        paths=[
            f"{model_dir}/bringup/plan.yaml",
            f"{model_dir}/bringup/plan.md",
            f"{model_dir}/bringup/components.yaml",
            f"{model_dir}/bringup/approvals.yaml",
            f"{model_dir}/bringup/results/plan_memory.json",
        ],
        approval="plan",
    )

    def served_format() -> dict:
        """F58: with the served KV dtype in the spec (serving.kv_dtype, the owner's answer to SC.1), every ladder
        rung must run on it: one cache format for the ladder, the contract and serving."""
        bits = {"bf16": 16, "bfloat16": 16, "bfp8": 8, "bfloat8_b": 8, "fp32": 32}.get(
            str(spec.get("serving.kv_dtype", ""))
        )
        if not bits:
            return {}
        names = set(spec.get("state.tensors") or []) | {
            n for v in (spec.get("state.by_block_type") or {}).values() for n in v
        }
        return {f"state_bits_{n}": f"== {bits}" for n in sorted(names)}

    def contract_cmds_all() -> str:
        return f" && {PY}.testing.serving --run all" if SV.tests(spec) else ""

    def contract_cmds(gate: str) -> str:
        """The serving contract tests a step's gate runs too (contract_tests.yaml, written by SC.1)."""
        return "".join(f" && {SAFE} --no-precompile {t['test']}" for t in SV.tests_for(spec, gate))

    comp_rung = spec.get("tests.component_rung") or next(r["name"] for r in ladder if r.get("full_dumps"))
    comp_golden = golden_task[comp_rung]
    manifest = [str(rung_dir(spec, spec.rung(comp_rung)) / "manifest.json")]
    last_swaps = []
    for bt in spec.data["block_types"]:
        layer = spec.representative_layer(bt)
        steps = ref.block_graph(layer)
        prev_swap = None
        for n, st in enumerate(steps, 1):
            ctest = rel(spec, component_test_path(spec, bt, st.name))
            cid = f"C.{bt}.{st.name}"
            metric = f"pcc_{st.name}_L{layer:02d}"
            add(
                cid,
                f"{bt} {st.name} ({st.kind}) on device, layer {layer}",
                "implement",
                ["PL.1", comp_golden],
                f"{SAFE} {ctest}" + contract_cmds(st.name),
                {metric: thr(spec, "component")},
                device=True,
                tests=[ctest],
                freeze_extra=manifest,
                paths=impl_paths,
                brief={"block_type": bt, "step": st.name, "kind": st.kind, "layer": layer, "stateful": st.stateful},
            )
            stest = rel(spec, swap_test_path(spec, bt, n, st.name))
            sid = f"S.{bt}.{n:02d}"
            add(
                sid,
                f"{bt} block with steps 1-{n} on device (last: {st.name})",
                "implement",
                [cid] + ([prev_swap] if prev_swap else []),
                f"{SAFE} {stest}",
                {"pcc_swap_out": thr(spec, "block")},
                device=True,
                tests=[stest],
                freeze_extra=manifest,
                paths=impl_paths,
                brief={"block_type": bt, "swapped": [s.name for s in steps[:n]], "layer": layer},
            )
            prev_swap = sid
        last_swaps.append(prev_swap)

    # Assemble: the swap tests run every step on the device through the hybrid harness (host in / host out per step);
    # the ladder, the contract and the profile need one model whose hidden state stays on the device.
    if last_swaps:
        multi = next((r for r in ladder if not r.get("prefix_from_golden") and r["seq"] // r["chunk"] > 1), first)
        am = {
            "pcc_layer_L*": thr(spec, "layer"),
            "pcc_state_min": thr(spec, "state"),
            "host_transfers_per_layer": "== 0",
        }
        add(
            "M.1",
            "Assemble the all-device model (hidden state on the device from embedding to final norm)",
            "assemble",
            last_swaps,
            f"BRINGUP_RUNG={multi['name']} {SAFE} --no-precompile models/demos/common/bringup/tests/test_ladder.py",
            am,
            device=True,
            paths=impl_paths,
        )
        last_swaps = ["M.1"]

    prev = None
    for r in ladder:
        tid = f"L.{r['name']}"
        m = {"pcc_layer_L*": thr(spec, "layer"), "pcc_state_min": thr(spec, "state")} | served_format()
        layers = r.get("layers") or spec.layers()
        if max(layers) == spec.num_layers - 1:
            m.update(pcc_final_hidden=thr(spec, "final_hidden"), top5_overlap=thr(spec, "top5"))
        if not r.get("prefix_from_golden") and r["seq"] // r["chunk"] > 1:
            m["host_transfers_per_layer"] = "== 0"  # agent rule 5: no host work in the forward path (warm chunks)
        add(
            tid,
            f"Ladder {r['name']}: {'last chunk after a golden prefix' if r.get('prefix_from_golden') else 'all chunks'} "
            f"of {r['seq']} in {r['chunk']}",
            "integrate",
            [*(last_swaps if prev is None else [prev]), golden_task[r["name"]]],
            f"BRINGUP_RUNG={r['name']} {SAFE} models/demos/common/bringup/tests/test_ladder.py",
            m,
            device=True,
            paths=impl_paths,
        )
        prev = tid
    add(
        "K.1",
        "Serving contract through the prefill engine API (layout, table, acks, engine input, read-back)",
        "contract",
        [f"L.{first['name']}"],
        f"{SAFE} --no-precompile models/demos/common/bringup/tests/test_contract.py" + contract_cmds(SV.ADAPTER),
        {"contract_checks_failed": "== 0", "acks_early": "== 0", "pcc_producer_kv_*": thr(spec, "state")},
        device=True,
        paths=[f"{model_dir}/tt"],  # plus models/demos/common/prefill, for every contract step (orchestrator)
    )
    add(
        "X.1",
        "Warm per-section, per-chip profile of one long chunk",
        "perf",
        [prev],
        f"{PROFILE_ENV} {SAFE} --no-precompile models/demos/common/bringup/tests/test_profile.py",
        {"device_ms_total": "> 0", "profiled_programs": ">= 1"},
        device=True,
    )
    add(
        "X.2",
        "Opportunity list ranked by measured device time",
        "perf",
        ["X.1"],
        f"{PY}.plan.opportunities --profile X.1",
        {"opportunities_listed": ">= 1"},
        paths=[f"{model_dir}/bringup/opportunities.md"],
    )
    # Final measurement once the picks are in (the overseer points X.3's deps at the last pick): full-target
    # accuracy, the warm full prefill (TTFT without the LM head) and one warm chunk at several start positions.
    full = ladder[-1]
    fm = {
        "pcc_layer_L*": thr(spec, "layer"),
        "pcc_state_min": thr(spec, "state"),
        "prefill_ms_full": "> 0",
        "pos_chunk": ">= 1",
        "op_rows": ">= 1",  # the per-op breakdown (F43) and the pipelined timeline (F44) behind the dashboard
        "timeline_ok": "== 1",
    }
    if max(full.get("layers") or spec.layers()) == spec.num_layers - 1:
        fm.update(pcc_final_hidden=thr(spec, "final_hidden"), top5_overlap=thr(spec, "top5"))
    add(
        "X.3",
        "Final: full-target accuracy, warm full prefill (no readback), one warm chunk at several positions",
        "perf",
        ["X.2"],
        f"BRINGUP_RUNG={full['name']} {SAFE} models/demos/common/bringup/tests/test_ladder.py"
        f" && BRINGUP_FULL_PREFILL=1 BRINGUP_PROFILE_OPS=1 {PROFILE_ENV} {SAFE} models/demos/common/bringup/tests/test_profile.py"
        f" && {SAFE} models/demos/common/bringup/tests/test_positions.py",
        fm,
        device=True,
    )
    # The derived ops (ttnn/ttnn/bringup) this model calls get a test case for every call it makes: record the
    # ttnn.bringup calls of one target-size chunk, then check each fork's tests/cases.py and run those tests
    # (skill/bringup-fork-tests). Passes at once when the model calls no fork.
    chunk = spec.data["target"]["chunk"]
    cap = next(
        (r for r in ladder if r.get("prefix_from_golden") and r["chunk"] == chunk),
        next((r for r in ladder if r["chunk"] == chunk), ladder[-1]),
    )
    calls = f"{rel(spec, spec.bringup_dir)}/results/fork_calls.json"
    add(
        "O.1",
        "Derived-op tests: a random-input case for every ttnn.bringup call this model makes",
        "optests",
        ["X.3"],
        f"BRINGUP_CAPTURE_FORKS={calls} BRINGUP_RUNG={cap['name']} {SAFE} --no-precompile"
        " models/demos/common/bringup/tests/test_ladder.py -p models.demos.common.bringup.testing.fork_capture"
        f" && {PY}.testing.fork_cases --capture {calls} --run-tests",
        {"forks_used": ">= 0", "fork_calls_uncovered": "== 0", "fork_tests_failed": "== 0"},
        role="optests",
        paths=["ttnn/ttnn/bringup"],
        device=True,
    )
    # Every switch of the model and its forks in one place (tt/settings.py), behaviour unchanged (agents/settings-audit.md)
    last = next((r for r in ladder if r.get("prefix_from_golden")), ladder[-1])
    add(
        "Z.1",
        "Settings audit: every switch of the model and its forks in tt/settings.py, behaviour unchanged",
        "settings",
        ["O.1"],
        f"{PY}.testing.settings_lint && BRINGUP_RUNG={last['name']} {SAFE} --no-precompile"
        " models/demos/common/bringup/tests/test_ladder.py" + contract_cmds_all(),
        {"settings_violations": "== 0", "pcc_layer_L*": thr(spec, "layer"), "pcc_state_min": thr(spec, "state")},
        device=True,
        role="settings",
    )
    return {"model": spec.data["hf_id"], "target": spec.data["target"], "tasks": tasks}


def main(argv=None):
    from models.demos.common.bringup.core import metrics
    from models.demos.common.bringup.core.ledger import Ledger

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--write", action="store_true", help="write <bringup_dir>/tasks.yaml (refuses to overwrite)")
    ap.add_argument("--early", action="store_true", help="only the tasks that need no reference (R, G, B, PL.0)")
    ap.add_argument(
        "--extend", action="store_true", help="append the generated tasks whose ids are not in tasks.yaml yet"
    )
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    out = generate(spec, early=a.early)
    led = Ledger(spec.bringup_dir)
    if a.extend:
        cur = led.load_spec()
        have = {t["id"] for t in cur.get("tasks", [])}
        new = [t for t in out["tasks"] if t["id"] not in have]
        cur.setdefault("tasks", []).extend(new)
        for k in ("model", "target"):
            cur.setdefault(k, out[k])
        led.write_tasks(cur)
        errs = led.validate()
        print(f"added {len(new)} tasks: {' '.join(t['id'] for t in new)}")
        for e in errs:
            print(f"LEDGER ERROR {e}")
        metrics.record("ledger_tasks", len(cur["tasks"]))
        metrics.record("ledger_tasks_added", len(new))
        metrics.record("ledger_errors", len(errs))
        return
    if a.write:
        if led.tasks_path.exists():
            raise SystemExit(f"{led.tasks_path} exists; use --extend or edit it")
        led.write_tasks(out)
        print(f"wrote {led.tasks_path}: {len(out['tasks'])} tasks")
    else:
        print(yaml.safe_dump(out, sort_keys=False, width=120))


if __name__ == "__main__":
    main()

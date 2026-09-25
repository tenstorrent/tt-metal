# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The standard task ledger for a model, generated from its spec and its reference's block graphs.

    python -m models.demos.common.bringup.plan.ledger_gen --spec S [--write]

Task ids by pipeline step:
    R.1 checkpoint   R.2 HF parity   R.3 chunked + graph replay         (reference)
    G.<rung>         one golden per rung that owns its golden            (goldens)
    B.1              box: mesh opens, collectives work                   (box)
    PL.1             plan fits DRAM, components mapped, ledger valid, approved   (plan)
    C.<bt>.<step>    component test of one step of the representative layer      (implement, parallel)
    S.<bt>.<nn>      swap test: steps 1..nn of the block on device, in graph order (implement, sequential)
    L.<rung>         ladder rung on device                               (integrate)
    K.1              serving contract through the engine API             (contract)
    X.1 profile      X.2 opportunity list                                (perf)
The plan agent may add, split or annotate tasks (e.g. a hooks-only task, a model-level embedding task) before the
plan is approved; the generator only writes the skeleton.
"""

from __future__ import annotations

import argparse

import torch
import yaml

from models.demos.common.bringup.reference.golden import load_spec, rung_dir
from models.demos.common.bringup.testing.harness import DEFAULT_THRESHOLDS
from models.demos.common.bringup.testing.templates import component_test_path, swap_test_path

SAFE = "scripts/run_safe_pytest.sh --run-all"
PY = "python -m models.demos.common.bringup"
PROFILE_ENV = "TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1"


def thr(spec, key: str) -> str:
    return f">= {spec.threshold(key, DEFAULT_THRESHOLDS[key])}"


def rel(spec, p) -> str:
    return str(p.relative_to(spec.repo))


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

    add(
        "R.1",
        "Checkpoint has every tensor the spec expects; intake approved; canonical prompt built",
        "intake",
        [],
        f"{PY}.intake.check_checkpoint && {PY}.reference.prompt",
        {
            "intake_approved": "== 1",
            "prompt_hash_ok": "== 1",
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
        "Reference matches HF per layer and on logits (fp32)",
        "reference",
        ["R.1"],
        f"{PY}.reference.check_hf --seq {spec.get('hf.parity_seq', 512)}",
        {"pcc_hidden_L*": ">= 0.9999", "pcc_logits": ">= 0.9999", "top1_match_frac": ">= 0.99"},
        paths=ref_paths,
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
    add(
        "PL.0",
        "Ledger: add the component, swap, ladder, contract and perf tasks from the block graphs",
        "plan",
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
        ["PL.0", "B.1"],
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
    )

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
                f"{SAFE} {ctest}",
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

    prev = None
    for r in ladder:
        tid = f"L.{r['name']}"
        m = {"pcc_layer_L*": thr(spec, "layer"), "pcc_state_min": thr(spec, "state")}
        layers = r.get("layers") or spec.layers()
        if max(layers) == spec.num_layers - 1:
            m.update(pcc_final_hidden=thr(spec, "final_hidden"), top5_overlap=thr(spec, "top5"))
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
        f"{SAFE} --no-precompile models/demos/common/bringup/tests/test_contract.py",
        {"contract_checks_failed": "== 0", "acks_early": "== 0", "pcc_producer_kv_*": thr(spec, "state")},
        device=True,
        paths=[f"{model_dir}/tt", "models/demos/common/prefill/adapter.py"],
    )
    add(
        "X.1",
        "Warm per-section, per-chip profile of one long chunk",
        "perf",
        [prev],
        f"{PROFILE_ENV} {SAFE} models/demos/common/bringup/tests/test_profile.py",
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

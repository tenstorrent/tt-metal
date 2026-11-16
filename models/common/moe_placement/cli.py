# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line pipeline. Every stage reads and writes one work directory, so stages can run on different
machines (host stages on any CPU box with numpy; device stages on the Galaxy) and resume after interruption.

  host    download           public multilingual corpus -> raw/                     (once, any model)
  host*   pack               tokenize into sequences/ with the model's tokenizer    (*needs the HF tokenizer)
  device  capture            routing of every sequence -> captures/<seq>/routing.npz
  host    import-routing     routing from any other source (reference model, vLLM, ...)
  host    calibrate          train plans, score on held-out traffic -> calibration/
  host    heldout            held-out balanced sequence for the A/B -> sequences/heldout_mix_bal
  host    preset             plan -> presets/<model>_<plan>.json (validated)
  host    screen             original-layout imbalance, modeled gain, go/no-go -> screen.json
  device  prepare            placed weight cache for a preset
  device  ab                 counterbalanced original vs placed A/B + ship gate -> ab/summary.json
  device  profile            MoE stage shares (routed-expert share of prefill time) -> profile/
  host    report             report.md from everything above

``--model`` selects a registered adapter (``models`` lists them). Host stages also accept ``--spec spec.json``
(a MoeSpec as JSON) for models without an adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from . import abtest, calibrate, corpus, routing, screen, traffic
from . import preset as presetlib
from .spec import MoeSpec

HOST_STAGES = {"download", "import-routing", "calibrate", "heldout", "preset", "screen", "report"}


def _log(msg):
    print(msg, flush=True)


def _spec_and_adapter(a, need_adapter: bool):
    adapter = None
    if a.model:
        from .adapters import get_adapter

        adapter = get_adapter(a.model)
        spec = adapter.spec
    elif a.spec:
        spec = MoeSpec.from_json(json.loads(Path(a.spec).read_text()))
    else:
        saved = a.workdir / "spec.json"
        if not saved.exists():
            raise SystemExit("give --model (registered adapter) or --spec spec.json")
        spec = MoeSpec.from_json(json.loads(saved.read_text()))
    if need_adapter and adapter is None:
        raise SystemExit("this stage runs on the device and needs --model with a registered adapter")
    saved = a.workdir / "spec.json"
    if saved.exists():
        old = MoeSpec.from_json(json.loads(saved.read_text()))
        if old != spec:
            raise SystemExit(
                f"{saved} was written for {old.name} ({old.num_experts} experts); use a separate work directory per model"
            )
    else:
        a.workdir.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps(spec.to_json(), indent=1) + "\n")
    return spec, adapter


def _captures(workdir: Path):
    return (
        sorted(d.name for d in (workdir / "captures").iterdir() if (d / "routing.npz").exists())
        if (workdir / "captures").exists()
        else []
    )


# --- stages --------------------------------------------------------------------------------------
def cmd_models(a):
    from .adapters import available

    print(json.dumps(available(), indent=1))


def cmd_download(a):
    print(json.dumps(corpus.download(a.workdir, a.only, a.max_chars)))


def cmd_pack(a):
    spec, adapter = _spec_and_adapter(a, need_adapter=a.tokenizer is None)
    tok = a.tokenizer or adapter.tokenizer_path()
    chat = a.chat_format or (adapter.chat_format if adapter else "hf")
    control = a.control_trace or (adapter.golden_sequence() if adapter else None)
    idx = corpus.pack(a.workdir, tok, spec.sequence_tokens, chat, control, "control", a.cpp_root)
    print(json.dumps(dict(sequences=len(idx)), indent=1))


def cmd_capture(a):
    spec, adapter = _spec_and_adapter(a, need_adapter=True)
    seqs = sorted(d.name for d in (a.workdir / "sequences").iterdir() if (d / "metadata.json").exists())
    if a.only:
        seqs = [s for s in seqs if s in a.only]
    seqs.sort(key=lambda s: (not s.startswith("control"), s))  # control first: it validates the setup
    (a.workdir / "captures").mkdir(parents=True, exist_ok=True)
    st = abtest.State(a.workdir / "captures" / "state.json", model=spec.name)
    for s in seqs:
        out = a.workdir / "captures" / s
        if (out / "routing.npz").exists():
            continue
        if out.exists():
            out.rename(out.with_name(f"{s}.failed-{int(time.time())}"))
        st.save(current=s)
        t0 = time.time()
        try:
            m = adapter.capture(a.workdir / "sequences" / s, out, a.workdir / "captures" / f"{s}.log")
            st.data["runs"][s] = dict(
                status="complete",
                minutes=round((time.time() - t0) / 60, 1),
                routed_layers=m.get("layers"),
                moe_transport=m.get("moe_transport"),
            )
            _log(f"captured {s} in {st.data['runs'][s]['minutes']} min")
        except Exception as e:
            st.data["runs"][s] = dict(status="failed", error=str(e)[:500])
            _log(f"capture {s} FAILED: {e}")
            if s.startswith("control"):
                st.save(current=None, status="failed")
                raise
        st.save(current=None)
    st.save(status="done")


def cmd_import_routing(a):
    spec, _ = _spec_and_adapter(a, need_adapter=False)
    z = np.load(a.npz)
    layers = z["layers"].tolist() if "layers" in z.files else list(range(z["expert_ids"].shape[0]))
    info = routing.import_routing(
        a.workdir / "captures" / a.name / "routing.npz",
        z["expert_ids"],
        layers,
        spec.num_experts,
        spec.top_k,
        z["expert_weights"] if "expert_weights" in z.files else None,
    )
    print(json.dumps(info))


def cmd_calibrate(a):
    spec, _ = _spec_and_adapter(a, need_adapter=False)
    plans = calibrate.FULL_PLANS if a.plans == "full" else a.plans.split(",")
    extra = {}
    for item in a.extra_preset or []:
        name, path = item.split("=", 1)
        corpus_layers = routing.load(a.workdir / "captures" / _captures(a.workdir)[0] / "routing.npz")[1]
        extra[name] = presetlib.owners_per_layer(presetlib.load(Path(path), spec), spec, corpus_layers)
    eval_plans = a.workdir / a.eval_from / "plans_owners.npz" if a.eval_from else None
    calibrate.run(
        a.workdir,
        spec,
        a.out,
        a.steps,
        a.constraint,
        plans,
        extra,
        eval_plans,
        a.layers.split(",") if a.layers else None,
        a.skip_prefix,
        a.workers,
        _log,
    )


def cmd_heldout(a):
    spec, _ = _spec_and_adapter(a, need_adapter=False)
    print(json.dumps(traffic.pack_heldout(a.workdir, a.name, a.seed, spec.sequence_tokens), ensure_ascii=False))


def _preset_path(a, spec):
    return Path(a.out) if getattr(a, "out", None) else a.workdir / "presets" / f"{spec.name}_{a.plan}.json"


def cmd_preset(a):
    spec, _ = _spec_and_adapter(a, need_adapter=False)
    layers, owners = calibrate.load_plan(a.workdir, a.calibration, a.plan)
    p, calib, stats = presetlib.build(
        spec, layers, owners, a.plan, str(a.workdir / a.calibration / "plans_owners.npz"), a.source_main or ""
    )
    path = _preset_path(a, spec)
    presetlib.write(path, spec, p, calib)
    presetlib.load(path, spec)
    print(json.dumps(dict(preset=str(path), **stats)))


def cmd_screen(a):
    spec, _ = _spec_and_adapter(a, need_adapter=False)
    res = dict(model=spec.name, experts_per_chip=spec.experts_per_chip)
    summ = a.workdir / a.calibration / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        res["held_out_utilization_original"] = s["utilization_identity"].get("mix_balanced")
        res["held_out_modeled_peak_change_pct"] = s["peak_change_pct"].get("mix_balanced", {})
    seqs = {}
    held = a.workdir / "heldout" / a.heldout / "routing_spliced.npz"
    if held.exists():
        seqs["heldout"] = routing.load(held)[0]
    for n in _captures(a.workdir):
        if n.startswith("control") and traffic.usable_capture(n):
            seqs[n] = routing.load(a.workdir / "captures" / n / "routing.npz")[0]
    plans = {}
    for pth in a.preset or []:
        p = presetlib.load(Path(pth), spec)
        layers = routing.load(a.workdir / "captures" / _captures(a.workdir)[0] / "routing.npz")[1]
        plans[Path(pth).stem] = presetlib.owners_per_layer(p, spec, layers)
    res["sequences"] = {}
    for name, ids in seqs.items():
        r = screen.imbalance(ids, spec)
        if plans:
            r["modeled_peak_change_pct"] = screen.modeled_gain(ids, spec, plans)
        res["sequences"][name] = r
    prof = sorted((a.workdir / "profile").glob("*/measurement.json")) if (a.workdir / "profile").exists() else []
    shares = [
        json.loads(p.read_text())["profile"]["routed_expert_share"]
        for p in prof
        if json.loads(p.read_text()).get("status") == "complete" and "O" in p.parent.name
    ]
    share = statistics.median(shares) if shares else None
    u = res.get("held_out_utilization_original") or (res["sequences"].get("heldout") or {}).get("utilization")
    plan_name = a.plan
    g = res.get("held_out_modeled_peak_change_pct", {}).get(plan_name)
    if u is not None and g is not None:
        res["verdict"] = screen.verdict(u, g, a.min_modeled_pct, a.max_utilization)
        res["expected"] = screen.expected_gain(g, share)
    (a.workdir / "screen.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def cmd_prepare(a):
    spec, adapter = _spec_and_adapter(a, need_adapter=True)
    presetlib.load(Path(a.preset), spec)
    dest = Path(a.destination)
    if dest.exists():
        raise SystemExit(f"{dest} exists; remove it or choose another destination")
    t0 = time.time()
    adapter.prepare_cache(Path(a.preset), dest, a.workdir / "prepare.log")
    (dest / "moe_placement_preset.json").write_text(Path(a.preset).read_text())
    _log(f"prepared {dest} in {(time.time() - t0) / 60:.1f} min")


def _workloads(a, adapter):
    w = {}
    g = adapter.golden_sequence()
    if g and not a.no_golden:
        w["control"] = dict(sequence=g, golden=True)
    held = a.workdir / "sequences" / a.heldout
    if (held / "metadata.json").exists():
        w["mix"] = dict(sequence=held, golden=False)
    if not w:
        raise SystemExit("no workloads: run `heldout` first and/or provide a golden sequence")
    return w


def cmd_ab(a):
    spec, adapter = _spec_and_adapter(a, need_adapter=True)
    caches = dict(O=adapter.original_cache(), B=Path(a.placed_cache))
    if not (caches["B"] / "8x4").exists() and not any(caches["B"].glob("**/*.expert_placement.json")):
        raise SystemExit(f"{caches['B']} does not look like a placed cache (run `prepare`)")
    res = abtest.run_ab(adapter, a.workdir, caches, _workloads(a, adapter), log=_log)
    print(json.dumps(dict(sides=res["sides"], gate=res["gate"]), indent=1))


def cmd_profile(a):
    spec, adapter = _spec_and_adapter(a, need_adapter=True)
    held = a.workdir / "sequences" / a.heldout
    caches = dict(O=adapter.original_cache())
    if a.placed_cache:
        caches["B"] = Path(a.placed_cache)
    pd = a.workdir / "profile"
    pd.mkdir(exist_ok=True)
    for side, cache in caches.items():
        out = pd / f"profile-{side}"
        if (out / "measurement.json").exists() and json.loads((out / "measurement.json").read_text()).get(
            "status"
        ) == "complete":
            continue
        if out.exists():
            out.rename(out.with_name(f"{out.name}.failed-{int(time.time())}"))
        adapter.check_device_free()
        rc = adapter.run("profile", held, cache, out, pd / f"profile-{side}.log")
        _log(f"profile {side}: rc {rc}")
        if rc == 0:
            prof = json.loads((out / "measurement.json").read_text())["profile"]
            _log(
                json.dumps(
                    dict(side=side, moe_share=prof["moe_share"], routed_expert_share=prof["routed_expert_share"])
                )
            )


def cmd_report(a):
    from .report import write_report

    path = write_report(a.workdir, a.plan)
    _log(f"wrote {path}")


# --- parser ----------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m models.common.moe_placement",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="stage", required=True)

    def stage(name, fn, help_):
        s = sub.add_parser(name, help=help_)
        s.set_defaults(fn=fn)
        if name != "models":
            s.add_argument("--workdir", type=Path, required=True)
        if name not in ("models", "download"):
            s.add_argument("--model", help="registered adapter (see `models`)")
            s.add_argument("--spec", type=Path, help="MoeSpec JSON, for host stages of models without an adapter")
        return s

    stage("models", cmd_models, "list registered adapters")
    s = stage("download", cmd_download, "download the raw corpus")
    s.add_argument("--only", nargs="*")
    s.add_argument("--max-chars", type=int, default=600_000)
    s = stage("pack", cmd_pack, "tokenize the corpus into sequences")
    s.add_argument("--tokenizer", help="HF tokenizer directory (default: the adapter's model)")
    s.add_argument("--chat-format", choices=["hf", "glm"])
    s.add_argument("--control-trace", type=Path, help="golden sequence directory (default: the adapter's)")
    s.add_argument("--cpp-root", type=Path, help="source tree to sample C++ files from (e.g. this repo)")
    s = stage("capture", cmd_capture, "capture routing on the device")
    s.add_argument("--only", nargs="*")
    s = stage("import-routing", cmd_import_routing, "import routing from another source")
    s.add_argument(
        "--name", required=True, help="sequence name (its sequences/<name> must exist for request-level use)"
    )
    s.add_argument(
        "--npz", type=Path, required=True, help="npz with expert_ids [layers, tokens, top_k] (+ layers, expert_weights)"
    )
    s = stage("calibrate", cmd_calibrate, "calibrate and score plans")
    s.add_argument("--out", default="calibration")
    s.add_argument("--plans", default="full", help="'full' or a comma list, e.g. bal128_s1")
    s.add_argument("--constraint", default="aggregate", choices=["aggregate", "frame", "cap10"])
    s.add_argument("--steps", type=int)
    s.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    s.add_argument(
        "--skip-prefix", type=int, default=0, help="drop each request's first N tokens (e.g. a fixed chat preamble)"
    )
    s.add_argument("--layers", help="comma list of layer ids (smoke test)")
    s.add_argument("--eval-from", help="score the plans of an earlier calibration directory instead of training")
    s.add_argument("--extra-preset", action="append", help="name=preset.json: score an existing preset alongside")
    s = stage("heldout", cmd_heldout, "pack the held-out balanced sequence")
    s.add_argument("--name", default="heldout_mix_bal")
    s.add_argument("--seed", type=int, default=777)
    s = stage("preset", cmd_preset, "build and validate a preset from a calibrated plan")
    s.add_argument("--calibration", default="calibration")
    s.add_argument("--plan", default="bal128_s1")
    s.add_argument("--source-main", help="commit the preset is meant for (recorded in the preset)")
    s.add_argument("--out")
    s = stage("screen", cmd_screen, "go/no-go screen before device time")
    s.add_argument("--calibration", default="calibration")
    s.add_argument("--plan", default="bal128_s1")
    s.add_argument("--heldout", default="heldout_mix_bal")
    s.add_argument("--preset", action="append", help="preset(s) to score on the held-out and control sequences")
    s.add_argument("--min-modeled-pct", type=float, default=-3.0)
    s.add_argument("--max-utilization", type=float, default=0.8)
    s = stage("prepare", cmd_prepare, "write a placed weight cache")
    s.add_argument("--preset", required=True)
    s.add_argument(
        "--destination",
        required=True,
        help="new directory, e.g. on node-local scratch (needs ~ the routed-expert size)",
    )
    s = stage("ab", cmd_ab, "on-device A/B and ship gate")
    s.add_argument("--placed-cache", required=True)
    s.add_argument("--heldout", default="heldout_mix_bal")
    s.add_argument("--no-golden", action="store_true")
    s = stage("profile", cmd_profile, "MoE stage shares on the held-out sequence")
    s.add_argument("--placed-cache")
    s.add_argument("--heldout", default="heldout_mix_bal")
    s = stage("report", cmd_report, "write report.md")
    s.add_argument("--plan", default="bal128_s1")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

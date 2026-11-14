#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Sampling-path benchmark: one process per (arm, batch, setting), model loaded once,
3 warm-up + 10 measured runs, each run = prefill (not timed) + N decode steps; per run the decode time per
output token is the mean wall time of steps 1..N-1 (step 0 of each run is excluded in every arm: it carries
the first-call input reload / capture). Reported: median (min, max) over the measured runs.

    python -u tools/l2cpu/examples/qwen3/bench.py --arm host --batch 1 --setting T0.7k50p0.9
    python tools/l2cpu/examples/qwen3/bench.py --table   # markdown table of every JSON in $L2CPU_QWEN3_OUT/bench

Arms
  host        decode_harness Plan A (row-major logits, whole-tensor readback, x280s C library on the host);
              --sampler hostref | hostref-mt:8 (thread pool over users)
  tensix      stock tt-transformers on-device sampling via Generator.decode_forward(sampling_params=...),
              driven like simple_text_demo.py (traced decode + sampling trace, token fed back on device,
              blocking per-step token read); --no-token-read: read_from_device=False, one sync at the end
  x280-planA  decode_harness Plan A with a Sampler instance from --sampler-factory module:callable
              (callable(args) -> dh_samplers.Sampler)
  x280-planB  --planb module:callable; callable(harness, prompts, params, seeds, n_tokens, run_idx) -> dict
              {"steps": [{"wall": s, ...optional breakdown...}, ...], "tokens": [[...] per user]}
              (harness = a loaded decode_harness.DecodeHarness; the callable owns its trace/arena)
Settings: T0.7k50p0.9, T0.7k32p0.9 (the stock Tensix sampler caps top-k at 32), greedy.
"""

import argparse
import glob
import importlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deps  # noqa: E402,F401  (bindings to the other l2cpu components, see deps.py)

OUTDIR = deps.out_dir("bench")
SETTINGS = {
    "T0.7k50p0.9": (0.7, 50, 0.9),
    "T0.7k32p0.9": (0.7, 32, 0.9),
    "T1.0k0p1.0": (1.0, 0, 1.0),
    "greedy": (0.0, 0, 1.0),
}


def log(*a):
    print("[bench %s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def load_callable(spec):
    mod, attr = spec.split(":")
    if mod.endswith(".py"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(mod)))
        mod = os.path.splitext(os.path.basename(mod))[0]
    return getattr(importlib.import_module(mod), attr)


def run_stats(steps, key="wall"):
    v = [s[key] for s in steps[1:] if s.get(key) is not None]
    return float(np.mean(v)) if v else None


def breakdown(steps):
    out = {}
    for k in sorted({k for s in steps for k, v in s.items() if isinstance(v, float)}):
        v = [s[k] for s in steps[1:] if isinstance(s.get(k), float)]
        if v:
            out[k + "_ms_median"] = 1e3 * float(np.median(v))
    return out


class HArgs(SimpleNamespace):
    """decode_harness args; options added to the harness later (e.g. --x280) default to None/off."""

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return None


def harness_args(a):
    return HArgs(
        batch=a.batch,
        precision="performance",
        max_seq_len=1024,
        trace_region_size=0,
        prefill_mode="batched",
        convert="auto",
        one_trace=True,
        input_tokens=a.input_tokens,
        max_new_tokens=a.output_tokens,
        clear_kv=1,
        hash_rows=0,
        x280=a.arm.startswith("x280"),  # x280 arms: harness boots arena + firmware before capture
        x280_tiles=a.tiles,
    )


def prompts_for_run(all_prompts, B, r):
    """Same prompts in every arm: batch 1 rotates over the first 5 demo prompts, batch 32 uses all 32."""
    return [all_prompts[r % 5]] if B == 1 else all_prompts[:32]


# ---------------------------------------------------------------- arms built on decode_harness
def arm_harness(a, sampler_or_planb, planb=False):
    from decode_harness import DecodeHarness, load_prompts

    h = DecodeHarness(harness_args(a))
    T, k, p = SETTINGS[a.setting]
    B = a.batch
    params = [{"temperature": T, "top_k": k, "top_p": p}] * B
    seeds = [a.seed] * B
    allp = load_prompts(os.environ["TT_METAL_HOME"], 32)
    runs = []
    for r in range(a.warmup + a.runs):
        prompts = prompts_for_run(allp, B, r)
        t0 = time.time()
        n_stats = len(getattr(sampler_or_planb, "stats", None) or [])
        if planb:
            g = sampler_or_planb(h, prompts, params, seeds, a.output_tokens, r)
        else:
            g = h.run_group(r, prompts, sampler_or_planb, params, seeds, None)
        toks = [u["tokens"] if isinstance(u, dict) else u for u in (g.get("users") or g.get("tokens") or [])]
        d = finish_run(r, a, g["steps"], toks, time.time() - t0, h.tokenizer)
        if r == 0 and toks:
            d["tokens"] = toks  # full token lists of run 0 (token sanity across arms)
        if not planb and hasattr(sampler_or_planb, "stats"):  # x280 Plan A: host push+sync and firmware records
            st = sampler_or_planb.stats[n_stats:]
            d["x280"] = x280_split([t for _, _, t in st[1:]])
            d["x280"]["push_sync_us"] = 1e6 * float(np.median([p_ for p_, _, _ in st[1:]]))
            d["x280"]["host_round_trip_us"] = 1e6 * float(np.median([q for _, q, _ in st[1:]]))
            d["x280"]["wake_and_host_us"] = d["x280"]["host_round_trip_us"] - d["x280"]["total_us"]
        if planb and g.get("timing"):
            d["x280"] = x280_split(g["timing"][1:])  # several tiles: per step the slowest tile's record
            if g.get("timing_tiles"):
                d["x280_tiles"] = [x280_split(t[1:]) for t in g["timing_tiles"]]
                log("   x280 split per tile (us):", [{k_: round(v, 1) for k_, v in x.items()} for x in d["x280_tiles"]])
            if g.get("diag"):
                cnt, gsum, gmax, esum = g["diag"][:4]
                if cnt:
                    d["x280"]["gap_notify_wait_us"] = gsum / cnt / TENSIX_MHZ
                    d["x280"]["gap_notify_wait_max_us"] = gmax / TENSIX_MHZ
                    d["x280"]["wake_derived_us"] = d["x280"]["gap_notify_wait_us"] - d["x280"]["total_us"]
        if "x280" in d:
            log("   x280 split (us):", {k_: round(v, 1) for k_, v in d["x280"].items()})
        runs.append(d)
    meta = {"logits_rm_address": hex(h.logits.buffer_address())} if getattr(h, "logits", None) is not None else {}
    h.ttnn.close_mesh_device(h.mesh)
    return runs, meta


CORE_MHZ = 1750.0  # x280 cycle counter (l2cpu_sampler.CORE_MHZ)
TENSIX_MHZ = 1350.0  # device timestamps of the notify/wait diag (measured AI clock under load on P300)


def x280_split(tms):
    """Firmware timing records (hart 0, cycles) -> per-step medians in us. wake-up latency is
    not in the record (it starts at the wake); see wake_* fields derived from host / device timestamps."""
    if not tms:
        return {}
    med = lambda f: float(np.median([f(t) for t in tms])) / CORE_MHZ  # noqa: E731
    return {
        "read_us": med(lambda t: t["cyc_read"]),
        "sample_us": med(lambda t: t["cyc_sample"]),
        "write_us": med(lambda t: t["cyc_write"]),
        "wait_workers_us": med(lambda t: t["cyc_wait"]),
        "publish_other_us": med(
            lambda t: t["cyc_total"] - t["cyc_read"] - t["cyc_sample"] - t["cyc_write"] - t["cyc_wait"]
        ),
        "total_us": med(lambda t: t["cyc_total"]),
    }


def planb_diag(harness, prompts, params, seeds, n_tokens, run_idx):
    """plan_b.planb_bench with the notify/wait device timestamps on (diag kernels in the captured trace)."""
    import plan_b

    if "pb" not in plan_b._BENCH:
        from l2cpu_sampler import session

        plan_b._BENCH["pb"] = plan_b.PlanB(harness.mesh, session=session(), uncached=len(params) > 1)
    return plan_b.run_group_planb(harness, plan_b._BENCH["pb"], prompts, params, seeds, n_tokens, diag=True)


def fw_image():
    from l2cpu.sampling.fw import DEFAULT_IMAGE  # sampling firmware image (L2S_FW_IMAGE overrides)

    path = os.environ.get("L2S_FW_IMAGE") or DEFAULT_IMAGE
    import hashlib

    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    label = os.environ.get("L2S_FW_LABEL") or os.path.basename(path)
    return {"path": path, "sha256": sha, "label": label}


def command_line(a, home, image):
    cmd = "$PY -u " + " ".join([os.path.relpath(os.path.abspath(sys.argv[0]), home)] + sys.argv[1:])
    if image is None:
        return cmd
    return "tt-smi -r 0,1,2,3 && L2S_FW_IMAGE=%s %s" % (os.path.basename(image["path"]), cmd)


def finish_run(r, a, steps, tokens, wall, tok):
    ms = run_stats(steps)
    d = {
        "run": r,
        "warmup": r < a.warmup,
        "ms_per_token": 1e3 * ms if ms else None,
        "tokens_per_s_per_user": 1.0 / ms if ms else None,
        "steps": len(steps),
        "run_wall_s": wall,
        "breakdown": breakdown(steps),
    }
    if tokens:
        d["user0_tokens_head"] = list(tokens[0][:16])
        d["user0_text_head"] = tok.decode(tokens[0][:48])
    log(
        "run %d%s: %.2f ms/token" % (r, " (warm-up)" if d["warmup"] else "", d["ms_per_token"] or -1),
        {k: round(v, 3) for k, v in d["breakdown"].items()},
    )
    return d


# ---------------------------------------------------------------- tensix arm (stock on-device sampling)
def arm_tensix(a):
    import torch
    import ttnn
    from decode_harness import DecodeHarness, encode, load_prompts
    from models.tt_transformers.tt.generator import SamplingParams

    h = DecodeHarness(harness_args(a))
    gen, B = h.generator, a.batch
    T, k, p = SETTINGS[a.setting]
    if T == 0:  # the demo's greedy dict -> force_argmax
        T, k, p = 0.0, 32, 0.08
    lst = (lambda x: [x] * B) if B > 1 else (lambda x: x)  # batch > 1 needs per-user lists (tt_penalties)
    sp = SamplingParams(
        temperature=lst(T),
        top_k=lst(k),
        top_p=lst(p),
        seed=None,
        frequency_penalty=lst(0.0),
        presence_penalty=lst(0.0),
        repetition_penalty=lst(1.0),
        enable_log_probs=lst(False),
    )
    allp = load_prompts(os.environ["TT_METAL_HOME"], 32)
    # compile the in-place KV reset before any trace exists (the demo's repeat_batches rule)
    for layer in h.model.layers:
        kc, vc = layer.attention.layer_past
        ttnn.mul(kc, 0, output_tensor=kc)
        ttnn.mul(vc, 0, output_tensor=vc)
    runs = []
    for r in range(a.warmup + a.runs):
        prompts = prompts_for_run(allp, B, r)
        ids = [encode(h.model_args, h.tokenizer, x, a.input_tokens)[0] for x in prompts]
        lens = [len(x) for x in ids]
        L = max(lens)
        toks = torch.zeros((B, L), dtype=torch.int32)
        for i, x in enumerate(ids):
            toks[i, : len(x)] = torch.tensor(x, dtype=torch.int32)
        for layer in h.model.layers:
            kc, vc = layer.attention.layer_past
            ttnn.mul(kc, 0, output_tensor=kc)
            ttnn.mul(vc, 0, output_tensor=vc)
        t_run = time.time()
        out = gen.prefill_forward_text(
            toks,
            page_table=h.page_table,
            kv_cache=[h.kv_cache],
            prompt_lens=lens,
            sampling_params=sp,
            warmup_prefill=True,
            enable_trace=True,
        )
        first = out[0] if isinstance(out, tuple) else torch.argmax(out, dim=-1)
        out_tok = first.reshape(B, 1) if first.dim() == 1 else first
        gen_tokens = [[int(out_tok[b].item())] for b in range(B)]
        pos = torch.tensor(lens)
        steps = []
        t_loop = time.perf_counter()
        for it in range(a.output_tokens):
            t0 = time.perf_counter()
            res = gen.decode_forward(
                out_tok,
                pos,
                enable_trace=True,
                page_table=h.page_table,
                kv_cache=[h.kv_cache],
                sampling_params=sp,
                prompt_tokens=toks,
                output_tokens=out_tok,
                reload_inputs=it == 0,
                reload_page_table=False,
                reload_sampling_params=True,
                reset_sampling_state=it == 0,
                read_from_device=not a.no_token_read,
            )
            if not a.no_token_read:
                tk, _ = res
                out_tok = tk.unsqueeze(1)
                for b in range(B):
                    gen_tokens[b].append(int(out_tok[b].item()))
            if it == 0 and a.no_token_read:
                ttnn.synchronize_device(h.mesh)
                t_loop = time.perf_counter()
            pos = pos + 1
            steps.append({"wall": time.perf_counter() - t0})
        if a.no_token_read:  # one sync at the end; per-step wall = (loop time after step 0) / (N - 1)
            ttnn.synchronize_device(h.mesh)
            per = (time.perf_counter() - t_loop) / (a.output_tokens - 1)
            steps = [steps[0]] + [{"wall": per} for _ in steps[1:]]
            gen_tokens = None
        runs.append(finish_run(r, a, steps, gen_tokens, time.time() - t_run, h.tokenizer))
    h.ttnn.close_mesh_device(h.mesh)
    return runs, {
        "sampling_params": {"temperature": T, "top_k": k, "top_p": p, "seed": None, "penalties": "none"},
        "per_step_host_work": "execute_trace(decode) + sampling reset_params host writes (not when "
        "force_argmax) + execute_trace(sampling) + blocking cpu() of the 32 uint32 tokens"
        + (" [skipped: --no-token-read]" if a.no_token_read else ""),
    }


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["host", "tensix", "x280-planA", "x280-planB"])
    ap.add_argument("--batch", type=int, default=1, choices=[1, 32])
    ap.add_argument("--setting", default="T0.7k50p0.9", choices=list(SETTINGS))
    ap.add_argument("--sampler", default="hostref", help="host arm: hostref | hostref-mt:N")
    ap.add_argument("--sampler-factory", help="x280-planA: module:callable(args) -> Sampler")
    ap.add_argument("--planb", help="x280-planB: module:callable (see docstring)")
    ap.add_argument("--no-token-read", action="store_true", help="tensix arm: no per-step token read")
    ap.add_argument("--planb-diag", action="store_true", help="x280-planB: capture notify/wait with device timestamps")
    ap.add_argument(
        "--tiles",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help="x280 arms: L2CPU tiles started; x280-planB at batch > 1 splits the batch over them",
    )
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--input-tokens", type=int, default=128)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=1234)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out")
    ap.add_argument("--report", action="store_true", help="print the consolidated report tables and exit")
    ap.add_argument("--table", action="store_true", help="print the markdown table of all bench JSONs and exit")
    a = ap.parse_args()
    if a.report:
        print(report())
        return 0
    if a.table:
        print(table())
        print()
        print(sanity())
        return 0
    if not a.arm:
        ap.error("--arm is required")

    home = os.environ["TT_METAL_HOME"]
    import ttnn

    assert os.path.realpath(ttnn.__file__).startswith(os.path.realpath(home)), ttnn.__file__
    os.chdir(home)
    os.environ["HF_MODEL"] = a.model
    os.environ["TT_CACHE_PATH"] = deps.weights_cache(a.model)
    import torch

    torch.manual_seed(213919)
    commit = subprocess.run(["git", "-C", home, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", home, "status", "--porcelain", "tools/l2cpu"], capture_output=True, text=True
        ).stdout.strip()
    )
    variant = (
        a.sampler
        if a.arm == "host"
        else ("no-token-read" if a.no_token_read else "token-read")
        if a.arm == "tensix"
        else a.arm
    )
    image = fw_image() if a.arm.startswith("x280") else None
    if image:
        variant = "%s img=%s" % (variant, image["label"]) + (" diag" if a.planb_diag else "")
        if a.tiles > 1:
            variant = variant.replace(" img=", " tiles=%d img=" % a.tiles)
    name = "%s_b%d_%s_%s%s" % (
        a.arm,
        a.batch,
        a.setting,
        variant.replace(":", "").replace(" ", "_").replace("=", "-"),
        ("_" + a.tag) if a.tag else "",
    )
    out = a.out or os.path.join(OUTDIR, name + ".json")
    t0 = time.time()
    if a.arm == "host":
        from dh_samplers import make_sampler

        runs, meta = arm_harness(a, make_sampler(a.sampler, "full"))
    elif a.arm == "x280-planA":
        from dh_samplers import make_sampler

        runs, meta = arm_harness(
            a, load_callable(a.sampler_factory)(a) if a.sampler_factory else make_sampler("x280", "full")
        )
    elif a.arm == "x280-planB":
        runs, meta = arm_harness(a, planb_diag if a.planb_diag else load_callable(a.planb), planb=True)
    else:
        runs, meta = arm_tensix(a)
    meas = [r for r in runs if not r["warmup"]]
    ms = [r["ms_per_token"] for r in meas]
    res = {
        "name": name,
        "arm": a.arm,
        "variant": variant,
        "batch": a.batch,
        "setting": a.setting,
        "setting_values": SETTINGS[a.setting],
        "commit": commit,
        "tree_dirty_x280": dirty,
        "command": command_line(a, home, image),
        "input_tokens": a.input_tokens,
        "output_tokens": a.output_tokens,
        "warmup": a.warmup,
        "runs_measured": a.runs,
        "ms_per_token": {"median": float(np.median(ms)), "min": float(np.min(ms)), "max": float(np.max(ms))},
        "tokens_per_s_per_user": {
            "median": float(np.median([1e3 / x for x in ms])),
            "min": float(np.min([1e3 / x for x in ms])),
            "max": float(np.max([1e3 / x for x in ms])),
        },
        "breakdown_median_of_runs": {
            k: float(np.median([r["breakdown"][k] for r in meas if k in r["breakdown"]])) for k in meas[0]["breakdown"]
        },
        "meta": meta,
        "fw_image": image,
        "wall_s": time.time() - t0,
        "runs": runs,
    }
    xs = [r_["x280"] for r_ in meas if r_.get("x280")]
    if xs:
        res["x280_split_median_of_runs_us"] = {k_: float(np.median([x[k_] for x in xs if k_ in x])) for k_ in xs[0]}
    xt = [r_["x280_tiles"] for r_ in meas if r_.get("x280_tiles")]
    if xt:
        res["x280_split_per_tile_median_of_runs_us"] = [
            {k_: float(np.median([x[t][k_] for x in xt])) for k_ in xt[0][t]} for t in range(len(xt[0]))
        ]
    res["tiles"] = a.tiles
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    log(
        "%s: %.2f ms/token median (min %.2f, max %.2f), %.2f tok/s/user -> %s"
        % (
            name,
            res["ms_per_token"]["median"],
            res["ms_per_token"]["min"],
            res["ms_per_token"]["max"],
            res["tokens_per_s_per_user"]["median"],
            out,
        )
    )
    return 0


def _rows():
    out = []
    for f in sorted(glob.glob(os.path.join(OUTDIR, "*.json"))):
        r = json.load(open(f))
        if r["name"].endswith("_ref"):
            continue  # token-reference runs (warmup 0, runs 1), not benchmark rows
        out.append(r)
    order = {"host": 0, "tensix": 1, "x280-planA": 2, "x280-planB": 3}
    return sorted(
        out, key=lambda r: (r["batch"], r["setting"] != "greedy", r["setting"], order.get(r["arm"], 9), r["variant"])
    )


def image_label(r):
    if not r["arm"].startswith("x280"):
        return "-"
    im = r.get("fw_image")
    return "%s %s" % (im["label"], im["sha256"][:8]) if im else "unknown image"


def table():
    head = (
        "| B | setting | arm | variant | fw image | ms/token median (min-max) | tok/s/user | host per-step medians (ms) "
        "| x280 per step (us): read / sample / write / wait+publish / total | push+sync / gap (us) | command | commit |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for r in _rows():
        b = r["breakdown_median_of_runs"]
        parts = [
            "%s %.2f" % (lab, b[k + "_ms_median"])
            for k, lab in (("fwd_cvt", "dev"), ("read", "read"), ("sample", "sample"), ("write", "write"))
            if k + "_ms_median" in b
        ]
        x = r.get("x280_split_median_of_runs_us") or {}
        xs = (
            "%.0f / %.0f / %.0f / %.0f / %.0f"
            % (x["read_us"], x["sample_us"], x["write_us"], x["wait_workers_us"] + x["publish_other_us"], x["total_us"])
            if x
            else "-"
        )
        ex = []
        if "push_sync_us" in x:
            ex.append("push+sync %.0f, host rt %.0f" % (x["push_sync_us"], x["host_round_trip_us"]))
        if "gap_notify_wait_us" in x:
            ex.append("gap %.0f (wake %.1f)" % (x["gap_notify_wait_us"], x["wake_derived_us"]))
        variant = r["variant"].split(" img=")[0] + (" diag" if r["variant"].endswith(" diag") else "")
        rows.append(
            "| %d | %s | %s | %s | %s | %.2f (%.2f-%.2f) | %.1f | %s | %s | %s | `%s` | %s |"
            % (
                r["batch"],
                r["setting"],
                r["arm"],
                variant,
                image_label(r),
                r["ms_per_token"]["median"],
                r["ms_per_token"]["min"],
                r["ms_per_token"]["max"],
                r["tokens_per_s_per_user"]["median"],
                ", ".join(parts) or "-",
                xs,
                "; ".join(ex) or "-",
                r["command"],
                r["commit"][:11] + ("+dirty" if r.get("tree_dirty_x280") else ""),
            )
        )
    return head + "\n" + "\n".join(rows)


def sanity():
    """Run-0 tokens of every x280 row vs the host reference run (same batch/setting/prompts/seeds)."""
    refs = {}
    for f in glob.glob(os.path.join(OUTDIR, "*.json")):
        r = json.load(open(f))
        if r["arm"] == "host" and r["runs"] and "tokens" in r["runs"][0]:
            refs.setdefault((r["batch"], r["setting"]), r["runs"][0]["tokens"])
    lines = []
    for r in _rows():
        if not r["arm"].startswith("x280"):
            continue
        ref = refs.get((r["batch"], r["setting"]))
        t = r["runs"][0].get("tokens") if r["runs"] else None
        if ref is None or t is None:
            lines.append("- %s %s: no %s" % (r["name"], image_label(r), "reference" if ref is None else "run-0 tokens"))
            continue
        nu = sum(a == b for a, b in zip(t, ref))
        nt = sum(x == y for a, b in zip(t, ref) for x, y in zip(a, b))
        tot = sum(len(a) for a in ref)
        lines.append(
            "- %s (%s): run-0 tokens == host run 0: %d/%d users, %d/%d tokens"
            % (r["name"], image_label(r), nu, len(ref), nt, tot)
        )
    return "\n".join(lines)


def _all():
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUTDIR, "*.json")))]


def _pick(rs, arm, B, setting, pred=lambda r: True):
    c = [
        r
        for r in rs
        if r["arm"] == arm
        and r["batch"] == B
        and r["setting"] == setting
        and pred(r)
        and not r["name"].endswith("_ref")
    ]
    return sorted(c, key=lambda r: r["name"])[-1] if c else None


def _img(r, label):
    return (r.get("fw_image") or {}).get("label") == label


def _cell(r):
    if r is None:
        return "not run", "-"
    m, t = r["ms_per_token"], r["tokens_per_s_per_user"]
    return "%.2f (%.2f-%.2f)" % (m["median"], m["min"], m["max"]), "%.1f" % t["median"]


def _cmd(r):
    return "`%s`" % r["command"] if r else "-"


def _commit(r):
    return (r["commit"][:11] + ("+dirty" if r.get("tree_dirty_x280") else "")) if r else "-"


def report():
    rs = _all()
    streamed = lambda r: r["name"].endswith("_streamed")  # noqa: E731
    plain_b = lambda r: "streamed" not in r["name"]  # noqa: E731
    out = [
        "## Headline: Qwen3-8B, 128 in / 256 out, median of 10 runs (min-max)",
        "",
        "| B | setting | arm | ms/token | tok/s/user | commit | command |",
        "|---|---|---|---|---|---|---|",
    ]
    for B in (1, 32):
        for S in ("T0.7k50p0.9", "greedy"):
            host = _pick(rs, "host", B, S, lambda r: r["variant"] == ("hostref" if B == 1 else "hostref-mt:8"))
            ten = _pick(rs, "tensix", B, "T0.7k32p0.9" if S != "greedy" else S, lambda r: r["variant"] == "token-read")
            pa = _pick(rs, "x280-planA", B, S, lambda r: True)
            pb = _pick(rs, "x280-planB", B, S, lambda r: plain_b(r))
            pbs = _pick(rs, "x280-planB", B, S, lambda r: streamed(r))
            for lab, r in (
                ("host (%s)" % ("1 thread" if B == 1 else "8 threads"), host),
                ("tensix (stock%s)" % (", k32" if S != "greedy" else ""), ten),
                ("x280 Plan A", pa),
                ("x280 Plan B", pb),
                ("x280 Plan B streamed", pbs),
            ):
                ms, tk = _cell(r)
                out.append("| %d | %s | %s | %s | %s | %s | %s |" % (B, S, lab, ms, tk, _commit(r), _cmd(r)))
    out += [
        "",
        "## x280 per-step split (hart 0 firmware timing records, medians, us) and device gaps",
        "",
        "| arm | image | B | setting | logits read | sampling | write-back | wait workers + publish | x280 total | "
        "push+sync (host, Plan A) | host round trip (Plan A) | notify end -> wait release (Plan B, mean / max) | wake-up |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(
        [r for r in rs if r["arm"].startswith("x280") and r.get("x280_split_median_of_runs_us")],
        key=lambda r: (r["arm"], r["batch"], r["setting"], r["name"]),
    ):
        x = r["x280_split_median_of_runs_us"]
        var = (r["fw_image"] or {}).get("label", "?") + (" streamed" if streamed(r) else "")
        gap = (
            "%.0f / %.0f" % (x["gap_notify_wait_us"], x["gap_notify_wait_max_us"]) if "gap_notify_wait_us" in x else "-"
        )
        wake = (
            ("%.1f (gap - total)" % x["wake_derived_us"])
            if "wake_derived_us" in x
            else ("<= %.0f (round trip - total, incl. host)" % x["wake_and_host_us"])
            if "wake_and_host_us" in x
            else "-"
        )
        out.append(
            "| %s | %s | %d | %s | %.1f | %.0f | %.1f | %.0f | %.0f | %s | %s | %s | %s |"
            % (
                r["arm"],
                var,
                r["batch"],
                r["setting"],
                x["read_us"],
                x["sample_us"],
                x["write_us"],
                x["wait_workers_us"] + x["publish_other_us"],
                x["total_us"],
                "%.0f" % x["push_sync_us"] if "push_sync_us" in x else "-",
                "%.0f" % x["host_round_trip_us"] if "host_round_trip_us" in x else "-",
                gap,
                wake,
            )
        )
    out += [
        "",
        "## Token sanity (run 0 of every x280 row vs the host reference run 0, same prompts/seeds)",
        "",
        sanity(),
    ]
    out += ["", "## All rows", "", table()]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())

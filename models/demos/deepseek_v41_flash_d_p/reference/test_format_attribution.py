# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.21 (owner 2026-11-01): which of the device's format / precision choices make its error larger than DeepSeek's
reference? CPU only, no device.

The CPU reference at numerics official, dtype bf16 (DeepSeek's code at its shipped precision: the margin source) with
the device-format sites of ds41_ref.FORMAT_SITES, each 'official' (the reference's own quantizer / dtype) or 'device'
(what tt/settings.py ships). Layers 0-9 chained as the ladder runs them (reference/drift.py:replay, the ladder's feed):
rung ``last`` = s56320's last chunk, positions 51200-56319, the golden's state prefix at 51200 loaded. Why this rung:
one 5120-token chunk per run (s10240 needs two), at a deep position (51200 keys of compressed history: the index /
compressed-KV formats matter there), and the device's own numbers on it are FIX.20's ladder (err_layer_L*, err_state_*).

Runs, each against the native golden (err_layer_L00..L09 = rel L2 of the block output, err_state_* = the ladder's
state metrics: float rel L2 over the positions the device computed, index lists 1 - overlap):
    device         every site 'device' (a): the emulation. VALIDATED against FIX.20's device errors: per metric
                   err_emu / err_device, median and worst (emu_match_*; owner: +-25 % per metric, median +-15 %, a check
                   of the emulation, never of the model, F138)
    loo_<site>     all 'device' but <site> 'official' (b, leave-one-out)
    add_<site>     all 'official' but <site> 'device' (c, add-one)
    official       every site 'official' (d) = the reference's own drift (equals drift-last.json: a check of the feed)
Shares, per layer and at L09: gap_closed_<site> = (err_device - err_loo) / (err_device - err_official) (the part of the
gap the site's device format causes, given the others), addone_<site> = (err_add - err_official) / (err_device -
err_official) (the part it causes alone); *_median = over L00..L09.

Each run is cached as JSON (generated/fix21_format_attribution/<golden hash>/<EMU_VERSION>/<run>.json; EMU_VERSION
names the emulation's semantics: bump it with any change to ds41_ref's format sites). FIX21_ONLY=<run,run,...>
computes only those runs and records nothing (to fill the cache from several jobs); the gate run assembles the rest.
CPU: half the physical cores (BRINGUP_CPU_THREADS overrides), nice 19.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.drift import narrow, replay
from models.demos.common.bringup.reference.golden import Golden, load_spec, reference_at
from models.demos.common.bringup.testing import e2e_limits as E
from models.demos.common.bringup.testing.ladder import index_overlap, rel
from models.demos.deepseek_v41_flash_d_p.reference import ds41_ref as R

pytestmark = pytest.mark.timeout(12 * 3600)  # a run not yet cached is minutes of CPU (pytest.ini: 300 s)

EMU_VERSION = "v1"
RUNG = "last"
LAYERS = list(range(10))
SITES = list(R.FORMAT_SITES)
DEVICE_TASK = "FIX.20"  # its ladder rung last = the device's errors on this chunk (shipped settings)
SECOND_TASK = "P.26"  # the same rung under P.26 (recorded beside, not the check)
OUT = Path(__file__).resolve().parents[4] / "generated" / "fix21_format_attribution"
RESULTS = Path(__file__).resolve().parents[1] / "bringup" / "ledgers" / "galaxy_8x4-full" / "results"


def run_formats(name: str) -> dict:
    if name == "device":
        return {s: "device" for s in SITES}
    if name == "official":
        return {}
    kind, site = name.split("_", 1)
    assert site in SITES, name
    if kind == "loo":
        return {s: "device" for s in SITES if s != site}
    assert kind == "add", name
    return {site: "device"}


def run_names() -> list[str]:
    return ["device", "official"] + [f"loo_{s}" for s in SITES] + [f"add_{s}" for s in SITES]


def _threads() -> int:
    if os.environ.get("BRINGUP_CPU_THREADS"):
        return int(os.environ["BRINGUP_CPU_THREADS"])
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or os.cpu_count()
    except ImportError:
        cores = os.cpu_count()
    return max(1, cores // 2)


def _errors(ref, rep: dict, g: Golden, drift: dict) -> dict:
    """The ladder's metrics of one run: err_layer_L<i>, err_state_<name>_L<i> (index lists: 1 - overlap)."""
    out, last, seq = {}, g.n_chunks - 1, g.seq
    for i in LAYERS:
        out[f"err_layer_L{i:02d}"] = rel(rep["trail"][i], g.layer(last, i)["out"].float())
        got = ref.state_tensors(rep["state"], i, seq)
        for name, want in g.state(i).items():
            key = f"state_{name}_L{i:02d}"
            region = (drift.get(key) or {}).get("region")
            o = got[name].reshape(want.shape) if got[name].numel() == want.numel() else got[name]
            o, w = narrow(o, region), narrow(want, region)
            if w.is_floating_point():
                out[f"err_{key}"] = rel(o, w)
            else:
                out[f"err_{key}"] = 1.0 - index_overlap(o, w)
    return out


class Runs:
    """The attribution runs on one reference (weights loaded once), each cached."""

    def __init__(self):
        self.s = load_spec()
        self.g = Golden.for_rung(self.s, RUNG)
        self.dir = OUT / str(self.g.pinned_hash())[:16] / EMU_VERSION
        self.dir.mkdir(parents=True, exist_ok=True)
        self.drift = E.load(self.g.dir, RUNG)
        self._ref = None

    def ref40(self):
        """All the model's layers (the token runs); the 10-layer one is dropped first (host memory)."""
        self._ref = None
        if getattr(self, "_ref40", None) is None:
            self._ref40 = reference_at(self.s, "official", layers=None, dtype=torch.bfloat16)
        return self._ref40

    def ref(self):
        if self._ref is None:
            t0 = time.time()
            self._ref = reference_at(self.s, "official", layers=LAYERS, dtype=torch.bfloat16)
            print(f"[fix21] reference layers {LAYERS[0]}-{LAYERS[-1]} loaded in {time.time() - t0:.0f} s", flush=True)
        return self._ref

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def get(self, name: str, compute: bool = True) -> dict | None:
        p = self.path(name)
        if p.exists():
            return json.loads(p.read_text())
        if not compute:
            return None
        fm = run_formats(name)
        ref = self.ref().with_formats(fm)
        t0 = time.time()
        rep = replay(self.s, ref, self.g, LAYERS, self.g.n_chunks - 1)
        err = _errors(ref, rep, self.g, self.drift)
        res = {
            "run": name,
            "formats": sorted(fm),
            "seconds": round(time.time() - t0, 1),
            "threads": torch.get_num_threads(),
            "errors": err,
            "made": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tmp = p.with_suffix(f".{os.getpid()}.tmp")  # several fill jobs may share the cache
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(p)
        print(
            f"[fix21] run {name}: {res['seconds']:.0f} s, L09 {err['err_layer_L09']:.5f} L00 {err['err_layer_L00']:.5f}",
            flush=True,
        )
        return res


def _device_metrics(task: str) -> dict:
    p = RESULTS / f"{task}.json"
    m = json.loads(p.read_text())["metrics"]
    return {k: v["value"] for k, v in m.items()}


def compared_keys(err: dict, dev: dict) -> list[str]:
    """The metrics the emulation is checked on: every err_layer / err_state of layers 0-9 the device recorded, except
    the exact ones (0 on the device: compress_partial, engram_tail)."""
    return sorted(k for k in err if k in dev and dev[k] > 0)


def share(e_dev: float, e_x: float, e_off: float, loo: bool) -> float:
    gap = e_dev - e_off
    if abs(gap) < 1e-12:
        return float("nan")
    return (e_dev - e_x) / gap if loo else (e_x - e_off) / gap


@pytest.fixture(scope="module")
def runs():
    torch.set_num_threads(_threads())
    try:
        os.nice(19 - os.nice(0))
    except OSError:
        pass
    return Runs()


def test_format_attribution(runs):
    only = [n for n in os.environ.get("FIX21_ONLY", "").split(",") if n]
    if only:  # fill the cache from a job; the gate run records
        for n in only:
            assert n in run_names(), f"{n}: not a run ({run_names()})"
            runs.get(n)
        pytest.skip(f"FIX21_ONLY: computed {only}; nothing recorded")

    # (2) the emulation first: all-device vs the device's own errors on the same rung
    dev = _device_metrics(DEVICE_TASK)
    assert dev.get("rung_start") == (runs.g.n_chunks - 1) * runs.g.chunk, "FIX.20's ladder ran another rung"
    emu = runs.get("device")["errors"]
    off = runs.get("official")["errors"]
    keys = compared_keys(emu, dev)
    ratios = {k: emu[k] / dev[k] for k in keys}
    lay = [ratios[f"err_layer_L{i:02d}"] for i in LAYERS]
    med = statistics.median(ratios.values())
    worst_k = max(ratios, key=lambda k: abs(ratios[k] - 1.0))
    print(f"\n[fix21] emulation (all device) vs {DEVICE_TASK} device, rung {RUNG}, layers 0-9:")
    print(f"{'metric':34s} {'device':>9s} {'emu':>9s} {'ratio':>7s} {'official':>9s}")
    for k in keys:
        print(f"{k:34s} {dev[k]:9.5f} {emu[k]:9.5f} {ratios[k]:7.3f} {off.get(k, float('nan')):9.5f}")
    ok = all(abs(r - 1) <= 0.25 for r in ratios.values()) and abs(med - 1) <= 0.15
    print(
        f"[fix21] emu_match_median {med:.3f} (layers {statistics.median(lay):.3f}), worst {worst_k} "
        f"{ratios[worst_k]:.3f}: {'MATCH' if ok else 'NO MATCH'} (+-25 % per metric, median +-15 %)"
    )
    metrics.record("emu_match_median", med)
    metrics.record("emu_match_median_layers", statistics.median(lay))
    metrics.record("emu_match_worst", ratios[worst_k])
    metrics.record("emu_match_worst_metric", worst_k)
    metrics.record("emu_match_ok", int(ok))
    metrics.record("emu_metrics", len(keys))
    for k in keys:
        metrics.record(f"emu_ratio_{k}", ratios[k])
    second = _device_metrics(SECOND_TASK)
    k2 = compared_keys(emu, second)
    if k2:
        metrics.record("emu_match_median_p26", statistics.median(emu[k] / second[k] for k in k2))
    drift_l = [runs.drift.get(f"layer_L{i:02d}", {}).get("rel") for i in LAYERS]
    if all(d is not None for d in drift_l):  # the official run is the drift file's reference (a check of the feed)
        metrics.record(
            "official_vs_drift_median", statistics.median(off[f"err_layer_L{i:02d}"] / drift_l[i] for i in LAYERS)
        )

    # (3) attribution: leave-one-out and add-one for every site
    metrics.record("sites", len(SITES))
    res = {"device": runs.get("device"), "official": runs.get("official")}
    for s in SITES:
        res[f"loo_{s}"] = runs.get(f"loo_{s}")
        res[f"add_{s}"] = runs.get(f"add_{s}")
    for name, r in res.items():
        metrics.record(f"run_seconds_{name}", r["seconds"])
        for k, v in r["errors"].items():
            if k.endswith("_L09"):
                metrics.record(f"{k}_{name}", v)
    print(f"\n[fix21] gap device - official at L09: {emu['err_layer_L09']:.5f} - {off['err_layer_L09']:.5f}")
    print(
        f"{'site':12s} {'loo L09':>8s} {'gap_closed':>10s} {'(median)':>9s} {'add L09':>8s} {'addone':>7s} {'(median)':>9s}"
    )
    tot_loo, tot_add = 0.0, 0.0
    for s in SITES:
        lo, ad = res[f"loo_{s}"]["errors"], res[f"add_{s}"]["errors"]
        per_l = [share(emu[k], lo[k], off[k], True) for k in (f"err_layer_L{i:02d}" for i in LAYERS)]
        per_a = [share(emu[k], ad[k], off[k], False) for k in (f"err_layer_L{i:02d}" for i in LAYERS)]
        k9 = "err_layer_L09"
        g9, a9 = share(emu[k9], lo[k9], off[k9], True), share(emu[k9], ad[k9], off[k9], False)
        gm = statistics.median(x for x in per_l if x == x)
        am = statistics.median(x for x in per_a if x == x)
        tot_loo, tot_add = tot_loo + g9, tot_add + a9
        metrics.record(f"gap_closed_{s}", g9)
        metrics.record(f"gap_closed_{s}_median", gm)
        metrics.record(f"addone_{s}", a9)
        metrics.record(f"addone_{s}_median", am)
        print(f"{s:12s} {lo[k9]:8.5f} {g9:10.3f} {gm:9.3f} {ad[k9]:8.5f} {a9:7.3f} {am:9.3f}")
    metrics.record("gap_closed_sum", tot_loo)
    metrics.record("addone_sum", tot_add)
    print(f"[fix21] sums at L09: leave-one-out {tot_loo:.3f}, add-one {tot_add:.3f} (1 = additive sites)")
    (runs.dir / "attribution.json").write_text(json.dumps(res, indent=1))  # every run's errors, beside the cache
    assert len(keys) > 0 and med > 0


def test_bfp_rules_match_packers():
    """ds41_ref.bfp_round is the repo's packer rules, not a new format: tests/test_bfp_packer_rounding.py's
    ``rne_host`` (ttnn's host packer; the SFPU packs are byte-equal to it) and ``device`` (bfp8_pack_precise, half
    away) on its crafted bf16 tiles and random ones, and tt/experts.py:_bfp4_round, value for value (normal groups;
    a group whose largest value is denormal packs as zeros)."""
    from models.demos.deepseek_v41_flash_d_p.tests import test_bfp_packer_rounding as P
    from models.demos.deepseek_v41_flash_d_p.tt.experts import _bfp4_round

    def decode(codes, shared, m):
        sign, q = (codes >> m) & 1, codes & ((1 << m) - 1)
        v = q.double() * torch.pow(2.0, (shared.repeat_interleave(16, dim=1) - 127 - (m - 1)).double())
        return torch.where(sign == 1, -v, v).float()

    g = torch.Generator().manual_seed(0)
    for m, rule, ties in ((3, "rne_host", "even"), (7, "rne_host", "even"), (7, "device", "away")):
        x = torch.cat(
            [P.crafted(m, False), torch.randn(2048, 32, generator=g) * torch.randn(2048, 1, generator=g).exp()]
        )
        x = x.bfloat16().float()
        normal = (x.abs().reshape(-1, 16).amax(-1) >= 2.0**-120).repeat_interleave(16).reshape(x.shape)
        want, got = decode(*P._quantize(x, m, rule), m), R.bfp_round(x, m, ties)
        assert torch.equal(want[normal], got[normal]), (m, rule, int((want != got)[normal].sum()))
        if m == 3:
            rows = normal.all(-1)
            assert torch.equal(_bfp4_round(x[rows]), R.bfp_round(x[rows], 3))


# (4) what the dominant site(s) buy at the tokens: 40 layers, rung last's chunk, FIX.20's logit KL
TOKEN_SITES: tuple[str, ...] = ("experts_w",)  # add_experts_w 0.95 of the L09 gap alone (FIX.21 attribution)
TOKEN_RUNS = ("device40", "minus40")


def tokens_formats(name: str) -> dict:
    fm = run_formats("device")
    if name == "minus40":
        for s in TOKEN_SITES:
            fm.pop(s)
    return fm


def tokens_run(runs: Runs, name: str, compute: bool) -> dict | None:
    """All 40 layers from the golden prefix (as test_logit_kl feeds the device), the final norm's token statistics
    against the native golden (test_logit_kl.token_stats / summary), cached."""
    p = runs.dir / f"{name}-{'+'.join(TOKEN_SITES)}.json"
    if p.exists():
        return json.loads(p.read_text())
    if not compute:
        return None
    from models.demos.deepseek_v41_flash_d_p.tests.bringup import test_logit_kl as KL

    g, gold, _, nxt, _ = KL.rung_inputs(RUNG)
    layers = list(range(runs.s.num_layers))
    ref = runs.ref40().with_formats(tokens_formats(name))
    t0 = time.time()
    rep = replay(runs.s, ref, g, layers, g.n_chunks - 1)
    fn = rep["final_norm"].float().reshape(gold.shape)
    secs = round(time.time() - t0, 1)
    errs = {f"err_layer_L{i:02d}": rel(rep["trail"][i], g.layer(g.n_chunks - 1, i)["out"].float()) for i in layers}
    res = {
        "run": name,
        "sites_off": list(TOKEN_SITES) if name == "minus40" else [],
        "seconds": secs,
        "err_final_norm": rel(fn, gold),
        "errors": errs,
        "tokens": KL.summary(KL.token_stats(fn, gold, nxt)),
    }
    del rep
    p.write_text(json.dumps(res, indent=1))
    print(f"[fix21] {name}: {secs:.0f} s, final_norm rel {res['err_final_norm']:.4f}, {res['tokens']}", flush=True)
    return res


def test_format_attribution_tokens(runs):
    """(4): the all-device emulation and the same without TOKEN_SITES over all 40 layers, at the tokens (FIX.20's
    metric on rung last's chunk; the official side is test_logit_kl's cached official final norm). Computed only by
    a fill job (FIX21_TOKENS=1); the gate run records the cached results."""
    if not TOKEN_SITES:
        pytest.skip("no dominant site chosen (TOKEN_SITES empty)")
    compute = os.environ.get("FIX21_TOKENS") == "1"
    res = {n: tokens_run(runs, n, compute) for n in TOKEN_RUNS}
    if compute and os.environ.get("FIX21_ONLY"):
        pytest.skip("fill job")
    missing = [n for n, r in res.items() if r is None]
    assert not missing, f"40-layer runs {missing} not made: FIX21_TOKENS=1 (CPU, about an hour each)"
    from models.demos.deepseek_v41_flash_d_p.tests.bringup import official_final_norm as OFN
    from models.demos.deepseek_v41_flash_d_p.tests.bringup import test_logit_kl as KL

    g, gold, _, nxt, _ = KL.rung_inputs(RUNG)
    off, _ = OFN.load(g, RUNG)
    osum = KL.summary(KL.token_stats(off.reshape(gold.shape), gold, nxt))
    dev = _device_metrics(DEVICE_TASK)
    metrics.record("tokens_sites_off", "+".join(TOKEN_SITES))
    for n, r in res.items():
        metrics.record(f"run_seconds_{n}", r["seconds"])
        metrics.record(f"err_final_norm_{n}", r["err_final_norm"])
        for k, v in r["tokens"].items():
            metrics.record(f"{k}_{n}", v)
    for k, v in osum.items():
        metrics.record(f"{k}_official40", v)
    metrics.record("kl_mean_device_fix20", dev.get(f"kl_mean_{RUNG}"))
    print(f"\n[fix21] tokens, rung {RUNG} (40 layers):")
    print(f"  {'':22s} {'kl_mean':>9s} {'top1':>7s} {'top5':>7s} {'nll':>7s} {'final rel':>9s}")
    rows = [
        (
            "device (FIX.20)",
            dev.get(f"kl_mean_{RUNG}"),
            dev.get(f"top1_agree_{RUNG}"),
            None,
            None,
            dev.get(f"err_final_norm_{RUNG}"),
        )
    ]
    rows += [
        (
            f"emu {n}",
            r["tokens"]["kl_mean"],
            r["tokens"]["top1_agree"],
            r["tokens"]["top5_ovl"],
            r["tokens"]["nll"],
            r["err_final_norm"],
        )
        for n, r in res.items()
    ]
    rows += [
        (
            "official",
            osum["kl_mean"],
            osum["top1_agree"],
            osum["top5_ovl"],
            osum["nll"],
            dev.get(f"err_final_norm_{RUNG}_official"),
        )
    ]
    for name, *v in rows:
        print(f"  {name:22s} " + " ".join("     None" if x is None else f"{x:9.5f}" for x in v))

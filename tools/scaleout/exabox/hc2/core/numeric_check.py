#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""numeric_check.py: per-chip numeric check of the stress matmul (DIDT-style 'ND' check).

Every chip of the mesh runs the same bf16 matmul on the same replicated inputs, so every chip's output must be
bit-identical. During the burst, ub0_repro.py reads back the output of some matmuls (the first one always: the cold
start) and compares each chip with the golden output, bit for bit:
  - the golden for an activation is the output most chips agree on at its first check (a chip that is wrong on the
    first, cold, matmul cannot become the reference), sanity-checked against a CPU fp32 matmul of a row slice;
  - like tests/didt/op_test_base.py (determinism check) the checked matmuls alternate between two activations, so a
    chip that stopped writing its output buffer cannot pass by returning the previous result;
  - per chip: checks, mismatched checks, first mismatched iteration, mismatched elements, max abs error, NaN, Inf.
A mismatch never stops the load: it is counted and reported (numeric_report.py prints the per-chip lines).

Chip naming: metal device id -> PCIe id (/dev/tenstorrent/N) from UMD's 'Opening local chip ids/PCIe ids' log line
-> PCI address -> tray 1-4 (bus & 0xF0, ub0_repro.py's BUS_TO_TRAY).
"""
import os
import re
import time

UMD_MAP_RE = re.compile(r"Opening local chip ids/PCIe ids: \{([0-9, ]*)\}/\[([0-9, ]*)\]")


def umd_map_from_text(text):
    """UMD's 'Opening local chip ids/PCIe ids: {0, 1, ...}/[0, 1, ...]' line -> {logical (metal) id: PCIe id}."""
    m = None
    for m in UMD_MAP_RE.finditer(text or ""):
        pass                                    # the last open in the log wins
    if not m:
        return {}
    lg = [int(v) for v in m.group(1).replace(" ", "").split(",") if v != ""]
    pc = [int(v) for v in m.group(2).replace(" ", "").split(",") if v != ""]
    return dict(zip(lg, pc)) if len(lg) == len(pc) else {}


def umd_map_from_stdout():
    """The map from this process's own stdout when it goes to a file (run.sh: > try<i>.log); {} otherwise."""
    for fd in (1, 2):
        try:
            p = os.readlink(f"/proc/self/fd/{fd}")
            if os.path.isfile(p):
                with open(p, "r", errors="replace") as f:
                    got = umd_map_from_text(f.read())
                if got:
                    return got
        except OSError:
            pass
    return {}


class NumericCheck:
    """Bit-exact per-chip comparison of replicated matmul outputs. Never raises out of check(): an error in the check
    itself is recorded and turns the checker off for the rest of the run (the load goes on)."""

    def __init__(self, torch, ttnn, mesh, every, budget_s, cpu_rows, log, max_checks=0):
        self.torch, self.ttnn, self.mesh, self.log = torch, ttnn, mesh, log
        self.every, self.budget_s, self.cpu_rows, self.max_checks = every, budget_s, cpu_rows, max_checks
        self.golden = {}                 # activation index -> golden torch tensor (one chip's output)
        self.golden_info = {}            # activation index -> how it was chosen (agree count, CPU pcc, max err)
        self.cpu_inputs = {}             # activation index -> (x, w) CPU bf16 tensors for the CPU slice check
        self.chips = {}                  # mesh index -> per-chip counters
        self.checks, self.spent, self.error, self.off = [], 0.0, None, False
        self.ids = self._metal_ids()

    def _metal_ids(self):
        try:
            return [int(v) for v in self.mesh.get_device_ids()]
        except Exception:
            try:
                n = self.mesh.get_num_devices()
            except Exception:
                n = 0
            return list(range(n))

    def set_inputs(self, k, x, w):
        self.cpu_inputs[k] = (x, w)

    def want(self, n, last=False):
        """Check matmul n (1-based)? The first always; then every --numeric-every while the budget lasts; the last."""
        if self.off:
            return False
        if self.max_checks and len(self.checks) >= self.max_checks:
            return False
        if n == 1:
            return True
        if self.spent >= self.budget_s:
            return False
        return last or (self.every > 0 and n % self.every == 0)

    def _per_device(self, y):
        ttnn = self.ttnn
        host = y.cpu()
        return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(host)]

    def _cpu_slice(self, k):
        torch = self.torch
        x, w = self.cpu_inputs.get(k, (None, None))
        if x is None or self.cpu_rows <= 0:
            return None
        r = min(self.cpu_rows, x.shape[-2])
        return torch.matmul(x[..., :r, :].float(), w.float())

    def _set_golden(self, k, outs):
        torch = self.torch
        reps = []                                   # [(tensor, [mesh indices])]
        for i, t in enumerate(outs):
            for rep in reps:
                if rep[0].shape == t.shape and torch.equal(rep[0], t):
                    rep[1].append(i)
                    break
            else:
                reps.append((t, [i]))
        reps.sort(key=lambda r: -len(r[1]))
        g, agree = reps[0][0], reps[0][1]
        info = dict(agree=len(agree), of=len(outs), variants=len(reps), finite=bool(torch.isfinite(g.float()).all()))
        ref = self._cpu_slice(k)
        if ref is not None:
            gs = g.float()[..., :ref.shape[-2], :]
            d = (gs - ref).abs()
            info["cpu_rows"] = int(ref.shape[-2])
            info["cpu_max_abs_err"] = float(d.max()) if bool(torch.isfinite(d).all()) else float("nan")
            a, b = gs.flatten().double(), ref.flatten().double()
            a, b = a - a.mean(), b - b.mean()
            den = float((a.norm() * b.norm()))
            info["cpu_pcc"] = round(float((a * b).sum()) / den, 6) if den > 0 else None
            info["cpu_ok"] = bool(info["finite"] and info["cpu_pcc"] is not None and info["cpu_pcc"] >= 0.99)
        else:
            info["cpu_ok"] = None
        if len(agree) * 2 <= len(outs):
            info["warning"] = "no majority: golden is the largest group only"
        self.golden[k], self.golden_info[k] = g, info
        self.log(f"NUMERIC golden act{k}: {info['agree']}/{info['of']} chips bit-identical ({info['variants']} "
                 f"distinct outputs), CPU {info.get('cpu_rows', 0)}-row slice pcc {info.get('cpu_pcc')} "
                 f"max abs err {info.get('cpu_max_abs_err')}" + (f"; {info['warning']}" if "warning" in info else ""))

    def check(self, y, n, k, t_burst):
        """Compare matmul n's output y (activation k) on every chip with the golden. Returns seconds spent."""
        t0 = time.monotonic()
        torch = self.torch
        try:
            outs = self._per_device(y)
            if k not in self.golden:
                self._set_golden(k, outs)
            g = self.golden[k]
            gi = g.view(torch.int16) if g.dtype == torch.bfloat16 else g
            gf = g.float()
            bad = []
            for i, t in enumerate(outs):
                c = self.chips.setdefault(i, dict(metal=self.ids[i] if i < len(self.ids) else i, checks=0, mismatched=0,
                                                  first_iter=None, first_s=None, bad_elems=0, max_bad_elems=0,
                                                  max_abs_err=0.0, nan=0, inf=0, first_check_bad=None))
                c["checks"] += 1
                if t.shape == g.shape and torch.equal(t, g):
                    if c["first_check_bad"] is None:
                        c["first_check_bad"] = False
                    continue
                tf = t.float()
                nan, inf = int(torch.isnan(tf).sum()), int(torch.isinf(tf).sum())
                if t.shape == g.shape:
                    ti = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
                    ne = int((ti != gi).sum())
                    d = (tf - gf).abs()
                    d = d[torch.isfinite(d)]
                    mx = float(d.max()) if d.numel() else float("inf")
                else:
                    ne, mx = t.numel(), float("inf")
                c["mismatched"] += 1
                c["bad_elems"] += ne
                c["max_bad_elems"] = max(c["max_bad_elems"], ne)
                c["max_abs_err"] = max(c["max_abs_err"], mx)
                c["nan"] += nan
                c["inf"] += inf
                if c["first_iter"] is None:
                    c["first_iter"], c["first_s"] = n, round(t_burst, 2)
                if c["first_check_bad"] is None:
                    c["first_check_bad"] = True
                bad.append(f"mesh{i}/metal{c['metal']}: {ne} elems, max abs err {mx:.3g}, NaN {nan}, Inf {inf}")
            dt = time.monotonic() - t0
            self.spent += dt
            self.checks.append(dict(iter=n, act=k, s=round(dt, 2), t_burst=round(t_burst, 2), bad=len(bad)))
            if bad or len(self.checks) <= 2:
                self.log(f"NUMERIC check {len(self.checks)} matmul {n} act{k} ({dt:.1f}s): "
                         + (f"{len(bad)} chip(s) MISMATCH: " + "; ".join(bad[:8]) if bad else
                            f"{len(outs)}/{len(outs)} chips bit-exact"))
            return dt
        except Exception as e:      # the check failed, not the chip: stop checking, never stop the load
            dt = time.monotonic() - t0
            self.spent += dt
            self.off = True
            self.error = f"{type(e).__name__}: {str(e)[:300]}"
            self.log(f"NUMERIC check error at matmul {n} (checks off for the rest of this try, load goes on): "
                     f"{self.error}")
            return dt

    def result(self, ids, tray_of, umd_map=None):
        """Per-chip record for the result JSON: mesh index -> metal id -> PCIe id -> PCI address -> tray."""
        umd_map = umd_map if umd_map is not None else umd_map_from_stdout()
        src = "umd_log" if umd_map else "assumed_identity"
        chips = {}
        for i, c in sorted(self.chips.items()):
            dev = umd_map.get(c["metal"], c["metal"]) if umd_map else c["metal"]
            pci = ids.get(dev)
            tray = tray_of(pci) if pci else None
            chips[str(i)] = dict(c, dev=dev, pci=pci, tray=tray, ubb=None if tray is None else tray - 1,
                                 max_abs_err=None if c["max_abs_err"] == float("inf") else round(c["max_abs_err"], 6))
        return dict(schema=1, enabled=True, every=self.every, budget_s=self.budget_s, cpu_rows=self.cpu_rows,
                    checks=len(self.checks), check_iters=[c["iter"] for c in self.checks],
                    check_s=round(self.spent, 2), map_source=src, golden={str(k): v for k, v in self.golden_info.items()},
                    error=self.error, chips=chips,
                    mismatched_chips=sorted(int(i) for i, c in chips.items() if c["mismatched"]))


# SFPU ops in the heavy numeric vote: each checked heavy matmul output's first H x W
# block, height-sharded one 32-row shard per core (shard s on core x,y = s % gx, s // gx, ROW_MAJOR), goes through one
# SFPU op per launch, rotating these; every chip's result must be bit-identical, like the matmul's.
SFPU_OPS = ("exp", "reciprocal", "sqrt", "gelu")   # sqrt runs on abs(x): a NaN would never compare equal to itself


def sfpu_layout(gx, gy, w=64):
    """(rows, cols) of the sharded SFPU input on a gx x gy core grid: one 32-row shard per core."""
    return 32 * gx * gy, w


def sfpu_core(row, gx):
    """Core 'x,y' that computed a row of the height-sharded SFPU output."""
    s = int(row) // 32
    return f"{s % gx},{s // gx}"


def cores_from_rows(rows, gx):
    """{'x,y': wrong elements} from the rows of the wrong elements, most first."""
    out = {}
    for r in rows:
        k = sfpu_core(r, gx)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def cores_from_counter(torch, counter, gx):
    """A chip's 2-D mismatch counter -> {'x,y': elements that mismatched at least once}."""
    c = counter.reshape(-1, counter.shape[-1])
    return cores_from_rows(torch.nonzero(c)[:, 0].tolist(), gx)


def sfpu_vote(torch, outs, gx):
    """Cross-chip bit-exact vote of one SFPU output per chip (bf16 compared by int16 view). The majority output is the
    reference; {chip index: {'elems': n, 'cores': {'x,y': n}}} for each chip off it; {} with no strict majority."""
    def bits(t):
        return t.view(torch.int16) if t.dtype == torch.bfloat16 else t
    keys = [hash(bits(t).numpy().tobytes()) for t in outs]
    top = max(set(keys), key=keys.count) if keys else None
    if top is None or keys.count(top) * 2 <= len(keys):
        return {}
    g = bits(outs[keys.index(top)])
    bad = {}
    for i, t in enumerate(outs):
        if keys[i] == top:
            continue
        d = bits(t) != g if t.shape == g.shape else torch.ones(g.shape, dtype=torch.bool)
        d = d.reshape(-1, d.shape[-1])
        bad[i] = dict(elems=int(d.sum()), cores=cores_from_rows(torch.nonzero(d)[:, 0].tolist(), gx))
    return bad


def sfpu_cores_text(sf):
    """', SFPU vote core x,y X,Y (op, k elems)...' for a chip's {'<op>': {'x,y': n}}; '' when none.
    Test helper (tests/test_numeric_sfpu.py): run.sh's report path uses fleet/chunk_sched.sfpu_text."""
    items = [(n, op, xy) for op, cs in (sf or {}).items() for xy, n in cs.items()]
    if not items:
        return ""
    items.sort(key=lambda t: (-t[0], t[1], t[2]))
    return ", SFPU vote " + "; ".join(f"core x,y {xy} ({op}, {n} elems)" for n, op, xy in items[:8])


def sfpu_lines(bad, ids, tray_of, umd_map, metal_ids, op):
    """'NUMERIC FAIL <BDF> (tray N) chip D [sfpu:<op>]: ...core x,y X,Y' per chip of a sfpu_vote() result.
    Test helper for the CPU vote model (tests/test_numeric_sfpu.py), not in run.sh's report path: the heavy chunk's FAIL
    line comes from fleet/chunk_sched.report with sfpu_count and sfpu_text."""
    out = []
    for i, b in sorted(bad.items()):
        metal = metal_ids[i] if i < len(metal_ids) else i
        dev = umd_map.get(metal, metal) if umd_map else metal
        pci = ids.get(dev)
        out.append(f"NUMERIC FAIL {pci or '?'} (tray {(tray_of(pci) if pci else None) or '?'}) chip {dev} [sfpu:{op}]: "
                   f"{b['elems']} elements off the cross-chip majority" + sfpu_cores_text({op: b["cores"]}))
    return out

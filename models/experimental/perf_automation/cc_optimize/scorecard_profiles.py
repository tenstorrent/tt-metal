# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

_METRIC_KEYS = ("prefill_time_to_first_token", "decode_t/s/u", "decode_t/s", "prefill_t/s")

_TOL_KEYS = {
    "prefill_time_to_first_token": "prefill_time_to_first_token_tolerance",
    "decode_t/s/u": "decode_t_s_u_tolerance",
    "decode_t/s": "decode_t_s_tolerance",
    "prefill_t/s": "prefill_t_s_tolerance",
}

_MEASURED_KEYS = {
    "prefill_time_to_first_token": "TTFT_ms",
    "decode_t/s/u": "TSU",
    "decode_t/s": "TS",
}


def _repo_root_default():
    return Path(__file__).resolve().parents[4]


def _load_targets(repo_root):
    base = Path(repo_root) if repo_root else _repo_root_default()
    p = base / "models" / "model_targets.yaml"
    if not p.is_file():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}
    return data.get("targets", {}) or {}


def sku_for(arch, chips):
    a = (arch or "").lower()
    try:
        c = int(chips or 1)
    except Exception:
        c = 1
    if any(t in a for t in ("blackhole", "bh", "p300", "p150", "p100")):
        return {1: "bh_p100", 2: "bh_p300", 4: "p300x2"}.get(c, "bh_p300")
    if any(t in a for t in ("wormhole", "wh", "n300", "n150")):
        return {1: "wh_n150", 2: "wh_n300"}.get(c, "wh_n300")
    return None


def _match(targets, model_id):
    mid = (model_id or "").lower()
    short = mid.split("/")[-1]
    for name, spec in targets.items():
        nlow = str(name).lower()
        if nlow in (mid, short):
            return name, spec
        for al in spec.get("aliases") or []:
            if str(al).lower() in (mid, short):
                return name, spec
    for name, spec in targets.items():
        nlow = str(name).lower()
        if short and (short in nlow or nlow in short):
            return name, spec
    return None, None


def resolve(model_id, arch, chips, repo_root=None):
    targets = _load_targets(repo_root)
    name, spec = _match(targets, model_id)
    out = {
        "found": False,
        "matched_model": name,
        "sku": sku_for(arch, chips),
        "batch_size": None,
        "seq_len": None,
        "targets": {},
        "tolerances": {},
    }
    if not spec:
        return out
    sku_spec = (spec.get("skus", {}) or {}).get(out["sku"])
    if not sku_spec:
        return out
    entries = sku_spec.get("entries", []) or []
    active = [e for e in entries if e.get("status") == "active"]
    entry = (active or entries or [{}])[0]
    perf = entry.get("perf", {}) or {}
    out["found"] = True
    out["batch_size"] = entry.get("batch_size")
    out["seq_len"] = entry.get("seq_len")
    out["targets"] = {k: perf.get(k) for k in _METRIC_KEYS if perf.get(k) is not None}
    out["tolerances"] = {k: perf.get(v) for k, v in _TOL_KEYS.items() if perf.get(v) is not None}
    return out


def _fmt(v):
    if v is None:
        return "—"
    try:
        return "%.2f" % float(v)
    except Exception:
        return str(v)


def _within(measured, target, tol):
    try:
        lo, hi = float(target) * (1.0 - float(tol)), float(target) * (1.0 + float(tol))
        return "PASS" if lo <= float(measured) <= hi else "OUT"
    except Exception:
        return "—"


def render(model_id, arch, chips, measured, repo_root=None):
    prof = resolve(model_id, arch, chips, repo_root)
    sku = prof.get("sku") or "?"
    bsz = measured.get("batch") or prof.get("batch_size")
    seq = measured.get("ISL") or prof.get("seq_len")
    osl = measured.get("OSL")
    meas = {k: measured.get(v) for k, v in _MEASURED_KEYS.items()}
    tgt = prof.get("targets", {})
    tol = prof.get("tolerances", {})
    rule = "  " + "─" * 78
    note = "" if prof.get("found") else "   (not in model_targets.yaml — measured-only)"
    lines = [
        rule,
        "  MODEL_TARGETS SCORECARD — %s" % model_id,
        "  sku=%s  batch_size=%s  seq_len=%s  OSL=%s%s" % (sku, bsz, seq, osl, note),
        rule,
        "  %-32s %-11s %-11s %-8s %s" % ("metric", "measured", "target", "tol", "within?"),
    ]
    for k in ("prefill_time_to_first_token", "decode_t/s/u", "decode_t/s"):
        m, t, to = meas.get(k), tgt.get(k), tol.get(k)
        verdict = _within(m, t, to) if (m is not None and t is not None and to is not None) else "—"
        lines.append("  %-32s %-11s %-11s %-8s %s" % (k, _fmt(m), _fmt(t), _fmt(to), verdict))
    lines += [
        rule,
        "  model_targets.yaml block (measured @ sku=%s):" % sku,
        "    %s:" % model_id,
        "      skus:",
        "        %s:" % sku,
        "          entries:",
        "            - batch_size: %s" % (bsz if bsz is not None else "null"),
        "              seq_len: %s" % (seq if seq is not None else "null"),
        "              status: active",
        "              perf:",
        "                prefill_time_to_first_token: %s" % _fmt(meas["prefill_time_to_first_token"]),
        "                decode_t/s/u: %s" % _fmt(meas["decode_t/s/u"]),
        "                decode_t/s: %s" % _fmt(meas["decode_t/s"]),
        rule,
    ]
    return "\n".join(lines)

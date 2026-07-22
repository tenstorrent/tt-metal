# SPDX-License-Identifier: Apache-2.0
"""DRAM-bandwidth roofline for decode.

Two entry points, one scoring path:
  * compute_target(model_facts, hw_facts, ...)  — MODEL-LEVEL tok/s ceiling + band from
    active_bytes / peak_BW (full-pipeline optimize).
  * target_from_floor_ms(modeled_floor_ms)      — PER-MODULE band from the module's aggregate
    bandwidth floor (roofline.residual_report.modeled_floor_ms); expressed as invocations/ms
    (no decode loop, so no true tok/s) but scored identically.

score(target, forward_ms) returns measured rate, bw_util (fraction of the achievable ceiling),
and BELOW_BAND | IN_BAND | ABOVE_BAND | UNKNOWN. Decode only; prefill is FLOP-bound (stub).
Pure + unit-testable — no device, no pipeline, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_ELEM = {
    "bfloat16": 2.0,
    "bfloat8_b": 1.0625,
    "bfloat4_b": 0.5625,
    "float16": 2.0,
    "float32": 4.0,
    "int8": 1.0,
    "bf16": 2.0,
    "bf8_b": 1.0625,
    "bf4_b": 0.5625,
    "fp16": 2.0,
    "fp32": 4.0,
}
_DEFAULT_BYTES_PER_ELEM = 2.0
_BAND_LO_FRAC = 0.60
_BAND_HI_FRAC = 0.80


def _bytes_per_elem(dtype) -> float:
    return BYTES_PER_ELEM.get(str(dtype or "").strip().lower(), _DEFAULT_BYTES_PER_ELEM)


def _scalar(v, default=0):
    """Coerce a config value that may arrive as a list/dict (e.g. per-layer top_k) to a scalar,
    so a structured value degrades instead of crashing (fixes-plan Point 1)."""
    if isinstance(v, (list, tuple)):
        return _scalar(v[0], default) if v else default
    if isinstance(v, dict):
        vals = list(v.values())
        return _scalar(vals[0], default) if vals else default
    try:
        return type(default)(v)
    except (TypeError, ValueError):
        return default


@dataclass
class PerfTarget:
    active_bytes: int
    peak_bw_bytes_s: float
    theoretical_tok_s: float
    band: tuple[float, float]
    regime: str = "decode"
    tp_degree: int = 1
    seq_len: int = 0


def active_bytes(model_facts: dict, *, regime: str = "decode", seq_len: int = 0) -> int:
    """Bytes streamed from DRAM per decode step, summed per-tensor at each tensor's real dtype.

    Dense: Σ tensor_bytes(all weight tensors). MoE: shared_bytes + top_k * per_expert_bytes
    (the reachable read set — NOT all experts). Optional KV term when seq_len>0."""
    if regime != "decode":
        raise NotImplementedError("perf_target models the decode regime only (prefill is FLOP-bound)")
    mf = model_facts or {}

    if mf.get("is_moe"):
        dt = mf.get("dominant_dtype") or mf.get("torch_dtype") or "bfloat16"
        shared = _shared_bytes(mf, dt)
        per_expert = float(mf.get("per_expert_bytes") or (float(mf.get("per_expert_params", 0)) * _bytes_per_elem(dt)))
        top_k = _scalar(mf.get("top_k", 0), 0)
        wb = shared + top_k * per_expert
    else:
        tensors = mf.get("weight_tensors")
        if tensors:
            wb = sum(float(t.get("numel", 0)) * _bytes_per_elem(t.get("dtype")) for t in tensors)
        else:
            dt = mf.get("dominant_dtype") or mf.get("torch_dtype") or "bfloat16"
            wb = float(mf.get("total_params", 0)) * _bytes_per_elem(dt)

    kv = 0.0
    if seq_len and mf.get("layers") and mf.get("kv_heads") and mf.get("head_dim"):
        kv_dt = mf.get("kv_dtype") or mf.get("dominant_dtype") or "bfloat16"
        kv = 2.0 * int(mf["layers"]) * int(mf["kv_heads"]) * int(mf["head_dim"]) * int(seq_len) * _bytes_per_elem(kv_dt)

    return int(round(wb + kv))


def _shared_bytes(mf: dict, dt) -> float:
    """Always-on MoE bytes: attention + router + shared experts + resident embeddings."""
    if mf.get("shared_tensors"):
        return sum(float(t.get("numel", 0)) * _bytes_per_elem(t.get("dtype")) for t in mf["shared_tensors"])
    if mf.get("shared_bytes") is not None:
        return float(mf["shared_bytes"])
    return float(mf.get("shared_params", 0)) * _bytes_per_elem(dt)


def compute_target(model_facts: dict, hw_facts: dict, *, tp_degree: int = 1, seq_len: int = 0) -> PerfTarget:
    """MODEL-LEVEL decode ceiling. Per-device convention: per-device bytes vs single-chip BW
    (never per-device bytes against mesh-aggregate BW — that is the 4-8x error)."""
    ab = active_bytes(model_facts, seq_len=seq_len)
    peak_bw = float((hw_facts or {}).get("dram_bw_gbps", 0.0)) * 1e9
    tp = max(1, int(tp_degree or 1))
    ab_per_device = ab / tp
    theo = (peak_bw / ab_per_device) if ab_per_device > 0 else 0.0
    band = (_BAND_LO_FRAC * theo, _BAND_HI_FRAC * theo)
    return PerfTarget(
        active_bytes=ab,
        peak_bw_bytes_s=peak_bw,
        theoretical_tok_s=theo,
        band=band,
        tp_degree=tp,
        seq_len=seq_len,
    )


def target_from_floor_ms(modeled_floor_ms: float) -> PerfTarget:
    """PER-MODULE target from the module's aggregate DRAM-bandwidth floor (roofline
    residual_report.modeled_floor_ms). 'Rate' is invocations/s (1000/ms) — there is no decode
    loop, so this is a relative achievable rate, scored the same way as the model-level ceiling."""
    theo = (1000.0 / modeled_floor_ms) if modeled_floor_ms and modeled_floor_ms > 0 else 0.0
    band = (_BAND_LO_FRAC * theo, _BAND_HI_FRAC * theo)
    return PerfTarget(active_bytes=0, peak_bw_bytes_s=0.0, theoretical_tok_s=theo, band=band)


def score(target: PerfTarget, forward_ms: float) -> dict:
    """Compare a measured decode-step / invocation time against the target.

    bw_util = measured / theoretical = effective_BW / peak_BW (identical), i.e. the fraction of
    the achievable ceiling reached. status: BELOW_BAND (keep optimizing) | IN_BAND (>= 60% of
    ceiling, done) | ABOVE_BAND (beat the ceiling -> active_bytes/floor suspect, assert never win)
    | UNKNOWN (no valid target or measurement)."""
    theo = target.theoretical_tok_s if target else 0.0
    if not theo or theo <= 0 or not forward_ms or forward_ms <= 0:
        return {"status": "UNKNOWN", "measured_tok_s": None, "bw_util": None, "theoretical_tok_s": theo or None}
    measured = 1000.0 / forward_ms
    bw_util = measured / theo
    lo, _hi = target.band
    if measured > theo:
        status = "ABOVE_BAND"
    elif measured >= lo:
        status = "IN_BAND"
    else:
        status = "BELOW_BAND"
    eff_bw = None
    if target.peak_bw_bytes_s > 0 and target.active_bytes > 0:
        eff_bw = (target.active_bytes / max(1, target.tp_degree)) / (forward_ms / 1000.0)
    return {
        "status": status,
        "measured_tok_s": round(measured, 3),
        "theoretical_tok_s": round(theo, 3),
        "bw_util": round(bw_util, 4),
        "band": (round(target.band[0], 3), round(target.band[1], 3)),
        "effective_bw_bytes_s": eff_bw,
    }


def prefill_ceiling(*_a, **_k):
    raise NotImplementedError("prefill is FLOP-bound; v1 models decode only (peak_TFLOPs/model_FLOPs stub)")

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

import math
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

# ---------------------------------------------------------------------------
# THE CEILING IS PARAMS-BASED AND SUSTAINED (team decision, 2026-07-29).
#
#   dense: (peak_BW * 0.80) / params_GB          8B on a 512 GB/s part -> (512*0.8)/8 = 51.2 tok/s/u
#   MoE:   (peak_BW * 0.50) / active_params_GB   3B active -> (512*0.5)/3 = 85.3 tok/s/u
#
# 1 B params -> 1 GB streamed. Deliberately an approximation: the exact on-device byte count (6.095 GB
# for Llama-3.1-8B served as bf4/bf8) is more accurate, but establishing it costs a per-model
# investigation of what width each tensor group is actually served at. A param count is
# dtype-independent, so it needs no such work, and xB -> xGB lands close enough to steer optimization.
#
# The DRAM efficiency the part sustains (~80% of spec on dense streams; ~50% on MoE, whose per-token
# expert reads are scattered) is folded INTO the ceiling, so theoretical_rate is ACHIEVABLE rather than
# spec. MoE cards publish the active-parameter count ("30B-A3B" -> 3B active), which is this input.
_BYTES_PER_PARAM = 1.0
_DENSE_BW_FRACTION = 0.80
_MOE_BW_FRACTION = 0.50
# Band lo as a fraction of the ACHIEVABLE ceiling. 0.75 is chosen so the DENSE stop threshold lands
# exactly where it did before the efficiency factor moved inside the ceiling:
# 0.60 * peak == 0.75 * (0.80 * peak). Keeping 0.60 here instead would move the "done" line from
# 38.4 to 30.7 tok/s/u for an 8B model -- i.e. the run would stop optimizing sooner as a pure
# side effect of restating the same physics.
_ACHIEVABLE_BAND_LO_FRAC = 0.75


_UNKNOWN_DTYPES: set = set()
_SPELLING_ALIASES = {
    "bfp8_b": "bfloat8_b",
    "bfp8": "bfloat8_b",
    "bfp4_b": "bfloat4_b",
    "bfp4": "bfloat4_b",
    "bf16": "bfloat16",
    "fp16": "float16",
    "float32": "fp32",
    "f32": "fp32",
}
_EXTRA_WIDTHS = {"int8": 1.0, "uint8": 1.0, "fp8": 1.0, "int4": 0.5, "nf4": 0.5, "int32": 4.0}


_KNOWN_SPELLINGS = dict(
    list(BYTES_PER_ELEM.items())
    + [(k, BYTES_PER_ELEM[v]) for k, v in _SPELLING_ALIASES.items() if v in BYTES_PER_ELEM]
    + list(_EXTRA_WIDTHS.items())
)


def _bytes_per_elem(dtype) -> float:
    """Bytes per element for a dtype spelling.

    A miss used to fall silently to 2.0, so a bf8_b model's active_bytes was overstated ~2x, its
    theoretical tok/s understated ~2x, and score() could return IN_BAND -> the run stopped
    declaring success from a string-key miss. Normalise the spellings that actually occur and
    record anything still unknown so the caller can report a degraded ceiling.
    """
    s = str(dtype or "").strip().lower()
    s = s.rsplit(".", 1)[-1]  # DataType.BFLOAT8_B -> bfloat8_b
    if s in BYTES_PER_ELEM:
        return BYTES_PER_ELEM[s]
    # Known spellings first: free, exact, and correct with no agent available.
    for cand, width in _KNOWN_SPELLINGS.items():
        if s == cand:
            return width
    # Only a genuinely unrecognised spelling reaches the agent (cached), so new dtype names from a
    # future model resolve by MEANING instead of silently taking the default byte width.
    try:
        from agent import integrity as _integrity

        resolved = _integrity.classify(
            s,
            set(BYTES_PER_ELEM),
            what="dtype",
            evidence="Tenstorrent/torch tensor element type; map to the equivalent known dtype by bit width.",
        )
        if resolved:
            return BYTES_PER_ELEM[resolved]
    except Exception:  # noqa: BLE001
        pass
    if s:
        _UNKNOWN_DTYPES.add(s)
    return _DEFAULT_BYTES_PER_ELEM


def unknown_dtypes() -> list:
    """Dtype spellings that fell back to the default -- the ceiling derived from them is a GUESS."""
    return sorted(_UNKNOWN_DTYPES)


def _scalar(v, default=0):
    """Coerce a config value that may arrive as a list/dict (e.g. per-layer top_k) to a scalar,
    so a structured value degrades instead of crashing (fixes-plan Point 1)."""
    if isinstance(v, (list, tuple)):
        return _scalar(v[0], default) if v else default
    if isinstance(v, dict):
        vals = list(v.values())
        return _scalar(vals[0], default) if vals else default
    # NaN/inf are UNKNOWN, not enormous. json.loads accepts `Infinity`, so a hand-edited or corrupted
    # perf_target_inputs.json reaches here, and int(inf) raises OverflowError -- which is neither a
    # TypeError nor a ValueError, so it escaped this guard and crashed the whole ceiling path.
    if isinstance(v, float) and not math.isfinite(v):
        return default
    try:
        return type(default)(v)
    except (TypeError, ValueError, OverflowError):
        return default


@dataclass
class PerfTarget:
    active_bytes: int
    peak_bw_bytes_s: float
    # NOT "tok_s". This is the ceiling per UNIT OF WORK -- tokens/s for an autoregressive model,
    # steps/s for a diffusion model, inferences/s for a single-pass one -- and the unit is in `unit`
    # below. The field was called theoretical_tok_s, which read as a promise that the number was
    # per-token; that is what led a reader to hardcode depth="token" when looking up the byte anchor,
    # so every non-LLM model silently fell back to a stale snapshot. A name that cannot be misread is
    # cheaper than the bug it prevents.
    theoretical_rate: float
    band: tuple[float, float]
    regime: str = "decode"
    tp_degree: int = 1
    seq_len: int = 0
    # THE UNIT THE CEILING IS PER. peak_BW / active_bytes is a rate only if `active_bytes` is what ONE
    # unit of work reads, and that unit differs by model: a token (autoregressive), a denoise step
    # (diffusion), one forward pass (classifier/encoder). Carried here so the band can check that the
    # measurement it is handed counts the SAME unit -- a per-step ceiling scored against a per-token
    # reading is not a comparison, and nothing downstream could previously tell.
    unit: str = "token"
    # Fraction of peak DRAM BW folded into theoretical_rate (0.80 dense / 0.50 MoE). Carried so a
    # report can state the ceiling is SUSTAINED rather than spec, and so the spec number stays
    # recoverable (theoretical_rate / bw_fraction) instead of being lost inside one multiplication.
    bw_fraction: float = 1.0
    # How the divisor was obtained: the params rule, or the exact per-tensor bytes it falls back to.
    bytes_source: str = ""


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
        elif mf.get("weight_bytes"):
            # The checkpoint's own byte count, when the producer measured it instead of inferring it
            # from params x dtype. This is the "model size" term of the decode bound -- 8 GB / 512 GB/s
            # = 64 tok/s/u for Llama-3.1-8B -- and reading it off disk avoids the params-times-dtype
            # round trip, which is wrong for any mixed- or quantised-dtype checkpoint.
            wb = float(mf["weight_bytes"])
        else:
            dt = mf.get("dominant_dtype") or mf.get("torch_dtype") or "bfloat16"
            wb = float(mf.get("total_params", 0)) * _bytes_per_elem(dt)

    kv = 0.0
    if seq_len and mf.get("layers") and mf.get("kv_heads") and mf.get("head_dim"):
        kv_dt = mf.get("kv_dtype") or mf.get("dominant_dtype") or "bfloat16"
        kv = 2.0 * int(mf["layers"]) * int(mf["kv_heads"]) * int(mf["head_dim"]) * int(seq_len) * _bytes_per_elem(kv_dt)

    total = wb + kv
    # Non-finite means the facts are junk (a corrupted/hand-edited perf_target_inputs.json: json.loads
    # accepts `Infinity`). int(round(inf)) raises OverflowError and took the whole ceiling path with it;
    # 0 reads as "no byte count", which is what an unusable input is.
    if not math.isfinite(total):
        return 0
    return int(round(total))


def ceiling_params(model_facts: dict) -> int:
    """Params streamed per unit of work: ACTIVE params for MoE, total params for dense.

    A routed token reads the shared trunk plus only the experts it selects, so `active_params` is the
    read set -- taken directly (what a model card publishes) or derived from
    shared_params + top_k * per_expert_params when that split is known. 0 when unknown.
    """
    mf = model_facts or {}
    if mf.get("is_moe"):
        active = _scalar(mf.get("active_params", 0), 0)
        if active > 0:
            return int(active)
        shared = _scalar(mf.get("shared_params", 0), 0)
        per_expert = _scalar(mf.get("per_expert_params", 0), 0)
        top_k = _scalar(mf.get("top_k", 0), 0)
        if shared > 0 and per_expert > 0 and top_k > 0:
            return int(shared + top_k * per_expert)
        return 0
    return int(_scalar(mf.get("total_params", 0), 0))


def simple_active_bytes(model_facts: dict) -> int:
    """Bytes streamed per unit of work under the xB -> xGB rule. 0 when the param count is unknown."""
    return int(round(ceiling_params(model_facts) * _BYTES_PER_PARAM))


def bw_fraction(model_facts: dict) -> float:
    """Fraction of peak DRAM bandwidth this model's read pattern actually sustains."""
    return _MOE_BW_FRACTION if (model_facts or {}).get("is_moe") else _DENSE_BW_FRACTION


def rate_and_band(
    bytes_per_unit: float, peak_bw_bytes_s: float, *, frac: float = _DENSE_BW_FRACTION, tp_degree: int = 1
):
    """(ceiling, (band_lo, band_hi)) from an ALREADY-KNOWN byte count.

    The one place the ceiling arithmetic lives. Exists because a second caller (the report renderer,
    recomputing from the ledger's byte anchor) had its own copy: `peak / bytes` with a hardcoded
    (0.60, 0.80) band. That copy silently kept the pre-sustained-fraction physics, so an anchored run
    printed a 84.0 ceiling while the stop gate it shares was judging against 51.2 -- the report and the
    gate disagreeing about the same run. Callers pass bytes; nobody else multiplies.
    """
    tp = max(1, int(tp_degree or 1))
    per_dev = (float(bytes_per_unit) / tp) if bytes_per_unit else 0.0
    # A NEGATIVE input is unknown, not fast. A junk dram_bw_gbps (or frac) used to divide straight
    # through to a negative ceiling, which then set a negative band and scored as BELOW_BAND -- a
    # verdict on a target that does not exist. Clamping to 0 makes score() report UNKNOWN instead.
    pk, fr = max(0.0, float(peak_bw_bytes_s or 0.0)), max(0.0, float(frac or 0.0))
    if per_dev <= 0:
        return 0.0, (0.0, 0.0)
    theo = (pk * fr) / per_dev
    return theo, (_ACHIEVABLE_BAND_LO_FRAC * theo, theo)


def _shared_bytes(mf: dict, dt) -> float:
    """Always-on MoE bytes: attention + router + shared experts + resident embeddings."""
    if mf.get("shared_tensors"):
        return sum(float(t.get("numel", 0)) * _bytes_per_elem(t.get("dtype")) for t in mf["shared_tensors"])
    if mf.get("shared_bytes") is not None:
        return float(mf["shared_bytes"])
    return float(mf.get("shared_params", 0)) * _bytes_per_elem(dt)


def compute_target(
    model_facts: dict, hw_facts: dict, *, tp_degree: int = 1, seq_len: int = 0, bytes_per_unit: float = 0.0
) -> PerfTarget:
    """MODEL-LEVEL decode ceiling, SUSTAINED (not spec peak).

        ceiling = (peak_BW * bw_fraction) / bytes_per_unit

    with bytes from the param count under the xB -> xGB rule and bw_fraction 0.80 dense / 0.50 MoE
    (see the block comment at the top of this module). Llama-3.1-8B on a 512 GB/s part:
    (512*0.8)/8 = 51.2 tok/s/u.

    Falls back to the exact per-tensor byte count ONLY when no param count is available, so facts
    written before this change still yield a ceiling instead of dropping to the weaker ms floor.
    Per-device convention: per-device bytes vs single-chip BW (never per-device bytes against
    mesh-aggregate BW — that is the 4-8x error).

    `bytes_per_unit` overrides the divisor with a caller-supplied one -- how the PINNED BASELINE byte
    count is passed in. The optimize loop reverts the model directory between attempts, so facts read
    from it describe whichever vintage is on disk; the report already divided by the ledger's write-once
    anchor while this function divided by the facts, so the report and the stop gate could judge one run
    against two ceilings. A caller holding the anchor passes it here instead of recomputing.
    """
    mf = model_facts or {}
    ab, src = 0, ""
    if bytes_per_unit and float(bytes_per_unit) > 0:
        ab = int(round(float(bytes_per_unit)))
        src = "anchored baseline bytes"
    if ab <= 0:
        ab = simple_active_bytes(mf)
        src = "params rule: %.3gB x %.2f B/param" % (ceiling_params(mf) / 1e9, _BYTES_PER_PARAM)
    if ab <= 0:
        ab = active_bytes(mf, seq_len=seq_len)
        src = "per-tensor exact bytes (no param count available)"
    frac = bw_fraction(mf)
    peak_bw = float((hw_facts or {}).get("dram_bw_gbps", 0.0)) * 1e9
    tp = max(1, int(tp_degree or 1))
    theo, band = rate_and_band(ab, peak_bw, frac=frac, tp_degree=tp)
    return PerfTarget(
        active_bytes=ab,
        peak_bw_bytes_s=peak_bw,
        theoretical_rate=theo,
        band=band,
        tp_degree=tp,
        seq_len=seq_len,
        unit=str(mf.get("unit") or "token").strip().lower(),
        bw_fraction=frac,
        bytes_source=src,
    )


def target_from_floor_ms(modeled_floor_ms: float) -> PerfTarget:
    """Fallback target from a per-profile roofline floor. Carries NO BAND, deliberately.

    The 60-80% band is a statement about DRAM BANDWIDTH: 60-80% of peak sustained against the bytes a
    token must stream. Applying those fractions to 1000/floor produces a number that looks like the
    same thing and is not -- the floor is a sum of per-op minimum times over one profiling window, so
    1000/floor is an invocations-per-second figure for that window, not a rate the hardware has a peak
    for. Doing it anyway printed "achievable (60-80%) : 671.54 - 895.38 ms" next to a 534 ms
    measurement, and it is the SAME band the optimize stop gate consults -- so a run could be told it
    had reached an achievable target that was never derived from the hardware's bandwidth at all.

    Returning (0.0, 0.0) means "no band here". score() then reports the floor ratio without a verdict,
    and the band stop cannot fire until a real ceiling exists (compute_target, from active_bytes).
    """
    theo = (1000.0 / modeled_floor_ms) if modeled_floor_ms and modeled_floor_ms > 0 else 0.0
    return PerfTarget(active_bytes=0, peak_bw_bytes_s=0.0, theoretical_rate=theo, band=(0.0, 0.0))


def score(target: PerfTarget, forward_ms: float) -> dict:
    """Compare a measured decode-step / invocation time against the target.

    bw_util = measured / theoretical_rate, the fraction of the SUSTAINED ceiling reached (the
    efficiency factor lives inside theoretical_rate now, so this is no longer a fraction of spec
    peak -- `bw_util_of_peak` is, and the two differ by exactly target.bw_fraction). Formerly: the fraction of
    the achievable ceiling reached. status: BELOW_BAND (keep optimizing) | IN_BAND (>= 60% of
    ceiling, done) | ABOVE_BAND (beat the ceiling -> active_bytes/floor suspect, assert never win)
    | UNKNOWN (no valid target or measurement)."""
    theo = target.theoretical_rate if target else 0.0
    if not theo or theo <= 0 or not forward_ms or forward_ms <= 0:
        return {"status": "UNKNOWN", "measured_tok_s": None, "bw_util": None, "theoretical_rate": theo or None}
    measured = 1000.0 / forward_ms
    bw_util = measured / theo
    lo, _hi = target.band
    if measured > theo:
        # Ordered BEFORE the no-band check on purpose, and it is NOT a stop signal -- exit_policy
        # terminates on IN_BAND only. Beating the number the target carries is a SANITY flag that the
        # floor form needs too: it is how the report separates "past the pinned BASELINE floor", which
        # an optimized build legitimately does, from "beats this build's OWN floor", which no kernel
        # does and which means a stale pairing (a 2-layer measurement against a 16-layer floor). Moving
        # the no-band check above this silently deleted that distinction.
        status = "ABOVE_BAND"
    elif not lo:
        # NO BAND to be in. A zero band means the target carries no bandwidth-derived achievable range
        # (target_from_floor_ms), and `measured >= 0` would otherwise read IN_BAND for every measurement
        # ever taken -- declaring an arbitrary run "done" against a range never computed from hardware.
        status = "NO_BAND"
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
        "theoretical_rate": round(theo, 3),
        "bw_util": round(bw_util, 4),
        # bw_util is now the fraction of the SUSTAINED ceiling reached; this is the fraction of SPEC
        # peak, which is what a bandwidth number in a report should be comparable to.
        "bw_util_of_peak": round(bw_util * target.bw_fraction, 4) if target.bw_fraction else None,
        "bw_fraction": target.bw_fraction,
        "bytes_source": target.bytes_source or None,
        "band": (round(target.band[0], 3), round(target.band[1], 3)),
        "effective_bw_bytes_s": eff_bw,
    }


def prefill_ceiling(*_a, **_k):
    raise NotImplementedError("prefill is FLOP-bound; v1 models decode only (peak_TFLOPs/model_FLOPs stub)")

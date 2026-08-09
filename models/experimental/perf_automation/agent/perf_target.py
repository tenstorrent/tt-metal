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

# ---------------------------------------------------------------------------
# THE CEILING IS PARAMS-BASED (team decision, 2026-07-29).
#
#   theoretical ceiling = (peak_BW * TP) / params_GB          <- SPEC, nothing folded in
#   achievable band     = (0.75 * f) * ceiling .. f * ceiling
#       dense: 512 / 8 GB = 64.0 tok/s/u   band 38.4 - 51.2   f = 0.80
#       MoE:   512 / 3 GB = 170.7          band 64.0 - 85.3   f = 0.50, over ACTIVE params
#   The band's TOP is what the ceiling used to report, so targets are unchanged.
#
# THE CEILING IS SPEC, NOT A TARGET. peak_bw / bytes_per_token, nothing folded in. To emit one token
# the chip must read every weight from DRAM and cannot use a weight it has not fetched, so no software
# can beat bytes/bandwidth. A ceiling that has already been discounted is not a ceiling -- a good run
# passes it, and then it says nothing. gemma3 showed exactly that: 28.7 tok/s/u reported as 84% of a
# "ceiling" of 34.1, i.e. ABOVE the achievable band, when the real wall is 512/12 = 42.7.
#
# The sustained fraction has not gone away, it has moved to the BAND, where it belongs as a statement
# about what a run can reach rather than about what the hardware permits.
#
# 1 B params -> 1 GB streamed. Deliberately an approximation: the exact on-device byte count (6.095 GB
# for Llama-3.1-8B served as bf4/bf8, 8.79 GB for gemma3-12b) is more accurate, but establishing it
# costs a per-model investigation of what width each tensor group is served at. A param count is
# dtype-independent, so it needs no such work, and xB -> xGB lands close enough to steer optimization
# -- conservatively, since a bf4-heavy model streams less than 1 B/param and so has a HIGHER real
# ceiling than this reports.
_BYTES_PER_PARAM = 1.0

# The DRAM efficiency each read pattern sustains: ~80% of spec on a dense stream, ~50% on MoE, whose
# per-token expert reads are scattered. This is the TOP of the achievable band.
_DENSE_BAND_HI = 0.80
_MOE_BAND_HI = 0.50
# The bottom of the band, as a fraction of its top. One ratio for both patterns rather than a fourth
# hand-picked constant: dense keeps its familiar 0.60-0.80 (0.60 = 0.75 x 0.80), and MoE follows the
# same shape at 0.375-0.50.
_BAND_LO_OF_HI = 0.75

# --- COMPUTE term -----------------------------------------------------------------------------------
# A weight is used in one multiply-accumulate per token, and a MAC is 2 FLOPs, so the work ONE unit of
# work costs is 2 x (params it reads) x (tokens in that unit). Model-agnostic: it needs the same param
# count the bandwidth term already has, and no architecture formulas.
#
# tokens_per_unit is 1 for a decode step and the sequence length for a prefill -- which is exactly why
# prefill is compute-bound and decode is not: the bytes stay put while the FLOPs scale with the
# sequence. At batch 1 decode the arithmetic intensity is ~2 FLOP/byte against a part that wants ~100,
# so the bandwidth term dominates by a wide margin and this one never binds. It exists so that prefill,
# diffusion at resolution and batched encoders are bounded by the constraint that actually binds THEM.
_FLOPS_PER_PARAM_PER_TOKEN = 2.0
# Fraction of matrix-engine peak a real kernel sustains. Deliberately generous (the compute term is a
# ceiling, not a prediction) and separate from the DRAM fraction, which describes a different unit.
_COMPUTE_PEAK_FRACTION = 0.80


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
    # DATA parallelism. Never in theoretical_rate -- replicas do not make one unit of work faster.
    # Kept so a report can state system throughput (aggregate_rate) beside the per-unit ceiling.
    dp_degree: int = 1
    # Which constraint set the ceiling: "memory" (bytes / BW) or "compute" (FLOPs / peak). The standard
    # roofline takes the lower rate; naming the winner is what routes the lever class.
    bound_by: str = "memory"
    # Tokens inside one unit of work: 1 for a decode step, the sequence length for a prefill. The bytes
    # are flat in this, the FLOPs are linear -- which is the whole reason prefill is compute-bound.
    tokens_per_unit: int = 1
    # theoretical_rate * dp: total units/s the mesh can retire. NOT the number the band scores.
    aggregate_rate: float = 0.0


def active_bytes(model_facts: dict, *, regime: str = "decode", seq_len: int = 0, batch: int = 1) -> int:
    """Bytes streamed from DRAM per unit of work, summed per-tensor at each tensor's real dtype.

    `regime` selects the unit: "decode" is one token, "prefill" is one request of `seq_len` tokens.
    Both stream the whole weight set exactly once -- that is why a decode token and a prefill request
    share a floor -- and prefill adds the KV it writes plus its activations, both linear in seq_len.

    Dense: Σ tensor_bytes(all weight tensors). MoE: shared_bytes + top_k * per_expert_bytes
    (the reachable read set — NOT all experts). Optional KV term when seq_len>0.

    BATCH SCALES THE KV TERM AND NOTHING ELSE, which is the whole reason batching pays: the weights
    are read ONCE and amortised across every user in the step, while each user carries their own
    KV history and reads all of it. So doubling the batch does not double the bytes -- it adds one
    more KV history to a fixed weight cost, and the per-user ceiling falls only by that much.

    Omitting the factor made batch free: an 8-user step was costed as a 1-user step, the ceiling came
    out too high, and every at-floor verdict computed against it inherited the error."""
    if regime not in ("decode", "prefill"):
        raise NotImplementedError("regimes: decode | prefill (got %r)" % (regime,))
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
        kv *= max(1, int(batch or 1))

    # PREFILL MOVES MORE THAN THE WEIGHTS. Refusing the regime outright meant its caller had nothing
    # to use and fell back to the DECODE read set, so a report printed the same memory ceiling for
    # both stages -- 21.84 ms each -- which reads as physics and is really one number used twice.
    #
    # Same weights (read once, whatever the prompt length), plus two terms that scale with the
    # prompt: the KV it WRITES for every token, and activations, which are the per-token hidden and
    # intermediate widths carried through each layer. Written from the same facts the decode path
    # uses; no new inputs, no per-model table.
    act = 0.0
    if regime == "prefill":
        kv *= 2.0  # written on the way in, then read back by attention
        _n, _h = _scalar(mf.get("layers", 0), 0), _scalar(mf.get("hidden_size", 0), 0)
        _i = _scalar(mf.get("intermediate_size", 0), 0) or (4 * _h)
        if seq_len and _n and _h:
            a_dt = mf.get("dominant_dtype") or mf.get("torch_dtype") or "bfloat16"
            # per layer: the residual stream in and out, and the MLP intermediate in and out
            act = float(_n) * (2.0 * _h + 2.0 * _i) * int(seq_len) * _bytes_per_elem(a_dt)
            act *= max(1, int(batch or 1))

    total = wb + kv + act
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
    """Fraction of peak DRAM bandwidth this model's read pattern sustains -- the BAND's top.

    Kept under its old name because callers persist it as `bw_fraction` in the throughput snapshot
    and the report reads it back. What changed is where it applies: it used to be multiplied into the
    ceiling, so `theoretical_rate` was already a sustained figure; now the ceiling is spec and this
    sets the top of the achievable band.
    """
    return _MOE_BAND_HI if (model_facts or {}).get("is_moe") else _DENSE_BAND_HI


def rate_and_band(bytes_per_unit: float, peak_bw_bytes_s: float, *, frac: float = _DENSE_BAND_HI, tp_degree: int = 1):
    """(ceiling, (band_lo, band_hi)) from an ALREADY-KNOWN byte count.

    The one place the ceiling arithmetic lives. Exists because a second caller (the report renderer,
    recomputing from the ledger's byte anchor) had its own copy: `peak / bytes` with a hardcoded
    (0.60, 0.80) band. That copy silently kept the pre-sustained-fraction physics, so an anchored run
    printed a 84.0 ceiling while the stop gate it shares was judging against 51.2 -- the report and the
    gate disagreeing about the same run. Callers pass bytes; nobody else multiplies.

    THE CEILING IS SPEC: peak / bytes, with `frac` no longer multiplied into it. The fraction now sets
    the band's TOP, so the band spans frac*0.75 .. frac of the ceiling -- 0.60-0.80 dense, exactly the
    range the report has always printed, and 0.375-0.50 for MoE. Folding it into the ceiling made a
    "ceiling" a run could pass: gemma3 read 84% of 34.1 and sat ABOVE its own achievable band, when
    the wall is 512/12 = 42.7 and 28.7 is 67% of it -- inside the band, with headroom left to chase.
    """
    tp = max(1, int(tp_degree or 1))
    per_dev = (float(bytes_per_unit) / tp) if bytes_per_unit else 0.0
    # A NEGATIVE input is unknown, not fast. A junk dram_bw_gbps (or frac) used to divide straight
    # through to a negative ceiling, which then set a negative band and scored as BELOW_BAND -- a
    # verdict on a target that does not exist. Clamping to 0 makes score() report UNKNOWN instead.
    pk, fr = max(0.0, float(peak_bw_bytes_s or 0.0)), max(0.0, float(frac or 0.0))
    if per_dev <= 0:
        return 0.0, (0.0, 0.0)
    # SPEC CEILING, then the band. peak / bytes is the wall; `fr` is what the read pattern sustains
    # and so sets the band's top, with the bottom a fixed ratio below it.
    theo = pk / per_dev
    hi = fr * theo
    return theo, (_BAND_LO_OF_HI * hi, hi)


def chip_peak_flops(hw_facts: dict, fidelity: str = "") -> float:
    """Per-CHIP matrix-engine peak FLOP/s, or 0.0 when the arch facts carry no peak table.

    Per-chip for the same reason the bandwidth term is per-chip: the bytes and the FLOPs of one unit of
    work are both per-device, so pairing either with a mesh-aggregate figure applies the chip count
    twice. `worker_cores` in the env is already multiplied by mesh_chips, so it is divided back out.
    """
    hw = hw_facts or {}
    peaks = hw.get("peak_tflops_per_core") or {}
    if not peaks:
        return 0.0
    per_core = peaks.get(str(fidelity or "").strip().lower()) or peaks.get("hifi4")
    if not per_core:
        return 0.0
    cores = _scalar(hw.get("worker_cores", 0), 0)
    chips = max(1, _scalar(hw.get("mesh_chips", 1), 1))
    if cores <= 0:
        gx, gy = _scalar(hw.get("grid_x", 0), 0), _scalar(hw.get("grid_y", 0), 0)
        cores = gx * gy * chips
    per_chip_cores = max(1.0, float(cores) / float(chips))
    return float(per_core) * 1e12 * per_chip_cores


def compute_ceiling(model_facts: dict, hw_facts: dict, *, tp_degree: int = 1, tokens_per_unit: int = 1) -> float:
    """Per-unit rate ceiling from COMPUTE: peak_FLOPs / FLOPs_per_unit. 0.0 when unknown.

    FLOPs_per_unit = 2 * params_read * tokens_per_unit, sharded across TP the same way the bytes are.
    """
    params = ceiling_params(model_facts)
    toks = max(1, _scalar(tokens_per_unit, 1))
    tp = max(1, int(tp_degree or 1))
    if params <= 0:
        return 0.0
    flops_per_unit = _FLOPS_PER_PARAM_PER_TOKEN * float(params) * float(toks) / tp
    peak = chip_peak_flops(hw_facts, str((model_facts or {}).get("fidelity") or ""))
    if peak <= 0 or flops_per_unit <= 0:
        return 0.0
    return (peak * _COMPUTE_PEAK_FRACTION) / flops_per_unit


def _shared_bytes(mf: dict, dt) -> float:
    """Always-on MoE bytes: attention + router + shared experts + resident embeddings."""
    if mf.get("shared_tensors"):
        return sum(float(t.get("numel", 0)) * _bytes_per_elem(t.get("dtype")) for t in mf["shared_tensors"])
    if mf.get("shared_bytes") is not None:
        return float(mf["shared_bytes"])
    return float(mf.get("shared_params", 0)) * _bytes_per_elem(dt)


def measured_bytes_per_unit(profile: dict) -> int:
    """Bytes the PROFILED ops actually read, summed. 0 when the profile carries none.

    The params rule cannot express three shapes, and each fails in a way no byte-width estimate fixes:
      * MULTI-TOWER -- a token reads the language backbone, not the audio encoder or the vocoder, but the
        param count is over the whole checkpoint;
      * CONV-HEAVY -- a weight is reused across every spatial position, so bytes and FLOPs are not
        proportional to params at all;
      * ROUTED with no published active count -- only the selected experts stream.

    Summing what the profile RECORDED sidesteps all three, because a tower that did not run has no ops to
    sum: it stops predicting the read set and adds up the one observed. Deliberately NOT the default --
    the params rule is the agreed number for dense and MoE LLMs, and this must never quietly override it.

    Partial coverage understates the total, which yields a ceiling that is too HIGH -- the safe direction,
    since a run then keeps optimizing rather than stopping early. It is still not evidence for ending a
    run; that is what the arming check is for.
    """
    total = 0.0
    for b in (profile or {}).get("buckets") or []:
        if not isinstance(b, dict) or b.get("id") == "host_overhead":
            continue
        for op in b.get("top_ops") or []:
            try:
                v = float((op or {}).get("bytes") or 0.0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v) and v > 0:
                total += v
    return int(round(total)) if total > 0 else 0


def params_rule_expresses(model_facts: dict) -> bool:
    """Is the xB -> xGB rule a fair statement of THIS model's read set?

    False for a routed model with no active count (total params overstate it) and for a model the caller
    has flagged as multi-tower or conv-shaped. Nothing here inspects an architecture by name.
    """
    mf = model_facts or {}
    if mf.get("is_moe") and not _scalar(mf.get("active_params", 0), 0):
        return False
    if mf.get("multi_tower") or mf.get("weight_reuse_per_unit"):
        return False
    return True


def compute_target(
    model_facts: dict,
    hw_facts: dict,
    *,
    tp_degree: int = 1,
    dp_degree: int = 1,
    seq_len: int = 0,
    bytes_per_unit: float = 0.0,
    tokens_per_unit: int = 1,
    profile: dict | None = None,
) -> PerfTarget:
    """MODEL-LEVEL per-unit ceiling: the LOWER of the bandwidth and compute bounds.

        memory  : (peak_BW_per_chip * bw_fraction * TP) / bytes_per_unit
        compute : (peak_FLOPs_per_chip * 0.80 * TP) / (2 * params * tokens_per_unit)
        ceiling : min(the two)          -- the binding constraint, `bound_by` names it

    TP scales both (weights and work shard together). DP and PP scale NEITHER -- they multiply how many
    units run at once, not how fast one is, so they appear only in `aggregate_rate`. Model-agnostic: no
    architecture formulas, only a param count, a unit, and the mesh split.

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
    # MEASURED BYTES, only where the params rule cannot express the read set (see params_rule_expresses).
    # Never an override of the agreed dense/MoE number -- it is consulted before the params fallback ONLY
    # for the shapes the rule misstates, and after the pinned anchor either way.
    if not (bytes_per_unit and float(bytes_per_unit) > 0) and not params_rule_expresses(mf):
        _mb = measured_bytes_per_unit(profile or {})
        if _mb > 0:
            ab, src = _mb, "measured per-op bytes (params rule cannot express this read set)"
    if bytes_per_unit and float(bytes_per_unit) > 0:
        ab = int(round(float(bytes_per_unit)))
        src = "anchored baseline bytes"
    # THE CENSUS OUTRANKS EVERY RULE, because it is not a rule. agent/weight_census walks the BUILT
    # model and sums each resident tensor's element count at its REAL dtype -- the only place the
    # served width exists, since the checkpoint records what was on disk and not what the loader
    # decided. Every other branch here predicts that width: params x 1.0 published voxtral at
    # 141.8 tok/s/u against a true 54.7, and the checkpoint's own byte count gives gemma-3 a 21.0
    # ceiling for a model measuring 30.8. An INCOMPLETE census (a dtype the census has no width for)
    # is refused rather than used as a lower bound: too few bytes reads as too HIGH a ceiling, which
    # is the direction that ends a run early believing it is at the wall.
    if ab <= 0:
        _dwb = _scalar(mf.get("device_weight_bytes", 0), 0)
        if _dwb > 0 and mf.get("device_census_complete", True):
            ab = int(_dwb)
            src = "device census: %.3g GB resident at served dtype" % (_dwb / 1e9)
    if ab <= 0:
        ab = simple_active_bytes(mf)
        src = "params rule: %.3gB x %.2f B/param" % (ceiling_params(mf) / 1e9, _BYTES_PER_PARAM)
    if ab <= 0:
        ab = active_bytes(mf, seq_len=seq_len)
        src = "per-tensor exact bytes (no param count available)"
    frac = bw_fraction(mf)
    # PER-CHIP BANDWIDTH, NOT MESH-AGGREGATE. The bytes are already per-device (ab / tp), so pairing
    # them with mesh-aggregate bandwidth applies the chip count TWICE:
    #     today     (per_chip * chips) / (B / TP) = per_chip * chips * TP / B
    #     physics    per_chip * TP / B            -- each of TP chips streams B/TP at per_chip
    # The ratio is `chips`, so EVERY mesh run was inflated by its chip count -- 4x for a TP=4 model on
    # 4 chips, 8x for a replicated model on 8 (decide_parallelism's single-chip route returns
    # tp=1, dp=total_chips, which is exactly the case environment.py flags its aggregate as invalid
    # for). Only a 1-chip run was correct, which is why nothing caught it.
    #
    # DP and PP are deliberately absent: replicas and pipeline stages do not reduce the bytes ONE unit
    # of work streams, so they raise aggregate throughput, never the per-unit ceiling this scores.
    hw = hw_facts or {}
    peak_bw = float(hw.get("dram_bw_per_chip_gbps") or hw.get("dram_bw_gbps", 0.0)) * 1e9
    tp = max(1, int(tp_degree or 1))
    theo, band = rate_and_band(ab, peak_bw, frac=frac, tp_degree=tp)
    bound = "memory"
    # THE BINDING CONSTRAINT WINS. Bandwidth and compute are both ceilings, so the real one is the
    # LOWER rate -- the standard roofline. Decode at batch 1 is memory-bound by a wide margin, so this
    # changes nothing there; it is what bounds prefill, diffusion at resolution and batched encoders,
    # which were previously handed a bandwidth number they could never be limited by.
    _toks = max(1, _scalar(tokens_per_unit, 1))
    theo_c = compute_ceiling(mf, hw, tp_degree=tp, tokens_per_unit=_toks)
    if theo_c > 0 and (theo <= 0 or theo_c < theo):
        theo, bound = theo_c, "compute"
        _hi = frac * theo
        band = (_BAND_LO_OF_HI * _hi, _hi)
    # DP AND PP ARE NOT IN THE CEILING. Replicas and pipeline stages do not reduce what ONE unit of work
    # reads or computes, so they cannot make one token faster -- they multiply how many run at once.
    # Folding them in is what inflated every mesh run by its chip count. Carried separately so a report
    # can state total system throughput without ever contaminating the per-unit target the gate scores.
    dp = max(1, _scalar(dp_degree, 1))
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
        dp_degree=dp,
        bound_by=bound,
        tokens_per_unit=_toks,
        aggregate_rate=theo * dp,
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

    bw_util = measured / theoretical_rate, the fraction of the SPEC ceiling reached. The sustained
    fraction sets the band's top rather than being folded into the ceiling, so this IS already a
    fraction of spec bandwidth and cannot exceed 1.0 for a real measurement; `bw_util_of_peak` is
    kept as an alias of it for readers that ask for the peak-relative number by name. status: BELOW_BAND (keep optimizing) | IN_BAND (>= 60% of
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
        # the ceiling IS spec, so bw_util is already peak-relative; the alias stays for callers.
        "bw_util_of_peak": round(bw_util, 4),
        "bw_fraction": target.bw_fraction,
        "bytes_source": target.bytes_source or None,
        "band": (round(target.band[0], 3), round(target.band[1], 3)),
        "effective_bw_bytes_s": eff_bw,
    }


def prefill_ceiling(*_a, **_k):
    raise NotImplementedError("prefill is FLOP-bound; v1 models decode only (peak_TFLOPs/model_FLOPs stub)")

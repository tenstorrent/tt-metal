# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The full-model evidence harness, re-run for the optimized stage.

This is deliberately **not** a new harness.  ``$optimize`` asks for before/after
in the same regime, so this loads ``doc/full_model/bench/evidence.py`` by path and
changes exactly three things:

* artifacts land under ``doc/optimized_full_model/`` instead of ``doc/full_model/``;
* the layer-stack lower bound is recomputed from *this* stage's per-layer traced
  decode, measured with the decoder stage's own ``layer_ab.py`` on the shipped
  default (``logs/layer_ab_after.log``), because the SwiGLU reshard changed it;
* ``--baseline`` reverts the three decode-path changes this stage ships, so the
  ``before`` row comes out of the same script, the same host and the same code.

Everything else -- the readiness runners, the accuracy gates, the split-sampling
contract assertions, the prompt-shape sweep, the fallback audit, the perf windows
-- is the full-model stage's code, unmodified.

Usage::

    python doc/optimized_full_model/bench/evidence.py --stages perf --out evidence_perf.json
    python doc/optimized_full_model/bench/evidence.py --stages perf --baseline --out evidence_perf_before.json
    python doc/optimized_full_model/bench/evidence.py --stages perf --prefill-trace --out evidence_perf_prefill_trace.json
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

#: Per-layer traced decode at context 2048 on the shipped optimized default,
#: measured by ``doc/optimized_multichip_decoder/bench/layer_ab.py --candidates
#: tp4,tp4b --decode-context 2048`` (``logs/layer_ab_after.log``).  The full-model
#: stage's floor used 0.4546 / 0.4238; the SwiGLU multiply reshard moved both.
SLIDING_MS_PER_LAYER = 0.4473
FULL_MS_PER_LAYER = 0.4164
SLIDING_LAYERS = 39
FULL_LAYERS = 13

#: The same measurement on the *pre-stage* default, from the optimized multichip
#: decoder stage's README.  Used for the ``--baseline`` arm so its floor is the
#: floor that arm's layers actually have.
BASELINE_SLIDING_MS_PER_LAYER = 0.4546
BASELINE_FULL_MS_PER_LAYER = 0.4238


def _load():
    path = ROOT / "doc/full_model/bench/evidence.py"
    spec = importlib.util.spec_from_file_location("muse_glimmer_full_model_evidence", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ev = _load()
    ev.OUT = ROOT / "doc/optimized_full_model"
    (ev.OUT / "logs").mkdir(parents=True, exist_ok=True)

    prefill_trace = "--prefill-trace" in sys.argv
    if prefill_trace:
        sys.argv.remove("--prefill-trace")
        original_build = ev.build_generator

        def build_with_prefill_trace(*a, **k):
            k.setdefault("prefill_trace", True)
            return original_build(*a, **k)

        ev.build_generator = build_with_prefill_trace
        ev.say("EV prefill_trace=True: the opt-in traced prefill is on for this run")

    baseline = "--baseline" in sys.argv
    if baseline:
        sys.argv.remove("--baseline")
        from models.autoports.meta_models_muse_glimmer_30b.tt import model as model_mod
        from models.autoports.meta_models_muse_glimmer_30b.tt import optimized_decoder as dec_mod

        model_mod.LM_HEAD_SOFTCAP_IN_L1 = False
        model_mod.EMBED_DECODE_GATHER_SHARDED = False
        dec_mod.DECODE_SWIGLU_MUL_CORES = None
        ev.say("EV baseline: the full-model stage's decode path (softcap in DRAM, gather to DRAM, no SwiGLU reshard)")

    original = ev.stage_perf

    def stage_perf(generator, summary, args):
        original(generator, summary, args)
        out = summary["performance"]
        sliding = BASELINE_SLIDING_MS_PER_LAYER if baseline else SLIDING_MS_PER_LAYER
        full = BASELINE_FULL_MS_PER_LAYER if baseline else FULL_MS_PER_LAYER
        total = SLIDING_LAYERS * sliding + FULL_LAYERS * full
        out["layer_stack_lower_bound_ms_per_token"] = {
            "sliding_layers": SLIDING_LAYERS,
            "full_layers": FULL_LAYERS,
            "sliding_ms_per_layer": sliding,
            "full_ms_per_layer": full,
            "total_ms": total,
            "tok_s_u": 1e3 / total,
            "source": (
                (
                    "the optimized multichip decoder stage's README, i.e. the *pre-stage* "
                    "per-layer decode, hardcoded here as BASELINE_SLIDING/FULL_MS_PER_LAYER "
                    "so the --baseline arm's floor is the floor its own layers have. Not "
                    "re-measured by this run and not from layer_ab_after.log, which contains "
                    "only the after-arms."
                )
                if baseline
                else (
                    "doc/optimized_multichip_decoder/bench/layer_ab.py --candidates tp4,tp4b "
                    "--decode-context 2048, on this stage's shipped default "
                    "(doc/optimized_full_model/logs/layer_ab_after.log), hardcoded here as "
                    "SLIDING/FULL_MS_PER_LAYER rather than re-measured by this run"
                )
            ),
        }
        out["baseline_arm"] = baseline
        out["prefill_trace_arm"] = prefill_trace
        traces = getattr(generator, "_prefill_traces", {})
        out["prefill_trace_buckets"] = sorted(traces)
        ev.say(f"EV perf layer-stack lower bound = {total:.3f} ms/token ({1e3/total:.2f} t/s/u)")

    ev.stage_perf = stage_perf
    return ev.main()


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The post-selection evidence harness, on the selected precision config.

Deliberately **not** a new harness.  This loads
``doc/full_model/bench/evidence.py`` by path -- the same code the full-model and
optimized-full-model stages ran -- and changes two things:

* artifacts land under ``doc/datatype_sweep/``;
* the layer-stack lower bound is *not* silently inherited.  The parent hardcodes
  the full-model stage's per-layer figures; those were measured on the
  carried-forward precision policy, so they are only a floor for a selection that
  keeps that policy.  This wrapper reads
  ``doc/datatype_sweep/sweep_results.json``, and stamps the floor with whether
  the selected config's decoder-layer dtype/fidelity policy is the one the floor
  was measured under, rather than presenting a stale number as a bound.

Nothing else moves: the readiness runners, the accuracy gates, the split-sampling
assertions, the non-aligned prompt-shape sweep, the fallback audit and the perf
windows are the full-model stage's code.  The perf stage in particular is the
**same warmed token-out no-readback benchmark** the optimized full model reported,
which is what ``$datatype-sweep`` asks for post-selection.

The generator is built by ``build_generator(model_dir, mesh_device)`` with no
precision knobs, so what it measures is whatever
``doc/datatype_sweep/selected_precision_config.json`` selects.

Usage::

    python doc/datatype_sweep/bench/evidence.py --stages capacity,prefill,teacher,shapes,fallback \\
        --out evidence_accuracy.json
    python doc/datatype_sweep/bench/evidence.py --stages perf --out evidence_perf.json
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

#: The optimized-full-model stage's per-layer traced decode at context 2048, on
#: its shipped precision policy (``doc/optimized_full_model/logs/layer_ab_after.log``,
#: ``layer_ab.py --candidates tp4,tp4b --decode-context 2048``).
SLIDING_MS_PER_LAYER = 0.4473
FULL_MS_PER_LAYER = 0.4164
SLIDING_LAYERS = 39
FULL_LAYERS = 13
#: The decoder-layer policy those figures were measured under.  If the selected
#: config matches it, the floor is this selection's floor; if not, the floor is
#: labelled as belonging to a different policy rather than quoted as a bound.
FLOOR_POLICY = {
    "weight_attn": "bfloat8_b",
    "weight_mlp_gate_up": "bfloat4_b",
    "weight_mlp_down": "bfloat4_b",
    "kv_cache_dtype": "bfloat8_b",
    "activation_dtype": "bfloat16",
    "ccl_prefill_dtype": "bfloat8_b",
    "ccl_decode_dtype": "bfloat16",
    "decode_fidelity": "LoFi",
    "prefill_fidelity": "LoFi",
    "layer_exceptions": "",
}


def _load():
    path = ROOT / "doc/full_model/bench/evidence.py"
    spec = importlib.util.spec_from_file_location("muse_glimmer_full_model_evidence", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selected_policy() -> tuple[str, dict | None]:
    """``(config_id, flattened policy)`` of the selection, from the sweep tables."""
    from models.autoports.meta_models_muse_glimmer_30b.tt import precision_config as pc

    config_id = str(pc.load_precision_config()["config_id"])
    results_path = ROOT / "doc/datatype_sweep/sweep_results.json"
    if not results_path.is_file():
        return config_id, None
    results = json.loads(results_path.read_text())
    for row in results["configs"]:
        if row["config_id"] == config_id:
            return config_id, row["policy"]
    return config_id, None


def main() -> int:
    ev = _load()
    ev.OUT = ROOT / "doc/datatype_sweep"
    (ev.OUT / "logs").mkdir(parents=True, exist_ok=True)

    config_id, policy = selected_policy()
    matches_floor = policy is not None and all(policy.get(k) == v for k, v in FLOOR_POLICY.items())
    ev.say(f"EV selected precision config = {config_id}")

    original = ev.stage_perf

    def stage_perf(generator, summary, args):
        original(generator, summary, args)
        out = summary["performance"]
        total = SLIDING_LAYERS * SLIDING_MS_PER_LAYER + FULL_LAYERS * FULL_MS_PER_LAYER
        out["layer_stack_lower_bound_ms_per_token"] = {
            "sliding_layers": SLIDING_LAYERS,
            "full_layers": FULL_LAYERS,
            "sliding_ms_per_layer": SLIDING_MS_PER_LAYER,
            "full_ms_per_layer": FULL_MS_PER_LAYER,
            "total_ms": total,
            "tok_s_u": 1e3 / total,
            "measured_under_the_selected_policy": matches_floor,
            "source": (
                "doc/optimized_multichip_decoder/bench/layer_ab.py --candidates tp4,tp4b "
                "--decode-context 2048 on the optimized-full-model stage's shipped policy "
                "(doc/optimized_full_model/logs/layer_ab_after.log), not re-measured by this run. "
                + (
                    "The selected config's decoder-layer dtype/fidelity policy is that policy, so "
                    "this is a floor for this selection too."
                    if matches_floor
                    else "The selected config's decoder-layer policy DIFFERS from the one the floor "
                    "was measured under, so this figure is a comparator for the previous policy "
                    "and not a bound on this one."
                )
            ),
        }
        out["precision_config_id"] = config_id
        ev.say(
            f"EV perf layer-stack floor = {total:.3f} ms/token " f"(measured_under_the_selected_policy={matches_floor})"
        )

    ev.stage_perf = stage_perf

    original_capacity = ev.stage_capacity

    def stage_capacity(generator, summary):
        original_capacity(generator, summary)
        summary["capacity"]["precision_config_id"] = config_id

    ev.stage_capacity = stage_capacity
    return ev.main()


if __name__ == "__main__":
    raise SystemExit(main())

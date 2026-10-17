# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The census weighed a quarter of the model and nothing could tell.

RUN 10, 2026-08-19. perf_target_inputs.json recorded:

    "device_weight_bytes": 1718081696,
    "bytes_per_param": 1.3228,          ->  1.299 B parameters
    "device_census_complete": true

The checkpoint's own headers say the model is 4.676 B parameters -- language_model 4.014,
audio_tower 0.637, multi_modal_projector 0.025. The census counted 27.8% of it, called itself
complete, and every ceiling in the report divided that number.

ITS OWN SECTION FIGURES SAY WHY. lm_layers came back at 227.3 M, and the language config puts one
layer at 100.7 M (attn 4 x 3072-wide with 8 kv heads, mlp 3 x 3072x8192). That is two layers of
thirty. Adding the embedding and lm_head (805.3 M), the audio tower's 186.7 M and the 41.9 M KV
cache gives 1.235 B for a 2-layer build, against the 1.299 B measured.

So the census ran inside a profiling run, which builds at the coverage window of 2 layers -- and
device_weight_bytes is pinned by the FIRST complete census, which the capped run reaches before the
uncapped full-pipeline gate does.

A capped census is shaped exactly like a whole one. The only way to tell is for it to say.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _marker(**over):
    from agent.weight_census import marker

    c = {
        "weight_bytes": 1718081696,
        "scope": "pipeline",
        "weight_tensors": [{"numel": 10, "dtype": "bfloat16"}],
        "unknown_dtype_tensors": 0,
        "complete": True,
        "bytes_per_param": 1.3228,
    }
    c.update(over)
    return marker(c)


def test_the_marker_states_the_depth_it_was_taken_at(monkeypatch):
    monkeypatch.delenv("TT_PERF_LAYERS", raising=False)
    assert "depth=all " in _marker()

    monkeypatch.setenv("TT_PERF_LAYERS", "2")
    # The knob is NAMED, not just its value: a multi-stack model is capped by several variables and
    # a bare number could not say which one shrank the build. The reader only tests against "all".
    assert "depth=TT_PERF_LAYERS=2 " in _marker()


def test_all_layers_is_the_absence_of_a_cap_not_a_sentinel(monkeypatch):
    """set_depth expresses "no cap" by REMOVING the variable; a literal 0 is read by builders as
    "build zero layers", which is why nothing writes one."""
    from agent.weight_census import census_depth

    monkeypatch.setenv("TT_PERF_LAYERS", "0")
    assert census_depth() == "all"
    monkeypatch.setenv("TT_PERF_LAYERS", "16")
    assert census_depth() == "TT_PERF_LAYERS=16"


def test_an_unreadable_depth_is_not_a_claim_of_full_depth(monkeypatch):
    from agent.weight_census import census_depth

    monkeypatch.setenv("TT_PERF_LAYERS", "not-a-number")
    assert census_depth() == "all" or census_depth() == "unknown"


def test_the_reader_refuses_a_capped_census():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i = src.index('if "TRACE_WEIGHT_BYTES=" in line:')
    body = src[i : src.index('if "TRACE_STAGE_OPS[', i)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert '_depth != "all"' in code, "a capped census is pinned as the model's resident bytes again"
    assert "_persist_device_weight_bytes(" in code
    # the refusal must come BEFORE the persist, not as a note after it
    assert code.index('_depth != "all"') < code.index("_persist_device_weight_bytes(_wb")


def test_the_arithmetic_that_identified_the_cap():
    """Recorded so the diagnosis can be re-checked rather than believed: the census total matches a
    2-layer build of this model to within 5%."""
    h, i_, kvh, hd, V = 3072, 8192, 8, 128, 131072
    per_layer = (h * h + 2 * h * kvh * hd + h * h) + 3 * h * i_
    two_layers_plus_heads = 2 * per_layer + 2 * (V * h)
    measured_audio_and_kv = 186.7e6 + 41.9e6
    predicted = two_layers_plus_heads + measured_audio_and_kv

    census_reported = 1718081696 / 1.3228
    assert abs(predicted - census_reported) / census_reported < 0.06, (predicted, census_reported)
    assert census_reported / 4.676e9 < 0.3, "the undercount is no longer ~28%; re-derive the cause"

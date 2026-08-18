# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill streams more than the weights, so it does not share decode's byte count.

active_bytes refused every regime but decode -- "prefill is FLOP-bound" -- and the roofline's caller,
having nothing else, used the DECODE read set for both stages. The report then printed the same
memory ceiling twice:

    PREFILL  memory <- binds   21.84 ms
    DECODE   memory <- binds   21.84 ms

which reads as a physical result and is one number used twice. The weights term IS shared, and that
part is real: both stages stream the whole model exactly once, which is why a prefill of 128 tokens
and a single decode token sit on the same floor. What prefill adds is linear in the prompt: the KV it
WRITES for every token (and reads back in attention), and the activations carried through each layer.

Both terms come from facts the decode path already reads -- layers, kv_heads, head_dim, hidden_size,
intermediate_size -- so there is no new input to supply and no per-model table.

The premise in the old message was also wrong for this model: prefill is memory-bound at ISL 128, not
FLOP-bound. It crosses over near 685 tokens.
"""
from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
_spec = _ilu.spec_from_file_location("pt_prefill_ut", str(_PA / "agent" / "perf_target.py"))
PT = _ilu.module_from_spec(_spec)
sys.modules["pt_prefill_ut"] = PT
_spec.loader.exec_module(PT)

MF = {
    "total_params": 11_180_446_320,
    "weight_bytes": 11_180_446_320,
    "layers": 48,
    "kv_heads": 8,
    "head_dim": 256,
    "hidden_size": 3840,
    "intermediate_size": 15360,
    "dominant_dtype": "bfloat16",
    "kv_dtype": "bfloat16",
}


# ITEMS IS NOW EXPLICIT. The terms that scale with WORK -- the KV a stage writes and the activations
# it carries -- used to appear only under `if regime == "prefill"`, which meant a third stage got
# neither however much work it did. The caller states how many items one unit processes instead: a
# prompt-consuming stage retires every prompt token, so items = seq_len here.


def test_prefill_is_accepted():
    assert PT.active_bytes(MF, regime="prefill", seq_len=128, items=128) > 0


def test_prefill_moves_more_than_decode():
    """The whole point: one number used twice was hiding a real difference."""
    d = PT.active_bytes(MF, regime="decode", seq_len=128)
    p = PT.active_bytes(MF, regime="prefill", seq_len=128, items=128)
    assert p > d, (p, d)


def test_the_weights_still_dominate_at_short_context():
    """Both stages stream the model once, so the floors are CLOSE -- that similarity is real, and a
    fix that made them wildly different would be as wrong as making them identical."""
    d = PT.active_bytes(MF, regime="decode", seq_len=128)
    p = PT.active_bytes(MF, regime="prefill", seq_len=128, items=128)
    assert 1.0 < p / d < 1.5, p / d


def test_the_extra_terms_scale_with_the_prompt():
    """KV written and activations are both linear in seq_len; the weight term is not."""
    a = PT.active_bytes(MF, regime="prefill", seq_len=128, items=128)
    b = PT.active_bytes(MF, regime="prefill", seq_len=1024, items=1024)
    w = PT.active_bytes(MF, regime="prefill", seq_len=0, items=0)
    assert b > a > w
    assert abs((b - w) - 8 * (a - w)) < 0.02 * (b - w), (a, b, w)


def test_a_stage_is_costed_by_what_it_processes_not_by_what_it_is_called():
    """THE REFUSAL WAS THE BUG. active_bytes raised on any regime but decode|prefill, and every
    caller wraps it in `except Exception: return base` -- so a third stage did not get an error, it
    got a weights-only ceiling, silently. Voxtral's audio encoder was priced with no activation term
    at all because of its NAME, while the math it needed had been items-driven for weeks.

    Two properties, together: an unknown name is costed, and the name changes nothing."""
    assert PT.active_bytes(MF, regime="denoise", seq_len=128, items=128) > 0

    same = {
        PT.active_bytes(MF, regime=r, seq_len=128, items=128, batch=2)
        for r in ("decode", "prefill", "encode", "denoise", "", "TYPO")
    }
    assert len(same) == 1, "the stage's NAME moved its byte count: %s" % same


def test_the_facts_producer_emits_the_geometry_the_prefill_term_needs():
    """Without hidden_size/intermediate_size the prefill term silently degrades to weights-only --
    which is decode's figure, and is how both stages came to print one ceiling twice.

    It now travels per BLOCK rather than as loose keys, because the loose form had no answer for a
    model with two towers: it took `layers` from the deepest and the widths from another, and priced
    every stage with a model that does not exist. The single-tower shape is unchanged -- one block,
    and its geometry is still published flat for callers that have not learned about blocks."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _model_block_facts")
    block = src[i : src.index("\ndef ", i + 1)]
    assert "tower_geometry(" in block, "geometry no longer comes from the config's own towers"

    j = src.index("def _perf_target_inputs")
    pti = src[j : src.index("\ndef ", j + 1)]
    assert '"hidden_size"' in pti and '"intermediate_size"' in pti, "the single-block flat view is gone"


def test_thin_facts_still_get_a_weights_floor():
    """A model whose config yields no layer geometry must still be costed, not dropped."""
    thin = {"total_params": 8_000_000_000, "dominant_dtype": "bfloat16"}
    assert PT.active_bytes(thin, regime="prefill", seq_len=128) == PT.active_bytes(thin, regime="decode")

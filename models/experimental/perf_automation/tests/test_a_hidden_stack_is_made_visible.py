# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A stack the model runs and the walk cannot see is repaired, not silently profiled at full depth.

THE WALK'S RULE IS NARROWER THAN REALITY. find_all_stacks counts a list as one stack when its
elements share a class, or when their classes share a base. A pipeline that wraps each layer in a
DIFFERENT class -- a counting proxy here, a parts-assembled layer there -- holds a real stack that
reads as unrelated objects. Measured on Voxtral-Mini-3B: the walk returns four stacks, of which the
device side contributes exactly one, while the model runs three sections. One coverage number, one
knob, one stack capped, the rest at full depth, and no error at any step.

WIDENING THE RULE BY INFERENCE FAILED TWICE, which is why this works from the discrepancy instead.
Comparing attribute sets scored every pair of torch modules as identical -- they all carry
_parameters, _buffers, _modules, training -- so three unrelated top-level submodules registered as a
stack and shadowed the real ones: 5 stacks became 3 and an encoder was lost. Comparing child-module
names with framework internals excluded still could not separate "three wrappers around one layer
kind" from "three submodules of a model", and the similarity mean included the reference compared
with itself, so any two-element list passed. Both attempts were worse than leaving the walk alone.

WHAT IS MEASURABLE. A pipeline built from HF weights carries the reference model, and torch holds
its stacks as ModuleLists of one class -- so the walk always sees those. Two reference stacks against
one device stack is a fact that needs no naming rule, no config and no per-model code. It does not
say what the missing stack looks like; an agent reading the source can, because it can see which list
is built from the reference's layers and run in sequence by the forward.

NOTHING HERE IS MODEL-SPECIFIC. The check is a count of two kinds of stack the walk already returns.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


class _Stack:
    def __init__(self, path, blocks):
        self.path, self.stack = path, blocks


def _ref(n):
    import torch

    class RefLayer(torch.nn.Module):
        pass

    return [RefLayer() for _ in range(n)]


def _dev(n):
    class TtBlock:
        def __call__(self):
            pass

    return [TtBlock() for _ in range(n)]


def test_a_hidden_stack_is_counted():
    """Voxtral's shape: two reference stacks, one device stack, one section unreadable."""
    from agent.stack_visibility import hidden_stack_count, split_stacks

    stacks = [
        _Stack("hf.model.audio_tower.layers", _ref(32)),
        _Stack("hf.model.language_model.layers", _ref(30)),
        _Stack("enc_a._inner.layers", _dev(32)),
    ]
    dev, ref = split_stacks(stacks)
    assert (len(dev), len(ref)) == (1, 2)
    assert hidden_stack_count(stacks) == 1


def test_a_repaired_model_reports_nothing_hidden():
    from agent.stack_visibility import hidden_stack_count

    stacks = [
        _Stack("hf.model.audio_tower.layers", _ref(32)),
        _Stack("hf.model.language_model.layers", _ref(30)),
        _Stack("enc_a._inner.layers", _dev(32)),
        _Stack("lm_layers", _dev(3)),
    ]
    assert hidden_stack_count(stacks) == 0


def test_more_device_stacks_than_reference_is_not_a_defect():
    """A pipeline may split ONE reference stack across two resident towers -- which is exactly what
    Voxtral does with its two encoders. More device stacks than reference is a design, not a gap."""
    from agent.stack_visibility import hidden_stack_count

    stacks = [_Stack("hf.model.layers", _ref(32)), _Stack("a.layers", _dev(32)), _Stack("b.layers", _dev(32))]
    assert hidden_stack_count(stacks) == 0


def test_a_model_holding_no_reference_never_false_fires():
    """Without a reference there is nothing to compare against, so this degrades to today's
    behaviour rather than inventing a discrepancy."""
    from agent.stack_visibility import hidden_stack_count

    assert hidden_stack_count([_Stack("tt.layers", _dev(8))]) == 0
    assert hidden_stack_count([]) == 0


def _walk():
    """find_all_stacks lifted out of the probe, so the rule the prompt describes is checked against
    the rule the walk applies rather than against my memory of it."""
    import ast
    import types

    src = (_PA / "cc_optimize" / "_op_sig_probe.py").read_text()
    tree = ast.parse(src)
    want = {
        "_shape_sig", "_is_atomic", "_stack_members", "_stack_tier", "_shared_base", "_is_composite",
        "_is_block_stack", "_uniform_kind", "_child_nodes", "_node_sequence", "_largest_repeated_stack",
        "_enclosing_stack", "StackInfo", "_dominant_type", "_walk_for_stacks", "find_all_stacks",
    }  # fmt: skip
    keep = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)) or getattr(n, "name", None) in want
    ]
    mod = types.ModuleType("_walk_only")
    mod.__dict__["__file__"] = str(_PA / "cc_optimize" / "_op_sig_probe.py")
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<probe>", "exec"), mod.__dict__)
    return mod.find_all_stacks


def test_the_advice_the_prompt_gives_actually_works_on_the_real_walk():
    """A PROMPT THAT PRESCRIBES A FIX THE WALK STILL REJECTS BURNS EVERY RETRY ROUND.

    Measured against find_all_stacks itself: a hybrid list needs a shared base AND at least 4
    elements -- three differently-classed wrappers stay invisible however they are based, because the
    >=4 bound is what stops an ordinary list of a few unrelated submodules from reading as a stack.
    Same class has no such bound. The lists get short exactly when the profiler caps the depth, so
    "just add a base class" is advice that works during discovery and fails under the cap.
    """
    import torch

    from agent.stack_visibility import census, hidden_stack_count, parse_census

    find_all_stacks = _walk()

    class Sub:
        def __call__(self):
            pass

    class Ref(torch.nn.Module):
        pass

    class HF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.ModuleList([Ref() for _ in range(32)])
            self.b = torch.nn.ModuleList([Ref() for _ in range(30)])

    class Enc:
        def __init__(self):
            self.attn = Sub()

        def __call__(self):
            pass

    def hidden(lm):
        class Pipe:
            def __init__(self):
                self.hf, self.enc, self.lm = HF(), [Enc() for _ in range(32)], lm

        return hidden_stack_count(parse_census(census(find_all_stacks(Pipe()))))

    class Base:
        pass

    def blk(base, n):
        cls = type(
            "W%d" % n,
            (base,) if base else (),
            {"__init__": lambda s: setattr(s, "attn", Sub()), "__call__": lambda s: None},
        )
        return cls

    A, B = blk(Base, 1), blk(Base, 2)
    NA, NB = blk(None, 3), blk(None, 4)

    assert hidden([NA(), NB(), NA(), NB()]) == 1, "unrelated classes must stay hidden"
    assert hidden([A(), B(), A(), B()]) == 0, "a shared base at 4 blocks is what the prompt promises"
    assert hidden([A(), B(), A()]) == 1, "THE TRAP: a shared base is not enough below 4 blocks"
    assert hidden([A(), A(), A()]) == 0, "one class has no length bound"


def test_the_prompt_does_not_promise_a_base_is_always_enough():
    from agent.stack_visibility import repair_prompt, retry_prompt

    stacks = [_Stack("hf.model.layers", _ref(30)), _Stack("enc.layers", _dev(32))]
    for p in (repair_prompt(stacks), retry_prompt(stacks)):
        assert "4" in p, "the length bound on a shared base is not stated"
    assert "PREFER ONE CLASS" in repair_prompt(stacks), "the fix that always works is not the one asked for"


def test_the_prompt_states_the_measurement_and_the_rule():
    from agent.stack_visibility import repair_prompt

    stacks = [
        _Stack("hf.model.language_model.layers", _ref(30)),
        _Stack("enc_a._inner.layers", _dev(32)),
    ]
    p = repair_prompt(stacks)
    assert "1 block stack(s); the device side exposes 1" in p or "reference model has" in p
    assert "share a common base" in p, "the rule the walk applies is not stated"
    assert "do not change what any of them does" in p.lower() or "do not touch numerics" in p
    assert "hf.model.language_model.layers" in p, "the agent is not told which stack has no counterpart"


def test_the_retry_reports_the_walk_not_a_verdict():
    """Same discipline as the depth repair: an agent told 'it failed' repeats itself; an agent told
    what the walk now returns fixes what the walk reads."""
    from agent.stack_visibility import retry_prompt

    stacks = [_Stack("hf.model.layers", _ref(30)), _Stack("enc_a.layers", _dev(32))]
    r = retry_prompt(stacks)
    assert "After the edit the walk now returns" in r
    assert "hf.model.layers" in r and "enc_a.layers" in r
    assert "WRAPPER's class is what the walk sees" in r


def test_the_census_survives_the_filter_that_hides_the_gap():
    """THE WALK RUNS IN THE PROBE AND THE RUN NEVER SEES ITS OBJECTS.

    _device_stacks drops the reference stacks before tagging -- correctly, sizing from one asked
    Voxtral for depth 32, its own full depth, so the cap changed no work. But that filter is also
    what erases the evidence: afterwards a model running three sections and exposing one looks
    exactly like a model with one section, and the run only ever receives a signpost sequence. The
    census states both kinds before the filter, and travels as one parseable line.
    """
    from agent.stack_visibility import census, hidden_stack_count, parse_census

    stacks = [
        _Stack("hf.model.audio_tower.layers", _ref(32)),
        _Stack("hf.model.language_model.layers", _ref(30)),
        _Stack("enc_a._inner.layers", _dev(32)),
    ]
    rows = parse_census("some pytest noise\n" + census(stacks) + "\nmore noise")
    assert len(rows) == 3
    assert {r["kind"] for r in rows} == {"reference", "device"}
    assert hidden_stack_count(rows) == 1, "the count does not survive the round trip"

    paths = {r["path"] for r in rows}
    assert "hf.model.audio_tower.layers" in paths and "enc_a._inner.layers" in paths


def test_a_probe_that_emits_no_census_changes_nothing():
    """An older probe, or a run that died before building -- degrades to today's behaviour."""
    from agent.stack_visibility import hidden_stack_count, parse_census

    assert parse_census("") == []
    assert parse_census("PERF_STACK_CENSUS=not json") == []
    assert hidden_stack_count(parse_census("")) == 0


def test_the_probe_states_the_walk_before_it_filters():
    src = (_PA / "cc_optimize" / "_op_sig_probe.py").read_text()
    i = src.index("_all = find_all_stacks(obj)")
    order = src[i : i + 200]
    assert order.index("_census(_all)") < order.index("_device_stacks(_all)"), "the census reads the filtered walk"


def test_the_run_repairs_before_it_sizes_anything():
    """Ordering is the whole point: the empty-walk refusal, the declared-sections comparison, the
    depth repair and the coverage sizing all read the walk. A half-visible model passes each of them
    on its visible part alone."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    vis = src.index("make_stacks_visible(_root,")
    assert vis < src.index('if not facts["full_blocks"]'), "sizing happens before the repair"
    assert vis < src.index("make_model_cappable(_root"), "the depth repair runs on the unrepaired walk"


def test_the_run_reads_its_facts_off_the_repaired_walk():
    """A repair the run then ignores is worse than none: it costs a device probe and reports success
    while every number downstream still describes the model as it was."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("make_stacks_visible(_root,")
    after = src[i : i + 800]
    assert "_facts_from(raw, sigs, seq)" in after, "the facts are not recomputed after a re-walk"
    assert "sigs, raw, seq = _last" in after, "the re-walk's own probe result is discarded"


def test_the_second_round_states_the_walk_not_the_failure():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def make_stacks_visible(")
    assert "feedback=i > 0" in src[i : i + 4000], "every round sends the same opening prompt"


def test_the_run_verifies_by_rewalking():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert "def make_stacks_visible(" in src, "the run never repairs a hidden stack"
    i = src.index("def make_stacks_visible(")
    body = src[i : i + 4000]
    assert "rewalk" in body, "the repair is not verified by re-walking"
    assert "hidden_stack_count(cur" in body, "the count is not re-measured after the edit"


def test_making_a_second_stack_visible_does_not_refuse_the_run():
    """SUCCEEDING CHANGED THE TOKEN FORMAT, AND THE READER ONLY KNEW THE OLD ONE.

    A single-stack model emits `PERF_BLOCK_SIGNPOST:7`. As soon as a second stack is discoverable the
    emitter switches to `PERF_BLOCK_SIGNPOST:stack2:7`, so each block can be attributed to its own
    stack. _blocks_ran split on the FIRST colon, so it parsed "stack2:7" as an integer, raised,
    swallowed it and returned 0 -- which reads downstream as "the built model exposes NO discoverable
    block stacks" and REFUSES the run.

    Measured on Voxtral, 2026-08-12: the repair made lm_layers visible, the walk went 1 -> 2 device
    stacks as designed, the probe emitted 155 signposts, and the run refused anyway. No single-stack
    model could ever reach this, so it sat latent until the repair started working.
    """
    import sys as _s
    from pathlib import Path as _P

    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "cc_optimize"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_run_for_blocks", _P(__file__).resolve().parent.parent / "cc_optimize" / "run.py"
    )
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    single = ["PERF_BLOCK_SIGNPOST:%d" % i for i in range(30)]
    assert run._blocks_ran(single) == 30, "the single-stack form regressed"

    multi = ["PERF_BLOCK_SIGNPOST:stack2:%d" % i for i in range(32)]
    multi += ["PERF_BLOCK_SIGNPOST:stack3:%d" % i for i in range(3)]
    assert run._blocks_ran(multi) == 32, "a multi-stack model still reports zero blocks ran"

    assert run._blocks_ran(["PERF_BLOCK_SIGNPOST:stack2:junk"]) == 0, "an unparseable index must not throw"
    assert run._blocks_ran([]) == 0


def test_every_signpost_reader_uses_the_one_parser():
    """THE BUG WAS DUPLICATION, NOT ONE BAD LINE.

    `_parse_signpost_payload` already handled both token forms correctly. Three readers had their own
    spelling instead -- `int(tok.split(":", 1)[1])` in _blocks_ran (which refused the run),
    `body.rsplit(":", 1)[-1]` in _stack_paths, and `t.split(":")[1]` in the stack-count comprehension.
    Fixing only the first would leave the next reader to rediscover the same trap, so the format is
    now decoded in exactly one place.
    """
    src = (_PA / "cc_optimize" / "run.py").read_text()
    body = src[src.index("def _blocks_ran(") : src.index("def _model_root_from_node(")]
    assert "_parse_signpost_payload" in body, "_blocks_ran still parses the token itself"
    paths = src[src.index("def _stack_paths(") :][:900]
    assert "_parse_signpost_payload" in paths, "_stack_paths still parses the token itself"
    assert 't.split(":")[1]' not in src, "the ad-hoc stack-id comprehension is still present"
    assert src.count('int(tok.split(":", 1)[1])') == 0, "the original broken spelling survives somewhere"

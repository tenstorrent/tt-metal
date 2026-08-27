"""`--batch` must pick its batching axis from what the MODEL reports.

`_batch_prompt_block` used to emit one fixed instruction block describing autoregressive decode
(a per-step cache, one slot per stream) for *every* model. A model with an autoregressive head
matched it and batched. A model without one -- an iterative/diffusion pipeline, which has no
per-step cache and therefore no decode tile to fill -- was handed instructions it could not
satisfy, hit the block's closing "STOP and report it as a hole" line, and emitted B=1. Batching B
independent samples was perfectly possible for it; nothing had asked for that axis.

The axis is now chosen from the task heads discovered by `_enumerate_task_heads`, each of which
records whether it exposes the framework's autoregressive generation contract. That signal is read
off the class object, never matched against a model, stage or class NAME, so a renamed head still
reports itself correctly.
"""

import inspect

from scripts.tt_hw_planner.commands.emit_e2e import (
    _BATCH_AUTOREGRESSIVE_AXIS,
    _BATCH_COMMON_RULES,
    _BATCH_INDEPENDENT_AXIS,
    _batch_prompt_block,
    _enumerate_task_heads,
)

_AR_HEAD = {"class": "AHead", "task": "a", "generates": True}
_PLAIN_HEAD = {"class": "BHead", "task": "b", "generates": False}


def test_batch_of_one_emits_nothing():
    """Unchanged default: B<=1 adds no instruction at all."""
    for value in (0, 1, None):
        assert _batch_prompt_block(value) == ""
        assert _batch_prompt_block(value, heads=[_AR_HEAD]) == ""


def test_autoregressive_head_gets_the_cache_axis():
    block = _batch_prompt_block(8, heads=[_AR_HEAD, _PLAIN_HEAD])
    assert "per-step cache holds B independent sequences" in block
    assert "NO autoregressive generation contract" not in block


def test_no_autoregressive_head_gets_the_independent_axis():
    """The regression: a model with no generation contract must still get a batch axis."""
    block = _batch_prompt_block(32, heads=[_PLAIN_HEAD])
    assert "NO autoregressive generation contract" in block
    assert "stack the 32 samples on the LEADING axis" in block
    assert "per-step cache holds B independent sequences" not in block


def test_empty_discovery_is_an_answer_not_a_gap():
    """`[]` means nothing reported generation -> independent axis, not the legacy path.

    Models whose config will not load as a transformers auto-model (the diffusion pipelines this
    fix is for) discover no heads at all. Treating that as "unknown" would route them straight
    back into the autoregressive text.
    """
    assert "NO autoregressive generation contract" in _batch_prompt_block(32, heads=[])


def test_omitting_heads_preserves_the_previous_behaviour():
    """Additive change: callers that pass no heads get exactly the old autoregressive text."""
    assert "per-step cache holds B independent sequences" in _batch_prompt_block(8)
    assert inspect.signature(_batch_prompt_block).parameters["heads"].default is None


def test_both_axes_carry_the_shared_invariants():
    """The rules that hold either way live in one constant, so the paths cannot drift."""
    for block in (
        _batch_prompt_block(4, heads=[_AR_HEAD]),
        _batch_prompt_block(4, heads=[_PLAIN_HEAD]),
    ):
        assert _BATCH_COMMON_RULES.format(batch=4) in block
        assert "do NOT shard on batch" in block
        assert "do NOT fake a batch axis" in block


def test_common_rules_tell_the_builder_to_unpin_b1_stubs():
    """The actual blocker for a non-autoregressive model: stubs PCC'd at B=1 hardcode a leading 1."""
    rules = _BATCH_COMMON_RULES.format(batch=32)
    assert "x.shape[0]" in rules
    assert "SILENTLY DROPS" in rules


def test_batch_count_is_threaded_into_every_axis():
    for heads in ([_AR_HEAD], [_PLAIN_HEAD], None):
        block = _batch_prompt_block(16, heads=heads)
        assert "BATCH = 16" in block
        assert "{batch}" not in block, "an unformatted placeholder reached the prompt"


def test_axis_choice_uses_no_model_or_stage_names():
    """Constraint: the decision is made from the reported signal, never from a typed-in name."""
    import ast
    import textwrap

    src = inspect.getsource(_batch_prompt_block)
    assert 'h.get("generates")' in src

    # Scan the EXECUTABLE body only. Prose may name an architecture as an example; the branch
    # itself must not, so the docstring is dropped before scanning rather than exempting words.
    fn = ast.parse(textwrap.dedent(src)).body[0]
    if ast.get_docstring(fn) is not None:
        fn.body = fn.body[1:]
    lowered = ast.unparse(fn).lower()
    for name in ("flux", "xtts", "seamless", "voxtral", "llama", "qwen", "diffusion", "vocoder"):
        assert name not in lowered, f"{name!r} is a name the axis choice must not depend on"


def test_independent_axis_text_names_no_stage():
    """The new block describes the model's own iteration, not a named stage."""
    lowered = _BATCH_INDEPENDENT_AXIS.lower()
    for name in ("denoise", "diffusion", "vae", "unet", "vocoder", "prefill"):
        assert name not in lowered, f"{name!r} would hardcode a stage name into the guidance"


def test_enumerate_task_heads_records_the_signal_and_keeps_its_contract():
    """`generates` is additive: existing callers read only `class`/`task`."""
    src = inspect.getsource(_enumerate_task_heads)
    assert '"generates": bool(has_gen)' in src
    assert '"class": cls_name' in src and '"task": slug' in src
    assert 'hasattr(obj, "generate")' in src, "the signal must be read off the class, not its name"


def test_autoregressive_axis_still_describes_the_cache_contract():
    """Guard the text the previously-working path depends on."""
    axis = _BATCH_AUTOREGRESSIVE_AXIS.format(batch=8)
    assert "one cache slot per batch row" in axis
    assert "[B, heads, C, head_dim]" in axis

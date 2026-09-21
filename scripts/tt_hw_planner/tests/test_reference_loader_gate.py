"""The reference-loader gate must not bank a file that defines no loader.

Every per-component PCC score for a non-transformers checkpoint is measured against whatever
`_reference_loader.py` returns, so "is this a loader?" is the question the whole gate rests on. It
used to be answered with ``"def load_reference_model" in source`` -- a substring, which the name
merely being MENTIONED satisfied. A file whose only occurrence was in a comment (the shape an agent
leaves when it writes a TODO and stops) defined nothing, yet resolved=True and bring-up moved on.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import importlib.util

import pytest

from scripts.tt_hw_planner.reference_loader_resolver import (
    checkpoint_coverage,
    _resolved,
    _validates,
    _NATIVE_CONFIG_FILE,
    assess,
    check_invariants,
    config_fidelity,
    loader_path,
    uses_random_weights,
    verify,
    weight_provenance,
)

# Scoped to the tests that need it: the structural checks below must keep running on a box with no
# torch, which is exactly where a contributor is most likely to run this file.
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("safetensors") is None,
    reason="needs torch + safetensors",
)


def _write(tmp_path: Path, src: str) -> Path:
    p = loader_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return tmp_path


def test_real_loader_validates(tmp_path: Path) -> None:
    assert _validates(
        _write(
            tmp_path,
            """
            def load_reference_model(model_id: str):
                return object()
            """,
        )
    )


def test_name_only_in_a_comment_is_not_a_loader(tmp_path: Path) -> None:
    # The substring gate accepted this: nothing is defined, so the import the PCC template does
    # would fail LATER, far from the cause, with bring-up already recorded as resolved.
    assert not _validates(
        _write(
            tmp_path,
            """
            # def load_reference_model(model_id: str): -- TODO, not written yet
            PLACEHOLDER = True
            """,
        )
    )


def test_name_only_inside_a_string_is_not_a_loader(tmp_path: Path) -> None:
    assert not _validates(
        _write(
            tmp_path,
            '''
            HELP = """
            def load_reference_model(model_id): ...
            """
            ''',
        )
    )


def test_zero_arg_stub_is_not_a_loader(tmp_path: Path) -> None:
    # Callers pass the model id; a no-arg def would TypeError on first use.
    assert not _validates(
        _write(
            tmp_path,
            """
            def load_reference_model():
                return object()
            """,
        )
    )


def test_unparseable_and_missing_files_are_not_loaders(tmp_path: Path) -> None:
    assert not _validates(tmp_path)  # nothing written at all
    assert not _validates(_write(tmp_path, "def load_reference_model(  <<< syntax error\n"))


def test_random_weight_fallback_travels_with_the_result(tmp_path: Path) -> None:
    """Strategy 5 builds the reference from random weights, so PCC against it verifies STRUCTURE
    only. That used to be recorded in a module docstring, which nothing reads -- a run could be
    scored against weights unrelated to the checkpoint and the result looked identical."""
    d = _write(
        tmp_path,
        """
        REFERENCE_USES_RANDOM_WEIGHTS = True

        def load_reference_model(model_id: str):
            return object()
        """,
    )
    assert uses_random_weights(d)
    out = _resolved(d, "loader written")
    assert out["resolved"] is True and out["random_weights"] is True
    assert "RANDOM weights" in out["caveat"]
    # And it must reach `reason`, which is the only field any caller of resolve() actually prints.
    # Carried solely under its own key, the caveat was written and read by nobody.
    assert "RANDOM weights" in out["reason"], out["reason"]


def test_real_weight_loader_carries_no_caveat(tmp_path: Path) -> None:
    d = _write(
        tmp_path,
        """
        def load_reference_model(model_id: str):
            return object()
        """,
    )
    assert not uses_random_weights(d)
    assert "random_weights" not in _resolved(d, "loader written")


# --- runtime gate: a file that parses is still not a loader that WORKS ------------------------
# Each case below is a loader that sails through the structural check and would have been banked as
# resolved, then failed somewhere downstream where the cause is no longer obvious.


@requires_torch
@pytest.mark.parametrize(
    ("body", "expect"),
    [
        ("raise RuntimeError('no weights here')", "RuntimeError"),
        ("return None", "returned None"),
        ("return {'state_dict': {}}", "not nn.Module"),
        ("import torch; return torch.nn.Module()", "no parameters"),
    ],
    ids=["raises", "returns-none", "returns-non-module", "no-parameters"],
)
def test_runtime_gate_rejects_loaders_that_cannot_produce_a_model(tmp_path: Path, body: str, expect: str) -> None:
    d = _write(tmp_path, f"def load_reference_model(model_id):\n    {body}\n")
    assert _validates(d), "precondition: the structural check accepts all of these"
    v = verify(d, "some/model")
    assert v["ok"] is False and v["status"] == "broken"
    assert expect in v["reason"], v["reason"]


@requires_torch
def test_runtime_gate_accepts_a_loader_that_returns_a_real_module(tmp_path: Path) -> None:
    d = _write(
        tmp_path,
        """
        import torch

        def load_reference_model(model_id):
            return torch.nn.Linear(8, 8)
        """,
    )
    v = verify(d, "some/model")
    assert v["ok"] is True and v["status"] == "verified", v["reason"]


# --- provenance: break the model on purpose ---------------------------------------------------


def _checkpoint(tmp_path: Path, *shards: dict) -> str:
    """Write one safetensors file per shard; a lone dict is the single-file case."""
    from safetensors.torch import save_file

    d = tmp_path / "ckpt"
    d.mkdir(parents=True, exist_ok=True)
    for i, tensors in enumerate(shards):
        save_file(tensors, str(d / f"model-{i:05d}.safetensors"))
    return str(d)


def _module_with(*weights) -> "torch.nn.Module":
    import torch

    m = torch.nn.Module()
    for i, w in enumerate(weights):
        setattr(m, f"w{i}", torch.nn.Parameter(w, requires_grad=False))
    return m


@requires_torch
def test_provenance_confirms_weights_that_came_from_the_checkpoint(tmp_path: Path) -> None:
    import torch

    w = torch.randn(4096)
    out = weight_provenance(_checkpoint(tmp_path, {"w": w}), _module_with(w.clone()))
    assert out["status"] == "from_checkpoint", out


@requires_torch
def test_provenance_flags_a_reference_that_never_loaded_the_weights(tmp_path: Path) -> None:
    """THE case worth catching: architecture built from config, weights left at random init.

    Such a reference loads cleanly, has the right shapes, and passes every other check -- while
    every PCC measured against it is meaningless.
    """
    import torch

    ckpt = _checkpoint(tmp_path, {"w": torch.randn(4096)})
    random_init = _module_with(torch.randn(4096) * 0.02)  # never read the checkpoint
    out = weight_provenance(ckpt, random_init)
    assert out["status"] == "no_match", out
    assert "randomly initialised" in out["reason"]


@requires_torch
def test_provenance_tolerates_a_permuted_but_correct_conversion(tmp_path: Path) -> None:
    """A correct loader may reorder weights (RoPE layouts differ); that must not read as no_match."""
    import torch

    w = torch.randn(4096)
    permuted = w[torch.randperm(w.numel())]
    out = weight_provenance(_checkpoint(tmp_path, {"w": w}), _module_with(permuted))
    assert out["status"] == "from_checkpoint", out


@requires_torch
def test_unreachable_checkpoint_is_unverified_not_a_failure(tmp_path: Path) -> None:
    """An environment that cannot check must not be reported as a bad loader."""
    import torch

    out = weight_provenance(str(tmp_path / "nothing-here"), _module_with(torch.randn(4096)))
    assert out["status"] == "unverified", out


@requires_torch
def test_a_loader_that_rewrites_one_shard_is_not_convicted_by_it(tmp_path: Path) -> None:
    """Why the sample spans shards: which file a tensor lands in is an artefact of the writer.

    This loader transformed everything in the largest shard and loaded the rest verbatim, which is
    what a real conversion does. Sampling only the largest shard would read zero matches and -- now
    that the verdict blocks -- refuse a correct loader.
    """
    import torch

    kept = torch.randn(4096)
    ckpt = _checkpoint(tmp_path, {"a": torch.randn(8192)}, {"b": kept})
    out = weight_provenance(ckpt, _module_with(torch.randn(8192) * 3.0, kept.clone()))
    assert out["status"] == "from_checkpoint", out


@requires_torch
def test_a_reference_that_loaded_nothing_matches_nothing_in_any_shard(tmp_path: Path) -> None:
    import torch

    ckpt = _checkpoint(tmp_path, {"a": torch.randn(8192)}, {"b": torch.randn(4096)})
    out = weight_provenance(ckpt, _module_with(torch.randn(8192) * 0.02, torch.randn(4096) * 0.02))
    assert out["status"] == "no_match", out
    assert "2 checkpoint file(s)" in out["reason"], out


@requires_torch
def test_weights_that_never_loaded_now_stop_the_run_rather_than_warn(tmp_path: Path) -> None:
    """The reason this blocks: the TT port is built FROM this module (`_build_ttnn_port`).

    So a randomly-initialised reference hands its random weights to the TT side too, both agree
    perfectly, and every component passes having compared noise to the same noise. The failure is
    not red, it is green -- which is exactly why warning about it was not enough.
    """
    import torch

    ckpt = _checkpoint(tmp_path, {"a": torch.randn(8192)}, {"b": torch.randn(4096)})
    verdict = assess(_module_with(torch.randn(8192) * 0.02, torch.randn(4096) * 0.02), ckpt)
    assert verdict["ok"] is False, verdict
    assert "randomly initialised" in verdict["reason"], verdict


@requires_torch
def test_a_checkpoint_that_stops_being_readable_is_not_blamed_on_the_loader(tmp_path: Path) -> None:
    """A read that dies partway cannot claim "nothing matched" -- the matches may be in the rest."""
    import torch

    ckpt = _checkpoint(tmp_path, {"a": torch.randn(8192)}, {"b": torch.randn(4096)})
    (Path(ckpt) / "model-00001.safetensors").write_bytes(b"truncated, not a safetensors file")
    out = weight_provenance(ckpt, _module_with(torch.randn(8192) * 0.02, torch.randn(4096) * 0.02))
    assert out["status"] == "unverified", out


# --------------------------------------------------------------------------------------------
# Self-consistency. Provenance proves the weights are the shipped ones; it says nothing about
# whether they are wired up correctly, and a miswired reference has no golden to be caught by.
# These models are deliberately broken in the three ways a generated loader actually breaks.
# --------------------------------------------------------------------------------------------


def _tiny_lm(flaw: str = "none"):
    """A minimal causal LM, optionally carrying one specific wiring bug."""
    import torch
    import torch.nn as nn

    class Cfg:
        vocab_size, hidden_size = 32, 8
        is_decoder = True  # how a decoder announces that causality applies to it

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.embed = nn.Embedding(Cfg.vocab_size, Cfg.hidden_size)
            self.head = nn.Linear(Cfg.hidden_size, Cfg.vocab_size)

        def forward(self, input_ids: torch.LongTensor):
            h = self.embed(input_ids)
            if flaw == "acausal":
                # Averages the WHOLE sequence, so every position can see the future. This is what a
                # forgotten causal mask looks like: it trains, it runs, the numbers look fine.
                h = h.mean(dim=1, keepdim=True).expand_as(h)
            elif flaw == "batch_mixing":
                # A transposed axis, so one sample's values land in another's row.
                h = h + h.mean(dim=0, keepdim=True)
            else:
                h = torch.cumsum(h, dim=1) / torch.arange(1, h.shape[1] + 1).view(1, -1, 1)
            out = self.head(h)
            return out + torch.randn_like(out) if flaw == "nondeterministic" else out

    return LM().eval()


@requires_torch
def test_a_correctly_wired_reference_satisfies_every_invariant() -> None:
    out = check_invariants(_tiny_lm())
    assert out["status"] == "holds", out
    assert out["checks"] == {"determinism": "pass", "batch_invariance": "pass", "causality": "pass"}


@requires_torch
def test_a_reference_that_can_see_its_own_future_is_caught() -> None:
    """The failure with no golden: right weights, missing causal mask, plausible numbers."""
    out = check_invariants(_tiny_lm("acausal"))
    assert out["status"] == "violated", out
    assert out["checks"]["causality"] == "FAIL", out


@requires_torch
def test_a_reference_that_leaks_across_the_batch_is_caught() -> None:
    out = check_invariants(_tiny_lm("batch_mixing"))
    assert out["status"] == "violated", out
    assert out["checks"]["batch_invariance"] == "FAIL", out


@requires_torch
def test_a_reference_that_will_not_reproduce_itself_is_caught() -> None:
    out = check_invariants(_tiny_lm("nondeterministic"))
    assert out["status"] == "violated", out
    assert out["checks"]["determinism"] == "FAIL", out


@requires_torch
def test_a_bidirectional_encoder_is_not_condemned_for_being_bidirectional() -> None:
    """An audio tower or text encoder sees the whole sequence BY DESIGN.

    The acausal model here is wired identically to the one two tests above that gets rejected --
    the only difference is that this one does not claim to be a decoder. Causality is a law for
    decoders, so for anything else the honest answer is `skipped`, not a failure.
    """
    encoder = _tiny_lm("acausal")
    encoder.config.is_decoder = False
    out = check_invariants(encoder)
    assert out["status"] == "holds", out
    assert out["checks"]["causality"] == "skipped", out


@requires_torch
def test_a_model_that_states_its_own_inputs_is_taken_at_its_word() -> None:
    """`dummy_inputs` is the model describing itself, which beats anything inferred about it."""
    import torch

    lm = _tiny_lm()
    lm.dummy_inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    assert check_invariants(lm)["status"] in {"holds", "unverified"}


@requires_torch
def test_an_output_field_this_has_never_heard_of_is_still_found() -> None:
    """The checks must not depend on a model naming its output the way other models do."""
    import torch
    import torch.nn as nn

    class OddlyNamed(nn.Module):
        """Wraps its result in a field no name table would contain."""

        def __init__(self):
            super().__init__()
            self.inner = _tiny_lm()
            self.config = self.inner.config

        def forward(self, input_ids: torch.LongTensor):
            result = self.inner(input_ids)

            class Wrapped:
                def to_tuple(self_inner):
                    return (result,)

            return Wrapped()

    out = check_invariants(OddlyNamed())
    assert out["status"] == "holds", out
    assert out["checks"]["causality"] == "pass", out


@requires_torch
def test_a_model_that_cannot_be_driven_is_unverified_not_broken() -> None:
    """Absence of evidence is not evidence of a bug -- this must never fail a good loader."""
    import torch.nn as nn

    class Opaque(nn.Module):
        def forward(self, mystery):  # no annotation, no config to size it from
            return mystery

    out = check_invariants(Opaque())
    assert out["status"] == "unverified", out


@requires_torch
def test_a_violated_invariant_fails_the_gate_and_says_why(tmp_path) -> None:
    """The verdict has to reach `verify`, or the check is a comment."""
    import scripts.tt_hw_planner.reference_loader_resolver as rlr

    demo = tmp_path / "demo"
    loader_path(demo).parent.mkdir(parents=True, exist_ok=True)
    loader_path(demo).write_text(
        textwrap.dedent(
            """
            def load_reference_model(model_id):
                raise AssertionError("patched out")
            """
        )
    )
    monkey = _tiny_lm("acausal")
    original, rlr.load_reference = rlr.load_reference, lambda *a, **k: monkey
    try:
        out = verify(demo, str(tmp_path))
    finally:
        rlr.load_reference = original
    assert out["ok"] is False and out["status"] == "broken", out
    assert "causality" in out["reason"], out


# --------------------------------------------------------------------------------------------
# Declared constants. The invariants above are blind here: a wrong epsilon or RoPE base keeps the
# model deterministic, batch-invariant and causal, and every shape identical. It is simply a
# different model, and it becomes the yardstick. The checkpoint states these numbers itself.
# --------------------------------------------------------------------------------------------


def _llama(**overrides):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, **overrides
    )
    return LlamaForCausalLM(cfg).eval()


@requires_torch
def test_a_wrong_constant_is_invisible_to_every_other_layer(tmp_path) -> None:
    """The premise for this check existing, pinned so it cannot be quietly assumed away.

    If a perturbed epsilon ever DID trip the invariants, this check would be redundant. It does
    not, which is exactly why the constants have to be read from the checkpoint instead.
    """
    assert check_invariants(_llama(rms_norm_eps=9.0))["status"] == "holds"
    assert check_invariants(_llama(rope_theta=2.0))["status"] == "holds"


@requires_torch
def test_a_reference_that_contradicts_its_checkpoint_is_caught(tmp_path) -> None:
    shipped = _llama()
    shipped.save_pretrained(tmp_path)
    # Written explicitly because save_pretrained omits values left at their default.
    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg["rms_norm_eps"] = 1e-05
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    assert config_fidelity(str(tmp_path), _llama(rms_norm_eps=1e-05))["status"] == "matches"
    wrong = config_fidelity(str(tmp_path), _llama(rms_norm_eps=1e-03))
    assert wrong["status"] == "diverges", wrong
    assert "rms_norm_eps" in wrong["mismatched"], wrong


@requires_torch
def test_a_native_checkpoint_is_checked_by_value_since_its_names_differ(tmp_path) -> None:
    """`params.json` says `norm_eps` where the converted model says `rms_norm_eps`.

    Translating between the two would need the hand-written name table that must not exist, so the
    declared values are looked for anywhere in the model instead.
    """
    (tmp_path / _NATIVE_CONFIG_FILE).write_text(json.dumps({"dim": 32, "n_layers": 2, "norm_eps": 1e-05}))
    (tmp_path / "weights.safetensors").write_bytes(b"")

    assert config_fidelity(str(tmp_path), _llama(rms_norm_eps=1e-05))["status"] == "present"
    missing = config_fidelity(str(tmp_path), _llama(rms_norm_eps=1e-03))
    assert missing["status"] == "absent", missing
    assert "norm_eps" in missing["mismatched"], missing


@requires_torch
def test_one_key_matching_by_name_does_not_wave_the_rest_through(tmp_path) -> None:
    """The realistic native config: SOME keys collide with the reference's names, some do not.

    `rope_theta` is spelled the same either side, `norm_eps` is not. Checking by value only when
    nothing matched by name would let that one collision stand in as a clean bill of health and
    report `matches` over a wrong epsilon -- worse than reporting nothing, because it reassures.
    """
    (tmp_path / _NATIVE_CONFIG_FILE).write_text(
        json.dumps({"dim": 32, "n_layers": 2, "vocab_size": 64, "rope_theta": 10000.0, "norm_eps": 1e-05})
    )
    (tmp_path / "weights.safetensors").write_bytes(b"")

    assert config_fidelity(str(tmp_path), _llama(rms_norm_eps=1e-05))["status"] == "matches"
    wrong = config_fidelity(str(tmp_path), _llama(rms_norm_eps=1e-03))
    assert wrong["status"] == "absent", wrong
    assert "norm_eps" in wrong["mismatched"], wrong


@requires_torch
def test_a_checkpoint_with_no_readable_config_is_unverified(tmp_path) -> None:
    assert config_fidelity(str(tmp_path), _llama())["status"] == "unverified"


@requires_torch
def test_a_contradicted_constant_fails_the_gate(tmp_path) -> None:
    """As with the invariants, the verdict has to reach `verify` to mean anything."""
    import scripts.tt_hw_planner.reference_loader_resolver as rlr

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _llama().save_pretrained(ckpt)
    cfg = json.loads((ckpt / "config.json").read_text())
    cfg["rms_norm_eps"] = 1e-05
    (ckpt / "config.json").write_text(json.dumps(cfg))

    demo = tmp_path / "demo"
    loader_path(demo).parent.mkdir(parents=True, exist_ok=True)
    loader_path(demo).write_text("def load_reference_model(model_id):\n    raise AssertionError('patched')\n")

    wrong = _llama(rms_norm_eps=1e-03)
    original, rlr.load_reference = rlr.load_reference, lambda *a, **k: wrong
    try:
        out = verify(demo, str(ckpt))
    finally:
        rlr.load_reference = original
    assert out["ok"] is False and out["status"] == "broken", out
    assert "rms_norm_eps" in out["reason"], out


@requires_torch
def test_the_output_unwrap_needs_no_field_names_at_all() -> None:
    """Shared helper, so the gate and the e2e harness cannot drift apart on what a result is."""
    import torch

    from scripts.tt_hw_planner.model_output import result_tensor

    inner = torch.ones(2, 3)

    class Renamed:
        """Names its result something no table would contain, and offers to unpack itself."""

        def to_tuple(self):
            return (None, inner)

    assert result_tensor(Renamed()) is inner
    assert result_tensor({"whatever": (inner,)}) is inner
    assert result_tensor(inner) is inner
    assert result_tensor("not a tensor anywhere") is None


# --------------------------------------------------------------------------------------------
# Coverage. Provenance asks whether the reference's weights are genuine and is satisfied by one
# matching tensor, so a reference implementing PART of a checkpoint passes it comfortably. Nothing
# asked whether anything in the checkpoint has no counterpart in the reference at all.
#
# mistralai/Voxtral-4B-TTS-2603: no reference existed for that architecture, so the resolver fell
# back to the nearest sibling -- a text-only causal LM. 116 of 386 tensors, the whole
# `audio_tokenizer` vocoder and 30% of the model, were never enumerated and never brought up. Every
# component the tool did enumerate graduated and it reported COMPLETE.


def _big(n, scale=1.0):
    import torch

    return torch.randn(n) * scale


def _named_checkpoint(tmp_path: Path, tensors: dict) -> str:
    from safetensors.torch import save_file

    d = tmp_path / "ckpt"
    d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "model-00000.safetensors"))
    return str(d)


@requires_torch
def test_a_submodel_the_reference_never_implements_is_named(tmp_path: Path) -> None:
    """The reported failure, reproduced: a reference holding only one of two submodels."""
    driven = {f"backbone.layers.{i}.weight": _big(4096 + i) for i in range(20)}
    absent = {f"some_vocoder.blocks.{i}.weight": _big(8192 + i) for i in range(20)}
    ckpt = _named_checkpoint(tmp_path, {**driven, **absent})

    out = checkpoint_coverage(ckpt, _module_with(*[t.clone() for t in driven.values()]))
    assert out["status"] == "undriven", out
    assert out["undriven"] == ["some_vocoder"], out
    assert "some_vocoder" in out["reason"] and "20 tensors" in out["reason"], out


@requires_torch
def test_a_fully_implemented_checkpoint_is_covered(tmp_path: Path) -> None:
    tensors = {f"backbone.layers.{i}.weight": _big(4096 + i) for i in range(20)}
    ckpt = _named_checkpoint(tmp_path, tensors)
    out = checkpoint_coverage(ckpt, _module_with(*[t.clone() for t in tensors.values()]))
    assert out["status"] == "covered", out


@requires_torch
def test_a_handful_of_transformed_tensors_is_not_an_absent_submodel(tmp_path: Path) -> None:
    """A loader may dequantise, merge or fuse individual tensors; those match nothing and are still
    correct. Only a submodel too large to be a conversion artefact counts as absent."""
    kept = {f"backbone.layers.{i}.weight": _big(4096 + i) for i in range(20)}
    rewritten = {"fused.qkv.weight": _big(9999)}
    ckpt = _named_checkpoint(tmp_path, {**kept, **rewritten})
    out = checkpoint_coverage(ckpt, _module_with(*[t.clone() for t in kept.values()]))
    assert out["status"] == "covered", out


@requires_torch
def test_a_small_submodel_is_below_the_floor_even_at_zero_matches(tmp_path: Path) -> None:
    """Both floors apply: too few tensors AND too small a share stay quiet, so a stray norm the
    loader folded away cannot refuse a correct reference."""
    kept = {f"backbone.layers.{i}.weight": _big(4096 + i) for i in range(60)}
    tiny = {f"odds.{i}.weight": _big(7777 + i) for i in range(3)}
    ckpt = _named_checkpoint(tmp_path, {**kept, **tiny})
    out = checkpoint_coverage(ckpt, _module_with(*[t.clone() for t in kept.values()]))
    assert out["status"] == "covered", out


@requires_torch
def test_the_group_name_is_read_off_the_checkpoint_not_matched_to_a_list(tmp_path: Path) -> None:
    """A checkpoint that calls its parts something nobody anticipated is still grouped correctly."""
    driven = {f"aaa.layers.{i}.weight": _big(4096 + i) for i in range(20)}
    absent = {f"zzz_unheard_of_thing.{i}.weight": _big(8192 + i) for i in range(20)}
    ckpt = _named_checkpoint(tmp_path, {**driven, **absent})
    out = checkpoint_coverage(ckpt, _module_with(*[t.clone() for t in driven.values()]))
    assert out["undriven"] == ["zzz_unheard_of_thing"], out


@requires_torch
def test_an_absent_submodel_stops_the_run_rather_than_reporting_complete(tmp_path: Path) -> None:
    """The point of the gate: bring-up must not grade itself against a reference that cannot reach
    part of the model, because every component it DOES enumerate will pass."""
    driven = {f"backbone.layers.{i}.weight": _big(4096 + i) for i in range(20)}
    absent = {f"some_vocoder.blocks.{i}.weight": _big(8192 + i) for i in range(20)}
    ckpt = _named_checkpoint(tmp_path, {**driven, **absent})
    verdict = assess(_module_with(*[t.clone() for t in driven.values()]), ckpt)
    assert verdict["ok"] is False, verdict
    assert "some_vocoder" in verdict["reason"], verdict
    assert verdict["coverage"]["status"] == "undriven", verdict


@requires_torch
def test_an_unreadable_checkpoint_is_unverified_not_undriven(tmp_path: Path) -> None:
    """Same rule the provenance check already follows: a read that failed blames the environment."""
    ckpt = _named_checkpoint(tmp_path, {"backbone.w": _big(4096)})
    (Path(ckpt) / "model-00000.safetensors").write_bytes(b"truncated")
    out = checkpoint_coverage(ckpt, _module_with(_big(4096)))
    assert out["status"] == "unverified", out


@requires_torch
def test_no_checkpoint_on_disk_is_unverified(tmp_path: Path) -> None:
    out = checkpoint_coverage(str(tmp_path / "nothing-here"), _module_with(_big(4096)))
    assert out["status"] == "unverified", out


# --------------------------------------------------------------------------------------------
# Looking outside the repo. Tiers 1-4 all search the model's OWN code, so a checkpoint shipping
# weights for a submodel implemented only in a third-party engine reads as "no reference exists".
# mistralai/Voxtral-4B-TTS-2603: its flow-matching sampler and codec decoder are implemented in
# vLLM-Omni and ported from there by at least two public projects, while the resolver -- which can
# only see the HF file list -- settled for a text-only sibling covering 70% of the checkpoint.


def test_the_prompt_tells_the_resolver_to_cover_the_whole_checkpoint(tmp_path: Path) -> None:
    from scripts.tt_hw_planner.reference_loader_resolver import build_prompt

    text = build_prompt("some-org/some-model", tmp_path, "AutoConfig raised Unrecognized model")
    low = text.lower()
    assert "top-level tensor groups" in low, "the resolver is never told to enumerate the checkpoint"
    assert "search the web" in low, "the resolver is never told to look outside the repo"
    assert "covering part of a checkpoint" in low, "the partial-reference failure is never named"


def test_the_resolver_is_given_tools_that_can_reach_outside_the_repo() -> None:
    """A tier that says 'search' is inert unless the agent can actually search."""
    import inspect

    from scripts.tt_hw_planner import reference_loader_resolver as rlr

    src = inspect.getsource(rlr.resolve) if hasattr(rlr, "resolve") else inspect.getsource(rlr)
    assert '"WebSearch"' in src and '"WebFetch"' in src, "no tool can reach past the HF file list"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A checkpoint that declares itself in its own words was refused at the door, then mis-sized, then
had no word to be onboarded under.

Three gates, one habit: each had been taught a single dialect and treated everything else as absent.

  1. "Is there a model in this directory" asked only for the transformers config, so a native
     checkpoint -- weights and a full declaration sitting right there -- came back as an invalid
     HuggingFace id, naming neither the real cause nor the thing it had been handed. Reached by a
     relative name it was worse than wrong: the folder has no separator in it, so it reads as a
     well-formed hub id, and the run died reporting the user's own directory as a repo that does
     not exist.
  2. Let in, it was then measured by a reader that knew only `hidden_size` / `num_hidden_layers`.
     A native config says `dim` and `n_layers`, so the model sized as zero-width and zero-layer --
     which the caller reports as a category downgrade to CNN, blaming a config.json the checkpoint
     never claimed to have. The identical model named by hub id planned in full.
  3. Onboarding kept its own hand-typed list of categories, three behind the probe's. A model the
     probe classified as speech had no word available to record it under, and the offline path
     wrote down a vision category instead.

These pin all three, and the single lists that keep them from drifting apart again.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.tt_hw_planner import auto_onboard as AO
from scripts.tt_hw_planner import probe as P
from scripts.tt_hw_planner.architecture import build_arch_spec, detect_architecture

# The same model, twice, in the two dialects a checkpoint declares itself in.
_HF_DIALECT = {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 32000,
    "max_position_embeddings": 8192,
}
_NATIVE_DIALECT = {
    "dim": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 32000,
    "max_seq_len": 8192,
    "norm_eps": 1e-05,
}


def _checkpoint(directory: Path, filename: str, config: dict) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(json.dumps(config))
    (directory / "model.safetensors").write_bytes(b"\0" * 4096)
    return str(directory)


@pytest.fixture()
def native_checkpoint(tmp_path):
    """A non-transformers drop: its own config, real bytes, and no transformers config at all."""
    return _checkpoint(tmp_path / "native", P.NATIVE_CONFIG_FILE, _NATIVE_DIALECT)


@pytest.fixture()
def hf_checkpoint(tmp_path):
    return _checkpoint(tmp_path / "hf", P.ROOT_CONFIG_FILE, _HF_DIALECT)


# --------------------------------------------------------------------------------------
# The door
# --------------------------------------------------------------------------------------


def test_a_checkpoint_that_declares_itself_is_a_model_directory(native_checkpoint) -> None:
    assert P._is_local_model_dir(native_checkpoint)


def test_and_so_is_the_transformers_one(hf_checkpoint) -> None:
    """The dialect that already worked must keep working."""
    assert P._is_local_model_dir(hf_checkpoint)


def test_a_directory_that_declares_nothing_is_not_a_model(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("no model here")
    assert not P._is_local_model_dir(str(tmp_path))


def test_a_checkpoint_is_no_longer_refused_as_a_malformed_hub_id(native_checkpoint) -> None:
    """The error the whole bug wore: the path was reported as an invalid model id."""
    assert P._validate_hf_id(native_checkpoint) == native_checkpoint


def test_a_folder_is_never_quietly_looked_up_on_the_hub(tmp_path, monkeypatch, expect_error) -> None:
    """A relative folder name is a well-formed hub id by shape, and used to be treated as one.

    The run then failed at the hub naming the user's own directory, having never opened it.
    """
    (tmp_path / "not-a-model").mkdir()
    monkeypatch.chdir(tmp_path)

    with expect_error(SystemExit, "is a directory") as excinfo:
        P.probe_model("not-a-model")

    message = str(excinfo.value)
    assert "not-a-model" in message
    assert "directory" in message
    for name in P.MODEL_CONFIG_FILES:
        assert name in message, "the refusal must say what it looked for"


# --------------------------------------------------------------------------------------
# Reading what it declares
# --------------------------------------------------------------------------------------


def test_the_architecture_step_reads_the_native_dialect() -> None:
    spec = build_arch_spec(_NATIVE_DIALECT, detect_architecture(_NATIVE_DIALECT))

    assert spec.hidden_size == _NATIVE_DIALECT["dim"]
    assert spec.num_layers == _NATIVE_DIALECT["n_layers"]
    assert spec.num_attention_heads == _NATIVE_DIALECT["n_heads"]
    assert spec.num_key_value_heads == _NATIVE_DIALECT["n_kv_heads"]
    assert spec.max_position_embeddings == _NATIVE_DIALECT["max_seq_len"]


def test_the_dialects_describe_the_same_model() -> None:
    """Not "both non-zero" -- the same numbers, because it is the same model written down twice."""
    native = build_arch_spec(_NATIVE_DIALECT, detect_architecture(_NATIVE_DIALECT))
    hf = build_arch_spec(_HF_DIALECT, detect_architecture(_HF_DIALECT))

    for field in ("hidden_size", "num_layers", "num_attention_heads", "num_key_value_heads", "max_position_embeddings"):
        assert getattr(native, field) == getattr(hf, field), field


def test_the_native_config_document_is_actually_fetched(native_checkpoint) -> None:
    """Reading the directory used to ask for the transformers config and nothing else."""
    assert P._maybe_fetch_config(native_checkpoint) == _NATIVE_DIALECT


@pytest.mark.parametrize("folder", ["native", "bert-checkpoints", "my-opt-run", "t5x_export", "llama-3-8b"])
def test_the_folder_s_name_never_chooses_the_architecture(tmp_path, folder) -> None:
    """Asking for the transformers config first was not merely useless here -- it ANSWERED.

    Given a directory holding no config.json, AutoConfig does not fail: it matches every registered
    model_type against the PATH STRING and returns that architecture's DEFAULTS. A checkpoint in a
    folder called `native` came back a fully-populated NatConfig -- 512-wide, 4 layers, a vision
    backbone's stage names -- and, that tier having "succeeded", the params.json lying beside the
    weights was never opened. Not about one unlucky word: every one of these names hijacks it.
    """
    checkpoint = _checkpoint(tmp_path / folder, P.NATIVE_CONFIG_FILE, _NATIVE_DIALECT)

    assert P._maybe_fetch_config(checkpoint) == _NATIVE_DIALECT


@pytest.mark.parametrize("folder", ["native", "bert-checkpoints"])
def test_and_the_loader_prompt_describes_the_checkpoint_it_was_given(tmp_path, folder) -> None:
    """Where the substitution did the most damage: this text is handed to the LLM as "the model's
    config" while it writes the loader, so the folder name was choosing the architecture the loader
    got written against -- for precisely the non-transformers checkpoints that module serves."""
    from scripts.tt_hw_planner import reference_loader_resolver as R

    summary = R._config_summary(_checkpoint(tmp_path / folder, P.NATIVE_CONFIG_FILE, _NATIVE_DIALECT))

    assert str(_NATIVE_DIALECT["dim"]) in summary
    assert str(_NATIVE_DIALECT["n_layers"]) in summary
    # The native dialect declares no model_type; every invented transformers config prints one.
    assert "model_type" not in summary


def test_a_transformers_checkpoint_is_still_read_by_transformers(hf_checkpoint) -> None:
    """The other side of that gate. AutoConfig is still asked wherever it has a document to read,
    and its answer carries the defaults the raw file leaves out -- which the sizing steps rely on."""
    cfg = P._maybe_fetch_config(hf_checkpoint)

    assert cfg["model_type"] == _HF_DIALECT["model_type"]
    assert cfg["hidden_size"] == _HF_DIALECT["hidden_size"]
    assert len(cfg) > len(_HF_DIALECT), "no longer the transformers-expanded config"


def test_a_native_checkpoint_is_planned_like_a_named_one(native_checkpoint) -> None:
    """The end of the road: sized, fitted, and still called what it is."""
    with patch.object(P, "_agent_classify_category", return_value=None):
        probe = P.probe_model(native_checkpoint)

    assert probe.arch_spec is not None, "native checkpoint still skips the architecture step"
    assert probe.memory_model is not None, "native checkpoint still has no memory model to plan against"
    assert probe.arch_spec.num_layers == _NATIVE_DIALECT["n_layers"]
    assert probe.arch_spec.hidden_size == _NATIVE_DIALECT["dim"]
    assert probe.category != "CNN", "a sized transformer must not be reported as a vision model"


def test_both_checkpoint_dialects_plan_identically(native_checkpoint, hf_checkpoint) -> None:
    with patch.object(P, "_agent_classify_category", return_value=None):
        native = P.probe_model(native_checkpoint)
        hf = P.probe_model(hf_checkpoint)

    assert native.arch_spec.num_layers == hf.arch_spec.num_layers
    assert native.arch_spec.hidden_size == hf.arch_spec.hidden_size
    assert (native.memory_model is None) == (hf.memory_model is None)


# --------------------------------------------------------------------------------------
# One list, so they cannot drift apart again
# --------------------------------------------------------------------------------------


def test_what_counts_as_a_model_declaration_is_said_once() -> None:
    """Two answers to "is this a model" is what refused a checkpoint another module was reading."""
    package = Path(__file__).resolve().parents[1]
    sources = [p for p in package.rglob("*.py") if "tests" not in p.parts]

    for filename in P.MODEL_CONFIG_FILES:
        declarations = [p for p in sources if f'= "{filename}"' in p.read_text()]
        assert len(declarations) == 1, f"{filename} is declared in {[p.name for p in declarations]}"

    assert P.NATIVE_CONFIG_FILE in P.MODEL_CONFIG_FILES
    assert P.ROOT_CONFIG_FILE in P.MODEL_CONFIG_FILES


def test_the_loader_resolver_reads_the_same_list() -> None:
    from scripts.tt_hw_planner import reference_loader_resolver as RLR

    assert RLR._NATIVE_CONFIG_FILE == P.NATIVE_CONFIG_FILE


# --------------------------------------------------------------------------------------
# A word for every model the probe can recognise
# --------------------------------------------------------------------------------------


def test_onboarding_knows_every_category_the_probe_can_produce() -> None:
    """The hand-typed copy sat three categories behind and nothing noticed."""
    classifiable = {c for c in P._VALID_CATEGORIES if c != AO._UNCLASSIFIED_CATEGORY}

    assert AO._ALLOWED_CATEGORIES == classifiable


def test_an_unclassified_model_is_not_a_category() -> None:
    assert AO._UNCLASSIFIED_CATEGORY in P._VALID_CATEGORIES
    assert AO._UNCLASSIFIED_CATEGORY not in AO._ALLOWED_CATEGORIES


def _proposal(category: str) -> dict:
    return {
        "category": category,
        "name": "a new speech arch",
        "demo_path": "models/demos/auto_onboard/speech",
        "routing_mode": "template",
        "canonical_hf_id": "org/some-speech-model",
        "notes": "",
        "model_type_keys": ["some_speech_arch"],
        "pipeline_tags": ["text-to-speech"],
        "smoke_test_entry": None,
        "use_module_tree": True,
    }


@pytest.mark.parametrize("category", sorted({c for c in P._VALID_CATEGORIES if c != "Unknown"}))
def test_every_category_the_probe_can_produce_can_be_onboarded(category) -> None:
    errors = AO._validate_proposal(_proposal(category), new_model_type="some_speech_arch")

    assert not [e for e in errors if "category" in e], f"{category} has no word in onboarding"


def test_an_unclassified_proposal_is_still_refused() -> None:
    errors = AO._validate_proposal(_proposal(AO._UNCLASSIFIED_CATEGORY), new_model_type="some_speech_arch")

    assert [e for e in errors if "category" in e]


def test_the_offline_path_reports_what_the_probe_found(tmp_path) -> None:
    """It used to rewrite anything it had no word for into a fixed category.

    A missing entry in a list thereby became a confident claim about the model: a speech model was
    recorded as a vision one, and the run said nothing.
    """
    from types import SimpleNamespace

    speech = SimpleNamespace(
        model_id="org/some-speech-model",
        category="TTS",
        saved_dtype="float32",
        memory_model=None,
        raw_config={"model_type": "some_speech_arch"},
        pipeline_tag="text-to-speech",
    )
    with patch.object(AO, "probe_model", return_value=speech), patch.object(
        AO, "discover_components_from_hf_id", return_value=[]
    ):
        proposal = AO.auto_onboard("org/some-speech-model", skip_llm=True)

    assert proposal.backend_python_repr["category"] == "TTS"
    assert proposal.validation_errors == []


def test_the_prompt_offers_exactly_what_validation_accepts() -> None:
    """Typed separately, the enum went stale and never named speech, so no agent could propose it."""
    prompt = AO._build_prompt(
        model_id="org/some-speech-model",
        new_model_type="some_speech_arch",
        new_pipeline_tag="text-to-speech",
        inferred_category="TTS",
        components=[],
        closest_existing=None,
        closest_score=0.0,
    )

    offered = [line for line in prompt.splitlines() if '"category"' in line]
    assert len(offered) == 1
    for category in AO._ALLOWED_CATEGORIES:
        assert category in offered[0], f"the prompt never offers {category}"
    assert AO._UNCLASSIFIED_CATEGORY not in offered[0]

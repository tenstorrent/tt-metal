# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The batch's inputs are sourced from the model, not authored by the builder.

`_BATCH_COMMON_RULES` used to ask only for "{batch} DISTINCT reference inputs". DISTINCT was the
whole specification, so a Qwen-Image-Edit bring-up shipped 32 prompts the agent wrote. Every sample
that later missed the PCC bar was one of the written ones -- and because each sample varied its
content AND its seed together, no miss could be attributed to either, nor told apart from a hardware
fault. These tests pin the replacement: read the model's own published example, hold it fixed across
the batch, and vary only the seed.
"""

from __future__ import annotations

import textwrap

import pytest

from scripts.tt_hw_planner import reference_inputs as RI
from scripts.tt_hw_planner.commands import emit_e2e as E


class _Pipe:
    """Stand-in for a declared entry-point class, so positional names come from a real signature."""

    def __call__(self, image=None, prompt=None, num_inference_steps=None, generator=None):
        raise AssertionError("discovery must never execute the example")


def _parse(src: str, cls=_Pipe):
    return RI._parse_fence(textwrap.dedent(src), cls, "test")


# --- what the model publishes -------------------------------------------------------------------


def test_the_published_example_is_read_as_literals():
    got = _parse(
        """
        pipe = _Pipe.from_pretrained("some/model")
        pipe.to("cuda")
        image = load_image("https://example.invalid/a.png")
        prompt = "do the thing"
        out = pipe(image, prompt, num_inference_steps=50)
        """
    )
    assert got.kwargs["prompt"] == "do the thing"
    assert got.kwargs["num_inference_steps"] == 50
    assert got.kwargs["image"].target == "https://example.invalid/a.png"
    # `.to("cuda")` is housekeeping, not the invocation: the richest call on the object wins, and no
    # list of method names to ignore is kept anywhere.
    assert "cuda" not in str(got.kwargs)


def test_positional_arguments_are_named_from_the_entry_points_own_signature():
    """`pipe(image, prompt)` names nothing. The names come from the class the MODEL declares."""
    got = _parse(
        """
        pipe = _Pipe.from_pretrained("some/model")
        out = pipe(load_image("https://example.invalid/a.png"), "do the thing")
        """
    )
    assert set(got.kwargs) == {"image", "prompt"}


def test_a_chained_loader_reports_what_it_loads_not_how_it_converts():
    """`Image.open(p).convert("RGB")` names the file in the INNER call; the outer link is a no-op
    for provenance. Reporting the outer one recorded the image as its own colour space."""
    got = _parse(
        """
        pipe = _Pipe.from_pretrained("some/model")
        image = Image.open("./input.png").convert("RGB")
        out = pipe(image, "do the thing")
        """
    )
    assert got.kwargs["image"].target == "./input.png"


def test_an_example_that_names_a_file_the_repo_does_not_ship_is_not_self_contained():
    got = _parse(
        """
        pipe = _Pipe.from_pretrained("some/model")
        out = pipe(Image.open("./nonexistent-input.png"), "do the thing")
        """
    )
    assert not got.resolvable
    assert "NOT SELF-CONTAINED" in got.describe()


def test_a_url_is_self_contained():
    got = _parse(
        """
        pipe = _Pipe.from_pretrained("some/model")
        out = pipe(load_image("https://example.invalid/a.png"), "do the thing")
        """
    )
    assert got.resolvable and "NOT SELF-CONTAINED" not in got.describe()


def test_the_seed_is_found_wherever_the_example_puts_it():
    """The card that prompted this passes it inside the mapping it splats, not as a keyword."""
    inline = _parse(
        """
        pipe = _Pipe.from_pretrained("some/model")
        out = pipe(image=load_image("https://example.invalid/a.png"), generator=torch.manual_seed(7))
        """
    )
    splatted = _parse(
        """
        pipe = _Pipe.from_pretrained("some/model")
        inputs = {"image": load_image("https://example.invalid/a.png"), "generator": torch.manual_seed(7)}
        out = pipe(**inputs)
        """
    )
    assert inline.seed == splatted.seed == 7
    # the generator is the seed, not an input to hold fixed -- it must not survive as a kwarg
    assert "generator" not in inline.kwargs and "generator" not in splatted.kwargs


def test_the_example_is_never_executed():
    """It comes off the hub. `_Pipe.__call__` raises if anything runs it."""
    assert _parse(
        """
        pipe = _Pipe.from_pretrained("some/model")
        out = pipe(load_image("https://example.invalid/a.png"), "x")
        """
    )


def test_a_fence_that_loads_nothing_is_not_an_example():
    assert _parse("x = 1\ny = x + 1\n") is None


def test_a_fence_that_does_not_parse_is_skipped_not_raised():
    assert _parse("this is prose, not python ===\n") is None


def test_content_is_kept_when_the_example_preprocesses_before_calling_the_model():
    """An LLM/VLM card tokenises first, so the model's own call carries only generation settings.
    Taking the model call alone reported an example with no inputs in it."""
    got = _parse(
        """
        model = _Pipe.from_pretrained("some/model")
        tok = AutoTokenizer.from_pretrained("some/model")
        enc = tok("the actual content", return_tensors="pt")
        out = model.generate(**enc, max_new_tokens=128)
        """
    )
    assert got.kwargs["max_new_tokens"] == 128
    assert "the actual content" in str(got.kwargs)


def test_another_objects_call_is_not_named_with_this_class_signature():
    """When the example loads via an Auto* factory the spelled name is not the model's class, so
    positionals are labelled by position. Naming them from the declared class would attach one
    object's parameter names to another object's call."""
    got = _parse(
        """
        model = AutoModelForCausalLM.from_pretrained("some/model")
        tok = AutoTokenizer.from_pretrained("some/model")
        enc = tok("the actual content", return_tensors="pt")
        """
    )
    assert "image" not in got.kwargs and "prompt" not in got.kwargs
    assert any(k.startswith("tok[") for k in got.kwargs)


# --- what the builder is told -------------------------------------------------------------------


def _block(reference):
    return E._batch_input_block(32, reference)


def test_the_builder_is_told_to_vary_only_the_seed():
    for ref in (None, RI.ExampleInputs(kwargs={"prompt": "x"}, seed=0, sources=("t",))):
        text = _block(ref)
        assert "IDENTICAL across all 32 samples" in text
        assert "Vary ONLY the sampling axis" in text
        assert "Do NOT author 32 different content inputs" in text


def test_a_published_example_is_handed_over_verbatim_with_its_source():
    ref = RI.ExampleInputs(
        kwargs={"prompt": "do the thing", "image": RI.Resource("load_image", ("https://example.invalid/a.png",))},
        seed=0,
        sources=("some docstring", "seed from the card"),
    )
    text = _block(ref)
    assert "PUBLISHES ITS OWN EXAMPLE" in text
    assert "do the thing" in text and "https://example.invalid/a.png" in text
    assert "some docstring" in text and "seed from the card" in text
    assert "Sample 0 uses 0 EXACTLY" in text  # sample 0 reproduces the published example


def test_no_example_means_author_one_and_say_so():
    """The fallback the user asked for: ONE authored input, reused, marked as authored."""
    text = _block(None)
    assert "NO PUBLISHED EXAMPLE WAS DISCOVERED" in text
    assert "exactly ONE content input" in text
    assert "TOOL-AUTHORED" in text
    assert "32 authored inputs are 32 of them" in text


def test_an_example_with_braces_in_its_values_does_not_break_the_block():
    """A chat template's messages are dicts. str.format would read their braces as fields."""
    ref = RI.ExampleInputs(kwargs={"messages": [{"role": "user", "content": "hi"}]}, seed=None, sources=("t",))
    assert "'role': 'user'" in _block(ref)


def test_deterministic_models_are_given_the_fallback_axis_not_a_licence_to_invent():
    text = _block(None)
    assert "no sampling axis" in text and "LOADED DATA" in text
    assert "do NOT invent 32 inputs" in text


# --- the existing contract still holds ------------------------------------------------------------


def test_the_policy_reaches_the_builder_through_the_batch_block():
    block = E._batch_prompt_block(32, heads=[{"generates": True}])
    assert E._BATCH_COMMON_RULES.format(batch=32) in block  # the axis rules are unchanged
    assert "HOW TO SOURCE THE 32 INPUTS" in block


def test_b1_is_untouched():
    assert E._batch_prompt_block(1) == "" and E._batch_prompt_block(1, reference=None) == ""


def test_discovery_never_raises_and_none_is_a_valid_answer(monkeypatch):
    """Discovery reads the hub. A failure there must not take the bring-up down."""
    monkeypatch.setattr(RI, "discover", lambda mid: (_ for _ in ()).throw(RuntimeError("hub down")))
    assert E._discover_reference_inputs("some/model") is None


def test_json_fetching_is_unchanged_by_the_shared_text_reader(tmp_path):
    """fetch_repo_json now parses what fetch_repo_text returns -- same answers, one download path."""
    from scripts.tt_hw_planner.probe import fetch_repo_json, fetch_repo_text

    (tmp_path / "config.json").write_text('{"a": 1}')
    (tmp_path / "bad.json").write_text("not json")
    (tmp_path / "list.json").write_text("[1, 2]")
    assert fetch_repo_json(str(tmp_path), "config.json") == {"a": 1}
    assert fetch_repo_json(str(tmp_path), "bad.json") is None  # unparseable -> None, as before
    assert fetch_repo_json(str(tmp_path), "list.json") is None  # non-dict -> None, as before
    assert fetch_repo_json(str(tmp_path), "missing.json") is None
    assert fetch_repo_text(str(tmp_path), "bad.json") == "not json"
    assert fetch_repo_text(str(tmp_path), "missing.json") is None


def test_a_local_model_directory_is_read_without_the_hub(tmp_path):
    """End to end on a directory: card fence in, example out, no network."""
    (tmp_path / "config.json").write_text('{"model_type": "x", "transformers_version": "4.0"}')
    (tmp_path / "README.md").write_text(
        "---\nlibrary_name: nonexistent_library_xyz\n---\n\n"
        '```python\npipe = Thing.from_pretrained("m")\nout = pipe(prompt="hello", generator=torch.manual_seed(3))\n```\n'
    )
    got = RI.discover(str(tmp_path))
    assert got.kwargs["prompt"] == "hello" and got.seed == 3
    assert RI.CARD_FILE in got.sources[0]


def test_the_discovery_code_names_no_model_or_stage():
    """Constraint: names come from the model's documents, never from a string typed here.

    The executable body is scanned, not the prose: a docstring may cite the bring-up this came
    from, but no branch may depend on the word."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(RI))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    lowered = ast.unparse(tree).lower()
    for name in ("qwen", "flux", "llama", "whisper", "diffusion", "unet", "vae", "prompt", "image", "decode"):
        assert name not in lowered, f"{name!r} is a name discovery must read, never assume"


def test_the_policy_text_names_no_stage_either():
    for block in (E._BATCH_INPUT_RULES, E._BATCH_INPUT_UNPUBLISHED, E._BATCH_INPUT_PUBLISHED):
        lowered = block.lower()
        for name in ("denoise", "diffusion", "vae", "unet", "vocoder", "prefill", "decoder"):
            assert name not in lowered, f"{name!r} would hardcode a stage name into the guidance"

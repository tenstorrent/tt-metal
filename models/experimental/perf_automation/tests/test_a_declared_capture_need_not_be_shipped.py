"""A model that declares its reference tensors need not ship them.

A pipeline's <stage>_trace_inputs() reads golden tensors a bring-up capture wrote into `_captured/`.
Those are large and are NOT committed -- only the manifest beside them is -- so on a tree that has
never run that capture, which is every model being optimised for the first time, every stage raises
FileNotFoundError and the per-stage split is lost. voxtral run 35, all three stages:

    [perf-adapter] stage 'encode'  cannot prepare its own inputs (FileNotFoundError ... args.pt)
    [perf-adapter] stage 'prefill' cannot prepare its own inputs (FileNotFoundError ... output.pt)
    STAGE_MARKS_RESULT=0

The earlier answers all leaned on the generated test to supply inputs instead -- find its preparer by
name, then by signature, then by scope, then fill that preparer's arguments from the scope. Each one
worked on the file in front of it and broke on the next, because the test is written fresh by a model
each time. This leans on nothing the generator writes: the manifest is the MODEL's own artifact, it
is committed, and it declares the shape and dtype of exactly what the hook would have loaded. Values
are irrelevant to a timing measurement.
"""
import json

import pytest

from agent.captured_stub import build, for_missing_file, install, missing_path

torch = pytest.importorskip("torch")


def _manifest(tmp_path, doc, name="voxtral_encoder"):
    d = tmp_path / "_captured" / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(doc))
    return d


_REAL = {
    "component": "voxtral_encoder",
    "submodule_path": "model.audio_tower",
    "args": {"kind": "tuple", "items": [{"kind": "tensor", "shape": [1, 128, 3000], "dtype": "torch.bfloat16"}]},
    "output": {
        "kind": "dict",
        "items": {
            "last_hidden_state": {"kind": "tensor", "shape": [1, 1500, 1280], "dtype": "torch.bfloat16"},
            "pooler_output": {"kind": "tensor", "shape": [375, 3072], "dtype": "torch.bfloat16"},
        },
    },
}


def test_the_declared_args_are_rebuilt_with_their_own_shape(tmp_path):
    d = _manifest(tmp_path, _REAL)
    obj, why = for_missing_file(d / "args.pt")
    assert isinstance(obj, tuple) and len(obj) == 1
    assert tuple(obj[0].shape) == (1, 128, 3000)
    assert obj[0].dtype is torch.bfloat16
    assert "synthesised" in why


def test_a_nested_output_is_rebuilt_field_by_field(tmp_path):
    d = _manifest(tmp_path, _REAL)
    obj, _ = for_missing_file(d / "output.pt")
    assert set(obj) == {"last_hidden_state", "pooler_output"}
    assert tuple(obj["last_hidden_state"].shape) == (1, 1500, 1280)
    assert tuple(obj["pooler_output"].shape) == (375, 3072)


def test_the_field_is_chosen_by_the_file_it_stands_in_for(tmp_path):
    """args.pt is the manifest's `args`, output.pt its `output` -- the capture's own naming."""
    d = _manifest(tmp_path, _REAL)
    assert isinstance(for_missing_file(d / "args.pt")[0], tuple)
    assert isinstance(for_missing_file(d / "output.pt")[0], dict)


def test_a_file_the_manifest_does_not_describe_is_refused_by_name(tmp_path):
    d = _manifest(tmp_path, _REAL)
    obj, why = for_missing_file(d / "hidden.pt")
    assert obj is None
    assert "hidden" in why and "describes" in why


def test_no_manifest_means_no_stand_in(tmp_path):
    d = tmp_path / "_captured" / "x"
    d.mkdir(parents=True)
    obj, why = for_missing_file(d / "args.pt")
    assert obj is None and "no manifest" in why


def test_an_existing_file_is_never_replaced(tmp_path):
    """This supplies what was never shipped; it must not shadow a real capture."""
    d = _manifest(tmp_path, _REAL)
    real = d / "args.pt"
    torch.save({"i am": "real"}, real)
    restore = install()
    try:
        assert torch.load(real, weights_only=False) == {"i am": "real"}
    finally:
        restore()


def test_a_missing_file_is_supplied_through_torch_load(tmp_path, capsys):
    """The hooks are the model's code and every one of them differs; torch.load is the single call
    they all make, which is why the stand-in goes there rather than into each hook."""
    d = _manifest(tmp_path, _REAL)
    restore = install()
    try:
        got = torch.load(d / "args.pt", weights_only=False)
    finally:
        restore()
    assert tuple(got[0].shape) == (1, 128, 3000)
    assert "synthesised args.pt" in capsys.readouterr().err


def test_the_patch_is_removed_afterwards(tmp_path):
    """Installed around the stage walk only -- the rest of the run must see plain torch.load."""
    before = torch.load
    restore = install()
    assert torch.load is not before
    restore()
    assert torch.load is before


def test_a_missing_file_with_no_manifest_still_raises(tmp_path):
    """Absent evidence is not a licence to invent one."""
    d = tmp_path / "_captured" / "y"
    d.mkdir(parents=True)
    restore = install()
    try:
        with pytest.raises(FileNotFoundError):
            torch.load(d / "args.pt", weights_only=False)
    finally:
        restore()


def test_the_path_is_taken_from_the_exception(tmp_path):
    err = FileNotFoundError(2, "No such file or directory")
    err.filename = str(tmp_path / "_captured" / "z" / "args.pt")
    assert missing_path(err).name == "args.pt"


def test_scalars_and_unknown_kinds_pass_through():
    assert build({"kind": "int", "value": 7}) == 7
    assert build("not a spec") == "not a spec"


def test_the_buffers_are_deterministic(tmp_path):
    """Two runs of the same stage should be comparable; random values would make them not."""
    d = _manifest(tmp_path, _REAL)
    a, _ = for_missing_file(d / "args.pt")
    b, _ = for_missing_file(d / "args.pt")
    assert torch.equal(a[0], b[0])

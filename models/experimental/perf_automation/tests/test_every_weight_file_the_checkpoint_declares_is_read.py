"""The weight readers follow the checkpoint's own layout, and a file they miss is reported, not ignored.

Qwen-Image-Edit (2026-09-29) is a diffusers pipeline: model_index.json lists its components and each
keeps its weights in the subfolder of that name -- transformer/ 9 files, text_encoder/ 4, vae/ 1, none
at the top. Every reader globbed the top level only, so the run had no param count and every roofline
value read "n/a -- not measured". A flat checkpoint must read exactly as it did before.
Component names here are the fixture's own; the code under test takes them from model_index.json.
"""

import importlib.util
import json
import struct
import subprocess
from pathlib import Path

from agent import model_bytes as mb

_PA = Path(__file__).resolve().parents[1]


def _st(path: Path, tensors: dict) -> None:
    """A minimal safetensors file: 8-byte header length, JSON header, no payload needed."""
    hdr, off = {}, 0
    for k, s in tensors.items():
        n = 2
        for dim in s:
            n *= dim
        hdr[k] = {"dtype": "BF16", "shape": list(s), "data_offsets": [off, off + n]}
        off += n
    hdr["__metadata__"] = {"format": "pt"}
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def _flat(tmp_path):
    d = tmp_path / "flat"
    _st(d / "model-00001-of-00002.safetensors", {"model.layers.0.w": (64, 32), "lm_head.weight": (10, 32)})
    _st(d / "model-00002-of-00002.safetensors", {"model.layers.1.w": (64, 32)})
    return d


def _components(tmp_path):
    d = tmp_path / "pipe"
    (d).mkdir()
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "SomePipeline", "alpha": ["lib", "A"], "beta": ["lib", "B"], "sched": ["lib", "S"]})
    )
    _st(d / "alpha" / "diffusion_pytorch_model-00001-of-00002.safetensors", {"blocks.0.w": (128, 64)})
    _st(d / "alpha" / "diffusion_pytorch_model-00002-of-00002.safetensors", {"blocks.1.w": (128, 64)})
    _st(d / "beta" / "model.safetensors", {"layers.0.w": (32, 32)})
    (d / "sched").mkdir()  # a component with no weights
    return d


def _old_module():
    src = subprocess.run(
        ["git", "show", "HEAD:models/experimental/perf_automation/agent/model_bytes.py"],
        cwd=_PA,
        capture_output=True,
        text=True,
    )
    if src.returncode != 0:
        return None
    p = _PA / "tests" / "_model_bytes_before.py"
    p.write_text(src.stdout)
    try:
        spec = importlib.util.spec_from_file_location("_model_bytes_before", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        p.unlink()


def test_a_flat_checkpoint_reads_exactly_as_before(tmp_path):
    d = _flat(tmp_path)
    old = _old_module()
    assert old is not None, "the pre-change reader must be available to compare against"
    for unit in ("token", "step"):
        new = mb.weight_bytes(d, unit=unit)
        assert new["params"] == 64 * 32 * 2 + 10 * 32
        assert new == old.weight_bytes(d, unit=unit), unit


def test_each_declared_component_is_read_under_its_own_name(tmp_path):
    d = _components(tmp_path)
    files = mb.weight_files(d)
    assert [(p, f.parent.name) for p, f in files] == [("alpha.", "alpha"), ("alpha.", "alpha"), ("beta.", "beta")]
    r = mb.weight_bytes(d, unit="step")
    assert r["params"] == 128 * 64 * 2 + 32 * 32 and r["shards"] == 3


def test_nothing_is_unread_for_either_layout(tmp_path):
    assert mb.unread_weight_files(_flat(tmp_path)) == []
    assert mb.unread_weight_files(_components(tmp_path)) == []


def test_a_weight_file_no_reader_reaches_is_reported(tmp_path):
    d = _components(tmp_path)
    _st(d / "undeclared" / "extra.safetensors", {"x": (4, 4)})  # a folder model_index.json does not name
    (d / "beta" / "legacy.bin").write_bytes(b"0")  # a stored-weight file the header walk cannot read
    assert mb.unread_weight_files(d) == ["undeclared/extra.safetensors"]
    assert "beta/legacy.bin" not in mb.unread_weight_files(d), "the size sum does count .bin files"


def test_sections_read_the_same_as_the_equivalent_flat_checkpoint(tmp_path):
    """A component's tensors carry its name, so tower detection sees exactly what it would if the same
    tensors sat in one flat file under their full module paths."""
    d = _components(tmp_path)
    flat = tmp_path / "same_flat"
    _st(
        flat / "model.safetensors",
        {"alpha.blocks.0.w": (128, 64), "alpha.blocks.1.w": (128, 64), "beta.layers.0.w": (32, 32)},
    )
    roots = {"s_big": "alpha.blocks", "s_small": "beta.layers"}
    assert mb.untowered_sections(d, roots) == mb.untowered_sections(flat, roots)
    assert mb.weight_bytes(d, unit="step")["params"] == mb.weight_bytes(flat, unit="step")["params"]


def test_the_size_sum_counts_component_files(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("_run_for_sizes", _PA / "cc_optimize" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    d = _components(tmp_path)
    monkeypatch.setattr(run, "_hf_snapshots", lambda mid: [d])
    expected = sum(f.stat().st_size for _, f in mb.weight_files(d, mb._WEIGHT_SUFFIXES))
    assert expected > 0 and run._hf_cache_weight_bytes("org/name") == expected


def test_the_facts_carry_every_unread_file():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _perf_target_inputs(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "_mb.unread_weight_files(_snap)" in body
    assert 'facts["weights_unread"]' in body and "ERROR" in body


def test_section_bytes_reads_the_components_as_their_own_sections(tmp_path):
    from agent.checkpoint_sections import section_bytes

    d = _components(tmp_path)
    flat = tmp_path / "same_flat"
    _st(
        flat / "model.safetensors",
        {"alpha.blocks.0.w": (128, 64), "alpha.blocks.1.w": (128, 64), "beta.layers.0.w": (32, 32)},
    )
    assert section_bytes(d) == section_bytes(flat) == {"alpha": 2 * 128 * 64 * 2, "beta": 32 * 32 * 2}
    assert section_bytes(_flat(tmp_path)) == {"model": (64 * 32 * 2) * 2, "lm_head": 10 * 32 * 2}

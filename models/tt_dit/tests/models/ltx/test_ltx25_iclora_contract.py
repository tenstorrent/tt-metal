# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests: real public wrapper, synthetic assets, no native imports."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

SOURCE = Path(__file__).resolve().parents[3] / "pipelines/ltx/pipeline_ltx25_iclora.py"


def load_module():
    spec = importlib.util.spec_from_file_location("iclora_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_callable_is_cpu_importable():
    assert callable(load_module().LTX25ICLoRAPipeline.create_pipeline)


@pytest.fixture
def contract(tmp_path, monkeypatch):
    for name in os.environ:
        if name.startswith(("LTX_", "LTX25_")) or name == "RUN_I2V":
            monkeypatch.delenv(name)
    module = load_module()
    example = SOURCE.parents[2] / "tests/fixtures/ltx25_iclora/fixture.example.json"
    fixture = json.loads(example.read_text())
    fixture["resolved"] = True

    def bind(name, image=False):
        path = tmp_path / (name + (".png" if image else ".bin"))
        if image:
            Image.new("RGB", (128, 128), "red").save(path)
        else:
            path.write_bytes(name.encode())
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    for name, asset in fixture["assets"].items():
        asset.update(bind(name), revision="a" * 40)
    fixture["fusion"]["fused_sha256"] = fixture["assets"]["transformer"]["sha256"]
    fixture["fusion"]["base"] = {"sha256": "b" * 64, "revision": "b" * 40}
    fixture["fusion"]["recipe"] = {"tool": "synthetic-test-only", "revision": "c" * 40, "description": "test double"}
    fixture["reference"].update(bind("sheet", True), permission="synthetic test image")
    fixture["generation"].update(height=128, width=128, num_frames=9, ref_frames=9, quality_variant="synthetic-test-only")
    fixture["acceptance"].update(prompt="synthetic", seeds=[10, 11, 12], criteria="synthetic tests only")
    fixture["acceptance"]["quality_reference"] = {"kind": "primary_recipe", "revision": "d" * 40, "uri": "synthetic:test"}
    fixture["acceptance"]["first_frame"] = dict(bind("first", True), permission="synthetic")
    fixture["acceptance"]["last_frame"] = dict(bind("last", True), permission="synthetic")
    path = tmp_path / "fixture.json"

    class NativeDouble:
        def __init__(self):
            self._i2v_cond_cache = {}
            self._ref_latent_cache = {}
            self.calls = []
            self.result = (object(), object())
            self.error = None

        def warmup_buffers(self, **kwargs):
            self.warmup = kwargs

        def generate(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            # Read exactly as the real path/resolution cache does: stale data is observable here.
            for image, _, _, _ in kwargs["images"] or []:
                for h, w in ((64, 64), (128, 128)):
                    key = (image, h, w)
                    fresh = Path(image).read_bytes()
                    assert self._i2v_cond_cache.setdefault(key, fresh) == fresh
            if self.error:
                raise self.error
            return kwargs["output_path"] if kwargs["output_path"] is not None else self.result

    native = NativeDouble()

    def factory(mesh, **kwargs):
        native.factory = kwargs
        return native

    monkeypatch.setattr(module, "_native_factory", lambda: factory)

    def create(**kwargs):
        path.write_text(json.dumps(fixture))
        return module.LTX25ICLoRAPipeline.create_pipeline(None, fixture_path=path, **kwargs)

    return SimpleNamespace(module=module, fixture=fixture, path=path, native=native, create=create, bind=bind)


def test_fused_conv_warmup_and_r2v_wiring(contract):
    c = contract
    pipe = c.create(traced=True)
    f = c.native.factory
    for name in c.fixture["assets"]:
        assert f[name] == c.fixture["assets"][name]["path"]
    assert f["diffusion_decoder"] is False
    assert f["run_warmup"] is False
    assert f["use_ancestral_sampler"] is False
    assert f["fps"] == 24
    assert "extra_transformer_variants" not in f
    pipe.warmup_buffers()
    assert c.native.warmup == dict(num_frames=9, height=128, width=128, ref_num_frames=9, stages=("s1_ref", "s2_ref"))
    first = c.fixture["acceptance"]["first_frame"]["path"]
    last = c.fixture["acceptance"]["last_frame"]["path"]
    assert pipe.generate_r2v("hello", first_frame=first, last_frame=last, seed=42) is c.native.result
    prompt, kw = c.native.calls[-1]
    assert prompt == "hello"
    assert kw == dict(output_path=None, output_type="rgb", images=[(first, 0, 1.0, 1.0), (last, 8, 1.0, 1.0)], reference_video=(c.fixture["reference"]["path"], 1.0), ref_frames=9, num_frames=9, height=128, width=128, fps=24, seed=42)
    assert pipe.generate_r2v("export", seed=1, output_path="once.mp4") == "once.mp4"
    assert len(c.native.calls) == 2
    for output_type in ("float", "yuv"):
        assert pipe.generate_r2v("frames", seed=1, output_type=output_type) is c.native.result
        assert c.native.calls[-1][1]["output_type"] == output_type
    resolved = pipe.resolved_fixture
    resolved["generation"]["fps"] = 99
    assert pipe.resolved_fixture["generation"]["fps"] == 24


@pytest.mark.parametrize("target,value", [
    (("resolved",), False), (("schema_version",), 2),
    (("assets", "video_vae", "kind"), "diffusion"), (("assets", "video_vae", "model_version"), "2.3"),
    (("assets", "transformer", "sha256"), "0" * 64), (("fusion", "fused_sha256"), "0" * 64),
    (("fusion", "adapter", "sha256"), "0" * 64), (("fusion", "adapter", "size_bytes"), 1),
    (("fusion", "recipe", "revision"), None), (("reference", "permission"), ""),
    (("generation", "height"), 127), (("generation", "num_frames"), 8),
    (("generation", "ref_frames"), 17), (("generation", "fps"), float("nan")),
    (("generation", "sampler"), "ancestral"), (("generation", "s1_sigmas"), [1, 0]),
    (("generation", "reference_strength"), -1), (("generation", "pin_s2_strength"), 2),
    (("acceptance", "seeds"), [1]), (("environment", "LTX_REF_IMG0"), "1"),
])
def test_invalid_fixture_rejected_before_native(contract, target, value):
    node = contract.fixture
    for key in target[:-1]:
        node = node[key]
    node[target[-1]] = value
    with pytest.raises((ValueError, FileNotFoundError)):
        contract.create()
    assert not hasattr(contract.native, "factory")


def test_bare_base_and_mutated_file_rejected(contract):
    c = contract
    c.fixture["fusion"]["base"]["sha256"] = c.fixture["assets"]["transformer"]["sha256"]
    with pytest.raises(ValueError, match="base"):
        c.create()
    c.fixture["fusion"]["base"]["sha256"] = "b" * 64
    Path(c.fixture["assets"]["audio_vae"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        c.create()


@pytest.mark.parametrize("option", ["transformer", "fps", "diffusion_decoder", "extra_transformer_variants", "run_warmup", "pipeline_class"])
def test_runtime_override_rejected(contract, option):
    with pytest.raises(ValueError, match="runtime"):
        contract.create(**{option: True})


@pytest.mark.parametrize("name,value", [("LTX_KF_APPEND_TOKEN", "True"), ("LTX25_VAE_FALLBACK_23", "1"), ("LTX25_TEXT_STACK", "gemma3"), ("LTX_S1_SIGMAS", "1,0"), ("LTX_VIDEO_ONLY", "1"), ("LTX_UNKNOWN_RECIPE", "1")])
def test_inherited_overrides_and_post_creation_drift_rejected(contract, monkeypatch, name, value):
    pipe = contract.create()
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="environment"):
        contract.create()
    with pytest.raises(ValueError, match="environment"):
        pipe.generate_r2v("x", seed=1)
    with pytest.raises(ValueError, match="environment"):
        pipe.warmup_buffers()
    assert not contract.native.calls


def test_endpoint_invalidation_precedes_consumption_and_state_is_bounded(contract):
    c = contract
    pipe = c.create()
    fixed_ref = ("digest-not-path", 9, 128, 128)
    c.native._ref_latent_cache[fixed_ref] = (object(), object())
    preserved = c.native._ref_latent_cache[fixed_ref]
    first = c.fixture["acceptance"]["first_frame"]["path"]
    for i in range(20):
        Image.new("RGB", (128, 128), (i, 0, 0)).save(first)
        last = c.bind("last" + str(i), True)["path"]
        pipe.generate_r2v("x", first_frame=first, last_frame=last, seed=i)
        assert len(c.native._i2v_cond_cache) == 4
        assert len(pipe._endpoint_digests) == 2
        assert c.native._ref_latent_cache == {fixed_ref: preserved}
    pipe.generate_r2v("x", seed=1)
    assert not c.native._i2v_cond_cache
    assert not pipe._endpoint_digests


@pytest.mark.parametrize("error", [RuntimeError("unexpected native failure"), KeyboardInterrupt()])
def test_failure_or_interrupt_cannot_leave_stale_or_unbounded_state(contract, error):
    c = contract
    pipe = c.create()
    for i in range(5):
        first = c.bind("failure" + str(i), True)["path"]
        c.native.error = error
        with pytest.raises(type(error)):
            pipe.generate_r2v("x", first_frame=first, seed=i)
        assert not c.native._i2v_cond_cache
        assert not pipe._endpoint_digests


def test_reference_mutation_fl2v_and_bad_endpoint(contract):
    c = contract
    c.fixture["generation"]["mode"] = "fl2v"
    pipe = c.create()
    with pytest.raises(ValueError, match="both"):
        pipe.generate_r2v("x", seed=1)
    first = c.fixture["acceptance"]["first_frame"]["path"]
    Image.new("RGB", (64, 64)).save(first)
    with pytest.raises(ValueError, match="geometry"):
        pipe.generate_r2v("x", seed=1, first_frame=first, last_frame=first)
    Image.new("RGB", (128, 128), "blue").save(c.fixture["reference"]["path"])
    with pytest.raises(ValueError, match="digest"):
        pipe.warmup_buffers()


def test_example_is_explicitly_unresolved():
    module = load_module()
    with pytest.raises(ValueError, match="unresolved"):
        module.load_fixture(SOURCE.parents[2] / "tests/fixtures/ltx25_iclora/fixture.example.json")

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests: real public wrapper, synthetic assets, no native imports."""
import ast
import hashlib
import importlib.util
import json
import os
import sys
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
    fixture["generation"].update(
        height=128, width=128, num_frames=9, ref_frames=9, quality_variant="synthetic-test-only"
    )
    fixture["acceptance"].update(prompt="synthetic", seeds=[10, 11, 12], criteria="synthetic tests only")
    fixture["acceptance"]["quality_reference"] = {
        "kind": "primary_recipe",
        "revision": "d" * 40,
        "uri": "synthetic:test",
    }
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

    raw_native_factory = module._native_factory
    monkeypatch.setattr(module, "_native_factory", lambda: factory)

    def create(**kwargs):
        path.write_text(json.dumps(fixture))
        return module.LTX25ICLoRAPipeline.create_pipeline(None, fixture_path=path, **kwargs)

    return SimpleNamespace(
        module=module,
        fixture=fixture,
        path=path,
        native=native,
        create=create,
        bind=bind,
        raw_native_factory=raw_native_factory,
    )


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
    assert kw == dict(
        output_path=None,
        output_type="rgb",
        images=[(first, 0, 1.0, 1.0), (last, 8, 1.0, 1.0)],
        reference_video=(c.fixture["reference"]["path"], 1.0),
        ref_frames=9,
        num_frames=9,
        height=128,
        width=128,
        fps=24,
        seed=42,
    )
    assert pipe.generate_r2v("export", seed=1, output_path="once.mp4") == "once.mp4"
    assert len(c.native.calls) == 2
    for output_type in ("float", "yuv"):
        assert pipe.generate_r2v("frames", seed=1, output_type=output_type) is c.native.result
        assert c.native.calls[-1][1]["output_type"] == output_type
    resolved = pipe.resolved_fixture
    resolved["generation"]["fps"] = 99
    assert pipe.resolved_fixture["generation"]["fps"] == 24


@pytest.mark.parametrize(
    "target,value",
    [
        (("resolved",), False),
        (("schema_version",), 2),
        (("assets", "video_vae", "kind"), "diffusion"),
        (("assets", "video_vae", "model_version"), "2.3"),
        (("assets", "transformer", "sha256"), "0" * 64),
        (("fusion", "fused_sha256"), "0" * 64),
        (("fusion", "adapter", "sha256"), "0" * 64),
        (("fusion", "adapter", "size_bytes"), 1),
        (("fusion", "recipe", "revision"), None),
        (("reference", "permission"), ""),
        (("generation", "height"), 127),
        (("generation", "num_frames"), 8),
        (("generation", "ref_frames"), 17),
        (("generation", "fps"), float("nan")),
        (("generation", "sampler"), "ancestral"),
        (("generation", "s1_sigmas"), [1, 0]),
        (("generation", "reference_strength"), -1),
        (("generation", "pin_s2_strength"), 2),
        (("acceptance", "seeds"), [1]),
        (("environment", "LTX_REF_IMG0"), "1"),
    ],
)
def test_invalid_fixture_rejected_before_native(contract, target, value):
    node = contract.fixture
    for key in target[:-1]:
        node = node[key]
    node[target[-1]] = value
    with pytest.raises((ValueError, FileNotFoundError)):  # allow-pytest.raises: CPU-only run excludes native conftest
        contract.create()
    assert not hasattr(contract.native, "factory")


def test_bare_base_and_mutated_file_rejected(contract):
    c = contract
    c.fixture["fusion"]["base"]["sha256"] = c.fixture["assets"]["transformer"]["sha256"]
    with pytest.raises(ValueError, match="base"):  # allow-pytest.raises: CPU-only run excludes native conftest
        c.create()
    c.fixture["fusion"]["base"]["sha256"] = "b" * 64
    Path(c.fixture["assets"]["audio_vae"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):  # allow-pytest.raises: CPU-only run excludes native conftest
        c.create()


@pytest.mark.parametrize(
    "option", ["transformer", "fps", "diffusion_decoder", "extra_transformer_variants", "run_warmup", "pipeline_class"]
)
def test_runtime_override_rejected(contract, option):
    with pytest.raises(ValueError, match="runtime"):  # allow-pytest.raises: CPU-only run excludes native conftest
        contract.create(**{option: True})


@pytest.mark.parametrize(
    "name,value",
    [
        ("LTX_KF_APPEND_TOKEN", "True"),
        ("LTX25_VAE_FALLBACK_23", "1"),
        ("LTX25_TEXT_STACK", "gemma3"),
        ("LTX_S1_SIGMAS", "1,0"),
        ("LTX_VIDEO_ONLY", "1"),
        ("LTX_UNKNOWN_RECIPE", "1"),
    ],
)
def test_inherited_overrides_and_post_creation_drift_rejected(contract, monkeypatch, name, value):
    pipe = contract.create()
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="environment"):  # allow-pytest.raises: CPU-only run excludes native conftest
        contract.create()
    with pytest.raises(ValueError, match="environment"):  # allow-pytest.raises: CPU-only run excludes native conftest
        pipe.generate_r2v("x", seed=1)
    with pytest.raises(ValueError, match="environment"):  # allow-pytest.raises: CPU-only run excludes native conftest
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
        with pytest.raises(type(error)):  # allow-pytest.raises: CPU-only run excludes native conftest
            pipe.generate_r2v("x", first_frame=first, seed=i)
        assert not c.native._i2v_cond_cache
        assert not pipe._endpoint_digests


def test_reference_mutation_fl2v_and_bad_endpoint(contract):
    c = contract
    c.fixture["generation"]["mode"] = "fl2v"
    pipe = c.create()
    with pytest.raises(ValueError, match="both"):  # allow-pytest.raises: CPU-only run excludes native conftest
        pipe.generate_r2v("x", seed=1)
    first = c.fixture["acceptance"]["first_frame"]["path"]
    Image.new("RGB", (64, 64)).save(first)
    with pytest.raises(ValueError, match="geometry"):  # allow-pytest.raises: CPU-only run excludes native conftest
        pipe.generate_r2v("x", seed=1, first_frame=first, last_frame=first)
    Image.new("RGB", (128, 128), "blue").save(c.fixture["reference"]["path"])
    with pytest.raises(ValueError, match="digest"):  # allow-pytest.raises: CPU-only run excludes native conftest
        pipe.warmup_buffers()


def test_example_is_explicitly_unresolved():
    module = load_module()
    with pytest.raises(ValueError, match="unresolved"):  # allow-pytest.raises: CPU-only run excludes native conftest
        module.load_fixture(SOURCE.parents[2] / "tests/fixtures/ltx25_iclora/fixture.example.json")


@pytest.mark.parametrize("form", ["duplicate", "unknown", "boolean_version", "oversized"])
def test_manifest_parser_bypasses(contract, form):
    c = contract
    c.create()
    data = c.path.read_text()
    if form == "duplicate":
        data = data.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    elif form == "unknown":
        value = json.loads(data)
        value["generation"]["extra_transformer_variants"] = ["bare"]
        data = json.dumps(value)
    elif form == "boolean_version":
        data = data.replace('"schema_version": 1', '"schema_version": true')
    else:
        data += " " * (1024 * 1024)
    c.path.write_text(data)
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only run excludes native conftest
        c.module.load_fixture(c.path)


def test_relative_paths_animation_and_missing_assets(contract):
    c = contract
    for asset in c.fixture["assets"].values():
        asset["path"] = Path(asset["path"]).name
    assert c.create().resolved_fixture["assets"]["video_vae"]["path"].startswith(str(c.path.parent))
    sheet = c.path.parent / "animated.gif"
    Image.new("RGB", (128, 128), "red").save(sheet, save_all=True, append_images=[Image.new("RGB", (128, 128), "blue")])
    c.fixture["reference"].update(path=str(sheet), sha256=hashlib.sha256(sheet.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="still"):  # allow-pytest.raises: CPU-only run excludes native conftest
        c.create()
    del c.fixture["assets"]["video_vae"]
    with pytest.raises(ValueError, match="five"):  # allow-pytest.raises: CPU-only run excludes native conftest
        c.create()


def test_concurrent_session_is_refused_without_wait(contract):
    pipe = contract.create()
    pipe._call_lock.acquire()
    try:
        with pytest.raises(  # allow-pytest.raises: CPU-only conftest
            ValueError, match="concurrent"
        ):  # allow-pytest.raises: CPU-only run excludes native conftest
            pipe.generate_r2v("x", seed=1)
        with pytest.raises(  # allow-pytest.raises: CPU-only conftest
            ValueError, match="concurrent"
        ):  # allow-pytest.raises: CPU-only run excludes native conftest
            pipe.warmup_buffers()
    finally:
        pipe._call_lock.release()


def test_native_factory_and_imported_schedule_drift(contract, monkeypatch):
    c = contract
    package = SimpleNamespace()
    distilled = SimpleNamespace(
        DISTILLED_SIGMA_VALUES=list(c.module.S1_SIGMAS),
        STAGE_2_DISTILLED_SIGMA_VALUES=list(c.module.S2_SIGMAS),
        KEYFRAME_S1_STRENGTH=0.5,
        KEYFRAME_S1_STRENGTH_COARSE=0.25,
    )
    factory = object()
    package.pipeline_ltx_distilled = distilled
    monkeypatch.setattr(c.module, "__package__", "synthetic_ltx")
    monkeypatch.setitem(sys.modules, "synthetic_ltx", package)
    monkeypatch.setitem(
        sys.modules,
        "synthetic_ltx.pipeline_ltx25_distilled",
        SimpleNamespace(LTX25DistilledPipeline=SimpleNamespace(create_pipeline=factory)),
    )
    assert c.raw_native_factory() is factory
    for field, wrong in [
        ("DISTILLED_SIGMA_VALUES", [1, 0]),
        ("STAGE_2_DISTILLED_SIGMA_VALUES", [1, 0]),
        ("KEYFRAME_S1_STRENGTH", 0.1),
        ("KEYFRAME_S1_STRENGTH_COARSE", 0.1),
    ]:
        original = getattr(distilled, field)
        setattr(distilled, field, wrong)
        with pytest.raises(ValueError, match="imported"):  # allow-pytest.raises: CPU-only run excludes native conftest
            c.raw_native_factory()
        setattr(distilled, field, original)


def test_existing_ltx25_reference_sampler_is_deterministic(monkeypatch):
    # Execute the unchanged native override against a recorder, without importing ttnn.
    import textwrap

    path = SOURCE.with_name("pipeline_ltx25_distilled.py")
    source = path.read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "LTX25DistilledPipeline")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_denoise_no_guidance")
    method_source = ast.get_source_segment(source, method)
    monkeypatch.setitem(
        sys.modules, "sampler_test.pipeline_ltx_distilled", SimpleNamespace(ANCESTRAL_NOISE_SEED_OFFSET=42)
    )

    class Recorder:
        def _denoise_no_guidance(self, *args, **kwargs):
            return kwargs

    namespace = dict(__package__="sampler_test", Recorder=Recorder)
    exec("class NativeOverride(Recorder):\n" + textwrap.indent(method_source, "    "), namespace)
    native = namespace["NativeOverride"]()
    native.use_ancestral_sampler = True
    assert native._denoise_no_guidance(None, None, trace_key="s1")["ancestral"] is True
    for trace_key in ("s1_ref", "s2_ref"):
        assert native._denoise_no_guidance(None, None, trace_key=trace_key)["ancestral"] is False


@pytest.fixture
def constructor_native(contract, monkeypatch):
    """Execute the native factories and constructors; stub only their collaborators."""
    import weakref
    from contextlib import nullcontext

    events = []
    state = SimpleNamespace(cold=False, capture_only=False, error=None)

    def capture_only(enabled):
        state.capture_only = enabled
        events.append(("capture_only", enabled))

    kp = SimpleNamespace(
        kernel_prewarm_cold_start_needed=lambda: state.cold,
        kernel_prewarm_set_capture_only=capture_only,
        kernel_prewarm_offline_compile=lambda: events.append(("compile",)) or 3,
    )
    namespace = dict(
        os=os,
        weakref=weakref,
        ttnn=SimpleNamespace(
            Topology=SimpleNamespace(Linear="linear", Ring="ring"),
            device=SimpleNamespace(is_blackhole=lambda: True),
            _ttnn=SimpleNamespace(device=kp),
        ),
        EncoderParallelConfig=SimpleNamespace,
        ParallelFactor=SimpleNamespace,
        DiTParallelConfig=SimpleNamespace,
        VaeHWParallelConfig=SimpleNamespace,
        CCLManager=lambda mesh, **kw: SimpleNamespace(**kw),
        StateTensor=lambda: object(),
        resolve_audio_traced=lambda traced, audio: traced if audio is None else audio,
        set_kernel_prewarm_capturing=lambda flag: events.append(("tracer_capture_only", flag)),
        logger=SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
        safe_open=lambda *a, **kw: nullcontext(SimpleNamespace(metadata=lambda: {"model_version": "2.5"})),
    )

    def load_class(filename, name, base):
        tree = ast.parse(SOURCE.with_name(filename).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        # Retain the exact production methods, including decorators and super() dispatch.
        node.bases = [ast.Name(id=base, ctx=ast.Load())] if base else []
        node.body = [
            n for n in node.body
            if isinstance(n, ast.AnnAssign)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name in {"__init__", "create_pipeline"})
        ]
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE.with_name(filename)), "exec"), namespace)
        return namespace[name]

    base = load_class("pipeline_ltx.py", "LTXPipeline", None)
    load_class("pipeline_ltx_distilled.py", "LTXDistilledPipeline", "LTXPipeline")
    native = load_class("pipeline_ltx25_distilled.py", "LTX25DistilledPipeline", "LTXDistilledPipeline")
    base._resolve_checkpoint_file = staticmethod(lambda path: path)
    base._make_gemma_encoder_pair = lambda self, path: SimpleNamespace(
        gemma_path=path, defer_trace_capture=lambda: events.append(("defer_encode",))
    )
    for name in ("_load_config_from_checkpoint", "_instantiate_modules", "_register_coresident_exclusions", "_maybe_apply_quant_config", "_prime_caches"):
        setattr(base, name, lambda *a, **kw: None)
    base.release_traces = lambda self: events.append(("release",))

    def warmup(self, **kwargs):
        stages = kwargs.get("stages", ("s1", "s2"))
        events.append(("warmup", kwargs.copy(), self._traced, state.capture_only))
        if state.error:
            raise state.error
        for stage in stages:
            if self._traced and stage not in self._trace_state:
                events.append(("allocate", stage))
                self._trace_state[stage] = object()
        if self._traced and not state.capture_only:
            events.append(("capture", stages))

    base.warmup_buffers = warmup
    distilled = SimpleNamespace(
        DISTILLED_SIGMA_VALUES=contract.module.S1_SIGMAS,
        STAGE_2_DISTILLED_SIGMA_VALUES=contract.module.S2_SIGMAS,
        KEYFRAME_S1_STRENGTH=0.5,
        KEYFRAME_S1_STRENGTH_COARSE=0.25,
    )
    monkeypatch.setattr(contract.module, "__package__", "constructor_ltx")
    monkeypatch.setitem(sys.modules, "constructor_ltx", SimpleNamespace(pipeline_ltx_distilled=distilled))
    monkeypatch.setitem(sys.modules, "constructor_ltx.pipeline_ltx25_distilled", SimpleNamespace(LTX25DistilledPipeline=native))
    mesh = SimpleNamespace(shape=(4, 8), clear_program_cache=lambda: events.append(("clear_cache",)))
    factory = contract.raw_native_factory
    monkeypatch.setattr(contract.module, "_native_factory", lambda: lambda ignored, **kw: factory()(mesh, **kw))
    return state, events


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("dynamic_load", [False, True])
def test_constructor_reference_allocation_before_capture(contract, constructor_native, traced, cold, dynamic_load):
    state, events = constructor_native
    state.cold = cold
    pipe = contract.create(traced=traced, dynamic_load=dynamic_load)
    native = pipe._pipeline
    assert native._traced is traced
    assert native._vae_traced is traced
    assert native._audio_traced is traced
    assert native.fps == 24
    assert (native._init_num_frames, native._init_height, native._init_width) == (9, 128, 128)
    assert native._diffusion_decoder is False
    assert native.use_ancestral_sampler is False
    implicit = [e for e in events if e[0] == "warmup"]
    assert len(implicit) == (2 if cold and traced and not dynamic_load else int(traced))
    pipe.warmup_buffers()
    for _, kw, actual_traced, _ in (e for e in events if e[0] == "warmup"):
        assert (kw["num_frames"], kw["height"], kw["width"]) == (9, 128, 128)
        assert kw.get("ref_num_frames") == 9
        assert kw.get("stages") == ("s1_ref", "s2_ref")
        assert actual_traced is traced
    captures = [i for i, e in enumerate(events) if e[0] == "capture"]
    if traced:
        assert captures
        assert [e for e in events[:captures[0]] if e[0] == "allocate"] == [
            ("allocate", "s1_ref"), ("allocate", "s2_ref")
        ]
        assert not any(e[0] == "allocate" for e in events[captures[0]:])
        assert implicit[-1][1]["capture_all"] is cold
    else:
        assert not captures
    if cold and traced and not dynamic_load:
        assert implicit[0][1]["capture_all"] is True
        assert implicit[0][1]["capture_traced"] is True
        assert implicit[0][3] is True
        assert events.index(("compile",)) < events.index(("release",)) < events.index(("clear_cache",)) < captures[0]
    else:
        assert ("compile",) not in events
    assert state.capture_only is False


@pytest.mark.parametrize("error", [RuntimeError("capture failed"), KeyboardInterrupt()])
def test_constructor_cold_capture_cleanup(contract, constructor_native, error):
    state, events = constructor_native
    state.cold = True
    state.error = error
    with pytest.raises(type(error)):  # allow-pytest.raises: CPU-only run excludes native conftest
        contract.create(traced=True, dynamic_load=False)
    assert state.capture_only is False
    assert events[-2:] == [("tracer_capture_only", False), ("capture_only", False)]
    assert ("compile",) not in events

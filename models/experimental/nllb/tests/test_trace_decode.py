# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""CPU ownership/failure control-flow tests; no claim of native TT recovery."""

from types import SimpleNamespace
import numpy as np
import pytest

from models.experimental.nllb.tt import backend
from models.experimental.nllb.tt import trace_decode

ProjectedDecoderTrace = trace_decode.ProjectedDecoderTrace


class Buffer:
    def buffer_address(self):
        return id(self)


@pytest.fixture
def harness(monkeypatch):
    # Legacy scope tests exercise the retained body-only/full-policy path.
    monkeypatch.setattr(trace_decode, "ProjectedDecoderTrace", trace_decode.DecoderTrace)
    b = backend.Backend.__new__(backend.Backend)
    b.device = object()
    b._trace_failures = []
    b.trace_decoder_enabled = True
    b.pad, b.vocab = 1, 16
    b.config = dict(
        vocab_size=16, max_position_embeddings=128, pad_token_id=1, eos_token_id=2, decoder_start_token_id=2
    )
    state = SimpleNamespace(
        active=False,
        begin_error=None,
        end_error=None,
        release_error=None,
        body_error=None,
        begin_active=True,
        events=[],
        owners=[],
        encodes=0,
        replays=0,
        input_count=4,
    )

    def encode(ids, mask):
        state.encodes += 1
        return Buffer(), mask[0]

    b.encode = encode
    b.decoder_memory = lambda enc, valid: (enc, valid)

    def project(output, length, **kw):
        values = np.zeros((1, 1, 16))
        values[0, 0, 2] = 1
        return values

    b.project_decoder = project

    def body(owner):
        if owner not in state.owners:
            state.owners.append(owner)
        owner.cross_kv.setdefault("model.decoder.layers.0.encoder_attn", (Buffer(), Buffer()))
        if state.active and state.body_error is not None:
            raise state.body_error
        return Buffer()

    monkeypatch.setattr(trace_decode.DecoderTrace, "host_inputs", lambda self, ids: [Buffer() for _ in range(4)])
    monkeypatch.setattr(trace_decode.DecoderTrace, "body", body)

    def begin(device, **kw):
        assert device is b.device
        state.events.append("begin")
        state.active = state.begin_active
        if state.begin_error is not None:
            raise state.begin_error
        return 71

    def end(device, trace_id, **kw):
        assert device is b.device and trace_id == 71
        state.events.append("end")
        if state.end_error is not None:
            raise state.end_error
        state.active = False

    def release(device, trace_id):
        assert device is b.device and trace_id == 71
        # Referenced buffers/cache must still be owned at the native boundary.
        owner = state.owners[-1]
        assert len(owner.inputs) == state.input_count and owner.cross_kv and owner.encoder is not None
        state.events.append("release")
        if state.release_error is not None:
            raise state.release_error

    def replay(device, trace_id, **kw):
        assert device is b.device and trace_id == 71 and not state.active
        state.replays += 1

    ops = SimpleNamespace(
        DRAM_MEMORY_CONFIG=object(),
        to_device=lambda host, device, **kw: Buffer(),
        synchronize_device=lambda device: None,
        begin_trace_capture=begin,
        end_trace_capture=end,
        release_trace=release,
        copy_host_to_device_tensor=lambda *a, **kw: None,
        execute_trace=replay,
    )
    monkeypatch.setattr(trace_decode, "ttnn", ops)
    return b, state


def inputs():
    return np.array([[5, 2, 1]]), np.array([[1, 1, 0]])


def assert_blocked(b):
    ids, mask = inputs()
    originals = ids.copy(), mask.copy()
    # Even disabled tracing, cap=1 and direct untraced decode must be blocked.
    for enabled in (False, True):
        b.trace_decoder_enabled = enabled
        for call in (
            lambda: b.generate(ids, mask, 4, 1),
            lambda: b.generate(ids, mask, 4, 3),
            lambda: b.forward(ids, mask, ids),
            lambda: b.decode(ids, object(), mask[0]),
            lambda: b.decode(ids, object(), mask[0], final_token_only=True, cross_kv={}),
        ):
            with pytest.raises(RuntimeError, match="cleanup unresolved"):  # allow-pytest.raises: CPU-only check.
                call()
    for actual, original in zip((ids, mask), originals):
        np.testing.assert_array_equal(actual, original)


def test_entry_guards_precede_validation_or_learned_work():
    b = backend.Backend.__new__(backend.Backend)
    b._trace_failures = [object()]
    # No config/weights/encode setup: touching these would fail the test.
    assert_blocked(b)


@pytest.mark.parametrize("active", [False, True])
def test_native_begin_without_id_retains_ownership_and_exact_error(harness, active):
    b, s = harness
    sentinel = RuntimeError("native begin")
    s.begin_error, s.begin_active = sentinel, active
    with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
        b.generate(*inputs(), 4, 3)
    assert caught.value is sentinel
    owner = s.owners[0]
    refs = tuple(owner.inputs)
    cache = dict(owner.cross_kv)
    assert owner.unresolved and owner.trace_id is None
    assert b._trace_failures == [owner] and b._decode_trace is None
    assert len(refs) == 4 and cache and owner.encoder is not None
    assert s.events == ["begin"] and s.replays == 0
    owner.close()
    owner.close()
    assert tuple(owner.inputs) == refs and owner.cross_kv == cache
    assert b._trace_failures == [owner]
    assert_blocked(b)
    with pytest.raises(RuntimeError, match="cleanup unresolved"):  # allow-pytest.raises: CPU-only check.
        owner.decode(inputs()[0])
    assert s.encodes == 1 and s.events == ["begin"]


@pytest.mark.parametrize("phase", ["end", "release"])
@pytest.mark.parametrize("body_fails", [False, True])
def test_native_cleanup_failure_is_sticky_and_preserves_primary(harness, phase, body_fails):
    b, s = harness
    cleanup = RuntimeError("native " + phase)
    primary = RuntimeError("body")
    if phase == "begin":
        s.begin_error = cleanup
    elif phase == "end":
        s.end_error = cleanup
    elif phase == "release":
        s.release_error = cleanup
    else:
        raise AssertionError("unknown failure phase")
    s.body_error = primary if body_fails else None
    with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
        b.generate(*inputs(), 4, 3)
    assert caught.value is (primary if body_fails else cleanup)
    owner = s.owners[0]
    assert owner.unresolved and b._trace_failures == [owner]
    assert b._decode_trace is None
    refs, cache = tuple(owner.inputs), dict(owner.cross_kv)
    assert len(refs) == 4 and cache and owner.encoder is not None
    assert s.events.count("end") == 1
    # A later release success does not prove arbitrary native state recovered.
    s.release_error = None
    owner.close()
    assert owner.trace_id is None
    assert tuple(owner.inputs) == refs and owner.cross_kv == cache
    assert "drop_buffers" not in owner.events[1:]
    assert_blocked(b)
    assert s.encodes == 1
    if body_fails or phase == "end":
        assert s.replays == 0


def test_successful_partial_cleanup_allows_same_device_reuse(harness):
    b, s = harness
    device = b.device
    sentinel = RuntimeError("body")
    s.body_error = sentinel
    ids, mask = inputs()
    originals = ids.copy(), mask.copy()
    with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
        b.generate(ids, mask, 4, 3)
    assert caught.value is sentinel
    owner = s.owners[0]
    assert s.events == ["begin", "end", "release"] and s.replays == 0
    assert owner.trace_id is None and not owner.unresolved
    assert not owner.inputs and owner.output is None and not owner.cross_kv
    # prepare() closes after the body exception; generate() closes again in
    # finally. Re-clearing empty references may log another drop, never another
    # native end/release. Ignore the initial empty close before bucket allocation.
    captured = owner.events[owner.events.index("begin:32") :]
    assert captured[:3] == ["begin:32", "end", "release"]
    assert captured[3:] and set(captured[3:]) == {"drop_buffers"}
    native_events = list(s.events)
    owner.close()
    owner.close()
    assert s.events == native_events
    assert owner.trace_id is None and not owner.unresolved
    assert not owner.inputs and owner.output is None and not owner.cross_kv
    assert not b._trace_failures and b._decode_trace is None and not s.active
    s.body_error = None
    np.testing.assert_array_equal(b.generate(ids, mask, 4, 3), [[2, 4, 2]])
    assert b.device is device and s.replays == 1 and s.encodes == 2
    assert s.events == ["begin", "end", "release"] * 2
    recovered = s.owners[-1]
    assert not recovered.inputs and recovered.output is None and not recovered.cross_kv
    assert recovered.trace_id is None and not recovered.unresolved
    assert not b._trace_failures and b._decode_trace is None and not s.active
    recovered.close()
    recovered.close()
    assert s.events == ["begin", "end", "release"] * 2
    for actual, original in zip((ids, mask), originals):
        np.testing.assert_array_equal(actual, original)


class SpecBuffer(Buffer):
    def __init__(self, rows=32, dim=8, dtype="bf16", layout="tile", memory="dram"):
        self.shape = (1, 1, rows, dim)
        self.padded_shape = self.shape
        self.dtype, self.layout, self.memory = dtype, layout, memory

    def memory_config(self):
        return self.memory


@pytest.fixture
def warming(harness, monkeypatch):
    b, s = harness
    b.dim, b.generation_projection = 8, "last"
    b.matrix_dtype = "bf16"
    b.precision_policy = {"mode": "bf16"}
    b.kernel = SimpleNamespace(math_fidelity="HiFi4", fp32_dest_acc_en=True)
    b.lm_weight = SpecBuffer(rows=16)
    original_encode = b.encode

    def encode(ids, mask):
        _, valid = original_encode(ids, mask)
        # Supply tensor metadata while retaining the harness counter and mask.
        rows = ((ids.shape[1] + 31) // 32) * 32
        return SpecBuffer(rows=rows, dim=b.dim), valid

    b.encode = encode
    s.projects, s.syncs, s.project_error, s.sync_error = [], 0, None, None

    def project(output, length, **kw):
        s.projects.append(length)
        if s.project_error is not None:
            raise s.project_error
        return np.zeros((1, 1, 16))

    def sync(device):
        assert device is b.device
        s.syncs += 1
        if s.sync_error is not None:
            raise s.sync_error

    b.project_decoder = project
    monkeypatch.setattr(trace_decode.ttnn, "synchronize_device", sync)
    return b, s


def test_opt_in_scope_reuses_only_metadata_and_exit_forgets(warming):
    b, s = warming
    with b.last_warmup_reuse() as scope:
        scope.warm(SpecBuffer(), 32, SpecBuffer())
        assert s.projects == list(range(1, 33)) and s.syncs == 1
        # New prompt tensor addresses with identical operator geometry.
        scope.warm(SpecBuffer(), 32, SpecBuffer())
        assert len(s.projects) == 32 and s.syncs == 2
        with pytest.raises(RuntimeError, match="already owned"):  # allow-pytest.raises: CPU-only check.
            with b.last_warmup_reuse():
                pass
        assert len(scope.variants) == 32
    assert not scope.variants and b._last_warmup_reuse is None
    with b.last_warmup_reuse() as fresh:
        fresh.warm(SpecBuffer(), 32, SpecBuffer())
    assert len(s.projects) == 64


@pytest.mark.parametrize("mutation", ["clear", "disable_clear_repopulate"])
def test_invalidation_before_cache_mutation_even_with_equal_counts(warming, mutation):
    b, s = warming
    # Cache counts intentionally remain identical; never consulted by production.
    native_cache = {"entries": 100, "enabled": True}
    with b.last_warmup_reuse() as scope:
        scope.warm(SpecBuffer(), 32, SpecBuffer())
        b.invalidate_last_warmup()
        assert not scope.variants
        native_cache["entries"] = 0
        native_cache["enabled"] = mutation == "clear"
        native_cache.update(entries=100, enabled=True)
        scope.warm(SpecBuffer(), 32, SpecBuffer())
        assert len(s.projects) == 64 and len(scope.variants) == 32


@pytest.mark.parametrize(
    "change",
    [
        "rows",
        "dim",
        "padding",
        "dtype",
        "layout",
        "memory",
        "weight",
        "encoder",
        "config",
        "precision",
        "kernel",
        "projection",
        "vocab",
    ],
)
def test_complete_geometry_and_configuration_changes_rewarm(warming, change):
    b, s = warming
    with b.last_warmup_reuse() as scope:
        scope.warm(SpecBuffer(), 32, SpecBuffer())
        output, encoder, rows = SpecBuffer(), SpecBuffer(), 32
        if change == "rows":
            output, rows = SpecBuffer(rows=64), 64
        elif change == "dim":
            output, b.dim = SpecBuffer(dim=16), 16
        elif change == "padding":
            output.padded_shape = (1, 1, 64, 8)
        elif change in ("dtype", "layout", "memory"):
            setattr(output, change, "changed")
        elif change == "weight":
            b.lm_weight = SpecBuffer(dtype="bfp8")
        elif change == "encoder":
            encoder = SpecBuffer(rows=64)
        elif change == "config":
            b.config["d_model"] = 16
        elif change == "precision":
            b.precision_policy["mode"] = "bfp8"
        elif change == "kernel":
            b.kernel.math_fidelity = "changed"
        elif change == "projection":
            b.generation_projection = "full"
        elif change == "vocab":
            b.vocab = 32
        scope.warm(output, rows, encoder)
        assert s.projects[32:] == list(range(1, rows + 1))


@pytest.mark.parametrize("phase", ["project", "sync"])
def test_failed_warmup_never_commits_and_preserves_exact_exception(warming, phase):
    b, s = warming
    sentinel = RuntimeError("warm " + phase)
    with b.last_warmup_reuse() as scope:
        if phase == "project":
            s.project_error = sentinel
        elif phase == "sync":
            s.sync_error = sentinel
        else:
            raise AssertionError("unknown failure phase")
        with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
            scope.warm(SpecBuffer(), 32, SpecBuffer())
        assert caught.value is sentinel and not scope.variants
        if phase == "project":
            s.project_error = None
        else:
            s.sync_error = None
        s.projects.clear()
        scope.warm(SpecBuffer(), 32, SpecBuffer())
        assert s.projects == list(range(1, 33)) and len(scope.variants) == 32


def test_invalidation_and_exit_reject_direct_active_owner(warming, monkeypatch):
    b, s = warming
    original_body = trace_decode.DecoderTrace.body

    def body(owner):
        original_body(owner)
        return SpecBuffer()

    monkeypatch.setattr(trace_decode.DecoderTrace, "body", body)
    scope = b.last_warmup_reuse().__enter__()
    owner = trace_decode.DecoderTrace(b, SpecBuffer(), np.ones(32), {})
    owner.prepare(inputs()[0], [Buffer() for _ in range(4)])
    assert owner.trace_id == 71 and owner in b._last_warmup_owners
    for call in (b.invalidate_last_warmup, lambda: scope.__exit__(None, None, None)):
        with pytest.raises(RuntimeError, match="active trace ownership"):  # allow-pytest.raises: CPU-only check.
            call()
    assert b._last_warmup_reuse is scope
    owner.close()
    b.invalidate_last_warmup()
    scope.__exit__(None, None, None)
    assert not b._last_warmup_owners and b.device is scope.device


def test_scope_rejects_device_replacement_and_unresolved_ownership(warming):
    b, s = warming
    with b.last_warmup_reuse() as scope:
        device = b.device
        b.device = object()
        with pytest.raises(RuntimeError, match="device changed"):  # allow-pytest.raises: CPU-only check.
            scope.warm(SpecBuffer(), 32, SpecBuffer())
        assert not s.projects
        b.device = device
        b._trace_failures = [object()]
        with pytest.raises(RuntimeError, match="cleanup unresolved"):  # allow-pytest.raises: CPU-only check.
            b.invalidate_last_warmup()
        b._trace_failures.clear()


def test_default_still_warms_all_rows_each_request(harness):
    b, s = harness
    calls = []
    original = b.project_decoder

    def project(output, length, **kw):
        calls.append(length)
        return original(output, length, **kw)

    b.project_decoder = project
    for _ in range(2):
        b.generate(*inputs(), 4, 3)
    assert calls == (list(range(1, 33)) + [2]) * 2
    assert getattr(b, "_last_warmup_reuse", None) is None
    assert not b._last_warmup_owners


@pytest.mark.parametrize("phase", ["begin", "end", "release"])
@pytest.mark.parametrize("body_fails", [False, True])
def test_scope_exit_preserves_native_primary_and_cleanup_diagnostics(warming, monkeypatch, phase, body_fails):
    b, s = warming
    original_body = trace_decode.DecoderTrace.body

    def body(owner):
        original_body(owner)
        return SpecBuffer()

    monkeypatch.setattr(trace_decode.DecoderTrace, "body", body)
    native = RuntimeError("native scope " + phase)
    primary = RuntimeError("scope body")
    if phase == "begin":
        s.begin_error = native
    elif phase == "end":
        s.end_error = native
    elif phase == "release":
        s.release_error = native
    else:
        raise AssertionError("unknown failure phase")
    s.body_error = primary if body_fails else None
    ids, mask = inputs()
    originals = ids.copy(), mask.copy()
    device = b.device
    with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
        with b.last_warmup_reuse() as scope:
            b.generate(ids, mask, 4, 3)
    expected = primary if body_fails and phase != "begin" else native
    assert caught.value is expected
    owner = s.owners[0]
    assert owner.unresolved and b._trace_failures == [owner]
    assert owner in b._last_warmup_owners and b._decode_trace is None
    assert scope.entered and b._last_warmup_reuse is scope
    assert scope.variants and len(owner.inputs) == 4 and owner.cross_kv
    assert owner.encoder is not None and b.device is device
    assert len(scope.cleanup_errors) == 1
    diagnostic = scope.cleanup_errors[0]
    assert "cleanup unresolved" in str(diagnostic)
    assert diagnostic.__context__ is expected
    if phase == "begin":
        assert s.events == ["begin"] and s.replays == 0
        assert any("begin_error:" in event for event in owner.events)
    elif phase == "end":
        assert s.events == ["begin", "end", "release"] and s.replays == 0
        if body_fails:
            assert "end_error:" + repr(native) in owner.events
    else:
        assert "release_error:" + repr(native) in owner.events
    assert_blocked(b)
    for actual, original in zip((ids, mask), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("unresolved", [False, True])
def test_scope_exit_without_primary_retains_ownership_and_raises(warming, unresolved):
    b, s = warming
    scope = b.last_warmup_reuse().__enter__()
    owner = object()
    b._last_warmup_owners = {owner}
    if unresolved:
        b._trace_failures = [owner]
    with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
        scope.__exit__(None, None, None)
    assert scope.cleanup_errors == [caught.value]
    assert ("cleanup unresolved" if unresolved else "active trace ownership") in str(caught.value)
    assert b._last_warmup_reuse is scope and scope.entered
    assert b._last_warmup_owners == {owner}
    assert b._trace_failures == ([owner] if unresolved else [])


def test_scope_exit_with_active_owner_preserves_baseexception(warming):
    b, s = warming
    primary = KeyboardInterrupt("scope sentinel")
    owner = object()
    with pytest.raises(KeyboardInterrupt) as caught:  # allow-pytest.raises: CPU-only check.
        with b.last_warmup_reuse() as scope:
            b._last_warmup_owners = {owner}
            raise primary
    assert caught.value is primary
    assert b._last_warmup_reuse is scope and scope.entered
    assert b._last_warmup_owners == {owner}
    assert "active trace ownership" in str(scope.cleanup_errors[0])
    # Explicitly resolved ownership permits a later exit; diagnostics survive.
    b._last_warmup_owners.clear()
    assert scope.__exit__(None, None, None) is False
    assert not scope.entered and b._last_warmup_reuse is None
    assert len(scope.cleanup_errors) == 1


def test_scope_exit_after_resolved_body_failure_preserves_primary(warming, monkeypatch):
    b, s = warming
    original_body = trace_decode.DecoderTrace.body

    def body(owner):
        original_body(owner)
        return SpecBuffer()

    monkeypatch.setattr(trace_decode.DecoderTrace, "body", body)
    primary = RuntimeError("resolved scope body")
    s.body_error = primary
    device = b.device
    ids, mask = inputs()
    originals = ids.copy(), mask.copy()
    with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
        with b.last_warmup_reuse() as scope:
            b.generate(ids, mask, 4, 3)
    assert caught.value is primary
    assert not scope.entered and not scope.variants and not scope.cleanup_errors
    assert b._last_warmup_reuse is None and not b._last_warmup_owners
    assert not b._trace_failures and s.replays == 0
    s.body_error = None
    with b.last_warmup_reuse() as recovered_scope:
        # All-zero logits select token 0, so generation reaches the cap.
        result = b.generate(ids, mask, 4, 3)
    np.testing.assert_array_equal(result, [[2, 4, 0, 0]])
    assert b.device is device and s.replays == 2 and s.encodes == 2
    assert s.events == ["begin", "end", "release"] * 2
    assert not recovered_scope.entered and not recovered_scope.variants
    assert not recovered_scope.cleanup_errors and b._last_warmup_reuse is None
    assert not b._last_warmup_owners and not b._trace_failures
    assert b._decode_trace is None and not s.active
    assert len(s.owners) == 2
    for owner in s.owners:
        assert owner.trace_id is None and not owner.unresolved
        assert not owner.inputs and owner.output is None and not owner.cross_kv
    for actual, original in zip((ids, mask), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.fixture
def projected(harness, monkeypatch):
    b, s = harness
    monkeypatch.setattr(trace_decode, "ProjectedDecoderTrace", ProjectedDecoderTrace)
    b.dim, b.generation_projection = 1024, "last"
    s.input_count = 5
    s.index_writes, s.steps, s.reads = [], [], []
    s.warm_token, s.replay_token = 5, 2
    b.upload = lambda value: Buffer()

    def host(owner, ids):
        return [Buffer() for _ in range(4)] + [ids.shape[1] - 1]

    monkeypatch.setattr(ProjectedDecoderTrace, "host_inputs", host)
    base_body = trace_decode.DecoderTrace.body

    def decoder(*args, **kwargs):
        owner = b._decode_trace
        s.steps.append(("decoder", s.active))
        return base_body(owner)

    b.decoder_body = decoder

    def selector(hidden, index, selected):
        s.steps.append(("selector", s.active))

    monkeypatch.setattr(trace_decode, "raw_selector", selector)

    def project(selected):
        s.steps.append(("project_layout", s.active))
        return Buffer()

    b.project_selected_device = project

    def read(output):
        assert not s.active
        s.reads.append(s.replays)
        values = np.zeros((1, 1, 16))
        values[0, 0, s.replay_token if s.replays else s.warm_token] = 1
        return values

    b.read_decoder_row = read

    def copy(src, dst, **kwargs):
        if isinstance(src, int):
            s.index_writes.append((src, dst.buffer_address()))

    monkeypatch.setattr(trace_decode.ttnn, "copy_host_to_device_tensor", copy)
    return b, s


def test_projected_warm_eos_skips_capture(projected):
    b, s = projected
    s.warm_token = 2
    np.testing.assert_array_equal(b.generate(*inputs(), 4, 5), [[2, 4, 2]])
    assert s.events == [] and s.replays == 0 and s.reads == [0]
    assert [x[0] for x in s.steps] == ["decoder", "selector", "project_layout"]
    assert not b._last_warmup_owners and not b._trace_failures
    assert s.owners[0].selected is None and not s.owners[0].cross_kv


def test_projected_warm_once_capture_then_execute(projected):
    b, s = projected
    np.testing.assert_array_equal(b.generate(*inputs(), 4, 5), [[2, 4, 5, 2]])
    assert s.events == ["begin", "end", "release"]
    assert s.replays == 1 and s.reads == [0, 1]
    assert s.owners[0].warm_results == 1
    assert [x[0] for x in s.steps] == ["decoder", "selector", "project_layout"] * 2
    assert [x[1] for x in s.steps] == [False] * 3 + [True] * 3
    assert [row for row, _ in s.index_writes] == [2]


def test_projected_transitions_refresh_and_cross_kv(projected):
    b, s = projected
    owner = ProjectedDecoderTrace(b, Buffer(), np.ones(32), {})
    b._decode_trace = owner
    cache = None
    selected = []
    for n in (2, 31, 32, 33, 63, 64, 2, 32):
        ids = np.full((1, n), 5)
        ids[0, -1] = 1  # Final row is PAD: index must still be n-1.
        owner.decode(ids)
        selected.append(owner.selected)
        if cache is None:
            cache = dict(owner.cross_kv)
        assert owner.cross_kv == cache
    assert owner.warm_results == 3 and owner.replays == 5
    assert selected[0] is selected[1] is selected[2]
    assert selected[3] is selected[4] is selected[5]
    assert selected[6] is selected[7] and selected[6] is not selected[0]
    assert [row for row, _ in s.index_writes] == [30, 31, 62, 63, 31]
    assert s.index_writes[0][1] == s.index_writes[1][1]
    owner.close()
    assert s.events.count("release") == 3
    assert not owner.inputs and owner.selected is None and not b._trace_failures


@pytest.mark.parametrize("phase", ["warm", "begin", "body", "end", "replay", "readback", "release"])
def test_projected_exact_failure_and_ownership(projected, monkeypatch, phase):
    b, s = projected
    sentinel = RuntimeError("projected " + phase)

    def fail(*a, **kw):
        raise sentinel

    if phase in ("begin", "end", "release"):
        if phase == "begin":
            s.begin_error = sentinel
        elif phase == "end":
            s.end_error = sentinel
        elif phase == "release":
            s.release_error = sentinel
        else:
            raise AssertionError("unknown failure phase")
    elif phase == "body":
        s.body_error = sentinel
    elif phase == "warm":
        b.project_selected_device = fail
    elif phase == "replay":
        monkeypatch.setattr(trace_decode.ttnn, "execute_trace", fail)
    else:
        original = b.read_decoder_row

        def read(output):
            if s.replays:
                raise sentinel
            return original(output)

        b.read_decoder_row = read
    with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
        b.generate(*inputs(), 4, 5)
    assert caught.value is sentinel
    owner = s.owners[0]
    if phase in ("begin", "end", "release"):
        assert owner.unresolved and owner.selected is not None
        assert len(owner.inputs) == 5 and owner.cross_kv
        assert b._trace_failures == [owner]
        assert_blocked(b)
    else:
        assert not owner.unresolved and not b._trace_failures
        assert owner.selected is None and not owner.inputs and not owner.cross_kv


def test_projected_host_index_is_real_final_offset(harness, monkeypatch):
    import torch

    b, s = harness
    monkeypatch.setattr(trace_decode.ttnn, "uint32", "uint32", raising=False)
    monkeypatch.setattr(trace_decode.ttnn, "ROW_MAJOR_LAYOUT", "rm", raising=False)

    def from_torch(value, **kw):
        assert kw == dict(dtype="uint32", layout="rm")
        return value

    monkeypatch.setattr(trace_decode.ttnn, "from_torch", from_torch, raising=False)
    owner = ProjectedDecoderTrace(b, Buffer(), np.ones(32), {})
    for n in (2, 31, 32, 33, 63, 64):
        ids = np.ones((1, n), dtype=np.int64)
        index = owner.host_inputs(ids)[-1]
        assert torch.equal(index, torch.tensor([[n - 1]], dtype=torch.int32))


# Qualified donor CPU regressions. Native campaign originals remain in
# qualified_batch/test_trace_decode.py; portable native validation is separate.
@pytest.fixture(params=("bf16", "bfp8_b"))
def packed_harness(harness, request):
    b, state = harness
    # Production dispatch identity on a tiny operator/control-flow fake.
    # Keep its small buffers/vocabulary and the original general fixture intact.
    b.config.update(backend._DISTILLED_600M)
    b.precision_policy = {"mode": request.param}
    return b, state


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("failure", [False, True])
def test_packed_request_eos_and_exception_cleanup(packed_harness, monkeypatch, batch, failure):
    b, s = packed_harness
    b.generation_projection = "last"
    seen, finished = [], []
    sentinel = RuntimeError("packed sentinel")

    def initialize(owner):
        pass

    def decode_rows(owner, prefixes, active):
        seen.append((list(active), [list(p) for p in prefixes]))
        if failure:
            raise sentinel
        logits = np.zeros((len(active), b.vocab))
        for slot, row in enumerate(active):
            logits[slot, 2 if row == 0 or len(prefixes[row]) == 4 else 7] = 1
        return logits

    original_finish = trace_decode.PackedLMRequest.finish

    def finish(owner, **kw):
        original_finish(owner, **kw)
        finished.append(owner)

    monkeypatch.setattr(trace_decode.PackedLMRequest, "initialize", initialize)
    monkeypatch.setattr(trace_decode.PackedLMRequest, "decode_rows", decode_rows)
    monkeypatch.setattr(trace_decode.PackedLMRequest, "finish", finish)
    ids, mask = [np.repeat(x, batch, axis=0) for x in inputs()]
    before = ids.copy(), mask.copy()
    if failure:
        with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
            b.generate(ids, mask, 4, 5)
        assert caught.value is sentinel
    else:
        result = b.generate(ids, mask, 4, 5)
        np.testing.assert_array_equal(result[0], [2, 4, 2, 1, 1])
        for row in range(1, batch):
            np.testing.assert_array_equal(result[row], [2, 4, 7, 7, 2])
        assert [a for a, _ in seen] == [list(range(batch)), list(range(1, batch)), list(range(1, batch))]
        assert seen[-1][1][0] == [2, 4, 2]
    assert len(finished) == 1 and not finished[0].rows
    assert b._batch_trace is None and not b._trace_failures
    for x, y in zip((ids, mask), before):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("failure", [None, "replay", "release"])
def test_per_row_survivor_trace_and_cleanup(packed_harness, monkeypatch, failure):
    b, s = packed_harness
    owner = trace_decode.PackedLMRequest(b, *inputs())
    owner.rows = [trace_decode.PersistentRowTrace(b, Buffer(), np.ones(32), {}) for _ in range(4)]
    for row in owner.rows:
        row.inputs, row.output, row.bucket = [Buffer() for _ in range(4)], Buffer(), 32
        row.cross_kv["k"] = Buffer()
    owner.inputs = [Buffer() for _ in range(4)]
    owner.output, owner.selected = Buffer(), Buffer()
    owner.bucket = (32,) * 4
    b._last_warmup_owners = {owner}
    monkeypatch.setattr(trace_decode.PersistentRowTrace, "body", lambda row: row.output)
    monkeypatch.setattr(trace_decode.PackedLMRequest, "body", lambda row: row.output)
    monkeypatch.setattr(owner, "index_host", lambda n: n)
    monkeypatch.setattr(owner, "read", lambda: np.zeros((len(owner.active), b.vocab)))
    captured, replayed, writes, released = [], [], [], []
    sentinel = RuntimeError("per-row " + str(failure))

    def begin(*args, **kw):
        captured.append(len(captured) + 1)
        return captured[-1]

    def execute(device, trace_id, **kw):
        replayed.append(trace_id)
        if failure == "replay" and len(replayed) == 7:
            raise sentinel

    def release(device, trace_id):
        released.append(trace_id)
        if failure == "release" and trace_id == 2:
            raise sentinel

    monkeypatch.setattr(trace_decode.ttnn, "begin_trace_capture", begin)
    monkeypatch.setattr(trace_decode.ttnn, "end_trace_capture", lambda *a, **k: None)
    monkeypatch.setattr(trace_decode.ttnn, "execute_trace", execute)
    monkeypatch.setattr(trace_decode.ttnn, "release_trace", release)
    monkeypatch.setattr(
        trace_decode.ttnn, "copy_host_to_device_tensor", lambda src, dst, **kw: writes.append((src, dst))
    )
    prefixes = [[2, 4, 7] for _ in range(4)]
    owner.decode_rows(prefixes, [0, 1, 2, 3])
    handles = [row.trace_id for row in owner.rows]
    prefixes[0].append(2)
    for i in (1, 2, 3):
        prefixes[i].append(7)
    if failure == "replay":
        with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
            owner.decode_rows(prefixes, [1, 2, 3])
        assert caught.value is sentinel
    else:
        owner.decode_rows(prefixes, [1, 2, 3])
    assert [row.trace_id for row in owner.rows] == handles
    assert len(captured) == 5  # Four row bodies plus one shared LM, no EOS recapture.
    assert replayed.count(handles[0]) == 1
    assert any(src == 32 and dst is owner.inputs[0] for src, dst in writes if isinstance(src, int))
    rows = list(owner.rows)
    if failure == "release":
        with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
            owner.finish()
        assert caught.value is sentinel
        assert owner.unresolved and b._trace_failures
        assert owner.rows and owner.selected is not None and owner.inputs
        assert len(released) == 5  # Continue cleanup after one release fails.
        assert_blocked(b)
    else:
        owner.finish()
        assert not owner.rows and owner.output is None and owner.selected is None
        assert not owner.inputs and owner.input_ids is None
        assert all(row.trace_id is None and not row.inputs and not row.cross_kv for row in rows)
        assert not b._trace_failures and not b._last_warmup_owners


@pytest.mark.parametrize("batch", [2, 4])
def test_persistent_transfers_use_same_layout_copy(packed_harness, monkeypatch, batch):
    b, _ = packed_harness
    b.dim, b.vocab, b.lm_weight, b.kernel = 32, 35, object(), object()
    # Keep signed zero, subnormal and NaN payloads as raw storage in this
    # operator-contract mock. Native bit preservation needs the leased TT gate.
    bits = np.array([0, 32768, 1, 32769, 0x7F80, 0xFF80, 0x7FC1, 0x7FA1], dtype=np.uint16)

    def tensor(shape, layout):
        return SimpleNamespace(shape=shape, layout=layout, bits=np.resize(bits, shape).copy())

    hidden = tensor((1, 1, 32, 32), "tile")
    row = trace_decode.PersistentRowTrace(b, Buffer(), np.ones(32), {})
    row.output = tensor(hidden.shape, "tile")
    monkeypatch.setattr(trace_decode.DecoderTrace, "body", lambda owner: hidden)
    calls = []

    def copy(src, dst):
        assert src.layout == dst.layout and src.shape == dst.shape
        calls.append((src, dst))
        dst.bits[...] = src.bits
        return dst

    def arithmetic(*a, **kw):
        pytest.fail("persistent transfer must not use arithmetic")

    monkeypatch.setattr(trace_decode.ttnn, "copy", copy, raising=False)
    monkeypatch.setattr(trace_decode.ttnn, "add", arithmetic, raising=False)
    destination = row.output
    assert row.body() is destination and row.output is destination
    np.testing.assert_array_equal(destination.bits, hidden.bits)
    owner = trace_decode.PackedLMRequest(b, *inputs())
    owner.rows = [row] * batch
    owner.inputs = [Buffer() for _ in range(batch)]
    owner.selected = tensor((1, 1, batch, 32), "tile")
    owner.output = tensor((1, 1, batch, 35), "row")
    destination = owner.output
    projected = tensor((1, 1, batch, 64), "tile")
    projections = []

    def linear(x, w, **kw):
        assert x is owner.selected and w is b.lm_weight
        projections.append(x.shape)
        return projected

    def sliced(x, start, end):
        assert x is projected and start == (0, 0, 0, 0)
        assert end == (1, 1, batch, b.vocab)
        return tensor(end, "tile")

    def layout(x, target):
        assert x.shape == destination.shape and target == "row"
        y = tensor(x.shape, target)
        y.bits[...] = x.bits
        return y

    monkeypatch.setattr(trace_decode, "pack_last_rows", lambda *a: owner.selected)
    for name, value in dict(
        linear=linear, slice=sliced, to_layout=layout, ROW_MAJOR_LAYOUT="row", bfloat16="bf16"
    ).items():
        monkeypatch.setattr(trace_decode.ttnn, name, value, raising=False)
    assert owner.body() is destination and owner.output is destination
    assert projections == [(1, 1, batch, 32)]
    assert len(calls) == 2 and calls[-1][1] is destination
    np.testing.assert_array_equal(destination.bits, calls[-1][0].bits)


def test_observer_trace_id_is_json_safe_without_mutating_native_ownership():
    import json
    from models.experimental.nllb.tests.probe_packed_integration import observer_trace_id as snapshot

    class MeshTraceId:
        def __str__(self):
            return "MeshTraceId(71)"

    native = MeshTraceId()
    owner = SimpleNamespace(trace_id=native, unresolved=True)
    failures = [owner]
    before = dict(trace_id=snapshot(owner.trace_id), unresolved=owner.unresolved)
    with pytest.raises(TypeError):  # allow-pytest.raises: CPU-only check.
        json.dumps(native)
    assert json.loads(json.dumps(before)) == {"trace_id": "MeshTraceId(71)", "unresolved": True}
    assert owner.trace_id is native and failures == [owner] and owner.unresolved
    assert snapshot(None) is None and snapshot(0) == "0"
    owner.trace_id = None  # Simulate successful native release, not observer work.
    assert json.loads(json.dumps(dict(before=before, after=snapshot(owner.trace_id)))) == {
        "before": before,
        "after": None,
    }


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("inherited", [False, True])
def test_canonical_decode_selects_packed(packed_harness, monkeypatch, batch, inherited):
    b, s = packed_harness
    if inherited:

        class Inherited(backend.Backend):
            pass

        b.__class__ = Inherited
    b.generation_projection = "last"
    calls = []
    expected = np.full((batch, 3), 7, dtype=np.int64)

    def packed(model, ids, mask, target, cap):
        assert model is b and target == 4 and cap == 3
        calls.append((ids, mask))
        return expected

    monkeypatch.setattr(trace_decode, "generate_packed", packed)
    ids, mask = [np.repeat(x, batch, axis=0) for x in inputs()]
    assert b.generate(ids, mask, 4, 3) is expected
    assert len(calls) == 1 and s.encodes == 0


@pytest.mark.parametrize("kind", ["instance", "decorated", "subclass", "class"])
@pytest.mark.parametrize("batch", [2, 4])
def test_custom_decode_behavior_and_restored_packing(packed_harness, monkeypatch, kind, batch):
    from functools import wraps

    b, s = packed_harness
    b.generation_projection = "last"
    original = b.decode
    calls, caches, packed_calls = [], [], []

    def wrapped(ids, encoder, valid, **kw):
        # Delegate real control flow, including populated per-layer cross-KV.
        logits = original(ids, encoder, valid, **kw)
        assert kw["final_token_only"] and kw["cross_kv"]
        calls.append(ids.copy())
        caches.append(kw["cross_kv"])
        logits[...] = 0
        logits[0, 0, 7] = 1
        return logits

    if kind == "decorated":
        wrapped = wraps(original)(wrapped)
    if kind in ("instance", "decorated"):
        b.decode = wrapped
    elif kind == "subclass":

        class Override(backend.Backend):
            def decode(self, *args, **kw):
                return wrapped(*args, **kw)

        b.__class__ = Override
    else:
        monkeypatch.setattr(backend.Backend, "decode", lambda self, *args, **kw: wrapped(*args, **kw))
    expected_packed = object()

    def packed(*args):
        packed_calls.append(args)
        return expected_packed

    monkeypatch.setattr(trace_decode, "generate_packed", packed)
    ids, mask = [np.repeat(x, batch, axis=0) for x in inputs()]
    before = ids.copy(), mask.copy()
    result = b.generate(ids, mask, 4, 3)
    np.testing.assert_array_equal(result, [[2, 4, 7, 7]] * batch)
    assert len(calls) == 2 * batch and not packed_calls
    assert all(not cache for cache in caches)
    assert b._decode_trace is None and not b._trace_failures
    assert not b._last_warmup_owners and not s.active
    for actual, saved in zip((ids, mask), before):
        np.testing.assert_array_equal(actual, saved)
    b.decode = original
    assert b.generate(ids, mask, 4, 3) is expected_packed
    assert len(packed_calls) == 1


def test_wrapped_batch_exact_failure_cleanup_and_recovery(packed_harness, monkeypatch):
    b, s = packed_harness
    b.generation_projection = "last"
    original, device = b.decode, b.device
    sentinel = RuntimeError("wrapped populated cache")
    caches, calls = [], []
    fail = False

    def wrapped(*args, **kw):
        result = original(*args, **kw)
        assert kw["cross_kv"]
        caches.append(kw["cross_kv"])
        calls.append(args[0].copy())
        if fail:
            raise sentinel
        return result

    b.decode = wrapped
    monkeypatch.setattr(trace_decode, "generate_packed", lambda *args: pytest.fail("custom decode bypassed"))
    ids, mask = [np.repeat(x, 2, axis=0) for x in inputs()]
    before = ids.copy(), mask.copy()
    expected = b.generate(ids, mask, 4, 3)
    fail = True
    with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
        b.generate(ids, mask, 4, 3)
    assert caught.value is sentinel
    assert all(not cache for cache in caches)
    assert all(not owner.cross_kv and not owner.inputs and owner.trace_id is None for owner in s.owners)
    assert b._decode_trace is None and not b._trace_failures
    assert not b._last_warmup_owners and not s.active
    fail = False
    np.testing.assert_array_equal(b.generate(ids, mask, 4, 3), expected)
    assert len(calls) == 5 and b.device is device
    for actual, saved in zip((ids, mask), before):
        np.testing.assert_array_equal(actual, saved)


@pytest.mark.parametrize(
    "batch,cap,projection,enabled",
    [(1, 3, "last", True), (2, 1, "last", True), (4, 3, "full", True), (2, 1, "last", False)],
)
def test_nonpacked_dispatch_boundaries(packed_harness, monkeypatch, batch, cap, projection, enabled):
    b, s = packed_harness
    b.generation_projection, b.trace_decoder_enabled = projection, enabled
    monkeypatch.setattr(trace_decode, "generate_packed", lambda *args: pytest.fail("unexpected packed dispatch"))
    ids, mask = [np.repeat(x, batch, axis=0) for x in inputs()]
    result = b.generate(ids, mask, 4, cap)
    np.testing.assert_array_equal(result, [[2, 4] if cap == 1 else [2, 4, 2]] * batch)
    assert s.encodes == batch and not b._trace_failures


@pytest.mark.parametrize("changed", tuple(backend._DISTILLED_600M))
def test_packed_non600m_architecture_falls_back(packed_harness, monkeypatch, changed):
    b, state = packed_harness
    b.generation_projection = "last"
    b.config[changed] += 1
    monkeypatch.setattr(trace_decode, "generate_packed", lambda *args: pytest.fail("non600M packed dispatch"))
    ids, mask = [np.repeat(x, 2, axis=0) for x in inputs()]
    before = ids.copy(), mask.copy()
    result = b.generate(ids, mask, 4, 3)
    np.testing.assert_array_equal(result, [[2, 4, 2]] * 2)
    assert state.encodes == 2 and not b._trace_failures
    assert b._decode_trace is None and not b._last_warmup_owners
    for actual, saved in zip((ids, mask), before):
        np.testing.assert_array_equal(actual, saved)


@pytest.mark.parametrize("precision", ("fp32", "bfp4_b"))
def test_packed_unsupported_precision_falls_back(packed_harness, monkeypatch, precision):
    b, state = packed_harness
    b.generation_projection = "last"
    b.precision_policy["mode"] = precision
    monkeypatch.setattr(
        trace_decode, "generate_packed", lambda *args: pytest.fail("unsupported precision packed dispatch")
    )
    ids, mask = [np.repeat(x, 4, axis=0) for x in inputs()]
    before = ids.copy(), mask.copy()
    result = b.generate(ids, mask, 4, 3)
    np.testing.assert_array_equal(result, [[2, 4, 2]] * 4)
    assert state.encodes == 4 and not b._trace_failures
    assert b._decode_trace is None and not b._last_warmup_owners
    for actual, saved in zip((ids, mask), before):
        np.testing.assert_array_equal(actual, saved)

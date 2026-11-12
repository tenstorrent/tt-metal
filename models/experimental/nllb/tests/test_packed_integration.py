# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""CPU packed/RuntimeOwner integration; simulated TT operations, no native proof."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.experimental.nllb.tests.process_runner import run_task
from models.experimental.nllb.tt import backend
from models.experimental.nllb.tt import trace_decode
from models.experimental.nllb.tests.test_runtime_integration import load_runtime
from models.experimental.nllb.tests.test_trace_decode import (
    Buffer,
    assert_blocked,
    harness as harness,
    inputs,
    packed_harness as packed_harness,
)


@pytest.fixture
def packed_runtime(packed_harness, monkeypatch):
    b, s = packed_harness
    runtime = load_runtime()  # Isolate retained fake devices, never clear real owners.
    b.dim, b.generation_projection = 8, "last"
    b.lm_weight, b.kernel = Buffer(), object()
    s.phase = None
    s.primary = RuntimeError("packed body sentinel")
    s.release_sentinel = RuntimeError("packed release sentinel")
    s.release_kind = None
    s.packed = None
    s.rows = []
    s.live = {}
    s.released = []
    s.closed = []
    s.next_id = 100
    s.open_options = []
    s.cache_enables = []
    s.visibility = []
    b.device = SimpleNamespace(enable_program_cache=lambda: s.cache_enables.append(True))
    b.upload = lambda value: Buffer()
    ops = trace_decode.ttnn

    def observe():
        owner = s.packed
        assert owner in b._last_warmup_owners
        assert b.decode.__func__ is backend._CANONICAL_DECODE
        s.visibility.append(owner)

    initialize = trace_decode.PackedLMRequest.initialize

    def init(owner):
        s.packed = owner
        initialize(owner)
        s.rows = list(owner.rows)

    monkeypatch.setattr(trace_decode.PackedLMRequest, "initialize", init)

    row_body = trace_decode.DecoderTrace.body

    def body(row):
        result = row_body(row)
        observe()
        if s.phase == ("capture_row" if s.active else "warm_row"):
            raise s.primary
        return result

    monkeypatch.setattr(trace_decode.DecoderTrace, "body", body)

    def pack(hidden, indices, output):
        observe()
        assert len(hidden) == len(indices) == len(s.rows)
        if s.phase == ("capture_shared" if s.active else "warm_shared"):
            raise s.primary
        return output

    monkeypatch.setattr(trace_decode, "pack_last_rows", pack)

    def begin(device, **kw):
        observe()
        assert device is b.device and not s.active
        s.active = True
        s.next_id += 1
        s.live[s.next_id] = True
        return s.next_id

    def end(device, trace_id, **kw):
        assert device is b.device and trace_id in s.live and s.active
        s.active = False

    def release(device, trace_id):
        assert device is b.device and not s.active and trace_id in s.live
        owner = next(o for o in [s.packed] + s.rows if o.trace_id == trace_id)
        assert owner.inputs and owner.output is not None
        assert s.packed.selected is not None
        kind = "shared" if owner is s.packed else "row"
        if kind == s.release_kind:
            raise s.release_sentinel
        del s.live[trace_id]
        s.released.append(trace_id)

    def replay(device, trace_id, **kw):
        assert device is b.device and trace_id in s.live and not s.active
        s.replays += 1

    def read(output):
        assert output is s.packed.output and not s.active
        value = torch.zeros(1, 1, len(s.rows), b.vocab)
        value[..., 2 if s.packed.replays else 7] = 1
        return value

    def open_device(**kw):
        s.open_options.append(kw)
        return b.device

    def close_device(device):
        assert device is b.device and not s.live and not s.active
        assert not b._trace_failures and not b._last_warmup_owners
        s.closed.append(device)

    for name, value in dict(
        begin_trace_capture=begin,
        end_trace_capture=end,
        release_trace=release,
        execute_trace=replay,
        copy=lambda *a, **kw: None,
        from_torch=lambda *a, **kw: Buffer(),
        to_torch=read,
        linear=lambda *a, **kw: Buffer(),
        slice=lambda *a, **kw: Buffer(),
        to_layout=lambda value, *a: value,
        bfloat16="bf16",
        uint32="uint32",
        ROW_MAJOR_LAYOUT="row",
        synchronize_device=lambda device: observe(),
        open_device=open_device,
        close_device=close_device,
        is_trace_capture_active=lambda device: s.active,
    ).items():
        monkeypatch.setattr(ops, name, value, raising=False)
    monkeypatch.setitem(sys.modules, "ttnn", ops)
    monkeypatch.setitem(sys.modules, "backend", backend)
    monkeypatch.setenv("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    monkeypatch.setenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    return b, s, runtime


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize(
    "phase,release_kind",
    [
        (None, None),
        ("warm_row", None),
        ("warm_shared", None),
        ("capture_row", None),
        ("capture_shared", None),
        (None, "row"),
        (None, "shared"),
        ("capture_row", "row"),
        ("capture_shared", "shared"),
    ],
)
def test_packed_runtime_lifecycle(packed_runtime, batch, borrowed, phase, release_kind):
    b, s, runtime = packed_runtime
    s.phase, s.release_kind = phase, release_kind
    device = b.device
    ids, mask = [np.repeat(x, batch, axis=0) for x in inputs()]
    before = ids.copy(), mask.copy()
    owner = runtime.RuntimeOwner(device if borrowed else None)
    expected = s.primary if phase else s.release_sentinel if release_kind else None

    def request():
        with owner:
            if not borrowed:
                assert owner.open(0) is device
            assert owner.bind(b) is b
            assert b.decode.__func__ is backend._CANONICAL_DECODE
            return b.generate(ids, mask, 4, 4)

    if expected is not None:
        with pytest.raises(RuntimeError) as caught:  # allow-pytest.raises: CPU-only check.
            request()
        assert caught.value is expected
    else:
        result = request()
        np.testing.assert_array_equal(result, [[2, 4, 7, 2]] * batch)
        assert s.packed.replays == 1
        assert all(row.replays == 1 for row in s.rows)
        assert len(s.released) == batch + 1
    assert s.visibility and all(o is s.packed for o in s.visibility)
    assert b._batch_trace is None and b.device is owner.device is device
    assert owner.models == [b] and not s.active
    assert s.open_options == ([] if borrowed else [{"device_id": 0, **backend.DEVICE_OPTIONS}])
    assert s.cache_enables == ([] if borrowed else [True])
    for value, original in zip((ids, mask), before):
        np.testing.assert_array_equal(value, original)

    if release_kind:
        assert not owner.closed and not s.closed and s.live
        assert runtime.retained_owners() == (owner,)
        assert s.packed.unresolved and s.packed in b._trace_failures
        assert s.packed in b._last_warmup_owners
        failed = [o for o in [s.packed] + s.rows if o.trace_id is not None]
        assert failed and all(o.unresolved and o in b._trace_failures for o in failed)
        assert s.packed.inputs and s.packed.output is not None and s.packed.selected is not None
        assert s.packed.input_ids is ids and s.packed.attention_mask is mask
        assert s.packed.rows == s.rows
        for row in s.rows:
            assert row.inputs and row.output is not None and row.encoder is not None
            assert row.cross_kv
        assert owner.cleanup_errors
        assert_blocked(b)
        with pytest.raises(RuntimeError, match="retained"):  # allow-pytest.raises: CPU-only check.
            runtime.RuntimeOwner(device)
        refs = tuple(s.packed.inputs), tuple(row.output for row in s.rows), dict(s.live)
        owner.finish(expected)
        assert refs == (tuple(s.packed.inputs), tuple(row.output for row in s.rows), dict(s.live))
        assert not s.closed and not owner.closed
    else:
        assert owner.closed and not owner.cleanup_errors
        assert not runtime.retained_owners() and not b._trace_failures
        assert not b._last_warmup_owners and not s.live
        assert s.closed == ([] if borrowed else [device])
        assert not s.packed.rows and s.packed.input_ids is None and s.packed.attention_mask is None
        assert not s.packed.inputs and s.packed.output is None and s.packed.selected is None
        for row in s.rows:
            assert row.trace_id is None and not row.unresolved
            assert not row.inputs and row.output is None and not row.cross_kv and row.encoder is None


def test_runtime_observes_packed_warmup_registry(packed_runtime):
    b, s, runtime = packed_runtime
    ids, mask = [np.repeat(x, 2, axis=0) for x in inputs()]
    owner = runtime.RuntimeOwner()
    owner.open(0)
    owner.bind(b)
    packed = trace_decode.PackedLMRequest(b, ids, mask)
    packed.initialize()
    packed.decode_rows([[2, 4], [2, 4]], [0, 1])
    packed.decode_rows([[2, 4, 7], [2, 4, 7]], [0, 1])
    assert not b._trace_failures and b._last_warmup_owners == {packed}
    assert getattr(b, "_batch_trace", None) is None
    assert getattr(b, "_decode_trace", None) is None
    assert len(s.live) == 3
    owner.finish()
    assert owner.closed and s.closed == [b.device]
    assert len(s.released) == 3 and not s.live and not b._last_warmup_owners
    assert packed.trace_id is None and packed.output is None and packed.selected is None
    assert all(row.trace_id is None and not row.inputs and row.output is None for row in s.rows)
    packed.finish()
    assert not packed.rows and not runtime.retained_owners()


def test_portable_packed_native(nllb_device_id, tmp_path):
    """Opt-in native observer; CPU fault matrix above remains independent."""
    import os
    from pathlib import Path
    import json

    checkpoint = os.environ.get("NLLB_TEST_CHECKPOINT")
    config = os.environ.get("NLLB_TEST_CONFIG")
    tokenizer = os.environ.get("NLLB_TEST_TOKENIZER")
    if not all((checkpoint, config, tokenizer)):
        pytest.skip("set checkpoint, config and tokenizer for native packed observation")
    output = Path(os.environ.get("NLLB_PACKED_OUTPUT", str(tmp_path / "packed")))
    result = run_task(
        "packed",
        dict(
            checkpoint=checkpoint,
            config=config,
            tokenizer_directory=tokenizer,
            device=nllb_device_id,
            output=str(output),
        ),
        timeout=240,
    )
    print(result.stdout)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "complete.json").read_text())
    assert report["passed"] and report["device_closed"] and not report["retained"]
    assert json.loads((output / "public_b4.json").read_text())["decode_canonical"]
    assert json.loads((output / "mixed_eos.json").read_text())["active_rows"] == [[0, 1, 2, 3], [1, 2, 3], [1, 2, 3]]


@pytest.fixture
def identity_package(tmp_path):
    import hashlib
    import json

    entries = []
    for name in ("tt/backend.py", "tt/trace_decode.py", "tt/runtime_setup.py"):
        data = (name + "\n").encode()
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(data)
        entries.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (tmp_path / "PACKAGE_FILES.json").write_text(json.dumps({"files": entries}))
    return tmp_path, entries


def test_manifest_identity_valid_and_refreshed(identity_package):
    import hashlib
    import json
    from models.experimental.nllb.tests.probe_packed_integration import package_identity

    root, entries = identity_package
    assert package_identity(root) == {e["path"]: e["sha256"] for e in entries}
    data = b"# formatted replacement\n"
    (root / entries[0]["path"]).write_bytes(data)
    entries[0].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest().upper())
    (root / "PACKAGE_FILES.json").write_text(json.dumps({"files": entries}))
    assert package_identity(root)[entries[0]["path"]] == entries[0]["sha256"].lower()


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("fault", ["missing", "duplicate", "hash", "size", "changed"])
def test_manifest_identity_rejects_invalid(identity_package, index, fault):
    import json
    from models.experimental.nllb.tests.probe_packed_integration import package_identity

    root, entries = identity_package
    if fault == "missing":
        entries.pop(index)
    elif fault == "duplicate":
        entries.append(dict(entries[index]))
    elif fault == "hash":
        entries[index]["sha256"] = "g" * 64
    elif fault == "size":
        entries[index]["bytes"] = True
    else:
        path = root / entries[index]["path"]
        data = path.read_bytes()
        path.write_bytes(b"!" + data[1:])  # Same size; digest must detect the change.
    (root / "PACKAGE_FILES.json").write_text(json.dumps({"files": entries}))
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only check.
        package_identity(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", None),
        ("sha256", 42),
        ("sha256", ""),
        ("sha256", "a" * 63),
        ("sha256", "0" * 64),
        ("bytes", None),
        ("bytes", "1"),
        ("bytes", -1),
        ("bytes", 1.0),
        ("bytes", 0),
    ],
)
def test_manifest_identity_malformed_fields(identity_package, field, value):
    import json
    from models.experimental.nllb.tests.probe_packed_integration import package_identity

    root, entries = identity_package
    entries[0][field] = value
    (root / "PACKAGE_FILES.json").write_text(json.dumps({"files": entries}))
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only check.
        package_identity(root)

"""Host test of the weight-cache checkpoint stamp (no device)."""
import json

import pytest

from models.demos.blackhole.deepseek_v41_flash.tt import ckpt_stamp
from models.demos.blackhole.deepseek_v41_flash.tt.ckpt_stamp import checkpoint_id, ensure_stamp


def _ckpt(root, name, sizes, index=b"{}", values=False):
    d = root / name
    d.mkdir()
    for i, n in enumerate(sizes):
        (d / f"model-0000{i}-of-0000{len(sizes)}.safetensors").write_bytes(b"x" * n)
    if values:
        (d / ("model-00000-of-0000%d.safetensors" % len(sizes))).write_bytes(b"y" * sizes[0])
    (d / "model.safetensors.index.json").write_bytes(index)
    return str(d)


@pytest.fixture(autouse=True)
def _clear():
    ckpt_stamp._checked.clear()
    ckpt_stamp._ids.clear()


def test_stamp_written_on_empty_root(tmp_path):
    a = _ckpt(tmp_path, "a", [3, 5])
    root = tmp_path / "cache" / "weights"  # does not exist yet
    ensure_stamp(str(root), a)
    assert json.load(open(root / "ckpt_id.json"))["ckpt_id"] == checkpoint_id(a)


def test_same_ckpt_passes_again(tmp_path):
    a = _ckpt(tmp_path, "a", [3, 5])
    root = tmp_path / "c"
    root.mkdir()
    ensure_stamp(str(root), a)
    ckpt_stamp._checked.clear()
    ckpt_stamp._ids.clear()
    ensure_stamp(str(root), a)


def test_different_ckpt_raises(tmp_path):
    a = _ckpt(tmp_path, "a", [3, 5])
    b = _ckpt(tmp_path, "b", [3, 6])  # same names, different size
    c = _ckpt(tmp_path, "c", [3, 5], index=b'{"x": 1}')  # different index
    d = _ckpt(tmp_path, "d", [3, 5], values=True)  # same names, sizes and index, different values
    root = tmp_path / "c_root"
    ensure_stamp(str(root), a)
    for other in (b, c, d):
        assert checkpoint_id(other) != checkpoint_id(a)
        try:
            ensure_stamp(str(root), other)
        except RuntimeError as e:
            msg = str(e)
        else:
            raise AssertionError("different checkpoint did not raise")
        assert checkpoint_id(a) in msg and checkpoint_id(other) in msg and str(root) in msg


def test_disabled_cache_is_noop(tmp_path):
    ensure_stamp(None, "/nonexistent")
    ensure_stamp("0", "/nonexistent")

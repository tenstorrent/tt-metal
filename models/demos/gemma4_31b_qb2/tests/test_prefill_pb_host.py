# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests of the paged-bucket prefill (tt/prefill_paged*.py, tt/*_pb.py). Nothing here imports ttnn, so
they never touch a device: the key arithmetic is imported from its import-free module, and the copied method bodies
are compared by AST source extraction.

  python -m pytest -q models/demos/gemma4_31b_qb2/tests/test_prefill_pb_host.py
"""

import ast
import difflib
import importlib.util
import random
import textwrap
from pathlib import Path

import pytest

TT = Path(__file__).resolve().parents[1] / "tt"


def _keys():
    spec = importlib.util.spec_from_file_location("prefill_paged_keys", TT / "prefill_paged_keys.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


K = _keys()


# ------------------------------------------------------------------ arithmetic


def test_pages128():
    assert [K.pages128(n) for n in (1, 127, 128, 129, 8191, 8192, 8193)] == [128, 128, 128, 256, 8192, 8192, 8320]
    with pytest.raises(ValueError):
        K.pages128(0)


def test_start_class_matches_sdpa_eligibility():
    """The shipped _prefill_sdpa_config takes the Q chunks c in (32, 64, 128, 256) with rows % c == 0 and
    start % c == 0; for 128-aligned starts that set depends only on start % 256 (the class)."""
    chunks = (32, 64, 128, 256)
    for start in range(128, 300_000, 128):
        eligible = {c for c in chunks if start % c == 0}
        rep = K.start_class(start)
        assert eligible == {c for c in chunks if rep % c == 0}, start
    assert K.start_class(0) == 0
    with pytest.raises(ValueError):
        K.start_class(100)


def test_history_rows():
    assert [K.history_rows(a) for a in (128, 1024, 1152, 1280, 262016)] == [128, 1024, 1152, 1152, 1152]


def test_row_key_first_and_continuation():
    assert K.row_key(1000, 0) == ("first", 1024)
    assert K.row_key(3200, 2176) == ("cont", 1024, 128, 1152, 0, None)
    assert K.row_key(3072, 2048) == ("cont", 1024, 256, 1152, 0, None)
    assert K.row_key(3200, 2176, flex=False) == ("cont", 1024, 128, 1152, 0, 2176)
    # an unaligned scheduler start rewinds to its page: rows [2048, 3100) -> 1052 -> 1152
    assert K.row_key(3100, 2100) == ("cont", 1152, 256, 1152, 1, None)
    # small starts carry their history length
    assert K.row_key(640, 384) == ("cont", 256, 128, 384, 0, None)


def test_call_keys_include_row_count():
    keys = K.call_keys([1000, 1001], [0, 0])
    assert keys == {("first", 1024), ("rows", 2)}
    with pytest.raises(ValueError):
        K.call_keys([1], [0, 0])


def test_first_buckets_and_warmed_keys():
    buckets = K.first_buckets()
    assert buckets[0] == 128 and buckets[-1] == 8192 and len(buckets) == 64
    warmed = K.warmed_keys(32)
    assert len(warmed) == 64 + 32
    # every first chunk of a prompt the 8,192-token budget admits is warmed
    for n in range(1, 8193):
        assert K.row_key(n, 0) in warmed


def test_terminal_slices_cover_every_length():
    for b in K.first_buckets():
        starts = set(K.terminal_slices(b))
        assert len(starts) == 4
        for n in range(b - 127, b + 1):
            assert (n - 1) // 32 * 32 in starts, (b, n)


def test_tap_windows_cover_every_length():
    seed = 2080  # DFlash2Drafter.SEED_ROWS; dflash.tap_window: end = ceil32(n), start = max(0, end - SEED_ROWS)
    for b in K.first_buckets():
        windows = set(K.tap_windows(b, seed))
        for n in range(b - 127, b + 1):
            end = (n + 31) // 32 * 32
            assert (max(0, end - seed), end) in windows, (b, n)


def test_continuation_starts_cover_every_class_and_history():
    reps = {(K.start_class(a), K.history_rows(a)) for a in K.continuation_starts()}
    for a in range(128, 262_144, 128):
        assert (K.start_class(a), K.history_rows(a)) in reps, a
    assert len(reps) == len(K.continuation_starts()) == 10


def test_continuation_key_space_is_bounded_and_complete():
    space = K.continuation_key_space()
    assert len(space) <= 10 * 65
    rng = random.Random(20261113)
    for _ in range(20_000):
        start = rng.randrange(128, 262_000, 128)  # the plugin aligns partial chunk ends to 128
        length = start + rng.randint(1, 8192)
        assert K.row_key(length, start) in space, (length, start)


def test_retirements_per_lifetime_bound():
    """Lazy keys (continuations) retire traces once each: at most the continuation key space (650 under the plugin's
    128 alignment; 1,300 if unaligned continuations also occur)."""
    assert len(K.continuation_key_space()) <= 650
    assert len(K.continuation_key_space(unaligned=True)) <= 1300
    rng = random.Random(7)
    space = K.continuation_key_space(unaligned=True)
    for _ in range(20_000):
        start = rng.randrange(128, 262_000)  # a continuation (aligned start >= 128)
        assert K.row_key(start + rng.randint(1, 8192), start) in space


# ------------------------------------------------------------------ copied bodies: only `# PB` hunks differ


def _method_source(path, cls, name):
    src = path.read_text()
    lines = src.splitlines()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return textwrap.dedent("\n".join(lines[f.lineno - 1 : f.end_lineno])).splitlines()
    raise KeyError((path.name, cls, name))


COPIES = [
    ("decoder.py", "Decoder", "prefill_forward", "prefill_paged.py", "DecoderPB"),
    ("decoder.py", "Decoder", "_prefill_qkv", "prefill_paged.py", "DecoderPB"),
    ("model.py", "Gemma4Model", "embed", "prefill_paged.py", "Gemma4ModelPB"),
    ("model.py", "Gemma4Model", "prefill_device", "prefill_paged.py", "Gemma4ModelPB"),
    ("generator.py", "Gemma4Generator", "prefill_forward", "prefill_paged.py", "GeneratorPBMixin"),
    (
        "generator_resident_spec_vllm.py",
        "Gemma4ResidentSpecForCausalLM",
        "prefill_forward",
        "generator_resident_spec_vllm_pb.py",
        "Gemma4ResidentSpecPBForCausalLM",
    ),
]


@pytest.mark.parametrize("shipped_file,shipped_cls,name,pb_file,pb_cls", COPIES)
def test_copies_differ_only_in_marked_hunks(shipped_file, shipped_cls, name, pb_file, pb_cls):
    shipped = _method_source(TT / shipped_file, shipped_cls, name)
    pb = _method_source(TT / pb_file, pb_cls, name)
    hunks = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, shipped, pb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        hunks += 1
        changed = pb[j1:j2] if j2 > j1 else pb[max(0, j1 - 1) : j1]
        assert any("# PB" in line for line in changed), (tag, shipped[i1:i2], pb[j1:j2])
        for line in pb[j1:j2]:
            # a changed line is either marked or a continuation inside a marked statement
            assert "# PB" in line or line.strip() in ("", ")", "],", "}", "):") or line.rstrip().endswith(","), line
    assert hunks > 0


def test_shipped_files_untouched_by_the_pb_build():
    """The PB build adds files only: the shared-path files equal the package tree's (update 1, e1d6a5f523c)."""
    pkg = Path("/home/ttuser/experiments/gemma4/tt-metal-pkg/models/demos/gemma4_31b_qb2/tt")
    if not pkg.exists():
        pytest.skip("package worktree not present")
    for name in ("decoder.py", "model.py", "generator.py", "generator_vllm.py", "generator_resident_vllm.py",
                 "generator_resident_spec_vllm.py", "dflash.py", "dflash2.py", "resident_taps.py"):
        assert (TT / name).read_bytes() == (pkg / name).read_bytes(), name

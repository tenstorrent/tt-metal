# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ast
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from layer_ttnn_cache import (
    FileResult,
    FileSet,
    GuardError,
    StageTimer,
    check_env_paths,
    check_out_root,
    check_ttnn_config,
    compare_tensorbin,
    data_equal,
    exit_code,
    files_equal,
    format_profile,
    format_summary,
    fstype_of,
    install_timers,
    layer_files,
    layer_verdict,
    main,
    make_dump_guard,
    parse_args,
    parse_layers,
    read_mounts,
    restore_timers,
    snapshot_diff,
    split_fileset,
    stage_for_file,
    weka_snapshot,
)

from models.demos.deepseek_v3_d_p.utils.transformer_helpers import block_cache_state_dict

MLA_KEYS = [
    "q_a_proj.weight",
    "q_a_layernorm.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
]
INDEXER_SRC = {
    "indexer.wq_b.weight": "self_attn.indexer.wq_b.weight",
    "indexer.wk.weight": "self_attn.indexer.wk.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
    "indexer.k_norm_bias.weight": "self_attn.indexer.k_norm.bias",
    "indexer.weights_proj.weight": "self_attn.indexer.weights_proj.weight",
}


def _layer(dense=True, indexer=False, n_routed=4):
    keys = ["input_layernorm.weight", "post_attention_layernorm.weight"]
    keys += [f"self_attn.{k}" for k in MLA_KEYS]
    if indexer:
        keys += list(INDEXER_SRC.values())
    if dense:
        keys += [f"mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    else:
        keys += [f"mlp.experts.{j}.{p}.weight" for j in range(n_routed) for p in ("gate_proj", "up_proj", "down_proj")]
        keys += ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]
        keys += [f"mlp.shared_experts.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    return {k: torch.zeros(1) for k in keys}


def test_block_cache_state_dict_dense_no_indexer():
    sd = _layer(dense=True)
    out = block_cache_state_dict(sd, is_dense=True, n_routed=4)
    assert list(out) == ["attn_norm_weight", "mla_weights", "ffn_norm_weight", "ffn_weights"]
    assert list(out["mla_weights"]) == MLA_KEYS
    assert out["attn_norm_weight"] is sd["input_layernorm.weight"]
    assert out["ffn_norm_weight"] is sd["post_attention_layernorm.weight"]
    for k in MLA_KEYS:
        assert out["mla_weights"][k] is sd[f"self_attn.{k}"]
    for p in ("gate_proj", "up_proj", "down_proj"):
        assert out["ffn_weights"][p] is sd[f"mlp.{p}.weight"]


def test_block_cache_state_dict_dense_with_indexer():
    sd = _layer(dense=True, indexer=True)
    out = block_cache_state_dict(sd, is_dense=True, n_routed=4)
    assert list(out["mla_weights"]) == MLA_KEYS + list(INDEXER_SRC)
    for dst, src in INDEXER_SRC.items():
        assert out["mla_weights"][dst] is sd[src]
    assert out["mla_weights"]["indexer.k_norm_bias.weight"] is sd["self_attn.indexer.k_norm.bias"]


def test_block_cache_state_dict_moe():
    sd = _layer(dense=False, n_routed=4)
    out = block_cache_state_dict(sd, is_dense=False, n_routed=4)
    assert "ffn_weights" not in out
    assert list(out) == [
        "attn_norm_weight",
        "mla_weights",
        "ffn_norm_weight",
        "gate_weights",
        "routed_expert_weights",
        "shared_expert_weights",
    ]
    assert out["gate_weights"]["weight"] is sd["mlp.gate.weight"]
    assert out["gate_weights"]["e_score_correction_bias"] is sd["mlp.gate.e_score_correction_bias"]
    assert len(out["routed_expert_weights"]) == 4
    for j, e in enumerate(out["routed_expert_weights"]):
        for p in ("gate_proj", "up_proj", "down_proj"):
            assert e[p] is sd[f"mlp.experts.{j}.{p}.weight"]
    for p in ("gate_proj", "up_proj", "down_proj"):
        assert out["shared_expert_weights"][p] is sd[f"mlp.shared_experts.{p}.weight"]


SCRIPT = Path(__file__).with_name("layer_ttnn_cache.py")
WEKA_GUARD_TEST_PATH = "/mnt/weka/model-cache/scratch/__layer_cache_guard_test__"


# ---- Task 6: CLI


@pytest.mark.parametrize(
    "spec, expected",
    [("5", [5]), ("0-10", list(range(11))), ("1,3,2", [1, 2, 3]), ("3-3", [3]), ("2,2,1-2", [1, 2])],
)
def test_parse_layers_ok(spec, expected):
    assert parse_layers(spec, 61) == expected


@pytest.mark.parametrize("spec", ["", "a", "5-3", "-1", "61", "1,,2", "1-", " ", "1-a", "60-61"])
def test_parse_layers_rejects(spec):
    with pytest.raises(ValueError):
        parse_layers(spec, 61)


def test_parse_args_defaults():
    a = parse_args(["--model", "kimi_k2_7", "--layers", "1"])
    assert a.out_root == "~/ttnn-layer-cache"
    assert a.keep is False
    assert a.min_free_gb == 40
    assert a.weka_cache is None and a.hf_model is None


def test_parse_args_rejects_unknown_model():
    with pytest.raises(SystemExit) as e:
        parse_args(["--model", "glm53", "--layers", "1"])
    assert e.value.code == 2


def test_script_has_no_top_level_heavy_imports():
    tree = ast.parse(SCRIPT.read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0])
    assert not top & {"ttnn", "torch", "models", "transformers", "numpy"}


# ---- Task 7: path and env guards


def _mounts(*extra):
    lines = ["/dev/root / ext4 rw 0 0", "weka0 /mnt/weka wekafs rw 0 0"] + [f"dev {mp} {fs} rw 0 0" for mp, fs in extra]
    return read_mounts("\n".join(lines))


def test_read_mounts_unescapes():
    m = read_mounts("dev /mnt/with\\040space ext4 rw 0 0\nweka0 /mnt/weka wekafs rw,relatime 0 0\n")
    assert m == [("/mnt/with space", "ext4"), ("/mnt/weka", "wekafs")]


def test_fstype_of_longest_prefix_of_nearest_existing(tmp_path):
    (tmp_path / "wk").mkdir()
    mounts = _mounts((str(tmp_path / "wk"), "wekafs"), (str(tmp_path), "xfs"))
    assert fstype_of(tmp_path / "wk" / "missing" / "deeper", mounts) == "wekafs"
    assert fstype_of(tmp_path / "other", mounts) == "xfs"
    assert fstype_of(Path("/usr"), mounts) == "ext4"


@pytest.fixture
def fake(tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "fakeweka"
    home.mkdir()
    root.mkdir()
    return home, root


def test_check_out_root_accepts_home_child(fake):
    home, root = fake
    out = check_out_root(str(home / "ttnn-layer-cache"), home, _mounts(), roots=(str(root),))
    assert out == (home / "ttnn-layer-cache").resolve()
    assert not out.exists()


def test_check_out_root_rejects_root_child(fake):
    home, root = fake
    with pytest.raises(GuardError):
        check_out_root(str(root / "x"), home, _mounts(), roots=(str(root),))
    assert not (root / "x").exists()


def test_check_out_root_rejects_symlink_into_root(fake):
    home, root = fake
    (home / "link").symlink_to(root)
    with pytest.raises(GuardError):
        check_out_root(str(home / "link" / "x"), home, _mounts(), roots=(str(root),))
    assert not (root / "x").exists()


def test_check_out_root_rejects_wekafs_mount(fake):
    home, root = fake
    (home / "wk").mkdir()
    mounts = _mounts((str(home / "wk"), "wekafs"))
    with pytest.raises(GuardError):
        check_out_root(str(home / "wk" / "x"), home, mounts, roots=(str(root),))
    assert not (home / "wk" / "x").exists()


def test_check_out_root_rejects_outside_home(fake, tmp_path):
    home, root = fake
    with pytest.raises(GuardError):
        check_out_root(str(tmp_path / "elsewhere"), home, _mounts(), roots=(str(root),))
    with pytest.raises(GuardError):
        check_out_root(str(home), home, _mounts(), roots=(str(root),))


def test_check_out_root_rejects_home_on_wekafs(fake):
    home, root = fake
    mounts = _mounts((str(home), "wekafs"))
    with pytest.raises(GuardError, match="home"):
        check_out_root(str(home / "ttnn-layer-cache"), home, mounts, roots=(str(root),))


ENV_VARS = [
    "TT_METAL_CACHE",
    "TT_METAL_LOGS_PATH",
    "TT_METAL_HOME",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_MODULES_CACHE",
    "TRANSFORMERS_CACHE",
    "PYTHONPYCACHEPREFIX",
]


@pytest.mark.parametrize("var", ENV_VARS)
def test_check_env_paths_rejects(fake, var):
    home, root = fake
    with pytest.raises(GuardError, match=var):
        check_env_paths({var: str(root / "cache")}, _mounts(), roots=(str(root),))


def test_check_env_paths_ignores_unset_and_safe(fake):
    home, root = fake
    check_env_paths({"HOME": str(home), "TMPDIR": "", "UNRELATED": str(root)}, _mounts(), roots=(str(root),))


def test_layer_files_prefix(tmp_path):
    for name, size in [("layer_1.a", 3), ("layer_1.b.tensorbin", 5), ("layer_10.a", 7), ("layer_1x", 1), ("x", 1)]:
        (tmp_path / name).write_bytes(b"\0" * size)
    (tmp_path / "layer_1.dir").mkdir()
    assert layer_files(tmp_path, 1) == {"layer_1.a": 3, "layer_1.b.tensorbin": 5}
    assert layer_files(tmp_path, 10) == {"layer_10.a": 7}
    assert layer_files(tmp_path / "missing", 1) == {}


def test_main_rejects_weka_out_root_before_run(monkeypatch):
    import layer_ttnn_cache as script

    monkeypatch.setattr(script, "run_phase", lambda ctx: pytest.fail("run_phase reached"))
    assert main(["--model", "kimi_k2_7", "--layers", "1", "--out-root", WEKA_GUARD_TEST_PATH]) == 2
    assert not os.path.exists(WEKA_GUARD_TEST_PATH)


def test_main_rejects_weka_env_before_run(monkeypatch):
    import layer_ttnn_cache as script

    monkeypatch.setattr(script, "run_phase", lambda ctx: pytest.fail("run_phase reached"))
    monkeypatch.setenv("TT_METAL_CACHE", WEKA_GUARD_TEST_PATH)
    assert main(["--model", "kimi_k2_7", "--layers", "1"]) == 2
    assert not os.path.exists(WEKA_GUARD_TEST_PATH)


def test_main_rejects_bad_layer_spec(monkeypatch, tmp_path):
    import layer_ttnn_cache as script

    monkeypatch.setattr(script, "run_phase", lambda ctx: pytest.fail("run_phase reached"))
    assert main(["--model", "kimi_k2_7", "--layers", "61", "--out-root", "~/ttnn-layer-cache"]) == 2


# ---- Task 8: runtime guards

AUDIT_CHILD = r"""
import os, shutil, sys
sys.path.insert(0, sys.argv[1])
from layer_ttnn_cache import make_audit_hook
fake, out = sys.argv[2], sys.argv[3]
sys.addaudithook(make_audit_hook(roots=(fake,)))

def blocked(fn):
    try:
        fn()
    except PermissionError:
        return True
    return False

def run(base, expect_blocked):
    j = os.path.join
    checks = {
        "open_w": lambda: open(j(base, "a"), "w").close(),
        "open_a": lambda: open(j(base, "existing"), "a").close(),
        "open_rplus": lambda: open(j(base, "existing"), "r+").close(),
        "os_open_creat": lambda: os.close(os.open(j(base, "c"), os.O_WRONLY | os.O_CREAT)),
        "mkdir": lambda: os.mkdir(j(base, "d")),
        "remove": lambda: os.remove(j(base, "victim1")),
        "rename": lambda: os.rename(j(base, "victim2"), j(base, "renamed")),
        "replace": lambda: os.replace(j(base, "victim3"), j(base, "replaced")),
        "rmdir": lambda: os.rmdir(j(base, "emptydir")),
        "truncate": lambda: os.truncate(j(base, "existing"), 0),
        "chmod": lambda: os.chmod(j(base, "existing"), 0o644),
        "utime": lambda: os.utime(j(base, "existing")),
        "symlink": lambda: os.symlink("/etc/hostname", j(base, "sl")),
        "link": lambda: os.link(j(base, "existing"), j(base, "hl")),
        "rmtree": lambda: shutil.rmtree(j(base, "tree")),
    }
    bad = [k for k, fn in checks.items() if blocked(fn) != expect_blocked]
    with open(j(base, "existing"), "rb") as f:
        f.read()
    os.close(os.open(j(base, "existing"), os.O_RDONLY))
    return bad

bad_fake = run(fake, True)
bad_out = run(out, False)
print("BAD_FAKE", bad_fake)
print("BAD_OUT", bad_out)
sys.exit(1 if bad_fake or bad_out else 0)
"""


def _seed(base: Path):
    base.mkdir(parents=True, exist_ok=True)
    for name in ("existing", "victim1", "victim2", "victim3"):
        (base / name).write_bytes(b"x")
    (base / "emptydir").mkdir()
    (base / "tree").mkdir()
    (base / "tree" / "f").write_bytes(b"x")


def _listing(base: Path):
    return sorted((str(p.relative_to(base)), p.stat().st_size, p.stat().st_mtime_ns) for p in base.rglob("*"))


def test_audit_hook_blocks_fake_root_and_allows_out(tmp_path):
    fake, out = tmp_path / "fakeweka", tmp_path / "out"
    _seed(fake)
    _seed(out)
    before = _listing(fake)
    r = subprocess.run(
        [sys.executable, "-c", AUDIT_CHILD, str(SCRIPT.parent), str(fake), str(out)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert _listing(fake) == before
    assert not (fake / "a").exists()
    assert (out / "a").exists() and (out / "renamed").exists() and not (out / "tree").exists()


def test_dump_guard(tmp_path):
    fake, out = tmp_path / "fakeweka", tmp_path / "out"
    fake.mkdir()
    out.mkdir()
    (out / "link").symlink_to(fake)
    calls = []
    guard = make_dump_guard(lambda path, *a, **k: calls.append((path, a, k)), out)
    guard(str(out / "x.tensorbin"), "T", "MODE")
    assert calls == [(str(out / "x.tensorbin"), ("T", "MODE"), {})]
    for bad in (fake / "x.tensorbin", out / ".." / "x.tensorbin", out / "link" / "x.tensorbin"):
        with pytest.raises(PermissionError):
            guard(str(bad), "T", "MODE")
    assert len(calls) == 1
    assert list(fake.iterdir()) == []


def test_check_ttnn_config(fake, tmp_path, monkeypatch):
    home, root = fake
    ok = {"cache_path": str(home / ".cache/ttnn"), "model_cache_path": "", "tmp_dir": "/tmp/ttnn"}
    check_ttnn_config({**ok, "root_report_path": "generated/ttnn/reports"}, _mounts(), roots=(str(root),))
    for key in ("cache_path", "model_cache_path", "tmp_dir", "root_report_path"):
        with pytest.raises(GuardError, match=key):
            check_ttnn_config({**ok, key: str(root / "c")}, _mounts(), roots=(str(root),))
    monkeypatch.chdir(root)
    with pytest.raises(GuardError, match="root_report_path"):
        check_ttnn_config({"root_report_path": "generated/ttnn/reports"}, _mounts(), roots=(str(root),))
    assert list(root.iterdir()) == []


# ---- Task 9: file set and .tensorbin compare


def write_tb(path: Path, header: bytes, data: bytes):
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def _meta(**over):
    m = {"shape": "S", "padded_shape": "P", "dtype": "BF16", "layout": "TILE", "tile": "T", "shard_count": "32"}
    m.update(over)
    return lambda p: dict(m)


def _no_meta(p):
    raise AssertionError("load_meta called")


def test_split_fileset():
    fs = split_fileset({"c": 1, "a": 1, "b": 1}, {"b": 1, "a": 1, "w": 1})
    assert fs == FileSet(common=["a", "b"], missing_on_weka=["c"], weka_only=["w"])


DATA = bytes(range(256)) * 40


def test_compare_identical(tmp_path):
    write_tb(tmp_path / "a", b"H" * 24, DATA)
    write_tb(tmp_path / "b", b"H" * 24, DATA)
    assert compare_tensorbin(tmp_path / "a", tmp_path / "b", _no_meta, chunk=7).status == "EQUAL"


def test_compare_header_only_diff(tmp_path):
    write_tb(tmp_path / "a", b"H" * 40, DATA)
    write_tb(tmp_path / "b", b"G" * 32, DATA)
    r = compare_tensorbin(tmp_path / "a", tmp_path / "b", _meta(), chunk=7)
    assert r.status == "EQUAL_DATA", r


def test_compare_data_diff(tmp_path):
    flipped = bytearray(DATA)
    flipped[1000] ^= 0xFF
    write_tb(tmp_path / "a", b"H" * 24, bytes(flipped))
    write_tb(tmp_path / "b", b"H" * 24, DATA)
    r = compare_tensorbin(tmp_path / "a", tmp_path / "b", _no_meta, chunk=7)
    assert r == FileResult("DATA_DIFF", "first data diff at data offset 1000")


def test_compare_data_size_diff(tmp_path):
    write_tb(tmp_path / "a", b"H" * 24, DATA + b"x")
    write_tb(tmp_path / "b", b"H" * 24, DATA)
    assert compare_tensorbin(tmp_path / "a", tmp_path / "b", _no_meta).status == "DATA_DIFF"


def test_compare_meta_diff(tmp_path):
    write_tb(tmp_path / "a", b"H" * 40, DATA)
    write_tb(tmp_path / "b", b"G" * 32, DATA)
    calls = iter([_meta(dtype="BFLOAT8_B")(None), _meta(dtype="BFLOAT4_B")(None)])
    r = compare_tensorbin(tmp_path / "a", tmp_path / "b", lambda p: next(calls))
    assert r == FileResult("META_DIFF", "dtype: BFLOAT8_B != BFLOAT4_B")


def test_compare_truncated_weka_header(tmp_path):
    write_tb(tmp_path / "a", b"H" * 24, DATA)
    (tmp_path / "b").write_bytes(struct.pack("<Q", 10_000) + b"H" * 20)
    r = compare_tensorbin(tmp_path / "a", tmp_path / "b", _no_meta)
    assert r.status == "COMPARE_ERROR" and r.detail.startswith("COMPARE_ERROR: ") and "truncated" in r.detail


def test_compare_missing_weka(tmp_path):
    write_tb(tmp_path / "a", b"H" * 24, DATA)
    r = compare_tensorbin(tmp_path / "a", tmp_path / "missing", _no_meta)
    assert r.status == "COMPARE_ERROR" and r.detail.startswith("COMPARE_ERROR: ")


def test_compare_load_meta_raises(tmp_path):
    write_tb(tmp_path / "a", b"H" * 40, DATA)
    write_tb(tmp_path / "b", b"G" * 32, DATA)

    def boom(p):
        raise RuntimeError("bad header")

    r = compare_tensorbin(tmp_path / "a", tmp_path / "b", boom)
    assert r.status == "COMPARE_ERROR" and "bad header" in r.detail


def test_files_and_data_equal_cross_chunk_edges(tmp_path):
    write_tb(tmp_path / "a", b"H" * 24, DATA)
    write_tb(tmp_path / "b", b"H" * 24, DATA)
    for chunk in (1, 7, 64, 1 << 20):
        assert files_equal(tmp_path / "a", tmp_path / "b", chunk)
        assert data_equal(tmp_path / "a", tmp_path / "b", chunk)
    last = bytearray(DATA)
    last[-1] ^= 1
    write_tb(tmp_path / "c", b"H" * 24, bytes(last))
    for chunk in (1, 7, 64, 1 << 20):
        assert not files_equal(tmp_path / "a", tmp_path / "c", chunk)
        assert not data_equal(tmp_path / "a", tmp_path / "c", chunk)


KIMI_WEKA = Path(
    "/mnt/weka/model-cache/scratch/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill/kimi_k2_7_bh_32dev/8x4"
)


@pytest.mark.skipif(not KIMI_WEKA.is_dir(), reason="Weka cache not mounted")
def test_compare_real_tensorbin_against_weka(tmp_path):
    from layer_ttnn_cache import tensor_meta

    src = min(KIMI_WEKA.glob("layer_0.attn_norm*.tensorbin"), key=lambda p: p.stat().st_size)
    copy = tmp_path / src.name
    shutil.copyfile(src, copy)
    assert compare_tensorbin(copy, src, tensor_meta).status == "EQUAL"
    meta = tensor_meta(copy)
    assert meta["shape"] and meta["dtype"] and meta["shard_count"] == "32"
    assert "TensorTopology" in meta["tensor_topology"]
    raw = bytearray(copy.read_bytes())
    raw[-1] ^= 0xFF
    copy.write_bytes(bytes(raw))
    assert compare_tensorbin(copy, src, tensor_meta).status == "DATA_DIFF"


# ---- Task 10: verdict, exit code, Weka snapshot

FS_OK = FileSet(["a", "b"], [], ["w"])


@pytest.mark.parametrize(
    "build_error, complete, fs, statuses, expected",
    [
        ("Traceback", True, FS_OK, ["EQUAL", "EQUAL"], "FAIL"),
        (None, False, FS_OK, ["EQUAL", "EQUAL"], "FAIL"),
        (None, True, FS_OK, ["EQUAL", "EQUAL"], "PASS"),
        (None, True, FS_OK, ["EQUAL_DATA", "EQUAL_DATA"], "PASS"),
        (None, True, FS_OK, ["EQUAL", "DATA_DIFF"], "DIFF"),
        (None, True, FS_OK, ["EQUAL", "META_DIFF"], "DIFF"),
        (None, True, FS_OK, ["EQUAL", "COMPARE_ERROR"], "DIFF"),
        (None, True, FileSet(["a", "b"], ["c"], []), ["EQUAL", "EQUAL"], "DIFF"),
    ],
)
def test_layer_verdict(build_error, complete, fs, statuses, expected):
    results = {n: FileResult(s, "") for n, s in zip(fs.common, statuses)}
    assert layer_verdict(build_error, complete, fs, results) == expected


@pytest.mark.parametrize(
    "verdicts, changed, interrupted, expected",
    [
        (["PASS", "PASS"], [], False, 0),
        (["PASS", "DIFF"], [], False, 1),
        (["DIFF", "FAIL"], [], False, 3),
        (["PASS"], ["x"], False, 4),
        (["FAIL"], ["x"], False, 4),
        (["FAIL", "DIFF"], [], True, 130),
        (["PASS"], ["x"], True, 4),
    ],
)
def test_exit_code(verdicts, changed, interrupted, expected):
    assert exit_code(verdicts, changed, interrupted) == expected


def test_weka_snapshot_and_diff(tmp_path):
    (tmp_path / "layer_1.a").write_bytes(b"12")
    (tmp_path / "layer_1.b").write_bytes(b"1")
    (tmp_path / "layer_10.a").write_bytes(b"1")
    s0 = weka_snapshot(tmp_path, 1)
    assert set(s0["files"]) == {"layer_1.a", "layer_1.b"}
    assert s0["files"]["layer_1.a"][0] == 2
    assert snapshot_diff(s0, weka_snapshot(tmp_path, 1)) == []

    s1 = {"dir_mtime_ns": s0["dir_mtime_ns"] + 1, "files": dict(s0["files"])}
    s1["files"]["layer_1.c"] = [1, 1]
    del s1["files"]["layer_1.b"]
    s1["files"]["layer_1.a"] = [3, s0["files"]["layer_1.a"][1]]
    lines = snapshot_diff(s0, s1)
    assert len(lines) == 4
    assert lines[0].startswith("dir mtime")
    assert "added layer_1.c" in lines and "removed layer_1.b" in lines
    assert any(line.startswith("changed layer_1.a: size 2 -> 3") for line in lines)

    s2 = {"dir_mtime_ns": s0["dir_mtime_ns"], "files": dict(s0["files"])}
    s2["files"]["layer_1.a"] = [2, s0["files"]["layer_1.a"][1] + 5]
    assert len(snapshot_diff(s0, s2)) == 1


# ---- Task 11: time profile


def _clock(ticks):
    it = iter(ticks)
    return lambda: next(it)


def test_stage_timer_nested_exclusive():
    t = StageTimer(clock=_clock([0, 3, 7, 10]))
    indexer = t.wrap(lambda: None, "indexer")
    mla = t.wrap(lambda: indexer(), "mla")
    mla()
    assert t.exclusive() == {"mla": 6, "indexer": 4}


def test_stage_timer_dense_ffn_owns_shared_expert():
    t = StageTimer(clock=_clock([0, 1, 9, 10, 20, 25]))
    shared = t.wrap(lambda: None, "shared_expert")
    ffn = t.wrap(lambda: shared(), "dense_ffn")
    ffn()
    shared()
    assert t.exclusive() == {"dense_ffn": 10, "shared_expert": 5}


def test_stage_timer_exception_pops_stack():
    t = StageTimer(clock=_clock([0, 2, 10, 13]))

    def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        t.wrap(boom, "mla")()
    t.wrap(lambda: None, "norms")()
    assert t.exclusive() == {"mla": 2, "norms": 3}


@pytest.mark.parametrize(
    "name, stage",
    [
        ("layer_3.attn_norm_weight_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin", "norms"),
        ("layer_3.ffn_norm_weight_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin", "norms"),
        ("layer_10.mla.indexer_wk_dtype_BFLOAT8_B_layout_TILE.tensorbin", "indexer"),
        ("layer_10.mla.indexer_k_norm_dtype_BFLOAT16_layout_TILE.tensorbin", "indexer"),
        ("layer_10.mla.q_a_proj_dtype_BFLOAT8_B_layout_TILE.tensorbin", "mla"),
        ("layer_10.mla.kv_a_layernorm_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin", "mla"),
        ("layer_1.gate.weight_dtype_BFLOAT16_layout_TILE.tensorbin", "moe_gate"),
        ("layer_1.routed_expert.local_0_gate_dtype_BFLOAT4_B_layout_TILE.tensorbin", "routed_experts"),
        ("layer_1.shared_expert.down_proj_dtype_BFLOAT8_B_layout_TILE.tensorbin", "shared_expert"),
        ("layer_0.ffn.up_proj_dtype_BFLOAT8_B_layout_TILE.tensorbin", "dense_ffn"),
        ("layer_0.something_else.tensorbin", "other"),
    ],
)
def test_stage_for_file(name, stage):
    assert stage_for_file(name) == stage


def _layer_report():
    rows = [
        "startup",
        "hf_read",
        "assemble",
        "norms",
        "mla",
        "indexer",
        "moe_gate",
        "routed_experts",
        "shared_expert",
        "dense_ffn",
        "other",
        "build_total",
        "check_complete",
        "validate.fileset",
        "validate.content",
        "cleanup",
    ]
    stages = {r: {"time_s": 0.0, "files": 0, "bytes": 0} for r in rows}
    stages.update(
        startup={"time_s": 18.04, "files": 0, "bytes": 0},
        hf_read={"time_s": 8.44, "files": 0, "bytes": 0},
        assemble={"time_s": 0.01, "files": 0, "bytes": 0},
        norms={"time_s": 0.02, "files": 2, "bytes": 34624},
        mla={"time_s": 0.6, "files": 8, "bytes": 107464904},
        moe_gate={"time_s": 0.0, "files": 2, "bytes": 5532944},
        routed_experts={"time_s": 90.66, "files": 36, "bytes": 9512789472},
        shared_expert={"time_s": 0.31, "files": 3, "bytes": 46801680},
        build_total={"time_s": 91.6, "files": 0, "bytes": 0},
        check_complete={"time_s": 0.0, "files": 0, "bytes": 0},
        **{"validate.fileset": {"time_s": 0.1, "files": 0, "bytes": 0}},
        **{"validate.content": {"time_s": 11.7, "files": 0, "bytes": 0}},
        cleanup={"time_s": 0.4, "files": 0, "bytes": 0},
    )
    return {
        "model": "kimi_k2_7",
        "out_dir": "/home/u/ttnn-layer-cache/kimi_k2_7/kimi_k2_7_bh_32dev/8x4",
        "layer": 1,
        "kind": "moe",
        "has_indexer": False,
        "verdict": "PASS",
        "build_s": 100.05,
        "valid_s": 11.8,
        "stages": stages,
        "called": ["moe_gate", "mla", "norms", "routed_experts", "shared_expert"],
        "files_built": 51,
        "bytes_built": 9672623624,
        "weka_bytes": 9719425352,
        "check_complete": True,
        "fileset": {"common": 51, "missing_on_weka": [], "weka_only": ["layer_1.shared_expert.x", "layer_1.y"]},
        "content": {"EQUAL": 0, "EQUAL_DATA": 51, "DATA_DIFF": 0, "META_DIFF": 0, "COMPARE_ERROR": 0},
        "header_only_diffs": 51,
        "files": {"layer_1.mla.q_a_proj": {"status": "EQUAL_DATA", "detail": "header 1 B vs 2 B"}},
        "peak_rss_gb": 72.13,
        "deleted_files": 51,
        "error": None,
    }


GOLDEN_PROFILE = """\
=== kimi_k2_7 layer 1 (moe) -> /home/u/ttnn-layer-cache/kimi_k2_7/kimi_k2_7_bh_32dev/8x4 ===
stage               time_s   files           bytes
startup               18.0       -               -
hf_read                8.4       -               -
assemble               0.0       -               -
norms                  0.0       2          34,624
mla                    0.6       8     107,464,904
moe_gate               0.0       2       5,532,944
routed_experts        90.7      36   9,512,789,472
shared_expert          0.3       3      46,801,680
build_total           91.6      51   9,672,623,624
check_complete         0.0   ok
validate.fileset       0.1   common=51 missing_on_weka=0 weka_only=2
validate.content      11.7   EQUAL=0 EQUAL_DATA=51 DATA_DIFF=0 META_DIFF=0 COMPARE_ERROR=0 header_only_diffs=51
cleanup                0.4   deleted 51 files
peak_rss_gb           72.1
verdict           PASS
INFO weka_only: layer_1.shared_expert.x
INFO weka_only: layer_1.y"""


def test_format_profile_golden():
    assert format_profile(_layer_report()) == GOLDEN_PROFILE


def test_format_profile_lists_problems():
    r = _layer_report()
    r["verdict"] = "DIFF"
    r["fileset"]["missing_on_weka"] = ["layer_1.new"]
    r["files"]["layer_1.mla.o_proj"] = {"status": "DATA_DIFF", "detail": "first data diff at data offset 9"}
    r["stages"]["startup"]["time_s"] = None
    out = format_profile(r)
    assert "MISSING on weka: layer_1.new" in out
    assert "DATA_DIFF: layer_1.mla.o_proj: first data diff at data offset 9" in out
    assert "\nstartup                  -       -               -\n" in out


def test_format_summary():
    a = _layer_report()
    b = _layer_report()
    b["layer"] = 2
    b["verdict"] = "DIFF"
    b["content"] = {"EQUAL": 0, "EQUAL_DATA": 50, "DATA_DIFF": 1, "META_DIFF": 0, "COMPARE_ERROR": 0}
    out = format_summary([a, b], 1).splitlines()
    assert out[0].split() == [
        "layer",
        "kind",
        "build_s",
        "valid_s",
        "bytes_GB",
        "files",
        "miss",
        "weka_only",
        "hdr_only",
        "routed_s",
        "mla_s",
        "content",
        "verdict",
    ]
    assert out[1].split() == ["1", "moe", "100.0", "11.8", "9.67", "51", "0", "2", "51", "90.7", "0.6"] + [
        "51",
        "EQUAL_DATA",
        "PASS",
    ]
    assert out[2].split()[-3:] == ["1", "DATA_DIFF", "DIFF"]
    assert out[3].startswith("total  2 layers  build_s 200.1  valid_s 23.6  19.35 GB written")
    assert out[3].endswith("exit 1")


class _Base:
    @classmethod
    def build_ttnn_cache(cls, x):
        return (cls.__name__, x)


class _Sub(_Base):
    pass


class _Static:
    @staticmethod
    def build_ttnn_cache(x):
        return ("static", x)


class _Override(_Static):
    @staticmethod
    def build_ttnn_cache(x):
        return ("override", _Static.build_ttnn_cache(x))


def test_install_and_restore_timers():
    orig_static = _Static.__dict__["build_ttnn_cache"]
    orig_override = _Override.__dict__["build_ttnn_cache"]
    orig_base = _Base.__dict__["build_ttnn_cache"]
    t = StageTimer(clock=_clock(range(100)))
    saved = install_timers(t, [(_Sub, "indexer"), (_Static, "shared_expert"), (_Override, "dense_ffn")])
    assert _Sub.build_ttnn_cache(1) == ("_Sub", 1)
    assert _Base.build_ttnn_cache(2) == ("_Base", 2)
    assert _Override.build_ttnn_cache(3) == ("override", ("static", 3))
    assert _Static.build_ttnn_cache(4) == ("static", 4)
    assert t.calls == {"indexer": 1, "dense_ffn": 2, "shared_expert": 1}
    assert set(t.exclusive()) == {"indexer", "dense_ffn", "shared_expert"}
    restore_timers(saved)
    assert "build_ttnn_cache" not in _Sub.__dict__
    assert _Static.__dict__["build_ttnn_cache"] is orig_static
    assert _Override.__dict__["build_ttnn_cache"] is orig_override
    assert _Base.__dict__["build_ttnn_cache"] is orig_base
    assert _Sub.build_ttnn_cache(5) == ("_Sub", 5)

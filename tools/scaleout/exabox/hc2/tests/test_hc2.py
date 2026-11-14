# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Deviceless tests of hc2: SYNC vs core/, the glue (glue/run_hc2.sh, glue/check_env.sh) and SPDX headers.

HC2_CORE names another core (a core/ copy or a tray-test checkout); the stub run uses it when hc2/core is absent.
"""

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

HC2 = Path(__file__).resolve().parents[1]
GLUE = HC2 / "glue" / "run_hc2.sh"
CHECK = HC2 / "glue" / "check_env.sh"
TT_METAL_HOME = HC2.parents[3]
STUB_LIMIT_S = 60


def core_dir():
    return Path(os.environ.get("HC2_CORE") or HC2 / "core")


def core_files(d):
    """{path relative to d: bytes}; a symlink's bytes are its target, as git stores it; python caches skipped."""
    out = {}
    for dp, dns, fns in os.walk(d):
        links = [n for n in dns if os.path.islink(os.path.join(dp, n))]
        dns[:] = [n for n in dns if n != "__pycache__" and n not in links]
        for n in fns + links:
            if n.endswith(".pyc"):
                continue
            p = os.path.join(dp, n)
            out[os.path.relpath(p, d)] = os.readlink(p).encode() if os.path.islink(p) else Path(p).read_bytes()
    return out


def read_sync(path):
    """(header dict, {path: sha256}) of a SYNC manifest."""
    hdr, hashes = {}, {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln or ln.startswith("#"):
            continue
        h, sep, p = ln.partition("  ")
        if sep and len(h) == 64:
            hashes[p] = h
        elif ": " in ln:
            k, v = ln.split(": ", 1)
            hdr[k] = v
    return hdr, hashes


def drift(core, hashes):
    """{path: A|M|D} of core/ vs the SYNC hashes."""
    have = core_files(core)
    d = {p: "D" for p in hashes if p not in have}
    d.update(
        {p: "A" if p not in hashes else "M" for p, c in have.items() if hashes.get(p) != hashlib.sha256(c).hexdigest()}
    )
    return d


def write_sync(path, core):
    lines = ["# test SYNC", "source: tenstorrent/tray-test", "path: traytest/hc", "commit: " + "0" * 40]
    lines += [f"{hashlib.sha256(c).hexdigest()}  {p}" for p, c in sorted(core_files(core).items())]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_sync_drift_detects_each_change(tmp_path):
    core = tmp_path / "core"
    (core / "lib" / "__pycache__").mkdir(parents=True)
    (core / "run.sh").write_text("echo run\n")
    (core / "lib" / "a.py").write_text("A = 1\n")
    (core / "lib" / "__pycache__" / "a.cpython-310.pyc").write_bytes(b"x")
    os.symlink("a.py", core / "lib" / "b.py")
    write_sync(tmp_path / "SYNC", core)
    hdr, hashes = read_sync(tmp_path / "SYNC")
    assert hdr["path"] == "traytest/hc" and sorted(hashes) == ["lib/a.py", "lib/b.py", "run.sh"]
    assert drift(core, hashes) == {}
    (core / "run.sh").write_text("echo changed\n")
    (core / "new.sh").write_text("x\n")
    (core / "lib" / "a.py").unlink()
    os.remove(core / "lib" / "b.py")
    os.symlink("other.py", core / "lib" / "b.py")
    assert drift(core, hashes) == {"run.sh": "M", "new.sh": "A", "lib/a.py": "D", "lib/b.py": "M"}


def test_sync_hashes_match_core():
    sync, core = HC2 / "SYNC", HC2 / "core"
    if not sync.is_file() or not core.is_dir():
        pytest.skip("no hc2/SYNC and hc2/core in this checkout yet (tray-test traytest/tools/hc2_sync.sh export)")
    hdr, hashes = read_sync(sync)
    assert hdr.get("source") == "tenstorrent/tray-test" and hdr.get("commit"), "SYNC header lacks source or commit"
    assert hashes, "SYNC lists no files"
    assert drift(core, hashes) == {}, "core/ differs from SYNC: edit traytest/hc in tray-test and re-export"


def run_core():
    c = core_dir()
    if not (c / "run.sh").is_file() and not (c / "traytest" / "run.sh").is_file():
        pytest.skip(f"no hc2 core at {c} (set HC2_CORE to a core/ copy or a tray-test checkout)")
    return c


def test_stub_run_through_glue(tmp_path):
    c = run_core()
    # max-util is skipped on purpose, whatever this tt-metal build holds, so the core's skip WARNING is always expected
    env = dict(os.environ, HC2_CORE=str(c), HC2_OUT_DIR=str(tmp_path / "out"), TMPDIR=str(tmp_path))
    env.pop("TRAY_MAXUTIL_BIN", None)
    env["FLEET_MAXUTIL_SKIP"] = "hc2 stub test"
    t = time.monotonic()
    r = subprocess.run(
        ["bash", str(GLUE), "--stub", "--mode=smoke"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=3 * STUB_LIMIT_S,
    )
    took = time.monotonic() - t
    out = r.stdout.decode(errors="replace")
    assert r.returncode == 0, out[-3000:]
    assert "VERDICT: STUB" in out and "Final verdict: PASS" in out, out[-3000:]
    assert "WARNING: max-util skipped" in out, "FLEET_MAXUTIL_SKIP must make the core skip max-util with its WARNING"
    assert "tt-smi -r " not in out
    assert took < STUB_LIMIT_S, f"stub run took {took:.0f} s (limit {STUB_LIMIT_S} s)"
    assert any((tmp_path / "out").glob("run_*")), "run folder not under HC2_OUT_DIR"


def fake_core(tmp_path, hooks=True):
    """a core whose run.sh prints the environment the glue gave it"""
    c = tmp_path / "fake-core"
    c.mkdir()
    keys = "TT_METAL_HOME UB0_PYTHON HC_SETUP HC_HOLDFILE HC_HOLD HC_SITE_FILE H3_TT_METAL_BUILD"
    keys += " FLEET_MAXUTIL_SKIP TRAY_MAXUTIL_BIN TRAY_RUN_PARENT"
    body = "#!/usr/bin/env bash\n" + ('[ -n "${HC_SETUP:-}" ]\n' if hooks else "")
    body += f'for k in {keys}; do echo "$k=${{!k-<unset>}}"; done\necho "args=$*"\necho "pwd=$PWD"\n'
    (c / "run.sh").write_text(body)
    return c


def glue_env(tmp_path, core, **kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HC_", "HC2_", "H3_", "TRAY_", "FLEET_"))}
    env.update(HC2_CORE=str(core), HC2_OUT_DIR=str(tmp_path / "out"), TT_METAL_HOME="/elsewhere", **kw)
    return env


def run_glue(env, *args):
    r = subprocess.run(["bash", str(GLUE), *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    vals = dict(ln.split("=", 1) for ln in r.stdout.decode().splitlines() if "=" in ln)
    return r.returncode, vals, r.stderr.decode()


def test_glue_sets_only_hc_setup(tmp_path):
    core = fake_core(tmp_path)
    rc, v, err = run_glue(glue_env(tmp_path, core), "--mode=full", "x y")
    assert rc == 0, err
    assert v["TT_METAL_HOME"] == str(TT_METAL_HOME)
    assert v["UB0_PYTHON"] == f"{TT_METAL_HOME}/python_env/bin/python"
    assert v["HC_SETUP"] == str(CHECK)
    assert v["HC_HOLDFILE"] == v["HC_HOLD"] == v["HC_SITE_FILE"] == "<unset>"
    assert v["args"] == "--mode=full x y" and v["pwd"] == str(core)
    assert v["TRAY_RUN_PARENT"] == str(tmp_path / "out")
    view = Path(v["H3_TT_METAL_BUILD"])
    assert view == tmp_path / "out" / ".h3-build" and (view / "READY").is_file()
    assert os.readlink(view / "tt-metal") == str(TT_METAL_HOME)
    assert os.readlink(view / "python_env") == f"{TT_METAL_HOME}/python_env"
    mu = TT_METAL_HOME / "build" / "test" / "didt" / "unit_tests_didt_max_util"
    if not os.access(mu, os.X_OK):
        assert "not built" in v["FLEET_MAXUTIL_SKIP"] and v["TRAY_MAXUTIL_BIN"] == "<unset>"


def test_glue_keeps_caller_hooks_and_env_dir(tmp_path):
    core = fake_core(tmp_path)
    env = glue_env(tmp_path, core, HC_HOLD="/h/hold.sh", PYTHON_ENV_DIR="/opt/venv", HC_SETUP="true")
    rc, v, err = run_glue(env)
    assert rc == 0, err
    assert v["HC_HOLD"] == "/h/hold.sh" and v["HC_SETUP"] == "true" and v["UB0_PYTHON"] == "/opt/venv/bin/python"


def test_glue_refuses_core_without_hooks(tmp_path):
    core = fake_core(tmp_path, hooks=False)
    rc, v, err = run_glue(glue_env(tmp_path, core))
    assert rc == 2 and "HC_SETUP" in err and "args" not in v
    rc, v, err = run_glue(glue_env(tmp_path, core), "--stub")
    assert rc == 0 and v["args"] == "--stub", err


def test_glue_refuses_missing_core(tmp_path):
    rc, v, err = run_glue(glue_env(tmp_path, tmp_path / "nothing"))
    assert rc == 2 and "no hc2 core" in err


def fake_env(tmp_path, ttnn_file, noise=False):
    """a python_env whose python prints check_env.sh's HC2_TTNN= line; noise adds warnings on stdout and stderr"""
    home = tmp_path / "tt-metal"
    py = home / "python_env" / "bin" / "python"
    py.parent.mkdir(parents=True)
    body = f'echo "HC2_TTNN={ttnn_file}"\n'
    if noise:
        body = f'echo "UserWarning: stdout noise"\n{body}echo "/usr/lib/python3/warnings.py: stderr noise" >&2\n'
    py.write_text("#!/usr/bin/env bash\n" + (body if ttnn_file else "exit 1\n"))
    py.chmod(0o755)
    return home


def run_check(home, **kw):
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_ENV_DIR"}
    env.update(TT_METAL_HOME=str(home), **kw)
    return subprocess.run(["bash", str(CHECK)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)


def test_check_env_passes_inside_tt_metal_home(tmp_path):
    home = fake_env(tmp_path, tmp_path / "tt-metal" / "ttnn" / "ttnn" / "__init__.py")
    r = run_check(home)
    assert r.returncode == 0 and b"ttnn from" in r.stdout, r.stderr


def test_check_env_fails_outside_tt_metal_home(tmp_path):
    home = fake_env(tmp_path, "/usr/lib/python3/site-packages/ttnn/__init__.py")
    r = run_check(home)
    assert r.returncode == 3 and b"not from TT_METAL_HOME" in r.stderr


def test_check_env_ignores_import_noise(tmp_path):
    home = fake_env(tmp_path, tmp_path / "tt-metal" / "ttnn" / "ttnn" / "__init__.py", noise=True)
    r = run_check(home)
    assert r.returncode == 0 and b"ttnn from " + str(tmp_path / "tt-metal" / "ttnn").encode() in r.stdout, r.stderr


def test_check_env_fails_without_ttnn_or_python(tmp_path):
    home = fake_env(tmp_path, None)
    assert run_check(home).returncode == 3
    r = run_check(home, PYTHON_ENV_DIR=str(tmp_path / "none"))
    assert r.returncode == 3 and b"no python" in r.stderr


TT_SMI_R = re.compile(r"tt-smi\s+(-r|--reset)\b")


def test_no_tt_smi_r_in_glue():
    """never tt-smi -r (a galaxy reset is tt-smi -glx_reset); the regex is checked on planted lines first"""
    assert TT_SMI_R.search("    tt-smi -r 0,1") and TT_SMI_R.search("tt-smi  --reset")
    assert not TT_SMI_R.search("tt-smi -glx_reset") and not TT_SMI_R.search("tt-smi -ls")
    bad = []
    for p in sorted((HC2 / "glue").iterdir()):
        for n, ln in enumerate(p.read_text().splitlines(), 1):
            if not ln.lstrip().startswith("#") and TT_SMI_R.search(ln):
                bad.append(f"{p.name}:{n}: {ln}")
    assert not bad, bad


def test_every_hc2_file_has_spdx():
    """glue/, tests/ and README; core/ and SYNC come from tray-test and are checked there"""
    missing = []
    for p in sorted(HC2.rglob("*")):
        rel = p.relative_to(HC2)
        if not p.is_file() or rel.parts[0] in ("core", "SYNC") or "__pycache__" in rel.parts or p.suffix == ".pyc":
            continue
        head = p.read_text(errors="replace")[:600]
        if "SPDX-License-Identifier: Apache-2.0" not in head or "SPDX-FileCopyrightText" not in head:
            missing.append(str(rel))
    assert not missing, f"no SPDX header: {missing}"

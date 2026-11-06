#!/usr/bin/env python3
"""The mainline gate tool (cov.py), with three extra torus cases (X and Y on the P150x8, Y on the Galaxy). After each
mock run, the case's JIT objects (*.o, *.d), XIP copies and precompiled headers are deleted (the ELFs, generated headers and cmds.json stay: what the gate
reads). Run with python3 -B (no __pycache__ in the mainline tree) and TT_COV_TREE / TT_COV_DIR set."""
import glob, os, shutil, subprocess, sys

sys.path.insert(0, "/home/aagarwal/tt-metal-mainline/.scratch")
import cov  # noqa: E402

cov.CASES["bh_torus_x"] = (cov.BH, cov.test_yaml("Torus", "      torus_config: X\n"), {})
cov.CASES["bh_torus_y"] = (cov.BH, cov.test_yaml("Torus", cov.TORUS_Y), {})
cov.CASES["glx_torus_y"] = (cov.GLX, cov.test_yaml("Torus", cov.TORUS_Y), {})

_run_mock = cov.run_mock


def run_mock(name):
    _run_mock(name)
    d = os.path.join(cov.COV, "cases", name, "cache")
    assert d.startswith("/home/aagarwal/.tt-metal-mainline-cov-bfc") and d.endswith("/cache"), d
    subprocess.run(["find", d, "-type", "f", "(", "-name", "*.o", "-o", "-name", "*.d", "-o", "-name", "*.xip.elf", ")",
                    "-delete"])
    for p in glob.glob(os.path.join(d, "*", "pch")):
        shutil.rmtree(p)


cov.run_mock = run_mock
cov.main()

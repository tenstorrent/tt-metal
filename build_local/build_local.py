#!/usr/bin/env python3
"""Rebuild ONLY the two translation units of the indexer_score op change (op unity + the nanobind unity that holds indexer_score_nanobind.cpp) from THIS worktree and relink
_ttnncpp.so / _ttnn.so into build_local/lib. The shared build (/mnt/tt-data/ssinghal/tests/tt-metal/build_Release) is only READ (object files, libs, compile commands via
``ninja -t commands``); nothing is written there. Usage: python3 build_local.py"""
import os
import shlex
import subprocess
import sys

SRC = "/mnt/tt-data/ssinghal/tests/tt-metal"
BLD = f"{SRC}/build_Release"
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{WT}/build_local"
os.makedirs(f"{OUT}/obj", exist_ok=True)
os.makedirs(f"{OUT}/lib", exist_ok=True)
OPDIR = "ttnn/cpp/ttnn/operations/experimental/indexer_score"


def ninja_cmd(target):
    out = (
        subprocess.run(["ninja", "-C", BLD, "-t", "commands", target], capture_output=True, text=True, check=True)
        .stdout.strip()
        .splitlines()
    )
    return out[-1]


def compile_unity(target_obj, unity_cxx_rel, name):
    cmd = shlex.split(ninja_cmd(target_obj))
    # drop dependency-file flags (they would write into the shared build dir)
    toks, skip = [], False
    for i, t in enumerate(cmd):
        if skip:
            skip = False
            continue
        if t == "-MD":
            continue
        if t in ("-MT", "-MF"):
            skip = True
            continue
        toks.append(t)
    cmd = toks
    unity_src = open(f"{BLD}/{unity_cxx_rel}").read().replace(f"{SRC}/{OPDIR}", f"{WT}/{OPDIR}")
    unity_path = f"{OUT}/obj/{name}.cxx"
    open(unity_path, "w").write(unity_src)
    o = f"{OUT}/obj/{name}.o"
    i_o = cmd.index("-o")
    cmd[i_o + 1] = o
    i_c = cmd.index("-c")
    cmd[i_c + 1] = unity_path
    print("compiling", name, flush=True)
    subprocess.run(cmd, cwd=BLD, check=True)
    return o


def link(target, replace, out):
    cmd = shlex.split(ninja_cmd(target))
    toks = list(cmd)
    # remove the '-Xlinker --dependency-file=...' pair
    cmd = []
    i = 0
    while i < len(toks):
        if toks[i] == "-Xlinker" and i + 1 < len(toks) and toks[i + 1].startswith("--dependency-file"):
            i += 2
            continue
        cmd.append(toks[i])
        i += 1
    cmd = [replace.get(t, t) for t in cmd]
    i_o = cmd.index("-o")
    cmd[i_o + 1] = out
    # _ttnn.so links ttnn/_ttnncpp.so: use the new one
    cmd = [f"{OUT}/lib/_ttnncpp.so" if t == "ttnn/_ttnncpp.so" else t for t in cmd]
    # relative libs stay relative to BLD (read only); drop the leading ': &&' and trailing '&& :'
    cmd = [t for t in cmd if t not in (":", "&&")]
    print("linking", out, flush=True)
    subprocess.run(cmd, cwd=BLD, check=True)


op_obj_rel = f"{OPDIR}/CMakeFiles/ttnn_op_experimental_indexer_score.dir/Unity/unity_0_cxx.cxx.o"
nb_obj_rel = "ttnn/CMakeFiles/ttnn.dir/Unity/unity_28_cxx.cxx.o"
what = sys.argv[1] if len(sys.argv) > 1 else "all"
if what in ("all", "compile"):
    op_o = compile_unity(op_obj_rel, op_obj_rel[: -len(".o")], "indexer_op_unity")
    nb_o = compile_unity(nb_obj_rel, nb_obj_rel[: -len(".o")], "ttnn_nb_unity28")
else:
    op_o, nb_o = f"{OUT}/obj/indexer_op_unity.o", f"{OUT}/obj/ttnn_nb_unity28.o"
link("ttnn/_ttnncpp.so", {op_obj_rel: op_o}, f"{OUT}/lib/_ttnncpp.so")
link("ttnn/_ttnn.so", {nb_obj_rel: nb_o}, f"{OUT}/lib/_ttnn.so")
print("DONE", flush=True)

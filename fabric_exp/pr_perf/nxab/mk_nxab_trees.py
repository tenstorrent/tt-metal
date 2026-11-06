#!/usr/bin/env python3
"""mk_nxab_trees.py: the Galaxy trees of the 200G NeighborExchange A/B (port e7fe1899049 only; local, never committed).

  mk_nxab_trees.py host            /data/aagarwal/frs-glx/trees/nxab_host: a farm (symlinks) of trees/port_e7fe189 whose
                                   host libraries and test_tt_fabric are this A/B's build
                                   (/home/aagarwal/tt-metal-port-nxab: e7fe1899049 + the test harness's telemetry switch
                                   + rx_slots_host.patch, TT_FABRIC_AB_VC0_RX_SLOTS; unset = the allocator's counts)
  mk_nxab_trees.py arm NAME TX RX IDLE
                                   trees/nxab_NAME: a farm of nxab_host with the router headers patched: spin_sites.patch
                                   + slack_study_spin.hpp (spin_mcycle.hpp.in, TX / RX core cycles per packet on E0 / E1;
                                   0 compiles a spin out) when TX or RX > 0, and IDLE_PASSES_BEFORE_PARTIAL_CREDIT = IDLE
                                   when IDLE != 4 (idle_passes64.patch's line)
  mk_nxab_trees.py tune NAME       trees/NAME: a farm of trees/port_e7fe189 with the worktree's host libraries and test binary
                                   (as host) and every source file its local diff changes (git diff HEAD --name-only), as
                                   real copies: the IDLE_PASSES sweep's tree (port_tune_e7fe.patch: the receiver constants
                                   from TT_FABRIC_EXP_IDLE_PASSES / _PROOF_CHECKS, the study's test infra)
Every tree gets .tree_sha naming what it holds. A tree that exists is left alone (remove it first to rebuild)."""
import os, re, shutil, subprocess, sys

T = "/data/aagarwal/frs-glx/trees"
SRC = "/home/aagarwal/tt-metal-port-nxab"
HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIRS = ["build_Release/tt_metal", "build_Release/tt_stl", "build_Release/lib", "build_Release/tt_metal/third_party/umd/lib"]
TEST_BIN = "build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric"
EDM = "tt_metal/fabric/hw/inc/edm_fabric"


def stamp(dst, text):
    """Writes dst/.tree_sha as a real file: the farm links the base's, and writing through that link would overwrite it."""
    p = os.path.join(dst, ".tree_sha")
    if os.path.lexists(p):
        os.unlink(p)
    open(p, "w").write(text)


def farm(base, dst, overrides):
    """dst = base as symlinks, with overrides {rel: ("file", src_path) | ("link", target) | ("data", bytes)}."""
    assert not os.path.exists(dst), dst
    real_dirs = {""}
    for rel in overrides:
        parts = rel.split("/")[:-1]
        for k in range(1, len(parts) + 1):
            real_dirs.add("/".join(parts[:k]))

    def walk(rel):
        out = os.path.join(dst, rel)
        os.makedirs(out, exist_ok=True)
        src_dir = os.path.join(base, rel)
        for e in sorted(os.listdir(src_dir)):
            r = f"{rel}/{e}" if rel else e
            if r in real_dirs:
                walk(r)
            elif r not in overrides:
                os.symlink(os.path.join(src_dir, e), os.path.join(out, e))
    walk("")
    for rel, (kind, v) in overrides.items():
        p = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if kind == "file":
            shutil.copy2(v, p)
        elif kind == "link":
            os.symlink(v, p)
        else:
            open(p, "wb").write(v)


def host():
    dst = f"{T}/nxab_host"
    if os.path.exists(dst):
        print(f"{dst} exists: left alone")
        return
    ov = {}
    for d in LIB_DIRS:
        for e in os.listdir(os.path.join(SRC, d)):
            p = os.path.join(SRC, d, e)
            if ".so" not in e or os.path.isdir(p):
                continue
            ov[f"{d}/{e}"] = ("link", os.readlink(p)) if os.path.islink(p) else ("file", p)
    ov[TEST_BIN] = ("file", os.path.join(SRC, TEST_BIN))
    # build_Release/lib holds installed copies the loader can reach first (it precedes umd/lib in the job's
    # LD_LIBRARY_PATH): the A/B's own builds of them, so every host library but fmt / benchmark is this build's.
    umd = "build_Release/tt_metal/third_party/umd/lib"
    for e in os.listdir(os.path.join(SRC, umd)):
        p = os.path.join(SRC, umd, e)
        ov[f"build_Release/lib/{e}"] = ("link", os.readlink(p)) if os.path.islink(p) else ("file", p)
    ov["build_Release/lib/libtt_metal.so"] = ("file", os.path.join(SRC, "build_Release/tt_metal/libtt_metal.so"))
    ov["build_Release/lib/libtt_stl.so"] = ("file", os.path.join(SRC, "build_Release/tt_stl/libtt_stl.so"))
    head = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    diff = subprocess.run(["git", "-C", SRC, "diff", "--stat"], capture_output=True, text=True, check=True).stdout
    farm(f"{T}/port_e7fe189", dst, ov)
    stamp(dst,
        f"{head} (farm of {T}/port_e7fe189; host libraries and test_tt_fabric from {SRC}/build_Release, "
        f"{len(ov)} files; local diff:\n{diff})\n")
    print(f"{dst}: {len(ov)} host files from {SRC}")


def host_overrides():
    SRC = os.environ.get("NXAB_SRC", globals()["SRC"])  # another built worktree (the flush check's)
    ov = {}
    for d in LIB_DIRS:
        for e in os.listdir(os.path.join(SRC, d)):
            p = os.path.join(SRC, d, e)
            if ".so" not in e or os.path.isdir(p):
                continue
            ov[f"{d}/{e}"] = ("link", os.readlink(p)) if os.path.islink(p) else ("file", p)
    ov[TEST_BIN] = ("file", os.path.join(SRC, TEST_BIN))
    umd = "build_Release/tt_metal/third_party/umd/lib"
    for e in os.listdir(os.path.join(SRC, umd)):
        p = os.path.join(SRC, umd, e)
        ov[f"build_Release/lib/{e}"] = ("link", os.readlink(p)) if os.path.islink(p) else ("file", p)
    ov["build_Release/lib/libtt_metal.so"] = ("file", os.path.join(SRC, "build_Release/tt_metal/libtt_metal.so"))
    ov["build_Release/lib/libtt_stl.so"] = ("file", os.path.join(SRC, "build_Release/tt_stl/libtt_stl.so"))
    # The jobs run ./build/test/...: point build at this farm's build_Release, not at the base tree's (whose test binary
    # is the base's own build).
    ov["build"] = ("link", "build_Release")
    return ov


def tune(name):
    dst = f"{T}/{name}"
    if os.path.exists(dst):
        print(f"{dst} exists: left alone")
        return
    ov = host_overrides()
    changed = subprocess.run(["git", "-C", SRC, "diff", "HEAD", "--name-only"], capture_output=True, text=True,
                             check=True).stdout.split()
    for f in changed:
        ov[f] = ("file", os.path.join(SRC, f))
    head = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    farm(f"{T}/port_e7fe189", dst, ov)
    stamp(dst, f"{head} (farm of {T}/port_e7fe189; host libraries and test_tt_fabric from {SRC}/build_Release; "
               f"{len(changed)} source files from its local diff: {' '.join(changed)})\n")
    print(f"{dst}: {len(ov)} files ({len(changed)} sources)")


def tree_from(name, src, extra_name=None, extra_file=None):
    """trees/NAME: a farm of trees/port_e7fe189 with src's host libraries and test binary and every file src differs in
    from e7fe1899049 (committed or not; files src deletes are left as the base has them, unused); with extra_name, a
    farm of trees/NAME's base instead, one file (extra_name) replaced by extra_file."""
    dst = f"{T}/{name}"
    if os.path.exists(dst):
        print(f"{dst} exists: left alone")
        return
    if extra_name:
        base = f"{T}/{src}"
        farm(base, dst, {extra_name: ("file", extra_file)})
        stamp(dst, f"{open(os.path.join(base, '.tree_sha')).read().split()[0]} with {extra_name} from {extra_file} "
                   f"(farm of {base})\n")
        print(f"{dst}: {base} with {extra_name} replaced")
        return
    os.environ["NXAB_SRC"] = src
    ov = host_overrides()
    changed = subprocess.run(["git", "-C", src, "diff", "e7fe1899049", "--name-only", "--diff-filter=d"],
                             capture_output=True, text=True, check=True).stdout.split()
    for f in changed:
        ov[f] = ("file", os.path.join(src, f))
    head = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    farm(f"{T}/port_e7fe189", dst, ov)
    stamp(dst, f"{head} (farm of {T}/port_e7fe189; host libraries and test_tt_fabric from {src}/build_Release; "
               f"{len(changed)} files that differ from e7fe1899049)\n")
    print(f"{dst}: {len(ov)} files ({len(changed)} sources)")


def arm(name, tx, rx, idle):
    dst = f"{T}/nxab_{name}"
    if os.path.exists(dst):
        print(f"{dst} exists: left alone")
        return
    base = f"{T}/nxab_host"
    work = f"/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/nxab_arm_{name}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    files = [f"{EDM}/router/router_receiver.hpp", f"{EDM}/router/router_sender.hpp"]
    for f in files:
        os.makedirs(os.path.join(work, os.path.dirname(f)), exist_ok=True)
        shutil.copy2(os.path.join(base, f), os.path.join(work, f))
    subprocess.run(["git", "init", "-q", work], check=True)
    ov = {}
    if tx or rx:
        subprocess.run(["git", "-C", work, "apply", os.path.join(HERE, "spin_sites.patch")], check=True)
        hdr = open(os.path.join(HERE, "spin_mcycle.hpp.in")).read().replace("@TX@", str(tx)).replace("@RX@", str(rx))
        ov[f"{EDM}/slack_study_spin.hpp"] = ("data", hdr.encode())
    if idle != 4:
        p = os.path.join(work, EDM, "router/router_receiver.hpp")
        s = open(p).read()
        old = "constexpr uint32_t IDLE_PASSES_BEFORE_PARTIAL_CREDIT = 4;"
        assert s.count(old) == 1
        s = s.replace(old, f"constexpr uint32_t IDLE_PASSES_BEFORE_PARTIAL_CREDIT = {idle};  "
                           "// LOCAL NX A/B arm (never committed); stock: 4")
        open(p, "w").write(s)
    for f in files:
        ov[f] = ("file", os.path.join(work, f))
    farm(base, dst, ov)
    stamp(dst,
        f"nxab arm {name}: farm of {base}; E0 spin {tx}, E1 spin {rx} core cycles per packet, "
        f"IDLE_PASSES_BEFORE_PARTIAL_CREDIT {idle}\n")
    print(f"{dst}: tx {tx}, rx {rx}, idle {idle}")


if __name__ == "__main__":
    if sys.argv[1] == "host":
        host()
    elif sys.argv[1] == "tune":
        tune(sys.argv[2])
    elif sys.argv[1] == "tree":  # tree NAME SRC_WORKTREE  |  tree NAME BASE_TREE_NAME REL_PATH FILE
        tree_from(*sys.argv[2:])
    else:
        arm(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))

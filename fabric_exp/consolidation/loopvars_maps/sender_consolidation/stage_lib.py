"""stage_lib.py: the sender consolidation's stage scripts (stage_c1.py and later) make a variant from its parent with a
unified diff (cN.patch, next to them; made with `diff -ruN PARENT NAME`), so the script is the reviewable record of the
stage. make(parent, name, patch): copies variants/PARENT to variants/NAME (refusing an existing NAME) and applies the
patch with no fuzz. With --check, builds into a scratch directory instead and compares it with the existing
variants/NAME file by file (exit 1 on any difference)."""
import filecmp, os, shutil, subprocess, sys, tempfile

V = "/home/aagarwal/tt-metal/fabric_exp/variants"
HERE = os.path.dirname(os.path.abspath(__file__))


def files(d):
    return sorted(os.path.relpath(os.path.join(p, f), d) for p, _, fs in os.walk(d) for f in fs)


def make(parent, name, patch):
    check = "--check" in sys.argv
    dst = os.path.join(tempfile.mkdtemp(), name) if check else f"{V}/{name}"
    if os.path.exists(dst):
        sys.exit(f"{dst} exists")
    shutil.copytree(f"{V}/{parent}", dst, symlinks=True)
    r = subprocess.run(["patch", "-p1", "-d", dst, "--fuzz=0", "--no-backup-if-mismatch", "-s", "-i",
                        os.path.join(HERE, patch)], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"patch failed:\n{r.stdout}{r.stderr}")
    if check:
        same = files(dst) == files(f"{V}/{name}") and all(
            filecmp.cmp(f"{dst}/{f}", f"{V}/{name}/{f}", shallow=False) for f in files(dst))
        print(f"{name}: {'reproduced exactly' if same else 'DIFFERS from variants/' + name}")
        shutil.rmtree(os.path.dirname(dst))
        sys.exit(0 if same else 1)
    print("made", dst)

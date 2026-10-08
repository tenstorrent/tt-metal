import json, re, subprocess, shlex, sys, os, concurrent.futures as cf

ROOT = "/home/smarton/fasth3/tt-metal"
W = ROOT + "/tt-project/worktrees/t297-mainpr"
cc = json.load(open("/tmp/t298_compdb.json"))
files = subprocess.check_output(["git", "-C", W, "show", "--name-only", "--format=", "70157c213e6"], text=True).split()
files = [f for f in files if f.endswith(".cpp") and f.startswith("ttnn/") and "/kernels/" not in f]
units = [e for e in cc if "/Unity/unity_" in e.get("file", "") and " -o " in e["command"]]


def find_cmd(rel):
    for e in units:
        try:
            if ROOT + "/" + rel in open(e["file"]).read():
                return e
        except OSError:
            pass


def rewrite(m):
    path = W + "/" + m.group(1)
    return path if os.path.exists(path) and ".cpmcache" not in path else m.group(0)


def job(rel):
    e = find_cmd(rel)
    if e is None:
        return rel, None, "no unity TU found"
    args = shlex.split(e["command"])
    out, skip = [], False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a in ("-o", "-c", "-MF", "-MT", "-include-pch", "-include"):
            skip = True
            continue
        if a in ("-MD", "-Winvalid-pch", "-fpch-instantiate-templates", "-fpch-validate-input-files-content"):
            continue
        if a == "-Xclang":
            nxt = args[i + 1] if i + 1 < len(args) else ""
            if "pch" in nxt or nxt == "-include" or nxt.endswith((".pch", ".hxx")):
                skip = True
                continue
            out.append(a)
            continue
        if a.endswith((".pch", ".hxx")) or "cmake_pch" in a:
            continue
        out.append(re.sub(re.escape(ROOT) + r"/(?!build_Release)([^\s:]*)", rewrite, a))
    out += ["-fsyntax-only", W + "/" + rel]
    p = subprocess.run(out, cwd=e["directory"], capture_output=True, text=True)
    return rel, p.returncode, p.stderr[-4000:]


rc = 0
with cf.ThreadPoolExecutor(8) as ex:
    for rel, code, err in ex.map(job, files):
        print(f"[{'OK' if code == 0 else 'FAIL'}] rc={code} {rel}")
        if code != 0:
            rc = 1
            print(err)
print("DONE rc", rc)
sys.exit(rc)

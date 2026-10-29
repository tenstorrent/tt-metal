#!/usr/bin/env python3
"""toolchain_ab.py OUT LOG...: rebuild the router kernels a run built (its logged compile and link commands,
TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1) twice, with the toolchain they were logged with and with OLD_SFPI (default
~/.sfpi-dl/7.86.0/sfpi), and compare the ELFs' loadable bytes. The run's kernel cache (generated headers, firmware
ELFs to link against) and the sources it compiled (the tree, with the same variant swapped in) must still be in
place. Kernels: both router slots (fabric_erisc_router, fabric_erisc_router_v2). Prints per log the configs compared,
identical and differing (and how many replays reproduce the run's own ELF, a check of the replay), and for each
differing config the loadable sizes."""
import os, re, subprocess, sys

LOGGED_SFPI = re.compile(r"[^\s'\"]*/runtime/sfpi(?=/)")
OLD_SFPI = os.environ.get("OLD_SFPI", os.path.expanduser("~/.sfpi-dl/7.86.0/sfpi"))
OBJCOPY = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy"
KDIR = re.compile(r"kernels/(fabric_erisc_router(?:_v2)?)/(\d+)/(\w+)/")


def commands(log):
    """{(kernel, hash, proc): {"compile": cmd, "link": cmd}} from a run log (first well-formed command of each)."""
    out = {}
    for line in open(log, errors="replace"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"g\+\+ (compile|link) cmd: (.*?)\s*\(build\.cpp:\d+\)\s*$", line)
        if not m or "<stdout>" in m.group(2) or "<stderr>" in m.group(2):
            continue
        k = KDIR.search(m.group(2))
        if not k or m.group(2).count("riscv-tt-elf-g++") != 1:
            continue
        # Two MPI ranks write to one VC1 log, so a line can be cut by the other's output: keep whole commands only.
        if m.group(1) == "compile" and not (m.group(2).count(" -c ") == 1 and m.group(2).endswith(".d")):
            continue
        if m.group(1) == "link" and not re.search(r"\S+\.o\s+-o\s+\S+\.elf", m.group(2)):
            continue
        out.setdefault((k.group(1), k.group(2), k.group(3)), {}).setdefault(m.group(1), m.group(2))
    return {k: v for k, v in out.items() if "compile" in v and "link" in v}


def build(cmds, wd, sfpi=None):
    """Compile and link into wd; returns the ELF path or an error string."""
    os.makedirs(wd, exist_ok=True)
    c, l = cmds["compile"], cmds["link"]
    # The JIT compiles in the kernel's cache directory (the -MF file's grandparent): the generated headers are there.
    kdir = os.path.dirname(os.path.dirname(re.search(r"\s-MF\s+(\S+)", c).group(1)))
    if sfpi:
        c, l = LOGGED_SFPI.sub(sfpi, c), LOGGED_SFPI.sub(sfpi, l)
    obj, elf = os.path.join(wd, "k.o"), os.path.join(wd, "k.elf")
    c = re.sub(r"\s-o\s+\S+", f" -o {obj}", c, count=1)
    c = re.sub(r"\s-MF\s+\S+", f" -MF {os.path.join(wd, 'k.d')}", c, count=1)
    lo = re.search(r"(\S+\.o)\s+-o\s+(\S+\.elf)", l)
    l = l.replace(lo.group(1), obj).replace(lo.group(2), elf)
    for cmd in (c, l):
        cwd = kdir
        m = re.match(r"cd (\S+) && (.*)", cmd)
        if m:
            cwd, cmd = m.group(1), m.group(2)
        r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if r.returncode:
            return "failed: " + r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "failed"
    return elf


def loadable(elf):
    return subprocess.run([OBJCOPY, "-O", "binary", elf, "/dev/stdout"], capture_output=True).stdout


def main():
    out, logs = sys.argv[1], sys.argv[2:]
    for log in logs:
        cmds = commands(log)
        same = diff = failed = 0
        faithful = checked = 0
        notes = []
        for (kernel, h, proc), c in sorted(cmds.items()):
            base = os.path.join(out, os.path.basename(log), kernel, h, proc)
            a, b = build(c, base + "/logged"), build(c, base + "/old", OLD_SFPI)
            if a.startswith("failed") or b.startswith("failed"):
                failed += 1
                notes.append(f"  {kernel} {h} {proc}: logged {a if a.startswith('failed') else 'ok'}, old {b if b.startswith('failed') else 'ok'}")
                continue
            la, lb = loadable(a), loadable(b)
            # the ELF the run itself built (the JIT links to a temporary name, then renames it <proc>.elf)
            jit = os.path.join(os.path.dirname(re.search(r"\S+\.o\s+-o\s+(\S+\.elf)", c["link"]).group(1)), proc + ".elf")
            if os.path.exists(jit):
                checked += 1
                faithful += loadable(jit) == la
            if la == lb:
                same += 1
            else:
                diff += 1
                notes.append(f"  {kernel} {h} {proc}: differs, loadable {len(la)} (logged) vs {len(lb)} (old) bytes")
        print(f"{os.path.basename(log)}: {len(cmds)} configs, identical {same}, differ {diff}, failed {failed}; "
              f"replay matches the run's own ELF in {faithful} of {checked}")
        for n in notes[:12]:
            print(n)


if __name__ == "__main__":
    main()

# Hash the loaded image (PROGBITS sections with the ALLOC flag) of every kernel ELF of a JIT cache, keyed by kernel name,
# a hash of the build's generated files (defines, descriptors, compile-time arguments, bindings, the per-thread wrappers
# with the farm directory name normalised) and the ELF path below the build directory. usage: elf_hash_cfg.py <cache>
import hashlib, os, re, subprocess, sys
READELF = os.environ.get("READELF", "/proj_sw/user_dev/mvlahovic/llk_analysis/worktrees/main-b3e9316/runtime/sfpi/compiler/bin/riscv-tt-elf-readelf")
root = sys.argv[1]
cfg = {}
for dp, dn, fn in os.walk(root):
    for f in fn:
        if not f.endswith(".elf") or f.endswith(".xip.elf"):
            continue
        p = os.path.join(dp, f)
        parts = p.split(os.sep)
        if "kernels" not in parts:
            continue
        i = parts.index("kernels")
        kdir = os.sep.join(parts[: i + 3])
        if kdir not in cfg:
            h = hashlib.sha256()
            for g in sorted(os.listdir(kdir)):
                gp = os.path.join(kdir, g)
                if os.path.isfile(gp) and (g.endswith(".h") or g.endswith(".cpp")):
                    h.update(g.encode()); h.update(re.sub(rb"/farm_[A-Za-z0-9_]+/", b"/farm/", open(gp, "rb").read()))
            cfg[kdir] = h.hexdigest()[:16]
        key = "/".join([parts[i + 1], cfg[kdir]] + parts[i + 3 :])
        secs = []
        for line in subprocess.run([READELF, "-S", "-W", p], capture_output=True, text=True).stdout.splitlines():
            m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)", line)
            if m and m.group(2) == "PROGBITS" and "A" in m.group(3):
                secs.append(m.group(1))
        hh = hashlib.sha256()
        for s in secs:
            hh.update(subprocess.run([READELF, "-x", s, p], capture_output=True, text=True).stdout.encode())
        print(f"{key} {hh.hexdigest()[:24]} {kdir}")

#!/usr/bin/env python3
# Scratch CI proof runner of the unary datacopy family (not for merge). On a scratch branch of main or of the PR head merged
# with main, it prints, for comparison between the two runs:
#   BITID <case> <sha256 of the output bytes> <elements>     device outputs of scratch_r3/bitid.py cases (Blackhole card)
#   ELF <arch> <kernel> <key> <t0> <t1> <t2>                  disassembly hashes of the JIT compute kernels built for the
#                                                             scratch_r3/prof_ops12.py cases on a mock cluster (arch wh or bh)
# usage: run_proofs.py bitid <case>... | elf <wh|bh> <case>...
import glob, hashlib, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def objdump():
    for p in (os.path.join(ROOT, "runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"), shutil.which("riscv-tt-elf-objdump") or ""):
        if p and os.path.exists(p):
            return p
    sys.exit("no riscv-tt-elf-objdump")


def code_hash(elf, od):
    txt = subprocess.run([od, "-d", "--no-show-raw-insn", elf], capture_output=True, text=True).stdout
    lines = [m.group(1) + " " + re.sub(r"\s*<[^>]*>", "", m.group(2).split("#")[0]).strip()
             for m in re.finditer(r"^\s*([0-9a-f]+):\s+(.*)$", txt, re.M)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16] + "/%d" % len(lines)


def key(d):
    h = hashlib.sha256()
    for f in ("kernel_args_generated.h", "defines_generated.h", "chlkc_descriptors.h", "named_ct_arg_map_generated.h"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            h.update("\n".join(l for l in open(p, errors="replace").read().splitlines() if "FULL_KERNEL_NAME" not in l).encode())
    return h.hexdigest()[:16]


def run_elf(arch, cases):
    desc = {"wh": os.path.join(ROOT, "tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"),
            "bh": os.path.join(HERE, "p100a_cluster.yaml")}[arch]
    cache = tempfile.mkdtemp(prefix=f"r3elf_{arch}_")
    env = dict(os.environ, TT_METAL_CACHE=cache, TT_METAL_MOCK_CLUSTER_DESC_PATH=desc, R3_NMEAS="1", R3_L1_SMALL="16384")
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    r = subprocess.run([sys.executable, os.path.join(HERE, "prof_ops12.py")] + cases, env=env, capture_output=True, text=True)
    print(f"ELFRUN {arch} rc={r.returncode} ok={r.stdout.count(' ok')} failed={r.stdout.count('FAILED')}")
    for line in r.stdout.splitlines():
        if "FAILED" in line:
            print("ELFFAIL", arch, line[:300])
    od = objdump()
    for d in sorted(glob.glob(os.path.join(cache, "**/kernels/*/*"), recursive=True)):
        if not os.path.isdir(os.path.join(d, "trisc1")):
            continue
        hs = [code_hash(os.path.join(d, t, t + ".elf"), od) if os.path.exists(os.path.join(d, t, t + ".elf")) else "-"
              for t in ("trisc0", "trisc1", "trisc2")]
        print("ELF", arch, os.path.basename(os.path.dirname(d)), key(d), *hs)


def run_bitid(cases):
    out = tempfile.mkdtemp(prefix="r3bitid_")
    env = dict(os.environ, HWLOCK_HELD="ci")
    r = subprocess.run([sys.executable, os.path.join(HERE, "bitid.py"), out] + cases, env=env, capture_output=True, text=True)
    print(f"BITIDRUN rc={r.returncode}")
    for line in r.stdout.splitlines():
        if line.startswith("BITID"):
            print("BITIDLOG", line[:300])
    import torch
    for f in sorted(glob.glob(os.path.join(out, "*.pt"))):
        t = torch.load(f)
        b = t.contiguous().numpy().tobytes()
        print("BITID", os.path.basename(f)[:-3], hashlib.sha256(b).hexdigest(), t.numel())


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "bitid":
        run_bitid(sys.argv[2:])
    elif mode == "elf":
        run_elf(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)

#!/usr/bin/env python3
"""Offline JIT compile check of the packed sparse_sdpa_msa kernels (no device): replays the JIT's own riscv
compile commands for the legacy kernels (captured in a run log) on the packed kernel sources with packed CT args.
usage: kcheck.py [G] [log]"""
import os
import re
import subprocess
import sys

G = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LOG = sys.argv[2] if len(sys.argv) > 2 else "/mnt/data/kernel-agent/dev/prefill-sdpa/runs/base0/log.txt"
W = os.environ.get("KROOT", "/mnt/data/kernel-agent/dev/prefill-sdpa/tt-metal")
KD = W + "/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels"
OUT = f"/mnt/data/kernel-agent/dev/prefill-sdpa/kcheck/{os.path.basename(W)}-G{G}"
lines = [l for l in open(LOG, errors="replace") if "riscv-tt-elf-g++" in l and "sparse_sdpa_msa_" in l]
Sqt = G // 2
cb_state_end = 18 + 6 * Sqt
fmt = [5] * 64
size = [2048] * 64
fmt[2] = fmt[3] = 6
size[2] = size[3] = 1088
for i in range(cb_state_end, 64):
    fmt[i] = 255
compute_ct = [G, 4, 4, 4, 1035273459, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 15, 16, 17, 18]
reader_head = [16, 1280, 16, 1, 256, 64, 16, 16, 8, 8, 0, 2, 3, 9, 10, 13, 14, 17, 1088, 1088, 1, 128, G, 0, 1, 1, 0, 0]
writer_head = [16, 1280, 1, 256, 4, 16, 16, 8, 8, 8, 4, 11, 13, 14, 1088, 1088, 15, 16]
ok = True
for l in lines:
    l = l.strip()
    m = re.search(r"-o (\S+)", l)
    o = m.group(1)
    kind = "reader" if "msa_reader" in o else "writer" if "msa_writer" in o else "compute"
    sub = os.path.basename(os.path.dirname(o))  # ncrisc / brisc / trisc0..2
    kdir = os.path.dirname(os.path.dirname(o))
    dst = f"{OUT}/{kind}"
    os.makedirs(f"{dst}/{sub}", exist_ok=True)
    for f in os.listdir(kdir):
        p = os.path.join(kdir, f)
        if os.path.isfile(p):
            t = open(p).read()
            t = t.replace(
                "/mnt/data/kernel-agent/dev/prefill-sdpa/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/",
                W + "/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/",
            )
            t = t.replace(
                f"kernels/compute/sparse_sdpa_msa_compute.cpp", "kernels/compute/sparse_sdpa_msa_packed_compute.cpp"
            )
            t = t.replace(
                f"kernels/dataflow/sparse_sdpa_msa_reader.cpp", "kernels/dataflow/sparse_sdpa_msa_packed_reader.cpp"
            )
            t = t.replace(
                f"kernels/dataflow/sparse_sdpa_msa_writer.cpp", "kernels/dataflow/sparse_sdpa_msa_packed_writer.cpp"
            )
            if f == "chlkc_descriptors.h":

                def rep(name, vals):
                    global t
                    t = re.sub(
                        r"(constexpr uint\d+_t " + name + r"\[64\] = \{\n)([^\n]*)",
                        lambda mm: mm.group(1) + "    " + ",".join(map(str, vals)),
                        t,
                    )

                for n in ("unpack_src_format", "unpack_dst_format", "pack_src_format", "pack_dst_format"):
                    rep(n, fmt)
                for n in ("unpack_tile_size", "pack_tile_size"):
                    rep(n, size)
            open(f"{dst}/{f}", "w").write(t)
    cta = re.search(r"-DKERNEL_COMPILE_TIME_ARGS=(\S+)", l).group(1).split(",")
    if kind == "compute":
        new = compute_ct
    elif kind == "reader":
        new = reader_head + [int(x) for x in cta[28:]]
    else:
        new = writer_head + [int(x) for x in cta[25:]]
    cmd = l.replace(
        f"-DKERNEL_COMPILE_TIME_ARGS={','.join(cta)}", "-DKERNEL_COMPILE_TIME_ARGS=" + ",".join(map(str, new))
    )
    no = f"{dst}/{sub}/k.o"
    cmd = cmd.replace(o, no)
    cmd = re.sub(r"-MF \S+", f"-MF {dst}/{sub}/k.d", cmd)
    r = subprocess.run(cmd, shell=True, cwd=f"{dst}/{sub}", capture_output=True, text=True)
    errs = [x for x in r.stderr.splitlines() if "error" in x.lower() and "pragma message" not in x]
    print(f"[kcheck G={G}] {kind}/{sub}: rc={r.returncode} {'OK' if r.returncode == 0 else ''}")
    if r.returncode:
        ok = False
        print("\n".join(errs[:40]))
        open(f"{dst}/{sub}/err.txt", "w").write(r.stderr)
sys.exit(0 if ok else 1)

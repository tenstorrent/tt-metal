import sys, re, subprocess, collections
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
A2L = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-addr2line"
e = sys.argv[1]
KEYS = ["execute_chip_unicast_to_local_chip_impl", "drain_pending_atomic_flush", "complete_pending_atomic_flush",
        "e1_rx_completions", "e1_local_delivery_shared", "e1_forward_slots_shared", "e1_admit_slots_shared",
        "e1_forward_2d", "e1_forward_to_rt", "e1_admit_2d", "e1_local_delivery", "e1_retire",
        "e1_init_downstreams", "e1_open_downstreams", "e1_setup_downstream_cmd_bufs", "e1_rx_step", "run_e1_receive_program",
        "teardown", "e1_kernel_main"]
out = subprocess.run([OD, "-d", "--section=.text", e], capture_output=True, text=True).stdout
addrs = [l.split(":")[0].strip() for l in out.splitlines() if re.match(r"\s+[0-9a-f]+:\t", l)]
a2l = subprocess.run([A2L, "-a", "-f", "-i", "-C", "-e", e] + ["0x" + a for a in addrs], capture_output=True, text=True).stdout.split("\n")
cnt = collections.Counter(); cur = None; names = []
def flush():
    if cur is None: return
    tag = "other"
    for n in names:  # innermost first
        hit = [k for k in KEYS if n.startswith(k) or ("::" + k) in n or n.split("<")[0].endswith(k)]
        if hit: tag = hit[0]; break
    if tag in ("e1_rx_step",):
        tag += "<%s>" % (re.search(r"e1_rx_step<(\d+)", " ".join(names)) or [None, "?"])[1]
    if tag == "e1_rx_completions":
        tag += "<%s>" % (re.search(r"e1_rx_completions<(\d+)", " ".join(names)) or [None, "?"])[1]
    if tag == "execute_chip_unicast_to_local_chip_impl":
        m = re.search(r"e1_rx_step<(\d+)", " ".join(names)); tag += "@" + (("step" + m.group(1)) if m else "shared/other")
    cnt[tag] += 4
i = 0
while i < len(a2l):
    l = a2l[i]
    if re.match(r"^0x[0-9a-f]+$", l):
        flush(); cur = l; names = []; i += 1; continue
    if cur is not None and l and i + 1 < len(a2l):
        names.append(l); i += 2; continue
    i += 1
flush()
for k, v in cnt.most_common(): print(f"{v:6d} {k}")
print(sum(cnt.values()))

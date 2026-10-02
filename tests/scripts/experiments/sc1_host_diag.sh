#!/usr/bin/env bash
# Host-side diagnostics for the bh_sc1 Galaxy runners (experiment branch only), round 2:
# thread environment, affinity, cgroup, uptime, hugepage consumers, and torch timings vs thread count.
set -x
hostname; uptime; who -b 2>/dev/null; cat /proc/sys/kernel/random/boot_id
env | grep -i -E 'OMP|MKL|OPENBLAS|KMP|TORCH|GOMP|OMPI|PMIX|MPI' | sort
grep -E 'Cpus_allowed_list|Threads' /proc/self/status
cat /proc/self/cgroup
ls /sys/fs/cgroup/ | head -30
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu.weight /sys/fs/cgroup/cpu.stat 2>&1 | head -8
nproc --all; nproc
grep -E 'HugePages_(Total|Free|Rsvd)' /proc/meminfo
cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages 2>/dev/null
ls -la /dev/hugepages 2>/dev/null | head -5; mount | grep -i huge
for p in /proc/[0-9]*; do h=$(grep -s '^HugetlbPages:' $p/status | awk '{print $2}'); [ -n "$h" ] && [ "$h" != "0" ] && echo "hugetlb_kB=$h pid=${p#/proc/} cmd=$(tr '\0' ' ' < $p/cmdline 2>/dev/null | cut -c1-80)"; done 2>/dev/null | sort -t= -k2 -rn | head -10
set +x
python3 - <<'EOF'
import os, time, torch
print("OMP_NUM_THREADS env:", os.environ.get("OMP_NUM_THREADS"), "| sched_getaffinity:", len(os.sched_getaffinity(0)))
print("torch", torch.__version__, "default threads", torch.get_num_threads(), "interop", torch.get_num_interop_threads())
x = torch.randn(81, 480, 832, 3)
for n in (1, 8, 32, 64):
    torch.set_num_threads(n)
    t = time.time(); y = (x.clamp(-1, 1) * 127.5 + 127.5).to(torch.uint8); a = time.time() - t
    t = time.time(); z = torch.cat([x[:, :, :416], x[:, :, 416:]], dim=2); b = time.time() - t
    t = time.time(); p = x.permute(0, 3, 1, 2).contiguous(); c = time.time() - t
    print(f"threads={n}: uint8 convert {a:.3f}s  concat {b:.3f}s  permute+contig {c:.3f}s")
EOF

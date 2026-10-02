#!/usr/bin/env bash
# Host-side diagnostics for the bh_sc1 Galaxy runners (experiment branch only).
# The Wan2.2 VAE section doubled on every sc1 host on 2026-09-15 with no tt-metal change involved;
# this prints the host facts the normal job logs do not show, plus a few host-only timings.
set -x
hostname
uname -a
nproc
cat /sys/fs/cgroup/cpu.max 2>/dev/null
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null
cat /sys/fs/cgroup/memory.max 2>/dev/null
cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null
lscpu | grep -E "Model name|^CPU\(s\)|MHz|NUMA node|Thread|Socket"
grep -i -E "huge|MemTotal|MemAvailable" /proc/meminfo
cat /sys/kernel/mm/transparent_hugepage/enabled
cat /proc/sys/kernel/numa_balancing
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
cat /proc/cmdline
dmesg 2>/dev/null | grep -i -E "iommu|dmar|tenstorrent" | tail -5
which tt-smi && tt-smi -s 2>/dev/null | head -80
set +x
python3 - <<'EOF'
import time, torch, numpy as np
print("torch", torch.__version__, "threads", torch.get_num_threads(), "interop", torch.get_num_interop_threads())
x = torch.randn(81, 480, 832, 3)
t = time.time(); y = (x.clamp(-1, 1) * 127.5 + 127.5).to(torch.uint8); print("uint8 convert 388MB s", round(time.time() - t, 3))
t = time.time(); z = x.clone(); print("clone 388MB s", round(time.time() - t, 3))
t = time.time(); w = torch.cat([x[:, :, :416], x[:, :, 416:]], dim=2); print("concat s", round(time.time() - t, 3))
t = time.time(); p = x.permute(0, 3, 1, 2).contiguous(); print("permute+contig s", round(time.time() - t, 3))
a = np.random.rand(50_000_000).astype(np.float32); t = time.time(); b = a.copy(); print("numpy copy 200MB s", round(time.time() - t, 3))
EOF

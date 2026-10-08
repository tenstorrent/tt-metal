# Kimi-K3 4xGLX traced prefill sweep

Reproduces the #53508 numbers: 100k cache hit + 10k cold prefill, DRAM slot capacity, and the
1k / 10k / 100k / 1M ISL sweep at 24 users (the 1M-window maximum).

Perf only. `K3_PERF_SHARED_KDA_CARRY=1` (set in `runner.yaml.in`) lets the traced runner serve more
than one user by replaying every slot on slot 0's KDA carry: same ops and bytes, wrong KDA numerics.
Main refuses traced multi-user K3 without it.

## Setup

Same commit on all four hosts (shared filesystem), then once:

```bash
./build_metal.sh --build-type Release && ./create_venv.sh
```

Weights, TTNN cache and the token source must be readable on every host (Weka paths are the
defaults in `runner.yaml.in`, the adapter and `producer.sh`). Passwordless ssh between the hosts.

## Run

From the first host in `HOSTS` (rank 0; the producer attaches to its H2D service):

```bash
S=models/demos/deepseek_v3_d_p/scripts/kimi_k3_sweep
export HOSTS="bh-glx-120-b06u08 bh-glx-120-b06u02 bh-glx-120-b07u02 bh-glx-120-b07u08"

# 1k/10k/100k, then 10k on top of the 100k each slot now holds (~25 min incl. ~7 min bringup)
bash $S/drive.sh /path/u24 24 warm:2:5120 i1k:24:1024 i10k:24:10240 i100k:24:102400 \
    hit:24:10240:102400 hit1:1:10240:102400

# 1M (~85 min)
bash $S/drive.sh /path/u24_1m 24 warm:2:5120 i1m:24:1048576

# capacity ceiling: both OOM in compile
bash $S/drive.sh /path/u25 25 warm:2:5120

python3 $S/capacity.py
```

A phase is `tag:users:isl[:prefix]`. `prefix` resumes every request after that many tokens already
in the slot (`PREFILL_PRODUCER_PREFIX_LEN`), so the `hit` phase times only the 10,240 new tokens.
`drive.sh` resets the four galaxies, launches the runner, runs the phases strictly in sequence,
stops the runner and prints `report.py`. Per-rank DRAM lines (`DRAM after ...`) are in `runner.log`.

`report.py` columns: `tok/s` is end-to-end (pipeline fill included); `steady tok/s/user` uses the
last rank's completion interval and is meaningless for phases with fewer chunks than stages.

## Knobs

| env | default | |
|---|---|---|
| `HOSTS` | the quad above | must follow the physical galaxy ring; rotations and reversal are fine |
| `ACK3` | `1` | `0` puts rank 3 on host-callback layer acks instead of D2H |
| `TCP_IFACE` | `ens5f0np0` | |
| `SCRATCH` | `/var/tmp/$USER-k3` | host-local JIT cache and TMPDIR; never NFS |

The published runs used `ACK3=0`: on 2026-10-08 b07u08 chip 15 (`0000:48:00.0`) was in an `identity`
IOMMU domain and the read-only D2H pin failed with `EINVAL`. Check before running:

```bash
for h in $HOSTS; do ssh $h 'cat /sys/bus/pci/drivers/tenstorrent/0000:*/iommu_group/type | sort | uniq -c'; done
```

Anything other than 32 x `DMA-FQ` on the host serving rank 3 needs `ACK3=0`; on any other rank it
needs a reboot.

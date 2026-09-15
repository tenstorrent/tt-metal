# Handoff: perf-test PR #56146 (tensorbin payload 64B alignment) on an IOMMU host

You are a fresh Claude Code session. Your job is to run a prepared perf experiment and
report numbers. This directory (`perf_handoff_56146/`) contains everything you need:
`wan_load_bench.py`, `run_all.sh`, and this brief. **Read this whole file first**, then
run the preflight before anything expensive.

## What the PR does (the mechanism you're measuring)

`ttnn.dump_tensor` writes a `.tensorbin` as `[u64 header_size][flatbuffer header][shard
payloads]`. `ttnn.load_tensor` mmaps the file at a page boundary and hands device pointers
straight into the mapping, so each shard payload lands at file offset `8 + header_size`.

The **pinned host→device DMA path** (used for weight uploads > 32 MB) DMAs directly from
that mapped pointer and requires it to be NoC-L1 aligned: **64 B on Blackhole, 16 B on
Wormhole** (`tt_metal/impl/buffers/dispatch.cpp:1198`). The old writer only 8-aligned the
header (payload at ~2984 = 8 mod 16), so **every shard was rejected** and fell back to a
slower staged hugepage copy, logging `Pinned source memory start address ... must be
aligned 64 B` per shard (tens of thousands per model load).

The fix (`ttnn/core/tensor/serialization.cpp`) pads `header_size` so `8 + header_size` is a
multiple of 64. Reader is unchanged.

## How the experiment isolates the fix

- **Aligned** files are produced by the fixed `dump_tensor` (payload offset % 64 == 0).
- **Unaligned** twins are synthesized from the aligned files by inserting 8 zero bytes
  before the payload and bumping `header_size` by 8. This reproduces the OLD writer's
  behavior exactly at the load path: identical payload bytes, but offset is now 8 mod 16
  → rejected on Wormhole (and 8 mod 64 → rejected on Blackhole). The 8-byte bump keeps the
  reader's required 8 B payload alignment, so the files still load correctly. This is a
  tighter A/B than two separate builds (same data, only the offset differs).

Weights are **real Wan2.2 T2V transformer** big linear layers (attn q/k/v/out + FFN),
each > 32 MB in bf16, loaded to the mesh (replicated → one > 32 MB upload per device), so
every upload is subject to the alignment check.

## HARD PREREQUISITE: IOMMU must be ON

The pinned path only exists when IOMMU is enabled
(`tt_metal/distributed/pinned_memory.cpp:GetMemoryPinningParameters` returns `max_pins=0`
otherwise). **With IOMMU off, aligned and unaligned are identical and there is nothing to
measure.** Verify first:

```bash
ls /sys/kernel/iommu_groups | wc -l      # must be > 0
grep -o 'intel_iommu=[^ ]*\|iommu=[^ ]*' /proc/cmdline   # expect intel_iommu=on iommu=pt
```

If these are empty, STOP and tell the user this host also has IOMMU off — do not proceed.
(The original box, a docker container on an IOMMU-off host, is exactly why this was handed
off.) The `preflight` step below also fails-closed if the pinned path isn't active.

## Other prerequisites

1. **tt-metal built WITH the fix.** You need this repo checked out on branch
   `jameslee/tensorbin-payload-alignment` (commit with `serialization.cpp` padding) and
   built. If not built:
   ```bash
   export TT_METAL_HOME=$(git rev-parse --show-toplevel)
   ./build_metal.sh -c          # -c = ccache
   ```
   Verify the fix is active (payload must be 64B-aligned):
   ```bash
   export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
   python - <<'PY'
   import torch, tempfile, pathlib, ttnn
   t = ttnn.Tensor(torch.rand((2,3,64,96), dtype=torch.bfloat16), ttnn.bfloat16)
   p = pathlib.Path(tempfile.mkdtemp())/"a.tensorbin"; ttnn.dump_tensor(str(p), t)
   hs = int.from_bytes(open(p,'rb').read(8),'little')
   print("payload_offset mod 64 =", (8+hs)%64, "(0 == fix active)")
   assert (8+hs)%64==0
   PY
   ```
2. **Wan2.2 T2V weights** in the HF cache. The harness auto-discovers
   `models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/.../transformer/` under `~/.cache/huggingface/hub`,
   `$HF_HOME/hub`, or `/localdev/*/.cache/huggingface/hub`. If they live elsewhere, set
   `WAN_TRANSFORMER_DIR=/path/to/.../transformer`. To download (~65 GB):
   `huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers`.
3. Adjust `DEVICES` to the mesh width available (default 4). `ARCH_NAME` defaults to
   `wormhole_b0`; set to `blackhole` on a BH box (the alignment threshold there is 64 B —
   the experiment still works, the unaligned offset 8 is rejected either way).

## Run it

```bash
cd perf_handoff_56146
# quick correctness/activation check (1 block, aligned vs unaligned):
TT_METAL_HOME=/path/to/tt-metal ./run_all.sh          # does preflight THEN full 40-block run
```
`run_all.sh` env knobs: `TT_METAL_HOME`, `WORK` (scratch, default `/localdev/$USER/wan22_align_exp`),
`BLOCKS` (default 40), `DEVICES` (default 4), `WAN_TRANSFORMER_DIR`.

To scope it down first, run just the preflight:
```bash
export TT_METAL_HOME=/path/to/tt-metal PYTHONPATH=$TT_METAL_HOME
export TT_METAL_LOGGER_LEVEL=Info
python wan_load_bench.py preflight --work /tmp/pf --devices 4
```

## Expected result & what to report

- **preflight**: `aligned rejects=0`, `unaligned rejects>0` → PASS (pinned path active).
- **full run** prints two lines like:
  ```
  aligned:   rejects=0     ... total_load_time=<Ta>s ...
  unaligned: rejects=<N>   ... total_load_time=<Tu>s ...
  ```
  where `N ≈ #tensors × #devices` (e.g. 40 blocks × 10 weights × 4 devices ≈ 1600).

Report to the user: the two summary lines (from `$WORK/logs/summary.txt`), i.e. aligned vs
unaligned **total_load_time** and **rejects**, plus GB loaded and `mean_ms`/`max_ms`. The
headline is the load-time delta (unaligned slower) and that aligned eliminates all the
`Pinned source memory ... must be aligned` rejections. Keep the logs under `$WORK/logs/`.

## Notes / gotchas
- Page cache: the bench measures warm loads (files may be in page cache), which cleanly
  isolates the pinned-vs-staged difference (that cost is CPU-side, paid regardless of page
  cache). If you want a cold-disk number too and have root: `echo 3 | sudo tee
  /proc/sys/vm/drop_caches` before a bench.
- Device memory: the bench loads each weight then deallocates, so only one weight is
  resident at a time — no full-model memory pressure.
- If `preflight` FAILS with rejects=0 on unaligned: IOMMU is off, or the KMD lacks
  read-only page pinning, or shards are ≤ 32 MB. Check IOMMU first.

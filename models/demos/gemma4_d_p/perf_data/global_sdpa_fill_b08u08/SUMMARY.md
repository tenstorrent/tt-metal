# Gemma4 global ring SDPA: improvements (bh-glx-120-b08u08, night of 2026-10-20)

Base: main `909748641c1`. Local branches only (not pushed, no PRs). Raw log: `results.txt`; per-run logs and
tracy reports alongside; runner `runner.sh` / `queue.txt`.

## Branches

| Branch | Commits | What |
|---|---|---|
| `mmanzoor/gemma4-gsdpa-ksplit8-q128` | 3 | Port of Kyle's 8-band K split (`c4142e621c8`, reducer sender-mask fix + test cases incl. bf16 q128/5); Gemma4 `L1_SMALL` (1 KB) in the adapter + test fixtures; chunk 2048 global attention at q128 / k256 / 5 bands |
| `mmanzoor/gemma4-gsdpa-v-from-k` | 1 | ring_joint_sdpa reader: V's overlapping packed columns copied core-locally from the K^T tiles; only V's last 4 tile columns read from DRAM (bit-exact) |
| `mmanzoor/gemma4-gsdpa-v-from-k-overlap` | +1 on v-from-k | Issue V's reads before the K multicast. Negative result, kept for record |

## Results (chunk 2048 unless noted)

| | main | ksplit8-q128 | v-from-k | v-from-k-overlap |
|---|---:|---:|---:|---:|
| 256k prefill, device (ms) | 10310.8 | **9906.1 (-3.9%)** | 10019.8 (-2.8%) | 10058.1 (-2.4%) |
| first / last chunk (ms) | 53.6 / 107.6 | 54.2 / 100.5 | 53.5 / 103.3 | 53.5 / 103.8 |
| global SDPA chunk 0 (us) | 145.5 | 213.4 | **140.5** | 142.8 |
| global SDPA chunk 64 (us) | 2666.9 | **2364.0** | 2459.7 | 2487.3 |
| global SDPA chunk 127 (us) | 5253.7 | **4611.7** | 4826.7 | 4899.7 |
| KV PCC overall / min head | 0.982643 / 0.941301 | 0.983214 / 0.943989 | identical | identical |
| SDPA Gemma4 global accuracy | 7/7 | 9/9 (one NFS JIT-cache flake, passed on rerun) | 7/7, bit-identical | 7/7, bit-identical |

4K / 8K chunk e2e (main only; the branch runs were cut off): 7211.0 ms / 6404.2 ms.

## Where the remaining time is (chunk 64)

- ksplit8-q128: remote ring iterations 265 us each (compute-bound, matmuls ~70% of peak); **first remote shard
  stall 150 us (6%)** — the local iteration now finishes before the first shard lands; partial readiness would
  hide it but its port deadlocks on main (not debugged; the compute stride filter and per-Q chunk count looked
  consistent). Rescale ~10% (ceiling, measured last session). 30 of 110 cores idle.
- v-from-k: remote iterations 269 us (same rate on 96 cores); injector chain ~14.8 us vs ~12.5 us step, so 1.17 us
  of V wait per step remains; the injector's single-core K DRAM read (139 KB at ~21.6 GB/s) is now the largest
  piece; slowest cores drift +86 us behind the median by the end.

## Not run / interrupted

At 05:55 another user (zbaczewski) started a galaxy reset on this host (tt-device-mcp reset); the ksplit8-q128
and v-from-k 4K/8K e2e runs failed at device open, and the runner was stopped. No device state was changed by me.

## Next

1. ksplit8-q128 is the branch to land (-3.9% e2e). Re-run its 4K/8K e2e (it only changes the 2K config).
2. Partial readiness on top of it (~150 us / 6% of global SDPA): debug the deadlock with DPRINT of the reader's
   semaphore thresholds, under `timeout`.
3. Lazy rescale with in-place accumulation (up to ~10%), then distributed K/V injection (several cores per band
   row read DRAM) for the q64 path.

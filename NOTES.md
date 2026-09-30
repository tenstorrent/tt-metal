# t32 notes: ring-joint SDPA chunk re-sweep + V2A split-K

- Branch base: t10 (e7295c8cfd8) + cherry-pick of t14's block trace harness (57497b94bd5).
  Sweep test: 65bc398d0a4 `test_ltx_ring_sdpa_chunk_sweep` in models/tt_dit/tests/models/ltx/test_transformer_ltx.py.
- Worktree links (gitignored): build, build_Release -> ../t10/build_Release, runtime -> ../t10/runtime,
  ttnn/ttnn/_ttnn.so -> t10's. Env: tmp/env.yaml (TT_METAL_CACHE=/var/tmp/t32-tt-metal-cache, on /, not /home).
- Shipped configs (4x8, attention_ltx.py): self S1 (96,256), S2 (192,512); V2A cross q=32, k=512 (fallback k).
- Driver: `nohup bash tmp/drive.sh > tmp/drive.log` = prewarm_and_submit per stage (600 s cap, hook-enforced).
  Stage-3 job IDs for stage_1 / stage_2 are the "Job N queued" lines after "stage 3/3" in tmp/drive.log.
- On resume: for each stage job ID: `tt-device-mcp logs -n 100000 <ID> | grep SWEEP`.
  Lines: `SWEEP <stage> self q= k= us= bitexact|maxabs rel_l2`, `SWEEP <stage> v2a q=32 k= ...`,
  `SWEEP <stage> v2a_splitk us= ...` (first config of each group is the shipped one = "ref").
- Next: pick best self/cross chunks per stage; if a win > noise, update ring_sdpa_chunk_by_n /
  cross k in attention_ltx.py, confirm with block A/B (test_ltx_transformer_block_trace_perf 8k) and an e2e run.
  If split-K clearly beats the ring cross at acceptable rel_l2, wire it behind a flag in attention_ltx.py.
- t22 (general denoise) works on noise prefetch / prompt handoff: no overlap with SDPA.

## 2026-09-30 16:xx, device pause (attempt 1 resumed)
- Job 656 was the capture-only prewarm (kernels never executed): its SWEEP numbers are meaningless. drive.sh
  is gone and none of our broker jobs are queued. No broker submissions while the pause holds.
- V2A split-K CPU reference: models/tt_dit/tests/models/ltx/test_v2a_split_k_reference.py (6 pass, torch only).
  Math matches dense attention to 1e-5 at S1/S2 shapes, incl. an all-padding shard and far-apart logits.
  bf16 emulation: split-K rel_l2 0.0063 (S1) / 0.0066 (S2) vs 0.0061 / 0.0064 unsplit. Dropping the
  shared-max rescale fails all 6, so the test guards the merge.
- Chunk candidates picked analytically (tmp/chunk_model.py): work units = 8 heads x q-chunks over 110 cores,
  partial last K chunk computed in full. S1 q=96 is already optimal (104 units, 1 wave); k=608 (19 tiles)
  divides the 38-tile shard. S2 q=192 (208 units, 2 waves) ties q=128/96; candidates (384,256), (192,448), (128,608).
- Sweep test now takes LTX_SWEEP_SELF / LTX_SWEEP_CROSS_K / LTX_SWEEP_SPLITK for single-config jobs and logs
  host_rel_l2 (vs fp32 host attention) for the ring cross and split-K.
- Next: when the pause lifts, run tmp/READY_32.md jobs one at a time (1 = S2 split-K first).
  If split-K wins: move _v2a_split_k into attention_ltx.py behind LTX_V2A_SPLIT_K (key_bias/row_zeros must be
  built before trace capture), then block A/B + e2e.

## 2026-09-30 18:37, sweep moved to blx03 (g15blx02 still paused)
- blx03 worktree: ~/fasth3/t32 (git worktree of blx03's ~/fasth3/tt-metal clone at 35a6d41811e). Links build_Release,
  runtime, ttnn/ttnn/_ttnn.so to ~/fasth3/tt-metal (t36 build; no host C++ difference vs this branch).
  tmp/ there holds sweep.sh, blx03_sweep.sh, env.yaml (untracked; copies in this worktree's tmp/).
  JIT cache: blx03 ~/fasth3/cache/t32-tt-metal-cache (delete when done).
- Driver (g15blx02): tmp/drive32.sh, log tmp/drive32.log. Job 865 = S2 split-K; then S1 split-K, S1 self, S2 self,
  one at a time, stops on first failure; ends with DRIVE32_DONE. Job IDs are the JOB[...] lines.
- Read results: `ssh g14blx03 'tt-device-mcp logs -n 100000 <ID>' | grep SWEEP | sed 's/.*SWEEP/SWEEP/'`.

## 2026-09-30 18:5x, S2 split-K measured (blx03 job 865)
- Test PASSED; the wrapper exited 2 only because I edited tmp/blx03_sweep.sh while bash was reading it. SWEEP numbers valid.
- stage_2 (5 ops): self q=192 k=512 4586.0 us (ref); V2A ring cross q=32 k=512 838.1 us, host_rel_l2 0.0268;
  V2A split-K 1638.3 us, rel_l2 vs ring 0.0789, host_rel_l2 0.0777. Split-K is 2x slower and 3x less accurate: rejected at S2.
- Driver part 2: tmp/drive32b.sh (log tmp/drive32b.log, ends DRIVE32B_DONE): S1 self 96,608/96,416 plus S1 split-K in one job,
  then S2 self 384,256/192,448/128,608. It waits while any smarton job runs on blx03 (t-other job 867/869 were ahead).
- Next: read JOB[...] ids in tmp/drive32b.log, grep SWEEP. If a self config beats ref by more than noise (~1-2%) and is bit-exact,
  set it in attention_ltx.py ring_sdpa_chunk_by_n and run block A/B. Otherwise report "shipped configs already optimal".

## 2026-09-30 19:2x, sweep complete (blx03 jobs 875, 876; both RUN_EXIT=0, device healthy after)
Per-op device time, 5 ops averaged, 1080p/145f shapes on 4x8. rel_l2/maxabs are vs the shipped config.

| stage | op | q | k | us | vs ref | note |
|---|---|---|---|---|---|---|
| S1 | self | 96 | 256 | 638.4 | ref (shipped) | |
| S1 | self | 96 | 608 | 664.2 | +4.0% | rel_l2 0.0171 |
| S1 | self | 96 | 416 | 660.8 | +3.5% | rel_l2 0.0161 |
| S1 | V2A ring cross | 32 | 512 | 259.0 | ref (shipped) | host_rel_l2 0.0241 |
| S1 | V2A split-K | - | - | 536.9 | +107% | host_rel_l2 0.0270 |
| S2 | self | 192 | 512 | 4579.5 | ref (shipped) | |
| S2 | self | 384 | 256 | 4712.8 | +2.9% | rel_l2 0.0220 |
| S2 | self | 192 | 448 | 4645.0 | +1.4% | rel_l2 0.0204 |
| S2 | self | 128 | 608 | 6514.6 | +42% | rel_l2 0.0213 |
| S2 | V2A ring cross | 32 | 512 | 837.4 | ref (shipped) | host_rel_l2 0.0268 |
| S2 | V2A split-K | - | - | 1638.3 | +96% | host_rel_l2 0.0777 (job 865) |

- Verdict: the shipped chunk sizes are already the fastest of every candidate, and V2A split-K is ~2x slower
  in both stages. No model change: denoise delta 0, output unchanged. The analytic model (tmp/chunk_model.py)
  predicted ties for the larger K chunks; in practice they lose 1.4-4%, likely L1/CB pressure, not core waves.
- Cleanup done: blx03 ~/fasth3/t32 worktree and /var/tmp/fasth3/cache/t32-tt-metal-cache (890 MB) removed.
- Remaining ideas (not pursued here): sparse S2 self-attention (S2 self is ~4.6 ms/op, the real cost), fusing
  the V2A cross into the self ring pass.

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

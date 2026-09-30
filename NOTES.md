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

# DeepSeek-V4.1-Flash: future work

## Device sampler: inverse-CDF draw returns zero-weight tokens (`tt/device_sampler.py`)

Found 2026-10-19 from a user bug report; reproduced on hardware (host .47, 4x8 mesh).

**Where it applies.** Only our own sampler (`DSV41DeviceSampler`, full-vocabulary draw). The default serving config uses the shared sampler (`DSV41_SHARED_SAMPLER=1`, `DSV41_SHARED_SAMPLER_SPEC=1`, set in `generator_vllm.py`), which never calls this draw, so image `0.24.0-9bf06a9-84b529b` is not affected. Affected: runs with either flag set to 0, the previous release image `0.24.0-c9cfb00-ffd1a57` (its spec verify used our sampler by default), and `text_demo` / other paths outside vLLM (`dsv41_model` defaults to our sampler there). The top-k path (`top_k <= 32`, `forward_topk`) never returned a zero-weight token.

**Cause.** `ttnn.matmul` on fp32 inputs loses about 2e-3 relative precision (`ttnn.sum` is accurate to about 1e-7). The draw builds the in-row prefix, the row offsets and the column CDF (`incl`) with matmuls, but takes the overall total with `ttnn.sum(tot_all)`, so the target `u * total` and the CDF disagree, and the row offsets are not continuous at row boundaries. Consequences: for `u` near 1 the selected column / index can fall past the last token with weight (the last vocabulary id 129279, weight 0), and over a stretch of zero-weight tokens the count `sum(cs <= t)` is wrong because the prefix is not monotone.

**Measured** (`tests/test_sampler_cdf_exact.py` on branch `ssinghal/dsv4p1-sampler-cdf`, 3.81M direct draws, against an exact float64 inverse CDF):
- Zero-weight tokens returned: 689 draws (7.5e-5 to 7.3e-4 per scenario depending on the weight pattern; 2.45e-4 for a real top-p 0.9 nucleus of 10 tokens).
- Draws differing from the exact answer by more than rounding: 4.5% overall; 0.6% for 32 tokens, 27% for a dense temperature-only row, 49% for a 2000-token run.
- Full production sampler, about 1M draws: top-p rows gave zero-weight tokens at 2.6e-4 to 7.6e-4 (229 draws), up to 4.9% wrong beyond rounding.
- The 8 mesh-column replicas always agreed. One token or two exactly-valued tokens: no error.
- Speed of the current code: 3.76 ms at 8 users per row, 9.47 ms at 32, 1.06 ms top-k path.

**Fix (done on the test host, not committed).** Use only exact sums: the total and the column CDF from the same accumulation, choose the column among those with positive mass, clamp the final index to the last position with positive weight. On the same 3.81M draws: 0 zero-weight tokens, 0 differences beyond rounding (all within about 1e-7 of the total), 0 replica disagreements. Worktree `/mnt/tt-data/ssinghal/wt/tm_sampler` (branch `ssinghal/dsv4p1-sampler-cdf`): commit `493e5e3d1e6` has the new test and the draw moved into `_draw` (same operations); the fix itself is an uncommitted edit of `tt/device_sampler.py` there.

**To do.**
- Commit the fix, time it (the draw must not regress the 3.4 ms / 1.06 ms step), run `test_device_sampler.py`, `test_sampler_cdf_exact.py` and `test_sampler_vs_shared.py` on the fix, including the full-sampler half that was not run with it.
- Decide whether to ship it for the `DSV41_SHARED_SAMPLER=0` and non-vLLM paths in the next release.

Logs: `/mnt/tt-data/ssinghal/dsv4-logs/samplercdf` (`before_cdf.log`, `after_cdf.log`, `before_cost*.log`, `before_vsshared.log`, `probe2.log`, `probe_fp32_ops.py`).

## Serving: batched throughput of the 0.24.0-9bf06a9-84b529b defaults (APC on, 4096 token budget)

Measured 2026-10-19 (`/mnt/tt-data/ssinghal/dsv4-logs/serving_bisect/README.md`, table in RELEASE_NOTES.md Known issues). Against the previous image at 16 concurrent requests, distinct prompts: ISL 1024 TTFT +14 % / TPOT +21 % / tok/s -16 %, ISL 8192 TTFT +100 % / TPOT +71 % / tok/s -44 %. APC explains the ISL 1024 rows (V3: APC off = within 5 %), the 4096 token budget explains most of the ISL 8192 rows (V4: 67584 = TPOT +7 %); both set back (V5, V6) = within 5 % of the previous image, the demo (no scheduler) is identical on both code versions: the code is not slower, the defaults trade batched speed for the features. Budgets 16384 / 32768 only recover part (16 x 8192 TPOT +39 % / +22 %).

To do (all need a code change and a rebuild unless noted):
- **APC snapshot cost (about 90 ms per snapshot, 182 ms per 1028-token prompt, synchronous on the engine thread, no decode runs meanwhile).** Options: snapshot only above a minimum prompt length (512 / 1024-token snapshots are cheap to recompute), make the device-to-host copy asynchronous / overlap it with the next prefill call, snapshot only prefixes that are likely to be reused. Target: 16 x ISL 1024 within 5 % of the previous image with APC on.
- **Chunked prefill packing at budget 4096.** An 8196-token prompt is cut into 4096 + 4100 tokens and 1-2 prompts share a call: 72-76 prefill calls and 286-304k prefilled tokens for 262k prompt tokens (about 10 % padding), against 15-18 calls with budget 67584. Options: pieces that fit the prompt (budget a multiple of the typical piece), pack more short remainders per call, avoid the per-call overhead. Target: 16 x ISL 8192 TPOT within 10 % of the previous image at budget 4096 (keeps the 32k-prompt decode stall at 9.1 s instead of 32.7 s).
- **Budget default (config only).** If APC and packing are not fixed: consider shipping the `_noapc` asset spec (APC off, budget 67584) as the batch-throughput config next to the default one.
- Re-measure with distinct seeds per run (the old default sweep re-sent seed-0 prompts, vLLM logged about a 40 % prefix hit rate in its 8192 rows, so it showed only +19 %).

## Host .40: device hangs on host-to-device writes (not a code problem)

Found 2026-10-19. On .40 the demo hangs in `ttnn.zeros` (`prefill_sparse.py:251`) blocked in `FDMeshCommandQueue::wait_for_outstanding_reads` (py-spy: `demo_ab/probe40_b16_old.pyspy.txt`) for B=4 old, B=4 new and B=16 old; the same runs pass on .32 (and .47, .45, .46). dmesg on .40: `tenstorrent: pin_user_pages_longterm failed: -14` (at the hang, and earlier the same day before our runs).
Only difference found between the hosts: kernel command line. .40: `iommu.passthrough=0 iommu.strict=0` (translated IOMMU), up 5 days (rebooted recently); .32 / .45 / .46 / .47: `iommu=pt`, up 4-7 weeks. Same tenstorrent driver 2.7.0, same hugepage setup (Hugetlb 32 GB, 1 GiB pages) on all.
To do: keep .40 out of device runs; have the host owner compare .40's boot config with the others (iommu=pt) and re-test after a reboot with the same command line. Not changed from here (needs root + reboot of a shared machine).

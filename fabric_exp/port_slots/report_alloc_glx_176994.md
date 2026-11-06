<!-- Galaxy Slurm 176994 (bh-glx-110-a01u02, 2026-11-05 05:36-05:43 UTC): ring-jam sanity of the 1D leftover order,
port-alloc 9ce0acbeaba staged against tier (alloc_glx.sbatch JAM=1; results /data/aagarwal/frs-glx/jobs/176994/results).
SLOT_ALLOCATOR.md section 12.15. Cycles per packet at 1.35 GHz from the summary CSVs; pass a / pass b. -->

# Ring-jam sanity on Galaxy: staged 26,25,32 against tier 16,16,32 (R 32, T 16 in both)

- Run: all 14 steps exit 0, every test passed, slot rows as captured, no hang. glx_frg_val (data-checked: Ring and XY
  torus full-ring multicast and all-to-all, unicast write and fused atomic inc, 4096 B) 10 / 10 on each arm, first.
- The validation's XY torus routers run 17,16,16 in both arms.

| row | tier | staged | staged vs tier | 172983 branch (120-c) | 172983 base, jammed (120-c) | §9.5 A16, 26,25,32 (110-a, 172449) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8-ring 3072 B (control) | 194.2 / 194.3 | 190.8 | -1.8% | 194.7 / 194.8 | 195.6 | 191.6 |
| 8-ring 3584 B | 227.9 / 228.1 | 209.0 / 209.1 | -8.3% | 228.5 / 228.4 | 310.1 / 312.6 | 209.1 |
| 8-ring 4352 B | 269.4 | 254.1 | -5.7% | 269.6 / 269.3 | 375.1 / 372.2 | 254.2 |
| 4-ring 4352 B | 254.9 | 255.1 | +0.1% | 254.9 / 255.0 | 323.7 / 339.9 | 255.1 |
| half ring 4352 B | 253.6 | 253.7 | +0.0% | 253.6 | 331.5 / 332.1 | 253.8 |
| ring 1-hop 4352 B | 249.6 | 249.5 | -0.0% | 249.5 | 249.5 | 249.5 |

- No row is at the jam level (310-375) in either arm or pass; the passes agree within 0.1% on every row.
- Tier on this 110-a node reproduces 172983's branch (a 120-c node) within 0.3% on every row.
- Staged reproduces the study's A16 arm (§9.5, the same 26,25,32 rows and T 16, on 110-a) within 0.5% on every row.

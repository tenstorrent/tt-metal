# B1 investigation - September 11

The local controlled-routing matrix is complete on branch `ssalice/mistral4-b1-investigation`. Reports and experiment files remain separate from the B3 test-only `akhan/mistral4-prefill-followups` branch. No GitHub issue or message has been posted.

Twelve successful fresh-process captures cover L18 and L23 with captured, placement-balanced, and source-shuffled routing, two processes per case. All use the same archived manager-free worker. Validation confirms1920 target operation rows: eight devices × ten iterations × two operations × twelve captures. After excluding the first iteration from each process,1728 rows remain.

Balanced placement reduced isolated Combine duration by38.36% on L18 and25.82% on L23; Dispatch fell about2.1% on each layer. Source shuffling reduced L18 Combine by4.06% while preserving aggregate destination counts, supporting further source-flow/ordering investigation. These are eager kernel results with an FFN surrogate, not model throughput gains or correctness-validated production changes.

The run used eight visible Galaxy chips and full128-expert SP8/TP1 placement. Initial stalled/failed attempts required targeted hardware recovery; another initialization failure caused recovery between some repeats. All failed or incomplete attempts are excluded. The final requested capture completed successfully; root is checking device release. No additional hardware work is launched by the analysis agent.

Read [LOCAL_FINDINGS.md](LOCAL_FINDINGS.md) for grouped means/ranges, changes, scope and remaining B1 questions; [MATRIX_RESULTS.md](MATRIX_RESULTS.md) for every process; [VALIDATION.md](VALIDATION.md) for recovery and extraction details. Rebuild with `python3 aggregate_timings.py`.

The next unresolved task is pairing actual PP selections with timings from the same full-model execution, followed by a placement implementation with numerical correctness and matched traced throughput validation. [PAIRED_CAPTURE_PLAN.md](PAIRED_CAPTURE_PLAN.md) scopes that work. Current results do not causally explain the historical PP L18/L23 timing gap.

Final device ownership check: `fuser /dev/tenstorrent/*` returned no holders after the final successful matrix process exited 0.

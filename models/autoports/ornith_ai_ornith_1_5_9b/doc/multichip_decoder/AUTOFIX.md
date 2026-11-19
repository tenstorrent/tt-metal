# Geometry hang repair — succeeded

The original one-reader down-projection sweep allocated 64 input storage cores ×
2 K tiles for a tensor with only 96 K tiles. The host factory divided 96 blocks by
64 cores, truncating to one block per shard. The reader kernel reconstructed 96
sender-coordinate entries although the factory supplied 64. Live operation 309 on
all four chips showed seven readiness credits at logical (0,4), while the intended
sender (0,0) waited with zero. Activation CB0 had no data; weight CB1 was full.
This was a matmul sender-coordinate mismatch. CCL was not involved in that op.

`AUTOTRIAGE.md` records the exact current-source factory/kernel locations and live
evidence. The main agent verified those locations before editing. Two isolated,
read-only AutoFix agents independently checked exact-K and padded-K hypotheses;
neither edited source or ran hardware.

The model-local `exact_k_shards` planner now sizes input storage independently of
output geometry, with `storage_cores * shard_tiles == K/32` and each block dividing
shard_tiles. K3072/requested64 uses 48 input cores × 2 tiles; K1024/requested64 uses
32 input cores × 1 tile. The existing 32-core control remains representable.
Padding both operands to K4096 would also repair the invariant but adds 33.3%
K-dependent traffic/MACs, 4.25 MiB weight payload/chip, and runtime activation
padding. It was rejected on that cost after the unpadded repair passed.

Changed implementation: `tt/multichip_decoder.py`. Changed validation:
`tests/multichip_geometry.py` and `tests/test_multichip_planner.py`. No shared kernel
or earlier-stage implementation changed. The sweep now prints each candidate before
dispatch, checkpoints completed cases, and supports resumption.

Commands from the checkout root:

```bash
python models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_planner.py
timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_geometry \
  --layer 0 --role mlp.down_proj --repair-probe \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/repair_geometry.json
```

Host planner regression passed. The device command completed with exit 0 using the
same real weights, one reader, requested output64 and exact input48. Both block1
and block2 passed all-device eager PCC against the known-passing 32-core reference;
minimum PCC was 0.9999999403953552. Each completed three groups of 40 warmed
nonblocking trace replays. Raw evidence: `repair_geometry.json` and
`repair_geometry.log`.

The resumed sweeps completed 156 candidates per layer (312 total), all eager PCC
at least 0.995 and all warmed trace groups complete. Evidence:
`geometry_{0,3}.json`, `geometry_{0,3}_after_repair.log`, and
`geometry_comparison.csv`. `geometry_0_before_repair.json` preserves the original
132 completed candidates; the next candidate was the failure. The initial parent
suspicion of a 32-core/eight-reader failure was disproved by that ledger and the
compiled runtime arguments.

Recovery was bounded and serialized: original timeout exit124 at 19:54:51 UTC;
post-timeout trivial probe exit124; no remaining device owners; one authorized
reset exit0; all four chips listed; all-chip add/synchronize smoke exit0. Logs:
`post_timeout_smoke.log`, `geometry_pre_reset_list.log`, `geometry_reset.log`,
`geometry_post_reset_list.log`, `geometry_recovered_smoke.log`. No manual signal,
other user's process termination, or second recovery reset was needed.

The shared kernel's unsafe oversharded geometry remains an upstream limitation;
the delivered planner prevents it. Final whole-decoder numerical, trace, watcher,
and performance gates are recorded separately in the stage README and audit.

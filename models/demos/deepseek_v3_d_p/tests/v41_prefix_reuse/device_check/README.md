# V4.1 prefill prefix reuse: device checks

These checks run the production 512k prefill runner on one galaxy. The settings are chunk 8192, MAXSEQ 524288, A3 + islands +
warm-all, device embed, Engram RM and the CP2 export. They go through the real common prefill runner: H2D socket, layer-ack
channel, and the KV table exported to a file. `runner_harness.py` adds test-only arms and digests to that runner.
`driver.py` plays tt-d-gen's prefill worker: the chunks of a prompt go back to back, the next prompt starts only after all
40 acks of every chunk, and headers carry `mark_last_chunk` bits 29 / 30.

**Preconditions (both checks).** Run from host 29 in a maintenance window. The serving prefill runner on host H (default 30)
must be down, and H's chips must be free; the script refuses otherwise. H needs at least 10 GB free on its root disk. The
Engram row servers (layer 1 on 31, layer 14 on 29) are reused when they are up, and started otherwise. The script never
restarts the serving stack: bring it back afterwards with `v41ax_stack_up.sh` / `stack_up.sh`.

## H1 (device-held prefix): `run_device_check.sh`

```
H=30 bash run_device_check.sh                  # PHASES=R,S by default; R,S,U,T for unmarked headers + timing
```

- **R:** cold references, each run after X. Prompt A runs twice; that pair is the determinism control.
- **S:** the reuse sequence A, B, A, A2, A2, AP, A.
- **U:** the same sequence with unmarked headers.
- **T:** timing.

**Reading the result.** `[driver] VERDICT determinism` must be PASS: the cold prefill is reproducible byte for byte.
`[driver] VERDICT reuse` must be PASS: every S / U digest equals the cold reference of the same prompt. Results are in
`$OUT` (`check.log`, `results.json`, `runner.log`). Run1 (11-11, host 30) is in `v41ax_logs/prefix_check/run1/`.

## H2 (host chunk store): `run_device_check_h2.sh`

```
H=30 bash run_device_check_h2.sh               # PHASES=R2,S2 by default; R2,S2,T2 adds timing
```

This is the same build and harness as H1, with the store added to the runner's environment:
- `V41_PREFILL_PREFIX_STORE_GB=8`: about 0.7 GiB is used;
- `V41_PREFILL_PREFIX_HOST_RESERVE_GB=32`;
- `V41_PREFILL_PREFIX_SPILL_TIPS=2`.

The tree is copied to `/home/ttuser/sdawle/tt-metal-1017-h2` on H. Results go to `v41ax_logs/prefix_check_h2/<date>/`.
Wall time is about 30-45 min: the build takes 15-25 min, then about 15 min of prompts and digests.

**Prompts.** The ids come from fixed seeds; every length is a multiple of 128, like the engine's `floor128(S - 2541)`.

| Prompt | Content | Chunks |
|---|---|---|
| X | 2C unrelated tokens | 2 |
| A | 3C + 4992 | 4 (ragged last) |
| A2 | A + 12032 = 5C + 640 | 6 |
| AP | A[:2C] | 2 |
| B | A[:C + 7040] + 20096 | 5 (shares A's chunk 0 only) |
| U | 2C + 1280 unrelated tokens | 3 |

**Phases.**

- **R2 (references).** Arm `off` is today's runner: reuse is bypassed, and the store is neither read nor written. Each
  reference runs after X. A runs twice, which is the determinism control.
- **S2 (the store sequence).** Arm `clear` empties the retention and the store, then runs X. After that, arm `reuse` runs the
  prompts below. Each one is digested and must equal its R2 reference. Each one's per-chunk pattern must also equal the
  driver's `EXPECT_S2`. The codes are: S = skipped (the device holds the chunk, H1); R = restored from the store; C =
  computed; C+Ns = computed after N silently rebuilt chunks.

  | # | Prompt | Expected | What it exercises |
  |---|---|---|---|
  | 1 | A | CCCC | computed; its 3 full chunks are read into the store |
  | 2 | U | CCC | the device leaves A; A's tips (3C, 2C) are spilled after U's last chunk |
  | 3 | A2 | RRRCCC | **the AgentX case**: A's chunks are restored although the device holds U; resumes on A's tip at 3C |
  | 4 | U | RRC | U restored; its ragged last chunk is computed on U's tip at 2C |
  | 5 | A | RRRC | an exact repeat after another prompt |
  | 6 | U | RRC | |
  | 7 | AP | RR | a strict prefix; its LAST chunk is restored with the stored carries at 2C (windows from them, export cleared past 2C) |
  | 8 | U | RRC | |
  | 9 | B | R,C+1s,C,C,C | chunk 0 restored; there is no tip at C, so it is rebuilt silently from 0 (restore, then silent compute) |
  | 10 | A2 | SRRRRC | H1 and H2 in one prompt: chunk 0 is held on the device (B's), 1..4 come from the store, resumes on A2's tip at 5C |
  | 11 | A2 | SSSSSC | held on the device; the ragged last chunk is recomputed |

- **T2 (timing, optional, 2 reps).** The order is X `clear`, A `off`, X `off`, A `reuse`, U `reuse`, A2 `off`, A2 `reuse`.

**Reading the result.** All of these are in `$OUT/check.log`.

1. `[driver] VERDICT determinism ... PASS`: R2's A twice is byte-identical in the export, model rows, slab entry rows and
   Engram history. If it FAILs, nothing else can be judged bit-exactly; see design risk R1.
2. `[driver] VERDICT reuse ... PASS`: every S2 digest equals its R2 reference. The digest covers:
   - the 7 export tensors, whole, from chip 0, with the window rings also checked on the last chip;
   - the model's entries and index keys `[0, S / ratio)`;
   - the path-A slab entry rows (`slab=`);
   - the Engram history (`engram=`).

   `results.json` → `verdict` names any failing prompt. Its `digests` → `tensors` holds per-tensor CRCs, which localize the
   failure: `swa` / `win` are the window rings, `ent*` / `idx*` are the restored rows.
3. `[driver] VERDICT h2 pattern ... PASS`: each S2 prompt did what it should, as listed above (the triples are prompt, got,
   expected). If the digests are equal but the pattern is not, the store missed or mis-decided, which is a performance
   defect, not a correctness one. Use the runner's `[v41-prefix]` lines to see why: `store slot 0`, `restored`, `stored`,
   `spilled`, `rebuilt the carries`, `NOT stored`.
4. Host-side numbers, from `runner.log`, are MEASURED on the device host:
   - `[v41-prefix] slot 0 chunk [s, e) stored: read X s, copy Y s`: the capture cost per computed full chunk;
   - `[v41-prefix] slot 0 restored [s, e) from the host store: model rows X s, export rows + Engram Y s (dispatch)`;
   - `[v41-prefix] slot 0 spilled N checkpoint(s) ... in X s`;
   - `[v41-prefix] ... new prompt ...: ...; store N/M chunk(s) ...; runner RSS X GiB`.
5. T2 walls, from `results.json` → `results[*].wall_to_last_ack_s`, phase `T2<rep>`:
   - `A reuse − A off` is the capture overhead on 3 full chunks;
   - `A2 reuse` against `A2 off` shows 3 restored chunks replacing 3 computed ones.

   Report both reps.

The store sequence is reproduced on the host fake by `test_v41_prefix_store.py::test_store_sequence_equals_cold`. The
`EXPECT_S2` patterns were checked against the real runtime code on the fake at the same chunk-count structure.

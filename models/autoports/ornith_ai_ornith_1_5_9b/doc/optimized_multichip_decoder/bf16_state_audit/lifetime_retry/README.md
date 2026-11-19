# RNE direct probe L1 lifetime AutoFix

Original direct run passed all configured wave104 batches and wave7 through B26, eight steps/four ranks including bitwise quantized state, then failed before B27 step0 candidate dispatch. Preserve `bf16_state_rne_direct_watcher.log/.json`: last recorded comparison is B26/wave7/step7/rank3. The allocation error says the candidate static CB region ends at542656 bytes while a live L1 allocation starts at440320; overlap is102336 bytes. This is a memory gate failure, not a PCC failure.

## Ownership and exact state-storage peak

A BF16 state per request contains8×128×128×2=262144 bytes; its FP32 counterpart is524288 bytes. All axes are tile aligned, so these figures already include tile storage (but not allocator/bank alignment overhead).

The old probe retains the preceding step's device objects via `tensors`, `ranks`, the last `values` tuple, and named output/roundtrip locals. At the B26→B27 transition, allocating new state variables does not release old state references held by those containers:

| Live allocation group at failing dispatch | BF16-request units | Bytes/chip |
|---|---:|---:|
| Old B26 candidate state | 26 | 6815744 |
| Old B26 quantized FP32 control | 52 | 13631488 |
| Old B26 unquantized FP32 control | 52 | 13631488 |
| Old B26 rounded BF16 control | 26 | 6815744 |
| Old B26 widened FP32 temporary | 52 | 13631488 |
| New B27 candidate + two FP32 controls | 135 | 35389440 |
| **State allocations alone** | **343** | **89915392 (85.75 MiB)** |

Rank views alias these buffers rather than adding their own state storage, but their references prevent release. Three prior L1 outputs and the newly allocated candidate output add at least256 KiB (each physical output tile group is65536 bytes at B<=32), plus allocator/bank rounding and other allocations. No memory snapshot was captured at failure, so exact per-bank fragmentation/total process L1 cannot be reconstructed solely from the error. The state-storage ownership above is source-determined.

The candidate CB payload is430080 bytes per active core; production-control CB payload is434176. Both are fixed by descriptors and independent of B/wave. Wave7 changes the number of heads processed sequentially, not per-core CB size. Source therefore identifies unnecessary test allocation overlap, not an inherent B27 CB shape limit; hardware retry must confirm that diagnosis. A real L1 capacity violation occurred in the harness, and is not ignored.

## Fresh variant and unchanged gates

New `tests/multichip_bf16_state_rne_lifetime_probe.py` calls the same RNE candidate. Both independent FP32 reference states and BF16/FP32 roundtrip temporaries now use DRAM, because their L1 residency is not part of the candidate storage contract. Candidate persistent state still uses DRAM at wave104 and L1 at wave7; input tensors still alternate DRAM/L1, and every original batch/wave/step remains. Thus candidate L1 reader/writer coverage is preserved.

After every rank's device tensors have been downloaded, delete the rank-view containers, output/temporary handles and uploaded input handles. After each batch, delete all three persistent state handles. This uses ordinary reference release; no forced deallocate, reset, cache flush, or diagnostic suppression is added. All kernel source is unchanged.

At B32, candidate persistent L1 state is8 MiB. All two-control/roundtrip state storage is now DRAM:2×16+8+16=56 MiB peak while rounding, without previous-step duplicate references. L1 inputs/outputs remain small tiled tensors; candidate CB430080 bytes/core still must fit physically. This isolates candidate L1 use from multiple test-reference copies rather than weakening the model contract.

The `stats`, `score`, and `compare` function ASTs are byte-structurally identical to the original probe; all call-site quality gates are retained. Output/global/per-head PCC>=.995, bitwise quantized state, exact roundtrip, finite/constant handling and failure tensor dumps remain required. `validation.json` records arithmetic and source hashes. Python compilation passed. New `bf16_state_rne_lifetime_retry_jobs.json` keeps Watcher and runs the complete default matrix; no hardware was launched by this agent.

## Decoder evidence remains a failure

The completed RNE decoder long run `bf16_state_rne_answer1024_32.log` fails at decode70/request8 with PCC0.993030071, below unchanged .995. The earlier direct-pack BF16 candidate failed the same step/request at0.993284. Root reports short RNE B6 latency0.541183 ms versus FP32 0.545082, and B32 0.669385 versus FP32 0.666156. These short timings are not acceptance evidence and show no B32 gain.

Even a fully clean direct retry would demonstrate equivalence to the specified quantized-state recurrence, not acceptable model accuracy. Recurrent BF16 precision loss remains a material explanation for the long failure. Preserve both long failures and unchanged FP32 control evidence; do not select BF16 or change production based on the partial direct pass.

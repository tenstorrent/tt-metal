# LM-head candidate screen

Real pinned BFP8 weights, HiFi2/FP32, TP4, one logical token with physical32 rows. Timings are traced host windows for the projection only, including split/concat. Actual reduced-model activations; no synthetic precision veto.

| Local split width | Sender cores | Readers/bank | K block | us | PCC | Outcome |
|---:|---:|---:|---:|---:|---:|---|
| interleaved | 64 compute | n/a | auto | 1181.06 | 0.99999994 | isolated screen; see integration disposition below |
| 8192 | 8 | 1 | 3 | 1365.68 | 0.99999994 | isolated screen; see integration disposition below |
| 8192 | 8 | 1 | 7 | 1288.49 | 0.99999994 | isolated screen; see integration disposition below |
| 8192 | 8 | 1 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 8192 | 8 | 2 | 3 | 1056.24 | 0.99999994 | isolated screen; see integration disposition below |
| 8192 | 8 | 2 | 7 | 1084.58 | 0.99999994 | isolated screen; see integration disposition below |
| 8192 | 8 | 2 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 8192 | 4 | 1 | 7 | 1297.33 | 0.99999994 | isolated screen; see integration disposition below |
| 8192 | 4 | 1 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 8192 | 4 | 1 | 42 | — | — | L1 allocation rejected; exact error in JSON |
| 8192 | 4 | 2 | 7 | 1372.83 | 0.99999994 | isolated screen; see integration disposition below |
| 8192 | 4 | 2 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 8192 | 4 | 2 | 42 | — | — | L1 allocation rejected; exact error in JSON |
| 16384 | 8 | 1 | 3 | 1293.56 | 0.99999994 | isolated screen; see integration disposition below |
| 16384 | 8 | 1 | 7 | — | — | L1 allocation rejected; exact error in JSON |
| 16384 | 8 | 1 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 16384 | 8 | 2 | 3 | 1041.31 | 0.99999994 | isolated screen; see integration disposition below |
| 16384 | 8 | 2 | 7 | 1069.57 | 0.99999994 | isolated screen; see integration disposition below |
| 16384 | 8 | 2 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 16384 | 4 | 1 | 7 | — | — | L1 allocation rejected; exact error in JSON |
| 16384 | 4 | 1 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 16384 | 4 | 1 | 42 | — | — | L1 allocation rejected; exact error in JSON |
| 16384 | 4 | 2 | 7 | 1358.21 | 0.99999994 | isolated screen; see integration disposition below |
| 16384 | 4 | 2 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 16384 | 4 | 2 | 42 | — | — | L1 allocation rejected; exact error in JSON |
| 32768 | 8 | 1 | 3 | — | — | L1 allocation rejected; exact error in JSON |
| 32768 | 8 | 1 | 7 | — | — | L1 allocation rejected; exact error in JSON |
| 32768 | 8 | 1 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 32768 | 8 | 2 | 3 | 1023.23 | 0.99999994 | isolated screen; see integration disposition below |
| 32768 | 8 | 2 | 7 | — | — | L1 allocation rejected; exact error in JSON |
| 32768 | 8 | 2 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 32768 | 4 | 1 | 7 | — | — | L1 allocation rejected; exact error in JSON |
| 32768 | 4 | 1 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 32768 | 4 | 1 | 42 | — | — | L1 allocation rejected; exact error in JSON |
| 32768 | 4 | 2 | 7 | — | — | L1 allocation rejected; exact error in JSON |
| 32768 | 4 | 2 | 21 | — | — | L1 allocation rejected; exact error in JSON |
| 32768 | 4 | 2 | 42 | — | — | L1 allocation rejected; exact error in JSON |
| interleaved | 64 compute | n/a | auto | 1180.69 | 0.99999994 | isolated screen; see integration disposition below |
| 65536 | 8 | 1 | 1 | — | — | L1 allocation rejected; exact error in JSON |
| 65536 | 8 | 1 | 3 | — | — | L1 allocation rejected; exact error in JSON |
| 65536 | 8 | 1 | 7 | — | — | L1 allocation rejected; exact error in JSON |
| 65536 | 8 | 2 | 1 | — | — | L1 allocation rejected; exact error in JSON |
| 65536 | 8 | 2 | 3 | — | — | L1 allocation rejected; exact error in JSON |
| 65536 | 8 | 2 | 7 | — | — | L1 allocation rejected; exact error in JSON |
| 65536 | 4 | 1 | 1 | — | — | L1 allocation rejected; exact error in JSON |
| 65536 | 4 | 1 | 3 | — | — | L1 allocation rejected; exact error in JSON |
| 65536 | 4 | 1 | 7 | — | — | L1 allocation rejected; exact error in JSON |
| 65536 | 4 | 2 | 1 | — | — | L1 allocation rejected; exact error in JSON |
| 65536 | 4 | 2 | 3 | — | — | L1 allocation rejected; exact error in JSON |
| 65536 | 4 | 2 | 7 | — | — | L1 allocation rejected; exact error in JSON |

## Integration disposition

The selected8192/8senders/2readers/Kblock3 candidate passes final full-model correctness, context and all-batch performance. The32768 winner overlaps live L1 during integrated warmup;16384 exceeds physical L1 at M544. Explicit output-block-height1 was then rejected by AutoDebug for unsafe partial-core input reads. Final8192 uses standard full output blocks and matches every baseline logit on all four chips at M32/512/544/768/800/1024. Exact allocation errors and the adapted retries are in AUTODEBUG.md, AUTOFIX.md and head_boundaries.json. All timing rows above remain isolated projection controls; final default performance is performance_b*.json.

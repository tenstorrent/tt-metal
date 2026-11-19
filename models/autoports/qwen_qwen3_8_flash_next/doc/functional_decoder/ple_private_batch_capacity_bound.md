# Current PLE tuple-decode batch capacity

Normal batch5 passes the explicitly requested packed-backend-control contract. Its literal strict flag remains false; this is not a literal strict numerical pass. Normal batch6 fails trace finalization. The measured byte counts independently establish a total-DRAM bound for the current resident representation and single tuple-decode trace, rather than relying on fragmentation alone.

| Required live allocation at batch6 | Bytes |
|---|---:|
| Resident model and prepared setup, measured before states | 33,659,240,448 |
| Actual six-request trace | 399,966,208 |
| Six mandatory state payloads | 26,739,072 |
| Activated expert input20MiB plus dense down output80MiB | 104,857,600 |
| Conservative sum | 34,190,803,328 |
| Available device DRAM | 34,178,731,008 |
| Excess | 12,072,320 |

Each request requires655,360B GDN convolution,3,145,728B FP32 recurrent state,655,360B PLE convolution, and64B token-history payload (4,456,512B before bank rounding). Both convolution histories have32 physical rows despite logical lengths3 and9. All six state sets persist throughout tuple replay. The private activated tensor must coexist with sparse down's80MiB output until `sparse_matmul` returns; its deletion follows that call. The repaired reduction does not allocate the previous80MiB clone, and this proof does not count it. Inputs, outputs, hash buffers, routing tensors, later preparation storage, alignment, and fragmentation are omitted conservatively.

Native batch6 trace finalization reports5,511,552B/bank address overlap (44,092,416B across8 banks). That address-footprint number is distinct from the12,072,320B conservative total-capacity excess above. Its trace occupies399,966,208B; the failed allocation snapshot has65,436,160B free after trace storage, below even100MiB before other transient requirements.

Normal batch5 uses333,316,096B trace storage and leaves138,514,944B free after capture. Watcher batch5 uses358,809,600B trace storage,25,493,504B more, and3,948,544B more prepared allocation before capture. Its1,111,040B trace-footprint overlap is an instrumentation limit; normal batch5 success invalidates interpreting it as a four-request physical capacity limit.

Exact artifacts, source hashes, provenance paths, and arithmetic are in `ple_private_batch_capacity_bound.json`. Normal batch5 and batch6 are `ple_private_batch5_normal_v1.json` and `ple_private_batch6_normal_v1.json`; Watcher-only failure is `ple_private_batch5_controlled_v1.json`. This establishes5 as the largest observed supported normal tuple batch under the present trace/resident policy; hypothetical alternate encodings or capture schedules are outside this proof.

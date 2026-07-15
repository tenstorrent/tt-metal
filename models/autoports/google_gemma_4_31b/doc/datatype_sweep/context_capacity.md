# Stage 08 KV-cache capacity recomputation

Target: four Blackhole P150b devices, TP4, production batch 1, advertised
context 262,144 tokens.

The Stage 07 per-device accounting separates the KV cache from all other
model, RoPE, sampler, page-table, trace-region, and allocator reservations:

```text
non-KV accounted bytes
  = 27,672,814,984 - 2,789,212,160
  = 24,883,602,824

usable DRAM bytes/device = 34,225,520,640
```

| Policy | KV storage | Physical weights/device | Batch-1 KV bytes/device | Accounted total | Margin | Largest physical full-context batch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline and `kv_cache_bf16` weight policy | BFP8 | 10,908,115,456 | 2,789,212,160 | 27,672,814,984 | 6,552,705,656 | 3 |
| `kv_cache_bf16` | BF16 | 10,908,115,456 | 5,578,424,320 | 30,462,027,144 | 3,763,493,496 | 1 |
| BFP8 LM-head candidates | BFP8 | 10,577,814,016 | 2,789,212,160 | 27,342,513,544 | 6,883,007,096 | 3 |
| canonical BFP8 MLP control | BFP8 | 15,532,335,616 | 2,789,212,160 | 32,297,035,144 | 1,928,485,496 | 1 |
| `canonical_accuracy_bfp8_hifi2_bf16commcache` | BF16 | 15,532,335,616 | 5,578,424,320 | 35,086,247,304 conservative | -860,726,664 conservative | runtime-proven batch 1 |

The canonical control changes all MLP weight placements from BFP4 to BFP8, so
it cannot reuse the baseline non-KV total. The Stage 05 payload establishes
43,352,064 bytes/layer/copy for TP-local BFP4 MLP weights. BFP8 and BFP4 tile
payloads use 1.0625 and 0.5625 bytes/value, respectively, so the BFP8 payload
is 81,887,232 bytes/layer/copy. Across 60 layers and the retained large-M plus
decode placements, canonical weights add 4,624,220,160 bytes/device. Combining
that increase with BF16 KV exceeds Stage 07's conservative accounting envelope
by 860,726,664 bytes/device at the advertised batch-1 context. The full Stage 08
run nevertheless constructed all 60 layers and the advertised-context cache,
executed 100 traced teacher-forcing tokens with 99 trace replays, and closed all
four devices normally. Thus the retained 12 GiB general allocator reserve is
not fully concurrent with this measured workload; the negative value is not a
hard physical limit. Runtime evidence proves at least batch 1 and 262,144 tokens,
so the context contract is not reduced. The canonical policy is rejected on
measured performance (17.800 t/s/u), not capacity.

The isolated `kv_cache_bf16` candidate retains baseline weights and still fits
the full advertised context at production batch 1. Its BF16 full-context batch
2 would account for 36,040,451,464 bytes/device and is physically short
1,814,930,824 bytes; this changes only the capacity upper bound, not the
advertised batch-1 contract.

The BFP8 LM-head rows replace the 704,643,072-byte/device BF16 projection with
the same tile value count at the BFP8-to-BF16 payload ratio `1.0625 / 2`, or
374,341,632 bytes/device. That saves 330,301,440 bytes/device and raises the
advertised-context margin to 6,883,007,096 bytes without changing the BFP8 KV
layout. If an LM-head row wins, these are the selected-config capacity values.

The selected Stage 08 row is copied into `../context_contract.json` after the
accuracy/performance winner is known. Candidate runtime JSON proves the actual
KV storage dtype used by each measured construction path.

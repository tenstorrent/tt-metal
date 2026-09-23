# How many KV slots fit

## Result

A slot costs **2176 bytes per token of capacity, per chip**, at bfloat8_b. That is the whole
divisor, and it is exact: measured DRAM deltas on a 4x8 Blackhole Galaxy matched the closed form to
the byte at three capacities.

| Capacity | Per slot, per chip (both caches) | Measured / predicted |
| --- | ---: | ---: |
| 8,192 | 17.0 MiB | 1.000 |
| 32,768 | 68.0 MiB | 1.000 |
| 131,072 | 272.0 MiB | 1.000 |

So the slot count follows from free DRAM and nothing else:

```
num_users = (free DRAM per chip - run reserve) / (2176 * max_seq_len)
```

`slot_bytes_per_chip` states the divisor, `max_user_slots` performs the division against DRAM that
is free at call time, and `allocate_kv_cache(num_users="max")` allocates the result.

## Where the numbers come from

Measured on `bh-glx-120-c01u08`, a 4x8 Blackhole Galaxy at the production shape (SP=4 x TP=8), 32
layers, real checkpoint, bfloat8_b caches, under Slurm jobs 109855, 109875, 109916, 109960 and
109976.

| Quantity | 8K capacity | 32K capacity |
| --- | ---: | ---: |
| DRAM per chip | 31.83 GiB | 31.83 GiB |
| Weights and model buffers | 2.73 GiB | 2.76 GiB |
| Free with weights resident | 29.10 GiB | 29.07 GiB |
| Slots `num_users="max"` allocates | 1,681 | 420 |
| Free left after allocating them | 1.19 GiB | 1.18 GiB |
| Whole prompt prefilled into the top slot | 8,192 tokens, slot 1,680 | 32,768 tokens, slot 419 |

The slot count is inverse-linear in capacity, as the divisor says: 4x the context buys a quarter of
the slots, and 1681/420 is 4.00.

### The 2K to 128K sweep

Every count below was allocated by `num_users="max"` and then used, under Slurm jobs 110427 (2K to
32K) and 110506 (64K, 128K): a whole prompt to capacity prefilled into the top slot, followed by a
second prompt in slot 0 while the top slot stayed full. One model build per capacity, weights
resident before the count is taken.

| Capacity | Slot cost per chip | Slots | Free left | Prefilled to capacity in |
| ---: | ---: | ---: | ---: | --- |
| 2,048 | 4.2 MiB | 6,727 | 1.18 GiB | slot 6,726 |
| 4,096 | 8.5 MiB | 3,363 | 1.19 GiB | slot 3,362 |
| 8,192 | 17.0 MiB | 1,681 | 1.19 GiB | slot 1,680 |
| 16,384 | 34.0 MiB | 840 | 1.20 GiB | slot 839 |
| 32,768 | 68.0 MiB | 420 | 1.18 GiB | slot 419 |
| 65,536 | 136.0 MiB | 209 | 1.28 GiB | slot 208 |
| 131,072 | 272.0 MiB | 104 | 1.35 GiB | slot 103 |

The count halves for every doubling of capacity: 6727, 3363, 1681, 840, 420, 209, 104. One row breaks
the pattern by a single slot -- 64K gives 209 where halving 420 would give 210 -- and it is the
contiguous run, not the byte count, that costs it: 209 slots leave 1.28 GiB/chip, so a 210th slot's
136 MiB would still clear the 1 GiB reserve, but not in one run per bank. 128K then halves 209
cleanly. What the count is *not* is a serving claim: it is how many slots fit in DRAM, and prefill
fills one at a time.

Startup does not vary with the count. Allocation took 157-167 s at every capacity, because what it
zeroes is the same ~28 GiB per chip either way.

Chunk times inside this probe are cold-path: it visits each chunk index once, so every chunk pays a
program build. The warm behaviour is the table below.

The arithmetic and the allocation are covered by `test_kv_cache.py`, which checks the divisor is
exact and that `num_users="max"` allocates what it promises and can be written across its range.
The rows above additionally prefilled each count through the real model, which is an acceptance
probe rather than a committed test: rerun it by building the model at a capacity, allocating with
`num_users="max"`, and prefilling a whole prompt into `num_users - 1`.

## Slots are free at run time

They cost DRAM and nothing else, which is worth stating with evidence because a packed batch
extent 840x larger is exactly the kind of thing that turns out to be walked rather than indexed.
Whole chunks through the real model at 8K, warm, median of three passes after a warmup pass:

| Slots | Cache per chip | Warm per-chunk | Throughput |
| ---: | ---: | ---: | ---: |
| 2 | 34 MiB | 151.8 ms | 6,746 tok/s |
| 1,681 | 27.9 GiB | 151.1 ms | 6,775 tok/s |

Ratio 1.00x. The per-op sweep agrees: from 2 to 1024 slots the cache write holds at 1.7 -> 1.6 ms
and the attention read at 3.3 -> 3.0 ms, measured at both slot 0 and the top slot, so neither op
walks the planes below the one it addresses.

A first, cold pass looks very different -- 13.7 s chunks at 1681 slots -- but that is the
per-chunk-index program build, not the slot count: the 2-slot run's cold pass has a 20.9 s chunk in
it. Anything timing this path has to warm the program cache first.

The one real cost is startup: zeroing 27.9 GiB per chip takes **157 s**, against 0.2 s for two
slots. A serving front end pays it once, but it belongs in a boot budget.

## Why the division needs two corrections

**Contiguity, not bytes, is what fails first.** Each cache is a single multi-GiB buffer that has to
land in one run per bank. Asking for 1,752 slots at 8K -- which the byte count allows -- failed with
1,953,371,520 B free per bank against 1,951,924,224 B needed, because the largest free block was
1,929,126,272 B. The bytes were there; the run was not. `max_user_slots` therefore divides the
largest contiguous block per bank rather than the free total.

**A cache that fits but cannot be used is not useful.** Activations, logits and per-op intermediates
come out of the same DRAM after the cache is allocated. That reserve was measured directly, by
holding the model fixed and allocating ballast until a chunk stopped completing: chunks still ran
with **0.45 GiB/chip free** at both 8K and 32K, the smallest figure probed, and never failed. A
chunk is a fixed 1024 tokens whatever the capacity, which is why the figure does not move with
`max_seq_len`.

The 1 GiB default reserve is therefore mostly placement margin rather than activation space.

## Allocation is device-side

`allocate_kv_cache` used to build the cache as a torch tensor and upload it. That tensor was fp32:
4 B/element against 1.0625 on device, so it wanted 44 GiB of host RAM for a slot count the device
holds in 21 GiB. The host, not DRAM, would have capped every auto-sized deployment. Zeroing on the
device removes that ceiling and the upload.

## Using it

```python
cache = allocate_kv_cache(mesh_device, mesh_config, num_users="max", max_seq_len=max_seq_len)
```

Call it with the weights already loaded: the count is a reading of the current allocator state, not
a property of the hardware, and the weights are the largest thing competing for it. For the adapter
path, which types `num_users` as an int in `PrefillRunParams`, resolve the number first with
`max_user_slots(mesh_device, max_seq_len=...)` and pass it through as usual.

Taking everything leaves ~1.2 GiB/chip free, which is enough to prefill but not enough for anything
else that may want DRAM later -- tracing, a wider head, a second resident model. `reserve_bytes`
raises the floor for those cases, and each GiB withheld costs 60 slots at 8K or 15 at 32K.

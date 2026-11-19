# Full-model DRAM arithmetic

The current source fits **one candidate request at context 262144**, with
235,069,952 bytes per rank remaining after the inherited 2 GiB runtime envelope.
At context 128, the configured 32-slot candidate leaves 2,411,022,848 bytes per
rank after that envelope. These are CPU planning results, not measured capacity
or a reduction of advertised capability. `doc/context_contract.json` is unchanged.

| Component | Bytes per rank |
|---|---:|
| Accepted decoder static tensors, including bank padding | 27,044,237,312 |
| Selected optimized phase-policy weight delta | 506,757,120 |
| Actual embedding, untied LM head and final mixer | 643,235,840 |
| **Full48 model static** | **28,194,230,272** |
| Known generator persistent buffers | 8,036,864 |
| Common sampler buffers allocated at construction | 87,612,928 |
| Persistent prefill output per slot | 3,981,312 |
| Shared index pair at context 262144 | 41,943,040 |
| Independent request state at context 262144 | 3,460,372,992 |
| **Known B1/262144 resident total** | **31,796,177,408** |
| Inherited observed allocatable DRAM reference | 34,178,731,008 |
| Headroom before runtime envelope | 2,382,553,600 |
| Headroom after separate 2 GiB envelope | 235,069,952 |

`capacity_plan.json` contains per-tensor ledgers and B1/B8/B16/B32 rows for
contexts 128 and 262144. The source-derived terminal allocation is 983,040 bytes
larger than the earlier unspecified final-mixer allowance. The token embedding
is hidden-dimension sharded (`[248320,640]` BF16 per rank); the separate LM head
is vocabulary sharded (`[2560,62080]` BF16 per rank). Final-mixer FP32 storage is
7,536,640 bytes per rank. No duplicate terminal weight copy is charged.

The preserved policy is EP4 BFP4 experts, the selected optimized dense/precise
projection groups, BF16 activations and CCL payloads with ordered FP32 reduction,
BF16 tiled paged K **and** V, BF16 compressed index keys, and FP32 GDN recurrence.
The accepted optimized-decoder README explicitly rejects BFP8 KV for latency;
the current decoder source has no selected K/V dtype split to extrapolate.
The residual remains one complete 2560-feature hyperconnection stream per rank.

The shared workspace and RoPE pair are charged once. Immutable block/token
index arrays are charged once per distinct configured capacity, across all
requests and attention layers. Mutable K/V, compression buffers, page tables,
positions, GDN/PLE state, and prepared hash inputs/outputs remain per request.
The common sampler allocates penalty history even when the default request has
zero penalties; its three local masks/counts and two full-vocabulary buffers are
included. The sampler's physical batch is 32 regardless of active slot count.
The generator also retains a tiled prefill-logit output for every configured
slot. When there is more than one slot, SlotStateMover allocates one additional
complete request cache for relocation; shared immutable indices remain shared.
Device seed state, updates, active flags and saved token slots are included.
These changes leave the arithmetic candidate maxima at B1/262144 and B32/128.

`shared_pool_profiles` separately plans the setup-time global KV pool. Every
unique BF16 K/V pair uses 32,768 bytes per 32-token block per rank; a logical
block across all twelve attention layers uses 393,216 bytes. Reusing one pair
across layers requires disjoint physical blocks and does not reduce this rate.
The scratch slot aliases the global pool and retains only its private non-KV
state. At physical context C rounded up to 128, each private non-KV cache uses
`34,675,200 + ceil(C/256)*196,608 + ceil(C/512)*6,144` bytes. Shared index arrays
use `(ceil(C/1024) + ceil(C/256))*32,768` bytes.

Subtract known static bytes, shared indices, B private caches, one additional
private scratch cache when B>1, B prefill outputs, and the runtime allowance
from allocatable DRAM. Divide the remainder by 393,216, round down to blocks,
and multiply by 32 to obtain a planning total-token bound, capped by B contexts.
With the unmeasured 2 GiB allowance, C262144 gives B1:262144 pooled tokens and
B8:123296 pooled tokens. B16 and B32 cannot fit even their private state under
that allowance. C128 gives the full B1/B8/B16/B32 context budgets.

Use the explicit `defer_kv_cache=True` setup option to realize these pooled
figures without private K/V construction. The default private-cache constructor
still incurs its original B-plus-scratch peak before any later replacement.
Deferred construction rejects execution until the allocator binds a pool. None
of these arithmetic bounds changes the advertised context or constitutes
measured serving readiness.

The runtime-only `tt/cache_budget.py` reproduces these byte formulas and exposes
all inputs with `CacheBudget.to_dict()`. Its pinned-plugin budget reserves the
worker's additional output page per slot and one null block, and rejects a
configuration if the pool cannot admit one complete configured context. This
configuration guard does not silently reduce the advertised context. Fourteen
CPU tests, including exact allocation-boundary and independent-ledger checks,
passed; see `cache_budget_tests.log`.

The arithmetic uses the inherited 8-bank, 64-byte DRAM page rules. For tiled
arrays, each dimension's tile padding and tile byte size are included. A
single-page row-major array reserves a page in each bank; merely multiplying
its logical element count is insufficient. The script recomputes every state
entry for each context instead of scaling a maximum-context total linearly.

Reproduce without TTNN or device access:

```bash
python models/autoports/qwen_qwen3_8_flash_next/doc/full_model/capacity_arithmetic.py
```

The remaining physical measurements must include trace reservations and live
captured outputs, every warmed program-cache signature, optional sampler/logprob
buffers created during preparation, fabric/CCL resources, transient activations
and scratch, allocator fragmentation, and construction overlap. The existing
1 GiB and 2 GiB envelopes are retained as unmeasured planning allowances. They
are not evidence that those runtime paths fit. In particular, B1 maximum context
has only about 224 MiB beyond the 2 GiB allowance, so keep prompt processing
bounded to internal chunks and measure the real allocation high-water mark.

## Runtime memory view

The binding in `ttnn/cpp/ttnn-nanobind/device.cpp:201` exposes scalar fields:
`num_banks`, `total_bytes_per_bank`, `total_bytes_allocated_per_bank`,
`total_bytes_free_per_bank`, and `largest_contiguous_bytes_free_per_bank`.
`block_table` is a list of string-to-string dictionaries. The implementation in
`tt_metal/detail/reports/memory_reporter.cpp:172` forwards allocator statistics;
these byte values are per bank, not lists and not per-device totals.

The device-owning probe can serialize each configured mesh/device view:

```python
fields = (
    "num_banks", "total_bytes_per_bank", "total_bytes_allocated_per_bank",
    "total_bytes_free_per_bank", "largest_contiguous_bytes_free_per_bank",
)
view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
record = {name: int(getattr(view, name)) for name in fields}
record["block_table"] = [dict(row) for row in view.block_table]
record["total_bytes"] = record["num_banks"] * record["total_bytes_per_bank"]
record["allocated_bytes"] = record["num_banks"] * record["total_bytes_allocated_per_bank"]
record["free_bytes"] = record["num_banks"] * record["total_bytes_free_per_bank"]
```

Collect views after mesh open, after full48 weights/terminal load before cache,
after request/cache creation, after all preparation, and after capture/replay.
Also record `ttnn.BufferType.TRACE`: a configured trace pool is separate from
allocatable DRAM in the memory view, so do not subtract its measured reservation
twice. Derive the KV budget from measured free DRAM after subtracting the
remaining known request state, sampler/generator state, and measured runtime
needs. Check largest contiguous bytes per bank as well as aggregate free bytes.
Only the real full-model acceptance run can establish maximum context; a physical
failure calls for one bracketing pass/fail pair, not a context sweep.

# Shared global KV pool contract

`tt/cache_pool.py` supplies CPU validation and lazy TTNN allocation/binding. It
does not by itself close the serving allocator row. No vLLM adapter is added.
The pinned plugin is `042a94863ad27d88207bcf71abdf67122c6f4c27`; its
`src/vllm_tt_plugin/model_runner.py::_kv_cache_shape` already folds TP, and
`_build_per_layer_specs` returns `(shape,dtype,tensor_idx)` in attention-layer
order. For the accepted TP4 decoder the physical shape is
`[global_blocks,1,32,256]`, **K and V both BF16 tiled DRAM**. The earlier BFP8
paged-cache candidate was rejected. GDN recurrence remains FP32; compressed
QSA keys remain BF16. No activation, fidelity, collective or residual layout
policy changes are introduced.

Each unique tensor index owns one K/V pair. Repeated tensor indices retain
actual aliases: the plugin permits different attention groups to share one
allocation because their global block IDs are disjoint. Such layers require
different per-layer page tables. Replacing aliases with duplicate buffers, or
using one common table across aliased layers, would change the contract.
Heterogeneous shapes for one tensor index are rejected before allocation; this
model's indexed-attention layers have uniform geometry.

Every request still owns its convolution/recurrent state, PLE convolution and
token history, QSA raw and compressed indexer history, current position, and
logical page-table buffer. `state.capacity` remains the absolute request
context; it is not the global pool size. The current custom gather independently
compiles physical block count and logical page-table width. Its wrapper creates
`LocalSelectedPagedKV` from current state tensors on each call; there is no
pre-bind stored gather object to repair. Binding after capture is forbidden.

## Integration API

The generator now implements `allocate_kv_cache_per_layer` and the uniform
`allocate_kv_cache` wrapper. Construct it with the internal setup argument
`defer_kv_cache=True`, call the allocator before preparation, and pass its opaque
`SharedKVPool` result as `kv_cache` to warmup/prefill/decode. No operator config
or capability switch is required. The default readiness constructor retains
its existing private allocation path. `page_tables_per_layer` follows the pinned
runner: per-layer inputs override group0 and may include padded runner rows;
rows describe submission order before scattering into fixed slots.

Deferred setup allocates no private K/V in `MultichipDecoder.create_state`;
all other state and decoder policies remain unchanged. Model and generator
reject unbound execution. For existing owned private caches, the allocator
validates the complete plan then frees private K/V before global allocation.
Failure after this transition permanently invalidates the instance and requires
reconstruction; no fallback or partial old-cache reuse is possible. Borrowed
external private storage cannot be silently freed.

The lower-level helper sequence is:

```python
pool = SharedKVPool(per_layer_specs, mesh)
displaced = pool.bind([*generator.caches, scratch])
# Deallocate displaced private K/V only after their old users are gone.
tables = pool.warmup_tables(batch_size=B, capacity=C)
# Upload tables[attention_index][slot:slot+1] to each persistent state table.
# Rebuild SlotStateMover with shared_kv=True caches/scratch before its warmup.
pool.freeze()
generator.prepare()
```

`bind` validates all caches before replacing any field, marks every cache and
scratch `shared_kv=True`, and attaches `shared_kv_pool` as a lifetime owner.
It preserves all non-KV objects. `pair_for_attention_layer(i)` enables direct
state construction with the real pools; that is the preferred full-capacity
setup to avoid temporarily retaining both old private and new shared K/V.
The compatibility binder returns old allocations for explicit disposal and
does not solve their transient peak by itself.

`normalize_tables(batch_size,capacity,page_table=...)` accepts one common table;
alternatively `page_tables_by_layer=...` supplies one per attention layer.
It rejects noninteger/out-of-pool IDs before writes and pads to physical
`ceil128(capacity)/32` width. Optional `lengths` describes valid tokens including
an about-to-be-written token, ignores scheduler padding, and verifies that live
pages do not overlap within each shared pool. A **complete live-slot snapshot**
is necessary to detect overlap with dormant requests. Lengths describe retained
history, so a dormant request with cached tokens must not be represented by zero.
These are scheduler transition checks, not per-token device readbacks.

Current `prepare` leaves slot0 at position32 before its cumulative width sweep;
`warmup_tables` reserves two pages per slot and shared group, enough for up to32
decode widths. It requires enough global blocks for these temporary reservations.
Canonical prefill executes attention through scratch after copying the current
request's private state and page tables. Scratch borrows that request's pages;
it has no independent simultaneous request ownership and reserves no extra
blocks. Scheduler tables must replace temporary warmup maps before real
requests. This is setup scratch space,
not a reason to reduce the advertised context; a tighter preparation sequence
could reset between warmup widths if a very small pool needs only one page/slot.

## Integrated lifecycle behavior (reduced proof; acceptance incomplete)

1. Bind before any decoder preparation, capture, or serving request. Cache
   ownership passed through warmup and forward must resolve to these fixed slots.
2. Slot migration copies all per-request tensors including page tables and
   indexer history, but never shared K/V. Scratch aliases the pool and is excluded
   from K/V copying. Device seed and token-feedback migrations remain separate.
3. Request reset clears recurrent/PLE/indexer/position state but never clears
   shared K/V or overwrites scheduler page mappings with identity. A new causal
   prefix writes its valid positions before reading them; no stale future token
   is eligible for attention. Page reuse requires scheduler ordering after prior
   work drains. KV-only prefix caching remains excluded for `recurrent_state`.
4. `_bind` must accept per-layer table inputs, retain per-slot/per-layer shadows,
   validate the complete update before any copy, and copy only changed tables.
   `tensor_idx` sharing requires this API; a single common table is insufficient.
5. Keep the pool alive until all traces are released. Do not deallocate or replace
   its tensor objects while captures reference them. Request release does not
   free global pool storage.

## Evidence and pending device proof

CPU command, passed **9 tests**:

```bash
python_env/bin/python -m unittest models.autoports.qwen_qwen3_8_flash_next.tests.test_cache_pool_cpu -v
```

The tests cover atomic rejection, dtype/shape/tensor-index validation, real alias
ownership, unmodified request-state objects, global IDs beyond private capacity,
padding/nonaligned context, disjoint shared-group IDs, and warmup reservations.

`tests/probe_shared_kv_pool.py` is prepared but **not run by its CPU-only author**.
Its docstring has the serialized timeout180 invocation. It uses the actual custom
gather, slot mover, and generator reset on a 1x4 mesh, with allocation tracking.
It checks global ID15 in a16-block pool despite a4-page logical table, unchanged
replay, slot remapping, reset isolation, changed ID13, and stable tensor addresses.
It does not exercise paged-update writes or model logits. A reduced generator
test using two disjoint global maps, real writes, remap, continued decode, and
one-slot release must still pass before allocator/ownership rows are supported.

Generator integration CPU command passed **49 tests plus62 subtests** (log `doc/full_model/generator_pool_cpu.log`):

```bash
python_env/bin/python -m pytest --noconftest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_generator_pool_cpu.py models/autoports/qwen_qwen3_8_flash_next/tests/test_decode_validation_cpu.py models/autoports/qwen_qwen3_8_flash_next/tests/test_prefill_validation_cpu.py models/autoports/qwen_qwen3_8_flash_next/tests/test_cache_pool_cpu.py models/autoports/qwen_qwen3_8_flash_next/tests/test_cache_budget.py
```

`tests/probe_shared_pool_generator.py` exercises real
layers0/1/3 with deferred allocation, real paged writes, mixed versus solo logits
and state, global IDs above private capacity, remap, release/reuse, dormant-page
alias rejection before mutation, and unchanged-table replay. Its docstring has
the serialized timeout1800 invocation. The v3 run supplies the partial evidence
below; the complete lifecycle probe must pass before marking integration
supported. `model_capabilities.fabric_config` now declares the accepted ring
configuration consumed by pinned `worker.set_fabric`; mesh ownership remains
with the caller.

### Decode geometry and exact cache checks

`doc/full_model/shared_pool_generator_v3.json` completed device teardown in
154.08 seconds. All **184 pooled-versus-private B2 comparisons were exact**:
mixed-prefill and first-decode logits, private recurrent/conv/PLE/indexer state,
positions, and logically addressed K/V across all four ranks. The same loaded
model served both generators; private traces and generator/cache owners were
released before pooled setup. Mixed-prefill versus solo and repeated solo
prefill/decode/state were also exact. This is evidence for those operations,
not a pass for the still-unreached migration/release/reuse tests.

The only failing v3 gate compared B2 decode against B1 byte for byte. That
comparison crosses the accepted `MeshMoe._reduce` policy: `n == 1` uses the
feature-parallel matmul with FP32 accumulation, whereas `n > 1` uses native
grouped BF16 `post_combine_reduce`. Both return BF16 outputs. Different rounding
after the first MoE consequently changes downstream recurrence without implying
a cache ownership failure. The two v3 row PCCs were 0.9999729249 and 0.9982841970;
maximum absolute differences were 0.0234375 and 0.208984375. Both retained the
same top-1 and top-5 membership. Full-stack/HF acceptance remains independent.

The corrected probe keeps exact mixed-prefill and repeated-solo gates. For
solo-versus-batched decode logits it uses the existing, unchanged
`tests/probe_full_model_batch.py` limits: finite values, whole-output PCC >=0.995,
minimum-row PCC >=0.99, top-1 agreement =1, top-5 agreement >=0.98, and top-100
agreement =1. It retains every cross-geometry cache difference as a diagnostic;
no tolerance is applied to a cache invariant.

The private baseline now follows the complete same-geometry sequence: B2 decode,
B2 permutation/decode, release, B1 survivor continuation, slot reuse, and B1
unchanged-table replay. Every corresponding pooled logit and logical cache state
must equal that private control exactly. Physical page IDs alone differ by
design (private IDs0..3 versus pool IDs8..15); each snapshot follows its own page
table to compare causal K/V contents. Pool-wide no-write-on-release, dormant-page
alias rejection, stable addresses, and unchanged-table upload checks remain
strict. Actual pooled page tables are also checked exactly against the scheduler
mapping after every lifecycle step, independently of the private comparison.
These extended lifecycle controls ran in v5; the scoped results and remaining numerical failure are recorded below.

`probe_full_model_serving.py` uses the same frozen cross-geometry logit gate.
Its dormant-feedback control now reproduces the initial B2 then B1 execution
history, and seeded B2 row-reordering compares against another B2 run. Exact
tokens, state, inactive rows, migration, seed state, output snapshots and
lifecycle invariants are retained; B1-only controls remain for B1-only histories.

CPU validation of this gate correction: both probes compile and format cleanly;
all three `test_shared_pool_diagnostics_cpu` tests pass. Five synthetic cases
matched the metrics from the AST-extracted original batch `comparison` function:
exact values and a small non-top-logit perturbation pass; a changed top-1, row
corruption, and nonfinite logits are rejected. No device run was made for these
test changes, and no runtime model, generator, dtype or compute policy was edited.

`get_max_tokens_all_users` now delegates to `tt/cache_budget.py`, using the
source-derived pooled-state budget, one null page, the pinned worker's one
output page per fixed slot, and an explicitly unmeasured2GiB runtime allowance.
It validates TP4/no lane DP and rejects a plan unable to admit even one complete
configured context; it does not return a silently shortened context. These
planning constants still require memory-high-water and max-context acceptance.
The plugin has no model-capability key for trace-region bytes: its documented
existing worker resource configuration must provide the accepted trace/L1
reservation; no invented model resource key is claimed to be consumed.


### V5 independent migration and full reduced lifecycle evidence

`shared_pool_generator_v5.json/log` completes the reduced lifecycle and normal
device teardown. The JSON elapsed time is162.36s (bounded process about172s).
The overall result remains **failed** on the unchanged cross-geometry decode
accuracy gates. It contains1734 output/state comparisons; the only nonexact
comparison families are solo B1 versus histories containing B2 decode.

An independent reference consumes forced tokens42/43/44 at B2 without ever
calling migration. The migrated B2 run equals the appropriately flipped
uninterrupted outputs and every cache field exactly at all three steps.
Reset/rebuilt private controls are exact. The pooled migrated43 output/logical
state also equals the flipped uninterrupted reference. This isolates migration
from the earlier control where both paths migrated: this tested permutation is
correct in private and shared storage. No sampled-token/seed combination is
claimed by this logits-only probe.

All same-geometry private-versus-pool lifecycle checks pass exactly: mixed
prefill, initial decode, migration, release, survivor continuation, slot reuse,
and unchanged-table replay. Global page IDs remain exact, releasing a slot
preserves all shared K/V bytes and the survivor, dormant aliases reject before
mutation, and pool addresses remain stable. These are reduced layers0/1/3 with
one actual QSA pool, not full48/12-pool proof. Full-model and combined sampler
serving acceptance remain separate.

The first42 B1/B2 comparison passes the frozen gate (whole PCC0.9989975,
minimum row0.9982631, top1/5/100 all1). Token43 fails: whole PCC0.9796497,
minimum row0.9702018, top1/top5 agreement0.5 and top1001. The44 survivor also
fails whole PCC0.9942773 despite top1/5/100 all1. These are real numerical
failures to repair, not accepted rounding drift or grounds to lower thresholds.
The exact migration controls rule out the tested state permutation as their
cause; capture real MoE inputs and pre-reduction tensors next to distinguish
upstream expert/routing divergence from the accepted B1/B2 reduction paths.

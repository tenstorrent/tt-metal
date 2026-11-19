# Traced request seed progression

`tt/device_seed.py` implements seed-state progression around the existing
common sampler; it does not implement a sampling algorithm. This CPU audit
does not establish hardware correctness or close serving acceptance gates.

## Exact source contract

The current `models/common/sampling/generator.py` derives request seeds with
`_hash_request_seed_to_device_seed(seed,counter,salt)`: modulo64 request seed
XOR modulo64(counter + golden-ratio constant), SplitMix64 finalization, then
`value % 1_000_000 + 1`. The new helper reproduces that function exactly with
salt0. Arbitrary Python seeds follow the common implementation's modulo64
semantics; it does not silently truncate to32 bits. Distinct integers congruent
modulo2^64 intentionally alias, just as in the common implementation.

Salt0 is intentional: the pinned plugin's
`tests/tt/test_seeding_and_variety.py::test_uniform_seed_deterministic` and
`test_same_seeds_reproduce_across_batches` expect the same seed to reproduce
across request rows. Common SeedManager's default concurrent duplicate-seed
salts would deliberately distinguish these requests. The helper has no
row/core/request-arrival salt, so batch reshuffling cannot alter the stream.

`models/common/sampling/tt_sampling.py` calls `ttnn.manual_seed` immediately
before `ttnn.sampling`, after its SFPU-sensitive tie-breaking. The seed tensor
is uint32 ROW_MAJOR `[32]`; user IDs0..31 route seeds to the sampler cores.
`manual_seed/.../manual_seed_receive_all_data.cpp` leaves hardware PRNG state
unchanged when seed is `UINT32_MAX`, otherwise calls `rand_tile_init(seed)`.
`api/compute/eltwise_unary/rand.h` adds no stream/core salt with its default
stream_id0. The stock sampling kernel generates its threshold from hardware
PRNG output; this helper never computes probabilities or selected tokens.

The helper retains the common bounded positive seed distribution. SplitMix64
is a deterministic, well-mixed noncryptographic mapping; modulo1e6 introduces
at most one 64-bit input count difference between seed bins for a uniform
64-bit input. Different request seeds are not promised distinct every-token
device seeds: collisions are expected in a bounded seed space. CPU exactness
checks prove arithmetic equivalence, not statistical quality of hardware
sampling. That quality remains the stock sampler's responsibility.

## Device storage and operations

All persistent buffers are replicated on the existing TP4 mesh, uint32
ROW_MAJOR interleaved DRAM:

- `state[32,16]`: seed low/high, counter low/high, mode, reserved words.
- `updates[32,16]`: transition-only reset flag and five new state words.
- `active[32]`: scheduler membership; updates only when slots change.
- `output[32]`: borrowed existing common-sampler seed tensor.

A prepared generic dataflow kernel runs on core(0,0) per device and uses4,224
bytes of local scratch. It reads 32 aligned64-byte state rows, reads one
128-byte active mask, writes updated state rows and one128-byte seed output.
Only active seeded rows increment. Mode2 supplies an entropy initialization
once for an active unseeded row, then changes to mode0/SKIP. Inactive rows
always supply SKIP and retain state. No sampled token, logit or counter is read
by the host. Arithmetic uses explicit uint32 limbs, shifts, XOR, multiplication
and carry; no float roundtrip, int64 TTNN arithmetic, RISC-V int64 division,
or unaligned four-byte NoC writes are required.

Transition reset runs a separately prepared kernel mode. It updates only
flagged rows, preserving counters that the host may not know for other live
requests. `reset_slots` receives values in reset-slot order. `remap` gathers
all seed/counter/mode rows on device, including off-batch requests; it retains
all allocation addresses. Identity-padded scheduler remaps cover physical32.
The activity buffer is scheduler control, supplied again for the new layout.

## Integration

1. Construct `DeviceSeedState(mesh,sampler.tt_sampling.seeds_tt_tensor)` at
   setup and call `prepare()` before capture. Warm identity
   `remap(range(max_batch_size),warmup=True)` before capture if slot migration
   will be used. Apply an initial reset and active-slot selection.
2. Choose an explicit counter convention. Recommended: next token's absolute
   position. Prefill ending at exclusive positionP predicts tokenP, so reset
   its counter toP. A decode consuming input at positionp predicts tokenp+1;
   a reset before that forward uses counterp+1. Thereafter `forward()` advances
   only device state. Resumed-prefill reset uses its new exclusive prompt end.
   Never recompute this counter from stale host positions when reload_inputs
   is false. The alternative generated-token ordinal must be used consistently
   across all entrypoints if chosen instead.
3. In the sampling trace call `seed_state.forward()` before the common
   `_run_sampling(...,tt_out_tok=decode_tokens)`. Stock manual_seed remains
   immediately before stock sampling. The existing post-draw SKIP copy is
   harmless, though redundant because the next seed helper forward overwrites
   all32 seed values.
4. `set_active_slots` runs at scheduler boundaries; an unchanged tuple performs
   no host upload. Reset seed state on explicit reset_sampling_state even for
   seed=None. Repeated resident decode performs neither reset nor host copies.
5. On a remap, call `seed_state.remap` once in both host/device sampling modes,
   alongside model and token-feedback migration. The hardware unseeded RNG
   remains non-slot-addressable; the plugin contract exempts it from gather
   and requires reinitialization on commanded sampling-state transitions.
6. This helper becomes the sole owner of device seed writes/advancement.
   Do not additionally call common SeedManager get_new_values/reset operations
   to write the same seed tensor. Common sampler parameter/penalty invalidation
   may still be enforced, but its host seed arrays are no longer authoritative
   for device progression. State reset and parameter reload are distinct;
   neither may silently rebuild the other's state.
7. Standalone host compatibility remains request-local HostSampler; plugin
   host mode receives raw logits and keeps its own RNG. No seed policy config
   option or invented capability is required for the new device route.

Warmup execution that advances seed state must be followed by an authoritative
reset before measured generation. Trace capture itself must follow the repo
capture semantics; do not assume capturing a seed op preserves state. The
helper's `prepare()` compiles without executing, matching DeviceNGramHash.

## CPU validation and remaining device proof

Command:

```text
timeout 60 env PYTHONPATH=/home/ttuser/qwen38-flash/tt-metal python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.test_device_seed_cpu
```

Seven tests passed. Tests compile the exact device integer source as C++ and
compare10,024 seed/counter pairs against the current common function extracted
through Python AST without importing TTNN. Coverage includes negative/large
Python seeds, high32-bit sensitivity, uint64 counter carry/wrap, inactive
rows, one-shot unseeded initialization, repeated same seeds, slot migration,
and paused requests. Evidence: `doc/full_model/device_seed_cpu.log`.

Required focused device proofs before acceptance:

- Kernel compiles on the current architecture and TensorAccessor page sizes
  match allocation pages, including the common output seed tensor.
- Eager and replayed seed outputs/counters equal the CPU formula on all4 ranks,
  including counter carry and reset of one slot without changing others.
- Traced common-sampler draws reproduce after seed reset and across different
  physical rows, with different seeds producing variety. No guarantee of
  bit-identical HF/CPU sampled tokens is implied by a shared numerical seed.
- Two/three-slot remaps preserve next seed/draw; inactive rows do not advance;
  authoritative host/device transitions restore counters coherently.
- Trace allocation tracking finds no unsafe state buffer lifetime, and host
  copy counters remain constant over steady repeated decode.
- Measure the extra single-core kernel cost. Use the helper only when seeded
  state is required if adding it to unseeded greedy traces costs materially;
  unseeded greedy can retain the proven existing seed path. Switching must
  remain an explicit traced-key/transition choice, not a per-token fallback.

The initial helper is not hardware-validated. Failures in these probes remain
stage work, not a reason to report auto-seeded serving support prematurely.

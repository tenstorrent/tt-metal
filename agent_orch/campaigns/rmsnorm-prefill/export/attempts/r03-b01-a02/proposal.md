# r03-b01-a02: land the all-gathered stat sticks in L1, not DRAM: the persistent stats scratch becomes an L1-interleaved mesh buffer, so the fabric writes and the worker's post-go stick read are L1 transactions

## Motivation
After the AG, every worker sits on a fixed DRAM round trip before compute can start the combine:
- r02-b02-a04 (`go.py`, all 4 chips, medians): W_DRAIN start - go = **0.65-0.72 µs on every shape**. That window is the
  CB reserve + 8 x 64 B reads of the worker's ring_size gathered sticks from the DRAM scratch + barrier + push. Its
  pre-staging of the addresses/reserve saved nothing, so the cost is the DRAM read round trip itself.
- That reflection's #3: "The stick read after go only goes away if the gathered data lands in L1. That needs a
  persistent, mesh-coherent L1 buffer". Nobody tried it.
- The fabric side also writes to DRAM: each forwarder's packet goes to a DRAM page on every chip with flush=true
  before the fused out_ready inc, so the AG completion (F_FABRIC end) includes a DRAM write commit too.
- This round (r03-b01..b04) all cut the same post-AG combine SFPU cost (~0.9 µs, now ~0.38-0.43 µs left). The stick
  read is the next-largest fixed post-AG cost on this path (AG end -> stick read -> combine -> POST -> drain), and the
  drain end follows POST start 1:1 (r03-b01-a01 reflection), so a fixed saving here should reach the kernel end.

## Mechanism
The scratch is the caller-owned persistent tensor built from `make_stats_tensor_spec` (the test allocates it through
`dit_fused_distributed_rmsnorm_create_stats_buffer`, which uses the same spec). Change the spec's buffer type from
DRAM to **L1 INTERLEAVED** (`device/dit_fused_distributed_rmsnorm_program_factory.cpp`).
- `create_device_tensor` makes it a mesh buffer with the same L1 address on every chip, and the allocator reserves
  that region on every core for the tensor's lifetime, so it is as persistent and as cross-chip safe as the DRAM
  version (same ping-ponged semaphores, same protocol).
- The 4 pages (4352 B each) land in L1 banks of 4 Tensix cores. Kernels need no change: the forwarder resolves the page
  NoC address through the same TensorAccessor (`addrgen_detail::get_noc_address`), and the worker reads with
  `noc.async_read(stats_accessor, ..., {.page_id, .offset_bytes})`, both generic over DRAM/L1 interleaved.
- `validate_on_program_cache_miss` already compares the caller buffer's buffer type with the spec; only its error
  text mentions DRAM, which I update. Comments that say "DRAM scratch" in the worker writer get updated.
Host change (rebuild); kernels and CBs unchanged. L1 cost: 4352 B at the top of each core's L1, well inside the
factory's 110 KB L1 margin.

## Why this is not a repeat
- r02-b02-a04 changed the go *release* (multicast) and pre-staged the read addresses: neutral, and it measured that
  the remaining cost is the DRAM round trip. This removes the DRAM from the round trip. It is its reflection's #3,
  which it called a larger change; it turns out to be a spec change because the scratch is a caller-owned tensor.
- All other nodes changed the read, PRE, POST/combine, the drain NoCs or placement. None changed where the AG data
  lands.

## Expected effect and risk
- Stick read after go: ~0.67 µs -> ~0.2-0.3 µs (L1-to-L1 NoC read). Possibly a bit off F_FABRIC (L1 write commit
  instead of DRAM). Fixed saving of ~0.3-0.5 µs per shape, so ~+2-3% (more on h3584/h4096). Accuracy bit-identical.
- Risks: (a) if an L1 bank->core mapping differed between chips the fabric would write the wrong core -> hang or
  accuracy_fail (BH uses translated coords and identical grids, so it shouldn't); (b) a CB/L1-buffer clash at program
  creation -> runtime_error; (c) the bank cores may be worker cores whose NoC ports are busy with the drain/reads of
  20 workers at once -> smaller gain. Judge with W_AGWAIT end -> W_DRAIN start per worker (was 0.65-0.72 µs) and
  F_FABRIC duration in the report.

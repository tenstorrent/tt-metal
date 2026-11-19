# AutoFix: initialized attention page-table tails

The failing contract was B32 with independent 4095/4096/4097-token prefixes,
17/18-token continuations, 128 evolving traced decode steps, allocation tracking
including program caches, and original PCC gates. Capacity is4256 tokens,133
32-token pages per request. It failed at decode0/request8. Saved output tensors
were identical across ranks but finite values reached4.8e34; float32 correlation
overflowed. The rank-equality assertion was a misleading diagnostic. The output
was still genuinely wrong: independent float64 reference PCC0.133719.

AutoDebug source review and a separate isolated page-reader check established
that SDPA with k_chunk_size256 reads136 page IDs at position4114. Three tail IDs
were uninitialized. Source references and exact historical hashes are in
[AUTODEBUG.md](AUTODEBUG.md). All-reduce input slots and reduction order are fixed
across ranks; ordinary BF16 accumulation order is not supported as the cause.

The minimal repair in `tt/multichip_decoder.py` builds a separate attention table
padded to eight page entries, initialized with valid local page zero before pool
offsets are added. The cache-update table remains separate because its width must
not exceed physical cache capacity. Both single and batched decode own padding.
Prefill's padded tail now points into the same request's pool region. Public input
lengths, page maps, context capacity, payload dtypes and PCC thresholds are unchanged.

Validation after the repair:

- `pool_contracts_padded_3.json`: full attention, contexts96/160/128, full/subset/
  reversed/single/mixed pools, output/contribution/state PCC, bitwise untouched
  neighbors, interleaved captures and input refresh;1788 checks, minimumPCC0.99999988.
- `integrated_long_stack_padded_32.json`: same failing fixture,4416 checks,
  128 replays, minimumPCC0.998714447. All rank checks pass.
- `integrated_padded_stack_32.json`:16-step state/benchmark check, minimumPCC0.999560058;
  1.997307ms with per-request table padding.
- `long_stack_stage4.json`: frozen serial/private control passes, minimumPCC0.998722792.
- `table_offsets_long_32.json`: subsequent batched table-construction candidate
  preserves the full long stress pass. Its16-step timing is1.654017ms in
  `table_offsets_32.json`; integration/final-source gates remain separate work.

Commands and exact source snapshots are in the matching `_command.json` files and
job lists. No precision option or weaker gate was needed. A diagnostic FP32-CCL
candidate was prepared but not used to repair this bug; the queue stopped before
running it once the page-tail cause was established. The exact bogus page IDs and
first corrupted arithmetic intermediate were not captured; source violation plus
the same-fixture repair contrast supports this fix without claiming that detail.

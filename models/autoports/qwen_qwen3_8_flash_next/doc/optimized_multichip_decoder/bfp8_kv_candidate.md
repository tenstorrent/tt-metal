# Non-default BFP8 local KV candidate

The candidate changes only QSA K/V storage to BFLOAT8_B. The cache shape remains
`[capacity/32, 1, 32, 256]` per rank, including the existing adjacent-rank KV-head
replication. Page tables, selected IDs, indexer compressed keys, raw history,
RoPE, BF16 Q/K/V update inputs, and BF16 SDPA inputs retain their contracts.
Logical query counts 1–32, including unaligned tails, remain valid.

`tt/bfp8_selected_paged_kv.py` and
`tt/kernels/local_selected_paged_kv_bfp8.cpp` implement selected-row unpacking.
A BFP8 tile contains 64 exponent bytes and 1024 sign/mantissa bytes. The reader
fetches aligned 64-byte exponent and mantissa lines only for selected rows. It
caches repeated lines within each 32-slot output group and normalizes each
7-bit mantissa into the corresponding BF16 encoding. Invalid selected IDs
produce zero output, as in the BF16 reader. The unchanged 40-KiB per-core scratch
reservation covers all accesses (last used byte 39999). There is no whole-cache
conversion on the forward or trace path.

The opt-in wrapper `tests/multichip_bfp8_kv_candidate.py` composes with the
parent's normal candidate harness. It changes newly created zero caches during
setup and replaces only the multichip QSA gather. The implementation currently
casts the initial zero cache once during setup; this is an experimental
allocation path, with a transient BF16 cache. A retained policy should allocate
BFP8 zeros directly after validating that constructor on the target mesh.
Neither default decoder code nor the parent's candidate-policy file is changed.

The existing mesh prefill uses sequential `paged_update_cache` for each logical
row, not `paged_fill_cache`. Its BF16 input is therefore preserved in prefill
and decode. The native update validator explicitly supports BFLOAT8_B storage.
If a future prefill implementation uses `paged_fill_cache`, its fill input must
be cast to BFP8 first; changing only the cache would violate that API contract.

CPU evidence: `bfp8_kv_cpu.json` covers 63,232 scalar sign/mantissa/exponent
combinations against independent `ldexp` plus IEEE encoding, every tile element,
aligned reads within the 1088-byte tile, and scratch bounds. This is not device
or precision acceptance evidence. No TTNN import, device open, or hardware
operation was performed by the CPU implementation subtask.

Parent-owned serialized runs:

```bash
python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_bfp8_selected_paged_kv \
  --queries 1,2,31,32 --replays 6 --output <component.json>

python -m models.autoports.qwen_qwen3_8_flash_next.tests.multichip_bfp8_kv_candidate \
  --candidate <same-best-topology-policy> <same-arguments-as-normal-probe>
```

The component run requires trace allocation tracking, uses all four ranks,
reversed/different rank-local page permutations, changed selected IDs, sentinel
slots, and all tile-row boundary cases. It compares exactly with native BFP8
unpacking to isolate reader/addressing bugs from quantization. Follow it with
same-input real-weight layer-3 short-prefix/B32 controls, traced replay and
cache-consuming continuation. Synthetic component values cannot reject BFP8
precision. A numerical collapse needs intermediate debugging, not a dtype veto.

Physical K/V bytes per token per rank change from 1024 to 544, excluding common
metadata and unchanged selected-output BF16 buffers. At capacity 262144 this
saves 125829120 bytes per QSA layer per rank. The final default KV contract is
unchanged unless parent measurements accept this policy. If accepted, update
`doc/context_contract.json` and perform the required max-context validation;
if rejected, record measured whole-layer latency/PCC and retain the BF16 contract.

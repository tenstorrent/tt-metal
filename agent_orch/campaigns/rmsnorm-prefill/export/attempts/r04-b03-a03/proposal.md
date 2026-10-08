# r04-b03-a03: de-phase DRAM banks per worker on the 20-core pull path: each worker walks its row's columns from a per-core rotation (CB slot k <-> column (k+rot) mod W, chosen so worker phase = tile_row % 8) for the input read, the BRISC gamma read and the output drain

## Motivation
- For h4096 / h6144 / h7168 the local width W is 32 / 48 / 56 tiles, all 0 mod 8. Page = row*W + c, so bank = c % 8
  on **every** worker: the 20 NCRISC input readers and the 20 BRISC output drains walk the 8 DRAM banks in lockstep.
  With block_size 4, every worker's drain block targets the same 4 banks, i.e. one DRAM column (banks 0-3 at x=0,
  4-7 at x=9) at a time. h3584 (W=28) only alternates two phases (rows offset by 4 banks).
- r01-b03-a03 (column-split lineage, 80 cores) de-phased exactly this and won +3.4%: the input read hot spot was
  worth -1.2 µs on h4096 and the drain tail shrank 0.4-0.6 µs on h4096/h6144/h7168. The current lineage (r01-b04 ->
  r02-b02 -> ...) never ported the input/drain rotation; only the gamma read is rotated (r01-b01-a03 / r01-b04-a03).
- Since then the drain became the post-AG tail (4.1-6.6 µs after AG release, r04-b03-a02), and r04-b02-a02/a03
  showed the drain is **contention-sensitive to synchronization**: when all 20 drains start within 0.11 µs (multicast
  go) each core drains 0.13-0.42 µs slower than when the go incs stagger them over 0.30 µs. Lockstep bank/column
  targeting is a direct candidate for that contention, and a per-core rotation removes it without delaying anyone.
- The input read is the longest single phase (R_INPUT end 3.4-6.3 µs, ~355-370 GB/s/chip aggregate) and its
  first block of every worker hits banks 0-3 at once.

## Mechanism
Per worker, a column rotation `rot` with `(tile_row_start*W + rot) % 8 == tile_row_start % 8`, so the 20 workers'
starting banks are spread evenly over the 8 banks (2-3 workers per bank phase, both DRAM columns busy in every block
step). CB slot k holds column `(k + rot) mod W` everywhere:
- reader (`dit_rmsnorm_fused_reader.cpp`, trid-pipelined resident pass): input page `row*W + (k+rot) mod W` into
  slot k (also the reader-side bcast gamma ride-along page, for completeness);
- worker writer (`dit_rmsnorm_fused_worker_writer.cpp`): gamma page p lands in slot `(p - rot) mod W` (the existing
  per-worker gamma page rotation is kept); the drain writes CB slot k to output page `row*W + (k+rot) mod W`
  (dual-NoC rule evaluated on the rotated page, unchanged otherwise);
- compute is untouched: it pairs input slot k with gamma slot k and emits output in slot order, which are now the
  same rotated column. sum(x^2) is order-independent up to fp32 rounding.
- factory: compile-time `col_rotate` flag (appended last to reader and writer CT args), enabled only for the plain
  RMSNorm AG path with resident POST: use_mux, !layernorm, !fuse_rope, !streaming_low_l1, !block_major_post,
  num_heads_per_device == 1, no bias, weight read by the writer (or no weight). Every other config keeps rot = 0.

## Why this is not a repeat
- r01-b03-a03 did the same de-phasing on the 80-core column-split lineage (since abandoned); never ported to the
  20-core lineage that every node since r01-b04 descends from. Different core count, different reader (trid
  lookahead), posted drain, and now a drain that is known to be contention-bound.
- r04-b02-a03 #1 suggested "rotate each core's first output bank" to stagger the drains; not implemented by anyone.
- Not a VC / cmd-buf / dual-NoC / posted change (r03-b02-a03, r03-b03-a03, r02-b0x, r04-b04-a01): issue path
  untouched, only the order of DRAM pages each core visits.
- Siblings are stacking HiFi2 + writer wins; this is orthogonal and stacks on all of them.

## Expected effect and risk
- Input read: h4096/h6144/h7168 R_INPUT end -0.1..-0.5 µs (the 20-core lockstep is milder than 80-core, and the
  4-block lookahead already spans all banks). h3584 small.
- Drain: -0.1..-0.4 µs on the wide shapes if the lockstep column targeting is part of the contention.
- Expected score ~1.40-1.42 (parent 1.3906). h3584 ~neutral.
- Risk: a reader/writer/gamma mapping mismatch scrambles columns -> PCC fail (obvious). Partial last block (h3584:
  28 = 7x4, fine; no partial) unaffected since the mapping is per tile. Rotation could also make things worse if
  lockstep actually helped DRAM page locality (row-buffer hits) — then R_INPUT / W_DRAIN would get longer; I'll
  compare zone ends with r04-b03-a02's push.py.

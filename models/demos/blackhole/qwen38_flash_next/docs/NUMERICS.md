# Numerics: the acceptance gate, the pinned tables, what "bitwise" means here

Every number here was measured on 4x p150 unless a date and host say otherwise.

## What "bitwise" means here

- **Bitwise repeatable**: the same request on the same build gives the same device token stream on every run.  The
  greedy argmax stream a request without sampling fields gets is bitwise equal to the greedy loop the acceptance replay
  and the evaluations measure, on the launcher's `--sampling` server too (`qwen38.decode_loop` is `greedy` in the
  response and the ledger; `temperature 0` and `greedy: true` are the same path).
- **Bitwise between two device paths**: `--long-chunks` (128-row prefill chunks) produces the same tokens as 32-row
  chunks alone, bitwise on all 48 layers.  The MTP path is not bitwise with plain decode on 4 of the 12 acceptance
  prompts (below).  The device token streams on tt-metal `28238f903b` (2026-09-06) are bitwise the same as on the
  previous base `d04395ed86` (2026-08-29).
- **Against the CPU reference**: the device is not bitwise with the CPU.  Chunked prefill is tolerance-class against the
  CPU reference on all 48 layers; the greedy token streams agree with the CPU records for a prefix and then leave them
  (the divergence indices below).  96/96 on the `json` record is the gate.
- **BF4 experts**: the routed experts run from a BF4 cache whose every tensorbin payload is byte for byte the CPU-staged
  corpus's (`tools/stage_full_bf4_cpu.py`); the CPU oracle can run with bf16 or BF4-emulated experts (`oracle` below).
  The cache is keyed by the converter's sources (`ttnn/bf4.py`, the `moe_compute` layout packer, tt-metal's BFP4
  packer), and every start re-packs one routed expert of the first cached layer from the checkpoint and compares the
  bytes with the cache; a cache converted by different code is refused (`SERVER.md`).

## The prompt-end snapshot restore (2026-09-28)

The chat server's prompt-end snapshot (`docs/SERVER.md`) copies the recurrent buffers (GDN states and ring slots, PLE
slots, QSA staging tiles and raw-key rings of the 48 layers and of the MTP layer) before the last prompt token and copies
them back on a restore; the KV and compressed caches are positional and not copied (rows past the position are never
gathered, blocks at or past the position are masked or rewritten before they are scored).  The copy is bitwise (pinned
at every start).  What a restore reproduces is the state of the request that captured it, and that state is
schedule-dependent: a 177-token prompt prefilled from position 0 computes positions 160-175 inside one padded 32-row
chunk, while the same prompt restored from a 174-token snapshot (the same messages rendered with thinking on) and
teacher-forced over its 3-token tail computes positions 174-175 in the 1-row FP32 step; the two states at 176 agree to
rounding, and a greedy reply parts from the fresh reply at the next near-tie.  On the served lanes smoke of 2026-09-28
(a 1x4 p150 line, `--mtp 4` greedy, the wrap form) that was token 40 of the `chat` record's max_tokens-40 reply ("crowd
levels" restored, "crowd flow" fresh; the ledger: the row before it restored at 174 with `prefill_forced_tokens` 3 and
recaptured at 176).  This is the tolerance class the 2026-09-06 design already stated for follow-up turns, not a
restore defect: `qwen38.snapshot_schedule` reads `forced-tail` on such a row and `chunked` on an exact repeat after a
fresh prefill, which restores the fresh state bitwise.  The rows of record and the acceptance replays are fresh
prefills.  Measured with the snapshot-restore audit tool, development side (every snapshot buffer, the QSA caches and the
position dumped and compared bitwise per device; 32k, `--mtp 4` greedy, both fold settings, 2026-09-28; the full tables
are in the development notes): the exact repeat restores all 215 snapshot buffers bitwise and the
caches differ only at rows at or past the position (KV rows >= 176, compressed blocks >= 44), the reply identical
(40/40, 256/256, the 1-row loop too); the smoke's history (the thinking-on render fresh, the thinking-off render restoring
it with a 2-token forced tail, then restored again at 176) differs from the fresh 177-token state in 172 of the 215
buffers: every GDN recurrent state (fp32, max abs 0.0024-0.050 per layer), the ring slots of positions 174 and 175 (max
abs 0.73 and 2.48: the 1-row q/k/v projection vs the chunk's 32-row matmul), the QSA staging rows of 174-175 (max abs up
to 23), the PLE slots of 174-175 (1-2 bf16 ulp), and the caches at KV rows 174-175 and compressed block 43; the landing
ring slot and the raw-key rings differ too but are dead or inert at that position.  The reply parts at token 40 under the
wrap (ids 6195 vs 5684, the 15th pass accepting 0 vs 3 after 14 identical passes) and at token 27 under the fold (and
on the 1-row loop under both).  A second fresh prefill reproduces the first bitwise, state and tokens, in every cell.

## Fused decode kernels and two-reader decode linears (2026-09-16)

Decode chains run as fused programs (`ttnn/fused/`, built on `ttnn.generic_op`) where a kernel is bitwise against the
chain it replaces on device, leaves every pinned table above unchanged and beats the previous step time in its own
timing slot. On by default: `gr_read` with `gr_recip_last` (its front with the rsqrt applied after the down projection,
COMPONENT class, its own section below; `QWEN38_FUSED_OFF=gr_recip_last` restores `gr_fold`), `gr_write`, `greedy_tail`, `moe_combine` (the prefill slab's
MoE combine as one program), `moe_post`, `ple`, `position_derive`,
`qsa_block`, `qsa_rows`, `router_tail`, `shared_expert`: the gated-residual read as two programs with its two all-gathers inside
them (the stats, their gather, normalize + down-project and the partial gather as one program whose transport cores send
the tiles over the 1D fabric line into the pages the stock collectives write, then low-rank + gate: 18 programs per read
as 2, the chain's LLK sequences call for call, its reduce scaler and spill/reload rounding included; a gather is data
movement, so the fold is bitwise, 2026-09-25); the gated-residual write as one program (SFPU multiply, FPU add, as the
chain); the tail's greedy epilogue (24 programs as 4 plus one gather); the MoE post program (fill, tilize, the
score-weighted reduce over the ten expert slots in slot order, the shared expert's x sigmoid and the partial add as one
program; its reader takes each routing tensor in one read from moe_compute's drain-core shard and seeds the score tiles
with the NoC's zeros, so the program is 7.5 -> 5.9 us at one row and flat in the row count, bitwise; at several rows the
routed dispatch untilizes the sharded hidden directly and takes the rows view of the result, 2026-09-25); the prologue's
position derivation (40 programs as 1); the sparse-attention block's decode glue as six programs (index tail, main tail,
post-attention, partial widen, selection row, score merge); the MoE router tail (softmax, top-10, sum, div, casts and
layouts: 12 programs per layer as one, its top-k on one core per eight-token group: the LLK sort's four independent
passes, so every token sees the chain's instructions, bitwise, 88 -> 52 us per layer at one row, 2026-09-18; its precise
exp runs over the vector pairs that hold the core's live rows only -- a dead row's exp is never read -- 52 -> 39 us
at one row and 53 -> 42 at the MTP verify's five, bitwise, 2026-09-26); the shared
expert as three programs (one DRAM-sharded linear over the concatenated [gate | up | scalar] weight, one silu / product
/ sigmoid program, the down linear); the layer-1 PLE (stats, group norm, gate, conv with the state shift and the layer's
permute + add: 56 programs as 9, the SFPU `mac_tile` of `ttnn.mac`, the accurate fp32 reduce of the gate's sum); and,
since 2026-09-25, `gdn_step`, the GDN decode step from the projection to the gated output as one program (the conv, the
head split, the gates, the l2 norms, the fp32 delta-rule update, the read-out, the gated RMSNorm and the sigmoid gate:
49 programs per layer as 1), a COMPONENT-class kernel that serves by default on its component-gate proof against the CPU
oracle (layer-0 probe, four p150: state error 0.0048 and gated output 0.0098 for the fused step, 0.0081 and 0.0234 for
the composed chain) and runs where its input contract holds (the one-row step and, since 2026-09-25, the batched-decode
lanes body on B rows, one item per (lane, value head) and one state slot per lane: 40.9 -> 35.0 ms per step at 4 lanes
and 50.4 -> 43.3 ms at 8 on the 200-replay lane sweep, 28.5 and 23.1 tok/s per user, 114 and 185 aggregate, the one-row
step unchanged; the MTP draft body and verify rows keep the chain). The composite lane body, kept as the fallback when
the fused GDN step is not admitted, differs from the one-row fused body at one near-tie in the acceptance chain (219/225
at 4 and 8 lanes); the fused lanes form matches 225/225.  The lanes' greedy tail scans (row, tile-group) items since
2026-09-25, one lane row per core on up to 128 cores (the scan program 164 -> 86 us at 4 lanes and 322 -> 173 at 8, the
step 34.7 -> 34.6 and 42.8 -> 42.7 ms at 4 and 8 lanes on the 200-replay lane sweep), bitwise the per-core all-rows scan
it replaces (`QWEN38_FUSED_GREEDY_TAIL_LANE_SPLIT=0`).  Since 2026-09-26 the MoE block's router top-k, the shared
expert's eltwise and down linear and the routed dispatch untilize run as ONE program (`moe_dense`: four kernel groups on
disjoint cores, the top-k's lane cores on a placement rectangle off the dense linears' storage cores, the eltwise on
those five storage cores multicasting its intermediate into 16 worker cores that run the down linear as a streaming
matmul with the DRAM-sharded matmul's spill and reload after every K tile, the untilize on eight cores), so the 13.5 us
of shared work per layer run under the 51 us top-k instead of after it: 144 programs per step fewer, the composite 51.2
us where the four programs took 64.9, the one-row step 26.50 -> 25.89 ms per token on the landed head beside the live-row exp (37.7 -> 38.6 tok/s, the greedy
request of the sampling server, 200 traced steps in one hold; alone on the previous head 27.24 -> 26.65), the MTP verify pass 59.3 -> 58.6 ms (json) and 57.9 -> 57.5 (prose)
with identical committed streams; bitwise the four programs on every row 1..32 at both down-weight formats and both
routing placements (the device test and the audit-capture microtest), the acceptance table unchanged (12/12);
`QWEN38_FUSED_OFF=moe_dense` restores the four programs.  And, since 2026-09-25, `gdn_rows_wrap`: the MTP verify
rows' GDN body between the projection's sharded-to-interleaved and the out-projection matmul as the prefill lane's
`gdn_pre_rows` + the two chunk prims `ttnn.prim.chunk_gdn_prep` / `chunk_gdn_scan` + `gdn_post_rows`, 6 programs per
GDN layer where the chain ran 53 (the layer's verify segment 11 against 58), bitwise on the die (the gated tile, o and
the final state against the composite's, unmasked and at every commit mask, eager and replayed: the rows micro-test's
wrap arm) and at the pass level (the served pass pair on the same tree: tokens per pass and the accept histograms
identical, the k = 4 pass 7.3-9.8 ms shorter; per GDN layer 0.4805 -> 0.3061 ms and 58 -> 12 device ops).  The
served pair on that tree (the acceptance lineage, so greedy ran the split verify form on both arms; fixed 256-token
answers through the like-for-like client; `QWEN38_FUSED=gdn_rows_wrap` against the switch unset): 560-token chat
55.40 vs 48.81 tok/s (+13.5 %), 177-token multi-turn chat 51.75 vs 45.81 (+13.0 %), json 82.58 vs 72.98 (+13.2 %),
code 82.79 vs 73.33 (+12.9 %), prose 40.99 vs 36.10 (+13.5 %), tokens per pass identical on every class (3.1205 /
2.8333 / 4.5536 / 4.5714 / 2.1983, the same pass counts); `A3-mtp4-32k` with the wrap on 12/12 with the sha, every
pinned divergence index identical (chat 43, code 32, list 56, math 63, ...), json 96/96 at 85.3 tok/s, the hand-offs
matched in both orders.  `QWEN38_FUSED_OFF=gdn_rows_wrap` restores the chain, and the prefill slab keeps its own form
(`gdn_prefill_rows`, on by default after its line gate).
The verify tile's attention-layer glue serves as two bitwise data-movement programs by default since 2026-09-26:
`qsa_score_pages` (the indexer's score slice to the resident blocks and the row-major reshape into the all-gather's
pages, one program) and `qsa_rows_post_attention` (the gate's copy, slices and concat, the local heads' slice, tilize,
sigmoid, multiply, slices, concat and the move into the out-projection shard, one 48-core program), bitwise the chain
on the 1x4 p150 line and on the served stream (the 12-record acceptance table identical to the verify-rows fold's
row on divergence index and token sha256); the pass pair measured -0.51 ms of verify replay and -0.7 ms of pass wall
per pass with tokens per pass unchanged.  `QWEN38_FUSED_OFF=qsa_score_pages,qsa_rows_post_attention` restores the
chain.  A third fold, the out-projection's widen at 32 rows, measured zero at the replay floor and was not kept.
Opt-in through `QWEN38_FUSED=<name>`: `final_mixer`, `position_advance`, `gdn_rows_prims_direct` (the MTP
verify rows' chunk recurrence as the composite's two phase prims called directly with the composite's own relayout ops
in its order, bitwise by construction and on the line 2026-09-25: the seam the wrap's programs were proven against, a
diagnostic beside the default).  `gdn_rows_scan` (2026-09-26, class COMPONENT against the served stream; on by default
since its component-gate record below, `QWEN38_FUSED_OFF=gdn_rows_scan` restores the wrap): the same
verify-rows body as ONE program per GDN layer that runs the fused `gdn_step`'s serial fp32 recurrence row after row
over the k + 1 rows and keeps every prefix state, so the commit is one pick of the prefix state after the accepted rows
instead of the masked re-run of the two chunk prims.  Measured on the 4x p150 line at k = 4 on real layers 1 and 33
(the rows micro-test's scan arm): every prefix state and every committed state bitwise the state after the same rows
of the served 1-row fused step, and the GDN layer's output rows bitwise plain decode's 1-row outputs (max abs 0.0),
where the wrap / chain form (the chunk / WY recurrence at TF32 operand precision) deviates from those outputs by 1.2e-3
on the state at a scale of 0.185 (layer 1; 9.2e-4 at 0.066 on layer 33) and up to 1.2e-2 on the bf16 output rows;
device operations per GDN layer 10 for the forward against the wrap's 12 (the chain's 58) and 3 for the commit against
5 (12); on one die the program runs in 113 us at k = 4 on 48 cores against 382 us for five sequential fused steps, the
pick in 11 us.  The fold's persistent DRAM: (k + 1) x 786,432 bytes per GDN layer per device (3.9 MB at k = 4;
141.6 MB over the 36 GDN layers); on the line at k = 4 with two verify forms the MTP chain's states term measured
31,093,824 bytes per bank against the admission's 13,785,664 estimated without the fold (+17,308,160, of which the
prefix states are 17,694,720 by construction and the rest sits inside the margin), its traces 5,832,256 against
12,218,695 estimated (one program per GDN layer where the wrap runs six), 87,394,112 per bank in all, so the admission's
estimate carries the prefix states -- derived from k + 1 and the GDN layer count -- whenever `QWEN38_FUSED` names the
kernel; the term is per drafting chain (a second chain of `QWEN38_MTP_DRAFTS_PER_REQUEST` allocates its own GDN
rows states and is charged its own k + 1).  The whole verify stream with the kernel on is measured in the MTP
section's opt-in row below (the served line, 2026-09-26); the QSA and MoE rows forms are unchanged by this kernel, and
the MTP table's pins move only at the kernel's default flip.  The kernels cover rows 1..32
(decode, the MTP verify rows); the 128-row prefill chunk and the slab keep their chains.
`qsa_rows` (2026-09-26, on by default; `QWEN38_FUSED_OFF=qsa_rows` restores the composed glue) runs the QSA verify
tile's glue as fused programs: program 1, the indexer scores' all-reduce composite and mask add as one all-gather plus the fused
score merge over the 32 rows (6 programs per layer as 2; bitwise on the four dies at three positions; the MTP lead's pass
pair on the line: -0.50 / -0.60 / -0.51 ms per pass on json / prose / the 560-token chat, the committed streams bitwise,
the A3-mtp4-32k pins 12/12, D1 25.904 against 25.889 ms); program 2, the main tail with the verify rows' KV stage -- the
decode `qsa_main_tail` program's norm, RoPE, head-split and query kernels on the tile's 32 rows with its staging cores
replaced by one KV core that reads the position P and the row count R from device scalars (the verify form's k + 1, the
commit form's accepted count: one program), reads the current cache block back, scatters rows j < R of the packed [v | k]
row into slot (P + j) & 31 of that block or of the next block's zero rows, zeroes the block's rows past the pass as the
chain's kept-0 rows are, and writes both blocks (the head split, two norms, two RoPEs, the concat, the block lookup, the
keep multiply, two one-hot placements, the add, two cache writes and the sparse-query concat / untilize / pad as one
program); bitwise the chain's on the four dies at rows 5 and 6 at the block's first row, inside, crossing the edge and at
its last row, in the single-row form and with the row count 0..5 read from the device, the untouched cache rows
byte-identical, 0.094 against the chain's 0.143 ms per traced call at five rows.  The family's gate on the line
(programs 1 + 2 against the composed glue, the `--mtp 4` server): the pass wall json 51.03 -> 49.65, prose 49.82 -> 48.56,
560-token chat 51.51 -> 50.25 ms (-1.38 / -1.26 / -1.26 ms per pass; the verify replay -1.0, the draft replay -0.33 --
the MTP layer's draft row runs the same main tail at one row), acceptance histograms and tokens per pass identical,
the A3-mtp4-32k pins 12/12 at the default and under `QWEN38_MTP_SAMPLED=0` (the json stream's sha equal to the composed
glue's), the plain-decode D1 band untouched (25.602 against 25.610 ms per step median over 200 steps).Since 2026-09-26 the slab has a fused form of its own, `gdn_prefill_rows` (on by default: bitwise the chain on the 4x p150
line in two independent runs -- pins 12/12, records 12/12, 3,232 agreement columns, 4/4 probes -- with the time to the first
token at 32k -13.1 %; a four-die device test guards the mesh-topology seam the one-die tests cannot see;
`QWEN38_FUSED_OFF=gdn_prefill_rows` restores the chain): the GDN body from the projection to the gated output as two rows-form programs around the unchanged chunk prims (called directly through
`ttnn.prim.chunk_gdn_prep` / `chunk_gdn_scan`), bitwise the chain's on one p150 die at 64 and 2048 rows, the recurrent
state and the FIR history included (PREFILL.md has the measured per-call figures).
`QWEN38_FUSED_OFF=<name>[,...]` (or `all`) in the server's environment falls back to the composed
chains; an unknown name in either variable refuses to start; `QWEN38_FUSED_OFF=gr_fold` runs the GR read's merged
three-program form with the stock collectives (5 programs per read); `QWEN38_FUSED_GR_READ_MERGED=0` runs the GR read's
split form (7 programs per read); `QWEN38_ROUTER_TAIL_LANES=0` runs the router tail's top-k on one core per tile (the
same program, the LLK's four passes on that core); `QWEN38_ROUTER_TAIL_EXP_LIVE=0` runs the router tail's exp over every
vector of every tile (the same bits, the full cost).

The DRAM-sharded decode linears of the GDN input and output projections, the sparse attention's query-gate and output
projections and the LM-head chunks read each DRAM bank with two worker cores (`num_workers_per_dram_bank=2`; the K/V and
index projections, the router, the gated-residual linears and the shared expert keep one).  The arithmetic is the same
(the same K order into the same fp32 destination): bitwise at rows 1..32 and at the model level, every pinned table
above holds.  `QWEN38_DRAM_WORKERS=1` restores one reader per bank.  The GDN input weight's bank shard holds a whole
number of tiles per reader, 18 instead of 17 (576 instead of 544 columns: 1.3 MB per layer, 47 MB per card of the
4.6 GB the 32k build leaves free), under its own cache file (`qkvzab_tile_aligned_ab_dram_sharded_bank576`); a cached
tensorbin loads in the layout it was written in, so every loaded member is checked against the layout its linear runs
and a one-reader file is never accepted for the wider allocation.

Measured 2026-09-16 on 4x p150 (200 traced decode steps, host wall, one hold): 49.89 ms per token with the
chains and one reader per bank (`QWEN38_FUSED_OFF=all QWEN38_DRAM_WORKERS=1`), 38.67 ms with the defaults
(25.9 tokens/s; 38.26 min, 38.85 p90); 3,370 programs and 34.5 ms of kernel time per
step on chip 0 under the device profiler (38.9 ms span).  The 2026-09-15 numbers (13 kernels, one reader per
bank): 42.14 ms, 4,618 programs, 37.9 ms of kernel time.

Measured 2026-09-25 on 4x p150 with the defaults of that day (BF8 dense weights, the fused GDN step, the compact expert
layout, the MoE post program's one-read routing, two readers per bank), the README's decode rows: one stream 27.2 ms per
token on the 200-step pin recipe (36.8 tokens/s greedy), flat with depth; the device sampler's step 27.4 ms (36.5
tokens/s sampled) on both model cards (thinking; instruct with its presence penalty); the batched-decode lane body at
4 / 8 lanes 34.6 / 42.7 ms per step, 28.9 / 23.4 tokens/s per user and 116 / 187 aggregate (the chat server serves one
stream; the lane sweep measures the lane body directly).

The slab's one-pass MoE combine (`moe_combine`, the default since 2026-09-26; `QWEN38_FUSED_OFF=moe_combine` restores the
512-row blocks) is bitwise class: it issues the fused reduce's own multiply-accumulate in its slot order on the
page's owned rows, and the line gate of 2026-09-26 (4x p150, 32k context, `--prefill-slab 2048`) read the twelve
acceptance records, the 3232 agreement columns and the four probes identical to the blocks', with the attention
identity through the stack unchanged; TTFT at 31,716 tokens 14.30 -> 12.38 s (`PREFILL.md`).  The one call's three
rings (`QWEN38_MOE_SLAB_RINGS` unset, the default since 2026-09-26) are bitwise the two rings on the same gate: records
12/12, columns 3232/3232, probes 4/4, in two runs (`PREFILL.md`).

The slab's routed rows were freed under the op (2026-09-26, fixed the same day): `ttnn.reshape` of the untilized
`[1, 1, T, 2560]` rows to the `[1, T, 2560]` moe_compute counts is a zero-cost view -- a new tensor id over the same
buffer -- and the model freed the source on a tensor-id test, so the op's next DRAM allocation (its packed token-list
page, 90 KB) landed on the freed input's first pages: DRAM bank 0's first six pages of the 8-bank interleave = token
rows 0, 8, 16, 24, 32 and 40 (measured in the slab body: those six input rows change during the call, to the list's
words; on one die the view and its source share the address and the next allocation lands exactly there, rows 0, 8,
.., 120 of a 2048-row input overwritten).  Whether the page landed on the freed rows depended on the allocator's
state: a fresh or cache-off launch, yes; a cached launch or a trace replay, no.  So the process's first slab body (the
server's warm pass, whose output is discarded) and the cache-off diagnostic tools computed layer 0 on six corrupted
rows (83 of 84 state buffers off from the second run on, first at layer 1's GDN recurrent state, worst 12.96 on a KV
row), while every served slab (a trace replay) computed the correct rows.  The fix keeps the source alive across the
call and frees a reshape's source only when the reshape copied (`ttnn/contracts.py` `same_buffer`, by buffer address;
the rule applied at every reshape-then-free site of the model, the eight others being copies today).  Gate: a fresh
process's slab body equals its second and third runs bitwise on all 84 state buffers, all three equal to the
trace-replay result; the served completions of the fixed tree equal the unfixed tree's token for token (below).

The 2048-row slab and the 128-row chunk form are tolerance class against each other (`PREFILL.md`), and on long
chat-shaped prompts the class is visible in the greedy stream: the served completions of the two forms differ from
token 30 on a 2,160-token document and from token 0 on a 20,612-token one (256 greedy tokens each, 2026-09-26, 4x
p150, 32k context), while the twelve acceptance records (37-434 tokens, no slab fires) stay identical.  This is the
forms' documented numerics, not the defect above (the defect never reached a served stream).

`moe_compute`'s W2 ring exchange runs its handshake in the first a2a iteration only (2026-09-26, a runtime patch: the
partials travel once per chunk and the compute's later iterations read the resident buffers; the dropped iterations'
wait / increment pairs disappear on every core alike): bitwise on every form the op serves here (the decode rows, the
128-token chunk, the 2048-row slab: the 12-prompt pins and their token-stream digests, the `--mtp 4` pins and tokens
per pass, a 41-routing soak of the combine pages), 12.98 -> 11.6 us per distinct local expert per launch on the 1x4
line (-10 %), the B=1 step 27.14 -> 26.89 ms on the 200-step pin recipe, the sampled `--mtp 4` pass -0.84 ms per token
served, the slab's one call 2.09 -> 1.94 ms per layer (-7 %) on one die.

`moe_compute`'s a2a pipeline (2026-09-26, a runtime patch, the streaming decode ring only): compute runs W0/W1 of the
next owned chunk while dm1 exchanges the current chunk's partials (two parity slots of a2a buffers; one chunk's credit
may be outstanding), dm1 exchanges the next chunk before it writes this one's output rows, and the feed carries three
chunk slots (a fourth does not fit the slab-mode server's L1); the prefill ring forms keep the serial order. The same matmuls in the same order: the combine pages are
bitwise on the rows bench (11 cases, three builds); 11.42 -> 10.91 us per distinct local expert per launch on the 1x4
line (-4.5 %); the ring core is now paced by its DRAM weight stream (9.7-9.9 us per expert, about 285 GB/s of the
die's 512). The served numbers follow with the rebuild that carries it. The streaming decode ring's weight CB holds
four blocks instead of three (2026-09-26): 10.88 -> 10.68 us per distinct local expert on the same bench, bitwise;
five and six blocks and dual-NoC reads are no better -- the ring core's stream is paced by its one DRAM bank (about
36 GB/s of the channel's 64); the prefill ring forms keep three (their L1 has no room for a fourth).

`moe_compute`'s rows form on the local output path with two rings (2026-09-27, `QWEN38_MOE_ROWS_FORM=rings2`, the default; a launch-form change in the model, the runtime unchanged): each of the two rings reads the weight slices of the experts it owns (a one-chunk expert goes whole to the least-loaded ring) and dm1 writes the expert rows into the combine buffer itself (no combine kernels: 20 cores instead of 16). Every ring's matmuls are the one-ring form's in the same order into the same accumulators, so the pages are bitwise: on the rows bench 12 cases x 4 devices (rows 1..5, n = 3..13 distinct local experts, identical and random routings) for every local output form (streaming ring, replay ring, two and three rings) against the fused-local-combine pages; the soak, 40 seeded random routings x 10 calls x 50 traced replays per form with the buffer after the replays bitwise the first call's pages; the A3-mtp4-32k, A3-chunked-32k and A3-slab2048-mtp4-32k tables identical (12/12, 12/12, 13/13 divergence indices and token shas, no re-seed). Per launch on the 1x4 p150 line (rows 5, n distinct local experts on the busiest device): 42.0 + 10.87 n us with the fused local combine -> 30.2 + 10.86 n on the local output path's streaming ring -> 36.1 + 7.39 n with two rings (29.6 + 8.20 n with three); the second reader per bank halves each reader's rate -- the bank stops at about 43 GB/s with two or three readers (the dense linears' ceiling; the die at about 340 GB/s of its 512), so three rings buy nothing over two, and a layout spreading each core's slice over the banks would add no bandwidth. In the served `--mtp 4` pass (chat560, greedy, the pass decomposition of record, 120 pipelined passes) the verify's device wall 37.89 -> 35.94 ms (-1.95 ms over 49 launches = the bench law at the served expert counts), the pass 49.19 -> 47.16 ms p50, tokens per pass identical; served rows (EOS honoured, the same server form): the 560-token chat prompt 62.19 -> 64.86 tok/s (+4.3 %), json 96.6 -> 101.01, the multi-turn chat 58.5 -> 61.04, prose 45.42 -> 47.25, code 93.34 -> 97.47, sampled TPOT 19.92 -> 19.08 ms; plain decode 37.95 -> 39.27 tok/s. `fulllocal` restores the one-ring fused-local-combine form.

The MTP layer's rows after a slab prefill (`--prefill-slab` with `--mtp`, 2026-09-26) are the 128-row twin's slab form:
every slice is the 128-row chunk extension's exact arithmetic (bitwise per row against the 128-row chunks given the same
input rows), but its input is the slab body's layer-47 residual, which is tolerance-class against the chunk bodies, so
the MTP layer's KV cache and the MTP stream after a slab prefill are tolerance-class against the chunked prefill's -- the
same class as the target's own state after a slab.  They carry their own pinned table (`A3-slab2048-mtp4-32k`: the twelve study
records' and the slab record's divergence indices and token-stream digests under `--mtp 4 --prefill-slab 2048`, generated
at the landing head and asserted identical on every later run) beside the target's `A3-slab2048-32k`.  The 1x4 proof
(a development tool, run 2026-09-27 on the fixed body): the slab body with the twin is bitwise run to run on the target's
84 state buffers, its 97 hand-off buffers and the MTP layer's 4 (three runs, the process's first slab body among them) and
bitwise to the body without the twin; the MTP layer's KV rows after the slab differ from the 16 x 128-row chunk bodies'
by at most 10.20 where the target's own slab-vs-chunks difference in the same run is 17.22 (the class holds), the
hand-off's integer fields are identical, and the slice form is bitwise to the 128-row form on identical residual rows.
Measured 2026-09-27 on the 1x4 p150 line at the landing head (e60ad5151037), `--prefill-slab 2048 --mtp 4` against
`--prefill-slab 2048` on the same build (one cold request per row as the server's first requests, 32 completion
tokens, host-timed TTFT): 2,100 prompt tokens 1.093 against 0.983 s, 20,860 tokens 7.701 against 6.713 s, 31,714
tokens 11.991 against 10.504 s -- +0.053 / +0.047 / +0.047 ms per prompt token, +110 / +99 / +99 ms per 2048-row slab
for the twin's 16 traced slices (1 / 10 / 15 slabs; the slab's 3,019 tok/s becomes 2,645). The 560-token chat prompt
decodes at 62.6 tok/s greedy (3.12 tokens per pass; 37.9 plain) and the sampled card profile at 19.74 ms per token
(25.78 plain); the acceptance replay pins the twelve study records 12/12 to the `--mtp 4` table (they stay under 2048
prompt tokens, so no slab fires on them) and the slab record (`document`, 2,228 prompt tokens, one slab) at divergence
index 12 (stream sha 9beadb60...) under `--mtp 4`, against 12 (7d56a970...) for the plain slab: the only pin
exercising a slab body under MTP.

### Two readers per bank on a mixed-harvest mesh (QuietBox 2, 2026-09-29)

One DRAM-sharded matmul program is placed on every device of the mesh; until 2026-09-29 tt-metal's two-reader placement
required every device to report the same optimal bank -> worker assignment, and a QuietBox 2, whose dies are harvested
in different columns (one die serves banks 4-7 from another worker column), fell back to one reader per bank
(`READY` `dram_workers_fallback`).  A DRAM-sharded reader addresses its shard by bank id, so any worker can read any
bank and the assignment is a locality choice: the placement is now the first device's on every device
(`ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp`, `get_dram_bank_reader_assignments`), the
devices whose own optimal readers or hop counts differ are named once in a warning with the largest hop deviation, and
`READY` carries `dram_workers_placement` ("identical", or the reference coordinate with the differing dies).  One
reader remains where the placement cannot be validated: a device whose worker or DRAM grid differs from the first
device's, or a mesh whose bank count is not the eight the table was qualified on.  Numerics: bitwise by construction
(data movement only); measured below.

Measured 2026-09-29 on a QuietBox 2 (2x p300c, harvested columns 7+9 / 0+9 / 6+13 / 1+9, one 1x4 line, the placement
`reference (0, 0); other optimal readers at [(0, 1), (0, 2)]`, largest hop deviation 3; host Ryzen 7 9700X, cpufreq
powersave, load 1.0-2.4; each arm its own hold, the fork launcher with `--profile qb2 --acceptance --require-json-96`,
the rows the like-for-like client's EOS-honoured medians of 3 after a warm-up, the sampled TPOT 6 chats x 200 tokens):

| form | 1 reader per bank (the fallback) | 2 readers per bank | 4x p150 line (2026-09-28 card) |
|---|---|---|---|
| `--mtp 4 --long-chunks --prefill-slab 2048`, greedy: 560-token chat / multi-turn / code / json / prose tok/s | 63.92 / 59.93 / 97.57 / 105.17 / 46.62 | 68.73 / 64.75 / 104.88 / 113.05 / 50.28 | 68.3 / 64.2 / 104.3 / 111.0 / 50.0 |
| the same, tokens per pass | 2.9425 / 2.6842 / 4.4483 / 4.75 / 2.0857 | identical | 2.94 / 2.68 / 4.45 / 4.75 / 2.09 |
| the same, sampled TPOT non-thinking / thinking ms | 17.81 / 19.30 | 16.50 / 18.08 | 16.7 / 18.3 |
| plain decode (no `--mtp`), greedy 560-token chat tok/s; sampled TPOT ms | 38.54; 25.27 | 40.52; 23.96 | (38.6; 2026-09-26 row) |
| the 13-record startup replay (A3-slab2048-mtp4 form) and the 12-record plain replay | the pins | identical to the one-reader arm, record by record | the pins |

Where the one-reader gap sat (the MTP pass decomposition and the device census on the same box, chat560, the fold form,
one reader; the line's fold-form census of 2026-09-26 as the reference): pass wall 46.19 ms pipelined against the line's
41.95; blocking replays verify 36.03 + draft 8.23 + commit 0.96; the exposed host 1.16 ms (smaller than the line's
1.5-2.2: PLE lookup 0.96 + upload 0.21); per program class on chip 0, Matmul 7.76 + 3.91 (verify + draft) against the
line's 5.62 + 2.52 = +3.5 ms, MoECompute 5.37 against ~4.3 expected at this head (~+1.0), the fused programs ~+0.7,
AllGather 1.88 against 1.82 and the reduce-scatters 2.54 against 3.19 (the ring's two-channel links are not a term),
host<->device transfers one 6.4 KB write and one 268 B readback per pass (per-call latencies 3-4x below the line's
2026-09-03 record).  With two readers the rows read the line's.

The image's `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` (pinned host memory not cached, set for the translated-IOMMU
long-pin hazard) costs the SAMPLED pass on this host: the split verify's tiny decision writes (three 4-byte host
tensors per pass) take 57 us at the median but 2.9 ms at the 90th percentile, `decision_write` 3.62 ms per pass, the
sampled pass 50.92 ms against 47.81 with the cache unset (the greedy pass unchanged, 46.19 / 46.07).  The line
measured no effect of the setting either way; the serving default is the release's decision.

## The slab's block-shared attention (`QWEN38_FUSED=sparse_sdpa_tiled`, 2026-09-25)

A tolerance-class fused kernel, opt-in: `sparse_sdpa_tiled` replaces the prefill slab's block-id expansion, zero V
half, head pad, `sparse_sdpa` and head slice with one program per QSA layer (`PREFILL.md`).  The attended set per query
and the bf16 scores are the chain's; the flash loop's order and chunk size differ, so the running max / sum / out round
elsewhere.  Judged as the slab itself is (the long windows against the references) and, per layer, against an fp32
oracle on the same bf16 inputs, where it must be no farther than the chain: on a QuietBox 2 die (2x p300c) with the
captured slab selections it reads PCC 0.99979 against the oracle (the chain 0.99978), row rel_err p50 0.021 (0.022),
PCC 0.99954 against the chain at P = 28672 and 0.99988 at P = 0; bit-exact run to run and across trace replays.  The
chunk forms and the decode path never see it.

The line gate (4x p150, 32k context, `--prefill-slab 2048`, 2026-09-26; the chain as the control; acceptance pins
12/12 and the acceptance columns identical on every arm; the 31,716-token long part identical through the first slab
and diverging from position 8128, where a query first attends across a slab boundary):

| arm | KL control -> arm, mean / max (long part) | top-1 vs the control | top-1 vs HF | TTFT 31,716 tokens | attention per 2048-row slab (device profiler, the prompt's first slab, 12 layers) |
|---|---|---|---|---|---|
| control (the chain) | - | - | 0.9875 | 15.92 s | 89.77 ms (`sparse_sdpa` 81.14 + expansion, pad, concat, slice 8.63); the slab's kernel time 899.5 ms |
| kernel, HiFi4 / fp32 DEST (`QWEN38_SPARSE_SDPA_TILED_FIDELITY=hifi4`) | 0.0059 / 0.55 | 159/160 | 0.9812 | 14.36 s | 18.45 ms; 818.6 ms |
| kernel, HiFi2 / bf16 DEST (the default) | 0.0028 / 0.084 | 160/160 | 0.9875 | 14.25 s | 16.28 ms; 816.6 ms (2,257 -> 2,484 tok/s over the slab) |

The HiFi2 form is the default: closer to the control and to the reference than the HiFi4 form (whose fp32
destination changes where the scores round) and 10 % faster to the first token at 32k; under the device profiler the
attention class of one slab is 5.5x shorter than the chain's (89.77 -> 16.28 ms) and the slab's kernel time 9 % shorter.

## The GR read with the rsqrt applied last (`QWEN38_FUSED=gr_recip_last`, 2026-09-27)

A COMPONENT-class fused kernel, on by default since 2026-09-27 with the component proof below recorded in its registry
entry (it needs `gr_read`; `QWEN38_FUSED_OFF=gr_recip_last` restores the fold's front): `gr_recip_last` replaces `gr_fold`'s one-program front (the stats, their line gather, normalize and
the down projection, the partials' line gather) with the same program re-associated so that nothing waits on the
statistics exchange.  The fold's 5-row zone timeline (the MTP verify's rows, 111 traced calls, one 1x4 p150 line) puts
22 of its 44.1 us per call between the two gathers waiting on rsqrt(mean): the norm compute 8.6, the 80-tile multicast
of the normalized row into the twelve down workers 7.8 and the matmul 6.1; the two data legs themselves are 3.8 and
7.0 us.  Here the norm cores compute u = bf16(x * gamma/4) right after the stats and multicast it at once; the workers
run the same 80 matmul blocks into one fp32 dest tile per residual stream while the stats round trip is in flight;
the norm cores turn the gathered stats into rsqrt(mean + eps) (the chain's reduce, eps and rsqrt, bitwise that far)
and multicast the four fp32 tiles to the workers, which scale each stream's partial (the chain's column-broadcast
multiply, on the fp32 partial) and sum the four in stream order into the partial tile the partials gather carries as
before; the gate's operand bf16(u * rsqrt) is written off the critical path.  The transports, the payload and
`low_rank_gate` are unchanged.  Rounding points that move against the chain: bf16(x * gamma/4) enters the matmul where
the chain rounds bf16(bf16(x * rsqrt) * gamma/4); no spill/reload of the running partial; rsqrt is applied to the fp32
partial; the gate's operand has one bf16 rounding where the chain has two.

The component gate (the 1x4 replica against the tt/ oracle's `read_tp4` on captured residual rows, layers 2/12/47 x
rows 1/5/32, the checkpoint's attention-block weights; the composed column is the bitwise fold): block-input rms_rel
against the oracle 0.0036-0.0045 where the composed chain reads 0.0038-0.0045 (the kernel is closer at layers 2 and
12 and 4 % farther at layer 47, within the component rule's 1.10x + 1e-3 on every row), |bias| <= 1.9e-4, ULP p99
8-16 against the chain's 8-16, the injection's rms_rel equal to the chain's to three digits; the two device forms
differ from each other by rms_rel 0.0036-0.0040:

| layer | rows | fused rms_rel | composed rms_rel | allowed (1.10x + 1e-3) | fused bias | fused / composed ULP p99 |
|---|---|---|---|---|---|---|
| 2 | 1 | 0.00385 | 0.00391 | 0.00530 | 9.1e-05 | 16 / 12 |
| 2 | 5 | 0.00408 | 0.00426 | 0.00568 | 7.3e-05 | 16 / 16 |
| 2 | 32 | 0.00413 | 0.00425 | 0.00568 | 5.6e-05 | 16 / 16 |
| 12 | 1 | 0.00363 | 0.00376 | 0.00514 | 4.7e-05 | 10.4 / 15.4 |
| 12 | 5 | 0.00408 | 0.00443 | 0.00587 | 4.0e-05 | 16 / 16 |
| 12 | 32 | 0.00428 | 0.00449 | 0.00594 | 4.5e-05 | 16 / 16 |
| 47 | 1 | 0.00445 | 0.00427 | 0.00569 | 1.9e-04 | 8 / 8 |
| 47 | 5 | 0.00436 | 0.00424 | 0.00567 | 1.9e-04 | 9 / 8 |
| 47 | 32 | 0.00412 | 0.00407 | 0.00548 | 7.4e-05 | 11 / 12 |

Program cache +1 on the first call then 0; 3 x 100 trace replays with
the rows refreshed bitwise against eager calls; every transport semaphore 0 afterwards.  Cost: the front 44.1 -> 33.1
us per call on the device (span, 5 rows), the read (front + `low_rank_gate`) 71.5-72.1 -> 59.7-60.7 us traced host wall
at rows 1/5/32; the new pacer is the u multicast (8.3 -> 17.5 us), then the partials exchange (8 us: six 4 KB packets
per transport core; a pool of packet headers in place of the per-packet flush was measured flat, so the sender's
slot wait paces those sends, not the flush; multicasting the u row in 4-tile pieces as it is packed measured 2.4-3.2 us
slower, so that multicast is bandwidth-bound, not issue-bound).

Measured 2026-09-27 on the 1x4 p150 line with the kernel on against the fold (the same tree, both arms): the served
`--mtp 4` pass p50 chat560 48.82 -> 47.58 ms, json 48.28 -> 46.86, prose 47.22 -> 45.51, chat560 sampled 51.27 -> 50.00
(100 pipelined passes per cell, the verify replay 37.84 -> 36.59 / 37.07 -> 35.87 / 36.39 -> 35.06, the draft and commit
replays unchanged); the paired 43-prompt EOS-honoured sample (the twelve acceptance records, sixteen served / eval
corpus items, six book excerpts, five synthetic chats; one greedy 256-token request each, `ignore_eos` false; a second
quiet 1x4 line, load under 2): 3108.2 against 3027.9 tok/s summed (+2.7 %), up on 28 prompts / down on 15, paired mean
+1.87 tok/s (s.e. 0.52; 95 % interval +0.85..+2.89), mean relative +2.99 % (s.e. 0.94), by part acceptance +3.3 %,
served +4.0 %, eval +1.4 %, book +0.7 %, synthetic +3.1 %.  The token streams change (a tolerance-class kernel), so the
tokens per pass move both ways per prompt while the pass wall gain is deterministic.  The acceptance replays under
the kernel (json 96/96 on every path): plain decode against the pinned chunked table keeps fact 15, list 56,
multilingual 9, prose 13, refactor 24, sky 19, story 6, summary 75 and leaves the reference later on chat (2 -> 56) and
code (32 -> none in 96), earlier on math (61 -> 56); `--mtp 4` against the pinned MTP table keeps nine indices (code 32,
fact 15, multilingual 9, prose 13, refactor 24, sky 19, story 6, summary 1, json none) and moves chat 43 -> 39, list 56 ->
46, math 63 -> 56 -- the positions the k = 5 study above records as 0.09-0.36-logit near-ties; `--prefill-slab 2048
--mtp 4` against its 13-row table keeps the slab record at 12 and reads the twelve study records exactly as the MTP
replay does.  The pins moved to this kernel's tables with the default (below).  The README's `--mtp 4` row, re-measured on the
combined head (the MoE rows form on two rings and this kernel by default) on a quiet 1x4 p150 line, 2026-09-27, the
EOS-honoured client (1 warm-up + median of 3, greedy, max 256): chat560 64.35 tok/s (256 tokens, 2.98 tokens per pass),
the 177-token multi-turn chat 56.77 (256, 2.56), code 102.2 (256, 4.65), json 103.89 (153 to its end marker, 4.78),
prose 50.18 (193, 2.24) against the rings row's 64.86 / 61.04 / 97.47 / 101.01 / 47.25 (the 177-token chat's stream
leaves the fold's at token 39 and accepts fewer drafts: 2.83 -> 2.56 tokens per pass; code and json accept more);
sampled TPOT (median of 6 chats) 17.60 ms non-thinking and 19.32 ms thinking against 19.8 / 19.1; the chat560 pass p50
45.84 ms (p10 45.15 / p90 46.39, 100 pipelined passes, 2.914 tokens per pass), json 45.11, prose 44.16.

## The acceptance mechanism

`--acceptance` replays the shipped CPU greedy records under `tools/acceptance/greedy-prompts/` against the CPU before
the server listens: the twelve study records, and on a server with a prefill slab the slab record
(`prompt-document-greedy.json`, 2,228 prompt tokens, a document made of the twelve records' answers: the only record
that runs a slab).  `--require-json-96` refuses to serve unless the `json` record matches the CPU 96/96 (the
other records diverge from the CPU after 2-75 tokens, the known greedy-prompt pattern; their first divergence index is
in `acceptance.json` in the run directory).  The records were rendered by the CPU study with the system prompt
`You are a helpful assistant.`; that system turn is inside their recorded prompt ids, which the replay feeds to the
device as they are.  The server itself adds no system prompt to a client's request (`SERVER.md`).

## The pinned table: chunked prefill, 32k, 2026-09-25

The pinned table (`tools/ci/baselines/A3-chunked-32k-divergence_index.json`), measured 2026-09-25 on two 4x p150 hosts
with identical device tokens record for record, with the chunked prefill, is the first index where each record leaves
the CPU greedy stream; a start whose replay leaves earlier is a regression:

| json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none (96/96) | 2 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |

The 2026-09-25 change is the BF8 dense weights, the fused GDN step and the compact expert layout by default: `chat`
leaves at 2 where the 2026-09-06 table had 43 (at index 2 the device's own candidate row holds its pick, 449, one bf16
step above the CPU's 264; the BF8 weights and the fused step each move that margin one step, alone either keeps 264),
the other eleven indices are unchanged, and six streams differ from the previous table after their divergence
(`fact` at 60, `multilingual` at 32, `prose` at 76, `refactor` at 83, `story` at 16).  Against the HF reference over the
36-item corpus the device column stays at top-1 0.9502 (the 2026-09-06 record, weighted the same way, 0.9518) and
against the bf16 CPU oracle at top-1 0.9539, truncated KL 0.0511.  The 2026-09-06 table read chat 43 with the rest as
above.  Before 2026-09-06 the table read chat 8, code 24, list 46, refactor 22 (the other eight as above).  The change is the
GDN decay gate: the fused `add + softplus` activation the gate used returned exactly 0 wherever its input was below
-5.02 (and was 1.5e-3 off elsewhere); the gate now runs `ttnn.add` then `ttnn.softplus`, which follows the reference
everywhere (4e-5 absolute).  Against the CPU oracle the device's GDN state error on a decode step fell from 0.29 to
0.013; with the chunked prefill the device stays on the CPU stream longer on 4 of the 12 prompts and no prompt leaves
earlier (the teacher-forced table, `A3-forced-32k-divergence_index.json`, moved later on four prompts and earlier on
three); the cost is one more program per layer, about 0.4 ms per decoded token (50.0 -> 50.4 ms).  The proofs in
`PROOFS.md` predate this fix where they quote the old indices.

## `--long-chunks`

The 128-row chunk path gives the same tokens as 32-row chunks alone, bitwise on all 48 layers; on the QuietBox
(2026-09-06, before the gate fix) the acceptance replay with `--long-chunks` was identical to the plain start (`json`
96/96, the same eleven divergence indices, 19.4-19.6 tokens/s).

## MTP (`--mtp 4`), 32k, 2026-09-25

The MTP path leaves the CPU reference at plain decode's token on 12 of the 12 acceptance prompts and is bitwise plain
decode's stream on 8 of 12 (measured 2026-09-26 with the verify rows' GDN body on the fold, `gdn_rows_scan`, bitwise
the 1-row fused step; the pinned table is that run's); `list`, `multilingual`, `prose` and `refactor` leave the
reference where plain decode does and differ after it, unattributed by measurement (the row below).  Before the fold
(2026-09-25, the verify rows on the composed chain, rounding apart from the 1-row loop) the committed stream left the
reference on `chat` at token 43 where plain decode leaves at 2, on `math` at 63 against 61 and on `summary` at 1
against 75 (`In 1947,` became `Invented at Bell Labs in`); the other nine records left the reference at the
plain-decode token (`json` 96/96), and every gate passed.
The 2026-09-06 table (below, the GDN gate fix of that day) had four such prompts: `chat` 56 against 43, `list` 46
against 56, `math` 56 against 61, `summary` 1 against 75; `list` now agrees at 56.  At
each of the three earlier tokens the device's own 1-row logits hold the CPU's token and the MTP token within one bf16
step (an exact tie on `summary` and `list`, which the 1-row loop breaks toward the lower id), the verify row landed one
step the other way, and the CPU oracle rates the two 0.4-1.0 logits apart: near-ties, not a defect; under the fold the
verify row is the 1-row step's (the pinned MTP table is `tools/ci/baselines/A3-mtp4-32k-divergence_index.json`).
Before the gate fix the two paths differed on `code` (44 against 24) and `fact` (16 against 15) only.  Throughput
on the 2026-09-16 body (14 fused kernels, two DRAM readers per bank; quiet host, the acceptance replay of the 12
prompts, both tables as pinned): `--mtp 4` 39.0 tokens/s median over the prompts and 67.9 on `json` (4.80 tokens per
pass, 70.7 ms per pass); `--mtp 3` 38.0 median, 55.6 on `json`.  Sampled drafting (2026-09-25, `--mtp 4` on the
27.7 ms D1 step, the card profiles non-thinking / thinking): 40.8 / 40.7 tokens/s at 24.5 / 24.6 ms per token against
33.5 / 33.9 plain sampled, 2.64 / 2.68 tokens per pass, 0 fallbacks.

| | json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain decode, 2026-09-25 | none (96/96) | 2 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| `--mtp 4` (the GDN rows fold, the default), 2026-09-26 | none (96/96) | 2 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| `--mtp 4` (the wrap, `QWEN38_FUSED_OFF=gdn_rows_scan`), 2026-09-25 | none (96/96) | 43 | 32 | 15 | 56 | 63 | 9 | 13 | 24 | 19 | 6 | 1 |
| plain decode, 2026-09-06 | none (96/96) | 43 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| `--mtp 4`, 2026-09-06 | none (96/96) | 56 | 32 | 15 | 46 | 56 | 9 | 13 | 24 | 19 | 6 | 1 |

The fold's row (`gdn_rows_scan`, the default since its component-gate record; 2026-09-26, the served `--mtp 4` line;
the pins above are this run's).  Every number of this row was measured on the sealed fork runtime built at `e74d3b864ff`,
at head `93b9e857bd2` (the 2026-09-26 unified `5307c930416` plus the fold); the runtime kit of record is
now the fork runtime built at `5eac9c778edb` (the MoE a2a pipeline lives in the runtime C++), the fold-against-wrap comparison
stands as measured on the sealed kit; the kit of record's numbers (the fork runtime built at `5eac9c778edb`, `_ttnncpp.so`
`211530d77b2b`, the same line, 2026-09-26, the tree = unified `5eac9c778ed` plus the fold) are the numbers of record
and the sealed kit's the original measurement, both given below.  The committed stream leaves the CPU
reference at plain decode's token on 12/12 prompts (`chat` 2, `math` 61, `summary` 75 where the served row above reads
43 / 63 / 1), and its device token stream is bitwise plain decode's pinned stream on 8/12 (`json`, `chat`, `code`, `fact`,
`math`, `sky`, `story`, `summary`; the served row's pinned streams equal plain decode's on 5/12).  The greedy server
(`QWEN38_MTP_SAMPLED=0`) with the kernel on reads the same twelve rows.  On `list`, `multilingual`, `prose` and `refactor`
the stream leaves the reference at plain decode's token but its later tokens differ from both pinned streams (on
`refactor` the served row's stream equals plain decode's where the fold's does not): under the fold the GDN rows are
measured bitwise the 1-row fused step, and no other verify-rows form is measured against its 1-row counterpart row for
row -- the linears are the same tile programs at 1 and 32 rows, the residual read / write, RoPE and norms the same
per-row kernels, the QSA glue the decode's own kernels over the rows (bitwise the chain at rows 5 and 6), the MoE rows
form measured bitwise the 1-row path (rows 5, 2026-09-03/04; the down linear at rows 1 / 5 / 32, 2026-09-25), the
candidates' argmax exact -- so the forms bitwise by construction only, the attention over the R query rows, the PLE rows
body and the compressed-index block mean, are where a rounding difference can still sit: by elimination, not by
measurement; the near-tie check (the device's own 1-row logits at the first differing token) is the test that settles it.
Plain decode's own step is untouched: 25.612 against 25.619 ms per token over 200 traced steps with the kernel on / off.

| | json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `--mtp 4` + `QWEN38_FUSED=gdn_rows_scan`, 2026-09-26 | none (96/96) | 2 | 32 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| stream bitwise plain decode's pin | yes | yes | yes | yes | no | yes | no | no | no | yes | yes | yes |
| stream bitwise the `--mtp 4` pin | yes | no | yes | yes | no | no | no | no | no | yes | no | no |
| fold stream sha256 (first 12) | a5b4defa72fa | 34686e9146e7 | 0234032be259 | 9afd57cc0303 | ea554701b6f5 | 19f55ccb4bb2 | e83d74f6b8bd | 129b10287cbd | 8a812a8ea01b | de86bbc09c48 | d1897b72f57d | d9d024681580 |

The fold's stream property, stated once (the fold alone, 2026-09-26): the committed stream leaves the CPU reference at
plain decode's token on 12 of 12 records (the indices above equal the plain-decode row's) and is bitwise plain decode's
stream on 8 of 12
(`json`, `chat`, `code`, `fact`, `math`, `sky`, `story`, `summary`); on `list`, `multilingual`, `prose` and
`refactor` it leaves at plain decode's token and differs after it (a drafted continuation is not plain decode's).

`--prefill-slab 2048 --mtp 4` under the fold (2026-09-27, one served startup replay on a 1x4 p150 line, the fork
runtime built at `5eac9c778edb`; gate 96/96, hand-off pass): the twelve study records read the fold's `--mtp 4` row
above on both metrics (12/12: no slab fires under 2048 prompt tokens), and the slab record `document` (2228 prompt
tokens, one 2048-row slab, the only pin exercising a slab body under MTP) leaves the reference at token 12 as plain
decode and the wrap do, with a device stream that differs from both after it (digest `8f95abdc2da3`; the wrap's
`9beadb60cc20`, plain decode's `7d56a9706729`).  The plain `--prefill-slab 2048` replay in the same hold read the
landed `A3-slab2048-32k` table 13/13 on both metrics: the fold touches no plain-decode or prefill stream.

| `--prefill-slab 2048 --mtp 4`, fold, 2026-09-27 | json | chat | code | document | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| divergence index | none (96/96) | 2 | 32 | 12 | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| stream bitwise the plain slab table | yes | yes | yes | no | yes | no | yes | no | no | no | yes | yes | yes |

`tools/ci/baselines/A3-slab2048-mtp4-32k-*` were that replay's rows until `gr_recip_last` served by default; since
2026-09-28 both MTP tables are the default set's replays (fold + rings2 + `gr_recip_last`, the paragraph below): the
MTP stream leaves the reference where plain decode does on 11 of 12 study records and later on `math` (61 against
plain decode's 56), the slab record `document` at index 33 with its own stream (the wrap and the fold-only form read
12; the plain `--prefill-slab 2048` table's 12 predates `gr_recip_last`).  Every value measured; none copied.

The fold program itself (2026-09-27, `ttnn/fused/gdn_rows_scan`: the gate scalars of every verify row computed
in one SFPU pass over the a / b tiles and replicated per row by the reader as exact fp32 copies, bitwise the
rows sequential `gdn_step` calls at rows 1..8 on one die): kernel 111.0 -> 85.8 us per GDN layer over the 36
layers and the verify trace's span 38.254 -> 37.098 ms (-1.16 ms per verify pass) in the census on the 1x4 line,
the pins above identical (the 12 rows and the 13-row slab table); on one die 105.9 -> 83.9 us of 48-core kernel
time (the program's device law ~58 + 5.2 x rows us).  The MTP lane's fold-levers development note (2026-09-27,
beside the law tool, not exported) holds the program law, this lever's record and the three levers measured and
dropped (the reader's reads, one pack per state form, the top-face activations).

The fold's lanes form (2026-09-28, `ttnn/fused/gdn_rows_scan` `run_lanes` / `run_pick_lanes`; the MTP lane body's B lanes x
R = k + 1 verify rows lane-major in one tile run the same recurrence on one (value head, lane) item per core with the
single form's compute kernel unchanged, the prefix states `[B R, 12, 128, 128]` and a per-lane pick as the commit; the
served single-stream fold is byte-identical): BITWISE per lane the sequential 1-row `gdn_step` from that lane's state at
every lane shape within the tile (B in 2, 4, 6, 8 x R in 1..5 on one die, every prefix slot and gated row; the pad rows
and the other lanes' rows may hold any finite value and change nothing), so the lanes carry the fold's stream and its
class -- COMPONENT against the chunk chain, exactly as the single form -- instead of the chain's.  One die, 200 traced
replays: 170.9 us per layer at 4 x 5 on 48 cores (198.8 at 6 x 5, 204.9 at 8 x 4; the single form 106.1 on its 48-core
split, 159.6 on the 12-core one), the lane pick 48.0 us; the lanes form is not a verify-time lever (the batched chunk
call it replaces reads about 151 us at B = 4), its lever is the commit (one pick per lane against the batched chunk
re-run).  On the 1x4 p150 line at 4 lanes x k = 4, 32k (2026-09-28, the lane tool's windows: 200 timed passes; the fold ON on both sides): the lanes' refereed passes against the head's B=1 chain are exact where the chain's lanes were not: the twelve STAGE-3 prompts in three batches of four, 3884/3884 refereed lane passes exact and every committed stream equal to the head's B=1 chain's (chat 409/409, code 293/293, fact 275/275, json 229/229, multilingual 248/248, prose 398/398, list 276/276, math 262/262, refactor 292/292, sky 257/257, story 571/571, summary 374/374; the chain fold-on read 26/400, 3/293, 11/275, 27/229 on the first batch), and the 4x4 pass wall is 60.71 / 60.75 / 59.64 ms p50 per batch (p10 58.0-58.9 / p90 62.6-62.8; the blocking probe 64.4-65.0) against the chain's 79.45 / 78.46 / 77.72: the commit replay 9.27 -> 2.17 ms (the picks) and the verify replay 62.20 -> 49.4-49.8 ms (the lanes' chunk chain was the composed one, about thirty programs per GDN layer, where the B=1 stream sat on the wrap), the draft replay equal (8.62-8.72 against 8.69).  The prefix states per bank: 35,389,440 / 70,778,880 / 106,168,320 / 113,246,208 B at 10 / 20 / 30 / 32 lane rows (36 GDN layers, 8 banks; `gdn_rows_scan.prefix_states_bytes`), measured at rows 20 as +70,789,120 B per bank of lane verify state against the chain's; the three lane traces shrink from 7,340,224 to 3,864,512 B per bank.  The lane body dispatches through the fold's registry step exactly as the single stream does (`QWEN38_FUSED_OFF=gdn_rows_scan` restores the chunk chain for both), and allocating the lane verify state allocates the prefix states.  The wave-F development note beside the MTP lane stage notes (2026-09-28, not exported) holds the die proof, the one-die law and the line tables.

The rows of record of the served default set -- the verify-rows fold (`gdn_rows_scan`) with its gate scalars in one
SFPU pass, the MoE rows on two rings (`rings2`) and the gated-residual read's re-associated norm (`gr_recip_last`) --
measured 2026-09-28 on one idle 1x4 p150 line (hold sb-case2-49b8a0f, host load 1.49 / 1.04 / 1.63 at the clients; the fork runtime built at
`5eac9c778edb`, `_ttnncpp.so` build `b7d6011943f3` on this host; the 2026-09-27 rings-on-wrap row was measured on
another host whose kit build of the same source reads `432a48451f4e`), tree `49b8a0f2ea3c`, EOS honoured, 1 warm-up +
median of 3: 560-token chat 68.06 tok/s (2.94 tokens per pass), json 111.40 (4.75; 153 tokens), chat
64.16 (2.68), prose 50.06 (2.09; 218 tokens), code 104.32 (4.45); sampled TPOT 17.46 ms thinking /
16.18 ms non-thinking (6 chats x 200 tokens); the 560-token chat pass p50 41.95 ms pipelined (p10 41.32, p90
42.64, n 175) and 43.28 ms on the blocking probe.  The same hold's startup replays are the pinned tables above
(`A3-mtp4-32k`, `A3-slab2048-mtp4-32k`: every value this replay's).

The pass (measured by the MTP lead on a second 1x4 p150 line at the landing head, 120 + 20 passes per cell,
`--stop-at-end`): the fold removes 2.9-3.1 ms of program time from a k = 4 pass (the commit's masked re-run of the two
chunk prims becomes one pick, 0.91 against 3.13 ms; the verify's six GDN programs per layer become one, -0.55..-0.62;
the draft replay equal) and 0.8-1.3 ms of the pipelined wall on every prompt, since the commit replay already overlapped
the host's PLE lookup: fused verify, greedy, p50 json 47.84 against 49.17 ms, prose 47.23 against 48.24, 560-token chat
49.23 against 50.38 (the all-blocking probe 49.15 / 48.21 / 50.07 against 52.03 / 51.07 / 53.19) on the sealed kit, and
on the kit of record (runs re5-decomp-on / re5-decomp-off) json 47.42 against 48.63, prose 46.69 against 47.78, 560-token
chat 48.76 against 49.73 with the same tokens per pass; split verify, sampled,
json 51.61 against 51.81, prose 50.59 against 51.55, 560-token chat 52.71 against 53.88 (probe -3.01 / -2.91 / -2.89,
the sampled tail unchanged; json's stream identical at 4.750 tokens per pass, prose 2.033 against 2.000 and the chat
2.951 against 2.527 a different sampled continuation of one seeded request).  Served tokens/s move with the stream, not
with the pass: the fold's stream is plain decode's, so a request's first ~96-150 tokens are identical under the fold and
the wrap and then a different continuation with its own draftability follows; no systematic acceptance loss is
attributed (the fold's drafter is not less accurate on its own committed stream than the wrap's on its: `fact`, an
identical 67-token answer, 2.58 against 2.48 tokens per pass, depth-1 acceptance 0.808 against 0.741, one acceptance
apart; `summary`, 3.31 against 2.67 tokens per pass over its own 86- / 64-token answer).  Served rows, EOS honoured
(the lmx-like client, one warm-up and the median of 3, greedy, at most 256 tokens): the kit of record's columns are the
numbers of record (runs re5-rows-on / re5-rows-off, servers `20260926T162840Z-3851428` fold and `20260926T162159Z-3842121`
wrap; the wrap's rows are the README's), the sealed kit's the original measurement:

| served greedy, tokens/s over the answer to its end marker | fold, kit of record | wrap, kit of record | fold, sealed kit | wrap, sealed kit | tokens per pass, fold / wrap |
|---|---|---|---|---|---|
| 560-token chat (256 tokens, finish length: the answer exceeds 256) | 62.52 | 61.30 | 61.70 | 60.55 | 3.08 / 3.12 |
| 177-token multi-turn chat (256, length; all answer) | 57.60 | 57.62 | 56.89 | 57.00 | 2.80 / 2.83 |
| code (256, length; the answer is 283 tokens) | 95.51 | 91.93 | 94.16 | 90.99 | 4.65 / 4.57 |
| json (153 tokens, stop) | 97.27 | 95.25 | 95.74 | 94.02 | 4.75 / 4.75 |
| prose (stop; the fold's answer 203 tokens, the wrap's 196) | 43.42 | 44.73 | 42.87 | 44.28 | 2.09 / 2.17 |
| sampled, ms per token (non-thinking / thinking) | 19.67 / 20.93 | 19.75 / 19.90 | 19.91 / 20.56 | 19.90 / 20.20 | |

The wrap's kit-of-record rows sit 0.7-1.3 tokens/s above the sealed kit's (the MoE exchange pipelining), lengths and
tokens per pass identical; at 256 forced tokens the kit of record reads fold 62.54 / 76.87 / 57.62 / 37.13 / 95.49
against wrap 61.29 / 92.03 / 57.68 / 45.49 / 91.92 (560-token chat / json / chat / prose / code).

At 256 forced tokens (`ignore_eos`, the previous form) the same requests read mixed by stream -- 560-token chat +1.9 %,
code +3.8 %, json -16.7 %, prose -18.4 %; the 12 acceptance records net -2.1 % at 256 tokens and +1.2 % on the 96-token
replay; `summary` -47.5 % where the wrap's post-marker text repeats and drafts at depth-1 0.864 and the fold's does not
(0.370), which never enters a verdict.  The 43-prompt pair in that form is contaminated the same way (sum +3.2 %, paired
mean +1.94 +- 1.49 tokens/s, n = 43); the flip-decision sample is its EOS-honoured re-run: Over 43 distinct prompts -- the 12 acceptance records, the reference corpus's 16 served renders (12 with
thinking on and 4 requests with tools), its 4 evaluation items, the two books' first 256 / 512 / 1024 tokens as raw user
text, and the synthetic chat prompt at 128 / 256 / 560 / 1024 / 2048 rendered tokens -- one greedy request each with EOS
honoured (max 256; 21 of the 43 stop before 256), tokens per pass from the server's pass counters: the fold reads 3005.8
against the wrap's 2947.4 tokens per second summed (+2.0 %), 30 prompts up and 13 down; the paired mean delta is +1.36
+- 0.61 tokens per second (standard error, n 43) and the mean relative delta +2.44 % +- 1.09 %; no part of the set is
negative on average (acceptance records +2.6 %, served renders +1.1 %, evaluation items +1.1 %, synthetic chat +5.9 %,
book prefixes +3.9 %).  Where the two arms' answers are identical the fold's tokens per pass is equal or one acceptance
higher.  The
lmx-like client's `ignore_eos` form matched the localmaxxing reference's 256 output tokens; the reference record does
not say whether EOS was forced.

The sampled law gate under the kernel (the host-fp32 derivation; json / chat / code / story at offsets 0 / 16 / 48,
4,096 seeds, 32 loop requests of 10 tokens) passes on both cards, 12 positions each, loop mismatches / fallbacks /
guard deviations 0 / 0 / 0; the rows-against-plain total variation is 0.0 at the median on both cards, max 0.0253
(thinking) and 0.2227 (non-thinking) where the wrap-era rows of record read 0.0083 / 0.0438.  The non-thinking maximum
is one position, read from the gate's position dump: at `chat` offset 48 the fold's verify row puts the top logit one
bf16 step below the 1-row tail's (20.625 against 20.75, the runner-up equal), 0.029 of probability before truncation;
the non-thinking nucleus cut at 0.8 sits between the two (0.8066 against 0.7772), so the runner-up survives under the
fold and not under plain decode -- the 0.2227 total variation is that one token's mass; the argmax is unchanged, and
the wrap's row equals plain decode's at this position.  The layer that moves the logit is not attributed (the fold's
identity with the 1-row step was measured on GDN layers 01 and 33).

Per GDN layer on the 1x4 line (30 traced replays, quiet host): the verify forward 0.459 -> 0.271 ms and the commit
0.136 -> 0.051-0.058 against the chain (the wrap's 0.306 / 0.110), device operations 58 -> 10 and 12 -> 3.  The fold's
DRAM stays the clause's (k + 1) x 786,432 bytes per GDN layer per device, 141.6 MB per device at k = 4, per drafting chain.

## The MTP pass decomposition (2026-09-25)

Measured on the 4-chip p150 line (a shared host) at the landed head with the chain opened as the server opens it,
`--mtp 4`, the `json` and `prose` acceptance prompts, 120-280 pipelined passes per cell and 20-40 all-blocking probe passes
(every trace replayed blocking on its own: its raw device wall), the device profiler for the per-family split (its per-program
markers inflate a trace by 3-4 %; the raw replay walls are the timing numbers).  The pass wall does not depend on what is
accepted: with the host decision stubbed to accept-all / reject-all the json pass is 62.20 / 62.00 ms against 62.14 normal.

| term of one k = 4 pass (json greedy, ms) | fused verify (`QWEN38_MTP_SAMPLED=0`) | split verify (the default, greedy) | split verify, sampled |
|---|---|---|---|
| commit trace (the previous pass's a*+1 rows; hides the host's PLE lookup) | 4.07 raw, ~1.3 exposed | 4.06 | 4.06 |
| PLE n-gram rows of the k+1 tokens (host; the table is host memory) | 2.8, hidden | 2.7 | 2.7 |
| verify body (48 layers on the 5-row tile, LM head, rows argmax, accept, alignment) | 47.35 raw | head 45.25 + tail 2.79 | same |
| host between head and tail | - | read 0.67, decision 0.02, writes 0.86 | read 0.67, decision 0.6 + 0.8 per row evaluated (4.6 at a* = 4; 0.7 batched), writes 0.84 |
| draft trace (k - 1 rows: embed, mixer, MTP layer, rows-1 MoE, final mixer, LM head, the fused greedy tail) | 7.38 raw = 3 x 2.46 | 7.33 = 3 x 2.44 | 7.33 |
| pass-row readback (268 B, the one host read) | 0.72 | 0.54 | 0.52 |
| pipelined pass wall, p50 [p90] | **60.28** [61.54] | **62.14** [63.40] | **66.92** [68.07] |
| tokens per pass / tok/s over the passes | 4.657 / 76.9 | 4.657 / 74.6 | 4.677 / 70.0 |

Prose: fused 58.92, split greedy 60.86 (2.529 tokens per pass, 41.3 tok/s), split sampled 63.28 (2.583, 40.5).  The verify
body is flat in k inside the rows-5 MoE form (raw head 44.2 / 45.2 / 46.0 / 45.3 ms for 2 / 3 / 4 / 5 rows) and 1.6x the
27.7 ms one-row decode step even at two rows: the rows forms are the cost, not the row count.  Each draft row adds 2.4 ms of
device time (draft trace 0.14 / 2.51 / 4.89 / 7.33 ms for 0 / 1 / 2 / 3 rows); k = 1 / 2 / 3 / 4 give json 1.98 / 2.92 / 3.18 /
4.66 tokens per pass at 53.3 / 56.2 / 60.6 / 62.1 ms (37.0 / 51.5 / 52.4 / 74.6 tok/s).  k = 5 is refused by the MTP DRAM
admission on this head (the 32-row MoE verify form's states).

The verify head by family (device profiler, chip 0 kernel ms of 43.0 over 4,479 programs; 36 GDN layers at 833 us, 12 QSA
layers at 986 us, head epilogue 1.2):

| family | GDN layers | QSA layers | total |
|---|---|---|---|
| MoE compute (156 us per layer at 5 rows against ~65 at 1 row) + the combine collective (58 us) | 7.73 | 2.58 | 10.30 |
| fused programs (gated-residual read/write, router top-k, dispatch, MoE post) | 7.75 | 2.48 | 10.23 |
| dense matmuls (projections, shared expert, router) | 3.95 | 1.63 | 5.57 |
| Chunk-GDN prep + scan (the recurrence over the rows) | 1.87 | - | 1.87 |
| composed glue (binary / unary / reshape / transpose / slice / concat / norms / rope / tilize / KV update / indexer / SDPA) | 8.70 | 5.15 | 13.85 |

One draft row (254 programs, 2.30 ms kernel, 0.18 ms dispatch): LM head 0.54 (eight weight-bound chunk matmuls), the MTP QSA
layer ~0.7, the input mixer / norm chain ~0.6, the rows-1 MoE ~0.33, the final mixer ~0.12, the resolve 0.055 (the fused greedy
tail).  Ceiling at five tokens per pass = 5 / pass wall: fused greedy 82.9 (json) / 84.9 (prose) tok/s, split greedy 80.5 / 82.2,
split sampled 74.7 / 79.0; accept-all realizes 80.0 on json.

k = 5 (`--mtp 5`, admissible since the k-aware DRAM estimate) against k = 4 on the same runtime (2026-09-26, the 4-chip p150
line, 32k): the pass costs 65.4-67.4 ms in the split form (k = 4: 62.1-63.3) and about 68.5 ms served on the fused greedy
form (k = 4: 60.2, from tokens per pass and tok/s), +4 to +8 ms per pass for one more draft row (+2.4 ms of draft trace):
the 6-row verify tile and the 32-row MoE verify form pay the rest.  Served 256-token greedy rows, k = 5 against k = 4: json
75.6 against 75.6 tok/s (5.18 against 4.55 tokens per pass), code 77.1 against 75.7 (5.24 / 4.57), the 177-token multi-turn
chat 45.6 against 47.5 (3.10 / 2.83), the 560-token chat 42.1 against 50.6 (2.88 / 3.12), prose 26.8 against 37.3 (1.76 /
2.20).  k = 4 stays the default; k = 5 is opt-in.  The k = 5 stream is not bitwise with k = 4's (the pass partition moves the
commit chunks and the verify MoE runs the 32-row instance): `json` stays 96/96, `code` holds the reference to 96 (k = 4
leaves at 32), `chat` leaves at 39 (43), `list` 46 (56), `math` 61 (63), `multilingual` 9 (9, a different tail): the
divergence tokens are the reference's #2 or #3 at fp32-oracle margins of 0.09-0.36 logits (chat 39: 0.27, list 46: 0.09,
math 61: 0.36, multilingual 9: 0.11; the k = 4 positions: chat 43 0.04, list 56 0.42, math 63 0.04, code 32 0.42), the
near-tie class above (the "0.4-1.0 logits apart" wording of the 2026-09-25 table holds for list and code; chat and math sit
at 0.04).  Per-depth acceptance
rows measured with a fixed completion budget and no stop ids overstate short-answer classes: `json` reaches `<|im_end|>`
after about 151 tokens and then repeats its answer, so a 600-token json row runs about 75 % in that loop; read the passes
before the first end marker.

### The PLE rows lookup after the reader trim (2026-09-26)

The "PLE n-gram rows" term above is the host's lookup of the k + 1 tokens' rows plus their one upload.  One k = 4 pass reads
80 rows of 320 B from about 62 of the 128 table parts (33 shard files); the lookup was 75 % syscalls: the row reader
re-proved the checkpoint file guard's identity with three `fstat` per touched PART (185 per pass) around 79 WILLNEED and
79 `pread` calls.  The identity is the shard FILE's, so the resident lookup now proves it once per touched file before and
after each batch, advises every distinct row before any is read and reads each once: 344 -> about 130 syscalls per pass, the
same bytes in the same order (`tests/test_ple_resident_lookup_no_device.py` pins fstat = 2 x files touched, fadvise = pread =
distinct rows, every WILLNEED before the first pread, the per-file fail-closed, and the oracle bitwise).  The WILLNEED stays:
a pass whose rows are all cold costs +0.24 ms with it, 3.8 ms without, 8.8 ms through mmap slices.

Measured on the 4-chip p150 line (the fork runtime built at `5eac9c778edb`, the GDN rows fold on, greedy fused, pipelined
p50, json / prose / chat560): the lookup 1.90 / 1.90 / 1.92 -> 1.41 / 1.41 / 1.41 ms per pass, the row write 0.70-0.78
unchanged (the call's own host cost, the same with the device idle), the `ple_rows` segment 2.62 / 2.62 / 2.66 -> 2.20 /
2.18 / 2.16, the pass wall 47.26 / 46.55 / 48.64 -> 46.87 / 46.17 / 48.11 ms, the exposed-host bucket (host work per pass
minus the 0.9 ms commit replay it hides under: the device idle from the commit's end to the verify launch) 1.98 / 2.07 / 2.19
-> 1.50 / 1.69 / 1.64 ms; the wrap arm's segment 2.75 -> 2.25.  The stream is unchanged: the A3 12-record table is identical
to the fold's row with the fold on and passes its pins with it off, tokens per pass identical on every cell.  What remains of
the segment scales with the main thread's core clock: in cells where the core ran at 3.70 GHz the lookup reads 0.70-0.72 and
the write 0.42-0.43 ms (the standalone host numbers), at 1.48-2.6 GHz both about twice that -- the thread sleeps 44 of every
47 ms on the readback and the frequency governor lowers its clock; neither the thread's NUMA node (both 2.15-2.20 ms) nor its
SMT sibling moves it.  The remaining ~1.0 ms/pass is the host's cpufreq policy, a host setting.

### The verify lanes read on a second command queue (2026-09-26)

The fused and the device-decided pass loops know `(a, t', d_1)` when the verify body ends, a draft's length (about 7 ms)
before the pass row the draft lands, but looked every PLE row of the next pass up after that row while the device idled
from the commit's end to the verify launch.  With `QWEN38_MTP_PLE_EARLY` (the server's default with `--mtp`; `=0` opts
out) the chain records an event on the main queue right after the verify launch; right after the draft launch the second
queue waits for it and reads the verify readback row (one buffer read of coordinate 0, 35 fp32 lanes -- nothing else ever
runs on that queue: a program or a trace there takes the sub-device's workers and the main queue's programs are refused,
a buffer read takes no ownership); the host looks the next pass's rows 0-1 up from the context this pass commits to while
the draft runs, and the next step looks rows 2..k up and uploads the R rows once under the commit.  The same rows from the
same contexts (the two-call lookup is the whole lookup row for row, pinned); the lanes are checked against the pass row
every pass.  The host-decided split form keeps its order (its head readback already blocks the host before the tail; a
reader given with it is refused).

Measured on the 4-chip p150 line (the fork runtime built at `5eac9c778edb`, the GDN rows fold on, pipelined p50 over
json / prose / chat560, mode-matched cells): the greedy fused pass 46.83 / 46.24 / 48.22 -> 46.52 / 45.60 / 47.71 ms
(-0.31 / -0.64 / -0.51); with the reader the segments read `early_readback` 37.3 / 36.4 / 38.2 (the second queue's wait for
the verify), `ple_rows_early` 0.83 (rows 0-1, hidden under the draft), `readback` 6.0-6.3, `ple_rows_late` 1.81-1.89 (rows
2..k plus the upload, on the critical path) against `ple_rows` 2.22 without it: the two calls cost 2.64-2.72 together,
+0.45 ms of their own in this clock mode (+0.12 standalone).  The device-decided sampled pass -0.71 / -0.72 ms (json /
prose).  The probe replays (verify, draft, commit) are identical with and without the second queue: the two-queue open costs
nothing on the device.  Bitwise: the A3 12-record table identical to the fold's row with the fold on and passing the pins
with it off, the server recording the switch; a 2,179-pass soak with the lanes asserted every pass, no hang, a clean
close.  The plain decode step (no pass loop) 25.43 against 25.61 ms, untouched.  Clock caveat: these are mode-matched
cells; a cell whose main thread runs in the boosted clock mode reads 1.5-2 ms lower on both arms (the section above), and a
process confined to the core pair of the busiest completion-queue reader thread kept its main thread at 3.70 GHz on every
cell, the pass 45.44 / 44.56 / 46.57 ms (-1.4 / -1.6 / -1.5 against unpinned): a launch-time placement under study, not
code.

## Served lanes (`--lanes 4 --mtp 4`), 32k and 128k, 2026-09-28

The lanes server (`docs/SERVER.md`, "Several requests at once") serves up to B greedy requests at once through the MTP
lane chain (`ttnn/mtp_lanes.py`: B lanes verify k + 1 rows each in one tile).  The verify-rows fold serves the lanes
through its lanes form (`ttnn/fused/gdn_rows_scan`), so a `--lanes` process runs the served default set and a lane's
stream is the DEFAULT single-stream server's.  (Before the lanes form landed the lanes ran the chunk chain and a
`--lanes` process turned the fold off: on that form the lanes matched a fold referee for 3-27 passes and then diverged on
every prompt -- the fold is COMPONENT class -- and matched a wrap referee on every pass; the fold-off rows are in the
dev note.)

**The pin table `A3-lanes4-mtp4-32k` (`tools/ci/baselines/`), the lanes' startup replay: the twelve records through the
lane scheduler four at a time (each admitted into a lane while the others decode, queue waits 0-11 s), against the CPU
records and the single-stream replay of the same process.**  12/12 streams equal the single-stream replay, `json`
96/96; the same twelve streams at 128k (bitwise the 32k replay's: the lane arithmetic carries no context term);
identical on a second start.  Divergence index against the CPU record and the stream sha256 (first 12): the fold's streams (12 of 12 equal the `--mtp 4` pin's sha).

| lanes 4, k=4, 2026-09-28 | json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| divergence index | none (96/96) | 56 | none (96/96) | 15 | 56 | 61 | 9 | 13 | 24 | 19 | 6 | 75 |
| lane stream sha256 (first 12) | a5b4defa72fa | 1bbbf060a22e | 159de7287947 | 9afd57cc0303 | 8253ba344696 | 19f55ccb4bb2 | 8b3efd744238 | 19538e64128d | 8a812a8ea01b | de86bbc09c48 | efaf8f20e25c | d9d024681580 |
| stream equals the single-stream replay of the same process | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |

**Served gates (a 1x4 p150 line, 32k, both servers `--mtp 4` greedy on the served default set, the same request bodies; Exact: 11/11 rows byte for byte (the four acceptance prompts at once, the fifth, 4 x chat560 at once, a stop-string row, a `max_tokens` 40 row); the forced `</think>` rows equal through the reasoning and the forced token 6/6; the lanes' own determinism 8/8 repeats byte-identical).**
Exact: the four acceptance prompts `chat` / `code` / `json` / `prose` served at once over their natural segments
(EOS honoured, `max_tokens` 512) and a fifth request behind them, a stop-string row and a `max_tokens` 40 row: byte for
byte the single stream's reply (reasoning, content, finish reason) on every row (11 of 11).  Tolerance, the forced
`</think>` of the thinking budget: budgets 3 / 5 / 6 / 7 / 9 / 12 on the 177-token chat record (the budget landing inside a
pass at 3 and 7, on a pass's last emitted token at 5, 6, 9 and 12) equal the single stream's reply through the reasoning
and the forced token on 6 of 6, byte-identical to the end on 3 (budgets 3, 5, 6) and parting 611-1023 characters into
the answer on the other 3 (12, 7, 9; on the wrap form of the interim record 0 of 6 held to the end and the parts came at
17-331 characters): the single stream feeds the forced token
(and the token before it in the last-token case) through its 1-row decode traces, a lane through the verify body, and
the rows path is not bitwise with the 1-row path (this section's MTP-vs-plain rows); the same budgeted request twice on
the lanes is byte-identical.  A second single-stream seam surfaced by these rows: the single stream served a
`max_tokens` 40 row from its restored prompt-end snapshot (the previous request had the same prompt, `prefill_tokens`
1) with a different 40th token than a fresh prefill of the same prompt gives (`crowd levels` / `crowd flow`); the
comparator therefore prefills fresh before every event row (a one-token request on another prompt displaces the
snapshot), and the restored form is recorded as a single-stream item of its own.

**DRAM (per bank, the allocator's view, MEASURED at the lanes server's start; the admission
`lanes_capacity_admission` decides on the live reading at the chain's warm hook, before any capture, and READY is refused
when the measured growth exceeds the estimate).**

| 4 x 4 lanes (the fold's lanes form; 20 rows of prefix states) | 32,768 | 131,072 |
|---|---|---|
| free at the hook (before the lane states) | 1,652,521,152 | 1,473,018,048 |
| estimate required (lane states + pagers + the fold's prefix states 70,778,880 + MoE rows instances, +7 % slope, 10 % margin; + traces) | 451,965,918 | 1,336,834,040 |
| reserved for the chain's own captures | 74,138,023 | 74,138,023 |
| MEASURED growth: lane states + pagers + prefix states / the three lane traces | 393,809,536 / 3,375,424 | 1,151,192,704 / 3,375,424 |
| free after every capture (the served headroom) | 1,242,104,320 | 305,185,280 |
| lane pass p50 in the startup replay (107 passes, 4 lanes) | 58.7 ms | 67.0 ms |
| READY, warm caches (lanes: allocate 2.7, imports 1.8, eager passes 5.9, captures 1.0 s) | 135 s | 176 s |

**Served rows of record (the same hold, the 1-min host load stated in the table; the lanes' per-user rate is
0.55-0.7x the single stream's: the 4-lane pass wall is 58.7 ms in the replay (67.0 at 128k) against the single stream's
46 -- the 20-row verify body -- and every other lane pauses for an admission's prefill, so a lane commits the same
tokens per pass at a lower pass rate while the four together commit 2.37x the single stream's tokens on the full
batch; the levers left -- a prefill beside the passes, the 20-row body -- are later waves').**

| row (`--lanes 4 --mtp 4`, 32k, the served default set; the 1-min host load per row group: the 4 x chat560 rows from the 23:11Z hold at 1.35-1.42 on the lanes rows and 2.1-2.3 on the comparator's (its own startup's tail), the other rows from the 22:08Z hold at 2.7-4.0 on the comparator rows and 3.1-7.0 on the lanes rows with another session's partition validation alongside) | lanes (4 at once) | single stream (the default server, in turn) |
|---|---|---|
| 4 x chat560 (thinking off, EOS honoured; 2048 tokens): aggregate | **134.0 tok/s** (2048 / 15.28 s wall; 147.2 from the first token) | 56.5 tok/s (2048 / 36.27 s) |
| 4 x chat560: per request | 36.9 / 40.8 / 45.7 / 51.9 tok/s; 177 passes each = 2.89 tok/pass | 67.1 / 67.2 / 67.3 / 67.7 |
| 4 x chat560: TTFT | 1.37 / 2.70 / 4.04 / 5.44 s (admissions 1.34-1.35 s each = prefill 1.19-1.20 + evict 0.05 + import 0.10-0.11, one per boundary) | 1.21 / 1.21 / 1.21 / 1.24 s |
| 4 x chat560: stalled by the others' admissions | 0.00 / 1.35 / 2.69 / 4.03 s | -- |
| mixed batch chat / code / json / prose (natural, EOS honoured): per user | 40.3 / 47.5 / 77.2 / 26.6 tok/s (194 / 71 / 32 / 105 passes) | 63.0 / 103.9 / 111.7 / 50.2 |
| mixed batch: aggregate over the batch wall | 83.5 tok/s (1194 / 14.30 s; the short answers end early) | 64.7 (1194 / 18.45 s in turn) |
| mixed batch: TTFT | 1.62 / 0.55 / 2.10 / 0.97 s | 0.54 / 0.28 / 0.34 / 0.29 s |
| the fifth request (chat, 2 s behind the four) | queue_wait 2.04 s, TTFT 2.73 s, 44.7 tok/s | -- |
| startup: READY / lane pass p50 in the replay | 135 s (warm caches) / 58.7 ms over 107 passes | -- |
| 128k: 4 x chat560 aggregate / per request / TTFT | 112.4 tok/s (2048 / 18.23 s); 31.1 / 34.7 / 39.1 / 44.7; TTFT 1.67 / 3.36 / 5.07 / 6.79 s (admissions 1.65-1.70 s) | -- |
| 128k: READY / lane pass p50 in the replay / free after captures | 176 s / 67.0 ms / 305,185,280 B per bank (growth states 1,151,192,704, traces 3,375,424) | -- |

The single stream's TTFT for the 560-token chat prompt is 1.21 s; a lane's is 1.4-5.4 s in a four-request burst: the
admissions serialise (one per pass boundary while a lane decodes; 1.34 s each = prefill 1.19 + evict 0.05 + import
0.10) and every other lane pauses for them -- the honest cost of admission in that wave; the next section is the
2026-09-29 stage that cut it.

### The admission in segments, the 128-row chunks as the served default (2026-09-29)

Two changes (`docs/SERVER.md`, "Several requests at once"): the chunked prefill runs 128-row chunks where the prompt
allows for every server form (the rows of record above ran both servers on 32-row chunks alone, `READY.mtp.admission`
`long_chunks false`; the 128-row chunks are bitwise the 32-row path, the section "`--long-chunks`" above), and an
admission runs in segments with the decoding lanes' passes between them under `--lanes-stall-budget` (the chunk driver
also prepares the next chunk's host inputs under the running replay for every caller, so only the first chunk's
preparation is exposed at any event cadence).  The pins, in the `--lanes 4 --mtp 4` process of the first hold at the new
head (a 1x4 p150 line, 32k, the served default set, host 1-min load 8-16 -- the gates are load-independent, the rows
are not): the single-stream replay reads the `A3-mtp4-32k` sha table 12/12 and `json` 96/96 (the divergence row above,
unchanged); the lanes replay under the interleave (budget 0.25 s, 13 interleaved passes over the 12 admissions, 111
passes, pass p50 59.9 ms) 12/12 lane streams equal the single-stream replay; the lanes' DRAM growth states
393,805,440 + traces 3,375,424 B per bank (the 2026-09-28 record 393,809,536 / 3,375,424), free after every capture
1,209,590,016 (before 1,242,104,320: the 128-row chunk state and trace, 32.5 MB), the estimate 451,965,918 holds,
program cache 828 -> 1272 across the captures and none after; READY 502 s (the first start at the head: the kernels
compiled).

| admission at 32k (prefill / evict / import, s; the startup replay, load 8-16) | tokens | 32-row chunks (2026-09-28) | 128-row chunks |
|---|---|---|---|
| `sky` / `code` / `fact` / `story` / `list` / `prose` / `multilingual` | 37-58 | -- | 0.27-0.29 / 0.05 / 0.09-0.11 |
| `json` | 85 | -- | 0.344 / 0.056 / 0.103 |
| `math` | 99 | -- | 0.385 / 0.047 / 0.101 |
| `chat` | 177 | -- | 0.455 / 0.051 / 0.101 |
| `summary` | 221 | -- | 0.542 / 0.048 / 0.098 |
| `refactor` | 434 | -- | 0.755 / 0.054 / 0.102 |
| the 560-token chat prompt (the served batch) | 560 | 1.19-1.20 / 0.05 / 0.10-0.11 | 0.90-0.96 / 0.05 / 0.10 (load 8; the quiet-host row below) |

The served byte gate at the new head (the same smoke set, the `--lanes 4 --mtp 4` server at the default budget against
the 2026-09-28 comparator's replies, the default single stream on 32-row chunks; host load 10-21): 11/11 rows byte for
byte (the four acceptance prompts at once, the fifth, 4 x chat560 at once, the stop-string and `max_tokens` 40 rows),
the forced `</think>` rows 6/6 through the forced token and parting afterwards on the same three budgets at the same
characters as before (the 1-row seam above, untouched), the lanes' own repeats 8/8; at 128k the same process form
reads the shas 12/12 and the lanes pin 12/12 (15 interleaved passes, pass p50 66.5 ms), the admission fits with 6.3 MB
per bank of headroom at the hook and 269,504,768 B per bank stay free after every capture (305,185,280 before the
128-row state).

**The stall budget sweep (RELATIVE, host 1-min load 11-62: other sessions' test sets and two masked full sets of
this branch ran alongside; the four settings interleaved in two rounds so the drift hits them alike; each hold its own
`--lanes 4 --mtp 4` server at 32k on a 1x4 p150 line, the same request bodies).  CAVEAT: the sweep ran BEFORE the
budget clock's fix (the landed form counts the admission work from the decoding lanes' last pass, or the first lane's
activation after an idle boundary; the sweep's clock restarted per admission), so the natural batch's pauses in this
table are the old clock's (a boundary of four short admissions never reached 0.5 s and the first lane waited the whole
boundary: 0.96-1.07 s at 0.5 s); the rows of record below run with the fix and supersede them; the burst rows (one
admission of 1.1 s or more at a time) are the same under either clock.**  4 x chat560 at once (2048 tokens), the natural batch chat / code / json / prose at once
with a fifth chat 2 s behind, and the stall row: three chat560 at once and a 4,031-token prompt 3 s later.

| budget (round a: load 26-62; round b: load 11-34) | 4 x chat560 aggregate a / b | TTFT by arrival (b) | per request (b) | the decoding lanes' longest pause (b) | natural batch aggregate a / b, TTFT (b) | fifth TTFT (b) | the 4k prompt: its admission wall / TTFT, the three lanes' longest pause (b) |
|---|---|---|---|---|---|---|---|
| `off` (whole admissions, the 2026-09-28 form) | 142.6 / **144.6** tok/s | 1.08 / 2.15 / 3.24 / 4.39 s | 39.3 / 42.8 / 47.1 / 52.3 | 1.1-2.2 s (a: 3.5) | 79.6 / 83.2; 0.48 / 1.03 / 1.52 / 2.19 | 2.63 s | 5.51 s / 5.83 s; **5.56 s** |
| 0.5 s (the default since this stage) | 130.3 / 133.8 | 1.11 / 2.25 / 3.52 / 4.71 | 36.8 / 39.9 / 43.8 / 48.2 | 0.62-0.64 s (1-2 interleaved passes per admission) | 79.5 / 82.5; 0.55 / 1.02 / 1.44 / 2.19 | 2.73 | 6.06 / 6.63 (9 interleaved passes); 0.76 |
| 0.25 s | 121.8 / 130.5 | 1.07 / 2.40 / 3.74 / 5.03 | 36.8 / 39.9 / 43.6 / 47.9 | 0.40-0.44 s (3 per admission) | 72.3 / 82.8; 0.57 / 1.22 / 1.77 / 2.32 | 2.76 | 6.54 / 7.23 (17); 0.40 |
| 0 (a pass at every segment) | 121.9 / 127.1 | 1.11 / 2.56 / 4.01 / 5.43 | 36.7 / 39.7 / 43.4 / 47.8 | 0.34-0.43 s (6 per admission) | 74.5 / 82.4; 0.78 / 1.38 / 2.02 / 2.66 | 3.06 | 7.49 / 8.51 (33); 0.32 |

What the table says: the device work is the same at every setting, so the interleave costs exactly the passes it
inserts -- on a four-burst the later admissions' passes serve one to three lanes and the batch ends one pass later per
inserted pass (`off` 14.16 s against 15.31 s at 0.5 and 15.70 s at 0.25: -8 % and -10 % aggregate), and a long prompt's
first token waits one pass per inserted pass (+0.8 s at 0.5, +1.4 s at 0.25, +2.7 s at 0) -- while it bounds what the
decoding lanes see: a pause of budget plus one 128-row segment (0.6-0.8 s at 0.5, 0.4 s at 0.25) instead of the whole
admission (1.1 s per 560-token prompt, 5.6 s for the 4k prompt).  The natural batch (short prompts, admissions of
0.43-0.66 s) is within noise across the settings.  The default is 0.5 s: half the interleave cost of 0.25 s for a
pause under a second; `--lanes-stall-budget off` is kept for an operator who wants the burst aggregate and accepts
multi-second freezes, `0` for the shortest pauses.  The rows of record at the default (a quiet host) follow.

**The rows of record of this stage (2026-09-29, a 1x4 p150 line at 32k, both servers `--mtp 4 --no-sampling` greedy on
the served default set, 128-row chunks, the fold on; ONE host and one session, the comparator first; the 1-min host load
per row group stated, the servers' own load included: the comparator 0.8-2.6, the lanes 0.9-1.6; the lanes at the default
budget 0.5 s, two runs).**  The byte gate: 13/13 rows equal the comparator's replies on both runs (the four acceptance
prompts at once, the fifth, 4 x chat560 at once, the three chat560 of the stall row and the 4k prompt).

| row (`--lanes 4 --mtp 4`, budget 0.5 s) | lanes (4 at once), run 1 / run 2 | single stream (the default server, in turn) |
|---|---|---|
| 4 x chat560 (thinking off, EOS honoured; 2048 tokens): aggregate | **132.2 / 131.3 tok/s** (2048 / 15.49 s; 15.60 s) | 57.6 tok/s (2048 / 35.53 s) |
| 4 x chat560: per request (by arrival) | 36.5 / 39.4 / 42.9 / 47.1 (36.1 / 39.0 / 42.4 / 46.4); 177 passes each | 66.7 / 66.8 / 66.7 / 66.7 |
| 4 x chat560: TTFT | 1.10 / 2.27 / 3.47 / 4.65 s (1.07 / 2.23 / 3.42 / 4.59) | 0.91 / 0.91 / 0.91 / 0.91 s |
| 4 x chat560: the admission | 1.04-1.09 s of segments (prefill 0.90-0.95 + evict 0.06 + import 0.08), the first whole, the next three with one interleaved pass each (wall 1.10-1.14 s) | -- |
| 4 x chat560: the decoding lanes' longest pause / stalled | 0.61-0.62 s; 3.20 / 2.14 / 1.06 / 0 s (3.17 / 2.13 / 1.04 / 0) | -- |
| mixed natural batch chat / code / json / prose (EOS honoured): per user | run 1: 37.7 / 46.6 / 74.1 / 28.1 tok/s; run 2: 36.7 / 60.3 / 52.8 / 28.3 (the arrival order differs between the runs) | 62.2 / 102.3 / 110.4 / 49.2 |
| mixed batch: aggregate over the batch wall | 80.5 / 81.8 tok/s (1194 / 14.83 s; 14.60 s) | 63.5 (1194 / 18.81 s in turn) |
| mixed batch: TTFT chat / code / json / prose | run 1: 1.26 / 0.60 / 2.30 / 1.77 s; run 2: 0.68 / 2.08 / 1.13 / 1.61 (one admission of 0.41-0.62 s per boundary, none reaching the budget; the longest pause 0.54-0.61 s) | 0.50 / 0.28 / 0.36 / 0.28 s |
| the fifth request (chat, 2 s behind the four) | TTFT 3.06 / 2.71 s, 43.1 / 43.0 tok/s, one interleaved pass | -- |
| the stall row: three chat560 at once, a 4,031-token prompt 3 s later | the long admission 5.49 s of segments (prefill 5.35 = 1.33 ms per prompt token, evict 0.06, import 0.08) as 8 interleaved passes, wall 5.95 s, its TTFT 6.35 / 6.42 s and 54.2 tok/s after; the three decoding lanes' longest pause 0.72-0.80 s (a 5.5 s freeze on the 2026-09-28 form), their stalled 7.57 / 6.53 / 5.49 s, 27.1-30.5 tok/s over their answers | the 4k prompt's TTFT 5.34 s (about 760 prompt tokens per second) and 73.8 tok/s after; the three chat560 in turn 66.6-66.8 |
| startup: READY (warm caches) / the pin | 142-151 s; 12/12 lane streams = the single-stream replay = the shipped shas, 4 interleaved passes in the replay, pass p50 61.0 ms | 370 s (the first `--no-sampling` start at the head) |

Against the 2026-09-28 rows of record (32-row chunks, whole admissions): the burst's aggregate 134.0 -> 132.2 tok/s (-1.3 %:
the 128-row chunks' 0.28 s per admission bought back by the interleave's passes), its TTFTs 1.37 / 2.70 / 4.04 / 5.44 ->
1.10 / 2.27 / 3.47 / 4.65 s, the other lanes' pause per admission 1.35 -> 0.62 s, stalled 4.03 / 2.69 / 1.35 -> 3.20 /
2.14 / 1.06 s; the natural batch 83.5 -> 80.5-81.8 (the same host-noise band as its 2026-09-28 rows at load 3-7); the
fifth 2.73 -> 2.71-3.06 s.  The single stream's own rows moved with the 128-row default: the 560-token chat prompt's TTFT
1.21 -> 0.91 s (-25 %), the natural rows 0.54 / 0.28 / 0.34 / 0.29 -> 0.50 / 0.28 / 0.36 / 0.28, the decode 67.1-67.7 ->
66.7-66.8 tok/s (the same, within noise).

`--lanes-stall-budget off` on the same host in the same session (04:35-04:39Z, load 0.8-1.6; the byte compare 13/13):
4 x chat560 134.3 tok/s (2048 / 15.25 s), per request 36.6 / 39.5 / 43.1 / 47.3, TTFT 1.09 / 2.20 / 3.34 / 4.45 s,
admissions 1.05-1.09 s whole, the decoding lanes' longest pause 1.10-1.15 s (the admission itself), stalled 3.19 / 2.14 /
1.05 / 0; the natural batch 82.0 tok/s, TTFT 0.62 / 1.21 / 1.62 / 2.10, the longest pause 1.54 s (json's boundary); the
fifth TTFT 2.68 s; the stall row: the 4k prompt's admission 5.46 s whole, its TTFT 5.76 s, the three decoding lanes
frozen 5.51 s.  Against the default on the quiet host the interleave costs 1.5-2.3 % of the burst's aggregate (the loud
sweep's 8 % carried the host load's share of every inserted pass's host work) and 0.6 s of the 4k prompt's first token,
for pauses of 0.6-0.8 s instead of 1.1-5.5 s.

128k (`--allocated-context 131072`, the same host and session, 04:39-04:45Z, load 1.0-1.7; READY 216 s, the pin 12/12,
growth states 1,151,188,608 + traces 3,375,424 B per bank, free after every capture 269,504,768): 4 x chat560 109.7 tok/s
(2048 / 18.67 s; 112.4 on 2026-09-28 with 1.65-1.70 s freezes), per request 30.7 / 33.7 / 37.3 / 41.8, TTFT 1.43 / 3.09 /
4.76 / 6.44 s (1.67 / 3.36 / 5.07 / 6.79), admissions 1.42-1.48 s of segments (prefill 1.00-1.01 + evict 0.19 + import
0.24) with two interleaved passes each after the first (the evict and import segments carry the 128k state), the
decoding lanes' longest pause 0.69-0.70 s, stalled 4.42 / 2.95 / 1.48 / 0; the natural batch 68.5 tok/s, TTFT 0.94 /
1.84 / 2.61 / 3.48 s, pauses under 0.84 s; the fifth TTFT 4.50 s, 38.5 tok/s; the stall row: the 4k prompt's admission
6.35 s (prefill 5.86 = 1.45 ms per prompt token at 128k) as 11 interleaved passes, its TTFT 8.91 s, the three decoding
lanes' longest pause 0.70 s.

256k (`--allocated-context 262144`, the single-stream server, no `--mtp`; the same host, 04:45-04:49Z): READY after 180 s
(the first 256k start on that host converted its caches inside the start), the 32-row and 128-row chunk traces
captured (the 128-row capture 2.36 s), the acceptance replay 12/12 with `json` 96/96, free after every capture
1,299,216,512 B per bank -- the 128-row default fits the largest context with the compact expert layout (the
2026-09-06 table's 94 MB per bank at 256k predates that layout).



## The device sampler's law (2026-09-25)

The on-device sampler (`sampler_tail`, one program on one core after the top-32 candidate row) is gated on its law, not
on tokens: at teacher-forced prefixes of the acceptance prompts (24 positions: chat, code, json, story at offsets 0, 16,
48, both model-card profiles) the tool draws 4096 device tokens per position with a fresh seeded stream and compares
their counts with the host sampler's law over the same candidate row (`sample_candidates` before its draw) by the
sampled-MTP lane's method: the G statistic with cells under five pooled and alpha 0.01 split over the positions
(Bonferroni, 0.000417 each), the total variation of the empirical law against the host law beside its null expectation,
the device sampler's own law (its draw intervals over the 2^24 uniform grid, exact) against the host law, and every
device token against the integer-exact host reference (`device_sampler_reference`, 0 mismatches required).  The presence
penalty runs on the device from the request's own history.  Verdict 2026-09-25: PASS: every position passes the G-test
(minimum p 0.0740), the largest empirical TV is 0.0079, the largest device-law TV is 0.0000022, mismatches 0.

The candidate row itself (`candidate_row`, 2026-09-25) is folded into greedy_tail's scan and merge: each scan core keeps
its sorted top-32 behind the argmax's compare (a stable insertion, so ties keep the lowest ids and the list's first
entry is the argmax pair; the greedy outputs are unchanged), and the merge takes the shard's 32 from the cores' lists in
core order and writes the fp32 row [values | global ids] the row's all_gather takes: the row's 10 programs and 312 us
per step become 2 and 8 (the gather and the copy) while the scan and merge grow 66 -> 128 us (net -242 us on the tail),
the sampled step 27.57 -> 27.37 ms. The values are bitwise the chain's (ttnn.topk's) and the ids equal above the kth
value; the boundary group is the lowest ids among the ties, where ttnn.topk's pick is implementation-defined (the
candidate row's agreement rule tolerates boundary ties). `QWEN38_FUSED_OFF=candidate_row` restores the chain's row.

Profile `thinking`:

| prompt | offset | kept lanes | G | df | p | TV empirical vs host | TV null expectation | TV device law vs host | draws | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| json | 0 | 2 | 0.04 | 1 | 0.8381 | 0.0008 | 0.0030 | 0.0000010 | 4096 | 0 |
| json | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| json | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| chat | 0 | 5 | 2.72 | 4 | 0.6060 | 0.0079 | 0.0095 | 0.0000022 | 4096 | 0 |
| chat | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| chat | 48 | 2 | 3.19 | 1 | 0.0740 | 0.0046 | 0.0020 | 0.0000004 | 4096 | 0 |
| code | 0 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| code | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| code | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| story | 0 | 3 | 1.70 | 2 | 0.4273 | 0.0056 | 0.0057 | 0.0000012 | 4096 | 0 |
| story | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| story | 48 | 3 | 1.82 | 2 | 0.4030 | 0.0049 | 0.0075 | 0.0000009 | 4096 | 0 |

Profile `instruct`:

| prompt | offset | kept lanes | G | df | p | TV empirical vs host | TV null expectation | TV device law vs host | draws | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| json | 0 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| json | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| json | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| chat | 0 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| chat | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| chat | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| code | 0 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| code | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| code | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| story | 0 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| story | 16 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |
| story | 48 | 1 | 0.00 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000000 | 4096 | 0 |

The instruct card where its policy keeps two or more lanes, so the draw is a real one: the tool walked each record's
offsets 1..95 reading the host law at every prefix and took the first 4 offsets per prompt with at least 2 kept lanes
(12 positions over chat, code, story), 4096 draws each, alpha 0.01 split over the 12 positions (0.000833 each).  Verdict
2026-09-25: PASS: G-test 12/12 (minimum p 0.1427), largest empirical TV 0.0119, largest device-law TV 0.0000011,
mismatches 0.

| prompt | offset | kept lanes | G | df | p | TV empirical vs host | TV null expectation | TV device law vs host | draws | mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chat | 1 | 2 | 0.00 | 1 | 0.9695 | 0.0003 | 0.0060 | 0.0000001 | 4096 | 0 |
| chat | 2 | 2 | 2.15 | 1 | 0.1427 | 0.0110 | 0.0060 | 0.0000001 | 4096 | 0 |
| chat | 3 | 3 | 1.22 | 2 | 0.5423 | 0.0070 | 0.0079 | 0.0000011 | 4096 | 0 |
| chat | 8 | 2 | 0.39 | 1 | 0.5320 | 0.0049 | 0.0062 | 0.0000000 | 4096 | 0 |
| code | 24 | 2 | 1.38 | 1 | 0.2396 | 0.0076 | 0.0052 | 0.0000011 | 4096 | 0 |
| code | 26 | 2 | 0.13 | 1 | 0.7211 | 0.0024 | 0.0054 | 0.0000006 | 4096 | 0 |
| code | 32 | 2 | 0.13 | 1 | 0.7152 | 0.0028 | 0.0062 | 0.0000005 | 4096 | 0 |
| code | 40 | 2 | 0.02 | 1 | 0.8773 | 0.0012 | 0.0060 | 0.0000004 | 4096 | 0 |
| story | 1 | 3 | 2.32 | 2 | 0.3141 | 0.0119 | 0.0085 | 0.0000005 | 4096 | 0 |
| story | 2 | 2 | 1.96 | 1 | 0.1612 | 0.0105 | 0.0060 | 0.0000004 | 4096 | 0 |
| story | 5 | 2 | 0.21 | 1 | 0.6466 | 0.0030 | 0.0052 | 0.0000011 | 4096 | 0 |
| story | 6 | 2 | 1.49 | 1 | 0.2216 | 0.0092 | 0.0060 | 0.0000004 | 4096 | 0 |

With `QWEN38_MTP_DEVICE_ACCEPT=1` an `--mtp --sampling` server decides its passes on the device under the same law
(`mtp_accept`, one program on one core between the verify head and its tail: row `j` accepts draft `d_{j+1}` iff
`fl32(u_j S_j) < w_j(d_{j+1})` over the table weights, a tie rejects, the first rejection and the bonus row draw with a
second uniform, no division); the response's `mtp_acceptance_arithmetic` names `device-theta` or `host-fp32` (the
host-decided pass, the host sampler's fp32 law) and the fingerprint carries `-device-accept`: one seed reproduces one
stream per arithmetic; the law gate runs per arithmetic (2026-09-26).

## Image prompts: a tolerance class against the CPU reference (2026-09-29)

An image prompt's device stream is compared with the CPU stream of `Qwen4ExpForConditionalGeneration` (transformers
5.16.1, bf16, the same processor render and pixels, greedy) by two quantities that are not the same: the divergence
index -- the first token where the device leaves the CPU's free-running greedy stream, as the text records are judged --
and the teacher-forced ranks and gaps -- the CPU model re-run on the prompt plus the device's own tokens, and at every
position the rank of the device token in the CPU distribution and the CPU log-prob of the CPU argmax minus the device
token's, in nats.  A near-tie can leave one stream at index 2 and another at 6 while every device token stays the CPU's
first or second choice.  The three-axis positions and the delta are exact against the reference on every record; the
tower's features against the FP32 reference are the tower's own record (PCC 0.995-0.998, mean absolute difference 0.002
on these fixtures).

The class table (one 1x4 p150 line, 32k, chunked prefill, greedy, 32 tokens per fixture; the fixtures are the
deterministic synthetic images of `tests/fixtures/vision_fixture.py` and `models/sample_data/demo.jpeg`, each asked to
describe the image in one sentence with thinking off):

| fixture | image tokens | features | device tokens in the CPU top-1 / top-2 / top-5 | worst rank | max gap (nats) | mean gap | divergence index |
|---|---|---|---|---|---|---|---|
| synthetic 512 x 512 | 256 | CPU tower | 29 / 32 / 32 | 1 | 0.375 | 0.027 | 6 |
| synthetic 384 x 640 | 240 | CPU tower | 26 / 30 / 32 | 3 | 2.0 | 0.164 | 5 |
| synthetic 1024 x 1024 | 1,024 | CPU tower | 29 / 30 / 32 | 2 | 1.5 | 0.106 | 1 |
| demo.jpeg 2048 x 1365 | 2,752 | CPU tower | 30 / 32 / 32 | 1 | 0.5 | 0.023 | 11 |
| synthetic 512 x 512 | 256 | device tower | 22 / 25 / 27 (EOS at 27) | 3 | 1.75 | 0.171 | 6 |
| synthetic 384 x 640 | 240 | device tower | 25 / 30 / 32 | 3 | 1.25 | 0.191 | 2 |
| synthetic 1024 x 1024 | 1,024 | device tower | 29 / 32 / 32 | 1 | 0.875 | 0.059 | 2 |
| demo.jpeg 2048 x 1365 | 2,752 | device tower | 30 / 32 / 32 | 1 | 0.5 | 0.023 | 11 |

The class bound: every device token inside the CPU top-5 and no gap above 2.0 nats.

The pins (`tools/ci/pins.json`, configuration `vision-chunked-32k`; `tools/ci/baselines/A3-vision-chunked-32k-*.json`)
are seeded from the served form with the device tower's features -- chunked prefill through the 32-row chunk trace, on
which the 128-row form is bitwise -- per fixture: the divergence index (6, 2, 2, 11) under the text records' rule
(`not_earlier`: it may not move earlier), the stream's token sha (`exact`, information: a changed stream past its index
is reported, not failed), and the class bound as a hard floor (`score_in_top5`, `score_max_gap`), judged by the
teacher-forcing pass of the CPU reference tool.  Comparisons against an earlier run or against the CPU tower's features
are information in the record.

The forms (the same fixtures; every hold under six minutes): `--long-chunks` gives the four image streams bitwise the
32-row form's; `--mtp 4` gives them bitwise too, with 2.3-2.6 accepted tokens per pass against the chat control's 2.77
on the same chain (about 90 %); `--prefill-slab 2048` gives three bitwise and leaves the 32-row stream on demo.jpeg at
token 14 (the slab's tolerance class; scored 30 / 32 / 32, worst rank 1, max gap 1.0).  Time to first token of an image
prompt is the same-length text prompt's or less on every form (0.55 vs 0.59 s at 277 tokens and 3.72 vs 3.75 s at 2,773
tokens with `--long-chunks`; 1.81 vs 1.87 s with the slab), the tower's own time apart (40-45 ms at 256 image tokens,
156-184 ms at 1,024, 643-653 ms at 2,752, beside the live chain).

Text requests on the same build are unchanged: the fused position derive at a zero shift is bitwise the previous rows
(194 device cases), the acceptance control leaves the CPU record at its pinned index before and after the image requests
in every form, and the previous head's chat stream is the same stream token for token; a text chunk reads the resident
table's rows bit for bit, the feature buffer's text lanes hold -0.0 (the identity of the add) and the decode shift is
zero.

The lanes (`--lanes 4 --mtp 4`, one 1x4 p150 line, 32k, 2026-09-29): the four fixtures at once on four lanes, their
same-length text prompts and a mixed wave of two text and two image requests reply byte for byte as the single stream
(19 of 19 rows against the single-stream comparator; the lanes' startup replay 12 of 12).  An image admission is a text
admission of the same length plus the tower: 0.732 against 0.711 s at 277 tokens (the 512 x 512 fixture), 0.787 against
0.689 at 261 (the 384 x 640 fixture pads to the 1,024-row bucket), 1.873 against 1.616 at 1,045, 5.162 for demo.jpeg's
2,773 tokens against 3.223 for a 2,321-token text prompt; the eviction and the import are unchanged.

Prefix reuse for image prompts (2026-09-29): the committed prefix and the prompt-end snapshot are keyed on the images'
pad spans with their digests beside the ids.  The second turn of an image conversation extends the committed state (24
of 328 tokens prefilled, 0.41 s against 0.64 s for the whole prompt) and its reply is in the prompt-end snapshot's
forced-tail class above -- the text path's second turn on the 177-token chat record (a 64-token first reply) parts from
its fresh prefill the same way, from the first token, while a one-line first turn extended byte-equal -- and a prompt
with another image of the same size (identical pad ids) resets.

The MTP chunk extension's feature row (2026-09-29): the MTP layer's next-token embedding inside an image span is the
tower's row (the zero sentinel plus the feature rows, as the backbone's chunk), not the pad's text embedding.  Image
acceptance with the device tower (k = 4, 32 greedy tokens; passes / accepted drafts / tokens per pass), before and
after:

| fixture | image tokens | before | after |
|---|---|---|---|
| synthetic 512 x 512 | 256 (27 tokens, EOS) | 12 / 16 / 2.33 | 10 / 18 / 2.80 |
| synthetic 384 x 640 | 240 | 14 / 19 / 2.36 | 14 / 19 / 2.36 |
| synthetic 1024 x 1024 | 1,024 | 14 / 18 / 2.29 | 13 / 18 / 2.38 |
| demo.jpeg 2048 x 1365 | 2,752 | 13 / 22 / 2.69 | 13 / 22 / 2.69 |
| all four | | 53 passes / 75 accepted | 50 / 77 |
| the chat control (97 text tokens) | | 35 / 62 / 2.77 | 35 / 62 / 2.77 |

Every image pin (the divergence index and the stream sha) and the text control hold on both sides: the drafts change,
the committed streams do not.  The text lanes' -0.0 rows are the add's identity, so text prompts are bitwise unchanged.

## The teacher-forced table

`tools/ci/baselines/A3-forced-32k-divergence_index.json` pins the startup replay with the teacher-forced prefill
(`--prefill-mode teacher_forced`), measured 2026-09-06 on 4x p150 with the GDN gate fix:

| json | chat | code | fact | list | math | multilingual | prose | refactor | sky | story | summary |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none (96/96) | 25 | none (96/96) | 15 | 56 | 61 | 43 | 13 | 24 | 19 | 6 | 75 |

Against the 2026-09-04 table the fix moved chat 10 -> 25, code 44 -> 96/96, list 46 -> 56 and summary 61 -> 75 later,
math 63 -> 61, multilingual 45 -> 43 and story 19 -> 6 earlier; fact, prose, refactor and sky are unchanged.

## The reference columns

`tools/reference/` freezes the teacher-forced corpus `Q38-REF-v1` (36 items, about 13k positions, with sha256s;
`TESTING.md` lists what is in it and how to run the tool).  `tools/qwen38_reference_corpus.py hf` keeps per position
the top-32 ids and log-probs and the teacher's log-prob from the Transformers `qwen4_exp` model on the CPU (bf16
weights, fp32 LM head); `oracle` does the same through `tt/` (bf16 or BF4-emulated experts); `device` converts a served
chain's agreement records; `score` compares two columns (top-1 / top-5 agreement, clear-margin top-1, truncated KL over
the shared top-32 support, first divergence).  The HF column is the acceptance reference the other columns are read
against.

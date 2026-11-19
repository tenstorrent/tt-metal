# Terminal and full-path optimization audit

CPU-only inspection, 2026-11-16. No device experiments or production edits. Read
`tt/model.py`, `tt/generator.py`, `tt/prefill.py`, selected multichip defaults,
the native v4 reduced profile, and optimize LM-head, matmul, accounting and
prefill references. Numerical and residual policies remain unchanged.

## Budget and measured bottleneck

The accepted isolated decoder FUNCTIONAL measurements give
`30 * 0.719504474 + 10 * 0.340836807 = 24.99350229 ms/token`.
Full-model `performance/lifetime_fixed.json` records queued token-out
25.081887921 ms/token, caller-observed 25.128267976, model-only 24.600408906,
and fixed-logit sampler-only 0.487787433. Queued token-out exceeds the isolated
stack estimate by only 0.088385631 ms (0.354%); there is no demonstrated 10–15%
gap to close. The negative model-only-minus-stack difference confirms these
different workload/context measurements are not an additive decomposition.
The fresh stage7 same-harness baseline should supersede these historical values.

Native `doc/full_model/performance/profile_reduced_v4_drains/token_out_op_groups.csv`
on rank0 identifies the terminal head as the largest individual model op:
397.9985 us, BF16×BFP8, local `1×2048×62080`, interleaved operands, 110 cores.
Terminal norm is 22.6397 us; logits padding and copying cost 23.6897/21.2469 us.
Sampler local large-indices top-k costs 266.0246 us plus 24.3902/20.4846 us
route preparation/finish. Candidate gathers cost 11.3008/11.2668 us. These are
per-rank kernel-work sums, not latency or a sum over four chips. Sampling is
about 1.945% of the separately measured all40 queued path, not a dominant
all40 bottleneck. The existing greedy probe rejects force-argmax (2.6647 ms)
and power-of-two padding (0.5218 ms) against split greedy (0.4883 ms).

## First candidate: DRAM-sharded terminal head

Keep terminal FP32 norm, BF16 cast, BFP8 head and HiFi2 with FP32 destination
accumulation exactly as selected. Change only head storage and execution
geometry. `tests/run_lm_head_geometry_probe.py` provides one fresh-process
candidate against the original terminal using real reduced-layer AIME
prefill activations, eager/replay full-vocab comparison, top1, and queued
traced whole-terminal timing. It includes norm, casts, reshards and a stable
output copy in both timing rows. The primary correctness bar is PCC≥0.995
plus top1 agreement; all40 readiness remains necessary for a kept candidate.

| Parameter | Legal initial family |
|---|---|
| Weight | Width-sharded DRAM over 8 banks, full K=2048 per bank |
| Logical local N | 62080, unchanged; sampler offsets remain unchanged |
| Storage bank width | `ceil(62080/(32*8*readers))*32*readers`: 7776 / 7808 / 7776 for readers1/2/3 |
| Input | BF16 L1 width-sharded, 16 cores×128 initially; 8×256 or 32×64 controls |
| Output | BF16 L1 width-sharded, `per_core_N=32`, 61 storage cores×1024 columns; then DRAM interleaved |
| K block | 1,2,4,8,16,32,64 divide Kt64; start readers2/3 at4/8/16, extend as L1 permits |
| Readers | 1/2/3; `use_multicore_pipeline=readers>1` |

Output `per_core_N` controls storage geometry independently of reader/compute
cores. Values20/32/64 give97/61/31 output cores. All fit the110-core grid.
Physical storage padding is internal to the DRAM shards; do not enlarge the
logical vocabulary or alter tokenizer IDs. The existing common sampler keeps
real-vocab masking. No added logical columns reach it.

The one-reader full-width control can exceed L1 even at small K blocks:
the native factory triples the weight staging CB and stores a wide output
intermediate. At bank width243 tiles its weight CB alone is
`3*243*block*1088` bytes. Record exact failures, then adapt by splitting the
local head into two/four tile-aligned pieces and trimming internal storage
padding before concatenation. A single L1 error does not reject the family.
The multicore pipeline uses bounded8-column passes and is the more promising
full-width route. Rough unavoidable stored weight traffic is135,086,080
bytes/rank (~0.26384 ms at512 GB/s); this is a bandwidth estimate, not a
measured terminal lower bound. Even a 0.1–0.2 ms terminal win is less than1%
of all40 decode, so two small kept candidates can activate the Stop Rule.

`tt-transformers-contract/src/tt_transformers/modules/lm_head/lm_head_1d.py`
provides the right semantic template: local-vocabulary DRAM-sharded chunks,
L1 output and localized reshard. Direct use requires its LazyWeight and4-D
interfaces; its current compute validator also names
`WormholeComputeKernelConfig` while accepting Blackhole architecture. This is
a source-level integration concern, not a tested blocker. The model-local
probe follows the same operation family without changing precision defaults.

## Other minimal candidates and constraints

1. `SamplerCCL.line_all_gather` currently ignores `buffer_key` and never passes
   a persistent output buffer. Add setup-time keyed candidate-value/index
   buffers and compare the whole sampler trace. Both existing gathers total
   only~22.6 us of rank0 kernel work, so expected all40 gain is small. Validate
   the actual shared-buffer lifetime against both live traces with tracking.
2. `model.embed` gathers hidden once per slot. Batch32 currently repeats
   embedding and collective for each fixed slot. A single packed32-row
   embedding/gather followed by per-slot extraction is the coherent candidate;
   preserve exact inactive rows and avoid changing decoder residual layout.
3. Prefill TTFT (~2.92 s historical) offers more absolute room than terminal
   decode. `BoundedPrefill` is eager, repeatedly transports/tilizes chunks,
   creates terminal outputs and uploads full-attention positions for each
   layer/chunk. Profile reduced prefill before selecting trace or buffer reuse.
   Traced chunks require stable offset/position/page inputs, global preparation,
   and logical tail/resume tests. Do not simply trace host-refresh helpers.
4. Full-attention continuation after an unaligned start uses singleton chunks
   until the next256 boundary. This is deliberate correctness fallback with
   potentially large TTFT cost. It requires an adapted cache-fill/attention
   probe, not removing the planner condition. Do not relitigate the accepted
   resumed-prefill numeric limitation while optimizing this path.

The existing steady-state model/sampling replay is already nonblocking with
device token/position advance. Historical no-readback counters are zero for
token, position, page and activity refreshes. `generate()` itself delivers each
token to the host for the caller and stopping; report its timing separately
from queued device production. Decoder workspace and replicated BF16 residuals
are the accepted baseline, not an accidental wrapper regression.

## Probe status

The new probe was CPU syntax-checked only; no hardware result is claimed.
Suggested first launches: readers3/block8/cores16, readers2/block8/cores16,
readers1/block1/cores16, each serialized under an external timeout. Continue
with block4/16 and input-grid controls around the winner, then larger blocks
where the measured allocation permits them. Preserve failures and record
setup memory before adopting a second head weight copy in production.

## Main-agent measurements and adapted one-reader probe

The main agent subsequently ran the probe serially; this audit agent did not
open hardware. The best reported full-width candidate is readers3/block4/cores16,
0.353635 ms versus same-run baseline0.452275 ms, with bit-equal logits. Nearby
reported candidate times are readers3/block8 0.367281 ms, readers2/block8
0.371851 ms, readers3/block16 0.382754 ms and readers2/block16 0.417201 ms.
The exact raw JSON files under this directory remain authoritative.

`head_r1_b1_c16.json` records the one-reader failure in terminal `ttnn.linear`:
"Statically allocated dataflow buffers on core range [0-0 - 10-9] grow to
2408960 B which is beyond max L1 size of 1572864 B". This occurs at compile/allocation,
not a numerical comparison. The readers3/block32 control also exceeds L1
(2080704 B versus1572864 B, as reported by the main agent).

The probe now accepts `--splits 1|2|4`. It slices each device's original local
62080-column vocabulary head into contiguous31040- or15520-column pieces,
DRAM-shards each piece with its own internal storage padding, projects with the
same input/norm/math policy, reshards each result, and concatenates in original
local order. It never rearranges device ownership or exposes padded vocabulary
IDs. For one reader the two/four-piece bank storage widths are3904/1952
columns (122/61 tiles), reducing the wide temporary buffers. Suggested adapted
control: `--readers 1 --block 1 --cores 16 --splits 4`; compare splits2 if it fits.
The complete terminal timing includes concatenation.

`--resident-l1-bytes 581696` requests a persistent per-bank allocation target
before terminal warmup/capture. The probe allocates one retained row-major
uint32 width shard on every worker to fill the difference from the measured
current L1 allocation, rounded up to32 bytes per worker. JSON records the
actual before/after memory view. This is a cheap capacity-pressure proxy;
it does not reproduce the full model's exact allocation addresses, cache
program buffers, or fragmentation and cannot replace final full40 validation.

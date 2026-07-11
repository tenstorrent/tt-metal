# Functional decoder work log

Target: `google/gemma-4-31B`

Branch: `odjuricic/agentic-research/graph-rewrite-skill`

Starting commit: `bc899b5b2f217dbe83a10b7800fdc1e37c35b136`

## Architecture and implementation

- Read the installed HF Gemma4 config/modeling code and the repo Gemma4 TTNN
  implementation.
- Confirmed 60 dense text layers in a 5 sliding + 1 full pattern, hidden size
  5376, GeGLU MLP 21504, four learned RMSNorms, layer scalar, 262144 context.
- Added a `LightweightModule` functional decoder with a real canonical
  `from_state_dict` boundary and host-free prefill/decode runtime.
- Added logical-length padding/slicing, paged fill/update/SDPA, tensor current
  positions, full-context streaming, long-sequence MLP chunking, and batch-2
  prefill, including batch 32.

## Focused repair evidence

1. Full prefill initially failed because single-core `nlp_concat_heads`
   requested 2,208,512 bytes L1 against 1,572,864 available. Replacing it in
   the autoport with device permute+reshape passed at PCC 0.999592.
2. Sequence 33 initially exposed K/V padded-length mismatch. Internal tile
   padding plus logical output slicing passed at PCC 0.999282/0.999309.
3. Batched reuse initially invalidated a caller-owned RoPE view. Preserving the
   caller buffer fixed batch-2 x 33 at PCC 0.999289/0.999287.
4. A 196577-token legacy full-attention chunk caused a NOC0 stall. Triage could
   not attach because NOC0 was hung; the exact output is in
   `triage/long_full_196577_tt-triage.txt`. The process was terminated, bounded
   list/reset/list recovered the P150, and a 1x1 mesh open/close passed.
5. Full prefill at 262144 then exposed an exact 2^32-element Q normalization
   crash. Streaming Q/K/V projection, normalization, RoPE, page fill, SDPA and
   output projection in 1024-token blocks resolved both the overflow and the
   long-kernel stall. The advertised context passed in 393.27 s cold and the
   near-context 262113 rerun passed in 94.53 s with compiled programs.

Ordinary focused experiments identified and verified each issue; no unresolved
hard bug remained that required `$autofix`.

## Commands and results

- Final standard suite: see `logs/final_revision_standard_suite.log`; 17
  passed, with 6 long/performance evidence tests explicitly gated in the
  ordinary invocation.
- Batch-32 follow-up: `logs/batch32_prefill_decode_real.log`; four real-weight
  sliding/full prefill/decode cases passed at PCC 0.999309--0.999480.
- Real-weight context commands set `GEMMA4_LONG_PREFILL` to 262144 and 262113
  for each representative layer. Exact-limit prefill PCC was 0.998786 sliding
  and 0.999089 full; after that prefill, traced decode replay at position
  262143 matched the prefill final token at 0.998715 and 0.996611.
- Advertised-position empty-history construction smokes also passed, but the
  context capability claim is based on the populated-cache runs above.
- Final watcher command and clean batch-32 traced result:
  `logs/final_watcher_batch32_stable_trace.log` and
  `watcher_batch32_stable_final/generated/watcher/watcher.log`.
- Tracy commands and reports: `logs/perf_*.log` and `tracy/<kind>/<mode>/`.

PCC and latency tables are in `README.md`. The acceptance threshold remained
0.995; no waiver or capability reduction was used.

## Initial review remediation

The initial independent review in `stage_review.md` returned
`more-work-needed`. Its findings were closed as follows:

- long-only branches now have real-weight PCC controls and exact-limit
  populated-cache traced decode evidence;
- every cache correctness test uses genuinely non-identity page mappings;
- batch-32 decode captures and replays the complete trace twice and requires
  deterministic repeated output;
- the runtime fallback audit covers the complete functional decoder runtime
  call graph and transitive Gemma attention, MLP, and RMSNorm helpers;
- the final standard suite and final batch-32 watcher run were regenerated;
- the failed triage attachment artifact records the exact failure and bounded
  recovery instead of being empty.

The delivered runtime source hash is
`315990f7e060a464d935efadf0dd50a8b035e187947ec320d08e4a15479e7a85`;
the final test source hash is
`45b5053b59e5a2ac7c1a56e1cfd1aac31989588cdfcc513b47e97a0581d09c45`.
Only tests, evidence documentation, and an isort-only import reordering changed
after the four profiler runs; functional runtime logic did not.

## Hardware recovery record

- Failing command: `GEMMA4_LONG_PREFILL=196577 ... test_long_nonaligned_prefill_capacity[5-full_attention]`.
- Triage signature: `NOC0 is hung on PCIe device ID 0`.
- Stopped only PIDs belonging to that pytest/timeout pipeline.
- `timeout 60 tt-smi -ls --local` returned `Read 0xffffffff ... board should be reset`.
- `timeout 180 tt-smi -r` completed; the next bounded list showed one P150.
- A 1x1 TTNN mesh open/close printed `MESH_SMOKE_OK`.
- The stage resumed and the redesigned advertised-context path passed.

## Stage review and checkpoint

The initial review returned `more-work-needed`; all findings were fixed. The
fresh rereview in `stage_review_rereview.md` returned `clean-pass`, and its
post-hook provenance follow-up kept that verdict.

Implementation and evidence checkpoint:
`dac92a78dbca5d4b2d3e85b1007b00064b1ccc42`. This work-log/review provenance
update is a stage-owned follow-up commit. No push was performed.

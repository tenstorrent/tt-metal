# Stage 03 — RVQ depth decoder (`rvq_depth_decoder`) on one Blackhole chip

Work log for MiniMax-Music3's local LM (`MiniMaxMusic3RVQDepthDecoder`) in TTNN.

* implementation: [`../../tt/depth_decoder.py`](../../tt/depth_decoder.py) (`DepthDecoder`, `DepthStepTrace`)
* host reference: [`../../reference/depth_decoder_ref.py`](../../reference/depth_decoder_ref.py) (torch
  transcription of the diffusers module + the teacher-forced depth loop)
* tests: [`../../tests/test_depth_decoder.py`](../../tests/test_depth_decoder.py) (gate),
  [`../../tests/test_depth_decoder_perf.py`](../../tests/test_depth_decoder_perf.py) (Tracy-signposted, `-m slow`)
* scripts: [`../../scripts/collect_depth_perf.sh`](../../scripts/collect_depth_perf.sh),
  [`../../scripts/trace_probe_depth.py`](../../scripts/trace_probe_depth.py)
* measured numbers: [`pcc/pcc_results.json`](pcc/pcc_results.json), [`perf/*.json`](perf/),
  [`tracy/{eager,traced}/perf_report.txt`](tracy/); hang evidence: [`triage/`](triage/)

## What was built

`DepthDecoder.from_pretrained(mesh_device, weights_dir, dtype=ttnn.bfloat16)` loads
`<weights_dir>/rvq_depth_decoder/diffusion_pytorch_model.safetensors` (bf16 checkpoint; about 1.35 GB
on device: 1.2 GB of transformer / embedding weights plus the seven heads stored both per head and
fused, and the one-hot selector tables) in about 1.2 s and exposes:

| method | contract |
|---|---|
| `audio_embed(codes, codebook)` | `audio_embeddings[code + (codebook-1)*1024]`; host `[B]` codes of codebook 1..7, or a `[1, 32]` uint32 device tensor of offset indices; returns the `[1, 1, 32, 4096]` "decode-row" tensor (rows 0/1 = batch rows) |
| `residual_embedding_sum(frame_codes [B, 8])` | sum of the 7 residual-code embeddings, `[1, 1, 32, 4096]` (the residual half of `MusicLLM.embed_frame`, for stage 04) |
| `project(x)` | `projection` linear (4096 -> 4096, no bias) of a host `[B, 4096]` or device `[1, 1, 32, 4096]` row tensor |
| `new_sequence()` / `place_step(seq, rows, step)` | build the depth sequence on device: `seq + E[step] @ rows` with a one-hot `[64, 32]` scatter matrix |
| `forward(inputs_embeds, *, num_steps)` | host `[2, num_steps, 4096]` (already projected, like the reference) or the device sequence `[1, 1, 64, 4096]`; returns the normed hidden of the LAST logical step as `[1, 1, 32, 4096]` (rows 0/1 = batch rows); `num_steps` 1..9 or a gather-selector tensor (traced path) |
| `head(k, hidden)` | `audio_heads[k-1]` logits `[1, 1, 32, 1024]`, k in 1..7 |
| `heads_all(hidden)` | all seven heads in one matmul `[1, 1, 32, 7168]` (column block k-1 = head k) |
| `teacher_forced_loop(global_hidden, semantic_embed, residual_codes)` | the 7-step depth loop with given codes, on device; returns the 7 last-step hiddens and 7 head logits |
| `DepthDecoder.rows_to_host(t)` | `[1, 1, 32, N]` -> fp32 `[2, N]` (the explicit test boundary) |

`DepthStepTrace(decoder)` captures two traces: a *seed* trace (project the backbone hidden and the
semantic-code embedding from two persistent `[1, 1, 32, 4096]` buffers and scatter them into steps
0 / 1 of the persistent sequence buffer) and a *step* trace (embed the previous code -> project ->
scatter into the sequence -> 4 layers + final norm -> gather the last step -> all 7 heads into a
persistent logits buffer). `begin_frame(global_hidden, semantic_embed)` takes host `[2, 4096]`
tensors or persistent `[1, 1, 32, 4096]` device row tensors (copied with `ttnn.copy`), `step(index,
prev_code)` replays the step trace for index 1..7 rewriting only the scatter selector, the code-id
row and the gather selector, `logits_for(k)` reads head k back. A whole frame allocates nothing on
device after construction (see "Trace lifetime" below). Stage 07 gets a trace-safe fixed-shape depth step.

### Capability contract (model-specific; there is no KV cache / context length here)

| claim | evidence | remaining risk |
|---|---|---|
| any logical depth length 1..9 (pipeline uses 2..8) is accepted; padding to 32 and the last-step read are internal | `test_forward_and_heads_vs_reference[2..8]` PCC >= 0.9994; `MAX_STEPS = 9` asserted in `forward` / selectors | length 9 is untested (never built by the pipeline; the 16-slot position table would allow up to 16) |
| batch 2 with independent CFG rows (row 0 conditional, row 1 unconditional) | `test_distinct_batch_rows`: rows fed golden frames 1 and 2 each reproduce their own golden depth hiddens (PCC 0.99995 / 0.99994), row-0-vs-row-1 golden PCC 0.83 (so the rows really differ) | none known |
| no KV cache; every step re-runs the whole sequence exactly as the reference | `teacher_forced_loop` / `DepthStepTrace._step_graph` re-run `hidden_states(seq)` per step; golden loop PCC 0.99995 | none |
| learned position embedding, 16 slots | pre-broadcast onto rows `b*32+s`, `s < 16`, zero beyond | none |
| padded rows never influence real rows | zero rows stay zero through RMSNorm (0 / sqrt(eps)), causal SDPA (`is_causal=True`) hides later rows; per-length PCC and the golden loop confirm | none |
| runtime free of host round-trips | `forward`, `hidden_states`, `teacher_forced_loop` and both traces only use device ops (the trace capture would reject a host write); reads happen only in `rows_to_host` / `logits_for` | none |
| fixed-shape, allocation-free traced depth step | `test_traced_step_matches_eager_and_perf`: bit-identical to eager, bit-identical across frames and between host- and device-seeded frames, and no `Allocating device buffers is unsafe` warning during the traced frames (asserted via `capfd`) | callers must keep post-capture device tensors dead across replays (documented contract) |

### Layout decision: pad to one tile, select with one-hot matmuls

The reference re-runs the whole depth sequence (2..8 positions, no KV cache) at every step. Both CFG
rows are kept in one tile-aligned `[1, 1, 64, 4096]` tensor (row `b*32 + s` = step `s` of batch row
`b`), so every op has a fixed shape for every step length: one program-cache entry per op, and one
trace serves all seven steps. Rows past the logical length are zero plus the (zero-padded) position
embedding; causal SDPA (`is_causal=True`, no explicit mask) guarantees real rows never read them.
The logical last step is read with a `[32, 64]` one-hot gather matmul, and new steps are appended with
a `[64, 32]` one-hot scatter matmul + add. Both selectors are plain device tensors, which is what
makes the traced step possible without host-side shape changes. Batch rows are tile-aligned splits
(`[1,1,64,N] <-> [2,1,32,N]`) done with `ttnn.experimental.view` (zero-copy).

Attention: fused QKV weight `[4096, 12288]` -> `nlp_create_qkv_heads` (16 heads x 256) -> SDPA
(HiFi4) -> `nlp_concat_heads` -> `to_out`; the QKV output and the head tensors live in L1. MLP:
`down(silu(gate(x)) * up(x))`; the SiLU is passed as the gate matmul's `activation` but, without a
program config, ttnn applies it as a separate unary op (7 us, visible in Tracy) - real fusion is a
stage-07 item. Norms: `ttnn.rms_norm` eps 1e-6, width-sharded over a 4x4 core grid (see
"Performance"). Matmuls: bf16 weights, HiFi2, fp32 accumulation, DRAM interleaved. RMSNorm runs in
bf16 (the reference computes the statistics in fp32); PCC shows no measurable loss at these lengths.

### Trace lifetime

tt-metal warns once per device on the first buffer allocation made while a trace is live
(`Allocating device buffers is unsafe due to the existence of an active trace`, allocator.cpp): such a
buffer is safe only if it dies before the next `execute_trace`, because replays reuse the addresses
captured for their intermediates. `DepthStepTrace` therefore allocates everything it needs (sequence,
seed rows, selectors, code ids, hidden / logits outputs) in `__init__`, runs both graphs once for the
program cache, and only then captures both traces back to back; `begin_frame` is two buffer writes (or
two `ttnn.copy` from persistent device tensors) plus a replay. The first version seeded the frame with
eager `project` / `place_step` calls, which allocated after capture and produced the warning (benign
in that usage - the temporaries were dead before `step()` - but unenforced); the gate now asserts
the warning is absent. Caller contract for stages 04/07: any device tensor created after
`DepthStepTrace(...)` must not be alive across `step()` / `begin_frame()`; pass the backbone hidden as
a persistent row tensor that already existed before the depth traces were built.

## How to run

```bash
source ~/mm3-bringup/common.sh && cd $MM3_WT
# gate (also: ~/mm3-bringup/checks/03.sh)
with_hw_lock timeout 3600 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_depth_decoder.py -m "not slow" -q
# Tracy device-op evidence, one warmed frame each
with_hw_lock $MM3_MODEL_DIR/scripts/collect_depth_perf.sh eager
with_hw_lock $MM3_MODEL_DIR/scripts/collect_depth_perf.sh traced
# watcher
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_LOGS_PATH=$MM3_MODEL_DIR/generated/watcher_depth \
  with_hw_lock timeout 1200 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_depth_decoder.py -m "not slow" -q
```

Watcher / profiler runs do not write `doc/` evidence (`_record` skips when `TT_METAL_WATCHER` or
`TT_METAL_DEVICE_PROFILER` is set), so `pcc/pcc_results.json` always holds the plain gate run.

Hardware: host `qbge-devex-02`, board `p300c`, board id `000004613193411b`, PCI `0000:01:00.0`
(`TT_METAL_VISIBLE_DEVICES=0`, opened as a 1x1 mesh, program cache on, trace region 90 MB, no fabric);
`with_hw_lock tt-smi -s` on 2026-09-09. Software: `~/tt-metal` prebuilt at `e946955cc15`
(shared `ttnn`), worktree branch `jashan/minimax-music3` (commits listed at the end).

## Evidence

### Correctness (`pcc/pcc_results.json`, gate run 2026-09-09 17:36, 11 passed)

Reference for (1): the fp32 torch transcription loaded from the same bf16 safetensors; input = the
reference's own projected 9-step sequence for golden frame 1 (backbone hidden, semantic-code
embedding, c1..c7), truncated to `steps`. Bar 0.995.

| steps | hidden PCC (last step) | min over 7 heads' logits PCC | fused-heads min PCC |
|---|---|---|---|
| 2 | 0.99994 | 0.99946 | 0.99946 |
| 3 | 0.99995 | 0.99992 | 0.99992 |
| 4 | 0.99994 | 0.99952 | 0.99952 |
| 5 | 0.99996 | 0.99968 | 0.99968 |
| 6 | 0.99995 | 0.99975 | 0.99975 |
| 7 | 0.99995 | 0.99971 | 0.99971 |
| 8 | 0.99994 | 0.99973 | 0.99973 |

(2) Golden frame 1, teacher-forced 7-step loop on device vs the diffusers fp32 golden
`frame_hiddens[0, 0, 4096:]` (bar 0.99): **PCC 0.99995** over all 7 x 4096 values; per step
0.99996 / 0.99997 / 0.99996 / 0.99995 / 0.99994 / 0.99996 / 0.99995. Control: the torch transcription
reproduces the golden with PCC 1.000000, so the loop, code offsets and step alignment are right.
The device argmax of every head equals the fp32 reference argmax (7/7); neither equals the golden
sampled code (0/7). Classified: the pipeline samples top-k(50) from CFG-guided logits and the
reference's conditional distributions are flat (the review's CPU check ranks the golden codes 0..60
in the reference logits, argmax probability 3-16 %), so argmax agreement is not expected; the
decoder-side control is the fp32 reference, which behaves identically.

(2b) Distinct rows (`test_distinct_batch_rows`): row 0 = golden frame 1, row 1 = golden frame 2; each row
vs its own golden: PCC 0.99995 / 0.99994; vs the torch reference: hidden 0.99995 / 0.99994, min head
logits 0.99983 / 0.99989; row 0 vs row 1's golden 0.83 (rows are genuinely different, not swapped).

(3) Determinism: two identical 7-step loops give bit-identical hiddens and logits.

(4) Traced step vs eager: all 7 heads' logits bit-identical (PCC 1.0, max |diff| 0); a second frame
and a device-seeded frame are bit-identical to the first (no state leaks through the persistent
sequence buffer); no post-capture allocation warning.

### Performance (warmed, one 7-step teacher-forced frame, batch 2)

| variant | host wall per frame | device time per frame (Tracy, sum of `Device Time` in `perf_report.csv`) | ops per frame |
|---|---|---|---|
| eager (`teacher_forced_loop` + 7 logits read-backs) | 29.7 ms (gate run, mean of 10) / 30.5 ms (under Tracy) | 28.69 ms | 576 |
| traced (`DepthStepTrace`, seed replay + 7 step replays + 7 read-backs) | 31.3 ms (gate run, mean of 20) / 32.3 ms (under Tracy) | 29.53 ms | 601 |

Before the two layout changes below (first version of this stage, commit `dabf3050d01`): eager 35.2 ms
wall / 34.39 ms device (450 ops), traced 37.5 ms / 35.26 ms (476 ops). The loop is device-bound
(device time = 94-97 % of wall), so tracing does not help yet; it costs ~0.8 ms of extra copies and
buffer rewrites. Device-time breakdown of the eager frame (`tracy/eager/perf_report.csv`):

| op | device time / frame | count | cores | note |
|---|---|---|---|---|
| Matmul (bf16 weights) | 22.4 ms (78 %) | 170 | 32-96 | weight matmuls run at 73-78 % of DRAM bandwidth (376-397 GB/s); 1.14 GB of weights per step |
| `nlp_create_qkv_heads` (L1) | 3.2 ms (11 %) | 28 | 2 | 113 us each (155 us from DRAM before) - one core per batch row's single tile row |
| `nlp_concat_heads` (L1) | 1.1 ms (4 %) | 28 | 2 | 38 us each (52 us before) |
| RMSNorm, width-sharded 4x4 + reshards | 0.87 ms (3 %) | 63 + 126 | 16 | 8.5 us norm + 2.4 + 2.9 us resharding (75 us on 2 cores before: the interleaved kernel parallelizes over the 2 tile rows) |
| eltwise add/mul, SiLU (unfused unary), SDPA, embedding | 1.2 ms (4 %) | 161 | 110 / 1 | SDPA is 8 us |

The RMSNorm and L1 choices were selected from a Tracy micro-benchmark of alternatives (scratch, not
committed): sharded norm grids 4x4 / 8x4 / 8x8 gave 7-8 us kernels but 8x4 / 8x8 had 30-45 us
dispatch gaps; a composite `x * rsqrt(mean(x^2))` was slower (its reduce also runs on 2 cores); a
"heads as batch" formulation with per-head weights (`[1, 16, 4096, 256]` batched matmuls) was 7x
slower (batched matmul over 16 cores at 6 % DRAM utilization), so `nlp_create_qkv_heads` stays.

Perf artifacts: `tracy/eager/`, `tracy/traced/` (`ops.csv.gz`, `perf_report.txt` for the signposted
window, `perf_report.csv`, `perf_report.summary.txt`, `pytest.log`; no dropped profiler markers -
buffers are drained after every warm frame). Column used: `Device Time` (us) of `perf_report.csv`
filtered by the `PERF_DEPTH_*` signposts. Tracy tooling notes (tools folder flag, tolerant
post-processing) are the same as stage 02 (`scripts/collect_llm_perf.sh`).

## Watcher

`TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1` over the whole gate test file (11 tests incl. both
traces): 11 passed in 19 s, `watcher.log` (2171 lines, kept under `generated/watcher_depth/`,
gitignored) has only attach / dump / detach lines - zero exception / assert / sanitizer / overflow /
fault lines.

## Decisions taken (nobody to ask)

* **Reference in `python_env`**: python_env's diffusers 0.38 lacks the class, so
  `reference/depth_decoder_ref.py` is a torch-only transcription (Apache-2.0 header kept, source cited);
  the golden-frame test proves it matches the real diffusers module (PCC 1.000000).
* **Unconditional row in the golden test**: the golden dump only holds the conditional row's backbone
  hidden, so `test_golden_frame_depth_loop` feeds both rows the same data and compares row 0, as the
  pipeline does (`hidden_parts.append(hidden[:1])`); row independence is covered separately by
  `test_distinct_batch_rows` (two different golden frames in the two rows).
* **`MAX_STEPS = 9`**: the pipeline builds at most 8 positions (hidden, c0, c1..c6) but the module
  accepts 9 so a caller may append c7 too (e.g. to embed a complete frame).
* **`ttnn.reshape` replaced by `ttnn.experimental.view`**: the batch/row splits are tile-aligned, so the
  zero-copy `view` is used explicitly instead of `reshape` (which may fall back to the host for tile
  tensors). `scripts/trace_probe_depth.py` checks every op of the step graph inside its own trace
  capture (16/16 ok, `triage/trace_probe.log`).
* **Trace capture must be preceded by a full compile run of the exact op sequence**: the first trace
  attempt failed with "Writes are not supported during trace capture" because the two output copies
  (`ttnn.copy` into the persistent hidden / logits buffers) were first executed inside the capture and
  compiled there (kernel load = host write). After such a failed capture the mesh close waits forever
  in `FDMeshCommandQueue::clear_expected_num_workers_completed` (gdb backtrace in
  `triage/gdb_main_thread.txt`; tt-triage: all checks pass, no worker kernels running, dispatch idle -
  `triage/triage-summary.txt`, `triage/tt-triage-callstacks-excerpt.txt`). The process was killed
  (`kill -9`), no reset was needed (`tt-smi -ls` healthy, next device open fine).
* **Heads fused for the traced step**: the trace computes all 7 heads (`heads_all`, 58 MB of weights,
  ~0.15 ms) because the head index changes per step; the caller slices the block it needs. The per-head
  weights are kept too for `head(k)` (58 MB duplicate, negligible).
* **Two correctness-neutral layout changes were taken in this stage** (sharded norm, L1 head tensors)
  because they were measured to remove ~17 % of the frame time with unchanged PCC; everything else is
  left for stage 07.

## Independent review

`stage-review` (fresh subagent, 2026-09-09 17:18-17:27, on commit `dabf3050d01`): `more-work-needed`
with three P2 findings, all addressed in the follow-up commit: (1) unclassified post-capture
allocation warning in `begin_frame` -> seed trace + persistent seed buffers, contract documented,
gate asserts the warning is absent; (2) `pcc_results.json` perf entry clobbered by the watcher run and
a README number (37.4) no log supported -> `_record` skips watcher / profiler runs, numbers re-taken
from the final gate run; (3) batch-row independence never exercised -> `test_distinct_batch_rows`.
Other concerns fixed: SiLU is not fused (text corrected), profiler buffers drained during warmup,
triage evidence copied into `doc/depth_decoder/triage/`, capability-contract table added, weight
footprint corrected, stale `_capture` comment fixed. A second review pass is recorded below.

## Open risks / hand-off to later stages

* **`nlp_create_qkv_heads` / `nlp_concat_heads` still run on 2 cores** (15 % of device time): the ops
  parallelize over tile rows and each batch row has one. Stage 07 options: a sharded head split, or
  restructuring the QKV projection so heads come out of the matmul directly; the per-head batched
  matmul was measured and rejected (above).
* **Matmuls are DRAM-bandwidth bound on bf16 weights** (1.14 GB read per step, 7 steps per frame,
  25 frames/s of audio => ~30 ms of depth decoding per generated frame, i.e. ~7.5 s per 10 s clip on
  top of the backbone). bfp8 weights would halve that; precision impact goes to the stage-07 dtype sweep.
* The traced step's per-step read-back (logits -> host for sampling) is inherent to the pipeline's
  top-k sampling; an on-device sampler is a stage-04/07 option.
* All PCC evidence uses golden frames 1 and 2 of one clip; the bars are met with margin (>= 0.9994).

## Commits

* `dabf3050d01` — first version: implementation, reference, tests, perf scripts, Tracy evidence, work log.
* follow-up (this log's final state) — review fixes, seed trace, distinct-rows test, sharded norm / L1
  heads, refreshed evidence; SHA recorded in the section below after committing.

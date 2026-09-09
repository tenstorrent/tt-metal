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
  [`tracy/{eager,traced}/perf_report.txt`](tracy/)

## What was built

`DepthDecoder.from_pretrained(mesh_device, weights_dir, dtype=ttnn.bfloat16)` loads
`<weights_dir>/rvq_depth_decoder/diffusion_pytorch_model.safetensors` (bf16 checkpoint, 1.2 GB of
weights on device in about 1.2 s) and exposes:

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

`DepthStepTrace(decoder)` captures one depth step (embed previous code -> project -> scatter into the
persistent sequence buffer -> 4 layers + final norm -> gather last step -> all 7 heads) as ONE
trace; `begin_frame(global_hidden, semantic_embed)` seeds steps 0/1, `step(index, prev_code)` replays
it for index 1..7 rewriting only the scatter selector, the code-id row and the gather selector,
`logits_for(k)` reads the head-k logits back. Stage 07 gets a trace-safe fixed-shape depth step.

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
(HiFi4) -> `nlp_concat_heads` -> `to_out`. MLP: `down(silu(gate(x)) * up(x))` with the SiLU fused into
the gate matmul. Norms: `ttnn.rms_norm` eps 1e-6. Matmuls: bf16 weights, HiFi2, fp32 accumulation,
DRAM interleaved. RMSNorm runs in bf16 (the reference computes the statistics in fp32); PCC shows no
measurable loss at these lengths.

## How to run

```bash
source ~/mm3-bringup/common.sh && cd $MM3_WT
# gate (also: ~/mm3-bringup/checks/03.sh)
with_hw_lock timeout 3600 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_depth_decoder.py -m "not slow" -q
# Tracy device-op evidence, one warmed frame each
with_hw_lock $MM3_MODEL_DIR/scripts/collect_depth_perf.sh eager
with_hw_lock $MM3_MODEL_DIR/scripts/collect_depth_perf.sh traced
```

Hardware: host `qbge-devex-02`, board `p300c`, board id `000004613193411b`, PCI `0000:01:00.0`
(`TT_METAL_VISIBLE_DEVICES=0`, opened as a 1x1 mesh, program cache on, trace region 90 MB, no fabric);
`with_hw_lock tt-smi -s` on 2026-09-09. Software: `~/tt-metal` prebuilt at `e946955cc15`
(shared `ttnn`), worktree branch `jashan/minimax-music3` (this stage's commit is recorded at the end).

## Evidence

### Correctness (`pcc/pcc_results.json`, 2026-09-09)

Reference for (1): the fp32 torch transcription loaded from the same bf16 safetensors; input = the
reference's own projected 9-step sequence for golden frame 1 (backbone hidden, semantic-code
embedding, c1..c7), truncated to `steps`. Bar 0.995.

| steps | hidden PCC (last step) | min over 7 heads' logits PCC | fused-heads min PCC |
|---|---|---|---|
| 2 | 0.99994 | 0.99973 | 0.99973 |
| 3 | 0.99995 | 0.99983 | 0.99983 |
| 4 | 0.99995 | 0.99958 | 0.99958 |
| 5 | 0.99994 | 0.99940 | 0.99940 |
| 6 | 0.99993 | 0.99922 | 0.99922 |
| 7 | 0.99995 | 0.99966 | 0.99966 |
| 8 | 0.99993 | 0.99963 | 0.99963 |

(2) Golden frame 1, teacher-forced 7-step loop on device vs the diffusers fp32 golden
`frame_hiddens[0, 0, 4096:]` (bar 0.99): **PCC 0.99994** over all 7 x 4096 values; per step
0.99993 / 0.99993 / 0.99993 / 0.99994 / 0.99995 / 0.99994 / 0.99993. Control: the torch transcription
reproduces the golden with PCC 1.000000, so the loop, code offsets and step alignment are right.
The device argmax of every head equals the fp32 reference argmax (7/7); neither equals the golden
sampled code (0/7) because the pipeline samples top-k from CFG-guided logits, which is expected.

(3) Determinism: two identical 7-step loops give bit-identical hiddens and logits.

(4) Traced step vs eager: all 7 heads' logits bit-identical (PCC 1.0, max |diff| 0); a second frame
after the first is bit-identical (no state leaks through the persistent sequence buffer).

### Performance (warmed, one 7-step teacher-forced frame, batch 2)

| variant | host wall per frame | device time per frame (Tracy, sum of "Device Time" in `perf_report.csv`) | ops per frame |
|---|---|---|---|
| eager (`teacher_forced_loop` + 7 logits read-backs) | 35.2 ms (gate run, mean of 10) / 36.1 ms (under Tracy) | 34.39 ms | 450 |
| traced (`DepthStepTrace`, 7 replays + 7 read-backs) | 37.4 ms (gate run, mean of 20) / 38.3 ms (under Tracy) | 35.26 ms | 476 |

The loop is device-bound (device time = 95 % of wall), so tracing does not help yet; it costs ~1 ms of
extra copies (persistent sequence/output buffers) and buffer rewrites. Device time breakdown (eager):

| op | device time / frame | count | note |
|---|---|---|---|
| Matmul (bf16 weights, DRAM interleaved) | 22.6 ms (66 %) | 170 | weight matmuls run at 73-78 % of DRAM bandwidth (376-397 GB/s); 1.14 GB of weights per step |
| RMSNorm (`LayerNormDeviceOperation`) | 4.7 ms (14 %) | 63 | **75 us each on 2 cores** (parallelizes over tile rows; the sequence is 2 tile rows) |
| `nlp_create_qkv_heads` | 4.4 ms (13 %) | 28 | **155 us each on 2 cores** |
| `nlp_concat_heads` | 1.5 ms (4 %) | 28 | 52 us each on 2 cores |
| eltwise add/mul/silu, SDPA, embedding | 1.3 ms (4 %) | 161 | SDPA is 10 us |

Perf artifacts: `tracy/eager/`, `tracy/traced/` (`ops.csv.gz`, `perf_report.txt` for the signposted
window, `perf_report.csv`, `perf_report.summary.txt`, `pytest.log`). Column used: `Device Time` (us) of
`perf_report.csv` filtered by the `PERF_DEPTH_*` signposts. Tracy tooling notes (tools folder flag,
tolerant post-processing) are the same as stage 02 (`scripts/collect_llm_perf.sh`).

## Decisions taken (nobody to ask)

* **Reference in `python_env`**: python_env's diffusers 0.38 lacks the class, so
  `reference/depth_decoder_ref.py` is a torch-only transcription (Apache-2.0 header kept, source cited);
  the golden-frame test proves it matches the real diffusers module (PCC 1.000000).
* **Unconditional row in the golden test**: the golden dump only holds the conditional row's backbone
  hidden, so both batch rows are fed row 0. Rows are independent (per-row causal attention), and only
  row 0 is compared, as in the pipeline (`hidden_parts.append(hidden[:1])`).
* **`MAX_STEPS = 9`**: the pipeline builds at most 8 positions (hidden, c0, c1..c6) but the module
  accepts 9 so a caller may append c7 too (e.g. to embed a complete frame); the position table has 16
  slots, so up to 16 would be representable, but nothing needs it.
* **`ttnn.reshape` replaced by `ttnn.experimental.view`**: the trace probe (`scripts/trace_probe_depth.py`)
  showed every op is trace-safe; `ttnn.reshape` on tile tensors would be a host fallback candidate, so
  the batch/row splits use the zero-copy `view` explicitly (the shapes are tile-aligned).
* **Trace capture must be preceded by a full compile run of the exact op sequence**: the first trace
  attempt failed with "Writes are not supported during trace capture" because the two output copies
  (`ttnn.copy` into the persistent hidden/logits buffers) were first executed inside the capture and
  compiled there (kernel load = host write). After such a failed capture the mesh close waits forever
  in `FDMeshCommandQueue::clear_expected_num_workers_completed` (tt-triage: no worker kernels running,
  dispatch idle; evidence kept under `generated/triage/`); the process had to be killed, no reset was
  needed (`tt-smi -ls` healthy, next open fine). `DepthStepTrace._capture` now runs the whole sequence
  including the copies before capturing.
* **Heads fused for the traced step**: the trace computes all 7 heads (`heads_all`, 58 MB of weights,
  ~0.15 ms) because the head index changes per step; the caller slices the block it needs.
* **No optimization in this stage** beyond correctness-neutral choices (fused QKV, fused SiLU, fp32
  accumulation). The 2-core norm / head-split ops are documented above for stage 07.

## Open risks / hand-off to later stages

* **30 % of the depth-loop device time is in three ops that run on 2 cores** (RMSNorm, create/concat
  heads) because the sequence occupies only 2 tile rows. Stage 07 candidates: width-sharded RMSNorm
  (as tt_transformers decode), or replacing create/concat heads by per-head batched matmuls with
  SDPA over `[heads, batch, 32, 256]`; expected ~1.5 ms/step saving (of ~4.9 ms).
* **Matmuls are DRAM-bandwidth bound on bf16 weights** (1.14 GB read per step, 7 steps per frame,
  25 frames/s of audio => ~35 ms of depth decoding per generated frame, i.e. ~9 s per 10 s clip on
  top of the backbone). bfp8 weights would halve that; precision impact goes to the stage-07 dtype sweep.
* The traced step's per-step read-back (logits -> host for sampling) is inherent to the pipeline's
  top-k sampling; an on-device sampler is a stage-04/07 option.
* Watcher run: not done in this stage (the stage prompt does not require it; the ops used are all
  stock ttnn ops already exercised watcher-clean in stage 02's backbone, except `nlp_create_qkv_heads`
  / `nlp_concat_heads` / `embedding` on these shapes).

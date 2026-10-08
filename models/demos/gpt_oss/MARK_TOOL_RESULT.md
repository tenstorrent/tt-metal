# gpt-oss-20b decode, optimized by Mark's tool (tt-model-bringup)

This branch is the result of an automated optimization run of `models/demos/gpt_oss` for **gpt-oss-20b decode at
batch 1 on a QuietBox 2** (4 Blackhole chips as a 1x4 mesh, tensor parallel over 4). The run used Mark's
tt-model-bringup skills with Claude Opus 5.5, in a comparison with two other optimizer tools that started from the same
code and were scored by the same tests.

| | Decode, ms per token | Tokens/s/user | Top-1 | Top-5 | Traced top-1 | Mean logits PCC |
|---|---:|---:|---:|---:|---:|---:|
| Start (06ea5e2a40) | 18.34 | 54.5 | 77 | 97 | 78 | 0.9677 |
| This branch (27439e82e9) | **2.42** | **413** | **90** | **100** | **87** | **0.9784** |

Measured 2026-10-08 on one QB2, back to back with the start code, a board reset before each tree, 5 runs of the
perf test each (median; second pass of this branch 2.42 ms), and the README demo as an independent check
(2.44 ms, 410 tokens/s/user).

## When the new code runs

The optimized decode path switches on only for gpt-oss-20b at batch 1 on a Blackhole 1x4 mesh: TP 4, no expert
parallelism, 32 experts, one token per device, an 11x10 core grid and 8 DRAM banks
(`fused_decode_layout_supported` in `tt/fused_decode.py`). Every other model, mesh or batch (120b, Galaxy, larger
batches) runs the original code unchanged. Prefill is unchanged.

## How to check it

1. Check out this branch and build tt-metal (`./build_metal.sh`). The branch changes one C++ file,
   `tt_metal/impl/profiler/profiler.cpp`, which is a profiler fix shared by all three tools' start code.
2. Get the accuracy test's reference data. It is not in git. Unpack `gptoss20b_accuracy_ref.tgz` in the tt-metal
   root. It adds two files:
   - `generated/optimizer_reference/gpt-oss-20b-logits.pt` (139 MB), the HF fp32 reference logits;
   - `generated/optimizer_accuracy_baseline_gpt-oss-20b.json` (299 B), the start code's pinned scores
     (77 / 97 / 78 / 0.9677).

   Never delete the pinned-score file and never set `PCC_GATE_PIN_BASELINE`. Either one makes the test pin the
   tree it runs on as its own baseline, so it would pass trivially.
3. Run, from the tt-metal root with the python env active:

   ```
   bash models/demos/gpt_oss/tests/optimizer/run_check.sh      # 5 perf runs + the accuracy test
   ```

   or the two tests directly:

   ```
   export HF_MODEL=openai/gpt-oss-20b TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
   PERF_GATE_ROLE=verdict OPTIMIZER_DECODE_ONLY=1 \
     pytest models/demos/gpt_oss/tests/optimizer/test_optimizer_perf.py::test_optimizer_direct_perf[blackhole-1x4]
   pytest models/demos/gpt_oss/tests/optimizer/test_optimizer_pcc.py::test_optimizer_full_model_pcc[blackhole-1x4]
   ```

   The first run builds the weight cache and takes a few minutes longer. To compare with the start code, run the
   same commands on commit 06ea5e2a40 (about 18.34 ms per token).

What the two tests do:

- `test_optimizer_perf.py`: prefills the demo's first prompt, then calls the shared generator's `decode_forward`
  once per token for 129 tokens, with the decode trace and on-device greedy sampling, reading each token back to
  the host. The first call captures the trace and is not timed. It prints the mean host wall time of the other 128
  calls as `TRACE_STAGE_MS[decode]`, and the first 300 characters of the generated text as `GATE_TEXT`.
- `test_optimizer_pcc.py`: 100 positions (one prefill position, 99 teacher-forced decode steps) compared with the
  HF fp32 reference: top-1, top-5, traced top-1, mean logits correlation and top-100 correlation, each held to
  the pinned start scores, plus an absolute floor of 0.95 on the mean correlation.

Neither test, the demo, the shared generator (`models/tt_transformers`), TTNN nor tt_metal is modified by the four
optimization commits. All their changes are under `models/demos/gpt_oss/tt/`.

## What changed, commit by commit

Each figure is the traced decode time that step saved when it was measured during the run (ms per token). Steps
inside one commit are listed largest first.

**1. Fused decoder, 42ba55233d (18.35 to 6.42 ms).** Python with stock TTNN ops; no new kernels.
- All-reduce as one op: `ttnn.experimental.all_reduce_async` with a persistent buffer and global semaphore,
  30 cores, 2 links, replacing `ttnn.all_reduce`, which ran as 7 ops for this tensor. Together with RMSNorm on the
  same 30 cores, a bf16 residual and an unpadded o_proj weight: -4.57.
- Experts through the indexed mode of `ttnn.sparse_matmul` (`indices=` the 4 routed experts, per-expert bias
  fused), so only 4 result rows are produced instead of 32; expert width padded 720 to 768 so gate/up use 24
  cores; one-op SwiGLU; 64-wide router: -4.33.
- Explicit 1D matmul program configs for the router, QKV and o_proj: -2.21.
- Router top-4 and softmax as one op (`generalized_moe_gate`): -0.32.
- Gate and up packed into one matmul: -0.26. Fused RoPE, KV-cache write and QKV bias: -0.11.
  Fused weighted sum of the 4 experts: -0.07. Stage-review fixes (bf16 embedding, precision pins): -0.07.

**2. Optimized decoder, 480ab5ba70 (6.41 to 4.65 ms).** Adds hand-written Metalium kernels launched through
`ttnn.generic_op` (`tt/experts/stream.py`, `tt/experts/kernels/stream_*`).
- Own DRAM-streaming kernels: expert gate/up with bias and SwiGLU inside (-0.60), expert down with bias and
  weighted sum inside (-0.38), QKV, o_proj and router (-0.28).
- All-reduce payload packed into 3 tiles instead of 90: -0.12. Streaming cores grouped into rectangles: -0.13.
- o_proj LoFi: -0.09. Gate/up block width: -0.07. QKV writes Q/K/V heads directly: -0.05.
  SDPA key chunk 256 on full-attention layers: -0.03. Router top-4 inside its kernel: -0.02.

**3. Optimized multichip, 1ad37e266a (4.64 to 3.99 ms).** `tt/decode_boundary.py`, `kernels/boundary_*`.
- One op for each layer boundary (all-reduce, residual add and RMSNorm), keeping the residual as one flat bf16 row
  on one core per chip: -0.46.
- The boundary runs inside the next op: -0.12. The fabric send moves into the producing matmul: -0.05.
  HiFi2 in the boundary op: -0.02.

**4. Optimized full model, 27439e82e9 (3.98 to 2.42 ms).** `tt/decode_terminal.py`, `tt/decode_inputs.py`,
`kernels/terminal_*`, `kernels/decode_inputs.cpp`.
- Own streamed LM-head kernel over an evenly split vocabulary (50,272 per chip, no padding), with the scores laid
  out as 32 rows so top-32 runs on short rows, a merge kernel, and a one-kernel greedy pick: -1.45.
- Each LM-head core keeps its own top 32; position advance on chip: -0.06. Decode inputs (embedding and RoPE rows)
  in one op: -0.06.

## Known loose ends

- All 18 kernel files sit in `tt/experts/kernels/`, and `tt/experts/stream.py` also serves QKV, o_proj, the router
  and the LM head; only the expert kernels belong under `experts/`.
- The prefill all-reduce in `config.py` (`reduce_scatter_minimal_async` + `all_gather_async`) can read
  uninitialized device memory, so prefill results can depend on what ran on the device before. This bug is in the
  original model code, and the original code here is unchanged. A board reset before running avoids it. The run
  found and fixed it during a later precision-sweep stage, but that fix is not on this branch.

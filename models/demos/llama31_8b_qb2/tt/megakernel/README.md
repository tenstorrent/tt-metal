# Experimental QB2 decode fusion

This is a **hardware-validated local MLP prototype**, not a complete decode
megakernel. On Blackhole QB2, real-checkpoint gate/up, SwiGLU and down stages,
complete-layer paged replay/remapping, and a batch-one context-128 full-model
comparison pass. All 32 teacher-forced logit rows and all-layer KV caches are
bitwise identical to the native traced baseline; all 32 greedy tokens agree.
Initial warmed host decode is slower: 9.246 versus 8.786 ms/token (31 decode
steps, median of three runs). These are host generation timings, not device
profiler or serving measurements. See `PROGRESS.md` and run artifacts for limits.

`FusedMLP` combines local gate/up projection, SwiGLU, and down projection into
one generic program per chip. Eight projection cores and sixteen SFPU cores
coordinate through four program-local semaphores. The packed gate/up and product
stay in L1. A 128-byte row in a DRAM address table selects a layer's weights.
One object owns all referenced weights and reusable scratch through trace release.

The initial geometry is batch one, hidden width 4096, TP-local intermediate
width 3584, eight Blackhole DRAM banks, BFP4 gate/up weights, BFP8 down weights,
and BF16 activations. It retains the baseline's LoFi projections, BF16 partials
with packer L1 accumulation, and BF16 rounding between SiLU and multiplication.
Exact numerical agreement was measured with real layer-0 and layer-31 weights. The separate `swiglu` mode
replaces only layout/slicing/activation work and retains native projections.

`enable_experimental_decode(model, mode="mlp")` installs the body across the 32
layers before generator warmup/capture. Prefill, normalization, attention,
paged KV operations, collectives, embedding, head, and sampling retain their
existing implementations. There are still 32 host-orchestrated MLP invocations
per token; the body does not loop over complete decoder layers on device.
Output aliases scratch, so calls must be sequential and consumed before reuse.
This integration is not qualified for vLLM or concurrent requests.

After verifying the four-chip mesh, run serially with a bounded device runner
that captures triage before terminating a hung process:

```bash
pytest -q -s models/demos/llama31_8b_qb2/tests/test_decoder.py -k '1-129'
pytest -q -s models/demos/llama31_8b_qb2/tests/test_megakernel_mlp.py
pytest -q -s models/demos/llama31_8b_qb2/tests/test_megakernel.py
python -m models.demos.llama31_8b_qb2.tests.benchmark_megakernel \
    --mode baseline --context 128 --output /outside/repo/baseline
python -m models.demos.llama31_8b_qb2.tests.benchmark_megakernel \
    --mode mlp --context 128 --reference /outside/repo/baseline \
    --output /outside/repo/mlp
```

Layer tests use real checkpoint weights and embedding inputs, cross positions
127/128 and 255/256, migrate physical pages and change the same captured page
table in place, compare complete cache tensors, and replay traces repeatedly.
Full-model tests exercise greedy token feedback and teacher-forced logits/KV
across all layers. The comparison checks matched checkpoint/precision/sampling.
The reported host generation duration includes enqueue and final readback;
it is not a device-profiler or vLLM serving measurement.

An optional CPU-only HF reference provides full-model accuracy evidence on a
fixed teacher stream, in addition to the matched TT fusion comparison:

```bash
python -m models.demos.llama31_8b_qb2.tests.reference_megakernel \
    --context 128 --tokens 32 --output /outside/repo/hf-reference
```

Pass `--hf-reference /outside/repo/hf-reference/reference.pt` to both TT
benchmark modes. The same prompt and teacher tokens are checked explicitly;
HF uses BF16 checkpoint/cache precision, so it is an accuracy reference, not
the matched performance baseline. Logits, predicted tokens and all-layer KV
are saved outside Git. Failed TT comparisons preserve actual tensors.
The focused MLP test separately checks packed gate/up, SiLU/product, and down
outputs with distinct real layer-0/layer-31 weight-table rows and trace replay.

Run profiling in a separate process, without Watcher or serving:

```bash
python -m tracy -r --device-trace-profiler -o /outside/repo/profile \
    -m models.demos.llama31_8b_qb2.tests.benchmark_megakernel \
    --mode mlp --context 128 --repeats 3 --profile --output /outside/repo/profile-run
```

Compare the matched baseline invocation. `QB2_DECODE_BEGIN_*` / `END_*`
signposts enclose complete one-token model/sampling traces at a fixed context;
input/position refreshes are outside. `MLP-*` device zones separate projection,
SFPU, output write, and barrier phases. Profiler overhead prevents using these
runs as the generation latency result.

The next implementation stage is device profiling and optimization, then
folding normalization and fabric reductions into the body, then attention/KV
and device layer-loop coordination. Current DRAM-sharded matmul uses Metal 2.0
ProgramArtifacts; the legacy Python sequential fuser cannot directly consume
it. Blackhole has 64 CB indices; this prototype's highest index is 31, so CB
index exhaustion is not its current constraint.

`reuse_scratch=True` shares input/weight/partial backing storage between the
sequential projections. Its weight buffer holds one block to fit alongside
native prefill allocations; the default uses separate double-buffered weights.
Both configurations pass stage and complete-layer tests. Only the default has
full-model evidence so far. A larger shared allocation was rejected by current
runtime prefill L1 collision validation; do not restore it without rechecking.

The BF16 HF reference is a separate accuracy diagnostic: the original selected
BFP4/BFP8 baseline has aggregate logit PCC 0.97753 on this prompt, despite 100%
teacher top-1 agreement. The prototype reproduces it exactly. The saved result
explicitly records `hf_pcc_099_passed=false`; this is not a 0.99 HF accuracy pass.
Use `--require-hf-pcc` to make an HF threshold mandatory after saving evidence.

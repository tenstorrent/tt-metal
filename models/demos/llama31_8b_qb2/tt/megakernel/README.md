# Experimental QB2 decode fusion

This is an **unvalidated partial prototype**, not a complete decode megakernel.
Its Blackhole kernels compile with current main. At initial bringup, the owned
QB2 failed topology mapping and an isolated mesh failed ERISC initialization;
numerical and performance tests require recovered, verified hardware.

`FusedMLP` combines local gate/up projection, SwiGLU, and down projection into
one generic program per chip. Eight projection cores and sixteen SFPU cores
coordinate through four program-local semaphores. The packed gate/up and product
stay in L1. A 128-byte row in a DRAM address table selects a layer's weights.
One object owns all referenced weights and reusable scratch through trace release.

The initial geometry is batch one, hidden width 4096, TP-local intermediate
width 3584, eight Blackhole DRAM banks, BFP4 gate/up weights, BFP8 down weights,
and BF16 activations. It retains the baseline's LoFi projections, BF16 partials
with packer L1 accumulation, and BF16 rounding between SiLU and multiplication.
Exact numerical agreement still needs measurement. The separate `swiglu` mode
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

The next implementation stage is measured local-body bringup, followed by
folding normalization and fabric reductions into the body, then attention/KV
and device layer-loop coordination. Current DRAM-sharded matmul uses Metal 2.0
ProgramArtifacts; the legacy Python sequential fuser cannot directly consume
it. Blackhole has 64 CB indices; this prototype's highest index is 31, so CB
index exhaustion is not its current constraint.

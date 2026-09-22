# Experimental QB2 decode megakernel

This experiment has a **hardware-qualified partial prototype** and a larger
**compiler-only decoder/32-layer-loop prototype awaiting hardware validation**.
See [PROGRESS.md](PROGRESS.md) for exact checkpoints, results and recovery state.
No measured variant is faster than the original traced model.

The body preserves the selected checkpoint policy: GU BFP4, QKV/O/down/head
and KV BFP8, BF16 activations/residuals/collectives, LoFi projections with BF16
partials and packer L1 accumulation, native HiFi4/FP32 norm/RoPE/SDPA and HiFi2
head. Prefill stays in the original implementation. Batch one only.

| Mode | Composed stages | Qualification |
|---|---|---|
| `swiglu` | packed gate/up through BF16 SiLU and multiply | Real layer/replay |
| `mlp` | gate/up, SwiGLU, down | Real full32-layer context128 |
| `mlp_reduce` | MLP and four-chip reduce-scatter | Real full32-layer context128 |
| `mlp_tail` | above plus residual | Real full32-layer context128 |
| `norm_mlp_tail` | native RMSNorm and MLP tail | Real full32-layer context128 |
| `gather_norm_mlp_tail` | all-gather and normalized MLP tail | Compiler only |
| `post_attention` | O projection, RS/add, AG/norm, MLP, RS/add | Compiler only |
| `attention_tail` | native paged SDPA, concat and post-attention path | Compiler only |
| `decoder` | complete single decoder layer, including QKV/RoPE/paged KV | Compiler only |
| `decoder_loop` | one device program loops over the layer weight/KV table | Compiler only |
| `decoder_loop_embedding` | token embedding and the device layer loop | Compiler only |
| `decoder_loop_head` | embedding/layer loop, then native AG and fused final norm/head | Compiler only |
| `decode_token` | embedding,32-layer loop, final AG/norm/head in one program | Compiler only |

Qualified modes produce bitwise-identical teacher logits and all64 KV tensors,
32/32 matching greedy outputs and stable repeated generations at B1/context128.
Page tests migrate physical pages, mutate the captured page table in place,
cross positions127/128 and255/256, and replay repeatedly. Real MLP intermediates
also match for distinct layers0/31. Host medians: baseline8.785510ms/token,
MLP9.246204, MLP+RS9.161502, MLP+RS/add9.133671, norm+tail9.336578.
These are31-step generation timings, not device or serving latency.

BF16 Hugging Face diagnostics are separate from the matched quantized TT
comparison: baseline and exact prototypes both have logitPCC0.977533 and100%
teacher top1 agreement on the fixed prompt. The provisional0.99 HF target fails
for both. This does not establish task accuracy. Context2048 baseline is saved
at9.223623ms/token; matched prototype testing remains pending.

`FusedMLP` owns weights and reusable scratch. A128-byte DRAM row selects a
layer's weights; `DecoderLoop` additionally binds each layer's two KV addresses
before warmup. Outputs alias scratch and must be consumed before reuse. Loop
workers use current DeepSeek cross-RISC synchronization and64-CB reset helpers;
start/end synchronization words are distinct. Global layer barriers protect
local semaphore/CB reset, while native fabric initialization coordinates chips.
Attention retains its native32-core grouping on separate cores and uses CB32.
The whole-layer layout requires Blackhole's real11x10 grid and GU8 workers.

The embedding variant reads BF16 row-major checkpoint rows directly into the
first residual buffer. `decode_token` additionally places sixteen HiFi2 head
workers and eight final norm workers on the remaining24 cores. A terminal
all-gather borrows the last reduction's completed output and open connections;
it preserves that reduction's counters. The final norm reuses the layer norm
scratch. Sampling remains the native traced boundary. The earlier modes retain
their native terminal operators. No persistent multi-token loop or vLLM
integration is implemented. Negative/inactive positions are not supported. Cache allocations cannot change while a loop/trace references
their address table; page-table contents and positions remain device inputs.

After health/ownership verification, run these **serially with bounded commands**:

```bash
pytest -q -s models/demos/llama31_8b_qb2/tests/test_megakernel_mlp.py
pytest -q -s models/demos/llama31_8b_qb2/tests/test_megakernel.py
pytest -q -s models/demos/llama31_8b_qb2/tests/test_megakernel_loop.py
pytest -q -s models/demos/llama31_8b_qb2/tests/test_megakernel_head.py
python -m models.demos.llama31_8b_qb2.tests.benchmark_megakernel \
    --mode baseline --context 128 --output /outside/repo/baseline
python -m models.demos.llama31_8b_qb2.tests.benchmark_megakernel \
    --mode decoder_loop_embedding --context 128 --reference /outside/repo/baseline \
    --output /outside/repo/prototype
```

Use `reference_megakernel --context 128 --tokens 32 --output ...` to prepare the
CPU HF reference, then pass its `reference.pt` with `--hf-reference` to both TT
runs. The harness verifies matched prompt/checkpoint/precision/sampling/teacher
stream and saves logits, tokens and all caches outside Git. Failures retain
actual tensors. `test_megakernel_loop.py` selects real layers0/31 and separate
cache allocations to exercise table selection, reset and page remapping.

Collect device profiling in a separate process, without serving or Watcher:

```bash
python -m tracy -r --device-memory-profiler --op-support-count 4000 -o /outside/repo/profile \
    -m models.demos.llama31_8b_qb2.tests.benchmark_megakernel \
    --mode norm_mlp_tail --context 128 --repeats 3 --profile --output /outside/repo/profile-run
```

The harness drains profiler buffers between windows; captures missing operations
must not support comparisons. Saved qualified baseline/GU16 reports have three
complete windows/all four devices. Device3 median firmware span9.393458ms
baseline versus9.853521ms GU16 MLP+RS. Kernel durations overlap with waits; do
not sum phase durations or firmware operation durations into latency. Profiler
zones include MLP, attention/concat, paged KV and loop barriers. Hardware DRAM
traffic counts and final-mode measurements remain outstanding.

Optional `reuse_scratch=True` aliases projection input/weight/partial storage,
using one weight block to coexist with native prefill allocations. It passes
real stage/layer tests; full-model performance is unqualified. GU16 passes the
partial model tests but is slower; complete-layer composition requires GU8.

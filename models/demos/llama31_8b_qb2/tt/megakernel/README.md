# Experimental QB2 decode megakernel

`decode_token` is a **hardware-qualified batch-one token-to-logits prototype**
with embedding, a 32-layer device loop, and final norm/head in one four-chip
program. Real-model checks pass at contexts 128 and 2048.
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
| `gather_norm_mlp_tail` | all-gather and normalized MLP tail | Real layer/remap/replay |
| `post_attention` | O projection, RS/add, AG/norm, MLP, RS/add | Real layer/remap/replay |
| `attention_tail` | native paged SDPA, concat and post-attention path | Real layer/remap/replay |
| `decoder` | complete single decoder layer, including QKV/RoPE/paged KV | Real layer/remap/replay |
| `decoder_loop` | one device program loops over the layer weight/KV table | Real layers 0/31, inactive warmup/remap/replay |
| `decoder_loop_embedding` | token embedding and the device layer loop | Components verified inside decode_token |
| `decoder_loop_head` | embedding/layer loop, then native AG and fused final norm/head | Separate composition unqualified; head alone exact |
| `decode_token` | embedding,32-layer loop, final AG/norm/head in one program | Real full model, contexts 128/2048 |

Full-model-qualified modes produce bitwise-identical teacher logits and all64 KV tensors,
32/32 matching greedy outputs and stable repeated generations at B1/context128.
Page tests migrate physical pages, mutate the captured page table in place,
cross positions127/128 and255/256, and replay repeatedly. Real MLP intermediates
also match for distinct layers0/31. Host medians: baseline8.785510ms/token,
MLP9.246204, MLP+RS9.161502, MLP+RS/add9.133671, norm+tail9.336578.
Full `decode_token` with NoC scratch clearing at context128 is 9.221568ms/token
versus 8.785510 baseline; context2048 is 9.639827 versus 9.219413 baseline. Both contexts
match all teacher logits/KV bitwise and all 32 greedy outputs. These are medians
of three 31-step warmed generations, not device-profiler or serving latency.

BF16 Hugging Face diagnostics are separate from the matched quantized TT
comparison: baseline and exact prototypes both have logitPCC0.977533 and100%
teacher top1 agreement on the fixed prompt. The provisional0.99 HF target fails
for both. Context2048 has HF logitPCC0.984128 and teacher top1 agreement93.75%
for both baseline and prototype on the same forced stream. These diagnostic
prompts do not establish task accuracy.

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
integration is implemented. Position -1 skips KV access for native inactive
warmup; three such warmups preserve all cache pages in the real loop tests.
Batches above one and serving are unqualified. Cache allocations cannot change
while a loop/trace references their address table; page-table contents and
positions remain device inputs. Every native fabric phase drains and closes
before PacketHeaderPool::reset reuses its headers. The complete body borrows
native attn/o/down workspace buffers, avoiding extra persistent storage that
otherwise conflicts with the context2048 prefill head.

After health/ownership verification, run these **serially with bounded commands**:

```bash
pytest -q -s models/demos/llama31_8b_qb2/tests/test_megakernel_mlp.py
pytest -q -s models/demos/llama31_8b_qb2/tests/test_megakernel.py
pytest -q -s models/demos/llama31_8b_qb2/tests/test_megakernel_loop.py
pytest -q -s models/demos/llama31_8b_qb2/tests/test_megakernel_head.py
python -m models.demos.llama31_8b_qb2.tests.benchmark_megakernel \
    --mode baseline --context 128 --output /outside/repo/baseline
python -m models.demos.llama31_8b_qb2.tests.benchmark_megakernel \
    --mode decode_token --context 128 --reference /outside/repo/baseline \
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
    --mode decode_token --context 128 --tokens 3 --repeats 3 --profile --output /outside/repo/profile-run
```

The harness drains profiler buffers between windows; captures missing operations
must not support comparisons. Saved qualified baseline/GU16 reports have three
complete windows/all four devices. Device3 median firmware span9.393458ms
baseline versus9.853521ms GU16 MLP+RS. Kernel durations overlap with waits; do
not sum phase durations or firmware operation durations into latency. Profiler
zones include MLP, attention/concat, paged KV and loop barriers. The first complete token profile has35ops/device/window versus999 baseline;
optimized timings and hardware DRAM-addressed NoC payload results are being
recorded in PROGRESS.md and the external REPORT.md.

Optional `reuse_scratch=True` aliases projection input/weight/partial storage,
using one weight block to coexist with native prefill allocations. It passes
real stage/layer tests; full-model performance is unqualified. GU16 passes the
partial model tests but is slower; complete-layer composition requires GU8.

For focused DRAM-addressed NoC payload accounting, keep traffic instrumentation
separate from latency captures:

```bash
python -m tracy -r --collect-noc-traces -o /outside/repo/traffic-profile \
    -m models.demos.llama31_8b_qb2.tests.profile_megakernel_traffic \
    --mode mlp --output /outside/repo/traffic-run
python -m models.demos.llama31_8b_qb2.tests.analyze_megakernel_traffic \
    --logs /outside/repo/traffic-profile/.logs \
    --ops-csv /outside/repo/traffic-profile/reports/DATE/ops_perf_results_DATE.csv \
    --extended-payload --expected-dram-read-bytes 32113664 \
    --output /outside/repo/traffic-counts.json
```

Repeat with `--mode baseline` at the same precision and geometry. The analyzer
joins all four devices' captured operations to signposted replay windows and
rejects missing files/durations, unequal operation counts, unresolved request
state, saturated sizes and unbalanced kernel endpoints. Also inspect capture
logs for dropped records. These are issued NoC payload bytes at32B resolution,
not DRAM-controller bus counters or serving latency. Qualified real layer0 captures cover three windows/all four chips: baseline
32,112,640 DRAM-addressed read bytes per chip/layer, fused32,113,664 (including
1,024 bytes of table reads). Use32112640 for the baseline expected-payload gate.
The fused body does not reduce weight bytes; local NoC read traffic increases
from1,146,880 to4,390,912 bytes while local writes fall from1,867,776 to262,144.
These focused MLP measurements do not represent full-model traffic.

This branch fixes a confirmed profiler encoding truncation: the old8-bit
payload field capped every request above8,160B, including the16KB projection
reads. Seven previously reserved bits now extend the cap to1,048,544B while
preserving old captures' low-byte/posted-bit positions. Use matching rebuilt
host and device profiler code; old analysis binaries do not understand the new
high bits. Host wire-format boundary tests, real SFPI builds and hardware captures pass.
A second profiler fix reserves the complete four-word event before flushing;
the old one-word test silently discarded later requests near a buffer boundary.
Independent expected-weight-payload checks exposed this even when kernel
endpoints balanced. Both `tar` and `tt_pybinds` install components must be
refreshed after rebuilding the host decoder.

The full-layer path reads metadata and RoPE rows into scratch whose low six
address bits match the DRAM source. Page entries remain individual4B reads,
including short page tables; RoPE rows are tiled by local copies after a256B
read. This fixes violations of Blackhole's64B read-congruence rule found during
review. A CPU address/face-layout audit, full-layer page migration tests and
full-model context128/2048 checks pass. Generator warmup deliberately uses position
-1. Cache compute receives that flag on a separate CB with mailbox forwarding to
all three TRISCs, avoiding divergent branches; the native cache arithmetic runs
only for active positions. Inactive attention scratch is zeroed and existing KV
pages remain the test's required invariant. This is not a serving qualification.

# Self-conditioning embedding prechunk (dg-08)

Date: 2026-07-09
Target: DiffusionGemma 26B-A4B-it, QB2 `P150x4`, TP=4, 256-token canvas, BF16, traced fixed-step serving
Build limitation: `ENABLE_TRACY=OFF`; synchronized component timing and Metal trace replay are used.

## Result

The 256K-vocabulary self-conditioning signal is evaluated as 32 ordered 8K-vocabulary chunks. The
old path stored one `[262144, 2816]` BF16 embedding table and copied an 8K-row device slice before
every chunk matmul, on every denoise step. The selected path constructs the same table as 32
persistent 8K-row tensors and directly supplies each tensor to the existing matmul.

This changes no values, dtype, matmul shape, chunk boundary, or accumulation order. It only removes
32 repeated embedding-table `ttnn.slice` operations per step. The chunks are built before trace
capture, so their addresses are persistent. Total embedding-weight payload is unchanged; the
monolithic tensor is replaced rather than retained.

`DG_SELFCOND_PRECHUNK_EMBED=0` is the diagnostic opt-out. Prechunking is the default after the
bit-identity and traced-performance gates below.

## Synchronized component gate

Harness: `doc/optimize_perf/prof_step_breakdown.py`, real checkpoint, two real MoE layers, five
warmed iterations, synchronized once after each timed batch.

| path | full two-layer step (ms) | soft embedding (ms) | other checked components |
|---|---:|---:|---|
| monolithic (`DG_SELFCOND_PRECHUNK_EMBED=0`) | 82.334 | 25.863 | hidden 48.223; LM head 4.361; terminal 28.906 |
| prechunked (`=1`) | **73.575** | **18.210** | hidden 47.576; LM head 4.362; terminal 28.899 |
| delta | **-8.759 (-10.6%)** | **-7.653 (-29.6%)** | controls flat |

The harness was repaired before this measurement because both `denoise_hidden_forward` and
`_apply_lm_head` consume/deallocate their inputs. Each timed invocation now receives a clone; the
persistent benchmark source remains allocated.

## Full-model traced A/B

Harness: `doc/optimize_perf/bench_lever_e2e.py`, full 30 layers, seed 0, RUN-first argmax sampling, sparse tuned
MoE, terminal argmax dedup, single-step Metal traces, three 256-token blocks. Steady latency is the
mean of blocks 2–3. Separate fresh processes were used because the weight representation is selected
at model construction.

| steps | path | steady block (s) | tokens/s | committed SHA over 3 blocks |
|---:|---|---:|---:|---|
| 48 | monolithic | 13.9946 | 18.293 | `a9f0d18709b07d1e` |
| 48 | prechunked | **13.6555** | **18.747 (+2.48%)** | `a9f0d18709b07d1e` |
| 12 | monolithic | 4.3943 | 58.258 | `24393ba7aad6077c` |
| 12 | prechunked | **4.3401** | **58.984 (+1.25%)** | `24393ba7aad6077c` |

The 48/12 block-time slope gives a warmed traced denoise step of 266.675 ms before and 258.761 ms
after: **7.914 ms/step saved (2.97%)**. That independently agrees with the synchronized
soft-embedding delta of 7.653 ms. The per-block 48-step saving is 339.1 ms.

The committed SHA is exact at both budgets (768 committed tokens per row), so this candidate clears
the stronger output-identity gate; no #48291-relative or HF-fidelity waiver is needed. Generated text
heads also match within each budget. Precision remains BF16 and the online-softmax math remains the
existing ordered 8K-chunk algorithm.

### Final unset-default reproduction

After changing the selector default, fresh one-budget processes were run with
`DG_TRACE_REGION_SIZE=10737418240` set before mesh open. The harness now fails before mesh open unless
the reservation is **exactly** 10 GiB and the workload is the canonical P150x4/TP=4, 256-canvas,
seed-0, three-block @48/@12 configuration. It records raw/resolved selectors, trace bytes, prefill,
TTFT, every block, total generation, steps, and commit SHA. `selfcond_prechunk_e2e.json` embeds the
five process rows plus checkpoint/tokenizer hashes, source HEAD, dirty-worktree source hashes, build
cache hash, mesh/TP, prompt, seed, allocation, and `ENABLE_TRACY=OFF` provenance.

| steps | path | steady block (s) | tokens/s | committed SHA |
|---:|---|---:|---:|---|
| 48 | monolithic control | 14.0971 | 18.160 | `a9f0d18709b07d1e` |
| 48 | explicit candidate (`=1`) | **13.6996** | **18.687 (+2.90%)** | `a9f0d18709b07d1e` |
| 48 | unset final default | **13.6817** | **18.711 (+3.03%)** | `a9f0d18709b07d1e` |
| 12 | monolithic control | 4.4479 | 57.555 | `24393ba7aad6077c` |
| 12 | unset final default | **4.3710** | **58.568 (+1.76%)** | `24393ba7aad6077c` |

The exact-provenance 48/12 slope is **268.033 → 258.631 ms/warmed traced step**, a 9.403 ms
(3.51% latency, 3.64% step-rate) improvement. The selected @48 row supersedes the preliminary
18.819 t/s row collected before trace-region provenance was emitted; the 0.57% difference is normal
run variance and does not change selection.

The complete serving-session model path also wins, including prefill, first-block trace capture, and
three commits; the timer ends after the third block and excludes post-run hashing/text formatting:

| @48 metric | control | explicit candidate | unset final default |
|---|---:|---:|---:|
| prefill (s) | 1.3273 | 1.3160 | 1.1120 |
| TTFT = prefill + block 0 (s) | 127.227 | 125.763 | **125.977** |
| all block latencies (s) | 125.8990, 14.1206, 14.0735 | 124.4468, 13.7019, 13.6974 | **124.8640, 13.7052, 13.6582** |
| full prefill + 3-block generation (s) | 155.4222 | 153.1637 | **153.3410** |

The unset default improves complete generation by 1.36% and the explicit/default rows agree within
0.13% at steady state. `selfcond_prechunk_e2e.json` is the compact machine-readable evidence.

## Exact diffusion-decision gate

`verify_selfcond_prechunk_decisions.py` first ran two fresh full 30-layer processes over one complete
48-step, 256-token trajectory with the same prompt tokens, initial-canvas hash, and 48 explicitly
injected random-renoise token tensors in the traced benchmark's RUN-first regime (Gumbel noise
`None`). Every step compared hashes of:

- clean argmax;
- sampled token ids;
- BF16 per-position entropy;
- entropy-budget accept mask;
- accepted/renoised next canvas;
- the explicit clean commit candidate for that step (DiffusionGemma commits the last such candidate).

Result: **48/48 steps exact for all six recorded fields**, including exact entropy means and accept counts;
the aggregate trajectory SHA is
`b2e74f4edd6a2e3562b81b04b2f94bdb9881011b225f6f353cb8668449d2ab51`, and the final commit SHA is
`e3b1344d8f795aa0c40a8d96c58e7d94bdb3c234ac9c67bf3d21faed687eafdc` in both runs.
The artifact also asserts that the last per-step candidate hash equals the trajectory's actual commit
hash in both processes. (No KV commit occurs during intermediate denoise steps; those rows are
candidates.) `selfcond_prechunk_decisions.json` persists both per-step hash sets and exactness flags.

The same gate then ran the production memory-bounded sampler with 48 deterministic
`ChunkedGumbelNoise` descriptors (seeds 2–49, chunk size 1024, FP32 noise). Control/default Gumbel
descriptor hash `fb8f0108c5516e18a00ae30ad67f81990e3e1886973497a13f3941e67e1d7aa3`,
initial canvas, all renoise tensors, and prompt tokens are identical. Result: **48/48 exact** for
sampled IDs, clean argmax, entropy, accept mask, renoised canvas, and every clean commit candidate;
trajectory SHA `55260b7946ce281c85449030f5f177fc320725adcd7d6757ec7265f103a9c0cf`.
See `selfcond_prechunk_gumbel_decisions.json`.

This closes production-sampler semantic fidelity, but not traced Gumbel performance:
`tt/traced_denoise.py` explicitly accepts only `gumbel_noise=None` and raises on a real
tensor/descriptor. Therefore all ranked traced numbers in this document are labeled RUN-first
argmax; no production-Gumbel traced throughput is claimed and the already-characterized traced-loop
implementation is not re-ground here.

## Prompt-correct qualitative gate

`qualitative_prechunk.py` ran three chat-template-rendered prompts at 16 fixed steps in fresh
monolithic-control and unset-default processes, first with traced RUN-first argmax and then with the
production chunked-Gumbel sampler. The checkpoint has a non-empty Gemma chat template; each committed
block SHA and complete decoded text matched exactly within both A/Bs:

| prompt | control/default committed SHA | classification |
|---|---|---|
| greeting | `372cc8350b7f0524` / `b2f37af19faecf20` | Chinese prefix, then punctuation/digit/whitespace degeneration; unchanged |
| explain diffusion | `e12f4556267c918b` / `e12f4556267c918b` | coherent answer, then mixed punctuation/Chinese token and severe repetition; unchanged |
| Moon phases | `4882bbccf6826532` / `e0578d3bef313f68` | correct answer, then repetition or emoji/replacement-glyph degeneration; unchanged |

The SHA pairs are argmax / chunked-Gumbel. Rendered prompts, token IDs, complete decoded outputs,
checkpoint/tokenizer hashes, explicit anomaly ledgers, and verdicts are in
`selfcond_prechunk_qualitative.json` and `selfcond_prechunk_gumbel_qualitative.json`. None of the
stored decoded outputs contains the literal word `user`; it appears only in rendered chat-template
prompts. The optimization introduces no qualitative regression, but the artifacts do not dismiss
the pre-existing #48291 quality defects.

## Persistent allocation and 256K capability

The table payload is exactly `262144 × 2816 × 2 = 1,476,395,008` bytes/chip (1408 MiB) before and
after. Storage changes from one 1408 MiB allocation to 32 × 44 MiB allocations; persistent bytes
change by zero and allocation count changes by +31.

A full 30-layer, `max_seq_len=262144`, 256-token-canvas final-default smoke passed with a non-aligned
24-token logical prompt and the production memory-bounded chunked-Gumbel sampler for the **full
48-step budget**: post-build 29.704 GiB/chip; post-prefill+one committed block 31.134 GiB/chip;
0.733 GiB/chip free; `DG_VLLM_SERVING_SMOKE_SUCCESS`; clean teardown. Durable provenance and complete
output are in `selfcond_prechunk_256k_chunked.json`.

The full-depth 256K allocation cannot currently be combined with Metal trace capture. With zero
reservation, model operations fit but `end_trace_capture` detects the trace buffer overlapping the
DRAM high-water mark. Reservations of 512, 256, 192, and 176 MiB avoid or approach that overlap but
leave no contiguous 128 MiB buffer for `token_entropy`'s `exp_shifted`/`expected_terms`. This is an
exact full-vocab-entropy-plus-trace capacity limit at max context; it does not reduce the advertised
256K eager chunked-Gumbel capability. The full 30-layer traced RUN-first argmax path and trace-safe canvas feedback remain verified
at the performance allocation (`max_seq_len=1024`) for 48 and 12 steps. The updated
`doc/context_contract.json` records both facts rather than implying traced 256K support.

## Commands

Environment common to hardware runs:

```bash
source /home/zni/venvs/tt-diffusion-gemma/bin/activate
export PYTHONPATH=/home/zni/tt-metal:/home/zni/tt-metal/ttnn
export TT_METAL_HOME=/home/zni/tt-metal TT_METAL_RUNTIME_ROOT=/home/zni/tt-metal
export DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARCH_NAME=blackhole
```

Component A/B:

```bash
DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 \
DG_SELFCOND_PRECHUNK_EMBED=0 python -u \
  models/experimental/diffusion_gemma/doc/optimize_perf/prof_step_breakdown.py \
  --num-layers 2 --iters 5

DG_SPARSE_MOE=1 DG_SPARSE_MOE_TUNED=1 DG_DEDUP_ARGMAX=1 \
DG_SELFCOND_PRECHUNK_EMBED=1 python -u \
  models/experimental/diffusion_gemma/doc/optimize_perf/prof_step_breakdown.py \
  --num-layers 2 --iters 5
```

Full traced rows (run in fresh processes with `DG_SELFCOND_PRECHUNK_EMBED=0`, `=1`, and unset):

```bash
DG_TRACE_REGION_SIZE=10737418240 DG_SELFCOND_PRECHUNK_EMBED=0 python -u \
  models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py \
  --levers baseline --budgets 48 --blocks 3 --out control.json

DG_TRACE_REGION_SIZE=10737418240 DG_SELFCOND_PRECHUNK_EMBED=1 python -u \
  models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py \
  --levers baseline --budgets 48 --blocks 3 --out candidate.json

env -u DG_SELFCOND_PRECHUNK_EMBED DG_TRACE_REGION_SIZE=10737418240 python -u \
  models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py \
  --levers baseline --budgets 48 --blocks 3 --out default.json
```

Exact @48 production chunked-Gumbel decision control/default (omit
`--gumbel-mode chunked` to regenerate the RUN-first argmax artifact):

```bash
DG_SELFCOND_PRECHUNK_EMBED=0 python -u \
  models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py \
  --steps 48 --gumbel-mode chunked --out decisions-control.json

env -u DG_SELFCOND_PRECHUNK_EMBED python -u \
  models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py \
  --steps 48 --gumbel-mode chunked --out decisions-default.json

python -u models/experimental/diffusion_gemma/doc/optimize_perf/verify_selfcond_prechunk_decisions.py \
  --compare decisions-control.json decisions-default.json --out decisions-comparison.json
```

Production chunked-Gumbel qualitative and 256K capability commands:

```bash
DG_SELFCOND_PRECHUNK_EMBED=0 python -u \
  models/experimental/diffusion_gemma/doc/optimize_perf/qualitative_prechunk.py \
  --steps 16 --gumbel-mode chunked --out qualitative-control.json

env -u DG_SELFCOND_PRECHUNK_EMBED python -u \
  models/experimental/diffusion_gemma/doc/optimize_perf/qualitative_prechunk.py \
  --steps 16 --gumbel-mode chunked --out qualitative-default.json

python -u models/experimental/diffusion_gemma/doc/optimize_perf/qualitative_prechunk.py \
  --compare qualitative-control.json qualitative-default.json \
  --out qualitative-comparison.json

env -u DG_SELFCOND_PRECHUNK_EMBED -u DG_DENOISE_TRACED python -u \
  models/experimental/diffusion_gemma/demo/serving_smoke.py \
  --num-blocks 1 --canvas-length 256 --max-denoising-steps 48 \
  --max-seq-len 262144 --gumbel-mode chunked \
  --disable-eos-stop --local-files-only \
  --metrics-json selfcond_prechunk_256k_chunked.json
```

Device-free gate:

```bash
pytest -q models/experimental/diffusion_gemma/tests/test_tt_self_conditioning.py \
  models/experimental/diffusion_gemma/tests/test_denoise_forward.py
```

Additional gates:

- `test_tt_self_conditioning.py` + `test_denoise_forward.py`: **40 passed** before the independent
  review follow-up; **40 passed** after adding the chunk-consumer partial-vocabulary/lifecycle
  assertions (the number of test functions is unchanged). The latter exposed two
  stale test doubles that did not accept the already-landed `canvas_rope_provider` keyword; the mocks
  now cover that trace-safe call contract.
- Separate watcher smoke: `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`, two real layers,
  traced 4-step/one-block path, non-aligned logical prompt length 24. Result:
  `DG_VLLM_SERVING_SMOKE_SUCCESS`, 256 tokens emitted; watcher attached/detached all four devices and
  reported no error/assert/NoC violation. The full 5,295,465-byte log has SHA-256
  `25a5a17fd87d7e84bf7f2ac2f12f9a2e70dbd427c520b578e8f947e23ae1a051`; all four exact attach and
  detach records, the zero-match error scan, metrics hash, and post-run four-device `tt-smi` listing
  are in `selfcond_prechunk_watcher_summary.json`.
- Batch-local no-shared-edits gate:
  `DG_BASE_REF=0472860c40c .../check_no_shared_gemma4_edits.sh` → **OK**. The branch merge-base gate
  still reports the pre-existing `models/demos/gemma4/tt/experts/operations.py` delta already
  attributed to dg-04 in the prior campaign review; this batch does not touch it.

Infrastructure recovery: an exploratory final-default run accidentally inherited
`TT_METAL_WATCHER=10`, so its 114.8 s watcher-instrumented blocks were discarded. Stopping that
contaminated run left one ERISC heartbeat stale; the next mesh open failed before model code.
Recovery followed the device skill exactly: bounded four-device list → `tt-smi -r` → bounded
four-device list → `(1,4)` open/close `MESH_SMOKE_OK`. All accepted measurements were collected after
recovery with watcher/profiler variables explicitly unset.

## Limits

- This optimization removes embedding-table slices only. The 32 dynamic logits slices and online
  softmax/matmul operations remain.
- The production vocabulary is exactly divisible by 8192. The builder and unit gate retain a final
  short chunk for non-divisible vocabularies.
- Model-load allocation count increases by 31, but weight bytes do not increase. No extra runtime
  copy or host fallback is introduced.
- Prompt length, canvas length, and KV state are untouched. The 256K eager production
  chunked-Gumbel capability is freshly reproduced at 48 steps; traced 256K remains blocked by the
  full-vocab entropy/trace-region capacity interaction detailed above.
- `traced_denoise.py` supports the RUN-first `gumbel_noise=None` regime only. Production
  chunked-Gumbel decisions and capacity pass, but no traced Gumbel throughput is claimed.

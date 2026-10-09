# Stage 1: functional layers (prefill only)

Model `perplexity-ai/pplx-decider-v1-27b` at revision `b01a5cbaca5391f73bd55103d4f27e8982cd5e60`, on one
Blackhole p150a (13x10 compute grid, 8 DRAM banks, 31.83 GiB DRAM). Code is at tt-metal branch
`gtobarTT/pplx-decider-bringup`. Labels: **measured** means command output recorded here or in
`work_log.md`; **inferred** means it follows from code or arithmetic and was not run.

## Contract

This stage uses an adapted prefill-only contract that the person approved. The source is
`tt-ops/runs/pplx-decider-v1-27b/notes.md`, amended 2026-10-09.

- **What runs.** Per-module and per-layer TTNN prefill for both decoder-layer kinds:
  - `linear_attention`: Gated DeltaNet, 48 layers.
  - `full_attention`: gated GQA attention with partial RoPE, 16 layers, at index 3 mod 4.
  - Also: embedding, zero-centred RMSNorm, SwiGLU MLP, final norm and the 5120x255 readout.
- **Lengths.** The tests use the prefill buckets **128, 1024, 2048, 4096 and 8192**. The app pads a
  prompt on the right to the next bucket. The classifier reads the hidden state of the last *real*
  token.
  - This rule replaces the plugin's arbitrary-logical-length requirement, by person decision.
  - Right padding cannot change a real token's hidden state, because both mixers are causal
    (inferred).
  - The last-real-token readout is validated end to end in stage 6.
  - The layer code still accepts any length from 1 to `max_seq_len`: chunks of 2048 plus a tail,
    with tile padding inside.
- **Context.** HF advertises 262144 tokens. The app limit is `max_length=8192`
  (`source/src/autojev/model.py:180`), and this stage supports and tests 8192.
  - The app contract is fully met: every case passes at S=8192.
  - S=16384 was also tried for one layer of each kind. It passes, as evidence only. See
    [`../context_contract.json`](../context_contract.json).
- **Decode: N/A.** The model is a prefill-only classifier. It has no KV cache across requests and no
  decode step. Paged-decode, decode-trace and decode-perf items do not apply.
- **Batch: 1.** Batch > 1 with left padding is not implemented (see Limitations).

### Public signatures

```python
PplxDecoderLayer.from_state_dict(state_dict, *, args: PplxDeciderArgs, layer_idx: int, optimizations: Optimizations)
PplxDecoderLayer.forward(x: ttnn.Tensor, rotary: PplxRotary | None = None) -> ttnn.Tensor
    # x [1, S, 5120] BF16 TILE DRAM, a fresh request; rotary is required for full_attention layers
PplxDecoderLayer.mixer_prefill(x_normed, rotary=None) -> ttnn.Tensor      # mixer alone, for module tests
PplxGatedAttention.forward(x, *, start_pos: int, cos, sin) -> ttnn.Tensor  # one chunk
PplxGatedDeltaNet.forward(x, state: DeltaState | None = None) -> (ttnn.Tensor, DeltaState)  # one chunk
PplxMLP.forward(x); PplxRMSNorm.forward(x); PplxEmbedding.forward(ids [1,S] uint32 RM)
PplxReadout.forward(hidden [1,S,5120]) -> [1,1,255]   # last row; stage 6 passes the last real token
Optimizations.build(device, *, policy=PrecisionPolicy.bfp8_weights(), max_seq_len=8192, prefill_chunk=2048)
```

Precision policy for every number in this file is `act_bf16__w_bfp8_all__hifi2`:

- BF16 activations.
- BFP8 weights for every projection.
- HiFi2 matmuls with fp32 accumulation.
- HiFi4 norms.
- BF16 K/V cache.
- fp32 DeltaNet recurrent state and scan.
- Prefill chunk of 2048 tokens.

## Commands

Environment: `source python_env/bin/activate; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_HOME=/local/ttuser/gtobar/hf`.

| purpose | command |
|---|---|
| goldens (CPU, once) | `python -m models.demos.pplx_decider_v1_27b.reference.hf_reference --seq-len 8192 --layers 0 3 61 63 --through-final` |
| PCC suite | `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q` |
| PCC table | `python models/demos/pplx_decider_v1_27b/tests/pcc_report.py <pcc_results.jsonl>` |
| warmed perf | `pytest models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py -q -s -k "test_layer_and_module_perf or test_embedding_and_head_perf"`, then `python .../tests/perf/test_prefill_perf.py` |
| device profile | `python -m tracy -r -p -v --no-web-server -o <dir> -m pytest "models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py::test_profile_layer[L3_full-S2048-device_params0]"` |
| perf report | `tt-perf-report <ops_perf_results.csv> --start-signpost PREFILL_START --end-signpost PREFILL_END --no-advice [--csv out.csv]` |
| fallback audit | `pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_runtime_audit.py -q` (also part of the suite) |
| watcher | `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=<dir> pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_decoder_layer.py -q -k "S2048 and (L0 or L3)"` |
| context probe | `python -m models.demos.pplx_decider_v1_27b.reference.hf_reference --seq-len 16384 --layers 0 3`, then `pytest models/demos/pplx_decider_v1_27b/tests/probe/test_context_probe.py -q -s` |

## PCC summary (measured)

`pytest models/demos/pplx_decider_v1_27b/tests/pcc -q -p no:cacheprovider` on HEAD `24e2635212e` plus this
stage's test changes. Result: **124 passed in 669.53 s, exit 0**, from 123 PCC cases with 0 failures.

- **Weights and inputs.** Real weights; the input is the HF layer-streamed real-prompt prefix. Each
  module's own HF intermediate is fed in BF16.
- **Lowest PCC.** **0.999897**, for `gated_deltanet` L61 S8192. The threshold is 0.995.
- **Lowest PCC per decoder layer:**
  - `linear_attention`: 0.999985 (L61).
  - `full_attention`: 0.999969 (L63).

The full table is in [`work_log.md`](work_log.md).

## Warmed prefill performance (measured)

Method:

- `tests/perf/test_prefill_perf.py`: batch 1, eager (no trace), real-prompt input.
- For each case: 2 warm-up passes are excluded, then 7 timed passes; the table gives the median.
- Host sync: `ttnn.synchronize_device` before and after each pass. A sample is the
  `time.perf_counter` delta around one full pass, so it includes host dispatch.
- Log: `artifacts/pplx_decider/perf/prefill_perf.jsonl`.

Median ms per pass:

| target | kind | S=128 | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---|---:|---:|---:|---:|---:|
| decoder layer | linear_attention (L0) | 2.11 | 6.16 | 10.98 | 22.47 | 46.95 |
| decoder layer | full_attention (L3) | 1.96 | 5.47 | 10.07 | 22.70 | 58.84 |
| GDN mixer | linear_attention | 0.90 | 3.09 | 5.60 | 11.69 | 24.32 |
| attention mixer | full_attention | 0.76 | 2.36 | 4.82 | 12.05 | 36.37 |
| MLP (per 2048 chunk, as the decoder runs it) | linear_attention | 1.08 | 2.78 | 4.79 | 9.83 | 21.83 |
| MLP | full_attention | 1.11 | 2.78 | 4.85 | 9.91 | 22.93 |
| embedding | - | 0.07 | 0.12 | 0.17 | 0.27 | 0.48 |
| final norm (last token) + readout | - | 0.26 | 0.28 | 0.33 | 0.43 | 0.63 |

Run-to-run spread is a few percent. An earlier run of the same harness gave a full_attention layer
time of 54.66 ms at S=8192, against 58.84 ms here.

### Device profile (measured, Tracy + tt-perf-report)

One warmed pass between `PREFILL_START` and `PREFILL_END`, after 2 warm-up passes. Reports are in
[`perf/`](perf/): a `*_perf_report.txt` table and a `*_perf_report.csv` / `*_stacked.csv` for each
capture. The raw `ops_perf_results` CSVs and traces are in `artifacts/pplx_decider/tracy/`. Device
time is the `Device Time` column, in µs, summed over the window.

| capture | device ops | host ops | device time | op-to-op gap | wall median (table above) | top ops |
|---|---:|---:|---:|---:|---:|---|
| linear_attention S2048 | 30 | 0 | 10.95 ms | 15 µs | 10.98 ms | minimal_matmul 56.0 %, GDN prep 10.7 %, eltwise 8.6 %, GDN scan 6.7 % |
| linear_attention S8192 | 125 | 0 | 44.51 ms | 66 µs | 46.95 ms | minimal_matmul 55.1 %, GDN prep 10.3 %, eltwise 8.4 %, GDN scan 6.5 % |
| full_attention S2048 | 33 | 0 | 9.95 ms | 24 µs | 10.07 ms | minimal_matmul 61.2 %, SDPA 12.9 %, eltwise 9.2 %, norms 5.5 % |
| full_attention S8192 | 137 | 0 | 53.50 ms | 99 µs | 58.84 ms | minimal_matmul 45.5 %, chunked SDPA 33.7 %, eltwise 6.9 %, slice 4.5 % |

tt-perf-report rates the big matmuls at 60-77 % in its `FLOPs %` column (HiFi2, 130 cores). The
pass is device-bound: device time is 91-99 % of wall time.

### Projected 64-layer prefill (inferred)

The projection is 48 x linear + 16 x full + embedding + final norm/readout, from the measured
per-layer medians:

- **S=2048:** 48 x 10.98 + 16 x 10.07 + 0.17 + 0.33 = **689 ms**. From profiler device time alone:
  685 ms.
- **S=8192:** 48 x 46.95 + 16 x 58.84 + 0.48 + 0.63 = **3196 ms**. From profiler device time alone:
  2993 ms.

The projection leaves out per-layer weight residency: the full model must hold all 64 layers in DRAM
(about 27.2 GiB, inferred, see context_contract.json). It also leaves out host time between layers.

### Candidates for stage 2/3 (not implemented)

- Full-attention S=8192 spends 18.0 ms in 4 chunked-SDPA calls.
  - Each chunk re-reads the paged prefix.
  - A request-local, non-paged causal SDPA over the whole bucket, or larger q/k chunks, should cut
    this.
- The K/V cache is a paged structure that a prefill-only classifier does not need. Writing K/V
  straight to SDPA would remove `paged_fill_cache`, the typecasts and the page-table slices.
- The BF16 slices, concats and tilize around RoPE (partial 64/256 dims) and the q/k head split cost
  about 5 % of a full layer. Candidates: a fused partial-rotary op, or a pre-permuted q/k layout.
- Several `BinaryNg` ops could be fused into the producing matmul or norm, at about 8-9 % of either
  layer:
  - residual adds;
  - SiLU*up;
  - sigmoid gate;
  - softplus/decay.
- Delta in-projection slices and conv row-major conversion.
- MLP gate/up: try a packed gate+up matmul (one weight read) or the `minimal_mlp` SwiGLU layout from
  Qwen3.8.
- Raise the prefill chunk (for example 4096 or one chunk per bucket). This cuts per-chunk overheads
  and SDPA re-reads, with an L1/DRAM headroom check.
- Trace capture for each bucket (5 shapes) to remove dispatch at small S.

## Runtime fallback audit (measured)

Test: `tests/pcc/test_runtime_audit.py`, with the counter in `tests/runtime_audit.py`.

1. Build one layer of each kind and upload the input.
2. Run one warm-up pass.
3. Run one full prefill pass, then `synchronize_device`, inside `count_host_calls()`.

`count_host_calls()` counts:

- these `ttnn` functions, patched on `ttnn` and on `ttnn.operations.core`: `from_torch`, `to_torch`,
  `from_device`, `to_device`, `copy_host_to_device_tensor(_partial)`, `copy_device_to_host_tensor`
  and `allocate_tensor_on_host`;
- the methods `ttnn.Tensor.cpu` and `ttnn.Tensor.to_torch`;
- every torch function call (`TorchFunctionMode`);
- every aten op (`TorchDispatchMode`).

Cases: L0 (linear) and L3 (full) at S=128 (one chunk) and S=4096 (two chunks with state carry).
**The count is 0 in all 4 cases** (4 passed, part of the 124). A positive control in the same test
sees `ttnn.to_torch` and a torch op, so the counter works. The Tracy windows above independently
show **0 host ops**.

## Determinism and watcher (measured)

- **Determinism.** `tests/pcc/test_prefill_contract.py` runs 8192 -> 128 -> 2048 -> 4096 on one
  loaded instance per kind. It then repeats the 4096 request, and `torch.equal` holds: the output is
  bit-identical. The test passes in the suite. All contract PCCs are >= 0.99999, so no state leaks
  between requests.
- **Watcher at a 10 s interval.**
  - Command: `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=artifacts/pplx_decider/watcher_1b pytest .../test_decoder_layer.py -k "S2048 and (L0 or L3)"`.
  - Result: 2 passed, exit 0. Log: `artifacts/pplx_decider/watcher_1b/generated/watcher/watcher.log`.
  - The log has 2 dumps, attach and detach, and a stack-usage summary. It has no assert, NOC,
    sanitize, L1 or stack-overflow error.
- **Watcher at a 1 s interval.**
  - Command: `TT_METAL_WATCHER=1`, `-k "(S2048 or S8192) and (L0 or L3)"`.
  - Result: 4 passed, exit 0, 14 dumps, no errors. Log:
    `artifacts/pplx_decider/watcher_1b_1s/generated/watcher/watcher.log`.

## Limitations

- **Batch > 1 with left padding is not implemented.** HF zeroes left-pad rows at the Gated DeltaNet
  input when batch > 1. The modules here take one request per call (they raise on `b != 1`). The
  app's `inference.py` runs batch 1. A future batched path would also need right padding to the
  bucket plus per-row last-real-token indices.
- **Decode is N/A** (prefill-only classifier). There is no decode trace, paged decode or decode perf.
- **Supported context is 8192** (app contract), not the HF-advertised 262144.
  - `scripts/check_context_contract.py` from the plugin exits 2 for this contract, because it only
    accepts DRAM-limited reductions. This one is an app-contract reduction approved by the person.
  - S=16384 passes for one layer of each kind (PCC 0.999993).
- **Eager execution.** Prefill is not traced in this stage.
- **Bucket-only testing.** Non-bucket lengths are no longer tested (person decision); the code
  path for them exists.
- **Text only.** mRoPE reduces to 1D RoPE for text; vision is stage 12.
- **Golden scope.** The HF goldens are fp32 layer-by-layer runs of one real prompt. Layers 0, 3, 61
  and 63 are checked; all 64 layers are checked in stage 6.

## Self-check against the functional-decoder SKILL.md objectives

| objective (SKILL.md "Evidence To Leave" and goal) | status / evidence |
|---|---|
| `FunctionalDecoder` (`LightweightModule`) with a `from_state_dict` boundary | Met in adapted form: `tt/decoder.py` `PplxDecoderLayer.from_state_dict(state_dict, *, args, layer_idx, optimizations)`, under `models/demos/pplx_decider_v1_27b/` (path set by the person, see notes.md) |
| `prefill_forward` / `decode_forward` | `forward` = prefill. `decode_forward`: N/A (prefill-only, adapted contract in notes.md) |
| HF-vs-TTNN prefill PCC for each layer kind | Met: 0.999969-0.999994 per layer at all 5 buckets, real weights (PCC section) |
| HF-vs-TTNN decode PCC | N/A (prefill-only, adapted contract in notes.md) |
| Warmed prefill perf with perf-report outputs | Met: perf table above plus `perf/*_perf_report.{txt,csv}` |
| Traced warmed decode perf | N/A (prefill-only, adapted contract in notes.md) |
| Paged KV-cache behavior, page table and current position | Prefill: a request-local paged K/V cache with an identity page table, chunk-local pages and absolute `start_pos`, exercised by multi-chunk cases and the contract test. Decode: N/A |
| Capability-contract evidence table | `context_contract.json` plus the table below |
| Full advertised prefill length or a hard-limit reduction | Adapted: app contract 8192 is met; 16384 probe passes; 262144 not attempted (app reduction approved by the person, not DRAM; the plugin checker flags this) |
| Non-aligned lengths: smoke, boundary, across boundary, long non-divisible | Replaced by person decision: bucket lengths 128 / 1024 / 2048 / 4096 / 8192 with right padding. 2048 is exactly one chunk; 4096 and 8192 are multi-chunk |
| `context_contract.json` | Met: `doc/context_contract.json` |
| Determinism | Met: bit-identical repeat in `test_prefill_contract.py` |
| No torch or host calls in a pass | Met: counter = 0 (`test_runtime_audit.py`); 0 host ops in Tracy |
| Watcher-clean run with `TT_METAL_WATCHER=10` | Met: 10 s and 1 s runs, no errors |
| Real-weight test passing per layer kind | Met: every test uses real weights |
| Batch > 1 correctness | Not done: batch 1 only (Limitations) |
| MoE | N/A (dense SwiGLU MLP) |
| Sliding window | N/A (no sliding-window layers) |
| README and work_log with commands, PCC, perf, limitations | Met: this file and `work_log.md` |

### Capability-contract evidence

| claim | evidence | remaining risk |
|---|---|---|
| Both layer kinds match HF at real shapes | 123 PCC cases >= 0.999897, layers 0, 3, 61 and 63 | The other 60 layers are only checked in stage 6 |
| 8192-token prefill (app max) works | All modules and layers at S8192; probe at 8192 | Full-model DRAM headroom about 4.6 GiB (inferred); transient peak not measured |
| > 8192 is possible per layer | S16384 PCC 0.999993 for both kinds | Not supported by contract; 262144 not tried |
| No state leaks between requests | Contract test 8192 -> 128 -> 2048 -> 4096, bit-identical repeat | No concurrent requests (batch 1) |
| Right padding to a bucket is exact for real tokens | Causal mixers (inferred) | Validated end to end in stage 6 |
| Runs fully on device | Audit counter 0; Tracy 0 host ops | none known |

# Qwen3.8-27B functional decoder

Status: **complete; independent stage review clean-pass**.
Local checkpoint provenance is recorded in [work_log.md](work_log.md).

Target: `Qwen/Qwen3.8-27B`, checkpoint revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, Transformers 5.12.1.
The text decoder is `Qwen3_5DecoderLayer`: 48 gated delta-rule linear-attention
layers and 16 gated full-attention layers. Representative real layers are 0 and
3. Hidden size 5120, MLP size 17408, full attention 24 query/4 KV heads of
width 256, linear attention 16 key/48 value heads of width 128, convolution
width 4. The advertised context is 262144. Vision/MTP are not decoder kinds.
Gates match installed Transformers 5.12.1: `Qwen3_5Attention.forward` uses
`sigmoid(gate)`, and linear attention uses SiLU. The saved config retains
`output_gate_type: "swish"`; the installed Qwen3_5 code does not read that
field. The HF package version and real checkpoint revision are pinned in the
evidence.

## API

`FunctionalDecoder` subclasses `LightweightModule`. Setup:

```python
decoder = FunctionalDecoder.from_state_dict(
    layer_state_dict, hf_config=text_config, layer_idx=layer_idx,
    mesh_device=mesh_device,
)
state = decoder.allocate_state(batch_size=batch_size, num_pages=num_pages)
```

`layer_state_dict` has the exact layer-local HF keys. Inputs and outputs have
logical shape `[B,S,5120]`, BF16 TILE layout on a single 1x1 mesh.

```python
decoder.prefill_forward(
    x, state=state, start_pos=0, page_table=page_table,
    cos=cos, sin=sin, positions=positions,
)
decoder.decode_forward(
    x, state=state, current_pos=current_pos, page_table=page_table,
    cos=cos, sin=sin,
)
```

Full attention owns BF16 K/V buffers `[num_pages,4,32,256]`; the caller supplies
the INT32 row-major page table `[B,pages_per_request]` with disjoint physical
pages. `current_pos` is zero-based INT32 row-major `[B]`, in `[0,262143]`.
RoPE cos/sin are device BF16
`[B,S,64]` from the target HF partial-RoPE contract. Prefix continuation uses
absolute `start_pos`; unaligned prefix continuation also supplies a device
INT32 `[S,B]` positions tensor. Prefill uses 128-token physical chunks and
retains every logical output. Fresh requests use zeroed/new state; continuation
keeps the preceding state. Linear attention uses FP32 recurrent state and BF16
convolution history and does not consume KV page tables or RoPE.

Decode uses stable caller-owned tensors. The harness warms, snapshots/restores
state, captures `decode_forward`, restores state, and compares trace replay
output to HF. Conversion and result comparison occur outside forward passes.

## Correctness evidence

| Kind | Largest tested prefill | Prefill PCC | Traced decode PCC at context 262144 | Repeated replay |
|---|---:|---:|---:|---|
| Full attention | 262144 | 0.99680092 | 0.99774247 | Bitwise equal |
| Linear attention | 262144 | 0.99905564 | 0.99970686 | Bitwise equal |

Raw context results: `full_context.json/log`, `linear_context.json/log`.
Both kinds pass the full advertised context with no reduction. Initial
32-token real-weight smokes are also retained in `full_smoke.log/json` and
`linear_smoke.log/json`. Acceptance is PCC >= 0.995. `weight_stats.json`
records every used tensor's name, shape, dtype,
mean and std, with strict HF state loading for both kinds. `hf_config.json`
preserves the pinned config. The harness reads only layer tensors from the
existing checkpoint shards.

Both kinds also pass batch-2 boundary/reuse sweeps and batch-32 unaligned
continuation, traced decode, and changed-input tests. Full attention additionally
passes refreshed per-user positions, page-table routing, and untouched unowned
pages. Runtime Torch/TTNN-host-conversion guards pass. Clean watcher runs are
`linear_b32_watcher_retry.log/json` and `full_b32_watcher.log/json`, with the
watcher logs under their respective `watcher_*/generated/watcher/` directories.
The native delta scan's per-core head limit is handled by splitting its
independent batch axis; see the work log for the failed probe and verified fix.

Warmed performance at batch 1, prefill 128 / decode context 129 is recorded in
`performance.json`. Human-readable tables and filtered CSVs are under
`tracy/{linear_attention,full_attention}/{prefill,decode}_perf_report.*`.
`Device Time` is microseconds; sums are kernel time, with op-to-op gaps recorded
separately. The decode window executes a warmed trace. Original Tracy ops CSVs
and collection logs are preserved with those reports.

Both context runners use lengths `4097,262143,262144,31` on one loaded
instance. All logical outputs are compared with HF. The HF oracle bounds
only its temporary work to 1024 tokens, retaining the complete cache and
all outputs. The 262143-token case tests decode at context 262144; the
262144-token case checks prefill without an out-of-contract extra decode.

The synthetic pytest suite passes both layer kinds, each at batch 2 with
`1,31,32,33,127,128,129,257,31`, continuation and refreshed trace inputs:
`synthetic_pytest.log/xml`, `synthetic_linear.json`, `synthetic_full.json`.
Minimum prefill/decode PCC is 0.99722046/0.99650061 for linear attention and
0.99837809/0.99779248 for full attention.

`linear_unchunked_control.log/json` and `full_unchunked_control.log/json`
compare 128-token outer chunks with one 257-token outer chunk, real weights,
batch 2, unchanged HF, and follow-on traced decode from each filled state.
Both pass. Between-path prefill/decode PCC is 0.99999989/0.99999672 for linear
attention and 1.0/1.0 for full attention. `linear_b3.log/json` additionally
passes an uneven scan batch split (2+1), continuation, and traced decode.

Warmed kernel sums at batch 1, prefill 128 / decode context 129:

| Kind | Prefill | Traced decode |
|---|---:|---:|
| Linear attention | 4.623045 ms | 3.009414 ms |
| Full attention | 3.407822 ms | 2.378159 ms |

Op-to-op gaps are reported separately in `performance.json` and
`../functional_decoder.md`; these are kernel sums, not end-to-end latency.

## Reproduce the synthetic suite

From the repository root, with the tested software in `environment.json`:

```bash
EVIDENCE=models/autoports/qwen_qwen3_8_27b/doc/functional_decoder
PYTHONPATH=. timeout -k 10 900 python_env/bin/python -m pytest \
  --confcutdir=models/autoports/qwen_qwen3_8_27b/tests \
  models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py \
  -o addopts='' -v -s --durations=2 \
  --basetemp="$EVIDENCE/synthetic_pytest_tmp" \
  --junitxml="$EVIDENCE/synthetic_pytest.xml"
```

Real-weight, context, control, watcher and profiler commands are recorded in
`work_log.md` and `../functional_decoder.md`. Run device commands serially.

## Capability contract

| Claim | Evidence | Remaining risk or caller responsibility |
|---|---|---|
| Both target decoder kinds at real HF shapes | Real layers 0 and 3; `weight_stats.json`, strict HF loads, boundary and watcher results | Later layer weights and accumulated full-model accuracy are outside this stage |
| Advertised context 262144 | `hf_config.json`; `linear_context.json/log` and `full_context.json/log` pass exact prefill and final-context traced decode | `../context_contract.json` records supported context 262144; this is decoder-layer capacity |
| Arbitrary logical lengths through internal 128-token chunks and 32-token pages/tiles | Both kinds at 1,31,32,33,127,128,129,257,4097; both kinds at 262143 and 262144; short reuse | Equal logical lengths and prefix positions within each prefill batch; caller owns request scheduling |
| Paged KV ownership and absolute positions | `full_audited_b2`, `full_continuation`, `full_b32_watcher` JSON/logs: permuted pages, unowned-page preservation, changed positions and page-table rows during replay | Caller provides enough disjoint pages, valid INT32 positions, and matching partial-RoPE tensors |
| Explicit continuation preserves state | Both batch-32 watcher runs split at 33 and compare all outputs plus follow-on decode to HF | Fresh requests require zero/new state; linear recurrence cannot rewind by changing a position tensor |
| Batch sizes 1, 2 and 32 for both kinds; also linear batch 3 | Batch-1 smokes/profiles/context; batch-2 sweeps; batch-32 watcher; `linear_b3.json` | Maximum context was tested at batch 1; batch 32 at prefill 257 / decode context 258, not the Cartesian product |
| Fully traced decode with refreshed inputs | Guarded capture/replay, changed tokens/positions, bitwise repeated restored-state replay | Caller refreshes stable captured tensors and restores prefix state after warmup/capture |
| Runtime stays on device | Torch dispatch and TTNN conversion guards in `tests/run_decoder.py`, source audit in `stage_review_2.md`, zero host ops in measured perf tables | Setup, trace orchestration, input refresh and result inspection are explicit host boundaries |
| No sliding-window/MoE mode omitted | Target HF config/source: gated full attention and gated delta-rule attention; dense SwiGLU | Vision and MTP are outside the requested text decoder layer contract |

Acceptance remains PCC >=0.995 for every prefill/decode comparison. No
model-specific relaxation is used. Hardware timings above cover one warmed
short-shape sample per mode; they are neither latency distributions nor
full-context performance claims. The detected firmware is 19.8.0, versus
19.8.1 documented by this checkout; reset recovery and successful watcher runs
are recorded, without claiming the original heartbeat fault cannot recur.

Independent [final stage review](stage_review_2.md) returned **clean-pass**,
with no required work remaining. The earlier review findings are closed.

## Hardware recovery

Initial open failed on a static ERISC heartbeat before model code. One reset
restored operation with the same firmware; see `AUTOTRIAGE.md`, `reset_1.log`,
`device_list_after_reset_1.log`, and `mesh_smoke_after_reset_1.log`.
Four Blackhole p300c chips are visible; the decoder uses one chip, grid 11x10.
Detected firmware is 19.8.0; no firmware modification was performed.

Exact commands and evolving gates are in `work_log.md`. `environment.json`
records software/hardware provenance. `raw_capture_manifest.json` records
paths and hashes for bulky Tracy captures retained locally. Compact ops CSVs
and reports are retained in the stage checkpoint. First-use long full-attention
prefill has substantial per-offset SDPA compilation cost; warmed short-shape
timings do not include it. Long-context full-attention PCC is lower than the
short-case PCC, remains above 0.995, and has no threshold relaxation.

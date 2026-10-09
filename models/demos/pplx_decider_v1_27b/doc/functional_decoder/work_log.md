# Functional decoder work log (stages 1A and 1B)

Model: `perplexity-ai/pplx-decider-v1-27b`, snapshot `b01a5cbaca5391f73bd55103d4f27e8982cd5e60`
(Qwen3.5 hybrid text backbone: 64 layers, 48 Gated DeltaNet + 16 gated full attention, prefill only).
Hardware: one Blackhole p150a (device 0, 13x10 compute grid, 8 DRAM banks).
Software: tt-metal `2c1e1ebdd63` (branch `gtobarTT/pplx-decider-bringup`), torch 2.11.0+cpu,
transformers 5.12.1 (`fla`/`causal_conv1d` absent, so HF runs its torch fallbacks).

Labels: **measured** = command output in this log; **inferred** = follows from code, not run.

## What exists

| path | content |
|---|---|
| `reference/hf_reference.py` | Layer-streamed HF golden generator. `SnapshotReader` reads single tensors from the sharded safetensors; `build_decoder_layer` / `build_embedding` / `build_final_norm` / `build_readout` load one module with `strict=True`. The CLI streams the 64 layers one at a time and saves the input of chosen layers. The full model is never built. |
| `tt/model_config.py` | `PplxDeciderArgs` from the snapshot `text_config`; fails on any shape the kernels do not support. |
| `tt/optimizations.py` | The only place for dtypes, fidelities, matmul / SDPA / scan configs (`PrecisionPolicy`, `Optimizations.build`). |
| `tt/weight_adapter.py` | HF -> TT transforms (transpose, `1 + w` folding, q/gate de-interleave + QKVG pack, GDN in-proj pack, conv taps, `-exp(A_log)`), output `LazyWeight` bundles. |
| `tt/norm.py`, `tt/mlp.py`, `tt/attention.py`, `tt/gated_deltanet.py`, `tt/rope.py`, `tt/embedding.py`, `tt/readout.py`, `tt/decoder.py` | TTv2-style modules: `<Name>Config` dataclass, `__init__` + `from_config`, lazy device weights, straight-line `forward`. |
| `tests/test_utils.py`, `tests/pcc/*.py`, `tests/pcc_report.py` | Real-weight PCC tests and the table renderer. |
| `tests/runtime_audit.py`, `tests/pcc/test_runtime_audit.py` (1B) | Counts host conversions, round trips and torch ops in a measured pass; must be 0. |
| `tests/perf/test_prefill_perf.py` (1B) | Warmed prefill latency per layer kind and module; Tracy capture target with `PREFILL_START`/`PREFILL_END` signposts. |
| `tests/probe/test_context_probe.py` (1B) | S=8192/16384 probe per layer kind with DRAM numbers. |

Key differences from `models/demos/qwen38_27b_qb2/tt/decoder.py` (the logic source):
- Snapshot key prefix is `language_model.layers.N.` (no `model.`); readout is `readout.safetensors:weight [255,5120]`.
- No decode path, no DRAM-sharded decode weight copies (`dram: True` in Qwen3.8 doubled the weight footprint).
- DeltaNet and attention state are carried functionally between prefill chunks (zero state tensors are never written), not by in-place `ttnn.copy` into caller state.
- The attention module owns a request-local paged K/V cache sized for `max_seq_len` (8192 tokens -> 256 pages of 32) with an identity page table, allocated at setup.

## Golden recipe

Command (**measured**: exit 0, ~10 min wall, about 7-15 s per layer at S=8192 fp32 on 24 CPU threads):

```bash
python -m models.demos.pplx_decider_v1_27b.reference.hf_reference \
  --seq-len 8192 --layers 0 3 61 63 --through-final \
  --out /local/ttuser/gtobar/artifacts/pplx_decider/goldens
# log: /local/ttuser/gtobar/artifacts/pplx_decider/logs/golden_stream_S8192.log
```

- Prompt: the chat template (`enable_thinking=False`, generation prompt) around the snapshot's README and
  `source/src/autojev/*.py` text, first 8192 token ids. Real text, deterministic.
- The stack runs in fp32 (BF16 weights upcast), batch 1, no padding, `position_ids = arange(S)`,
  sdpa causal attention, `use_cache=False` semantics. Saved: `ids`, `L{0,3,61,63}_input`, `final_input`
  (output of layer 63), 801 MB total.
- Each test takes the prefix `[:, :S]` of a saved layer input (the model is causal, so this is the HF
  hidden state of an S-token prompt), rounds it to BF16 (the exact device input), and runs the
  strict-loaded HF layer / submodule on it in fp32 for the golden output.
- Module tests feed each TT module the BF16 value of the matching HF intermediate (for example the MLP
  gets HF `post_attention_layernorm(h)`), so each PCC isolates one module.

## Precision policy used for every number below

`act_bf16__w_bfp8_all__hifi2` (`PrecisionPolicy.bfp8_weights()`): BF16 activations; BFP8 weights for every
projection (QKVG, o_proj, GDN in/out proj, MLP gate/up/down, readout); HiFi2 matmuls with fp32 accumulation;
BF16 embedding table and norm weights (`1 + w` folded in fp32, then cast); HiFi4 norms; BF16 K/V cache;
fp32 DeltaNet recurrent state and scan; BF16 conv state. Prefill chunk 2048 tokens.

## PCC table (stage 1B, bucket lengths)

Source: `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q -p no:cacheprovider` with
`PPLX_DECIDER_PCC_LOG=.../logs/pcc_results_1b.jsonl`. It ran on HEAD `24e2635212e` plus the 1B test
changes.

- **Result (measured 2026-10-09):** 124 passed in 669.53 s, exit 0.
- **Log:** `/local/ttuser/gtobar/artifacts/pplx_decider/logs/pytest_pcc_1b.log`.
- **Re-validation:** this run also re-validates 1A after the isort-only change.
- **Rendering:** the table is rendered by `tests/pcc_report.py`.
- **Row notes:**
  - The `contract_*` rows come from `test_prefill_contract.py` (8192 -> 128 -> 2048 -> 4096 on one
    instance).
  - `embedding` is also checked bit-exact with `torch.equal`.
  - Readout PCC is over the 255 logits of the last token.

Policy: act_bf16__w_bfp8_all__hifi2. Threshold 0.995. Cells: PCC (FAIL marks < threshold).

| module | layer | kind | S=128 | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---|---|---:|---:|---:|---:|---:|
| contract_full_attention | 3 | full_attention | 0.999992 |  | 0.999994 | 0.999993 | 0.999993 |
| contract_linear_attention | 0 | linear_attention | 0.999990 |  | 0.999992 | 0.999993 | 0.999993 |
| decoder_layer_full_attention | 3 | full_attention | 0.999992 | 0.999993 | 0.999994 | 0.999993 | 0.999993 |
| decoder_layer_full_attention | 63 | full_attention | 0.999969 | 0.999970 | 0.999969 | 0.999969 | 0.999970 |
| decoder_layer_linear_attention | 0 | linear_attention | 0.999990 | 0.999992 | 0.999992 | 0.999993 | 0.999993 |
| decoder_layer_linear_attention | 61 | linear_attention | 0.999988 | 0.999989 | 0.999990 | 0.999989 | 0.999988 |
| embedding | - | - | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| final_norm | - | - | 0.999996 | 0.999996 | 0.999996 | 0.999996 | 0.999996 |
| gated_attention | 3 | full_attention | 0.999964 | 0.999962 | 0.999960 | 0.999958 | 0.999955 |
| gated_attention | 63 | full_attention | 0.999925 | 0.999935 | 0.999936 | 0.999935 | 0.999934 |
| gated_deltanet | 0 | linear_attention | 0.999993 | 0.999994 | 0.999994 | 0.999994 | 0.999994 |
| gated_deltanet | 61 | linear_attention | 0.999916 | 0.999926 | 0.999927 | 0.999922 | 0.999897 |
| mlp | 0 | linear_attention | 0.999987 | 0.999986 | 0.999986 | 0.999986 | 0.999986 |
| mlp | 3 | full_attention | 0.999961 | 0.999956 | 0.999957 | 0.999959 | 0.999958 |
| mlp | 61 | linear_attention | 0.999931 | 0.999935 | 0.999937 | 0.999937 | 0.999937 |
| mlp | 63 | full_attention | 0.999969 | 0.999967 | 0.999965 | 0.999963 | 0.999963 |
| readout | - | - | 0.999969 | 0.999980 | 0.999975 | 0.999967 | 0.999961 |
| rmsnorm_input_norm | 0 | linear_attention | 0.999997 | 0.999997 | 0.999997 | 0.999997 | 0.999997 |
| rmsnorm_input_norm | 3 | full_attention | 0.999998 | 0.999999 | 0.999999 | 0.999999 | 0.999999 |
| rmsnorm_input_norm | 61 | linear_attention | 0.999996 | 0.999996 | 0.999996 | 0.999996 | 0.999996 |
| rmsnorm_input_norm | 63 | full_attention | 0.999997 | 0.999997 | 0.999997 | 0.999997 | 0.999997 |
| rmsnorm_post_norm | 0 | linear_attention | 0.999995 | 0.999995 | 0.999995 | 0.999995 | 0.999994 |
| rmsnorm_post_norm | 3 | full_attention | 0.999995 | 0.999995 | 0.999995 | 0.999995 | 0.999995 |
| rmsnorm_post_norm | 61 | linear_attention | 0.999996 | 0.999996 | 0.999996 | 0.999996 | 0.999996 |
| rmsnorm_post_norm | 63 | full_attention | 0.999995 | 0.999996 | 0.999996 | 0.999996 | 0.999996 |

Cases: 123; failures: 0; min PCC 0.999897 (gated_deltanet L61 S8192).
Diagnostic (not gated): min PCC excluding token 0 = 0.999897.

The 1A table (lengths 1, 31, 129, 1000, 2048, 8192; 143 passed) is superseded. An intermediate 1B run
used 11 lengths: 1, 31, 32, 33, 128, 129, 1000, 1024, 2047, 2048 and 8192. It also passed (262 passed
in 1060.96 s, exit 0, min PCC 0.999897; `logs/pytest_pcc_1b_superseded_11len.log`). The person then
switched the contract to buckets.

## Coverage and contract notes

- **Lengths (1B).** By person decision on 2026-10-09, every module and both layer kinds use the
  prefill buckets 128, 1024, 2048, 4096 and 8192.
  - Layers: 0 and 61 (linear_attention); 3 and 63 (full_attention).
  - The app pads on the right to the next bucket and reads the last real token.
  - S=4096 is the prefix of the 8192-token golden (the model is causal), so no new goldens were
    needed.
  - S=8192 is 4 physical chunks of 2048.
- **Request isolation.** `test_prefill_contract.py` runs 8192 -> 128 -> 2048 -> 4096 back to back on
  ONE loaded layer instance per kind. Stale K/V pages or DeltaNet state would lower PCC.
- **Determinism.** The contract test repeats the last request, and the output must be bit-identical
  (**measured**: passes).
- **Fallback audit, strict form (1B, measured).** `test_runtime_audit.py` counts host conversions, host
  round trips and torch ops during one measured pass (see README). The count is 0 for L0 and L3 at
  S128 and S4096.
- **Token-0 diagnostic.** PCC is also logged without token 0 (diagnostic, not gated).

## Stage 1B evidence log (2026-10-09)

| item | command | result |
|---|---|---|
| PCC suite (buckets) | `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q -p no:cacheprovider` | 124 passed, exit 0, min PCC 0.999897; `logs/pytest_pcc_1b.log` |
| runtime audit | part of the suite; first standalone run `logs/runtime_audit.log` (S33/S2049, before buckets) | host-call count 0 in every case |
| warmed perf | `pytest .../tests/perf/test_prefill_perf.py -q -s -k "test_layer_and_module_perf or test_embedding_and_head_perf"` | 3 passed, exit 0; `perf/prefill_perf.jsonl`, `logs/perf_1b.log` |
| Tracy | `python -m tracy -r -p -v --no-web-server -o <dir> -m pytest ".../test_prefill_perf.py::test_profile_layer[<L0_linear,L3_full>-S<2048,8192>-device_params0]"` | 4 captures, exit 0; `artifacts/pplx_decider/tracy/`; reports in `doc/functional_decoder/perf/` |
| tt-perf-report | `tt-perf-report <ops.csv> --start-signpost PREFILL_START --end-signpost PREFILL_END --no-advice --no-color [--csv ...]` | exit 0 for all 8 invocations; `Device Time` column in µs |
| watcher 10 s | `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=artifacts/pplx_decider/watcher_1b pytest .../test_decoder_layer.py -k "S2048 and (L0 or L3)"` | 2 passed, exit 0, no watcher errors |
| watcher 1 s | `TT_METAL_WATCHER=1 TT_METAL_LOGS_PATH=artifacts/pplx_decider/watcher_1b_1s ... -k "(S2048 or S8192) and (L0 or L3)"` | 4 passed, exit 0, 14 dumps, no errors |
| 16k golden | `python -m models.demos.pplx_decider_v1_27b.reference.hf_reference --seq-len 16384 --layers 0 3 --threads 12` | exit 0, about 20 s per layer; `logs/golden_stream_S16384.log` |
| context probe | `pytest .../tests/probe/test_context_probe.py -q -s` | 4 passed, exit 0; PCC 0.999993 at 8192 and 16384 for L0 and L3; `logs/context_probe.jsonl` |
| context checker | `python <plugin>/scripts/check_context_contract.py --model-dir models/demos/pplx_decider_v1_27b --hf-model perplexity-ai/pplx-decider-v1-27b --require-contract` | exit 2: "supports context 8192, below HF-advertised 262144, without device-DRAM capacity evidence" (expected; app-contract reduction approved by the person) |

Perf, profile, projection and stage-2/3 candidates: see `README.md`.

## Not in this stage

These items are left for later:

- **Batch > 1 with left padding.** DeltaNet would need the HF padding-mask zeroing. The app's
  `inference.py` runs batch 1.
- **Full-stack PCC and decision agreement:** stage 6.
- **Dtype sweep:** stage 8.
- **Tracing and fusion:** stage 2/3.

## Commands

```bash
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD HF_HOME=/local/ttuser/gtobar/hf
pytest models/demos/pplx_decider_v1_27b/tests/pcc -q          # every PCC case, appends to the PCC log
python models/demos/pplx_decider_v1_27b/tests/pcc_report.py    # renders the table above
pytest models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py -q -s -k "test_layer_and_module_perf or test_embedding_and_head_perf"
python models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py   # perf table
```

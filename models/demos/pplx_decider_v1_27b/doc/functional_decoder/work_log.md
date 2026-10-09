# Functional decoder work log (stage 1A)

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

## PCC table

Source: `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q -p no:cacheprovider` (**measured** 2026-10-09:
143 passed in 668 s, exit 0; log `/local/ttuser/gtobar/artifacts/pplx_decider/logs/pytest_pcc_final.log`,
records `.../logs/pcc_results.jsonl`), rendered by `tests/pcc_report.py`. An earlier full run (141 cases,
before `test_prefill_contract.py` existed) gave the same values to six decimals. The `contract_*` rows come
from `test_prefill_contract.py`; S=2049 and S=4129 appear only there. `embedding` is also checked
bit-exact with `torch.equal`. Readout PCC is over the 255 logits of the last token.

Policy: act_bf16__w_bfp8_all__hifi2. Threshold 0.995.

| module | layer | kind | S=1 | S=31 | S=129 | S=1000 | S=2048 | S=2049 | S=4129 | S=8192 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| contract_full_attention | 3 | full_attention |  |  | 0.999992 |  |  | 0.999994 | 0.999993 | 0.999993 |
| contract_linear_attention | 0 | linear_attention |  |  | 0.999990 |  |  | 0.999992 | 0.999993 | 0.999993 |
| decoder_layer_full_attention | 3 | full_attention | 0.999999 | 0.999991 | 0.999992 | 0.999993 | 0.999994 |  |  | 0.999993 |
| decoder_layer_full_attention | 63 | full_attention | 0.999989 | 0.999970 | 0.999969 | 0.999970 | 0.999969 |  |  | 0.999970 |
| decoder_layer_linear_attention | 0 | linear_attention | 0.999996 | 0.999987 | 0.999990 | 0.999992 | 0.999992 |  |  | 0.999993 |
| decoder_layer_linear_attention | 61 | linear_attention | 0.999988 | 0.999985 | 0.999988 | 0.999989 | 0.999990 |  |  | 0.999988 |
| embedding | - | - | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |  |  | 1.000000 |
| final_norm | - | - | 0.999997 | 0.999996 | 0.999996 | 0.999996 | 0.999996 |  |  | 0.999996 |
| gated_attention | 3 | full_attention | 0.999971 | 0.999957 | 0.999964 | 0.999962 | 0.999960 |  |  | 0.999955 |
| gated_attention | 63 | full_attention | 0.999951 | 0.999929 | 0.999926 | 0.999935 | 0.999936 |  |  | 0.999934 |
| gated_deltanet | 0 | linear_attention | 0.999996 | 0.999991 | 0.999993 | 0.999994 | 0.999994 |  |  | 0.999994 |
| gated_deltanet | 61 | linear_attention | 0.999919 | 0.999914 | 0.999917 | 0.999926 | 0.999927 |  |  | 0.999897 |
| mlp | 0 | linear_attention | 0.999992 | 0.999988 | 0.999987 | 0.999986 | 0.999986 |  |  | 0.999986 |
| mlp | 3 | full_attention | 0.999966 | 0.999961 | 0.999961 | 0.999956 | 0.999957 |  |  | 0.999958 |
| mlp | 61 | linear_attention | 0.999923 | 0.999928 | 0.999931 | 0.999936 | 0.999937 |  |  | 0.999937 |
| mlp | 63 | full_attention | 0.999988 | 0.999970 | 0.999969 | 0.999968 | 0.999965 |  |  | 0.999963 |
| readout | - | - | 0.999967 | 0.999965 | 0.999991 | 0.999964 | 0.999975 |  |  | 0.999961 |
| rmsnorm_input_norm | 0 | linear_attention | 0.999997 | 0.999997 | 0.999997 | 0.999997 | 0.999997 |  |  | 0.999997 |
| rmsnorm_input_norm | 3 | full_attention | 1.000000 | 0.999998 | 0.999998 | 0.999999 | 0.999999 |  |  | 0.999999 |
| rmsnorm_input_norm | 61 | linear_attention | 0.999996 | 0.999996 | 0.999996 | 0.999996 | 0.999996 |  |  | 0.999996 |
| rmsnorm_input_norm | 63 | full_attention | 0.999997 | 0.999997 | 0.999997 | 0.999997 | 0.999997 |  |  | 0.999997 |
| rmsnorm_post_norm | 0 | linear_attention | 0.999996 | 0.999995 | 0.999995 | 0.999995 | 0.999995 |  |  | 0.999994 |
| rmsnorm_post_norm | 3 | full_attention | 0.999996 | 0.999995 | 0.999995 | 0.999995 | 0.999995 |  |  | 0.999995 |
| rmsnorm_post_norm | 61 | linear_attention | 0.999996 | 0.999996 | 0.999996 | 0.999996 | 0.999996 |  |  | 0.999996 |
| rmsnorm_post_norm | 63 | full_attention | 0.999993 | 0.999995 | 0.999995 | 0.999996 | 0.999996 |  |  | 0.999996 |

Cases: 146; failures: 0; min PCC 0.999897 (gated_deltanet L61 S8192).
Diagnostic (not gated): min PCC excluding token 0 = 0.999897, so the first token does not inflate the numbers.
The `test_reference.py` host checks (3 cases: layer 0 and layer 3 strict load with exact tensor equality and
no cross-layer keys; embedding/final norm/readout strict load) pass in the same run.

## Coverage and contract notes

- Lengths 1, 31, 129, 1000, 2048, 8192 for every module and both layer kinds, at layers 0 and 61
  (linear_attention) and 3 and 63 (full_attention). S=8192 (the app `max_length`) runs: it is 4 physical
  chunks of 2048 (**measured**). No length reduction was needed.
- `test_prefill_contract.py` runs 8192 -> 129 -> 2049 -> 4129 back to back on ONE loaded layer instance per
  kind: stale K/V pages or DeltaNet state from the previous request would lower PCC; 2049 and 4129 exercise
  a 1-token and a 33-token tail chunk after full chunks; the last request is repeated and must be
  bit-identical (determinism, **measured**: passes).
- Fallback audit (**measured** with `grep -n "from_torch\|to_torch\|import torch" tt/*.py`): every host
  conversion sits in a setup method (`weight_adapter`, `PplxRotary._build_tables`,
  `PplxGatedAttention._setup`, `PplxGatedDeltaNet._setup`); forward paths are TTNN ops only.
- PCC is also logged without token 0 (diagnostic, not gated) to check that the first token's larger
  activations do not inflate the number; see the table footer.

## Not in this slice (1B or later)

Warmed prefill perf + `tt-perf-report`, watcher run, `context_contract.json`, README, batch > 1 with left
padding (DeltaNet would need the HF padding mask zeroing; the app's `inference.py` runs batch 1),
full-stack PCC and decision agreement (stage 6), dtype sweep (stage 8).

## Commands

```bash
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD HF_HOME=/local/ttuser/gtobar/hf
pytest models/demos/pplx_decider_v1_27b/tests/pcc -q          # every PCC case, appends to the PCC log
python models/demos/pplx_decider_v1_27b/tests/pcc_report.py    # renders the table above
```

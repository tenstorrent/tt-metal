# Llama 3.1 8B Instruct Functional Decoder

Model: `meta-llama/Llama-3.1-8B-Instruct`

Autoport path: `models/autoports/meta_llama_llama_3_1_8b_instruct`

Repo commit: `86f8bc022e6d526d9766539c6ea50137cabec799`

Hardware used: N300 Wormhole, single 1x1 `ttnn.MeshDevice`.

## Implementation

`tt/functional_decoder.py` implements `FunctionalDecoder`, a single HuggingFace
`LlamaDecoderLayer` equivalent for the target model. Llama 3.1 8B has one
meaningful decoder layer kind: dense self-attention plus SwiGLU MLP. The tests
exercise layer 0 with real target shapes; all 32 HF layers share this layer kind.

The implementation:

- subclasses `models.common.lightweightmodule.LightweightModule`;
- loads canonical HF state dict keys through `from_state_dict`;
- uses common `Attention1D` for paged prefill/decode and traced decode;
- uses common `RMSNorm1D`;
- uses an autoport-local TTNN SwiGLU MLP because the repo-local common MLP import
  path currently depends on unavailable `models.tt_transformers` modules;
- supports BF16 weights, BF16 activations, and BF16 paged KV cache;
- is intentionally single-chip only for this functional-decoder stage.

No optimized-decoder, multichip-decoder, full-model, or vLLM work was started.

## Forward Contract

```python
FunctionalDecoder.from_state_dict(
    state_dict,
    *,
    hf_config,
    layer_idx,
    mesh_device,
    max_batch_size=1,
    max_seq_len=None,
    page_block_size=64,
    max_num_blocks=None,
    weight_dtype=ttnn.bfloat16,
    activation_dtype=ttnn.bfloat16,
    kv_cache_dtype=ttnn.bfloat16,
    cache_dir=None,
)
```

`state_dict` may be a full HF model state dict with
`model.layers.<layer_idx>.*` keys or a layer-local state dict.

```python
prefill_forward(
    hidden_states,
    *,
    rot_mats,
    page_table,
    user_id=0,
    chunk_page_table=None,
    chunk_start_idx=None,
)
```

`hidden_states` is a TTNN tensor shaped `[1, 1, seq_len, 4096]` in tile layout.
`rot_mats` is `(cos, sin)` for the same sequence positions, each shaped
`[1, 1, seq_len, 128]`. `page_table` is the paged KV mapping for cache fill.
The return tensor has the same shape as `hidden_states`.

```python
decode_forward(
    hidden_states,
    *,
    current_pos,
    rot_mats,
    page_table,
)
```

`hidden_states` is a TTNN tensor shaped `[1, 1, batch, 4096]`. `current_pos` is
a TTNN int32 tensor shaped `[batch]` containing absolute decode positions.
`rot_mats` is `(cos, sin)` for those positions, each shaped `[1, batch, 1, 128]`.
`page_table` is the same paged KV mapping used for prefill. The return tensor
has the same shape as `hidden_states`.

Runtime prefill/decode paths do not call `torch`, `ttnn.from_torch`,
`ttnn.as_tensor`, or `ttnn.to_torch`. Setup and test-boundary conversions are
outside the audited hot paths.

## Correctness Evidence

Acceptance threshold: PCC >= 0.995 for both prefill and decode.

All decode PCC values below are measured from a replayed TTNN trace, not from an
eager-only decode call. The repeated-input and eager-vs-trace checks verify
trace determinism.

| Test case | Weights | Seq len | Decode context | Prefill PCC | Decode trace PCC | Repeated input PCC | Eager vs trace PCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic paged prefill/decode trace | synthetic from real stats | 128 | 129 | 0.9999777881890652 | 0.9999841394751932 | 1.0 | 1.0 |
| real weights paged prefill/decode trace | real HF layer 0 | 128 | 129 | 0.9999812906688174 | 0.9999836008747124 | 1.0 | 1.0 |
| synthetic long-context paged prefill/decode trace | synthetic from real stats | 32768 | 32769 | 0.9998921568476432 | 0.9999840371223072 | 1.0 | 1.0 |

Paged-cache coverage:

- tests use a non-identity random page-table permutation;
- prefill calls paged cache fill through `Attention1D.prefill_forward`;
- decode updates/reads the paged cache through `Attention1D.decode_forward`;
- `current_pos` is a TTNN tensor in decode so trace capture/replay does not
  depend on a host scalar;
- a separate full-cache contract test constructs a 128K-token paged cache:
  `max_seq_len=131072`, `page_block_size=64`, `max_num_blocks=2048`.

Sequence-capacity evidence:

- HF-vs-TTNN prefill/decode was verified up to `seq_len=32768` and decode
  context `32769`.
- A `seq_len=65536` HF-reference probe was attempted and the host OOM killer
  killed the Python process at `anon-rss:506740044kB` on this 503 GiB host.
- Full 128K cache allocation and page geometry were still verified independently.

## Performance Evidence

The warmed performance run used synthetic weights at `seq_len=128`, page block
size 64, BF16 activation/weights/cache, and signposted prefill and traced-decode
windows.

| Window | Report | Device ops | Host ops in report | Device time sum |
| --- | --- | ---: | ---: | ---: |
| warmed prefill | `tracy/dense/prefill_perf_report.txt` | 24 | 0 | 3494.848 us |
| traced warmed decode replay | `tracy/dense/decode_perf_report.txt` | 22 | 0 | 2482.910 us |

`Device Time` is read from the filtered `tt-perf-report` CSV output:

- `tracy/dense/prefill_perf_report.csv`
- `tracy/dense/decode_perf_report.csv`

The raw device profiler provenance is preserved in:

- `tracy/dense/raw_device_only_ops_perf_results.csv`
- `tracy/dense/raw_profile_log_device.csv`

The normalized `prefill_ops.csv` and `decode_ops.csv` contain real device timing
rows plus signposts. Their tensor metadata and FLOP/DRAM percentages are neutral
placeholders added only so the installed `tt-perf-report` can render a table
from the device-only fallback. The timing columns come from the raw device
profiler output.

## Runtime Fallback Audit

`tests/test_functional_decoder.py` wraps one measured prefill pass and one
decode pass with monkeypatch guards that raise on:

- `ttnn.from_torch`
- `ttnn.as_tensor`
- `ttnn.to_torch`
- common `torch` tensor constructors and math helpers
- `torch.nn.functional.linear`

The audited synthetic, real-weight, and long-context tests pass with
`runtime_fallback_audit='prefill_decode_clean'`.

## Watcher Evidence

Clean watcher run:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/functional_decoder/watcher/synthetic_disable_eth \
python_env/bin/pytest \
  --confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_functional_decoder.py \
  -k synthetic_paged_prefill_decode_trace -vv -s
```

Result: 1 passed, 4 deselected. The watcher log scan found no
`fatal`, `assert`, `exception`, `error`, `noc`, `l1`, `stack`, `sanitize`,
`overflow`, out-of-bounds, fault, hang, or timeout indicators.

Watcher artifacts:

- `watcher/synthetic_disable_eth/generated/watcher/watcher.log`
- `watcher/synthetic_disable_eth/generated/watcher/kernel_names.txt`
- `watcher/synthetic_disable_eth/generated/watcher/kernel_elf_paths.txt`
- `watcher/synthetic_disable_eth/generated/inspector/*.yaml`

A first watcher attempt without `TT_METAL_WATCHER_DISABLE_ETH=1` failed during
device open with an `idle_erisc.elf` code-size overflow before the decoder test
ran. The clean run above uses the CI-style ETH watcher disable flag documented
by the functional-decoder skill.

## Limitations

- Functional decoder only: no optimized, multichip, full-model, or vLLM stage.
- Single 1x1 mesh only; multi-device rejection is intentional in
  `from_state_dict`.
- BF16-only evidence for weights, activations, and KV cache.
- HF-vs-TTNN correctness is verified for the single dense layer kind using
  layer 0. The target model has homogeneous decoder layers.
- Full 128K paged cache geometry is verified, but HF-vs-TTNN full 128K prefill
  is not feasible with the eager HF reference on this host. The largest verified
  HF-vs-TTNN sequence in this run is 32768; the next power-of-two probe, 65536,
  OOM-killed the host process.
- Normal repo pytest collection currently needs
  `--confcutdir=models/autoports/meta_llama_llama_3_1_8b_instruct/tests`
  because the root `conftest.py` imports unavailable
  `models.tt_transformers.demo.trace_region_config` in this checkout.

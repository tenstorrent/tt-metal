# Qwen3.8-27B functional decoder

## Result

This stage implements both decoder-layer kinds advertised by `Qwen/Qwen3.8-27B` in
`tt/functional_decoder.py`: 48 Gated DeltaNet (`linear_attention`) layers and 16
paged full-attention layers. The checkpoint identifies as Qwen3.8 but its text
configuration uses the Qwen3.5 hybrid decoder architecture.

The public contract accepts an on-device BF16 tiled `[B,S,5120]` tensor for
prefill and `[B,1,5120]` for decode. Full attention additionally requires
on-device RoPE tensors, an int32 logical-to-physical page table, a chunk page
table for prefill, and an int32 per-user current-position tensor for decode.
DeltaNet owns fixed-address recurrent and causal-convolution state and internally
pads/chunks prefill to 128 rows, returning only `logical_seq_len` rows. The module
docstring documents the detailed lifetime and trace-input contract.

The exact public construction and forward signatures are:

```python
FunctionalDecoder.from_state_dict(
    state_dict,
    *,
    hf_config,
    layer_idx,
    mesh_device,
    tensor_cache_path: str | Path | None = None,
    max_context: int | None = None,
    page_block_size: int = 64,
    **_kwargs,
)

FunctionalDecoder.prefill_forward(
    hidden_states,
    *,
    cos=None,
    sin=None,
    page_table=None,
    chunk_page_table=None,
    chunk_start_idx=0,
    chunk_start_idx_tensor=None,
    logical_seq_len=None,
)

FunctionalDecoder.decode_forward(
    hidden_states,
    *,
    cos=None,
    sin=None,
    current_position=None,
    page_table=None,
)
```

## Real-weight provenance

The real-weight tests consumed all 25 canonical layer tensors for representative
DeltaNet layer 0 and full-attention layer 3 from checkpoint revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. `real_weight_stats.json`
records every consumed tensor's canonical name, shape, checkpoint dtype, float32
mean, population standard deviation, source shard, index SHA-256, snapshot path,
and revision. `generate_real_weight_stats.py` is the reproducible, local-only
generator; it computes statistics one tensor at a time from safetensors and does
not use TT hardware.

## Correctness

All PCC values below use real checkpoint weights and the target dimensions
(hidden 5120, MLP 17408, 24 Q heads, 4 KV heads, head dimension 256). The
acceptance threshold is the standard functional-decoder bar, PCC >= 0.995.

| Layer kind | Path | PCC |
|---|---|---:|
| full attention, layer 3 | paged prefill, S=128 | 0.9972974494 |
| full attention, layer 3 | traced paged decode, position 128 | 0.9976662041 |
| DeltaNet, layer 0 | prefill, S=128 | 0.9977131745 |
| DeltaNet, layer 0 | eager decode | 0.9988066013 |
| DeltaNet, layer 0 | in-place-state warmup decode | 0.9995071971 |
| DeltaNet, layer 0 | traced decode | 0.9993015776 |

The full-attention trace was replayed twice with identical inputs and produced
bitwise-identical host results. DeltaNet trace tests verify that recurrent and
convolution state buffer addresses remain fixed through warmup, capture, and
replay. Tests also exercise batch two, disjoint/permuted physical pages,
per-user current positions, and non-aligned lengths around tile (32), page (64),
and DeltaNet chunk (128) boundaries.

The complete suite passed: **28 passed**. Exact output is in
`logs/full_suite.log`.

## Context capability

The HF-advertised context is 262,144 tokens and there is no capability reduction.
The layer harness advances DeltaNet state through 2,048 128-token chunks, fills a
full-attention paged cache through 128 2,048-token chunks with a permuted 4,096
entry page table, and decodes at position 262,143. See `../context_contract.json`.

## Performance

One warmed invocation was measured on one Blackhole device. Decode is trace
replay, not eager dispatch. Device time is the sum reported by `tt-perf-report`;
the signpost wall interval includes host synchronization and is intentionally not
reported as kernel latency.

| Layer kind | Mode | Shape | Device ops | Device time |
|---|---|---|---:|---:|
| full attention | paged prefill | B=1, S=128 | 50 | 2.584 ms |
| full attention | traced paged decode | B=1, S=1, pos=128 | 58 | 1.528 ms |
| DeltaNet | prefill | B=1, S=128 | 224 | 5.086 ms |
| DeltaNet | traced decode | B=1, S=1 | 90 | 1.563 ms |

Each mode was captured in a fresh bounded process so the device event buffer did
not fill. Profiler source, capture console, filtered CSV, and human-readable
table are retained together under `perf/captures/<mode>/`.
`perf/provenance.txt` records exact commands and environment information.

## Runtime and device audits

The measured call graph has a source audit forbidding `torch`,
`ttnn.from_torch`, `ttnn.to_torch`, CPU, and NumPy conversion in a single
prefill/decode pass. Conversion is restricted to weight loading and test/input
preparation. The watcher-only real-weight run passed both layer kinds and its
device log contains no fatal, assertion, sanitizer, NoC, watcher, or circular
buffer error. Watcher and profiler were run separately. Evidence is under
`watcher/`. A post-review watcher rerun covering B=2 and the non-aligned maximum
context also passed 4/4; its JUnit result and device log are under
`watcher_rereview/`.

Python shutdown also prints nanobind leak diagnostics. AutoFix isolated this from
the stage: a bare TTNN import, direct decoder-test-module import, and direct
shared-Qwen imports exit cleanly, while an unrelated pre-existing CPU-only gated
attention pytest emits the same two leaked binding instances, 20 types, and 250
functions without opening a device. This is a shared pytest/binding teardown
diagnostic, not a decoder resource or device-close failure; all commands exit 0.
The focused report and raw controls are under `autofix/`.

## Reproduction

From the repository root:

```bash
pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -s

TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=$PWD/models/autoports/qwen_qwen3_8_27b/doc/functional_decoder/watcher \
pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py \
  -k 'real_weights_paged_prefill_and_decode_pcc or real_weights_deltanet_prefill_and_traced_decode_pcc' -s

for CASE in full_prefill full_decode gdn_prefill gdn_decode; do
  python -m tracy -r -p -v -m pytest \
    "models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py::test_profile_warmed_prefill_and_traced_decode[${CASE}-device_params0]" -s
done
```

## Limitations

This is deliberately a functional, single-device decoder-layer stage. It does
not implement an optimized decoder, multi-chip partitioning, a block stack,
full model, generator, or vLLM integration. Performance numbers are evidence of
the functional kernels and are not an end-to-end throughput claim.

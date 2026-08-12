# Functional decoder — `meta-models/Muse-Glimmer-30B`

Single-chip (1x1 Blackhole mesh) TTNN implementation of the HuggingFace
`MuseGlimmerTextDecoderLayer`, with paged prefill, paged traced decode, and
HF-vs-TTNN correctness evidence at the full advertised 131072-token context.

| item | value |
| --- | --- |
| implementation | `models/autoports/meta_models_muse_glimmer_30b/tt/functional_decoder.py` |
| tests | `models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py` |
| HF reference harness | `models/autoports/meta_models_muse_glimmer_30b/tests/reference.py` |
| capacity probe | `models/autoports/meta_models_muse_glimmer_30b/tests/functional_decoder_capacity_probe.py` |
| device | 1 x Blackhole (`ttnn.MeshShape(1, 1)`, 11x10 compute grid), `trace_region_size=0` |
| dtypes | BF16 weights / activations / KV cache, TILE layout, DRAM interleaved |
| transformers | 5.15.0 (the first release that ships `transformers.models.muse_glimmer`) |
| acceptance bar | PCC >= 0.995 (skill default; no model-specific exception needed) |
| result | **74/74 tests pass, 194 PCC checks, worst 0.99742** |

## Architecture

`config.json` advertises two decoder-layer kinds, selected per layer index by
`text_config.layer_types` and `text_config.layer_rope_theta`:

| kind | `layer_types` | `layer_rope_theta` (gate) | attention mask | first layer index |
| --- | --- | --- | --- | --- |
| `sliding` | `sliding_attention` | `500000.0` | causal, window 2048 | 0 |
| `full` | `full_attention` | `0` (NoPE) | causal, unbounded | 3 |

`layer_rope_theta[i]` is only HF's per-layer *gate* — the rotary base itself is
the model-level `rope_parameters["rope_theta"]` (limitation 14).

The pattern is `[sliding, sliding, sliding, full]` repeating over 52 layers.
Everything else about the two kinds is identical, so `FunctionalDecoder` is one
parameterised implementation and every test is parameterised over both kinds
(layer 0 and layer 3 are the canonical instances).

Layer body (verified line-by-line against
`transformers/models/muse_glimmer/modeling_muse_glimmer.py`):

```
residual = x
h = input_layernorm(x)                    # centered RMSNorm, eps = rms_norm_eps = 1e-5
h = self_attn(h)
h = post_attention_layernorm(h)           # centered RMSNorm, eps = post_norm_eps = 1e-8
x = residual + h
residual = x
h = pre_feedforward_layernorm(x)          # centered RMSNorm, eps = 1e-5
h = down_proj(silu(gate_proj(h)) * up_proj(h))
h = post_feedforward_layernorm(h)         # centered RMSNorm, eps = 1e-8
x = residual + h
```

"Centered" RMSNorm is `rms_norm(x) * (1 + w)`; the `1 +` is folded into the
device weight at setup time.

Attention (`MuseGlimmerTextAttention`), hidden 6656, 32 Q heads / 2 KV heads,
head_dim 128, no biases:

```
q,k,v = q_proj/k_proj/v_proj(h)                # fused into one [6656, 4608] matmul
q = rmsnorm_no_scale(q) * qk_scale_factor      # scale-less RMSNorm over head_dim, eps 1e-5
k = rmsnorm_no_scale(k)                        # v is NOT normed
q,k = rope(q,k)                                # sliding layers only (NoPE on full layers)
o = sdpa(q,k,v, scale=1/sqrt(128), window=2048 on sliding layers)
o = concat_heads(o) * sigmoid(attn_gate_proj(h))   # output gate on the pre-o_proj tensor
o = o_proj(o)
```

`qk_scale_factor = 3.87` multiplies Q before RoPE.  RoPE is a rotation and Q's
only consumer is `q @ k^T`, so the constant is folded into the SDPA `scale`
(`3.87 / sqrt(128)`) — algebraically identical, one device op cheaper.

## Forward contract

```python
FunctionalDecoder.from_state_dict(
    state_dict, *, hf_config, layer_idx, mesh_device,
    max_batch_size=1, max_seq_len=None, page_block_size=64, max_num_blocks=None,
    weight_dtype=ttnn.bfloat16, activation_dtype=ttnn.bfloat16,
    kv_cache_dtype=ttnn.bfloat16, prefill_chunk_size=8192,
) -> FunctionalDecoder

decoder.prefill_forward(hidden_states, *, page_table, user_id=0, start_pos=0,
                        sliding_kv_tail=None, return_sliding_kv_tail=False) -> ttnn.Tensor
decoder.decode_forward(hidden_states, *, current_pos, page_table, rope_pos_ids=None) -> ttnn.Tensor
decoder.sliding_kv_tail_len(start_pos) -> int
decoder.forward(hidden_states, *, mode="prefill"|"decode", **kwargs)
```

* `prefill_forward`
  * `hidden_states`: TTNN tile tensor `[1, 1, seq_len, 6656]`.  `seq_len` is the
    **logical** prompt length and may be any value in
    `[1, max_seq_len - start_pos]` — it does not have to be a multiple of the
    tile height (32), the page block size (64) or the internal prefill chunk
    size (8192).  The layer pads to a tile multiple, masks causally, writes only
    real-plus-padding K/V into the user's own pages, and slices the output back
    to `seq_len`.
  * `page_table`: `int32` ROW_MAJOR `[max_batch_size, ceil(max_seq_len/block)]`,
    virtual block -> physical block.
  * `user_id`: page-table row / cache slot.
  * `start_pos`: absolute position of the first token (`0` for a fresh prompt).
    `> 0` supports caller-level chunked prefill and must be a multiple of the
    page block size; positions `[0, start_pos)` must already be in the cache.
    The caller must keep `start_pos` equal to the *logical* length already
    prefilled — continuing a 100-token prefill at `start_pos=128` would attend
    this layer's own tile padding.
  * `sliding_kv_tail` / `return_sliding_kv_tail`: on `sliding` layers a
    continuation (`start_pos > 0`) **must** be handed the previous call's last
    `sliding_kv_tail_len(start_pos)` K/V rows as `(k, v)`, each
    `[1, 2, tail_len, 128]`.  `chunked_scaled_dot_product_attention` (the paged
    reader) has no sliding-window mask, so the window cannot be read back out
    of the paged cache the way `full` layers read their prefix; the tail is
    handed over explicitly, the same contract `models/demos/gemma4` uses at
    generator level.  Omitting it raises `ValueError` rather than silently
    computing a truncated window.  Get the tail by passing
    `return_sliding_kv_tail=True` to the previous call, which then returns
    `(output, tail)`; the tail is consumed by the call it is passed to.
  * returns `[1, 1, seq_len, 6656]` (or `(output, tail)`).
* `decode_forward`
  * `hidden_states`: TTNN tile tensor `[1, 1, batch, 6656]`.
  * `current_pos`: `int32` device tensor `[batch]` — the absolute position each
    user decodes (also the KV write index).  Per-user (ragged) positions are
    supported and tested.
  * `rope_pos_ids`: `uint32` device tensor `[1, batch]` with the same positions,
    used for the on-device `ttnn.embedding` cos/sin gather.  Required on
    `sliding` layers, ignored on `full` (NoPE) layers.
  * `page_table`: the same mapping prefill used.
  * returns `[1, 1, batch, 6656]`.

Both entry points are trace-safe: every runtime input is a device tensor whose
*contents* the caller refreshes outside the captured region.  The layer owns its
paged KV cache (`decoder.kv_cache -> (k_cache, v_cache)`) and its RoPE tables;
all `torch` / `ttnn.from_torch` work happens in `from_state_dict`.

### Internal prefill chunking

`prefill_forward` processes the prompt in `prefill_chunk_size` (default 8192)
slices through the *whole* layer.  A 131072-token prompt materialised in one
shot would need >10 GB of MLP activations (intermediate width 19968); chunking
bounds the transient working set to the chunk.  Per chunk:

* K/V are written to the paged cache with `ttnn.experimental.paged_fill_cache`
  at the chunk's absolute block offset;
* `full` layers attend the whole prefix from the paged cache with
  `ttnn.transformer.chunked_scaled_dot_product_attention` (chunk 0 uses the
  in-memory square SDPA);
* `sliding` layers attend a square `[previous-window tail | this chunk]` slice —
  the paged chunked SDPA op has no sliding-window mask — with the Q rows for the
  tail zero-filled and dropped afterwards.  The carried tail is taken from
  `[previous tail | this chunk]` so a chunk shorter than the 2048 window still
  hands over the full history.

## Correctness

`pytest models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py`
→ **74 passed** (`test_results.xml`, `logs/full_test_run.log`).
All 194 PCC checks are in `logs/pcc_summary.txt`, regenerated from the suite log
by `bench/summarize_pcc.py`; the eight lowest are:

| PCC | check |
| --- | --- |
| 0.997422 | real-weights decode[full] pos=2049 |
| 0.997619 | decode[full] prompt=2048 pos=2049 |
| 0.997646 | decode[sliding] vs **FP32** HF reference pos=2049 |
| 0.997681 | traced decode[sliding] pos=2048 |
| 0.997765 | prefill[full] full-context seq_len=130073 (last 32 rows) |
| 0.997775 | prefill[full] full-context seq_len=131072 (interior @65536, 32 rows) |
| 0.997775 | prefill[full] full-context seq_len=131072 (last 32 rows) |
| 0.997777 | decode[full] user_id=2 pos=12345 |

### Live re-verification (2026-08-12)

The stage was re-run end to end on live hardware.  First against the *unchanged*
sources — `tt/functional_decoder.py` blob
`0f949dc99d23e1148cb94da4bb39be9600b015cd` and `tests/test_functional_decoder.py`
blob `175b59cbeee9bd29fd51c2b0ac4322f43bf5bddc`, both identical to
`2e2acc13f96` — and then again after the round-5 review's config-resolution
hardening (limitation 14):

| re-run | result | artifact |
| --- | --- | --- |
| full suite, pre-change | **73 passed** in 394.88 s; all 194 PCC checks **bit-identical** to the previously committed run | `logs/reverify_2026_08_12_full_test_run.log`, `logs/reverify_2026_08_12_test_results.xml` |
| watcher, pre-change | **14 passed** in 105.47 s, log clean | `watcher_reverify/watcher.log.gz`, `logs/reverify_2026_08_12_watcher_run.log` |
| full suite, post-change (canonical) | **74 passed** in 395.38 s; the same 194 PCC values again, bit-identical | `logs/full_test_run.log`, `test_results.xml` |
| watcher, post-change (canonical) | **15 passed** in 109.81 s, log clean | `watcher/watcher.log.gz`, `logs/watcher_run.log` |
| perf | not re-captured; the six committed Tracy windows are re-derived **and reconciled against their raw captures** instead | `logs/perf_summary.txt` |

Three runs of 194 PCC checks — the original stage capture, a pre-change rerun and
a post-change rerun — agree to the last digit, which is both the determinism
evidence at suite scale and the proof that the config-resolution edit changed no
number.  The Tracy captures predate that edit by design: it touches only
`from_state_dict`-time config resolution, docstrings and one new test, adds and
removes no device op, and cannot move a cos/sin table whose PCC is unchanged.
Re-capturing would add measurement noise and desynchronise the fused stage's
committed `baseline` columns, which quote exactly these six windows.

```bash
D=models/autoports/meta_models_muse_glimmer_30b/doc/functional_decoder
python $D/bench/summarize_pcc.py --compare $D/logs/reverify_2026_08_12_full_test_run.log
# RERUN_IDENTICAL 194 checks bit-identical to the committed run
python $D/bench/check_watcher.py                                      # WATCHER_CLEAN
python $D/bench/check_watcher.py $D/watcher_reverify/watcher.log.gz   # WATCHER_CLEAN
python $D/bench/summarize_perf.py --check   # 6 windows, raw-vs-filtered delta 0.000 us
python $D/bench/refresh_context_contract.py --check                   # matches the run
```

### Sequence-length coverage

`test_prefill_pcc` runs both kinds at
`seq_len ∈ {1, 100, 128, 2048, 2049, 4097, 8192, 8193, 12345}`:

| length | why |
| --- | --- |
| 1 | minimal smoke, sub-tile |
| 100 | non-aligned, sub-tile-multiple, inside one page block |
| 128 | exactly 4 tiles / 2 page blocks |
| 2048 | exactly the sliding window |
| 2049 | one token past the sliding window |
| 4097 | one token past a page-block-aligned length, mid-chunk |
| 8192 | exactly the prefill chunk size |
| 8193 | one token past the chunk size (forces a 1-token second chunk) |
| 12345 | long, divisible by neither tile, page block nor chunk |

`test_full_context_prefill_tail_pcc` adds `131072` and the non-aligned
`130073`.  `test_decode_pcc` decodes four consecutive tokens past prompts of
100 / 2048 / 3000, and `test_full_context_decode_pcc` decodes at position
131071.

### Caller-chunked (continuation) prefill

`test_continuation_prefill_pcc` splits a prompt into two `start_pos`-separated
calls and compares the concatenated output against a *single-shot* HF prefill of
the whole prompt, then decodes past the join to prove the paged prefix is
intact.  `64+100` and `1024+1024` are below the 2048 sliding window, so the
handed-over tail is *shorter* than the window — the regime the internal
tail-carry bug (work-log bug 1) lived in:

| split | kind | continuation prefill PCC | decode-after-continuation PCC |
| --- | --- | --- | --- |
| 4096 + 3000 | sliding | 0.998494 | 0.998646 |
| 4096 + 3000 | full | 0.998287 | 0.998375 |
| 1024 + 1024 | sliding | 0.998601 | 0.998703 |
| 1024 + 1024 | full | 0.998454 | 0.998669 |
| 64 + 100 | sliding | 0.999110 | 0.999014 |
| 64 + 100 | full | 0.999000 | 0.999150 |

`test_continuation_prefill_requires_sliding_tail` asserts that a sliding
continuation without its window tail raises rather than silently truncating.

### FP32 reference control

Every other PCC number compares the BF16 TTNN layer against a BF16 HF layer, so
an error common-mode to two bfloat16 implementations would be invisible.
`test_prefill_decode_pcc_vs_fp32_reference` re-runs the same weights and the
same (BF16-rounded) inputs through an **FP32** HF layer:

| kind | prefill PCC vs FP32 HF | decode PCC vs FP32 HF |
| --- | --- | --- |
| sliding | 0.998623 | 0.997646 |
| full | 0.998487 | 0.998515 |

These match the BF16-reference numbers to ~1e-3, i.e. the residual error is
TTNN-side BF16 rounding, not a shared modelling mistake.

### Paged KV cache

Every test uses a **randomly permuted, non-identity** page table
(`make_page_table`) across all users, so a block-address or row-indexing bug
would show up immediately.  `test_batched_prefill_decode_pcc` prefills
4, 13 and 32 users with *ragged* prompt lengths (2000..3147, straddling the 2048
sliding window) into one shared cache and then decodes all of them in one
batched call with *different* `current_pos` per user.  Batch 13 is prime and
larger than the 11-wide device grid, so it has no `batch`-core rectangle and
forces the decode head-concat's shape-agnostic fallback.

### Sliding-window boundary control (decode kernel)

`test_decode_sdpa_sliding_window_semantics` probes
`ttnn.transformer.paged_scaled_dot_product_attention_decode` *directly* against
an explicit PyTorch `kv_idx > cur_pos - W` mask over a known permuted paged
cache, at `cur_pos ∈ {2047, 2048, 2049, 5000}` — i.e. either side of the window
boundary.  The prefill op's window is pinned separately by
`sdpa_sliding_window_chunk_repro.py`; decode uses a different kernel and an
off-by-one there would be invisible under the end-to-end BF16 PCC floor.
Measured 0.99959 / 0.99959 / 0.99959 / 0.99967 against a bar of 0.999.

### Non-zero cache slot with multi-chunk prefill

`test_multi_chunk_prefill_nonzero_user` prefills 12345 tokens (two internal
chunks) into `user_id=2` of a 4-slot cache and decodes only that slot while the
other three sit at position 0 — the page-table row slicing and the chunked paged
read both run off row 2.  PCC 0.998483 / 0.998226 prefill, 0.998409 / 0.997777
decode.

### Capability-contract evidence table

The machine-readable form of this table is the `functional_decoder` block of
`models/autoports/meta_models_muse_glimmer_30b/doc/context_contract.json`: HF
advertised context, supported context, capability reduction (`none`), the tested
prefill/decode contexts and PCCs, batch coverage, non-aligned lengths, real
weights, determinism, fallback audit, watcher, capacity probes, byte budget and
perf.  Its measured fields are generated by `bench/refresh_context_contract.py`.
That file's *top level* belongs to the newest stage that touched it, so each
stage keeps its own block.

| claim | evidence | remaining risk |
| --- | --- | --- |
| Advertised context 131072 supported in prefill | `test_full_context_prefill_tail_pcc[131072-{sliding,full}]` PCC 0.99852 / 0.99778; `logs/capacity_probe_131072_layer{0,3}.log` | the HF reference covers the last 32 query rows against the full prefix (a 131072-query HF forward is not CPU-tractable); earlier rows are covered at 12345 and below.  For `sliding` those 32 rows only depend on the last ~2080 tokens (limitation 7) |
| Caller-chunked (`start_pos > 0`) prefill is correct for both kinds | `test_continuation_prefill_pcc` compares 4096+3000 against a single-shot HF prefill, then decodes at 7096 | sliding continuations require the caller to thread `sliding_kv_tail`; omitting it raises (`test_continuation_prefill_requires_sliding_tail`) rather than silently truncating the window |
| BF16 result is not a common-mode error with the BF16 HF reference | `test_prefill_decode_pcc_vs_fp32_reference` (FP32 HF control) 0.99862/0.99765 sliding, 0.99849/0.99852 full | none identified |
| Advertised context 131072 supported in decode | `test_full_context_decode_pcc[{sliding,full}]` at `cur_pos=131071`, PCC 0.99844 / 0.99796 | none identified |
| Non-aligned logical lengths accepted | 9 prefill lengths above + 130073 | none identified |
| Both layer kinds implemented from one code path | every test parameterised over `sliding` / `full`; `test_resolve_layer_kind_rejects_unsupported_pairings` walks all 52 layers and asserts the two other `(layer_type, rope_theta)` pairings raise | layers 1,2,4,5,… are the same kinds as 0 and 3 by config; not each individually run on device |
| The prefill kernel's sliding window is exactly 2048 tokens inclusive of self | HF `sliding_window_overlay` is `kv_idx > q_idx - W`; the standalone `sdpa_sliding_window_chunk_repro.py` pins the prefill op against that mask at PCC 0.99987, and end-to-end prefill PCC holds at 2048/2049/3000 | none identified |
| Paged prefill + paged decode share one cache correctly | permuted page tables everywhere; batch 4/13/32 with ragged positions straddling the window; `test_multi_chunk_prefill_nonzero_user` runs a two-chunk prefill + decode off cache slot 2 | none identified |
| The decode kernel's sliding window is exactly 2048 tokens inclusive of self | `test_decode_sdpa_sliding_window_semantics` at cur_pos 2047/2048/2049/5000 vs an explicit torch mask, PCC >= 0.99959 | none identified |
| Batch 4 / 13 / 32 with ragged prompt lengths and ragged decode positions | `test_batched_prefill_decode_pcc[{4,13,32}-{sliding,full}]` at 4096 context. 13 is prime and > the 11-wide grid, so it has no `batch`-core rectangle and forces the decode head-concat's shape-agnostic fallback | batch 32 at the full 131072 context was not run (4.3 GB of KV alone); not required by the stage |
| Real checkpoint loads through `from_state_dict` | `test_real_weights_prefill_decode_pcc`, `test_real_state_dict_key_and_shape_contract` | none identified |
| Decode is genuinely traced | `test_traced_decode_pcc` measures PCC **from the replay output**; `test_traced_decode_advances_positions` replays one capture across 3 positions | none identified |
| No host fallback in a measured pass | `test_no_host_fallback_in_forward[{3000,12345}-{sliding,full}]` traps `ttnn.from_torch/to_torch/as_tensor` and 13 `torch` entry points; 12345 covers the multi-chunk paths (paged chunked SDPA, sliding tail concat/clone, page-table and RoPE-table slicing) | none identified |

### Real weights

`test_real_weights_prefill_decode_pcc` loads exactly the target layer's tensors
out of the released safetensors shards (`safe_open`, no 56 GB full-model load)
and passes at `seq_len=2049`:

| kind | prefill PCC | decode PCC |
| --- | --- | --- |
| sliding (layer 0) | 0.99815 | 0.99922 |
| full (layer 3) | 0.99807 | 0.99742 |

`test_real_state_dict_key_and_shape_contract` asserts the real checkpoint's key
set and shapes match what `from_state_dict` consumes.  Normal runs use
deterministic synthetic weights generated from the real per-tensor statistics
recorded in `tests/layer_weight_stats.json`, so CI needs no weight download.

### Determinism

`test_determinism_repeated_inputs` runs the same 1024-token prefill three times
and the same decode three times and asserts **bit-identical** outputs
(`torch.equal`), for both layer kinds.

### Watcher

```
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=<abs>/doc/functional_decoder/watcher \
python -m pytest <test file>::test_prefill_pcc[12345-sliding] ... -q
```

15 tests passed (multi-chunk prefill both kinds, decode both kinds, continuation
prefill including the short-tail 64+100 case, traced decode replay both kinds,
batch-13 (fallback head-concat) and batch-4 prefill/decode, the non-zero
cache-slot multi-chunk prefill, the decode sliding-window control, the
`seq_len == max_seq_len == chunk` regression, and the RoPE-base config guard).
`watcher/watcher.log.gz` (11867 lines, gzipped to stay under the repo's 500 KB
file-size hook) contains only the legend, 5809 `k_ids:` lines, 11 periodic dumps
with stack-usage summaries (22 `Dump #` boundary lines, 50 stack-usage rows), and
4 attach + 4 detach lines — zero occurrences of `Watcher detected`, `tripped`,
`sanitize`, `TT_ASSERT`, `DEBUG_ASSERT`, CB/L1/NOC out-of-bounds or
hardware-fault messages.  Console log: `logs/watcher_run.log`.  Re-derive with

```bash
python bench/check_watcher.py                            # WATCHER_CLEAN
python bench/check_watcher.py watcher_reverify/watcher.log.gz
```

`check_watcher.py` asserts the *benign structure* too (line count, dumps,
`k_ids:`, stack-usage rows, attach/detach), not just the absence of fatal
messages, so an empty or truncated log fails with
`WATCHER_LOG_NOT_A_REAL_RUN` instead of passing by having nothing in it.  Those
minimums are calibrated to this suite's ~11.8 k-line runs: they fail
conservatively (a bad log is never certified), but a legitimately shorter watcher
run would need them relaxed.
`watcher_reverify/watcher.log.gz` is the pre-code-change reproduction from the
same day (14 tests, 11868 lines, same benign structure, same zero fatal
messages).

## Performance

Warmed, signposted windows, profiled with Tracy in **separate runs from the
watcher run**.  Column used: `Device Time` from the `tt-perf-report --csv`
output, in **microseconds** (the report renders the same values as `µs`).
`Op-to-Op Gap` is reported separately.

The decode windows use **8 trace replays** (`MG_PERF_DECODE_ITERS=8`).  A first
attempt at 32 replays overflowed the device profiler's DRAM marker buffer
(`Profiler DRAM buffers were full, markers were dropped!`, 360 occurrences for
the sliding kind) and silently under-counted ops, which made decode look ~13 %
faster than it is.  All six committed captures are warning-free and every
op-code count is an exact multiple of the replay count:

```bash
grep -c "markers were dropped" doc/functional_decoder/logs/tracy_*.log   # all 0
# and every OP Code count in each *_perf_report.csv divides by its replay count
```

Both integrity checks, and every per-iteration number in the table below, are
re-derived from the committed CSVs by `bench/summarize_perf.py` into
`logs/perf_summary.txt` (`--check` exits non-zero if the summary drifts or a
capture fails an integrity check), so nothing in this section is a hand
transcription.

| kind | mode | context | window | ops/iter | device time / iter | incl. op-to-op gaps |
| --- | --- | --- | --- | --- | --- | --- |
| sliding | prefill, 8192 tokens, batch 1 | — | `PERF_PREFILL` | 42 | 101.23 ms (80.9 k tok/s of layer throughput) | 101.26 ms |
| full | prefill, 8192 tokens, batch 1 | — | `PERF_PREFILL` | 24 | 99.38 ms (82.4 k tok/s) | 99.39 ms |
| sliding | traced decode, batch 1 | 2048 | `PERF_DECODE` | 64 | **3.163 ms/token** | 3.226 ms |
| sliding | traced decode, batch 1 | **131071** | `PERF_DECODE` | 64 | **3.160 ms/token** | 3.223 ms |
| full | traced decode, batch 1 | 2048 | `PERF_DECODE` | 32 | **3.080 ms/token** | 3.114 ms |
| full | traced decode, batch 1 | **131071** | `PERF_DECODE` | 32 | **3.575 ms/token** | 3.608 ms |

Decode is measured at both ends of the advertised context because the two kinds
scale differently: the `sliding` layer's SDPA reads at most its 2048-token
window, so 2048 and 131071 are identical (36.1 vs 35.1 µs of SDPA); the `full`
(NoPE) layer reads the whole prefix, so its SDPA goes 35.7 µs → 529.7 µs and the
step costs +0.49 ms.  The long-context windows advance `current_pos` without a
131071-token prefill — decode cost depends on how many KV tokens the op reads,
not on their contents, and a profiled long prefill would overflow the marker
buffer.

Decode device-time split (per token, sliding@2048 / full@131071):

| share | op | cores | note |
| --- | --- | --- | --- |
| 80.1 % / 70.8 % | 6 x `MatmulDeviceOperation` (2533 / 2531 µs) | 64–104 | weight-bandwidth bound: one step reads `(6656*4608 + 6656*4096 + 4096*6656 + 3*6656*19968) * 2 B = 967,835,648 B ≈ 968 MB`, ≈2.5 ms at the observed effective DRAM bandwidth |
| 14.1 % / 12.5 % | 6 x `LayerNormDeviceOperation` (447 µs) | **1** | `ttnn.rms_norm` picks a single core for the `[1,1,1,6656]` decode-shaped tensors (the same op runs on 110 cores in prefill).  Not a correctness issue, but it is the largest non-matmul cost and a named target for the optimized-decoder stage — it is *not* a bandwidth effect |
| 1.1 % / 14.8 % | `SdpaDecodeDeviceOperation` (36.1 / 529.7 µs) | 110 | the only op that scales with context |
| ~4.7 % / ~1.9 % | everything else (heads split/concat, paged update, RoPE gather, elementwise) | mixed | |

Lowering weight precision, DRAM-sharding the matmuls and giving the decode
RMSNorms a real core grid is the optimized-decoder stage's job, not this one.

Artifacts (per kind, under `tracy/<kind>/`):

* `prefill_ops.csv`, `decode_ops.csv.gz`, `decode_131071_ops.csv.gz` — raw Tracy
  ops CSVs copied from `generated/profiler/reports/<ts>/` (the two decode ones
  are gzipped to stay under the repo's 500 KB file-size hook; `gunzip -k` them
  to re-run `tt-perf-report` on the raw capture)
* `*_perf_report.txt` — human-readable tt-perf-report tables
* `*_perf_report.csv` — filtered CSV for the signposted window
* `*_perf_report.console.log` — provenance for the `--csv` invocation
* `*_perf_report_stacked.{csv,png}` — tt-perf-report 1.2.8 stacked breakdown

Evidence-regeneration scripts (`bench/`), so every number above is a command
rather than a transcription:

| script | regenerates | `--check` |
| --- | --- | --- |
| `bench/summarize_pcc.py` | `logs/pcc_summary.txt` from `logs/full_test_run.log` | yes (also `--compare <other suite log>`) |
| `bench/summarize_perf.py` | `logs/perf_summary.txt` from the six `*_perf_report.csv` windows, their Tracy logs and their raw ops CSVs (raw-vs-filtered reconciliation) | yes |
| `bench/check_watcher.py` | the watcher-clean verdict for a given watcher log, including its benign-structure minimums | exit code is the verdict |
| `bench/refresh_context_contract.py` | the measured fields of the `functional_decoder` block of `doc/context_contract.json` | yes |

Commands are in `work_log.md`; the driver script shape is:

```bash
MG_PERF_DECODE_ITERS=8 python -m tracy -r -p -v -m pytest \
  models/autoports/meta_models_muse_glimmer_30b/tests/test_functional_decoder.py::test_perf_prefill[sliding]
cp generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv doc/functional_decoder/tracy/sliding/prefill_ops.csv
tt-perf-report doc/functional_decoder/tracy/sliding/prefill_ops.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-summary --no-advice \
  > doc/functional_decoder/tracy/sliding/prefill_perf_report.txt
```

## Limitations and known issues

1. **tt-metal bug — sliding-window prefill SDPA with `q_chunk_size == 2 *
   k_chunk_size`.** `ttnn.transformer.scaled_dot_product_attention(...,
   is_causal=True, sliding_window_size=2048)` returns wrong results (PCC ~0.97
   vs a plain PyTorch masked-softmax reference) at `seq_len ∈ {2080, 4128,
   8224}` for both `q256/k128` and `q128/k64`, while `q128/k128` and `q256/k256`
   stay at ~0.9998 on the same inputs.  Reproducer:
   `sdpa_sliding_window_chunk_repro.py`, log:
   `logs/sdpa_sliding_window_chunk_repro.log`.  Worked around by always using
   `q_chunk_size == k_chunk_size` in `_prefill_program_config`.
2. **tt-metal binding bug — `scale` is unusable with most Python floats on
   `chunked_scaled_dot_product_attention`.**  That op's nanobind registration
   (`ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp:521`, and
   `joint_scaled_dot_product_attention` at :571) declares `scale` as
   `std::optional<float>` with `nb::arg("scale").noconvert()`, so a Python
   double that is not exactly representable as a `float` raises a
   signature-mismatch `TypeError` — `scale=1.0` works, `scale=0.3` does not.
   The plain `scaled_dot_product_attention` and the decode ops do **not** use
   `.noconvert()` and accept any double.  Worked around uniformly by rounding
   `sdpa_scale` to the nearest float32 (`_as_float32`); relative error ~1e-8.
3. **`ttnn.slice` aliasing.** A full-range `ttnn.slice` returns the *input
   tensor*, so deallocating the result frees the input.  Hit twice: once on the
   page table (surfaced as `bad optional access` from a later slice of the freed
   buffer) and once, latently, on the persistent RoPE cos/sin tables when
   `start_pos == 0 and length == max_seq_len`.  Both sites now check ownership
   before deallocating (`_page_table_row`, `_prefill_rope_tables`); the RoPE one
   is regression-tested by `test_prefill_seq_len_equals_max_and_chunk`.
4. **`ttnn.pad` cannot front-pad a tiled tensor**, so the sliding chunked
   prefill builds its filler Q rows with `ttnn.zeros` + `ttnn.concat`.
5. **Full-context prefill reference is a reduced harness** — two 32-row blocks
   (interior @65536 and the tail) against the complete preceding prefix.
   Running 131072 HF queries on CPU is not tractable.
6. **Batch 32 at the full 131072 context was not exercised.**  The blocker is
   the HF *reference* harness, not the device: 32 x 131072 KV is 4.3 GB, which
   fits the chip comfortably (see `context_contract.byte_budget_at_full_context`),
   but building 32 independent 131072-token CPU references is not tractable.
   Nothing in the code or config caps batch or context.  Batches 4/13/32 are
   tested at 4096; batch 1 is tested at 131072.
   Batch sizes with no `batch`-core rectangle on the device grid (13, 17, 19,
   23, …) fall back to a slower shape-agnostic head-concat; correctness is
   covered by `test_batched_prefill_decode_pcc[13-*]`, speed is not.
7. **Sliding-layer full-context prefill evidence is window-bounded per row.**
   With a 2048-token window, any 32-row block of a sliding prefill depends only
   on the preceding ~2080 tokens, so no single block validates the whole
   131072-token prompt.  `test_full_context_prefill_tail_pcc[131072-*]`
   therefore checks two blocks: the last 32 rows *and* an interior block at
   row 65536 — the first row of internal chunk 9, which depends entirely on the
   8th carried window (PCC 0.998490 sliding / 0.997775 full).  The remaining 14
   tail carries are structurally identical and are covered directly at 8193 /
   12345 (multi-chunk) and by `test_continuation_prefill_pcc`.  The `full`
   variant's blocks *do* each depend on the whole prefix, because its chunked
   SDPA reads every prior page.
8. **Decode RMSNorm runs on one core** (~447 µs/token, 14 % of decode).  Named
   as an optimized-decoder target above; correctness is unaffected.
9. **The SDPA sliding-window chunk bug (1) has no upstream issue filed.**  This
   autonomous stage does not open GitHub issues; the reproducer and log are
   committed here so the next human can file one.  Every other model using
   `sliding_window_size` with `q_chunk == 2 * k_chunk` is silently affected.
10. **Folded norm weight is stored BF16, HF folds in FP32.**  Measured with
   `norm_weight_dtype_probe.py`: an FP32 device weight changes the norm's PCC by
   ~1e-6 (0.99994260 BF16 vs 0.99994183 FP32) because `ttnn.rms_norm` emits BF16
   either way, so the output rounding dominates.  Kept BF16.  Log:
   `logs/norm_weight_dtype_probe.log`.
11. **Scope.** Correctness-first: BF16 everywhere, DRAM interleaved, no weight
   quantisation, no fusion, no multi-device. This is the functional stage.
   Out of scope here but waiting for later stages: the checkpoint is
   `MuseGlimmerForConditionalGeneration` and also ships `MuseGlimmerVisionModel`
   + `MuseGlimmerVisionAdapter` with `image_token_id` / `video_token_id`, and the
   model level carries `final_logit_softcapping = 20.0` and
   `output_multiplier = 0.19611613513818404`.  None of those touch a text decoder layer, so
   none are implemented or tested here — but a full-model or serving stage that
   ignores the softcapping/multiplier or the vision tower will be wrong.
12. **`transformers` was upgraded to 5.15.0** in
   `/home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv` (the repo pins 5.12.1,
   which does not contain `transformers.models.muse_glimmer` at all, so the
   config cannot even be loaded there).  The venv is model-specific, so a CI job
   running this suite against the repo pin fails at import, not at PCC.
13. **tt-metal bug workaround — `nlp_create_qkv_heads_decode` with a DRAM
   input.**  On Blackhole that op's interleaved-DRAM reader zeroes odd-indexed Q
   rows (NoC DRAM-read alignment-match violation, tt-metal #16667), so decode
   stages the fused QKV in L1 with `ttnn.to_memory_config` before the split —
   the same workaround `models/demos/gemma4/tt/attention/operations.py:60-67`
   applies.  Correctness depends on it.  Cost: one `CopyDeviceOperation`,
   **2.13 µs/token** (0.07 % of a decode step), inside the "everything else" row
   of the perf split above.
14. **The RoPE base is read from `rope_parameters`, not `layer_rope_theta`.**  HF
   uses `layer_rope_theta[i]` only as a NoPE *gate*
   (`position_embeddings if config.layer_rope_theta[i] else None`) and takes the
   rotary base from the model-level `rope_parameters["rope_theta"]`.  This
   checkpoint stores `500000.0` in both, so reading the wrong one would be
   invisible in PCC; `from_state_dict` therefore pins `rope_parameters` in
   `_require_muse_glimmer_text_config` and
   `test_rope_base_is_the_model_level_rope_parameters` fails loudly if a future
   revision moves the base without moving the gate.

# E3: functional audio tower (stage report)

Scope: the EmbeddingGemma 2 audio path (HF `Gemma4AudioModel` + `embed_audio`) on one Blackhole
P150a. It covers the conv subsampling stack, 12 conformer layers, `output_proj` and `embed_audio`,
and produces the audio soft tokens that the text stream receives in place of `<|audio|>`
placeholders. Weights are bf16 (the checkpoint is bf16, so they are exact). Activations are bf16
between ops. Attention q/k/v, logits and softmax are fp32, as are the mel input, the depthwise-conv
accumulation and the tower output. Compute uses HiFi4 with fp32 accumulation, DRAM interleaved.
The reference is the unmodified HF modules in fp32 on CPU (transformers 5.20.2 overlay), computed
live from real speech. All numbers are measured on sjc-snva-tp100 on 2026-10-27 unless marked
otherwise.

## Architecture as implemented (checked against `gemma4/modeling_gemma4.py`, 5.20.2 overlay)

| item | HF (file:line) | TT |
|---|---|---|
| feature extractor (host) | `Gemma4AudioFeatureExtractor`: 16 kHz, 128 log-mel bins, frame 320 / hop 160 (100 frames/s), semicausal left pad, `input_features_mask` (True = valid); 30 s gives 2999 frames | used as is (host, it is the processor); `AudioEncoder.extract_features` wraps it |
| subsampling | `x * mask` -> `Conv2d(1->128, k3 s2 p1, no bias)` -> `LayerNorm(128, no bias)` over channels -> ReLU; `x * mask[::2]` -> `Conv2d(128->32)` -> LN(32) -> ReLU; flatten `(f, c)` -> `input_proj_linear 1024->1024` (`:363-418`) | `AudioSubsample`: both convs run on device with `ttnn.conv2d` (fp32 mel in, bf16 weights), `ttnn.layer_norm`, `relu`; both masks are multiplied on device; the NHWC conv output reshaped to `[B,1,S,1024]` is HF's flatten order |
| token mask | `mask[::2][::2]`; N = ceil(ceil(T/2)/2) tokens (`:388,417`) | `token_mask` on the host; the valid count per clip strips padded rows |
| rel-pos table | sinusoid `[sin, cos]` of distances 12..0, 1024 wide, timescales 1..1e4 (`:226-254`) | `rel_position_table` in fp32 at load time; `relative_k_proj` is folded with it into a per-layer constant |
| attention | q/k/v/post ClippableLinear; `q *= hd^-0.5/ln2 * softplus(per_dim_scale)`, `k *= ln(1+e)/ln2`; blocks of 12 queries x 24 keys; `ac + rel_shift(q relk)`; `tanh(l/50)*50`; masked fill -1e9; softmax fp32 (`:257-360`) | `AudioAttention`: an equivalent band on 32-query blocks x 64 keys (next section); per-channel q scale folded at load; fp32 q/k/v; Transformer-XL shift as a row-major view trick; softcap; additive -1e9 mask; manual fp32 softmax |
| mask | `create_bidirectional_mask` with `sliding_window_mask_function((12, 0))`: keep `0 <= q-kv < 12` and key valid; blocked to 5D (`:1874-1885, 1928-1966`) | host `attention_mask` `[B,1,S,64]` additive; `test_host_mask_matches_hf` checks the kept pairs against the HF 5D mask |
| FFN (x2, Macaron) | `x + 0.5 * post_norm(W2(silu(W1(pre_norm(x)))))`, 1024->4096->1024, clipped linears (`:421-453`) | `AudioFeedForward` |
| LightConv1d | pre RMSNorm -> `linear_start` 1024->2048 -> GLU -> causal depthwise conv k5 (left pad 4, no bias) -> `conv_norm` -> SiLU -> `linear_end` -> + residual (`:457-528`) | `AudioLightConv1d`: the causal shift is one exact 0/1 selection matmul per 32-row block (previous + own block), and the 5 taps accumulate in fp32 |
| layer | FFN1 -> `x + norm_post_attn(attn(norm_pre_attn(x)))` -> LConv -> FFN2 -> `norm_out` (`:531-579`) | `AudioConformerLayer`, same order |
| ClippableLinear | `clamp(in, in_min, in_max)` -> Linear (no bias) -> `clamp(out, out_min, out_max)`; 4 per-tensor scalars per linear, 84 linears (`:176-202`) | `ClippableLinear`: the checkpoint scalars are read once in `from_state_dict` and folded into `ttnn.clamp` as python floats |
| gradient clipping | `clamp(+-1e10)` before norms (`:442,449,517,559,566,574`) | not emitted. The largest \|activation\| HF produces at those sites is 301-340 (`audio_hf_max_abs_activation`, three cases), 7 orders of magnitude below the bound |
| output_proj | `Linear(1024->1536, bias)` (`:1978`) | `AudioOutputProj` (fp32 out) |
| embed_audio | RMSNorm(1536, no scale, eps 1e-6) -> `Linear 1536->512`, no bias (`modeling_embedding_gemma2.py:565-591, 856-878`) | `AudioEmbedder` (E1 `RMSNorm` with_scale=False) |

### Attention on 32-query blocks (why it equals HF's 12 x 24 blocking)

HF's blocked layout gives query `q = 12b + i` the keys `12b - 12 .. 12b + 11`. Its mask keeps
`0 <= q - kv < 12` and valid keys only. Since `q - 11 >= 12b - 12`, every kept key lies in the
window, so the computation is exactly the band `kv in [q-11, q]`, independent of the blocking. The
TT module uses blocks of 32 queries (tile aligned) with the previous 32 plus own 32 keys
(`block_context`), which also contains the band of every query. The rel-pos term is
`X = Q E^T` with `E` `[96, 128]` per head. Row `c = 64 - d` holds `relative_k_proj(sinusoid(d))`
for `d` in 0..12, and the other rows are zero. The term is then shifted with
`out[i, j] = X[i, 32 - i + j]` (row-major reshape / slice: the Transformer-XL trick that
`models/experimental/parakeet` uses, adapted). Column `j` of query `i` is key `32b - 32 + j`,
so `d = 32 + i - j`. For valid queries, the host mask keeps exactly HF's (query, key) pairs
(checked on short, b2, crop13 and crop25). Lengths not divisible by 12 and the first and last
chunk are covered by `test_attention_crop` (1, 11, 12, 13, 24, 25, 37 and 100 tokens) and by the
first / last 12-token chunk PCC in every attention test.

## Module contracts

`S` is the token bucket (32, 64, 128, 256, 512, 768; a clip runs in the smallest bucket >= its
tokens) and `T = 4S` the frame bucket. Rows past a clip's length are padding. Every module is
row-causal or row-wise, so padding never reaches valid rows: the band and the causal conv only
look back, FFN and norms work per row, and padding sits at the end.

| module (`tt/audio/`) | forward | in | out |
|---|---|---|---|
| `ClippableLinear` | `(x)` | `[B,1,S,in]` | `[B,1,S,out]` (`out_dtype`) |
| `AudioSubsample` | `(inputs)`; `conv_stack(inputs)`, `input_proj(flat)` | mel `[B,T,128,1]` fp32 RM, frame_mask `[B,1,T,1]` fp32, mask1 `[1,1,B*T/2*64,1]` bf16 | `[B,1,S,1024]` |
| `AudioFeedForward` | `(x)` | `[B,1,S,1024]` | `[B,1,S,1024]` |
| `AudioAttention` | `(x, mask)` | `[B,1,S,1024]`, mask `[B,1,S,64]` fp32 | `[B,1,S,1024]` |
| `AudioLightConv1d` | `(x)` | `[B,1,S,1024]` | `[B,1,S,1024]` |
| `AudioConformerLayer` | `(x, mask)` | as above | `[B,1,S,1024]` |
| `AudioOutputProj` | `(x)` | `[B,1,S,1024]` | `[B,1,S,1536]` fp32 |
| `AudioEmbedder` | `(hidden)` | `[B,1,S,1536]` fp32 | `[B,1,S,512]` bf16 |
| `AudioModel` | `(AudioDeviceInputs)`; `prepare(inputs)`; `forward_hidden(...)` | 4 device tensors | `(soft [B,1,S,512] bf16, hidden [B,1,S,1536] fp32)` |
| `AudioEncoder` (host) | `encode(features, mask, traced=)`; `extract_features(waveforms)`; `capture_all(shapes)` | processor outputs | list of `[n_b, 512]` per clip |

`AudioDeviceInputs` holds `mel`, `frame_mask`, `mask1` and `attn_mask`. `AudioEncoder.prepare_inputs`
builds all four on the host. They are uploaded before the forward and refreshed with
`copy_host_to_device_tensor` before a trace replay. `ttnn.conv2d` lays its weights out for one
input geometry, so `AudioModel.prepare` lays out both conv weights once per (batch, bucket), at
setup. The forward raises for an unprepared shape, and `capture_all` prepares every shape before
the first capture. Every `from_state_dict` checks tensor shapes and writes the ttnn weight cache
under `~/.cache/ttnn/models/embedding_gemma2/<snapshot>/audio.*`. The forward signatures are
keyword-friendly. `AudioModel.from_state_dict(..., act_dtype=ttnn.float32)` selects the
fp32-activation variant (see Precision).

## Inputs used (real speech from the tt-metal tree)

The 16 kHz mono speech clips of the Whisper demo
(`models/demos/audio/whisper/demo/dataset/conditional_generation/*.wav`, 4.0-7.6 s each, sorted
by name). No in-tree clip is 10 s or longer, so the long cases concatenate whole clips. The host
has no other speech or sound files outside build caches.

| case | audio | frames | tokens |
|---|---|---|---|
| short | clip 1, 0.50-1.14 s (0.64 s) | 63 | 16 |
| s2 | clip 2, 1.00-3.00 s | 199 | 50 |
| s10 | clips 1-2 concatenated, cut at 10.00 s | 999 | 250 |
| s30 | clips 1-5 concatenated, cut at 30.00 s (the maximum) | 2999 | 750 |
| b2_s10_s2 | s10 and s2 in one padded batch | 999 (199 valid in row 2) | 250, 50 |
| crop1 ... crop100 | first 4N frames of s30's features | 4N | N = 1, 11, 12, 13, 24, 25, 37, 100 |
| tone3 | **synthetic** 3 s, 440 Hz tone + 200-4000 Hz chirp (extra case) | 299 | 75 |

## PCC vs HF (bar 0.995 per layer kind; soft tokens >= 0.99 per clip)

Source: `pytest models/demos/blackhole/embedding_gemma2/tests/pcc/audio -k "not tracy"` (114
passed; metrics in `tt-ops/runs/embeddinggemma-2/e3/full_run1.jsonl`). Each module gets the HF
input of that module (hooks on the real HF modules). PCC is over valid tokens only.

| layer kind | tests | min PCC | worst case |
|---|---|---|---|
| conv stack (2 on-device convs + LN + ReLU + masks, flattened) | 8 | 0.999700 | crop13 |
| input_proj (teacher forced) | 3 | 0.999998 | |
| subsampling total (mel -> layer-0 input) | 8 | 0.999844 | crop13 |
| FFN1 / FFN2 (layers 0, 5, 11) | 18 | 0.999996 | |
| attention (layers 0, 5, 11; 3 cases) + crops 1-100 tokens (layer 0) | 17 | 0.999998 | b2, layer 11 |
| attention, first 12-token chunk / last chunk only | 17 / 17 | 0.999998 / 0.999998 | |
| LightConv1d (layers 0, 5, 11) | 9 | 0.999996 | |
| conformer layer 0, every case | 8 | 0.999986 | |
| all 12 layers, teacher forced (short, s30, b2) | 36 | 0.999986 | short, layer 2 |
| output_proj | 3 | 0.999999 | |
| embed_audio | 3 | 0.999999 | |
| host band mask == HF 5D mask (kept pairs, valid queries) | 4 | exact | |
| standalone reference vs `EmbeddingGemma2Model.get_audio_features` (valid rows) | 2 | max abs diff 0.0 | |

End to end, free running (`AudioEncoder.encode`, eager; processor output -> soft tokens), vs HF
fp32. The HF-bf16 columns are HF itself run in bf16 against the same fp32 reference:

| case | tower-out PCC | soft-token PCC | per-token cosine min (token) / mean | HF bf16: soft PCC / cosine min (token) |
|---|---|---|---|---|
| short (16) | 0.99842 | **0.99853** | 0.98157 (8) / 0.99846 | 0.99847 / 0.97941 (8) |
| s2 (50) | 0.99988 | **0.99989** | 0.99939 (10) / 0.99990 | 0.99992 / 0.99977 (39) |
| s10 (250) | 0.99826 | **0.99841** | 0.76366 (126) / 0.99847 | 0.99919 / 0.85131 (126) |
| s30 (750) | 0.99884 | **0.99905** | 0.77286 (267) / 0.99906 | 0.99924 / 0.81964 (267) |
| b2: s10 / s2 | 0.99826 / 0.99988 | **0.99841 / 0.99989** | same as single | same as single |
| tone3 (synthetic) | 0.99848 | **0.99806** | 0.97527 (43) / 0.99804 | 0.99844 / 0.98052 (32) |

Every clip passes the 0.99 bar. Per-layer PCC is at least 0.99970. Two comparisons with HF-bf16
are worth stating exactly:

- Soft-token PCC: TT is above HF-bf16 on short, and below it on s2, s10, s30 and tone3 (by up to
  0.00078, on s10).
- Worst per-token cosine: TT is above HF-bf16 on short, and below it on s2, s10, s30 and tone3.
  The two low tokens are the same tokens in both runs (s10 token 126, s30 token 267). TT's
  cosines there are 0.764 and 0.773, against 0.851 and 0.820 for HF-bf16.

### Precision lever (measured, not the default)

`test_soft_tokens_fp32_activations` reruns the end-to-end check with the conformer activations in
fp32 (`act_dtype=ttnn.float32`: input_proj, every clipped-linear output, residual adds and
norms; weights still bf16). Results (same jsonl, `act: fp32`):

| case | soft PCC bf16 -> fp32 act | cosine min bf16 -> fp32 act |
|---|---|---|
| short | 0.99853 -> 0.99987 | 0.98157 -> 0.99964 |
| s2 | 0.99989 -> 0.99992 | 0.99939 -> 0.99925 |
| s10 | 0.99841 -> 0.99919 | 0.76366 -> 0.85323 |
| s30 | 0.99905 -> 0.99917 | 0.77286 -> 0.75305 |
| tone3 | 0.99806 -> 0.99976 | 0.97527 -> 0.99564 |

fp32 activations raise soft PCC on every case. They do not fix the worst token of s30 (0.753),
which HF-bf16 also gets wrong (0.820). [measured, E6b] That token and s10 token 126 are
ill-conditioned in the model itself: 0.5-ulp bf16 perturbations of the HF fp32 reference's input
features move them by up to 0.16 (1 - cos) against about 5e-6 for the median token, the growth is
in conformer layers 10-11, and teacher-forced TT layers are as accurate on them as on any token
(`doc/datatype_sweep/README.md`, "Audio worst-token sensitivity"). The fp32-activation
cost is +16 % at 10 s and +20 % at 30 s of device time (Performance). bf16 stays the default,
as in E1/E2. The choice belongs to the E6 datatype sweep.

## Trace, determinism, fallback audit, watcher

| check | command | result |
|---|---|---|
| trace replay == eager | `test_audio_trace.py::test_trace_matches_eager`: traces (1,32), (1,64), (1,128), (1,256), (1,768), (2,256) captured once by `capture_all` | bit-identical soft tokens (max abs diff 0.0) on 8 cases / 9 clips; `short`, `crop13` and `crop25` replay the same (1,32) trace |
| trace allocation safety | full audio suite with `TT_METAL_TRACE_ALLOC_TRACKING=1` | 114 passed, no "still alive" report (`e3/logs/full_run1.log`) |
| determinism | `test_audio_trace.py::test_audio_determinism` (s10, b2) | 5 replays + 2 eager runs bit-identical |
| fallback audit | `test_audio_fallback_audit.py` (b2) | 0 torch calls in `AudioModel.forward` (TorchFunctionMode), no ttnn host-transfer entry point called, `throw_exception_on_fallback=True` did not raise; Tracy report of one forward: **1357 device ops, 0 host ops** |
| watcher | `TT_METAL_WATCHER=10 pytest tests/pcc/audio -k "not perf and not reference_matches"` (trace tests included) | 101 passed (189 s); `watcher.log` (1003 lines) has no line matching `error|assert|exception|overflow|sanitiz|fault` |

Static audit: in `tt/audio/`, `torch` is used only in setup (`from_state_dict`, the
`rel_key_matrix` / `causal_selection` constants, host conv weights for `prepare`) and on the host
side of `AudioEncoder` (`prepare_inputs`, `read_rows`, `_dummy_host`, `extract_features`) plus
`attention_mask`, which `prepare_inputs` calls. `AudioModel.prepare` (conv weight layout) is setup:
it runs once per shape, before any trace capture, and is not part of the measured forward.

## Performance (functional, unoptimized; traced, warmed, 20 replays)

`pytest models/demos/blackhole/embedding_gemma2/tests/pcc/audio/test_audio_perf.py -k traced -s`
(`device` = back-to-back `execute_trace` + one sync; `e2e` = input copy + replay + soft-token
readback; host prep = `prepare_inputs`, not part of e2e; log-mel extraction is not timed).

| batch x bucket | input | device ms | e2e ms | clips/s (device) | host prep ms |
|---|---|---|---|---|---|
| 1 x 256 | 10 s (s10) | **23.67** | 27.20 | 42.2 | 2.8 |
| 1 x 768 | 30 s (s30) | **47.79** | 57.71 | 20.9 | 7.1 |
| 4 x 256 | 4 x 10 s | **80.48** | 93.58 | 49.7 | 9.0 |
| 1 x 256, fp32 act | 10 s | 27.48 | 31.57 | 36.4 | 2.8 |
| 1 x 768, fp32 act | 30 s | 57.27 | 67.17 | 17.5 | 7.0 |

tt-perf-report for one warmed eager forward, b1 x 768 (s30), signposts `PERF_AUDIO` /
`PERF_AUDIO_END` (`tracy/b1_s768/forward_perf_report.{txt,csv}`, stacked
`forward_perf_report_stacked.csv`; `Device Time` in microseconds): **1357 device ops, 0 host
ops, 47,482 us device time + 746 us op-to-op gaps** (matches the 47.8 ms replay). This is the
representative report. The batch-4 and fp32 rows have traced device times only.

| op | device us | share | count |
|---|---|---|---|
| Matmul (FFN 4 per layer, q/k/v/post, lconv start/end, rel-pos, QK, PV, causal selection, input/output proj, embed) | 19,257 | 40.6 % | 171 |
| eltwise unary (clamps, silu, sigmoid, tanh, exp, reciprocal, relu) | 6,076 | 12.8 % | 326 |
| eltwise binary (residual adds, scales, masks, conv taps) | 5,528 | 11.6 % | 326 |
| reshape (codegen + view) | 6,119 | 12.9 % | 75 |
| permute / transpose (head split / merge) | 2,723 | 5.7 % | 48 |
| LayerNorm op (RMSNorm x 9 per layer, subsampling LN) | 2,307 | 4.9 % | 111 |
| slice / concat (block context, GLU, rel shift) | 3,948 | 8.3 % | 240 |
| conv2d + halo + resharding (subsampling) | 915 | 1.9 % | 10 |
| reduce (softmax max / sum), tilize / untilize (rel shift) | 608 | 1.3 % | 50 |

E5 levers [guess]: about 27 % of the time is reshape / permute / slice / concat, mostly around
the attention: the head split by reshape + permute, the block context concat, the row-major rel
shift. A fused QKV with an `nlp_create_qkv_heads`-style head split, a tile-aligned rel shift,
fused GLU / silu and sharded matmuls are the obvious candidates. In layer 0 the q/k/v clip bounds
are identical (input -20.375 / 20.25, output -34.5 / 34.25; other layers not checked), so a fused
QKV may keep one input clamp per layer.

## Context contract (audio)

`doc/context_contract.json` -> `audio`. The feature extractor caps a clip at 2999 frames: the
recon measured 60 s -> 750 tokens, and here 30.00 s gives 2999 frames and 750 tokens.
`TOKEN_BUCKETS` reaches 768, so the full length is supported and tested (s30, 750 tokens), with no
reduction. The recon judged `processor_config.audio_seq_length = 280` to be unused as a cap
[guess, not rechecked here]. Audio past 30 s is cut by the processor, as in HF.

## Out of scope (decoder-only items of the bring-up skill)

The audio tower is an encoder that runs once per clip. It has no KV cache, decode step, page
table, current-position tensor, sampling or chunked prefill, so none is built. Its attention is
causal-local, but the whole clip is available up front. Streaming (HF's commented-out
`padding_cache`) is not part of this model's embedding use. The "context length" axis is the
clip duration, bounded by the feature extractor (see above).

## Limitations

- Precision: the bf16 activation path is 0.0008 soft PCC below HF-bf16 on s10. See Precision for
  the fp32 lever.
- `AudioModel.prepare` must run per (batch, bucket) before the forward; `capture_all` does it.
- Buckets are fixed (32 ... 768 tokens). A 1-token clip runs 32 rows, and a 257-token clip runs
  512.
- Batch > 1 was tested up to 4 (perf) and 2 (PCC). Device memory was not probed beyond that.
- The multimodal merge (`masked_scatter` into the text embeddings) is E4.

## Artifacts

- `tt-ops/runs/embeddinggemma-2/e3/logs/`: `pcc_run2.log`, `trace_run1.log`, `tracy_run2.log`,
  `watcher_run1.log`, `full_run1.log`, `final_run.log`, `exp_fp32_act.log`
- `tt-ops/runs/embeddinggemma-2/e3/`: `pcc_run2.jsonl`, `trace_run1.jsonl`, `full_run1.jsonl`,
  `watcher_results.jsonl`, `final_results.jsonl`, `watcher/generated/watcher/watcher.log`
- `doc/functional_audio/tracy/b1_s768/`: tt-perf-report table, CSV and stacked CSV. The raw ops
  CSV stays at `generated/profiler/reports/2026_10_27_13_48_12/` (not committed).

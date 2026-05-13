# Converting bit_sculpt HF debug-traces to harness format

End-to-end recipe for turning a [bit_sculpt](https://github.com/tenstorrent/bit_sculpt) "Group A" debug-trace dump of a HuggingFace DeepSeek V3 run into the on-disk format consumed by `run_host_io_decoder_sweep.py`. Covers both halves of the cross-trace contract — the **hidden-state trace** (input to / reference output of one decoder layer) and the **KV-cache reference** (the cumulative MLA `kv_post_transform` for that layer).

Tooling: `convert_bit_sculpt_trace.py` (in this directory). Final consumer: `run_host_io_decoder_sweep.py` with `--validate-hidden-states-cross-trace` and/or `--validate-kv-cache-cross-trace` enabled.

---

## 1. Source format: bit_sculpt's on-disk layout

A bit_sculpt trace dir (e.g. `<bit_sculpt>/results/deepseek-r1-0528/debug_trace/cache-design-gen8192/`) contains:

### 1.1 Per-step decoder I/O (in tar.zst archives)

The raw `step_*/` directories are gitignored on the bit_sculpt side; committed copies live in **two zstd-compressed tarballs**, split by layer range:

| archive | layers | size |
|---|---|---|
| `artifacts-0-29.tar.zst` | 0–29 | ~2.3 GB |
| `artifacts-30-60.tar.zst` | 30–60 | ~2.4 GB |

Each tarball, when extracted, produces `step_{0..S}/` directories. For the cache-design-gen8192 trace specifically `S = 8192` (1 prefill step + 8192 decode steps = 8193 directories). Each step directory contains:

```
step_{s}/
    decoder_input_layer_0.safetensors                  # REAL FILE (only layer 0)
    decoder_input_layer_{1..N}.safetensors             # SYMLINK -> decoder_output_layer_{i-1}.safetensors
    decoder_output_layer_{0..N}.safetensors            # REAL FILE
```

Per-tensor layout:

- Tensor key inside each `.safetensors`: `decoder_input_layer_{i}` or `decoder_output_layer_{i}`.
- Shape: `(T_new, 7168)` bf16.
  - For `step_0` (prefill): `T_new = num_prefill_tokens` (= 65 for the cache-design-gen8192 prompt).
  - For `step_{1..S}` (decode): `T_new = 1`.

**Symlink gotcha.** `step_{s}/decoder_input_layer_{i}.safetensors` for `i >= 1` is a relative symlink to `step_{s}/decoder_output_layer_{i-1}.safetensors`. When you load a symlinked file with `safetensors.torch.load_file`, the returned dict's key reflects the **target file** (`decoder_output_layer_{i-1}`), not the symlink path. **Never look up by hard-coded key name** — always take the only-value of the dict.

There's also a per-tarball split caveat: `step_*/decoder_input_layer_30.safetensors` is in the 30–60 tarball but points to `decoder_output_layer_29.safetensors` which is in the 0–29 tarball. Extracting only one tarball leaves that single symlink dangling — extract both to be safe.

### 1.2 KV cache (cumulative, one file per layer at the trace root)

```
kv_cache_layer_{i}.safetensors                          # 61 files, layers 0..60
```

Each contains a single tensor:

- Key: `kv_post_transform_layer_{i}`
- Shape: `(T_total, 576)` bf16, where `T_total = num_prefill_tokens + num_decode_steps`.
- Row layout: rows `[0:num_prefill_tokens]` are the prefill cache, rows `[num_prefill_tokens : num_prefill_tokens+s+1]` are decode step `s`'s entries. Purely additive; rows never change once written.

Channel layout (MLA's compressed cache):

| channels | content | dim |
|---|---|---|
| `[:, :512]` | `kv_latent_normed` (post-RMSNorm shared compressed latent) | 512 |
| `[:, 512:]` | `k_pe_roped` (RoPE'd shared positional key) | 64 |

Total `576 = 512 + 64`. The 64-wide `k_pe_roped` is stored in **HF / split-halves / GPT-NeoX / LLaMA RoPE convention** — channels `[0:32]` are all 32 "real" parts contiguously, channels `[32:64]` are all 32 "imag" parts contiguously. This will matter later.

### 1.3 Metadata

`metadata.json` lives inside `artifacts-0-29.tar.zst` (and is also written to the trace root after extraction). Key fields used by the converter:

| field | value for cache-design-gen8192 | meaning |
|---|---|---|
| `n_tokens` | 65 | prefill token count |
| `decode_steps` | 8192 | number of decode steps captured |
| `n_layers` | 61 | total layers |
| `moe_layer_offset` | 3 | layers `[0, 3)` dense, `[3, 61)` MoE |
| `hidden_dim` | 7168 | matches `D.HIDDEN_SIZE` |
| `kv_lora_rank` | 512 | MLA compressed-latent dim |
| `qk_rope_head_dim` | 64 | k_pe channel count |
| `save_dtype` | `torch.bfloat16` | everything is bf16 on disk |

---

## 2. Target format: what `run_host_io_decoder_sweep.py` consumes

The harness loads per-prompt files from `--hidden-states-dir`. For each prompt name `<NAME>` passed via `--prompt`, the harness expects:

### 2.1 Hidden-state trace (always required)

```
<dir>/<NAME>.pt
```

`torch.save`d dict:

```python
{
    "input":  torch.Tensor,    # shape (L, 7168), dtype bfloat16
    "output": torch.Tensor,    # shape (L, 7168), dtype bfloat16
}
```

`L` is the total per-prompt sequence length the harness will sweep through. For the cache-design-gen8192 trace, `L = 65 + 8192 = 8257`. `trace["input"][p]` is the hidden state fed to the chosen decoder layer at sweep iteration `p`; `trace["output"]` is the per-position reference used by Phase 3b's cross-trace PCC.

### 2.2 KV-cache reference (required iff `--validate-kv-cache-cross-trace`)

```
<dir>/kv_cache_reference_<NAME>.pt
```

Slot-agnostic `torch.save`d tensor:

```python
ref.shape == (1, L, 576)
ref.dtype == torch.bfloat16
```

The leading `1` is the head dim (matching the per-(slot, prompt) slice the harness extracts from the on-device KV cache). The reference is stored in **HF / split-halves RoPE layout** — same as bit_sculpt's source format — because the harness applies the necessary permutation **in-memory at compare time** rather than baking it into the on-disk reference.

---

## 3. Step-by-step conversion

### Step 0. Extract the bit_sculpt tarballs

```bash
cd <bit_sculpt>/results/deepseek-r1-0528/debug_trace/cache-design-gen8192/
tar --zstd -xf artifacts-0-29.tar.zst
tar --zstd -xf artifacts-30-60.tar.zst
```

Expects ~8.5 GB of disk and 30–40 minutes total (~1 M small files; filesystem syscalls dominate). If you only need one layer, extract just the tarball that covers it (`0-29` for layers 0–29; `30-60` for layers 30–60).

### Step 1. Run the converter

```bash
cd <tt-metal>
python -m models.demos.deepseek_v3_b1.tests.unit_tests.convert_bit_sculpt_trace \
    --bit-sculpt-dir <bit_sculpt>/results/deepseek-r1-0528/debug_trace/cache-design-gen8192 \
    --layer-idx 4 \
    --prompt-name blitz_test \
    --out-dir /data/asaigal/pipeclean_traces/blitz_test
```

Mandatory flags:

| flag | meaning |
|---|---|
| `--bit-sculpt-dir` | the extracted trace dir (must contain `kv_cache_layer_*.safetensors` at root + `step_*/` subdirs) |
| `--layer-idx` | which decoder layer to extract (`0..60` for DeepSeek V3). The harness's `--decoder-layer-idx` must match exactly when you later run the sweep. |
| `--prompt-name` | output filename stem; this is what you pass to `--prompt` later |
| `--out-dir` | output directory (created if missing) |

Optional flags:

| flag | default | meaning |
|---|---|---|
| `--num-prefill-tokens` | 65 | number of token rows in `step_0` |
| `--num-decode-steps` | 8192 | number of `step_{1..S}` directories |
| `--no-kv-cache` | off | skip the KV-cache reference, emit only the hidden-state trace |

Internally the converter does:

1. For each step `s` in `[0, num_decode_steps]`:
   - load `step_{s}/decoder_input_layer_{layer_idx}.safetensors` (extracting the only tensor in the dict — symlink-safe)
   - load `step_{s}/decoder_output_layer_{layer_idx}.safetensors`
   - validate shape == `(num_prefill_tokens if s==0 else 1, 7168)` bf16
2. Concatenate inputs along dim 0 → `(num_prefill_tokens + num_decode_steps, 7168)` bf16. Same for outputs.
3. `torch.save({"input": …, "output": …}, "<out>/<prompt>.pt")`.
4. If KV cache is requested:
   - Load `kv_cache_layer_{layer_idx}.safetensors[kv_post_transform_layer_{layer_idx}]`.
   - Validate shape == `(num_prefill_tokens + num_decode_steps, 576)` bf16.
   - Reshape `(L, 576) -> (1, L, 576)` (add the head dim).
   - `torch.save(kv_ref, "<out>/kv_cache_reference_<prompt>.pt")`.

For the cache-design-gen8192 dataset, this finishes in ~25 s and produces:

```
<out_dir>/
    <prompt>.pt                            # ~236 MB, dict of two (8257, 7168) bf16 tensors
    kv_cache_reference_<prompt>.pt         #  ~9.5 MB, (1, 8257, 576) bf16 tensor
```

### Step 2. Run the sweep

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 \
python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \
    --decoder-layer-idx 4 \
    --hidden-states-dir /data/asaigal/pipeclean_traces/blitz_test \
    --prompt blitz_test \
    --num-replication-slots 1 \
    --validate-hidden-states-cross-trace \
    --validate-kv-cache-cross-trace \
    --pcc-threshold 0.97 --kv-cache-pcc-threshold 0.97
```

Critical: **`--decoder-layer-idx` MUST match the converter's `--layer-idx`.** Nothing on disk distinguishes a layer-4 trace from a layer-8 trace by filename; if they diverge, the cross-trace PCC silently degrades to a misleading ~0.92 (the layer outputs are correlated through the residual stream but not aligned).

If you want multiple layers, run the converter once per layer with a different `--prompt-name` (e.g. `blitz_test_layer4`, `blitz_test_layer8`) and pick the right one per sweep invocation.

---

## 4. The RoPE storage-convention subtlety

The single non-obvious transformation in this pipeline is **not done by the converter** — it's done in-memory by the harness during Phase 4b. Documented here so future contributors understand the on-disk contract.

### 4.1 What's stored

- **Bit_sculpt reference** (split-halves layout, HF/LLaMA/GPT-NeoX convention):
  ```
  channel idx:   0   1   ...  30  31  32  33  ...  62  63
  value:        r0  r1  ... r30 r31  i0  i1  ... i30 i31
  ```
  This is what HuggingFace's `rotate_half(x) = cat([-x[d/2:], x[:d/2]])` produces and consumes; it's the canonical form for LLaMA-family models. The converter writes the reference in this form, unchanged from bit_sculpt's source.

- **TT on-device KV cache** (interleaved / GPT-J / RoFormer convention):
  ```
  channel idx:   0   1   2   3   ...   60   61   62   63
  value:        r0  i0  r1  i1   ...  r30  i30  r31  i31
  ```
  TT's MLA RoPE kernel computes `output = x*cos + (x @ trans_mat)*sin` where `trans_mat` is a 32×32 sparse permutation matrix that maps each adjacent `(r_k, i_k)` pair to `(-i_k, r_k)`. This requires the interleaved layout. See `unified_kernels/rope.hpp` and `get_rot_transformation_mat()` for the definitive code.

Both layouts encode the same 32 rotated complex numbers; the difference is purely physical placement of bytes, and they're bijections of each other via a fixed permutation.

### 4.2 Where the permutation happens

Inside `host_io_decoder_harness.py`, helper `_split_halves_to_interleaved_kpe(kv)`:

```python
out = kv.clone()
kpe = kv[..., 512:]                              # (..., 64) split-halves
out[..., 512::2] = kpe[..., :32]                 # reals  -> even output positions
out[..., 513::2] = kpe[..., 32:]                 # imags  -> odd output positions
return out
```

This is invoked **once per prompt** inside Phase 4b, applied to the loaded reference, before PCC. The first 512 (`kv_latent_normed`) channels are layout-agnostic and pass through unchanged.

### 4.3 Why the on-disk reference is *not* pre-permuted

Two reasons:

1. The HF/split-halves form is the canonical, well-documented layout. Anyone re-using the reference with another tool (vLLM diff, hand-inspection, etc.) gets the standard form rather than a TT-specific one.
2. If the on-disk reference were pre-permuted, the converter would silently bake a TT-internal convention into a notionally GPU-side artifact, making the file uninspectable without out-of-band knowledge of the transformation.

If you ever produce a reference KV cache from a different source whose RoPE layout is already interleaved, you'll need to either pre-permute it back to split-halves before saving, or extend the harness to skip the permutation conditionally. There's no flag for this today.

---

## 5. Validation, diagnostics, and gotchas

### 5.1 What success looks like

For a correctly converted layer-4 reference on bf16-native hardware, expect:

- Hidden-state cross-trace PCC: `~0.99x` (Phase 3b).
- KV-cache cross-trace PCC: `~0.99x` overall (Phase 4b).
  - kv_latent_normed sub-PCC: `~0.998`.
  - k_pe_roped sub-PCC (after the in-memory permutation): `~0.999`.

PCC values noticeably below `0.99` for *just* k_pe_roped while kv_latent_normed remains high means **the RoPE permutation in the harness is wrong for your data** (e.g., your reference is already in interleaved form, or in some third convention). The harness's Phase 4b logs the sub-channel split automatically on failure for exactly this debugging path.

### 5.2 The standalone diagnostic

For finer-grained mismatch localization without re-running the device sweep, use the sibling tool:

```bash
python -m models.demos.deepseek_v3_b1.tests.unit_tests.diagnose_kv_cache_mismatch \
    --reference <out_dir>/kv_cache_reference_<prompt>.pt \
    --dump      ./dumps/kv_cache_slot_00_<prompt>.pt
```

It runs four stages:

1. baseline per-channel PCC,
2. interleaved ↔ split-halves permutation hypotheses on both sides (identifies RoPE-convention mismatches),
3. per-position PCC across a spread of positions (distinguishes layout issues from per-position RoPE-frequency / position-offset issues),
4. raw position-0 values from both sides for eyeballing.

This is what we used to discover the RoPE convention difference in the first place.

### 5.3 Gotchas, in order of how badly they will bite you

1. **Layer-idx mismatch between convert and sweep.** No on-disk metadata records which layer a `.pt` was built for. Symptom: low (~0.92) but non-zero cross-trace PCC. Fix: stay disciplined with `--prompt-name` (e.g. encode the layer in the stem).
2. **Symlink-driven safetensors key collision.** `step_{s}/decoder_input_layer_{i+1}.safetensors` is a symlink to `…_output_layer_{i}.safetensors`; the loaded dict's key is `decoder_output_layer_{i}`, not `decoder_input_layer_{i+1}`. Code that hard-codes the key by filename will silently load the wrong field. The converter avoids this by taking `next(iter(d.values()))` after validating `len(d) == 1`.
3. **Cross-tarball symlink dangle.** Extract both tarballs even if you only care about layers 0–29 — `step_*/decoder_input_layer_30.safetensors` (in the 30–60 tarball) symlinks across into the 0–29 tarball, and a few other cross-range edges exist.
4. **`get_kv_cache_host` returns fp32, not bf16.** The numpy-host path doesn't natively support bf16, so the host tensor lands as fp32. Comparisons are dtype-agnostic (both `comp_pcc` and `torch.equal` work), but dumps you save with `torch.save` will inherit that fp32 dtype and a backing-storage view that bloats the file (`(64, 1, 131072, 576) × 4 bytes ≈ 19 GB` instead of the 9.5 MB the per-prompt slice would suggest). Fix on the roadmap: `.clone().to(bfloat16)` the slice in `run_sweep`'s Phase 5 dump path.
5. **bit_sculpt is greedy 8192-decode-only.** The trace was generated with greedy decoding (argmax, no sampling, no repetition penalty). The model drifts off-topic by the end; that's a known property of the trace, not a bug, and doesn't affect cross-trace PCC since both sides see the same token stream.

---

## 6. End-to-end summary

```bash
# 0. extract once (slow: ~35 min, ~8.5 GB)
cd <bit_sculpt>/results/deepseek-r1-0528/debug_trace/cache-design-gen8192/
tar --zstd -xf artifacts-0-29.tar.zst
tar --zstd -xf artifacts-30-60.tar.zst

# 1. convert one layer (fast: ~25 s)
cd <tt-metal>
python -m models.demos.deepseek_v3_b1.tests.unit_tests.convert_bit_sculpt_trace \
    --bit-sculpt-dir <bit_sculpt>/results/deepseek-r1-0528/debug_trace/cache-design-gen8192 \
    --layer-idx 4 \
    --prompt-name blitz_test_layer4 \
    --out-dir /data/asaigal/pipeclean_traces/blitz_test_layer4

# 2. run sweep with cross-trace validation
TT_METAL_SLOW_DISPATCH_MODE=1 \
python -m models.demos.deepseek_v3_b1.tests.unit_tests.run_host_io_decoder_sweep \
    --decoder-layer-idx 4 \
    --hidden-states-dir /data/asaigal/pipeclean_traces/blitz_test_layer4 \
    --prompt blitz_test_layer4 \
    --num-replication-slots 1 \
    --validate-hidden-states-cross-trace \
    --validate-kv-cache-cross-trace \
    --pcc-threshold 0.97 --kv-cache-pcc-threshold 0.97

# 3. (optional) diagnose mismatches without re-running the device sweep
python -m models.demos.deepseek_v3_b1.tests.unit_tests.diagnose_kv_cache_mismatch \
    --reference /data/asaigal/pipeclean_traces/blitz_test_layer4/kv_cache_reference_blitz_test_layer4.pt \
    --dump      ./dumps/kv_cache_slot_00_blitz_test_layer4.pt
```

To convert additional layers, re-run step 1 with different `--layer-idx` / `--prompt-name`; step 0 is amortized.

# Voxtral-TTS on TTNN — status and resumption notes

**Read this first when picking the work back up.** It is written to be self-contained: state,
measurements, the traps that cost time, and what to do next. Architecture detail and the
reference-side findings live in `reference/PROVENANCE.md`; this file is the *work* state.

Branch: `lserbedzija/voxtral-tts-ttnn` (pushed). All work is under
`models/experimental/voxtral_tts/`. Nothing else in the repo is touched.

---

## 1. Where things stand

| Piece | State |
|---|---|
| CPU reference, 3 blocks + tokenizer + end-to-end pipeline | **done**, 30/30 vs upstream |
| Block 3 — codec decoder on TTNN | **CLOSED**, 242x real-time, see §4 |
| Block 1 — 3.4B AR backbone on TTNN | **not started** ← the next piece, see §7 |
| Block 2 — flow-matching transformer on TTNN | **not started** |
| Codec **encoder** | **impossible** — weights absent from the public release |

The model is 4 networks but only 3 are portable. 118 tests pass, 96 of them with no device
(the reference suite runs weight-free off a vendored tensor manifest).

**Block 3 is closed, not merely paused.** Six independent optimization attempts were measured and
all rejected (§4). The single remaining known win is worth ~0.1% end-to-end. Do not reopen it
without a new idea; read §4 and §5 first, because the obvious ideas are already disproven.

---

## 2. How to actually run things (do this before anything else)

There are **two** virtualenvs and neither is obvious.

**Main env — has ttnn, torch, torchaudio, transformers:**
```bash
source /localdev/lserbedzija/repos/xtts_ref_venv/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD      # from the repo root
```
This is the venv created for the XTTS-v2 work; the repo's own `python_env` is a docs env and does
**not** have torch/ttnn. `python_env` *does* have `black` if you need it (but see trap #8).

**Run the tests** (the repo `conftest.py` needs torchvision, which is absent, hence `--noconftest`):
```bash
python -m pytest models/experimental/voxtral_tts/tests/test_common_ref.py \
  models/experimental/voxtral_tts/tests/test_tokenizer_ref.py \
  models/experimental/voxtral_tts/tests/test_backbone_pcc.py \
  models/experimental/voxtral_tts/tests/test_flow_pcc.py \
  models/experimental/voxtral_tts/tests/test_codec_pcc.py \
  models/experimental/voxtral_tts/tests/test_codec_ttnn_pcc.py \
  --noconftest -p no:cacheprovider -q
# expect: 118 passed
```
Passing the directory instead of the file list collects nothing — pass the files.

**Opening a device for anything with convs** needs `l1_small_size`, or you get
`Out of Memory: ... bank size is 0 B`:
```python
dev = ttnn.open_device(device_id=0, l1_small_size=65536)   # as tests/test_codec_ttnn_pcc.py:31
```
Attention-only harnesses do not need it, which makes this an easy trap in a fresh script.

**Checkpoint** (8.0 GB, gitignored, CC BY-NC 4.0):
`reference/weights/` — `consolidated.safetensors`, `params.json`, `tekken.json`,
`voice_embedding/*.pt` (20 presets). Re-fetch per `reference/PROVENANCE.md`. Without it, 96
reference tests still pass; the 22 device/weight tests skip.

**Second env — upstream comparison only** (needs einops + mistral-inference, no GPU). Recipe and
rationale in `scripts/upstream_compare/README.md`. It lived in a session scratchpad and is **gone**;
recreate from the README. Also run `scripts/upstream_compare/fetch_upstream.py` to re-download the
two pinned vLLM-Omni files (gitignored on purpose).

**End-to-end reference speech** (~50 s of CPU for the reference backbone):
```bash
python models/experimental/voxtral_tts/reference/voxtral_pipeline_ref.py \
    --text "..." --voice neutral_male --threads $(nproc)
```

---

## 3. What is validated, and how

**Reference vs upstream: 30/30.** Harness in `scripts/upstream_compare/`.
- Block 1 vs `mistral_inference` (Mistral's own): RoPE table + application **bit-exact**,
  RMSNorm/SwiGLU/repeat_kv bit-exact, full 26-layer stack PCC 0.99999988.
- Block 1 input side + Blocks 2/3 vs vLLM-Omni's `nn.Module`s: 37-code frame **bit-identical
  integers**, waveform PCC 0.99999982, all 8 decoder stages >= 0.99999.
- Tokenizer vs `mistral_common`: **exact token ids**, 15 prompts, 8 languages. Ground truth
  vendored at `tests/prompt_fixture.json`.

**End-to-end reference:** 0.0% WER (Whisper) on 4 runs — 24 words x 2 voices, a **125-word
paragraph in a single pass** (469 frames, 37.5 s), and French via `fr_female`.

**Block 3 on device:** PCC 0.9994–0.9998 vs the reference across T=1…1536, plus a real-speech
fixture (`tests/real_frames_fixture.pt`, 6.4 KB of genuine Block 1+2 output) that asserts **both**
PCC > 0.9999 **and** worst sample < 2% of peak. Default config measures PCC 0.999984 / worst 1.16%.
That worst-sample bound is the gate that matters — see trap #9.

---

## 4. Block 3: performance, and everything that was tried

Warm, N150, defaults:

| T | audio | warm | RTF | vs real-time |
|---|---|---|---|---|
| 64 | 5.1 s | 43.9 ms | 0.0086 | 117x |
| 469 | 37.5 s | 155.5 ms | 0.0042 | 241x |
| 1500 | 120 s | 539.4 ms | 0.0045 | 222x |

Upstream report RTF **0.103 for their whole pipeline** on an H200, so Block 3 is ~4–8% of the total
budget. **It is not where the end-to-end answer gets decided** — Block 1 is (87% of the parameters,
12.5 sequential steps per second of audio). Do not over-invest here.

### Applied, with what each was worth
1. **bf16 attention** — best accuracy of four dtype configs *and* faster. Re-verified on the
   worst-sample metric, not just PCC (see below).
2. **Chunked windowed attention**, slab 512 — O(S²) → O(S·slab). At S=12000: warm 892→497 ms,
   cold 10580→1178 ms, mask 2304 MB→4.2 MB. **Exact**, not an approximation: chunked and unchunked
   give byte-identical worst-case error at all four stage lengths.
3. **Uniform slab-sized chunks** — one cached bias per window. Bias cache 23 tensors/53 MB →
   6/18.1 MB, and it stops growing with utterance length.
4. **Conv length bucketing**, 128 — on a stream of 12 distinct lengths, 120.9 s → 1.66 s (73x).
5. **Hoisted conv weight preparation** — 2.6x at T=128 (112.8→43.7 ms); host share 88%→24%.
6. **Content-deduplicated prepared weights** — 730 MB → 98 MB, bit-identical.

### Measured and REJECTED — do not retry without new information
| idea | result |
|---|---|
| **sdpa** for the attention interior | 1.44–2.27x FASTER, 3.3x worse worst-case error, failed 11 tests. See §4.1 |
| **Smaller slab** toward the compute optimum 2·window | slab=32 computes **9x fewer** scores and runs **9x slower** (334 vs 36 ms/pass) |
| **Device trace capture** | **1.00x** at every slab size, including 3570 ops. The async command queue already hides host dispatch behind device execution |
| **Batching the chunks** into one matmul | batched attention is 2.4x faster, but the gather costs more than it saves (11.30 vs 9.67 ms). Chunks overlap by `window`, and ttnn has no strided view, so the copy is unavoidable |
| **Unchunked attention** with a full [S,S] mask | identical accuracy, 3x slower at S=4096, mask grows to 268 MB. Only S<=1024 prefers it — a `chunk_min` question |
| **Changing the dtype default** | the current default already wins on *both* PCC and worst-sample |

The recurring lesson: **on this hardware, doing less arithmetic does not mean finishing sooner.**
Three of these reduce FLOPs and all three are slower. Cost is dominated by per-kernel overhead and
memory traffic.

### 4.1 The sdpa investigation (the deepest hole; read before touching attention)

sdpa **is** faster and its API **does** support ALiBi (via `attn_mask` with `is_causal=False`). The
old note claiming "no ALiBi support" was wrong. It is rejected on **accuracy**: against an exact
fp64 answer, worst-case error 1.95e-02 vs hand-rolled 5.85e-03.

Ruled out as the cause, each by measurement:
- our ALiBi mask — an **all-zero** mask still gives 4.7x (1.135e-03 vs 2.395e-04)
- `MASK_NEG` magnitude — identical to 4 digits from -1e4 to -1e30
- ALiBi precision — per-head error is uniform, *anti*-correlated with slope
- online-softmax rescaling — 1 k-block vs 4 k-blocks: identical
- `exp_approx_mode`, `math_fidelity` HiFi3/HiFi4, `fp32_dest_acc_en`
- legacy v1 vs streaming v2 kernel (`fp32_dest_acc_en` selects between them!) — 2.066e-02 vs 2.163e-02
- the fp32→bf16 input conversion — costs 3.70e-03, identically for both paths
- **patching `sdpa_program_factory.cpp` so `im_df`/`stats_df` are Float32, and rebuilding** —
  marker-verified live (`im_df=Float32`), error unchanged. So closed issue #13364 is a red herring.

Established: a **mask-independent precision floor intrinsic to the fused kernel**, 3.7–10.7x worse
worst-case. Telling detail: hand-rolled error *falls* with S (4.57e-04→1.27e-04 for S=128→2048)
because softmax averages more values, while sdpa stays flat (1.7e-03→1.4e-03). sdpa has a
fixed-magnitude error term that does not benefit from averaging. Numerical attribution matches
"bf16 accumulation of P@V" (6.8e-04) rather than "bf16 scores+probs" (1.1e-04, which is what
hand-rolled's 2.4e-04 is consistent with).

**Where it stopped:** the one-k-block result rules out repeated cross-block accumulation, so it is a
single fixed-precision step inside the fused pipeline. Pinning it further needs instrumenting
`kernels/compute/sdpa.cpp` to dump intermediates. Not done.

**This matters for Block 1** — see §7.

---

## 5. Traps that cost real time — read these

1. **A failed trace capture wedges the device.** If an exception escapes between
   `begin_trace_capture` and `end_trace_capture`, `close_device` hangs and **every later run on the
   card blocks**. Cost ~20 min and required killing by PID. Always `end_trace_capture` in a
   `finally`. (No trace code remains in Block 3, but Blocks 1/2 will want tracing.)
2. **Prepared conv weights are NOT length-independent.** Same shape at every length, *different
   values* — cross-length reuse computes PCC **0.19**. Shape equality and absence of a crash are not
   correctness. The cause: ttnn lowers conv to an implicit GEMM whose weight operand is
   `[C_in·K, C_out]` (no L in it, hence the constant shape), but the sharding/blocking plan is
   chosen from the GEMM's row count `N·L_out`, and the weight is pre-arranged to match that plan.
3. **`prepare_conv_*`'s `input_dtype` is the ACTIVATION dtype**, not the weight dtype. Getting it
   backwards silently produced PCC 0.008.
4. **A stale benchmark outlived what it measured.** bf16 weights looked ~20% faster; that gap was
   conv weight-prep cost. Re-measure defaults after any change to the hot path.
5. **"Time is in the TTNN wrapper" != "time is dispatch."** A profile showed 98% inside
   `ttnn/decorators.py:__call__` and I concluded tracing was the fix. It was work inside *one op*.
   Tracing later measured 1.00x. Ask *which ops*, not just how much.
6. **Synthetic gates let the audible path rot.** Four numerical changes landed without re-running
   real speech. Now pinned by a 6.4 KB fixture. If you optimize Blocks 1/2, add the equivalent
   fixture test *first*.
7. **Comments outlive code.** Every review pass has found one: the conv-upload comment, an
   "fp32 attention" docstring after bf16 became default, a wrong sdpa rationale, and the same
   optimization quoted as both 2.4x and 2.6x. Re-read the docstring of anything you change.
8. **`models/experimental` is NOT covered by the repo's black config** (pyproject `include` lists
   `models/demos`). Black flags 13/16 files; the sibling xtts_v2 code fails it too. **Do not
   reformat.** Do keep lines <= 120.
9. **PCC HIDES OUTLIERS — the single most expensive lesson.** It is a correlation: it can sit at
   0.9998 while individual samples are badly wrong, and for audio the outliers are what you hear.
   sdpa passed at PCC 0.9998 per slab and still failed 11 tests, one stage measuring PCC 0.916.
   Always pair PCC with a worst-sample bound.
10. **Verifying a C++ rebuild is harder than it looks.** Three separate traps in one experiment:
    - `nohup cmd &` inside a backgrounded tool call gets reaped when the wrapper exits; the harness
      reports the *wrapper's* exit code, not the build's.
    - `ldd` resolved `_ttnncpp.so` from `build_Release/lib/`, a **stale separate copy**, not the
      freshly linked `build_Release/ttnn/` one. `LD_LIBRARY_PATH` fixes it (it is RUNPATH, so
      LD_LIBRARY_PATH wins).
    - **A relinked `.so` with a fresh timestamp is NOT evidence your source change is in it.** The
      first build reported exit 0, relinked both `.so`s, and had never recompiled the edited file
      (it only rebuilt `tt_metal/fabric`). Only a runtime `log_warning` marker proved it. Note
      `log_debug` is compiled out in Release, so it cannot be used for this.
11. **`fp32_dest_acc_en` selects the sdpa kernel variant**, it is not just a precision knob:
    `can_use_streaming_compute(fp32_dest_acc_en) { return !fp32_dest_acc_en; }`. Passing `True`
    forces the legacy v1 kernel. Holding it `True` in every test meant streaming v2 was never
    exercised for a long time.

---

## 6. Open items

### Block 3 — closed, with two deferred decisions (not defects)
- **`chunk_min` should probably be 1024, not 512.** Measured crossover is S ~ 2000: at S=1024
  unchunked is 1.5x faster (1.40 vs 2.10 ms). Worth 1.06x on attention, ~1.7% of the block,
  **~0.1% end-to-end**, and costs a 16.8 MB bias. Recorded in the code; not worth a commit alone.
- **`bucket=128` is wrong for streaming.** A 1-second chunk costs the same 43.7 ms as a 10-second
  one, because everything below 128 frames pads to 128. Streaming wants 16 or 32 (floor: 12.8 ms at
  bucket 16). A **geometric** ladder (64,128,192,256,384,512,768,1024,1536) would beat the uniform
  grid — worst-case relative padding drops from 98% to ~50% — but tuning it needs a real
  utterance-length distribution.
- **Streaming needs boundary state throughout**, not just a bucket change. Three places fabricate
  or discard context that a continuing stream would supply: the transposed convs' trimmed tail, the
  forward convs' `_pad_causal` (which would replicate the chunk's own first frame instead of the
  previous chunk's real tail), and attention's first `window` rows. Ordinary work, but list it.
- **batch=1 only.** Upstream serve concurrency is 32. Not needed yet.
- **bf16 weights + bf16 attention has almost no margin**: 1.93% worst-sample against a 2.00% gate,
  for a ~1% speed gain. The knob exists for experiments; do not ship it.
- Minor: the suite tests to T=469; T=1500 is measured by hand but not pinned.

### Block 1 — the next big piece
3.4B params, 26 layers, dim 3072, GQA 32/8, head_dim 128, SwiGLU 9216, RMSNorm, RoPE θ=1e6, tied
embeddings, **`n_heads*head_dim` (4096) != `dim` (3072)** so wq/wo are not square.

**Risks investigated and CLEARED** (all in `models/tt_transformers/tt/`):
- `head_dim=128` is honoured: `model_config.py:2678` reads it from config, falling back to
  `dim // n_heads` (which would give a wrong 96).
- The non-square `wo` is handled: `model_config.py:1983` uses `k_dim = n_heads * head_dim`.
- **`forward()` takes embeddings, not token ids** (`model.py:852`, `x: ttnn.Tensor`), so our
  37-codebook summed input works — `embedding.py` is a separate module we simply bypass.
- `lm_head` is skippable, and we *should* skip it: Block 2 consumes the post-final-norm hidden
  state (`voxtral_backbone_ref.py:157`), and we never emit text tokens. That is 402M tied
  parameters (131072x3072) we do not pay for.
- RoPE convention: `use_hf_rope` defaults to `False` (Meta/interleaved), which matches our
  bit-exact reference. Do not let a default flip it to HF half-split.

**New constraint found:** `tt_transformers` **requires HF-format checkpoints** —
`model_config.py:588` raises unless `HF_MODEL` is set, and the Meta `consolidated.00.pth` path is
vestigial. Ours is Mistral-native `consolidated.safetensors` + `params.json`. So the shim is bigger
than a config-add: a `config.json` **plus a checkpoint transcription to HF parameter names**.
Mechanical and CPU-testable. `model_params/Mistral-7B-Instruct-v0.3/config.json` is the schema to
copy; the field mapping is dim→hidden_size, n_layers→num_hidden_layers, n_heads→num_attention_heads,
n_kv_heads→num_key_value_heads, hidden_dim→intermediate_size, norm_eps→rms_norm_eps,
tied_embeddings→tie_word_embeddings, plus an explicit `head_dim: 128`.

**Topology is settled:** this box has exactly one device (WORMHOLE_B0, 1 visible, 1 PCIe), so
single-N150. Memory: 3.03B params in the 26 layers + 402M embedding = 3.43B →
**bf16 6.86 GB, bfp8 3.64 GB**, plus ~218 MB KV cache at seq 2048 and ~260 MB for Block 3, against
~11 GB usable. bf16 fits with headroom; start there and only drop to bfp8 if speed demands it.

### Block 2 — smaller but more novel
390M, 3 layers, **3-token sequence**, bidirectional (no RoPE, no mask), 7 Euler steps per frame with
CFG batched to 2B. Fixed shapes and a fixed step count make it an ideal trace target — upstream's
own CUDA-graph version of exactly this gave them 47% latency / 2.5x RTF. Note Block 3's tracing null
result does **not** transfer: that was a long chain of ~30 us device-bound ops, whereas Block 2's
ops are tiny, which is exactly the regime where host dispatch can become exposed. Read trap #1
before writing trace code.

### Standing constraints (not fixable by us)
- Weights are **CC BY-NC 4.0**, non-commercial, including the reference voices. Same class of
  blocker as XTTS-v2's CPML. Needs legal sign-off before any product use.
- **Voice cloning from arbitrary audio is impossible** — the codec encoder is not in the release
  (0 of 386 tensors). Only the 20 shipped presets. A test asserts this so a future release that
  adds them fails loudly.

---

## 7. Suggested order when resuming

1. Re-read this file and `reference/PROVENANCE.md`. Recreate the two venvs (§2).
2. Run the 118 tests to confirm the environment before changing anything.
3. **The sdpa compounding probe — do this first, it is ~1 hour and needs no checkpoint.**
   Block 1 escapes sdpa's *minor* problem (the `attn_mask`/`is_causal` exclusivity: its attention is
   causal + RoPE with no additive bias, so the native fast path works and no mask tensor is needed)
   but **inherits the major one** — the mask-independent 3.7–10.7x worst-case penalty in §4.1, which
   was proven with an all-zero mask. Chain 26 sdpa attentions at Block 1's real shapes (GQA 32/8,
   head_dim 128, dim 3072, causal, seq ~500) with random weights against an fp64 reference, and
   report PCC **and worst-sample per layer**, to see whether the error compounds or stays bounded.
   Cover **both** kernels: `sdpa` for prefill and **`sdpa_decode`** for decode — a separate code
   path, and the XTTS-v2 work hit an odd-tile bug there needing the KV-cache length rounded to a
   multiple of 64.
   - If bounded: proceed with sdpa. `tt_transformers` ships it for Llama 3.1 8B, Mistral 7B, Qwen
     and Gemma at production scale, so a worst-case attention penalty is evidently tolerable for LLM
     hidden states — and Block 1's output feeds more processing rather than your ears, unlike
     Block 3's.
   - If not: that is a finding worth escalating, because it would affect every model in the repo.
4. Block 1 via `tt_transformers` — the `params.json`→`config.json` shim plus the checkpoint
   transcription (§6), validated on CPU against our reference (bit-exact RoPE, PCC 0.99999988 on the
   full stack) before any device work.
5. Wire it up: our embeddings into `forward(x=...)`, keep the final norm, skip `lm_head`. Gate on
   PCC **and** worst-sample.
6. Block 2, with the 7-step ODE in a single trace.
7. `pipeline.py` end-to-end on device, then revisit Block 3's `bucket` for the real chunk size.

The end-to-end performance question is decided by Block 1: 87% of the parameters and 12.5 sequential
steps per second of audio. Block 3 is done and should be left alone.

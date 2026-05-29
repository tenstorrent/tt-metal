---
name: generation
description: Build the autoregressive generation loop, demos, and end-to-end use-case validation for a brought-up model (KV cache, sampling, EOS, image/text I/O). Use when implementing generate(), demos, or e2e tests.
---

# SKILL: Generation (AR Loop + Demos + E2E Validation)

## Purpose
Wire a working end-to-end pipeline for one use case: encoder forward
→ (optional AR decode with KV cache + sampling + EOS) → (optional audio
post-processing) → output. Produce a demo CLI that runs both the TTNN
pipeline and the HF reference side-by-side, plus a parametric e2e test
gated against HF parity on a small known-good sample set.

## When to use
- After all `components_used` for the use case have `ttnn.status=done`
  AND `real_weights.status=done`. Both gates matter — the AR loop is
  fragile under real-checkpoint compounding error.
- Once per use case. The first use case that needs AR pays the cost
  of building `kv_cache.py` + a `{use_case}_generator.py` driver;
  subsequent use cases on the same model reuse that infrastructure.

## Prerequisites
- `tt/weight_loader.py` returns nested state dicts for every block in
  `use_case.components_used`. Full-config PCC > 0.97 per the integration
  skill's Stage 2.
- HF `generation_config.json` is on disk next to the checkpoint (read
  it for `decoder_start_token_id`, `eos_token_id`, language code maps).
- A handful of known-good `(src, ref)` pairs exist in the model's HF
  card or model README. Anything that HF itself translates / transcribes
  correctly at greedy `num_beams=1` is suitable seed data.

## Process

### 1. Determine the pipeline shape

Read the use case entry from `architecture_inventory.json` and answer
three questions:

- `needs_ar` — is there a `.generate()` on the HF wrapper? If false the
  generation phase is a one-shot encoder forward, no KV cache.
- `needs_audio_out` — is `output_modality == "audio"`? If true the
  pipeline ends in a vocoder (or NAR text-to-unit + vocoder for models
  that go through a discrete-unit bottleneck).
- `components_used` — the exact set of TTNN sub-models this pipeline
  composes. The `tt/<use_case>_model.py` wrapper imports these and
  nothing else.

Map this onto one of the three pipeline shapes:

| Shape | Modality | Components |
| --- | --- | --- |
| Encoder-only | text→embeddings, audio→features | `{encoder}` |
| Encoder + AR decoder | text out, captions | `{encoder, decoder, lm_head, kv_cache}` |
| Encoder + AR decoder + audio out | translation→speech, TTS | `{encoder, decoder, lm_head, kv_cache, vocoder}` and optionally `{nar_unit_generator}` |

The wrapper class lives in `tt/{use_case}_model.py` and exposes a
single use-case-specific verb: `translate()`, `transcribe()`,
`synthesize()`, `embed()`. Construction loads weights ONCE; the verb
is re-entrant across multiple calls.

### 2. KV cache (skip if `needs_ar=false`)

Two cache classes per layer. Both store `[batch, num_heads, seq_len,
head_dim]` in DRAM TILE_LAYOUT bfloat16, mirroring the post-projection
shape already produced by your MHA block (the `reshape -> transpose(1,2)`
lands K/V in this layout before SDPA).

- **`SelfAttentionKVCache`** — `seq_len = max_decode_seq_len`, padded to
  a tile multiple of 32. Written one token per step, read whole on
  every step.
- **`CrossAttentionKVCache`** — `seq_len = encoder_seq_len`. Populated
  ONCE per `generate()` call right after the encoder runs (the K/V
  projections are applied to encoder hidden states and the result is
  stored here). Reused for every decode step.

Pre-allocate persistent buffers at `__init__` and overwrite in place
across `generate()` calls. Captured decode traces hold a pointer to
these buffers — reallocating between calls would break trace reuse.
Provide a `reset()` that zeroes the contents but does NOT free the
device tensors.

Self-attention skeleton (model-agnostic):

```python
class SelfAttentionKVCache:
    def __init__(self, device, num_layers, batch, num_heads, max_seq_len,
                 head_dim, dtype=ttnn.bfloat16):
        zeros = torch.zeros(batch, num_heads, max_seq_len, head_dim,
                            dtype=torch.bfloat16)
        self.k_caches = [
            ttnn.from_torch(zeros, device=device, dtype=dtype,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(num_layers)
        ]
        self.v_caches = [...]  # same
        # Persistent int32 [batch] position buffer for paged_update_cache.
        self._persistent_pos_tt = ttnn.from_torch(
            torch.zeros(batch, dtype=torch.int32),
            device=device, dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def update(self, layer_idx, k_new, v_new, pos):
        # Resolve int -> persistent device tensor.
        if isinstance(pos, ttnn.Tensor):
            pos_tt = pos
        else:
            host = ttnn.from_torch(torch.tensor([int(pos)] * self.batch,
                                                dtype=torch.int32),
                                    dtype=ttnn.int32)
            ttnn.copy_host_to_device_tensor(host, self._persistent_pos_tt)
            pos_tt = self._persistent_pos_tt
        # k_new/v_new are [B, num_heads, 1, head_dim]; reshape +
        # interleaved_to_sharded into the layout paged_update_cache wants.
        ...
        ttnn.experimental.paged_update_cache(self.k_caches[layer_idx],
                                             k_sharded,
                                             update_idxs_tensor=pos_tt)
        ttnn.experimental.paged_update_cache(self.v_caches[layer_idx],
                                             v_sharded,
                                             update_idxs_tensor=pos_tt)
```

Use `ttnn.experimental.paged_update_cache` with a single logical page
(no actual paging — the cache is just a contiguous tensor) from day
one. `ttnn.update_cache(int_pos)` bakes the position into the kernel
args at capture time and BREAKS trace reuse across positions. The
perf skill explains why this is the only viable choice for tracing;
do not start with `update_cache` and "convert later".

Cross-attention is simpler — one `ttnn.fill_cache` per layer at the
end of the encoder prefill:

```python
def populate(self, layer_idx, k, v):
    ttnn.fill_cache(self.k_caches[layer_idx], k, 0)
    ttnn.fill_cache(self.v_caches[layer_idx], v, 0)
    ttnn.deallocate(k); ttnn.deallocate(v)
    self._populated[layer_idx] = True
```

### 3. `decode_step` contract

The decoder block exposes a `decode_step(...) -> hidden_tt` method
that takes:

- `input_ids`: `[1, 1]` int (single token) OR a persistent
  `persistent_input_ids_tt` device buffer (preferred for tracing).
- `position`: integer slot (untraced) OR a persistent int32 device
  buffer (traced).
- `past_key_values`: the bundled `{SelfAttnCache, CrossAttnCache}`.
- `precomputed_self_mask_tt`: a `[1, 1, 1, max_seq]` device tensor
  the caller streams per-step. The cache stores ALL slots; the mask
  hides slots beyond the current position with additive `-inf`.
- `precomputed_encoder_mask_tt`: a `[1, 1, 1, enc_seq_total]` device
  tensor invariant across the generate() call.

It returns a `[1, 1, embed_dim]` last-hidden-state tensor. The generator
wrapper then applies the LM head (`ttnn.linear` with the tied
`shared.weight`) to produce `[vocab]` logits.

Internally the step:
1. Embeds the single new token, adds positional embedding.
2. For each layer: self-attn — Q from current token, K/V from cache
   (after writing this step's K/V via `paged_update_cache`).
3. For each layer: cross-attn — Q from current hidden, K/V read from
   the static cross-attn cache.
4. FFN, final layer-norm.

Single-token Q against full-cache K/V via SDPA is the standard pattern.
Do not collapse to plain matmul — the SDPA kernel benefits from
the cache-resident K/V layout.

### 4. Prefill order

For an encoder + AR decoder use case:

1. Run the encoder. Capture `enc_hidden [1, S, H]` host-side (or keep on
   device if your wrapper supports it).
2. Tile-pad S to a multiple of 32. The cross-attention cache shape is
   keyed by this padded S — construct the generator with the matching
   `encoder_seq_len`.
3. Call `populate_encoder_cache(enc_hidden, encoder_attention_mask)`.
   For each layer: project encoder hidden states through that layer's
   cross-attn K/V weights and `fill_cache` the result.
4. Reset the self-attn cache.
5. Run the model-specific prefix tokens through `decode_step` to warm
   up the self-attn cache. Common shapes:
   - Multilingual encoder-decoder: `[decoder_start_token_id, tgt_lang_id]`.
   - Decoder-only causal LM: `[bos_token_id]` (or no prefix at all).
   - Conditional ASR: `[start_of_transcript, lang_id, task_id, no_timestamps]`.

The prefix is model-specific. Read HF's `.generate()` (or
`generation_config.json`) to find it:

```python
# In a debugger / scratch script:
from transformers import <HFClass>
m = <HFClass>.from_pretrained(...)
print(m.generation_config)
print(m.generation_config.decoder_start_token_id)
# Walk through .generate() in the HF source for the prefix-token
# construction. Copy that logic verbatim.
```

The logits produced at position N-1 (where N = prefix length) are the
FIRST logits you sample from. Logits at positions 0..N-2 are discarded.

### 5. AR loop + sampling + EOS

Greedy is `torch.argmax(logits)`. That gets you to the e2e test
green-light fastest. Top-k / top-p come from
`models/common/sampling/tt_sampling.py`; logits processors (no-repeat
n-gram, repetition penalty, etc.) live in
`models/common/generation_utils.py`. Use them only when the use case
requires non-greedy decoding — most translation / transcription
benchmarks score against `do_sample=False, num_beams=1`.

Stop on either:
- `next_token == eos_token_id` (model-specific, read from
  `generation_config.json`)
- `len(tokens) == max_new_tokens` (caller-supplied, also bounded by
  `max_decode_seq_len` cache capacity)

Skeleton (greedy, single batch):

```python
tokens = [decoder_start_token_id, tgt_lang_id]   # prefix

# Warm-up: feed every prefix token. Logits from position N-1 are the
# first sampled ones.
for i, tok in enumerate(tokens):
    logits = decode_step(prev_token=tok, position=i, ...)

for pos in range(len(tokens), max_total):
    next_token = int(torch.argmax(logits).item())
    tokens.append(next_token)
    if next_token == eos_token_id:
        break
    if pos + 1 >= max_total:
        break
    logits = decode_step(prev_token=next_token, position=pos, ...)
```

Return `tokens` as a Python list including the prefix — this matches
what HF's `model.generate()` returns. Strip via
`processor.decode(tokens, skip_special_tokens=True)` for display.

### 6. Demo CLI shape

One file per use case under `demo/demo_<use_case>.py`. Use `typer` —
all existing TT demos follow this convention.

CLI args by input modality:
- text in: `--src "..."` `--src-lang <code>` `--tgt-lang <code>`
- audio in: `--wav <path>` `--src-lang <code>` (optional, may default)
- text out: print to stdout
- audio out: `--out <path>` (writes 16-bit PCM 16kHz WAV)

Three flags are conventional across every demo:
- `--max-new-tokens` — AR budget (always includes prefix tokens for HF
  parity).
- `--device` — informational on single-device hosts.
- `--skip-hf` — opt out of the side-by-side HF run (faster iteration).

ALWAYS run HF reference alongside TTNN by default — the demo is the
spot where humans eyeball output quality, and the HF parity comparison
is the primary signal. The wrapper pattern:

```python
import typer
app = typer.Typer(add_completion=False, no_args_is_help=True)


def _run_hf_reference(...):
    """Invoke the matching HF class .generate() and return the decoded
    output. Free the HF model before TTNN runs to avoid ~8GB resident."""
    import torch
    from transformers import AutoProcessor, <HFClass>
    proc = AutoProcessor.from_pretrained(HF_PATH)
    model = <HFClass>.from_pretrained(HF_PATH, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(..., do_sample=False, num_beams=1,
                             max_new_tokens=max_new_tokens)
    text = proc.decode(out[0].tolist(), skip_special_tokens=True)
    del model
    return text


@app.command()
def main(
    src: str = typer.Option(..., "--src"),
    src_lang: str = typer.Option(..., "--src-lang"),
    tgt_lang: str = typer.Option(..., "--tgt-lang"),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens"),
    skip_hf: bool = typer.Option(False, "--skip-hf"),
):
    import ttnn
    from <model_pkg>.tt import weight_loader as wl
    from <model_pkg>.tt.<use_case>_model import <UseCaseModel>
    hf_sd = wl.load_hf_state_dict()
    processor = ...
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        model = <UseCaseModel>(device=dev, hf_state_dict=hf_sd,
                                processor=processor)
        ttnn_out = model.<verb>(src, src_lang, tgt_lang,
                                max_new_tokens=max_new_tokens)
    finally:
        ttnn.close_device(dev)
    typer.echo(f"Source: {src}")
    typer.echo(f"Translation: {ttnn_out}")
    if not skip_hf:
        typer.echo(f"HF: {_run_hf_reference(...)}")
```

The HF run lives in a helper, NOT inline — the validation test
imports it.

### 7. Validation by output modality

One helper module: `demo/validate.py`. One thin function per metric.
Add metrics as the use_case inventory demands them.

| Modality | Metric | Library | Helper signature |
| --- | --- | --- | --- |
| text out (translation) | corpus BLEU | `sacrebleu` | `bleu(hyps, refs) -> float` |
| text out (transcription) | WER | `jiwer` | `wer(hyps, refs) -> float` |
| audio out (TTS / S2ST) | ECAPA speaker-similarity cosine | `speechbrain` | `ecapa_cos(hyp_wav, ref_wav) -> float` |
| audio out (fallback) | char-edit-distance over re-ASR transcripts | stdlib | `char_similarity(hyp_text, ref_text) -> float` |
| embeddings | PCC | torch | `pcc(hyp, ref) -> float` |

Single-reference BLEU:

```python
def bleu(hyps, refs):
    import sacrebleu
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)  # 0..100
```

Audio similarity falls back to re-ASR when an ECAPA model is not on
disk: transcribe both `hyp_wav` and `ref_wav` with the same ASR
pipeline, then char-edit-distance on the two transcripts. The
re-ASR fallback is best-effort — wrap it in try/except and degrade
gracefully to the simpler duration + finite-value gates.

### 8. E2E test

One file per use case: `tests/test_e2e_<use_case>.py`. Same pytest
skeleton as the integration skill's `test_real_hf_weights.py`:

- Session-scoped `hf_sd`, `processor`, `hf_translations` fixtures.
- Function-scoped `device` fixture (fresh open per test).
- Read samples from `demo/inputs/<use_case>_samples.json` (or
  `.wav` for audio-in use cases).
- Compute the metric on TTNN outputs AND HF outputs over the SAME
  samples. Compare TTNN to HF — not TTNN to references — because the
  bf16 stack is allowed to drift slightly from fp32 HF as long as
  parity holds.

```python
SAMPLES_PATH = Path(__file__).resolve().parent.parent / "demo" / "inputs" / "<use_case>_samples.json"
MAX_NEW_TOKENS = 32       # short-form set; keep small for runtime
BLEU_TOLERANCE = 1.0      # "HF - 1.0" from use_case.validation_threshold


@pytest.fixture(scope="module")
def samples():
    return json.load(open(SAMPLES_PATH))


@pytest.fixture(scope="module")
def hf_translations(samples, processor):
    # Run HF reference once over all samples; cache for the module.
    ...


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def test_<use_case>_metric_matches_hf(samples, hf_sd, processor,
                                      hf_translations, device):
    refs = [s["ref"] for s in samples]
    model = <UseCaseModel>(device=device, hf_state_dict=hf_sd,
                            processor=processor)
    ttnn_outs = [model.<verb>(...) for s in samples]

    ttnn_score = bleu(ttnn_outs, refs)
    hf_score = bleu(hf_translations, refs)

    print(f"TTNN={ttnn_score:.3f}  HF={hf_score:.3f}  "
          f"drift={hf_score - ttnn_score:.3f}  (tolerance={BLEU_TOLERANCE})")
    for i, (s, hf_t, tt_t) in enumerate(zip(samples, hf_translations,
                                            ttnn_outs)):
        print(f"  [{i}] HF={hf_t!r}  TTNN={tt_t!r}")

    assert ttnn_score >= hf_score - BLEU_TOLERANCE, (
        f"TTNN {ttnn_score:.3f} fell more than {BLEU_TOLERANCE} below "
        f"HF {hf_score:.3f} — investigate AR drift / cross-attn cache."
    )
```

Parse `use_case.validation_threshold` into one of two forms:
- Parity-relative: `"HF - 1.0"` / `"HF + 0.05"` — the gate is `metric
  drift relative to HF`. Use when HF itself is imperfect on the
  samples (translation: HF scores ~30 BLEU, not 100).
- Absolute: `"≥ 0.95"` — the gate is `metric ≥ value`. Use when there's
  a ground-truth reference and HF is expected to nail it (audio
  ECAPA-cos, deterministic embeddings).

Keep `MAX_NEW_TOKENS` small (16-32). The test exists to gate parity,
not to measure long-form generation. Performance characterization is
the perf phase's job.

### 9. Hybrid host/device boundaries

Some HF pipelines have host-side bookkeeping that does NOT belong on
device. The orchestrator's "no shortcuts" guard for the ttnn-phase
does NOT extend to "no host ops anywhere downstream of bringup".
Document these in the use case's `hybrid_notes` field in
`architecture_inventory.json`.

Common legitimate host-side pieces:
- Tokenizer / processor — always on host. HF `AutoProcessor` is the
  ground truth for tokenization; replicating it in TTNN is wasted
  effort.
- Char-to-id maps for TTS char-level input (e.g. char tokenizer +
  per-token character-count tables produced by the HF processor for
  NAR text-to-unit pipelines). These are small Python dict lookups;
  copying them to device-side ops would add latency for zero gain.
- Audio post-processing — trim trailing silence, resample to 16kHz,
  WAV write. Trivial on host; not part of the perf-critical path.
- Sample-rate conversion (torchaudio) for non-16k input.

A NON-legitimate host-side piece is e.g. running a transformer layer
on host because the TTNN port hit a hang. That belongs in the debug
phase, not in `hybrid_notes`. Read `hybrid_notes` as "things the user
should expect to run on host" not "things we gave up on porting".

## Output artifacts

- `tt/<use_case>_model.py` — wrapper composing the per-component TTNN
  sub-models. One verb (`translate` / `transcribe` / `synthesize` /
  `embed`). Loads weights once; `generate()` is re-entrant.
- `tt/kv_cache.py` and `tt/<some>_generator.py` — created by the FIRST
  use case that needs AR; reused thereafter.
- `tt/t2u_generator.py` or similar — created by the FIRST use case
  that needs the NAR-unit/vocoder path; reused thereafter.
- `demo/demo_<use_case>.py` — typer CLI, side-by-side HF run by
  default.
- `demo/validate.py` — metric helpers; add a function per metric as
  use cases need them.
- `demo/inputs/<use_case>_samples.json` or `.wav` files — the 5-10
  known-good seeds used by the e2e test.
- `tests/test_e2e_<use_case>.py` — single parametric test, HF parity
  gate.
- One row in `BRINGUP_LOG.md` under the `use_cases.generation` field,
  with the measured metric vs HF and the resolved threshold.

## Failure modes

- **TTNN tokens diverge from HF mid-generation.** Greedy decoding is
  deterministic, so identical logits-argmax MUST produce identical
  tokens. If they don't, the logits PCC is dropping mid-loop. Reproduce
  with an argmax-token-match harness: feed the SAME prefix to TTNN
  and HF, compare `argmax(logits)` at every step, find the first
  position they disagree. The block whose output PCC drops at that
  position is the culprit. Re-run that block's full-config PCC test
  with a realistic (post-LN) input — see the integration skill's
  realistic-input trick.

- **BLEU / WER below gate.** Enable the side-by-side `[idx] HF=... TTNN=...`
  print and inspect. Distinguish three cases:
  - TTNN output is identical to HF on most samples but truncates one
    early → EOS predicted prematurely. Audit the EOS-detection
    branch (off-by-one in position counting is common).
  - TTNN output rambles past the EOS point → EOS missed. Same fix.
  - TTNN output is semantically right but lexically different → bf16
    drift in late-position logits. Raise the threshold, or raise the
    sample count so the corpus BLEU averages out the noise.

- **Audio output is silent / garbled.** Three usual suspects, in order:
  - Wrong vocoder lang id. The vocoder has its OWN language code map
    (`vocoder_lang_code_to_id`), NOT the text decoder's
    `text_decoder_lang_to_code_id`. They are different integers even
    when the language code string matches.
  - Unit-token offset off-by-one. After the NAR text-to-unit pass,
    HF subtracts `vocoder_offset` (usually 4) from the unit IDs
    before indexing `vocoder.unit_embedding`. Missing this offsets
    every embedding lookup and produces noise.
  - Duration upsampler underflow. The NAR decoder's
    `char_count_per_id` includes leading/trailing zero pads for the
    lang/EOS tokens — drop them and the upsampled sequence is too
    short, clipping the audio.

- **Hangs on the first decode step but not in untraced mode.** The
  trace captured a non-persistent buffer address. Audit every input
  to `decode_step` — anything passed by Python int (position, token
  id) must be threaded through a persistent device buffer before
  capture. See `text_generator.py`'s `_ensure_persistent_buffers` for
  the full list.

- **`generate()` works on the first call, fails on the second.**
  Captured trace holds a pointer to a buffer that got reallocated. The
  cross-attn cache is the most common culprit — `populate()` must
  OVERWRITE the persistent buffers via `fill_cache`, not allocate
  fresh ones. Same for the encoder-mask: in trace mode, `_precompute_
  encoder_mask` must `copy_host_to_device_tensor` into the persistent
  buffer, not return a new tensor.

- **Shape mismatch on KV-cache update.** `paged_update_cache` wants
  `[1, B, num_heads, head_dim]` HEIGHT_SHARDED on L1. The K/V leaving
  the projection is `[B, num_heads, 1, head_dim]` TILE_LAYOUT DRAM.
  The cache wrapper does the reshape + `interleaved_to_sharded` —
  don't try to skip it.

## Reference implementation

- `models/demos/facebook_seamless_m4t_v2_large/tt/kv_cache.py` —
  full Self + Cross attention KV caches with persistent position
  buffer for trace reuse. ~400 lines.
- `models/demos/facebook_seamless_m4t_v2_large/tt/text_generator.py` —
  AR loop with prefill / `decode_step` / sampling / EOS / language
  conditioning. Single-trace capture across all decode positions via
  `paged_update_cache` + persistent buffers. ~870 lines.
- `models/demos/facebook_seamless_m4t_v2_large/tt/text_to_text_model.py`
  — encoder + AR decoder integration wrapper. ~260 lines.
- `models/demos/facebook_seamless_m4t_v2_large/tt/speech_to_text_model.py`
  — same but with an audio-input front end.
- `models/demos/facebook_seamless_m4t_v2_large/tt/text_to_speech_model.py`
  and `speech_to_speech_model.py` — variants that add the NAR
  text-to-unit pass + vocoder for audio output.
- `models/demos/facebook_seamless_m4t_v2_large/tt/t2u_generator.py` —
  NAR pass (encoder + decoder + argmax, no AR loop) for audio-output
  use cases.
- `models/demos/facebook_seamless_m4t_v2_large/demo/demo_t2tt.py`,
  `demo_s2tt.py`, `demo_t2st.py`, `demo_s2st.py`, `demo_asr.py` —
  one typer CLI per use case, all with side-by-side HF runs.
- `models/demos/facebook_seamless_m4t_v2_large/demo/validate.py` —
  thin BLEU wrapper around `sacrebleu`.
- `models/demos/facebook_seamless_m4t_v2_large/tests/test_e2e_t2tt.py`,
  `test_e2e_s2tt.py`, `test_e2e_t2st.py`, `test_e2e_s2st.py` — five
  e2e tests, one per use case, each with a HF-parity gate.

## Cross-references

- `skills/integration/SKILL.md` — produces the `real_weights.status=done`
  precondition. The Stage 2 full-config PCC>0.97 gate is the only
  thing standing between a green per-block PCC and a working AR loop.
- `skills/perf/SKILL.md` — characterizes the end-to-end pipeline this
  skill produces. Explains why `paged_update_cache` + persistent
  buffers + single-trace capture is the only viable path; do NOT
  rebuild the AR loop without those primitives.
- `skills/optimization/SKILL.md` — block-level performance work.
  Generation phase consumes the optimization-phase output (op fusion,
  L1 sharding) as-is — do not re-tune blocks during generation work.
- `models/common/sampling/tt_sampling.py` — top-k / top-p sampling
  primitives for non-greedy decoding.
- `models/common/generation_utils.py` — logits processors
  (no-repeat-ngram, repetition penalty, etc.) when the use case needs
  them.

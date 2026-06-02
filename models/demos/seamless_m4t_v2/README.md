# SeamlessM4Tv2 — Speech-to-Text Translation (S2TT) on Blackhole

TT port of Meta's [SeamlessM4Tv2](https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2)
**speech-to-text translation** path (`SeamlessM4Tv2ForSpeechToText`, checkpoint
`facebook/seamless-m4t-v2-large`) targeting a single Blackhole P150a.

Primary target direction: **English speech → Japanese text (en→ja)**.

Scope: S2TT only. Text-to-unit, vocoder, S2ST and T2ST are out of scope.

## Architecture

Two components (≈1.4B params, BF16 ≈ 2.8GB — fits one P150a):

1. **Conformer speech encoder** (w2v-BERT 2.0): 24 layers, dim 1024, 16 heads,
   FFN 4096, SiLU. Each block = macaron FFN (×0.5) → self-attention (Shaw
   `relative_key`) → causal depthwise conv (k=31) → FFN (×0.5) → LayerNorm.
   Followed by a stride-8 adapter that downsamples the sequence ~8×.
   - Chunk attention is a no-op at realistic clip lengths (`chunk_size=20000`),
     so encoder self-attention is full bidirectional (padding mask only).
2. **Text decoder** (NLLB-style): 24 layers, dim 1024, 16 heads, FFN 8192,
   vocab 256102. Causal self-attention + cross-attention to encoder + FFN,
   autoregressive with KV cache.

The fbank feature extractor (`SeamlessM4TFeatureExtractor`) runs on CPU.

## Layout

```
reference/reference_s2tt.py   HF reference + intermediate-tensor capture (PCC golden)
tt/model_config.py            SeamlessS2TTConfig (reads HF config.json)
tt/load_weights.py            HF safetensors -> prefix-sliced torch tensors
tt/conformer_conv.py          conv module (Phase 1)
tt/conformer_attention.py     Shaw rel-pos self-attention (Phase 2)
tt/conformer_encoder.py       conformer layer + 24-stack + adapter (Phase 3)
tt/text_decoder.py            decoder + lm_head (Phase 4)
tt/generator.py               build() + autoregressive greedy loop (Phase 5)
demo/demo_s2tt.py             wav -> Japanese text
tests/                        per-phase PCC tests
evaluation/                   BLEU (en->ja) + RTF benchmark (Phase 8)
```

## Running

CPU reference tests (host venv with transformers + sentencepiece):

```bash
cd /home/yito/ttwork/tt-metal
PYTHONPATH=$(pwd) MPLCONFIGDIR=/tmp/mpl \
  ../tt/bin/python -m pytest models/demos/seamless_m4t_v2/tests/test_phase0_reference.py -q --noconftest
```

On-device (TT) tests run inside the `metalcon:may11build` container bound to
`/dev/tenstorrent/1`:

```bash
sudo docker run --rm -v /home:/home -v /dev/hugepages-1G:/dev/hugepages-1G \
  --device /dev/tenstorrent/2 metalcon:may11build bash -c '
  export PYTHONPATH=/home/yito/ttwork/tt-metal:$PYTHONPATH MPLCONFIGDIR=/tmp/mpl
  cd /home/yito/ttwork/tt-metal
  python3 -m pytest models/demos/seamless_m4t_v2/tests/test_conformer_conv_pcc.py -q'
```

The container provides its own `/tt-metal` build (ttnn); the mounted repo provides
the `models.demos.seamless_m4t_v2` package.

## Status

| Phase | Description | State |
|---|---|---|
| 0 | Reference harness + config + weight map | done (encoder/decoder PCC golden validated) |
| 1 | Conformer conv module (depthwise = CPU fallback) | done (PCC ≥0.99 @ seq 128/512/1500 on device 2) |
| 2 | Conformer rel-pos self-attention | done (PCC ≥0.99, with/without pos bias) |
| 3 | Conformer layer + encoder + adapter | done (full encoder PCC ≥0.98, real weights) |
| 4 | Text decoder single-step | done (hidden PCC ≥0.95, greedy argmax agreement 1.0 vs HF) |
| 5 | E2E greedy generation (en→ja) | done (TT tokens identical to HF, agreement 1.0) |
| 6 | Performance | done (constant-shape bucketed decode — no per-step recompile; ~77% JIT cache hits). Full incremental KV-cache + ttnn trace = further opt |
| 7 | On-device conv hardening | done (depthwise + adapter stride-8 convs on-device; full encoder PCC ≥0.98; switchable via `conv_cpu_fallback`) |
| 8 | Evaluation (chrF/BLEU en→ja + RTF) | done (harness + metrics + FLEURS/smoke manifest builders) |

## Evaluation

```bash
# smoke (synthetic audio + HF references → measures TT-vs-HF agreement + RTF)
python3 models/demos/seamless_m4t_v2/evaluation/make_smoke_manifest.py --n 3
python3 models/demos/seamless_m4t_v2/evaluation/run_benchmark.py \
    --manifest models/demos/seamless_m4t_v2/evaluation/smoke_manifest.json --bucket_len 96

# real en→ja (FLEURS English audio + Japanese parallel transcription; needs network)
python3 models/demos/seamless_m4t_v2/evaluation/build_fleurs_manifest.py --max_samples 50
python3 models/demos/seamless_m4t_v2/evaluation/run_benchmark.py \
    --manifest models/demos/seamless_m4t_v2/evaluation/fleurs_en_ja/manifest.json --compare_hf
```

chrF is the primary metric for Japanese (tokenizer-free); BLEU uses sacrebleu's
ja-mecab when available, else a built-in char-BLEU (`evaluation/metrics.py`).

# Reference Audio for Voice Cloning

Qwen3-TTS uses In-Context Learning (ICL) for voice cloning. This requires a reference audio sample of the target speaker's voice.

## Included Reference Audio

- `jim_reference.wav` - 4 second English male voice sample
- `jim_reference.refcache.pt` - Pre-computed codec embeddings (speeds up inference)

## How Reference Audio Works

1. **Speech Tokenizer Encoder** converts audio to RVQ codes (16 codebooks @ 12Hz)
2. **Speaker Encoder (ECAPA-TDNN)** extracts a 2048-dim speaker embedding from mel spectrogram
3. These are combined with text embeddings to create ICL input for the Talker model

## Generating Your Own Reference

### Option 1: Let the Demo Auto-Generate
Simply provide any `.wav` file and the pipeline will automatically:
- Resample to 24kHz mono
- Extract codec codes and speaker embedding
- Cache results to `<audio_path>.refcache.pt` for faster subsequent runs

```bash
python demo_full_ttnn_tts.py \
    --text "Hello, this is a test." \
    --ref-audio /path/to/your/reference.wav \
    --ref-text "Text spoken in the reference audio" \
    --output /tmp/output.wav
```

### Option 2: Extract from Video
```bash
# Extract audio from video (e.g., MP4)
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 24000 -ac 1 reference.wav
```

### Option 3: Pre-Generate Cache Manually
```python
from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import encode_reference_audio, load_weights

main_weights, _ = load_weights()
ref_codes, audio_data = encode_reference_audio(
    "reference.wav",
    main_weights,
    cache_path="reference.refcache.pt"  # Optional custom path
)
print(f"Codes: {ref_codes.shape}, Audio: {audio_data.shape}")
```

## Reference Audio Requirements

| Requirement | Recommendation |
|-------------|----------------|
| Duration | 2-10 seconds (4-6s optimal) |
| Sample rate | Any (auto-resampled to 24kHz) |
| Channels | Mono or stereo (auto-converted to mono) |
| Format | WAV, FLAC, MP3 (any soundfile-supported format) |
| Quality | Clear speech, minimal background noise |
| Content | Natural speech in target language |

## Cache File Format

The `.refcache.pt` file contains:
```python
{
    "ref_codes": torch.Tensor,   # [seq_len, 16] - RVQ codes
    "audio_data": torch.Tensor,  # [num_samples] - Raw waveform @ 24kHz
}
```

## Cross-Lingual Voice Cloning

Qwen3-TTS supports using a reference speaker from one language to generate speech in another:

```bash
# English speaker -> French output
python demo_full_ttnn_tts.py \
    --text "Bonjour, comment allez-vous?" \
    --ref-audio jim_reference.wav \
    --ref-text "Let me also go over the review slides." \
    --language french \
    --output /tmp/french_with_english_voice.wav
```

Supported languages: english, chinese, french, german, italian, japanese, korean, portuguese, russian, spanish

## Troubleshooting

### Reference audio bleeding into output
Use `--auto-trim-bleed` to automatically detect and remove reference echo:
```bash
python demo_full_ttnn_tts.py \
    --text "Hello world" \
    --ref-audio reference.wav \
    --ref-text "Reference text" \
    --trim-frames 0 \
    --auto-trim-bleed
```

### Stale cache issues
Delete the `.refcache.pt` file to regenerate:
```bash
rm reference.refcache.pt
```

### Voice quality issues
- Ensure reference audio is clean (no music, minimal noise)
- Try a longer reference (4-6 seconds)
- Use reference audio in the same language as target text for best results

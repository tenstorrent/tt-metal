# AudioX - Anything-to-Audio Generation on Tenstorrent Hardware

AudioX is a unified framework for anything-to-audio generation supporting text, video, image, and audio inputs.

## Architecture

- **Multimodal encoders**: Text (CLAP), Video, Image, Audio encoders
- **Multimodal Adaptive Fusion**: Fuses diverse modality inputs
- **Diffusion Transformer**: Denoising network for audio generation
- **Vocoder**: Converts latent representations to 16 kHz waveforms

## Supported Tasks

- Text-to-audio
- Text-to-music
- Video-to-audio
- Video-to-music
- Audio inpainting
- Music completion

## Usage

```bash
# Text-to-audio
python models/experimental/audiox/demo_ttnn.py \
    --prompt "gentle rain falling on leaves" \
    --output rain.wav

# Text-to-music
python models/experimental/audiox/demo_ttnn.py \
    --prompt "upbeat electronic dance music" \
    --output music.wav \
    --mode text-to-music

# Run tests
pytest models/experimental/audiox/tests/
```

## Performance Targets

- Stage 1: 20+ tokens/s diffusion sampling, <30s generation for 10s audio
- Stage 2: Sharded memory configs, fused ops, L1 optimization
- Stage 3: 50+ tokens/s, <10s generation time, flash attention

## References

- [AudioX Paper (arXiv:2503.10522)](https://arxiv.org/abs/2503.10522)
- [AudioX Project Page](https://zeyuet.github.io/AudioX/)
- [TTNN Model Bring-up Guide](../../../tech_reports/ttnn/TTNN-model-bringup.md)

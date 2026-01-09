# SpeechT5 TTS (Text-to-Speech)

## Platforms:
    Wormhole (n150, n300)

### Introduction
SpeechT5 is a unified-modal encoder-decoder framework that converts text input to speech output. It was introduced in the paper ["SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing"](https://arxiv.org/abs/2110.07205). </br>
The SpeechT5 TTS model has been pre-trained on large-scale text-to-speech datasets and can generate high-quality natural speech from text input. It uses a combination of text encoder, speech decoder with postnet, and HiFi-GAN vocoder for generating 16kHz mono audio.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Login to huggingface with: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
  - To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

## How to Run

### Demo Script
Run the main demo script with text input:
```bash
cd models/experimental/speecht5_tts
python demo_ttnn.py "Hello world, this is text to speech synthesis on Tenstorrent hardware."
```

### Pytest Demo
Use the following command to run the model via pytest:
```bash
pytest models/experimental/speecht5_tts/demo_ttnn.py::test_demo -v
```

### Multi-Device Support
The demo supports both N150 and N300 hardware:

#### N150 (Single Device)
```bash
export MESH_DEVICE=N150
python demo_ttnn.py "Hello from N150" --max_steps 24
```

#### N300 (Multi-Device)
```bash
export MESH_DEVICE=N300
python demo_ttnn.py "Hello from N300" --max_steps 24
```

## Architecture

The SpeechT5 TTS pipeline consists of:
1. **Text Encoder**: Processes input text into hidden representations
2. **Speech Decoder**: Autoregressive decoder generating mel spectrograms
3. **Postnet**: Refines mel spectrograms for better quality
4. **Vocoder**: HiFi-GAN converts mel spectrograms to audio waveforms

## References

- [SpeechT5 Paper](https://arxiv.org/abs/2110.07205)
- [Microsoft SpeechT5 Models](https://huggingface.co/microsoft/speecht5_tts)
- [HiFi-GAN Vocoder](https://huggingface.co/microsoft/speecht5_hifigan)
- [CMU ARCTIC Dataset](https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors)

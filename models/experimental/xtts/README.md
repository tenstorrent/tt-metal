# XTTS

TTNN port of XTTS (text-to-speech) for Tenstorrent hardware.

## Layout

This directory follows the same structure as the `dots_ocr` demo
(see tenstorrent/tt-metal#43362):

```
xtts/
├── README.md
├── requirements.txt        # extra Python deps for the reference model
├── reference/              # pure-PyTorch reference implementation (CPU, no hardware)
├── tt/                     # TTNN ports of the layers (target Tenstorrent hardware)
└── tests/
    ├── conftest.py         # shared fixtures
    ├── demo/               # end-to-end demos (TT + reference)
    └── pcc/                # per-layer PCC validation against the reference
```

## Reference demo

The reference demo ([`reference/demo.py`](reference/demo.py)) runs the upstream
Coqui XTTS-v2 model on CPU, mirroring the model card
(https://huggingface.co/coqui/XTTS-v2). It is the ground truth the `tt/` port is
validated against.

### Setup (one-time)

Install the reference dependencies. CPU torch packages must come from the
PyTorch CPU index, and FFmpeg is a system dependency:

```bash
source python_env/bin/activate
sudo apt-get install -y ffmpeg
pip install -r models/experimental/xtts/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ This downgrades `transformers` to 4.57.x in the shared env (coqui-tts is
> incompatible with transformers 5.x), which may break tt-metal's own
> transformers-5 models. Restore with `pip install transformers==5.10.2
> huggingface_hub==1.18.0` when done. See `requirements.txt` for details.

### Run

`--mode api` uses the high-level `TTS.api`; `--speaker_wav` is any short voice
clip to clone (its words are irrelevant). `COQUI_TOS_AGREED=1` accepts the CPML
license non-interactively. The first run downloads ~1.9 GB of weights.

```bash
COQUI_TOS_AGREED=1 python models/experimental/xtts/reference/demo.py \
    --mode api \
    --text "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent." \
    --speaker_wav models/demos/audio/whisper/demo/dataset/conditional_generation/17646385371758249908.wav \
    --language en \
    --output models/experimental/xtts/output.wav
```

`--mode direct` loads a local checkpoint explicitly (model card example #3):

```bash
COQUI_TOS_AGREED=1 python models/experimental/xtts/reference/demo.py \
    --mode direct \
    --checkpoint_dir /path/to/xtts/ \
    --speaker_wav /path/to/speaker.wav \
    --language en \
    --output output.wav
```

Output is 24 kHz mono wav.

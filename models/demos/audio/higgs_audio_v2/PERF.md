# Model Performance

Promoted public performance path:

- single device
- `batch_size=1`
- greedy generation
- traced TTNN decode
- performance gate lives in [test_performance.py](/vol/stor-vol-1/loc/tt-metal/models/demos/audio/higgs_audio_v2/tests/test_performance.py)

## Reproduce

Set the repo environment first:

```bash
export TT_METAL_HOME=/vol/stor-vol-1/loc/tt-metal
export HIGGS_AUDIO_REPO=/abs/path/to/higgs-audio
export PYTHONPATH=$HIGGS_AUDIO_REPO:$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}
```

Then run:

```bash
python_env/bin/python -m pytest models/demos/audio/higgs_audio_v2/tests/test_performance.py -q
```

## Current Result

Latest traced run on March 27, 2026 for the minimal three-case public suite:

- aggregate autoregressive throughput: `64.61` tokens/s
- aggregate `RTF`: `0.3978`
- aggregate decode-only `RTF`: `0.3869`

Per-case results from the same run:

- `tts_es_project_update_long`: `64.62` tokens/s, `RTF 0.3923`
- `clone_en_review_update_long`: `64.62` tokens/s, `RTF 0.4002`
- `dialog_en_review_long`: `64.60` tokens/s, `RTF 0.4008`

Public gate:

- tokens/s `>= 60`
- `RTF < 0.5`

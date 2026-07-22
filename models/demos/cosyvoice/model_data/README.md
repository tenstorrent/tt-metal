# `model_data/` — heavy on-disk artifacts (not committed to tt-metal)

This directory holds everything needed to *run* the bring-up that the repo
itself does not commit. Contents are produced by the setup scripts and
are intentionally excluded from git via `../.gitignore`. Regenerate with:

```bash
source /root/tt-metal/python_env/bin/activate
cd /root/tt-metal/models/demos/cosyvoice
python scripts/download_model.py        # -> model_data/cosyvoice2-0.5B/
python scripts/clone_reference.py      # -> model_data/CosyVoice_src/
python scripts/gen_golden.py           # -> model_data/golden/
```

| subdir | contents | size | produced by |
|---|---|---|---|
| `CosyVoice_src/` | FunAudioLLM/CosyVoice reference repo at pinned SHA | ~50 MB | `scripts/clone_reference.py` |
| `cosyvoice2-0.5B/` | HF `snapshot_download('FunAudioLLM/CosyVoice2-0.5B')` | ~4.86 GB | `scripts/download_model.py` |
| `golden/` | per-component `.pt` fixtures + reference WAVs (greedy, seed=1986) | variable | `scripts/gen_golden.py` |


Do NOT edit anything under `CosyVoice_src/` — it is the frozen reference.

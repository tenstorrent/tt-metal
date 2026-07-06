# MiniMax-M3 — downloading the weights

The M3 text backbone ships **bf16** (no fp8/quant). The full checkpoint is **~869 GB** (59
safetensors shards). It's a **public** HF repo — **no token needed** (verified by pulling a file
unauthenticated). On-device we quantize to bf4/bf8, but the download itself is the bf16 source.

## Requirements
- `git-lfs` installed (`git lfs version`)
- **~900 GB free disk** (`df -h <target>`)
- Network — we saw ~124 MB/s → ~2 h total; scales with bandwidth.

## Method A — git-lfs (what we used)

```bash
# 1. Clone WITHOUT pulling the big blobs (fast — grabs config / tokenizer / modeling code + LFS stubs)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/MiniMaxAI/MiniMax-M3 /path/to/MiniMax-M3
cd /path/to/MiniMax-M3

# 2. Pull just the weight shards (the 869 GB; --exclude="" overrides any default excludes; skips the figure)
git lfs pull --include="*.safetensors" --exclude=""
```

## Method B — huggingface CLI (alternative)

```bash
pip install -U "huggingface_hub[cli]"
hf download MiniMaxAI/MiniMax-M3 --local-dir /path/to/MiniMax-M3
```

## What our TT path needs from the checkpoint
- the 59 `model-*-of-00059.safetensors` (the weights)
- `model.safetensors.index.json` (shard map)
- `config.json` (VL-wrapped; `ModelArgs` unwraps `text_config`)
- `configuration_minimax_m3_vl.py` (for `AutoConfig(trust_remote_code=True)`)
- the tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`)

Method A's clone grabs everything small; `git lfs pull` adds the weights. Method B grabs it all.
(We do NOT need the multimodal tensors — `vision_tower` / `multi_modal_projector` / `patch_merge_mlp`
— the loader drops them.)

## Use it
```bash
export HF_MODEL=/path/to/MiniMax-M3     # any "MiniMax-M3*" dir name works (Step A assert allows the prefix)
```

## Verify the download finished (not LFS stubs)
```bash
du -sh /path/to/MiniMax-M3                                    # expect ~870 GB
find /path/to/MiniMax-M3 -name "model-*.safetensors" -size +1G | wc -l   # expect 59 (stubs are ~135 B)
```

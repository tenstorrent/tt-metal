# Exact Hugging Face reference

This directory records the CPU reference for `Qwen/Qwen3.8-Flash-Next` at revision
`812d96e10a0dd82b69d749e68693698cf6c36daa`. The reference uses all 48 original HF
text decoder layers and original checkpoint BF16 weights. Vision and MTP do not
participate in the text autoregressive path.

The available host RAM cannot hold the entire 360 GB checkpoint. The repo-local
`tests/full_model_hf_reference.py` bootstraps the installed, unmodified
`readiness_check.generate` CLI with a memory-safe model factory: non-PLE tensors
are file-backed safetensors mappings; PLE embedding lookup reads the exact
requested BF16 checkpoint rows. The original HF text model, decoder, PLE,
attention, routing, cache, and LM-head forward implementations remain unchanged.
The loader does not quantize, cast, clone, offload model arithmetic to TT, or
shorten the model. `loader.json` checks the complete state-dict contract and
reports the bytes mapped. File pages may be evicted by the OS; CPU runtime is
therefore sensitive to storage and concurrent host use.

The pinned snapshot initially contained 70 broken shard links, corresponding to
184,268,773,096 logical bytes. Calling `hf_hub_download` for those files at the
pinned revision restored them from the existing shared content-addressed cache.
The LM-head shard was restored first. No existing evidence was removed.
`download_plan.json`, `download.log`, and `capacity.json` record the inventory.
The hub's `main` revision was explicitly checked against the pinned SHA before
fetching `config.json` to establish the normal cached revision alias.

Run from the tt-metal checkout:

```bash
source python_env/bin/activate
source /home/ttuser/qwen38-flash/lane-env.sh
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HUB_OFFLINE=1 timeout 10800 \
  python -u models/autoports/qwen_qwen3_8_flash_next/tests/full_model_hf_reference.py \
  --hf-model Qwen/Qwen3.8-Flash-Next --prompt-source aime24 --chat-template \
  --gen-len 100 --top-k 100 \
  --model-dir models/autoports/qwen_qwen3_8_flash_next \
  --output models/autoports/qwen_qwen3_8_flash_next/readiness_aime24_chat.refpt \
  > models/autoports/qwen_qwen3_8_flash_next/doc/full_model/hf_reference/generate.log 2>&1
```

The bootstrap forwards those arguments unchanged to the installed runner using
`runpy.run_module("readiness_check.generate", run_name="__main__")`. The packaged
runner owns prompt rendering, greedy continuation, top-100 extraction, reference
format and cache lookup/write. `--model-dir` enables its normal exact-match
reference cache. The installed plugin is read-only and has not been modified.

`qualitative_prompt_format.json`, `aime24_prompt.txt`, and `aime24_tokens.json`
record the chat template and 203-token rendered AIME24 prompt. Greedy reference
settings differ deliberately from the model's normal sampled recipe (temperature
1.0, top-k 20, top-p 0.95); matched sampled controls must use that recipe on both
HF and TT before interpreting a qualitative failure. A token-cap stop is not an
EOS quality gate. This reference alone is not a TT accuracy or coherence verdict.

The run completed successfully: `result.json` records the 100-by-100 top-k
reference and timing. `aime24_completion.txt` contains the actual 100-token HF
continuation. It is coherent English reasoning restating the AIME problem, with
no mechanical repetition or wrong-language drift in the observed prefix. It
ends at the requested token cap; no mathematical-solution verdict is claimed.

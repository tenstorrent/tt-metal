# vLLM model bundles for repo-local autoports

Each subdirectory is a TT vLLM plugin model bundle: a folder holding a
`vllm_metadata.json` with

- `arch` — the HuggingFace architecture name from the checkpoint's `config.json`
- `main_class` — `"module:Class"` implementing the vLLM generator adapter

The TT plugin discovers every bundle under `EXTRA_MODELS_DIR` and registers it
**before** its own built-in model map, so an implementation living in this
checkout can be served with **no source change to the plugin repo**. From the
plugin's own comment: *"Runs first so a distributed bundle can supply a model
without touching this file."*

## Why this exists for Qwen3.6/3.8-27B

`Qwen3_5ForConditionalGeneration` resolves in the plugin's built-in map to
`models.demos.blackhole.qwen36.tt.qwen36_vllm`. A server started without a
bundle therefore serves the *demo*, which silently invalidates any autoport
release report — stage 11 rules such a report invalid even when `run.py` exits 0.

The alternative fix is a plugin patch adding `qwen36_autoport` to the
`TT_QWEN35_TEXT_VER` selector. That selector exists upstream but whitelists only
`qwen36_blackhole` and **raises** on any other value:

```python
raise ValueError(f"Unsupported TT Qwen3.5 version: {qwen35_text_version}, "
                 "pick one of [qwen36_blackhole]")
```

so setting `TT_QWEN35_TEXT_VER=qwen36_autoport` hard-fails at registration. And
`tenstorrent/vllm-tt-plugin` is not writable from this account (`"push": false`),
so the patch cannot be landed here. Bundles avoid that entirely: they live in
this repo, which is pushable.

## Usage

Point `EXTRA_MODELS_DIR` at this directory and leave `TT_QWEN35_TEXT_VER` unset
(or `qwen36_blackhole`). Registration order does the rest: the bundle claims
`TTQwen3_5ForConditionalGeneration` first, and the built-in map's
`_register_model_if_missing` then skips it.

```bash
EXTRA_MODELS_DIR=<checkout>/models/autoports/vllm_bundles
```

Verify positively rather than assuming — the failure mode is silent.

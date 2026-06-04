# Vendored ACE-Step-1.5 (`acestep` package)

This folder contains a **filtered copy** of the upstream
[ACE-Step-1.5](https://github.com/ACE-Step/ACE-Step-1.5) `acestep` Python package, vendored so the
tt-metal demo (`run_prompt_to_wav.py` and friends) can run **without** an external
`--ace-step-repo-root` checkout. Everything required to instantiate
`acestep.handler.AceStepHandler` and call `initialize_service`, `preprocess_batch`,
`prepare_condition`, and the various `infer_*` / `_decode_audio_codes_to_latents` hooks the demo
overrides on TTNN is present.

## How it gets imported

The demo's `_ensure_acestep_on_path()` prepends this folder to `sys.path`, so plain
`import acestep` resolves to `_vendored_acestep/acestep/`. If the user passes
`--ace-step-repo-root /path/to/external` the external copy takes precedence (legacy escape hatch);
otherwise the vendored tree is used.

## What is included

| Subtree | Included? | Notes |
|---|---|---|
| Top-level modules: `handler.py`, `inference.py`, `gpu_config.py`, `model_downloader.py`, `audio_utils.py`, `constants.py`, `constrained_logits_processor.py`, `debug_utils.py`, `llm_inference.py`, `llm_backend_compat.py`, `local_cache.py`, `__init__.py` | **Yes** | All non-test, non-CLI top-level modules. |
| `core/audio/`, `core/generation/`, `core/llm/`, `core/lora/`, `core/scoring/`, `core/system/` | **Yes** | The full inference / handler / scoring stack. |
| `models/{base,common,turbo,sft,xl_base,xl_sft,xl_turbo}/` | **Yes** | DiT model definitions and configs (needed for HF `trust_remote_code` resolution at checkpoint load). |
| `genres_vocab.txt` | **Yes** | Asset used by `prepare_condition`. |
| `LICENSE` | **Yes** | MIT — copy of upstream license. |

## What is **excluded** (and why)

| Subtree / file | Why removed |
|---|---|
| `*_test.py`, `_*_test_support.py` | Tests; not needed at runtime, and they pull pytest / heavy fixtures. |
| `api/`, `api_server.py` | REST server, unused by the demo. |
| `ui/` | Gradio web UI, unused. |
| `training/`, `training_v2/` | Fine-tuning / training scripts, unused for inference. |
| `text_tasks/` | Stand-alone text utilities, unused. |
| `third_parts/` | Bundled third-party deps (e.g. `nano-vllm`); we don't use the upstream LM backend. |
| `openrouter*` | OpenRouter LLM proxy, unused. |
| `dataset/`, `dataset_handler.py` | Data loading utilities for training. |
| `mlx_*.py`, `models/mlx/` | Apple Silicon / MLX backend, unused on Tenstorrent. |
| `cli_args.py`, `launcher*.py`, `acestep_v15_pipeline*.py` | Entry-point CLI / launcher / pipeline wrapper, unused (the demo has its own CLI). |
| `__pycache__/` | Build artifacts. |

## Refreshing this vendor

When you bump the ACE-Step-1.5 commit you're tracking, refresh the vendor tree with:

```bash
DEST=models/experimental/ace_step_v1_5/torch_ref/_vendored_acestep
SRC=/path/to/ACE-Step-1.5/acestep
rm -rf "$DEST/acestep"
mkdir -p "$DEST/acestep"
rsync -a \
  --exclude='__pycache__/' \
  --exclude='*_test.py' \
  --exclude='_*_test_support.py' \
  --exclude='api/' \
  --exclude='api_server.py' \
  --exclude='ui/' \
  --exclude='training/' \
  --exclude='training_v2/' \
  --exclude='text_tasks/' \
  --exclude='third_parts/' \
  --exclude='openrouter*' \
  --exclude='dataset/' \
  --exclude='dataset_handler.py' \
  --exclude='mlx_*' \
  --exclude='models/mlx/' \
  --exclude='cli_args.py' \
  --exclude='launcher*' \
  --exclude='acestep_v15_pipeline*' \
  "$SRC/" "$DEST/acestep/"
cp /path/to/ACE-Step-1.5/LICENSE "$DEST/LICENSE"
```

## License

The vendored sources retain their MIT license — see `LICENSE` next to this file. tt-metal's own
license (Apache-2.0) applies to the rest of this repository.

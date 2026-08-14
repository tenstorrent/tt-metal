# vLLM model bundles for repo-local autoports

Each subdirectory is a TT vLLM plugin model bundle: a folder containing a
`vllm_metadata.json` with

- `arch` — the HuggingFace architecture name from the checkpoint's `config.json`
- `main_class` — `"module:Class"` implementing the vLLM generator adapter

The TT plugin discovers every bundle under `EXTRA_MODELS_DIR` and registers it
**before** its own built-in model map, so an implementation living in this
checkout can be served with **no source change to `tenstorrent/vllm`**. From the
plugin's own docstring: "Any distribution tool can drop a bundle folder here and
have it registered with no source edit to this plugin."

## Why this exists

Gemma4 architectures resolve to `models.demos.gemma4.tt.generator_vllm` in the
plugin's built-in map. A server started against a base `google/gemma-4-31B`
checkout without a bundle therefore comes up cleanly and serves the *demos*
implementation, which silently invalidates any autoport release report — Stage 11
rules such a report invalid even when `run.py` exits 0.

The previous fix was a patch to the plugin's `platform.py` adding a
`TT_GEMMA4_TEXT_VER` selector. That patch lived only in a local checkout, was
never pushed, and is absent from `tenstorrent/vllm` (`git ls-remote` and the
commit API both confirm). Bundles avoid that failure mode entirely because they
live here, in a repo we can push.

## How selection works

`TTPlatform.check_and_update_config` prepends `TT` to each architecture the
checkpoint declares and then requires a matching TT-prefixed registration. Bundle
registration applies the same prefix (`tt_arch = arch if arch.startswith("TT")
else "TT" + arch`), and the built-in map uses `_register_model_if_missing`, so a
bundle registered first wins for that name while the built-in bare-name entries
still prevent vLLM's multimodal-fallback crash.

So `arch: "Gemma4ForConditionalGeneration"` here becomes
`TTGemma4ForConditionalGeneration -> models.autoports.google_gemma_4_31b...`,
which is what actually loads. No `--hf-overrides` is required.

Verified on pristine upstream vLLM `dev` (`bf98d55`), no plugin patch, on
2x Blackhole p300:

```text
Registered TT model TTGemma4ForConditionalGeneration ->
  models.autoports.google_gemma_4_31b.tt.generator_vllm:Gemma4ForCausalLM
  (from EXTRA_MODELS_DIR/gemma4_31b_autoport)
Prefix caching is not supported in TT backend for
  models.autoports.google_gemma_4_31b.tt.generator_vllm, disabling it
```

## Usage

```bash
export EXTRA_MODELS_DIR=$TT_METAL_HOME/models/autoports/vllm_bundles
export GEMMA4_31B_AUTOPORT_DIR=$TT_METAL_HOME/models/autoports/google_gemma_4_31b
```

`main_class` is a full dotted path, so tt-metal on `PYTHONPATH` is sufficient to
import it and the bundle folder needs no packaging.

`TT_VLLM_BUILTIN_MODELS=0` additionally disables the built-in map entirely — the
plugin calls that "the intended end-state once all models ship as bundles". Use
it when you want a hard guarantee that no other implementation is registered;
note it also unregisters every other TT model in the image.

tt-inference-server sets `EXTRA_MODELS_DIR` automatically from the resolved
tt-metal checkout when this directory exists (see `workflows/run_local_server.py`
on branch `mvasiljevic/fast-models-fast/gemma4-31b`).

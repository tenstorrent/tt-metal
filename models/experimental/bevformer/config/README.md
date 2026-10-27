# BEVFormer model configs

The bundled JSON files are the validated inference-facing equivalents of upstream's
`projects/configs/bevformer/bevformer_{tiny,small,base}.py`. They contain the architecture fields,
image preprocessing and ordered camera contract consumed by this port; training-only optimizer,
loss, schedule and dataset-path settings are intentionally omitted.

`model_config.py` remains a compatibility facade for existing reference, TTNN and test imports. It
loads `bevformer_base.json` by default, or the bundled file selected by
`BEVFORMER_VARIANT=tiny|small|base`. The sequence demo activates the config supplied through its
`--config` argument before importing the model.

The loader accepts a local JSON file or an HTTPS URL. Remote configs are cached and may be pinned
with `--config-sha256`. The demo also recognizes official-style `.py` sources and sends them through
the restricted converter below; Python configs downloaded from a URL are never executed.

An official-style Python config can be translated to normalized JSON without importing or
executing it. The converter interprets only a small, data-oriented AST subset and rejects configs
whose model dictionary cannot be read statically:

```sh
python -m models.experimental.bevformer.config.convert_upstream \
  --input https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py \
  --output /tmp/bevformer_tiny.json
```

Pin remote source contents with `--sha256` for a reproducible conversion. Fields that upstream
keeps outside the model config, notably the learned camera order contract, come from the matching
bundled variant and are written explicitly to the normalized result.

Camera names and their order belong here because the model has learned per-camera embeddings.
Camera intrinsics and extrinsics do not: real calibration is stored per frame in the demo manifest
as `lidar2img`, while deterministic approximate rigs used by tests live in `tests/fixtures/`.

# BEVFormer model configs

The bundled JSON files are the validated inference-facing equivalents of upstream's
`projects/configs/bevformer/bevformer_{tiny,small,base}.py`. They contain the architecture fields,
image preprocessing and ordered camera contract consumed by this port; training-only optimizer,
loss, schedule and dataset-path settings are intentionally omitted.

| Variant | Bundled config | Upstream counterpart | Pre-pad image | Backbone | FPN levels | BEV grid | Encoder layers |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| tiny | [`bevformer_tiny.json`](bevformer_tiny.json) | [`bevformer_tiny.py`](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py) | 800 x 450 | ResNet-50 | 1 | 50 x 50 | 3 |
| small | [`bevformer_small.json`](bevformer_small.json) | [`bevformer_small.py`](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_small.py) | 1280 x 720 | ResNet-101 DCN | 1 | 150 x 150 | 3 |
| base | [`bevformer_base.json`](bevformer_base.json) | [`bevformer_base.py`](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py) | 1600 x 900 | ResNet-101 DCN | 4 | 200 x 200 | 6 |

All three variants use six cameras, 256 embedding dimensions, six decoder layers, 900 object
queries and the same ten nuScenes classes. The image sizes above are after `image_scale` and before
padding to the configured size divisor.

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

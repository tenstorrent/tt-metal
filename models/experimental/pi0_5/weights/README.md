# weights/

Checkpoints are **not** tracked in git. Use the download script to fetch and
prepare the upstream openpi **pi05_libero** checkpoint in the torch/safetensors
layout this package expects.

```bash
# gated repo → authenticate first
huggingface-cli login          # or: export HF_TOKEN=...

python_env/bin/python models/experimental/pi0_5/weights/download_pi05_libero.py \
  --out $HOME/pi05_cache/pi05_libero_upstream
export PI05_CHECKPOINT_DIR=$HOME/pi05_cache/pi05_libero_upstream
```

The script:
1. Downloads `model.safetensors` + `config.json` + `assets/` from the HF torch
   mirror (`--repo-id`, default `openpi/pi05_libero`). This mirror is openpi/lerobot's
   JAX→PyTorch conversion of the canonical Orbax checkpoint — i.e. the "convert to
   torch" step already applied.
2. Ensures `config.json` (writes the 5-key header with `action_horizon=10` if the
   repo omits it — without it `from_checkpoint()` wrongly defaults to 50).
3. Ensures `assets/physical-intelligence/libero/norm_stats.json` (fetches from the
   public GCS bucket if missing).
4. **Verifies** the result with this package's own `Pi0_5WeightLoader` +
   `action_horizon_from_checkpoint` — the definitive "it works" check.

Resulting layout:
```
<out>/model.safetensors                                     ~7.2 GB bf16
<out>/config.json                                           {action_dim, action_horizon=10, ...}
<out>/assets/physical-intelligence/libero/norm_stats.json
```

**JAX/Orbax only?** If you can only reach the canonical Orbax checkpoint
(`gs://openpi-assets/checkpoints/pi05_libero/`, no config.json), convert it with
openpi's exporter first, then re-run with `--skip-download` to add
config.json/norm_stats + verify. See the header of `download_pi05_libero.py`.

## Checkpoint tensor-name contract

For a checkpoint to load in this package, the **expert** must contain the adaRMS
modulation tensors per layer:

```
model.layers.{i}.input_layernorm.dense.weight     # (3 * width, width)
model.layers.{i}.input_layernorm.dense.bias       # (3 * width,)             optional
model.layers.{i}.post_attention_layernorm.dense.weight
model.layers.{i}.post_attention_layernorm.dense.bias
```

…and the **suffix** must contain `time_mlp_in.{weight,bias}` / `time_mlp_out.{weight,bias}`
in addition to `action_in_proj` and `action_out_proj`. `state_proj` and
`action_time_mlp_*` from PI0 are **not** used.

If your checkpoint uses different names, add a rename pass in
`Pi0_5WeightLoader.state_dict` (already strips lerobot's `model.` prefix automatically).

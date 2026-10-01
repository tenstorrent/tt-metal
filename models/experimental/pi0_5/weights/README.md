# weights/

Checkpoints are **not** tracked in git.

## pi05_base — single-chip PCC/perf

Model checkpoint: **`gs://openpi-assets/checkpoints/pi05_base`** (openpi JAX/Orbax, public
bucket, ~12.4 GB). The single-chip PCC/perf numbers are measured with this checkpoint — use
it, not the HF `lerobot/pi05_base` mirror. [`download_pi05_base.sh`](download_pi05_base.sh)
converts it to torch safetensors with openpi's own exporter, in one command:

```bash
models/experimental/pi0_5/weights/download_pi05_base.sh
# PI05_CACHE=<dir>  work/output dir, outside the repo (default $HOME/pi05_cache)
# KEEP_JAX=1        keep the downloaded JAX checkpoint (default: deleted after conversion)
export PI05_CHECKPOINT_DIR=$HOME/pi05_cache/pi05_base
```

Needs `curl`, `git`, `uv`, `python3`, ~27 GB free disk while converting (~14.5 GB after) and
~44 GB RAM (CPU-only conversion, ~1 min). The script:

1. downloads the Orbax checkpoint from the public bucket via `curl` (resumable, sizes verified);
2. clones openpi (pinned commit) into `$PI05_CACHE/openpi`, runs `uv sync`, and applies
   openpi's `transformers_replace` patch (AdaRMS) — with `UV_LINK_MODE=copy`, so the patch
   stays inside openpi's `.venv` (uv's default hardlink mode would also patch the shared uv cache);
3. runs `examples/convert_jax_model_to_pytorch.py --config_name pi05_aloha --precision float32`;
4. verifies the result with this package's `Pi0_5WeightLoader` (horizon 50, adaRMS + `time_mlp_*` tensors);
5. links it at `weights/pi05_base` (the PCC/perf test default) and deletes the JAX checkpoint.

Exporter details the script handles:

- `--config_name pi05_aloha` is `Pi0Config(pi05=True)` with all defaults, so `config.json`
  gets `action_horizon=50` (pi05_base's horizon). The weights don't depend on the config.
- `--checkpoint_dir` must contain `pi05` — the exporter picks the adaRMS weight mapping by
  that substring.
- `--precision float32`: the Orbax params are fp32; the exporter's default `bfloat16` rounds them.
- The exporter looks for `assets/` next to `--checkpoint_dir`, not inside it, so the script copies them.

Result: `model.safetensors` (~14.5 GB fp32, 812 tensors, no `model.` prefix), `config.json`,
`assets/<robot>/norm_stats.json`.

## pi05_libero — LIBERO task success

Use the download script to fetch and prepare the upstream openpi **pi05_libero**
checkpoint in the torch/safetensors layout this package expects.

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

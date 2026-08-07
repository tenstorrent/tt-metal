# Wan2.2 T2V-A14B Style LoRA (Tenstorrent)

Train a style LoRA on [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)
with a flow-matching objective on Tenstorrent hardware, and sample from it through
tt_dit's runtime-LoRA pipeline.

This is a port of a CUDA/diffusers reference pipeline. Stage boundaries, flag names,
cache format, and training semantics are kept identical so the two runs are directly
comparable — see [Parity](#parity-with-the-cuda-reference).

> **Status:** the four stages, LoRA plumbing, and adapter export are in place. The
> `train` stage additionally requires the ttml port of the Wan DiT
> (`ttml.models.wan2_2`), which is **not implemented yet** — `preprocess`,
> `precompute`, and `infer` work without it.

## Model

A14B is a 2-expert MoE with two-stage denoising:

| Expert | HF subfolder | Timestep range |
|---|---|---|
| high-noise | `transformer` | `t >= 0.875` |
| low-noise | `transformer_2` | `t < 0.875` |

A style LoRA usually only needs the low-noise expert (`train_experts: low`), which also
keeps a single 14B expert resident instead of both.

## Usage

Every knob lives in one YAML file —
[`configs/training_configs/wan2_2_t2v_a14b_lora.yaml`](../../../configs/training_configs/wan2_2_t2v_a14b_lora.yaml).
The CLI takes a stage, an optional config path, and nothing else.

Each stage is its own process. `precompute` and `infer` drive ttnn/tt_dit; `train`
drives ttml — the two frameworks cannot hold the device at the same time.

```bash
export TT_METAL_HOME=/path/to/tt-metal
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto
cd $TT_METAL_HOME/tt-train/sources/examples/lora_wan2_2

python pipeline.py preprocess   # OmniConsistency style subset -> images + captions
python pipeline.py precompute   # VAE latents + UMT5 embeddings, encoded on the mesh
python pipeline.py train        # LoRA flow-matching training
python pipeline.py infer        # sample a clip from the trained adapter
```

Use a different config with `-c`, and override individual values without editing it
using `--set KEY=VALUE` (repeatable, uppercase field names):

```bash
python pipeline.py train -c my_run.yaml --set MAX_STEPS=2500 --set MESH_SHAPE=1,8 --set GRAD_CLIP=0
```

Unknown keys and out-of-range values are rejected at load time rather than failing
part-way into a run.

### Equivalent of the CUDA invocation

```bash
uv run python src/wan2_2_14b/pipeline.py train \
    --cache-dir cache/wan22_14b_lg --experts both --max-steps 2500 \
    --device-map auto --lora-path cache/wan22_14b_lg/lora.safetensors
```

becomes, in the YAML: `data.cache_dir: cache/wan22_14b_lg`, `model.train_experts: both`,
`training.max_steps: 2500`, `lora.lora_path: cache/wan22_14b_lg/lora.safetensors`,
`device.mesh_shape: [1, 8]` (the analogue of `--device-map auto`), and
`optimizer.grad_clip: 0` (see [known differences](#known-differences)). It writes
`cache/wan22_14b_lg/lora_high.safetensors` and `lora_low.safetensors`.

`train_experts: both` keeps **both** 14B experts resident — ~56 GB of bf16 weights, so
it needs the weights sharded across the mesh (`[1, 8]` on a Loud Box, wider on a
Galaxy). Only the expert selected by the sampled timestep receives gradients on a given
step. `low` halves the resident footprint if memory is tight; a style LoRA often only
needs the low-noise expert.

## Layout

| File | Role |
|---|---|
| `pipeline.py` | the four-subcommand CLI |
| `pipeline_config.py` | `Config` dataclass + YAML loader, `--set` overrides, validation |
| `preprocess.py` | stage 1: dataset download + caption pairing (no device work) |
| `precompute.py` | stage 2: VAE + UMT5 encode -> cache, on the mesh |
| `utils/tt_encoders.py` | tt_dit `WanEncoder` + `TextEncoder` driven on device for stage 2 |
| `train.py` | stage 3: LoRA flow-matching training loop (ttml, on device) |
| `infer.py` | stage 4: sampling via tt_dit `WanPipelineRuntimeLoRA` |
| `utils/dataset.py` | cache dataset + CFG-dropout collate |
| `utils/lora_targets.py` | regex target sets for `ttml.modules.LoraConfig` |
| `utils/lora_export.py` | adapter save/load in PEFT/diffusers key format |
| `utils/device_setup.py` | ttml mesh setup |
| `utils/logger.py` | wandb with stdout fallback |

Cache layout is unchanged from the reference, so the training stage reads either
source. Note the *values* differ: latents encoded on the mesh land at ~0.995 PCC
against the torch VAE, so a cache built here is not bit-identical to a CUDA-built one.

```
<CACHE_DIR>/samples/sample_%04d.pt   {"latent": (C,F,H,W), "caption": str}
<CACHE_DIR>/embeds.pt                {caption: (MAX_SEQ, 4096)}, includes ""
<CACHE_DIR>/metadata.json            [{"idx", "caption"}, ...]
```

## Adapter format

Adapters are written with PEFT/diffusers keys — `transformer.blocks.0.attn1.to_q.lora_A.weight`,
2-D tensors, float32 — so one file loads three ways with no conversion:

- **diffusers**: `pipe.load_lora_weights(load_file(path))` (use `load_into_transformer_2=True` for the low-noise file)
- **tt_dit**: `register_lora(...)` at runtime, or `fuse_lora_state_dict` for CPU-side fusion
- **this example**: resume/eval

`utils/lora_export.py` bridges three naming differences between the ttml tree and the
checkpoint: `to_out` → `to_out.0`, `ffn.ff1` → `ffn.net.0.proj`, `ffn.ff2` → `ffn.net.2`.

## Config

The YAML is fully commented; these are the entries worth knowing before a first run.

| Key | Default | Description |
|---|---|---|
| `model.train_experts` | `both` | `low` / `high` / `both` — which MoE expert(s) to adapt |
| `training.max_steps` | 3000 | optimizer steps (batch 1 × grad-accum 4) |
| `lora.lora_rank` / `lora_alpha` | 32 / 32 | scaling is `alpha/rank`, `use_rslora=False` |
| `lora.lora_target_set` | `attn` | `attn` / `attn+ffn` — see the note below |
| `optimizer.grad_clip` | 1.0 | must be `0` when `mesh_shape` has TP > 1 |
| `device.mesh_shape` | `[1, 1]` | `[DP, TP]` for train; VAE height/width factors for precompute |
| `logging.wandb_enabled` | true | false logs to stdout only |

Sections are cosmetic — a key is matched against the `Config` field of the same name
(uppercased) wherever it appears — but a mistyped *section* is rejected so a whole
block cannot be silently ignored.

## Parity with the CUDA reference

Timestep sampling, noise, and `x_t` are built on the host **with torch, using the
reference's generators and seeds**, then uploaded — so given the same latents, the
tensors entering the model are bitwise identical. Only model math runs on device.

- **Step-level (the strong gate):** dump `(t, noise, latent, text_embed)` from one CUDA
  `flow_matching_step` and replay those exact tensors here; losses should agree to bf16
  tolerance. This works because the inputs are injected rather than sampled, so it does
  not depend on how the cache was built.
- **Curve-level (approximate):** same seed, ~200 steps on both; compare `train/loss_ema`
  and `val/loss` (validation noise is seeded per sample index, so it is deterministic on
  both sides). Expect small systematic offset — the device VAE's latents differ from the
  torch VAE's, so the two runs are fitting slightly different targets. Encode the cache
  with diffusers on a CUDA box if you need this comparison to be exact.
- **Adapter portability:** train here, then load the file with diffusers and with
  tt_dit; tt_dit's aggregate L2 check confirms the base weights actually changed.

### Known differences

1. **`attn` vs `attn+ffn` targets.** The reference's target list contains
   `"ff.net.0.proj"`, but Wan's feedforward module is named `ffn`, so under PEFT's
   name-suffix matching it never matched `...ffn.net.0.proj` — the FFN was silently
   left un-adapted, and `assert 0 < trainable < total // 20` still passed on the
   attention targets alone. The default here (`attn`) reproduces that *effective*
   behaviour; `--lora-targets attn+ffn` is the *intended* one. `ttml.modules.LoraModel`
   raises when a pattern matches nothing, so this class of typo cannot pass silently here.
2. **LoRA A init.** PEFT's `init_lora_weights="gaussian"` vs ttml's kaiming-uniform
   `_create_lora_A`. Same distribution family, different scale — affects the early
   loss curve, not final quality (`LORA_A_INIT` documents the knob).
3. **Gradient clipping under TP** — the one behavioural gap in the normal
   configuration. The reference clips against the true global norm; `ttml.core.clip_grad_norm`
   computes a per-device norm with no cross-mesh reduction, so at TP>1 it would clip
   against shard-local norms. `train.py` refuses that combination rather than clip
   incorrectly, so TP runs need `--grad-clip 0`.

   A correct global norm is implementable and worth doing: each LoRA pair has exactly
   one sharded half (`lora_B` for column-parallel, `lora_A` for row-parallel) whose
   shards *partition* its elements, so summing per-device sums-of-squares and
   all-reducing over the TP axis is exact, with the replicated half counted once. It
   needs a scalar all-reduce that ttml does not currently expose.

## References

- `models/tt_dit/models/Wan2_2.md` — base architecture, supported meshes, perf
- `models/tt_dit/experimental/models/Wan2_2_LoRA.md` — adapter key formats and fusion
- `tt-train/docs/DISTRIBUTED_TRAINING.md` — MGD files and mesh setup

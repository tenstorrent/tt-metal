# Running Wan 2.2, SD 3.5 and SDXL on Blackhole Galaxy

Exact, copy-pasteable invocations for the three diffusion demos. Each section is
self-contained — open a fresh shell per model, since the environment variables
(notably `HF_HOME` and the cache dirs) differ between them.

---

## 1. Wan 2.2 (T2V) — DBCache A/B

720p, 40 denoising steps, ring topology on a 4x8 submesh.

```bash
cd /home/ttuser/sdawle/jon/tt-metal
source python_env/bin/activate

export TT_METAL_HOME=$PWD
export PYTHONPATH=$PWD
export HF_HOME=/mnt/tt-data/hf-home
export TT_DIT_CACHE_DIR=/mnt/tt-data/nkira/tt_dit_cache_wan
export WAN_DBCACHE_RUNS=dbcache
export TT_METAL_TDP_LIMIT_WATTS=190               # higher perf
unset WAN_DBCACHE_PROMPT TT_METAL_WATCHER HF_HUB_OFFLINE

pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_dbcache.py \
  -k "bh_4x8sp1tp0nl2_ring and 720p" -s
```

**DBCache disabled (baseline)** — same commands, only the run selector changes:

```bash
export WAN_DBCACHE_RUNS=baseline
```

`WAN_DBCACHE_RUNS` accepts a comma-separated list, so `baseline,dbcache` runs
both back to back in one pipeline instance and prints PSNR/PCC of DBCache
against the baseline video. With a single run in the list those columns are
`nan` — there is nothing to compare against.

### Reference numbers (720p, 40 steps)

| run     | denoise (s) | step (s) | cached steps | PSNR vs base | PCC vs base |
|---------|-------------|----------|--------------|--------------|-------------|
| dbcache | 82.7        | 2.04     | 30           | nan          | nan         |

### Prompt

Defaults to:

> Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.

Override with `export WAN_DBCACHE_PROMPT="..."` (the command above `unset`s it
so the default is used).

### Outputs

Written to the current working directory as
`wan_dbcache_<run>_<W>x<H>_s<steps>.mp4` plus a `_strip.png` 5-frame contact
sheet, e.g. `wan_dbcache_dbcache_1280x720_s40.mp4`.

### Useful knobs

| Variable | Default | Meaning |
|---|---|---|
| `WAN_DBCACHE_RUNS` | `baseline,split_nocache,dbcache` | Which runs to execute |
| `WAN_DBCACHE_STEPS` | `40` | Denoising steps |
| `WAN_DBCACHE_RDT` | preset | DBCache residual-diff threshold override |
| `WAN_DBCACHE_PROMPT` | cats-boxing prompt | Prompt text |

Other `-k` selectors: `480p` instead of `720p`; `bh_4x8sp1tp0nl2_traced_ring`
for the traced path; `bh_4x8sp1tp0nl2_linear` / `bh_2x4sp1tp0nl2_linear` for
line topologies.

---

## 2. Stable Diffusion 3.5

20 steps, 2x2 submesh, traced, bf16.

```bash
cd ~/tt-metal-sd35
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_HOME=/mnt/tt-data/hf-home
export TT_LOGGER_LEVEL=Error
unset TT_DIT_CACHE_DIR
export SD35_STEPS=20
export TT_METAL_TDP_LIMIT_WATTS=190               # higher perf
python models/tt_dit/tests/models/sd35/run_sd35_submesh.py
```

> The run script lives at `models/tt_dit/tests/models/sd35/run_sd35_submesh.py`.
> `~/tt-metal-sd35` is a separate checkout of the SD 3.5 work; the same script
> is also present in this tree (`/home/ttuser/sdawle/jon/tt-metal`), so
> substitute that path if the dedicated checkout is not present on the machine.
> Note `TT_DIT_CACHE_DIR` must be **unset** — unlike Wan 2.2, SD 3.5 uses its
> own default cache location.

### Prompt

Defaults to:

> An epic, high-definition cinematic shot of a rustic snowy cabin glowing warmly at dusk,
> nestled in a serene winter landscape. Surrounded by gentle snow-covered pines and delicate
> falling snowflakes — captured in a rich, atmospheric, wide-angle scene with deep cinematic
> depth and warmth.

### Outputs

`sd35_2x2_<tag>_it<N>.png` in the current working directory, where `<tag>`
defaults to `<layout>_<topology>_<quant>_s<steps>` — e.g.
`sd35_2x2_1x4tp_ring_bf16_s20_it0.png`. Two iterations are produced by default
(the first includes warm-up/compile, the second is the steady-state timing).

### Useful knobs

| Variable | Default | Meaning |
|---|---|---|
| `SD35_STEPS` | `28` | Denoising steps |
| `SD35_ITERS` | `2` | Number of image iterations |
| `SD35_TRACED` | `1` | Trace the denoise loop |
| `SD35_CFG` | `1` | Classifier-free guidance |
| `SD35_LAYOUT` | `auto` | Parallel layout |
| `SD35_TOPOLOGY` / `SD35_FABRIC` | `auto` | Ring vs line |
| `SD35_LINKS` | `2` | Fabric links |
| `SD35_T5` | `0` | Enable the T5 text encoder |
| `SD35_TAG` | derived | Output filename tag |

---

## 3. SDXL (Stable Diffusion XL base)

Text-to-image, 1024x1024, 20 steps, VAE and text encoders on device, traced —
the README's canonical invocation.

```bash
cd tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$PWD
export PYTHONPATH=$PWD
export HF_HOME=/mnt/tt-data/nkira/hf              # SDXL weights live under $HF_HOME/hub
export TT_METAL_CACHE=/home/ttuser/ttm_cache_42   # keep JIT kernels off the NFS tree
unset TT_MM_THROTTLE_PERF                         # bare metal: no throttle
export TT_METAL_TDP_LIMIT_WATTS=190               # higher perf
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=10,9

pytest models/demos/stable_diffusion_xl_base/demo/demo.py \
  -k "device_vae and device_encoders and with_trace and no_cfg_parallel and 1024x1024 and steps20"
```

**Single-device run** — same environment, prefix with `TT_VISIBLE_DEVICES`:

```bash
TT_VISIBLE_DEVICES=0 pytest models/demos/stable_diffusion_xl_base/demo/demo.py \
  -k "device_vae and device_encoders and with_trace and no_cfg_parallel and 1024x1024 and steps20"
```

### Prompt

Defaults to:

> An astronaut riding a green horse

with negative prompt `disturbing`.

### Outputs

`output/output<N>.png`, relative to the current working directory (the
directory is created on first run).

### `-k` selector vocabulary

| Axis | Options |
|---|---|
| VAE | `device_vae`, `host_vae` |
| Encoders | `device_encoders`, `host_encoders` |
| Trace | `with_trace`, `no_trace` |
| CFG | `no_cfg_parallel`, `use_cfg_parallel` |
| Resolution | `1024x1024`, ... |
| Steps | `steps20`, `steps50` |

See `models/demos/stable_diffusion_xl_base/README.md` for the full matrix.

---

## Gotchas

- **One shell per model.** `HF_HOME` differs (Wan 2.2 / SD 3.5 use
  `/mnt/tt-data/hf-home`, SDXL uses `/mnt/tt-data/nkira/hf`), as do the cache
  variables. Re-exporting into a dirty shell is the usual cause of "weights not
  found".
- **`TT_DIT_CACHE_DIR`** is set for Wan 2.2 and must be unset for SD 3.5.
- **Outputs land in `$PWD`** for Wan 2.2 and SD 3.5, and in `$PWD/output/` for
  SDXL. Run from a scratch directory if you do not want artifacts dropped into
  the repo root.

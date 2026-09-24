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

### Reference numbers (720p, 40 steps, one video)

Measured 2026-09-24 on a 32-chip BH Galaxy, firmware 19.11.0.0, warm caches.

| TDP   | denoise (s) | step (s) | cached steps |
|-------|-------------|----------|--------------|
| 190 W | 65.1        | 1.60     | 30           |
| 130 W | 82.8        | 2.04     | 30           |

`TT_METAL_TDP_LIMIT_WATTS=190` is worth 21% here, so do not omit it. PSNR/PCC
are `nan` with a single run in `WAN_DBCACHE_RUNS` -- add `baseline` to get them.

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

### Reference numbers (1024x1024, 20 steps)

Measured 2026-09-24 on a 32-chip BH Galaxy, firmware 19.11.0.0, traced, CFG on,
warm caches. Iteration 1 is the steady-state figure; iteration 0 carries warm-up.

**One process** — one image on a 4-chip submesh of the full mesh, layout `1x4tp`,
ring. This is what the command above runs.

| TDP   | encoder (s) | vae (s) | denoising (s) | step (s) | total (s) |
|-------|-------------|---------|---------------|----------|-----------|
| 190 W | 0.04        | 0.20    | 3.89          | 0.194    | **4.13**  |
| 130 W | 0.04        | 0.18    | 4.09          | 0.204    | **4.32**  |

**Eight processes in parallel** — one per 4-chip column, each pinned with
`TT_VISIBLE_DEVICES`, layout `4x1tp`, ring. Mean over the 8 workers, all 32 chips
busy. See "Eight parallel processes" below.

| TDP   | denoising (s) | step (s)      | total, mean (s) | total, range (s) |
|-------|---------------|---------------|-----------------|------------------|
| 190 W | 3.84 - 3.90   | 0.192 - 0.195 | **4.116**       | 4.08 - 4.15      |
| 130 W | 4.03 - 4.13   | 0.201 - 0.207 | **4.339**       | 4.31 - 4.37      |

Running eight in parallel is effectively free: -0.014 s at 190 W and +0.019 s at
130 W against a single process, both inside the 0.07 s spread between workers.
Eight columns do not contend, so throughput is ~8x for the same per-image
latency.

For reference, the tt-inference-server container in its 8-column `DEVICE_IDS`
mode measures 4.53 s (190 W) and 4.68 s (130 W) per image. The ~0.4 s gap over
these numbers is request-path overhead -- queueing, scheduler dispatch, base64
encoding, HTTP -- not device contention.

### Eight parallel processes

One process per 4-chip column, pinned with `TT_VISIBLE_DEVICES`. The columns are
non-contiguous because a column is a slice of the (4, 8) mesh:

```bash
0,4,12,8      1,5,13,9      2,6,14,10     3,7,15,11
27,31,23,19   26,30,22,18   25,29,21,17   24,28,20,16
```

Per process, on top of the single-process environment above:

```bash
export TT_VISIBLE_DEVICES="0,4,12,8"           # one column per process
export TT_METAL_CACHE=/home/ttuser/ttm_cache_8proc/w0   # must be per-process
export SD35_TAG=p0                             # else the PNGs collide
export SD35_LAYOUT=4x1tp SD35_TOPOLOGY=ring
```

Two caveats:

- `TT_METAL_CACHE` **must** differ per process. Eight processes JIT-compiling
  into one cache directory race each other.
- `run_sd35_submesh.py` asserts `full.shape == SystemMeshDescriptor().shape()`.
  Under `TT_VISIBLE_DEVICES` the descriptor reports `(2, 2)` for four visible
  chips while the device opens as `(4, 1)`, so that assert has to be relaxed to
  a warning for this mode. The bare `(4, 1)` itself opens fine on ring fabric --
  the "native 4x1 hangs on Galaxy" note above applies to slicing a column out of
  the full 32-chip mesh, not to four chips opened alone.

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

### Reference numbers (1024x1024, 20 steps, 32 prompts across 32 chips)

Measured 2026-09-24. Note the unit: SDXL runs one full pipeline per chip, so
this is 32 images in the quoted wall time, not one. Divide by 32 for a
per-image figure (~0.13 s at 190 W). Not comparable to the Wan 2.2 or SD 3.5
rows above, and measured with `TT_MM_THROTTLE_PERF` unset.

| TDP   | denoising loop, 32 prompts (s) | image gen, 32 prompts (s) |
|-------|--------------------------------|---------------------------|
| 190 W | 3.27                           | **4.22**                  |
| 130 W | 3.31                           | **4.27**                  |

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

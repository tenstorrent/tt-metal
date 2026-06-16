# Installing Sulphur (self-contained) on a Galaxy or loudbox

Goal: a fully self-contained `sulphur OUTPUT.mp4 "prompt"` command that generates a 6 s
1080p audio-video clip with `SulphurAI/Sulphur-2-base` through tt-metal's LTX-2.3 distilled
pipeline — depending on **nothing** in any individual user's home dir, submitted through the
shared tt-device broker.

This is the exact recipe used for the `/home/sulphur` install on g03blx02, with the
corrections learned doing it. Another agent can follow it top-to-bottom.

---

## 0. What Sulphur actually is (so you don't take a wrong turn)

- Sulphur-2-base is a **fine-tune of `Lightricks/LTX-2.3`** (22B AV DiT). It is **not** a
  standalone model and **not** runnable by `diffusers` on TT.
- `sulphur_distil_bf16.safetensors` (46 GB) is the **full LTX-2.3 distilled bundle**
  (DiT + video VAE + `audio_vae` + `vocoder` + `text_embedding_projection`) with the Sulphur
  fine-tune baked in. Its safetensors keys are **byte-identical** to
  `ltx-2.3-22b-distilled-1.1.safetensors`, so it is a **drop-in checkpoint** for the existing
  pipeline. No model port, no diffusers — just point `LTX_CHECKPOINT` at it.
- The pipeline additionally needs, loaded separately: the **Gemma text encoder**
  (`google/gemma-3-12b-it-qat-q4_0-unquantized`, ~23 GB) and the **LTX spatial upscaler**
  (`Lightricks/LTX-2.3:ltx-2.3-spatial-upscaler-x2-1.1.safetensors`, ~1 GB).

## 1. Prerequisites on the target box

- TT devices present (`/dev/tenstorrent/*`) and the **tt-device-mcp broker** running
  (`tt-device-mcp status` works). All device runs go through it — never bare `pytest`/`tt-smi`.
- A C++ build toolchain for tt-metal (the standard dev image has it).
- **~90 GB free** on a **large data disk** (NOT the OS rootfs). Check `df -h`; pick the mount
  with hundreds of GB free (usually `/home`). Build artifacts are small (~5 GB); the 3 weight
  blobs are the bulk (~70 GB) + clone (~5 GB) + transient build intermediates.
- **Gemma is gated.** You cannot download it without accepting Google's license + an HF token.
  Easiest: copy it from an existing local HF cache on the box (find one with
  `find /home -maxdepth 5 -type d -name 'models--google--gemma-3-12b-it-qat-q4_0-unquantized' 2>/dev/null`).
  Otherwise set `HF_TOKEN` and let step 4 download it.
- Sulphur and the LTX upscaler are **public** — no token needed.

## 2. Parameters — set these for the target box

```bash
ROOT=/home/sulphur                 # install dir (real data; see step 3 for the /homes alias)
PIN=6182a3b34c0                    # PROVEN commit. Do NOT use main (see warning below).
GEMMA_SRC=/home/<someone>/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized
MESH_PARAM=bh_4x8sp1tp0_linear     # match the hardware (see step 6)
```

> **Do NOT clone `main`.** `main` is missing the LTX audio-decode fixes (submesh routing,
> polyphase upsample, mel-VAE blocking, vocoder trace gating). The proven commit `6182a3b34c0`
> is on `origin/ltx-perf` and contains them. A `main`-based install hangs/garbles audio on 4x8.

## 3. Create the install dir on the big disk

If you want it visible at a tidy path like `/homes/sulphur` but the rootfs is small, put the
real data on the big disk and symlink:

```bash
sudo mkdir -p "$ROOT" /homes
sudo ln -sfn "$ROOT" /homes/sulphur          # optional alias; data lives on the big disk
sudo chown "$USER:$USER" "$ROOT"
```

## 4. Clone + pin (self-contained objects)

```bash
# Fast, self-contained clone from a local checkout if one exists, else from GitHub.
if [ -d "$HOME/tt-metal/.git" ]; then
  git clone --no-hardlinks "file://$HOME/tt-metal" "$ROOT/tt-metal"
else
  git clone git@github.com:tenstorrent/tt-metal.git "$ROOT/tt-metal"
fi
cd "$ROOT/tt-metal"
git remote set-url origin git@github.com:tenstorrent/tt-metal.git   # drop any local-path origin
git fetch origin "$PIN" 2>/dev/null || git fetch origin
git checkout -q "$PIN"
```

## 5. Build in place (~30–45 min; do NOT do this on a box at high load)

```bash
cd "$ROOT/tt-metal"
git submodule update --init --recursive
bash build_metal.sh -e
bash create_venv.sh
# REQUIRED — the LTX pipeline's supplemental deps. create_venv.sh does NOT install these,
# and without them the run does ALL the device compute then dies at the final mp4 mux with
# `ModuleNotFoundError: No module named 'av'`. --no-deps keeps the dev env's stable torch/
# safetensors (diffusers 0.38.0 over-declares safetensors but never gates it at import).
source python_env/bin/activate
uv pip install --no-deps -r models/tt_dit/pipelines/ltx/requirements.txt   # av==17.0.1, diffusers==0.38.0
deactivate
# sanity:
TT_METAL_HOME="$ROOT/tt-metal" PYTHONPATH="$ROOT/tt-metal" python_env/bin/python -c "import ttnn, av, diffusers; print('ttnn OK', av.__version__, diffusers.__version__)"
```

> Build is much faster than 30–45 min (≈7 min observed) when `$HOME/tt-metal` was already
> built on the box — the local clone + ccache reuse the prior artifacts.

A copied prebuilt `build_Release`/`python_env` is NOT portable — it embeds absolute paths.
Build in place.

## 6. Copy the weights — self-contained, no token where possible

On btrfs and same filesystem, `cp --reflink=auto` is near-instant and **data-independent**
(deleting the source later does not corrupt the copy).

```bash
mkdir -p "$ROOT/models" "$ROOT/hf/hub" "$ROOT/tt_dit_cache" "$ROOT/tt-metal/generated/test_reports"

# Sulphur distil (public, 46 GB):
HF_HOME="$ROOT/hf" "$ROOT/tt-metal/python_env/bin/python" -c "
from huggingface_hub import hf_hub_download
p=hf_hub_download('SulphurAI/Sulphur-2-base','sulphur_distil_bf16.safetensors')
import shutil; shutil.copyfile(p,'$ROOT/models/sulphur_distil_bf16.safetensors')"

# LTX upscaler (public, ~1 GB) — into the offline HF cache:
HF_HOME="$ROOT/hf" "$ROOT/tt-metal/python_env/bin/python" -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Lightricks/LTX-2.3','ltx-2.3-spatial-upscaler-x2-1.1.safetensors')"

# Gemma (GATED) — copy from an existing cache (preferred):
cp -a --reflink=auto "$GEMMA_SRC" "$ROOT/hf/hub/"
#   …or, with a token instead of GEMMA_SRC:
#   HF_HOME="$ROOT/hf" HF_TOKEN=hf_xxx "$ROOT/tt-metal/python_env/bin/python" -c \
#     "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-12b-it-qat-q4_0-unquantized')"
```

Mesh param (step 2 `MESH_PARAM`): the distilled test is parametrized for `2x2sp0tp1`,
`2x4sp0tp1`, `bh_2x4sp1tp0`, `wh_4x8sp1tp0`, `bh_4x8sp1tp0_linear`, `bh_4x8sp1tp0_ring`,
`bh_4x32sp1tp0`. Pick the one matching the box. A 32-chip Blackhole Galaxy →
`bh_4x8sp1tp0_linear` (linear/untraced is the safe default; ring uses trace); an 8-chip
Blackhole loudbox → `bh_2x4sp1tp0`.

Detect arch + chip count **without touching the device** (don't run bare `tt-smi`): chip count
is `ls /dev/tenstorrent/` (numeric nodes); arch is in `/dev/tenstorrent/by-id` (e.g.
`blackhole-*`) or PCI device id `0xb140` = Blackhole
(`for d in /sys/bus/pci/devices/*; do [ "$(cat $d/vendor)" = 0x1e52 ] && cat $d/device; done`).

## 6b. (Recommended) Build the LoRA variant so both are always available

Sulphur ships a separate `sulphur_lora_rank_768.safetensors` (10 GB, public) — a **video-style
LoRA** delta. Fused onto the Lightricks distilled base it approximates the merged
`sulphur_distil` (CPU analysis: it reconstructs ~85–94% of the Sulphur video-attention delta;
it does not touch the audio branch). Building it as a second standalone checkpoint lets
`sulphur --lora …` run the LoRA variant while plain `sulphur …` runs the merged one — both
always available, no pipeline code change.

```bash
# Download the LoRA (10 GB, public) + the Lightricks distilled BASE it fuses onto (46 GB, public):
HF_HOME="$ROOT/hf" "$ROOT/tt-metal/python_env/bin/python" -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('SulphurAI/Sulphur-2-base','sulphur_lora_rank_768.safetensors'))
print(hf_hub_download('Lightricks/LTX-2.3','ltx-2.3-22b-distilled-1.1.safetensors'))"

# Fuse with the repo's OWN fuse_loras_into (strength 1.0), keeping non-transformer weights from
# the base. CRITICAL: pass metadata= — the audio decoder reads json.loads(f.metadata()["config"]),
# so dropping the safetensors __metadata__ makes the run die with 'NoneType is not subscriptable'.
cd "$ROOT/tt-metal" && PYTHONPATH="$ROOT/tt-metal" HF_HOME="$ROOT/hf" python_env/bin/python - <<'PY'
import glob, os
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from models.tt_dit.utils.fuse_loras import fuse_loras_into, LoraSpec
hub = os.path.expanduser(os.environ["HF_HOME"]) + "/hub"
base = glob.glob(f"{hub}/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-22b-distilled-1.1.safetensors")[0]
lora = glob.glob(f"{hub}/models--SulphurAI--Sulphur-2-base/snapshots/*/sulphur_lora_rank_768.safetensors")[0]
out  = os.environ["ROOT"] + "/models/sulphur_lora_fused_distil.safetensors"
with safe_open(base, "pt") as f: meta = f.metadata() or {}
full = load_file(base); pre = "model.diffusion_model."
tf   = {k[len(pre):]: v for k, v in full.items() if k.startswith(pre)}
rest = {k: v for k, v in full.items() if not k.startswith(pre)}
fused = {pre + k: v for k, v in fuse_loras_into(tf, [LoraSpec(path=lora, strength=1.0)]).items()}
fused.update(rest)
save_file(fused, out, metadata=meta)          # metadata= is mandatory
print("wrote", out)
PY
```

This produces `$ROOT/models/sulphur_lora_fused_distil.safetensors`. The base + LoRA blobs can be
deleted afterward (the fused checkpoint is standalone), or kept to re-fuse at other strengths.
On btrfs the base read is instant if you `cp --reflink=auto` it from an existing cache instead.

## 6c. Other LTX-2.3 drop-in fine-tunes (e.g. TenStrip/LTX2.3-10Eros)

Any HF model whose bf16 checkpoint is **byte-compatible with `ltx-2.3-22b-distilled`** (5947
tensors, identical key set + shapes, `config` in `__metadata__`) runs through this pipeline
unchanged — no code, no conversion. Sulphur and 10Eros are both examples. **Verify compatibility
first, cheaply (header range-read, no 46 GB download):**

```bash
python - <<'PY'   # checks <repo>'s bf16 file against the Lightricks distilled key set
import urllib.request, json, struct
REPO="TenStrip/LTX2.3-10Eros"; FILE="10Eros_v1_bf16.safetensors"   # <-- edit
url=f"https://huggingface.co/{REPO}/resolve/main/{FILE}"
n=struct.unpack("<Q",urllib.request.urlopen(urllib.request.Request(url,headers={"Range":"bytes=0-7"})).read(8))[0]
hdr=json.loads(urllib.request.urlopen(urllib.request.Request(url,headers={"Range":f"bytes=8-{8+n-1}"})).read(n))
ks=[k for k in hdr if k!="__metadata__"]
print("tensors",len(ks),"| config-meta","config" in hdr.get("__metadata__",{}),
      "| has adaln", "model.diffusion_model.adaln_single.linear.weight" in ks)
PY
# Expect: tensors 5947 | config-meta True | has adaln True   -> drop-in. Then download + run:
HF_HOME="$ROOT/hf" "$ROOT/tt-metal/python_env/bin/python" -c "
from huggingface_hub import hf_hub_download; import shutil
shutil.copyfile(hf_hub_download('TenStrip/LTX2.3-10Eros','10Eros_v1_bf16.safetensors'),
                '$ROOT/models/10Eros_v1_bf16.safetensors')"
CKPT_PATH=$ROOT/models/10Eros_v1_bf16.safetensors sulphur out.mp4 "your prompt"
```

`CKPT_PATH=<file>` overrides the checkpoint for any drop-in. The gemma encoder is
checkpoint-independent, so a box with a populated `gemma-…/text_encoder` cache (Known issue #2)
runs any of these variants without re-hitting the gemma hang.

## 7. The launcher

Write `"$ROOT/sulphur"` (chmod 755). It submits through the broker and is fully self-contained
(offline HF, local checkpoint, shared device-weight cache). Copy the working one verbatim:

```bash
cp /home/sulphur/sulphur "$ROOT/sulphur"   # if installing from a box that already has it
# else recreate it — see the reference at the bottom of this file.
sed -i "s#^ROOT=.*#ROOT=$ROOT#" "$ROOT/sulphur"
sed -i "s#-k bh_4x8sp1tp0_linear#-k $MESH_PARAM#" "$ROOT/sulphur"
chmod 755 "$ROOT/sulphur"
```

## 8. World-accessible permissions (shared use)

```bash
chmod -R a+rX "$ROOT"
chmod -R a+rwX "$ROOT/tt_dit_cache" "$ROOT/tt-metal/generated"   # jobs (any user) write here
chmod 755 "$ROOT/sulphur"
# Put it on every user's PATH so `sulphur ...` works from anywhere (no `./`, no cd):
sudo ln -sfn "$ROOT/sulphur" /usr/local/bin/sulphur
# verify another user can read+exec:
sudo -n -u nobody test -r "$ROOT/models/sulphur_distil_bf16.safetensors" && echo readable
```

(If `sulphur` still says "command not found" in a shell that already missed it once, run
`hash -r` or open a new shell — bash caches the failed lookup.)

JIT kernel cache goes to each caller's `~/.cache/ttnn` (per-user) — no shared-write needed.

## 9. Verify

```bash
# Clear any wedged ETH fabric first (see Known Issues), then (works from any dir, any user):
sulphur /tmp/sulphur_test.mp4 "a fluffy corgi DJ at a neon party, upbeat music"
# LoRA variant (needs §6b's sulphur_lora_fused_distil.safetensors):
sulphur --lora /tmp/sulphur_lora_test.mp4 "a fluffy corgi DJ at a neon party, upbeat music"
```

First run is slow: it converts the 46 GB checkpoint and populates `$ROOT/tt_dit_cache`
(watch the per-stage `… loaded from cache in Xs` / `reading source weights …` / watchdog
`… still working, Ns elapsed` lines). **Every subsequent run is a fast cache hit.** Output is
a playable H.264 1080p + AAC MP4 at the path you gave.

---

## Known issues (real, observed on g03blx02 — tell the user, don't paper over)

1. **ETH fabric flakiness.** On a 4x8 Galaxy the inter-chip ETH links sometimes fail to train
   (`Device N: timed out waiting for active ethernet core 27-25` / `tt_fabric::ControlPlane`).
   Reset via the broker — never bare `tt-smi`: **`tt-device-mcp reset`** (refused if a foreign
   tenant holds the device; kill your own queued/running job first with `tt-device-mcp kill`).
   On a Galaxy this runs the IPMI `-glx_reset` and re-inits all 32 chips (~60 s). It usually
   trains the links up but is non-deterministic; may need a repeat. A full power-cycle is the
   hard fix if resets won't hold. Triage with `tools/tt-triage.py --dev=all --run=check_eth_status`.
   **Confirmed recovery on g03blx04:** `tt-device-mcp kill <job>` → `tt-device-mcp reset` →
   resubmit; the post-reset run passed.
   **Reset is uid-gated, not agent-gated — and `status` races.** The gate only refuses *foreign-uid*
   holders, so it will silently kill a *concurrent agent's* job running under your own uid. And
   `tt-device-mcp status` is a snapshot: a job can start in the gap between your check and the reset
   (observed — killed two other agents' runs this way). Re-check `status` *immediately* before any reset
   and abort if anything is RUNNING (yours or not). Better: rely on the op-timeout (launcher reference
   below) so hung jobs self-abort and you rarely need a reset at all.
2. **GIL-held hang on first run (4x8).** Observed twice: the Gemma text-encoder convert on
   first cache-miss load, AND mid-denoise (2nd pass, `Step 1/3`). Signature: worker pinned at
   ~135–175% CPU, RSS flat, zero new output — the GIL is held so the in-process watchdog can't
   surface it. Once the device-weight cache is populated this path is skipped. Remedy: kill and
   rerun — the `tt_dit_cache` + JIT kernel cache from the partial run persist, so the retry is
   faster and the (non-deterministic) hang usually doesn't recur. f07cs04 (8-chip) never hit
   either hang; it's 4x8/Galaxy-specific.
   **On g03blx02 it did NOT clear** — hung on the gemma cache-miss convert across ~8 reset+reruns,
   deterministically. Confirmed a real device-write hang, not compilation, via the doc's own test:
   0 cache-file churn + worker pinned ~350% CPU + RSS flat + C-stack in `finish_nolock` (waiting on a
   sharded mesh write that never completes). The confirmed *successes* are g03blx04 + f07cs04.
   **Best sidestep when a box won't clear it:** the hang is only on the gemma *cache-miss* convert; once
   `$ROOT/tt_dit_cache/gemma-…/text_encoder/TP8_1_mesh4x8_bf16` exists, that path is skipped (cache-hit).
   That cache is mesh-keyed and data-independent — copy it from a box where the convert succeeded (e.g.
   `cp -a --reflink=auto <good-box>/tt_dit_cache/gemma-*/text_encoder $ROOT/tt_dit_cache/gemma-*/`) and the
   gemma load becomes a hit, avoiding the hang entirely.
3. **Host oversubscription.** Concurrent agent workloads can drive load >100 on a 64-core box;
   everything then crawls. Check `uptime` before blaming the code.
4. **Pin the commit.** Re-stated because it matters: `main` lacks the audio fixes.
5. **First run on a 4x8 is long and looks stalled — it usually isn't.** Cold-cache compute was
   ~825 s (≈14 min) on g03blx04 vs ~47 s on a warm cache hit. Most of it is silent JIT kernel
   compilation, during which the launcher's staleness watchdog prints `⚠ NO OUTPUT for Ns`
   even though the job is healthy. Distinguish a real hang (#2) from honest compilation by
   watching cache-file churn rather than stdout:
   `find ~/.cache/tt-metal-cache -type f -newermt '-90 seconds' | wc -l` — hundreds/thousands
   of fresh files = compiling fine; zero + CPU still pinned = hung, kill & rerun.

## Reference: the `sulphur` launcher

See `/home/sulphur/sulphur` on g03blx02 for the canonical version. Usage is
`sulphur [--lora] OUTPUT.mp4 "prompt"` — `--lora` swaps `LTX_CHECKPOINT` to
`$ROOT/models/sulphur_lora_fused_distil.safetensors` (§6b), default is the merged
`sulphur_distil_bf16.safetensors`; `CKPT_PATH=<file> sulphur …` overrides with any drop-in (§6c). Shape:
`tt-device-mcp run-bg -w $ROOT/tt-metal -t 5400 "<cmd>"` + a streaming staleness watchdog, where `<cmd>` exports
`TT_METAL_HOME`/`PYTHONPATH`/`PYTHON_ENV_DIR=$ROOT/tt-metal[/python_env]`,
`HF_HOME=$ROOT/hf`, `HF_HUB_OFFLINE=1`, `GEMMA_PATH=google/gemma-3-12b-it-qat-q4_0-unquantized`,
`TT_DIT_CACHE_DIR=$ROOT/tt_dit_cache`, `LTX_CHECKPOINT=$ROOT/models/sulphur_distil_bf16.safetensors`,
`OUTPUT_PATH`/`PROMPT`/`NO_PROMPT=1`/`LTX_TRACED=0`/`SEED`/`HEIGHT=1088`/`WIDTH=1920`/`NUM_FRAMES=145`,
and `TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-180}` (bounds a stuck device op — e.g. the issue-#2
gemma write — so it aborts with `TIMEOUT: device timeout, potential hang detected` instead of hanging
forever; it's per-op, so it does NOT false-trip long cold compiles whose individual ops are sub-second),
then runs `pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled -k $MESH_PARAM -s -v --timeout=0`.
The `PYTHON_ENV_DIR` export is required or the broker mis-locates the venv as `$ROOT/tt-metal/tt-metal/python_env`.

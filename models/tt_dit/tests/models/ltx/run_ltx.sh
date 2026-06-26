#!/usr/bin/env bash
#
# run_ltx.sh — one-shot launcher for the LTX-2.3 TT-DiT video pipelines.
#
# Handles everything: activates the venv, auto-downloads the required checkpoints
# + Gemma weights from HuggingFace, writes an env file (the device broker does NOT
# inherit your shell exports), and launches the pytest entrypoint through the
# tt-device-mcp broker (or directly with --direct).
#
# Examples:
#   ./run_ltx.sh                              # distilled T2V on Galaxy 4x8 ring (default prompt)
#   ./run_ltx.sh -p "a dragon over a castle"  # custom prompt
#   ./run_ltx.sh --i2v ~/pics/hero.png        # image-to-video
#   ./run_ltx.sh --i2v-chained                # t2v->i2v chained ~12s video (4x8)
#   ./run_ltx.sh --variant pro --mesh 2x4     # Pro one-stage on a Loud Box
#   ./run_ltx.sh --traced                     # traced steady-state timing
#   ./run_ltx.sh --interactive                # type prompts at a REPL
#
set -euo pipefail

# ----------------------------------------------------------------------------- defaults
TT_METAL_DIR="${TT_METAL_HOME:-$HOME/tt-metal}"
CKPT_DIR="$HOME/.cache/ltx-checkpoints"

VARIANT="distilled"          # distilled | pro | two-stages
MESH="4x8"                   # 4x8 | 2x4 | 2x2
TOPOLOGY="ring"              # ring | linear   (only meaningful on 4x8)

# Text-to-video default (used when I2V is OFF): the fairy warrior.
PROMPT_T2V_DEFAULT="A lithe fairy warrior hovers in a sun-dappled enchanted forest, iridescent dragonfly wings shimmering as they beat, clad in ornate leaf-and-silver armor etched with glowing runes. She grips a slender glowing longsword wreathed in soft blue arcane light, hair drifting in the breeze, motes of golden pollen and faint magic sparks floating through shafts of light. Cinematic fantasy, painterly D&D concept-art style, dramatic rim lighting, shallow depth of field, slow graceful camera push-in, highly detailed."
# Image-to-video default (used when I2V is ON): the woman in I2V_DEFAULT_IMAGE singing, written
# for tight lip-sync since the distilled pipeline generates audio alongside the video.
PROMPT_I2V_DEFAULT="A young woman with shoulder-length curly brown hair, wearing a deep green blouse and a delicate gold necklace, stands in a warm, plant-filled modern cafe lit by soft natural daylight. She smiles brightly and sings cheerfully the words 'doobie do, doobie day, oh what a sunny day', her lips moving in precise sync with every syllable — mouth opening and closing naturally on each word, jaw and lips articulating the consonants, white teeth flashing as she enunciates. Joyful, playful expression: eyebrows lifting on the upbeat notes, eyes sparkling, head swaying gently to the rhythm with her curls bouncing softly, subtle lifelike facial micro-expressions and natural breathing between phrases. Cinematic shallow depth of field, soft golden rim light, warm blurred greenery and gentle bokeh behind her, slow graceful camera push-in, photorealistic, highly detailed."
PROMPT=""

# Default conditioning image used when I2V is activated without an explicit path.
I2V_DEFAULT_IMAGE="$HOME/pics/stock-photo-portrait-of-young-smiling-woman-looking-at-camera-with-crossed-arms-happy-girl-standing-in-1865153395.jpg"
I2V_IMAGE=""
I2V_STRENGTH="1.0"
FRAMES=""; HEIGHT=""; WIDTH=""; SEED=""; STEPS=""
OUTPUT=""
CHAINED=0                    # --i2v-chained: t2v->i2v ~12s video (two broker passes + splice, 4x8)
I2V_SEED=""                  # i2v-phase seed in --i2v-chained (default 12); t2v uses --seed (default 10)
GATE=0                       # --i2v-gate: run the i2v assertion test (PCC + seam) on an image
# Fast-path defaults: traced steady-state on (gen #0 captures, gen #1 is the pure-replay
# measurement — the "after warmup" number), warmup on (required before capture), and the
# post-gen VBench/CLIP quality gates off (they load extra models + add minutes, and don't
# affect the gen timing). Flip with --no-trace / --quality-gates.
TRACED=1
WARMUP=1
QUALITY_GATES=0
RUN_I2V_OVERRIDE=""         # "" = auto (0 for T2V fast path, 1 for I2V); set by --run-i2v
HOST_VAE_OVERRIDE=""        # "" = off (device-only); set to 1 via --host-vae-encoder to enable
INTERACTIVE=0
TIMEOUT=7200                 # broker wall-clock cap (max 7200)
RUN_MODE="broker"           # broker | direct
DO_DOWNLOAD=1
HF_TOKEN_ARG="${HF_TOKEN:-}"

usage() {
  cat <<'EOF'
Usage: ./run_ltx.sh [options]

  -v, --variant <distilled|pro|two-stages>   pipeline (default: distilled)
  -m, --mesh <4x8|2x4|2x2>                    device mesh (default: 4x8 = Galaxy)
      --topology <ring|linear>               fabric on 4x8 (default: ring)
  -p, --prompt "<text>"                       text prompt (default: fairy warrior)
      --i2v [image_path]                      image-to-video conditioning image
                                              (bare --i2v uses the default singing-woman image)
      --i2v-chained                           t2v->i2v chained ~12s video (forces 4x8 ring):
                                              two broker passes (t2v-only, then i2v on its
                                              last frame) + host splice. No -p reproduces the
                                              e2e baseline (test DEFAULT_LTX_PROMPT).
      --i2v-seed <N>                          i2v-phase seed in --i2v-chained (default 12)
      --i2v-gate [image_path]                 run the i2v ASSERTION test (frame-0 PCC>0.85 +
                                              grid-seam gate) on an image at 4x8 ring, single
                                              pass (bare uses the default conditioning image)
      --strength <float>                      i2v conditioning strength (default: 1.0)
      --frames <N>                            num frames (distilled only; default 145)
      --height <N>                            height, mult of 64 (distilled only; default 1088)
      --width  <N>                            width,  mult of 64 (distilled only; default 1920)
      --seed   <N>                            RNG seed (default 10)
      --steps  <N>                            override denoise steps
  -o, --output <path.mp4>                     output video path
      --traced                                traced steady-state (default; warmup + replay)
      --no-trace                              disable tracing (single eager gen, no replay)
      --no-warmup                             disable RUN_WARMUP
      --quality-gates                         run VBench + CLIP gates after gen (default: off)
      --run-i2v <0|1>                         force the per-token I2V path on/off
                                              (default: auto — 0/fast for T2V, 1 for --i2v)
      --host-vae-encoder <0|1>                encode the I2V conditioning image on the host
                                              (ltx_core reference) and log device-vs-host parity,
                                              using the host latent for the run. Default: 0 (off,
                                              device-only). Pass 1 to enable; needs the LTX-2
                                              reference at <repo>/LTX-2 (or LTX_REFERENCE_ROOT).
      --interactive                           prompt REPL (sets NO_PROMPT=0)
  -t, --timeout <sec>                         broker timeout (default 7200, max 7200)
      --direct                                run pytest directly (skip the broker)
      --skip-download                         assume weights already present
      --hf-token <tok>                        HuggingFace token (or set HF_TOKEN)
  -h, --help                                  this help
EOF
}

# ----------------------------------------------------------------------------- arg parse
while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--variant)   VARIANT="$2"; shift 2 ;;
    -m|--mesh)      MESH="$2"; shift 2 ;;
    --topology)     TOPOLOGY="$2"; shift 2 ;;
    -p|--prompt)    PROMPT="$2"; shift 2 ;;
    --i2v)
      # Optional path: "--i2v <path>" uses that image; bare "--i2v" uses I2V_DEFAULT_IMAGE.
      if [[ -n "${2:-}" && "$2" != -* ]]; then I2V_IMAGE="$2"; shift 2
      else I2V_IMAGE="$I2V_DEFAULT_IMAGE"; shift; fi ;;
    --i2v-chained)  CHAINED=1; shift ;;
    --i2v-seed)     I2V_SEED="$2"; shift 2 ;;
    --i2v-gate)
      GATE=1
      if [[ -n "${2:-}" && "$2" != -* ]]; then I2V_IMAGE="$2"; shift 2; else shift; fi ;;
    --strength)     I2V_STRENGTH="$2"; shift 2 ;;
    --frames)       FRAMES="$2"; shift 2 ;;
    --height)       HEIGHT="$2"; shift 2 ;;
    --width)        WIDTH="$2"; shift 2 ;;
    --seed)         SEED="$2"; shift 2 ;;
    --steps)        STEPS="$2"; shift 2 ;;
    -o|--output)    OUTPUT="$2"; shift 2 ;;
    --traced)       TRACED=1; shift ;;
    --no-trace)     TRACED=0; shift ;;
    --no-warmup)    WARMUP=0; shift ;;
    --quality-gates) QUALITY_GATES=1; shift ;;
    --run-i2v)      RUN_I2V_OVERRIDE="$2"; shift 2 ;;
    --host-vae-encoder) HOST_VAE_OVERRIDE="$2"; shift 2 ;;
    --interactive)  INTERACTIVE=1; shift ;;
    -t|--timeout)   TIMEOUT="$2"; shift 2 ;;
    --direct)       RUN_MODE="direct"; shift ;;
    --skip-download) DO_DOWNLOAD=0; shift ;;
    --hf-token)     HF_TOKEN_ARG="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# Chained and gate both run the distilled i2v e2e on 4x8 ring with the per-token i2v path on.
if [[ "$CHAINED" -eq 1 || "$GATE" -eq 1 ]]; then
  VARIANT="distilled"; MESH="4x8"; TOPOLOGY="ring"; RUN_I2V_OVERRIDE=1
fi
# Gate is a single pass conditioned on an image; default to the standard conditioning image.
[[ "$GATE" -eq 1 && -z "$I2V_IMAGE" ]] && I2V_IMAGE="$I2V_DEFAULT_IMAGE"
# Prompt default by mode (singing woman for I2V, fairy warrior for T2V). Chained is the exception:
# leave PROMPT empty so both passes fall back to the test's DEFAULT_LTX_PROMPT (the canonical e2e
# baseline) — overriding it desyncs the chain. A user -p still wins everywhere.
if [[ "$CHAINED" -ne 1 && -z "$PROMPT" ]]; then
  if [[ -n "$I2V_IMAGE" ]]; then PROMPT="$PROMPT_I2V_DEFAULT"; else PROMPT="$PROMPT_T2V_DEFAULT"; fi
fi

cd "$TT_METAL_DIR"

# ----------------------------------------------------------------------------- venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$TT_METAL_DIR/python_env/bin/activate"
fi
export TT_METAL_HOME="$TT_METAL_DIR"
export PYTHONPATH="$TT_METAL_DIR"
# The broker does NOT inherit our shell venv when an env file (-e) is used, and its
# auto-locator guesses a wrong (doubled) path. Tell it explicitly.
export PYTHON_ENV_DIR="$TT_METAL_DIR/python_env"
export VIRTUAL_ENV="${VIRTUAL_ENV:-$TT_METAL_DIR/python_env}"
[[ -n "$HF_TOKEN_ARG" ]] && export HF_TOKEN="$HF_TOKEN_ARG"

# ----------------------------------------------------------------------------- variant -> test file + checkpoint
case "$VARIANT" in
  distilled)
    TEST_FILE="models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py"
    CKPT_FILE="ltx-2.3-22b-distilled-1.1.safetensors"
    NEED_UPSAMPLER=1; NEED_LORA=0 ;;
  pro|one-stage|onestage)
    TEST_FILE="models/tt_dit/tests/models/ltx/test_pipeline_ltx_one_stage.py"
    CKPT_FILE="ltx-2.3-22b-dev.safetensors"
    NEED_UPSAMPLER=0; NEED_LORA=0 ;;
  two-stages|twostages)
    TEST_FILE="models/tt_dit/tests/models/ltx/test_pipeline_ltx_two_stages.py"
    CKPT_FILE="ltx-2.3-22b-dev.safetensors"
    NEED_UPSAMPLER=1; NEED_LORA=1 ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 1 ;;
esac

# ----------------------------------------------------------------------------- mesh/topology -> pytest -k id
case "$MESH" in
  4x8)
    [[ "$TOPOLOGY" == "linear" ]] && KID="bh_4x8sp1tp0_linear" || KID="bh_4x8sp1tp0_ring" ;;
  2x4) KID="bh_2x4sp1tp0" ;;
  2x2) KID="2x2sp0tp1" ;;
  *) echo "Unknown mesh: $MESH" >&2; exit 1 ;;
esac

# Chained/gate target the i2v e2e test, whose 4x8 case carries a distinct id (i2v_4x8sp1tp0_ring)
# kept out of the bh_* namespace so a bare t2v `-k bh_4x8sp1tp0_ring` never selects it.
[[ "$CHAINED" -eq 1 || "$GATE" -eq 1 ]] && KID="test_pipeline_distilled_i2v and i2v_4x8sp1tp0_ring"

# ----------------------------------------------------------------------------- download weights
if [[ "$DO_DOWNLOAD" -eq 1 ]]; then
  echo ">> Ensuring weights are present (auto-download from HuggingFace if missing)..."
  python - "$CKPT_DIR" "$CKPT_FILE" "$NEED_UPSAMPLER" "$NEED_LORA" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download, snapshot_download

ckpt_dir, ckpt_file, need_ups, need_lora = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4] == "1"
os.makedirs(ckpt_dir, exist_ok=True)
REPO = "Lightricks/LTX-2.3"

# Main checkpoint -> local ckpt dir, so the test's skipif (os.path.exists) passes and
# default_ltx_checkpoint() resolves it without any env var.
dest = os.path.join(ckpt_dir, ckpt_file)
if os.path.exists(dest):
    print(f"   [ok]   {ckpt_file} (cached)")
else:
    print(f"   [pull] {ckpt_file} -> {ckpt_dir}")
    hf_hub_download(repo_id=REPO, filename=ckpt_file, local_dir=ckpt_dir)

# Upsampler + distilled LoRA are fetched by the pipeline via plain hf_hub_download
# (HF hub cache). Pre-pull into the same cache so the run doesn't stall on them.
if need_ups:
    print("   [pull] spatial upsampler")
    hf_hub_download(repo_id=REPO, filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
if need_lora:
    # LoRA resolver checks the local ckpt dir first, then HF.
    lora = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    if not os.path.exists(os.path.join(ckpt_dir, lora)):
        print("   [pull] distilled LoRA")
        hf_hub_download(repo_id=REPO, filename=lora, local_dir=ckpt_dir)

# Gemma-3-12B text encoder. GATED on HF (needs HF_TOKEN + accepted license). If
# GEMMA_PATH points at an existing local directory, use it as-is and skip the gated
# download entirely (no token required).
gemma = os.environ.get("GEMMA_PATH", "google/gemma-3-12b-it-qat-q4_0-unquantized")
if os.path.isdir(gemma):
    print(f"   [ok]   gemma (local dir {gemma})")
else:
    print(f"   [pull] {gemma}")
    snapshot_download(repo_id=gemma)
print(">> Weights ready.")
PY
fi

# ----------------------------------------------------------------------------- env for the run
export LTX_CHECKPOINT="$CKPT_DIR/$CKPT_FILE"
# Gemma text encoder. The HF repo is GATED (needs HF_TOKEN + accepted license); on boxes
# without a token, point GEMMA_PATH at a local copy of the weights and it loads offline.
export GEMMA_PATH="${GEMMA_PATH:-google/gemma-3-12b-it-qat-q4_0-unquantized}"
export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-$HOME/.cache/tt-dit}"
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
# Persist compiled kernels. Unset, TT_METAL_CACHE defaults to an ephemeral temp dir,
# so every cold start recompiles ALL of the 22B model's kernels (the slow startup).
# Pinning it to a stable path means only the first run compiles; later runs reuse it.
export TT_METAL_CACHE="${TT_METAL_CACHE:-$HOME/.cache/tt-metal-cache}"
mkdir -p "$TT_METAL_CACHE"
[[ -n "$PROMPT" ]] && export PROMPT   # empty (chained, no -p) -> test uses DEFAULT_LTX_PROMPT
export RUN_WARMUP="$WARMUP"
export NO_PROMPT=$([[ "$INTERACTIVE" -eq 1 ]] && echo 0 || echo 1)
export LTX_TRACED="$TRACED"

# RUN_I2V: per-token image-conditioning path. For pure T2V the fast scalar-AdaLN path
# (RUN_I2V=0) avoids the per-token video timesteps + dense modulation, so default it off
# when there's no conditioning image. With --i2v it MUST be on (the transformer asserts on
# images= otherwise). --run-i2v overrides the auto choice.
if [[ -n "$RUN_I2V_OVERRIDE" ]]; then
  RUN_I2V="$RUN_I2V_OVERRIDE"
elif [[ -n "$I2V_IMAGE" ]]; then
  RUN_I2V=1
else
  RUN_I2V=0
fi
export RUN_I2V

# LTX_VAE_ENCODER_HOST: encode the I2V conditioning image on the host (ltx_core reference) and log
# device-vs-host parity, using the host latent for the run. Off by default (device-only fast path);
# turn it on explicitly with --host-vae-encoder 1 to self-check the device VAE encoder.
if [[ -n "$HOST_VAE_OVERRIDE" ]]; then
  LTX_VAE_ENCODER_HOST="$HOST_VAE_OVERRIDE"
else
  LTX_VAE_ENCODER_HOST=0
fi
export LTX_VAE_ENCODER_HOST

# VBench + CLIP quality gates run AFTER generation (don't affect the gen number) but load
# extra models and add minutes. Off by default for a perf run; --quality-gates turns them on.
export RUN_VBENCH="$QUALITY_GATES"
export RUN_CLIP="$QUALITY_GATES"

[[ -n "$I2V_IMAGE" ]] && export LTX_I2V_IMAGE="$I2V_IMAGE"
[[ -n "$I2V_IMAGE" ]] && export LTX_I2V_STRENGTH="$I2V_STRENGTH"
[[ -n "$FRAMES" ]] && export NUM_FRAMES="$FRAMES"
[[ -n "$HEIGHT" ]] && export HEIGHT
[[ -n "$WIDTH"  ]] && export WIDTH
[[ -n "$SEED"   ]] && export SEED
[[ -n "$STEPS"  ]] && export NUM_STEPS="$STEPS"
[[ -n "$OUTPUT" ]] && export OUTPUT_PATH="$OUTPUT"

# Names of env vars to forward to the run. The first group makes the broker use our
# venv + repo instead of its (wrong) auto-located path.
ENV_NAMES=(PYTHON_ENV_DIR VIRTUAL_ENV TT_METAL_HOME PYTHONPATH \
           LTX_CHECKPOINT GEMMA_PATH TT_DIT_CACHE_DIR TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES \
           TT_METAL_CACHE PROMPT RUN_WARMUP NO_PROMPT LTX_TRACED RUN_I2V LTX_VAE_ENCODER_HOST \
           RUN_VBENCH RUN_CLIP)
[[ -n "$I2V_IMAGE" ]] && ENV_NAMES+=(LTX_I2V_IMAGE LTX_I2V_STRENGTH)
[[ -n "$FRAMES" ]] && ENV_NAMES+=(NUM_FRAMES)
[[ -n "$HEIGHT" ]] && ENV_NAMES+=(HEIGHT)
[[ -n "$WIDTH"  ]] && ENV_NAMES+=(WIDTH)
[[ -n "$SEED"   ]] && ENV_NAMES+=(SEED)
[[ -n "$STEPS"  ]] && ENV_NAMES+=(NUM_STEPS)
[[ -n "$OUTPUT" ]] && ENV_NAMES+=(OUTPUT_PATH)
[[ -n "${HF_TOKEN:-}" ]] && ENV_NAMES+=(HF_TOKEN)

PYTEST_CMD="pytest $TEST_FILE -k \"$KID\" -s --timeout $TIMEOUT"

echo
echo "==================== LTX run ===================="
echo " variant : $VARIANT      mesh: $MESH  topology: $TOPOLOGY"
echo " test    : $TEST_FILE"
echo " -k      : $KID"
echo " ckpt    : $LTX_CHECKPOINT"
 echo " i2v     : $([[ "$CHAINED" -eq 1 ]] && echo "chained t2v->i2v 12s (seeds ${SEED:-10}/${I2V_SEED:-12}) -> ${OUTPUT:-$HOME/t2v_i2v_chained_12s.mp4}" || echo "${I2V_IMAGE:-<none> (text-to-video)}")"
echo " mode    : $RUN_MODE"
echo " traced  : $([[ "$TRACED" -eq 1 ]] && echo "yes (gen #1 = steady-state replay)" || echo "no")"
echo " warmup  : $([[ "$WARMUP" -eq 1 ]] && echo "yes" || echo "no")    RUN_I2V: $RUN_I2V    quality-gates: $([[ "$QUALITY_GATES" -eq 1 ]] && echo "on" || echo "off")"
echo " host-vae-encoder: $([[ "$LTX_VAE_ENCODER_HOST" == "1" ]] && echo "on (device-vs-host parity logged; host latent used)" || echo "off")"
echo " prompt  : ${PROMPT:0:70}..."
echo "================================================="
echo

# ----------------------------------------------------------------------------- launch
# Emit "KEY: <quoted>" YAML lines for the broker env file. json.dumps gives a properly escaped
# double-quoted scalar, which YAML reads back as the exact string (spaces, commas, quotes).
# Only vars actually present in the environment are written, so an unexported name is a no-op.
emit_env() {
  python - "$@" <<'PY'
import json, os, sys
out, names = sys.argv[1], sys.argv[2:]
with open(out, "w") as f:
    for n in names:
        if n in os.environ:
            f.write(f"{n}: {json.dumps(os.environ[n])}\n")
PY
}

if [[ "$CHAINED" -eq 1 ]]; then
  [[ "$RUN_MODE" == "direct" ]] && { echo "--i2v-chained needs the broker (two passes); drop --direct" >&2; exit 1; }
  T2V_OUT="$HOME/chain_t2v.mp4"; COND="$HOME/chain_cond.png"; I2V_OUT="$HOME/chain_i2v.mp4"
  FINAL="${OUTPUT:-$HOME/t2v_i2v_chained_12s.mp4}"
  T2V_SEED="${SEED:-10}"; I2V_SEED="${I2V_SEED:-12}"

  ENVA="$(mktemp /tmp/ltx_chain_a.XXXXXX.yaml)"; ENVB="$(mktemp /tmp/ltx_chain_b.XXXXXX.yaml)"
  # An autoupdate broker restart mid-run kills the job; pause it, always re-arm + clean on exit.
  TIMER_PAUSED=0
  sudo -n systemctl stop tt-device-reconcile.timer 2>/dev/null && { TIMER_PAUSED=1; echo "reconcile timer paused"; } || echo "WARN: could not pause reconcile timer"
  cleanup_chain() {
    rm -f "$ENVA" "$ENVB"
    [[ "$TIMER_PAUSED" == 1 ]] && { sudo -n systemctl start tt-device-reconcile.timer 2>/dev/null && echo "reconcile timer re-armed" || echo "WARN: could not re-arm reconcile timer"; }
  }
  trap cleanup_chain EXIT

  # The 4x8 one-process chain wedges: the i2v-time image-encoder reload clobbers t2v's captured
  # traces. So pass A persists the t2v clip + its last frame, pass B conditions a fresh process.
  echo ">> [chained 1/2] t2v-only (seed $T2V_SEED) -> $T2V_OUT"
  rm -f "$T2V_OUT" "$COND"
  export SEED="$T2V_SEED" LTX_T2V_ONLY=1 LTX_T2V_OUT="$T2V_OUT" LTX_COND_OUT="$COND"
  emit_env "$ENVA" "${ENV_NAMES[@]}" SEED LTX_T2V_ONLY LTX_T2V_OUT LTX_COND_OUT
  tt-device-mcp run "$PYTEST_CMD" -w "$TT_METAL_DIR" -e "$ENVA" -t "$TIMEOUT" -o 40
  [[ -f "$T2V_OUT" && -f "$COND" ]] || { echo "chained pass A failed (missing $T2V_OUT / $COND)" >&2; exit 1; }
  unset LTX_T2V_ONLY LTX_T2V_OUT LTX_COND_OUT

  echo ">> [chained 2/2] i2v on the t2v last frame (seed $I2V_SEED) -> $I2V_OUT"
  rm -f "$I2V_OUT"
  export SEED="$I2V_SEED" LTX_I2V_IMAGE="$COND" LTX_I2V_STRENGTH="$I2V_STRENGTH" LTX_I2V_OUT="$I2V_OUT"
  emit_env "$ENVB" "${ENV_NAMES[@]}" SEED LTX_I2V_IMAGE LTX_I2V_STRENGTH LTX_I2V_OUT
  tt-device-mcp run "$PYTEST_CMD" -w "$TT_METAL_DIR" -e "$ENVB" -t "$TIMEOUT" -o 40
  [[ -f "$I2V_OUT" ]] || { echo "chained pass B failed (missing $I2V_OUT)" >&2; exit 1; }

  echo ">> [chained splice] -> $FINAL"
  FF="$(command -v ffmpeg || true)"
  [[ -z "$FF" ]] && FF="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
  [[ -n "$FF" ]] || { echo "no ffmpeg (system or imageio-ffmpeg)" >&2; exit 1; }
  "$FF" -v error -i "$T2V_OUT" -i "$I2V_OUT" \
    -filter_complex "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]" \
    -map "[v]" -map "[a]" -y "$FINAL"
  echo "CHAINED 12s VIDEO: $FINAL"
  "$FF" -hide_banner -i "$FINAL" 2>&1 | grep -E "Duration|Stream" || true
elif [[ "$RUN_MODE" == "direct" ]]; then
  eval "$PYTEST_CMD"
else
  ENV_FILE="$(mktemp /tmp/ltx_env.XXXXXX.yaml)"
  trap 'rm -f "$ENV_FILE"' EXIT
  emit_env "$ENV_FILE" "${ENV_NAMES[@]}"
  echo ">> Submitting to tt-device-mcp broker (timeout ${TIMEOUT}s)..."
  tt-device-mcp run "$PYTEST_CMD" -w "$TT_METAL_DIR" -e "$ENV_FILE" -t "$TIMEOUT" -o 40
fi

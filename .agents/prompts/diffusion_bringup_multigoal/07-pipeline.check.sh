#!/usr/bin/env bash
# Runner-side gate for the diffusion end-to-end pipeline stage: a generated artifact must exist and must
# not be mechanically degenerate (frozen/black video, silent audio, NaN), and the capability contract
# must be present and sane. Scoped to this run via env so stale artifacts cannot pass/fail it.
# Exit 0 pass, 1 advisory, 2 critical, 3 error.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../../.." && pwd)"
scripts="$repo_root/.agents/scripts"

# Locate the generated artifact directory (must contain frames/ and/or audio.wav).
if [ -n "${DIFFUSION_OUT:-}" ]; then
  out="$DIFFUSION_OUT"
elif [ -n "${MODEL_DIR:-}" ] && [ -d "${MODEL_DIR}/doc" ]; then
  # newest sample dir under the model's doc tree
  out="$(find "${MODEL_DIR}/doc" -type d -name frames -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
else
  echo "Set DIFFUSION_OUT to the generated artifact dir (with frames/ and/or audio.wav)." >&2
  exit 3
fi
[ -z "$out" ] && { echo "no generated artifact directory found." >&2; exit 2; }

frames_arg=(); wav_arg=()
[ -d "$out/frames" ] && frames_arg=(--frames "$out/frames")
[ -f "$out/audio.wav" ] && wav_arg=(--wav "$out/audio.wav")
[ ${#frames_arg[@]} -eq 0 ] && [ ${#wav_arg[@]} -eq 0 ] && { echo "no frames/ or audio.wav under $out" >&2; exit 2; }

python "$scripts/check_diffusion_degenerate.py" "${frames_arg[@]}" "${wav_arg[@]}" --missing critical
deg=$?

python "$scripts/check_diffusion_contract.py" \
  --model-dir "${MODEL_DIR:-}" --hf-model "${HF_MODEL:-}" --stage pipeline --require-contract
con=$?

# worst of the two (3 error > 2 critical > 1 advisory > 0)
worst=$deg; [ $con -gt $worst ] && worst=$con
exit $worst

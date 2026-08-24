#!/bin/bash
# Real test: can the resolver write a loader for the native checkpoint with NO existing
# PyTorch implementation of the architecture reachable on the machine?
#
# Everything moved here is restored by the EXIT trap, including on error or kill.
set -u

STASH=/localdev/lserbedzija/resolver_test/stash
PR=/localdev/lserbedzija/repos/tt-metal-pr46283
HP=/localdev/lserbedzija/repos/tt-metal
LATEST=/localdev/lserbedzija/repos/tt-metal-latest

mkdir -p "$STASH"
MANIFEST="$STASH/manifest.tsv"
: > "$MANIFEST"

hide() {   # hide <path>  -> move it into the stash, remember where it came from
  local src="$1"
  [ -e "$src" ] || { echo "  skip (absent): $src"; return 0; }
  local dst="$STASH/$(echo "$src" | tr '/' '_')"
  mv "$src" "$dst"
  printf '%s\t%s\n' "$dst" "$src" >> "$MANIFEST"
  echo "  hidden: $src"
}

restore() {
  echo; echo "=== RESTORING ==="
  chmod 755 "$STASH" 2>/dev/null || true
  # reverse order, so parents come back before children
  tac "$MANIFEST" | while IFS=$'\t' read -r dst src; do
    if [ -e "$dst" ]; then
      mkdir -p "$(dirname "$src")"
      mv "$dst" "$src" && echo "  restored: $src"
    else
      echo "  MISSING IN STASH: $dst -> $src"
    fi
  done
  echo "=== git status, hand-port repo (must match the 'before' snapshot) ==="
  git -C "$HP" status --short | head
  echo "=== git status, PR repo ==="
  git -C "$PR" status --short | grep -v perfauto_bak | head
}
trap restore EXIT

echo "=== BEFORE: git status, hand-port repo ==="
git -C "$HP" status --short | head

echo; echo "=== HIDING every PyTorch implementation and architecture-revealing copy ==="
hide "$HP/models/experimental/voxtral_tts/reference/voxtral_common_ref.py"
hide "$HP/models/experimental/voxtral_tts/reference/voxtral_backbone_ref.py"
hide "$HP/models/experimental/voxtral_tts/reference/voxtral_flow_ref.py"
hide "$HP/models/experimental/voxtral_tts/reference/voxtral_codec_ref.py"
hide "$HP/models/experimental/voxtral_tts/reference/voxtral_pipeline_ref.py"
hide "$HP/models/experimental/voxtral_tts/reference/voxtral_tokenizer_ref.py"
hide "$HP/models/experimental/voxtral_tts/tt"
hide "$HP/models/demos/voxtral_tts_backbone"
hide "$LATEST/models/experimental/voxtral_tts"
hide "/localdev/lserbedzija/hf_models/voxtral-tts-full"
hide "$PR/models/demos/voxtral_tts_full"
hide "$PR/TOOL_FINDINGS.md"
hide "/localdev/lserbedzija/pr46283_evidence/prior_run_state/overlay_store"
hide "/localdev/lserbedzija/resolver_test/demo_dir"
hide "/localdev/lserbedzija/resolver_test/voxtral-tts-native"
chmod 000 "$STASH"

echo; echo "=== VERIFY nothing is findable ==="
echo -n "  voxtral_common_ref.py anywhere readable: "
find /localdev/lserbedzija -name "voxtral_common_ref.py" 2>/dev/null | wc -l
echo -n "  modeling_voxtral_tts.py anywhere readable: "
find /localdev/lserbedzija -name "modeling_voxtral_tts.py" 2>/dev/null | wc -l

echo; echo "=== RUNNING the resolver against the isolated native checkpoint ==="
cd /tmp
env -u VOXTRAL_REPO_ROOT TT_METAL_HOME="$PR" PYTHONPATH="$PR" \
    TT_HW_PLANNER_LOADER_RESOLVER=1 \
    "$PR/python_env/bin/python" -u - <<'PY'
import sys, json
sys.path.insert(0, "/localdev/lserbedzija/repos/tt-metal-pr46283")
from pathlib import Path
from scripts.tt_hw_planner import reference_loader_resolver as r

N = "/localdev/lserbedzija/resolver_test/native_iso"
D = Path("/localdev/lserbedzija/resolver_test/demo_dir_iso")
failure = ("RuntimeError: Could not load %s via AutoModelForCausalLM or AutoModel; last error: "
           "ValueError: Unrecognized model in %s. Should have a `model_type` key in its config.json"
           % (N, N))
res = r.resolve(model_id=N, demo_dir=D, failure_text=failure, timeout_s=1500, cwd=Path("/tmp"))
print("RESULT:", json.dumps(res, indent=1), flush=True)
PY
echo "=== resolver exit: $? ==="

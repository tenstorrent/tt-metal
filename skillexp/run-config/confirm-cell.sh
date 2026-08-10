#!/usr/bin/env bash
# Post-run isolation confirmation for one advchal-v3 cell.
#
#   bash confirm-cell.sh <short> <model_dir> <v2_output_commit_or_->
#
# Asked of every cell AFTER it finishes, because the pre-flight can only prove the tree was clean at
# LAUNCH. Contamination in the v2 corpus arrived mid-run (an agent reading the object store) and after
# it (a publish step assembling the wrong tree), and every guard that inspected the checkout passed
# honestly while the object store carried the leak. So: re-derive, do not trust.
#
# Nothing here writes to the repo. Read-only, host-side.
set -uo pipefail
SHORT=${1:?short name, e.g. nmFN}
MD=${2:?model dir}
V2=${3:-'-'}
METAL=/home/mvasiljevic/tt-metal
LOGROOT=/home/mvasiljevic/skillexp-logs
W="skillexp-cell/advchal-v3/$SHORT"
P="models/autoports/$MD/doc/advisor_challenger"
D="$LOGROOT/p-advchal-v3-$SHORT"
fail=0
ok()   { echo "  ok   $*"; }
bad()  { echo "  FAIL $*"; fail=1; }
note() { echo "  --   $*"; }

echo "=== advchal-v3 post-run confirmation: $SHORT ($MD) ==="
echo "--- 1. the stage's own outcome ---"
rc=$(cat "$D/exit_code" 2>/dev/null || echo '?')
ts=$(grep -hoE 'stage_[0-9]+_terminal_status=[a-zA-Z]+' "$D/manifest.txt" 2>/dev/null | tail -1 | cut -d= -f2)
echo "  rc=$rc terminal_status=${ts:-unknown}"

echo "--- 2. did the tree stay the tree under test? ---"
a_now=$(git -C "$METAL" rev-parse -q "$W:.agents" 2>/dev/null)
a_skill=$(git -C "$METAL" rev-parse -q "mvasiljevic/qb2/skillexp/challenger-skill-v3:.agents" 2>/dev/null)
[ -n "$a_now" ] && [ "$a_now" = "$a_skill" ] \
  && ok ".agents still identical to the frozen skill branch (no factor drift)" \
  || bad ".agents DIVERGED from the skill branch -- the stage changed under the measurement"
grep -qiE 'FACTOR DRIFT|ARM ANCESTRY LOST|CONTAMINATED|run_stage exit=71' "$D/console.log" 2>/dev/null \
  && bad "the driver recorded a contamination marker in this run's console.log" \
  || ok "no contamination marker in this run's console.log"

echo "--- 3. is the output this cell's own work? (author date vs stage start) ---"
t0=$(stat -c %Y "$D" 2>/dev/null || echo 0)
n=0; old=0
while read -r c ad; do
  [ -n "$c" ] || continue; n=$((n+1))
  [ "$ad" -lt $((t0 - 120)) ] && { bad "commit ${c:0:11} touching $P predates the stage ($(date -Iseconds -d @"$ad"))"; old=1; }
done < <(git -C "$METAL" log --format='%H %at' "mvasiljevic/qb2/skillexp/challenger-skill-v3..$W" -- "$P" 2>/dev/null)
[ "$old" = 0 ] && ok "$n commit(s) touching the output, none predating the stage"
cp=$(git -C "$METAL" reflog show "$W" 2>/dev/null | grep -icE 'cherry-pick|rebase' || true)
[ "${cp:-0}" = 0 ] && ok "work-branch reflog records no cherry-pick or rebase" \
                    || bad "work-branch reflog records ${cp} cherry-pick/rebase event(s)"

echo "--- 4. blob identity: nothing reused from v2 or from a sibling ---"
# THE CHECK THAT CAUGHT THE V2 CONTAMINATION, and it caught it by hand, days late. Cheap to automate:
# an artefact byte-identical to another cell's is either inherited or copied, never independently measured.
mapfile -t mine < <(git -C "$METAL" ls-tree -r "$W" -- "$P" 2>/dev/null | awk '{print $3" "$4}')
echo "  this cell produced ${#mine[@]} artefact blob(s)"
if [ "$V2" != "-" ] && git -C "$METAL" rev-parse -q --verify "$V2^{commit}" >/dev/null 2>&1; then
  shared=0
  while read -r sha path; do
    [ -n "$sha" ] || continue
    for m in "${mine[@]}"; do
      [ "${m%% *}" = "$sha" ] && { note "shares a blob with v2: ${path##*/}"; shared=$((shared+1)); }
    done
  done < <(git -C "$METAL" ls-tree -r "$V2" -- "$P" 2>/dev/null | awk '{print $3" "$4}')
  [ "$shared" = 0 ] && ok "no artefact blob shared with the v2 run of this cell ($V2)" \
                    || bad "$shared artefact blob(s) byte-identical to the v2 run -- inherited, not measured"
else
  note "no v2 output commit given or it is unreachable; blob comparison vs v2 skipped"
fi

echo "--- 5. did any parked ref come back during the run? ---"
back=0
while read -r sha ref; do
  [ -n "$ref" ] || continue
  git -C "$METAL" rev-parse -q --verify "$ref" >/dev/null 2>&1 && { note "restored: $ref"; back=$((back+1)); }
done < "$LOGROOT/advchal-v3/parked-refs.txt"
[ "$back" = 0 ] && ok "all 156 parked refs are still unnamed" \
               || bad "$back parked ref(s) are named again -- the isolation lapsed mid-run"

n_td=$(git -C "$METAL" ls-tree -r --name-only "$W" -- "$P" 2>/dev/null | grep -c 'traced_dtypes.json')
echo "--- 6. provenance the artefacts must carry (population: $n_td traced_dtypes.json) ---"
[ "${n_td:-0}" -gt 0 ] || bad "no traced_dtypes.json under $P at all -- provenance is unverifiable"
for f in $(git -C "$METAL" ls-tree -r --name-only "$W" -- "$P" 2>/dev/null | grep 'traced_dtypes.json'); do
  # No f-strings and no nested quotes: an earlier version of this block was itself malformed Python, and
  # the script reported "did not parse" and then PASSED. Keep it boring.
  git -C "$METAL" show "$W:$f" 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
bad = 0
for k, expect in (("tracer_matches_checkout", True), ("optimizer_files_changed_since_pin", [])):
    v = d.get(k)
    good = (v == expect)
    bad += 0 if good else 1
    print(("  ok   " if good else "  FAIL ") + k + "=" + repr(v))
print("  --   advisor_commit=" + str(d.get("advisor_commit"))[:12] + " host=" + str(d.get("host")))
print("  --   capture_scope stopped_at=" + repr((d.get("capture_scope") or {}).get("stopped_at")))
sys.exit(1 if bad else 0)
'
  [ "$?" -eq 0 ] || bad "$f: provenance fields missing, wrong, or unreadable"
done
for f in $(git -C "$METAL" ls-tree -r --name-only "$W" -- "$P" 2>/dev/null | grep -E 'incumbent.*\.json'); do
  git -C "$METAL" show "$W:$f" 2>/dev/null | python3 -c '
import json, sys, os
d = json.load(sys.stdin)
c = [(d.get("device_users_at_start") or {}).get("count"), (d.get("device_users_at_end") or {}).get("count")]
tag = "  ok   " if c == [0, 0] else "  --   "
print(tag + os.environ.get("F", "?") + " process_ordinal=" + str(d.get("process_ordinal")) +
      " device_users=" + str(c))
' F="${f##*/}" || note "${f##*/}: could not read exclusivity fields"
done

echo "--- 7. the gate ---"
g=$(ls -t "$D"/*check*.log 2>/dev/null | head -1)
if [ -n "$g" ]; then
  grep -cE '^CRITICAL' "$g" >/dev/null 2>&1 && echo "  CRITICAL: $(grep -cE '^CRITICAL' "$g")  WARN: $(grep -cE '^WARN' "$g")"
  tail -1 "$g"
else note "no gate log found in $D"; fi

echo
[ "$fail" = 0 ] && echo "CONFIRMATION PASSED for $SHORT" || echo "CONFIRMATION FAILED for $SHORT -- read the FAIL lines above"
exit "$fail"

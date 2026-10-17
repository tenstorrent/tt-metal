#!/bin/bash
# usage: test_legs.sh
# Smoke test for legs.sh — no device, no network. Builds a temp repo with a bare `origin`, a branch
# `legs-test` that adds marker.txt (main has none), points TT_DEVICE_LOCK at a temp file (so the server
# skips tt-smi) and uses --cwd . because the default tt_metal/tt-llk/tests does not exist there.
# Cases: branch leg passes in the serving worktree · patch leg passes and leaves the repo clean ·
# --timeout 1 on `sleep 5` reports 124 · heartbeat file + status serve line · status lists 3 done · then a few refusals (nothing open → 3,
# other-arch request stays open, dirty repo → 96, bad args → 2). Prints PASS/FAIL per case; exit 1 on any FAIL.
set -uo pipefail
legs=$(dirname "$(readlink -f "$0")")/legs.sh
tmp=$(mktemp -d "${TMPDIR:-/tmp}/legs-test.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
export TT_DEVICE_LOCK=$tmp/lock GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_AUTHOR_NAME=legs GIT_AUTHOR_EMAIL=legs@test GIT_COMMITTER_NAME=legs GIT_COMMITTER_EMAIL=legs@test
repo=$tmp/repo; bare=$tmp/origin.git; D=$tmp/run
fail=0
check() { if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2${3:+ -- $3}"; fail=1; fi; }

git init -q -b main "$repo" 2>/dev/null || { git init -q "$repo" && git -C "$repo" symbolic-ref HEAD refs/heads/main; }
echo base > "$repo/README"; git -C "$repo" add README; git -C "$repo" commit -q -m base
git init -q --bare "$bare"; git -C "$repo" remote add origin "$bare"; git -C "$repo" push -q origin main
git -C "$repo" checkout -q -b legs-test; echo m > "$repo/marker.txt"; git -C "$repo" add marker.txt; git -C "$repo" commit -q -m marker
sha=$(git -C "$repo" rev-parse HEAD); git -C "$repo" push -q origin legs-test; git -C "$repo" checkout -q main
[ ! -e "$repo/marker.txt" ] && git -C "$repo" rev-parse -q --verify origin/main > /dev/null; check $? "fixture: repo on main, origin has both branches"
serve=("$legs" serve --run-dir "$D" --arch wormhole --repo "$repo" --once)

# 1. branch mode: served in the disposable worktree, not in the repo
id=$("$legs" request --run-dir "$D" --arch wormhole --label branch --branch legs-test --sha "$sha" --cmd 'echo hello; test -f marker.txt' --cwd .)
"${serve[@]}" > /dev/null 2>&1; src=$?
out=$("$legs" wait --run-dir "$D" "$id"); rc=$?
[ "$src" = 0 ] && [ "$rc" = 0 ] && grep -q hello "$D/legs/$id/leg.log"; check $? "branch leg passes" "serve=$src wait=$rc $out"
[ -e "$tmp/wt-legs-wormhole/marker.txt" ] && [ ! -e "$repo/marker.txt" ] && grep -q "\"head\": \"$sha\"" "$D/legs/$id/result.json"
check $? "branch leg ran at $sha in wt-legs-wormhole, repo untouched"

# 2. patch mode: applied to the repo, run, reverted
echo m2 > "$repo/marker2.txt"; git -C "$repo" add marker2.txt; git -C "$repo" diff --cached > "$tmp/m2.diff"
git -C "$repo" rm -q --cached marker2.txt; rm -f "$repo/marker2.txt"
id=$("$legs" request --run-dir "$D" --arch wormhole --label patch --patch "$tmp/m2.diff" --cmd 'test -f marker2.txt && ! test -f marker.txt && test "$CHIP_ARCH" = wormhole && test "$FOO" = bar' --cwd . --env FOO=bar)
"${serve[@]}" > /dev/null 2>&1; src=$?
out=$("$legs" wait --run-dir "$D" "$id"); rc=$?
[ "$src" = 0 ] && [ "$rc" = 0 ]; check $? "patch leg passes (env + CHIP_ARCH seen)" "serve=$src wait=$rc $out $(cat "$D/legs/$id/leg.log")"
[ -z "$(git -C "$repo" status --porcelain)" ] && grep -q '"revert_ok": true' "$D/legs/$id/result.json"; check $? "repo clean after the patch leg, revert_ok recorded"

# 3. timeout
id=$("$legs" request --run-dir "$D" --arch wormhole --label slow --branch legs-test --sha "$sha" --cmd 'sleep 5' --timeout 1 --cwd .)
"${serve[@]}" > /dev/null 2>&1
out=$("$legs" wait --run-dir "$D" "$id"); rc=$?
[ "$rc" = 124 ]; check $? "timeout leg reports 124" "wait=$rc $out"

# 4. heartbeat + status
alive=$D/legs/serve-wormhole@$(hostname).alive
grep -qE "^wormhole $(hostname) pid=[0-9]+ [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z busy=-$" "$alive" 2>/dev/null; check $? "serve left a heartbeat" "$(cat "$alive" 2>/dev/null)"
out=$("$legs" status --run-dir "$D")
head -1 <<< "$out" | grep -qE "^serve wormhole: $(hostname) pid=[0-9]+ last seen [0-9]+s ago \(busy=-\)$"; check $? "status shows the wormhole server" "$(head -1 <<< "$out")"
[ "$(grep -c ' done exit=' <<< "$out")" = 3 ] && [ "$(grep -vc '^serve ' <<< "$out")" = 3 ]; check $? "status lists 3 done requests" "$out"
echo "blackhole otherbox pid=1 $(date -u -d '-10 minutes' +%FT%TZ) busy=-" > "$D/legs/serve-blackhole@otherbox.alive"
"$legs" status --run-dir "$D" | grep -qE '^serve blackhole: otherbox pid=1 last seen [0-9]+s ago STALE \(busy=-\)$'; check $? "status marks a 10-minute-old heartbeat STALE"
rm -f "$D/legs/serve-blackhole@otherbox.alive"

# 5. refusals
"${serve[@]}" > /dev/null 2>&1; [ $? = 3 ]; check $? "serve --once with nothing open exits 3"
id=$("$legs" request --run-dir "$D" --arch blackhole --label other --branch legs-test --sha "$sha" --cmd true --cwd .)
"${serve[@]}" > /dev/null 2>&1; [ $? = 3 ] && "$legs" status --run-dir "$D" | grep -q "^$id blackhole other open "; check $? "wormhole server leaves a blackhole request open"
echo dirty >> "$repo/README"
id=$("$legs" request --run-dir "$D" --arch wormhole --label dirty --patch "$tmp/m2.diff" --cmd true --cwd .)
"${serve[@]}" > /dev/null 2>&1; out=$("$legs" wait --run-dir "$D" "$id"); rc=$?
git -C "$repo" checkout -q README
# untracked files (sfpi/.venv left by the harness) must NOT block patch mode
mkdir -p "$repo/tt_metal/tt-llk/tests/sfpi"; echo x > "$repo/tt_metal/tt-llk/tests/sfpi/untracked.txt"
uid=$("$legs" request --run-dir "$D" --arch wormhole --label untracked --patch "$tmp/m2.diff" --cmd 'test -f marker2.txt' --cwd .)
"${serve[@]}" > /dev/null 2>&1
"$legs" wait --run-dir "$D" "$uid" --timeout 60 > /dev/null 2>&1; check $? "patch leg runs despite untracked files" "$(tail -3 "$D/legs/$uid/leg.log" 2>/dev/null)"
grep -q '"revert_ok": true' "$D/legs/$uid/result.json"; check $? "revert_ok true with untracked files present"
rm -rf "$repo/tt_metal/tt-llk/tests/sfpi"
[ "$rc" = 96 ] && grep -q "modified tracked files" "$D/legs/$id/leg.log"; check $? "dirty repo refuses a patch leg with 96" "wait=$rc $out"
"$legs" request --run-dir "$D" --arch wormhole --label bad --cmd true > /dev/null 2>&1; [ $? = 2 ]; check $? "request without --branch/--sha or --patch exits 2"
"$legs" request --run-dir "$D" --arch 'Not-An-Arch' --label bad --branch b --sha s --cmd true > /dev/null 2>&1; [ $? = 2 ]; check $? "request with a bad arch exits 2"
"$legs" > /dev/null 2>&1; [ $? = 2 ]; check $? "no subcommand prints usage and exits 2"

[ "$fail" = 0 ] && echo "ALL PASS" || echo "SOME FAILED (artefacts in $tmp, kept)"; [ "$fail" = 0 ] || trap - EXIT
exit "$fail"

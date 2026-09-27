#!/usr/bin/env bash
# Host-only regression for the sweep setup-state provenance gate.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=setup_state_lib.sh
source "$HERE/setup_state_lib.sh" || exit 2

TMP=$(mktemp -d "${TMPDIR:-/tmp}/selftest-setup-state.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
BRANCH=nkapre/sfpi
FAILS=0

new_repo() {
  local dir=$1
  git init -q -b "$BRANCH" "$dir"
  git -C "$dir" config user.name "setup-state selftest"
  git -C "$dir" config user.email "setup-state-selftest@example.invalid"
  printf '%s\n' "fixture" > "$dir/tracked"
  git -C "$dir" add tracked
  git -C "$dir" commit -q -m fixture
}

for repo in sfpi gcc metal blaze; do new_repo "$TMP/$repo"; done
GCC_SHA=$(git -C "$TMP/gcc" rev-parse HEAD)
git -C "$TMP/sfpi" update-index --add --cacheinfo "160000,$GCC_SHA,gcc"
git -C "$TMP/sfpi" commit -q -m 'pin gcc'
mkdir -p "$TMP/install"
printf '#!/usr/bin/env bash\nexit 0\n' > "$TMP/install/g++"
chmod +x "$TMP/install/g++"
STATE=$TMP/SETUP-STATE.env

write_state() {
  {
    echo "CRAQ_SETUP_BRANCH=$BRANCH"
    echo "CRAQ_SFPI_DIR=$TMP/sfpi"
    echo "CRAQ_GCC_DIR=$TMP/gcc"
    echo "CRAQ_METAL_DIR=$TMP/metal"
    echo "CRAQ_BLAZE_DIR=$TMP/blaze"
    echo "CRAQ_COMPILER=$TMP/install/g++"
    echo "CRAQ_GCC_PIN_SHA=$GCC_SHA"
    for pair in SFPI:sfpi GCC:gcc METAL:metal BLAZE:blaze; do
      name=${pair%%:*}; dir=${pair#*:}
      echo "CRAQ_${name}_SHA=$(git -C "$TMP/$dir" rev-parse HEAD)"
      echo "CRAQ_${name}_BRANCH=$(git -C "$TMP/$dir" rev-parse --abbrev-ref HEAD)"
    done
  } > "$STATE"
}

expect() {
  local name=$1 want=$2 rc
  shift 2
  "$@" >/dev/null 2>&1; rc=$?
  if [ "$rc" = "$want" ]; then
    echo "  PASS: $name"
  else
    echo "  FAIL: $name (want rc=$want got rc=$rc)"
    FAILS=$((FAILS + 1))
  fi
}

write_state
expect "matching branch checkouts pass" 0 verify_craq_setup_state "$STATE"

git -C "$TMP/gcc" checkout -q --detach
write_state
expect "detached GCC at recorded SFPI gitlink passes" 0 verify_craq_setup_state "$STATE"

printf '%s\n' moved >> "$TMP/metal/tracked"
git -C "$TMP/metal" commit -qam moved
expect "checkout moved after state was written is refused" 2 verify_craq_setup_state "$STATE"
git -C "$TMP/metal" reset -q --hard HEAD~1

printf '%s\n' alternate >> "$TMP/gcc/tracked"
git -C "$TMP/gcc" commit -qam alternate
write_state
expect "GCC not matching SFPI gitlink is refused" 2 verify_craq_setup_state "$STATE"
git -C "$TMP/gcc" reset -q --hard "$GCC_SHA"

git -C "$TMP/blaze" checkout -q --detach
write_state
expect "detached non-GCC checkout is refused" 2 verify_craq_setup_state "$STATE"

if [ "$FAILS" -eq 0 ]; then
  echo "setup-state self-test: ALL PASS"
  exit 0
fi
echo "setup-state self-test: $FAILS FAILURE(S)"
exit 1

#!/bin/bash
# C0 item 2 and condition 6 (torrentci#3525): replay consecutive tt-metal
# main commits k .. k+10 in one pod, each from a cleaned build tree.
#
#   replay.sh warm   k cold (writes); k again (the all-hit floor); then each
#                    k+i reads and writes one cumulative cache: the
#                    warm-per-push case (r2.4), where k+i's run is the queue
#                    entry on top of every earlier main commit's markers.
#   replay.sh stale  k cold (writes); then each k+i reads, read-only, a fresh
#                    copy of k's markers: up to 10 commits stale (r2.8 item 2).
#
# Every k+i in this range has the same submodule pins and CPM inputs as the
# checked-out tip (checked below), so the workspace's submodules and CPM
# cache stay valid.
set -euo pipefail
mode=$1
commits=(
  7068c52b5d4eee67850df2302fc408da1a9cc275
  8f710be1532fb2a24526a76396dae8c4b2fd5af9
  c909fa92d575926bc2f4a5e18bfca7785d0ba9be
  28569d6cc10b0b94638a45edadcc743ef8a675fa
  19a05e4e92eccd344720986a341e160a7085c9eb
  2a1236fa6e78c49d61986e403d1fe09dd1b981cb
  e81af37a2b7471d3f7ddfacfe38272021c8fc22b
  fb1591e4ef978c91b09e8292e0fb4eabc85b1f96
  e755b189d06f78fd831bbce98c51b54b45f99b97
  a695405830522a4fc3f51e1b06e02f7bd3c372a2
  79951bd44b7484a136b3619e0d81f26baa8a1936
)
rm -rf /tmp/c0 && cp -a .torrentci/c0-ctcache /tmp/c0
export C0_TOOLS=/tmp/c0
g() { git -c safe.directory='*' "$@"; }
pins() { g ls-tree "$1" tt_metal/third_party/umd tt_metal/third_party/tracy tt_metal/third_party/tt-cluster-descriptors | awk '{print $3}' | tr '\n' ' '; }
head_pins=$(pins HEAD)
for c in "${commits[@]}"; do
  g cat-file -e "$c^{commit}" || { echo "commit $c is not in the workspace"; exit 1; }
  [ "$(pins "$c")" = "$head_pins" ] || { echo "submodule pins differ at $c"; exit 1; }
done
checkout() {
  g checkout --quiet --detach "$1"
  echo "== $(g log -1 --format='%h %s' | cut -c1-100)"
  if [ -n "${2:-}" ]; then
    local changed headers
    changed=$(g diff --name-only "$2" "$1" | grep -c -E '\.(cpp|cc|c|h|hpp|inl|ipp|tpp)$' || true)
    headers=$(g diff --name-only "$2" "$1" | grep -E '\.(h|hpp|inl|ipp|tpp)$' | tr '\n' ' ' || true)
    echo "C0COMMIT $1 cxx_files_changed=$changed headers_changed=[${headers}]"
  fi
}
export CTCACHE_DIR=/tmp/ctcache-main
rm -rf "$CTCACHE_DIR" /tmp/ctcache-k
checkout "${commits[0]}"
"$C0_TOOLS/analyse.sh" k0-cold ctcache
cp -a "$CTCACHE_DIR" /tmp/ctcache-k
echo "markers after k0: $(find /tmp/ctcache-k -type f | wc -l)"
if [ "$mode" = warm ]; then
  "$C0_TOOLS/analyse.sh" k0-floor ctcache
  export C0_REF_DIR=/tmp/ctcache-k
  for i in $(seq 1 10); do
    checkout "${commits[$i]}" "${commits[$((i - 1))]}"
    "$C0_TOOLS/analyse.sh" "k$i-warm" ctcache
  done
else
  for i in $(seq 1 10); do
    checkout "${commits[$i]}" "${commits[0]}"
    rm -rf /tmp/ctcache-stale && cp -a /tmp/ctcache-k /tmp/ctcache-stale
    CTCACHE_DIR=/tmp/ctcache-stale C0_READ_ONLY=1 "$C0_TOOLS/analyse.sh" "k$i-stale" ctcache
  done
fi
echo "== C0 results ($mode)"
cat /tmp/c0-out/results.txt

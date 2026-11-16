#!/bin/sh
# Seed-pack checkout measurement (torrentci design #2852, slice S0).
#
# Prod CPU and disk only, over file://: TLS and network are out of scope.
# Builds a bare "mirror" from this job's own workspace (the default branch and
# the tags it carries), builds a week-old default-branch seed from it, then
# times, per run, the plain path (today's fetch + checkout) against the seed
# path (install seed pack + hidden ref, delta fetch, drift check, checkout,
# commit-DAG rehash, HEAD-tree rebuild), and checks the two results are equal.
#
# Usage: seed-measure.sh <cpus> <runs>
set -eu
CPUS=${1:-4}
RUNS=${2:-3}
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="*"
WS=$(pwd)
X=${TMPDIR:-/tmp}/seedm
rm -rf "$X"
mkdir -p "$X"

up() { awk '{print $1}' /proc/uptime; }
cpu() { awk '/^user_usec|^system_usec/{printf "%s ", $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || true; }
d() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.2f", b-a}'; }
measure() {
  label=$1
  shift
  s=$(up)
  c0=$(cpu)
  "$@"
  e=$(up)
  c1=$(cpu)
  echo "$c0 $c1" | awk -v l="$label" -v s="$s" -v e="$e" \
    '{printf "M %s real=%.2f user=%.2f sys=%.2f\n", l, e-s, ($3-$1)/1e6, ($4-$2)/1e6}'
}

echo "== env"
nproc
cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo no-cpu.max
git --version
(ldd --version 2>&1 || true) | head -1
sha256sum --version 2>&1 | head -1 || true
grep -m1 'model name' /proc/cpuinfo
echo "host_cpus=$(grep -c ^processor /proc/cpuinfo) cpus=$CPUS runs=$RUNS"
df -h "$WS" "$X" | sed 's/^/df /'

SHA=$(git rev-parse HEAD)
B=main
git rev-parse -q --verify refs/remotes/origin/$B >/dev/null || {
  echo "no refs/remotes/origin/$B in the workspace"
  exit 1
}
MAIN=$(git rev-parse refs/remotes/origin/$B)
OLD=$(git rev-list -1 --first-parent --before="7 days ago" refs/remotes/origin/$B)
echo "SHA=$SHA main=$MAIN seed_tip=$OLD ($(git log -1 --format=%cI "$OLD"))"
echo "behind=$(git rev-list --count "$OLD..$MAIN") ahead_of_main=$(git rev-list --count "$MAIN..$SHA")"

G="git -c protocol.version=2 -c pack.threads=$CPUS -c core.fsync=none"

echo "== mirror (bare, bitmapped, from the workspace; setup, not part of the result)"
M=$X/mirror.git
git init -q --bare "$M"
echo "$WS/.git/objects" > "$M/objects/info/alternates"
git for-each-ref --format='%(objectname) %(refname)' refs/tags refs/remotes/origin/$B |
  sed "s# refs/remotes/origin/$B\$# refs/heads/$B#" |
  awk '{print "create " $2 " " $1}' | git -C "$M" update-ref --stdin
# The job's own commit stands in for a pull-request head: the real mirror
# carries every branch.
git -C "$M" update-ref refs/heads/seed-measure-head "$SHA"
git -C "$M" config uploadpack.allowAnySHA1InWant true
git -C "$M" symbolic-ref HEAD refs/heads/$B
measure mirror-repack git -C "$M" -c pack.threads="$CPUS" repack -a -d -b -q
rm -f "$M/objects/info/alternates"
git -C "$M" count-objects -v | tr '\n' ' '
echo
echo "mirror_refs=$(git -C "$M" for-each-ref | wc -l)"

echo "== seed build (mirror side, pack-objects --revs from the week-old tip, pack.threads=1)"
S=$X/seed
mkdir -p "$S"
mkseed() {
  echo "$OLD" | git -C "$M" -c pack.threads=1 pack-objects --revs -q "$S/pack" > "$S/id"
}
measure seed-build mkseed
ID=$(cat "$S/id")
[ -f "$S/pack-$ID.rev" ] || git -C "$M" index-pack --rev-index "$S/pack-$ID.pack" >/dev/null 2>&1 || true
ls -l "$S"
echo "seed_id=$ID seed_objects=$(git show-index < "$S/pack-$ID.idx" | wc -l)"
fsck_seed() {
  rm -rf "$X/fsck.git"
  git init -q --bare "$X/fsck.git"
  cp "$S"/pack-"$ID".* "$X/fsck.git/objects/pack/"
  git -C "$X/fsck.git" update-ref refs/heads/$B "$OLD"
  git -C "$X/fsck.git" fsck --strict --no-dangling --no-progress
}
measure seed-fsck-strict fsck_seed
rm -rf "$X/fsck.git"
measure seed-sha256 sha256sum "$S/pack-$ID.pack" "$S/pack-$ID.idx" "$S/pack-$ID.rev"

# Plain path: today's sequence (init, remote, fetch SHA + default, checkout).
plain() {
  R=$1
  rm -rf "$R"
  $G init -q "$R"
  git -C "$R" remote add origin "file://$M"
  a=$(up)
  (cd "$R" && $G fetch -q -- origin "$SHA" "refs/heads/$B:refs/remotes/origin/$B")
  b=$(up)
  (cd "$R" && $G -c checkout.workers="$CPUS" checkout -q --detach "$SHA")
  c=$(up)
  echo "P plain fetch=$(d "$a" "$b") checkout=$(d "$b" "$c") total=$(d "$a" "$c")"
}

# DAG rehash: pack every commit reachable from SHA and the default branch with
# no deltas, index-pack it in a scratch repository, compare the names.
dag_verify() {
  R=$1
  V=$X/dagv.git
  rm -rf "$V"
  git init -q --bare "$V"
  git -C "$R" rev-list "$SHA" "refs/remotes/origin/$B" | sort > "$X/want"
  git -C "$R" -c pack.threads="$CPUS" pack-objects --stdout -q --window=0 < "$X/want" |
    git -C "$V" -c pack.threads="$CPUS" index-pack --stdin > /dev/null
  git -C "$V" show-index < "$(ls "$V"/objects/pack/*.idx)" | awk '{print $2}' | sort > "$X/got"
  rm -rf "$V"
  cmp -s "$X/want" "$X/got"
}

# HEAD tree rebuild: rehash every worktree file in CPUS ways, rebuild the tree
# in a scratch index, compare with the commit's tree.
tree_verify() {
  R=$1
  (
    cd "$R"
    [ "$(git cat-file commit "$SHA" | git hash-object -t commit --stdin)" = "$SHA" ] || exit 1
    T=$(git rev-parse "$SHA^{tree}")
    git ls-files -s > "$X/ls"
    awk -F'\t' '{split($1,m," "); if (m[1]=="100644"||m[1]=="100755") print $2}' "$X/ls" > "$X/files"
    rm -f "$X"/p.*
    awk -v n="$CPUS" -v x="$X" '{print > (x "/p." (NR % n))}' "$X/files"
    for p in "$X"/p.*; do
      git hash-object --stdin-paths < "$p" > "$p.oid" &
    done
    wait
    : > "$X/hashed"
    for p in "$X"/p.?; do
      [ -f "$p" ] || continue
      paste "$p.oid" "$p" >> "$X/hashed"
    done
    awk -F'\t' '{split($1,m," "); if (m[1]=="120000") print $2}' "$X/ls" | while IFS= read -r l; do
      printf '%s\t%s\n' "$(printf %s "$(readlink "$l")" | git hash-object --stdin)" "$l"
    done >> "$X/hashed"
    awk -F'\t' 'NR==FNR{h[$2]=$1; next} {split($1,m," "); o=(m[1]=="160000")?m[2]:h[$2]; print m[1]" "o"\t"$2}' \
      "$X/hashed" "$X/ls" > "$X/info"
    rm -f "$X/vidx"
    GIT_INDEX_FILE=$X/vidx git update-index --index-info < "$X/info"
    VT=$(GIT_INDEX_FILE=$X/vidx git write-tree)
    rm -f "$X/vidx" "$X"/p.* "$X/info" "$X/hashed"
    [ "$VT" = "$T" ]
  )
}

# Seed path, with the hidden ref.
seeded() {
  R=$1
  rm -rf "$R"
  $G init -q "$R"
  git -C "$R" remote add origin "file://$M"
  a=$(up)
  mkdir -p "$R/.git/torrentci-seed"
  for f in pack rev idx; do
    cp "$S/pack-$ID.$f" "$R/.git/torrentci-seed/"
  done
  b=$(up)
  (cd "$R/.git/torrentci-seed" && sha256sum "pack-$ID.pack" "pack-$ID.idx" "pack-$ID.rev" > /dev/null)
  c=$(up)
  for f in pack rev idx; do
    mv "$R/.git/torrentci-seed/pack-$ID.$f" "$R/.git/objects/pack/"
  done
  rm -rf "$R/.git/torrentci-seed"
  git -C "$R" update-ref refs/torrentci-seed/tip "$OLD"
  e=$(up)
  (cd "$R" && $G fetch -q -- origin "$SHA" "refs/heads/$B:refs/remotes/origin/$B")
  f=$(up)
  git -C "$R" merge-base --is-ancestor "$OLD" "refs/remotes/origin/$B"
  git -C "$R" update-ref -d refs/torrentci-seed/tip
  g=$(up)
  (cd "$R" && $G -c checkout.workers="$CPUS" checkout -q --detach "$SHA")
  h=$(up)
  dag_verify "$R" || echo "DAG MISMATCH"
  i=$(up)
  tree_verify "$R" || echo "TREE MISMATCH"
  j=$(up)
  echo "P seed copy=$(d "$a" "$b") sha256=$(d "$b" "$c") install=$(d "$c" "$e") fetch=$(d "$e" "$f") drift=$(d "$f" "$g") checkout=$(d "$g" "$h") dag_verify=$(d "$h" "$i") tree_verify=$(d "$i" "$j") total=$(d "$a" "$j") total_unverified=$(d "$a" "$h")"
}

snapshot() {
  R=$1
  out=$2
  (
    cd "$R"
    git for-each-ref > "$out.refs"
    git rev-list --objects --all | sort > "$out.objs"
    git count-objects -v | awk '/^count|^in-pack/' > "$out.count"
    # Every distinct object in the ODB, reachable or not: equal to the
    # reachable set when the seed adds nothing outside the closure.
    git cat-file --batch-all-objects --batch-check='%(objectname)' | sort -u > "$out.odb"
    git rev-list --objects --all | cut -d' ' -f1 | sort -u > "$out.reach"
    cut -d' ' -f3- .git/logs/refs/remotes/origin/$B | sed 's/^[^>]*> [0-9]* [-+][0-9]*\t//' > "$out.reflog"
    cp .git/config "$out.config"
    git rev-parse HEAD > "$out.head"
    git status --porcelain | wc -l > "$out.status"
    ls -l .git/objects/pack > "$out.packs"
  )
}

n=1
while [ "$n" -le "$RUNS" ]; do
  echo "== run $n"
  measure plain-total plain "$X/plain"
  snapshot "$X/plain" "$X/plain"
  rm -rf "$X/plain"
  measure seed-total seeded "$X/seeded"
  snapshot "$X/seeded" "$X/seeded"
  du -sk "$X/seeded/.git/objects/pack" | sed 's/^/seeded_pack_kib /'
  sed "s/^/seeded_packs /" "$X/seeded.packs"
  sed "s/^/plain_packs /" "$X/plain.packs"
  rm -rf "$X/seeded"
  for k in refs objs odb reflog config head status; do
    if cmp -s "$X/plain.$k" "$X/seeded.$k"; then echo "EQ $k equal"; else echo "EQ $k DIFFER"; diff "$X/plain.$k" "$X/seeded.$k" | head -10; fi
  done
  echo "refs=$(wc -l < "$X/plain.refs") reachable_objects=$(wc -l < "$X/plain.objs")"
  for v in plain seeded; do
    echo "$v odb_distinct=$(wc -l < "$X/$v.odb") reachable_distinct=$(wc -l < "$X/$v.reach") $(tr '\n' ' ' < "$X/$v.count")"
    cmp -s "$X/$v.odb" "$X/$v.reach" && echo "EQ $v odb == reachable" || echo "EQ $v odb DIFFERS from reachable"
  done
  cat "$X/plain.count" "$X/seeded.count" | tr '\n' ' '
  echo
  n=$((n + 1))
done
rm -rf "$X"
echo "== done"

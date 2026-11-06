#!/bin/bash
# Two-tree A/B on the extra LoudBoxes (lb2): the regress trees (manifest trees, e.g. main and new_288afd2) as zstd
# tarballs on /data, extracted on the LB node at their own paths (/home/aagarwal/tt-metal-main, -new), so ab_pair.sh
# runs both routers in one lb2 job on one host (coordinator 2026-11-04).
#
#   lb2_trees.sh pack TREE...   on a09u32: tar + zstd of the tree's path -> /data/aagarwal/lb2/regress_trees/TREE.tar.zst
#                               (+ .sha256 of the archive, read back after the write). Build objects, .git, caches and
#                               generated output left out (as run_group.py stage does: runtime/**/*.o stay).
#   lb2_trees.sh sync TREE...   on an lb2 node, first thing in the job command: when the node's copy is not the staged
#                               archive (stamp), check /home space, verify the archive's sha256, extract it next to the
#                               tree path and swap it in. A path that exists without our stamp is never touched.
set -uo pipefail
D=/data/aagarwal/lb2/regress_trees
TP=/home/aagarwal/tt-metal/fabric_exp/testplan
MIN_FREE_GB=${LB2_TREES_MIN_FREE_GB:-40}
mode=${1:-}; shift || true
say() { echo "[lb2_trees $(date -u +%T)] $*"; }
tree_path() {
    python3 - "$TP/manifest.yaml" "$1" <<'EOF'
import sys, yaml
t = yaml.safe_load(open(sys.argv[1]))["trees"].get(sys.argv[2]) or {}
p = t.get("path") or ""
print(p if p.startswith("/home/aagarwal/") else "")
EOF
}
[ $# -gt 0 ] || { echo "usage: lb2_trees.sh pack|sync TREE..."; exit 2; }
case $mode in
pack)
    [ "$(hostname)" = bh-lb-120-a09u32 ] || { say "!!! pack on bh-lb-120-a09u32 (where the trees live)"; exit 1; }
    mkdir -p $D
    for n in "$@"; do
        p=$(tree_path "$n"); [ -n "$p" ] && [ -d "$p" ] || { say "!!! tree $n: no /home/aagarwal path"; exit 1; }
        part=$D/.$n.tar.zst.part
        say "pack $n: $p -> $D/$n.tar.zst"
        tar -C "$p" --exclude=./.git --exclude=./.cpmcache --exclude=./generated --exclude=./.pytest_cache \
            --exclude=./tt-train --exclude='__pycache__' --exclude='*.pyc' \
            --exclude='./build_Release/*.o' --exclude='./build_Release/*.o.d' --exclude='./build_Release/*.a' \
            --exclude='./build_Release/*CMakeFiles*' --exclude='./build_Release/_deps/*-build' \
            --exclude='./build_Release/_deps/*-subbuild' --exclude=./build_Release/Testing \
            --exclude='./build_Release/.ninja_*' --exclude=./build_Release/build.ninja \
            --exclude=./build_Release/compile_commands.json -cf - . | zstd -T8 -3 -q -f -o "$part" || { say "!!! tar/zstd failed"; exit 1; }
        sum=$(sha256sum "$part" | cut -d' ' -f1)           # read back from /data: the bytes the node will read
        zstd -t -q "$part" || { say "!!! archive test failed"; exit 1; }
        mv -f "$part" $D/$n.tar.zst && echo "$sum" > $D/$n.tar.zst.sha256
        say "packed $n: $(du -h $D/$n.tar.zst | cut -f1), sha256 ${sum:0:16}, tree $(python3 -c "import json;print(json.load(open('$p/.regress_tree.json'))['sha'][:12])" 2>/dev/null)"
    done ;;
sync)
    for n in "$@"; do
        p=$(tree_path "$n"); [ -n "$p" ] || { say "!!! tree $n: no /home/aagarwal path in the manifest"; exit 1; }
        tb=$D/$n.tar.zst; [ -f "$tb" ] && [ -f "$tb.sha256" ] || { say "!!! no staged archive for $n"; exit 1; }
        want=$(cat "$tb.sha256")
        if [ -f "$p/.lb2_tree_stamp" ] && [ "$(cat "$p/.lb2_tree_stamp")" = "$want" ]; then
            say "$n: $p is the staged archive (${want:0:16})"; continue
        fi
        if [ -e "$p" ] && [ ! -f "$p/.lb2_tree_stamp" ]; then
            say "!!! $p exists on $(hostname) and is not an lb2 tree: not touching it"; exit 1
        fi
        free=$(df -BG --output=avail /home | tail -1 | tr -dc 0-9)
        [ "${free:-0}" -ge "$MIN_FREE_GB" ] || { say "!!! /home has ${free} GB free (< $MIN_FREE_GB)"; exit 1; }
        got=$(sha256sum "$tb" | cut -d' ' -f1)
        [ "$got" = "$want" ] || { say "!!! $tb sha256 ${got:0:16} != staged ${want:0:16}"; exit 1; }
        rm -rf "$p.lb2new"; mkdir -p "$p.lb2new"
        zstd -dc "$tb" | tar -x -C "$p.lb2new" || { rm -rf "$p.lb2new"; say "!!! extract failed"; exit 1; }
        "$p.lb2new/python_env/bin/python" -c 'import sys' || { rm -rf "$p.lb2new"; say "!!! the venv python does not start (uv Python missing on the node?)"; exit 1; }
        rm -rf "$p"; mv "$p.lb2new" "$p"; echo "$want" > "$p/.lb2_tree_stamp"
        say "$n: extracted to $p (${want:0:16}), $(df -BG --output=avail /home | tail -1 | tr -dc 0-9) GB free on /home"
    done ;;
*) echo "usage: lb2_trees.sh pack|sync TREE..."; exit 2 ;;
esac

#!/bin/bash
# t477/t480: build 63329812d9f (HEAD + submodule tarballs) on blx01 under /var/tmp/fasth3/t477/b (ionice -c3 nice, no /home writes).
set -eo pipefail
T=/var/tmp/fasth3/t477
trap 'echo $? > $T/build.rc' EXIT
export HOME=/var/tmp/fasth3/home XDG_CACHE_HOME=/var/tmp/fasth3/home/.cache TMPDIR=/var/tmp/fasth3/tmp
export CCACHE_DISABLE=1 CPM_SOURCE_CACHE=/var/tmp/fasth3/.cpmcache
free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9); [ "$free" -ge 300 ] || { echo "[t477] / free ${free}G"; exit 5; }
mkdir -p $T/b
echo "extract $(date -u)"
for f in head tracy tt-cluster-descriptors umd; do ionice -c3 nice -n 19 tar -x -f $T/$f.tar -C $T/b; done
cd $T/b
echo "build start $(date -u)"
ionice -c3 nice -n 19 ./build_metal.sh --build-type Release
echo "build done $(date -u)"
ls -la ttnn/ttnn/_ttnn*.so

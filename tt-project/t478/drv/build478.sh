#!/bin/bash
# t478: extract and build base then bundle trees on blx01 under /var/tmp/fasth3/t478 (ionice -c3 nice, no /home writes).
set -eo pipefail
T=/var/tmp/fasth3/t478
trap 'echo $? > $T/build.rc' EXIT
export HOME=/var/tmp/fasth3/home XDG_CACHE_HOME=/var/tmp/fasth3/home/.cache TMPDIR=/var/tmp/fasth3/tmp
export CCACHE_DISABLE=1 CPM_SOURCE_CACHE=/var/tmp/fasth3/.cpmcache
# attempt 1 died on pthread_create EAGAIN in a link step under load; cap parallelism
export CMAKE_BUILD_PARALLEL_LEVEL=24
for a in base bundle; do
  if [ ! -f $T/$a/build_metal.sh ]; then
    mkdir -p $T/$a
    echo "[$a] extract $(date -u)"
    ionice -c3 nice -n 19 tar -xzf $T/$a.tar.gz -C $T/$a
  fi
  cd $T/$a
  echo "[$a] build start $(date -u)"
  ionice -c3 nice -n 19 ./build_metal.sh --build-type Release
  echo "[$a] build done $(date -u) rc=$?"
  ls -la ttnn/ttnn/_ttnn*.so
done
rm -f $T/base.tar.gz $T/bundle.tar.gz
echo "all done $(date -u)"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# fleet/maxutil_pack.sh: the prebuilt tt-metal max-util archive that run.sh's max-util step unpacks.
# Nothing here opens a device: pack and check run no test, only --gtest_list_tests.
#   maxutil_pack.sh pack <tt-metal dir> <out dir> [<mpi prefix>]   write <out dir>/maxutil-<sha12>.tar.zst, print its pin
#   maxutil_pack.sh pin <archive>                                  print the pin lines for that archive
#   maxutil_pack.sh verify <archive> [<pin file>]                  exit 0 when the archive's sha256 is the pin's
#                                                                  (default pin file fleet/maxutil.pin)
#   maxutil_pack.sh repack <archive> <out dir>                     add the system libs it lacks (syslib/ below) to an
#                                                                  archive packed before syslib/, print the new pin
#   maxutil_pack.sh check <archive> <dir>                          unpack into <dir> (new or empty); exit 0 when ldd finds
#                                                                  every lib in the archive or glibc or libstdc++ (base_lib)
#                                                                  and the binary lists its tests
# The archive holds one top dir, maxutil/:
#   maxutil/MANIFEST   bin=, home=, lib= and opal_prefix= (empty without mpi), each relative to maxutil/
#   maxutil/tt-metal/  the unit_tests_didt_max_util binary, the .so files of the build_Release lib dirs (LIBDIRS),
#                      tt_metal/ (mesh graph descriptors included), runtime/sfpi, runtime/hw, tests/didt and
#                      tests/tt_metal/tt_fabric/physical_groupings: what the JIT kernel build and the mesh open read
#   maxutil/openmpi/   the mpi prefix, when given: libtt_metal.so needs libmpi.so.40 (ldd)
#   maxutil/syslib/    every lib from the build host's system dirs (MAXUTIL_SYSDIRS) that the binary or a lib in a MANIFEST
#                      lib dir loads and that is not glibc or the C++ runtime (base_lib): libevent_core and
#                      libevent_pthreads (openmpi), libhwloc, libnuma, libudev, libnl-3, krb5 ... (goal 53: a host
#                      without libevent-pthreads-2.1-7 ended every max-util run INCOMPLETE); last in MANIFEST lib=, which
#                      maxutil.py puts on LD_LIBRARY_PATH
# The tar is reproducible (sorted names, mtime 0, owner 0), so the same tree packs to the same sha256; the pin is that
# sha256 of the archive file, and the file name carries its first 12 hex.
set -euo pipefail
BIN=build_Release/test/didt/unit_tests_didt_max_util
LIBDIRS="build_Release/lib build_Release/tt_metal build_Release/tt_metal/third_party/umd/lib build_Release/tt_stl"
TREE="tt_metal runtime/sfpi runtime/hw runtime/sfpi-version.cmake tests/didt tests/tt_metal/tt_fabric/physical_groupings"
# the max-util fixture reads this at SetUpTestSuite (an archive without it ran 0 s of its 60 s).
# With no TT_METAL_RUNTIME_ROOT, tt-metal takes build_Release (beside libtt_metal.so) as its root, so the source
# tt_metal/soc_descriptors also goes in as build_Release/tt_metal/soc_descriptors; MANIFEST runtime_root names tt-metal.
SOC=build_Release/tt_metal/soc_descriptors/blackhole_140_arch.yaml
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
SYSDIRS=${MAXUTIL_SYSDIRS:-/lib /lib64 /usr/lib /usr/lib64}
die() { echo "maxutil_pack: $*" >&2; exit 1; }

# base_lib <soname>: glibc, the dynamic loader and the C++ runtime, which every host has and which stay out of the archive
base_lib() {
  case $1 in
    linux-vdso.so.*|ld-linux*.so.*|libc.so.*|libm.so.*|libmvec.so.*|libdl.so.*|libpthread.so.*|librt.so.*|libresolv.so.*| \
      libutil.so.*|libanl.so.*|libnsl.so.1|libBrokenLocale.so.*|libthread_db.so.*|libnss_*.so.*|libstdc++.so.*|libgcc_s.so.*) return 0 ;;
  esac
  return 1
}

# outside_libs <maxutil dir>: one "<soname> <path or not found>" line per lib that the binary or an ELF .so in a MANIFEST
# lib dir loads (ldd, LD_LIBRARY_PATH from MANIFEST lib=) from outside <maxutil dir> and that is not a base lib, sorted
outside_libs() {
  local m=$1 lib bin f d
  lib=$(sed -n 's/^lib=//p' "$m/MANIFEST" | tr ':' '\n' | sed "s,^,$m/," | paste -sd:)
  bin=$m/$(sed -n 's/^bin=//p' "$m/MANIFEST")
  { echo "$bin"; for d in $(tr ':' ' ' <<< "$lib"); do [ -d "$d" ] && find "$d" -type f -name '*.so*'; done; } | LC_ALL=C sort -u | while read -r f; do
    head -c 4 "$f" | cmp -s - <(printf '\177ELF') || continue
    LD_LIBRARY_PATH=$lib ldd "$f" 2> /dev/null | sed -n 's/^[[:space:]]*\([^ ]*\) => \(not found\|[^ ]*\).*/\1 \2/p' || true
  done | while read -r n p; do
    case $p in "$m"/*) continue ;; esac
    base_lib "$n" || echo "$n $p"
  done | LC_ALL=C sort -u
}

# add_syslibs <maxutil dir>: copy each outside lib found in a system dir into syslib/ (by soname, symlinks followed) and
# put syslib last in MANIFEST lib=; a lib not found or outside the system dirs is an error
add_syslibs() {
  local m=$1 n p d sys out
  out=$(outside_libs "$m")
  [ -n "$out" ] || return 0
  mkdir -p "$m/syslib"
  while read -r n p; do
    [ "$p" != "not found" ] || die "ldd: $n not found"
    sys=""; for d in $SYSDIRS; do case $p in "$d"/*) sys=1 ;; esac; done
    [ -n "$sys" ] || die "ldd: $n resolves to $p, outside the archive and the system dirs ($SYSDIRS)"
    cp -L "$p" "$m/syslib/$n"; chmod 644 "$m/syslib/$n"
  done <<< "$out"
  grep -Eq '^lib=(.*:)?syslib(:|$)' "$m/MANIFEST" || sed -i 's/^lib=\(.*\)$/lib=\1:syslib/' "$m/MANIFEST"
  out=$(outside_libs "$m")
  [ -z "$out" ] || die "libs still outside the archive after syslib/: $(paste -sd' ' <<< "$out")"
}

# tar_maxutil <dir holding maxutil/> <out dir>: the reproducible archive maxutil-<sha12>.tar.zst in <out dir>; prints the pin
tar_maxutil() {
  local x=$1 out=$2 t r
  t=$(mktemp "$out/.maxutil-tar.XXXXXX")
  (cd "$x" && tar --sort=name --format=gnu --mtime=@0 --owner=0 --group=0 --numeric-owner --mode=go-w -cf - maxutil) \
    | zstd -q -f -T8 -10 -o "$t" || { rm -f "$t"; die "tar of $x/maxutil failed"; }
  r=$(sha256sum "$t" | cut -c1-12)
  chmod 644 "$t"
  mv -f "$t" "$out/maxutil-$r.tar.zst"
  pin_lines "$out/maxutil-$r.tar.zst"
}

pin_lines() {  # pin_lines <archive>
  local s
  s=$(sha256sum "$1" | cut -d' ' -f1)
  echo "sha256=$s"; echo "file=$(basename "$1")"; echo "bytes=$(stat -c %s "$1")"
}

pack() {
  local src=$1 out=$2 mpi=${3:-} libs="" d f p r lp
  [ -x "$src/$BIN" ] || die "no binary $src/$BIN"
  for d in $LIBDIRS; do [ -d "$src/$d" ] || die "no lib dir $src/$d"; done
  for d in $TREE; do [ -e "$src/$d" ] || die "no $src/$d"; done
  [ -f "$src/tt_metal/soc_descriptors/${SOC##*/}" ] || die "no $src/tt_metal/soc_descriptors/${SOC##*/}"
  [ -z "$mpi" ] || [ -d "$mpi/lib" ] || die "no mpi lib dir $mpi/lib"
  lp=$(for d in $LIBDIRS; do printf '%s:' "$src/$d"; done)${mpi:+$mpi/lib}
  # every lib the binary loads must come from the archive or the base system, else it is missing on another host
  while read -r f p; do
    case $p in
      "not found") die "ldd: $f not found (LD_LIBRARY_PATH=$lp)" ;;
      "$src"/*) continue ;;
    esac
    [ -n "$mpi" ] && case $p in "$mpi"/*) continue ;; esac
    r=""; for d in $SYSDIRS; do case $p in "$d"/*) r=1 ;; esac; done   # a system lib: base, or syslib/ (add_syslibs)
    [ -n "$r" ] || die "ldd: $f resolves to $p, outside the archive and the system dirs ($SYSDIRS) (pass its prefix as <mpi prefix>)"
  done < <(LD_LIBRARY_PATH=$lp ldd "$src/$BIN" | sed -n 's/^[[:space:]]*\([^ ]*\) => \(not found\|[^ ]*\).*/\1 \2/p')
  mkdir -p "$out"
  PACK_TMP=$(mktemp -d "$out/.maxutil-pack.XXXXXX")   # global: the EXIT trap runs after pack returns
  trap 'rm -rf "${PACK_TMP:?}"' EXIT
  local tmp=$PACK_TMP
  for d in $LIBDIRS; do libs="$libs:tt-metal/$d"; done
  mkdir -p "$tmp/x/maxutil/tt-metal"
  printf 'bin=tt-metal/%s\nhome=tt-metal\nlib=%s\nopal_prefix=%s\nruntime_root=tt-metal\n' "$BIN" "${libs#:}${mpi:+:openmpi/lib}" \
    "${mpi:+openmpi}" > "$tmp/x/maxutil/MANIFEST"
  # the .so files (and their symlinks) at the top of each lib dir, not the build objects beside them
  (cd "$src" && for d in $LIBDIRS; do find "$d" -maxdepth 1 \( -type f -o -type l \) -name '*.so*'; done) \
    | LC_ALL=C sort > "$tmp/libs.lst"
  [ -s "$tmp/libs.lst" ] || die "no .so files in $LIBDIRS"
  # stage the tree (the build stays untouched), then one reproducible tar of it
  # shellcheck disable=SC2086  # TREE is a word list
  (cd "$src" && tar -cf - "$BIN" -T "$tmp/libs.lst" $TREE) | tar -xf - -C "$tmp/x/maxutil/tt-metal"
  cp -a "$src/tt_metal/soc_descriptors" "$tmp/x/maxutil/tt-metal/build_Release/tt_metal/"
  [ -z "$mpi" ] || cp -a "$mpi/." "$tmp/x/maxutil/openmpi"
  add_syslibs "$tmp/x/maxutil"
  tar_maxutil "$tmp/x" "$out"
}

repack() {
  local a=$1 out=$2
  mkdir -p "$out"
  PACK_TMP=$(mktemp -d "$out/.maxutil-pack.XXXXXX")
  trap 'rm -rf "${PACK_TMP:?}"' EXIT
  mkdir "$PACK_TMP/x"
  zstd -q -dc "$a" | tar -xf - -C "$PACK_TMP/x" || die "cannot unpack $a"
  [ -f "$PACK_TMP/x/maxutil/MANIFEST" ] || die "no maxutil/MANIFEST in $a"
  add_syslibs "$PACK_TMP/x/maxutil"
  tar_maxutil "$PACK_TMP/x" "$out"
}

verify() {
  local a=$1 pin=${2:-$HERE/maxutil.pin} want got
  want=$(sed -n 's/^sha256=//p' "$pin")
  [ -n "$want" ] || die "no sha256= line in $pin"
  got=$(sha256sum "$a" | cut -d' ' -f1)
  [ "$got" = "$want" ] || die "$a sha256 $got is not the pin's $want ($pin)"
  echo "maxutil_pack: $a matches the pin ($want)"
}

check() {
  local a=$1 dir=$2 m bin home lib opal miss list n out
  [ -d "$dir" ] && [ -n "$(ls -A "$dir")" ] && die "$dir is not empty"
  mkdir -p "$dir"; dir=$(cd "$dir" && pwd)   # absolute: the test list runs from inside the tree
  zstd -q -dc "$a" | tar -xf - -C "$dir"
  m=$dir/maxutil
  [ -f "$m/MANIFEST" ] || die "no maxutil/MANIFEST in $a"
  bin=$m/$(sed -n 's/^bin=//p' "$m/MANIFEST"); home=$m/$(sed -n 's/^home=//p' "$m/MANIFEST")
  lib=$(sed -n 's/^lib=//p' "$m/MANIFEST" | tr ':' '\n' | sed "s,^,$m/," | paste -sd:)
  opal=$(sed -n 's/^opal_prefix=//p' "$m/MANIFEST")
  [ -x "$bin" ] || die "no binary $bin"
  [ -f "$home/$SOC" ] || die "no $SOC in $a"
  echo "ldd $bin (LD_LIBRARY_PATH from MANIFEST):"
  LD_LIBRARY_PATH=$lib ldd "$bin" | tee "$dir/ldd.txt"
  miss=$(grep -c 'not found' "$dir/ldd.txt" || true)
  [ "$miss" = 0 ] || die "ldd: $miss lib(s) not found"
  grep -q "=> $m/" "$dir/ldd.txt" || die "ldd: no lib resolved from the archive"
  # every lib the binary and the archive's libs load is in the archive or glibc/libstdc++: a host without it cannot run
  out=$(outside_libs "$m")
  [ -z "$out" ] || die "libs outside the archive and not glibc/libstdc++ (repack bundles them in syslib/): $(paste -sd' ' <<< "$out")"
  # the test list only: no fixture runs, so no device opens
  list=$(cd "$home" && env TT_METAL_HOME="$home" TT_METAL_RUNTIME_ROOT="$home" ARCH_NAME=blackhole LD_LIBRARY_PATH="$lib" \
    ${opal:+OPAL_PREFIX="$m/$opal"} timeout 120 "$bin" --gtest_list_tests) || die "$bin --gtest_list_tests failed"
  echo "$list"
  n=$(grep -c '^  ' <<< "$list" || true)
  [ "$n" -gt 0 ] || die "$bin lists no tests"
  grep -q 'All' <<< "$list" || die "$bin lists no *All* test (maxutil.py runs --gtest_filter=*All*)"
  echo "maxutil_pack: check PASS: 0 libs missing, 0 outside the archive but glibc/libstdc++, $n tests listed, no device opened"
}

case ${1:-} in
  pack) [ $# -ge 3 ] && [ $# -le 4 ] || die "usage: pack <tt-metal dir> <out dir> [<mpi prefix>]"; shift; pack "$@" ;;
  repack) [ $# -eq 3 ] || die "usage: repack <archive> <out dir>"; shift; repack "$@" ;;
  pin) [ $# -eq 2 ] || die "usage: pin <archive>"; pin_lines "$2" ;;
  verify) [ $# -ge 2 ] && [ $# -le 3 ] || die "usage: verify <archive> [<pin file>]"; shift; verify "$@" ;;
  check) [ $# -eq 3 ] || die "usage: check <archive> <dir>"; shift; check "$@" ;;
  *) sed -n '7,17p' "$0" >&2; exit 2 ;;
esac

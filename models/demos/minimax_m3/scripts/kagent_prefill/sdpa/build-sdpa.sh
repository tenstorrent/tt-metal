#!/bin/bash
# Surgical host rebuild for the sparse_sdpa_msa factory: recompile ONLY the transformer unity blob that holds
# sparse_sdpa_msa_{device_operation,program_factory}.cpp (worktree copies), with the canonical build's exact
# flags + PCH, and relink _ttnncpp.so from the canonical object files + this one object into
# $W/build_sdpa/lib. Everything else in build_sdpa is a symlink to the canonical build. Nothing is written
# under the canonical tree. Point the worktree's `build` symlink at build_sdpa to use it.
set -euo pipefail
C=/mnt/data/kernel-agent/canonical/tt-metal
CB=$C/build_RelWithDebInfo
W=/mnt/data/kernel-agent/dev/prefill-sdpa/tt-metal
B=$W/build_sdpa
SP=/mnt/data/kernel-agent/dev/prefill-sdpa/buildcmds
mkdir -p $B/obj $B/lib $SP
# 1) symlink farm (once)
for e in $(ls -A $CB); do
  [ "$e" = lib ] && continue
  [ -e $B/$e ] || ln -s $CB/$e $B/$e
done
for e in $(ls -A $CB/lib); do
  [ "$e" = _ttnncpp.so ] && continue
  [ -e $B/lib/$e ] || ln -s $CB/lib/$e $B/lib/$e
done
# 2) unity source: canonical copies of every file, except the two msa sources from the worktree
U=ttnn/cpp/ttnn/operations/transformer/CMakeFiles/ttnn_op_transformer.dir/Unity/unity_2_cxx.cxx
sed -e "s#\"$C/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_program_factory.cpp\"#\"$W/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_program_factory.cpp\"#" \
    -e "s#\"$C/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_device_operation.cpp\"#\"$W/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sparse_sdpa_msa_device_operation.cpp\"#" \
    $CB/$U > $B/obj/unity_2_cxx.cxx
grep -c "$W" $B/obj/unity_2_cxx.cxx | grep -q 2 || { echo "unity rewrite failed"; exit 1; }
# 3) compile with the canonical command (run from the canonical build dir: relative paths in the flags)
if [ ! -f $SP/compile.txt ]; then (cd $CB && ninja -t commands $U.o | tail -1) > $SP/compile.txt; fi
if [ ! -f $SP/link.txt ]; then (cd $CB && ninja -t commands ttnn/_ttnncpp.so | tail -1) > $SP/link.txt; fi
cc=$(cat $SP/compile.txt)
cc=${cc//"-MT $U.o -MF $U.o.d -o $U.o -c $CB/$U"/"-MT $B/obj/u2.o -MF $B/obj/u2.o.d -o $B/obj/u2.o -c $B/obj/unity_2_cxx.cxx"}
[[ "$cc" == *"$B/obj/unity_2_cxx.cxx"* ]] || { echo "compile cmd rewrite failed"; exit 1; }
t0=$(date +%s)
(cd $CB && nice -n 19 bash -c "$cc")
t1=$(date +%s); echo "[build-sdpa] compiled in $((t1-t0)) s"
# 4) link: canonical objects + ours, RUNPATH as installed ($ORIGIN/build/lib:$ORIGIN)
ln=$(cat $SP/link.txt)
ln=${ln//" $U.o "/" $B/obj/u2.o "}
ln=${ln//"-o ttnn/_ttnncpp.so"/"-o $B/lib/_ttnncpp.so.new"}
ln=${ln//"--dependency-file=ttnn/CMakeFiles/ttnncpp.dir/link.d"/"--dependency-file=$B/obj/link.d"}
ln=$(echo "$ln" | sed -E "s#-Wl,-rpath,[^ ]*#-Wl,-rpath,'\\\$ORIGIN/build/lib:\\\$ORIGIN'#")
[[ "$ln" == *"$B/obj/u2.o"* && "$ln" == *"$B/lib/_ttnncpp.so.new"* ]] || { echo "link cmd rewrite failed"; exit 1; }
(cd $CB && nice -n 19 bash -c "$ln")
mv -f $B/lib/_ttnncpp.so.new $B/lib/_ttnncpp.so
t2=$(date +%s); echo "[build-sdpa] linked in $((t2-t1)) s -> $B/lib/_ttnncpp.so"

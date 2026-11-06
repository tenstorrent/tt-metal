#!/bin/bash
# Placement audit, step 1: router-only offline builds of the base (7cd3c13a0db) and each arm from the credit gate's
# capture (~/.tt-metal-mainline-cov-bfc-rjcred, host df64d0ad677 = the base's), without PCH, in the bisect worktree;
# each build stripped to loadable sections + function symbols after it.
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad
B=/home/aagarwal/tt-metal-mainline-bisect
export TT_COV_DIR=/home/aagarwal/.tt-metal-mainline-cov-bfc-rjcred TT_COV_TREE=/home/aagarwal/tt-metal-mainline-glx
OC=/home/aagarwal/tt-metal-mainline/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy
cd /home/aagarwal/tt-metal-mainline/.scratch
for arm in ${ARMS:-base rg_s537 rg_s583 rg_s605 rg_s673 rg_r755 rg_d414 rg_all op_s617 kim_s755 kim_s816 kim_r713}; do
    (cd $B && git checkout -q -- . && git checkout -q --detach 7cd3c13a0db && for m in tracy tt-cluster-descriptors umd; do
        rm -rf tt_metal/third_party/$m; ln -s /home/aagarwal/tt-metal-mainline/tt_metal/third_party/$m tt_metal/third_party/$m; done)
    python3 $S/arms.py $B $arm || { echo "!!! arm $arm did not apply"; continue; }
    echo "== $arm $(date +%T)"
    taskset -c 2-15,17-32,34-47,51-63 python3 -B $S/covnp.py build --name a_$arm --router-only --jobs 48 --alt-tree $B 2>&1 | grep -E "^built|compiled"
    find $TT_COV_DIR/builds/a_$arm -name "*.elf" -size +0 | xargs -P 32 -n 1 sh -c "$OC --strip-debug --discard-locals -R '.rela*' \"\$0\" 2>/dev/null"
    find $TT_COV_DIR/builds/a_$arm -name "*.o" -delete
    du -sh $TT_COV_DIR/builds/a_$arm
done
(cd $B && git checkout -q -- .)
echo AUDIT_BUILDS_DONE

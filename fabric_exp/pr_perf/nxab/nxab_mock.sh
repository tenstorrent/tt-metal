#!/bin/bash
# Mock dry-run of the 200G NX A/B (CPU only, no device): every arm's /data tree, as frs_glx_nxab.sbatch runs it (a job
# farm with its own generated/, the arm's environment, the host libraries in LD_LIBRARY_PATH), on the single-Galaxy mock
# with pr_glx_nxab.yaml at one config (4352 B, benchmark off). Pass: every arm reaches "Launching programs" with no fatal
# error; c16 / c64 log the receiver override and their router compile args carry the slot counts; b / d / e compile the
# patched headers (checked in the kernel cache's sources). Caches are deleted afterwards.
set -uo pipefail
G=/data/aagarwal/frs-glx
V=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/nxab_mock
DESC=/home/aagarwal/tt-metal-main/tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml
declare -A SRC=([a]=$G/trees/nxab_host [b10]=$G/trees/nxab_b10 [b30]=$G/trees/nxab_b30 [b60]=$G/trees/nxab_b60
    [c16]=$G/trees/nxab_host [c64]=$G/trees/nxab_host [d64]=$G/trees/nxab_d64 [e60]=$G/trees/nxab_e60)
declare -A ARM_ENV=([c16]="TT_FABRIC_AB_VC0_RX_SLOTS=16" [c64]="TT_FABRIC_AB_VC0_RX_SLOTS=64")
rm -rf $V; mkdir -p $V
python3 - $G/yamls/pr_e7fe/pr_glx_nxab.yaml $V/nx1.yaml <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); t = d["Tests"][0]; t["benchmark_mode"] = False
t["parametrization_params"]["size"] = [4352]
yaml.safe_dump(d, open(sys.argv[2], "w"), sort_keys=False)
PY
for v in $(env | grep -oE '^TT_[A-Za-z0-9_]+'); do unset $v; done
fail=0
for c in ${NXAB_ARMS:-a b10 b30 b60 c16 c64 d64 e60}; do
    T=${SRC[$c]} F=$V/farm_$c lib=${SRC[$c]}/build_Release
    mkdir -p $F/generated
    for e in $T/* $T/.[!.]*; do b=$(basename $e); [ "$b" = generated ] || ln -s $e $F/$b; done
    (cd $F && timeout 600 env TT_METAL_HOME=$F TT_METAL_RUNTIME_ROOT=$F TT_METAL_CACHE=$V/cache_$c \
        TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
        LD_LIBRARY_PATH=$lib/tt_metal:$lib/tt_stl:$lib/lib:$lib/tt_metal/third_party/umd/lib ${ARM_ENV[$c]:-} \
        ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $V/nx1.yaml > $V/$c.log 2>&1)
    L=$(sed 's/\x1b\[[0-9;]*m//g' $V/$c.log)
    ok=$(echo "$L" | grep -ac "Launching programs")
    # before the launch only: the mock has no device, so the test then fails its senders (status UNKNOWN) by design
    fatal=$(echo "$L" | grep -avE "compile cmd|link cmd" | awk '/Launching programs/{exit} {print}' | grep -acE "TT_FATAL|TT_THROW|error:|Segmentation")
    args=$(echo "$L" | grep -a "compile cmd" | grep -a fabric_erisc_router | grep -aoE "KERNEL_COMPILE_TIME_ARGS=[0-9,]+" | sort -u | head -1 | cut -d, -f1-8)
    ovr=$(echo "$L" | grep -aoE "LOCAL NX A/B: .*slots" | head -1)
    spin=$(grep -rhoE "constexpr uint32_t (tx|rx)_spin_cycles = [0-9]+" $F/tt_metal/fabric/hw/inc/edm_fabric/slack_study_spin.hpp 2>/dev/null | tr '\n' ' ')
    idle=$(grep -oE "IDLE_PASSES_BEFORE_PARTIAL_CREDIT = [0-9]+" $F/tt_metal/fabric/hw/inc/edm_fabric/router/router_receiver.hpp)
    built=$(find $V/cache_$c -path "*fabric_erisc_router*" -name "*.elf" 2>/dev/null | wc -l)
    fp=$(find $V/cache_$c -path "*fabric_erisc_router*" -name "*.elf" -exec md5sum {} + 2>/dev/null | cut -d" " -f1 | sort | md5sum | cut -c1-10)
    r=ok; { [ "$ok" -ge 1 ] && [ "$fatal" = 0 ] && [ "$built" -gt 0 ]; } || { r=FAIL; fail=1; }
    echo "$c: $r (launching $ok, fatal $fatal, router ELFs $built, image set $fp) args ${args#KERNEL_COMPILE_TIME_ARGS=} | ${ovr:-no override} | ${spin:-no spin} | $idle"
    [ $r = ok ] || echo "$L" | grep -avE "compile cmd|link cmd" | grep -aE "TT_FATAL|TT_THROW|error" | head -3 | sed 's/^/   | /'
    rm -rf $V/cache_$c
done
exit $fail

#!/bin/bash
# usage: CARD=n phase.sh <tag> steps...
TAG=$1; shift
RUN=/localdev/cglagovich/whcheck/run.sh
SP="scripts/run_safe_pytest_card.sh --run-all"
T=/localdev/cglagovich/whcheck/tools
N=tests/ttnn/nightly/unit_tests/operations/sdpa
U=models/tt_dit/tests/unit
LK="flock /tmp/tt-device-card${CARD}.lock"
for s in "$@"; do
  case $s in
    matrix) $RUN $TAG ${TAG}_matrix $LK python $T/sdpa_matrix.py ;;
    unit) $RUN $TAG ${TAG}_unit $SP tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py tests/ttnn/unit_tests/operations/sdpa/test_windowed_sdpa.py ;;
    joint) $RUN $TAG ${TAG}_joint $SP $N/test_sdpa_joint.py -k "program_cache or (b1 and nh3)" ;;
    chunked) $RUN $TAG ${TAG}_chunked $SP $N/test_sdpa_chunked.py ;;
    nprefill) $RUN $TAG ${TAG}_nprefill $SP $N/test_sdpa_prefill.py -k "noncausal_mask_streaming or unequal_seqlen or large_score or partial_k or with_attention_sink or test_sdpa_tt_with_program_cache" ;;
    ring) $RUN $TAG ${TAG}_ring $SP $U/test_whcheck_ring_joint.py ;;
    expring) $RUN $TAG ${TAG}_expring $SP $U/test_whcheck_exp_ring.py ;;
    dit) $RUN $TAG ${TAG}_dit $SP $U/test_whcheck_dit_legacy.py -s ;;
    wmatrix) TT_METAL_WATCHER=1 $RUN ${TAG}_watch ${TAG}_wmatrix $LK python $T/sdpa_matrix.py 0 1 5 6 7 12 13 15 18 ;;
    wring) TT_METAL_WATCHER=1 $RUN ${TAG}_watch ${TAG}_wring $SP $U/test_whcheck_ring_joint.py -k "bf16acc and bf16 and not bf8b" ;;
  esac
done
echo PHASE_DONE >> /localdev/cglagovich/whcheck/logs/phase_${TAG}_c${CARD}.done
#!/bin/bash
# usage: CARD=n run.sh <tag> <logname> <cmd...>
TAG=$1; LOG=$2; shift 2
R=/localdev/cglagovich/whcheck/tt-metal
export TT_METAL_HOME=$R PYTHONPATH=$R:$R/ttnn:$R/tools LD_LIBRARY_PATH=$R/build/lib ARCH_NAME=wormhole_b0
export CARD=${CARD:?} TT_VISIBLE_DEVICES=$CARD
export TT_METAL_CACHE=/localdev/cglagovich/whcheck/cache_${TAG}/c$CARD WHCHECK_OUT=/localdev/cglagovich/whcheck/out_$TAG
export PATH=/opt/venv/bin:$PATH
mkdir -p /localdev/cglagovich/whcheck/logs $TT_METAL_CACHE
cd $R
echo "START $(date -u) $(git rev-parse --short HEAD) card=$CARD $*" > /localdev/cglagovich/whcheck/logs/$LOG.log
"$@" >> /localdev/cglagovich/whcheck/logs/$LOG.log 2>&1
echo "EXIT $? $(date -u)" >> /localdev/cglagovich/whcheck/logs/$LOG.log

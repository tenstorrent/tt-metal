#!/bin/bash
# twins.sh VARIANT_DIR TAG: debug twins (pcs_sections.py twins) of the four profiled logs from VARIANT_DIR into
# /home/aagarwal/.tt-metal-v3r-builds/design_c/tw_<TAG>_{tv,tso,hr,js1e} (design_a/twins.sh with design_c's output dir).
set -u
cd /home/aagarwal/tt-metal
D=/home/aagarwal/.tt-metal-v3r-builds/design_c
for t in tv:pcsv_rx1tv_compiletorusy tso:pcsv_rx1tso_compiletorusy hr:pcsv_rx1hr_compilering js1e:pcsv_rx1js1e_compiletorusy; do
    n=${t%%:*}; l=${t#*:}
    OFFLINE_BASE=/home/aagarwal/.tt-metal-p1-base-m nice -n 10 python3 fabric_exp/consolidation/pcs_sections.py twins \
        fabric_exp/results/${l}_13129088_trial.log "$1" $D/tw_$2_$n > $D/tw_$2_$n.log 2>&1 &
done
wait
for n in tv tso hr js1e; do echo "$n $(tail -1 $D/tw_$2_$n.log)"; done

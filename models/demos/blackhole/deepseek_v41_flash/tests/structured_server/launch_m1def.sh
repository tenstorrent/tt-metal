#!/bin/bash
# .33: boot with EXACTLY the proposed M1 defaults as env (no mode file, no timing/mask-live files); rc3 image + m2b model/plugin read-only.
cd /mnt/tt-data/ssinghal/dsv4-logs/structured_m2b
W=/mnt/tt-data/ssinghal/wt
export IMAGE=dsv41-flash-local:rc3-bf46d461048-213b63e9-25ca837
export EXTRA_DOCKER="--mount type=bind,src=$W/structured_m2b/models/demos/blackhole/deepseek_v41_flash,dst=/home/container_app_user/tt-metal/models/demos/blackhole/deepseek_v41_flash,readonly --mount type=bind,src=$W/structured_plugin_m2b/src/vllm_tt_plugin,dst=/home/container_app_user/vllm-tt-plugin/src/vllm_tt_plugin,readonly -e DSV41_STRUCTURED_SPEC=1 -e VLLM_TT_STRUCTURED_SPEC=1 -e DSV41_INTRACE_SAMPLE=1 -e VLLM_TT_STRUCTURED_M2=0 -e DSV41_STRUCTURED_M2=0 -e DSV41_VLLM_STATS_EVERY=16"
export LOAD=m1def RUNLOAD=run_load_m1def.sh HOLD_MAX=7000 BOOT_TIMEOUT=2700
echo "HOLD $(hostname -s) m1def (M1 default env boot: plain c8, GSM8K 250, structured suite, 4-worker sweep, tool matrix x2) start $(date) flock; boot ~25 min + ~95 min load, releases by itself" >> ANNOUNCE
exec flock -w 14400 /tmp/dsv4_dev.lock ./hold.sh m1def

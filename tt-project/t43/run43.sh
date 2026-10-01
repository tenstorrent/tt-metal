#!/bin/bash
# blx03 broker job for #43: Tracy device profile of one conv VAE decode on 2x4. JIT/profiler output under /var/tmp.
# hostfmax.py 1150 below is not a drop guard (ltx-host job 995 dropped tray 1 at 1150, see tt-project/research/blx03_drop_0722.md).
# Safety comes from: full mesh then create_submesh(2,4), one job at a time, no 4x8. At any device stop, kill our
# queued broker jobs on every box (broker re-queues after reboot); never other tenants' jobs; skip boxes mid-upgrade.
M=/home/smarton/fasth3/tt-metal; D=/home/smarton/fasth3/t43; V=/var/tmp/fasth3/t43; LOG=$D/run43.log
source $M/python_env/bin/activate
export TT_METAL_HOME=$M PYTHONPATH=$D:$M:$M/ttnn:$M/tools HF_HUB_OFFLINE=1
# Production decode runs the fused on-device YUV output (job 879 compiled yuv_compute); match it.
export LTX_FUSE_YUV_OUTPUT=1
export TT_METAL_CACHE=$V/jit TT_METAL_PROFILER_DIR=$V/prof TT_METAL_PROFILER_CPP_POST_PROCESS=1
mkdir -p $V/jit $V/prof
cd $M
echo "[t43] commit=$(git rev-parse --short HEAD) clock: $(python /home/smarton/tray-stress/hostfmax.py 1150 | tail -1)" | tee $LOG
timeout 1000 python -m tracy -p -r -v -o $V/prof -m pytest -p conftest -c $M/pytest.ini --rootdir=$M -sv --timeout=900 $D/test_prof_conv_vae_2x4.py 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
python /home/smarton/tray-stress/hostfmax.py 0 >/dev/null 2>&1
# tracy starts a WASM GUI server and copies the .tracy into the tree; neither is needed here.
pkill -u smarton -f serve_wasm; rm -f $M/build/profiler/build_wasm/traces/test_prof_conv_vae_2x4_*.tracy
echo "T43_EXIT=$rc" | tee -a $LOG
exit $rc

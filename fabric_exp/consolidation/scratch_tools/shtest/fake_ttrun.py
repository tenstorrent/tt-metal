# Fake "tt-run" for offline tests: an MPI-like tree (ranks in their own process groups, one ignores SIGTERM).
import os, signal, subprocess, sys, time
cnt = os.environ["FAKE_COUNTER"]
i = int(open(cnt).read()) + 1 if os.path.exists(cnt) else 1
open(cnt, "w").write(str(i))
hang_at = int(os.environ.get("FAKE_HANG_AT", "0"))
cache = os.environ["TT_METAL_CACHE"]
print(f"fake run {i}, TT_METAL_CACHE {cache}, TT_VISIBLE_DEVICES {os.environ.get('TT_VISIBLE_DEVICES')}", flush=True)
if not os.listdir(cache):  # "compile": the real cache as symlinks, compile commands into the log
    subprocess.run(f"cp -rs /home/aagarwal/.tt-metal-cache-p1vc1-v3l3-O3/. {cache}/", shell=True, check=True)
    subprocess.run("grep -a 'g++ compile cmd' /home/aagarwal/tt-metal/fabric_exp/results/p11avc1_o3_13129088_vc1.log"
                   " | grep -a fabric_erisc_router_v2 | head -300", shell=True)
if i != hang_at:
    for k in range(24):
        print(f"[1,0]<stdout>: info | Test | Test T{k} Finished. (test_tt_fabric.cpp:324)", flush=True)
    sys.exit(0)

def rank(ignore_term):
    pid = os.fork()
    if pid == 0:
        os.setpgid(0, 0)
        if ignore_term:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(1)
    return pid

mp = os.fork()
if mp == 0:  # "mpirun": two ranks, no signal forwarding
    rank(False)
    rank(True)
    for k in range(9):
        print(f"[1,0]<stdout>: info | Test | Running Test: MultiMeshAllToAll_size_4096_ntype_atomic_inc", flush=True)
        print(f"[1,0]<stdout>: info | Test | Test T{k} Finished. (test_tt_fabric.cpp:324)", flush=True)
    while True:
        time.sleep(1)
os.waitpid(mp, 0)

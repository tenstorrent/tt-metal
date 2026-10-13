# Writes the t460 job specs (blx03-runner format, for tt-project/t288/blx01-submit.sh) into specs/.
import os

F = "/var/tmp/fasth3"
T = f"{F}/t460"
NEEDS = ",".join(
    [
        f"{F}/models/MiniMax-H3/READ_OK",
        f"{F}/models/lightx2v-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
        f"{F}/t422/b/ttnn/ttnn/_ttnn.so",
        f"{F}/t408/kf_t2va_f0.png",
        f"{F}/t286/t286_skipvaewarm.py",
        f"{T}/python_env/bin/python",
        f"{T}/src/models/tt_dit/models/vae/minimax_h3/upscale_minimax_h3.py",
    ]
)
# Timing pairs at seed 0 (6 s with 2 replays, 10 s with 1), then the quality seeds 1-4 at 6 s.
jobs = [("b6s0", "base", 6, 0, 2), ("i6s0", "int", 6, 0, 2), ("b10s0", "base", 10, 0, 1), ("i10s0", "int", 10, 0, 1)]
for s in range(1, 5):
    jobs += [(f"b6s{s}", "base", 6, s, 1), (f"i6s{s}", "int", 6, s, 1)]
os.makedirs("specs", exist_ok=True)
for i, (tag, arm, dur, seed, rep) in enumerate(jobs):
    with open(f"specs/{i:02d}_{tag}.txt", "w") as f:
        f.write(
            f"""ID=t460-{tag}
TASK=t460
WORKDIR={T}
ENV={T}/env460.yaml
CONFIG=h3turbo-v10-fl2va-{dur}s-seed{seed}-{arm}
TIMEOUT=600
CMD=env T286_WARM_DEADLINE_S=500 T286_WARM_LATE_S=430 T408_REPLAYS={rep} T408_SAVE=1 T408_PY_TIMEOUT=560 bash {T}/run460.sh {tag} {arm} {dur} {seed}
NEEDS={NEEDS}
"""
        )

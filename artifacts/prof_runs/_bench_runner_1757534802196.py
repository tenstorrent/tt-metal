import subprocess, sys

subprocess.run(
    [
        "build/test/tt_metal/tt_fabric/bench_unicast",
        "--src-dev",
        "0:0",
        "--dst-dev",
        "0:1",
        "--size",
        "131072",
        "--page",
        "2048",
        "--send-core",
        "0,0",
        "--recv-core",
        "0,0",
        "--iters",
        "5",
        "--warmup",
        "1",
        "--format",
        "csv",
        "--csv",
        "artifacts/unicast_sweep.csv",
    ],
    check=True,
)

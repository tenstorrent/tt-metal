# SPDX-License-Identifier: Apache-2.0
"""Serial warmed one-replay profiler captures, separate from watcher."""

import json
import os
import subprocess
import sys
import time

from .multichip_sweep import OUT

if __name__ == "__main__":
    assert not os.environ.get("TT_METAL_WATCHER")
    for layer, tokens in ((0, 128), (4, 128), (0, 8193)):
        target = OUT / f"profile_final_{layer}_{tokens}"
        command = [
            sys.executable,
            "-m",
            "tracy",
            "-r",
            "-p",
            "--no-web-server",
            "-o",
            str(target),
            "-m",
            "models.autoports.aleph_alpha_kolibri_1_bf16.tests.multichip_checks",
            "--layer",
            str(layer),
            "--tokens",
            str(tokens),
            "--repetitions",
            "1",
            "--tag",
            "profiled_final",
        ]
        env = os.environ | {"TT_METAL_TRACE_ALLOC_TRACKING": "0", "MC_PROFILE_DRAIN": "1"}
        log = OUT / f"profile_final_{layer}_{tokens}.log"
        with log.open("w") as f:
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT)
        with (OUT / "profile_commands.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        command=command,
                        exit_code=result.returncode,
                        log=str(log),
                        time=time.time(),
                        environment={
                            "TT_METAL_TRACE_ALLOC_TRACKING": "0",
                            "TT_METAL_WATCHER": None,
                            "MC_PROFILE_DRAIN": "1",
                        },
                    )
                )
                + "\n"
            )
        if result.returncode:
            raise SystemExit(result.returncode)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "models.autoports.aleph_alpha_kolibri_1_bf16.tests.multichip_render_profile",
                str(target),
            ],
            check=True,
        )
        print(target, "COMPLETE", flush=True)

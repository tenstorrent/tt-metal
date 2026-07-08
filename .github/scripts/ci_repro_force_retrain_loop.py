# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""THROWAWAY diagnostic for the ci-repro/cluster-teardown-indexerror investigation.

Round 3 re-ran the real production test a few times per CI job and hoped one sample would land
on the ethernet-retrain race in MetalContext's implicit init (see metal_env.cpp's
verify_fw_capabilities()) — with a ~9h runner queue per job, that's an expensive way to get very
few, un-forced samples.

This script instead loops, in a single process, forcing the race on demand each iteration:

  1. open a mesh device (first iteration: normal implicit MetalContext creation; later
     iterations: reconstructs a fresh MetalContext/Cluster, exactly the crash site)
  2. force an ethernet retrain on every active+up ethernet core, with no settle wait
  3. close the mesh device
  4. tear down MetalContext (ForceMetalContextReinit) so the retrain from step 2 is still
     in flight when step 1 reconstructs it on the next loop iteration

No model/tensor work is involved — the crash is purely at device-open time, before any model
code runs, so dropping it keeps each iteration to a couple of seconds instead of the ~30-90s a
full model test costs.

Usage: python3 ci_repro_force_retrain_loop.py [RUNS]  (default 30)
"""

import os
import sys
import traceback

import ttnn


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    repro_count = 0
    fail_count = 0

    for i in range(1, runs + 1):
        print(f"::group::run {i}/{runs}", flush=True)
        try:
            mesh_device = ttnn.open_mesh_device()
            ttnn._ttnn.multi_device.force_ethernet_retrain(mesh_device)
            ttnn.close_mesh_device(mesh_device)
            ttnn._ttnn.device.ForceMetalContextReinit()
            print(f"run {i}: PASSED", flush=True)
        except Exception as e:  # noqa: BLE001 - deliberately broad, we want every failure mode
            fail_count += 1
            message = str(e)
            if "map::at" in message or isinstance(e, IndexError):
                repro_count += 1
                print(f"run {i}: FAILED - reproduced the IndexError: {message}", flush=True)
            else:
                print(f"run {i}: FAILED - different error: {message}", flush=True)
            traceback.print_exc()
            # Best-effort: MetalContext may be left partially constructed after a throw. Force a
            # clean slate before the next iteration rather than letting the failure cascade.
            try:
                ttnn._ttnn.device.ForceMetalContextReinit()
            except Exception:
                pass
        print("::endgroup::", flush=True)

    summary = f"{repro_count}/{runs} reproduced the IndexError, {fail_count}/{runs} failed total."
    print(summary, flush=True)

    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        with open(step_summary_path, "a", encoding="utf-8") as f:
            f.write("## Forced-retrain repro loop\n\n")
            f.write(summary + "\n")


if __name__ == "__main__":
    main()

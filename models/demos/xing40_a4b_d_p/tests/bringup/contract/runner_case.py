# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The runner test's runner process: tt-metal's prefill_runner.main, unchanged, with two hooks.

    python -m models.demos.xing40_a4b_d_p.tests.bringup.contract.runner_case <out_dir>

  - model_config: FABRIC_PAYLOAD_SIZE in the fabric's range, checked before the mesh opens
  - build_runtime: the runtime the runner gets is checked for every call the runner makes on it; a missing one writes
    <out>/not_built.txt and exits 3 instead of a TypeError deep in the serving setup
  - run_request_loop: starts the producer (producer_case.py) and runs the runner's own loop; after the shutdown
    sentinel it dumps every slot's final KV through the exported table over UMD into <out>/final (kv_dram_poke file
    names), for the parent test to compare with the server's kv_dump_compare
Environment: the PREFILL_* settings the parent test sets (FABRIC_2D, mock migration with the migration-enabled table
path, D2H layer acks)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def main(out: str) -> int:
    out = Path(out)
    from models.demos.common.prefill.runners import prefill_runner as PR
    from models.demos.common.prefill.runners.migration import migration_table_path
    from models.demos.xing40_a4b_d_p.tests.bringup.contract import engine as E
    from models.demos.xing40_a4b_d_p.tests.bringup.contract import server_rules as R

    result = {"errors": []}
    miss = E.check_model_config(PR.MODEL_CFG)
    if miss:
        (out / "not_built.txt").write_text("; ".join(miss))
        return 3
    build = PR.ADAPTER.build_runtime

    def checked_build(**kw):
        rt = build(**kw)
        miss = E.check_runtime(rt)
        if miss:
            (out / "not_built.txt").write_text("; ".join(miss))
            os._exit(3)
        return rt

    PR.ADAPTER.build_runtime = checked_build
    loop = PR.run_request_loop

    def checked_loop(runtime, kv_caches, *a, **k):
        log = (out / "producer.log").open("w")
        # Under sh so the producer's exit code lands in <out>/producer.rc even on a crash: the parent test watches it,
        # because this process cannot (the request loop spins in C++ holding the GIL; no thread or signal handler
        # runs until a chunk arrives).
        cmd = [
            "/bin/sh",
            "-c",
            '"$0" -m models.demos.xing40_a4b_d_p.tests.bringup.contract.producer_case "$1"; echo $? > "$1/producer.rc"',
            sys.executable,
            str(out),
        ]
        prod = subprocess.Popen(cmd, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT)
        try:
            loop(runtime, kv_caches, *a, **k)
            rc = prod.wait(timeout=120)
            if rc != 0:
                result["errors"].append(f"producer exited {rc} (see producer.log / producer.json)")
            pj = json.loads((out / "producer.json").read_text())
            dmap = {
                tuple(int(x) for x in key.split(":")): int(v)
                for key, v in json.loads(Path(os.environ["PREFILL_MIGRATION_DEVICE_MAP_PATH"]).read_text()).items()
            }
            reader = E.TableReader(migration_table_path(), dmap)
            for s, e in pj["ends"].items():
                reader.dump(out / "final", int(s), range(R.NUM_LAYERS), 0, int(e))
            result["replica_mismatch"] = reader.replica_mismatch[:16]
            result["ends"] = pj["ends"]
        except BaseException as err:  # a rejected chunk lands here: report it, let the runner shut down
            result["errors"].append(f"{type(err).__name__}: {err}")
            result["traceback"] = traceback.format_exc()
        finally:
            if prod.poll() is None:
                prod.terminate()
                prod.wait(timeout=30)
            log.close()
            (out / "result.json").write_text(json.dumps(result, indent=1))

    PR.run_request_loop = checked_loop
    try:
        PR.main()
    except BaseException as err:
        result["errors"].append(f"runner: {type(err).__name__}: {err}")
        result["traceback"] = traceback.format_exc()
        (out / "result.json").write_text(json.dumps(result, indent=1))
        return 1
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The runner test's runner process: tt-metal's prefill_runner.main, unchanged, with two hooks.

    python -m models.demos.xing40_a4b_d_p.tests.bringup.contract.runner_case <out_dir>

  - model_config: FABRIC_PAYLOAD_SIZE in the fabric's range, checked before the mesh opens
  - build_runtime: the runtime the runner gets is checked for every call the runner makes on it; a missing one writes
    <out>/not_built.txt and exits 3 instead of a TypeError deep in the serving setup
  - run_request_loop: starts the feeder and runs the runner's own loop; after the shutdown sentinel it dumps every
    slot's final KV through the exported table over UMD into <out>/final (kv_dram_poke file names), for the parent
    test to compare with the server's kv_dump_compare. The feeder is
      engine    (XING_CONTRACT_FEEDER_CMD set) tt-d-gen's engine: the sh command the parent built with
                testing/dgen_engine.driver_shell (dgen_prefill_driver.py under tt-d-gen's Python, then the shutdown
                sentinel); its log is <out>/engine.log, its exit code <out>/feeder.rc
      producer  (fallback) producer_case.py, tt-metal's prefill_producer; <out>/producer.log, <out>/producer.rc
  - runtime.prefill_chunk (engine mode): records every chunk the runner is handed (slot, start, end) in result.json
    "chunks" (ends = each slot's last chunk end), and before a chunk named in XING_CONTRACT_SNAPSHOTS
    ({name: {"before": [start, end], "lo", "hi"}}) syncs the mesh and dumps that chunk's slot [lo, hi) through the
    table into <out>/<name>: the KV the earlier chunks left, before this one rewrites any of it
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

    feeder_cmd = (
        json.loads(os.environ["XING_CONTRACT_FEEDER_CMD"]) if os.environ.get("XING_CONTRACT_FEEDER_CMD") else None
    )
    snaps = json.loads(os.environ.get("XING_CONTRACT_SNAPSHOTS", "{}")) if feeder_cmd else {}
    result["chunks"], result["snapshots"] = [], {}
    st = {"reader": None}

    def table_reader():
        if st["reader"] is None:
            dmap = {
                tuple(int(x) for x in key.split(":")): int(v)
                for key, v in json.loads(Path(os.environ["PREFILL_MIGRATION_DEVICE_MAP_PATH"]).read_text()).items()
            }
            st["reader"] = E.TableReader(migration_table_path(), dmap)
        return st["reader"]

    def checked_build(**kw):
        rt = build(**kw)
        miss = E.check_runtime(rt)
        if miss:
            (out / "not_built.txt").write_text("; ".join(miss))
            os._exit(3)
        if feeder_cmd:
            import functools

            import ttnn

            chunk = rt.prefill_chunk

            @functools.wraps(chunk)
            def recorded_chunk(inp, kv, *a, slot_id, actual_start, actual_end, **k):
                result["chunks"].append([int(slot_id), int(actual_start), int(actual_end)])
                for name, sn in snaps.items():
                    if name not in result["snapshots"] and list(sn["before"]) == [actual_start, actual_end]:
                        ttnn.synchronize_device(rt.mesh_device)
                        table_reader().dump(out / name, int(slot_id), range(R.NUM_LAYERS), sn["lo"], sn["hi"])
                        result["snapshots"][name] = {
                            "taken": True,
                            "slot": int(slot_id),
                            "lo": sn["lo"],
                            "hi": sn["hi"],
                            "before_chunk": len(result["chunks"]) - 1,
                        }
                return chunk(inp, kv, *a, slot_id=slot_id, actual_start=actual_start, actual_end=actual_end, **k)

            rt.prefill_chunk = recorded_chunk
        return rt

    PR.ADAPTER.build_runtime = checked_build
    loop = PR.run_request_loop

    def checked_loop(runtime, kv_caches, *a, **k):
        log = (out / ("engine.log" if feeder_cmd else "producer.log")).open("w")
        # Under sh so the feeder's exit code lands in <out>/feeder.rc (<out>/producer.rc) even on a crash: the parent
        # test watches it, because this process cannot (the request loop spins in C++ holding the GIL; no thread or
        # signal handler runs until a chunk arrives).
        if feeder_cmd:
            cmd, what = feeder_cmd, "engine driver"
        else:
            cmd, what = [
                "/bin/sh",
                "-c",
                '"$0" -m models.demos.xing40_a4b_d_p.tests.bringup.contract.producer_case "$1"; echo $? > "$1/producer.rc"',
                sys.executable,
                str(out),
            ], "producer"
        # In this process group, so the parent's group kill (run_child) takes the feeder too.
        prod = subprocess.Popen(cmd, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT)
        try:
            loop(runtime, kv_caches, *a, **k)
            rc = prod.wait(timeout=120)
            if rc != 0 or (feeder_cmd and (out / "feeder.rc").read_text().strip() != "0"):
                result["errors"].append(f"{what} exited {rc} (see {log.name})")
            if feeder_cmd:
                ends = {}
                for s, _, e in result["chunks"]:
                    ends[str(s)] = e
            else:
                pj = json.loads((out / "producer.json").read_text())
                ends = pj["ends"]
            reader = table_reader()
            for s, e in ends.items():
                reader.dump(out / "final", int(s), range(R.NUM_LAYERS), 0, int(e))
            result["replica_mismatch"] = reader.replica_mismatch[:16]
            result["ends"] = ends
        except BaseException as err:  # a rejected chunk lands here: report it, let the runner shut down
            result["errors"].append(f"{type(err).__name__}: {err}")
            result["traceback"] = traceback.format_exc()
        finally:
            if prod.poll() is None:
                subprocess.run(["pkill", "-TERM", "-P", str(prod.pid)])  # sh's children (the driver / producer)
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

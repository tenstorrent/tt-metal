# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The runner tests' runner process: tt-metal's prefill_runner.main, unchanged, with checks around its calls.

    python -m models.demos.deepseek_v41_flash_d_p.tests.bringup.contract.runner_case <out_dir>

  - import: prefill_runner resolves PREFILL_MODEL at import (adapter.py get_adapter); an unregistered adapter writes
    <out>/not_built.txt and exits 3
  - model_config: FABRIC_PAYLOAD_SIZE in the fabric's range (K-79), checked before the mesh opens
  - build_runtime / allocate_kv_cache: every call the runner makes on the runtime exists and the KvCaches handle names
    its served tensors (caches.check_runtime / check_kv); else not_built.txt, exit 3
  - run_request_loop: starts the feeder and runs the runner's own loop; after the shutdown sentinel it dumps every
    slot's final KV of every served config through the exported table over UMD into <out>/final/<config> (kv_dram_poke
    file names). The feeder is
      engine    (DS41_CONTRACT_FEEDER_CMD set) tt-d-gen's engine: the sh command the parent built with
                testing/dgen_engine.driver_shell; its log <out>/engine.log, its exit code <out>/feeder.rc
      producer  (fallback) producer_case.py, tt-metal's prefill_producer sending the server's plan; <out>/producer.log,
                <out>/producer.rc
  - runtime.prefill_chunk: records every chunk the runner is handed (slot, start, end) in result.json "chunks", and in
    engine mode, before a chunk named in DS41_CONTRACT_SNAPSHOTS ({name: {"before": [start, end], "lo", "hi"}}), syncs
    the mesh and dumps that chunk's slot [lo, hi) through the table into <out>/<name>/<config>
Environment: the PREFILL_* settings the parent test sets (2D fabric, mock migration with the migration-enabled table
path, D2H layer acks)."""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

MIN_FABRIC_PAYLOAD, MAX_FABRIC_PAYLOAD = 4352, 15232  # K-79; tt_metal/fabric/erisc_datamover_builder.hpp (Blackhole)


def main(out: str) -> int:
    out = Path(out)
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C

    try:
        from models.demos.common.prefill.runners import prefill_runner as PR
    except KeyError as e:  # get_adapter: unknown PREFILL_MODEL
        (out / "not_built.txt").write_text(f"adapter not registered in models/demos/common/prefill/adapter.py: {e}")
        return 3
    except Exception as e:
        (out / "not_built.txt").write_text(f"prefill_runner import failed: {type(e).__name__}: {e}")
        return 3
    from models.demos.common.prefill.runners.migration import migration_table_path

    result = {"errors": [], "chunks": [], "snapshots": {}}
    n = int(getattr(PR.MODEL_CFG, "FABRIC_PAYLOAD_SIZE", 0))
    if not MIN_FABRIC_PAYLOAD <= n <= MAX_FABRIC_PAYLOAD:
        (out / "not_built.txt").write_text(
            f"model_config.FABRIC_PAYLOAD_SIZE = {n}: the runner's fabric packet payload must be in "
            f"[{MIN_FABRIC_PAYLOAD}, {MAX_FABRIC_PAYLOAD}] B (runner_utils.open_mesh_device)"
        )
        return 3
    layers = list(range(PR.NUM_LAYERS))
    feeder_cmd = (
        json.loads(os.environ["DS41_CONTRACT_FEEDER_CMD"]) if os.environ.get("DS41_CONTRACT_FEEDER_CMD") else None
    )
    snaps = json.loads(os.environ.get("DS41_CONTRACT_SNAPSHOTS", "{}")) if feeder_cmd else {}
    st = {"reader": None}

    def table_reader():
        if st["reader"] is None:
            st["reader"] = C.TableReader(
                migration_table_path(), C.read_device_map(os.environ["PREFILL_MIGRATION_DEVICE_MAP_PATH"])
            )
        return st["reader"]

    build, alloc = PR.ADAPTER.build_runtime, PR.ADAPTER.allocate_kv_cache

    def checked_build(**kw):
        rt = build(**kw)
        miss = C.check_runtime(rt)
        if miss:
            (out / "not_built.txt").write_text("; ".join(miss))
            os._exit(3)
        chunk = rt.prefill_chunk

        @functools.wraps(chunk)
        def recorded_chunk(inp, kv, *a, slot_id, actual_start, actual_end, **k):
            result["chunks"].append([int(slot_id), int(actual_start), int(actual_end)])
            for name, sn in snaps.items():
                if name not in result["snapshots"] and list(sn["before"]) == [actual_start, actual_end]:
                    import ttnn

                    ttnn.synchronize_device(rt.mesh_device)
                    table_reader().dump(out / name, int(slot_id), layers, sn["lo"], sn["hi"])
                    result["snapshots"][name] = {"taken": True, "slot": int(slot_id), "lo": sn["lo"], "hi": sn["hi"]}
            return chunk(inp, kv, *a, slot_id=slot_id, actual_start=actual_start, actual_end=actual_end, **k)

        rt.prefill_chunk = recorded_chunk
        return rt

    def checked_alloc(**kw):
        kv = alloc(**kw)
        miss = C.check_kv(kv, layers)
        if miss:
            (out / "not_built.txt").write_text("; ".join(miss))
            os._exit(3)
        geo = {k: {"dtype": str(t.dtype), "layout": str(t.layout)} for k, t in kv.by_config.items()}
        (out / "geometry.json").write_text(json.dumps(geo, indent=1))
        return kv

    PR.ADAPTER.build_runtime = checked_build
    PR.ADAPTER.allocate_kv_cache = checked_alloc
    loop = PR.run_request_loop

    def checked_loop(runtime, kv_caches, *a, **k):
        log = (out / ("engine.log" if feeder_cmd else "producer.log")).open("w")
        # Under sh so the feeder's exit code lands in its rc file even on a crash: the parent test watches it (this
        # process cannot: the request loop blocks in C++ holding the GIL until a chunk arrives).
        if feeder_cmd:
            cmd, what = feeder_cmd, "engine driver"
        else:
            cmd, what = [
                "/bin/sh",
                "-c",
                '"$0" -m models.demos.deepseek_v41_flash_d_p.tests.bringup.contract.producer_case "$1"; '
                'echo $? > "$1/producer.rc"',
                sys.executable,
                str(out),
            ], "producer"
        prod = subprocess.Popen(cmd, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT)
        try:
            loop(runtime, kv_caches, *a, **k)
            rc = prod.wait(timeout=120)
            frc = (out / "feeder.rc").read_text().strip() if feeder_cmd and (out / "feeder.rc").exists() else "0"
            if rc != 0 or frc != "0":
                result["errors"].append(f"{what} exited {rc} / {frc} (see {log.name})")
            if feeder_cmd:
                ends = {}
                for s, _, e in result["chunks"]:
                    ends[str(s)] = e
            else:
                ends = json.loads((out / "producer.json").read_text())["ends"]
            reader = table_reader()
            for s, e in ends.items():
                reader.dump(out / "final", int(s), layers, 0, int(e))
            result["replica_mismatch"] = reader.replica_mismatch[:16]
            result["ends"] = ends
        except BaseException as err:  # a rejected chunk lands here: report it, let the runner shut down
            result["errors"].append(f"{type(err).__name__}: {err}")
            result["traceback"] = traceback.format_exc()
        finally:
            if prod.poll() is None:
                subprocess.run(["pkill", "-TERM", "-P", str(prod.pid)])
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

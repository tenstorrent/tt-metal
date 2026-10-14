# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The runner tests' producer process (fallback feeder): tt-metal's prefill_producer.main, made to send what tt-d-gen
sends.

    python -m models.demos.deepseek_v41_flash_d_p.tests.bringup.contract.producer_case <out_dir>

prefill_producer differs from the server (K-78); each difference is replaced here, the rest (H2D connect, push, the
layer-ack channel, the shutdown sentinel) is the producer's own code:
  - schedule: the server's chunk plan and round-robin slot interleave (server_rules.interleave) instead of run_schedule
    (whose follow-ups start at a 32-aligned end and whose last chunk is never pulled back)
  - payload: PAD_ID past actual_end and ring_sdpa_reshuffle by actual_start (server_rules.server_payload) instead of
    the producer's real-token padding and rotation
  - acks: counted here, PREFILL_NUM_LAYERS per chunk (layers_per_chunk), and waited for before each snapshot
  - verify: off (the parent test reads the dumps with the server's own comparer)

Turns: server_rules.RUNNER_TURNS (slot 0 1000 then 4000 over 960, pulled back; slot 1 1500 then 3500 over 1472), or
DS41_CONTRACT_TURNS (JSON {"slot": [len, ...]}, the runner smoke). Snapshots, read through the exported table over
UMD right after the named chunk's acks (what the KV Manager ships then):
  snap_s0_t0     slot 0 [0, its first turn) every served layer
  snap_overlap   slot 0 [max_seq - chunk, end of the chunk before the pulled-back one), before the pulled-back chunk
"""

from __future__ import annotations

import json
import os
import signal
import struct
import sys
import time
from pathlib import Path

import numpy as np

from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as _R

TURNS = (
    {int(k): [int(n) for n in v] for k, v in json.loads(os.environ["DS41_CONTRACT_TURNS"]).items()}
    if os.environ.get("DS41_CONTRACT_TURNS")
    else _R.RUNNER_TURNS
)


def snapshots() -> dict:
    t0 = TURNS.get(0, [])
    if len(t0) < 2:
        return {}
    plan = _R.follow_up(t0[0], t0[1])[1]
    out = {"snap_s0_t0": {"after": (0, 0, t0[0]), "slot": 0, "lo": 0, "hi": t0[0]}}
    if len(plan) >= 2 and plan[-1][0] < plan[-2][1]:
        out["snap_overlap"] = {"after": (0, *plan[-2]), "slot": 0, "lo": plan[-1][0], "hi": plan[-2][1]}
    return out


SNAPSHOTS = snapshots()
ACK_TIMEOUT_S = float(os.environ.get("DS41_CONTRACT_ACK_TIMEOUT_S", "300"))


def main(out: str) -> int:
    out = Path(out)
    import ttnn
    from models.demos.common.prefill.runners import prefill_producer as P
    from models.demos.common.prefill.runners.migration import migration_table_path
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import caches as C
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.contract import server_rules as R

    n_layers = int(os.environ["PREFILL_NUM_LAYERS"])
    layers = list(range(n_layers))
    plan = R.interleave(TURNS)
    rec = {"plan": [vars(p) for p in plan], "acks_drained": 0, "errors": [], "snapshots": {}}
    st = {"ch": None, "drained": 0, "dead": False}

    class Service:
        """H2D service whose chunk payload is the server's (PAD_ID tail, ring_sdpa_reshuffle)."""

        def __init__(self, real):
            self.real = real

        def __getattr__(self, n):
            return getattr(self.real, n)

        def forward_to_tensor_bytes(self, chunk, metadata=None):
            slot, start, end = struct.unpack("<III", metadata)
            if (slot, start, end) != (0xFFFFFFFF,) * 3:  # the shutdown sentinel passes as is
                chunk = np.ascontiguousarray(R.server_payload(np.asarray(chunk).reshape(-1), start, end))
            return self.real.forward_to_tensor_bytes(chunk, metadata=metadata)

    class TT:
        def __getattr__(self, n):
            return getattr(ttnn, n)

        class H2DStreamService:
            @staticmethod
            def connect(*a, **k):
                return Service(ttnn.H2DStreamService.connect(*a, **k))

    P.ttnn = TT()

    def natural_rows(tokens, actual_start=0):  # the producer's rotation is replaced by the server's in Service
        assert len(tokens) == R.CHUNK, f"expected {R.CHUNK} tokens, got {len(tokens)}"
        return np.asarray(tokens, dtype=np.int64).astype(np.uint32).reshape(R.SP, 1, R.W)

    P._h2d_rows = natural_rows
    connect = P._connect_layer_ack_channel

    def connect_acks(timeout_s):
        st["ch"] = connect(timeout_s)
        return st["ch"]

    P._connect_layer_ack_channel = connect_acks

    def wait_acks(n: int) -> bool:
        if st["dead"]:
            return False
        t0 = time.perf_counter()
        while st["drained"] < n:
            if st["ch"] is None:
                rec["errors"].append("no layer-ack channel")
                return False
            st["drained"] += st["ch"].try_consume_all()
            if time.perf_counter() - t0 > ACK_TIMEOUT_S:
                rec["errors"].append(
                    f"layer acks: {st['drained']} < {n} after {ACK_TIMEOUT_S:.0f} s ({n_layers} per chunk)"
                )
                st["dead"] = True
                return False
            time.sleep(0.005)
        return True

    def drain(ack_channel, expected, timeout_s=600.0):  # the server's count (layers_per_chunk x chunks)
        wait_acks(n_layers * len(plan))
        rec["acks_drained"] = st["drained"]
        return st["drained"]

    P._drain_layer_acks = drain
    P._verify_resident_slots = lambda *a, **k: True

    def server_schedule(cfg, *, push_fn, now_fn=time.perf_counter, sleep_fn=time.sleep, rng=None):
        t0 = now_fn()
        reader = None
        push_ms, ends = [], {}
        for k, p in enumerate(plan):
            push_ms.append(push_fn(p.slot, p.chunk_idx, p.start, p.end, TURNS[p.slot][p.turn]))
            ends[p.slot] = p.end
            for name, sn in SNAPSHOTS.items():
                if tuple(sn["after"]) != (p.slot, p.start, p.end):
                    continue
                ok = wait_acks(n_layers * (k + 1))
                if reader is None:
                    reader = C.TableReader(migration_table_path(), P._read_device_map(60))
                if ok:
                    reader.dump(out / name, sn["slot"], layers, sn["lo"], sn["hi"])
                rec["snapshots"][name] = {"after_push": k, "taken": ok, **{x: sn[x] for x in ("slot", "lo", "hi")}}
        if reader is not None:
            rec["replica_mismatch"] = reader.replica_mismatch[:16]
        rec["ends"] = ends
        return P.RunStats(
            resident={s: P._SlotFill(real_len=e) for s, e in ends.items()},
            total_pushes=len(plan),
            push_ms=push_ms,
            completed=sum(len(t) for t in TURNS.values()),
            wall_s=now_fn() - t0,
        )

    P.run_schedule = server_schedule
    sys.argv = sys.argv[:1]  # prefill_producer.main parses argv (--manifest only)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # stopped by the runner: keep the record
    try:
        P.main()
    except SystemExit as e:
        if e.code == 143:
            rec["errors"].append("stopped by the runner (its request loop failed)")
        elif e.code not in (0, None):
            rec["errors"].append(f"prefill_producer exited {e.code}")
    except Exception as e:
        rec["errors"].append(f"prefill_producer: {type(e).__name__}: {e}")
    finally:
        rec["acks_expected"] = n_layers * len(plan)
        rec["acks_drained"] = st["drained"]
        rec["ends"] = rec.get("ends") or {}
        (out / "producer.json").write_text(json.dumps(rec, indent=1))
    return 1 if rec["errors"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

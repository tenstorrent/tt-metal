# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""K.2: the real prefill runner (mock migration, D2H acks, all 30 layers) fed by tt-d-gen's engine. Two engine runs
on one runner (each driver is a fresh engine: its prefix index starts empty), the cases of server_rules.scenario at
s4096 / chunk 2048, T = the golden's tokens:

  run 1  a = T[:1000] cold, then a2 = T[:3008] after it (resident 960, one chunk [960, 3008)); on the other slot,
         interleaved, c = T[1:3018] cold over two chunks, the last ending mid-record
  run 2  e = T[:1000] cold, then e2 = T[:4000] after it: chunks [960, 3008) and [2048, 4000), the last pulled back
         (prefill_writer.cpp:63-67) over rows the chunk before it wrote

Pass: every request reaches PREFILL_DONE at its length (the acks retire every chunk), the follow-ups start at 960,
the table has every record with the KV Manager's geometry and equal replicas, e2's KV matches the golden (PCC >=
max(serving.min_kv_pcc, state threshold)), and e2's slot over [0, 3008) equals a2's to the byte (tt-d-gen
kv_dump_compare.bytecmp): the reused prefix is untouched and the pulled-back rows are rewritten with the same bytes.
"""

import tempfile
from pathlib import Path

import pytest
from bringup_fw.core import metrics
from bringup_fw.testing import kv_table, server_rules
from bringup_fw.testing.contract import contract_rung, engine_env
from bringup_fw.testing.harness import device_timeout, impl_mode, spec

from models.demos.gemma4_a4b_glx.tests.bringup.contract import _runner as R

S = spec()
engine_env(S)
pytestmark = device_timeout(S)


def _dump(reader, table, out_dir, slot, layers, lo, hi) -> dict:
    """Records of [lo, hi) of every config into out_dir/c<config>; {config: record bytes}."""
    sizes = {}
    for c in range(table.num_configs()):
        reader.dump(Path(out_dir) / f"c{c}", slot, layers, lo, hi, c)
        sizes[c] = int(table.lookup(0, 0, 0, c).size_bytes)
    return sizes


# The runner opens the mesh itself: no mesh_device fixture here (it would hold the chips the runner needs).
@pytest.mark.parametrize("mesh_device", [None] if impl_mode() != "device" else ["runner"], ids=["runner"])
def test_runner_contract(mesh_device):
    if mesh_device is None:
        return
    kind, build = R.preflight(S)
    if kind != "tt-d-gen engine":
        pytest.skip(f"feeder is {kind}: this test drives tt-d-gen's engine (the producer path is K.1's dgen-verify)")
    g = R.golden(S)
    rung = S.rung(contract_rung(S))
    seq, chunk = rung["seq"], rung["chunk"]
    sc = server_rules.scenario(chunk, seq)
    cold_len = sc["cold_mid_end"][1]  # 3017
    res, follow_len = sc["pulled_back"]  # (960, 4000)
    first_len = res + 40
    assert server_rules.follow_up_resident(first_len, follow_len) == res
    plan_e2 = server_rules.chunk_plan(follow_len, res, chunk, seq)  # [(960, 3008), (2048, 4000)]
    assert server_rules.pulled_back(plan_e2), plan_e2
    mid_len = plan_e2[0][1]  # 3008: a2 is e2's first chunk alone
    assert (
        server_rules.chunk_plan(mid_len, server_rules.follow_up_resident(first_len, mid_len), chunk, seq) == plan_e2[:1]
    )
    tok = g.tokens().tolist()
    print(f"e2 chunks {plan_e2}; compare [0, {mid_len}) with a2; cold c {cold_len} tokens")

    failed = []
    with tempfile.TemporaryDirectory(prefix="g4glx_runner_") as td:
        td = Path(td)
        runner = R.Runner(S, td, trace_dir=str(g.dir))
        try:
            runner.wait_ready()
            layers = range(runner.n_layers)
            out = runner.drive(
                build,
                [
                    {"name": "a", "token_ids": tok[:first_len]},
                    {"name": "c", "token_ids": tok[1 : cold_len + 1]},
                    {"name": "a2", "token_ids": tok[:mid_len], "after": "a"},
                ],
                "run1",
            )["requests"]
            if out["a"]["slot"] == out["c"]["slot"]:
                failed.append(f"a and c share slot {out['a']['slot']} (two slots must interleave)")
            if int(out["a2"]["resident"]) != res:
                failed.append(f"a2 resident {out['a2']['resident']} != {res} (server rule follow_up_resident)")

            table, dmap = runner.table(), runner.dmap()
            n_slots = int(S.get("contract.num_users"))
            layout = kv_table.layout_failures(
                kv_table.records_from_table(table, layers, n_slots, seq), layers, n_slots, seq, table.num_configs()
            )
            metrics.record("runner_layout_failures", len(layout))
            failed += [f"layout: {f}" for f in layout]
            reader = kv_table.TableReader(table, dmap)
            sizes = _dump(reader, table, td / "a2", int(out["a2"]["slot"]), layers, 0, mid_len)

            out = runner.drive(
                build,
                [
                    {"name": "e", "token_ids": tok[:first_len]},
                    {"name": "e2", "token_ids": tok[:follow_len], "after": "e"},
                ],
                "run2",
            )["requests"]
            if int(out["e2"]["resident"]) != res:
                failed.append(f"e2 resident {out['e2']['resident']} != {res}")
            metrics.record("runner_reused_tokens", int(out["e2"].get("reused_tokens") or 0))
            s_e2 = int(out["e2"]["slot"])
            _dump(reader, table, td / "e2", s_e2, layers, 0, mid_len)

            kdc = R.dump_compare(S)
            for c, size in sizes.items():
                a = kdc.load_dump(td / "a2" / f"c{c}", chunk_bytes=size)
                b = kdc.load_dump(td / "e2" / f"c{c}", chunk_bytes=size)
                if not kdc.bytecmp(
                    a,
                    b,
                    layers=list(layers),
                    start=0,
                    end=mid_len,
                    dtype="bfp8",
                    width=size // R.BFP8_TILE_BYTES * 32,
                    storage="tile",
                ):
                    failed.append(
                        f"config {c}: e2's [0, {mid_len}) != a2's (reused prefix or pulled-back rows "
                        f"[{plan_e2[1][0]}, {mid_len}) rewritten with other bytes)"
                    )

            thr, worst = R.kv_threshold(S), 1.0
            got = S.hooks().contract_read_slot_kv(S, table, dmap, s_e2, follow_len)
            for layer in S.layers():
                want = g.state(layer)
                for k in ("key", "value"):
                    p = metrics.pcc(R.rows(got[layer][k], 0, follow_len), R.rows(want[k], 0, follow_len))
                    worst = min(worst, p)
                    if p < thr:
                        failed.append(f"e2 layer {layer} {k}: PCC {p:.4f} < {thr}")
            metrics.record("runner_kv_pcc_min", worst)
            if reader.replica_mismatch:
                failed.append(
                    f"{len(reader.replica_mismatch)} records differ between replicas, e.g. "
                    f"{reader.replica_mismatch[:3]}"
                )
        finally:
            runner.stop()
    metrics.record("runner_contract_failed", len(failed))
    assert not failed, "\n".join(failed)

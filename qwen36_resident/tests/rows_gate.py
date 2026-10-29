# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user rows mode gate (WP6), random weights on a 1 x 4 mesh.

One rows-mode model (ResidentModel rows=True, paged KV with a shuffled block table per slot, 2 launches of 8 rows)
runs a probe user P through the same inputs in several placements:
  alone      P in slot 0, nobody else;
  neighbours P in slot 0 beside 7 users;
  row        P in slot 5 beside 7 users;
  launch     P in slot 11 (launch 1) beside 7 users there and 8 in launch 0;
  churn      P in slot 2; users join slot 6 (state written, then a masked first-step launch with only them active and
             preserve set) and leave it, a neighbour pauses, and P itself pauses for some steps (preserve set).
P's x after each of its steps must be bit-identical in every placement. The state of every slot that has live state
but sits out a launch (P while paused or during a neighbour's join, the paused neighbour) must be bit-unchanged by
that launch: GDN states, its row of the conv histories, its KV pages. The alone run is also checked against
torch_step (relative error / PCC, as e0_bringup random).

usage: python -m qwen36_resident.tests.rows_gate [--layers 4] [--steps 16] [--max-pos 1024] [--out rows_gate.json]
"""

import argparse
import copy
import faulthandler
import json
import sys
import time

import torch

import ttnn
from qwen36_resident.model import HIDDEN, Dims, ResidentModel, State, is_attn, random_weights, torch_step
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh, pcc, rel, shuffled_tables

SLOTS = 16


class Gate:
    def __init__(self, mesh, layers, max_pos, interval=4):
        self.mesh = mesh
        self.d = d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x)
        self.layers, self.interval, self.max_pos = layers, interval, max_pos
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        self.n_gdn, self.n_attn = layers - n_attn, n_attn
        self.w = random_weights(d, self.n_gdn, n_attn, layers)
        blocks, self.tables = shuffled_tables(SLOTS, max_pos, seed=5)
        empty = [State(d, self.n_gdn, n_attn, 0, max_pos, zero=True) for _ in range(SLOTS)]
        t0 = time.time()
        self.model = ResidentModel(
            mesh, d, self.w, empty, layers, interval, kv_pool_blocks=blocks, block_tables=self.tables, rows=True
        )
        print(
            f"rows model: {SLOTS} slots in {self.model.n_sub} launches, G={self.model.head_groups}, "
            f"pool {blocks} blocks, built in {time.time() - t0:.1f} s",
            flush=True,
        )
        self.model.capture_trace()
        self.users = {}

    def user(self, name, start, seed, steps):
        """a user's initial State and its x0 per own step"""
        g = torch.Generator().manual_seed(1000 + seed)
        st = State(self.d, self.n_gdn, self.n_attn, start, self.max_pos, seed=seed)
        self.users[name] = dict(st=st, x0=[torch.randn(HIDDEN, generator=g) for _ in range(steps)])

    def run(self, plan, check_frozen=True):
        """plan: per launch step a dict slot -> user name that runs it (each user's own step count advances),
        plus optional 'join': {slot: user} written before that step and 'preserve': True. Returns per user its x
        after each of its steps, and the frozen-state violations (a live slot not run, whose state changed)."""
        m = self.model
        live, progress, xs, frozen_bad = {}, {}, {}, []
        for k, spec in enumerate(plan):
            for slot, name in spec.get("join", {}).items():
                m.write_slot(slot, self.users[name]["st"])
                live[slot] = name
                progress.setdefault(name, 0)
            runs = spec["run"]
            for slot, name in runs.items():
                live[slot] = name
            x0 = torch.zeros(SLOTS, HIDDEN)
            pos = [0] * SLOTS
            active = [False] * SLOTS
            for slot, name in runs.items():
                u = self.users[name]
                x0[slot] = u["x0"][progress[name]]
                pos[slot] = u["st"].pos + progress[name]
                active[slot] = True
            # every live slot that sits this step out must come through unchanged
            idle = [s for s in live if s not in runs] if check_frozen else []
            before = {s: m.read_slot(s) for s in idle}
            launches = sorted({s // m.batch for s in runs})
            toks = m.token_state(x0, pos, active=active, preserve=spec.get("preserve", False))
            m.step(toks, launches=launches)
            ttnn.synchronize_device(self.mesh)
            x = m.x()
            for slot, name in runs.items():
                xs.setdefault(name, []).append(x[slot].clone())
                progress[name] += 1
            for s in idle:
                after = m.read_slot(s)
                for key in ("state", "hist", "K", "V"):
                    if not torch.equal(before[s][key], after[key]):
                        frozen_bad.append(dict(step=k, slot=s, user=live[s], part=key))
        return xs, frozen_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--max-pos", type=int, default=1024)
    ap.add_argument("--out")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    S = a.steps
    mesh = open_mesh()
    out = dict(scenarios={}, frozen={})
    try:
        gate = Gate(mesh, a.layers, a.max_pos)
        # P near a tile boundary so its steps cross position tiles; neighbours spread over several blocks
        gate.user("P", 120, 7, S)
        for i in range(16):
            gate.user(f"N{i}", 33 + 41 * i, 20 + i, 3 * S)
        for i in range(3):
            gate.user(f"J{i}", 200 + 77 * i, 60 + i, S)

        def with_neighbours(p_slot, slots, k):
            return {**{s: f"N{(s + k) % 16}" for s in slots if s != p_slot}, p_slot: "P"}

        plans = {
            "alone": [dict(join={0: "P"}, run={0: "P"})] + [dict(run={0: "P"}) for _ in range(S - 1)],
            "neighbours": [dict(join=with_neighbours(0, range(8), 0), run=with_neighbours(0, range(8), 0))]
            + [dict(run=with_neighbours(0, range(8), 0)) for _ in range(S - 1)],
            "row": [dict(join=with_neighbours(5, range(8), 3), run=with_neighbours(5, range(8), 3))]
            + [dict(run=with_neighbours(5, range(8), 3)) for _ in range(S - 1)],
            "launch": [dict(join=with_neighbours(11, range(16), 5), run=with_neighbours(11, range(16), 5))]
            + [dict(run=with_neighbours(11, range(16), 5)) for _ in range(S - 1)],
        }
        # churn: P in slot 2, neighbour N1 in slot 4; J0 joins slot 6 (masked first step), runs, leaves; P pauses
        # (its neighbours run with preserve); J1 joins slot 6, N1 pauses meanwhile; J2 joins launch 1's slot 9
        churn = [dict(join={2: "P", 4: "N1"}, run={2: "P", 4: "N1"})]
        while sum("P" in st["run"].values() for st in churn) < S:
            k = len(churn)
            if k == 2:
                churn.append(dict(join={6: "J0"}, run={6: "J0"}, preserve=True))
            elif k in (5, 6):
                churn.append(dict(run={4: "N1", 6: "J0"}, preserve=True))  # P paused
            elif k == 8:
                churn.append(dict(join={6: "J1"}, run={6: "J1"}, preserve=True))  # J0 left
            elif k in (9, 10):
                churn.append(dict(run={2: "P", 6: "J1"}, preserve=True))  # N1 paused
            elif k == 11:
                churn.append(dict(join={9: "J2"}, run={9: "J2"}, preserve=True))
            else:
                run = {2: "P", 4: "N1"}
                run.update({6: "J0"} if 2 < k < 8 else {6: "J1"} if k > 8 else {})
                run.update({9: "J2"} if k > 11 else {})
                churn.append(dict(run=run))
        plans["churn"] = churn

        results = {}
        for name, plan in plans.items():
            t0 = time.time()
            xs, bad = gate.run(plan)
            results[name] = xs["P"]
            out["frozen"][name] = bad
            print(
                f"{name}: {len(plan)} launch steps, P ran {len(xs['P'])}, frozen-state violations {len(bad)} "
                f"({time.time() - t0:.1f} s)",
                flush=True,
            )
            for b in bad[:5]:
                print(f"    frozen violation {b}", flush=True)

        ref = results["alone"]
        st = copy.deepcopy(gate.users["P"]["st"])
        worst = (0.0, 1.0)
        for t in range(S):
            want = torch_step(gate.users["P"]["x0"][t], gate.d, gate.w, st, a.layers, gate.interval)
            worst = (max(worst[0], rel(ref[t], want)), min(worst[1], pcc(ref[t], want)))
        print(f"alone vs torch_step: worst rel {worst[0]:.4f} pcc {worst[1]:.5f}", flush=True)
        out["alone_vs_torch"] = worst
        for name, xs in results.items():
            same = sum(torch.equal(x, r) for x, r in zip(xs, ref))
            diff = max(float((x - r).abs().max()) for x, r in zip(xs, ref))
            out["scenarios"][name] = dict(identical=same, of=len(ref), max_abs_diff=diff)
            print(f"P {name} vs alone: {same}/{len(ref)} steps bit-identical (max |diff| {diff})", flush=True)
        # clear_slot: the slot's GDN states and history row zero, every other slot's state as it was
        m = gate.model
        before = {s: m.read_slot(s) for s in (2, 4, 6, 8)}
        m.clear_slot(9)
        cleared = m.read_slot(9)
        zeroed = not cleared["state"].any() and not cleared["hist"].any()
        kept = all(
            torch.equal(before[s][key], m.read_slot(s)[key]) for s in before for key in ("state", "hist", "K", "V")
        )
        print(f"clear_slot(9): state and history zero {zeroed}; other slots unchanged {kept}", flush=True)
        out["clear_slot"] = dict(zeroed=zeroed, others_kept=kept)
        ok = all(v["identical"] == v["of"] for v in out["scenarios"].values()) and not any(out["frozen"].values())
        ok = ok and zeroed and kept
        print(f"rows gate: {'PASS' if ok else 'FAIL'}", flush=True)
        out["pass"] = ok
    finally:
        close_mesh(mesh)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

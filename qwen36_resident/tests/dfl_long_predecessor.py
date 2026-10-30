# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A served DFlash2 prompt after a long-decoded one equals the verify model's own greedy decode (lane ttft2,
2026-10-30). The OSL 1024 sweep of ttft2_session_20261030-1248 and stale_state_repro gave such prompts (ISL >= 2048)
the first token 0 ("!") from all-zero first-step logits, in requests that then accepted no draft at all. Not stale decode
state (stale_state_repro's full variant, the whole initial state restored before the prompt, failed alike) but a race:
ResidentDFlash._prompt enqueued the prompt's verify step right after the drafter's context fill, a resident step's hubs
push their first round into the peers' hub slots (CB 0, mlp_hub.cpp) as soon as they start, and a chip still running the
fill's last ops (their CBs at those L1 addresses) lost its peers' slots and corrupted its own fill. It shows when one chip
lags, e.g. chip 1's AICLK throttle after ~80 s of sustained decode: after long-decoded requests.

The serving class (generator_vllm.Qwen36ResidentDFlashForCausalLM) driven as the plugin drives it (prefill_forward,
decode_forward blocks, release_request). Per case isl:prev:cur (the prompts[isl, 1024] of --prompts):
  ref:   cur through the verify model alone (ResidentDFlash.greedy: no PP prefill, handoff, fill or drafter)
  skew:  prev served for --long tokens, then cur for --tokens, cur's lag made certain: chip --skew-chip runs
         kernels/l1_scribble.cpp (its L1 CB region written over and over) right after the drafter's fill, before the
         verify step. The fill's synchronization (dflash.FILL_SYNC) makes it harmless; without it the step reads
         overwritten hub slots. cur must equal ref
  long:  the same without the skew, --repeat times (after --heat s of long requests back to back, once: sustained
         decode throttles chip 1, the lag of the bug's sessions)
Each served request logs its first token, its prefill logits (NaNs, max), the drafter's mean acceptance and the chips'
AICLK (sysfs, MHz). Exit 1 if any cur differs from ref.

usage (vLLM venv, plugin on PYTHONPATH): python -m qwen36_resident.tests.dfl_long_predecessor [--cases 8192:4:5]
    [--long 1024] [--tokens 64] [--repeat 1] [--heat 0] [--skew-chip 1] [--skew-passes 100] [--no-skew]
"""
import argparse
import faulthandler
import glob
import os
import sys
import time

import torch

import ttnn
from qwen36_resident import dflash
from qwen36_resident import generator_vllm as G
from qwen36_resident import weights as Q
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

KERNEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels", "l1_scribble.cpp")

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default="8192:4:5")
ap.add_argument("--prompts", default="/home/ttuser/experiments/qwen36_27b/profiles/resident/dfl/ttft/ttft2_prompts.pt")
ap.add_argument("--max-seq-len", type=int, default=16384)
ap.add_argument("--long", type=int, default=1024)
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--heat", type=float, default=0.0)
ap.add_argument("--skew-chip", type=int, default=1)
ap.add_argument("--skew-passes", type=int, default=100)
ap.add_argument("--skew-bytes", type=int, default=384 * 1024)
ap.add_argument("--no-skew", action="store_true")
a = ap.parse_args()
faulthandler.dump_traceback_later(1800, repeat=True, file=sys.stderr)


def aiclk():
    """the chips' AICLK (MHz) from sysfs (tt-kmd; opens no device), in tenstorrent!<i> order"""
    out = []
    for d in sorted(glob.glob("/sys/class/tenstorrent/tenstorrent!*"), key=lambda p: int(p.rsplit("!", 1)[1])):
        try:
            out.append(int(open(f"{d}/tt_aiclk").read()))
        except (OSError, ValueError):
            out.append(-1)
    return out


def prompts(isl):
    """8 prompts of isl tokens: the ttft2 session's (vllm bench random, seed 0), else seeded random ids"""
    if os.path.exists(a.prompts):
        return [list(map(int, p)) for p in torch.load(a.prompts)[isl, 1024]]
    return torch.randint(0, 150000, (8, isl), generator=torch.Generator().manual_seed(isl)).tolist()


mesh = open_mesh()
failed = []
try:
    t0 = time.time()
    obj = G.Qwen36ResidentDFlashForCausalLM(mesh, Q.Checkpoint(Q.checkpoint_dir()), a.max_seq_len, False)
    spec = obj.spec
    log(
        f"{type(obj).__name__} built in {time.time() - t0:.0f} s; dflash.FILL_SYNC "
        f"{getattr(dflash, 'FILL_SYNC', 'absent (a tree before the fix)')}; AICLK {aiclk()}"
    )
    # the class returns start()'s logits masked to the argmax: keep the raw ones
    raw, start = {}, spec.start

    def start_raw(ids):
        raw["lg"] = start(ids)
        return raw["lg"]

    spec.start = start_raw

    def serve(ids, n):
        """one request as the plugin serves it: (prefill logits [vocab], n tokens, the drafter's stats)"""
        T = len(ids)
        obj.prefill_forward(torch.tensor([ids]), None, None, torch.tensor([T]))
        lg = raw.pop("lg").reshape(-1).float()
        out = [int(lg.argmax())]
        while len(out) < n:
            out += obj.decode_forward(torch.tensor([[out[0]]]), torch.tensor([T])).reshape(-1).tolist()
        st = spec.stats()
        obj.release_request(0)
        return lg, out[:n], st

    def describe(lg, out, st):
        return (
            f"first token {out[0]}, prefill logits nan {int(torch.isnan(lg).sum())} max "
            f"{float(lg.nan_to_num().max()):.3g}, mean accept {st.get('mean_accept', 0):.2f} "
            f"({st.get('iterations', 0)} iterations)"
        )

    # the lag: l1_scribble on every worker core of chip --skew-chip (no passes on the others), enqueued right after the
    # drafter's fill (inside _fill_context: before its synchronization)
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    skew_pd = ttnn.MeshProgramDescriptor()
    for chip in range(mesh.get_num_devices()):
        rt = ttnn.RuntimeArgs()
        for x in range(grid.x):
            for y in range(grid.y):
                rt[x][y] = [a.skew_passes if chip == a.skew_chip else 0, 0x7F807F80]  # (bf16 +inf pairs)
        coord = ttnn.MeshCoordinate(0, chip)
        skew_pd[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[a.skew_bytes],
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                )
            ],
            semaphores=[],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=a.skew_bytes,
                    core_ranges=cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=a.skew_bytes)
                    ],
                )
            ],
        )
    fill = spec.drafter.fill_context

    def fill_then_skew(*args, **kw):
        fill(*args, **kw)
        ttnn.generic_op([spec.taps[0], spec.taps[0]], skew_pd)  # (the kernel touches neither)

    def heat():
        """--heat s of long requests back to back (sustained decode: chip 1's AICLK throttles after ~80 s)"""
        h0, k, P = time.time(), 0, prompts(int(a.cases.split(",")[0].split(":")[0]))
        while time.time() - h0 < a.heat:
            lg, out, st = serve(P[k % len(P)], a.long)
            log(f"heat request {k}: {describe(lg, out, st)}; AICLK {aiclk()} ({time.time() - h0:.0f} s)")
            k += 1

    for n_case, case in enumerate(a.cases.split(",")):
        isl, prev, cur = (int(x) for x in case.split(":"))
        P = prompts(isl)
        ref = spec.greedy(P[cur], a.tokens)
        log(f"[{case}] ref (verify model alone) first 12 {ref[:12]}")
        # skew first (on cool chips: the lag it makes is certain), then the heat, then the long-predecessor runs
        for kind in ([] if a.no_skew else ["skew"]) + ["heat"] * (n_case == 0 and a.heat > 0) + ["long"] * a.repeat:
            if kind == "heat":
                heat()
                continue
            t1 = time.time()
            lg, out, st = serve(P[prev], a.long)
            log(f"[{case}] {kind}: prev {prev} ({a.long} tokens): {describe(lg, out, st)}")
            clk = aiclk()
            if kind == "skew":
                spec.drafter.fill_context = fill_then_skew
            try:
                lg, out, st = serve(P[cur], a.tokens)
            finally:
                spec.drafter.__dict__.pop("fill_context", None)
            same = next((i for i, (x, y) in enumerate(zip(out, ref)) if x != y), len(out))
            if same < len(out):
                failed.append(f"{case} {kind}")
            log(
                f"[{case}] {kind}: cur {cur}: {describe(lg, out, st)}; == ref for {same}/{len(out)} "
                f"{'ok' if same == len(out) else 'WRONG'}; first 12 {out[:12]}; AICLK at its start {clk} "
                f"({time.time() - t1:.0f} s)"
            )
finally:
    close_mesh(mesh)
log(f"dfl_long_predecessor: {'FAIL ' + ', '.join(failed) if failed else 'PASS'}")
sys.exit(1 if failed else 0)

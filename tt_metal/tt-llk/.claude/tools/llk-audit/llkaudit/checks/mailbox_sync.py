# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
mailbox-sync (lite) checker — in-tree RISC<->RISC mailbox FIFO endpoints.

Mailboxes are directed point-to-point FIFOs between the baby-RISCV cores. A
misuse deadlocks (blocking read on a never-written FIFO / writer stalled on a
full one) or loses ordering. The functional mailbox surface INSIDE tt-llk is
tiny — the T1->T0 dst_index hand-off (cmath writes, cunpack reads) plus the
debug halt/unhalt handshake; the large surface (the CB tile-address/value
broadcast) lives in the compute API / kernels, OUTSIDE the headers this tool
parses (that is the kernel/JIT tier — run.sh --full-jit).

So this is a pure AUGMENTOR over the in-tree surface: it enumerates every FIFO
endpoint, decodes its directed channel (source_thread -> dest_thread) via the
write-dest / read-src convention, and pairs writers with readers of the *same*
resolved channel. The balance-over-iterations, call-count symmetry across
branches, overflow, and fence-ordering VERDICTS are semantic and stay with the
/mailbox-sync-audit skill (see blind_spots).

Hints:
  PAIRED_CHANNEL      — endpoint whose directed FIFO has a matching opposite-end
                        (writer<->reader) site in tt-llk.
  UNPAIRED_ENDPOINT   — resolved channel with NO in-tree partner: the partner is
                        likely in the compute-API/kernel tier, OR it is a real
                        imbalance. LLM adjudicates.
  UNRESOLVED_ENDPOINT — the issuing thread is not derivable from the file (e.g.
                        ckernel_debug.h), so the channel can't be resolved; hand
                        to the skill.
"""
from __future__ import annotations

from .. import registry
from ..factbase import FactBase
from .base import Check, Finding


class MailboxSync(Check):
    name = "mailbox-sync"
    description = "In-tree RISC<->RISC mailbox FIFO endpoints + pairing status"
    blind_spots = (
        "Only the IN-TREE mailbox surface is seen; the CB tile-address/value "
        "broadcast (tt_metal/hw/inc/api dataflow/compute) and user kernels are "
        "OUT of scope (kernel/JIT tier). Pairing is a static same-channel match "
        "only — it does NOT verify push/pop BALANCE over a complete iteration, "
        "call-count symmetry across control-flow branches (the deadlock trap), "
        "FIFO overflow (depth-4), or the ordering caveat (a mailbox write does "
        "not fence a prior store to a different region; a plain fence is a no-op "
        "on WH and insufficient on BH). The issuing thread is inferred from the "
        "file, so endpoints in thread-agnostic files (ckernel_debug.h) are "
        "UNRESOLVED. Quasar mailbox HW semantics are unverified here."
    )

    def run(self, fb: FactBase) -> list[Finding]:
        # 1. Gather unique endpoints (dedup nested TU re-parses at one site).
        eps: list[dict] = []
        seen: set = set()
        for c in fb.family("call"):
            op = registry.mailbox_op(c.get("name", ""))
            if not op:
                continue
            key = (c["file"], c.get("line"), c.get("name"), c.get("arg0", ""))
            if key in seen:
                continue
            seen.add(key)
            target = registry.mailbox_thread_arg(c.get("arg0", ""))
            issuing = registry.thread_of(c["file"])
            issuing = None if issuing == "UNKNOWN" else issuing
            # Directed channel (source -> dest): a write flows issuing->target;
            # a read/query drains target->issuing.
            if op == "write":
                src, dst = issuing, target
            else:
                src, dst = target, issuing
            channel = (src, dst) if (src and dst) else None
            eps.append(
                {"fact": c, "op": op, "src": src, "dst": dst, "channel": channel}
            )

        # 2. Which resolved channels are written / have a REAL blocking read?
        # A `not_empty` query is non-blocking and does NOT drain the FIFO, so it
        # does not make a write "paired" — a write with only a query (no read) is
        # a potential no-reader deadlock and must stay UNPAIRED. A query endpoint
        # itself is "paired" if the channel is written (it expects a producer).
        written = {e["channel"] for e in eps if e["op"] == "write" and e["channel"]}
        read_ch = {e["channel"] for e in eps if e["op"] == "read" and e["channel"]}

        # 3. Emit one recall finding per endpoint with its pairing status.
        findings: list[Finding] = []
        for e in eps:
            c, ch = e["fact"], e["channel"]
            fn = fb.enclosing(c["file"], c["off"])
            if ch is None:
                hint = "UNRESOLVED_ENDPOINT"
                chan_str = f"{e['src'] or '?'}->{e['dst'] or '?'}"
            else:
                # write needs a real read; read/query need a write.
                paired = (ch in read_ch) if e["op"] == "write" else (ch in written)
                hint = "PAIRED_CHANNEL" if paired else "UNPAIRED_ENDPOINT"
                chan_str = f"{ch[0]}->{ch[1]}"
            # Evidence: the opposite-end sites on the same channel (if any).
            partners = [
                self._ev(o["fact"], f'{o["op"]} {o["fact"].get("arg0","")}')
                for o in eps
                if ch is not None and o["channel"] == ch and o["op"] != e["op"]
            ]
            findings.append(
                Finding(
                    file=c["file"],
                    line=c.get("line", 0),
                    function=fn.name if fn else c.get("function", ""),
                    kind=f"mailbox:{e['op']}",
                    hint=hint,
                    detail=f"mailbox {e['op']} on channel {chan_str}",
                    evidence=[self._ev(c, f'{e["op"]} {c.get("arg0","")}')] + partners,
                )
            )
        return findings

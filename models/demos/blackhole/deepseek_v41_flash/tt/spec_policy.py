# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Default decode mode of DeepSeek-V4.1-Flash: which speculative-decoding mode a (padded) batch size gets, resolved in ONE place (no ttnn import: pure python, unit tested on the CPU).

Used by ``tt.common._env_setup`` (build time: DSV41_RING_ROWS), ``demo/text_demo.py`` (which pass runs) and ``tt.generator.Generator.enable_spec`` (which runners are built), so the model
build and the decode always agree, also after ``Generator.reconfigure`` (the choice is made per batch size; the pool, hence the ring rows, is rebuilt by every reconfigure).

Default (no DSV41_SPEC* variable set), U = users per mesh row = ceil(batch / mesh rows):
  B = 4 (U=1) adaptive {3,5} | B = 8 (U=2), 16 (U=4) adaptive {1,3,5} | B = 32 (U=8) adaptive {0,1,3} (0 = plain rounds, the remedy where spec loses at low acceptance)
  B >= 64: plain decode (opt-in: DSV41_SPEC=<k>, DSV41_SPEC_ADAPT=1, DSV41_SPEC_SET or DSV41_SPEC_ROWS=1; B >= 128 additionally needs DSV41_SPEC_B128=1)
  contexts too long for the spec runners' DRAM (or the matmul indexer backend): plain, with a loud log line.
Explicit variables always win: DSV41_SPEC=0 -> plain (no spec runner, ring rows untouched), DSV41_SPEC=<k> -> fixed k (adaptive with DSV41_SPEC_ADAPT=1), DSV41_SPEC_SET, DSV41_SPEC_ROWS.
"""

import os
from dataclasses import dataclass, field

RING_ROWS_SPEC = 288  # window ring rows of a spec-capable pool (>= 128 + 127 replay warm-up + k), tt/spec_paged.py
MAX_ROWS_DEFAULT_B = 32  # largest batch with default (adaptive) spec decode
B_PLAIN_DEFAULT = 64  # B >= 64: plain by default (opt-in only)
B_PLAIN_FORCE = 128  # B >= 128: plain unless DSV41_SPEC_B128=1

# Longest ``max_seq_len`` (the build / pool context, prompt + generated) the spec runners are expected to fit in DRAM per users-per-mesh-row U. Measured MEMLOG (40 layers, bf16 pool, this
# package's notes): after the prefills the free DRAM must cover the first runner (+111 MiB/bank, +0.7 per extra runner) and its trace capture (+31 MiB/bank) plus headroom (SPEC_NEED_FREE_MIB).
# B=8/16 at ISL 60453 (max_seq_len 70000): 239 / 159 MiB/bank left AFTER the runners: fits. B=32 at ISL 60k: 172 MiB free before the runner build -> OOM in the runner build (log
# spec_adapt_g32_isl64k_h33); B=32 at 32k: 270 MiB free before the runner (spec not run there, the plain pass fit): projected to fit.
SPEC_MAX_CTX = {1: 70000, 2: 70000, 4: 70000, 8: 40000}
SPEC_MAX_CTX_BACKEND = 131072  # beyond this the indexer is not the matmul backend (n_entries = ctx / ratio 2 > 65536, tt/indexer.py default_backend): 'spec verify needs the matmul indexer backend'
SPEC_NEED_FREE_MIB = (
    190  # runtime guard (free DRAM / bank before building the runners): 111 runner + 31 trace + ~48 headroom
)
SPEC_NEED_LARGEST_MIB = 40

_applied = (
    {}
)  # env variables this module set (so a later resolve / reconfigure does not mistake them for explicit user settings)


def default_ks(U, rows=False):
    """Candidate verification lengths per users-per-mesh-row with fast / supported row counts T = U * (1 + k) (mHC fast paths 4 / 8 / 16 / 24 / 32; T <= 32; T = 5..7 pad to 8;
    T = 2 / 3 are not supported). T > 32 needs the chunked verify (DSV41_SPEC_ROWS=1): B=64 (U=16) {1: T=32, 3: T=64}, B=128 (U=32) {1: T=64, 3: T=128}.
    """
    table = {1: [3, 5], 2: [1, 3, 5], 4: [1, 3, 5], 8: [1, 3]}
    if rows and U in (16, 32):
        return [1, 3]
    return table.get(U, [k for k in (1, 3) if U * (1 + k) <= 32] or [1])


def parse_ks(spec, default):
    """'1,3,5' -> [1, 3, 5]"""
    return sorted({int(x) for x in str(spec or default).split(",") if x.strip()})


@dataclass
class SpecChoice:
    mode: str  # 'plain' | 'fixed' | 'adaptive'
    k: int = 0  # DSV41_SPEC value: fixed k, or the largest candidate of the adaptive set
    ks: list = field(default_factory=list)  # adaptive candidate set (0 = plain rounds)
    rows: bool = False  # chunked > 32-row verify (DSV41_SPEC_ROWS=1 behaviour)
    reason: str = ""
    explicit: bool = False
    ring_rows: int = 0  # DSV41_RING_ROWS this choice needs at build time (0 = leave the default)

    @property
    def on(self):
        return self.mode != "plain"

    def describe(self):
        if self.mode == "plain":
            return f"spec decode: plain ({self.reason})"
        if self.mode == "adaptive":
            return f"spec decode: adaptive {{{','.join(map(str, self.ks))}}} ({self.reason})"
        return f"spec decode: fixed k={self.k} ({self.reason})"


def _get(env, name):
    """User-set value of an env variable ('' counts as unset); values this module applied itself are not user settings."""
    v = env.get(name)
    if v in (None, "") or (env is os.environ and _applied.get(name) == v):
        return None
    return v


def users_per_row(batch, mesh_rows=4):
    return -(-batch // mesh_rows)


def resolve(batch, mesh_rows=4, max_seq_len=None, env=None):
    """-> ``SpecChoice`` for this (padded) batch size and build context. ``env`` defaults to ``os.environ``."""
    env = os.environ if env is None else env
    U = users_per_row(batch, mesh_rows)
    raw = _get(env, "DSV41_SPEC")
    adapt = _get(env, "DSV41_SPEC_ADAPT")
    sset = _get(env, "DSV41_SPEC_SET")
    rows_env = _get(env, "DSV41_SPEC_ROWS") == "1"
    forced128 = _get(env, "DSV41_SPEC_B128") == "1"
    if raw is not None and int(raw) == 0:
        return SpecChoice("plain", reason="explicit DSV41_SPEC=0", explicit=True)
    opt_in = raw is not None or adapt == "1" or sset is not None or rows_env
    if batch >= B_PLAIN_FORCE and not forced128:
        return SpecChoice(
            "plain",
            reason=f"B>=128"
            + (f"; DSV41_SPEC={raw} ignored, DSV41_SPEC_B128=1 forces spec" if raw is not None else ""),
            explicit=raw is not None,
        )
    if batch >= B_PLAIN_DEFAULT and not opt_in:
        return SpecChoice(
            "plain", reason="B>=64 default; opt in with DSV41_SPEC / DSV41_SPEC_ADAPT=1 / DSV41_SPEC_ROWS=1"
        )
    rows = rows_env or (U in (16, 32) and opt_in)  # B>=64 opt-in switches the chunked verify on
    if raw is None:
        # default / opt-in without DSV41_SPEC: adaptive (DSV41_SPEC_ADAPT=0 -> fixed k from the default set)
        dks = default_ks(U, rows)
        if U >= 8:
            dks = [0] + dks  # plain rounds as an option where spec loses at low acceptance (B>=32)
        ks = parse_ks(sset, ",".join(map(str, dks)))
        why = f"default for B={batch}" if not opt_in else f"opt-in for B={batch}"
        if sset is not None:
            why += ", DSV41_SPEC_SET"
        if adapt == "0":
            k = 3 if 3 in ks else max(ks)
            c = SpecChoice("fixed", k, [], rows, f"DSV41_SPEC_ADAPT=0, {why}", opt_in)
        else:
            c = SpecChoice("adaptive", max(ks), ks, rows, why, opt_in)
        if max(ks) == 0:
            return SpecChoice("plain", reason="DSV41_SPEC_SET has only 0", explicit=True)
    else:
        k = int(raw)
        if adapt == "1":
            ks = parse_ks(sset, ",".join(map(str, default_ks(U, rows))))
            c = SpecChoice("adaptive", max(ks), ks, rows, f"explicit DSV41_SPEC={k} DSV41_SPEC_ADAPT=1", True)
        else:
            c = SpecChoice("fixed", k, [], rows, f"explicit DSV41_SPEC={k}", True)
    c.ring_rows = RING_ROWS_SPEC
    if not c.explicit and max_seq_len is not None:
        lim = min(SPEC_MAX_CTX.get(U, SPEC_MAX_CTX[8]), SPEC_MAX_CTX_BACKEND)
        if max_seq_len > lim:
            return SpecChoice(
                "plain", reason=f"context too long for spec: max_seq_len {max_seq_len} > {lim} at B={batch}", rows=False
            )
    return c


def apply(choice, env=None):
    """Make the process environment agree with ``choice``: DSV41_SPEC_ROWS for the chunked verify (read at runtime by tt/spec_model.py, spec_decoder.py, attention.py) and, at BUILD time,
    DSV41_RING_ROWS=288 for a pool that spec runners will use (a user-set DSV41_RING_ROWS wins). Undoes what an earlier ``apply`` set (reconfigure to another batch size).
    """
    env = os.environ if env is None else env
    for k, v in list(_applied.items()):
        if env.get(k) == v:
            env.pop(k)
        _applied.pop(k)
    want = {}
    if choice.on and choice.rows and env.get("DSV41_SPEC_ROWS") != "1":
        want["DSV41_SPEC_ROWS"] = "1"
    if choice.on and choice.ring_rows and not env.get("DSV41_RING_ROWS"):
        want["DSV41_RING_ROWS"] = str(choice.ring_rows)
    for k, v in want.items():
        env[k] = v
        _applied[k] = v
    return choice


def resolve_apply(batch, mesh_rows=4, max_seq_len=None, env=None):
    return apply(resolve(batch, mesh_rows, max_seq_len, env), env)


def dram_check(free_mib, largest_mib):
    """Runtime guard before the runners are built (after the prefills): -> (ok, message)."""
    if free_mib >= SPEC_NEED_FREE_MIB and largest_mib >= SPEC_NEED_LARGEST_MIB:
        return (
            True,
            f"DRAM free {free_mib:.0f} MiB/bank (largest block {largest_mib:.0f}) >= {SPEC_NEED_FREE_MIB}/{SPEC_NEED_LARGEST_MIB}",
        )
    return False, (
        f"DRAM free {free_mib:.0f} MiB/bank (largest block {largest_mib:.0f}) < the {SPEC_NEED_FREE_MIB} (largest {SPEC_NEED_LARGEST_MIB}) the spec runners + trace need"
    )

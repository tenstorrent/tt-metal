# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_ATTN_KV_SPLIT=user: KV-head pair split of the spec-verify and MTP-drafter SDPA (kernel plan K1, form P0).

At TP8 the 4 KV heads are replicated in chip pairs: devices 2h and 2h+1 both hold KV head h for every user and position,
and each runs its own 3 q heads over that full KV, so every pair streams its KV head twice. The user split halves that:
device d = 2h + r runs ALL 6 q heads of pair h, but only for its own users (user u belongs to rank u % 2), over the full
context. Its SDPA then has half the groups (B/2 instead of B) with twice the cores each, and reads half the KV bytes.

No pair exchange: the device multiplies its own rows by the pair's 6-head o-projection (wo of heads 6h..6h+5, "wo_pair")
and leaves the other rows zero, so the existing all-reduce of the o-proj partial sums the four owner devices of every row.
The KV cache, the KV writes (every device writes every user) and prefill are unchanged.

This module holds the host side (no ttnn import at module level, so it is CPU-testable): the ownership rule, the
per-device selection matrices / index vectors, and the per-device own page tables. ``KVSplitPlan`` turns them into
per-device device constants (ttnn.ShardTensorToMesh over the TP axis) before any trace capture, and derives the
per-device own positions inside a trace body (``own_positions``). ``KVSplitGate`` (QWEN36_ATTN_KV_SPLIT_MIN_CTX /
_MIN_CTX_DRAFT) is the context gate: which row buckets get split trace variants and when a step replays them. The
"Slot parity" and "DRAM budget" sections below say what a row's output depends on and what the split costs in
prefix-cache slots (QWEN36_ATTN_KV_SPLIT_SLOT_TRADE).

Layouts (rows are user-major, row u*T + j = user u, candidate j; T = 1 for the drafter):
  own_users(n, r)    users u < n with u % 2 == r (interleaved, so a row bucket's live rows 0..n-1 stay balanced)
  own_rows(n, T, r)  [u*T + j for u in own_users(n, r) for j in range(T)]   (Ro = len(own_rows) entries)
  sel   [tp, NHp, Ro, R]  sel[d, h, i, own_rows[i]] = 1: q_own = sel @ q (per head, dense [1, NHp, R, HD] layout)
  scat  [tp, 1, R, Ro]    scat[d, 0, own_rows[i], i] = 1: attn rows back to their global row, zeros elsewhere
  pos   [tp, Ro] uint32   own_rows: the gather index of the own positions out of the [R] row positions
  pt    [tp * n_own, nb]  own users' page-table rows (ShardTensorToMesh dim 0 -> [n_own, nb] per device)
"""

import os

import torch


def kv_split_mode():
    """QWEN36_ATTN_KV_SPLIT (default "0" = off, nothing differs): "user" = the P0 user split above. "seq" (the position
    split, kernel plan K11) needs an LSE-returning SDPA and is rejected."""
    v = os.environ.get("QWEN36_ATTN_KV_SPLIT", "0").strip().lower()
    if v in ("", "0", "off", "false"):
        return None
    if v == "user":
        return "user"
    raise ValueError(f"QWEN36_ATTN_KV_SPLIT={v!r}: only 'user' (P0) is implemented ('seq' is kernel plan K11)")


def kv_split_parts():
    """QWEN36_ATTN_KV_SPLIT_PARTS (default "verify,draft"): which SDPAs take the split when QWEN36_ATTN_KV_SPLIT=user --
    the spec verify ("verify") and / or the MTP drafter steps ("draft"). Read at trace capture (A/B isolation knob)."""
    v = os.environ.get("QWEN36_ATTN_KV_SPLIT_PARTS", "verify,draft")
    parts = {p.strip().lower() for p in v.split(",") if p.strip()}
    bad = parts - {"verify", "draft"}
    if bad:
        raise ValueError(f"QWEN36_ATTN_KV_SPLIT_PARTS={v!r}: unknown part(s) {sorted(bad)} (use verify, draft)")
    return parts


# --------------------------------------------------------------------------------------------------------------------
# Context gate (QWEN36_ATTN_KV_SPLIT_MIN_CTX / QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT)
#
# The split costs a fixed ~35-40 us per attention layer (the pair in-projection reads 3584 instead of 2048 weight columns,
# the 6-head o-projection 1536 instead of 768 rows, plus the row select / scatter) and saves SDPA time that grows with the
# context, so it only wins past a break-even context, which also depends on the row bucket (fewer users already give each
# SDPA group more cores, so the split gains less) and on the part (the drafter's decode SDPA breaks even earlier than the
# verify's packed SDPA). Measured on d21u20 (production-loop harness, 8 users, T = 4): verify +0.60 ms at 4K / -2.11 ms at
# 32K (break-even ~10.4K), drafter +0.04 / -0.40 ms (~6.8K); 4 users at 32K: verify -0.43, drafter -0.12 ms. The 4- and
# 2-user thresholds come from a ttsim-perf sweep of the same single-chip op chain (one verify layer / one draft step at
# 2 / 4 / 8 users x 4K / 16K / 32K / 64K, split vs unsplit) mapped onto the device with the 8-user points (device delta =
# 10.9 + 1.04 x sim per verify layer, 16.5 + 0.96 x sim per draft step): break-even verify 9.9K / 22.3K / 43.8K and
# drafter 6.8K / 20.7K / 40.6K at 8 / 4 / 2 users (the 4-user device point at 32K puts the verify nearer 25K). The
# 2-user buckets gain at most ~0.6 ms per step even at 64K, so they get no variant by default ("2:45056" enables them).
#
# With a gate (any threshold > 0), every row bucket keeps its unsplit traces (byte for byte r5's programs) and the buckets
# with a threshold get a second, split variant of the verify (main + logits + topk) and / or drafter trace. Per step the
# engine replays the split variant when the step's context -- the largest (start position + T) over its live rows, which
# is what the slowest SDPA group scans -- reaches the bucket's threshold. Both variants read and write the same persistent
# buffers (inputs, KV, GDN ring, shared outputs), so switching between them is free (no chain break, no restaging).
# --------------------------------------------------------------------------------------------------------------------
# Default thresholds per row bucket (None = never split that bucket): 8 users from the device (10.4K -> 10240 verify,
# 6.8K -> 7168 drafter), 4 users from the calibrated sim (22-25K -> 24576 verify, 20.7K -> 20480 drafter), 2 users off.
GATE_DEFAULT = {
    "verify": {8: 10240, 4: 24576, 2: None},
    "draft": {8: 7168, 4: 20480, 2: None},
}
GATE_PARTS = ("verify", "draft")
_GATE_ENV = {"verify": "QWEN36_ATTN_KV_SPLIT_MIN_CTX", "draft": "QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT"}


def parse_min_ctx(text, default):
    """Parse one gate table: "" -> ``default`` (a {bucket: threshold or None} dict, copied); "off" -> {} (never split);
    an integer N >= 0 -> N for every even bucket 2..32 (0 = split at every context); a comma list "8:10240,4:24576,2:off"
    -> per bucket (buckets not listed never split). Raises ValueError on anything else."""
    t = "" if text is None else str(text).strip().lower()
    if not t:
        return dict(default)
    if t == "off":
        return {}
    if ":" not in t:
        try:
            n = int(t)
        except ValueError:
            raise ValueError(f"KV split gate {text!r}: an integer, 'off' or a 'bucket:ctx,...' list") from None
        if n < 0:
            raise ValueError(f"KV split gate {text!r}: a negative context")
        return {b: n for b in range(2, 33, 2)}
    out = {}
    for part in t.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(":")
        try:
            b = int(k)
        except ValueError:
            raise ValueError(f"KV split gate {text!r}: bucket {k!r} is not an integer") from None
        if b < 2 or b % 2 or b > 32:
            raise ValueError(
                f"KV split gate {text!r}: bucket {b} is not an even width in 2..32 (the split needs pairs)"
            )
        v = v.strip()
        if v == "off":
            out[b] = None
            continue
        try:
            n = int(v)
        except ValueError:
            raise ValueError(
                f"KV split gate {text!r}: context {v!r} of bucket {b} is not an integer or 'off'"
            ) from None
        if n < 0:
            raise ValueError(f"KV split gate {text!r}: a negative context for bucket {b}")
        out[b] = n
    return out


class KVSplitGate:
    """The context gate: per part ("verify" / "draft") a {row bucket: min context or None} table, and the mode it implies:

      * "always" -- both tables are the uniform 0 table (QWEN36_ATTN_KV_SPLIT_MIN_CTX=0, the K1 measurement setting):
        every trace of an even width is captured split, as the K1 commit did, and nothing is gated;
      * "never"  -- both tables are empty: every trace unsplit (the pair weights are loaded but unused);
      * "gated"  -- anything else: unsplit traces everywhere plus split variants for the buckets with a threshold,
        picked per step by ``want``.

    Pure host object (no ttnn): built from the environment once per engine / model (``from_env``)."""

    def __init__(self, verify, draft):
        self.tables = {"verify": dict(verify), "draft": dict(draft)}
        uniform0 = {b: 0 for b in range(2, 33, 2)}
        if all(self.tables[p] == uniform0 for p in GATE_PARTS):
            self.mode = "always"
        elif all(not any(v is not None for v in self.tables[p].values()) for p in GATE_PARTS):
            self.mode = "never"
        else:
            self.mode = "gated"

    @classmethod
    def from_env(cls):
        """QWEN36_ATTN_KV_SPLIT_MIN_CTX (verify) and QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT (drafter). An unset drafter value
        follows an explicitly set verify value (so QWEN36_ATTN_KV_SPLIT_MIN_CTX=0 alone is the K1 always-split setting);
        both unset -> GATE_DEFAULT."""
        vt = os.environ.get(_GATE_ENV["verify"], "")
        dt = os.environ.get(_GATE_ENV["draft"], "")
        if not dt.strip() and vt.strip():
            dt = vt
        return cls(parse_min_ctx(vt, GATE_DEFAULT["verify"]), parse_min_ctx(dt, GATE_DEFAULT["draft"]))

    def threshold(self, part, b):
        return self.tables[part].get(int(b))

    def want(self, part, b, ctx):
        """True when bucket ``b``'s ``part`` should replay its split variant at context ``ctx`` (gated mode only)."""
        if self.mode != "gated" or ctx is None:
            return False
        t = self.threshold(part, b)
        return t is not None and int(ctx) >= t

    def variant_buckets(self, part, buckets):
        """The even buckets of ``buckets`` that get a split variant of ``part`` (gated mode; descending)."""
        if self.mode != "gated":
            return ()
        return tuple(
            sorted(
                (int(b) for b in buckets if split_n_users_ok(int(b)) and self.threshold(part, b) is not None),
                reverse=True,
            )
        )

    def canonical_split(self):
        """Whether the canonical (non-variant) traces of an even width are captured split: only in "always" mode."""
        return self.mode == "always"

    def describe(self):
        def one(p):
            t = self.tables[p]
            if not t:
                return "off"
            return ",".join(f"{b}:{'off' if v is None else v}" for b, v in sorted(t.items(), reverse=True))

        return f"mode={self.mode} verify[{one('verify')}] draft[{one('draft')}]"


def kv_split_canonical():
    """Whether the canonical traces of an even width are captured split (KVSplitGate "always" mode). Read per capture."""
    return KVSplitGate.from_env().canonical_split()


def kv_split_gate_ctx(start_pos, active, T):
    """The context a verify / drafter step's split decision uses: the largest ``start_pos[u] + T`` over the live rows (the
    KV span the slowest SDPA group scans: every group gets the same core share, so the step waits for the longest row).
    ``start_pos`` / ``active``: per engine row (inactive rows run at position 0 and are ignored). Pure; 0 for no live row.
    In deferred serving the start positions are the runner's (chain rows: its upper bound of the step's first position),
    never the engine's own bounds, whose tightness depends on whether the previous step had been read back: the decision
    is then a function of the request stream alone, not of host timing."""
    sp = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
    act = (
        torch.as_tensor(active).reshape(-1).to(torch.bool)
        if active is not None
        else torch.ones_like(sp, dtype=torch.bool)
    )
    if not bool(act.any()):
        return 0
    return int(sp[act].max()) + int(T)


def kv_split_supported(args):
    """QWEN36_ATTN_KV_SPLIT=user is on and the TP layout has KV pairs (TP == 2 * n_kv_heads: devices 2h and 2h+1 hold
    the replicated KV head h, one local KV head each)."""
    return kv_split_mode() == "user" and kv_split_layout_ok(
        getattr(args, "num_devices", 1), getattr(args, "n_kv_heads", 0), getattr(args, "n_local_kv_heads", 0)
    )


# --------------------------------------------------------------------------------------------------------------------
# Slot parity (what a row's output depends on with the split on)
#
# Engine row u is owned by pair rank u % 2 in every pair: all six q heads of pair h for row u run on device 2h + u % 2,
# over THAT device's replica of KV head h, and its o-proj partial reaches the all-reduce from that device only. The two
# replicas of a KV head agree on every position written in decode (both devices compute k / v with the same weights and
# program) but not bit for bit on prefill-written positions (the prefill all-gather matmul accumulates the gathered
# shards in a device-dependent order). So with the split on, a row's tokens are a deterministic function of its prompt,
# its sampling, the PARITY of its engine row and the per-step (bucket, split) choices -- not of the row index itself:
# rows of equal parity running the same prompt decode identically, rows of different parity may not (K1 evidence:
# replicated-prompt user 1 left user 0 at token 47). A live request never changes engine row (row swaps only move free
# rows), so its parity is fixed for its lifetime. Unsplit steps keep every row equal (each row's heads read one replica
# of each pair half and half, independent of the row).
# --------------------------------------------------------------------------------------------------------------------
def kv_split_owner(row):
    """Pair rank (0 = device 2h, 1 = device 2h+1 of every pair h) that owns engine row ``row`` under the split."""
    return int(row) % 2


def kv_split_identity_classes(n_rows, split_engaged):
    """Groups of engine rows that must decode identically when they run the same prompt with greedy decoding: every row
    together when no split step ran, else one class per pair rank (even rows, odd rows)."""
    rows = list(range(int(n_rows)))
    if not split_engaged:
        return [rows]
    return [r for r in ([u for u in rows if u % 2 == 0], [u for u in rows if u % 2 == 1]) if r]


# --------------------------------------------------------------------------------------------------------------------
# DRAM budget against the prefix-cache slots (QWEN36_ATTN_KV_SPLIT_SLOT_TRADE)
#
# The split adds per device: the pair weights of every attention layer it serves (16 verify layers + the MTP layer:
# [q pair | k | v | gate pair] dim x (4 NH + 2 NKV) HD and wo_pair 2 NH HD x dim in the attention weight dtype, plus the
# pair q / k norm rows), the plan constants (a few KB) and, gated, the split variants' bucket-local verify outputs
# (normed rows, shard logits, full logits TILE + ROW_MAJOR). At Qwen3.8-27B TP8 BFP8 that is 453 MiB of weights (measured
# 455.2 MiB on d21u20) = 21 prefix-cache slots of 21.75 MiB (48 GDN layers x (384 KiB fp32 state + 80 KiB padded conv
# carry)); with the default gate's two verify variants (V8s, V4s: ~54 MiB of retained outputs) the total is ~507 MiB =
# 24 slots. Production (525312-token KV pool, 256 slots) leaves ~13 GiB free per device, so nothing has to give; the
# trade below keeps the device footprint of a DRAM-tight configuration unchanged by taking the split's bytes out of the
# slot pool instead. The variants also take trace region (~10.7 MiB per verify variant, ~1.2 MiB per drafter variant,
# as the row buckets: ~24 MiB for the default gate, against ~260 of 1024 MiB used in serving).
# --------------------------------------------------------------------------------------------------------------------
_DTYPE_BYTES = {"bfloat8_b": 1088 / 1024, "bfloat4_b": 576 / 1024, "bfloat16": 2.0, "float32": 4.0}


def tile_bytes(shape, dtype_name):
    """Device bytes of a TILE tensor of ``shape`` (last two dims padded to 32) in ``dtype_name`` (block-float tiles
    include their shared exponents)."""
    s = [int(d) for d in shape]
    if len(s) >= 2:
        s[-2] = -(-s[-2] // 32) * 32
    s[-1] = -(-s[-1] // 32) * 32
    n = 1
    for d in s:
        n *= d
    return int(n * _DTYPE_BYTES[dtype_name])


def kv_split_pair_weight_bytes(dim, nh_local, n_local_kv, hd, n_layers, dtype_name="bfloat8_b"):
    """Per-device bytes of the split's pair weights over ``n_layers`` attention layers (wqkvg_pair, wo_pair in
    ``dtype_name``; the bf16 pair q / k norm rows)."""
    per = (
        tile_bytes([dim, (4 * nh_local + 2 * n_local_kv) * hd], dtype_name)
        + tile_bytes([2 * nh_local * hd, dim], dtype_name)
        + tile_bytes([1, 2 * nh_local + n_local_kv, 1, hd], "bfloat16")
    )
    return int(n_layers) * per


def kv_split_variant_out_bytes(variant_rows, dim, vocab, tp, topk=True):
    """Per-device bytes the split verify variants with ``variant_rows`` rows each keep allocated: normed [rows, dim] bf16,
    shard logits [rows, vocab/tp] bf16 (with the device top-k), logits [rows, vocab] bf16 TILE and its ROW_MAJOR copy, plus
    one more full-vocab tile per variant: what a row bucket's capture is measured to retain beyond those (DRAM after
    V4 / V2 / V1: +24.8 / +21.0 / +21.0 MiB on d21u20, against 24.7 / 23.0 / 21.8 from this formula)."""
    tot = 0
    for rows in variant_rows:
        r32 = -(-int(rows) // 32) * 32
        tot += tile_bytes([1, 1, r32, dim], "bfloat16")
        if topk:
            tot += tile_bytes([1, 1, r32, vocab // tp], "bfloat16")
        tot += 2 * tile_bytes([1, 1, r32, vocab], "bfloat16") + int(rows) * int(vocab) * 2
    return tot


def prefix_slot_bytes(n_gdn_layers, rec_shape, carry_shape):
    """Per-device bytes of one prefix-cache slot: one fp32 recurrent state + one bf16 conv carry per GDN layer (TILE)."""
    return int(n_gdn_layers) * (tile_bytes(rec_shape, "float32") + tile_bytes(carry_shape, "bfloat16"))


def kv_split_slot_trade_enabled():
    """QWEN36_ATTN_KV_SPLIT_SLOT_TRADE (default "0" = report only): "1" takes the split's DRAM out of the prefix-cache slot
    pool (QWEN36_PREFIX_CACHE_SLOTS minus the slot equivalent), so enabling the split leaves the device footprint as it was.
    Read once, when the prefix cache is built."""
    return os.environ.get("QWEN36_ATTN_KV_SPLIT_SLOT_TRADE", "0") == "1"


def kv_split_slot_equivalent(extra_bytes, slot_bytes):
    """Prefix-cache slots whose DRAM equals ``extra_bytes`` (rounded up)."""
    if int(slot_bytes) <= 0:
        return 0
    return -(-int(extra_bytes) // int(slot_bytes))


def kv_split_slots_after_trade(num_slots, extra_bytes, slot_bytes, trade):
    """The prefix-cache slot count once the split's ``extra_bytes`` are budgeted: ``num_slots`` unchanged without the trade,
    else ``num_slots`` minus the slot equivalent (never below 1)."""
    if not trade or int(extra_bytes) <= 0:
        return int(num_slots)
    return max(1, int(num_slots) - kv_split_slot_equivalent(extra_bytes, slot_bytes))


def kv_split_layout_ok(num_devices, n_kv_heads, n_local_kv_heads):
    """The split needs KV pairs: TP == 2 * n_kv_heads (devices 2h and 2h+1 hold the replicated KV head h)."""
    return num_devices == 2 * n_kv_heads and n_local_kv_heads == 1 and num_devices % 2 == 0


def split_n_users_ok(n_users):
    """Widths the split serves: an even user count >= 2 (both pair members own the same number of users, so every
    device runs the same SDPA shape). 1 user (and any odd width) keeps the unsplit path."""
    return n_users >= 2 and n_users % 2 == 0


def own_users(n_users, rank):
    return [u for u in range(n_users) if u % 2 == rank]


def own_rows(n_users, T, rank):
    return [u * T + j for u in own_users(n_users, rank) for j in range(T)]


def sel_host(n_users, T, tp, n_heads):
    """[tp, n_heads, Ro, R] bf16 one-hot row selection (see the module docstring)."""
    R = n_users * T
    rows = [own_rows(n_users, T, d % 2) for d in range(tp)]
    Ro = len(rows[0])
    s = torch.zeros(tp, n_heads, Ro, R, dtype=torch.float32)
    for d in range(tp):
        for i, r in enumerate(rows[d]):
            s[d, :, i, r] = 1.0
    return s.to(torch.bfloat16)


def scatter_host(n_users, T, tp):
    """[tp, 1, R, Ro] bf16 one-hot placement of the own rows back into the R global rows."""
    R = n_users * T
    rows = [own_rows(n_users, T, d % 2) for d in range(tp)]
    Ro = len(rows[0])
    s = torch.zeros(tp, 1, R, Ro, dtype=torch.float32)
    for d in range(tp):
        for i, r in enumerate(rows[d]):
            s[d, 0, r, i] = 1.0
    return s.to(torch.bfloat16)


def pos_index_host(n_users, T, tp):
    """[tp, Ro] int32 (staged as uint32): own_rows per device, the gather index of the own row positions."""
    return torch.tensor([own_rows(n_users, T, d % 2) for d in range(tp)], dtype=torch.int32)


def own_page_table_host(pt, tp, groups_per_user=1):
    """Host page table [B, nb] (one row per user) -> [tp * B/2 * g, nb]: device d's block is its own users' rows (the
    rows of users d % 2, d % 2 + 2, ...), each repeated g = groups_per_user times (the fused spec SDPA wants one row
    per candidate group: T = 8 runs 2 groups per user, T = 12 three). A row bucket b uses the first b/2 * g rows."""
    pt = torch.as_tensor(pt).to(torch.int32)
    B = int(pt.shape[0])
    assert split_n_users_ok(B), f"own page table needs an even user count, got {B}"
    return torch.cat(
        [pt[own_users(B, d % 2)].repeat_interleave(int(groups_per_user), dim=0) for d in range(tp)], dim=0
    ).contiguous()


def own_positions_host(row_positions, n_users, T, rank):
    """Reference for KVSplitPlan.own_positions: the own rows' entries of the [n_users * T] row positions."""
    p = torch.as_tensor(row_positions).reshape(-1)
    return p[own_rows(n_users, T, rank)].clone()


def split_attention_reference(q, k_cache_rows, v_cache_rows, positions, wo_pair_rows, gate, n_users, T, rank, scale):
    """Pure torch model of one device's split attention output (for host tests): q [R, NH2, HD] (pair-global heads),
    per-user KV [n_users, S, HD], positions [R], gate [R, NH2*HD] (pre-sigmoid), wo_pair_rows [NH2*HD, dim].
    Returns the [R, dim] o-proj partial: own rows = sum over the pair's heads, other rows = 0."""
    R, NH2, HD = q.shape
    out = torch.zeros(R, NH2 * HD, dtype=torch.float32)
    for u in own_users(n_users, rank):
        for j in range(T):
            r = u * T + j
            p = int(positions[r])
            kk = k_cache_rows[u, : p + 1].float()
            vv = v_cache_rows[u, : p + 1].float()
            s = (q[r].float() @ kk.T) * scale  # [NH2, p+1]
            a = torch.softmax(s, dim=-1) @ vv  # [NH2, HD]
            out[r] = a.reshape(-1)
    gated = out * torch.sigmoid(gate.float())
    rows = torch.zeros(R, 1)
    rows[own_rows(n_users, T, rank)] = 1.0
    return (gated @ wo_pair_rows.float()) * rows


class KVSplitPlan:
    """Per-device device constants of the split, keyed by (n_users, T); built eagerly by ``prepare`` (never inside a
    trace capture). T = K+1 for the spec verify, 1 for the drafter steps."""

    def __init__(self, mesh, tp, nh_local):
        self.mesh = mesh
        self.tp = tp
        self.nh2 = 2 * nh_local  # q heads per device in the split (the pair's heads)
        self.parts = kv_split_parts()  # which SDPAs split (QWEN36_ATTN_KV_SPLIT_PARTS), fixed for the process
        self._c = {}

    def _shard0(self, t, dtype, layout):
        import ttnn

        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )

    def prepare(self, n_users, T):
        """Build the (n_users, T) constants (no-op when present or when the width is not split)."""
        import ttnn

        key = (int(n_users), int(T))
        if key in self._c or not split_n_users_ok(n_users):
            return
        n, tp = key[0], self.tp
        R = n * T
        Ro = (n // 2) * T
        pos = pos_index_host(n, T, tp)  # [tp, Ro] -> [1, Ro] per device (the gather wants the source's rank)
        self._c[key] = {
            "sel": self._shard0(sel_host(n, T, tp, self.nh2), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            "scatter": self._shard0(scatter_host(n, T, tp), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            # gather index of the own positions: [1, Ro] uint32 per device, in both layouts (the RM gather is one op;
            # the TILE path is the fallback when the RM gather does not take int32)
            "pos_rm": self._shard0(pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            "pos_tile": self._shard0(pos, ttnn.uint32, ttnn.TILE_LAYOUT),
            "R": R,
            "Ro": Ro,
            "n_own": n // 2,
        }

    def has(self, n_users, T):
        return (int(n_users), int(T)) in self._c

    def consts(self, n_users, T):
        c = self._c.get((int(n_users), int(T)))
        assert c is not None, f"KVSplitPlan: ({n_users} users, T={T}) not prepared before capture"
        return c

    def own_positions(self, positions, n_users, T):
        """In-trace: replicated int32 [R] ROW_MAJOR row positions -> per-device int32 [Ro] ROW_MAJOR own positions
        (``own_positions_host`` on every device). QWEN36_ATTN_KV_SPLIT_POS = "rm" (default): one ROW_MAJOR gather;
        "tile": to_layout + TILE gather + to_layout. Every own row has a non-negative position in serving (padding
        rows run at a real position over the null block), so no clamp: a -1 here would skip that group in the SDPA."""
        import ttnn

        c = self.consts(n_users, T)
        R, Ro = c["R"], c["Ro"]
        n = int(positions.shape[-1])
        assert n >= R, f"own_positions: {n} row positions for {n_users} users x T={T}"
        cut = ttnn.slice(positions, [0], [R]) if n > R else None  # never full span when taken
        src = ttnn.reshape(cut if cut is not None else positions, (1, R))
        mode = os.environ.get("QWEN36_ATTN_KV_SPLIT_POS", "rm")
        if mode == "tile":
            t = ttnn.to_layout(src, ttnn.TILE_LAYOUT)
            g = ttnn.gather(t, -1, index=c["pos_tile"])
            ttnn.deallocate(t)
            out = ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(g)
        else:
            out = ttnn.gather(src, -1, index=c["pos_rm"])
        if cut is not None:
            ttnn.deallocate(cut)
        return ttnn.reshape(out, (Ro,))

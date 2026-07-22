#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Analyze a ttnn graph capture for read-after-write (RAW) dependencies between DEVICE PROGRAMS.

Input: JSON produced by ttnn.graph.end_graph_capture_to_file(...) (a list of node dicts).
Granularity: DEVICE PROGRAMS = leaf function nodes (no nested function_start) that own >=1
circular_buffer_allocate (CBs are program-scoped). This is the barrier-bounded unit we care about.

Metrics (all at the device-program level; the DAG, not naive execution-order adjacency):
  - adjacent-RAW %      : does op n+1 read a tensor op n wrote (the no-reorder / today view)
  - distance-to-first-consumer distribution : for each op output, how many ops until its first
                          reader (>1 => barrier can overlap that many ops regardless of ordering)
  - critical path       : longest producer->consumer chain / total ops (independence ceiling)
"""
import json
import re
import sys
from collections import defaultdict


def parse_device_ops(nodes):
    """Return ordered list of device-program ops: {counter,name,in_ids(set),out_ids(set)}.

    A device op = an innermost (leaf) function_start/function_end pair that contains >=1
    circular_buffer_allocate between start and end. tensor identity = params['tensor_id'].
    """
    by_counter = {n["counter"]: n for n in nodes}

    def tid(node_counter):
        # Buffer ADDRESS is the reliable cross-op identity: the ttnn API layer re-wraps tensors
        # between device ops so tensor_id changes (producer out=21, consumer in=22) while the address
        # is preserved. Fall back to tensor_id for synthetic graphs with no address (self-test).
        n = by_counter.get(node_counter)
        if not (n and n["node_type"] == "tensor"):
            return None
        p = n.get("params") or {}
        return p.get("address") or p.get("tensor_id")

    # Pair function_start with function_end via a stack (counter order).
    stack, pairs = [], []
    for n in nodes:
        if n["node_type"] == "function_start":
            stack.append(n)
        elif n["node_type"] == "function_end":
            if stack:
                pairs.append((stack.pop(), n))

    # A device operation is the function node carrying tensor I/O: non-empty input_tensors (the ttnn
    # API wrapper above it has none, and helpers like create_device_tensor have none). This fires per
    # invocation regardless of program-cache reuse -- unlike circular_buffer_allocate, which a
    # cache-reused identical op (e.g. the 2nd/3rd of q/k/v) does NOT re-emit.
    ops = []
    for fs, fe in pairs:
        if not fs.get("input_tensors"):
            continue
        in_ids = {tid(c) for c in fs.get("input_tensors", [])}
        out_ids = {tid(c) for c in fe.get("connections", [])}
        ops.append(
            {
                "counter": fs["counter"],
                "name": (fs.get("params") or {}).get("name", "?"),
                "in_ids": {x for x in in_ids if x is not None},
                "out_ids": {x for x in out_ids if x is not None},
            }
        )
    ops.sort(key=lambda o: o["counter"])
    return ops


def analyze(ops):
    n = len(ops)

    def producer_before(addr, i):  # most-recent prior op that wrote addr (reuse-safe RAW source)
        p = None
        for k in range(i):
            if addr in ops[k]["out_ids"]:
                p = k
        return p

    # adjacent RAW: does the immediate next op read anything this op wrote
    adj_raw = sum(1 for i in range(n - 1) if ops[i]["out_ids"] & ops[i + 1]["in_ids"])
    # UNRESOLVED boundaries: op_n records NO output (in-place ops -- RoPE rotates Q/K in place,
    # PagedUpdateCache writes the cache in place -- ttnn graph capture stores no output tensor for
    # them). We CANNOT confirm op_{n+1} is independent, so these are NOT counted as headroom
    # (conservative: an op whose output isn't recorded is treated as a possible producer).
    adj_unresolved = sum(1 for i in range(n - 1) if not ops[i]["out_ids"])
    adj_nonraw = (n - 1) - adj_raw
    adj_reorderable = adj_nonraw - adj_unresolved  # non-RAW AND op_n has a recorded output

    # distance to first consumer of each op's output (stop if the buffer is overwritten before any
    # read -> that write is dead / WAW, not a RAW producer)
    dists = []
    for i, op in enumerate(ops):
        best = None
        for a in op["out_ids"]:
            for j in range(i + 1, n):
                if a in ops[j]["out_ids"] and a not in ops[j]["in_ids"]:
                    break  # overwritten before read
                if a in ops[j]["in_ids"]:
                    d = j - i
                    best = d if best is None else min(best, d)
                    break
        if best is not None:
            dists.append(best)

    # critical path: longest chain via most-recent-producer edges (ops in program/topo order)
    depth = [1] * n
    for i, op in enumerate(ops):
        for a in op["in_ids"]:
            p = producer_before(a, i)
            if p is not None:
                depth[i] = max(depth[i], depth[p] + 1)
    return {
        "n_ops": n,
        "adj_boundaries": n - 1,
        "adj_raw": adj_raw,
        "adj_nonraw": adj_nonraw,
        "adj_nonraw_pct": 100.0 * adj_nonraw / (n - 1) if n > 1 else 0.0,
        "adj_reorderable": adj_reorderable,
        "adj_reorderable_pct": 100.0 * adj_reorderable / (n - 1) if n > 1 else 0.0,
        "adj_unresolved": adj_unresolved,
        "dist_hist": {d: dists.count(d) for d in sorted(set(dists))},
        "dist_gt1_pct": 100.0 * sum(1 for d in dists if d > 1) / len(dists) if dists else 0.0,
        "critical_path": max(depth) if depth else 0,
    }


def analyze_waw(ops):
    """Write-after-write between device programs. At buffer granularity a shared output address is
    EITHER a true WAW (same live buffer written twice: in-place / accumulate / KV-cache slice) OR
    allocator ADDRESS-REUSE (op A's output freed after its consumer, op B's fresh output recycles the
    address) OR in-place READ-MODIFY-WRITE (op B reads addr a and writes it back -- really a RAW, not a
    WAW). Only the first (a genuine overwrite of still-live data with no reader) is a barrier hazard.
    Split: if addr a is read BETWEEN the two writes (consumed then recycled) OR read BY the second
    writer itself (in-place RMW), it's NOT a true WAW. CAVEAT: buffer granularity can't see sub-tensor
    regions, so genuine PARTIAL/slice updates to disjoint regions of one buffer (e.g. KV cache) would
    also register here -- flagged, not resolved, by this pass."""
    n = len(ops)
    writers = defaultdict(list)
    for i, op in enumerate(ops):
        for a in op["out_ids"]:
            writers[a].append(i)
    collisions = {a: w for a, w in writers.items() if len(w) > 1}
    adj_waw = sum(1 for i in range(n - 1) if ops[i]["out_ids"] & ops[i + 1]["out_ids"])
    reuse = overwrite = 0
    for a, w in collisions.items():
        for x in range(len(w) - 1):
            i, j = w[x], w[x + 1]
            # consumed (allocator-reuse) if read between the writes; RMW (RAW, not WAW) if op j reads a
            if a in ops[j]["in_ids"] or any(a in ops[k]["in_ids"] for k in range(i + 1, j)):
                reuse += 1
            else:
                overwrite += 1
    return {
        "collision_addrs": len(collisions),
        "waw_pairs": reuse + overwrite,
        "alloc_reuse": reuse,
        "true_overwrite": overwrite,
        "adj_waw": adj_waw,
    }


def report(ops, m, title):
    print(f"\n=== {title} ===")
    print(f"device programs: {m['n_ops']}   adjacency boundaries: {m['adj_boundaries']}")
    print(
        f"adjacent RAW: {m['adj_raw']}   NON-RAW: {m['adj_nonraw']} ({m['adj_nonraw_pct']:.0f}%) = "
        f"{m['adj_reorderable']} confirmed-reorderable ({m['adj_reorderable_pct']:.0f}%) "
        f"+ {m['adj_unresolved']} UNRESOLVED (op_n output not recorded -- in-place)"
    )
    print(f"distance-to-first-consumer histogram (ops-until-first-reader): {m['dist_hist']}")
    print(f"  outputs whose first reader is >1 op away: {m['dist_gt1_pct']:.0f}%")
    print(
        f"critical path: {m['critical_path']} of {m['n_ops']} ops  "
        f"(=> up to {m['n_ops'] - m['critical_path']} ops of slack/independence)"
    )
    w = analyze_waw(ops)
    print(
        f"WAW: {w['collision_addrs']} shared output addr(s), {w['waw_pairs']} write-pairs "
        f"-> {w['alloc_reuse']} allocator-reuse (NOT a hazard), {w['true_overwrite']} candidate true-WAW; "
        f"adjacent WAW {w['adj_waw']}"
    )


def _barrier_ops(nodes):
    """Parse to (counter, name, ins{addr:(lay,buf,grid)}, outs{...}) -- shared by the locality classifiers."""

    def mc(node):
        s = (node.get("params") or {}).get("memory_config") if node else None
        if not s:
            return (None, None, None)
        lay = re.search(r"memory_layout=TensorMemoryLayout::(\w+)", s)
        buf = re.search(r"buffer_type=BufferType::(\w+)", s)
        grid = re.search(r"grid=(\[.*?\])\s*,\s*shape", s)
        return (lay.group(1) if lay else None, buf.group(1) if buf else None, grid.group(1) if grid else None)

    byc = {n["counter"]: n for n in nodes}

    def addr(c):
        n = byc.get(c)
        p = (n.get("params") or {}) if n else {}
        return (p.get("address") or p.get("tensor_id")) if n and n["node_type"] == "tensor" else None

    stack, pairs = [], []
    for n in nodes:
        if n["node_type"] == "function_start":
            stack.append(n)
        elif n["node_type"] == "function_end" and stack:
            pairs.append((stack.pop(), n))
    ops = []
    for fs, fe in pairs:
        if not fs.get("input_tensors"):
            continue
        ins = {addr(c): mc(byc.get(c)) for c in fs.get("input_tensors", []) if addr(c) is not None}
        outs = {addr(c): mc(byc.get(c)) for c in fe.get("connections", []) if addr(c) is not None}
        ops.append((fs["counter"], (fs.get("params") or {}).get("name", ""), ins, outs))
    ops.sort()
    return ops


# --- barrier-strength classifier (signatures validated against kernel source, 2026-07-21) ---
# Two kinds of downgradable boundary (Paul's taxonomy), keyed on the CONSUMER'S READ of the producer's
# output -- NOT the op's write-side scatter (a scatter-on-write op like ShardedToInterleaved still READS
# local, so its input boundary is NONE):
#   NONE  (Type 1, no barrier)   : producer writes its shard to local L1, consumer reads its own local L1.
#                                  Same core's SRAM, ordered by that core's kernel sequence -> nothing to sync.
#   LOCAL (Type 2, write-drain)  : DATA goes through DRAM (write must commit before read-back) BUT no core
#                                  reads another core's just-written shard -> core c drains only its OWN
#                                  DRAM writes. Requires DRAM-SHARDED (contiguous per core), NOT interleaved
#                                  (round-robin across banks = cross). Partition-preservation needs the
#                                  work-split to confirm -> reported as a Type-2 CANDIDATE.
#   GLOBAL                       : consumer reads other cores' data (gather/mcast/regrid/reduction/interleaved).
#
# Consumer reads only its own L1 shard (VERIFIED from source): binary_ng CB aliased to shard buffer + reader
# no-op under #if SRC_SHARDED; ShardedToInterleaved reads local shard via globally-allocated input CB
# (scatters only on write). UnaryNg follows the binary_ng sharded path.
_LOCAL_READ_OPS = (
    "binaryng",
    "binary_ng",
    "unaryng",
    "unary_ng",
    "eltwise",
    "shardedtointerleaved",
    "sharded_to_interleaved",
)
# Consumer reads OTHER cores' data (VERIFIED: reshard reads remote L1 via runtime (noc_x,noc_y); matmul mcasts
# in0; InterleavedToSharded reads interleaved DRAM round-robin across banks). concat_heads gathers heads.
_CROSS_READ_OPS = (
    "reshard",
    "halo",
    "gather",
    "all_gather",
    "allgather",
    "concat_heads",
    "concatheads",
    "reduce_scatter",
    "reducescatter",
    "all_to_all",
    "alltoall",
    "matmul",
    "sdpa",
    "scaled_dot_product",
    "interleavedtosharded",
    "interleaved_to_sharded",
)
_NORM_OPS = ("layernorm", "groupnorm", "rmsnorm", "softmax")  # cross when the reduced dim is width-sharded
_VIEW_OPS = ("reshape", "reshapeview", "reshape_view")  # view/metadata; may be a no-op -> undecidable


def classify_barriers(nodes):
    """Classify each adjacent RAW boundary by barrier strength: NONE (Type 1, no barrier), LOCAL (Type 2,
    per-core write-drain), GLOBAL (cross-core), or UNDECIDABLE. Keyed on the CONSUMER'S READ of the shared
    buffer (see the taxonomy comment above). Returns (rows, summary), rows = [(prod_idx, prod, cons, verdict, reason)].
    """
    ops = _barrier_ops(nodes)
    rows, summary = [], {"NONE": 0, "NONE_no_noc": 0, "LOCAL": 0, "GLOBAL": 0, "UNDECIDABLE": 0}
    for i in range(len(ops) - 1):
        shared = set(ops[i][3]) & set(ops[i + 1][2])
        if not shared:
            continue
        a = next(iter(shared))
        cons = ops[i + 1][1].lower()
        cn = ops[i + 1][1].split("::")[-1]
        lay, buf, cons_grid = ops[i + 1][2][a]
        prod_grid = ops[i][3][a][2]
        sharded = lay and lay.endswith("SHARDED")
        if any(k in cons for k in _VIEW_OPS):
            v, r = "UNDECIDABLE", f"{cn} is a view (may be a no-op; read pattern not modeled)"
        elif lay == "INTERLEAVED":
            v, r = "GLOBAL", "reads interleaved (round-robin across banks/cores)"
        elif prod_grid and cons_grid and prod_grid != cons_grid:
            v, r = "GLOBAL", f"regrid {prod_grid}->{cons_grid}"
        elif any(k in cons for k in _CROSS_READ_OPS):
            v, r = "GLOBAL", f"{cn} reads other cores (gather/mcast)"
        elif any(k in cons for k in _NORM_OPS) and lay == "WIDTH_SHARDED":
            v, r = "GLOBAL", f"{cn} reduces across width-sharded dim"
        elif buf == "DRAM" and sharded:
            v, r = "LOCAL", f"DRAM-sharded same grid -> Type-2 candidate (needs work-split to confirm partition)"
        elif buf == "L1" and sharded and any(k in cons for k in _LOCAL_READ_OPS):
            # sub-label: does the consumer op issue ANY noc_async_read? reader is a no-op only when EVERY
            # operand is sharded-L1 (CB aliased to the shard buffer); an interleaved/DRAM operand forces a
            # NoC read. All-sharded-L1 inputs -> data already resident, nothing to fetch (strongest Type-1).
            ins = list(ops[i + 1][2].values())
            no_noc = bool(ins) and all(b == "L1" and l and l.endswith("SHARDED") for (l, b, g) in ins)
            v = "NONE"
            r = f"{cn} reads only its own L1 shard ({lay})" + (
                " [zero NoC reads: all inputs sharded-L1, already resident]"
                if no_noc
                else " [local shared read; op has other non-sharded inputs -> some NoC read]"
            )
            summary["NONE_no_noc"] += no_noc
        else:
            v, r = "UNDECIDABLE", f"{cn} read pattern not verified"
        summary[v] += 1
        rows.append((ops[i][0], ops[i][1].split("::")[-1], cn, v, r))
    return rows, summary


def shard_locality(nodes):
    """Bound how many RAW boundaries are cross-core, from tensor shard specs (memory_config).

    IMPORTANT LIMITATION: memory_config gives where a shard RESIDES (its grid), NOT which core READS
    it. The two differ for gather ops -- e.g. Halo (conv neighbor-exchange) reads border rows from
    ADJACENT cores while keeping the same residence grid. So a preserved residence grid does NOT prove
    same-core access; the true access pattern lives in the kernel's NoC addressing / runtime args,
    which the capture does not expose at the tensor level. Therefore:
      - CROSS-CORE is RELIABLE where we can see it: DRAM (no worker-core affinity), a residence-grid
        CHANGE (reshard/regrid), or a known gather/scatter op (Halo, all_gather, concat_heads, ...).
        Their sum is a LOWER bound on cross-core.
      - 'residence_preserved_ambiguous' is everything else: an UPPER bound on same-core -- it may still
        be cross-core (e.g. a width-sharded matmul/norm that reduces across cores). NOT confirmed local.
    Returns counts dict."""

    def mc(node):
        s = (node.get("params") or {}).get("memory_config") if node else None
        if not s:
            return (None, None, None)
        lay = re.search(r"memory_layout=TensorMemoryLayout::(\w+)", s)
        buf = re.search(r"buffer_type=BufferType::(\w+)", s)
        grid = re.search(r"grid=(\[.*?\])\s*,\s*shape", s)
        return (lay.group(1) if lay else None, buf.group(1) if buf else None, grid.group(1) if grid else None)

    byc = {n["counter"]: n for n in nodes}

    def addr(c):
        n = byc.get(c)
        p = (n.get("params") or {}) if n else {}
        return (p.get("address") or p.get("tensor_id")) if n and n["node_type"] == "tensor" else None

    stack, pairs = [], []
    for n in nodes:
        if n["node_type"] == "function_start":
            stack.append(n)
        elif n["node_type"] == "function_end" and stack:
            pairs.append((stack.pop(), n))
    ops = []
    for fs, fe in pairs:
        if not fs.get("input_tensors"):
            continue
        ins = {addr(c): mc(byc.get(c)) for c in fs.get("input_tensors", []) if addr(c) is not None}
        outs = {addr(c): mc(byc.get(c)) for c in fe.get("connections", []) if addr(c) is not None}
        ops.append((fs["counter"], (fs.get("params") or {}).get("name", ""), ins, outs))
    ops.sort()
    # ops that read cross-core even when the residence grid is preserved (gather/scatter)
    GATHER = (
        "halo",
        "gather",
        "all_gather",
        "allgather",
        "concat_heads",
        "concatheads",
        "create_qkv",
        "createqkv",
        "reduce_scatter",
        "reducescatter",
        "all_to_all",
        "alltoall",
    )
    cats = {"residence_preserved_ambiguous": 0, "cross_DRAM": 0, "cross_regrid": 0, "cross_known_gather": 0}
    for i in range(len(ops) - 1):
        shared = set(ops[i][3]) & set(ops[i + 1][2])  # producer outs ∩ consumer ins
        if not shared:
            continue
        a = next(iter(shared))
        name = ops[i + 1][1].lower()
        lay, buf, grid = ops[i + 1][2][a]  # consumed buffer's layout/buffer/grid
        out_grids = {g for (_, _, g) in ops[i + 1][3].values() if g}
        if any(k in name for k in GATHER):
            cats["cross_known_gather"] += 1
        elif buf == "DRAM":
            cats["cross_DRAM"] += 1
        elif buf == "L1" and lay and lay.endswith("SHARDED") and grid and out_grids == {grid}:
            cats["residence_preserved_ambiguous"] += 1
        else:
            cats["cross_regrid"] += 1
    return cats


# ---- self-test: synthetic Llama-style decoder layer with KNOWN dependencies ----
def _synthetic_graph():
    """Hand-built graph in the ttnn schema. Decoder layer at device-program granularity:
    0 rmsnorm(x)->h1  1 qkv(h1)->qkv  2 attn(qkv)->attn  3 o_proj(attn)->o  4 add(x,o)->r1
    5 rmsnorm(r1)->h2  6 gate(h2)->g  7 up(h2)->u  8 silu_mul(g,u)->gu  9 down(gu)->d  10 add(r1,d)->r2
    KEY: op7(up) reads h2 (op5's out), NOT op6's out -> op6->op7 is the one NON-RAW adjacency.
    """
    T = {}  # tensor_id -> node counter
    nodes = [{"counter": 0, "node_type": "capture_start", "params": {}, "connections": [], "input_tensors": []}]
    c = [1]

    def tensor(tid):
        node = {"counter": c[0], "node_type": "tensor", "params": {"tensor_id": str(tid)}, "connections": []}
        nodes.append(node)
        T[tid] = c[0]
        c[0] += 1
        return T[tid]

    tensor(0)  # x (external embedding input)

    def op(name, in_tids, out_tid):
        fs = {
            "counter": c[0],
            "node_type": "function_start",
            "params": {"name": name},
            "input_tensors": [T[t] for t in in_tids],
            "connections": [],
        }
        nodes.append(fs)
        c[0] += 1
        nodes.append({"counter": c[0], "node_type": "circular_buffer_allocate", "params": {}, "connections": []})
        c[0] += 1
        ot = tensor(out_tid)
        fe = {"counter": c[0], "node_type": "function_end", "params": {"name": name}, "connections": [ot]}
        nodes.append(fe)
        c[0] += 1

    op("rmsnorm", [0], 1)  # h1
    op("matmul_qkv", [1], 2)  # qkv
    op("attention", [2], 3)  # attn
    op("matmul_o", [3], 4)  # o
    op("add_residual", [0, 4], 5)  # r1 = x + o
    op("rmsnorm", [5], 6)  # h2
    op("matmul_gate", [6], 7)  # g
    op("matmul_up", [6], 8)  # u   <-- reads h2 (op5), independent of op6 => NON-RAW adjacency
    op("silu_mul", [7, 8], 9)  # gu
    op("matmul_down", [9], 10)  # d
    op("add_residual", [5, 10], 11)  # r2 = r1 + d
    return nodes


def infer_attention_edges(ops):
    """create-heads -> SDPA/softmax edges the capture drops. NLPCreateHeads emits NO output tensor, so
    the SDPA that consumes its Q/K/V heads shows up producerless (a false root). Attention is emitted as
    [q/k/v proj ...][create-heads ...] SDPA, so each attention core (SDPA or explicit softmax) depends on
    the create-heads ops since the previous core. Returns [(create_heads_idx, core_idx)] pairs.
    """
    pairs, pending = [], []
    for i, o in enumerate(ops):
        nm = o["name"].lower()
        if "create" in nm and "head" in nm:
            pending.append(i)
        elif "sdpa" in nm or "scaled_dot_product" in nm or "softmax" in nm:
            pairs.extend((p, i) for p in pending)
            pending = []
    return pairs


def apply_resolutions(ops, resolve_inplace=False, add_edges=(), infer_attention=None):
    """Reconnect dependency edges the buffer-address capture cannot see (opt-in, conservative).

    resolve_inplace: an op with NO recorded output is an in-place RMW (e.g. RotaryEmbedding rotates Q,
      PagedFusedUpdateCache writes the KV cache) -- it mutates its input buffer(s). Treat its inputs AS
      its outputs so a downstream consumer of those buffers depends on it (and it on the prior writer).
      Only ADDS edges, so it can only lengthen the critical path (never fabricate slack).
    infer_attention: also add create-heads->SDPA edges (see infer_attention_edges). Defaults to
      resolve_inplace, so --resolve-inplace stops drawing SDPAs as false roots.
    add_edges: [(producer_idx, consumer_idx)] semantic deps known real but invisible in the capture --
      e.g. a reshape VIEW whose output drops its buffer address (tensor-id fallback) so producer and
      consumer tids differ. Injected via a synthetic shared address; each is an explicit, auditable claim.
    """
    if infer_attention is None:
        infer_attention = resolve_inplace
    if resolve_inplace:
        for o in ops:
            if not o["out_ids"]:
                o["out_ids"] = set(o["in_ids"])
    edges = list(add_edges) + (infer_attention_edges(ops) if infer_attention else [])
    for p, c in edges:
        tag = f"__edge_{p}_{c}"
        ops[p]["out_ids"].add(tag)
        ops[c]["in_ids"].add(tag)
    return ops


if __name__ == "__main__":
    argv = sys.argv[1:]
    resolve_inplace = "--resolve-inplace" in argv
    verbose = "--classify-verbose" in argv
    add_edges, files, i = [], [], 0
    while i < len(argv):
        if argv[i] == "--add-edge":
            i += 1
            p, c = argv[i].split(":")
            add_edges.append((int(p), int(c)))
        elif argv[i] not in ("--resolve-inplace", "--classify-verbose"):
            files.append(argv[i])
        i += 1
    if files and files[0] != "--self-test":
        nodes = json.load(open(files[0]))
        ops = parse_device_ops(nodes)
        apply_resolutions(ops, resolve_inplace, add_edges)
        tags = [t for t, on in (("resolve-inplace", resolve_inplace), (f"add-edge {add_edges}", bool(add_edges))) if on]
        report(ops, analyze(ops), f"capture: {files[0]}" + (f"  [{'; '.join(tags)}]" if tags else ""))
        s = shard_locality(nodes)
        tot = sum(s.values())
        if tot:
            xcore = s["cross_DRAM"] + s["cross_regrid"] + s["cross_known_gather"]
            amb = s["residence_preserved_ambiguous"]
            print(
                f"RAW-boundary shard locality (residence != access -- see docstring):\n"
                f"  cross-core >= {xcore}/{tot} ({100*xcore/tot:.0f}%, RELIABLE floor: "
                f"{s['cross_DRAM']} DRAM + {s['cross_regrid']} regrid + {s['cross_known_gather']} known-gather)\n"
                f"  same-core <= {amb}/{tot} ({100*amb/tot:.0f}%, residence-preserved but NOT confirmed local "
                f"-- kernel read pattern not visible in the capture)"
            )
        rows, cls = classify_barriers(nodes)
        ctot = cls["NONE"] + cls["LOCAL"] + cls["GLOBAL"] + cls["UNDECIDABLE"]  # NONE_no_noc is a NONE subset
        if ctot:
            dg = cls["NONE"] + cls["LOCAL"]
            nn = cls["NONE_no_noc"]
            print(
                f"\nBarrier strength over adjacent-RAW boundaries (keyed on consumer read; signatures validated "
                f"vs kernel source):\n"
                f"  NONE        {cls['NONE']:4}/{ctot} ({100*cls['NONE']/ctot:.0f}%)  Type 1: local L1->L1, no barrier\n"
                f"     of which  {nn:4}      zero NoC reads (all inputs sharded-L1, already resident)\n"
                f"  LOCAL       {cls['LOCAL']:4}/{ctot} ({100*cls['LOCAL']/ctot:.0f}%)  Type 2: DRAM-sharded, per-core write-drain (candidate, needs work-split)\n"
                f"  GLOBAL      {cls['GLOBAL']:4}/{ctot} ({100*cls['GLOBAL']/ctot:.0f}%)  cross-core read\n"
                f"  UNDECIDABLE {cls['UNDECIDABLE']:4}/{ctot} ({100*cls['UNDECIDABLE']/ctot:.0f}%)\n"
                f"  => downgradable (NONE+LOCAL): {dg}/{ctot} ({100*dg/ctot:.0f}%)"
            )
            if verbose:
                for pidx, pn, cn, v, r in rows:
                    print(f"    {pidx:5} {pn[:22]:22}->{cn[:22]:22} {v:11} {r}")
    else:
        ops = parse_device_ops(_synthetic_graph())
        m = analyze(ops)
        report(ops, m, "SELF-TEST synthetic decoder layer (schema validation, NOT measured)")
        # assertions on the known structure
        assert m["n_ops"] == 11, m["n_ops"]
        assert m["adj_nonraw"] == 1, m["adj_nonraw"]  # only op6(gate)->op7(up)
        names = [o["name"] for o in ops]
        i = names.index("matmul_gate")
        assert not (ops[i]["out_ids"] & ops[i + 1]["in_ids"]), "gate->up should be non-RAW"
        print("\nSELF-TEST PASSED: analyzer recovers the known dependency structure.")

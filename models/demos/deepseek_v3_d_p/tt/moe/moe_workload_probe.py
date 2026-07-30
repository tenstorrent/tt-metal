# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Per-(chunk, layer, dispatch-column) MoE routing-workload probe.

Answers "which dispatch/combine column carries the most sends, and in which layer of
which chunk?" — the (layer, col) hot-spot table that `tests/perf/test_prefill_dispatch_combine.py`
hardcodes (`_REAL_INDICES_PICKS`), but measured live over a chunked-prefill run instead of
being derived offline from a single 25600-token capture.

Metric
------
On an 8x4 Galaxy with ``cluster_axis=0``, dispatch groups run along mesh **columns**: column
``c`` hosts routed experts ``[c * experts_per_col, (c + 1) * experts_per_col)`` where
``experts_per_col = num_routed_experts // num_dispatch_groups`` (Kimi K2.6: 384/4 = 96,
DeepSeek V3: 256/4 = 64). A token's top-k pick therefore lands on column ``e // experts_per_col``,
so a column's dispatch load is simply the number of picks falling in its expert range:

    in_col_picks[c] = count(e for e in all picks if e // experts_per_col == c)

Of those, picks whose expert lives on the *same* mesh row as the source token travel by NOC;
the rest go over fabric. Both are recorded (``in_col_picks`` and ``fabric_picks``).

Combine mirrors dispatch: every dispatched pick produces exactly one returning contribution on
the same column, so these counts characterise the combine column load too. Combine's own
``forward`` never receives ``indices`` (only the metadata buffer), which is why the probe hooks
dispatch and reports both sides from one readback.

Cost / correctness notes
------------------------
Tokens are sharded across mesh **rows** and replicated across **columns**
(``ShardTensor2dMesh(dims=(0, None))``), so one column's 8 devices already hold every token of
the chunk. The probe reads back only those 8 index tensors per call, not all 32.

This forces a host sync per dispatch call, so it perturbs timing — run it with
``num_iters=1``. It is off unless ``TT_DS_MOE_WORKLOAD_PROBE=1``.

Usage
-----
    export TT_DS_MOE_WORKLOAD_PROBE=1
    export TT_DS_MOE_WORKLOAD_PROBE_OUT=/path/prefix   # optional, defaults under /tmp

The chunked-prefill driver calls :func:`set_context` around each chunk and :func:`dump` at the
end; ``TtDispatchModule.forward`` calls :func:`record_dispatch`.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import torch
from loguru import logger

import ttnn

ENV_ENABLE = "TT_DS_MOE_WORKLOAD_PROBE"
ENV_OUT = "TT_DS_MOE_WORKLOAD_PROBE_OUT"
ENV_CAPTURE = "TT_DS_MOE_ROUTING_CAPTURE"

# (iter, chunk, layer, col) -> {"in_col": int, "fabric": int, "per_row": [int, ...]}
_rows: dict[tuple[int, int, int, int], dict] = {}
# layer -> {chunk: LongTensor[chunk_tokens, top_k]} of global expert ids, iteration 0 only.
_captured: dict[int, dict[int, "torch.Tensor"]] = {}
_ctx = {"iter": -1, "chunk": -1, "tag": "unknown"}
_warned = False


def enabled() -> bool:
    return os.getenv(ENV_ENABLE) == "1" or capture_enabled()


def capture_enabled() -> bool:
    """Dump the raw per-layer routing to an `expert_routing.safetensors` for replay.

    Produces the exact format `load_captured_routing` consumes (keys ``expert_ids_layer_{N}``,
    Galaxy-global expert ids), so a chunked-prefill run can supply captured routing for models
    or datasets that have no ``routing/`` stream in their golden trace — notably the DeepSeek
    code_debug trace, whose own tensor_mapping.json records "NO routing/ stream (expert
    ids/weights not saved)".
    """
    return os.getenv(ENV_CAPTURE) == "1"


def set_context(iter_idx: int, chunk_idx: int, tag: str | None = None) -> None:
    """Mark which (iteration, chunk) the following dispatch calls belong to."""
    _ctx["iter"] = int(iter_idx)
    _ctx["chunk"] = int(chunk_idx)
    if tag is not None:
        _ctx["tag"] = tag


def set_tag(tag: str) -> None:
    _ctx["tag"] = tag


def record_dispatch(
    indices: ttnn.Tensor,
    layer_idx: int,
    num_routed_experts: int,
    mesh_shape: tuple[int, int],
    num_dispatch_groups: int = 4,
) -> None:
    """Histogram this dispatch call's top-k picks by owning column.

    ``indices`` is the per-device top-k tensor (1, seq_len_per_chip, num_experts_per_tok).
    Only column 0's devices are read back — columns are token-replicas of each other.
    """
    global _warned
    if not enabled():
        return
    try:
        rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
        experts_per_col = num_routed_experts // num_dispatch_groups
        if experts_per_col <= 0:
            return
        experts_per_chip = experts_per_col // rows

        dev_tensors = ttnn.get_device_tensors(indices)
        # Column 0 of every mesh row: linear device index = row * cols.
        col0 = [dev_tensors[r * cols] for r in range(rows) if r * cols < len(dev_tensors)]
        if not col0:
            col0 = dev_tensors

        # per_col[c] = total picks owned by column c; per_col_fabric[c] excludes same-row (NOC) picks.
        per_col = [0] * num_dispatch_groups
        per_col_fabric = [0] * num_dispatch_groups
        # per_col_row[c][r] = picks landing on the chip at (row r, col c) — intra-column skew.
        per_col_row = [[0] * rows for _ in range(num_dispatch_groups)]

        # Raw per-row picks, kept 2-D ([tokens_on_this_row, top_k]) for the routing capture. The
        # loader reads a flat tensor as view(dispatch_group_size, seq_len_per_chip, top_k), i.e.
        # chip-major, so stacking rows in mesh-row order is already the layout it expects.
        row_picks_2d: list[torch.Tensor] = []

        for src_row, t in enumerate(col0):
            raw = ttnn.to_torch(t).to(torch.int64)
            top_k = int(raw.shape[-1])
            if capture_enabled() and _ctx["iter"] == 0:
                row_picks_2d.append(raw.reshape(-1, top_k).clone())
            picks = raw.reshape(-1)
            picks = picks[(picks >= 0) & (picks < num_routed_experts)]
            if picks.numel() == 0:
                continue
            owner_col = picks // experts_per_col
            owner_row = (picks - owner_col * experts_per_col) // max(1, experts_per_chip)
            for c in range(num_dispatch_groups):
                in_c = owner_col == c
                n_in = int(in_c.sum())
                if n_in == 0:
                    continue
                per_col[c] += n_in
                # Same source row -> local NOC write, not a fabric send.
                per_col_fabric[c] += int((in_c & (owner_row != src_row)).sum())
                rows_hist = torch.bincount(owner_row[in_c], minlength=rows)
                for r in range(rows):
                    per_col_row[c][r] += int(rows_hist[r])

        for c in range(num_dispatch_groups):
            key = (_ctx["iter"], _ctx["chunk"], int(layer_idx), c)
            slot = _rows.setdefault(key, {"in_col": 0, "fabric": 0, "per_row": [0] * rows})
            slot["in_col"] += per_col[c]
            slot["fabric"] += per_col_fabric[c]
            for r in range(rows):
                slot["per_row"][r] += per_col_row[c][r]

        # [rows, tokens_per_row_per_chunk, top_k]; the builder concatenates chunks along the
        # token axis per row, so the final flat tensor stays chip-major.
        if row_picks_2d and len(row_picks_2d) == rows:
            _captured.setdefault(int(layer_idx), {})[int(_ctx["chunk"])] = torch.stack(row_picks_2d, dim=0)
    except Exception as e:  # instrumentation must never break a run
        if not _warned:
            _warned = True
            logger.warning(f"[moe-workload-probe] disabled after error: {e!r}")


def _out_prefix() -> str:
    base = os.getenv(ENV_OUT) or f"/tmp/moe_workload_{_ctx['tag']}"
    return base


def write_captured_routing() -> str | None:
    """Write the captured routing as ``<prefix>_expert_routing.safetensors``.

    Layout matches what :func:`load_captured_routing` expects: one flat int32 tensor per key
    ``expert_ids_layer_{N}``, read back as ``view(dispatch_group_size, seq_len_per_chip, top_k)``.
    Chunks are concatenated along each chip's token axis, so ``seq_len_per_chip`` ends up
    ``n_chunks * chunk_tokens_per_chip`` (11 chunks x 640 = 7040 for an 8x4 mesh at CHUNK=5120).
    Slice it down to a worker's expected token count with
    ``scripts/build_captured_routing.py --tokens N``.
    """
    if not capture_enabled() or not _captured:
        return None
    try:
        from safetensors.torch import save_file
    except ImportError:
        logger.warning("[moe-workload-probe] safetensors not importable; routing capture not written")
        return None
    out = f"{_out_prefix()}_expert_routing.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    for layer, per_chunk in sorted(_captured.items()):
        # [rows, tokens_per_chunk, top_k] per chunk -> concat on the token axis -> flat, chip-major.
        stacked = torch.cat([per_chunk[c] for c in sorted(per_chunk)], dim=1)
        tensors[f"expert_ids_layer_{layer}"] = stacked.reshape(-1).to(torch.int32).contiguous()
    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        save_file(tensors, out)
    except Exception as e:
        logger.warning(f"[moe-workload-probe] routing capture write failed: {e!r}")
        return None
    any_key = next(iter(tensors))
    logger.success(
        f"[moe-workload-probe] wrote routing capture: {len(tensors)} layers, "
        f"{tensors[any_key].numel()} ids/layer -> {out}"
    )
    return out


def dump(top_n: int = 4) -> str | None:
    """Write the full (iter, chunk, layer, col) table to CSV and log the top-N hot columns."""
    if not enabled() or not _rows:
        return None
    prefix = _out_prefix()
    csv_path = f"{prefix}.csv"
    try:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        n_rows = len(next(iter(_rows.values()))["per_row"])
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["iter", "chunk", "layer", "col", "in_col_picks", "fabric_picks", "col_share_pct"]
                + [f"row{r}_picks" for r in range(n_rows)]
            )
            for it, ch, layer, col in sorted(_rows):
                slot = _rows[(it, ch, layer, col)]
                total = sum(_rows[(it, ch, layer, c)]["in_col"] for c in range(4) if (it, ch, layer, c) in _rows)
                share = (100.0 * slot["in_col"] / total) if total else 0.0
                w.writerow([it, ch, layer, col, slot["in_col"], slot["fabric"], f"{share:.2f}"] + slot["per_row"])
        logger.success(f"[moe-workload-probe] wrote {len(_rows)} rows -> {csv_path}")
    except Exception as e:
        logger.warning(f"[moe-workload-probe] CSV write failed: {e!r}")

    # Hottest (layer, col) pairs by column share, aggregated over chunks — the
    # _REAL_INDICES_PICKS-style table.
    by_layer_col: dict[tuple[int, int], int] = defaultdict(int)
    by_layer: dict[int, int] = defaultdict(int)
    for (it, ch, layer, col), slot in _rows.items():
        by_layer_col[(layer, col)] += slot["in_col"]
        by_layer[layer] += slot["in_col"]
    ranked = sorted(by_layer_col.items(), key=lambda kv: -(kv[1] / max(1, by_layer[kv[0][0]])))
    logger.info(f"[moe-workload-probe] top {top_n} hottest (layer, col) by in-col share:")
    for (layer, col), picks in ranked[:top_n]:
        share = 100.0 * picks / max(1, by_layer[layer])
        logger.info(f"[moe-workload-probe]   (layer={layer:3d}, col={col}) share={share:5.2f}%  picks={picks}")

    # Per-chunk hottest column, so chunk-to-chunk drift is visible.
    per_chunk: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for (it, ch, layer, col), slot in _rows.items():
        per_chunk[ch][col] += slot["in_col"]
    logger.info("[moe-workload-probe] per-chunk column totals (in-col picks):")
    for ch in sorted(per_chunk):
        counts = [per_chunk[ch].get(c, 0) for c in range(4)]
        hottest = max(range(4), key=lambda c: counts[c])
        spread = (max(counts) / min(counts)) if min(counts) else float("inf")
        logger.info(f"[moe-workload-probe]   chunk {ch:2d}: {counts}  hottest=col{hottest}  max/min={spread:.2f}x")
    return csv_path


def reset() -> None:
    _rows.clear()

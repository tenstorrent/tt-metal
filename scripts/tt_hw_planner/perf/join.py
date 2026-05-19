# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Join Tracy's ops CSV with model_tracer's master JSON.

Tracy gives us a row per device-side op call with timing + utilization +
kernel hashes. model_tracer gives us a structured `arguments` dict per op
call. Together they let every Tracy row carry the literal Python kwargs
the model issued, enabling configuration-level clustering.

Join key: (operation_name, occurrence_index_within_name).

This is robust to absolute-order drift between Tracy (execution-end order)
and tracer (issuance order): the 47th matmul in Tracy maps to the 47th
matmul in tracer regardless of any non-matmul ops shuffled between them.

The result is a list of `JoinedRow` records. We don't depend on pandas
to keep the module importable in minimal environments; the chart layer
converts to a DataFrame when it needs one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# Argument canonicalization
# ---------------------------------------------------------------------------


def _canonicalize(value: Any) -> Any:
    """Sort keys recursively + coerce non-JSON types so identical configs
    hash to the same `args_hash` regardless of how Python serialized them."""
    if isinstance(value, Mapping):
        return {k: _canonicalize(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return value
    return repr(value)


def args_hash(arguments: Mapping[str, Any]) -> str:
    """Deterministic hash of an op's normalized argument dict (sha256, 12hex)."""
    canon = _canonicalize(arguments)
    blob = json.dumps(canon, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Tracy CSV parsing
# ---------------------------------------------------------------------------


@dataclass
class TracyRow:
    """A subset of the Tracy `ops_perf_results_*.csv` columns we use."""

    op_code: str
    op_type: str
    global_call_count: int
    device_id: Optional[int]
    attributes: str
    math_fidelity: str
    core_count: Optional[int]
    host_duration_ns: Optional[float]
    device_fw_duration_ns: Optional[float]
    device_kernel_duration_ns: Optional[float]
    op_to_op_latency_ns: Optional[float]
    brisc_ns: Optional[float]
    ncrisc_ns: Optional[float]
    trisc0_ns: Optional[float]
    trisc1_ns: Optional[float]
    trisc2_ns: Optional[float]
    erisc_ns: Optional[float]
    compute_cb_wait_front_ns: Optional[float]
    inputs_summary: str
    outputs_summary: str
    metal_trace_id: Optional[int]
    compute_kernel_source: str
    compute_kernel_hash: str
    dm_kernel_source: str
    dm_kernel_hash: str
    program_hash: str
    program_cache_hit: Optional[bool]
    pm_ideal_ns: Optional[float]
    pm_compute_ns: Optional[float]
    pm_bandwidth_ns: Optional[float]
    pm_req_i_bw: Optional[float]
    pm_req_o_bw: Optional[float]
    pm_fpu_util_pct: Optional[float]
    noc_util_pct: Optional[float]
    multicast_noc_util_pct: Optional[float]
    dram_bw_util_pct: Optional[float]
    eth_bw_util_pct: Optional[float]
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, str] = field(default_factory=dict)
    is_signpost: bool = False
    signpost_header: str = ""
    signpost_message: str = ""


_SIGNPOST_RE = re.compile(r"`TT_SIGNPOST:\s*(?P<header>[^\n`]+)(?:\n(?P<message>[^`]*))?`")


def _parse_signpost(attributes: str) -> Tuple[bool, str, str]:
    if not attributes:
        return False, "", ""
    m = _SIGNPOST_RE.search(attributes)
    if m is None:
        return False, "", ""
    return True, m.group("header").strip(), (m.group("message") or "").strip()


def _maybe_float(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v.replace(";", ".") if isinstance(v, str) and v.count(";") == 1 and "." not in v else v)
    except (ValueError, TypeError):
        return None


def _maybe_int(v: Optional[str]) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _maybe_bool(v: Optional[str]) -> Optional[bool]:
    if v is None or v == "":
        return None
    s = v.strip().lower()
    if s in ("1", "true", "yes", "hit"):
        return True
    if s in ("0", "false", "no", "miss"):
        return False
    return None


_INPUT_COL_RE = re.compile(r"^INPUT_(?P<idx>\d+)_(?P<field>[A-Za-z]+(?:\s+[A-Za-z]+)*)$")
_OUTPUT_COL_RE = re.compile(r"^OUTPUT_(?P<idx>\d+)_(?P<field>[A-Za-z]+(?:\s+[A-Za-z]+)*)$")


def _infer_role_from_op_code(op_code: str) -> str:
    """Best-effort semantic role from Tracy op code.

    Used only when true nn.Module attribution is unavailable. The role
    suffix makes block-scope fallback useful for comparisons/suggestions:

      decoder.layers.00.matmul
      decoder.layers.00.sdpa
      decoder.layers.00.norm
      ...
    """
    c = (op_code or "").lower()
    if "sdpa" in c or "scaleddotproduct" in c:
        return "sdpa"
    if "qkv" in c or "createheads" in c:
        return "qkv_heads"
    if "concatheads" in c:
        return "concat_heads"
    if "rotary" in c:
        return "rotary"
    if "norm" in c:
        return "norm"
    if "matmul" in c:
        return "matmul"
    if "allgather" in c:
        return "all_gather"
    if "reducescatter" in c:
        return "reduce_scatter"
    if "typecast" in c:
        return "typecast"
    if "reshard" in c or "interleaved" in c or "sharded" in c:
        return "layout_move"
    if "binary" in c:
        return "binary"
    return "other"


def _collect_tensor_columns(row: Mapping[str, str], pattern: re.Pattern) -> List[Dict[str, Any]]:
    """Group INPUT_<idx>_<field> / OUTPUT_<idx>_<field> columns into per-tensor dicts."""
    grouped: Dict[int, Dict[str, Any]] = {}
    for col, val in row.items():
        if val in (None, ""):
            continue
        m = pattern.match(col)
        if m is None:
            continue
        idx = int(m.group("idx"))
        field_name = m.group("field").strip().upper()
        grouped.setdefault(idx, {})[field_name] = val
    return [grouped[k] for k in sorted(grouped.keys())]


def parse_tracy_csv(path: Path) -> List[TracyRow]:
    """Parse a Tracy `ops_perf_results_*.csv` file into TracyRow records.

    Signpost rows (OP TYPE == "signpost") are kept in the same stream and
    flagged with `is_signpost=True`; downstream code uses them to attribute
    surrounding op rows to a logical block.
    """
    rows: List[TracyRow] = []
    if not path.exists():
        return rows
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            attributes = (raw.get("ATTRIBUTES") or "").strip()
            is_sp, sp_header, sp_message = _parse_signpost(attributes)
            tr = TracyRow(
                op_code=(raw.get("OP CODE") or "").strip(),
                op_type=(raw.get("OP TYPE") or "").strip(),
                global_call_count=_maybe_int(raw.get("GLOBAL CALL COUNT")) or -1,
                device_id=_maybe_int(raw.get("DEVICE ID")),
                attributes=attributes,
                math_fidelity=(raw.get("MATH FIDELITY") or "").strip(),
                core_count=_maybe_int(raw.get("CORE COUNT")),
                host_duration_ns=_maybe_float(raw.get("HOST DURATION [ns]")),
                device_fw_duration_ns=_maybe_float(raw.get("DEVICE FW DURATION [ns]")),
                device_kernel_duration_ns=_maybe_float(raw.get("DEVICE KERNEL DURATION [ns]")),
                op_to_op_latency_ns=_maybe_float(raw.get("OP TO OP LATENCY [ns]")),
                brisc_ns=_maybe_float(raw.get("DEVICE BRISC KERNEL DURATION [ns]")),
                ncrisc_ns=_maybe_float(raw.get("DEVICE NCRISC KERNEL DURATION [ns]")),
                trisc0_ns=_maybe_float(raw.get("DEVICE TRISC0 KERNEL DURATION [ns]")),
                trisc1_ns=_maybe_float(raw.get("DEVICE TRISC1 KERNEL DURATION [ns]")),
                trisc2_ns=_maybe_float(raw.get("DEVICE TRISC2 KERNEL DURATION [ns]")),
                erisc_ns=_maybe_float(raw.get("DEVICE ERISC KERNEL DURATION [ns]")),
                compute_cb_wait_front_ns=_maybe_float(raw.get("DEVICE COMPUTE CB WAIT FRONT [ns]")),
                inputs_summary=(raw.get("INPUTS") or "").strip(),
                outputs_summary=(raw.get("OUTPUTS") or "").strip(),
                metal_trace_id=_maybe_int(raw.get("METAL TRACE ID")),
                compute_kernel_source=(raw.get("COMPUTE KERNEL SOURCE") or "").strip(),
                compute_kernel_hash=(raw.get("COMPUTE KERNEL HASH") or "").strip(),
                dm_kernel_source=(raw.get("DATA MOVEMENT KERNEL SOURCE") or "").strip(),
                dm_kernel_hash=(raw.get("DATA MOVEMENT KERNEL HASH") or "").strip(),
                program_hash=(raw.get("PROGRAM HASH") or "").strip(),
                program_cache_hit=_maybe_bool(raw.get("PROGRAM CACHE HIT")),
                pm_ideal_ns=_maybe_float(raw.get("PM IDEAL [ns]")),
                pm_compute_ns=_maybe_float(raw.get("PM COMPUTE [ns]")),
                pm_bandwidth_ns=_maybe_float(raw.get("PM BANDWIDTH [ns]")),
                pm_req_i_bw=_maybe_float(raw.get("PM REQ I BW")),
                pm_req_o_bw=_maybe_float(raw.get("PM REQ O BW")),
                pm_fpu_util_pct=_maybe_float(raw.get("PM FPU UTIL (%)")),
                noc_util_pct=_maybe_float(raw.get("NOC UTIL (%)")),
                multicast_noc_util_pct=_maybe_float(raw.get("MULTICAST NOC UTIL (%)")),
                dram_bw_util_pct=_maybe_float(raw.get("DRAM BW UTIL (%)")),
                eth_bw_util_pct=_maybe_float(raw.get("ETH BW UTIL (%)")),
                inputs=_collect_tensor_columns(raw, _INPUT_COL_RE),
                outputs=_collect_tensor_columns(raw, _OUTPUT_COL_RE),
                is_signpost=is_sp,
                signpost_header=sp_header,
                signpost_message=sp_message,
            )
            rows.append(tr)
    return rows


# ---------------------------------------------------------------------------
# model_tracer master parsing
# ---------------------------------------------------------------------------


@dataclass
class TracerCall:
    """One concrete call extracted from `ttnn_operations_master.json`.

    The master file groups calls by (operation, config) with an executions
    list. Each executions entry has a `counter` of how many times that
    config was invoked from a given source. We expand back to individual
    calls so we can pair them positionally with Tracy rows.
    """

    operation_name: str
    arguments: Dict[str, Any]
    arguments_hash: str
    sweep_source_hash: Optional[str]
    config_id: Optional[str]
    return_value: Optional[Any]


def parse_tracer_master(path: Path) -> List[TracerCall]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    expanded: List[TracerCall] = []
    operations = data.get("operations") or {}
    for op_name, op_block in operations.items():
        configs = op_block.get("configurations") or []
        for cfg in configs:
            arguments = cfg.get("arguments") or {}
            executions = cfg.get("executions") or []
            ah = args_hash(arguments)
            for ex in executions:
                count = int(ex.get("counter") or 1)
                for _ in range(count):
                    expanded.append(
                        TracerCall(
                            operation_name=op_name,
                            arguments=dict(arguments),
                            arguments_hash=ah,
                            sweep_source_hash=cfg.get("sweep_source_hash"),
                            config_id=cfg.get("config_id"),
                            return_value=cfg.get("return_value"),
                        )
                    )
    return expanded


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------


@dataclass
class JoinedRow:
    """Tracy + tracer + block attribution for one device op call."""

    # provenance
    run_id: str
    row_index: int
    global_call_count: int
    device_id: Optional[int]

    # logical block from signpost walker
    block_path: str

    # op identity
    op_code: str
    op_type: str
    tracer_op_name: Optional[str]

    # configuration
    args_hash: Optional[str]
    arguments: Dict[str, Any]
    math_fidelity: str
    inputs_summary: str
    outputs_summary: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]

    # kernels
    compute_kernel_source: str
    compute_kernel_hash: str
    dm_kernel_source: str
    dm_kernel_hash: str
    program_hash: str
    program_cache_hit: Optional[bool]
    core_count: Optional[int]

    # timing
    device_kernel_ns: Optional[float]
    device_fw_ns: Optional[float]
    op_to_op_latency_ns: Optional[float]
    brisc_ns: Optional[float]
    ncrisc_ns: Optional[float]
    trisc0_ns: Optional[float]
    trisc1_ns: Optional[float]
    trisc2_ns: Optional[float]
    erisc_ns: Optional[float]
    compute_cb_wait_front_ns: Optional[float]

    # roofline
    pm_ideal_ns: Optional[float]
    pm_compute_ns: Optional[float]
    pm_bandwidth_ns: Optional[float]
    pm_req_i_bw: Optional[float]
    pm_req_o_bw: Optional[float]
    pm_fpu_util_pct: Optional[float]
    noc_util_pct: Optional[float]
    multicast_noc_util_pct: Optional[float]
    dram_bw_util_pct: Optional[float]
    eth_bw_util_pct: Optional[float]

    # region label (filled in by regions.py)
    region: Optional[str] = None
    region_reason: Optional[str] = None

    # cluster id (filled in by cluster.py)
    cluster_id: Optional[str] = None

    # nn.Module attribution (filled in from ttnn_module_hierarchy.json
    # if the run produced one). `module_path` is the FULL attribute path
    # of the innermost Module whose forward() owned this op (e.g.
    # "model.layers.0.self_attn.q_proj"). `module_class` is that
    # Module's Python class name (e.g. "Linear"). Both None for runs
    # without the sidecar — callers gracefully degrade to flat op view.
    module_path: Optional[str] = None
    module_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _walk_signposts(rows: List[TracyRow]) -> List[Tuple[int, str]]:
    """Maintain a block-path stack as we encounter signposts.

    Convention used by `tracy.block_scope`:
      header == "block_start <name>"  pushes <name>
      header == "block_end <name>"    pops <name>
    A plain header is treated as a one-shot marker that updates the
    current scope but doesn't push.
    Returns: list of (row_index, block_path) for every non-signpost row.
    """
    stack: List[str] = []
    out: List[Tuple[int, str]] = []
    for i, r in enumerate(rows):
        if r.is_signpost:
            h = r.signpost_header
            if h.startswith("block_start "):
                stack.append(h[len("block_start ") :].strip())
            elif h.startswith("block_end "):
                name = h[len("block_end ") :].strip()
                if stack and stack[-1] == name:
                    stack.pop()
                elif name in stack:
                    while stack and stack[-1] != name:
                        stack.pop()
                    if stack:
                        stack.pop()
            else:
                pass
            continue
        out.append((i, ".".join(stack) if stack else "root"))
    return out


def join_run(
    *,
    run_id: str,
    tracy_csv: Optional[Path],
    tracer_master: Optional[Path],
    num_hidden_layers: Optional[int] = None,
    module_hierarchy: Optional[Path] = None,
) -> List[JoinedRow]:
    """Tracy ⋈ tracer on (op_name, occurrence_index_within_name).

    Tracer-orphan Tracy rows keep `args_hash=None` and `arguments={}`.
    Tracer entries that don't have a matching Tracy row are silently
    dropped (they typically correspond to host-only ttnn calls that
    didn't produce a device kernel).

    When ``module_hierarchy`` is provided and the sidecar exists, the
    ``module_path`` and ``module_class`` fields on every JoinedRow are
    populated from the sidecar's op-counter index. Runs without the
    sidecar leave both fields None and downstream consumers (cytoscape
    view, suggestion engine) degrade gracefully.
    """
    tracy_rows = parse_tracy_csv(tracy_csv) if tracy_csv else []
    tracer_calls = parse_tracer_master(tracer_master) if tracer_master else []

    # Late import to avoid a cycle: module_graph.py imports JoinedRow.
    hierarchy_idx = None
    if module_hierarchy is not None:
        from .module_graph import parse_hierarchy_sidecar

        hierarchy_idx = parse_hierarchy_sidecar(module_hierarchy)

    tracer_by_name: Dict[str, List[TracerCall]] = {}
    for call in tracer_calls:
        tracer_by_name.setdefault(call.operation_name, []).append(call)
    tracer_consumed: Dict[str, int] = {k: 0 for k in tracer_by_name}

    block_assignments = dict(_walk_signposts(tracy_rows))

    joined: List[JoinedRow] = []
    op_name_counter: Dict[str, int] = {}

    for i, r in enumerate(tracy_rows):
        if r.is_signpost:
            continue

        op_key = r.op_code
        op_name_counter[op_key] = op_name_counter.get(op_key, -1) + 1

        # We try several aliases because tracy's OP CODE and tracer's
        # operation_name don't always agree on prefix (ttnn.add vs add).
        candidates: List[str] = []
        if op_key:
            candidates.append(op_key)
            if op_key.startswith("ttnn."):
                candidates.append(op_key[len("ttnn.") :])
            else:
                candidates.append(f"ttnn.{op_key}")

        matched: Optional[TracerCall] = None
        for cand in candidates:
            seq = tracer_by_name.get(cand)
            if seq is None:
                continue
            idx = tracer_consumed.get(cand, 0)
            if idx < len(seq):
                matched = seq[idx]
                tracer_consumed[cand] = idx + 1
                break

        module_path_val: Optional[str] = None
        module_class_val: Optional[str] = None
        if hierarchy_idx is not None and not hierarchy_idx.is_empty:
            attr = hierarchy_idx.op_attribution.get(r.global_call_count)
            if attr is not None:
                module_path_val, module_class_val = attr

        joined.append(
            JoinedRow(
                run_id=run_id,
                row_index=i,
                global_call_count=r.global_call_count,
                device_id=r.device_id,
                block_path=block_assignments.get(i, "root"),
                module_path=module_path_val,
                module_class=module_class_val,
                op_code=r.op_code,
                op_type=r.op_type,
                tracer_op_name=matched.operation_name if matched else None,
                args_hash=matched.arguments_hash if matched else None,
                arguments=matched.arguments if matched else {},
                math_fidelity=r.math_fidelity,
                inputs_summary=r.inputs_summary,
                outputs_summary=r.outputs_summary,
                inputs=r.inputs,
                outputs=r.outputs,
                compute_kernel_source=r.compute_kernel_source,
                compute_kernel_hash=r.compute_kernel_hash,
                dm_kernel_source=r.dm_kernel_source,
                dm_kernel_hash=r.dm_kernel_hash,
                program_hash=r.program_hash,
                program_cache_hit=r.program_cache_hit,
                core_count=r.core_count,
                device_kernel_ns=r.device_kernel_duration_ns,
                device_fw_ns=r.device_fw_duration_ns,
                op_to_op_latency_ns=r.op_to_op_latency_ns,
                brisc_ns=r.brisc_ns,
                ncrisc_ns=r.ncrisc_ns,
                trisc0_ns=r.trisc0_ns,
                trisc1_ns=r.trisc1_ns,
                trisc2_ns=r.trisc2_ns,
                erisc_ns=r.erisc_ns,
                compute_cb_wait_front_ns=r.compute_cb_wait_front_ns,
                pm_ideal_ns=r.pm_ideal_ns,
                pm_compute_ns=r.pm_compute_ns,
                pm_bandwidth_ns=r.pm_bandwidth_ns,
                pm_req_i_bw=r.pm_req_i_bw,
                pm_req_o_bw=r.pm_req_o_bw,
                pm_fpu_util_pct=r.pm_fpu_util_pct,
                noc_util_pct=r.noc_util_pct,
                multicast_noc_util_pct=r.multicast_noc_util_pct,
                dram_bw_util_pct=r.dram_bw_util_pct,
                eth_bw_util_pct=r.eth_bw_util_pct,
            )
        )

    # Auto-infer transformer-block boundaries if the demo did not emit
    # `tracy.block_scope` signposts AND we know the model's layer count.
    # This is what populates the "per-block utilization vs runtime" chart
    # for any HF model on any TT box, without any demo instrumentation.
    if num_hidden_layers and joined and all(j.block_path == "root" for j in joined):
        from .block_inference import infer_block_paths

        inferred = infer_block_paths(
            [j.op_code for j in joined],
            num_hidden_layers,
        )
        for j, label in zip(joined, inferred):
            j.block_path = label

    # If module-hierarchy sidecar exists but did not attribute any ops
    # (or the run had no sidecar), fall back to block_path attribution so
    # graph/timeline tooling still has a navigable module-like hierarchy.
    if joined and all(not j.module_path for j in joined):
        for j in joined:
            block = j.block_path or "root"
            role = _infer_role_from_op_code(j.op_code)
            j.module_path = f"{block}.{role}" if block != "root" else role
            j.module_class = "BlockScope"
    return joined


def write_joined_json(rows: List[JoinedRow], path: Path) -> Path:
    """Write joined rows to JSON (no pandas dependency); chart layer reads it back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [r.to_dict() for r in rows]
    path.write_text(json.dumps(serializable, indent=2, default=str))
    return path


def read_joined_json(path: Path) -> List[JoinedRow]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out: List[JoinedRow] = []
    for d in data:
        if "device_id" not in d:
            d["device_id"] = None
        out.append(JoinedRow(**d))
    return out

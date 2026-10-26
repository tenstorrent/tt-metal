# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Per-call / per-layer device-cycle extraction for the Quasar ResNet50 e2e run (Polaris LUT input).

Merges, keyed by device-op runtime ID (GLOBAL CALL COUNT = call_id << 10 | device_id):
  * profile_log_device.csv           -- device FW / KERNEL zones per core per RISC (cycles)
  * generated/inspector/*.yaml       -- op type, input TensorSpecs, workload -> program -> kernels
  * layer_markers.jsonl              -- RESNET_LAYER_MARKERS from _log_op: layer name <-> call-ID window
  * ops_perf_results_*.csv (tracy)   -- optional: attributes, output tensors, math fidelity, op hash
  * torchvision resnet50 (static)    -- conv/pool/linear params, NCHW shapes, MACs

Outputs (into --out):
  per_call.csv / per_call.jsonl  one row per call ID that has device records
  call_id_map.csv                every allocated call ID 0..max, with the reason when it has no kernel
  per_core.csv                   per call ID, per core, per RISC zone timing + kernel name
  layers.csv                     per model layer (marker window) aggregate
  manifest.json                  run manifest

All device times are raw cycles ("time[cycles since reset]"). Nanosecond columns are only emitted when
the device log reports a non-zero CHIP_FREQ; with CHIP_FREQ 0 (emulator) nothing is converted.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

DEVICE_ID_NUM_BITS = 10
HOST_FALLBACK_BIT = 31
FIRST_CALL_ID = 1  # ttnn::CoreIDs::device_operation_id starts at 1

TRISC_ROLE = {"TRISC0": "unpack", "TRISC1": "math", "TRISC2": "pack", "TRISC3": "sfpu"}
NOT_MEASURABLE = "not_measurable"
UTIL_NOTE = (
    "Quasar exposes no FPU/SFPU perf counters to tt-metal (perf_counters.hpp is WH/BH only), so matrix- and "
    "vector-pipe utilization are not measured. *_kernel_frac columns are TRISC kernel-zone lifetime / op kernel "
    "duration: an UPPER BOUND on pipe occupancy (includes CB waits/stalls), not utilization."
)
NOC_NOTE = (
    "No NoC-level timing: Quasar device profiler is L1-only (no NoC event tracing), so NoC cycles are not "
    "measured. read/write cycles are DM-RISC kernel-zone lifetimes attributed by the kernel bound to that RISC."
)


# ----------------------------------------------------------------------------------------------- utils
def decode_run_id(run_id):
    return {
        "call_id": (run_id & ((1 << HOST_FALLBACK_BIT) - 1)) >> DEVICE_ID_NUM_BITS,
        "device_id": run_id & ((1 << DEVICE_ID_NUM_BITS) - 1),
        "host_fallback": bool(run_id >> HOST_FALLBACK_BIT),
    }


def yaml_load(path):
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path) as f:
        return yaml.load(f, Loader=loader) or []


def git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", repo, *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def repo_info(path):
    if not path or not os.path.isdir(path):
        return None
    sha = git(path, "rev-parse", "HEAD")
    if sha is None:
        return None
    dirty = git(path, "status", "--porcelain", "--untracked-files=no")
    return {
        "path": path,
        "commit": sha,
        "branch": git(path, "rev-parse", "--abbrev-ref", "HEAD"),
        "describe": git(path, "describe", "--tags", "--always", "--dirty"),
        "dirty_files": dirty.splitlines() if dirty else [],
    }


# ---------------------------------------------------------------------------------- device log parsing
def parse_device_log(path):
    with open(path) as f:
        header = f.readline()
        m = re.search(r"ARCH:\s*(\w+),\s*CHIP_FREQ\[MHz\]:\s*(\d+)(?:,\s*Max Compute Cores:\s*(\d+))?", header)
        info = {
            "arch": m.group(1) if m else None,
            "chip_freq_mhz": int(m.group(2)) if m else None,
            "max_compute_cores": int(m.group(3)) if m and m.group(3) else None,
            "header": header.strip(),
        }
        reader = csv.reader(f)
        cols = [c.strip() for c in next(reader)]
        idx = {c: i for i, c in enumerate(cols)}
        records = []
        for row in reader:
            if not row or len(row) < len(cols) - 1:
                continue
            records.append(
                (
                    int(row[idx["run host ID"]]),
                    (int(row[idx["core_x"]]), int(row[idx["core_y"]])),
                    row[idx["RISC processor type"]].strip(),
                    row[idx["zone name"]].strip(),
                    row[idx["type"]].strip(),
                    int(row[idx["time[cycles since reset]"]]),
                )
            )
    info["n_records"] = len(records)
    info["run_id_source"] = "device log run host ID"
    return info, records


def program_windows(records):
    """Time windows of the programs in the log: DM0-FW spans grouped across cores (overlap => same program)."""
    spans, open_ = [], {}
    for run_id, core, risc, zone, typ, t in sorted(records, key=lambda r: r[5]):
        if zone != "DM0-FW":
            continue
        if typ == "ZONE_START":
            open_[core] = t
        elif typ == "ZONE_END" and core in open_:
            spans.append((open_.pop(core), t))
    spans.sort()
    groups = []
    for s, e in spans:
        if groups and s <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], e)
        else:
            groups.append([s, e])
    return groups


def assign_run_ids_by_order(records, runtime_ids):
    """Accumulate-mode logs carry run host ID 0 for every marker. With slow dispatch programs run strictly one
    after another, so the k-th program window in time is the k-th enqueued runtime ID. Only applied when the
    window count equals the runtime-entry count exactly."""
    groups = program_windows(records)
    if len(groups) != len(runtime_ids):
        return None, f"{len(groups)} program windows vs {len(runtime_ids)} runtime entries; not assigning"
    bounds = [g[0] for g in groups]
    import bisect

    out = []
    for run_id, core, risc, zone, typ, t in records:
        k = bisect.bisect_right(bounds, t) - 1
        if k < 0 or t > groups[k][1]:
            return None, f"marker at cycle {t} ({risc} {zone}) falls outside every program window"
        out.append((runtime_ids[k] << DEVICE_ID_NUM_BITS, core, risc, zone, typ, t))
    return out, f"time order: {len(groups)} program windows -> runtime IDs {runtime_ids[0]}..{runtime_ids[-1]}"


def build_zones(records):
    # zones[run_id][(x, y)][risc][zone] = {"start": min, "end": max, "n_start", "n_end"}
    zones = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for run_id, core, risc, zone, typ, t in records:
        z = zones[run_id][core][risc].setdefault(zone, {"start": None, "end": None, "n_start": 0, "n_end": 0})
        if typ == "ZONE_START":
            z["start"] = t if z["start"] is None else min(z["start"], t)
            z["n_start"] += 1
        elif typ == "ZONE_END":
            z["end"] = t if z["end"] is None else max(z["end"], t)
            z["n_end"] += 1
    return zones


def short_risc(risc):
    # QUASAR_DM2 -> DM2, QUASAR_NEO0_TRISC1 -> TRISC1 (single Neo cluster per Tensix on this config)
    r = risc.replace("QUASAR_", "")
    return re.sub(r"^NEO\d+_", "", r)


# ----------------------------------------------------------------------------------- inspector parsing
def parse_inspector(inspector_dir, kernel_cache_root=None):
    out = {"runtime_entries": {}, "workload_programs": defaultdict(list), "program_kernels": defaultdict(list)}
    if not inspector_dir or not os.path.isdir(inspector_dir):
        return None
    mw = os.path.join(inspector_dir, "mesh_workloads_log.yaml")
    if os.path.exists(mw):
        for ev in yaml_load(mw):
            if "runtime_entry" in ev:
                e = ev["runtime_entry"]
                out["runtime_entries"][int(e["runtime_id"])] = {
                    "mesh_workload_id": int(e["mesh_workload_id"]),
                    "name": str(e.get("name", "")),
                    "parameters": str(e.get("parameters", "")),
                    "trace_id": e.get("trace_id"),
                }
            elif "mesh_workload_add_program" in ev:
                e = ev["mesh_workload_add_program"]
                out["workload_programs"][int(e["mesh_workload_id"])].append(int(e["program_id"]))
    kp = os.path.join(inspector_dir, "kernels.yaml")
    if os.path.exists(kp):
        for ev in yaml_load(kp):
            k = ev.get("kernel")
            if not k:
                continue
            path = k.get("path", "") or ""
            procs = []
            if path and os.path.isdir(path):
                for sub in sorted(os.listdir(path)):
                    if re.fullmatch(r"dm\d+", sub) and os.path.isdir(os.path.join(path, sub)):
                        procs.append(sub.upper())
                if not procs and any(f.startswith("chlkc_") for f in os.listdir(path)):
                    procs = ["TRISC0", "TRISC1", "TRISC2", "TRISC3"]
            if not procs:
                src = k.get("source", "")
                procs = ["TRISC*"] if "/compute/" in src else ["DM?"]
            out["program_kernels"][int(k["program_id"])].append(
                {
                    "name": k.get("name"),
                    "source": k.get("source"),
                    "watcher_kernel_id": k.get("watcher_kernel_id"),
                    "processors": procs,
                }
            )
    out["has_runtime_entries"] = bool(out["runtime_entries"])
    return out


def kernels_for_call(insp, call_id):
    if not insp:
        return []
    e = insp["runtime_entries"].get(call_id)
    if not e:
        return []
    ks = []
    for pid in insp["workload_programs"].get(e["mesh_workload_id"], []):
        ks.extend(insp["program_kernels"].get(pid, []))
    return ks


_SPEC_SPLIT = re.compile(r"(?:^|, )\[(\d+)\]: ")


def parse_tensor_specs(s):
    """Parse Inspector's '[0]: TensorSpec(...), [1]: TensorSpec(...)' into dicts (best effort, raw kept)."""
    if not s or s == "Not captured":
        return []
    parts = _SPEC_SPLIT.split(s)
    specs = []
    for i in range(1, len(parts) - 1, 2):
        specs.append(parse_one_spec(parts[i + 1]))
    return specs


def _m(pat, s, g=1):
    m = re.search(pat, s)
    return m.group(g) if m else None


_GRID_RE = re.compile(r'"start":\{"x":(\d+),"y":(\d+)\},"end":\{"x":(\d+),"y":(\d+)\}')


def parse_one_spec(s):
    """Fields from fmt(TensorSpec): TensorSpec(logical_shape=Shape([..]),tensor_layout=TensorLayout(dtype=..,
    page_config=..,memory_config=MemoryConfig(memory_layout=..,buffer_type=..,shard_spec=ShardSpec{grid=[..],
    shape=[h, w], orientation=..}|std::nullopt,nd_shard_spec={"shard_shape":[..],"grid":[..],..}|std::nullopt,..))."""
    shape = _m(r"logical_shape=Shape\(\[([^\]]*)\]\)", s) or _m(r"Shape\(\[([^\]]*)\]\)", s)
    layout = None
    if re.search(r"TilePageConfig|Tile\(|tile_shape", s):
        layout = "TILE"
    elif "RowMajorPageConfig" in s:
        layout = "ROW_MAJOR"
    shard_sec = _m(r"shard_spec=(ShardSpec\{.*?\}),nd_shard_spec=", s)
    nd_sec = _m(r"nd_shard_spec=(\{.*?\}),created_with", s)
    grid_src = shard_sec or nd_sec or ""
    grid = _GRID_RE.findall(grid_src)
    shard_cores = sum((int(ex) - int(sx) + 1) * (int(ey) - int(sy) + 1) for sx, sy, ex, ey in grid) if grid else None
    shard_shape = None
    if shard_sec:
        m = re.search(r"\],\s*shape=\[([\d,\s]+)\]", shard_sec)
        shard_shape = m and "x".join(x.strip() for x in m.group(1).split(","))
    elif nd_sec:
        m = re.search(r'"shard_shape":\[([\d,\s]+)\]', nd_sec)
        shard_shape = m and "x".join(x.strip() for x in m.group(1).split(","))
    return {
        "shape": "x".join(x.strip() for x in shape.split(",")) if shape else None,
        "dtype": _m(r"DataType::(\w+)", s),
        "layout": layout,
        "memory_layout": _m(r"TensorMemoryLayout::(\w+)", s),
        "buffer_type": _m(r"BufferType::(\w+)", s),
        "shard_grid": ";".join(f"({a},{b})-({c},{d})" for a, b, c, d in grid) if grid else None,
        "shard_cores": shard_cores,
        "shard_shape": shard_shape,
        "shard_orientation": _m(r"ShardOrientation::(\w+)", grid_src),
        "raw": s,
    }


# --------------------------------------------------------------------------------------- tracy ops csv
def parse_tracy_ops(path):
    if not path or not os.path.exists(path):
        return None
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                gcc = int(r.get("GLOBAL CALL COUNT", "") or -1)
            except ValueError:
                continue
            if gcc < 0:
                continue
            if "OP CODE" not in r:
                continue
            rows[decode_run_id(gcc)["call_id"]] = r
    return rows or None


# -------------------------------------------------------------------------------------- torch reference
def torch_reference(batch, h, w):
    """Static torchvision resnet50 layer table: params + NCHW in/out shapes + MACs (weights not needed)."""
    try:
        import torch
        import torchvision
    except Exception as e:
        return {}, f"torchvision unavailable: {e}"
    model = torchvision.models.resnet50().eval()
    table = {}

    def hook(name):
        def fn(mod, inp, out):
            rec = table.setdefault(name, {"torch_module": name, "torch_type": type(mod).__name__})
            rec["torch_in_shape_nchw"] = list(inp[0].shape)
            rec["torch_out_shape_nchw"] = list(out.shape)
            if isinstance(mod, torch.nn.Conv2d):
                kh, kw = mod.kernel_size
                rec.update(
                    in_channels=mod.in_channels,
                    out_channels=mod.out_channels,
                    kernel=f"{kh}x{kw}",
                    stride="x".join(map(str, mod.stride)),
                    padding="x".join(map(str, mod.padding)),
                    dilation="x".join(map(str, mod.dilation)),
                    groups=mod.groups,
                    bias=mod.bias is not None,
                    macs=int(out.numel() * (mod.in_channels // mod.groups) * kh * kw),
                )
            elif isinstance(mod, torch.nn.MaxPool2d):
                k = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size,) * 2
                s = mod.stride if isinstance(mod.stride, tuple) else (mod.stride,) * 2
                p = mod.padding if isinstance(mod.padding, tuple) else (mod.padding,) * 2
                d = mod.dilation if isinstance(mod.dilation, tuple) else (mod.dilation,) * 2
                rec.update(
                    kernel="x".join(map(str, k)),
                    stride="x".join(map(str, s)),
                    padding="x".join(map(str, p)),
                    dilation="x".join(map(str, d)),
                    groups=1,
                )
            elif isinstance(mod, torch.nn.Linear):
                rec.update(
                    in_channels=mod.in_features,
                    out_channels=mod.out_features,
                    bias=mod.bias is not None,
                    macs=int(out.numel() * mod.in_features),
                )
            elif isinstance(mod, torch.nn.AdaptiveAvgPool2d):
                rec.update(kernel=f"{inp[0].shape[2]}x{inp[0].shape[3]}", stride="1x1", padding="0x0", groups=1)

        return fn

    for name, mod in model.named_modules():
        if isinstance(
            mod,
            (
                torch.nn.Conv2d,
                torch.nn.MaxPool2d,
                torch.nn.Linear,
                torch.nn.AdaptiveAvgPool2d,
                torchvision.models.resnet.Bottleneck,
            ),
        ):
            mod.register_forward_hook(hook(name))
    with torch.no_grad():
        model(torch.zeros(batch, 3, h, w))
    return table, None


def marker_to_torch(name):
    """Map the model's _log_op name to (torch module path, fused torch modules, layer kind)."""
    if name == "stem_conv1":
        return "conv1", ["bn1 (folded into weights)", "relu (conv activation)"], "conv"
    if name == "stem_maxpool":
        return "maxpool", [], "pool"
    if name in ("fold", "stem_tilize", "output"):
        return None, [], "layout"
    if name == "avgpool":
        return "avgpool", [], "pool"
    if name == "fc":
        return "fc", ["flatten (view)"], "matmul"
    m = re.fullmatch(r"layer(\d+)_module(\d+)\.(conv1|conv2|conv3|downsample|add)", name)
    if not m:
        return None, [], "unknown"
    blk = f"layer{m.group(1)}.{int(m.group(2)) - 1}"
    part = m.group(3)
    if part in ("conv1", "conv2"):
        return f"{blk}.{part}", [f"{blk}.bn{part[-1]} (folded into weights)", "relu (conv activation)"], "conv"
    if part == "conv3":
        return f"{blk}.conv3", [f"{blk}.bn3 (folded into weights)"], "conv"
    if part == "downsample":
        return f"{blk}.downsample.0", [f"{blk}.downsample.1 (BN folded into weights)"], "conv"
    return f"{blk}.residual_add", [f"{blk}.relu (add activation)"], "eltwise"


def op_category(op_type):
    t = (op_type or "").lower()
    if not t:
        return None
    for pat, cat in (
        (r"conv", "conv"),
        (r"matmul|bmm|linear", "matmul"),
        (r"halo|sliding", "conv.halo"),
        (r"pool", "pool"),
        (r"binary|add|eltwise|unary", "eltwise"),
        (r"reduce", "reduce"),
        (
            r"reshard|shard|interleave|tilize|untilize|reshape|copy|move|slice|pad|concat|transpose|fold|"
            r"to_memory|typecast|permute",
            "data_movement",
        ),
    ):
        if re.search(pat, t):
            return cat
    return "other"


def dm_role(kernel_name):
    n = (kernel_name or "").lower()
    r, w = "reader" in n or "read" in n or "gather" in n, "writer" in n or "write" in n or "drain" in n
    return "read+write" if r and w else "read" if r else "write" if w else "unknown"


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-log", required=True, help="profile_log_device.csv")
    ap.add_argument("--inspector-dir", help="generated/inspector (needs TT_METAL_INSPECTOR_LOG_RUNTIME_ENTRIES=1)")
    ap.add_argument("--markers", help="layer_markers.jsonl written via RESNET_LAYER_MARKERS")
    ap.add_argument("--tracy-ops", help="optional tracy ops_perf_results_*.csv (with host op columns)")
    ap.add_argument("--run-env", help="file of KEY=VALUE lines captured at launch (profiler/emulator settings)")
    ap.add_argument("--pytest-log", help="optional pytest log, for PCC / top-1 in the manifest")
    ap.add_argument("--soc-descriptor", help="simulator soc_descriptor.yaml (topology)")
    ap.add_argument("--repo", default=os.getcwd(), help="tt-metal checkout")
    ap.add_argument("--extra-repo", action="append", default=[], help="NAME=PATH extra git repo for the manifest")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--input-hw", default="224x224")
    ap.add_argument("--max-call-id", type=int, help="last allocated call ID (default: max seen in any source)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev_info, records = parse_device_log(args.device_log)
    freq = dev_info["chip_freq_mhz"]
    emit_ns = bool(freq)
    insp = parse_inspector(args.inspector_dir)
    if records and all(r[0] == 0 for r in records):
        rids = sorted(insp["runtime_entries"]) if insp else []
        assigned, msg = assign_run_ids_by_order(records, rids) if rids else (None, "no inspector runtime entries")
        print(f"device log has run host ID 0 on every marker (accumulate mode): {msg}")
        if assigned is None:
            sys.exit("cannot attribute device records to call IDs; rerun with TT_METAL_INSPECTOR_LOG_RUNTIME_ENTRIES=1")
        records = assigned
        dev_info["run_id_source"] = msg
    zones = build_zones(records)
    tracy = parse_tracy_ops(args.tracy_ops)
    markers = []
    if args.markers and os.path.exists(args.markers):
        with open(args.markers) as f:
            markers = [json.loads(l) for l in f if l.strip()]
        for mk in markers:
            for k in ("dtype", "layout", "memory_layout", "buffer_type"):
                if isinstance(mk.get(k), str):
                    mk[k] = mk[k].split(".")[-1].split("::")[-1]
    h, w = map(int, args.input_hw.split("x"))
    torch_tab, torch_err = torch_reference(args.batch_size, h, w)

    # ---- call-ID -> layer window
    window = {}  # call_id -> (marker, "model"|"instrumentation")
    prev_end = FIRST_CALL_ID
    for mk in markers:
        for cid in range(prev_end, mk["call_id_at_hook"]):
            window[cid] = (mk, "model")
        for cid in range(mk["call_id_at_hook"], mk["call_id_after_readback"]):
            window[cid] = (mk, "instrumentation")
        prev_end = mk["call_id_after_readback"]
    marker_end = prev_end if markers else None

    dev_calls = {}
    for run_id in zones:
        d = decode_run_id(run_id)
        dev_calls[d["call_id"]] = (run_id, d)

    max_id = args.max_call_id
    if max_id is None:
        cands = list(dev_calls) + (list(insp["runtime_entries"]) if insp else []) + (list(tracy) if tracy else [])
        if marker_end:
            cands.append(marker_end - 1)
        max_id = max(cands) if cands else -1

    per_call, per_core_rows, call_map = [], [], []
    for cid in range(FIRST_CALL_ID, max_id + 1):
        mk, win_kind = window.get(cid, (None, "after_last_marker" if markers else None))
        layer_name = mk["name"] if mk else None
        torch_path, fused, kind = marker_to_torch(layer_name) if layer_name else (None, [], None)
        ie = insp["runtime_entries"].get(cid) if insp else None
        tr = tracy.get(cid) if tracy else None
        op_type = (ie or {}).get("name") or (tr or {}).get("OP CODE")
        tr_type = (tr or {}).get("OP TYPE")

        # ---- call-ID map row (every ID)
        if cid in dev_calls:
            status, reason = "kernel", ""
        elif tr_type and "fallback" in tr_type.lower() or (tr_type and "host" in tr_type.lower()):
            status, reason = "no_kernel", "host-only op"
        elif win_kind == "instrumentation":
            status, reason = "no_kernel", "filtered op (PCC-hook to_torch readback; not part of the model)"
        elif ie or tr:
            nm = (op_type or "").lower()
            if "reshape" in nm:
                reason = "reshape (device op launched, no device profiler record)"
            elif "view" in nm:
                reason = "view (device op launched, no device profiler record)"
            else:
                reason = "missing record (enqueued on device, no profiler record: dropped/overwritten)"
            status = "no_kernel"
        elif insp and insp.get("has_runtime_entries"):
            # Every enqueue_mesh_workload logs a runtime entry; an ID with none was allocated outside it.
            status, reason = "no_kernel", "host-only op (ID allocated without a device workload enqueue)"
        else:
            status = "no_kernel"
            reason = "missing record (no host metadata captured; device record absent/overwritten)"
        if cid in dev_calls and win_kind == "instrumentation":
            reason = "filtered op (PCC-hook to_torch readback; has kernel but is not part of the model)"
        call_map.append(
            {
                "call_id": cid,
                "global_call_count": (cid << DEVICE_ID_NUM_BITS),
                "status": status,
                "reason": reason,
                "op_type": op_type,
                "layer_name": layer_name,
                "torch_module": torch_path,
                "window": win_kind,
                "host_source": "+".join(s for s, v in (("inspector", ie), ("tracy", tr)) if v) or None,
            }
        )
        if cid not in dev_calls:
            continue

        # ---- timing from device zones
        run_id, d = dev_calls[cid]
        cores = zones[run_id]
        kernels = kernels_for_call(insp, cid)
        proc_kernel = defaultdict(list)
        for k in kernels:
            for p in k["processors"]:
                proc_kernel[p].append(k["name"])
        fw_s, fw_e, k_s, k_e, dm0_s, dm0_e = [], [], [], [], [], []
        core_kernel = {}
        # risc -> [min start, max end, max per-core cycles, max start]
        risc_span = defaultdict(lambda: [None, None, 0, 0])
        for core, riscs in sorted(cores.items()):
            ck_s, ck_e = [], []
            for risc, zs in riscs.items():
                r = short_risc(risc)
                fw = next((z for n, z in zs.items() if n.endswith("-FW")), None)
                kn = next((z for n, z in zs.items() if n.endswith("-KERNEL")), None)
                if r == "DM0":
                    if fw and fw["start"] is not None:
                        dm0_s.append(fw["start"])
                    if fw and fw["end"] is not None:
                        dm0_e.append(fw["end"])
                else:
                    if fw and fw["start"] is not None:
                        fw_s.append(fw["start"])
                    if fw and fw["end"] is not None:
                        fw_e.append(fw["end"])
                if kn and kn["start"] is not None and kn["end"] is not None:
                    ck_s.append(kn["start"])
                    ck_e.append(kn["end"])
                    sp = risc_span[r]
                    sp[0] = kn["start"] if sp[0] is None else min(sp[0], kn["start"])
                    sp[1] = kn["end"] if sp[1] is None else max(sp[1], kn["end"])
                    sp[2] = max(sp[2], kn["end"] - kn["start"])
                    sp[3] = max(sp[3], kn["start"])
                kname = ";".join(sorted(set(proc_kernel.get(r, [])))) or None
                per_core_rows.append(
                    {
                        "call_id": cid,
                        "global_call_count": run_id,
                        "layer_name": layer_name,
                        "op_type": op_type,
                        "core_x": core[0],
                        "core_y": core[1],
                        "risc": r,
                        "risc_role": TRISC_ROLE.get(r, "dispatch_fw" if r == "DM0" else dm_role(kname)),
                        "kernel_name": kname,
                        "fw_start_cycle": fw["start"] if fw else None,
                        "fw_end_cycle": fw["end"] if fw else None,
                        "fw_cycles": (
                            (fw["end"] - fw["start"]) if fw and fw["start"] is not None and fw["end"] else None
                        ),
                        "kernel_start_cycle": kn["start"] if kn else None,
                        "kernel_end_cycle": kn["end"] if kn else None,
                        "kernel_cycles": (
                            (kn["end"] - kn["start"])
                            if kn and kn["start"] is not None and kn["end"] is not None
                            else None
                        ),
                        "zone_markers": sum(z["n_start"] + z["n_end"] for z in zs.values()),
                    }
                )
            if ck_s:
                core_kernel[core] = (min(ck_s), max(ck_e))
                k_s.append(min(ck_s))
                k_e.append(max(ck_e))

        per_core_dur = {c: e - s for c, (s, e) in core_kernel.items()}
        durs = list(per_core_dur.values())
        kernel_cycles = (max(k_e) - min(k_s)) if k_s else None

        def span(prefix):
            ss = [v for r, v in risc_span.items() if r.startswith(prefix)]
            if not ss:
                return None
            return max(v[1] for v in ss) - min(v[0] for v in ss)

        rec = {
            "global_call_count": run_id,
            "call_id": cid,
            "device_id": d["device_id"],
            "layer_name": layer_name,
            "torch_module": torch_path,
            "fused_torch_modules": "; ".join(fused) if fused else None,
            "layer_kind": kind,
            "window": win_kind,
            "op_type": op_type,
            "op_category": op_category(op_type),
            "is_primary_for_layer": None,
            "compute_kernels": ";".join(sorted({k["name"] for k in kernels if k["processors"][0].startswith("TRISC")}))
            or None,
            "dm2_kernel": ";".join(sorted(set(proc_kernel.get("DM2", [])))) or None,
            "dm3_kernel": ";".join(sorted(set(proc_kernel.get("DM3", [])))) or None,
            "math_fidelity": (tr or {}).get("MATH FIDELITY"),
            "cores_with_kernel": len(core_kernel),
            "cores": ";".join(f"({x},{y})" for x, y in sorted(core_kernel)),
            "device_fw_start_cycle": min(fw_s) if fw_s else None,
            "device_fw_end_cycle": max(fw_e) if fw_e else None,
            "device_fw_cycles": (max(fw_e) - min(fw_s)) if fw_s and fw_e else None,
            "dm0_dispatch_fw_start_cycle": min(dm0_s) if dm0_s else None,
            "dm0_dispatch_fw_end_cycle": max(dm0_e) if dm0_e else None,
            "device_kernel_start_cycle": min(k_s) if k_s else None,
            "device_kernel_end_cycle": max(k_e) if k_e else None,
            "device_kernel_cycles": kernel_cycles,
            "per_core_kernel_min_cycles": min(durs) if durs else None,
            "per_core_kernel_max_cycles": max(durs) if durs else None,
            "per_core_kernel_avg_cycles": round(sum(durs) / len(durs), 1) if durs else None,
            "per_core_imbalance_pct": (
                round(100.0 * (max(durs) - min(durs)) / max(durs), 3) if durs and max(durs) else None
            ),
            "per_core_max_over_avg": round(max(durs) / (sum(durs) / len(durs)), 4) if durs and sum(durs) else None,
            "kernel_first_to_last_start_cycles": (max(v[3] for v in risc_span.values()) - min(k_s)) if k_s else None,
            "core_first_to_last_start_cycles": (max(k_s) - min(k_s)) if k_s else None,
            "compute_cycles": span("TRISC"),
            "unpack_trisc0_cycles": risc_span["TRISC0"][2] if "TRISC0" in risc_span else None,
            "math_trisc1_cycles": risc_span["TRISC1"][2] if "TRISC1" in risc_span else None,
            "pack_trisc2_cycles": risc_span["TRISC2"][2] if "TRISC2" in risc_span else None,
            "sfpu_trisc3_cycles": risc_span["TRISC3"][2] if "TRISC3" in risc_span else None,
            "data_movement_cycles": span("DM"),
            "dm2_cycles": risc_span["DM2"][2] if "DM2" in risc_span else None,
            "dm3_cycles": risc_span["DM3"][2] if "DM3" in risc_span else None,
            "read_cycles": None,
            "write_cycles": None,
            "noc_cycles": NOT_MEASURABLE,
            "matrix_pipe_util": NOT_MEASURABLE,
            "vector_pipe_util": NOT_MEASURABLE,
            "math_trisc1_kernel_frac": None,
            "sfpu_trisc3_kernel_frac": None,
        }
        for dm in ("DM2", "DM3"):
            if dm in risc_span:
                role = dm_role(rec[f"{dm.lower()}_kernel"])
                rec[f"{dm.lower()}_role"] = role
                if role in ("read", "write"):
                    key = f"{role}_cycles"
                    rec[key] = max(rec[key] or 0, risc_span[dm][2])
            else:
                rec[f"{dm.lower()}_role"] = None
        if kernel_cycles:
            for r, key in (("TRISC1", "math_trisc1_kernel_frac"), ("TRISC3", "sfpu_trisc3_kernel_frac")):
                if r in risc_span:
                    rec[key] = round(risc_span[r][2] / kernel_cycles, 4)

        # ---- tensors: inspector inputs, tracy outputs, marker output for the layer's last op
        ins = parse_tensor_specs(ie["parameters"]) if ie else []
        for i, s in enumerate(ins[:3]):
            for k, v in s.items():
                if k != "raw":
                    rec[f"in{i}_{k}"] = v
        rec["inputs_raw"] = " | ".join(s["raw"] for s in ins) if ins else None
        rec["num_inputs"] = len(ins) if ie else None
        if tr:
            for k, v in tr.items():
                if k.startswith("OUTPUT_0_") or (k.startswith("INPUT_") and not ins):
                    rec[k.lower()] = v
            rec["tracy_attributes"] = tr.get("ATTRIBUTES")
            rec["tracy_program_cache_hit"] = tr.get("PROGRAM CACHE HIT")
            rec["tracy_program_hash"] = tr.get("PROGRAM HASH")

        # ---- torch reference (conv params / NCHW shapes / MACs)
        tref = torch_tab.get(torch_path.replace(".residual_add", "")) if torch_path else None
        for k in (
            "torch_type",
            "torch_in_shape_nchw",
            "torch_out_shape_nchw",
            "in_channels",
            "out_channels",
            "kernel",
            "stride",
            "padding",
            "dilation",
            "groups",
            "macs",
        ):
            v = tref.get(k) if tref else None
            if (
                torch_path
                and torch_path.endswith(".residual_add")
                and k
                not in (
                    "torch_in_shape_nchw",
                    "torch_out_shape_nchw",
                )
            ):
                v = "Bottleneck.add" if k == "torch_type" else None
            rec[k] = "x".join(map(str, v)) if isinstance(v, list) else v
        if emit_ns:
            for k in list(rec):
                if k.endswith("_cycles") and isinstance(rec[k], (int, float)):
                    rec[k.replace("_cycles", "_ns")] = round(rec[k] * 1000.0 / freq, 1)
        per_call.append(rec)

    # ---- primary op per layer window + marker output tensor on the window's last model op
    by_layer = defaultdict(list)
    for rec in per_call:
        if rec["layer_name"] and rec["window"] == "model":
            by_layer[rec["layer_name"]].append(rec)
    mk_by_name = {mk["name"]: mk for mk in markers}
    for name, recs in by_layer.items():
        kind = recs[0]["layer_kind"]
        want = {"conv": ("conv", "matmul"), "matmul": ("matmul",), "pool": ("pool",), "eltwise": ("eltwise",)}.get(
            kind, ()
        )
        prim = [r for r in recs if r["op_category"] in want]
        prim = prim or ([max(recs, key=lambda r: r["device_kernel_cycles"] or 0)] if not want else [])
        for r in recs:
            r["is_primary_for_layer"] = r in prim
        last = max(recs, key=lambda r: r["call_id"])
        mk = mk_by_name.get(name)
        if mk:
            for k in ("shape", "padded_shape", "dtype", "layout", "memory_layout", "buffer_type", "shard_spec"):
                v = mk.get(k)
                last[f"layer_out_{k}"] = "x".join(map(str, v)) if isinstance(v, list) else v

    # ---- layers.csv
    layers = []
    for mk in markers:
        recs = [r for r in per_call if r["layer_name"] == mk["name"] and r["window"] == "model"]
        ids_model = [c for c, (m, wk) in window.items() if m is mk and wk == "model"]
        tp, fused, kind = marker_to_torch(mk["name"])
        tref = torch_tab.get(tp) if tp else None
        layers.append(
            {
                "op_idx": mk["op_idx"],
                "layer_name": mk["name"],
                "torch_module": tp,
                "layer_kind": kind,
                "fused_torch_modules": "; ".join(fused) if fused else None,
                "call_ids": ";".join(map(str, sorted(ids_model))),
                "call_ids_with_kernel": ";".join(str(r["call_id"]) for r in recs),
                "op_types": ";".join(r["op_type"] or "?" for r in recs),
                "sum_device_kernel_cycles": sum(r["device_kernel_cycles"] or 0 for r in recs) if recs else None,
                "sum_device_fw_cycles": sum(r["device_fw_cycles"] or 0 for r in recs) if recs else None,
                "primary_kernel_cycles": (
                    sum(r["device_kernel_cycles"] or 0 for r in recs if r["is_primary_for_layer"]) if recs else None
                ),
                "kernel_coverage": f"{len(recs)}/{len(ids_model)}",
                "out_shape": "x".join(map(str, mk.get("shape", []))),
                "out_dtype": mk.get("dtype"),
                "out_layout": mk.get("layout"),
                "out_memory_layout": mk.get("memory_layout"),
                "out_buffer_type": mk.get("buffer_type"),
                "out_shard_spec": mk.get("shard_spec"),
                "kernel": (tref or {}).get("kernel"),
                "stride": (tref or {}).get("stride"),
                "padding": (tref or {}).get("padding"),
                "dilation": (tref or {}).get("dilation"),
                "groups": (tref or {}).get("groups"),
                "macs": (tref or {}).get("macs"),
            }
        )

    # ---- write tables
    def write_csv(path, rows):
        if not rows:
            open(path, "w").close()
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)

    write_csv(os.path.join(args.out, "per_call.csv"), per_call)
    with open(os.path.join(args.out, "per_call.jsonl"), "w") as f:
        for r in per_call:
            f.write(json.dumps(r) + "\n")
    write_csv(os.path.join(args.out, "call_id_map.csv"), call_map)
    write_csv(os.path.join(args.out, "per_core.csv"), per_core_rows)
    write_csv(os.path.join(args.out, "layers.csv"), layers)

    # ---- manifest
    run_env = {}
    if args.run_env and os.path.exists(args.run_env):
        with open(args.run_env) as f:
            for line in f:
                if "=" in line:
                    k, v = line.rstrip("\n").split("=", 1)
                    run_env[k] = v
    soc = None
    if args.soc_descriptor and os.path.exists(args.soc_descriptor):
        import yaml

        with open(args.soc_descriptor) as f:
            sd = yaml.safe_load(f)
        soc = {
            k: sd.get(k)
            for k in (
                "arch_name",
                "grid",
                "functional_workers",
                "harvested_workers",
                "dram",
                "dram_views",
                "dram_bank_size",
                "worker_l1_size",
                "features",
            )
        }
        soc["path"] = args.soc_descriptor
    repos = {"tt-metal": repo_info(args.repo)}
    for sub in ("tt_metal/third_party/umd", "tt_metal/third_party/tracy"):
        repos[os.path.basename(sub)] = repo_info(os.path.join(args.repo, sub))
    repos["tt-llk"] = {"note": "in-tree (tt_metal/tt-llk), same commit as tt-metal"}
    for spec in args.extra_repo:
        n, p = spec.split("=", 1)
        repos[n] = repo_info(p)
    pytest = {}
    if args.pytest_log and os.path.exists(args.pytest_log):
        txt = open(args.pytest_log, errors="replace").read()
        for k, pat in (
            ("pcc", r"PCC[^0-9]{0,20}([01]\.\d+)"),
            ("top1", r"top-?1[^0-9]{0,20}(\d+)"),
            ("result", r"=+ (\d+ passed[^=]*|\d+ failed[^=]*) =+"),
        ):
            m = re.findall(pat, txt, re.I)
            pytest[k] = m[-1] if m else None
    covered = sum(1 for r in call_map if r["status"] == "kernel")
    manifest = {
        "model": "ResNet50 (torchvision), models/demos/vision/classification/resnet50/quasar",
        "test": "models/demos/vision/classification/resnet50/quasar/tests/test_resnet50_e2e.py -k pretrained",
        "batch_size": args.batch_size,
        "input_shape_nchw": [args.batch_size, 3, h, w],
        "input": "torch.rand(..., dtype=torch.float32) with torch.manual_seed(0) in test infra",
        "activations_dtype": "bfloat16",
        "weights_dtype": "bfloat16",
        "math_fidelity": "LoFi",
        "weights_identifier": "torchvision ResNet50_Weights.IMAGENET1K_V1 (IMAGENET1K_V1.pt via model_location_generator)",
        "quasar_revision": run_env.get("AETHER_TAG"),
        "emulator": {
            k: run_env.get(k)
            for k in (
                "AETHER_TAG",
                "AETHER_CONFIG",
                "TT_UMD_SIMULATOR",
                "TT_METAL_SIMULATOR",
                "NNG_SOCKET_ADDR",
                "EMULATOR_TIMEOUT",
            )
            if k in run_env
        },
        "topology": soc,
        "device_log": {**dev_info, "path": os.path.abspath(args.device_log)},
        "time_unit": "cycles" if not freq else f"cycles (ns columns use {freq} MHz)",
        "cores_seen": sorted({f"({r['core_x']},{r['core_y']})" for r in per_core_rows}),
        "riscs_seen": sorted({r["risc"] for r in per_core_rows}),
        "trisc_roles": TRISC_ROLE,
        "profiler_settings": {k: v for k, v in run_env.items() if k.startswith(("TT_METAL", "TTNN", "RESNET_"))},
        "repos": repos,
        "host_sources": {
            "inspector_dir": os.path.abspath(args.inspector_dir) if args.inspector_dir else None,
            "inspector_runtime_entries": len(insp["runtime_entries"]) if insp else 0,
            "markers": len(markers),
            "tracy_ops": len(tracy) if tracy else 0,
            "torch_reference": torch_err or "ok",
        },
        "coverage": {
            "allocated_call_ids": max_id + 1 - FIRST_CALL_ID,
            "first_call_id": FIRST_CALL_ID,
            "call_ids_with_kernel": covered,
            "by_reason": {
                k: sum(1 for r in call_map if (r["reason"] or "kernel") == k)
                for k in sorted({r["reason"] or "kernel" for r in call_map})
            },
        },
        "notes": {"utilization": UTIL_NOTE, "noc": NOC_NOTE},
        "pytest": pytest,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"time unit: {manifest['time_unit']}  (CHIP_FREQ[MHz]={freq})")
    print(
        f"call IDs {FIRST_CALL_ID}..{max_id}: {covered} with kernel records, {max_id + 1 - FIRST_CALL_ID - covered} without"
    )
    for k, v in manifest["coverage"]["by_reason"].items():
        print(f"  {v:4d}  {k}")
    print(f"layers: {len(layers)}  per-call rows: {len(per_call)}  per-core rows: {len(per_core_rows)}")
    print(f"wrote {args.out}/{{per_call.csv,per_call.jsonl,call_id_map.csv,per_core.csv,layers.csv,manifest.json}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

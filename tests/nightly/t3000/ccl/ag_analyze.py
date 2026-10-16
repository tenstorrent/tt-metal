#!/usr/bin/env python3
"""Summarize AllGather device-kernel perf from tt-metal profiler ops_perf_results CSVs.

Everything (configs, iteration counts, device count) is derived from the CSV itself.

Method, per config:
  * config key  = topology|layout|dtype|memory|output shape|dim|links|op|pkt<packet_size>[|tuning],
    all parsed from the INPUT_0_* columns, ATTRIBUTES and OP CODE. `tuning` exists only in
    reports from builds that still had the (since removed) all_gather tuning test hook: its
    ATTRIBUTES 'tuning' struct, printed as key=value with only the non-default keys ('-' = all
    defaults, and always '-' for current builds). So rows that differ in packet size or tuning never
    pool. The `config` column shows the key without the pkt/tuning parts; they get their own
    `pkt` and `tuning` columns.
  * iteration   = rank of a row within (config, device), ordered by GLOBAL CALL COUNT and
    then METAL TRACE REPLAY SESSION ID (all replays of one trace share a call count; the
    trace-capture placeholder has no session id and sorts first).
  * warmup      = the first --warmup invocations (default 2: compile run + trace capture)
    are dropped, counting the NaN-duration placeholder that trace capture leaves. NaN rows
    are then dropped.
  * per iteration take the MAX duration over devices (slowest device); report the median
    over iterations, spread = (max-min)/median, n, and a bootstrap 95% CI of the median.
    A config with <= --warmup invocations keeps all of them (nothing to drop) and is warned about.
  * efficiency  = median(PM IDEAL [ns]) / median duration.
  * eff50%      = the same, but with the roofline recomputed by ideal_us() (a Python port of
    AllGatherDeviceOperation::create_op_performance_model) at 50 GB/s per link direction instead
    of the 25 GB/s the C++ assumes on BH. --link-gbps G (not 25) renames it eff@G%. The port is
    checked against PM IDEAL for every config and the max relative error is printed.
Durations > --desync-ns (default 1e8 ns) are treated as Blackhole wall-clock desync
artifacts: counted, warned about, and the WHOLE iteration (every device's row for that config
and iteration) is excluded, so the per-iteration max is never taken over a partial device set.

--compare joins A and B rows on the full key; a config (key minus pkt/tuning) that occurs once on
each side is joined on that instead, so a packet-size or tuning A/B still lines up (the differing
ATTRIBUTES are listed in a NOTE). verdict = "noise" when |delta| < --min-delta (default 5%; the CI
only captures within-run variation, run-to-run drift is larger) OR the two bootstrap 95% CIs of the
median overlap, else "B faster" / "B slower".

--vs-default A/Bs tunings WITHIN one run (only meaningful for reports from the removed tuning hook,
where one report held several tuning variants of the same shapes): per (config, pkt) the all-defaults row ('-') is A and every other
tuning is B, with the same verdict rule. --vs TUNING uses that compact tuning string (exact match, as
shown in the `tuning` column) as A instead. A summary per B tuning follows: configs faster / slower /
noise and the geometric-mean delta over the non-noise rows.

Usage:
  ag_analyze.py [REPORT_DIR_OR_CSV ...] [--warmup N] [--devices N] [--csv OUT]
                [--link-gbps G] [--aiclk-mhz F] [--num-cores C]
  ag_analyze.py --compare A B [--bootstrap N] [--min-delta PCT]   (A/B: dirs or CSVs)
  ag_analyze.py [REPORT_DIR_OR_CSV ...] --vs-default [--vs TUNING] [--min-delta PCT] [--csv OUT]
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DUR = "DEVICE KERNEL DURATION [ns]"
IDEAL = "PM IDEAL [ns]"
OPC = "OP CODE"
GCC = "GLOBAL CALL COUNT"
DEV = "DEVICE ID"
ATTR = "ATTRIBUTES"
SESSION = "METAL TRACE REPLAY SESSION ID"
SHAPE_COLS = [f"INPUT_0_{a}_PAD[LOGICAL]" for a in "WZYX"]
LAYOUT = "INPUT_0_LAYOUT"
DTYPE = "INPUT_0_DATATYPE"
MEMORY = "INPUT_0_MEMORY"
OUT_MEMORY = "OUTPUT_0_MEMORY"
ARCH = "DEVICE ARCH"
AVAIL_CORES = "AVAILABLE WORKER CORE COUNT"
DEFAULT_REPORTS = Path("generated/profiler/reports")

# bytes per element (block-float formats include their shared exponents: 1088 B / 576 B per 1024-elem tile)
DTYPE_BYTES = {
    "BFLOAT16": (2.0, "bf16"),
    "FLOAT32": (4.0, "f32"),
    "BFLOAT8_B": (1088 / 1024, "bfp8"),
    "BFLOAT4_B": (576 / 1024, "bfp4"),
    "UINT32": (4.0, "u32"),
    "INT32": (4.0, "i32"),
    "UINT16": (2.0, "u16"),
    "UINT8": (1.0, "u8"),
}

OUT_COLUMNS = [
    "key", "topology", "layout", "dtype", "shape", "dim", "op", "out_bytes",
    "n", "median_us", "spread_pct", "ideal_us", "eff_pct", "us_per_mb",
    "model_us", "ideal_alt_us", "eff_alt_pct",
    "config", "packet_size", "tuning", "ci_lo_us", "ci_hi_us",
]  # fmt: skip

# ----------------------------------------------------------------------------- legacy tuning hook
# Reports from builds with the removed all_gather tuning test hook carry an ATTRIBUTES 'tuning' struct
# (ttsl reflection, e.g. `<struct>(force_factory=AllGatherAlgorithm::Unicast;...;packet_size_override=std::nullopt)`,
# possibly namespace-qualified; the profiler turns ',' into ';'). Parsed only so those older reports still
# key and compare correctly; current builds have no such attribute. Field -> short key, in the hook's
# field order.
TUNING_FIELDS = {
    "force_factory": "factory", "walk_k": "k", "xfer_policy": "xfer", "rotation": "rotation", "entry_policy": "entry", "padded_slot": "padded_slot",
    "run_cap_bytes": "run_cap", "workers_per_dir": "workers", "signals_per_stripe": "signals",
    "mcast_cb_depth": "mcast_cb_depth", "packet_size_override": "packet",
}  # fmt: skip
# unset optionals, rotation None, entry Legacy, padded_slot 0
TUNING_DEFAULTS = {"nullopt", "none", "legacy", "0", "false"}
# fields whose default is not in TUNING_DEFAULTS, and per-field value spellings (walk_k: 1 = default, 0 = auto)
TUNING_FIELD_DEFAULTS = {"walk_k": {"1"}}
TUNING_FIELD_VALUES = {"walk_k": {"0": "auto"}}


def compact_tuning(raw: str | None) -> str:
    """Reflection string of the legacy tuning struct -> non-default keys as key=value ('' = all defaults).

    Unparseable input is returned raw (with ';' put back to ',') rather than dropped, so it still
    separates configs."""
    if not raw or raw in ("std::nullopt", "nan"):
        return ""
    m = re.search(r"\((.*)\)\s*$", raw.strip())
    pairs = re.findall(r"(\w+)=([^;,()]*)", m.group(1)) if m else []
    if not pairs or any(k not in TUNING_FIELDS for k, _ in pairs):
        return raw.replace(";", ",")
    out = []
    for k, v in pairs:
        v = v.strip().split("::")[-1]
        v = re.sub(r"(?<!^)(?=[A-Z])", "_", v).lower()  # AbsoluteDirStagger -> absolute_dir_stagger
        v = {"true": "1"}.get(v, v)
        if v in TUNING_FIELD_DEFAULTS.get(k, ()):
            continue
        v = TUNING_FIELD_VALUES.get(k, {}).get(v, v)
        if k in TUNING_FIELD_VALUES or v not in TUNING_DEFAULTS:
            out.append(f"{TUNING_FIELDS[k]}={v}")
    return ",".join(out)


def bootstrap_median_ci(x, n_boot: int, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile-bootstrap (1-alpha) CI of the median of x. Fixed seed -> reproducible verdicts."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or n_boot <= 0:
        m = float(np.median(x)) if len(x) else float("nan")
        return m, m
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    lo, hi = np.quantile(meds, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------- roofline model
# Port of AllGatherDeviceOperation::create_op_performance_model
#   (ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.cpp:147-239)
# plus the helpers it calls and the OpPerformanceModelGeneral ctor that turns cycles into PM IDEAL [ns].
# Constants copied verbatim from:
#   link BW GB/s  WH 12.5 / BH 25.0          ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:2217,2219
#   hop latency ns WH 711|874, BH 515|619 (1D|2D fabric)          ccl_common.cpp:2235-2236
#   DRAM / L1 latency cycles: dram WH 358 BH 529; l1 write 256/404; l1 read 259/403
#                                            ttnn/cpp/ttnn/operations/data_movement/common/common.cpp:210-216
#   NOC "device frequency" used to turn bytes into cycles: WH 1e9, BH 1.2e9          common.cpp:221
#   NOC BW tables {size: (WH, BH) GB/s}      common.cpp:93-136
#   peak DRAM BW for PM BANDWIDTH: WH 6*2*21.5=258, BH 512 GB/s (GiB in the formula)
#                                            ttnn/core/operation.cpp:39-42,46-49
#   ideal_ns = max(ceil(cycles / aiclk_GHz), int(bandwidth_ns))     ttnn/core/operation.cpp:30,122
FABRIC_LINK_GBPS = {"wormhole_b0": 12.5, "blackhole": 25.0}
FABRIC_HOP_NS = {"wormhole_b0": (711.0, 874.0), "blackhole": (515.0, 619.0)}  # (1D, 2D)
NOC_FREQ_HZ = {"wormhole_b0": 1e9, "blackhole": 1.2e9}
PEAK_DRAM_GBPS = {"wormhole_b0": 6 * 2 * 21.5, "blackhole": 512.0}
LAT_CYCLES = {  # (dram, l1 write, l1 read)
    "wormhole_b0": (358, 256, 259),
    "blackhole": (529, 404, 403),
}
_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
NOC_DRAM_BW = dict(zip(_SIZES, [
    (0.436, 0.387), (0.868, 0.772), (1.736, 1.545), (3.489, 3.088), (6.975, 6.176), (13.889, 12.361),
    (27.891, 24.71), (28.411, 49.164), (28.227, 50.238), (28.537, 50.393), (27.831, 50.636),
    (27.758, 50.695), (28.694, 50.626)]))  # fmt: skip
NOC_L1_READ_BW = dict(zip(_SIZES, [
    (0.868, 0.671), (1.724, 1.336), (3.477, 2.673), (6.885, 5.354), (13.794, 10.691), (27.143, 21.382),
    (28.976, 42.771), (29.742, 47.977), (29.544, 49.34), (28.728, 49.961), (28.7, 50.287),
    (28.618, 50.335), (28.7, 50.403)]))  # fmt: skip
NOC_L1_WRITE_BW = dict(zip(_SIZES, [
    (0.681, 0.511), (1.254, 1.018), (2.709, 2.036), (5.417, 4.072), (10.823, 8.143), (21.668, 16.284),
    (27.837, 32.593), (27.811, 48.644), (27.811, 49.828), (27.808, 50.219), (27.808, 50.578),
    (27.811, 50.412), (28.808, 50.383)]))  # fmt: skip
# Tensor::element_size(): block-float formats count 1 byte/elem (ttnn/core/tensor/tensor.cpp:386-399);
# the buffer page size still uses the real tile size (DTYPE_BYTES * 1024).
MODEL_ELEM_BYTES = {"bf16": 2, "f32": 4, "bfp8": 1, "bfp4": 1, "u32": 4, "i32": 4, "u16": 2, "u8": 1}
# Defaults for model inputs the CSV may not carry
DEFAULT_ARCH = "blackhole"
DEFAULT_AICLK_MHZ = 1350  # BH AICLK; PM IDEAL rounds through cycles at this clock
DEFAULT_NUM_CORES = 110  # BH compute_with_storage_grid 11x10, as reported by AVAILABLE WORKER CORE COUNT


def _noc_bw(size: int, is_dram: bool, is_read: bool, arch: str) -> float:
    """get_transaction_noc_bw (common.cpp:153-178), is_local=false."""
    table = NOC_DRAM_BW if is_dram else NOC_L1_READ_BW if is_read else NOC_L1_WRITE_BW
    idx = 0 if arch == "wormhole_b0" else 1
    keys = sorted(table)
    lo = 2 ** math.floor(math.log2(size))
    hi = 2 ** math.ceil(math.log2(size))
    lo_k = next((k for k in keys if k >= lo), keys[0])
    hi_k = next((k for k in keys if k >= hi), keys[-1])
    return table[lo_k][idx] if size - lo < hi - size else table[hi_k][idx]


def _txn_cycles(size: int, is_dram: bool, n: int, arch: str, is_read: bool) -> tuple[int, int]:
    """get_cycles_for_transaction_size (common.cpp:204-225), is_local=false -> (bw_cycles, latency_cycles)."""
    dram_lat, wr_lat, rd_lat = LAT_CYCLES[arch]
    lat = dram_lat if is_dram else rd_lat if is_read else wr_lat
    size = max(size, 16)
    bw = _noc_bw(size, is_dram, is_read, arch)
    return math.ceil(n * size * NOC_FREQ_HZ[arch] / (bw * 1e9)), lat


def ideal_us(
    out_bytes: int, num_devices: int, num_links: int, topology: str, link_gbps: float, *,
    arch: str = DEFAULT_ARCH, fabric_2d: bool = False, aiclk_mhz: float = DEFAULT_AICLK_MHZ,
    num_cores: int = DEFAULT_NUM_CORES, in_page: int = 2048, out_page: int = 2048,
    in_dram: bool = True, out_dram: bool = True, axes: list[tuple[int, str, int]] | None = None,
) -> float:  # fmt: skip
    """AllGather roofline in us, same structure as the C++ (all_gather_device_operation.cpp:147-239).

    out_bytes = N*S (S = physical bytes per device, Tensor::element_size convention). `axes` is
    [(devices, topology, links)] per mesh axis; default is one axis (num_devices, topology, num_links).
    """
    N = num_devices
    S = out_bytes // N
    cyc_per_ns = aiclk_mhz / 1000.0
    # fabric: K inbound links at the worst node (wrapped axis 2x), hops = network diameter
    K = hops = 0
    for nd, topo, links in axes or [(N, topology, num_links)]:
        if nd <= 1:
            continue
        wraps = topo.lower() in ("ring", "torus")
        K += (2 if wraps else 1) * links
        hops += nd // 2 if wraps else nd - 1
    fabric_bytes = (N - 1) * S
    bw_ns = fabric_bytes / (link_gbps * K) if K > 0 else 0.0
    fabric_bw_cyc = math.ceil(bw_ns * cyc_per_ns)
    fabric_fill_cyc = math.ceil(FABRIC_HOP_NS[arch][1 if fabric_2d else 0] * hops * cyc_per_ns)
    # local DRAM: read S, write N*S, spread over all compute cores
    rd_ppc = math.ceil(math.ceil(S / in_page) / num_cores)
    wr_ppc = math.ceil(math.ceil(N * S / out_page) / num_cores)
    rd_bw, rd_lat = _txn_cycles(in_page, in_dram, rd_ppc, arch, is_read=True)
    wr_bw, wr_lat = _txn_cycles(out_page, out_dram, wr_ppc, arch, is_read=False)
    throughput = max(max(rd_bw, wr_bw), fabric_bw_cyc)
    fill = max(rd_lat + wr_lat, fabric_fill_cyc)
    cycles = max(throughput, fill)
    # OpPerformanceModelGeneral: ns = max(ceil(cycles / GHz), tensor DRAM-transfer floor)
    compute_ns = math.ceil(cycles / cyc_per_ns)
    gib_ns = 1e9 / (PEAK_DRAM_GBPS[arch] * 1024**3)
    tensor_ns = [b * gib_ns if dram else 1.0 for b, dram in ((S, in_dram), (N * S, out_dram))]
    return max(compute_ns, int(max([1.0, *tensor_ns]))) / 1e3


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


# ----------------------------------------------------------------------------- input


def resolve_csv(p: str | None) -> Path:
    """A CSV path, a report dir containing one ops_perf_results*.csv, or None (= newest dir)."""
    if p is None:
        if not DEFAULT_REPORTS.is_dir():
            sys.exit(f"error: {DEFAULT_REPORTS} not found (run from repo root or pass a path)")
        dirs = sorted(d for d in DEFAULT_REPORTS.iterdir() if d.is_dir() and d.name.replace("_", "").isdigit())
        if not dirs:
            sys.exit(f"error: no timestamped report dirs under {DEFAULT_REPORTS}")
        path = dirs[-1]
    else:
        path = Path(p)
    if path.is_file():
        return path
    if path.is_dir():
        csvs = sorted(path.glob("ops_perf_results*.csv"))
        if len(csvs) != 1:
            sys.exit(f"error: expected exactly one ops_perf_results*.csv in {path}, found {len(csvs)}")
        return csvs[0]
    sys.exit(f"error: not found: {path}")


# ----------------------------------------------------------------------------- parsing


def parse_attrs(s) -> dict[str, str]:
    if not isinstance(s, str):
        return {}
    return dict(re.findall(r"'([^']+)':\s*'([^']*)'", s))


def brace_list(v: str | None) -> list[str] | None:
    """'{Topology::Linear; Topology::Ring}' -> ['Topology::Linear', 'Topology::Ring']."""
    if not v:
        return None
    v = v.strip()
    if v.startswith("{") and v.endswith("}"):
        return [x.strip() for x in v[1:-1].split(";") if x.strip()]
    return None


def to_int(v) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def pad_dim(cell) -> int | None:
    """'32[30]' -> 32 (the padded size before the brackets); '32' -> 32."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None
    m = re.match(r"\s*(\d+)", str(cell))
    return int(m.group(1)) if m else None


def logical_dim(cell) -> int | None:
    """'32[32]' -> 32 (the logical size in brackets); '32' -> 32."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None
    s = str(cell)
    m = re.search(r"\[(\d+)\]", s)
    if m:
        return int(m.group(1))
    try:
        return int(float(s))
    except ValueError:
        return None


def describe_row(row: pd.Series, attrs: dict[str, str], devices_flag: int | None) -> dict:
    in_shape = [logical_dim(row.get(c)) for c in SHAPE_COLS]
    in_shape = [d for d in in_shape if d is not None] or None
    rank = len(in_shape) if in_shape else 4

    # gather axis: explicit cluster_axis, else the only mesh axis with >1 device
    axis_ndev = [to_int(x) for x in (brace_list(attrs.get("axis_num_devices")) or [])]
    axis = to_int(attrs.get("cluster_axis"))
    if axis is None and axis_ndev:
        multi = [i for i, n in enumerate(axis_ndev) if n and n > 1]
        axis = multi[0] if len(multi) == 1 else None

    def per_axis(name: str) -> str | None:
        lst = brace_list(attrs.get(name))
        if lst is None:
            return None
        if axis is not None and axis < len(lst):
            return lst[axis]
        return lst[-1] if len(set(lst)) == 1 else None

    topo = attrs.get("topology") or per_axis("axis_topology") or "?"
    topo = topo.split("::")[-1].lower()
    links = to_int(attrs.get("num_links")) or to_int(per_axis("axis_num_links"))

    ndev = (
        to_int(attrs.get("num_devices"))
        or to_int(attrs.get("ring_size"))
        or (axis_ndev[axis] if axis is not None and axis < len(axis_ndev) else None)
        or devices_flag
    )

    # output shape from output_spec if present
    out_shape = None
    m = re.search(r"logical_shape=Shape\(\[([^\]]*)\]", attrs.get("output_spec", ""))
    if m:
        out_shape = [int(x) for x in re.split(r"[;,]\s*", m.group(1)) if x.strip()]

    dim = to_int(attrs.get("dim"))
    if dim is None:
        dfe = to_int(attrs.get("dim_from_end"))
        if dfe is not None:
            dim = rank + dfe if dfe < 0 else rank - 1 - dfe
    if dim is not None and dim < 0:
        dim += rank
    if dim is None and in_shape and out_shape and len(in_shape) == len(out_shape):
        diff = [i for i, (a, b) in enumerate(zip(in_shape, out_shape)) if a != b]
        dim = diff[0] if len(diff) == 1 else None
    if ndev is None and dim is not None and in_shape and out_shape and in_shape[dim]:
        ndev = out_shape[dim] // in_shape[dim]

    if out_shape is None and in_shape and dim is not None and ndev:
        out_shape = list(in_shape)
        out_shape[dim] *= ndev

    raw_dt = str(row.get(DTYPE, "?")).split("::")[-1].upper()
    bpe, dt = DTYPE_BYTES.get(raw_dt, (None, raw_dt.lower()))
    out_bytes = None
    if in_shape and bpe and ndev:
        out_bytes = int(round(math.prod(in_shape) * bpe * ndev))
    elif out_shape and bpe:
        out_bytes = int(round(math.prod(out_shape) * bpe))

    lay = str(row.get(LAYOUT, "?")).split("::")[-1].upper()
    lay = {"TILE": "tile", "ROW_MAJOR": "rm"}.get(lay, lay.lower())

    # roofline-model inputs: physical (padded) bytes, buffer page sizes, buffer types, per-axis fabric
    model = None
    pad_shape = [pad_dim(row.get(c)) for c in SHAPE_COLS]
    pad_shape = [d for d in pad_shape if d is not None]
    eb = MODEL_ELEM_BYTES.get(dt)
    if pad_shape and eb and bpe and ndev and links is not None:
        if lay == "tile":
            in_page = out_page = int(round(1024 * bpe))
        else:
            in_page = pad_shape[-1] * eb
            out_page = in_page * (ndev if dim == len(pad_shape) - 1 else 1)
        a_nd = brace_list(attrs.get("axis_num_devices"))
        a_tp = brace_list(attrs.get("axis_topology"))
        a_nl = brace_list(attrs.get("axis_num_links"))
        axes = None
        if a_nd and a_tp and a_nl and len(a_nd) == len(a_tp) == len(a_nl):
            axes = [(to_int(n) or 0, t.split("::")[-1].lower(), to_int(l) or 0) for n, t, l in zip(a_nd, a_tp, a_nl)]
        is_dram = lambda col: "DRAM" in str(row.get(col, row.get(MEMORY, "DRAM")))
        model = dict(
            out_bytes=math.prod(pad_shape) * eb * ndev, num_devices=ndev, num_links=links, topology=topo,
            fabric_2d="2D" in attrs.get("fabric_config", ""), in_page=in_page, out_page=out_page,
            in_dram=is_dram(MEMORY), out_dram=is_dram(OUT_MEMORY), axes=axes,
        )  # fmt: skip

    mem = re.sub(r"^DEV_\d+_", "", str(row.get(MEMORY, "?")))
    mem = "DRAM" if mem == "DRAM_INTERLEAVED" else "L1" if mem == "L1_INTERLEAVED" else mem
    op = re.sub(r"DeviceOperation$", "", str(row[OPC]))

    shape_s = "[" + ",".join(map(str, out_shape)) + "]" if out_shape else "?"
    dim_s = f"dim{dim}" if dim is not None else "dim?"
    links_s = f"L{links}" if links is not None else "L?"
    config = "|".join([topo, lay, dt, mem, shape_s, dim_s, links_s, op])
    packet = to_int(attrs.get("packet_size"))
    tuning = compact_tuning(attrs.get("tuning"))
    key = config + (f"|pkt{packet}" if packet is not None else "") + (f"|{tuning}" if tuning else "")
    # everything in ATTRIBUTES except output_spec, to flag non-key differences in A/B mode
    sig = {k: v for k, v in attrs.items() if k != "output_spec"}
    return dict(
        key=key, config=config, packet_size=packet, tuning=tuning, topology=topo, layout=lay, dtype=dt, mem=mem,
        shape=shape_s, dim=dim, op=op, links=links, num_devices=ndev, out_bytes=out_bytes,
        attr_sig=tuple(sorted(sig.items())), model=model,
    )  # fmt: skip


# ----------------------------------------------------------------------------- analysis


def analyze(
    csv_path: Path,
    warmup: int,
    devices_flag: int | None,
    desync_ns: float,
    model_opts: dict | None = None,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """model_opts: link_gbps (None = C++ value), alt_gbps, aiclk_mhz, num_cores (None = from CSV)."""
    mo = {**dict(link_gbps=None, alt_gbps=None, aiclk_mhz=DEFAULT_AICLK_MHZ, num_cores=None), **(model_opts or {})}
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in (OPC, DUR, GCC, DEV) if c not in df.columns]
    if missing:
        sys.exit(f"error: {csv_path}: missing required columns {missing}")
    df["_row"] = range(len(df))
    ag = df[df[OPC].astype(str).str.startswith("AllGather")].copy()
    print(f"# {csv_path}: {len(ag)} AllGather rows (of {len(df)})")
    if ag.empty:
        return pd.DataFrame()

    # describe each distinct (op, ATTRIBUTES, input columns) combo once
    desc_cols = [c for c in [OPC, ATTR, LAYOUT, DTYPE, MEMORY, OUT_MEMORY, *SHAPE_COLS] if c in ag.columns]
    ag["_dk"] = ag[desc_cols].astype(str).agg("\x1f".join, axis=1)
    descs = {}
    for dk, row in ag.drop_duplicates("_dk").set_index("_dk").iterrows():
        descs[dk] = describe_row(row, parse_attrs(row.get(ATTR)), devices_flag)
    ag["key"] = ag["_dk"].map(lambda k: descs[k]["key"])

    # same key run twice in one report (separated by other AllGather configs) -> separate segments
    ag = ag.sort_values("_row")
    seg = (ag["key"] != ag["key"].shift()).cumsum()
    seg_keys = ag.assign(_seg=seg).groupby("_seg", sort=False)["key"].first()
    repeat = seg_keys.groupby(seg_keys).cumcount() + 1
    ag["_seg"] = seg
    ag["key"] = [k if repeat[s] == 1 else f"{k}#{repeat[s]}" for k, s in zip(ag["key"], ag["_seg"])]
    if (repeat > 1).any():
        warn(f"{csv_path.name}: {int((repeat > 1).sum())} config(s) appear in non-contiguous blocks; suffixed '#N'")

    ag["_dur"] = pd.to_numeric(ag[DUR], errors="coerce")
    ag["_gcc"] = pd.to_numeric(ag[GCC], errors="coerce")
    ag["_sess"] = pd.to_numeric(ag[SESSION], errors="coerce") if SESSION in ag.columns else float("nan")
    ag["_ideal"] = pd.to_numeric(ag[IDEAL], errors="coerce") if IDEAL in ag.columns else float("nan")

    # invocation index within (config, device). Trace replays share one GLOBAL CALL COUNT, so the
    # replay session id breaks ties; the capture placeholder (no session id) sorts first.
    ag = ag.sort_values(["key", DEV, "_gcc", "_sess", "_row"], na_position="first")
    ag["_inv"] = ag.groupby(["key", DEV]).cumcount()
    inv_per = ag.groupby(["key", DEV])["_inv"].transform("size")
    drop_warm = (ag["_inv"] < warmup) & (inv_per > warmup)
    if warmup > 0:
        short = inv_per[inv_per <= warmup].groupby(ag["key"]).max()  # invocations per device, short configs only
        for k, n_inv in short.items():
            warn(
                f"{csv_path.name}: {k}: only {int(n_inv)} invocation(s) per device <= --warmup {warmup}; "
                f"warmup NOT dropped, stats include compile/capture runs"
            )
    meas = ag[~drop_warm & ag["_dur"].notna()].copy()

    # desync: drop the whole iteration (every device's row for that config + invocation index), so the
    # per-iteration max is never taken over a partial device set
    bad_row = meas["_dur"] > desync_ns
    if bad_row.any():
        bad_it = meas.loc[bad_row, ["key", "_inv"]].drop_duplicates()
        bad = pd.MultiIndex.from_frame(meas[["key", "_inv"]]).isin(pd.MultiIndex.from_frame(bad_it))
        per = bad_it.groupby("key").size()
        print("!" * 78, file=sys.stderr)
        warn(
            f"{csv_path.name}: {int(bad_row.sum())} duration(s) > {desync_ns:.0e} ns -- likely BH wall-clock "
            f"desync; {len(bad_it)} whole iteration(s) ({int(bad.sum())} rows, all devices) EXCLUDED from stats. "
            f"Iterations per config: " + ", ".join(f"{k}={v}" for k, v in per.items())
        )
        print("!" * 78, file=sys.stderr)
        meas = meas[~bad]
    # a NaN placeholder outside the warmup window is unexpected; mention it
    late_nan = ag[~drop_warm & ag["_dur"].isna()]
    if len(late_nan):
        warn(f"{csv_path.name}: {len(late_nan)} NaN-duration row(s) after warmup were dropped")

    def first_valid(rows: pd.DataFrame, col: str):
        v = rows[col].dropna() if col in rows.columns else []
        return v.iloc[0] if len(v) else None

    out = []
    max_err, max_err_key = 0.0, None
    for key, g in meas.groupby("key", sort=False):
        all_rows = ag[ag["key"] == key]
        d = descs[all_rows["_dk"].iloc[0]]
        per_iter = g.groupby("_inv")["_dur"].max()  # slowest device per iteration
        ndev_iter = g.groupby("_inv")[DEV].nunique()
        if ndev_iter.nunique() > 1:
            warn(f"{key}: device count per iteration varies ({sorted(ndev_iter.unique())})")
        med = float(per_iter.median())
        spread = (per_iter.max() - per_iter.min()) / med * 100 if med > 0 else float("nan")
        ci_lo, ci_hi = bootstrap_median_ci(per_iter.to_numpy(), n_boot)
        ideal = float(g["_ideal"].median()) if g["_ideal"].notna().any() else float("nan")
        if g["_ideal"].nunique() > 1:
            warn(f"{key}: PM IDEAL not constant ({g['_ideal'].min():.0f}..{g['_ideal'].max():.0f} ns); using median")
        ob = d["out_bytes"]

        # roofline port: at the C++ link rate (reproduces PM IDEAL) and at the alternative rate
        model_us = alt_us = float("nan")
        if d["model"] is not None:
            arch = str(first_valid(all_rows, ARCH) or DEFAULT_ARCH).lower()
            arch = "wormhole_b0" if arch.startswith("wormhole") else arch
            if arch in FABRIC_LINK_GBPS:
                cores = mo["num_cores"] or to_int(first_valid(all_rows, AVAIL_CORES)) or DEFAULT_NUM_CORES
                kw = dict(d["model"], arch=arch, aiclk_mhz=mo["aiclk_mhz"], num_cores=cores)
                base = mo["link_gbps"] or FABRIC_LINK_GBPS[arch]
                model_us = ideal_us(link_gbps=base, **kw)
                alt_us = ideal_us(link_gbps=mo["alt_gbps"] or 2 * base, **kw)
                if ideal > 1:
                    err = abs(model_us * 1e3 - ideal) / ideal
                    if err > 0.01:
                        warn(f"{key}: model {model_us:.2f} us vs PM IDEAL {ideal / 1e3:.2f} us ({err * 100:.2f}% off)")
                    if err >= max_err:
                        max_err, max_err_key = err, key
            else:
                warn(f"{key}: no roofline constants for arch '{arch}'")
        out.append(
            dict(
                key=key, topology=d["topology"], layout=d["layout"], dtype=d["dtype"], shape=d["shape"],
                dim=d["dim"], op=d["op"], out_bytes=ob, n=len(per_iter), median_us=med / 1e3,
                spread_pct=spread, ideal_us=ideal / 1e3 if ideal > 1 else float("nan"),
                eff_pct=ideal / med * 100 if ideal > 1 and med > 0 else float("nan"),
                us_per_mb=(med / 1e3) / (ob / 1e6) if ob else float("nan"),
                model_us=model_us, ideal_alt_us=alt_us,
                eff_alt_pct=alt_us * 1e3 / med * 100 if alt_us > 0 and med > 0 else float("nan"),
                config=d["config"] + (re.search(r"#\d+$", key).group(0) if re.search(r"#\d+$", key) else ""),
                packet_size=d["packet_size"], tuning=d["tuning"], ci_lo_us=ci_lo / 1e3, ci_hi_us=ci_hi / 1e3,
                devices=int(g[DEV].nunique()), attr_sig=d["attr_sig"],
            )
        )  # fmt: skip
    res = pd.DataFrame(out)
    if res.empty:
        return res
    if max_err_key is not None:
        rate = f"{mo['link_gbps']:g} GB/s" if mo["link_gbps"] else "C++ link rate"
        print(f"# roofline port ({rate}) vs PM IDEAL: max rel error {max_err * 100:.3f}% ({max_err_key})")
    return res.sort_values(["out_bytes", "topology", "key"], na_position="last").reset_index(drop=True)


# ----------------------------------------------------------------------------- output


def fmt_bytes(b) -> str:
    if b is None or (isinstance(b, float) and math.isnan(b)):
        return "?"
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b / div:.1f}{unit}"
    return f"{int(b)}B"


def f(v, spec: str) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def ci_pct(lo, hi, med) -> float:
    """Half-width of a CI as % of the median."""
    return (hi - lo) / 2 / med * 100 if med and med > 0 else float("nan")


def pkt_s(v) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else str(int(v))


def print_table(res: pd.DataFrame, alt_label: str = "eff50%") -> None:
    kw = max(len("config"), res["config"].str.len().max())
    aw = max(7, len(alt_label))
    hdr = (
        f"{'config':<{kw}}  {'out':>8}  {'n':>3}  {'median_us':>10}  {'spread%':>7}  {'ideal_us':>9}  {'eff%':>6}  "
        f"{alt_label:>{aw}}  {'us/MB':>7}  {'ci95%':>6}  {'pkt':>5}  tuning"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in res.itertuples():
        print(
            f"{r.config:<{kw}}  {fmt_bytes(r.out_bytes):>8}  {r.n:>3}  {f(r.median_us, '10.2f')}  "
            f"{f(r.spread_pct, '7.1f')}  {f(r.ideal_us, '9.2f')}  {f(r.eff_pct, '6.1f')}  "
            f"{f(r.eff_alt_pct, f'{aw}.1f')}  {f(r.us_per_mb, '7.2f')}  "
            f"{f(ci_pct(r.ci_lo_us, r.ci_hi_us, r.median_us), '6.1f')}  {pkt_s(r.packet_size):>5}  {r.tuning or '-'}"
        )


def join_keys(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add `jkey`: `config` where it is unique on both sides (so a packet/tuning A/B lines up), else `key`."""
    ca, cb = a["config"].value_counts(), b["config"].value_counts()
    uniq = lambda c: ca.get(c, 0) <= 1 and cb.get(c, 0) <= 1
    a = a.assign(jkey=[c if uniq(c) else k for c, k in zip(a["config"], a["key"])])
    b = b.assign(jkey=[c if uniq(c) else k for c, k in zip(b["config"], b["key"])])
    return a, b


def verdict(a_med, a_lo, a_hi, b_med, b_lo, b_hi, min_delta: float) -> tuple[float, str]:
    """(delta% of B vs A, verdict). noise if below the run-to-run floor, or if the bootstrap 95% CIs of
    the two medians overlap."""
    delta = (b_med - a_med) / a_med * 100
    ci_overlap = a_lo <= b_hi and b_lo <= a_hi
    noise = abs(delta) < min_delta or ci_overlap
    return delta, "noise" if noise else ("B faster" if delta < 0 else "B slower")


def print_compare(a: pd.DataFrame, b: pd.DataFrame, la: str, lb: str, min_delta: float = 5.0) -> None:
    a, b = join_keys(a, b)
    m = a.merge(b, on="jkey", how="outer", suffixes=("_a", "_b"), indicator=True)
    m["key"] = m["jkey"]
    both = m[m["_merge"] == "both"].copy()
    only_a = m.loc[m["_merge"] == "left_only", "key_a"].tolist()
    only_b = m.loc[m["_merge"] == "right_only", "key_b"].tolist()
    print(f"A = {la}\nB = {lb}")
    if both.empty:
        print("\nno configs in common.")
    else:
        both = both.sort_values(["out_bytes_a", "topology_a", "key"])
        kw = max(len("config"), both["key"].str.len().max())
        hdr = (
            f"{'config':<{kw}}  {'out':>8}  {'nA':>3} {'nB':>3}  {'A_us':>9}  {'B_us':>9}  {'delta%':>7}  "
            f"{'sprA%':>6}  {'sprB%':>6}  {'ciA%':>5}  {'ciB%':>5}  {'effA%':>6}  {'effB%':>6}  verdict"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in both.itertuples():
            delta, verd = verdict(
                r.median_us_a, r.ci_lo_us_a, r.ci_hi_us_a, r.median_us_b, r.ci_lo_us_b, r.ci_hi_us_b, min_delta
            )
            print(
                f"{r.key:<{kw}}  {fmt_bytes(r.out_bytes_a):>8}  {int(r.n_a):>3} {int(r.n_b):>3}  {r.median_us_a:9.2f}  "
                f"{r.median_us_b:9.2f}  {delta:+7.1f}  {f(r.spread_pct_a, '6.1f')}  {f(r.spread_pct_b, '6.1f')}  "
                f"{f(ci_pct(r.ci_lo_us_a, r.ci_hi_us_a, r.median_us_a), '5.1f')}  "
                f"{f(ci_pct(r.ci_lo_us_b, r.ci_hi_us_b, r.median_us_b), '5.1f')}  "
                f"{f(r.eff_pct_a, '6.1f')}  {f(r.eff_pct_b, '6.1f')}  {verd}"
            )
        # differences in ATTRIBUTES between joined rows (packet_size / tuning when joined on config)
        diffs = {}
        for r in both.itertuples():
            da, db = dict(r.attr_sig_a), dict(r.attr_sig_b)
            for k in sorted(set(da) | set(db)):
                if da.get(k) != db.get(k):
                    diffs.setdefault(k, set()).add((da.get(k), db.get(k)))
        if diffs:
            print("\nNOTE: joined configs differ in ATTRIBUTES fields:")
            for k, vals in diffs.items():
                print(f"  {k}: " + ", ".join(f"A={va} B={vb}" for va, vb in sorted(vals, key=str)))
    for lbl, keys in (("A", only_a), ("B", only_b)):
        if keys:
            print(f"\nonly in {lbl} ({len(keys)}):")
            for k in keys:
                print(f"  {k}")


def vs_tuning(res: pd.DataFrame, ref: str, min_delta: float) -> tuple[pd.DataFrame, list[str]]:
    """Within-report A/B: per (config, pkt), A = the row whose tuning == ref, B = every other tuning.

    Returns (one row per B, sorted by config/pkt/tuning; groups that have other tunings but no ref row)."""
    rows, no_ref = [], []
    pkt = res["packet_size"].map(pkt_s)
    for (config, pk), g in res.assign(_pkt=pkt).groupby(["config", "_pkt"], sort=True):
        a = g[g["tuning"] == ref]
        others = g[g["tuning"] != ref]
        if others.empty:
            continue
        if a.empty:
            no_ref.append(f"{config}|pkt{pk}: " + ", ".join(t or "-" for t in sorted(others["tuning"])))
            continue
        a = a.iloc[0]
        for b in others.itertuples():
            delta, verd = verdict(
                a.median_us, a.ci_lo_us, a.ci_hi_us, b.median_us, b.ci_lo_us, b.ci_hi_us, min_delta
            )  # fmt: skip
            rows.append(
                dict(
                    config=config, packet_size=pk, out_bytes=a.out_bytes, tuning_a=ref or "-", tuning=b.tuning or "-",
                    n_a=int(a.n), n_b=int(b.n), median_us_a=a.median_us, median_us_b=b.median_us, delta_pct=delta,
                    ci_lo_us_a=a.ci_lo_us, ci_hi_us_a=a.ci_hi_us, ci_lo_us_b=b.ci_lo_us, ci_hi_us_b=b.ci_hi_us,
                    ci_pct_a=ci_pct(a.ci_lo_us, a.ci_hi_us, a.median_us),
                    ci_pct_b=ci_pct(b.ci_lo_us, b.ci_hi_us, b.median_us), verdict=verd,
                )
            )  # fmt: skip
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["config", "packet_size", "tuning"]).reset_index(drop=True)
    return out, no_ref


def print_vs_tuning(vs: pd.DataFrame, no_ref: list[str], ref: str) -> None:
    print(f"A = tuning {ref or '-'} (per config, pkt)\nB = every other tuning")
    if no_ref:
        print(f"\nno reference row (tuning {ref or '-'}) for {len(no_ref)} config(s); skipped:")
        for s in no_ref:
            print(f"  {s}")
    if vs.empty:
        print("\nnothing to compare." if no_ref else f"\nno non-{'default' if not ref else 'reference'} tuning rows.")
        return
    kw = max(len("config"), vs["config"].str.len().max())
    tw = max(len("tuning"), vs["tuning"].str.len().max())
    hdr = (
        f"{'config':<{kw}}  {'pkt':>5}  {'tuning':<{tw}}  {'nA':>3} {'nB':>3}  {'A_us':>9}  {'B_us':>9}  "
        f"{'delta%':>7}  {'ciA%':>5}  {'ciB%':>5}  verdict"
    )
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in vs.itertuples():
        print(
            f"{r.config:<{kw}}  {r.packet_size:>5}  {r.tuning:<{tw}}  {r.n_a:>3} {r.n_b:>3}  {r.median_us_a:9.2f}  "
            f"{r.median_us_b:9.2f}  {r.delta_pct:+7.1f}  {f(r.ci_pct_a, '5.1f')}  {f(r.ci_pct_b, '5.1f')}  "
            f"{r.verdict}"
        )
    # per B tuning: verdict counts and geometric-mean delta over the non-noise rows
    print(f"\nsummary per tuning (vs {ref or '-'}; geomean delta over non-noise rows):")
    hdr = f"{'tuning':<{tw}}  {'configs':>7}  {'faster':>6}  {'slower':>6}  {'noise':>5}  {'geomean%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for t, g in vs.groupby("tuning", sort=True):
        sig = g[g["verdict"] != "noise"]
        ratio = sig["median_us_b"] / sig["median_us_a"]
        gm = (float(np.exp(np.log(ratio).mean())) - 1) * 100 if len(sig) else float("nan")
        cnt = g["verdict"].value_counts()
        print(
            f"{t:<{tw}}  {len(g):>7}  {cnt.get('B faster', 0):>6}  {cnt.get('B slower', 0):>6}  "
            f"{cnt.get('noise', 0):>5}  {f(gm, '+8.1f'):>8}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("inputs", nargs="*", help="report dirs or ops_perf_results CSVs (default: newest report dir)")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="A/B compare two report dirs or CSVs")
    p.add_argument("--warmup", type=int, default=2, help="leading invocations to drop per config/device (default 2)")
    p.add_argument("--devices", type=int, default=None, help="num devices if ATTRIBUTES lacks it")
    p.add_argument("--desync-ns", type=float, default=1e8, help="durations above this are excluded (default 1e8)")
    p.add_argument("--csv", metavar="PATH", help="write the per-config table to PATH")
    p.add_argument(
        "--vs-default", action="store_true",
        help="A/B every tuning against the all-defaults tuning of the same config+pkt, within the input report(s)",
    )  # fmt: skip
    p.add_argument(
        "--vs", metavar="TUNING", default=None,
        help="with --vs-default (implied): use this compact tuning string (as in the `tuning` column) as A",
    )  # fmt: skip
    p.add_argument(
        "--min-delta", type=float, default=5.0, metavar="PCT",
        help="--compare / --vs-default: |delta%%| below this is 'noise' regardless of the CIs (default 5.0)",
    )  # fmt: skip
    p.add_argument(
        "--bootstrap", type=int, default=2000, metavar="N",
        help="bootstrap resamples for the 95%% CI of the median (default 2000; 0 = CI collapses to the median)",
    )  # fmt: skip
    p.add_argument(
        "--link-gbps", type=float, default=25.0,
        help="fabric GB/s per link direction (default 25 = the C++ BH constant, reproduces PM IDEAL). "
        "The extra column is eff50%% at the default, else eff@<G>%% at G",
    )  # fmt: skip
    p.add_argument(
        "--aiclk-mhz", type=float, default=DEFAULT_AICLK_MHZ,
        help=f"device AICLK the model rounds cycles through (not in the CSV; default {DEFAULT_AICLK_MHZ}, BH)",
    )  # fmt: skip
    p.add_argument(
        "--num-cores", type=int, default=None,
        help="compute cores for the DRAM term (default: CSV 'AVAILABLE WORKER CORE COUNT', "
        f"else {DEFAULT_NUM_CORES} = BH 11x10)",
    )  # fmt: skip
    args = p.parse_args()

    default_rate = args.link_gbps == 25.0
    alt_gbps = 50.0 if default_rate else args.link_gbps
    alt_label = "eff50%" if default_rate else f"eff@{args.link_gbps:g}%"
    alt_tag = "50" if default_rate else f"at{args.link_gbps:g}"
    model_opts = dict(link_gbps=None, alt_gbps=alt_gbps, aiclk_mhz=args.aiclk_mhz, num_cores=args.num_cores)
    csv_names = {"ideal_alt_us": f"ideal{alt_tag}_us", "eff_alt_pct": f"eff{alt_tag}_pct"}

    run = lambda path: analyze(
        resolve_csv(path), args.warmup, args.devices, args.desync_ns, model_opts, n_boot=args.bootstrap
    )

    vs_mode = args.vs_default or args.vs is not None
    if args.compare and vs_mode:
        p.error("--compare and --vs-default/--vs are mutually exclusive")
    if args.compare:
        if args.inputs:
            p.error("--compare takes its own two inputs; do not also pass positional inputs")
        ra, rb = run(args.compare[0]), run(args.compare[1])
        if ra.empty or rb.empty:
            print("error: zero AllGather rows on one side", file=sys.stderr)
            return 1
        print()
        print_compare(ra, rb, args.compare[0], args.compare[1], args.min_delta)
        if args.csv:
            ja, jb = join_keys(ra[OUT_COLUMNS], rb[OUT_COLUMNS])
            m = ja.merge(jb, on="jkey", how="outer", suffixes=("_a", "_b"))
            m.insert(0, "key", m.pop("jkey"))  # the join key, as before (config when that is unambiguous)
            m.rename(columns={f"{k}{s}": f"{v}{s}" for k, v in csv_names.items() for s in ("_a", "_b")}).to_csv(
                args.csv, index=False
            )
            print(f"\nwrote {args.csv}")
        return 0

    frames = []
    for path in args.inputs or [None]:
        r = run(path)
        if not r.empty:
            r.insert(0, "source", str(resolve_csv(path)))
            frames.append(r)
    if not frames:
        print("error: zero AllGather rows found", file=sys.stderr)
        return 1
    res = pd.concat(frames, ignore_index=True)
    multi = len(frames) > 1
    if multi:
        src_tag = res["source"].map(lambda s: Path(s).parent.name) + ":"
        res["key"] = src_tag + res["key"]
        res["config"] = src_tag + res["config"]
        res = res.sort_values(["out_bytes", "topology", "key"]).reset_index(drop=True)
    print()
    if vs_mode:
        ref = "" if args.vs in (None, "-") else args.vs
        vs, no_ref = vs_tuning(res, ref, args.min_delta)
        print_vs_tuning(vs, no_ref, ref)
        if args.csv and not vs.empty:
            vs.to_csv(args.csv, index=False)
            print(f"\nwrote {args.csv}")
        return 0
    print_table(res, alt_label)
    if args.csv:
        res[OUT_COLUMNS + (["source"] if multi else [])].rename(columns=csv_names).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

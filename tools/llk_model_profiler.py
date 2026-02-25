#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LLK Model Profiler: Correlates Tracy performance CSV with JIT build cache
to produce a per-TTNN-op summary of LLK API calls, data formats, tile sizes,
and configuration parameters for an entire model run.

Usage:
    python tools/llk_model_profiler.py \
        --csv generated/profiler/reports/<date>/ops_perf_results_<date>.csv \
        [--cache-dir ~/.cache/tt-metal-cache/] \
        [--output report.txt] \
        [--format text|json]
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Decode tables
# ---------------------------------------------------------------------------

DATA_FORMAT_MAP = {
    0: "Float32",
    1: "Float16",
    2: "Bfp8",
    3: "Bfp4",
    4: "Tf32",
    5: "Float16_b",
    6: "Bfp8_b",
    7: "Bfp4_b",
    8: "Int32",
    9: "UInt16",
    10: "Lf8",
    11: "Bfp2",
    14: "Int8",
    15: "Bfp2_b",
    24: "UInt32",
    26: "Fp8_e4m3",
    30: "UInt8",
    255: "Invalid",
}

MATH_FIDELITY_MAP = {
    0: "LoFi",
    2: "HiFi2",
    3: "HiFi3",
    4: "HiFi4",
    255: "Invalid",
}

# ---------------------------------------------------------------------------
# Kernel source -> LLK API mapping
# ---------------------------------------------------------------------------

_BIN_RELU = ["llk_unpack_AB", "llk_math_eltwise_binary", "llk_pack", "llk_pack_relu_config"]
_SFPU_BIN = ["llk_unpack_A", "llk_math_eltwise_unary_datacopy", "llk_pack", "llk_pack_relu_config"]
_SFPU_TRN = ["llk_unpack_A", "llk_math_eltwise_unary_datacopy", "llk_pack"]

KERNEL_LLK_MAP: dict[str, list[str]] = {
    "eltwise_binary_kernel.cpp": ["llk_unpack_AB", "llk_math_eltwise_binary", "llk_pack"],
    "eltwise_binary.cpp": ["llk_unpack_AB", "llk_math_eltwise_binary", "llk_pack"],
    "eltwise_binary_no_bcast.cpp": _BIN_RELU,
    "eltwise_binary_scalar.cpp": _BIN_RELU,
    "eltwise_binary_sfpu.cpp": _SFPU_BIN,
    "eltwise_binary_sfpu_no_bcast.cpp": _SFPU_BIN,
    "eltwise_binary_sfpu_scalar.cpp": _SFPU_BIN,
    "eltwise_where_sfpu.cpp": _SFPU_TRN,
    "ternary_sfpu_no_bcast_ttt.cpp": _SFPU_TRN,
    "bmm_large_block_zm_fused_bias_activation.cpp": [
        "llk_unpack_AB_matmul",
        "llk_math_matmul",
        "llk_pack",
    ],
    "bmm_large_block_zm.cpp": [
        "llk_unpack_AB_matmul",
        "llk_math_matmul",
        "llk_pack",
    ],
    "eltwise_sfpu.cpp": [
        "llk_unpack_A",
        "llk_math_unary_sfpu",
        "llk_pack",
    ],
    "conv_bmm_tilize.cpp": [
        "llk_unpack_tilize",
        "llk_math_matmul",
        "llk_pack",
    ],
    "compute_pool_2d.cpp": [
        "llk_unpack_A",
        "llk_math_reduce",
        "llk_pack",
    ],
    "pack_untilize.cpp": [
        "llk_unpack_A",
        "llk_pack_untilize",
    ],
    "reduce.cpp": [
        "llk_unpack_reduce",
        "llk_math_reduce",
        "llk_pack",
    ],
    "eltwise_copy.cpp": [
        "llk_unpack_A",
        "llk_pack",
    ],
    "tilize.cpp": [
        "llk_unpack_tilize",
        "llk_pack",
    ],
    "untilize.cpp": [
        "llk_unpack_A",
        "llk_pack_untilize",
    ],
}

# ---------------------------------------------------------------------------
# SFPU compute-API function -> underlying LLK API
# ---------------------------------------------------------------------------

SFPU_FUNC_TO_LLK: dict[str, str] = {
    "add_binary_tile": "llk_math_eltwise_binary_sfpu_add",
    "add_binary_tile_init": "llk_math_eltwise_binary_sfpu_add_init",
    "sub_binary_tile": "llk_math_eltwise_binary_sfpu_sub",
    "sub_binary_tile_init": "llk_math_eltwise_binary_sfpu_sub_init",
    "mul_binary_tile": "llk_math_eltwise_binary_sfpu_mul",
    "mul_binary_tile_init": "llk_math_eltwise_binary_sfpu_mul_init",
    "div_binary_tile": "llk_math_eltwise_binary_sfpu_div",
    "div_binary_tile_init": "llk_math_eltwise_binary_sfpu_div_init",
    "add_int_tile": "llk_math_eltwise_binary_sfpu_add_int",
    "add_int_tile_init": "llk_math_eltwise_binary_sfpu_add_int_init",
    "mul_int_tile": "llk_math_eltwise_binary_sfpu_mul_int",
    "mul_int_tile_init": "llk_math_eltwise_binary_sfpu_mul_int_init",
    "gt_int32_tile": "llk_math_eltwise_binary_sfpu_gt_int32",
    "gt_int32_tile_init": "llk_math_eltwise_binary_sfpu_gt_int32_init",
    "where_tile": "llk_math_eltwise_binary_sfpu_where",
    "where_tile_init": "llk_math_eltwise_binary_sfpu_where_init",
    "lerp_tile": "llk_math_eltwise_binary_sfpu_lerp",
    "lerp_tile_init": "llk_math_eltwise_binary_sfpu_lerp_init",
    "addcmul_tile": "llk_math_eltwise_binary_sfpu_addcmul",
    "addcmul_tile_init": "llk_math_eltwise_binary_sfpu_addcmul_init",
    "addcdiv_tile": "llk_math_eltwise_binary_sfpu_addcdiv",
    "addcdiv_tile_init": "llk_math_eltwise_binary_sfpu_addcdiv_init",
    **{
        f"{op}_tile": f"llk_math_eltwise_unary_sfpu_{op}"
        for op in (
            "silu",
            "relu",
            "sigmoid",
            "gelu",
            "sqrt",
            "recip",
            "tanh",
            "log",
            "gtz",
            "ltz",
            "eqz",
            "nez",
            "gez",
            "lez",
        )
    },
    **{
        f"{op}_tile_init": f"llk_math_eltwise_unary_sfpu_{op}_init"
        for op in (
            "silu",
            "relu",
            "sigmoid",
            "gelu",
            "sqrt",
            "recip",
            "tanh",
            "log",
            "gtz",
            "ltz",
            "eqz",
            "nez",
            "gez",
            "lez",
        )
    },
    "exp_tile": "llk_math_eltwise_unary_sfpu_exponential",
    "exp_tile_init": "llk_math_eltwise_unary_sfpu_exponential_init",
}

_SFPU_DISPATCH_DEFINES = {
    "BINARY_SFPU_OP",
    "BINARY_SFPU_INIT",
    "TERNARY_SFPU_OP_FUNC",
    "TERNARY_SFPU_OP_INIT",
    "PROCESS_LHS_ACTIVATIONS",
    "PROCESS_RHS_ACTIVATIONS",
    "PROCESS_POST_ACTIVATIONS",
}

_SFPU_FUNC_RE = re.compile(r"\b([a-z][a-z0-9_]*_tile(?:_init)?)\b")


def resolve_sfpu_llk_apis(defines: dict[str, str]) -> list[str]:
    """Map SFPU dispatch defines to their underlying LLK API names."""
    apis: list[str] = []
    seen: set[str] = set()
    for key in _SFPU_DISPATCH_DEFINES:
        val = defines.get(key, "")
        if not val:
            continue
        for m in _SFPU_FUNC_RE.finditer(val):
            llk = SFPU_FUNC_TO_LLK.get(m.group(1))
            if llk and llk not in seen:
                seen.add(llk)
                apis.append(llk)
    return apis


def infer_llk_apis_from_source(kernel_source_path: str, search_root: str) -> list[str]:
    """Fallback: scan the kernel source for llk_* API calls."""
    basename = os.path.basename(kernel_source_path)
    if basename in KERNEL_LLK_MAP:
        return list(KERNEL_LLK_MAP[basename])

    full_path = os.path.join(search_root, kernel_source_path)
    if not os.path.isfile(full_path):
        return [f"<unknown: {basename}>"]

    apis = set()
    llk_pattern = re.compile(r"\b(llk_(?:unpack|math|pack)[a-zA-Z_]*)\b")
    with open(full_path) as f:
        for line in f:
            for match in llk_pattern.finditer(line):
                apis.add(match.group(1))
    return sorted(apis) if apis else [f"<unknown: {basename}>"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CBFormats:
    """Data formats and tile dimensions for a single circular buffer index."""

    index: int
    src_format: str = "Invalid"
    dst_format: str = "Invalid"
    tile_r_dim: int = 0
    tile_c_dim: int = 0
    face_r_dim: int = 0
    num_faces: int = 0
    num_faces_r: int = 0
    num_faces_c: int = 0


@dataclass
class KernelConfig:
    """All extracted LLK-level configuration for one kernel build."""

    kernel_source: str = ""
    kernel_hash: str = ""

    math_fidelity: str = ""
    math_approx: Optional[bool] = None
    dst_accum_mode: Optional[bool] = None
    dst_sync_mode: str = ""

    unpack_cbs: list[CBFormats] = field(default_factory=list)
    pack_cbs: list[CBFormats] = field(default_factory=list)

    defines: dict[str, str] = field(default_factory=dict)
    llk_apis: list[str] = field(default_factory=list)


@dataclass
class OpSummary:
    """Summary for one TTNN operation invocation group."""

    op_code: str = ""
    math_fidelity_csv: str = ""
    attributes: str = ""
    input_datatypes: list[str] = field(default_factory=list)
    output_datatypes: list[str] = field(default_factory=list)
    invocation_count: int = 0
    kernel_configs: list[KernelConfig] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_csv(csv_path: str) -> list[OpSummary]:
    """Parse Tracy CSV and group rows by OP CODE + unique kernel hash set."""

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    groups: dict[str, OpSummary] = {}

    for row in rows:
        op_code = row.get("OP CODE", "").strip()
        if not op_code:
            continue

        compute_hashes_raw = row.get("COMPUTE KERNEL HASH", "").strip()
        if not compute_hashes_raw or compute_hashes_raw == "[]":
            continue

        compute_sources_raw = row.get("COMPUTE KERNEL SOURCE", "").strip()
        math_fidelity = row.get("MATH FIDELITY", "").strip()

        hashes = _parse_list_field(compute_hashes_raw)
        sources = _parse_list_field(compute_sources_raw)

        hash_key = ";".join(sorted(hashes))
        group_key = f"{op_code}|{hash_key}"

        if group_key not in groups:
            input_dts = []
            for i in range(3):
                dt = row.get(f"INPUT_{i}_DATATYPE", "").strip()
                if dt:
                    input_dts.append(dt)
            output_dts = []
            dt = row.get("OUTPUT_0_DATATYPE", "").strip()
            if dt:
                output_dts.append(dt)

            summary = OpSummary(
                op_code=op_code,
                math_fidelity_csv=math_fidelity,
                attributes=row.get("ATTRIBUTES", "").strip(),
                input_datatypes=input_dts,
                output_datatypes=output_dts,
                invocation_count=1,
            )

            for src, h in zip(sources, hashes):
                kc = KernelConfig(kernel_source=src, kernel_hash=h)
                summary.kernel_configs.append(kc)

            if len(hashes) > len(sources):
                for h in hashes[len(sources) :]:
                    summary.kernel_configs.append(KernelConfig(kernel_hash=h))

            groups[group_key] = summary
        else:
            groups[group_key].invocation_count += 1

    return list(groups.values())


def _parse_list_field(raw: str) -> list[str]:
    """Parse a CSV field like "['a'; 'b']" into a list of strings."""
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    parts = re.split(r"[;,]\s*", raw)
    result = []
    for p in parts:
        p = p.strip().strip("'\"")
        if p:
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Build cache parsing
# ---------------------------------------------------------------------------


def find_cache_dir(cache_root: str, build_key: Optional[str] = None) -> Optional[str]:
    """Find the build cache base directory (git_hash/build_key/kernels/).

    If *build_key* is given it is treated as a ``git_hash/build_key`` path
    fragment (e.g. ``90f18bad94/15603236145992786089``) and is resolved
    directly under *cache_root*.  Otherwise the first valid
    ``git_hash/build_key/kernels/`` discovered automatically is returned.
    """
    cache_root = os.path.expanduser(cache_root)
    if not os.path.isdir(cache_root):
        return None

    if build_key:
        candidate = os.path.join(cache_root, build_key.strip("/"), "kernels")
        if os.path.isdir(candidate):
            return candidate
        candidate = os.path.join(cache_root, build_key.strip("/"))
        if os.path.isdir(candidate):
            return candidate
        return None

    for git_hash in os.listdir(cache_root):
        git_dir = os.path.join(cache_root, git_hash)
        if not os.path.isdir(git_dir):
            continue
        for bk in os.listdir(git_dir):
            kernels_dir = os.path.join(git_dir, bk, "kernels")
            if os.path.isdir(kernels_dir):
                return kernels_dir
    return None


def find_kernel_dir(kernels_base: str, kernel_hash: str) -> Optional[str]:
    """Resolve a kernel hash like 'eltwise_binary_kernel/1162.../' to full path."""
    kernel_hash = kernel_hash.strip().rstrip("/")
    candidate = os.path.join(kernels_base, kernel_hash)
    if os.path.isdir(candidate):
        return candidate
    return None


def parse_descriptors(descriptor_path: str) -> dict:
    """Parse chlkc_descriptors.h and extract all arrays and scalar values."""
    result = {
        "math_fidelity_raw": None,
        "approx": None,
        "dst_accum_mode": None,
        "dst_sync_mode": "",
        "unpack": {},
        "pack": {},
    }

    if not os.path.isfile(descriptor_path):
        return result

    with open(descriptor_path) as f:
        content = f.read()

    mf = re.search(r"MathFidelity\s*=\s*static_cast<ckernel::MathFidelity>\((\d+)\)", content)
    if mf:
        result["math_fidelity_raw"] = int(mf.group(1))

    approx = re.search(r"APPROX\s*=\s*(true|false)", content)
    if approx:
        result["approx"] = approx.group(1) == "true"

    dst_accum = re.search(r"DST_ACCUM_MODE\s*=\s*(true|false)", content)
    if dst_accum:
        result["dst_accum_mode"] = dst_accum.group(1) == "true"

    dst_sync = re.search(r"#define\s+DST_SYNC_MODE\s+DstSync::(\w+)", content)
    if dst_sync:
        result["dst_sync_mode"] = dst_sync.group(1)

    array_pattern = re.compile(
        r"constexpr\s+(?:std::int32_t|unsigned\s+char|uint8_t|uint16_t)\s+"
        r"((?:unpack|pack)_\w+)\[32\]\s*=\s*\{([^}]+)\}"
    )
    for m in array_pattern.finditer(content):
        name = m.group(1)
        values = [int(v.strip()) for v in m.group(2).split(",")]
        prefix = "unpack" if name.startswith("unpack") else "pack"
        short_name = name[len(prefix) + 1 :]
        result[prefix][short_name] = values

    return result


def parse_defines(defines_path: str) -> dict[str, str]:
    """Parse defines_generated.h into a dict."""
    defines = {}
    if not os.path.isfile(defines_path):
        return defines

    with open(defines_path) as f:
        for line in f:
            m = re.match(r"#define\s+(\w+)\s+(.*)", line.strip())
            if m:
                defines[m.group(1)] = m.group(2).strip()
    return defines


def build_cb_formats(arrays: dict[str, list[int]], prefix_map: dict[str, str]) -> list[CBFormats]:
    """Build a list of CBFormats for active (non-255/non-0) circular buffers."""
    src_fmt = arrays.get("src_format", [])
    dst_fmt = arrays.get("dst_format", [])
    tile_r = arrays.get("tile_r_dim", [])
    tile_c = arrays.get("tile_c_dim", [])
    face_r = arrays.get("tile_face_r_dim", [])
    num_faces_arr = arrays.get("tile_num_faces", [])
    num_faces_r = arrays.get("num_faces_r_dim", [])
    num_faces_c = arrays.get("num_faces_c_dim", [])

    cbs = []
    for i in range(min(32, len(src_fmt) if src_fmt else 0)):
        sf = src_fmt[i] if i < len(src_fmt) else 255
        df = dst_fmt[i] if i < len(dst_fmt) else 255
        if sf == 255 and df == 255:
            continue
        cb = CBFormats(
            index=i,
            src_format=DATA_FORMAT_MAP.get(sf, f"Unknown({sf})"),
            dst_format=DATA_FORMAT_MAP.get(df, f"Unknown({df})"),
            tile_r_dim=tile_r[i] if i < len(tile_r) else 0,
            tile_c_dim=tile_c[i] if i < len(tile_c) else 0,
            face_r_dim=face_r[i] if i < len(face_r) else 0,
            num_faces=num_faces_arr[i] if i < len(num_faces_arr) else 0,
            num_faces_r=num_faces_r[i] if i < len(num_faces_r) else 0,
            num_faces_c=num_faces_c[i] if i < len(num_faces_c) else 0,
        )
        cbs.append(cb)
    return cbs


def enrich_kernel_config(kc: KernelConfig, kernels_base: str, search_root: str) -> None:
    """Fill in a KernelConfig from the build cache and kernel source."""
    kernel_dir = find_kernel_dir(kernels_base, kc.kernel_hash) if kernels_base else None

    if kernel_dir:
        desc = parse_descriptors(os.path.join(kernel_dir, "chlkc_descriptors.h"))
        kc.defines = parse_defines(os.path.join(kernel_dir, "defines_generated.h"))

        raw_mf = desc.get("math_fidelity_raw")
        if raw_mf is not None:
            kc.math_fidelity = MATH_FIDELITY_MAP.get(raw_mf, f"Unknown({raw_mf})")
        kc.math_approx = desc.get("approx")
        kc.dst_accum_mode = desc.get("dst_accum_mode")
        kc.dst_sync_mode = desc.get("dst_sync_mode", "")

        kc.unpack_cbs = build_cb_formats(desc.get("unpack", {}), {})
        kc.pack_cbs = build_cb_formats(desc.get("pack", {}), {})

    kc.llk_apis = infer_llk_apis_from_source(kc.kernel_source, search_root)
    sfpu_apis = resolve_sfpu_llk_apis(kc.defines)
    if sfpu_apis:
        existing = set(kc.llk_apis)
        kc.llk_apis.extend(a for a in sfpu_apis if a not in existing)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_cb_line(cb: CBFormats) -> str:
    tile = f"{cb.tile_r_dim}x{cb.tile_c_dim}"
    return (
        f"CB{cb.index}: src={cb.src_format}, dst={cb.dst_format}, "
        f"tile={tile}, faces={cb.num_faces} ({cb.num_faces_r}r x {cb.num_faces_c}c), "
        f"face_r={cb.face_r_dim}"
    )


def format_text(summaries: list[OpSummary]) -> str:
    lines = []
    op_groups: dict[str, list[OpSummary]] = defaultdict(list)
    for s in summaries:
        op_groups[s.op_code].append(s)

    for op_code, group in op_groups.items():
        total_invocations = sum(s.invocation_count for s in group)
        lines.append("=" * 80)
        lines.append(f"TTNN OP: {op_code}  " f"({len(group)} unique config(s), {total_invocations} total invocations)")
        lines.append("=" * 80)

        for cfg_idx, summary in enumerate(group, 1):
            lines.append("")
            lines.append(f"  Config {cfg_idx} ({summary.invocation_count} invocations):")
            lines.append(f"    Math Fidelity (CSV): {summary.math_fidelity_csv}")
            lines.append(f"    Input Datatypes:  {', '.join(summary.input_datatypes) or 'N/A'}")
            lines.append(f"    Output Datatypes: {', '.join(summary.output_datatypes) or 'N/A'}")

            for kc in summary.kernel_configs:
                lines.append("")
                src_basename = os.path.basename(kc.kernel_source) if kc.kernel_source else "N/A"
                lines.append(f"    Kernel: {src_basename}")
                lines.append(f"    Hash:   {kc.kernel_hash}")

                if kc.defines:
                    defines_str = ", ".join(f"{k}={v}" for k, v in kc.defines.items())
                    lines.append(f"    Defines: {defines_str}")

                if kc.math_fidelity:
                    lines.append(f"    Math Fidelity (HW): {kc.math_fidelity}")
                if kc.math_approx is not None:
                    lines.append(f"    Math Approx Mode:   {kc.math_approx}")
                if kc.dst_accum_mode is not None:
                    lines.append(f"    FP32 Dest Accum:    {'yes' if kc.dst_accum_mode else 'no'}")
                if kc.dst_sync_mode:
                    lines.append(f"    Dst Sync Mode:      {kc.dst_sync_mode}")

                if kc.llk_apis:
                    lines.append(f"    LLK APIs: {', '.join(kc.llk_apis)}")

                if kc.unpack_cbs:
                    lines.append("    Unpack CBs:")
                    for cb in kc.unpack_cbs:
                        lines.append(f"      {format_cb_line(cb)}")

                if kc.pack_cbs:
                    lines.append("    Pack CBs:")
                    for cb in kc.pack_cbs:
                        lines.append(f"      {format_cb_line(cb)}")

        lines.append("")

    return "\n".join(lines)


def _fmt_set(values: set) -> str:
    """Format a set of values into a compact string."""
    if not values:
        return ""
    sorted_vals = sorted(str(v) for v in values)
    return ", ".join(sorted_vals)


def _collect_api_stats(summaries: list[OpSummary]) -> dict:
    """Collect per-LLK-API statistics across all op configs."""

    @dataclass
    class APIStat:
        total_invocations: int = 0
        config_count: int = 0
        op_codes: set = field(default_factory=set)
        math_fidelities: set = field(default_factory=set)
        math_approx: set = field(default_factory=set)
        fp32_dest_accum: set = field(default_factory=set)
        dst_sync_modes: set = field(default_factory=set)
        unpack_src_formats: set = field(default_factory=set)
        unpack_dst_formats: set = field(default_factory=set)
        pack_src_formats: set = field(default_factory=set)
        pack_dst_formats: set = field(default_factory=set)
        tile_dims: set = field(default_factory=set)
        face_r_dims: set = field(default_factory=set)
        num_faces: set = field(default_factory=set)
        num_faces_r: set = field(default_factory=set)
        num_faces_c: set = field(default_factory=set)
        defines_keys: set = field(default_factory=set)
        define_values: dict = field(default_factory=lambda: defaultdict(set))

    api_stats: dict[str, APIStat] = defaultdict(APIStat)

    for summary in summaries:
        for kc in summary.kernel_configs:
            for api in kc.llk_apis:
                if api.startswith("<unknown"):
                    continue
                st = api_stats[api]
                st.total_invocations += summary.invocation_count
                st.config_count += 1
                st.op_codes.add(summary.op_code)

                if kc.math_fidelity:
                    st.math_fidelities.add(kc.math_fidelity)
                if kc.math_approx is not None:
                    st.math_approx.add(kc.math_approx)
                if kc.dst_accum_mode is not None:
                    st.fp32_dest_accum.add(kc.dst_accum_mode)
                if kc.dst_sync_mode:
                    st.dst_sync_modes.add(kc.dst_sync_mode)

                for cb in kc.unpack_cbs:
                    st.unpack_src_formats.add(cb.src_format)
                    st.unpack_dst_formats.add(cb.dst_format)
                    st.tile_dims.add(f"{cb.tile_r_dim}x{cb.tile_c_dim}")
                    st.face_r_dims.add(cb.face_r_dim)
                    st.num_faces.add(cb.num_faces)
                    st.num_faces_r.add(cb.num_faces_r)
                    st.num_faces_c.add(cb.num_faces_c)

                for cb in kc.pack_cbs:
                    st.pack_src_formats.add(cb.src_format)
                    st.pack_dst_formats.add(cb.dst_format)

                for k, v in kc.defines.items():
                    st.defines_keys.add(k)
                    st.define_values[k].add(v)

    return api_stats


OP_ARG_DEFINES = {
    "ELTWISE_OP",
    "ELTWISE_OP_TYPE",
    "REDUCE_DIM",
    "REDUCE_OP",
    "SFPU_OP_CHAIN_0",
    "SFPU_OP_INIT_0",
    "BINARY_SFPU_OP",
    "BINARY_SFPU_INIT",
    "BINARY_OP",
    "BINARY_OP_TYPE",
    "TERNARY_SFPU_OP_FUNC",
    "TERNARY_SFPU_OP_INIT",
    "PROCESS_LHS_ACTIVATIONS",
    "PROCESS_RHS_ACTIVATIONS",
    "PROCESS_POST_ACTIVATIONS",
}


def write_summary_md(summaries: list[OpSummary], md_path: str) -> None:
    """Write a Markdown table summary with one row per LLK API."""
    api_stats = _collect_api_stats(summaries)

    headers = [
        "LLK API",
        "Configs",
        "Total Invocations",
        "TTNN Ops",
        "Op Args",
        "Input Data Formats",
        "Output Data Formats",
        "Tile Dims",
        "Math Fidelity",
        "Math Approx",
        "FP32 Dest Accum",
        "Dst Sync Mode",
        "Kernel Defines",
    ]

    rows = []
    for api in sorted(api_stats.keys()):
        st = api_stats[api]

        op_arg_parts = []
        generic_define_parts = []
        for k in sorted(st.defines_keys):
            vals = _fmt_set(st.define_values[k])
            if k in OP_ARG_DEFINES:
                op_arg_parts.append(f"{k}={vals}")
            elif k != "NOC_MODE":
                generic_define_parts.append(f"{k}={vals}")

        is_sfpu = "sfpu" in api.lower()
        math_approx_str = _fmt_set(st.math_approx) if is_sfpu else "False"

        if api.startswith("llk_unpack"):
            input_fmts = _fmt_set(st.unpack_src_formats)
            output_fmts = _fmt_set(st.unpack_dst_formats)
        elif api.startswith("llk_math"):
            input_fmts = _fmt_set(st.unpack_dst_formats)
            output_fmts = _fmt_set(st.pack_src_formats)
        else:
            input_fmts = _fmt_set(st.pack_src_formats)
            output_fmts = _fmt_set(st.pack_dst_formats)

        rows.append(
            [
                f"`{api}`",
                str(st.config_count),
                str(st.total_invocations),
                _fmt_set(st.op_codes),
                "; ".join(op_arg_parts),
                input_fmts,
                output_fmts,
                _fmt_set(st.tile_dims),
                _fmt_set(st.math_fidelities) or "LoFi",
                math_approx_str or "False",
                _fmt_set(st.fp32_dest_accum),
                _fmt_set(st.dst_sync_modes),
                "; ".join(generic_define_parts),
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        padded = [c.ljust(col_widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    lines = [
        "# LLK API Cross-Model Summary",
        "",
        _fmt_row(headers),
        "| " + " | ".join("-" * w for w in col_widths) + " |",
    ]
    for row in rows:
        lines.append(_fmt_row(row))
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def format_json(summaries: list[OpSummary]) -> str:
    data = []
    for s in summaries:
        entry = {
            "op_code": s.op_code,
            "math_fidelity": s.math_fidelity_csv,
            "input_datatypes": s.input_datatypes,
            "output_datatypes": s.output_datatypes,
            "invocation_count": s.invocation_count,
            "kernel_configs": [],
        }
        for kc in s.kernel_configs:
            kc_dict = {
                "kernel_source": kc.kernel_source,
                "kernel_hash": kc.kernel_hash,
                "math_fidelity_hw": kc.math_fidelity,
                "math_approx": kc.math_approx,
                "fp32_dest_accum": kc.dst_accum_mode,
                "dst_sync_mode": kc.dst_sync_mode,
                "defines": kc.defines,
                "llk_apis": kc.llk_apis,
                "unpack_cbs": [asdict(cb) for cb in kc.unpack_cbs],
                "pack_cbs": [asdict(cb) for cb in kc.pack_cbs],
            }
            entry["kernel_configs"].append(kc_dict)
        data.append(entry)
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLK API usage across a model run by correlating "
        "Tracy perf CSV with JIT build cache artifacts.",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to ops_perf_results_*.csv from Tracy profiler",
    )
    parser.add_argument(
        "--cache-dir",
        default="~/.cache/tt-metal-cache/",
        help="Root of the JIT build cache (default: ~/.cache/tt-metal-cache/)",
    )
    parser.add_argument(
        "--build-key",
        default=None,
        help="Explicit git_hash/build_key path under --cache-dir "
        "(e.g. '90f18bad94/15603236145992786089'). "
        "When omitted the first valid build key is auto-detected.",
    )
    parser.add_argument(
        "--source-root",
        default=".",
        help="Root of the tt-metal source tree for kernel source lookup (default: cwd)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Write a Markdown table summary of unique LLK API parameter combinations to this file",
    )
    args = parser.parse_args()

    csv_path = os.path.expanduser(args.csv)
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing CSV: {csv_path}", file=sys.stderr)
    summaries = parse_csv(csv_path)
    print(f"Found {len(summaries)} unique op configurations", file=sys.stderr)

    kernels_base = find_cache_dir(args.cache_dir, args.build_key)
    if kernels_base:
        print(f"Using build cache: {kernels_base}", file=sys.stderr)
    else:
        print(
            f"Warning: No build cache found at {args.cache_dir}. " "Output will have limited detail.",
            file=sys.stderr,
        )

    source_root = os.path.expanduser(args.source_root)
    for summary in summaries:
        for kc in summary.kernel_configs:
            enrich_kernel_config(kc, kernels_base, source_root)

    if args.format == "json":
        output = format_json(summaries)
    else:
        output = format_text(summaries)

    if args.summary:
        summary_path = os.path.expanduser(args.summary)
        write_summary_md(summaries, summary_path)
        print(f"Summary written to: {summary_path}", file=sys.stderr)

    if args.output:
        out_path = os.path.expanduser(args.output)
        with open(out_path, "w") as f:
            f.write(output)
        print(f"Report written to: {out_path}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

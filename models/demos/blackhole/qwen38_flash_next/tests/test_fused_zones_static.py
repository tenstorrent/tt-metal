# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of the fused kernels' phase zones (the study build's device profiler zones): one header with one
define, every kernel source marks its phases through it with per-kernel names, and nothing is compiled in -- the
kernel descriptors are the served build's, byte for byte -- unless the environment says ``QWEN38_FUSED_ZONES=1``."""

import hashlib
import inspect
import re
from pathlib import Path

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail

FUSED = Path(fp.__file__).parent
HEADER = FUSED / "kernels" / "zones.h"
# Keep Samuel Jett's frozen cd9 source chunk files byte-identical. They retain
# their original profiling zones rather than the optional model zone macro.
SOURCE_CHUNK_SHA256 = {
    "compute_prep.cpp": "922d2b203e2d43c369e320cdc5a186da115d808da11dea644e2dce53652eff6d",
    "reader_prep.cpp": "a125d1141506f18150eef80b6e6dc60fc5c14615fa9c637eca144bf7590be300",
    "writer_prep.cpp": "1bca1d55b8f0a335760d16784df7c1983ff7a413e5bc4a20bc387797fe67bec2",
    "compute_scan.cpp": "8ab5fa5d85585c7325be1a4b0bc1b6c50416d870738a4e812351076502b5a82b",
    "reader_scan.cpp": "8c6dce017af78ede0c4c6ebeaa6e5e2ba8c496b3a59eced8fdd62655c823daf0",
    "writer_scan.cpp": "1c515df4f939a07b94d935e63d72a1da21ae01dcda52768441d83e0f6d709ea8",
}
ALL_KERNELS = sorted(FUSED.glob("*/kernels/*.cpp"))
KERNELS = [p for p in ALL_KERNELS if p.parent.parent.name != "gdn_source_chunk"]
BUILDERS = sorted(FUSED.glob("**/*.py"))
INCLUDE = '#include "../../kernels/zones.h"'
ZONE = re.compile(r'FUSED_ZONE\("([^"]+)"\)')
NAME = re.compile(r"^fz_[a-z0-9_]+$")
KERNEL_FILES = 178
ZONES = 414  # zone marks over the kernels; a variant program's copy (gdn_step conv_split, gr_recip_last front2, ...) reuses its original's names
# Sources that run no phase of the served decode (microbenchmark / idle / empty stand-ins and the KV page scatter): no zones, no include.
FUNCTION_SCOPE_ZONE = {  # kernels added after the one-block-per-zone rule: a zone that does not follow a "{" line (the kernel keeps its own scoping)
    "gdn_step/kernels/conv_reader.cpp",
    "gdn_step/kernels/conv_writer.cpp",
    "gr_recip_last/kernels/front_reader.cpp",
    "gr_recip_last/kernels/front_reader3.cpp",
    "gr_recip_last/kernels/raw_writer.cpp",
    "gr_recip_last/kernels/raw_writer3.cpp",
    "gr_recip_last/kernels/transport4.cpp",
    "mm_stream/kernels/compute_mm_stream.cpp",
    "mm_stream/kernels/hub_mm_stream.cpp",
    "mm_stream/kernels/reader_mm_stream.cpp",
    "mm_stream/kernels/shard_mm_stream.cpp",
    "mm_stream/kernels/writer_mm_stream.cpp",
    "mm_stream2/kernels/hub_mm_stream2.cpp",
    "mm_stream2/kernels/reader_mm_stream2.cpp",
    "ple/kernels/gate_prod_compute.cpp",
    "ple/kernels/gate_prod_reader.cpp",
    "ple/kernels/gate_prod_writer.cpp",
    "router_tail/kernels/compute_router_tail.cpp",
    "router_tail_sc/kernels/compute_rt_sort.cpp",
    "router_tail_sc/kernels/compute_rtf_sort.cpp",
    "router_tail_sc/kernels/writer_rtf_index.cpp",
    "row_reduce/kernels/compute3.cpp",
}
ZONELESS = {
    "kv_page_scatter/kernels/scatter_rows.cpp",
    "mm_stream/kernels/bench_mm_compute.cpp",
    "mm_stream/kernels/idle_mm_stream.cpp",
    "mm_stream2/kernels/empty_mm_stream2.cpp",
    "moe_experts/kernels/bench_compute_moe_experts.cpp",
    "moe_experts/kernels/bench_reader_moe_experts.cpp",
}
# every kernel's zone names carry its prefix (the census table's phase column reads them)
PREFIX = {
    "final_mixer": "fz_fm_",
    "gdn_post_rows": "fz_gpo_",
    "gdn_pre_rows": "fz_gpr_",
    "gdn_rows_scan": "fz_gsc_",
    "gdn_step": ("fz_gs_", "fz_gv_", "fz_gcv_"),
    "gr_fold": "fz_gf_",
    "gr_read": "fz_gr_",
    "gr_recip_last": ("fz_gl_", "fz_gr_", "fz_gr2_"),  # front2 / front3 reuse the gr_read and row_gather2 names
    "gr_write": "fz_gw_",
    "greedy_tail": "fz_gt_",
    "moe_combine": "fz_mc_",
    "moe_dense": "fz_md_",
    "moe_post": "fz_mp_",
    "mtp_accept": "fz_ma_",
    "ple": "fz_pl_",
    "position_derive": "fz_pd_",
    "qsa_block": "fz_qs_",
    "qsa_rows": "fz_qr_",
    "qsa_trim": "fz_qt_",
    "moe_experts": "fz_me_",
    "mm_stream": "fz_mm_",
    "mm_stream2": "fz_mm_",
    "router_tail": "fz_rt_",
    "router_tail_sc": "fz_rt_",
    "row_gather": "fz_rg_",
    "row_gather2": "fz_rg2_",
    "row_gather3": "fz_rg3_",
    "row_reduce": "fz_rr_",
    "sampler_tail": "fz_st_",
    "shared_expert": "fz_se_",
    "sparse_sdpa_tiled": "fz_ss_",
    "untilize_rows": "fz_ur_",
}
OLD_SWITCHES = ("FMP_ZONES", "FGS_ZONES", "QWEN38_MOE_POST_ZONES", "QWEN38_GDN_STEP_ZONES", "FMP_ZONE(", "FGS_ZONE(")


def _code(text: str) -> str:
    """The source without comments and string literals (for brace counting)."""

    return re.sub(r'"(?:\\.|[^"\\])*"', "", re.sub(r"//.*", "", text))


def test_header_is_one_define_that_compiles_to_nothing_without_the_switch():
    text = HEADER.read_text()
    assert "#pragma once" in text
    order = [
        text.index("#ifdef QWEN38_FUSED_ZONES"),
        text.index('#include "api/debug/kernel_profiler.hpp"'),
        text.index("#define FUSED_ZONE(name) DeviceZoneScopedN(name)"),
        text.index("#else"),
        text.index("#define FUSED_ZONE(name)\n#endif"),
    ]
    assert order == sorted(order)
    assert text.count("#define FUSED_ZONE(name)") == 2 and _code(text).count("DeviceZoneScopedN") == 1
    assert fp.ZONES_DEFINE == "QWEN38_FUSED_ZONES" == fp.ZONES_ENV


def test_every_kernel_marks_its_phases_through_the_header():
    zoned = [p for p in KERNELS if str(p.relative_to(FUSED)) not in ZONELESS]
    assert len(KERNELS) == KERNEL_FILES and set(PREFIX) == {p.parent.parent.name for p in zoned}
    for path in KERNELS:
        if path not in zoned:  # a zoneless source marks nothing and includes nothing
            text = path.read_text()
            assert INCLUDE not in text and "FUSED_ZONE(" not in text, path
    names: list[str] = []
    for path in zoned:
        text = path.read_text()
        kernel = path.parent.parent.name
        prefixes = PREFIX[kernel] if isinstance(PREFIX[kernel], tuple) else (PREFIX[kernel],)
        assert INCLUDE in text, path
        assert "DeviceZoneScopedN" not in text and "kernel_profiler.hpp" not in text, path
        assert not any(token in text for token in OLD_SWITCHES), path
        found = ZONE.findall(text)
        assert found, f"{path} marks no phase"
        for name in found:
            assert NAME.match(name) and name.startswith(prefixes), (path, name)
        # one zone per block: the macro declares `zone`, so every FUSED_ZONE opens a block of its own
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "FUSED_ZONE(" in line:
                assert ZONE.fullmatch(re.sub(r"\s*//.*", "", line).strip().rstrip(";")), (
                    path,
                    line,
                )  # a trailing comment is fine
                previous = next(
                    l for l in reversed(lines[:i]) if l.strip() and not l.lstrip().startswith("#")
                )  # skip #if/#endif lines
                opens_block = re.sub(r"//.*", "", previous).rstrip().endswith("{")
                # the listed kernels (added after this rule) scope their zones themselves
                assert opens_block or (str(path.relative_to(FUSED)) in FUNCTION_SCOPE_ZONE), (
                    path,
                    i + 1,
                    previous,
                )
        code = _code(text)
        assert code.count("{") == code.count("}") or "#else" in text, path  # an #if/#else pair may open one brace twice
        names.extend(found)
    assert len(names) == ZONES


def test_source_chunk_keeps_exact_pinned_kernel_bytes():
    paths = [p for p in ALL_KERNELS if p.parent.parent.name == "gdn_source_chunk"]
    assert {p.name for p in paths} == set(SOURCE_CHUNK_SHA256)
    for path in paths:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_CHUNK_SHA256[path.name]


def test_zone_defines_follow_the_environment_switch():
    assert fp.zone_defines({}) == []
    assert fp.zone_defines({"QWEN38_FUSED_ZONES": "0"}) == []
    assert fp.zone_defines({"QWEN38_FUSED_ZONES": "1"}) == [("QWEN38_FUSED_ZONES", "1")]
    source = inspect.getsource(fp._kernel)
    assert "defines=[*defines, *zone_defines()]" in source
    builder = inspect.getsource(router_tail.program_parts)  # the descriptors' builder (router_tail_program wraps it)
    assert (
        "defines=fp.zone_defines()," in builder
        and "compute.defines = _compute_defines() + fp.zone_defines()" in builder
    )


def test_descriptors_are_byte_identical_with_the_switch_unset(monkeypatch):
    captured = []

    class Descriptor:
        SourceType = ttnn.KernelDescriptor.SourceType

        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(ttnn, "KernelDescriptor", Descriptor)
    monkeypatch.delenv("QWEN38_FUSED_ZONES", raising=False)
    fp.reader_kernel("a.cpp", "cores", [1, 2], [("c", [3])], defines=[("A", "1")])
    fp.compute_kernel("b.cpp", "cores", [4], [], named={"n": 5})
    assert captured[0]["defines"] == [("A", "1")] and captured[1]["defines"] == []  # the served build: nothing added
    monkeypatch.setenv("QWEN38_FUSED_ZONES", "1")
    fp.reader_kernel("a.cpp", "cores", [1, 2], [("c", [3])], defines=[("A", "1")])
    fp.writer_kernel("w.cpp", "cores", [], [])
    assert captured[2]["defines"] == [("A", "1"), ("QWEN38_FUSED_ZONES", "1")]
    assert captured[3]["defines"] == [("QWEN38_FUSED_ZONES", "1")]
    for kwargs in captured:  # the zone define is the only difference
        assert kwargs["kernel_source"] in ("a.cpp", "b.cpp", "w.cpp") and kwargs["core_ranges"] == "cores"
    assert captured[0]["compile_time_args"] == captured[2]["compile_time_args"] == [1, 2]
    assert captured[0]["runtime_args"] == captured[2]["runtime_args"] == [("c", [3])]


def test_no_builder_keeps_the_old_per_kernel_switches():
    sites = []
    for path in BUILDERS:
        text = path.read_text()
        assert not any(token in text for token in OLD_SWITCHES), path
        if "ttnn.KernelDescriptor(" in text:
            sites.append(path.name if path.name != "__init__.py" else path.parent.name)
    assert sorted(sites) == [
        "program.py",
        "router_tail",
        "router_tail_sc",
    ]  # every descriptor is built where the zone define is added

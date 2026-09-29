# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of the fused kernels' phase zones (the study build's device profiler zones): one header with one
define, every kernel source marks its phases through it with per-kernel names, and nothing is compiled in -- the
kernel descriptors are the served build's, byte for byte -- unless the environment says ``QWEN38_FUSED_ZONES=1``."""

import inspect
import re
from pathlib import Path

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import router_tail

FUSED = Path(fp.__file__).parent
HEADER = FUSED / "kernels" / "zones.h"
KERNELS = sorted(FUSED.glob("*/kernels/*.cpp"))
BUILDERS = sorted(FUSED.glob("**/*.py"))
INCLUDE = '#include "../../kernels/zones.h"'
ZONE = re.compile(r'FUSED_ZONE\("([^"]+)"\)')
NAME = re.compile(r"^fz_[a-z0-9_]+$")
KERNEL_FILES = 110
ZONES = 248
# every kernel's zone names carry its prefix (the census table's phase column reads them)
PREFIX = {
    "final_mixer": "fz_fm_",
    "gdn_post_rows": "fz_gpo_",
    "gdn_pre_rows": "fz_gpr_",
    "gdn_rows_scan": "fz_gsc_",
    "gdn_step": "fz_gs_",
    "gr_fold": "fz_gf_",
    "gr_read": "fz_gr_",
    "gr_recip_last": "fz_gl_",
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
    "router_tail": "fz_rt_",
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
        text.index('#include "tools/profiler/kernel_profiler.hpp"'),
        text.index("#define FUSED_ZONE(name) DeviceZoneScopedN(name)"),
        text.index("#else"),
        text.index("#define FUSED_ZONE(name)\n#endif"),
    ]
    assert order == sorted(order)
    assert text.count("#define FUSED_ZONE(name)") == 2 and _code(text).count("DeviceZoneScopedN") == 1
    assert fp.ZONES_DEFINE == "QWEN38_FUSED_ZONES" == fp.ZONES_ENV


def test_every_kernel_marks_its_phases_through_the_header():
    assert len(KERNELS) == KERNEL_FILES and set(PREFIX) == {p.parent.parent.name for p in KERNELS}
    names: list[str] = []
    for path in KERNELS:
        text = path.read_text()
        kernel = path.parent.parent.name
        assert INCLUDE in text, path
        assert "DeviceZoneScopedN" not in text and "kernel_profiler.hpp" not in text, path
        assert not any(token in text for token in OLD_SWITCHES), path
        found = ZONE.findall(text)
        assert found, f"{path} marks no phase"
        for name in found:
            assert NAME.match(name) and name.startswith(PREFIX[kernel]), (path, name)
        # one zone per block: the macro declares `zone`, so every FUSED_ZONE opens a block of its own
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "FUSED_ZONE(" in line:
                assert ZONE.fullmatch(line.strip().rstrip(";")), (path, line)
                previous = next(l for l in reversed(lines[:i]) if l.strip())
                assert re.sub(r"//.*", "", previous).rstrip().endswith("{"), (path, i + 1, previous)
        code = _code(text)
        assert code.count("{") == code.count("}"), path
        names.extend(found)
    assert len(names) == ZONES and len(set(names)) == ZONES, "zone names are distinct over the kernels"


def test_zone_defines_follow_the_environment_switch():
    assert fp.zone_defines({}) == []
    assert fp.zone_defines({"QWEN38_FUSED_ZONES": "0"}) == []
    assert fp.zone_defines({"QWEN38_FUSED_ZONES": "1"}) == [("QWEN38_FUSED_ZONES", "1")]
    source = inspect.getsource(fp._kernel)
    assert "defines=[*defines, *zone_defines()]" in source
    builder = inspect.getsource(router_tail.program_parts)  # the descriptors' builder (router_tail_program wraps it)
    assert "defines=fp.zone_defines()," in builder and "compute.defines = _dev_defines() + fp.zone_defines()" in builder


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
    assert sorted(sites) == ["program.py", "router_tail"]  # every descriptor is built where the zone define is added

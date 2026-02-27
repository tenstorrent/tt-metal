# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generalized fused-kernel micro-op profiler."""

import json
import sys
from pathlib import Path

import pytest

PROFILING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROFILING_DIR))

from fused_kernel_profiler import (
    ClassifiedInterval,
    CoreRoleConfig,
    KernelConfig,
    MicroOpConfig,
    OpType,
    Participant,
    ZoneInterval,
    _adjusted_max_duration,
    _adjusted_span_cycles,
    _trisc_pipeline_wall_clock,
    _uniform_gate,
    classify_intervals,
    compute_critical_path,
    compute_op_duration,
    compute_predecessor_gates,
    cycles_to_ns,
    extract_chip_freq_mhz,
    format_json_report,
    format_text_report,
    infer_core_roles,
    load_kernel_config,
    parse_device_log,
    risc_matches,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iv(core, risc, zone, start, end, chip=0, tid=1):
    """Shorthand to create a ZoneInterval."""
    return ZoneInterval(
        chip_id=chip,
        core=core,
        risc=risc,
        zone_name=zone,
        start_cycle=start,
        end_cycle=end,
        timer_id=tid,
    )


def _ci(iv, func="unknown", role="unknown"):
    """Shorthand to create a ClassifiedInterval."""
    return ClassifiedInterval(
        interval=iv,
        micro_op_zone=iv.zone_name,
        function=func,
        core_role=role,
    )


def _simple_config():
    """Minimal config with RMSNORM (compute), MCAST, MATMUL, GATHER."""
    return KernelConfig(
        kernel_name="test",
        core_roles={
            "input_core": CoreRoleConfig(infer_from_zone="RMSNORM"),
            "matmul_core": CoreRoleConfig(infer_from_zone="MATMUL"),
        },
        micro_ops=[
            MicroOpConfig(
                zone="RMSNORM",
                type=OpType.COMPUTE,
                participants=[Participant("input_core", "TRISC", "compute")],
            ),
            MicroOpConfig(
                zone="MCAST",
                type=OpType.MCAST,
                participants=[
                    Participant("input_core", "BRISC", "sender"),
                    Participant("matmul_core", "NCRISC", "receiver"),
                ],
            ),
            MicroOpConfig(
                zone="MATMUL",
                type=OpType.COMPUTE,
                participants=[Participant("matmul_core", "TRISC", "compute")],
            ),
            MicroOpConfig(
                zone="GATHER",
                type=OpType.GATHER_REDUCE,
                participants=[
                    Participant("matmul_core", "NCRISC", "sender"),
                    Participant("input_core", "BRISC", "receiver"),
                    Participant("input_core", "TRISC", "reducer"),
                ],
            ),
        ],
        dependencies=[
            ("MCAST", "RMSNORM"),
            ("MATMUL", "MCAST"),
            ("GATHER", "MATMUL"),
        ],
        display_order=["RMSNORM", "MCAST", "MATMUL", "GATHER"],
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_load_pre_sdpa_config(self):
        config = load_kernel_config("pre_sdpa")
        assert config.kernel_name == "pre_sdpa"
        assert len(config.micro_ops) == 17
        assert "input_core" in config.core_roles
        assert "matmul_core" in config.core_roles
        assert config.get_micro_op("MATMUL") is not None
        assert config.get_micro_op("NONEXISTENT") is None

    def test_load_all_built_in_configs(self):
        for name in ["pre_sdpa", "post_sdpa", "shared_expert", "kv_cache_branch", "lm_head_sampling", "moe"]:
            config = load_kernel_config(name)
            assert config.kernel_name == name
            assert len(config.micro_ops) > 0
            assert len(config.display_order) > 0

    def test_config_validation_bad_dependency(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(
            "kernel_name: bad\n"
            "core_roles:\n"
            "  c1:\n"
            "    infer_from_zone: Z1\n"
            "micro_ops:\n"
            "  - zone: Z1\n"
            "    type: compute\n"
            "    participants:\n"
            "      - role: c1\n"
            "        risc: TRISC\n"
            "        function: compute\n"
            "dependencies:\n"
            "  - [Z1, NONEXISTENT]\n"
        )
        with pytest.raises(ValueError, match="NONEXISTENT"):
            load_kernel_config(bad_yaml)

    def test_config_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_kernel_config("definitely_not_a_real_kernel_name_xyz")


# ---------------------------------------------------------------------------
# RISC matching
# ---------------------------------------------------------------------------


class TestRiscMatching:
    def test_exact_match(self):
        assert risc_matches("BRISC", "BRISC")
        assert risc_matches("NCRISC", "NCRISC")

    def test_trisc_wildcard(self):
        assert risc_matches("TRISC", "TRISC_0")
        assert risc_matches("TRISC", "TRISC_1")
        assert risc_matches("TRISC", "TRISC_2")

    def test_no_match(self):
        assert not risc_matches("BRISC", "NCRISC")
        assert not risc_matches("BRISC", "TRISC_0")
        assert not risc_matches("NCRISC", "TRISC_1")


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

SAMPLE_LOG = """ARCH: wormhole, CHIP_FREQ[MHz]: 1200
PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], stat value, Run ID, zone name, zone phase, source line, source file
0,0,0,TRISC_0,1001,10000000000,0,1,RMSNORM,begin,100,k.cpp
0,0,0,TRISC_0,1001,10000120000,0,1,RMSNORM,end,100,k.cpp
0,0,0,TRISC_1,1002,10000010000,0,1,RMSNORM,begin,100,k.cpp
0,0,0,TRISC_1,1002,10000100000,0,1,RMSNORM,end,100,k.cpp
0,0,0,TRISC_2,1003,10000020000,0,1,RMSNORM,begin,100,k.cpp
0,0,0,TRISC_2,1003,10000130000,0,1,RMSNORM,end,100,k.cpp
0,0,0,BRISC,1004,10000130000,0,1,MCAST,begin,200,k.cpp
0,0,0,BRISC,1004,10000180000,0,1,MCAST,end,200,k.cpp
0,1,0,NCRISC,1005,10000180000,0,1,MCAST,begin,210,k.cpp
0,1,0,NCRISC,1005,10000200000,0,1,MCAST,end,210,k.cpp
0,2,0,NCRISC,1006,10000180000,0,1,MCAST,begin,210,k.cpp
0,2,0,NCRISC,1006,10000210000,0,1,MCAST,end,210,k.cpp
0,1,0,TRISC_0,1007,10000210000,0,1,MATMUL,begin,300,k.cpp
0,1,0,TRISC_0,1007,10001010000,0,1,MATMUL,end,300,k.cpp
0,1,0,TRISC_1,1008,10000220000,0,1,MATMUL,begin,300,k.cpp
0,1,0,TRISC_1,1008,10001020000,0,1,MATMUL,end,300,k.cpp
0,1,0,TRISC_2,1009,10000230000,0,1,MATMUL,begin,300,k.cpp
0,1,0,TRISC_2,1009,10001050000,0,1,MATMUL,end,300,k.cpp
0,2,0,TRISC_0,1010,10000210000,0,1,MATMUL,begin,300,k.cpp
0,2,0,TRISC_0,1010,10001000000,0,1,MATMUL,end,300,k.cpp
0,2,0,TRISC_1,1011,10000220000,0,1,MATMUL,begin,300,k.cpp
0,2,0,TRISC_1,1011,10001010000,0,1,MATMUL,end,300,k.cpp
0,2,0,TRISC_2,1012,10000230000,0,1,MATMUL,begin,300,k.cpp
0,2,0,TRISC_2,1012,10001040000,0,1,MATMUL,end,300,k.cpp
"""


class TestLogParsing:
    @pytest.fixture
    def log_path(self, tmp_path):
        p = tmp_path / "device_log.csv"
        p.write_text(SAMPLE_LOG)
        return p

    def test_extract_freq(self, log_path):
        assert extract_chip_freq_mhz(log_path) == 1200.0

    def test_parse_all_zones(self, log_path):
        intervals, freq = parse_device_log(log_path)
        assert freq == 1200.0
        zones = {iv.zone_name for iv in intervals}
        assert zones == {"RMSNORM", "MCAST", "MATMUL"}

    def test_parse_with_zone_filter(self, log_path):
        intervals, _ = parse_device_log(log_path, allowed_zones={"RMSNORM"})
        assert all(iv.zone_name == "RMSNORM" for iv in intervals)
        assert len(intervals) == 3  # TRISC_0, TRISC_1, TRISC_2

    def test_interval_duration(self, log_path):
        intervals, _ = parse_device_log(log_path, allowed_zones={"RMSNORM"})
        t0 = next(iv for iv in intervals if iv.risc == "TRISC_0")
        assert t0.duration_cycles == 120000

    def test_cycles_to_ns(self):
        assert cycles_to_ns(1200000, 1200.0) == 1000000.0  # 1ms


# ---------------------------------------------------------------------------
# Core role inference
# ---------------------------------------------------------------------------


class TestCoreRoleInference:
    def test_infer_roles(self, tmp_path):
        intervals = [
            _iv((0, 0), "TRISC_0", "RMSNORM", 0, 100000),
            _iv((0, 0), "TRISC_1", "RMSNORM", 0, 90000),
            _iv((1, 0), "TRISC_0", "RMSNORM", 0, 50),  # no-op
            _iv((1, 0), "TRISC_0", "MATMUL", 0, 800000),
            _iv((2, 0), "TRISC_0", "MATMUL", 0, 750000),
        ]
        config = _simple_config()
        roles = infer_core_roles(intervals, config)
        assert roles["input_core"] == {(0, 0)}
        assert (1, 0) in roles["matmul_core"]
        assert (2, 0) in roles["matmul_core"]
        # (1,0) has RMSNORM with 50 cycles = 0.05% of 100000 -> filtered out
        assert (1, 0) not in roles["input_core"]


# ---------------------------------------------------------------------------
# Interval classification
# ---------------------------------------------------------------------------


class TestIntervalClassification:
    def test_classify_mcast(self):
        intervals = [
            _iv((0, 0), "BRISC", "MCAST", 0, 50000),
            _iv((1, 0), "NCRISC", "MCAST", 50000, 70000),
            _iv((2, 0), "NCRISC", "MCAST", 50000, 80000),
        ]
        config = _simple_config()
        core_roles = {
            "input_core": {(0, 0)},
            "matmul_core": {(1, 0), (2, 0)},
        }
        classified = classify_intervals(intervals, config, core_roles)
        mcast_cls = classified["MCAST"]
        senders = [c for c in mcast_cls if c.function == "sender"]
        receivers = [c for c in mcast_cls if c.function == "receiver"]
        assert len(senders) == 1
        assert senders[0].interval.core == (0, 0)
        assert len(receivers) == 2

    def test_classify_trisc_compute(self):
        intervals = [
            _iv((0, 0), "TRISC_0", "RMSNORM", 0, 100000),
            _iv((0, 0), "TRISC_1", "RMSNORM", 10000, 95000),
            _iv((0, 0), "TRISC_2", "RMSNORM", 20000, 110000),
        ]
        config = _simple_config()
        core_roles = {"input_core": {(0, 0)}, "matmul_core": set()}
        classified = classify_intervals(intervals, config, core_roles)
        rms_cls = classified["RMSNORM"]
        assert all(c.function == "compute" for c in rms_cls)
        assert len(rms_cls) == 3  # all 3 TRISCs classified


# ---------------------------------------------------------------------------
# Timing engine
# ---------------------------------------------------------------------------


class TestTimingEngine:
    def test_compute_trisc_pipeline(self):
        """Per-core wall clock = max(end) - min(start) across TRISC sub-procs."""
        intervals = [
            _ci(_iv((0, 0), "TRISC_0", "X", 1000, 1100), "compute"),
            _ci(_iv((0, 0), "TRISC_1", "X", 1020, 1110), "compute"),
            _ci(_iv((0, 0), "TRISC_2", "X", 1040, 1120), "compute"),
        ]
        dur, core = _trisc_pipeline_wall_clock(intervals)
        # wall clock = 1120 - 1000 = 120 (NOT max of durations 100, 90, 80)
        assert dur == 120
        assert core == (0, 0)

    def test_compute_multi_core(self):
        """Max across cores of per-core pipeline wall clock."""
        intervals = [
            _ci(_iv((0, 0), "TRISC_0", "X", 1000, 1100), "compute"),
            _ci(_iv((0, 0), "TRISC_1", "X", 1020, 1110), "compute"),
            _ci(_iv((0, 0), "TRISC_2", "X", 1040, 1120), "compute"),
            _ci(_iv((1, 0), "TRISC_0", "X", 2000, 2200), "compute"),
            _ci(_iv((1, 0), "TRISC_1", "X", 2030, 2210), "compute"),
            _ci(_iv((1, 0), "TRISC_2", "X", 2050, 2250), "compute"),
        ]
        dur, core = _trisc_pipeline_wall_clock(intervals)
        # core (0,0): 1120-1000=120, core (1,0): 2250-2000=250 -> max=250
        assert dur == 250
        assert core == (1, 0)

    def test_mcast_sender_plus_max_receiver(self):
        """Mcast duration = sender + max(receivers)."""
        micro_op = MicroOpConfig(
            zone="MCAST",
            type=OpType.MCAST,
            participants=[
                Participant("input_core", "BRISC", "sender"),
                Participant("matmul_core", "NCRISC", "receiver"),
            ],
        )
        classified = [
            _ci(_iv((0, 0), "BRISC", "MCAST", 0, 50000), "sender", "input_core"),
            _ci(_iv((1, 0), "NCRISC", "MCAST", 50000, 70000), "receiver", "matmul_core"),
            _ci(_iv((2, 0), "NCRISC", "MCAST", 50000, 80000), "receiver", "matmul_core"),
        ]
        result = compute_op_duration(micro_op, classified, 1000.0)
        # sender=50000, max(receiver)=30000, total=80000
        assert result.duration_cycles == 80000
        assert result.breakdown["sender"] == 50000
        assert result.breakdown["receiver"] == 30000

    def test_gather_reduce(self):
        """gather_reduce = max(senders) + receiver + reducer TRISC pipeline."""
        micro_op = MicroOpConfig(
            zone="GATHER",
            type=OpType.GATHER_REDUCE,
            participants=[
                Participant("matmul_core", "NCRISC", "sender"),
                Participant("input_core", "BRISC", "receiver"),
                Participant("input_core", "TRISC", "reducer"),
            ],
        )
        classified = [
            _ci(_iv((1, 0), "NCRISC", "GATHER", 0, 10000), "sender"),
            _ci(_iv((2, 0), "NCRISC", "GATHER", 0, 12000), "sender"),
            _ci(_iv((0, 0), "BRISC", "GATHER", 12000, 17000), "receiver"),
            _ci(_iv((0, 0), "TRISC_0", "GATHER", 17000, 22000), "reducer"),
            _ci(_iv((0, 0), "TRISC_1", "GATHER", 17500, 22500), "reducer"),
            _ci(_iv((0, 0), "TRISC_2", "GATHER", 18000, 23000), "reducer"),
        ]
        result = compute_op_duration(micro_op, classified, 1000.0)
        # max(sender)=12000, receiver=5000, reducer=23000-17000=6000
        assert result.duration_cycles == 12000 + 5000 + 6000

    def test_dram_write(self):
        micro_op = MicroOpConfig(
            zone="KV_CACHE_UPDATE",
            type=OpType.DRAM_WRITE,
            participants=[Participant("kv_core", "BRISC", "writer")],
        )
        classified = [
            _ci(_iv((3, 0), "BRISC", "KV_CACHE_UPDATE", 0, 5000), "writer"),
            _ci(_iv((4, 0), "BRISC", "KV_CACHE_UPDATE", 0, 7000), "writer"),
        ]
        result = compute_op_duration(micro_op, classified, 1000.0)
        assert result.duration_cycles == 7000

    def test_span_type(self):
        micro_op = MicroOpConfig(
            zone="FLASH_MLA",
            type=OpType.SPAN,
            participants=[Participant("kv_core", "TRISC", "compute")],
        )
        classified = [
            _ci(_iv((3, 0), "TRISC_0", "FLASH_MLA", 100, 500), "compute"),
            _ci(_iv((3, 0), "TRISC_1", "FLASH_MLA", 200, 600), "compute"),
            _ci(_iv((4, 0), "TRISC_0", "FLASH_MLA", 50, 700), "compute"),
        ]
        result = compute_op_duration(micro_op, classified, 1000.0)
        # span = 700 - 50 = 650
        assert result.duration_cycles == 650

    def test_empty_classified(self):
        micro_op = MicroOpConfig(
            zone="X",
            type=OpType.COMPUTE,
            participants=[],
        )
        result = compute_op_duration(micro_op, [], 1000.0)
        assert result.duration_cycles == 0


# ---------------------------------------------------------------------------
# Critical-path solver
# ---------------------------------------------------------------------------


class TestCriticalPath:
    def test_linear_chain(self):
        durations = {"A": 100, "B": 200, "C": 50}
        deps = [("B", "A"), ("C", "B")]
        ef, path, total = compute_critical_path(durations, deps)
        assert total == 350
        assert path == ["A", "B", "C"]

    def test_parallel_branches_take_max(self):
        durations = {"ROOT": 10, "FAST": 20, "SLOW": 100, "JOIN": 5}
        deps = [("FAST", "ROOT"), ("SLOW", "ROOT"), ("JOIN", "FAST"), ("JOIN", "SLOW")]
        ef, path, total = compute_critical_path(durations, deps)
        # critical path: ROOT(10) -> SLOW(100) -> JOIN(5) = 115
        assert total == 115
        assert "SLOW" in path
        assert "FAST" not in path

    def test_independent_ops(self):
        durations = {"A": 100, "B": 200}
        deps = []
        ef, path, total = compute_critical_path(durations, deps)
        assert total == 200
        assert path == ["B"]

    def test_empty(self):
        ef, path, total = compute_critical_path({}, [])
        assert total == 0
        assert path == []


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    def _run_pipeline(self, log_path):
        from fused_kernel_profiler import analyze

        config = _simple_config()
        return analyze(log_path, config)

    @pytest.fixture
    def log_path(self, tmp_path):
        p = tmp_path / "device_log.csv"
        p.write_text(SAMPLE_LOG)
        return p

    def test_text_report_contains_key_info(self, log_path):
        (op_results, ef, cp, total_cp, roles, ivs, freq) = self._run_pipeline(log_path)
        config = _simple_config()
        report = format_text_report(config, op_results, ef, cp, total_cp, roles, ivs, freq)
        assert "test" in report
        assert "MATMUL" in report
        assert "Critical-path" in report

    def test_json_report_structure(self, log_path):
        (op_results, ef, cp, total_cp, roles, ivs, freq) = self._run_pipeline(log_path)
        config = _simple_config()
        data = format_json_report(config, op_results, ef, cp, total_cp, roles, freq)
        assert data["kernel_name"] == "test"
        assert "critical_path" in data
        assert "micro_ops" in data
        assert "MATMUL" in data["micro_ops"]


# ---------------------------------------------------------------------------
# End-to-end with pre_sdpa config
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.fixture
    def log_path(self, tmp_path):
        p = tmp_path / "device_log.csv"
        p.write_text(SAMPLE_LOG)
        return p

    def test_pre_sdpa_config_loads_and_runs(self, log_path):
        """Load the real pre_sdpa config and run against sample data."""
        from fused_kernel_profiler import analyze

        config = load_kernel_config("pre_sdpa")
        (op_results, ef, cp, total_cp, roles, ivs, freq) = analyze(log_path, config)
        assert freq == 1200.0
        # MATMUL should be found
        assert "MATMUL" in op_results
        matmul_result = op_results["MATMUL"]
        assert matmul_result.duration_cycles > 0

    def test_main_cli_json(self, log_path, tmp_path):
        """Test CLI --json output."""
        import sys

        from fused_kernel_profiler import main

        out_path = tmp_path / "out.json"
        orig = sys.argv
        try:
            sys.argv = [
                "fused_kernel_profiler",
                "--kernel",
                "pre_sdpa",
                "--log",
                str(log_path),
                "--json",
                "--output",
                str(out_path),
            ]
            code = main()
            assert code == 0
            data = json.loads(out_path.read_text())
            assert data["kernel_name"] == "pre_sdpa"
        finally:
            sys.argv = orig

    def test_main_cli_missing_log(self, tmp_path):
        import sys

        from fused_kernel_profiler import main

        orig = sys.argv
        try:
            sys.argv = [
                "fused_kernel_profiler",
                "--kernel",
                "pre_sdpa",
                "--log",
                str(tmp_path / "nonexistent.csv"),
            ]
            code = main()
            assert code == 1
        finally:
            sys.argv = orig


# ---------------------------------------------------------------------------
# Predecessor gate computation
# ---------------------------------------------------------------------------


class TestPredecessorGates:
    def test_no_predecessors_returns_empty(self):
        config = _simple_config()
        classified = {}
        gates = compute_predecessor_gates("RMSNORM", config, classified)
        assert gates == {}

    def test_same_core_gate(self):
        """MATMUL depends on MCAST; both have matmul_core participants."""
        config = _simple_config()
        classified = {
            "MCAST": [
                _ci(_iv((0, 0), "BRISC", "MCAST", 100, 200), "sender", "input_core"),
                _ci(_iv((1, 0), "NCRISC", "MCAST", 100, 300), "receiver", "matmul_core"),
                _ci(_iv((2, 0), "NCRISC", "MCAST", 100, 350), "receiver", "matmul_core"),
            ],
            "MATMUL": [
                _ci(_iv((1, 0), "TRISC_0", "MATMUL", 250, 900), "compute", "matmul_core"),
                _ci(_iv((2, 0), "TRISC_0", "MATMUL", 250, 920), "compute", "matmul_core"),
            ],
        }
        gates = compute_predecessor_gates("MATMUL", config, classified)
        assert gates[(1, 0)] == 300
        assert gates[(2, 0)] == 350
        assert (0, 0) not in gates

    def test_cross_core_gate(self):
        """MCAST depends on RMSNORM; MCAST sender (input_core) shares role,
        MCAST receivers (matmul_core) do NOT share a role with RMSNORM."""
        config = _simple_config()
        classified = {
            "RMSNORM": [
                _ci(_iv((0, 0), "TRISC_0", "RMSNORM", 0, 150), "compute", "input_core"),
            ],
            "MCAST": [
                _ci(_iv((0, 0), "BRISC", "MCAST", 100, 260), "sender", "input_core"),
                _ci(_iv((1, 0), "NCRISC", "MCAST", 100, 300), "receiver", "matmul_core"),
            ],
        }
        gates = compute_predecessor_gates("MCAST", config, classified)
        assert gates[(0, 0)] == 150

    def test_multiple_predecessors_take_max(self):
        """CREATE_Q_HEADS depends on both QNOPE and QROPE in pre_sdpa."""
        config = KernelConfig(
            kernel_name="test_multi_pred",
            core_roles={
                "core_a": CoreRoleConfig(infer_from_zone="OP_A"),
                "core_b": CoreRoleConfig(infer_from_zone="OP_B"),
            },
            micro_ops=[
                MicroOpConfig("OP_A", OpType.COMPUTE, [Participant("core_a", "TRISC", "compute")]),
                MicroOpConfig("OP_B", OpType.COMPUTE, [Participant("core_b", "TRISC", "compute")]),
                MicroOpConfig(
                    "JOIN",
                    OpType.SPAN,
                    [Participant("core_a", "NCRISC", "sender"), Participant("core_b", "NCRISC", "sender")],
                ),
            ],
            dependencies=[("JOIN", "OP_A"), ("JOIN", "OP_B")],
            display_order=["OP_A", "OP_B", "JOIN"],
        )
        classified = {
            "OP_A": [_ci(_iv((1, 0), "TRISC_0", "OP_A", 0, 500), "compute", "core_a")],
            "OP_B": [_ci(_iv((2, 0), "TRISC_0", "OP_B", 0, 800), "compute", "core_b")],
            "JOIN": [
                _ci(_iv((1, 0), "NCRISC", "JOIN", 100, 900), "sender", "core_a"),
                _ci(_iv((2, 0), "NCRISC", "JOIN", 100, 950), "sender", "core_b"),
            ],
        }
        gates = compute_predecessor_gates("JOIN", config, classified)
        assert gates[(1, 0)] == 500
        assert gates[(2, 0)] == 800


# ---------------------------------------------------------------------------
# Gate-adjusted timing helpers
# ---------------------------------------------------------------------------


class TestGateAdjustedHelpers:
    def test_trisc_pipeline_with_gate_subtracts_blocking(self):
        """TRISC on core (1,0): starts at 200, ends at 900.
        Gate (MCAST NCRISC end) on that core = 500.
        Blocking = 500-200 = 300.  Adjusted = 900-500 = 400."""
        intervals = [
            _ci(_iv((1, 0), "TRISC_0", "X", 200, 850), "compute"),
            _ci(_iv((1, 0), "TRISC_1", "X", 220, 870), "compute"),
            _ci(_iv((1, 0), "TRISC_2", "X", 240, 900), "compute"),
        ]
        gates = {(1, 0): 500}
        adj, core = _trisc_pipeline_wall_clock(intervals, gates)
        assert adj == 900 - 500
        assert core == (1, 0)

    def test_trisc_pipeline_no_blocking_when_gate_before_start(self):
        """Gate is before the zone starts -> no adjustment."""
        intervals = [
            _ci(_iv((1, 0), "TRISC_0", "X", 600, 900), "compute"),
            _ci(_iv((1, 0), "TRISC_1", "X", 610, 910), "compute"),
        ]
        gates = {(1, 0): 500}
        adj, core = _trisc_pipeline_wall_clock(intervals, gates)
        assert adj == 910 - 600
        assert core == (1, 0)

    def test_trisc_pipeline_multi_core_with_different_gates(self):
        """Two cores, each with different gate times."""
        intervals = [
            _ci(_iv((1, 0), "TRISC_0", "X", 100, 800), "compute"),
            _ci(_iv((2, 0), "TRISC_0", "X", 100, 900), "compute"),
        ]
        gates = {(1, 0): 300, (2, 0): 600}
        adj, core = _trisc_pipeline_wall_clock(intervals, gates)
        # core (1,0): 800 - max(100,300) = 500
        # core (2,0): 900 - max(100,600) = 300
        assert adj == 500
        assert core == (1, 0)

    def test_adjusted_max_duration_with_gate(self):
        intervals = [
            _ci(_iv((0, 0), "BRISC", "X", 100, 500), "sender"),
            _ci(_iv((1, 0), "BRISC", "X", 100, 600), "sender"),
        ]
        gates = {(0, 0): 400, (1, 0): 400}
        result = _adjusted_max_duration(intervals, gates)
        # (0,0): 500 - max(100,400) = 100
        # (1,0): 600 - max(100,400) = 200
        assert result == 200

    def test_adjusted_max_duration_no_gate(self):
        intervals = [
            _ci(_iv((0, 0), "BRISC", "X", 100, 500), "sender"),
            _ci(_iv((1, 0), "BRISC", "X", 100, 300), "sender"),
        ]
        result = _adjusted_max_duration(intervals, None)
        assert result == 400

    def test_adjusted_span_cycles_with_gate(self):
        intervals = [
            _ci(_iv((0, 0), "TRISC_0", "X", 100, 700), "compute"),
            _ci(_iv((1, 0), "TRISC_0", "X", 200, 800), "compute"),
        ]
        gates = {(0, 0): 500, (1, 0): 500}
        result = _adjusted_span_cycles(intervals, gates)
        # min_start=100, max_end=800, max_gate=500
        # 800 - max(100, 500) = 300
        assert result == 300

    def test_adjusted_span_no_gate_unchanged(self):
        intervals = [
            _ci(_iv((0, 0), "TRISC_0", "X", 100, 700), "compute"),
        ]
        result = _adjusted_span_cycles(intervals)
        assert result == 600

    def test_uniform_gate(self):
        intervals = [
            _ci(_iv((1, 0), "NCRISC", "X", 0, 100), "receiver"),
            _ci(_iv((2, 0), "NCRISC", "X", 0, 200), "receiver"),
        ]
        gate = _uniform_gate(intervals, 50)
        assert gate == {(1, 0): 50, (2, 0): 50}


# ---------------------------------------------------------------------------
# Gate-adjusted op timing (integrated)
# ---------------------------------------------------------------------------


class TestGateAdjustedOpTiming:
    def test_compute_with_gate_subtracts_blocking(self):
        """MATMUL TRISC starts at t=200, predecessor MCAST ends at t=500."""
        micro_op = MicroOpConfig(
            zone="MATMUL",
            type=OpType.COMPUTE,
            participants=[Participant("matmul_core", "TRISC", "compute")],
        )
        classified = [
            _ci(_iv((1, 0), "TRISC_0", "MATMUL", 200, 850), "compute", "matmul_core"),
            _ci(_iv((1, 0), "TRISC_1", "MATMUL", 220, 870), "compute", "matmul_core"),
            _ci(_iv((1, 0), "TRISC_2", "MATMUL", 240, 900), "compute", "matmul_core"),
        ]
        gates = {(1, 0): 500}
        result = compute_op_duration(micro_op, classified, 1000.0, gates)
        assert result.duration_cycles == 900 - 500  # 400 adjusted
        assert result.raw_duration_cycles == 900 - 200  # 700 raw
        assert result.blocking_cycles == 300
        assert result.duration_us == pytest.approx(0.4)
        assert result.raw_duration_us == pytest.approx(0.7)
        assert result.blocking_us == pytest.approx(0.3)

    def test_compute_no_gate_raw_equals_adjusted(self):
        micro_op = MicroOpConfig(
            zone="MATMUL",
            type=OpType.COMPUTE,
            participants=[Participant("matmul_core", "TRISC", "compute")],
        )
        classified = [
            _ci(_iv((1, 0), "TRISC_0", "MATMUL", 200, 900), "compute", "matmul_core"),
        ]
        result = compute_op_duration(micro_op, classified, 1000.0)
        assert result.duration_cycles == result.raw_duration_cycles
        assert result.blocking_cycles == 0

    def test_mcast_cross_op_and_within_op_gate(self):
        """
        MCAST sender on input_core blocks on RMSNORM (cross-op gate).
        MCAST receiver on matmul_core blocks on sender (within-op gate).
        """
        micro_op = MicroOpConfig(
            zone="MCAST",
            type=OpType.MCAST,
            participants=[
                Participant("input_core", "BRISC", "sender"),
                Participant("matmul_core", "NCRISC", "receiver"),
            ],
        )
        # Sender enters zone at 100, RMSNORM ends at 200, sender finishes at 260
        # Receiver enters zone at 95, blocks on sender until ~260, finishes at 280
        classified = [
            _ci(_iv((0, 0), "BRISC", "MCAST", 100, 260), "sender", "input_core"),
            _ci(_iv((1, 0), "NCRISC", "MCAST", 95, 280), "receiver", "matmul_core"),
        ]
        cross_op_gate = {(0, 0): 200}  # RMSNORM TRISC end on input_core
        result = compute_op_duration(micro_op, classified, 1000.0, cross_op_gate)

        # adjusted sender = 260 - max(100, 200) = 60
        # within-op gate for receiver = sender_end = 260
        # adjusted receiver = 280 - max(95, 260) = 20
        # adjusted total = 60 + 20 = 80
        assert result.duration_cycles == 80
        assert result.breakdown["sender"] == 60
        assert result.breakdown["receiver"] == 20

        # raw sender = 160, raw receiver = 185, raw total = 345
        assert result.raw_duration_cycles == 160 + 185

    def test_gather_reduce_full_adjustment(self):
        """
        GATHER: senders (matmul_core) block on MATMUL (cross-op gate).
        Receiver (input_core) blocks on senders (within-op gate).
        Reducer (input_core) blocks on receiver (within-op gate).
        """
        micro_op = MicroOpConfig(
            zone="GATHER",
            type=OpType.GATHER_REDUCE,
            participants=[
                Participant("matmul_core", "NCRISC", "sender"),
                Participant("input_core", "BRISC", "receiver"),
                Participant("input_core", "TRISC", "reducer"),
            ],
        )
        classified = [
            _ci(_iv((1, 0), "NCRISC", "GATHER", 800, 1050), "sender", "matmul_core"),
            _ci(_iv((2, 0), "NCRISC", "GATHER", 800, 1100), "sender", "matmul_core"),
            _ci(_iv((0, 0), "BRISC", "GATHER", 900, 1200), "receiver", "input_core"),
            _ci(_iv((0, 0), "TRISC_0", "GATHER", 1000, 1400), "reducer", "input_core"),
            _ci(_iv((0, 0), "TRISC_1", "GATHER", 1020, 1420), "reducer", "input_core"),
            _ci(_iv((0, 0), "TRISC_2", "GATHER", 1040, 1450), "reducer", "input_core"),
        ]
        # MATMUL TRISC ends at 950 on core (1,0) and 980 on core (2,0)
        cross_op_gate = {(1, 0): 950, (2, 0): 980}
        result = compute_op_duration(micro_op, classified, 1000.0, cross_op_gate)

        # adj sender (1,0): 1050 - max(800, 950) = 100
        # adj sender (2,0): 1100 - max(800, 980) = 120
        # max adj sender = 120
        adj_s = 120

        # within-op gate for receiver = max(sender_ends) = 1100
        # adj receiver: 1200 - max(900, 1100) = 100
        adj_r = 100

        # within-op gate for reducer = receiver_end = 1200
        # reducer TRISC: starts 1000,1020,1040 ends 1400,1420,1450
        # adj on core (0,0): max(1450) - max(min(1000), 1200) = 1450 - 1200 = 250
        adj_red = 250

        assert result.duration_cycles == adj_s + adj_r + adj_red
        assert result.breakdown["sender"] == adj_s
        assert result.breakdown["receiver"] == adj_r
        assert result.breakdown["reducer"] == adj_red

    def test_dram_write_with_gate(self):
        micro_op = MicroOpConfig(
            zone="KV_CACHE",
            type=OpType.DRAM_WRITE,
            participants=[Participant("kv_core", "BRISC", "writer")],
        )
        classified = [
            _ci(_iv((3, 0), "BRISC", "KV_CACHE", 100, 500), "writer", "kv_core"),
        ]
        gates = {(3, 0): 400}
        result = compute_op_duration(micro_op, classified, 1000.0, gates)
        assert result.duration_cycles == 100  # 500 - max(100, 400)
        assert result.raw_duration_cycles == 400
        assert result.blocking_cycles == 300

    def test_span_with_gate(self):
        micro_op = MicroOpConfig(
            zone="FLASH",
            type=OpType.SPAN,
            participants=[Participant("kv_core", "TRISC", "compute")],
        )
        classified = [
            _ci(_iv((3, 0), "TRISC_0", "FLASH", 100, 800), "compute", "kv_core"),
            _ci(_iv((4, 0), "TRISC_0", "FLASH", 200, 900), "compute", "kv_core"),
        ]
        gates = {(3, 0): 500, (4, 0): 500}
        result = compute_op_duration(micro_op, classified, 1000.0, gates)
        # raw span = 900-100 = 800
        # adjusted: max_end=900, min_start=100, max_gate=500
        # 900 - max(100, 500) = 400
        assert result.duration_cycles == 400
        assert result.raw_duration_cycles == 800

    def test_json_report_includes_blocking_fields(self):
        """JSON report should contain raw_duration_us and blocking_us."""
        config = _simple_config()
        micro_op = config.micro_ops[0]  # RMSNORM
        result = compute_op_duration(
            micro_op,
            [
                _ci(_iv((0, 0), "TRISC_0", "RMSNORM", 0, 1000), "compute", "input_core"),
            ],
            1000.0,
        )
        op_results = {"RMSNORM": result}
        data = format_json_report(
            config,
            op_results,
            {"RMSNORM": 1000},
            ["RMSNORM"],
            1000,
            {"input_core": {(0, 0)}, "matmul_core": set()},
            1000.0,
        )
        rmsnorm_data = data["micro_ops"]["RMSNORM"]
        assert "raw_duration_us" in rmsnorm_data
        assert "blocking_us" in rmsnorm_data
        assert "blocking_cycles" in rmsnorm_data

    def test_text_report_shows_adjusted_column(self):
        """Text report header should include Adjusted and Blocked columns."""
        config = _simple_config()
        result = compute_op_duration(
            config.micro_ops[0],
            [
                _ci(_iv((0, 0), "TRISC_0", "RMSNORM", 0, 1000), "compute", "input_core"),
            ],
            1000.0,
        )
        report = format_text_report(
            config,
            {"RMSNORM": result},
            {"RMSNORM": 1000},
            ["RMSNORM"],
            1000,
            {"input_core": {(0, 0)}, "matmul_core": set()},
            [_iv((0, 0), "TRISC_0", "RMSNORM", 0, 1000)],
            1000.0,
        )
        assert "Adjusted(us)" in report
        assert "Blocked(us)" in report

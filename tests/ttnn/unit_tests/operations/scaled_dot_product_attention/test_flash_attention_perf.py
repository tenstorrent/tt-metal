# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import statistics

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.operations.scaled_dot_product_attention import FlashAttentionProgramConfig, flash_attention


_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_DEVICE_CSV = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/profiler/.logs/profile_log_device.csv")
_RISC_LABEL = {"TRISC_0": "unpack", "TRISC_1": "math", "TRISC_2": "pack", "NCRISC": "reader", "BRISC": "writer"}
_COMPUTE_PHASE_ZONES = (
    "FA_Q_SCALE",
    "FA_QK_MATMUL",
    "FA_BLOCK_MAX",
    "FA_ONLINE_RESCALE",
    "FA_PROBS_EXP",
    "FA_BLOCK_SUM",
    "FA_PV_MATMUL",
    "FA_STATE_O_UPDATE",
    "FA_FINAL_NORMALIZE",
)
_DATAFLOW_PHASE_ZONES = (
    "FA_SENDER_Q_READ",
    "FA_SENDER_KV_RESERVE",
    "FA_SENDER_KV_DRAM",
    "FA_SENDER_KV_MCAST",
    "FA_RECEIVER_Q_READ",
    "FA_RECEIVER_KV_MCAST",
    "FA_WRITER_WAIT",
    "FA_WRITER_DRAM",
)
_PHASE_ZONES = _COMPUTE_PHASE_ZONES + _DATAFLOW_PHASE_ZONES


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total = 0.0
    found = False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, run, warmup=3, trials=10):
    for _ in range(warmup):
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    for _ in range(trials):
        run()
    total = _read_kernel_ns(device)
    return None if total is None else total / trials


def _read_device_csv(path):
    with open(path) as csv_file:
        lines = csv_file.read().splitlines()
    frequency_mhz = 1000.0
    for part in lines[0].split(","):
        if "CHIP_FREQ" in part:
            frequency_mhz = float(part.split(":")[1])
    rows = [[field.strip() for field in line.split(",")] for line in lines[2:] if line.strip()]
    return [row for row in rows if len(row) >= 12], 1000.0 / frequency_mhz


def _all_device_run_ids(path):
    if not os.path.exists(path):
        return set()
    rows, _ = _read_device_csv(path)
    return {row[7] for row in rows}


def _phase_profile(path, seen_run_ids):
    rows, ns_per_cycle = _read_device_csv(path)
    starts = {}
    ends = {}
    zone_names = set(_PHASE_ZONES)
    for row in rows:
        core = (int(row[1]), int(row[2]))
        risc, cycle, run_id, zone, event = row[3], int(row[5]), row[7], row[10], row[11]
        if run_id in seen_run_ids or zone not in zone_names:
            continue
        key = (run_id, core, risc, zone)
        (starts if event == "ZONE_START" else ends).setdefault(key, []).append(cycle)

    samples = {}
    for key, start_cycles in starts.items():
        end_cycles = sorted(ends.get(key, []))
        durations = [(end - start) * ns_per_cycle for start, end in zip(sorted(start_cycles), end_cycles)]
        if durations:
            run_id, core, risc, zone = key
            samples.setdefault(zone, {}).setdefault(core, {})[_RISC_LABEL.get(risc, risc)] = durations

    result = {}
    for zone, by_core in samples.items():
        core_walls = []
        calls_per_core = []
        engine_totals = {}
        engine_calls = {}
        for by_engine in by_core.values():
            totals = {engine: sum(durations) for engine, durations in by_engine.items()}
            core_walls.append(max(totals.values()))
            calls_per_core.append(max(len(durations) for durations in by_engine.values()))
            for engine, durations in by_engine.items():
                engine_totals.setdefault(engine, []).append(sum(durations))
                engine_calls.setdefault(engine, []).extend(durations)
        median_engine_totals = {engine: statistics.median(values) for engine, values in engine_totals.items()}
        limiting_engine = max(median_engine_totals, key=median_engine_totals.get)
        result[zone] = {
            "cores": len(by_core),
            "calls": statistics.median(calls_per_core),
            "median_total_ns": statistics.median(core_walls),
            "max_total_ns": max(core_walls),
            "median_call_ns": max(statistics.median(values) for values in engine_calls.values()),
            "limiting_engine": limiting_engine,
            "engine_total_ns": median_engine_totals,
        }
    return result


def _format_phase_profile(phases, kernel_ns_by_zone):
    lines = [
        "",
        "=== flash_attention device-zone phase profile ===",
        "zone | kernel ms | cores | calls/core | median wall/core us | unpack us | math us | pack us | "
        "max wall/core us | median call us | limiting engine",
    ]
    for zone in _PHASE_ZONES:
        phase = phases.get(zone)
        if phase is None:
            lines.append(f"{zone} | MISSING")
            continue
        engine_totals = phase["engine_total_ns"]
        lines.append(
            f"{zone} | {kernel_ns_by_zone[zone] / 1e6:.3f} | {phase['cores']} | {phase['calls']:.0f} | "
            f"{phase['median_total_ns'] / 1e3:.1f} | {engine_totals.get('unpack', 0.0) / 1e3:.1f} | "
            f"{engine_totals.get('math', 0.0) / 1e3:.1f} | {engine_totals.get('pack', 0.0) / 1e3:.1f} | "
            f"{phase['max_total_ns'] / 1e3:.1f} | {phase['median_call_ns'] / 1e3:.2f} | "
            f"{phase['limiting_engine']}"
        )
    for engine in ("unpack", "math", "pack"):
        accounted = sum(
            phases[zone]["engine_total_ns"].get(engine, 0.0) for zone in _COMPUTE_PHASE_ZONES if zone in phases
        )
        lines.append(f"compute zones, summed median {engine}/core: {accounted / 1e6:.3f} ms")
    return "\n".join(lines)


def _make_inputs(device, shape):
    torch.manual_seed(7)
    inputs = [torch.randn(shape, dtype=torch.bfloat16) * 0.25 for _ in range(3)]
    return [
        ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for tensor in inputs
    ]


@pytest.mark.skipif(os.environ.get("TTNN_RUN_FLASH_ATTN_DEFAULT_PERF") != "1", reason="default performance test")
def test_flash_attention_prefill_default_device_perf(device):
    """Report the tuned default path without compiling the full policy sweep."""
    shape = tuple(int(value) for value in os.environ.get("TTNN_FLASH_ATTN_PERF_SHAPE", "1,8,4096,128").split(","))
    tt_inputs = _make_inputs(device, shape)
    ns = _measure(device, lambda: flash_attention(*tt_inputs, program_config=FlashAttentionProgramConfig()))
    assert ns is not None, "device profiler returned no duration for the default path"
    batch, heads, sequence, head_dim = shape
    effective_tflops = 4.0 * batch * heads * sequence * sequence * head_dim / ns / 1e3
    logger.info(
        "\n=== flash_attention tuned-default device performance ===\n"
        f"shape={shape}, arch={device.arch()}\n"
        f"device time: {ns / 1e6:.3f} ms\n"
        f"effective QK+PV: {effective_tflops:.2f} TFLOP/s"
    )


@pytest.mark.skipif(os.environ.get("TTNN_RUN_FLASH_ATTN_PERF") != "1", reason="performance test")
def test_flash_attention_prefill_device_perf(device):
    """Report 8-head/4K device time for direct K/V reads and per-head multicast."""
    shape = tuple(int(value) for value in os.environ.get("TTNN_FLASH_ATTN_PERF_SHAPE", "1,8,4096,128").split(","))
    tt_inputs = _make_inputs(device, shape)

    configs = {
        "direct_dram": FlashAttentionProgramConfig(
            use_kv_multicast=False,
            softmax_block_tiles=4,
            fp32_dest_acc_en=True,
            rescale_exp_approx_mode="fast",
        ),
        "kv_mcast": FlashAttentionProgramConfig(
            use_kv_multicast=True,
            softmax_block_tiles=4,
            fp32_dest_acc_en=True,
            rescale_exp_approx_mode="fast",
        ),
        "kv_mcast_fp32_dest_exact": FlashAttentionProgramConfig(
            use_kv_multicast=True,
            softmax_block_tiles=4,
            fp32_dest_acc_en=True,
            exp_approx_mode="exact",
        ),
        "kv_mcast_bf16_dest_same_geometry": FlashAttentionProgramConfig(
            use_kv_multicast=True,
            qk_output_subblock=(2, 2),
            pv_output_subblock=(2, 2),
            fp32_dest_acc_en=False,
            rescale_exp_approx_mode="fast",
        ),
        "kv_mcast_bf16_dest_8tile": FlashAttentionProgramConfig(
            use_kv_multicast=True,
            qk_output_subblock=(2, 4),
            pv_output_subblock=(2, 4),
            softmax_block_tiles=8,
            fp32_dest_acc_en=False,
            rescale_exp_approx_mode="fast",
        ),
        "kv_mcast_bf16_dest_accurate_fast_same_geometry": FlashAttentionProgramConfig(
            use_kv_multicast=True,
            qk_output_subblock=(2, 2),
            pv_output_subblock=(2, 2),
            fp32_dest_acc_en=False,
            exp_approx_mode="accurate_fast",
        ),
        "kv_mcast_bf16_dest_accurate_fast_8tile": FlashAttentionProgramConfig(
            use_kv_multicast=True,
            qk_output_subblock=(2, 4),
            pv_output_subblock=(2, 4),
            softmax_block_tiles=8,
            fp32_dest_acc_en=False,
            exp_approx_mode="accurate_fast",
        ),
        "kv_mcast_bf16_dest_exact_8tile": FlashAttentionProgramConfig(
            use_kv_multicast=True,
            qk_output_subblock=(2, 4),
            pv_output_subblock=(2, 4),
            softmax_block_tiles=8,
            fp32_dest_acc_en=False,
            exp_approx_mode="exact",
            rescale_exp_approx_mode="exact",
        ),
        "kv_mcast_bf16_dest_fast_probs_accurate_fast_rescale": FlashAttentionProgramConfig(
            use_kv_multicast=True,
            qk_output_subblock=(2, 4),
            pv_output_subblock=(2, 4),
            softmax_block_tiles=8,
            fp32_dest_acc_en=False,
            exp_approx_mode="fast",
            rescale_exp_approx_mode="accurate_fast",
        ),
        "kv_mcast_bf16_dest_accurate_fast_probs_fast_rescale": FlashAttentionProgramConfig(
            use_kv_multicast=True,
            qk_output_subblock=(2, 4),
            pv_output_subblock=(2, 4),
            softmax_block_tiles=8,
            fp32_dest_acc_en=False,
            exp_approx_mode="accurate_fast",
            rescale_exp_approx_mode="fast",
        ),
    }
    ns = {}
    for name, config in configs.items():
        ns[name] = _measure(device, lambda c=config: flash_attention(*tt_inputs, program_config=c))
        assert ns[name] is not None, f"device profiler returned no duration for {name}"

    ratio = ns["direct_dram"] / ns["kv_mcast"]
    logger.info(
        "\n=== flash_attention prefill device performance ===\n"
        f"shape={shape}, arch={device.arch()}\n"
        f"direct DRAM K/V: {ns['direct_dram'] / 1e6:.3f} ms\n"
        f"per-head K/V multicast: {ns['kv_mcast'] / 1e6:.3f} ms\n"
        f"FP32 DEST exact exp: {ns['kv_mcast_fp32_dest_exact'] / 1e6:.3f} ms\n"
        f"BF16 DEST, same 2x2 geometry: {ns['kv_mcast_bf16_dest_same_geometry'] / 1e6:.3f} ms\n"
        f"BF16 DEST, 2x4/8-tile geometry: {ns['kv_mcast_bf16_dest_8tile'] / 1e6:.3f} ms\n"
        "BF16 DEST accurate-fast, same 2x2 geometry: "
        f"{ns['kv_mcast_bf16_dest_accurate_fast_same_geometry'] / 1e6:.3f} ms\n"
        "BF16 DEST accurate-fast, 2x4/8-tile geometry: "
        f"{ns['kv_mcast_bf16_dest_accurate_fast_8tile'] / 1e6:.3f} ms\n"
        f"BF16 DEST exact/clamped, 2x4/8-tile geometry: {ns['kv_mcast_bf16_dest_exact_8tile'] / 1e6:.3f} ms\n"
        "BF16 DEST fast probabilities / accurate-fast rescale: "
        f"{ns['kv_mcast_bf16_dest_fast_probs_accurate_fast_rescale'] / 1e6:.3f} ms\n"
        "BF16 DEST accurate-fast probabilities / fast rescale: "
        f"{ns['kv_mcast_bf16_dest_accurate_fast_probs_fast_rescale'] / 1e6:.3f} ms\n"
        f"multicast speedup: {ratio:.2f}x"
    )


@pytest.mark.skipif(os.environ.get("TTNN_RUN_FLASH_ATTN_READER_PLACEMENT") != "1", reason="performance placement sweep")
def test_flash_attention_prefill_reader_placement(device):
    """Compare first-column and head-rotated K/V DRAM sender placement."""
    shape = (1, 8, 4096, 128)
    tt_inputs = _make_inputs(device, shape)
    variants = {
        "first_column": FlashAttentionProgramConfig(spread_kv_readers=False),
        "spread_by_head": FlashAttentionProgramConfig(spread_kv_readers=True),
    }
    ns = {
        name: _measure(device, lambda c=config: flash_attention(*tt_inputs, program_config=c), warmup=3, trials=10)
        for name, config in variants.items()
    }
    assert all(value is not None for value in ns.values())
    best = min(ns, key=ns.get)
    logger.info(
        "\n=== flash_attention K/V reader placement ===\n"
        + "\n".join(f"{name}: {value / 1e6:.3f} ms ({value / ns[best]:.3f}x best)" for name, value in ns.items())
        + f"\nbest: {best}"
    )


@pytest.mark.skipif(os.environ.get("TTNN_RUN_FLASH_ATTN_PHASE_PROFILE") != "1", reason="device-zone phase profile")
def test_flash_attention_prefill_phase_profile(device):
    """Report named reader/compute/writer device zones for the primary prefill shape."""
    tt_inputs = _make_inputs(device, (1, 8, 4096, 128))
    phases = {}
    kernel_ns_by_zone = {}
    requested = os.environ.get("TTNN_FLASH_ATTN_PROFILE_PHASES")
    zones = _PHASE_ZONES if requested is None else tuple(zone.strip() for zone in requested.split(",") if zone.strip())
    unknown = set(zones) - set(_PHASE_ZONES)
    assert not unknown, f"unknown flash-attention profile zones: {sorted(unknown)}"

    # One named zone per compiled launch keeps the large fused compute kernel
    # below Wormhole's kernel-config size limit while retaining exact stage scopes.
    for zone in zones:
        config = FlashAttentionProgramConfig(profile_phase=zone)
        flash_attention(*tt_inputs, program_config=config)
        ttnn.synchronize_device(device)
        _read_kernel_ns(device)
        seen_run_ids = _all_device_run_ids(_DEVICE_CSV)

        flash_attention(*tt_inputs, program_config=config)
        ttnn.synchronize_device(device)
        kernel_ns = _read_kernel_ns(device)
        assert kernel_ns is not None
        zone_profile = _phase_profile(_DEVICE_CSV, seen_run_ids)
        assert zone in zone_profile, f"missing flash-attention device zone: {zone}"
        phases[zone] = zone_profile[zone]
        kernel_ns_by_zone[zone] = kernel_ns

    logger.info(_format_phase_profile(phases, kernel_ns_by_zone))


@pytest.mark.skipif(os.environ.get("TTNN_RUN_FLASH_ATTN_SWEEP") != "1", reason="performance sweep")
def test_flash_attention_prefill_block_sweep(device):
    """Measure the main L1/phase-amortization block choices on 8-head/4K prefill."""
    shape = (1, 8, 4096, 128)
    tt_inputs = _make_inputs(device, shape)
    variants = {
        "q1_k8": FlashAttentionProgramConfig(query_block_tiles=1, key_block_tiles=8),
        "q2_k4": FlashAttentionProgramConfig(query_block_tiles=2, key_block_tiles=4),
        "q2_k8": FlashAttentionProgramConfig(query_block_tiles=2, key_block_tiles=8),
        "q4_k4": FlashAttentionProgramConfig(query_block_tiles=4, key_block_tiles=4),
        "q4_k8": FlashAttentionProgramConfig(query_block_tiles=4, key_block_tiles=8),
        "q4_k16": FlashAttentionProgramConfig(query_block_tiles=4, key_block_tiles=16),
        "q8_k8_out1": FlashAttentionProgramConfig(
            query_block_tiles=8,
            key_block_tiles=8,
            output_buffer_depth=1,
        ),
        "q8_k16_kv1_out1": FlashAttentionProgramConfig(
            query_block_tiles=8,
            key_block_tiles=16,
            kv_buffer_depth=1,
            output_buffer_depth=1,
        ),
        "q4_k8_sb2x2": FlashAttentionProgramConfig(
            query_block_tiles=4,
            key_block_tiles=8,
            qk_output_subblock=(2, 2),
            pv_output_subblock=(2, 2),
        ),
    }
    ns = {
        name: _measure(device, lambda c=config: flash_attention(*tt_inputs, program_config=c), warmup=2, trials=5)
        for name, config in variants.items()
    }
    assert all(value is not None for value in ns.values())
    best = min(ns, key=ns.get)
    lines = ["", "=== flash_attention 8-head/4K block sweep ==="]
    lines.extend(f"{name}: {value / 1e6:.3f} ms ({value / ns[best]:.2f}x best)" for name, value in ns.items())
    lines.append(f"best: {best}")
    logger.info("\n".join(lines))


@pytest.mark.skipif(os.environ.get("TTNN_RUN_FLASH_ATTN_SUBBLOCK_SWEEP") != "1", reason="performance sweep")
def test_flash_attention_prefill_bf16_dest_subblock_sweep(device):
    """Sweep all eight-tile QK/PV output subblocks on the default BF16-DEST path."""
    shape = (1, 8, 4096, 128)
    tt_inputs = _make_inputs(device, shape)
    qk_shapes = ((1, 8), (2, 4), (4, 2))
    pv_shapes = ((2, 4), (4, 2))
    variants = {
        f"qk{qk_h}x{qk_w}_pv{pv_h}x{pv_w}": FlashAttentionProgramConfig(
            query_block_tiles=4,
            key_block_tiles=16,
            qk_output_subblock=(qk_h, qk_w),
            pv_output_subblock=(pv_h, pv_w),
        )
        for qk_h, qk_w in qk_shapes
        for pv_h, pv_w in pv_shapes
    }
    ns = {
        name: _measure(device, lambda c=config: flash_attention(*tt_inputs, program_config=c), warmup=3, trials=10)
        for name, config in variants.items()
    }
    assert all(value is not None for value in ns.values())
    best = min(ns, key=ns.get)
    lines = ["", "=== flash_attention BF16-DEST 8-tile subblock sweep ==="]
    lines.extend(f"{name}: {value / 1e6:.3f} ms ({value / ns[best]:.3f}x best)" for name, value in ns.items())
    lines.append(f"best: {best}")
    logger.info("\n".join(lines))


@pytest.mark.skipif(os.environ.get("TTNN_RUN_FLASH_ATTN_SOFTMAX_SWEEP") != "1", reason="performance sweep")
def test_flash_attention_prefill_softmax_block_sweep(device):
    """Sweep the elementwise/reduction DEST-lane batch on the tuned 8-head/4K geometry."""
    shape = (1, 8, 4096, 128)
    tt_inputs = _make_inputs(device, shape)
    variants = {
        f"softmax_block_{block}": FlashAttentionProgramConfig(softmax_block_tiles=block) for block in (1, 2, 4, 8)
    }
    ns = {
        name: _measure(device, lambda c=config: flash_attention(*tt_inputs, program_config=c), warmup=3, trials=10)
        for name, config in variants.items()
    }
    assert all(value is not None for value in ns.values())
    best = min(ns, key=ns.get)
    lines = ["", "=== flash_attention softmax block sweep ==="]
    lines.extend(f"{name}: {value / 1e6:.3f} ms ({value / ns[best]:.3f}x best)" for name, value in ns.items())
    lines.append(f"best: {best}")
    logger.info("\n".join(lines))

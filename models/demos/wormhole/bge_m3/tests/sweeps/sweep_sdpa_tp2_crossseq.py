# SPDX-License-Identifier: Apache-2.0
"""Device-profiler sweep for BGE-M3 TP2 cross-sequence stock SDPA."""

import os

import pytest
import torch
from tracy import signpost

import ttnn
from models.tt_dit.utils.sweep_mm_block_sizes import parse_ops_log

COMBOS = [
    (8, 8, 256, 256, False, True, True),
    (8, 8, 256, 512, False, True, True),
    (8, 8, 512, 256, False, True, True),
    (8, 8, 512, 512, False, True, True),
    (8, 7, 512, 512, False, True, True),
    (7, 8, 512, 512, False, True, True),
    (8, 6, 512, 512, False, True, True),
    (6, 8, 512, 512, False, True, True),
    (8, 8, 512, 512, True, True, True),
    (8, 8, 512, 512, False, False, True),
    (8, 8, 512, 512, True, False, True),
]
STREAMING_COMBOS = [
    (8, 8, 128, 128, False, True, False),
    (8, 8, 128, 256, False, True, False),
    (8, 8, 256, 256, False, True, False),
    (8, 8, 256, 512, False, True, False),
]


def selected_combos():
    if os.environ.get("BGE_SDPA_STREAMING_ONLY", "0") != "1":
        return COMBOS
    index = os.environ.get("BGE_SDPA_STREAMING_INDEX")
    if index is None:
        return STREAMING_COMBOS
    return [STREAMING_COMBOS[int(index)]]


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["tp2_n300"])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 10_000_000}],
    indirect=True,
)
def test_sdpa_tp2_worker(mesh_device):
    batch, heads, seq_len, head_dim = 12, 16, 8192, 64
    input_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 1), dims=(2, None))
    replicated_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    print("SDPA_SWEEP_STEP allocate exact-shape tensors", flush=True)
    q = ttnn.from_torch(
        torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=input_mapper,
    )

    def make_replicated():
        return ttnn.from_torch(
            torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicated_mapper,
        )

    k, v = make_replicated(), make_replicated()
    combos = selected_combos()

    def run(combo):
        gx, gy, q_chunk, k_chunk, math_approx, packer_l1, fp32_dest = combo
        compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=math_approx,
            fp32_dest_acc_en=fp32_dest,
            packer_l1_acc=packer_l1,
        )
        return ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=1.0,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=True,
            ),
            compute_kernel_config=compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    print(f"SDPA_SWEEP_STEP warmup {len(combos)} configurations", flush=True)
    for index, combo in enumerate(combos, start=1):
        print(
            f"SDPA_SWEEP_WARMUP [{index}/{len(combos)}] "
            f"g{combo[0]}x{combo[1]} q{combo[2]} k{combo[3]} "
            f"approx={combo[4]} packer={combo[5]} fp32={combo[6]}",
            flush=True,
        )
        output = run(combo)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)

    print("SDPA_SWEEP_STEP profile configurations", flush=True)
    signpost("start")
    for index, combo in enumerate(combos, start=1):
        print(
            f"SDPA_SWEEP_PROFILE [{index}/{len(combos)}] "
            f"g{combo[0]}x{combo[1]} q{combo[2]} k{combo[3]} "
            f"approx={combo[4]} packer={combo[5]} fp32={combo[6]}",
            flush=True,
        )
        output = run(combo)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(output)
    signpost("stop")
    print("SDPA_SWEEP_STEP profile complete", flush=True)


@pytest.mark.timeout(1800)
def test_sdpa_tp2_sweep():
    from tracy.process_model_log import run_device_profiler

    combos = selected_combos()
    subdir = "bge_sdpa_tp2_crossseq"
    command = (
        "pytest models/demos/wormhole/bge_m3/tests/sweeps/"
        "sweep_sdpa_tp2_crossseq.py::test_sdpa_tp2_worker -s -q"
    )
    print(f"SDPA_SWEEP_START total_configs={len(combos)} device-profiler worker", flush=True)
    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])

    durations = parse_ops_log(subdir, expected_ops=len(combos))
    assert len(durations) == len(combos), f"Expected {len(combos)} SDPA timings, got {len(durations)}"

    ranked = sorted(zip(durations, combos))
    print("SDPA_SWEEP_RANKED", flush=True)
    for duration_ns, (gx, gy, q_chunk, k_chunk, math_approx, packer_l1, fp32_dest) in ranked:
        print(
            f"  {duration_ns / 1e6:8.3f} ms  g{gx}x{gy} q{q_chunk} k{k_chunk} "
            f"approx={math_approx} packer={packer_l1} fp32={fp32_dest}",
            flush=True,
        )

    best_ns, best = ranked[0]
    print(
        f"SDPA_SWEEP_BEST device_ms={best_ns / 1e6:.3f} "
        f"g{best[0]}x{best[1]} q{best[2]} k{best[3]} "
        f"approx={best[4]} packer={best[5]} fp32={best[6]}",
        flush=True,
    )

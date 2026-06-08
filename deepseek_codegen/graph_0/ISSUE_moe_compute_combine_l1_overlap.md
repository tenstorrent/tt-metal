# Bug report draft — moe_compute combine mux L1 overlap on Blackhole Galaxy

> Drafted for `.github/ISSUE_TEMPLATE/bug_report.yml`. Repro:
> `deepseek_codegen/graph_0/repro_moe_compute_combine_l1_overlap.py` (verified to FATAL on
> unmodified tt-metal @ 918fad97336).

**Title:** `[ops/ccl]: moe_compute combine mux L1 overlaps compute tensor (TT_FATAL) on Blackhole Galaxy for experts_per_device>2 (DeepSeek hidden=7168)`

**Labels:** bug

---

## Component / Area
ops / experimental CCL — `ttnn.experimental.moe_compute` combine stage
(`selective_reduce_combine`), specifically
`ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.cpp::launch_mux_workers`.

## Issue Type
Runtime Crash (deterministic `TT_FATAL` at program-creation time; root cause is an L1-budget / mux sizing overflow).

## Observed
On a Blackhole 4×8 Galaxy (32 chips), calling `ttnn.experimental.moe_compute` with `cluster_axis=0`
and a DeepSeek-V3-scale MoE shape — **experts_per_device=8**, hidden_size=7168, intermediate_size=2048,
tokens_per_device=32, selected_experts_k=8 — aborts during program creation (before any device
execution) with:

```
TT_FATAL: Mux L1 memory [base=0x1b200, end=0x95330] overlaps with L1 tensor 0x93f00 and is in
danger of being clobbered. (assert.hpp:104)
@ .../moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.cpp:90
```

The fabric-mux L1 region (sized in `launch_mux_workers`) ends at `0x95330`, ~5 KB **above** the
lowest occupied compute L1 tensor (`0x93f00`). The BH mux buffer count is hardcoded
(`num_buffers_full_size_channels = num_buffers_header_only_channels = 14`) and, per the function's
own comment, was calibrated for **experts_per_device=2** (deepseek_v3 on WH 6U, ~21 KB headroom at
hidden=7168). At experts_per_device=8 the per-core compute L1 tensor grows / starts lower and the
mux no longer fits, so an otherwise-valid MoE shape can't build its program.

## Expected
`moe_compute` should support `experts_per_device > 2` at hidden=7168 on Blackhole — either by sizing
the combine mux L1 region from the actual available L1 budget (reduce `num_buffers` adaptively when
the compute tensor leaves less headroom), or by documenting/validating the supported
(experts_per_device, hidden_size) envelope up front with an actionable error instead of a raw
overlap `TT_FATAL`.

## Steps (exact commands)
Save the script below as `repro_moe_compute_combine_l1_overlap.py` and run it on **unmodified** tt-metal:
```bash
source <tt-metal>/python_env/bin/activate
export TT_METAL_HOME=<tt-metal> PYTHONPATH=<tt-metal> ARCH_NAME=blackhole
timeout 300 python3 repro_moe_compute_combine_l1_overlap.py
```
Expected: prints `REPRO CONFIRMED` and the `Mux L1 memory [...] overlaps with L1 tensor [...]` line.
~1 min; all inputs synthetic (no model data); fails at program-creation (no device hang).

<details>
<summary><b>repro_moe_compute_combine_l1_overlap.py</b> (full source — verified to FATAL on tt-metal 918fad97336)</summary>

```python
# SPDX-License-Identifier: Apache-2.0
"""Minimal standalone repro: ttnn.experimental.moe_compute raises a mux-L1-overlap TT_FATAL
during program creation on a Blackhole 4x8 Galaxy (cluster_axis=0) for a DeepSeek-V3-scale MoE
shape: experts_per_device=8, hidden_size=7168, intermediate_size=2048, tokens_per_device=32.

The combine stage (selective_reduce_combine) sizes its fabric-mux L1 region with a Blackhole
buffer count (`num_buffers=14`) calibrated for experts_per_device=2 (deepseek_v3 on WH 6U). At
epd=8 the per-core compute L1 tensor sits ~8 KB below the mux end, tripping:
    TT_FATAL ... Mux L1 memory [base=..,end=..] overlaps with L1 tensor .. (assert.hpp)
in selective_reduce_combine_program_factory.cpp::launch_mux_workers.

All inputs are synthetic (no model data). The op FATALs at program-creation time, before any
device execution, so this is deterministic and does not hang.

Run (on UNMODIFIED tt-metal):
    source <tt-metal>/python_env/bin/activate
    export TT_METAL_HOME=<tt-metal> PYTHONPATH=<tt-metal> ARCH_NAME=blackhole
    timeout 300 python3 repro_moe_compute_combine_l1_overlap.py
Expected: RuntimeError with "Mux L1 memory [...] overlaps with L1 tensor [...]".
"""
import torch
import ttnn
from ttnn.operations.ccl import MoEActivationFunction
from ttnn.experimental.moe_compute_utils import get_tilize_drain_core

MESH = (4, 8)
CLUSTER_AXIS = 0
EPD = 8                 # experts per device  (vs the epd=2 BH calibration) -> the L1 driver
HIDDEN = 7168
N = 2048                # intermediate_size
K = 8                   # selected experts per token
TOKENS_PER_DEV = 32

num_devices = MESH[0] * MESH[1]                 # 32
num_dispatch = MESH[CLUSTER_AXIS]               # 4
experts = EPD * num_devices                     # 256 -> op infers experts_per_device = 256/32 = 8
total_tokens = TOKENS_PER_DEV * num_dispatch    # 128
shard_dims = (0, None)                          # cluster_axis=0: tokens sharded on mesh axis 0
DRAM = ttnn.DRAM_MEMORY_CONFIG

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
dev = ttnn.open_mesh_device(ttnn.MeshShape(MESH), l1_small_size=32768)
print(f"opened {dev}", flush=True)
drain = get_tilize_drain_core()

# ---- expert_mapping [num_devices, experts] uint16, replicated (contiguous: expert e -> device e//EPD) ----
em_t = (torch.arange(experts, dtype=torch.int32) // EPD).reshape(1, experts).repeat(num_devices, 1)
tt_em = ttnn.from_torch(em_t.to(torch.int32), device=dev, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16,
                        memory_config=DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(dev))

# ---- weights: synthetic, REPLICATED (each device holds EPD=8 experts), C++ prepare -> bfloat4_b ----
def _up_rm(t):  # ROW_MAJOR bf16 replicated (the layout the C++ prepare_* helpers expect)
    return ttnn.from_torch(t, device=dev, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                           memory_config=DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(dev))

w0 = _up_rm(torch.rand(1, EPD, HIDDEN, N, dtype=torch.bfloat16) - 0.5)
w1 = _up_rm(torch.rand(1, EPD, HIDDEN, N, dtype=torch.bfloat16) - 0.5)
w2 = _up_rm(torch.rand(1, EPD, N, HIDDEN, dtype=torch.bfloat16) - 0.5)
mc = ttnn.experimental.get_weight_mem_configs(
    dev, num_layers=1, experts_per_device=EPD, hidden_size=HIDDEN, intermediate_size=N, has_bias=False)
w0w1_p = ttnn.experimental.prepare_w0_w1_tensor_for_moe_compute(w0, w1, L=1, E=EPD, K=HIDDEN, N=N)
tt_w0w1 = ttnn.experimental.quantize_weights_via_host(w0w1_p, dtype=ttnn.bfloat4_b, memory_config=mc.w0_w1)
w2_p = ttnn.experimental.prepare_w2_tensor_for_moe_compute(w2, L=1, E=EPD, N=N, K=HIDDEN)
tt_w2 = ttnn.experimental.quantize_weights_via_host(w2_p, dtype=ttnn.bfloat4_b, memory_config=mc.w2)
print("weights prepared (bfloat4_b)", flush=True)

# ---- moe_compute inputs (the dispatch outputs), synthetic zeros with the expected shapes/mem-configs ----
sparse = ttnn.from_torch(
    torch.zeros(num_dispatch, total_tokens, HIDDEN, dtype=torch.bfloat16), device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH))
out_shard = ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(drain, drain)}), [total_tokens, K], ttnn.ShardOrientation.ROW_MAJOR)
out_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard)
oi = ttnn.from_torch(
    torch.zeros(num_dispatch, total_tokens, K, dtype=torch.uint16), device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16, memory_config=out_l1,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH))
os_ = ttnn.from_torch(
    torch.zeros(num_dispatch, total_tokens, K, dtype=torch.bfloat16), device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=out_l1,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH))
combine_out = ttnn.from_torch(
    torch.zeros(K, TOKENS_PER_DEV, HIDDEN, dtype=torch.bfloat16), device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM,
    mesh_mapper=ttnn.ReplicateTensorToMesh(dev))
worker = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
         ttnn.CoreCoord(dev.compute_with_storage_grid_size().x - 1, dev.compute_with_storage_grid_size().y - 1))})
comb_sem = ttnn.create_global_semaphore(dev, worker, 0)
mux = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))])

print("calling moe_compute (expect mux-L1-overlap TT_FATAL during program creation)...", flush=True)
try:
    outs = ttnn.experimental.moe_compute(
        sparse, oi, os_, tt_em, tt_w0w1, tt_w2,
        layer_id=0, output_height_shard_dim=4, intermediate_size=N, has_bias=False,
        cluster_axis=CLUSTER_AXIS, mux_core_range_set=mux,
        optional_output_tensor=combine_out, optional_cross_device_semaphore=comb_sem,
        activation_type=MoEActivationFunction.SILU)
    ttnn.synchronize_device(dev)
    print("UNEXPECTED: moe_compute did NOT raise (no L1 overlap on this build/shape)", flush=True)
except RuntimeError as e:
    msg = str(e)
    hit = "Mux L1 memory" in msg and "overlaps with L1 tensor" in msg
    print(f"REPRO {'CONFIRMED' if hit else 'raised (different error)'}:", flush=True)
    print(msg.splitlines()[0] if msg else "", flush=True)
    for line in msg.splitlines():
        if "Mux L1 memory" in line:
            print(line, flush=True)

ttnn.close_mesh_device(dev)
```
</details>

## Input data / link or description
None — all inputs are synthetic (generated in the repro script). No model weights or captured tensors needed.

## Frequency
Always (deterministic; fires at program creation, independent of input values).

## System Details
- **Software:** tt-metal `918fad97336` (`v0.73.0-dev20260604-17-g918fad97336`); OS Ubuntu 24.04.4
  LTS; kernel 6.8.0; KMD 2.5.0; firmware bundle 19.6.0. Build: `build_Release` (Release).
- **Hardware:** Blackhole, Galaxy 6U — 32 chips exposed as a 4×8 mesh (`cluster_axis=0` → 4-device
  dispatch ring, 8 replicated). `FabricConfig::FABRIC_1D_RING`.

## Logs & Diagnostics
```
critical | Always | TT_FATAL: Mux L1 memory [base=0x1b200, end=0x95330] overlaps with L1 tensor
0x93f00 and is in danger of being clobbered. (assert.hpp:104)
RuntimeError: TT_FATAL @ .../selective_reduce_combine_program_factory.cpp:90:
  mux_kernel_config.get_memory_map_end_address() <= *occupied_l1_tensor_addr
```

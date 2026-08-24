"""Minimal reproducer: 2-chip (N300) small-message all-gather LATENCY.
No model, no prefill. Times the synchronized trace loop host-side (no device profiler).
Decode collectives are tiny/latency-bound; this is the smallest faithful test of the
inter-chip collective path that the scale sweep (§32) localized the f10cs05 deficit to.
Run on f10cs05 vs a good host; compare per-op latency."""
import time, pytest, ttnn
import tests.nightly.t3000.ccl.test_all_gather as agmod
from tests.nightly.t3000.ccl.test_all_gather import run_all_gather_impl, create_fabric_router_config

# tiny -> small decode-scale shapes (bf16), gathered on last dim across the 2 chips
SHAPES = [([1,1,32,2048],3,"tiny_32x2048"), ([1,1,32,4096],3,"decode_32x4096"), ([1,1,128,4096],3,"small_128x4096")]
IDS=[s[2] for s in SHAPES]

@pytest.mark.parametrize("mesh_device", [(1,2)], indirect=True)
@pytest.mark.parametrize("device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
      "fabric_router_config": create_fabric_router_config(6144), "trace_region_size": 90112}],
    indirect=True, ids=["fabric_ring"])
@pytest.mark.parametrize("shp", SHAPES, ids=IDS)
def test_n300_ag_latency(mesh_device, shp):
    shape, dim, name = shp
    iters = 100
    timings = {}
    orig = agmod.signpost
    agmod.signpost = lambda m,*a,**k: timings.__setitem__(m, time.perf_counter())
    try:
        run_all_gather_impl(mesh_device, shape, dim, ttnn.bfloat16, ttnn.TILE_LAYOUT,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            num_links=1, all_gather_topology=ttnn.Topology.Ring, num_iters=iters,
            enable_trace=True, use_persistent_buffers=True, skip_check=True)
    finally:
        agmod.signpost = orig
    per_us = (timings["stop"]-timings["start"])/iters*1e6
    print(f"\nN300_AG_LAT shape={name}{shape} per_op={per_us:.1f}us\n")

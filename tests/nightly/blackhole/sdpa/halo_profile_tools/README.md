# Gemma4 sliding halo / lanes profiling (local, not for merge)

Probe zones (`HW_OPEN`, `HW_SEND`, `HW_CLOSE`, `HR_READ`, `HR_ARRIVE`, `SDPA_HALO_WAIT`, `R_Q`, `R_KCHUNK`, `C_MERGE`)
are in the halo reader/writer and the ring joint SDPA reader/compute kernels. They compile to nothing unless the
device profiler is on.

Profile one sliding call (HP_B=1) or a lanes call (HP_B=4), traced, model layout (row-sharded cache and halo
buffers, ring fabric with 8 KB packets):

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SYNC=1 TT_METAL_PROFILER_DIR=/tmp/prof_b4 \
    LANES_AB_FABRIC_RING=1 LANES_AB_FABRIC_PAYLOAD=8192 HP_B=4 HP_TRACE=1 HP_CALLS=5 \
    scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_halo_profile_local.py -s

`HP_ND=0` without the fabric variables gives the setup of `test_lanes_ab_local.py` (interleaved buffers, 1D fabric).
`HP_KS` sets k_splits.

Summaries:

    python tests/nightly/blackhole/sdpa/halo_profile_tools/parse_halo_prof.py /tmp/prof_b4/.logs/profile_log_device.csv 5
    python tests/nightly/blackhole/sdpa/halo_profile_tools/parse_attn.py /tmp/prof_b4/.logs/profile_log_device.csv 5
    python tests/nightly/blackhole/sdpa/halo_profile_tools/per_dev.py /tmp/prof_b4/.logs/profile_log_device.csv 5

Model step A/B (`models/demos/gemma4_d_p/tests/test_batched_prefill.py::test_batched_step_perf`):
`GEMMA4_SLIDING_LANES=0` runs one ring SDPA per request instead of lanes; `GEMMA4_SLIDING_QC` /
`GEMMA4_SLIDING_KSPLIT` override the sliding config; `PERF_LANE_COUNTS`, `PERF_MIXES`, `PERF_SKIP_ALONE=1` trim the run.

Two SP4 x TP4 halves (`models/demos/gemma4_d_p/tests/test_request_parallel_local.py`) need `GEMMA4_ALLOW_SUBMESH=1`,
`GEMMA4_SLIDING_LANES=0` and a TT_CACHE_PATH holding a `tensor_cache_bf16_mesh4x4` cache (built on first run with
`GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1`).

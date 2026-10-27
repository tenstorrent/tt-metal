# MoeFusedSwiGlu intermittent hang (GLM-5.3, 4x4 stage, SP on axis 1)

## What happened

Three hangs, all on 2026-10-26, each on the **first** `MoeFusedSwiGlu` call of the process:

| Node | Workload | Slurm job | Raw tt-triage |
| --- | --- | --- | --- |
| bh-glx-120-c07u20 | `test_glm_prefill_transformer_chunked`, 4x4, L10 | 154634 | [triage-c07u20-job154634-chunked-L10.txt.gz](triage-c07u20-job154634-chunked-L10.txt.gz) |
| bh-glx-120-c06u20 | 2-rank prefill runner (12 layers over two 4x4 stages) | 154783 | [triage-c06u20-job154783-runner.txt.gz](triage-c06u20-job154783-runner.txt.gz) |
| bh-glx-120-c08u08 | `../pcc/test_fused_swiglu_hang_repro.py` | 154922 | [triage-c08u08-job154922-repro.txt.gz](triage-c08u08-job154922-repro.txt.gz) |

tt-triage signature (same all three times):

- Reader NCRISCs spin in `noc_async_read_barrier_with_trid(NEXT_X_TRID)` (trid 15), the NEXT_X
  barrier in `moe_fused_swiglu_reader.cpp` (line 1271 before 7cadfcab288, 1297 after).
- Compute, writer, combine and reduce-scatter wait behind the readers.
- NIU counters: read requests sent == read responses received, so no reads were lost; only the
  trid 15 outstanding counter is nonzero.

Ruled out: lost reads; a trid 15 counter left dirty by earlier ops (after a reset no GLM op leaves it
dirty, and an unfixed control did not hang on a dirty counter). A fix based on that theory was dropped.

## Repro

Both tests need a 4x4 mesh: use `run_4x4_stage_tests.sh` (`--half top` or `--half bottom`) on a Galaxy.

**Standalone loop** (cheapest; `tests/pcc/test_fused_swiglu_hang_repro.py`, not for CI). It runs

NOTE: The hang is very intermittent from the minimal repro alone.

`test_glm_moe`'s 4x4 1280-rows-per-chip row with the GLM hybrid split (320) and loops the forward. A
watchdog writes `HANG_MARKER` when an iteration stalls and leaves the device hung for triage:

```bash
cd $TT_METAL_HOME
REPRO_ITERS=400 REPRO_ITER_TIMEOUT_S=420 HANG_MARKER=/tmp/fused_swiglu_hang \
  models/demos/deepseek_v3_d_p/tests/run_4x4_stage_tests.sh --half top -- \
  models/demos/deepseek_v3_d_p/tests/pcc/test_fused_swiglu_hang_repro.py \
  -k "skew0.0 and not skew0.05" -rA -s
```

`skew0.05` / `skew0.2` add a random gate bias so experts land on both sides of the split. Since the
hangs were all on the first call, many fresh processes with `REPRO_ITERS=3` cover that case better
than one long loop. `TT_DS_PREFILL_DEBUG_TOKEN_COUNT=1` matches the c08u08 run.

**The workload that hung first** (needs the 12-layer 4x4 cache, ~51 GB):

NOTE: This is the most reliable way to repro the hang.

```bash
export GLM53_HF_MODEL=/mnt/weka/model-weights/llm/zai-org/GLM-5.3-fp8-aca966e4 HF_HUB_OFFLINE=1
export TT_GLM53_PREFILL_TTNN_CACHE=/mnt/weka/model-cache/scratch/rkim/glm53_ttnn_cache   # full 78-layer 4x4_sp1 cache, Galaxy nodes only
models/demos/deepseek_v3_d_p/tests/run_4x4_stage_tests.sh --half top -- \
  models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm_prefill_transformer_chunked \
  -k "4x4 and L10 and cold_cache and notrace" -rA
```

Build the cache first with `test_prefill_transformer.py::test_glm_build_ttnn_cache -k "4x4 and 12_layers"`.

The raw dumps (gzipped to stay under the repo's 500 KB file limit; read with `zless` or `zgrep`) are the `dump_aggregated_callstacks` and `dump_op_window` output, taken while each
process was still hung. Their line numbers are from the reader before 7cadfcab288, so the NEXT_X barrier
shows as `moe_fused_swiglu_reader.cpp 1271`. A full triage run of the c08u08 hang (every script, 6 MB,
plus its sqlite) is at `/data/amorrison/jobs/glm_hang/triage-full-154922.{txt,sqlite}` on exabox.

## Stall report (opt-in)

Running with DPRINT on compiles in a stall report at the NEXT_X barrier (7cadfcab288). It is off by
default and costs nothing when off:

```bash
export TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_RISCVS=NC TT_METAL_DPRINT_FILE=/tmp/dprint.txt
```

After 50M spins each stuck reader prints two `NEXT_X STALL` lines once: expert/block, the trid 15
counter at kernel entry and before/after its last x-row prefetch, the trid 14 counter, and NIU reads
sent/received. Then it keeps waiting, so tt-triage still works. `grep "NEXT_X STALL" /tmp/dprint.txt`.

NOTE: Enanling DPRINT seems to make the hang go away.

## tt-triage

```bash
cd $TT_METAL_HOME && source python_env/bin/activate
TT_METAL_HOME=$PWD PYTHONPATH=$PWD python tools/triage/triage.py --llm-output --disable-progress \
  --skip-version-check --run=dump_aggregated_callstacks --run=dump_op_window > triage.txt 2>&1
```

Under Slurm, attach to the live job from the login node. `--propagate=NONE` is required; without it
the step inherits the login node's 512-process limit and can't fork:

```bash
srun --propagate=NONE --jobid=<job> --overlap -w <node> bash -c '<the triage command above>'
```

Look for readers in `noc_async_read_barrier_with_trid` at the NEXT_X line. To read the per-trid NIU
counters directly with tt-exalens: NOC0 `NIU_MST_REQS_OUTSTANDING_ID(id)` is at
`0xFFB20200 + (0x10 + id) * 4` (NOC1 base `0xFFB30200`). RD_REQ_SENT is register 0x5 and
RD_RESP_RECEIVED is 0x2. The counters are 8 bits and are shared by reads and non-posted writes. A
dump script is at `/data/amorrison/jobs/glm_hang/dump_trid_counters.py` on exabox.

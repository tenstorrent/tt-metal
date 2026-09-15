# Task 4 report: dense Llama SwiGLU MLP

## Summary

PASS. The dense Llama-3.1-8B SwiGLU MLP is implemented and validated on the assigned
4x8 Blackhole Galaxy. The final gate ran the exact TP collective probe and five full MLP calls
over all 32 chips, including real layer-0 checkpoint weights, with two passing tests and no skips.

Tested base commit: 1ff25da02aec95996e5ccc4005b70cb51a943563
(Add Llama RMSNorm for Galaxy prefill).

Tested owned-file hashes:

- tt/mlp.py: acc1eaecafdcf1f23393a628d4af1ed5da0c2a39447f4a1123cfafef39ba236e
- tests/unit/test_mlp_vs_ref.py:
  7e1e2525dce3f10698a89f6dd2763c925bd22ce48d2c70e0d6af05e1b88d43d7

This report is included in the single Task 4 commit. Its final SHA is recorded in the controller
handoff because a commit cannot contain its own hash.

## Changes

- Added models/demos/llama_3p1_8b_d_p/tt/mlp.py.
  - Accepts MLP(mesh_device, mesh_config, state_dict) with the three HF-orientation tensors.
  - Validates the fixed 4x8 mesh, SP=4 on axis 0, TP=8 on axis 1, actual mesh/config agreement,
    32-device coverage, CPU tensor ownership, and exact HF weight shapes before any upload.
  - Transposes HF weights to TTNN matmul orientation, stores BF16 TILE DRAM weights, shards gate/up
    output width with MeshConfig.column_parallel, shards down input width with
    MeshConfig.row_parallel, and replicates both over SP.
  - Proves forwarding indices 0 and 1 for both TP neighbors from every mesh coordinate before
    uploading weights. Every call requires public get_usable_topology with Ring on axis 1 to
    resolve to Ring before compute.
  - Runs two column-parallel matmuls, fused lhs SiLU/multiply, one row-parallel down matmul, and
    public two-link TP Ring all_reduce. It leaves the caller input live and deallocates only owned
    intermediates after their final consumer.
- Added models/demos/llama_3p1_8b_d_p/tests/unit/test_mlp_vs_ref.py.
  - Uses installed Transformers 5.12.1 LlamaMLP in float32 as the independent oracle after BF16
    rounding the same input and weights.
  - Loads only the three layer-0 MLP tensors through the checkpoint safetensors index.
  - Checks every chip, SP row placement, TP replication, finite output, exact unchanged input,
    exact zero, numerical thresholds, validation failures, program-cache reuse, and changed
    allocation addresses.
  - Includes an exact BF16 TP-sum probe and a structured fixture that distinguishes an omitted
    TP reduction, swapped gate/up projections, and GPT-style clamping while exercising positive
    and negative gate activations beyond magnitude seven.

No RMSNorm, RoPE, decoder-layer, attention, cache, migration, or generalized checkpoint-loading
module was edited.

## Required source and synchronization preflight

Before launch, both controller-owned audits were read:

- /data/divanovic/llama31-8b-disagg/evidence/mlp-matmul-preflight.md
- /data/divanovic/llama31-8b-disagg/evidence/mlp-collective-preflight.md

The matmul audit covered the selected standard dense-interleaved
matmul_multicore_reuse_mcast_2d path, including balanced CB traffic, four matched
sender/receiver semaphores, output barriers, no dependency cycle, and cached address overrides.
It also covered the selected no-broadcast binary-ng SiLU/multiply path: one gate tile and one up
tile feed one SiLU intermediate and one output tile per iteration, without semaphores or fabric.

The collective audit covered the standard TP8 Ring reduce-scatter and unicast all-gather selected
by public ttnn.all_reduce: two links, two workers per direction, matched CB page counts and
cross-device semaphores, startup/credit/teardown ordering, destination addresses, cached address
overrides, one fabric connection per direction/link, and route helpers. No synchronization cycle
was found in the scoped source walk.

These were conditional source audits. The final run supplied the required live proof:

~~~text
mesh shape=(4, 8), devices=32
committed physical grouping=4x8_Mesh_flat_torus_xy (TORUSXY)
fabric config=FABRIC_1D_RING
all 64 directed +/-TP edges: forwarding_link_indices=(0, 1)
get_usable_topology(probe, Ring, cluster_axis=1)=Topology.Ring
~~~

The exact-sum collective completed and matched every value on every chip. The runtime selected the
audited unicast all-gather factory, visible in the raw all_gather_unicast_factory.cpp warning.

## Runtime configuration and L1

~~~text
gate/up: grid=7x8, in0_block_w=4, subblock=1x4,
         per_core_M=1, per_core_N=8, transpose_mcast=0,
         fused_activation=null, fuse_batch=0
down:    grid=8x8, in0_block_w=4, subblock=1x4,
         per_core_M=1, per_core_N=16, transpose_mcast=0,
         fused_activation=null, fuse_batch=0
compute: HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
~~~

The audited named L1 requirements were 196,608 bytes/core for each gate/up matmul,
376,832 bytes/core for down, and 14,336 bytes/core for fused multiply. Final live values:

~~~text
total_bytes_per_bank=1461248
largest_contiguous_bytes_free_per_bank=1461248
required_named_bytes_per_core=376832
~~~

The largest live free block exceeded the maximum scoped requirement by 1,084,416 bytes.

## TDD evidence

RED command:

~~~bash
source /data/divanovic/llama31-8b-disagg/tools/prefill_env.sh
cd /data/divanovic/llama31-8b-disagg/repos/tt-metal
"$PREFILL_PYTHON" -m pytest models/demos/llama_3p1_8b_d_p/tests/unit/test_mlp_vs_ref.py --collect-only -q > /data/divanovic/llama31-8b-disagg/evidence/task-4-mlp/red.log 2>&1
~~~

RED result: exit 2 after 83.08 seconds with the expected missing production module:

~~~text
ModuleNotFoundError: No module named 'models.demos.llama_3p1_8b_d_p.tt.mlp'
no tests collected, 1 error
~~~

Raw RED log: /data/divanovic/llama31-8b-disagg/evidence/task-4-mlp/red.log.

After implementation, host collection found both tests and the host structured fixture plus
installed Transformers one-row oracle passed:

~~~text
2 tests collected in 53.17s
HOST_HELPERS_PASS (1, 1, 1, 4096) torch.float32 True
~~~

Raw host logs:

- /data/divanovic/llama31-8b-disagg/evidence/task-4-mlp/host-collect-green.log
- /data/divanovic/llama31-8b-disagg/evidence/task-4-mlp/host-helpers.log

## Galaxy command and results

Both attempts sourced tools/prefill_env.sh, set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, retained the
repository root conftest, and used the exact torus-xy graph descriptor.

The shared flock covered the complete command:

~~~bash
set -o pipefail; flock -x /data/divanovic/llama31-8b-disagg/tmp/prefill-device.lock srun --overlap --jobid 105036 --nodes=1 --ntasks=1 --cpu-bind=none -w bh-glx-110-c07u20 bash /data/divanovic/llama31-8b-disagg/tmp/task4_mlp_run.sh 2>&1 | tee /data/divanovic/llama31-8b-disagg/evidence/task-4-mlp/green-device-attempt2.log
~~~

The script ran:

~~~bash
timeout 5400 "$PREFILL_PYTHON" -m pytest models/demos/llama_3p1_8b_d_p/tests/unit/test_mlp_vs_ref.py --tb=native -p no:cacheprovider -v -s
~~~

Attempt 1 ran from 19:57:52 to 20:00:14 UTC. The exact collective and all numerical checks passed,
but the test ended 1 failed/1 passed because the expected-error message used unescaped parentheses
as a pytest regex. Production behavior was correct. The expectation changed to the stable
requires mesh_shape prefix; no production code, fixture, oracle, or threshold changed.

Raw failed attempt:
/data/divanovic/llama31-8b-disagg/evidence/task-4-mlp/green-device-attempt1.log.

Final attempt:

~~~text
TASK4_MLP_START_UTC=2026-09-15T20:01:08Z
2 passed, 1 warning in 91.72s (0:01:31)
TASK4_MLP_END_UTC=2026-09-15T20:03:15Z
TASK4_MLP_EXIT=0
~~~

There were no skips. Device teardown completed cleanly.

| Case | Minimum PCC | Maximum normalized L2 |
| --- | ---: | ---: |
| synthetic structured | 0.9999951 | 0.0040297 |
| real layer-0 checkpoint | 0.9999905 | 0.0047243 |
| synthetic changed input | 0.9999945 | 0.0041023 |
| synthetic exact zero | exact zero | 0.0000000 |
| synthetic structured return | 0.9999951 | 0.0040297 |

Thresholds remained PCC >= 0.999 and normalized L2 <= 0.02.

The first call used input/output addresses 0x140080/0x188080 on every chip. Two retained guards
forced every later call to 0x1c0080/0x208080. The program cache remained at eight entries across
all five MLP calls while switching synthetic to real weights, changing inputs, and returning to
the first synthetic case. Final JIT summary: 187/202 hits.

Raw final GREEN log:
/data/divanovic/llama31-8b-disagg/evidence/task-4-mlp/green-device-attempt2.log.

## Hooks and self-review

Initial owned-file hooks passed, including black, autoflake, isort, expected-error enforcement,
merge-conflict, large-file, and Metalium API checks:
/data/divanovic/llama31-8b-disagg/evidence/task-4-mlp/precommit-initial.log.

Final hooks passed after this report was added:
/data/divanovic/llama31-8b-disagg/evidence/task-4-mlp/precommit-final.log.

Self-review findings and resolutions:

- Replaced a mistakenly large wrong-shape validation tensor with a 1x1 tensor.
- Added explicit runtime configuration and validated-edge logging before the first MLP launch.
- Fixed only the expected-error regex after attempt 1; numerical logic and thresholds stayed fixed.
- Re-read both owned files against the brief. Weight orientation, sharding axes, input contract,
  lifetime ownership, program configs, collective arguments, independent oracle, checkpoint
  selection, sensitivity checks, all-chip checks, and required comments are present.

## Concerns and limitations

- Final pytest reports one Unknown config option: timeout warning because plugin autoload was
  deliberately disabled as required; the run had an external timeout 5400.
- TTNN warns that all-gather semaphores in L1 can fragment later headroom and that the default
  4352-byte packet is suboptimal for 2048-byte pages. Live capacity passed with large margin and
  correctness is unaffected; these are future performance/allocation concerns.
- The module intentionally supports only the required 4x8 Galaxy, local [1,1,256,4096] BF16 TILE
  interleaved-DRAM activation, and host-provided BF16-convertible weights. General caching/loading
  and other chunk sizes remain later-task scope.

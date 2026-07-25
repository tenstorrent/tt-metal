# Serialized overlapping-mesh repro

Capture time: 2026-07-17T07:08:58Z

## Process

```text
wrapper PID: 1026256
pytest PID:  1026257
command: timeout 600 pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_synthetic_non_aligned_prefill_matches_optimized_and_cache_is_head_local --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/synthetic_prefill_cache.junit.xml
elapsed at final capture: 00:05:02
```

The test had been changed to call
`ttnn.synchronize_device(baseline.mesh_device)` and materialize the baseline
result before calling `multichip.prefill_forward`.  Both decoder objects were
still constructed before either forward: the baseline owns a `1x1` child view
of the same physical mesh on which the multichip decoder and its CCL resources
were constructed.

## Host stacks

Main thread:

```text
__pthread_cond_wait
tt::tt_metal::distributed::FDMeshCommandQueue::finish_nolock
tt::tt_metal::distributed::FDMeshCommandQueue::finish
tt::tt_metal::distributed::Synchronize
ttnn.synchronize_device
```

Completion-reader LWP 1026364:

```text
tt::umd::SysmemManager::read_from_sysmem
tt::tt_metal::read_cq_host_ptr<true>
tt::tt_metal::SystemMemoryManager::completion_queue_wait_front
tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue_event
tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue
```

The main thread is therefore blocked in the explicit baseline-child
synchronization.  It has not entered the parent-mesh multichip forward or an
all-reduce in this reproduction.

## Running operations

Captured with the diagnostics-only compatible triage shim:

```text
python models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/triage/run_compatible_triage.py \
  --run=dump_running_operations --llm-output --disable-progress --disable-colors
```

Only physical device 1 reported decoder work:

| Op | Key logical shapes | Devices | Active cores |
| --- | --- | --- | --- |
| 59 SDPA | Q `[1,64,17,64]`; K/V `[1,8,17,64]` | `1` | 24 |
| 61 Matmul | `[1,17,4096] @ [4096,2880]` | `1` | 2 |
| 73 Softmax | `[1,1,17,4]` | `1` | 79 |
| 74 Unary | `[17,32]` | `1` | 1 |

These are the full-width optimized single-chip shapes.  The TP=2 path would
have 32 local Q heads, 4 local KV heads, and a 2048-wide local attention result
per rank.  Device 0 had no running decoder operation.

## Comparison with first capture

The first unsynchronized reproduction reached parent-mesh all-reduce semaphore
setup and blocked while draining the older device-1 baseline work.  This
serialized reproduction never reaches the multichip forward: the explicit
child-mesh drain itself blocks on the same single-chip operations.  Baseline-
only immediate synchronization/materialization and multichip-only controls
both passed after reset.  The differentiator is therefore the simultaneous
lifetime/use of the overlapping parent and child mesh views, including parent-
mesh CCL/global-semaphore initialization before the child forward, rather than
the absence of a synchronization call.

No process was killed and no device was reset while collecting this evidence.

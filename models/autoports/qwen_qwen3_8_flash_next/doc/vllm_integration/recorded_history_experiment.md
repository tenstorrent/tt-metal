# Fixed scheduler history experiment (both replay modes passed)

This diagnostic isolates read finalization timing while preserving one actual
served scheduler history. It addresses the physical-width confound in the failed
mixed v3/v4 comparisons without changing those artifacts or model numerics.
The factory returns the production `QwenFlashNextForCausalLM` object. Its hooks
record CPU metadata and existing waits; they add no device wait or tensor read.
File writes and deep copies add host overhead, so this is not performance evidence.

Use a fresh absolute experiment directory. The dedicated launcher is required:
`run_reduced_server.sh` unconditionally exports the production factory and would
overwrite a diagnostic override supplied by its caller. This launcher uses layers
0,1,3, context1024, B2, page32, chunk budget256, async scheduling, and device sampling.

```bash
cd /home/ttuser/qwen38-flash/tt-metal
experiment=/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/recorded_history_v1
bash models/autoports/qwen_qwen3_8_flash_next/tests/run_vllm_history_server.sh "$experiment"
```

Once that diagnostic server is ready, in a second shell run exactly one pair:

```bash
source /home/ttuser/qwen38-flash/vllm-env/bin/activate
source /home/ttuser/qwen38-flash/lane-env.sh
cd /home/ttuser/qwen38-flash/tt-metal
experiment=/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/recorded_history_v1
timeout 1800 python models/autoports/qwen_qwen3_8_flash_next/tests/check_vllm_recorded_pair.py --recording "$experiment/recording"
```

The client admits prompt129 after the first generated token of prompt513, with
96 greedy output tokens each, seed42, and EOS ignored. It preserves streamed
chunks, exact IDs, response identities, usage and partial failures. An exclusive
attempt marker prevents reuse. Recordings preserve the complete SchedulerOutput,
sampling configuration, scheduler-owned KVCacheConfig, CPU model inputs, reload
plans, resident positions, page tables, row identities, scheduler commits and
the existing executor's read-finalization order. Only trusted local pickles may
be loaded.

Stop this server cleanly and audit EngineCore/device ownership before replay.
Run each replay in a separate process, serialized with all other device jobs:

```bash
source /home/ttuser/qwen38-flash/vllm-env/bin/activate
source /home/ttuser/qwen38-flash/lane-env.sh
cd /home/ttuser/qwen38-flash/tt-metal
export VLLM_TT_PLUGIN_ROOT=/home/ttuser/qwen38-flash/vllm-tt-plugin
export TT_VISIBLE_DEVICES=0,1,2,3 MESH_DEVICE=P300x2
export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
export QWEN_VLLM_LAYER_INDICES=0,1,3
export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export OMP_NUM_THREADS=8
unset TT_METAL_DEVICE_PROFILER TT_METAL_WATCHER QWEN_VLLM_HISTORY_DIR
experiment=/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/recorded_history_v1
timeout 3600 python -m models.autoports.qwen_qwen3_8_flash_next.tests.replay_vllm_history --recording "$experiment/recording" --read-mode immediate --output "$experiment/immediate.json"
timeout 3600 python -m models.autoports.qwen_qwen3_8_flash_next.tests.replay_vllm_history --recording "$experiment/recording" --read-mode recorded_queue --output "$experiment/recorded_queue.json"
```

Replay restores the real UniProcExecutor/TTWorker, production adapter, scheduler
pool configuration and warmup. Both columns retain async configuration; one
resolves each returned handle immediately, the other follows the recorded
executor resolution order. Every returned/committed request/token array must
match the serving recording, and recorded scheduler emissions must match HTTP
IDs. Host advisory decode tokens/positions are excluded only when comparing
model-input bookkeeping; effective reload-authoritative tokens/positions,
resident positions, active masks, sampling parameters, page tables, physical
widths and remaps are checked separately. Queued replay must show actual pending
overlap and zero unsafe overlaps. Failure artifacts preserve the first differing
boundary or contract. No comparison is weakened to text, prefixes or tolerances.

The pinned runner's `_update_states` consumes cached request positions/blocks,
not recorded cached output token arrays; generated outputs are produced by each
replay. This is a fixed-input runner replay, not a rerun of scheduler policy.
Recorded commits provide the scheduler/API correctness oracle. It assumes the
pinned UniProc executor shares scheduler object identities in one process and
that its VllmConfig/cache configuration are pickleable after removing the live
mesh. Those integration assumptions remain runtime checks, not demonstrated facts.
The recording must include mixed prefill/decode and actual overlap; inspect the
reported execution signatures (including chunk starts) before accepting coverage.
No full-layer correctness, maximum-context, cache tensor equality or performance
claim follows from this reduced experiment. Existing direct transition v2 covers
full state comparisons under its own reduced fixture. Preserve all original
failed controls and any failed recording/replay; use a new directory for a rerun.

Source-only checks passed: Black formatting, compilation of all three Python
files, `bash -n` on the launcher, and stdlib-only extracted helper checks for
contiguous events, rejection of truncated/noncontiguous recordings, and separate
prefill/decode contract projection without input mutation. No runtime TTNN
imports, HTTP requests, device jobs or servers were executed for these checks.

Hardware result: recording v1 completed; immediate_v2 and recorded_queue_v2 both passed200 history and307 exact output checks/103 commits. Queued overlap91, unsafe0. Immediate overlap0. The initial pre-device identity check failed because upstream appends an eight-character random suffix; corrected to require a unique matching API-choice prefix and exact token equality. Original log retained. See `AUTOFIX_async.md` for scope and limitations.

# AutoDebug: P1 retains Qwen trace buffers between precision candidates

## Finding and repair boundary

**The P1 runner never calls QwenGenerator.close(). Its cleanup hook only looks for teardown(), which QwenGenerator does not implement. Consequently the first four candidates leave their trace buffers owned by the shared mesh, and the fifth candidate exhausts the trace region.** This is a concrete lifecycle mismatch, with high confidence from source inspection, matching logs, and a passing standalone contrast. Hardware confirmation of the proposed repair remains outstanding.

The minimum repair for this sweep is a **stage-local P1 entry-point adapter under doc/datatype_sweep/** that synchronizes and explicitly closes each returned Qwen generator before the runner deletes it. Keep the installed readiness runtime, model `tt/` source, precision policies, and scoring settings unchanged. Resume in a fresh process/mesh, reuse the four valid score caches, and finish the missing candidate. Do not treat its allocation error as a numerical rejection or a reason to end the sweep.

This investigation wrote only this report. It did not edit source, invoke another AutoDebug launcher, execute the model, access devices, reset hardware, or run the sweep. Implementation and continued hardware work belong to the calling repair session under the inspection-only contract.

Paths below are repository-relative unless prefixed with `R/`. `R/` means:

`/home/ttuser/qwen36-27b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/`.

## Direct observations

The primary command receipt records the five candidates on one P300x2 mesh, FABRIC_1D, `READINESS_TRACE_REGION_SIZE=100000000`, four texts, and `--hf-positions 0` (`models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/p1_primary.command.json:15-64`; candidate map in `p1_primary_candidates.json`). Candidate order and captures in `p1_primary.log` are:

| Candidate | Build line | Capture lines | Outcome |
| --- | ---: | --- | --- |
| bf16_all_hifi4 | 21 | 108-112 | decode, sample, reset, prefill_128; score cache exists |
| bfp8_baseline | 113 | 199-202 | same four captures; score cache exists |
| gdn_attn8_hifi2_head8_hifi2 | 203 | 289-292 | same four captures; score cache exists |
| gateup4_lofi | 293 | 379-382 | same four captures; score cache exists |
| down4_lofi | 383 | 469 | first decode capture fails; no score cache |

The error at log lines 470-484 identifies allocation of a **TRACE** buffer during `end_trace_capture -> end_mesh_trace -> populate_mesh_buffer -> BankManager::allocate_buffer`. It is not a KV-cache, weight-tensor, or L1 allocation error.

| Allocator quantity | Per bank, bytes | Across eight banks, bytes |
| --- | ---: | ---: |
| Reserved capacity | 12,500,992 | 100,007,936 |
| Already allocated | 11,558,912 | 92,471,296 |
| Free / largest free block | 942,080 | 7,536,640 |
| New bank footprint requested | 1,007,616 | 8,060,928 |
| Shortfall | 65,536 | 524,288 |

The requested logical trace buffer is 8,019,968 bytes. With the current 8,192-byte trace page size on this large trace, that is 979 pages, rounded to 123 pages per bank by `calculate_bank_size_spread()` (`tt_metal/impl/allocator/allocator.cpp:784-793`; trace page selection in `tt_metal/impl/trace/dispatch.cpp:211-252`). Thus `123 * 8192 = 1,007,616` bytes per bank. The reserve calculation rounds `ceil(100000000 / 8)` to an 8,192-byte boundary, giving exactly the reported 12,500,992 (`allocator.cpp:52-64`; constants in `tt_metal/impl/trace/trace_buffer.hpp:17-19`). Free space equals the largest free block: this failure needs no fragmentation explanation.

There is a controlled passing sibling: `full_down4_lofi.json:2-5` records a pass at `max_seq_len=262144` with the same down4 config path. Its command receipt has exactly the same ten source hashes as primary and the same 100,000,000-byte trace region; `full_down4_lofi.log:107-111` records all four captures successfully. This supports standalone trace-set capacity for the policy. It is not a completed P1 reasoning measurement, because the sibling uses different input texts/lengths.

I recomputed all ten source hashes from the primary receipt; every one matches the prepared tree, including `tt/generator.py` SHA256 `43c6d811d3f5d1008a3cd9d6ca92552f130188bed17d1c395bdf7c0a5a505566`. The ownership investigation below uses current C++ source; the receipt does not independently fingerprint all allocator/trace C++ source or the loaded binary.

## Causal lifecycle

1. **One mesh survives all candidates.** `R/run_precision_reasoning.py:928-932` opens the mesh once, calls `run()`, then closes it in the outer `finally`. There is no inter-candidate mesh close or `deallocate_buffers()` call.

2. **P1 legitimately captures traces even though its main accuracy calculation is prefill.** `score_generator()` first scores a text's prefill logits, then runs traced teacher forcing (`R/run_precision_reasoning.py:428-443,447-459`). `generate()` calls `prepare()` (`tt/generator.py:881`). Preparation captures decode, sample, reset, and prefill_128 (`tt/generator.py:509-547`). The fifth failure therefore occurs during preparation for traced decode, before its first decode replay; some first-text prefill work may already have completed. No completed candidate rows are saved until the whole scoring call returns (`R/run_precision_reasoning.py:673-678`).

3. **The expected cleanup is skipped for every successful candidate.** `R/run_precision_reasoning.py:462-466` does only `getattr(generator, "teardown", None)`, optionally calls it, and collects Python garbage. Candidate `finally` calls that function, deletes the generator, and collects again (`:684-687`). Qwen implements `close()` at `tt/generator.py:933-936`, which calls `ttnn.release_trace()` for all entries and clears the dictionary. It has no `teardown()` or `__del__()`. Its `Generator` ABC also provides neither cleanup hook (`R/contract.py`), so inheritance does not bridge the names.

4. **Python garbage collection cannot release mesh-owned traces.** The Python dictionary contains trace IDs, not owning handles. `MeshTraceId` is a `StrongType<uint32_t>` (`tt_metal/api/tt-metalium/mesh_trace_id.hpp:13`). At capture begin, the mesh creates an entry in the active sub-device manager's trace pool (`tt_metal/distributed/mesh_device.cpp:1733-1741`; `tt_metal/impl/sub_device/sub_device_manager.cpp:121-125`). That pool owns shared trace buffers independently of the generator. The explicit release chain is `ttnn.release_trace()` (`ttnn/cpp/ttnn/operations/trace.cpp:31-33`) -> mesh release (`mesh_device.cpp:1686-1703`) -> pool erase (`sub_device_manager.cpp:128`) -> trace-buffer destruction/accounting decrement (`tt_metal/distributed/mesh_trace.cpp:170-174`) and owned buffer deallocation. Losing a numeric ID does not execute this chain.

5. **Accumulation matches the observed boundary.** Four successful four-trace sets leave 16 traces in this mesh by the inspected control flow. The fifth first decode trace requests about 8 MB with only about 7.54 MB of bank capacity left. The pool has no dependency on the prior Python generator remaining alive, so deletion/collection does not prevent this accumulation. The source does contain a Python reference cycle (`tt/generator.py:223-225` -> `tt/prefill_trace.py:27` -> generator); the existing post-delete collection helps ordinary tensor lifetime but still does not invoke trace release.

The log does not list individual trace sizes/IDs or a trace-pool dump. Attribution of all existing TRACE bytes to those 16 captures follows from the inspected ownership/control flow, rather than from a measured per-ID inventory. This limits the byte-by-byte proof, not the direct missing-cleanup finding.

## Completion and failure boundaries

Replay is asynchronous: `tt/generator.py:459-460` uses `blocking=False`. Logit/token reads provide host-visible results (`:278-284`), but the teacher-forcing loop can enqueue its final token write after the last read (`:918-924`). `close()` itself has no synchronization. A stage adapter should therefore synchronize the mesh before closing the candidate; successful CPU scoring or saving is not the general completion boundary.

The observed exception is raised during trace storage creation, not a later replay: C++ `end_mesh_trace()` calls `record_end()` before populating the buffer (`tt_metal/distributed/mesh_device.cpp:1772,1785-1790`). Successful trace uploads use blocking writes (`tt_metal/distributed/mesh_trace.cpp:164-165`). This explains why the visible failure follows the `CAPTURE decode` print immediately.

There are secondary exception-cleanup defects, **not the cause of the previous four successful candidates retaining traces**:

- `tt/generator.py:455-456` records an ID only after `end_trace_capture()` succeeds. On this allocation exception the ID is missing from `self.traces`, so adding a later `close()` cannot discover that failed capture. In the graph-exception branch (`:450-452`), an exception from `end_trace_capture()` similarly skips release and can mask the original exception.
- Current C++ increments `trace_buffers_size` before allocation (`tt_metal/distributed/mesh_trace.cpp:55-56,95-96`). If allocation fails with no assigned mesh buffer, the destructor's guarded decrement (`:170-174`) cannot undo that increment. C++ also registers the trace as active on exceptional exit (`mesh_device.cpp:1755-1757`). Releasing an orphan ID alone is therefore not a demonstrated recovery of all accounting.
- Generator construction failures are outside the scoring `finally` (`R/run_precision_reasoning.py:653-659`). Qwen construction allocates tensors but captures no traces; this is a separate partial-construction robustness gap.

For the minimum repair, close the failed process/mesh and resume fresh. Exception-safe capture registration/release and allocator accounting rollback can be addressed separately; neither requires changing completed numerical measurements to resume this sweep.

## Minimum stage-local repair

Add a new `doc/datatype_sweep/p1_runner.py` entry point with the following behavior (suggested code; **not installed by this investigation**):

```python
import ttnn
from readiness_check import run_precision_reasoning as p1


def main(argv=None):
    original_release = p1.release

    def release_candidate(generator):
        try:
            ttnn.synchronize_device(generator.mesh)
            generator.close()
        finally:
            original_release(generator)

    p1.release = release_candidate
    try:
        return p1.main(argv)
    finally:
        p1.release = original_release


if __name__ == "__main__":
    raise SystemExit(main())
```

This is intentionally Qwen-stage-specific: every returned object is a Qwen generator with `mesh` and `close()`, and there is no teardown hook to double-close it. The original release retains collection behavior; the runner retains its subsequent `del gen` and collection. Synchronization/close failures propagate, preventing silent continuation. The outer runtime `main()` still closes its mesh in `finally`.

In `run_p1.py:51-54`, replace the subprocess's `-m readiness_check.run_precision_reasoning` entry point with the absolute path to this adapter, keeping `-u` and all probe arguments. Calling the unmodified installed `main()` from the adapter preserves tokenizer setup, common build kwargs, argument defaults, mesh/fabric lifecycle, scoring, cache checks, and verdict generation. The adapter must run in the child process that executes P1; patching the launching parent alone would not affect that subprocess.

Record the adapter path/content hash alongside the new run receipt, since existing model-source identity deliberately does not hash stage driver files (`tests/full_model_evidence.py:10-26`). No changes to model `tt/` source or installed runtime are necessary. Adding `teardown = close` in `tt/generator.py` would also bridge the hook, but would change the model code hash and invalidate all four score caches plus exact-source eligibility of completed sweep evidence; that is a larger intervention for this task.

## Preserve measurements and continue

I inspected CPU score files safely with `torch.load(..., map_location="cpu", weights_only=True)` and recomputed the runner's hash functions extracted from its AST, without importing model/device code. All cache admission fields match current source/config/text/reference:

| Cached candidate | code_sha | Answer positions | Decode predictions |
| --- | --- | ---: | ---: |
| bf16_all_hifi4 | ad0bfd54d23e61e4 | 23,517 | 1,024 |
| bfp8_baseline | 97b65b4657d81ed2 | 23,517 | 1,024 |
| gdn_attn8_hifi2_head8_hifi2 | ec3fda5374ad888e | 23,517 | 1,024 |
| gateup4_lofi | 7c66634c00eb4245 | 23,517 | 1,024 |

Each cache has four text rows with answer lengths `[6144, 6144, 6144, 5085]`, decode lengths `[256, 256, 256, 256]`, and text SHA `00aad6a2d8bd6fd5`. There is no `down4_lofi.pt`.

Continuation for the calling session after implementing the adapter:

1. Use a new run name, for example `p1_primary_resume`, preserving primary's candidate map, log, command receipt, verdict archive, text cache, and existing score files. `run_p1.py:35-38` already refuses an existing candidate-map filename and `:77-78` archives each changed verdict.
2. Supply the full original five-candidate list with reference `bf16_all_hifi4` and selected `gdn_attn8_hifi2_head8_hifi2`, keeping `max_seq_len=262144`, the same absolute precision-config paths/contents, and the same scoring parameters. Explicitly retain `--hf-positions 0` for this recovery run; the stage wrapper otherwise defaults to 2048 (`run_p1.py:24`). Keep text SHA unchanged and do not disable the score cache.
3. **Omit `--only`.** Despite its help text, current `run()` restricts iteration to reference plus `--only` (`R/run_precision_reasoning.py:623-628`), and loads caches only inside that loop (`:641-648`). Other cached candidates would be absent from the new combined verdict. With the full list, the four exact cache hits skip construction and only down4 builds/scores.
4. The resulting intended invocation of the repaired stage driver is:

   ```bash
   python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/datatype_sweep/run_p1.py p1_primary_resume \
     --ids bfp8_baseline gdn_attn8_hifi2_head8_hifi2 gateup4_lofi down4_lofi \
     --selected gdn_attn8_hifi2_head8_hifi2 --hf-positions 0
   ```

   This command was **not run** here. It requires the proposed adapter wiring first and the calling session's device workflow.
5. Verify the new archive has measured statistics for all four non-reference candidates and no scoring errors. Rebuild the sweep report from preserved archives plus the new receipt, then continue required combination/selection/final-HF work. The sibling short-reference down4 pass does not substitute for its missing P1 measurement.

The cache key checks text SHA, exact candidate kwargs, reference role/kwargs, all model `tt/**/*.py`, and files named by string kwargs (`R/run_precision_reasoning.py:541-590`). It does **not** include every scoring option, such as top-k/decode settings, so retain primary defaults `topk=64`, `decode_tokens=256`, `decode_texts=4`; unchanged hashes alone do not authorize reinterpreting scores under changed options.

Primary's aggregate `precision_reasoning: pass` and exit 0 (`p1_primary.log:500`; command receipt `:65-66`) mean its selected-candidate critical checks passed. The runtime records other candidate errors separately (`R/run_precision_reasoning.py:701-702,748-760,786-792`). Thus primary is incomplete for down4 even though the selected candidate passed. `check_stage_artifacts.py:79-87` correctly requires a measured boolean result for unresolved faster candidates. Do not convert this runtime error into a measured precision failure.

## Alternatives adjudicated

| Hypothesis / intervention | Evidence and adjudication |
| --- | --- |
| Missing close between candidates | Direct hook mismatch plus independent mesh ownership; matches four complete trace sets then fifth first-capture failure. Leading cause. |
| down4 alone requires more than 100 MB of traces | Its failing decode request needs only about 8 MB; same-source standalone down4 captured all four trace types in the same region. Does not explain the cumulative allocation. Long P1 numerical quality is still unmeasured. |
| Trace fragmentation | Reported largest free block equals total free space per bank. Capacity depletion already explains failure. |
| Full-context KV or weight memory exhaustion | Failing allocator explicitly says TRACE, during trace-buffer creation; all five use full-context allocation. Distinct from ordinary tensor capacity. |
| Python cycle / insufficient gc.collect | Real cycle exists, but collection already runs after deletion and the trace pool is independently mesh-owned. More collection does not call close. |
| Asynchronous pending replay alone | Completion matters for correct release ordering, but there is no release call at all and failure occurs at capture storage allocation. Sync alone does not erase traces. |
| Increase trace region | Can defer exhaustion, leaves candidate-to-candidate retention intact, and changes available ordinary DRAM. Unnecessary for a policy already shown to fit alone. |
| Call deallocate_buffers between candidates | Not used by this runner. Current allocator resets TRACE banks as well as ordinary banks (`allocator.cpp:353-358`), but does not perform mesh trace-pool erasure/active-trace unregister/accounting cleanup. Use the explicit close API. |
| Blame the final bank validator or an old binary | Current trace admission counts logical padded buffer sizes (`mesh_trace.cpp:50-76`), while bank spreading rounds each allocation to full per-bank pages (`allocator.cpp:784-793`). These totals can differ, so a bank-level failure does not by itself prove revision drift. Prior per-trace sizes are not logged. No allocator rewrite is needed to remove the demonstrated retained resources. |

## Validation performed and discriminating follow-up

Performed without devices:

- Read the runner/model contract, sibling stage evaluator, trace API, mesh pool ownership, allocation/release path, asynchronous boundaries, archived logs/receipts, and report/cache identity rules.
- Recomputed all ten primary source hashes and all four cache identities; all match. Read cached CPU row/decode lengths; no down4 score file exists.
- Executed the actual runner `run()` and `release()` bodies extracted via AST with in-memory fake generators/scoring/mesh operations. Original cleanup left all fake close-owned resources live; the proposed synchronized-close hook closed every built candidate. An injected scoring exception still reached close and retained its candidate error. Four cache hits plus one miss built/closed only the missing candidate while retaining all five score entries. These are host-control-flow checks, not evidence of silicon deallocation or numerical equivalence.
- Revisited the headline against the code after drafting. Kept failure-only registration/accounting defects secondary, retained the distinction between logical and bank footprint, and did not claim a completed P1 numerical result for down4.
- An independent xhigh reviewer checked the draft against source/logs, recomputed the cache identities and allocator arithmetic, and reran the mocked lifecycle/cache cases. Review found no substantive corrections. The proposed adapter also passes a syntax check, and this report passes `git diff --check`.

Hardware follow-up for the repair session:

1. **Continuation check:** run the fresh-process recovery above. Require four unchanged score caches, one newly measured down4 cache, text SHA `00aad6a2d8bd6fd5`, all candidate statistics present, and an empty scoring-error map. Preserve original cache digests/mtimes to demonstrate reuse.
2. **Direct lifecycle check:** a cache-reusing resume would bypass the original multi-build stress. Separately run a bounded same-mesh repeated build/prepare/synchronize/close/delete/collect sequence across at least five candidates (or repeated baseline instances), without replacing preserved score files. Delete the generator and collect its cycles before the next build, matching the runner's ordinary tensor cleanup. Record TRACE allocation before build, after preparation, and after synchronization/close. Usage should return to its initial baseline after each close rather than grow across candidates. This distinguishes a true lifecycle repair from success merely caused by a fresh mesh and four cache hits.
3. **Numerical preservation:** require cached paired metrics to remain identical; obtain down4's actual numerical verdict without relaxing thresholds. The repair changes only cleanup after scoring. Any new numerical or trace-allocation failure on a fresh mesh remains actionable evidence and needs separate investigation.

No runtime trace-pool measurement, repaired five-build run, or continued sweep was performed in this inspection-only session. Those remain explicit acceptance work for the calling session, not grounds to stop after documenting the diagnosis.

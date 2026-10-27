# Masked recurrent state commit experiment

Status: independently accepted local optimization, default off. Serving/SWE
promotion is pending; neither original SWE8 image includes it.
`QWEN38_MASKED_RECURRENCE=1` selects direct commit only for canonical BF16 tiled
[B,1,1] masks in non-prefill one-token decode, batches8..32. Other paths retain
the existing selection/copy. No precision, cache, collective or context change.

The graph-fusing audit inspected the native recurrent GDN contract, its dataflow
writers, ternary `where` implementation/tests, and model usages. The existing GDN
interface has no active-mask commit option and replacing the selected FP32
compute with its other precision path would not preserve this contract.

| Existing subgraph | Shape/dtype and movement | Candidate |
|---|---|---|
| Normalize/reformat Q/K/V, FP32 exp/gates | Existing DRAM/FP32 operands, unchanged | Same |
| HiFi4/FP32 recurrence | One/two/three/four heads per core; initial state read before final output | Same compute and reader |
| Write temporary updated state | FP32 [B,12,128,128] per chip | Write active heads directly to their original disjoint state addresses |
| `where(mask,new,old)` | Mask broadcasts over each user's heads; FP32 old/new read, selected state written | Predicate in writer; inactive state is untouched |
| Copy selected state back | Full FP32 tensor transfer | Removed |
| Output normalization/gate/projection | Existing output for every row | Same; inactive outputs are still computed |

At B16 the state payload is12,582,912bytes/chip. The old final write, selection
and copy move6× that logical payload after the common initial-state read;
`where(...,output_tensor=state)` reduces this to4×, while conditional commit
writes at most1×. These are logical payload counts, not measured DRAM traffic.
The writer reads one32-byte mask fragment/core and uses32bytes of extra local
scratch. Each admitted head count divides12, so a core's sequential heads belong
to one user. Inactive heads still drain their output-state CB; no compute or
output arithmetic is skipped. No persistent model buffer is added.

Both alternatives were measured with strict tracking and nonzero random state,
full Qwen per-chip dimensions, five changing masks, four ASICs, five100-replay
windows and exact checks after all timed updates. Primitive results:

| B | Direct-commit speedup |
|---|---:|
|8|1.500×|
|16|1.783×|
|19|1.700×|
|32|1.946×|

All outputs/states match bitwise, so PCC is1 for these nonconstant fixtures;
all inactive nonzero state rows remain bitwise unchanged. The simpler existing
`where` output-buffer alternative also passes but gives1.192× at B16,
versus about1.79× for direct commit. The faster candidate is being carried
forward; primitive ratios are not model/SWE speedups.

A first compile failed due to a missing `api/core_local_mem.h` include, before
execution. The include was added; original source/error evidence is retained.
The clean optional integration shares the existing grouped factory/reader/compute
rather than keeping a second copy. A real-weight layer0 varying-mask B16 test
passes72 exact checks, including inactive convolution/FP32 state preservation;
masked layer replay1261.079→1093.519µs (1.153×). Final-source B8/B19/B32 real-layer checks also pass72 checks each, with
1.088×/1.153×/1.214× replay gains. Layer timing follows the final all-active
mask in the semantic schedule; those ratios are for the masked variant with
all rows active. Primitive timing uses partial masks, and the full-model pair
below uses8 actual requests in16 resident rows. Separate B32 watcher10/no-inline passes
with all watcher features enabled. The full64 strict pair now passes at B16 with8 active requests, physical
prefill chunk512, logical prompt4096/output128, tracking1/cache-skip0. All
fixed tokens, five maps of384 state hashes, continuation logits, three shared
qualitative prompts (all8 outputs complete within512 tokens), and ragged
continuations match exactly. Both policies retain4 traces/4 captures and no
inner-loop host refreshes/readbacks. Strict-run diagnostic decode9.150958→
8.082468s (1.1322×); TTFT8773.174→8773.615ms. These single-repeat numbers
are correctness-run diagnostics. A separate clean tracking0/cache-skip0 pair
with three timing repeats gives median13.87818→15.71310 tokens/s/user
(1.132216×), exact timed/ragged tokens, and TTFT8770.078→8771.916ms.
The clean runs skip the already accepted shared qualitative suite and state
readbacks; the strict pair above supplies those correctness gates.
A matched repaired-clock profile at the same8/16,512physical fixture
now completes: two real layers0/3 plus terminal/sampler/feedback decode
3723.083→3540.077µs (1.051695×),896→888 device ops, exactly two removed
ops per ASIC. Prefill283000.725→283002.281µs,7404 ops both arms; no prefill
gain is claimed. All sources/fixture/precision match except the masked flag.
See [compact profile comparison](masked-fixed-clock-profile-comparison.json).
The previously documented boundary-gap and fused-kernel fidelity reporting
interpretations still apply; raw timestamps/durations and parser assertions
are preserved. Original NFS loading stalled before measurements and resumed
without restarting the process; separately verified local weights prevent
that shared-storage dependence in subsequent tests. The [final independent review](MASKED_FINAL_REVIEW.md) passes the bounded
local stage. Serving/SWE promotion remains separate.

Raw code snapshots, checks and timing windows stay under
`/home/mvasiljev/build/qwen38-swe/masked-recurrence/` and local logs; only compact
code/documentation will be published. The existing reviewed grouped-only
optimization and CI image remain separately pinned.

As a scope check,183.006µs saved in the one linear layer of the reduced
profile predicts about8.784ms across48 linear layers. The clean full-model
pair saves8.414ms per decode step (127 measured steps per request), within
about4.2% of that simple estimate. This is an extrapolation check across
profiling/non-profiling builds; the directly measured full-model result is
the13.2% gain above.

Final review found and corrected a generic-op declaration issue: outputs must
be last in its IO tensor list, so the read-only mask now precedes out/final.
The old declared return was the BF16 mask, even though kernels wrote the
correct FP32 buffers and the Python caller ignored that return. All descriptors,
runtime addresses, kernels, math and helper return values are unchanged. The
[exact source/ordering proof](masked-io-order-proof.json) and a final strict B16
real-weight varying-mask run pass72 exact checks, including24 preserved
nonzero inactive-state checks. Existing full-model/profile receipts retain
their original pre-correction source hashes; their raw output metadata is not
rewritten. Final helper SHA is
`f0f9f46ce31ffdf0a6f79de724bf16ba112c1fffd4ed7e2484f886b75d904645`.

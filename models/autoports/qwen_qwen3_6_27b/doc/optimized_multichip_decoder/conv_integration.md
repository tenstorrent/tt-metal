# Local fused convolution: proposed production integration

The accompanying [unified patch](conv_integration.patch) is **unapplied**.
It adds `local_fused_conv=False` to `MultichipDecoder`'s policy and a normal
class-method branch in `_local_batched_conv`. Setting the policy true uses the
validated candidate for compact `[1,B,2560]` input, B1–32. Legacy layouts and
unsupported shapes keep the existing implementation. The policy does not alter
the choice of convolution route in `_gdn`.

The only second-file change is the focused probe's synthetic owner: it now
provides `policy={"local_fused_conv": False}`. This keeps its explicit reference
call unfused after production integration. No runtime hook, closure, model
monkeypatch or new persistent buffer is introduced. The frozen optimized
decoder and existing recurrence arithmetic are untouched.

Source inspected: `tt/multichip_decoder.py`, `tt/flat_prefill.py`,
`tt/recurrent_next_candidate.py`, `tests/run_multichip_decoder.py`,
`tests/probe_multichip_conv_candidate.py`, and
`tests/check_multichip_stack_resources.py`. Patch construction used main-model
SHA256 `618acffb99546ca0dffbfd24d63e13d1f8e60836a2dc40f524eb5103fd03f8e7`.
Run `git apply --check` again if the parent changes the model before applying.

## B1 and prefill behavior

| Policy / input | Convolution and recurrence route |
|---|---|
| B>1 compact decode, `local_fused_conv=true` | Fused local convolution; existing raw-packed recurrence policy |
| B1, `batched_conv_b1=false` | Existing native convolution route; `local_fused_conv` is not reached; baseline recurrence |
| B1, `batched_conv_b1=true`, `local_fused_conv=false` | Unfused local convolution; requested raw-packed recurrence variant |
| B1, both flags true | Fused local convolution; same requested raw-packed recurrence variant |
| Legacy `[B,1,2560]` with B>1 | Existing local convolution implementation |
| Prefill with sequence length >1 | Existing chunk convolution/recurrence path |

Keep `batched_conv_b1` explicit. Enabling `local_fused_conv` must not implicitly
change B1 recurrence from the native route to `cumulative_exp_scalar`. Measure
the B1 same-flag control and the original-policy control separately.

`flat_prefill._can_flatten` excludes one-token tails. `_legacy_chunk` passes
`[B,1,H]` through `_gdn`, so B>1 tails reach `_local_batched_conv` with legacy
`[B,1,2560]`; the new guard retains their implementation. B1 single-token
prefill is physically `[1,1,2560]`, indistinguishable from compact B1 decode;
when `batched_conv_b1=true` it uses the fused helper. This is intentional and
matches the earlier candidate instance hook. Validate a B1 single-token tail
when selecting the B1 family. The helper's output remains `[B,1,2560]`, and
the recurrence consumes it directly.

## Ownership and trace lifetime

`check_multichip_stack_resources.py` creates 64 owners with `copy.copy(original)`
and explicitly replaces `_ccl_buffers`, CCL semaphores, and convolution exchange
storage. An instance closure capturing the original model would survive a
shallow copy and execute against the wrong owner. A class-defined method binds
to the current owner when accessed, so the proposed branch reads that owner's
policy/taps and receives that invocation's history. Read-only shared taps are
consistent with the resource test's shared weights.

The minimal patch calls `local_conv(..., output=None)`. Its output is temporary
within `_gdn` and is consumed by the recurrence before leaving that invocation.
It adds no mutable model attribute and requires no new resource-clone reset.
Native untilize and output allocations remain part of the captured graph;
their replay addresses have the same lifetime discipline as other decoder
temporaries. Prepare fixtures and eager references before capture, as the
existing full-layer/resource runners already do. A second live trace must not
retain newly allocated outputs that overlap a first trace's temporaries.

The focused two-trace probe already avoids that separate hazard by allocating
its candidate output before either capture and passing `output=...`; production
does not need to copy that scratch into a model attribute. If persistent output
is later justified, allocate it before capture for each independent owner and
shape, retain it through trace release, and recreate the scratch dictionary on
every shallow clone. Record/assert unique addresses across the 48 GDN owners.
Never shallow-share a prototype `_conv_outputs` dictionary or a captured output
between owners. Do not add lazy first-use persistent allocation inside a live
trace's lifetime. This optional extension is deliberately absent from the patch.

## Applying and checking the proposal

```bash
git apply --check models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/conv_integration.patch
git apply models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/conv_integration.patch
```

Parent should run the normal `run_multichip_decoder` entrypoint with
`"local_fused_conv":true` added to the selected policy, without the candidate
instance-hook CLI. For B1 add `"batched_conv_b1":true` only for the explicitly
selected B1 family. Re-run the mandatory focused exact-output/history smoke
after applying: its baseline fixture patch is necessary. Preserve the normal
full-layer PCC/state/changed-input-trace gates and run the resource test against
the selected policy to verify the direct class method across copied owners.

CPU checks: proposed files compile; `git apply --check` passes; routing and
shallow-copy method binding were checked without importing TTNN. No main file
was edited, no patch was applied, and no device operation was run by this worker.

## Stream-state low-core audit

The separate streaming probe inherits `_local12_head_partition` unchanged.
`--low-cores 1` appends B3/one-core to the regular cases, once per dtype/mode;
it is not a global one-core override. `--cores 1` applies to every requested
batch. The one-core B3 case owns all 36 heads, crosses both user boundaries,
and retains grouped Q/K only within each global triple. Default B1 disables
grouping even if `--grouped-qk on`; this is the existing partition contract.

| One-core case | Heads | Q/K preparations, grouping on | Four-page state groups | Full CB11 wraps |
|---|---:|---:|---:|---:|
| B1 | 12 | 12 | 48 | 6 |
| B3 | 36 | 12 | 144 | 18 |
| B32 | 384 | 128 | 1536 | 192 |

With grouping off the preparation count equals the number of heads. Gate
cache refreshes occur at global user boundaries (h0/12/24 for B3), plus each
core's first head. The source uses the resolved partition's grouping for both
matched cumulative readers and the compute compile argument. Candidate writer
binding is fixed per cloned function; no per-call global mutation occurs.

The streaming writer waits for four CB11 tiles, writes/barriers/pops them, and
only waits for CB12 after all four state-column groups. Compute publishes each
state group and publishes CB12 after the fourth; therefore writer's output
wait does not block the state-group consumption needed for compute progress.
CB11 has 32 pages, CB12 eight, and CB5 32; their group sizes divide capacity.
Old state is already in CB5 before any updated-column publication. Multi-head
reader prefetch addresses distinct heads; earlier state writes do not race a
future head's old-state read. Output tail-padding writes remain last-head-only.

A one-core case is expected to take substantially longer than the default
parallel case; it is a wrap/ownership control, not a competitive configuration.
Four state-write barriers plus the final output barrier per head may also make
streaming slower. A stall is not explained away by low core count: after normal
warmup it should complete finite, exact outputs/state and changed-input traces
with `TT_METAL_TRACE_ALLOC_TRACKING=1`. CPU checks verify the actual partition
counts above; they do not prove absence of a device protocol bug.

# P2 AutoFix repair

Resumed by owner 2026-11-16; prior cap superseded by explicit resume.
Repair investigation began about 23:12 UTC. P2 top_k40/seed42 at32users routes
to host with ratio4.275190600 >1.15. Preserve requested profile and context.

Source diagnosis: sampling compute fixes K32, bitonic merge retains only32,
writer scans only two faces in one tile; simply raising declared cap is unsafe.
Independent CPU audit confirms local_sort supports64 across two tiles.
Proposed repair: explicit TTNN max_top_k64 restricted to W64; skip lossy merge,
use both tiles in softmax and tile-aware writer indexing. Existing max32 remains
default and unchanged. Canonical TTSampling reduces gathered candidates to64
and gathers matching globalIDs before the same sampler/seed/token-out call.

Also found canonical Qwen configure passed public T directly to low-level 1/T
parameter. Correct only the device upload, preserving logical stored T for
request remap/reload. Validate reciprocal temperature and boundaries.

No device owners or leftover servers on resume; tt-smi list shows all4chips.
No reset used. AutoDebug child sandbox unsupported by host namespace policy;
explicit retry without child sandbox under existing unrestricted authorization.

Validation pending: focused candidate membership/distribution and traced seed
replay, canonical generator/adapter reduced checks, affected final serving gates.

Focused validation completed: C++/binding build passes (`build_retry.log`,
`build_formatted.log`), host_probe_retry.log passes five cap/T/reload checks.
Legacy default32 k40 reproduced missing all ranks33..40 (`legacy_width_retry.json`).
Wide op probe passes96 k/p/reciprocal-T combinations, existing32 controls,
seed/remap/trace and persistent-output checks, allocation tracker positive
control (`width64.json/.log`). Initial TT_VISIBLE_DEVICES=0 probe could not
construct custom fabric descriptor; removed visibility filter on exclusively
owned idle host and opened only device0. No reset/hang occurred.
Canonical host-companion route test now requests k128, above new cap64.

## TP4 adapter result and independent diagnosis

`adapter_width64.json` passes on reduced real layers 0/3, four serving rows,
external8196-page cache, strict allocation tracking including program cache.
The public T0.5/1/2/0.7 parameters become2/1/0.5/1.4296875 on device.
128 draws reach every pooled rank33–40, stay inside each k33/40/63/64 set,
and reproduce exactly after reset_rng. Active seeds advance128; inactive seeds
remain unchanged. Greedy and counted penalty sibling checks pass. All five
trace IDs and the persistent token-out addresses remain unchanged, with no
full-logits readback. Log: `adapter_width64.log`.

The independent source-only AutoDebug report is `AUTODEBUG.md`. Its baseline
analysis agrees with the retained32/one-tile writer diagnosis, the scoped64
repair and the reciprocal-temperature correction. Its proposed additional
stress cases are suggestions, not claimed executed evidence. Independent final
source review found no concrete defect in default32 preservation, two-tile
buffer/index handling, seed ordering, global compaction or tensor lifetimes.
No plugin/upstream-vLLM changes are part of this repair.

Full-model tracked server launched in `readiness_vllm/p2_final_async`; acceptance
and the original P2 threshold remain pending until their actual results exist.
Historical overwritten canonical evidence is copied into `prior_artifacts/`.
The earlier max-context capacity acceptance is retained once: context, cache,
layout, positions and model weights are unchanged; current exact greedy controls
and mixed-shape lifecycle checks validate the changed sampler in serving.

## P2 verified fixed

All four default profiles pass unchanged users1/32,128 output tokens and1.15
load ratio. `readiness_vllm/eval_probes/p2_repair/harness_requests.json` and
`doc/eval_readiness/harness_requests/runs.json` contain current results.
Reasoning top-k40/T1/p.95/seed42 at32users: {'ratio': 0.9502695731812192, 'profile_tpot_ms': 78.23860912957764, 'plain_tpot_ms': 82.33306772904251, 'device_sampling_lines': 0, 'host_sampling_lines': 0}.
All profile runs report zero host-sampling lines. Removed the resolved P2 item
from active limitations; its original record remains in prior_artifacts.
No waiver or threshold/profile change was used.

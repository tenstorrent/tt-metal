# Focused runtime and cache ownership audit

CPU-only independent implementation inspection; no stage-review verdict and no device commands. Implementation files and the running queue were not modified.

## Result

No concrete host fallback or stale page ownership bug was found in the current integrated cache-fill path. The unmodified runtime auditor passes with 111 reachable functions, 14 source modules and zero violations (`runtime_ownership_audit_current.json`). Its imported-function traversal reaches the new `_prefill_cache_fill`, `_attention_page_table`, `_attention_state_view`, `page_table_for_attention`, `page_table_certified` and `batched_fill_pair` graph. The runtime branch does not reach `create_paged_state`, `replace_paged_table_from_host`, host certification, or table upload.

The six CPU lease checks were rerun against **production** `tt/page_table_ownership.py`, using the existing aliasing stand-in: shallow/deep copies share revocation; replacing one state revokes old aliases and creates a fresh private lease; negative skip entries do not imply ownership; duplicate physical pages fail certification; external plain states remain ordered. See `runtime_ownership_production_cpu.json`. This is host semantics evidence, not a device result.

## Findings and proposed isolated changes

1. The existing `check_page_table_ownership_cpu.py` executes `doc/page_table_ownership_reviewed.py`, not the integrated production module. Thus its historical pass alone does not establish current production semantics. The isolated patch switches its positive check to production source while preserving the original-patch negative control. The independent production check above already passes.
2. The auditor automatically reaches the new helpers today, but `REQUIRED_HELPERS` and negative controls cover only the earlier packed-correction/local-convolution graph. The isolated patch adds the cache graph, SwiGLU, local affine gate, flat packed GDN and tiled convolution entrypoints to the required set and adds the host table replacement to setup-only entrypoints. The resulting audit passes, and all 22 forbidden-call/binding negative controls pass (`runtime_ownership_audit_proposed.json`).
3. The audit's runtime sentinel prose is stale: the runner now guards six public transfer APIs during initial eager prefill, initial eager decode and decode capture. Timed prefill remains outside that context; it independently checks exact output and full state against guarded initial execution. The isolated patch corrects the scope text rather than claiming all eager executions are guarded.
4. The AST auditor does not model property getter/setter dispatch: indexing a setter can overwrite a same-named getter. The patch explicitly records this limitation. Manual inspection of this new ownership graph finds no uninspected property read on the certified runtime path: it reads ordinary `_table_lease.table` and `_table_lease.certified` fields. The public `page_table` property only revokes certification and returns the device handle; metadata properties read shape/address. Host work is confined to the explicit setup method. This limitation must not be described as general proof of arbitrary user-defined state subclasses.
5. `attach_runtime_results` matches only the main decoder source hash. Final evidence still needs the existing wrapper's complete helper-source provenance checked against the frozen final source tree; a main-file hash alone cannot establish freshness after helper-only edits. No stale final evidence is alleged because final runs have not occurred.

Patch: `runtime_ownership_audit_followup.patch`. It edits only the CPU auditor and CPU lease checker; no model/runner implementation is included. It has been evaluated from an in-memory source copy with all negative controls. Parent can apply it without modifying active device code.

## Ownership and capture reasoning

- Setup uploads a fresh device page table. Only globally disjoint nonnegative physical pages certify; this is conservative even for duplicates within one row.
- Every public raw-table exposure revokes the shared lease permanently. Setter assignment revokes the old lease before installing an uncertified new lease. Deepcopy explicitly retains lease identity, so Python wrappers cannot retain an independent stale certificate for the same device storage.
- Internal raw views are confined to private attention methods and the inherited ordered path; those methods do not publish or retain a new public alias.
- Eager bulk fill requires batch greater than one, policy enabled, valid lease, setup-created indices and inactive trace capture. The captured path always chooses ordered calls, so later public exposure/mutation cannot invalidate a captured parallel-write branch.
- `replace_paged_table_from_host` is explicitly setup-only. Its assertion prevents replacement during capture, and its API documentation requires releasing/recapturing existing traces before changing addresses. It does not purport to permit table replacement while a live trace still references the old table. Call sites in the runner replace before decode capture.
- Trace tests intentionally use private access to arrange controlled device mutations; production callers must obey the underscore-private boundary. Arbitrary mutation through private lease fields cannot be made safe by this public API.

## Commands

```text
python3 models/autoports/qwen_qwen3_6_27b/tests/audit_optimized_multichip_runtime.py --negative-controls --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/runtime_ownership_audit_current.json
```

Additional CPU scripts loaded the audit module via source compilation (no model imports), executed the proposed diff in memory, ran its negative controls, and called the existing lease-check function on production ownership source. No TTNN import, mesh open, reset, test device process, or implementation edit was performed.

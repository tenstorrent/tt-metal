# PR #53130 — drafted replies

> **Status: drafted, NOT posted.** Rewritten **2026-08-20** against the live thread list: of the
> 36 review comments on the PR, 15 already have a reply and the **21 below do not**. Supersedes
> the 2026-08-18 draft, which covered 11 (all of them still here, re-checked). One thread the
> old draft answered — `r3783240232`, a reviewdog import-order nit — is gone from the API
> listing entirely, so it needs nothing.
>
> Still not posted for the same reason as last time: no GitHub write access from the session
> that wrote them (`gh` is not installed on this box and outbound `curl` is blocked, so reads
> work and writes do not). Everything below is copy-paste ready.
>
> Check each thread is still open before pasting, and note the last group is deliberate
> won't-fix with a rationale rather than a change.

Base URL for each: `https://github.com/tenstorrent/tt-metal/pull/53130#discussion_r<id>`

SHAs are on `ldjurovic/llk-tests-blaze-promotions`.

---

## A. Fixed in this session (5 new commits)

### 3813108243 — `llk_unpack_A_rmsnorm.h:112`, should `UNP_SEL` be `UNP_A`?
> Fixed in 99dc2eea1c5, as `UNP_AB`.
>
> You are right about the mechanism. `x-start/x-end` is per-unpacker state, and this MOP
> issues its real UNPACRs against SrcA (unpacker 0) — one per face — while SrcB gets only a
> ZEROSRC dummy dvalid, because SrcB is filled by MOVD2B on the math thread and reads no L1.
> The selector was derived the way generic `llk_unpack_A.h` derives it,
> `(BType == NONE) ? UNP_A : UNP_B`, which is correct there (a SCALAR broadcast means the
> broadcast operand *is* SrcB) and wrong here (SCALAR only routes the base address to upk0).
> So it programmed the datum count on the one unpacker that does not use it, and left
> unpacker 0 at whatever the preceding op left.
>
> Programmed on both rather than on UNP_A alone: A is the one that needs it, keeping B costs
> one instruction and makes the change unable to regress the previous state — which matters
> because it cannot be measured here. Both of today's callers (the api wrapper and
> `rmsnorm_bcast_scalar_dest_reuse_test.cpp`) pass `FACE_R_DIM` and run after an unpack that
> already left unpacker 0 there, so old and new selectors agree and the existing sweep cannot
> discriminate them. Discriminating it needs a driver that unpacks at a narrow `face_r_dim`
> after a wide one; noted as follow-up. What the change removes is the dependence on that
> agreement continuing to hold.

### 3813108603 — `llk_unpack_A_top32_rm.h:25`, unused params
> Dropped in bf2e02a37dd. Confirmed on both counts: `unpack_src_format` is never read in the
> init (the y-stride comes from `unpack_dst_format & 0x3`), and the `unpack_to_dest` template
> parameter is unused there too — only the execute half reads either, and both stay there.
> The only caller is `llk_unpack_A_top32_rm_api.h`; with the template parameter gone its
> if/else collapsed, since the two arms differed solely in `within_face_16x16_transpose`, so
> it is now one call with the flag named at the call site.

### 3813108970 — `test_custom_mm_uninit_parity.py:112`, `REMAINING_WORK.md` does not exist
> Fixed in 0c19b7e0e3e, all four sites. The file went away with the strategy docs (your
> earlier comment on those) and these pointers outlived it. Agreed that :112 was the worst of
> the four, since it is the message a maintainer reads exactly when the guard fires. Each site
> now states the thing the item said instead of naming where it was written down — for the two
> parity ones, that the real test has to be metal-side, able to include
> `tt_metal/hw/inc/api/compute` and call both entry points, which a tt-llk test cannot. No
> issue link because the follow-up is not filed; the constraint now travels with the code.

### 3813109380 — `custom_mm_uninit_restore_test.cpp:38`, "must be 4" unqualified
> Scoped in fe7bfe01f13. You are right that it is false of the file as a whole —
> `test_custom_mm_uninit_restore.py:260` builds this source at `block_mop_num_faces=2` and
> asserts `not passed_test(...)`. The header now says what each value is for: 4 for the
> W-stride tests (at a mismatched geometry run 1 is wrong whatever the stride is, so the
> restore is unobservable), 2 for `test_custom_mm_uninit_leaves_the_caller_mop_installed`, and
> that deleting the 2-face case deletes the only coverage that would catch a MOP restore being
> re-added.

### 3813109712 — `llk_unpack_A_rmsnorm.h:26`, no doxygen
> Added in 99dc2eea1c5, on both functions, in the style of `llk_unpack_A.h`'s own blocks. It
> names which template arguments are actually reachable rather than presenting them as free
> choices — the static_asserts pin the pair to SCALAR + acc_to_dest + DEST_TO_SRCB — and
> records the missing-format-parameter note so the next reader does not go looking for the
> arguments this branch removed.

---

## B. Already fixed on the branch before this session — reply and resolve

### 3783809559 — `sort_headers_coexist_test.cpp:125`, the runtime half has no signal
> Agreed and taken. The datacopy does open with `math::set_dst_write_addr<Tile32x32, SrcRegs>`,
> so the offset the helper leaves is discarded and the copy lands identically whether the
> helper is right, wrong, or deleted. The offset sweep is gone (one fixed constant remains, so
> the helper is code-generated rather than only parsed), the runtime assertion message no
> longer claims to cover the helper's value, and both the source header and the module
> docstring now say the value is in the build succeeding. The helper stays covered in its real
> context by the topk_xl and deepseek_top32_rm kernels.

### 3783809910 — `test_custom_mm_uninit_restore.py`, the `family` axis never reaches the build
> Dropped. Both halves of the finding were right: it produced the same `variant_id` so the ELF
> was reused, and the rationale was inverted — the driver replicates the shared uninit body
> rather than calling either entry point, so a divergence between the two headers is precisely
> what the axis could not catch. The comment where the axis used to be now says that, and
> `test_custom_mm_uninit_parity.py` guards the divergence textually until there is a
> metal-side test that calls the real functions.

### 3783810855 — `test_sfpu_add_rsqrt.py`, atol gates the `(No, Float32)` cell
> Fixed: `_ATOL` is 0, so the per-cell rtol is what gates every cell. Your arithmetic was the
> point — golden in ~[0.45, 3.2] puts `rtol*|r|` around 3.2e-6, three orders under the old
> 1e-3 floor, and `torch.isclose` adds the two, so the floor competed with the relative term
> instead of backing it up. The "50000x tighter" claim went with it.

### 3783812081 — `BLAZE_PROMOTION_TEST_STRATEGY.md`, sprint tracker not documentation
> Agreed; both docs are off the branch. The genre argument is the right one — per-item effort
> estimates, an open-questions section, investigation diaries and cross-references to five
> in-flight PRs do not belong next to `DPRINT.md` / `LOGGING.md` / `TTSIM.md`. The findings
> worth keeping travel with the code that they constrain instead: the
> `_top32_rm_init_` / `_topk_xl_init_` same-kernel hang is the scope note in
> `sort_headers_coexist_test.cpp`, the `vConstFloatPrgm0` hazard is the driver comment and the
> hazard test in `test_sfpu_sampling.py`, and the `dense_packing` W-stride defect is the
> thread above plus the xfailed measurement in `test_custom_mm_uninit_restore.py`.

### 3783813409 — `llk_math_deepseek_top32_rm.h`, seven wrappers with no caller
> Dropped — the file is no longer in this branch's diff. Your read was right: the in-tree
> consumers drive `ckernel::sfpu::_bitonic_top32_*` / `_top32_rm_init_` directly via
> `SFPU_UNARY_CALL` and bypass the wrapper entirely, so it would have shipped seven public
> entry points with no caller and no test. It can land with the top32_rm test that needs it.

### 3783813739 — `compressed_custom_mm.h:72`, doxygen still shows `split_acc`/`finalize` as live
> Added. Each affected table row now carries the caveat inline — NOT FORWARDED on this family,
> accepted for call-site compatibility, LLK always instantiated with the flag false, sibling
> `custom_mm.h` does forward it — pointing at the NOTEs in the bodies. See the thread on
> Copilot's `split_acc` comment for why the behaviour itself is not changed here.

### 3796119145 (Copilot) — `llk_unpack_AB_compressed_custom_mm.h:250`, remainder word read
> Fixed in 54e218ebbce: the `meta_ptr[full_iters]` load is now inside `if (rem_iters != 0)`.
> Correct as reported — with `kt_dim * ct_dim` a multiple of 10 the buffer holds exactly
> `full_iters` words and the load ran anyway, e.g. `kt_dim=10, ct_dim=1`. The math thread had
> the same defect in three places; guarded in 506984e522a with
> `num_meta_words = (kt_dim * ct_dim + 9) / 10`, the expression the caller sizes the buffer
> with. Scope caveat both times: the elided word was never *used*, so this is memory safety
> rather than a wrong answer, and no golden can observe it.

### 3796908960 — `test_custom_mm_uninit_restore.py:210`, imperative `pytest.xfail()`
> Fixed, exactly as suggested: the test takes `request` and calls
> `request.node.add_marker(pytest.mark.xfail(reason=_DENSE_FP32_XFAIL, strict=False))`, so the
> body runs and the cell can report XPASS once the W-stride constants become format-aware.
> Your reading was right that the imperative form was `pytest.skip` with a different label,
> and the note about `pytest.param(..., marks=...)` not working here saved a wrong fix — the
> local `parametrize()` helper builds raw tuples and calls `.name`/`str()` on each value, so it
> does not unwrap a `ParameterSet`.

### 3796909307 — `test_sfpu_add_rsqrt.py:268`, the negative-sign assertion is fragile
> Taken. The `negative_lanes > 0` assertion on the FAST_APPROX=true side is gone; the test now
> asserts (1) FAST_APPROX=false leaves no negative lane, which follows from the guard itself,
> and (2) the two builds differ somewhere, which is what makes the flag observable at all —
> your suggested framing. The docstring says why: the sign falls out of `vConstIntPrgm0 - i`
> going negative for these exponents, i.e. from the current LUT seed and not from an
> invariant, so an accuracy-motivated retune could break the old assertion with the guard
> still working. The "sign guarantee" wording at the old :241 went with it.

### 3796909504 — `llk_unpack_A_rmsnorm.h:30`, `[[maybe_unused]]` params are dead all the way down
> Deleted end-to-end rather than suppressed — both parameters are gone from
> `_llk_unpack_A_rmsnorm_mop_config_`, `_llk_unpack_A_rmsnorm_init_` and the api wrapper. The
> one thing that referenced `unpack_dst_format`, the commented-out UInt16 src-zero-flag TODO,
> now carries a note that reviving it means plumbing the format back in. The same cleanup has
> since been applied to `llk_unpack_A_top32_rm.h` (bf2e02a37dd) per your later comment.

### 3796909555 — `sort_headers_coexist_test.cpp:120`, bare `0`
> Annotated: `0 /*addr*/`, matching the call above it.

### 3812155565 — reviewdog, trailing whitespace in `test_matmul_custom_mm.py:45`
> Fixed; `pre-commit run --files <branch files>` is clean on every hook.

---

## C. Not changed in code, deliberately — reply with the rationale and resolve

### 3796119098 (Copilot) — `custom_mm_uninit_restore_test.cpp`, exercise the real uninit API
> The finding is accurate and the constraint is structural: a tt-llk test cannot include
> `tt_metal/hw/inc/api/compute`, so this driver replicates the uninit body rather than calling
> `custom_mm_block_uninit` / `compressed_custom_mm_block_uninit`. Moving the scenario to the
> metal kernel tests is the right end state and is not something this PR does; it is a
> different test tree with a different harness.
>
> What this branch adds instead of pretending otherwise: the limitation is stated in both the
> driver header and the module docstring, and `test_custom_mm_uninit_parity.py` closes the two
> failure modes textually — a divergence between the two bodies, and the driver's replicated
> W-stride expressions drifting from the headers'. A text match cannot tell you the functions
> work, only that they still say the same thing, which is why it is called an interim guard.

### 3812324720 (Copilot) — `compressed_custom_mm.h:39`, do not silently discard `split_acc`
> The claim checks out: `matmul_expert_compressed_dram.hpp:645` `<false, true, false>` and
> `matmul_expert_compressed_sram.hpp:188` `<false, true, true>` both request `split_acc=true`
> and silently get false, and the LLK does implement the flag (it selects the DEST addr_mod).
>
> Not changed here, deliberately. Forwarding it changes what two shipping DeepSeek kernels
> compute; rejecting `true` at compile time breaks their build and requires migrating them.
> #52727 deferred this as a semantic change belonging in its own PR, and that call is the
> header owner's, not a test PR's. What this branch does is make the hazard concrete: the body
> NOTE now names both callers instead of saying "all existing consumers were validated against
> this behavior", which read as reassurance where there is a mismatch, and the doxygen rows
> carry the caveat.

### 3796909124 — `custom_mm.h:267`, pair the fused caller with `pack_block_contiguous_uninit`
> Your analysis is right and the flag it is about did not survive: `restore_tile_pack_mop` is
> not in main, and `custom_mm.h` on this branch is byte-identical to main —
> `git diff origin/main HEAD -- tt_metal/hw/inc/api/compute/experimental/custom_mm.h` is empty.
> It appears in the diff only because the merge-base predates #52727's merge, so there is
> nothing here to change. Recording the substance for whoever picks the contract up: the body
> was byte-identical to `pack_block_contiguous_uninit()`, the restore was weaker than it read
> (hardcoded 32x32/4-face vs `llk_pack_init` deriving geometry from `out_cb_id`, nothing to
> restore at all after `custom_mm_block_init_short`), and it left the
> `set_packer_strides`/`SETADCXX` that `_llk_pack_init_` sets untouched.
> `test_custom_mm_uninit_restore.py` has been narrowed to what actually merged and now pins the
> *absence* of a MOP restore, which is what a change re-adding one would break.

### 3796119166 (Copilot) — PR metadata still the template
> Fair. Note the packaging half no longer applies: this branch makes no change to
> `hw/sources.cmake` any more (`git diff origin/main...HEAD -- tt_metal/hw/sources.cmake` is
> empty) — those two header entries reached main with #52727, and they showed here only
> because the merge-base predates it. What is left is tt-llk tests plus the LLK-side
> promotions and cleanups. The title and body still want updating to say that.
> **Needs the PR description and title edited by hand — no `gh` on the box this was worked
> from.**

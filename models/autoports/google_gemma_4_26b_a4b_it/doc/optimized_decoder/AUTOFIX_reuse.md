# AutoFix: full-attention request-reuse accuracy

Status: the exact failure has a verified router-boundary repair. The parent
enabled existing FP32 score centering for full-attention generalized routing.
Original-contract and broader final-policy revalidation are pending; this report
does not mark stage 03 passed.

## Starting evidence

Fresh source investigation: `AUTODEBUG_reuse.md`. Original failure:
`final_request_reuse_layer5.json`/`.log`, request index 6, length/position 2049,
decode PCC .9948247973191795 versus unchanged .995. All nine prefills and the
other eight decodes pass. Exact real-input fixture:
`actual_reuse_failure_2049_1.pt`, rows 768:2818 of the recorded 4096-row layer-5
fixture, rebased to positions 0..2049.

All focused commands and return codes are recorded verbatim in
`reuse_control_commands.json`. They use `tests.run_optimized_decoder --defaults`
with `--layer 5 --length 2049 --real --decode --steps 1` and that input fixture.
The original runtime SHA256 is
`373afe87025f3bd33160cd80ea0930412ceaa1e1c51ce03d84dfb42957811f6a`.

## Hypothesis experiments

| Hypothesis/control | Artifact | Result | Verdict |
| --- | --- | --- | --- |
| Prior-request state or trace captured at length 31 is necessary | `reuse_control_fresh.json` | Fresh run .9948247936074835; deterministic trace | Refuted as necessary cause |
| LoFi output projection is repaired by HiFi4 alone | `reuse_control_output_hifi4.json` | .9948778278726624 | Proposed repair refuted |
| Shared MLP reader count is repaired by one reader alone | `reuse_control_shared_readers1.json` | .9948247936074835 | Proposed repair refuted |
| Centering FP32 scores before generalized-gate BF16 conversion repairs the miss | `reuse_control_router_center.json` | .9981819549518332 | Verified exact-input repair |
| Replacing generalized gate with ordinary FP32 routing repairs the miss | `reuse_control_router_fp32.json` | .9981727130158773 | Verified alternate boundary control |
| BFP8 expert activations are repaired by BF16 alone | `reuse_control_expert_activation16.json` | .9948212067944865 | Proposed repair refuted |

Every control has the same passing prefill PCC, clean runtime audits, and equal
repeated traced output. Program-cache-miss guards were not enabled in these
focused runs. Centering retains the cache dtype, page table, update/read ops,
attention compute configuration, expert precision, and composite implementation.

Rounded-window check: at position 2049, native FP32 SDPA uses a 128-token chunk
and reads through exclusive position 2176 (68 pages). Reuse allocates 4096 tokens
(128 pages); the fresh control allocates 3072 (96 pages). The failure is logical
page 64, row 1, chunk 16. There is no allocation overrun or dynamic chunk-size
change versus passing position 2047. Exact source references and arithmetic are
in `AUTODEBUG_reuse.md`.

## Fix and mechanism limits

The parent changed `tt/optimized_decoder.py:337` from
`generalized_router_center = sliding and generalized_router` to
`generalized_router_center = generalized_router`. Sliding already used this
default. Full attention now uses the already-tested optional implementation;
explicit overrides still apply. New runtime SHA256:
`d6f4d858d7358f6d52332d9d1747a8bdb25cfa359f5c6f5aedde2f2e4f5d9f93`.

`GeneralizedRouter` subtracts a shared row maximum in FP32 before converting the
scores to BF16 (`tt/optimized_decoder.py:739-743`). Exact ranking and softmax
are invariant to a shared shift, while BF16 rounding is not. The matched pair
therefore verifies a useful intervention at the routing representation boundary.
Neither final-output PCC nor the FP32 alternate control proves changed expert
IDs, a rank-8/9 tie, or a composite-kernel defect. No such mechanism is claimed.

No failed position was excluded and no threshold was lowered. Other failed
single-variable controls were not incorporated. Higher precision elsewhere is
not justified as a repair by these results.

## Final status and required verification

Exact-fixture repair: verified. Default integration: applied by the parent.
Original nine-request reuse and broader contracts under the new runtime hash:
pending the parent's `v2validated_*` runs. Also require the actual 4096/128
headline, actual 1025/512 stress, relevant maximum-context coverage, and final
complete-layer performance measurement before stage acceptance or speed claims.

This investigator inspected source/artifacts and wrote reports only; it did not
access hardware or edit the implementation. No build is required for the Python
default/report changes. Parent-owned formatting/static checks and final runtime
verification should be recorded when complete.

# Multi-user MTP speculation on the 8x4 stack (GLM53_SPEC_MULTI)

Status: design + host-side plan (generator.spec_multi_plan / spec_multi_k_by_rows, the adapter's opt-in SpecPlan).
The device step is the work list in section 3; nothing below is validated on device yet.

## 1. Why the served MTP is single-user today

`tt/spec_mtp.py` runs ONE user's step (slot 0) as one trace replay: verify (t = K+1 rows of that user), SpecCtl ACCEPT,
MTP catch-up + K-1 chained rows, SpecCtl GATHER. Every piece assumes one user:

| piece | single-user assumption |
|---|---|
| state rollback | candidate KDA states / conv histories in rows 0..K of the shared 32-row state; the next step reads row `sel` (word 0) |
| `kda_rec` REC_CHAIN | "the rows are consecutive tokens of ONE user", `static_assert(REC_UG_ROWS_OK, "one user group")`; built with GLM53_REC_UGROUPS=1 |
| `kda_pre` KDA_VERIFY | ROWS mode, B <= 4, one row group; the conv window of row j reads slot `sel`'s 3 history rows |
| sparse_dsa verify | MLA_DSA_ONE_SLOT on one index core; the indexer ring rows of slot 0 (ring tile rows 4g..4g+3), moved by ACCEPT |
| SpecCtl | one `sel` word, one record page, slot 0's DSA rings |
| MTP layer | catch-up rows 0..K of one user, chained single rows, a private latent cache for one slot |
| adapter / generator | `_spec_lone`: speculation only while exactly one request is live; plan k_by_rows ((1, K), (B, 1)) |

The plugin contract already supports per-row multi-request speculation (`num_valid_drafts` per row, `k_by_rows`
caps by live rows, SPEC_DECODE_CONTRACT.md section 2): no plugin change is needed.

## 2. Target: the B 16 1 M build, row groups

Build: max_batch 16 at SEQ 1 M with GLM53_STACK_NARROW_WIDTHS (1, 2, 4) + GLM53_STACK_NATIVE_WIDTHS=8: 16 slots
resident, small live counts decode on the narrow / native programs, and the t = 16 programs exist. `spec_multi_plan`:

| live users | rows / user (g) | drafts k | slots (group bases) | verify width | chained-row width |
|---|---|---|---|---|---|
| 1 | 4 | 3 | 0 | 4 (narrow; = today's lone step) | 1 |
| 2 | 4 | 3 | 0, 4 | 8 (native) | 2 |
| 3-4 | 4 | 3 | 0, 4, 8, 12 | 16 | 4 |
| 5 | 3 | 2 | 0, 3, 6, 9, 12 | 16 | 8 |
| 6-8 | 2 | 1 | 0, 2, .., 14 | 16 | 8 |
| 9-16 | - | 0 (plain decode) | - | - | - |

User u's verify group is rows [u g, u g + g): token rows (bonus + drafts), positions c_u + j, page-table rows = user u's
page row, candidate state rows. The users sit on the group bases (the generator's slot moves, as the lone step's
slot copies), so a group's other rows hold no live request -- the same invariant the lone step relies on for rows 1..K.
Plugin `k_by_rows` (K 3, 16 slots): ((4, 3), (5, 2), (8, 1), (16, 1)); rows past 8 live users decline per row.

## 3. Device work list (per module / kernel)

1. `kda_rec` REC_CHAIN: G groups of g rows. Row b's predecessor is b - 1 when b % g != 0 (this core's writer, the
   existing REC_CHAIN_SEM hand-off); a group start reads state row `sel[b / g]` (selector words 0..G-1). Relax
   `REC_UG_ROWS_OK` to "rows of one group share the user group"; build the verify stacks with GLM53_REC_UGROUPS=G.
2. `kda_pre` KDA_VERIFY: the ROWS-mode B <= 4 limit -> the t = 8 / 16 layouts (RSPLIT / pair packing of the native
   programs): row j of group u builds its conv window from slot `sel[u]`'s history rows (read once per group, before
   any write-back) and the launch's own pre-conv rows u g .. j - 1.
3. sparse_dsa verify: MLA_DSA_ONE_SLOT -> one index core per group (or the plain t = 16 lanes with per-row slots): each
   group's indexer writes its slot's ring rows; ACCEPT moves each group's ring rows (R rings x G slots). MLA attention
   stays per-row (per-row positions and page-table rows already make it causal inside a group and independent across
   groups); check the shard's token lanes (KPH auto: 8 lanes at t 8-16) with g consecutive positions per slot.
4. SpecCtl ACCEPT / BOOT / GATHER: per group: n_u, bonus -> token row u g, sel[u] = u g + n_u, pos_v rows of the group,
   mpos_cu / mpos_ch per group, one record page per group (or [G, 64]); GATHER fills token rows u g + 1 .. u g + k.
5. MTP drafting: the catch-up at width cu (all groups' rows; the MtpLayer TPMla / TPMoE at t = 16 -> watch the L1
   floor: GLM53_MLA_S1_DRAM=1 and trimmed `ts` sets were needed for narrow widths on the shard, 50tok_plan 2026-11-09);
   chained rows at width ch with row u = user u; one private MTP latent cache per group slot (4 x GLM53_SPEC_MAX_CTX)
   or one cache with per-group page rows.
6. Heads: the verify head clone at wv (16), the MTP head at cu / ch.
7. Generator / adapter: `spec_enter(slots)` places the users on the group bases (slot moves) and seeds every group;
   `spec_step` returns per-group (n_u, tokens); the adapter's `_spec_lone` gate becomes "live <= full // 2 and every live
   row greedy and below GLM53_SPEC_MAX_CTX", switching to plain decode (num_valid 0) otherwise; `spec_multi_capable` on
   the backend turns the opt-in plan into multi-row verifies.
8. Long context (phase 2): the MTP layer's attention is dense (+~2.4 ms per step per 32 K positions); past 256 K run it
   on the main sparse_dsa path (the indexer's top-k over the MTP latent cache) or keep GLM53_SPEC_MAX_CTX per user.

## 4. Validation rows (main session queue, in order)

- Host: test_stack_adapter_host.py (spec_widths / spec_plan_for extended), test_spec_decode_host.py.
- e2e, B 16 build at SEQ 1 M (the slot build), plain: live 1 / 4 / 16 (baseline step, ids).
- e2e spec A/B, B 16 build, GLM53_SPEC_MULTI=1 GLM53_SPEC_K=3 GLM53_MK_LIVE=4 GLM53_MK_DISTINCT=1, ISL 128 and 4 K:
  every user's ids equal its plain run (greedy lossless), tau per user, step ms vs the 16-row plain step.
- Probe: the t = 16 verify step time (traced) vs plain t = 16 and t = 4 steps.
- Served: the 1 M x 16 profile with GLM53_STACK_VLLM_SPEC=1 GLM53_SPEC_MULTI=1, 4 concurrent greedy users: texts equal to
  the plain profile's, TPOT per user; then AgentX (after the prefill / slot work lands).

## 5. Expected gain

At 4 live users the verify runs 16 rows: the plain 16-row step costs ~29 ms vs ~21 ms at 4 rows (probe 2026-11-12),
and the MTP rows add the catch-up (16 rows) + 2 chained 4-row launches. With tau ~3.2 (served, single user) the 4-user
rate is ~3.2 tokens per (29 + MTP ~8) ms ~ 86 tok/s per user vs ~47 plain at 4 users: ~1.8x; at 1-2 users the
narrow / native widths keep today's ~2.5x. Agentic traffic accepts less (Gemma 31B AgentX: draft-1 acceptance
0.36-0.49): measure tau on AgentX before assuming the gain.

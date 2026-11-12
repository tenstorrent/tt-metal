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

## 6. Batch 2 implementation plan (not started; anchors in tt/spec_mtp.py, tt/generator.py, the adapter, test_mk_e2e.py)

Batches 1 + item 3 (commits 0df437eed8d, 8336baae69c) give the verify kernels their groups: REC_CHAIN_GROUPS,
KDA_VERIFY_GROUPS, SPEC_GROUPS, MLA_DSA_GROUP_ROWS. Batch 2 builds and drives them. Fixed-G first: one SpecMTP per
process for exactly G = GLM53_SPEC_MULTI_USERS live users (default 4), speculating only while exactly G requests are
live on the group bases; variable live counts (a G = 4 build with inactive groups, SpecCtl skipping c < 0) after.

1. SpecMTP.__init__ (spec_mtp.py ~432): G from GLM53_SPEC_MULTI / _USERS; self.w = G (K + 1); wv = the smallest build
   width >= w (B 16 build, K 3: 8 at G 2, 16 at G 4; spec_widths grows a `groups` argument, the single-user rules
   unchanged at G 0); drop the `wv <= 8` assert for G > 1; masks_v rows < w; self.ctl = SpecCtl(mesh, K, groups=G);
   self.rec = dram_rows(G pages of 64 words); mpos_ch pages: row u = user u (ch = the smallest width >= G);
   mw_cu = wv, mw_ch = ch; MtpLayer ts {cu, ch} (watch L1 at t 16: GLM53_MLA_S1_DRAM=1, trimmed ts).
2. _build_verify_stacks (~600): around the KDA layer builds `env(GLM53_REC_UGROUPS=str(G))` and
   `wts.attn.kda_verify_groups = True` (deleted / restored with kda_verify_sel); the sparse_dsa verify launches are
   built with GLM53_MLA_VERIFY_GROUPS=str(G) wherever GLM53_MLA_ATT_VERIFY=1 is set around a verify launch (prepare /
   capture), so mk_mla emits MLA_DSA_GROUP_ROWS (item 3 builds here first).
3. set_page_table (~1113) -> set_page_tables(rows[G]): pt_v rows u g .. u g + K = user u's row (shard: each user's
   local rows via be.shard_local_rows); the MTP private cache becomes G caches of nb_cap blocks: pt_m rows u g + j =
   group u's identity row offset by u nb_cap; chain row u -> group u's.
4. _dsa_rings (~870): group-major [dsa_slot_addrs(be._dsa_st(kv, i, u g), 1)[1] for u in range(G) for i in dsa].
5. MTP chain source rows: MkEh.norm_h_proj(sel=) reads word 0 (kernels/spec/eh_dm0.cpp EH_SEL, T = 1). New define
   EH_SEL_ROWS: output row t's source = selector word t (the chained launch's row u = MTP output row u (K + 1) + n_u =
   SpecCtl's selector word u). The catch-up (cu = wv rows) needs no selection (row r = verify row r).
6. Heads (_make_heads ~691): verify = head_clone(be.head, wv); mtp_cu at cu; mtp_ch at ch (K - 1 clones).
7. Generator (generator.py ~2316-2420): spec_enter(kv, slots=(0, g, 2 g, ..)) seeds every group (BOOT); spec_step /
   spec_step_submit / spec_step_collect return per-group {n, tokens}; _spec_settle per slot; spec_room per slot;
   generate_batch(spec_k=K) with GLM53_SPEC_MULTI and live == G places the prompts on the group bases (no slot moves
   in v1) and walks every user's accepted tokens.
8. Adapter: backend.spec.groups > 0 -> spec_multi_capable; the gate "live == G (v1; later live <= 8), every live row
   greedy, every position < GLM53_SPEC_MAX_CTX" turns num_valid on for the live rows; else plain (num_valid 0).
9. test_mk_e2e.py _spec_ab: live > 1 -> every user's spec ids == its plain ids (the first mismatch, the top-2 gap),
   tau per user, the step ms.

First device rows after batch 2 (frozen b3): e2e knob-on B 16 at SEQ 1 M (GLM53_STACK_NARROW_WIDTHS=1,2,4
GLM53_STACK_NATIVE_WIDTHS=8), GLM53_SPEC_MULTI=1 GLM53_SPEC_MULTI_USERS=4 / 2 with GLM53_MK_LIVE=4 / 2, ISL 128 and
4 K: per-user ids == plain; a knob-off e2e regression (single-user K 3 at ISL 128 on B 4) on the same snapshot.

Batch 2 as implemented (host-tested only; device rows above pending on the frozen b3), deviations from the list:
- item 2: `kda_verify_groups` carries G (an int): kda_pre reads it as the group count (`True` would build ONE group;
  test_mk_layer_verify_groups set `True` up to b2 -- its U = 2 / 4 rows on b2 cannot pass, fixed for b3).
- item 5 also needed the MTP catch-up at cu = 16 > 4 on the FLAT residual: eh_expand.cpp / eh_dm0.cpp EH_FLAT (the
  expand writes, the first chained launch reads [1, 1, t, 16384]), and the MTP layer's dense attention per group:
  att_dm0.cpp MLA_ATT_GROUP_ROWS (a row's latent hand-off rule = its group base's position tile, not row 0's; mk_mla
  emits it for the dense MkMla under GLM53_MLA_VERIFY_GROUPS).
- item 3: the chained launches get their own MTP page table (pt_mc, row u = user u) -- the catch-up's pt_m rows
  u g + j index the same table by launch row; the MTP-cache prefill arms every group base (StackBackend
  arm_mtp_prefill / TPPrefill._mtp_write per slot, SpecMTP.mtp_row(slot=)); generate_batch prefills the G users on
  the per-user path (no batched / packed admission: those fill no MTP rows). The launch-mode prefill (M0) is refused.
- items 7 / 8: the step speculates while exactly the G requests are live; the first user to finish ends it for all
  (spec_leave, then plain decode); the adapter admits new requests on free group bases first, and a group base
  released mid-speculation LEAVES (the others' committed rows move to their bases) instead of aborting.

## 7. K 1 at 4-8 users, the open-pool trap, the validation geometry (2026-11-12)

Zoe Carver's regen measurements (notes glm-glx-mtp2, TP8 trays): K 1 pays at >= 4 users (B 4 49 -> 70, B 8 39 -> 53
t/s/u), 8 users x 2 rows on a 16-row verify; agentic long-context acceptance 0.80-0.84. The fixed-G step covers it as
GLM53_SPEC_K=1 with GLM53_SPEC_MULTI_USERS = 8 (B 16 build, w = 16) or 4 (w = 8: a build whose decode widths hold 8,
e.g. B 8 -- a B 16 build has no 8-row width: native widths need the 32-row fabric phase).

Per grouped define at g = 2: REC_CHAIN_GROUPS (kda_rec: ug = G, rows / ug = 2 <= 8 per core group), KDA_VERIFY_GROUPS
(kda_pre: G <= 16, one row pair per group, the selector page's 16 words), SPEC_GROUPS (G (K + 1) <= 32), MLA_DSA_GROUP_ROWS
/ MLA_ATT_GROUP_ROWS (2 rows per group, t <= 16) hold for G 8; EH_SEL_ROWS is not used (K 1 has no chained launch: the
catch-up head drafts d'1 and GATHER reads mout row u (K + 1) + n_u). Two changes:
- SpecCtl ACCEPT's DSA rings (SPEC_REV 4): the kernel took R x G slot bases (<= 64) -- G 8 x the 11 sparse_dsa layers =
  88 failed the host assert. Now R tensor bases (slot 0) and user u's ring = slot u (K + 1) of each (pages slot x 8 + i:
  the generator slot is the DSA state slot, StackBackend._dsa_st), for any G. The single-user kernel is unchanged.
- spec_widths(groups) at K 1: ch = w (no chained launch, so no extra MTP width / eh buffers for it).

The trap (glm-glx-mtp2 "MLA, index pool": their batch MLA saved the open-pool record only when the user's last row left a
pool open; with rows [p = 2, p + 1 = 3 mod 4] the record lost row p, and a rejected p + 1's repeat pooled the key from a
stale row). Ours is not affected: ix_dm1 writes every row's ring row back to DRAM (MLA_DSA_IX_ROWS: rows 0..3 of its
pool's group, 4 (q - q0) + p % 4 in a one-slot verify) on every chip, the owner chip writes the pooled key of p / 4 from
the pool's 4 ring rows, rows score only pools < (p + 1) / 4 after every row's writes, a pool a rejected row closed is
rewritten when that pool closes again, and ACCEPT moves the next base's pool rows to 0..3 per group (q0 = c_u / 4 in
both kernels). tt/shard_plan.py spec_ring_emulate models exactly that against sequential decode of the committed tokens
(K 1-3, G 1-8, random accepts, plain steps between verifies): 0 mismatches; Zoe's save-only-if-open variant fails at the
trap's case (prompt 6, K 1, the closing draft at 7 rejected: row 6 lost, pool 1 scored stale) and so does ACCEPT without
the move (tests/test_shard_plan_8x4.py test_spec_ring_rollback).

The L1 floor of the B 16 build (cores 0-1..0-7 = column 0's mhc_pre K members, i.e. sparse_dsa owners): every failing
row ended a program's static CBs there at the same 1,253,888 B (inferred: the t = 16 sparse_dsa program, the owners'
largest DSA CBs); what moved is the lowest L1 buffer -- 1,250,240 (narrow 4), 1,224,768 (narrow 1, 2, 4 + MTP G 4),
1,217,856 (narrow 4 + GLM53_MLA_S1_DRAM). The owners' DSA CBs scale with the slot's local pooled-key tiles (tp = SEQ / 4
/ 32 pools per chip; shard_plan.owner_l1, R8, t 16 on 8 lanes, KPH 4): CB_AKEY + CB_ASEL + mailbox 40,448 B at 1 M,
30,208 at 256 K, 25,088 at 128 K (+ the ASCR page-table row: 2,048 / 512 / 256 B) -- 256 K lowers that program's CB end
by ~11.8 KB, 128 K by ~17.2 KB; the narrow widths' and the MTP's L1 tensors do not scale with SEQ. The multi-user MTP
build WITHOUT narrow widths (and without S1_DRAM) has not run at 1 M yet. Validation order
(_logs/mesh/chain_mmtp_k1.sh): the 8x4 layer tests U 8 K 1 / U 4 K 1, then the e2e G 8 K 1 on B 16 without narrow widths
at 1 M, else 256 K, else 128 K (GLM53_SPEC_L1_DEBUG / GLM53_L1_REPORT attribute the L1), G 4 K 1 on a B 8 build, G 4 K 3
on B 16 at the fitting SEQ, and the single-user K 3 knob-off regression.

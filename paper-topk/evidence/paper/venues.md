# Venue + Deadline Scan — "Exact Top-K Selection on a Dataflow Many-Core"

Scan date: **2026-08-16**. Target: September 2026 submission window. All dates AoE unless noted.
Verification status is marked per venue: **[VERIFIED]** = read from official 2027 CFP this week; **[PATTERN]** = 2027 CFP not yet posted, estimated from the immediately preceding edition (source cited).

## Headline findings

1. **PPoPP 2027 is already closed.** The paper deadline was **August 3, 2026** (passed 13 days ago) — PPoPP moved its deadline from the historical Aug/Sept slot to early August. The user's "PPoPP September" assumption no longer holds.
2. **HPCA 2027 is also closed** (abstract registration July 24, 2026).
3. Three venues have live deadlines in or adjacent to September 2026: **ASPLOS 2027 September cycle (Sept 9)**, **EuroSys 2027 fall cycle (abs Sept 17 / paper Sept 24)**, and **IPDPS 2027 (abs Oct 1 / paper Oct 8)** — IPDPS is the best topical fit and the recommended primary.
4. No verified venue blocks an arXiv preprint. IPDPS explicitly permits arXiv submissions; ASPLOS permits with a "no advertising" quiet period; PPoPP explicitly encourages normal dissemination.

---

## 1. Per-venue table

| Venue | Deadline(s) | Status for Sept 2026 plan | Page limit | Artifact eval | Double-blind | Fit (1–5) |
|---|---|---|---|---|---|---|
| **IPDPS 2027** (Seattle, Jun 1–5 2027) | **Abs Oct 1, paper Oct 8, 2026** (firm) [VERIFIED] | **LIVE — primary match** | 10 pp double-column incl. figs, refs unlimited | No formal AE; post-acceptance Best Open-Source Contribution Award (self-nominate) | Yes (double-anonymous) | **5** |
| **ASPLOS 2027** (Sept cycle) | **Full paper Sept 9, 2026** (no abstract deadline); notify Dec 21, 2026 [VERIFIED] | LIVE but only ~3.5 weeks out | 11 pp excl. refs/appendices; +2 pp on acceptance | Yes, established AE with badges; artifacts valued | Yes; rapid-review round on first 2 pages | 3 |
| **EuroSys 2027** (Rabat, Apr 19–24 2027; fall cycle) | **Abs Sept 17, paper Sept 24, 2026**; notify Jan 29, 2027 [VERIFIED] | LIVE | 12 pp technical content, refs unlimited | AE traditional at EuroSys (CFP fetch didn't enumerate) | Yes | 2.5 |
| **PPoPP 2027** (Salt Lake City, Mar 20–24 2027) | Paper **Aug 3, 2026** — **PASSED**; notify Oct 26 [VERIFIED] | **MISSED** — next window PPoPP 2028, expect ~Aug 2027 | 10 pp excl. refs | Optional, encouraged (post-acceptance, Nov 9) | Yes | 5 (but closed) |
| **HPCA 2027** (Salt Lake City, Mar 20–24 2027) | Abs **Jul 24, 2026** — **PASSED** (paper ~1 wk later) [VERIFIED] | **MISSED** — HPCA 2028 ~Jul 2027 | ~11 pp (typical) | Yes (badges) | Yes | 2.5 (but closed) |
| **MLSys 2027** (Bellevue, May 17–22 2027) | **Paper Oct 30, 2026**; rebuttal Jan 12–16; notify Jan 26, 2027 [VERIFIED via deadline trackers; single deadline, not rolling] | Live, 6 weeks after target | ~10 pp (typical MLSys) | Yes, AE encouraged | Yes | 3 |
| **ICS 2027** | CFP not posted. ICS 2026 pattern: **two cycles — abs Dec 9/paper Dec 16, 2025 and abs Feb 2/paper Feb 9, 2026** → expect ~Dec 2026 and ~Feb 2027 cycles [PATTERN] | Backup (winter) | ~10 pp excl. refs (recent pattern) | ACM badges, optional (pattern) | Yes (pattern) | 4 |
| **SC27** (Denver, Nov 14+ 2027) | CFP not posted. SC26 pattern: **abs Apr 1 / paper Apr 8** (no ext.), mandatory AD/AE appendix Apr 28, notify Jul 1 → expect ~Apr 2027 [PATTERN] | Backup (spring) | ~10 pp excl. refs + reproducibility appendix (pattern) | **Mandatory** AD (Artifact Description) appendix; AE optional | Yes (pattern since SC22) | 4 |
| **ISPASS 2027** | CFP not posted. ISPASS 2026 pattern: **abs Dec 8 / paper Dec 15, 2025** (EST), conf late Apr → expect ~Dec 2026 [PATTERN] | Backup (winter) | ~11 pp (pattern) | Light/optional | Yes (pattern) | 4 |
| **DaMoN 2027** (w/ SIGMOD, ~Jun 2027) | CFP not posted. Pattern: **~mid-Feb–mid-Mar 2027** (2026: Feb 20; 2025: Mar 14) [PATTERN] | Spin-off slot | **6 pp excl. bib** (camera-ready 10); short papers 2 pp | Light | **Single-blind** (2026 verified) | 3.5 (as a 6-pp spin-off of C1+C4, not the full paper) |
| **ACM JEA** (J. Experimental Algorithmics) | **Rolling** — no deadline | Archival anytime | No hard limit (journal) | Encourages code/data availability | Single-blind (editor-mediated) | **5** (experimental algorithmics is literally the paper's genre) |
| **IEEE TPDS / ACM TOPC** | **Rolling** — no deadline | Archival anytime | TPDS ~14 pp double-column incl. refs (over-length fees); TOPC journal-length | Reproducibility badges optional | Single-blind | 4 |

### Fit rationale (compressed)

- **IPDPS (5/5):** The CFP's own track list is the paper's outline — *Algorithms* (C2 log-tree, C3 chunk-skip + skip law), *Measurements, Modeling, and Experiments* (C1 negative result, C4 silicon characterization, calibrated forecast), *Architecture*. IPDPS has a long tradition of "algorithm X mapped to novel parallel machine Y with measurements," and measured negative results with cost models are in-scope for the M&M&E area. 10 pp is tight but workable if the characterization section leans on an arXiv extended version.
- **PPoPP (5/5, closed):** Exact-fit venue (parallel algorithm design + real-machine measurement) — missed by 13 days; keep for 2028 only if everything else falls through.
- **ICS (4/5):** Supercomputing-systems + algorithms venue; accelerator design-space studies land well; two cycles give schedule flexibility.
- **SC (4/5):** Strong home for measurement studies with reproducibility appendices; large-scale angle is thinner (single chip, 104 cores), which SC reviewers may probe; the mandatory AD appendix is easy given the campaign ledger.
- **ISPASS (4/5):** The C4 silicon characterization + measured floors + calibrated host-simulation forecast is core ISPASS material; the algorithmic contributions (C2/C3) get less credit there.
- **MLSys (3/5):** Top-k is an ML-serving kernel and Tenstorrent hardware is topical, but MLSys prefers system-level ML workloads over a single-kernel design-space study; framing would need an ML-inference wrapper.
- **ASPLOS (3/5):** In-scope (architecture/software interplay on new silicon), but ASPLOS expects a systems/architecture *idea*; a measured algorithm-mapping study is a harder sell, and Sept 9 leaves ~3.5 weeks — only viable if the paper is essentially written.
- **EuroSys (2.5/5):** OS/distributed-systems center of gravity; a kernel-level algorithm study on an accelerator is peripheral.
- **HPCA (2.5/5, closed):** Characterization angle fits, but HPCA wants architectural mechanisms/proposals.
- **DaMoN (3.5/5):** Top-k selection is a classic DB operator; "data management on new hardware" is exactly the C1+C4 story — but at 6 pages it's a spin-off, not the main paper. Single-blind, so no anonymity friction at all.
- **JEA (5/5 archival):** Measured design-space study + cost models + soundness proof + negative result = textbook experimental algorithmics. Rolling, no page pressure, allows the full campaign ledger. Slow reviews (6–12 mo typical) — use as archival terminal, not first strike.

---

## 2. What's actually live in/near September 2026

| Date | Event |
|---|---|
| Sept 9, 2026 | ASPLOS 2027 fall-cycle full paper (no abstract step) — feasible only if draft is ~done |
| Sept 17 / 24, 2026 | EuroSys 2027 fall abstract / paper — weak fit, not recommended |
| **Oct 1 / 8, 2026** | **IPDPS 2027 abstract / paper — the real September target** (write in Sept, abstract Oct 1) |
| Oct 30, 2026 | MLSys 2027 paper — fallback if IPDPS draft slips or an ML framing emerges |

The user's expected pair was "PPoPP + IPDPS." **Half survives contact with 2026 CFPs: IPDPS yes (Oct 1/8), PPoPP no (closed Aug 3).**

---

## 3. arXiv / double-blind interaction (verified policies)

- **IPDPS 2027 [VERIFIED]:** "Having an arXiv paper does not prohibit authors from submitting a paper to IPDPS 2027." Requirements: keep the submitted PDF anonymized, don't cite/point reviewers to the arXiv version, don't deanonymize via the paper itself. → **Safe to post the preprint before the Oct 8 submission.** Mild prudence: use a slightly different title/abstract on arXiv so a trivial title search doesn't hit, though the policy doesn't require it.
- **ASPLOS 2027 [VERIFIED]:** arXiv posting allowed and not prior publication, but **no "advertising"** (social media, talks pointing at it) from two weeks before the deadline until decisions. If targeting the Sept 9 cycle, an arXiv post should have gone up before ~Aug 26 or wait until after notification.
- **PPoPP 2027 [VERIFIED, moot]:** explicitly encourages normal dissemination including web drafts.
- **EuroSys 2027 [as fetched]:** preprints permitted; the CFP (per fetch) asks the submission to differ in title/system name from the public preprint — unusually strict, double-check the CFP text directly if EuroSys is ever pursued.
- **SC / ICS / ISPASS [PATTERN]:** all have accepted arXiv-preprinted work under double-blind in recent editions; same "don't self-cite deanonymizingly" hygiene applies.
- **Journals (JEA/TPDS/TOPC):** single-blind, arXiv irrelevant.

**Net: the arXiv-first strategy is compatible with every recommended target.** Post the arXiv preprint at will; only ASPLOS imposes a publicity quiet period.

---

## 4. Recommendation

**Primary: IPDPS 2027** — abstract **Oct 1, 2026**, paper **Oct 8, 2026** (firm). Best-fit tracks: *Algorithms* or *Measurements, Modeling, and Experiments*. September becomes the writing month, which matches the requested timeline exactly. Bonus: the review pipeline gives an early-reject signal by **Nov 30, 2026** and a revision round (revised paper Jan 18, final decision Feb 1, 2027) — the revise-and-resubmit stage materially raises acceptance odds for a well-measured study.

**Backup chain (calendar):**

| When | Action |
|---|---|
| **Sept 2026** | Write. Post arXiv preprint (v1) any time — no conflict with IPDPS. Skip ASPLOS Sept 9 unless the draft is unexpectedly finished; skip EuroSys (fit). |
| **Oct 1 / Oct 8, 2026** | IPDPS 2027 abstract / paper. |
| **Oct 30, 2026** | (Contingency only) MLSys 2027 if the IPDPS submission was aborted and an ML-serving framing is preferred. |
| **Nov 30, 2026** | IPDPS early-reject signal. If early-rejected → immediately retarget **ISPASS 2027** (~Dec 2026 deadline, pattern; confirm CFP in Oct/Nov) with a characterization-forward reframe, or **ICS 2027 cycle 1** (~Dec 2026, pattern). |
| **Dec 18, 2026 – Feb 1, 2027** | IPDPS first-round decision → revision → final decision. |
| **~Feb 2027** | If IPDPS rejects at final decision: **ICS 2027 cycle 2** (~Feb 2027, pattern) — closest full-length refit. In parallel, carve the 6-page **DaMoN 2027** spin-off (C1 negative result + C4 silicon characterization, top-k-as-DB-operator framing; single-blind, deadline ~Feb–Mar 2027 pattern). |
| **~Apr 2027** | If ICS also misses: **SC27** (abs ~Apr 1 / paper ~Apr 8, 2027 pattern) with the mandatory AD appendix built from the campaign ledger. |
| **Any time / terminal** | **ACM JEA** (rolling) as the archival home for the full-length version with all four contributions, proofs, and the complete measurement ledger — submit the extended version here even if a conference version is accepted (JEA welcomes extended versions of conference papers). IEEE TPDS is the faster-turnaround rolling alternative. |
| **~Jul–Aug 2027** | Last-resort loop-back: HPCA 2028 (~Jul 2027) / PPoPP 2028 (~Aug 2027) with whatever the paper has become. |

**One structural suggestion falling out of the scan:** the paper's four contributions over-fill a 10-page IPDPS budget. Plan the split now — full story on arXiv (no limit), IPDPS 10-pager centered on C1+C2+C3 with C4 compressed to a half-page table pointing at the arXiv version, and the DaMoN 6-pager (C1+C4) plus the JEA archival version as pre-planned derivatives rather than salvage.

---

## Sources

- [PPoPP 2027 papers track (sigplan.org)](https://ppopp27.sigplan.org/track/PPoPP-2027-papers) — Aug 3, 2026 deadline, 10 pp, DB, AE optional
- [IPDPS 2027 Call for Papers (ipdps.org)](http://www.ipdps.org/ipdps2027/2027-call-for-papers.html) — Oct 1/8, 2026; 10 pp; double-anonymous; arXiv OK; review timeline
- [ASPLOS 2027 CFP (asplos-conference.org)](https://www.asplos-conference.org/asplos2027/cfp/) — Sept 9, 2026 cycle; 11 pp; DB; arXiv quiet period
- [EuroSys 2027 CFP (2027.eurosys.org)](https://2027.eurosys.org/cfp.html) — Sept 17/24, 2026 fall cycle; 12 pp; DB
- [HPCA 2027 (SIGARCH call)](https://www.sigarch.org/call-contributions/hpca-2027/) — abstract Jul 24, 2026 (passed)
- [MLSys deadlines tracker](https://mlsys-deadlines.github.io/) and [mlciv MLSys 2027](https://mlciv.com/ai-deadlines/conference/?id=mlsys27) — Oct 30, 2026; Bellevue May 17–22, 2027
- [ICS 2026 CFP (dipsa-qub.github.io)](https://dipsa-qub.github.io/ICS2026-webpage/call-for/call-for-papers.html) — two-cycle pattern (Dec/Feb)
- [SC26 dates & deadlines (sc26.supercomputing.org)](https://sc26.supercomputing.org/all-dates-deadlines/) — Apr 1/8 pattern, mandatory AD appendix
- [ISPASS 2026 submissions (ispass.org)](https://ispass.org/ispass2026/submission.php) — Dec 8/15 pattern
- [DaMoN (damon-db.org)](https://damon-db.org/) — Feb 20, 2026 deadline; 6 pp excl. bib; single-blind
- [SC27 listing (showsbee)](https://www.showsbee.com/fairs/100096-SuperComputing-Conference-2027.html) — Denver, Nov 2027

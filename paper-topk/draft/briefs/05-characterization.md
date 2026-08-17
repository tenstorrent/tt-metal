# Brief 05 — Characterization: Tensix Datapath Numerics and Selection Primitives (C4)

**Section file:** `sections/05-characterization.tex`
**Budget:** 2.5 columns (1.25 pages).

## Single job

Bank the silicon facts as an archival reference: the numeric semantics and
sorting-relevant primitive costs of the Tensix datapath, none of which appear
in any prior Tensix publication (verified against the citation graph of the
Wormhole stencil paper, related-work.md §4). The section's claim is NARROW by
design: *datapath numerics + selection primitives*, not "first Blackhole
characterization" — the informal microbench report (blackholemicrobench2025)
owns the general axis and must be cited in the first paragraph.

## Content plan (mirror Tab C1's row order)

### 5.1 bf16 datapath canonicalization (C4-1)

- NaN (any payload) → same-sign Inf; −0 → +0; ±subnormal → +0 (sign
  dropped); index positions preserved; fp32 passes bit-exactly.
- Found by identity-op probe; corroborated by vendor ISA docs (Dst
  behavior); baked into all reference models; enforced by a 62-cell contract
  suite with a JSONL divergence ledger.
- Framing: any bit-exactness claim on this datapath must model
  canonicalization first — this is the fact that makes the paper's
  "exact/bit-exact, never PCC" gates possible.

### 5.2 Sign-magnitude total order including NaN (C4-2)

- −NaN < −Inf < … < −0 < +0 < … < +Inf < +NaN, implemented natively by
  SFPGT, SFPSWAP, and the packer threshold unit; discovered via a failing
  ±NaN differential test. Packer NaN asymmetry (+NaN survives the threshold,
  −NaN zeroed) is a local finding.
- Software prior-art paragraph (related-work.md §4 rule): the bit-flip
  float→orderable-uint trick (herf2001radix, merrill2011radix, CUB/Thrust)
  and IEEE 754-2019 totalOrder are *software encodings*; the finding here is
  that the *hardware datapath itself* imposes this order — state it exactly
  that way.

### 5.3 Packer exponent histogram: free, sampled, aliased (C4-3)

- Enable cost zero (25.175 → 25.104 cyc/tile). But: samples 1-in-8
  positionally (p mod 64 < 8 — exactly 128 increments per 1024-datum tile,
  format-independent); bins alias exponent & 31 (32 bins; exp 127 ≡ 159);
  8-bit counters saturate at 255; sign-blind; the mode-6/7 WhichPackers
  field is ignored (a silent 4× trap); CLREXPHIST doesn't fence in-flight
  packs (≈39-count leak).
- Contradicts the Wormhole-era doc (per-datum counting) — on Blackhole it is
  a 12.5% positional sample. 38/38 functional + 6/6 perf tests; sampling
  proved by construction (marker patterns + mod-8 phase sweep).
- One sentence on consequence: kills naive histogram-select designs (§3),
  but remains usable as a threshold *seed* (reopening condition texture).

### 5.4 Packer threshold filter (C4-4)

- T ≥ 0: free compare-and-zero (4.034 cyc/vec vs 3.855 floor, +4.6%).
- T < 0: documented UB, measured as |T| rounded up to the next power of two
  (mantissa ignored) — unusable. SFPU fallback (SFPGT(SET_VD)+SFPAND,
  2 issues/vec) is the ISA floor.

### 5.5 Dst window layout + SFPSWAP operand order (C4-7, C4-8)

- word(r) = (r mod (K/16))·16 + r/(K/16) — the descending window is
  column-major over K/16 physical rows; validated exhaustively at
  K=512/1024/2048 by full-region dumps against distinct-monotone input.
  Honest sentence: an a-priori derivation was wrong; calibration replaced it.
- SFPSWAP mod1=1: max→VC, min→VD — the in-tree comment reads backwards;
  caught by a decision tracer accumulating −inf. (Both facts are what make
  the C3 rendezvous implementable; say so.)

### 5.6 Measured floors table (C4-5, C4-6)

- Count rate 2.0000 cyc/vec exactly; unpack-to-dest floor 3.938 cyc/vec;
  rendezvous 81/101/98 cyc by sync arm; MMIO ≈10 cyc/word; PassSync ≥25.1.
- SFPLOADMACRO co-issue law: Load+Simple+{MAD|Store} maps at 1.000–1.003
  cyc/vec (three sub-units co-issue free); reductions need ≥2 issues
  (SFPIADD not macro-schedulable) → count = 2 cyc/vec. This is the law that
  explains both C1's counting ceiling and the shipped micro-op wins (H10:
  merge 1.195/1.221/1.235×, step 1.053/1.074/1.090× at K=512/1024/2048 —
  cite as the law's application, one sentence).

## Claims owned

C4-1..C4-9 (evidence.md §1.5). C4-9 (harness findings: PerfConfig stimuli
bug, fill-arm semaphore leak, one-module-one-schema) is OPTIONAL here — one
sentence max, or push entirely to §7's methodology subsection (default:
push to §7).

## Figures/tables owned

- **Tab C1** (1.5–2 col, the section's anchor): characterization table —
  rows: canonicalization rules; sign-magnitude order; histogram properties
  (sample rate / aliasing / saturation / sign-blind / WhichPackers trap);
  threshold filter T≥0 and T<0; Dst window formula; SFPSWAP operand order;
  rendezvous floors; SFPLOADMACRO co-issue law. Columns: property | measured
  behavior | bench/probe | consequence for selection. Evidence rows
  C4-1..C4-8 each name their bench file.
- **(Optional) Fig C1** (0.5 col): Dst window layout diagram (word(r)
  formula as a picture) — include only if §6's figures fit; the formula
  alone may suffice.

## Style directives

1. This is the Jia-et-al.-genre section (jia2019ipu is the template
   citation) — findings stated as laws with their probe named in the same
   sentence.
2. Every quirk gets its consequence: "sampled 1-in-8 → any exact-count
   design on the histogram is dead on arrival; sampled seeding survives."
   No consequence, no column-inches.
3. Cycle figures here are slope-measured (two-point, marker-canceled) —
   quote to the precision the harness earned (2.0000, 3.938, 25.104) and
   state the method once at the top, pointing to §6's methodology for the
   tripwire details.
4. The [local finding] tags from the working notes become explicit
   "measured here, absent from vendor documentation" sentences — that
   contrast IS the contribution.
5. Candor pattern on 5.5: "derivation failed, calibration decided" — one
   clause, no self-flagellation; it sets up the methodology section's
   trust-nothing theme.

## Hazards

- NEVER print "first Blackhole characterization." The narrowed claim
  (datapath numeric semantics + sorting-relevant primitive costs) is
  uncontested AND archival where the closest competitor is a dead link —
  make that archival point once, politely (related-work.md §4 close).
- The 3.855 vs 3.938 floor pair: 3.855 is the threshold-filter bench's
  local floor, 3.938 the unpack_to_dest floor — different quantities; keep
  them in their own rows and never "reconcile" them.
- Vendor ISA doc references (Dst.md, SFPSWAP.md) are citable as
  "vendor ISA documentation" — no repo URLs (anonymity).
- µs conversions of any cycle figure carry the 1.35 GHz caveat; prefer
  leaving this section in cycles entirely.

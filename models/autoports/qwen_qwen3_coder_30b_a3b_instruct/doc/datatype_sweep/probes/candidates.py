# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The stage-07 datatype sweep candidate set, with the reason each row exists.

Every candidate is a ``PrecisionConfig`` expressed as a **delta from
``DEFAULT_PRECISION``**, because the question the sweep asks is "what does
moving this field cost and buy", not "what does this 20-field config score".
Writing the rows as deltas also makes the CSV's ``dtype_policy`` column a short
string rather than twenty repeated defaults.

Design of the set
-----------------

Decode at batch 1 is bound on **weight bytes pulled per token**, so the rows are
ordered by how many bytes each group actually moves per die per token:

===================  ==================================================  ==========
group                per-die bytes read per decode token (default)       lever
===================  ==================================================  ==========
experts (8 of 128)   bfloat4_b already -- the cheapest block float       block width
lm_head              2048 x 37984 at bfloat8_b ~ 82 MB, read every token dtype
attention qkv+wo     ~ 1024x1280 + 1280x1024 per layer at bfloat8_b      dtype
CCL wire             one reduce-scatter + all-gather per layer x 48      ccl_dtype
KV cache             512 B/token/layer/die, grows with context           kv_cache_dtype
===================  ==================================================  ==========

The **lm_head** is the single biggest non-expert read on the decode path and is
the only weight matrix read in full for every token, so it leads the set. The
experts are already at ``bfloat4_b``, which is why the expert rows here are
*block-width* rows and one reverse-direction ``bfloat8_b`` row that prices the
shipped choice rather than trying to beat it.

The BFP4 + LoFi obligation
--------------------------

The goal requires that **for every material BFP4 matmul group considered or
selected, a BFP4+LoFi candidate for that same group is included**. Stage 02
found LoFi free on ``bfloat4_b`` -- four mantissa bits leave HiFi4 nothing to
resolve -- so each group this sweep moves to ``bfloat4_b`` has a paired row that
also drops its fidelity to ``LoFi``:

* experts: already ``bfloat4_b`` + ``LoFi``; that *is* the shipped pair.
* lm_head: ``R01`` is bfp4 at the shipped ``HiFi2``, ``R02`` is the bfp4+LoFi pair.
* attention: ``R06`` is bfp4 qkv+wo at the op-default fidelity, ``R07`` is the
  bfp4+LoFi pair.

``attention_fidelity`` has no measured default (it is ``None`` == "the op
picks"), so ``R08`` sets ``LoFi`` with **no dtype change** -- the control row
that makes ``R07`` interpretable. Without it, an ``R07`` regression could not be
attributed to the dtype or the fidelity.

Block widths are co-tuned wherever a dtype changes
--------------------------------------------------

Stage 02 proved that sweeping dtype or ``in0_block_w`` alone finds the wrong
optimum: expert packing looked like 1.66x against an untuned baseline and was
1.09x against a tuned one. So ``R13`` (experts to ``bfloat8_b``) carries the
width that dtype wants rather than the width ``bfloat4_b`` wanted.

Legal widths only. ``_tuned_sparse_matmul_config`` **silently clamps** an
``in0_block_w`` that does not divide K in tiles, so a row asking for an illegal
width runs anyway and would be recorded under the width it asked for. The widths
below are divisors of the relevant K, and every row's *resolved* width is read
back from ``fallback_audit`` regardless.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SWEEP_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.precision import (  # noqa: E402
    DEFAULT_PRECISION,
    PrecisionConfig,
)

#: The **stage-06 shipped policy**, which every row here is a delta from.
#:
#: Pinned rather than taken from ``DEFAULT_PRECISION``, because stage 07 *moved*
#: ``DEFAULT_PRECISION``: the selected block widths are now its values, so a
#: candidate set defined against "the default" silently redefines itself the
#: moment a selection lands. Re-running tier A after the selection would then
#: report ``R00_default`` at 64/24 -- contradicting the 48-layer row that was
#: measured at 16/12 and is the baseline every gain in this sweep is quoted
#: against. Pinning the two fields makes the whole set reproducible from the
#: post-selection tree: re-running any probe rebuilds exactly the config that
#: row was measured at, which is also what ``configs/R00_default.json`` (written
#: at sweep time, before the default moved) still contains.
#:
#: Only the two fields stage 07 moved are named. If a later stage moves another
#: field of ``DEFAULT_PRECISION``, it must be added here or this set stops
#: reproducing itself -- ``_assert_baseline_is_stage06`` below is the tripwire.
BASELINE_PRECISION: PrecisionConfig = DEFAULT_PRECISION.with_overrides(
    experts_gate_up_in0_block_w=16,
    experts_down_in0_block_w=12,
)


class Candidate:
    """One sweep row: an id, a delta from the stage-06 baseline, and why it is here."""

    def __init__(self, cid: str, group: str, why: str, **overrides):
        self.cid = cid
        self.group = group
        self.why = why
        self.overrides = overrides
        self.config: PrecisionConfig = BASELINE_PRECISION.with_overrides(**overrides)

    @property
    def delta(self) -> str:
        """Short human-readable policy string for the CSV."""
        if not self.overrides:
            return "default (shipped stage-06 policy)"
        return ", ".join(f"{k}={v}" for k, v in sorted(self.overrides.items()))

    @property
    def dtype_policy(self) -> str:
        d = self.config.to_dict()
        return ", ".join(
            f"{k}={d[k]}"
            for k in (
                "experts_gate_up_dtype",
                "experts_down_dtype",
                "attention_qkv_dtype",
                "attention_wo_dtype",
                "lm_head_dtype",
                "activation_dtype",
                "ccl_dtype",
                "kv_cache_dtype",
                "logits_dtype",
                "sampling_dtype",
                "router_dtype",
                "embedding_dtype",
                "norm_weight_dtype",
            )
        )

    @property
    def fidelity_policy(self) -> str:
        d = self.config.to_dict()
        return (
            ", ".join(
                f"{k}={d[k]}"
                for k in (
                    "experts_fidelity",
                    "attention_fidelity",
                    "router_window_fidelity",
                    "lm_head_fidelity",
                    "norm_fidelity",
                )
            )
            + f", experts_gate_up_in0_block_w={d['experts_gate_up_in0_block_w']}, experts_down_in0_block_w={d['experts_down_in0_block_w']}"
        )

    def __repr__(self):
        return f"<Candidate {self.cid}: {self.delta}>"


# --- the set ------------------------------------------------------------------
#
# ``bfloat4_b`` weights, ``bfloat8_b`` weights and ``LoFi`` are the only levers
# that move real bytes or real cycles here. Rows that would only move a few KB
# (norm weights) are represented once, for completeness of the picture, and are
# expected to be flat.

CANDIDATES: list[Candidate] = [
    # -- 0. the control -------------------------------------------------------
    Candidate(
        "R00_default",
        "baseline",
        "The shipped stage-06 policy, re-measured in this sweep's own session so "
        "every other row is compared against a number taken on the same tree and "
        "the same silicon rather than against a quoted one.",
    ),
    # -- 1. lm_head: the largest per-token non-expert weight read --------------
    Candidate(
        "R01_lmhead_bfp4",
        "lm_head",
        "The LM head is 2048x37984 per die and is read in full for every decode "
        "token; bfloat8_b -> bfloat4_b halves that read. Biggest single decode "
        "lever the sweep has. Fidelity left at the shipped HiFi2 so this row "
        "isolates the dtype.",
        lm_head_dtype="bfloat4_b",
    ),
    Candidate(
        "R02_lmhead_bfp4_lofi",
        "lm_head",
        "REQUIRED BFP4+LoFi pair for R01. Four mantissa bits leave HiFi2 nothing "
        "to resolve, so if stage 02's finding generalises this is free accuracy-wise "
        "and cheaper in cycles.",
        lm_head_dtype="bfloat4_b",
        lm_head_fidelity="LoFi",
    ),
    Candidate(
        "R03_lmhead_lofi",
        "lm_head",
        "Control for R02: LoFi at the *shipped* bfloat8_b. Separates 'LoFi is free' "
        "from 'bfp4 is free' in R02, exactly as R08 does for attention.",
        lm_head_fidelity="LoFi",
    ),
    # -- 2. attention projections ---------------------------------------------
    Candidate(
        "R04_qkv_bfp4",
        "attention",
        "qkv alone to bfloat4_b. Separate from wo because wo's output feeds the "
        "attention all-reduce directly and is the more sensitive of the two.",
        attention_qkv_dtype="bfloat4_b",
    ),
    Candidate(
        "R05_wo_bfp4",
        "attention",
        "wo alone to bfloat4_b, to price the sensitive half apart from qkv.",
        attention_wo_dtype="bfloat4_b",
    ),
    Candidate(
        "R06_attn_bfp4",
        "attention",
        "Both attention projections to bfloat4_b: the full attention-weight lever, " "48 layers of it.",
        attention_qkv_dtype="bfloat4_b",
        attention_wo_dtype="bfloat4_b",
    ),
    Candidate(
        "R07_attn_bfp4_lofi",
        "attention",
        "REQUIRED BFP4+LoFi pair for R06.",
        attention_qkv_dtype="bfloat4_b",
        attention_wo_dtype="bfloat4_b",
        attention_fidelity="LoFi",
    ),
    Candidate(
        "R08_attn_lofi",
        "attention",
        "Control row for R07 and the first non-None attention_fidelity the model "
        "has ever run. attention_fidelity defaults to None ('the op picks'), so "
        "R07 is uninterpretable without this: it is the only thing that says "
        "whether a delta belongs to the dtype or to leaving the op default.",
        attention_fidelity="LoFi",
    ),
    # -- 3. experts: block widths, and the reverse-direction price ------------
    Candidate(
        "R09_gateup_bw32",
        "experts",
        "Co-tuning check. EXPERT_IN0_BLOCK_W_GATE_UP's own comment records 16 "
        "winning at LoFi and 32 at HiFi4; this re-measures that on the shipped "
        "bfp4+LoFi at 48 layers rather than trusting the comment. 32 divides K "
        "(2048/32 = 64 tiles) so it will not be clamped.",
        experts_gate_up_in0_block_w=32,
    ),
    Candidate(
        "R10_gateup_bw8",
        "experts",
        "The other side of the gate_up width optimum, so R09/R00/R10 bracket it " "rather than testing one direction.",
        experts_gate_up_in0_block_w=8,
    ),
    Candidate(
        "R11_down_bw24",
        "experts",
        "down's width bracket, upper side. 24 divides down's K in tiles; 12 ships.",
        experts_down_in0_block_w=24,
    ),
    Candidate(
        "R12_down_bw6",
        "experts",
        "down's width bracket, lower side.",
        experts_down_in0_block_w=6,
    ),
    Candidate(
        "R13_experts_bfp8_cotuned",
        "experts",
        "Reverse direction: experts back up to bfloat8_b, WITH the block widths "
        "that dtype wants (32/24) rather than bfloat4_b's. This is the row that "
        "prices the shipped bfp4 expert choice -- it should buy top-1 and cost "
        "t/s/u, and if it did not, the shipped default would be wrong. Co-tuned "
        "because stage 02 proved sweeping dtype at an untuned width finds the "
        "wrong optimum.",
        experts_gate_up_dtype="bfloat8_b",
        experts_down_dtype="bfloat8_b",
        experts_gate_up_in0_block_w=32,
        experts_down_in0_block_w=24,
    ),
    Candidate(
        "R14_gateup_bfp8_only",
        "experts",
        "gate_up alone to bfloat8_b, down left at bfp4. The two expert groups are "
        "separate fields precisely so the sweep can price them apart; down feeds "
        "the residual directly and is the likelier of the two to want precision.",
        experts_gate_up_dtype="bfloat8_b",
        experts_gate_up_in0_block_w=32,
    ),
    Candidate(
        "R15_down_bfp8_only",
        "experts",
        "down alone to bfloat8_b -- the sensitive half -- at down's co-tuned width.",
        experts_down_dtype="bfloat8_b",
        experts_down_in0_block_w=24,
    ),
    # -- 4. activations and the wire ------------------------------------------
    Candidate(
        "R16_ccl_bfp8",
        "ccl",
        "Halve the reduce-scatter/all-gather payload, twice per layer for 48 "
        "layers. Costs a cast in and out; the question is whether the wire saving "
        "beats the cast at batch 1. NOTE: ccl_dtype allocates a second persistent "
        "CCL buffer set (the cache keys on dtype), so this value gets its own "
        "process -- which it does anyway, one runner invocation per row.",
        ccl_dtype="bfloat8_b",
    ),
    Candidate(
        "R17_act_bfp8",
        "activation",
        "Activations and the inter-layer residual to bfloat8_b. Small bytes at "
        "batch 1, but it changes the input dtype of every matmul in the model, so "
        "it is the row most likely to move accuracy for no speed.",
        activation_dtype="bfloat8_b",
    ),
    Candidate(
        "R18_act_ccl_bfp8",
        "activation",
        "Both together: with activations already bfloat8_b the CCL cast becomes a "
        "no-op (effective_ccl_dtype inherits), so this prices the wire saving "
        "without the cast that R16 pays.",
        activation_dtype="bfloat8_b",
        ccl_dtype="bfloat8_b",
    ),
    # -- 5. kv cache ----------------------------------------------------------
    Candidate(
        "R19_kv_bfp8",
        "kv_cache",
        "Halves KV bytes per token per layer per die (512 -> 256 B). At the gate's "
        "158-token prompt this is nearly invisible, but it is the one row that "
        "could move the CONTEXT CONTRACT -- and it moves it upward (more capacity "
        "per byte), so it cannot reduce advertised capability. Measured so the "
        "contract question is answered with a number.",
        kv_cache_dtype="bfloat8_b",
    ),
    # -- 6. the small stuff, for completeness of the picture -------------------
    Candidate(
        "R20_embed_bfp8",
        "embedding",
        "The embedding table is a gather, not a matmul: one row read per token. "
        "Expected flat on t/s/u and near-flat on accuracy. Here so the report can "
        "say the group was evaluated rather than assumed.",
        embedding_dtype="bfloat8_b",
    ),
    Candidate(
        "R21_norm_hifi2",
        "norm",
        "RMSNorm at HiFi2 instead of HiFi4. Norm weights are 4 KB; this is a "
        "fidelity-only row on a reduction, expected flat.",
        norm_fidelity="HiFi2",
    ),
    Candidate(
        "R22_logits_sampling_bfp8",
        "terminal",
        "The terminal path: logits and the sampler's input to bfloat8_b. Touches "
        "one 37984-wide tensor per token per die. Included because the goal "
        "enumerates logits/sampling dtype as part of the selected config, so it "
        "needs a measured row rather than an inherited default.",
        logits_dtype="bfloat8_b",
        sampling_dtype="bfloat8_b",
    ),
]

#: Rows added after the first pass, built from what the first pass showed.
#:
#: The gate_up block-width bracket came back monotonic upward -- 8 -> 41.33,
#: 16 (shipped) -> 42.34, 32 -> 42.94 t/s/u, at unchanged accuracy -- which
#: means the optimum may sit above the largest width the first pass tried. So
#: the stacked set walks gate_up to its ceiling and then measures the combined
#: gate_up + down winner rather than assuming the two gains add. Stage 02's
#: whole lesson was that these knobs interact.
#:
#: **The two ceilings are not symmetric, and that is arithmetic, not an
#: oversight.** ``_tuned_sparse_matmul_config`` needs ``in0_block_w`` to divide
#: K in tiles:
#:
#: * gate_up's K is ``hidden_size`` = 2048 = **64 tiles** -> legal widths
#:   1, 2, 4, 8, 16, 32, 64. The shipped 16 is a quarter of full-K, so there are
#:   two rungs above it (32, 64).
#: * down's K is ``moe_intermediate_size`` = 768 = **24 tiles** -> legal widths
#:   1, 2, 3, 4, 6, 8, 12, 24. ``R11_down_bw24`` is therefore **already down's
#:   full-K row**; there is no wider width to try and no ``down`` analogue of
#:   the bw64 row.
#:
#: At full-K there is no inner-dimension blocking left to remove, so a further
#: gain has nowhere structural to come from: a modest improvement at 64 over 32
#: is plausible, a large one would be more suspicious than reassuring and is
#: treated as a suspected measurement artifact until repeated.
STACKED: list[Candidate] = [
    Candidate(
        "R23_gateup_bw64",
        "stacked",
        "gate_up at full-K (64 tiles), the top of its legal bracket. The 8/16/32 "
        "bracket was monotonic upward so the optimum may be here -- but this is "
        "also the width most likely to fail quietly: it is 4x the shipped block "
        "and 2x the winner, so it is the row where L1 pressure would first bite. "
        "_tuned_sparse_matmul_config clamps only on divisibility (k_tiles % blk), "
        "never on L1 capacity, so it CANNOT silently rescue an oversized block -- "
        "which means a slowdown here is a real spill and not an invalid config. "
        "The resolved width is read back from fallback_audit regardless.",
        experts_gate_up_in0_block_w=64,
    ),
    Candidate(
        "R24_gateup32_down24",
        "stacked",
        "The combined block-width winner: gate_up 32 with down at its full-K 24. "
        "Measured rather than inferred -- stage 02 proved these interact, so the "
        "two individual gains are not assumed to add.",
        experts_gate_up_in0_block_w=32,
        experts_down_in0_block_w=24,
    ),
    Candidate(
        "R25_gateup64_down24",
        "stacked",
        "The same combination at gate_up's full-K, so the stacked row is measured "
        "at both of the top two gate_up widths rather than only at whichever won "
        "alone.",
        experts_gate_up_in0_block_w=64,
        experts_down_in0_block_w=24,
    ),
    # -- the orthogonal stack: the one eligible dtype row, on top of the widths --
    #
    # ``R06_attn_bfp4`` is the only *dtype* row that clears every clause of the
    # selection rule on its own: +0.45% decode, beyond the 0.368% band, at top-1
    # 0.980 -- exactly on the floor, so it clears it. It changes attention weight
    # dtypes; ``R25`` changes expert matmul block widths. The two touch disjoint
    # ops, so the stage's own argument -- gains do not add, stacked rows must be
    # measured -- applies here just as it did to the two width rows.
    Candidate(
        "R26_attn_bfp4_bw64_24",
        "stacked",
        "R06 composed with R25: qkv+wo at bfloat4_b on top of the two full-K expert "
        "block widths. R06 and R25 are orthogonal (attention projection weights vs "
        "expert matmul scheduling), so if the attention dtype gain survives on top "
        "of the width gain this is the fastest config the sweep can build. Measured "
        "rather than inferred, for the same reason R25 itself was measured rather "
        "than summed from R23 and R11. The verdict may turn on top-1: R06 sits "
        "exactly on the 0.980 floor and this reference resolves top-1 to +/-0.01, "
        "so one token decides whether this row is eligible at all.",
        attention_qkv_dtype="bfloat4_b",
        attention_wo_dtype="bfloat4_b",
        experts_gate_up_in0_block_w=64,
        experts_down_in0_block_w=24,
    ),
    Candidate(
        "R27_attn_bfp4_lofi_bw64_24",
        "stacked",
        "REQUIRED BFP4+LoFi pair for R26 -- the same obligation R07 discharges for "
        "R06, re-run at the selected block widths so the pair is measured in the "
        "regime the stacked row is judged in.",
        attention_qkv_dtype="bfloat4_b",
        attention_wo_dtype="bfloat4_b",
        attention_fidelity="LoFi",
        experts_gate_up_in0_block_w=64,
        experts_down_in0_block_w=24,
    ),
    Candidate(
        "R28_kv_bfp8_bw64_24",
        "stacked",
        "bfloat8_b KV on top of the selected block widths, measured after the "
        "prefill cache writer was taught to cast K/V to the cache dtype "
        "(tt/functional_decoder.match_cache_dtype). R19 answers 'what does bfp8 KV "
        "cost against the stage-06 baseline'; this row answers the question the "
        "context contract actually needs -- 'what does it cost against the config "
        "we ship'. Same accuracy regime, so the KV capacity lever can be priced "
        "against the shipped model rather than against a config nobody runs.",
        kv_cache_dtype="bfloat8_b",
        experts_gate_up_in0_block_w=64,
        experts_down_in0_block_w=24,
    ),
]


def _assert_baseline_is_stage06() -> None:
    """Fail loudly if ``BASELINE_PRECISION`` has drifted from the measured rows.

    ``configs/R00_default.json`` was written by ``sweep_runner`` at sweep time --
    before the selection moved ``DEFAULT_PRECISION`` -- and is therefore a
    byte-level record of the config the ``R00_default`` 48-layer row was
    actually measured at. If a later stage moves another default field without
    naming it above, every gain in ``sweep_results.json`` would quietly start
    being quoted against a baseline nobody measured. This catches that at
    import.
    """
    recorded = SWEEP_DIR / "configs" / "R00_default.json"
    if not recorded.exists():
        return
    want = json.loads(recorded.read_text())
    got = BASELINE_PRECISION.to_dict()
    drifted = {k: (want[k], got[k]) for k in want if want.get(k) != got.get(k)}
    assert not drifted, (
        "BASELINE_PRECISION no longer reproduces the config R00_default was measured at "
        f"(configs/R00_default.json). Drifted fields {{field: (measured, now)}}: {drifted}. "
        "Add the moved field to BASELINE_PRECISION, or re-run the whole sweep."
    )


_assert_baseline_is_stage06()


def stack_winners(name: str, why: str, **overrides) -> Candidate:
    c = Candidate(name, "stacked", why, **overrides)
    STACKED.append(c)
    return c


def by_id(cid: str) -> Candidate:
    for c in CANDIDATES + STACKED:
        if c.cid == cid:
            return c
    raise KeyError(cid)


if __name__ == "__main__":
    for c in CANDIDATES:
        print(f"{c.cid:28s} [{c.group:10s}] {c.delta}")
    print(f"\n{len(CANDIDATES)} candidates")

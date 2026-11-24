<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hy4-preview fused decode

The decode step of tencent/Hy4-preview on one Blackhole Galaxy (32 chips, 8x4 mesh) as programs built in Python
from kernel sources and launched with `ttnn.generic_op` (kernels are JIT-compiled; nothing here needs a host C++
change), replayed as one device trace. The ttnn-op decode in `../tt/` is the functional path and the comparison.

State on 2026-11-23: one decode step of all 78 layers runs on the device as one trace, token ids in and token ids
out, with nothing from the host in between; it is bit-equal from replay to replay; its tokens are the reference's on
the contexts checked (see "Accuracy"). The 21 full-indexer layers are chained into the step (the selection is made
on the chips in every pass), and `model/decode_model.py` generates from an empty context: it appends its own rows
and keys, so a conversation needs no prefill stack (a prompt is fed one token per pass). Not done: the prefill stack
in front of that form at 78 layers, one set of expert weights for both, the MLP half and the full-indexer layer at
16 rows.

## Design
1. Programs are non-persistent and run under fast dispatch inside one device trace, so they share the open device
   and the KV storage with the ttnn prefill stack.
2. All 32 chips work on the same layer; the users of a pass are rows of `[8, 32]` (or `[16, 32]`) row tiles, so
   every weight byte is read from DRAM once per pass.
3. The 4 fp32 iHC residual streams live whole on one root chip: pre-mix, post-mix and the norm are chip-local.
   What crosses chips is the sublayer's input (broadcast) and output (reduced).
4. Decode reads the prefill KV storage in place (packed rows of 704 bytes: 512 FP8 latent values, 4 fp32 scales,
   64 BF16 RoPE values; BFP8 index keys in 32-row blocks) and appends its own rows and keys to it.
5. Attention never moves rows between chips: each chip scans its own stripe of index keys, the chips agree on the
   global top-2048 by a merge of counts, each chip reads its own selected rows by id and attends in latent space,
   and the partial results merge across chips in a fixed order.
6. Routed experts are streamed from DRAM for the union of the rows' picks only, 1/32 of every expert on each chip.

## Pieces
| directory | what | state |
|---|---|---|
| `collective/`, `layer_common/` | cross-chip sum / gather / unicast inside a program; the transports of a layer | 32 chips |
| `ihc_site/` | iHC pre-mix, post-mix, norm on the root; also the head's readout and final norm | 32 chips |
| `projections/` | the attention's dense projections as streamed units, 8 and 16 rows | 32 chips |
| `sparse_mla/` | selected rows read by id from the packed storage, latent attention with the sink | 32 chips |
| `indexer_scan/`, `merges/` | index-key scan with a chip-local top-2048; the cross-chip cut | 32 chips, exact ids |
| `layer_attn/` | the attention half of a layer: two programs (P1, P2); latent and index-key append; the full-indexer layer as one traced unit; a shared-indexer layer at 16 rows per pass; a fixed merge order | 32 chips |
| `moe/`, `layer_mlp/` | router, union of picks, DRAM-streamed experts, shared expert, dense FFN; the MLP half; two expert layouts: 1/32 of every expert per chip (tp32, the default) and 8-chip groups (tp8g4: a mesh column owns 64 experts, each chip a 1/8 slice of each) | 32 chips at 8 rows |
| `fidelity/` | a multi-phase matmul for BFP weights (`hifi_mm.h`) and what each fidelity costs | measured |
| `kv_store/` | a paged KV store (8192-token frames, frame tables in L1, bank-local copies, LRU by tokens) | 32 chips, with the prefill stack |
| `model/` | the whole step: N layers as one trace, shared-indexer and full-indexer layers (`Stack.set_indexer`); the embedding gather and the LM head with a greedy merge as programs of this piece; `DecodeModel` (`reset_row`, `step`, `open` / `close`): generation from an empty context; a restore program and a start-up rule for sharing the device with the prefill stack | 78 layers |

## Measured (32 chips, traced replay, 8 rows unless said)
| | |
|---|---|
| one decode step, 78 layers, ids to ids (every layer in shared-indexer form) | 43.6 to 46.2 ms by context (mean expert union 33 to 49 per layer) |
| the same step replayed 200 times | bit-equal in residual streams, logits and ids |
| attention half of a shared-indexer layer | 275 us (281 us with the fixed merge order) |
| attention half of a full-indexer layer, selection on the chips | 395 us (the selection 110 us of it) |
| attention half of a shared-indexer layer at 16 rows / the 8-row step twice | 389 us / 550 us (with 8 live rows or fewer the 8-row programs are 27 to 54 us faster; both sets share one layout and alternate bit for bit) |
| MLP half at a union of 48 | 250 to 300 us |
| embedding gather / LM head with greedy merge | 20 us / 189 us inside the step |
| prefill pass by trace replay between decode replays (6 layers) | about 30 ms, then 30 us to restore the decode step's L1 state |

The step's time follows the expert union (about 2.2 us per expert per layer). With the 8 rows taken from different
contexts (mean union 48.5) the 78-layer step is 46.2 ms, measured, tokens 8 of 8; eight unrelated conversations
have a union near 53 by the reference router's picks (about 47 ms, estimated). These figures are of the step with
every layer in shared-indexer form; the step with the full-indexer layers chained in (`DecodeModel`) has not been
timed as a step of record (the selection was about 2.3 ms by the unit's own measurement). With the tp8g4 expert
layout the same shared-form step is 42.2 to 43.9 ms; `DecodeModel(mlp_split="tp8g4")` runs the full-indexer form
with it (6 layers on the board: the same 216 of 256 golden tokens as tp32; the MLP program's tightest core keeps
72,960 B of L1 free, against 275,712 B with tp32). A row's result does not depend on the other rows of its pass
(`DecodeModel`, 78 layers: each row alone equals the same row among the others in every bit).

## Accuracy
Against the HF reference with the RoPE pairing of `reference/attention.py` set to `interleaved` (see below),
78 layers, greedy tokens of the device against the reference's: 8 of 8 on three contexts of 1,000 to 3,000 tokens
(one above the top-2048), 29 of 32 on a 31-token chat prompt. A host control with exact arithmetic on the same
storage formats loses the same tokens: what is left is the routed experts' single BFP4 term (a second, BFP2 term
on layers 42 to 77 recovers it at +26 % expert bytes per pass; not built). The FP8 latent rows are stored truncated
toward zero (about 4 % low); a scalar folded into the attention weights (`AttnLayer(latent_scale=1.0430)`)
compensates.
That scalar is a stopgap: it removes the mean bias only, and it cannot stay once prefill and decode read the same
weights. Rows packed with rounding to nearest need none (the append here has that form, `append="nearest"`); the
scalar goes when the prefill pack rounds.

From an empty context (`DecodeModel`: its own rows, rounded to nearest, no scalar; the selection of every full-indexer
layer made on the chips): 29 of 32 on the same chat prompt, the three lost tokens the ones its host control loses
with the same expert format, and 48 of 48 row tokens behind a 1,022-token context it wrote itself.

### RoPE pairing of the reference
`modeling_hy_v4.py` rotates half-split pairs `(i, i+32)` of the checkpoint's 64 rope rows; vLLM's Hy4 implementation
rotates interleaved pairs `(2i, 2i+1)`, and the checkpoint's row norms pair up that way. With the half-split
pairing the reference fails simple probes beyond a few dozen tokens of context; with the interleaved one it passes
them at 2,500 tokens. `HY4_ROPE_PAIRING=half_split|interleaved` selects it in `reference/attention.py` and
`tt/frames.py` (default `interleaved`; `half_split` reads the goldens recorded before); `reference/hf_harness.py` applies the same
switch to the HF model and records the pairing in a golden's metadata.
Both on-disk weight caches carry the pairing in their path (`tt/model.py` `WeightCache`, the projections' stream
cache), so a cache of one pairing is never loaded under the other. The default becomes `interleaved` in the same
change as the prefill stack's conversion, so that the two never disagree on one branch.

## Notes on this tree
- Comments cite sections of `PLAN.md`: the pieces' development notes, which are not part of this tree.
- Paths under `generated/hy4_preview/` are where tests read converted weights and goldens and write measurements;
  `model/convert.py` writes the decode weights: the routed experts' streams, one directory per layer and expert
  layout.
- The expert files are a container with a format number. `moe/prepare.py`'s docstring is its description: the
  values and their rounding, the order inside a DRAM bank, which entries of a chip are routed experts, and the
  shared expert as two BFP4 terms behind them (with tp8g4: on the chips of one group, zeros on the others). Every
  directory carries the stamp of what it holds in `manifest.json`; the loaders compare it with the layout they
  are about to run before they read a file, and the conversion counts a layer as complete only with it.
- The serving loop (an OpenAI server over `DecodeModel`) and the MLP half's latency work (router and expert
  hand-over changes) are not in this tree.

## Running
Each piece has `tests/`. Compile-only on a simulated 32-chip Galaxy (no board; kernels are JIT-compiled and the
circular-buffer, L1 and kernel-size limits checked):

    TT_METAL_MOCK_CLUSTER_DESC_PATH=tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_galaxy.yaml \
        pytest models/experimental/hy4_preview/fused/<piece>/tests

On a board, run a piece's smallest case first, alone, with `TT_METAL_OPERATION_TIMEOUT_SECONDS` set (15 to 30 s):
a compute kernel that stalls is not recovered by that timeout, so new kernels go one chip first. Tests that read
the prefill KV layout need the Hy4 prefill adapter (`models/demos/deepseek_v3_d_p/tt/hy4`).

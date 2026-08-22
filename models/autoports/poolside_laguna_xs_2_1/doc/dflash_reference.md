<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-XS-2.1 DFlash reference contract

Status: CPU reference, hardware-qualified TT one-round primitive, and a
fail-closed, default-off vLLM served controller. The controller has device-free
contracts and a bounded P150x2 correctness exercise, but it is **not performance
qualified**: the first real-context gate regressed the qualified baseline TPOT.
It must not be promoted or enabled by default.

## Published geometry

| Field | Value |
|---|---:|
| Draft decoder layers | 5 |
| Hidden / dense intermediate | 2,048 / 8,192 |
| Query heads / KV heads | 64 / 8 |
| Head dimension | 128 |
| Fused QKV rows | 8,192 Q + 1,024 K + 1,024 V |
| Attention | causal sliding window, 512 tokens |
| Attention gate | one softplus scalar per query head |
| RoPE | full 128 dimensions, NeoX layout, theta 500,000 |
| Proposal block | 16 query rows: one known bonus token + 15 masks |
| Mask token ID | 12 |
| Vocabulary | 100,352, shared with the target |

The draft checkpoint contains neither `embed_tokens.weight` nor
`lm_head.weight`. Both are supplied by Laguna-XS-2.1. The mask therefore uses row
12 of the target embedding table; it is not a separate learned draft tensor.

## Target auxiliary hidden states

The DFlash config exposes the same five locations with two indexing conventions:

| Zero-based target decoder layer | Captured post-layer hidden-state ID |
|---:|---:|
| 1 | 2 |
| 13 | 14 |
| 25 | 26 |
| 33 | 34 |
| 39 | 40 |

The target must return the post-layer states in that order. The drafter applies a
different learned RMSNorm to each 2,048-wide slice, concatenates the five slices
to width 10,240, projects them through `fc.weight` to width 2,048, then applies
`hidden_norm`.

For context K/V precomputation, every draft layer independently:

1. applies its own input RMSNorm to the combined target state;
2. projects with its fused QKV matrix and retains only K and V;
3. applies per-head K RMSNorm and NeoX RoPE to K (not V); and
4. inserts those tensors at the target states' absolute positions.

This per-layer input norm is Laguna-specific and must not be replaced by one
shared context norm.

## Parallel proposal block

For a target prefix ending at absolute position `P`, a full proposal step builds:

```text
query offset       0        1        2       ...       15
input token     bonus      mask     mask      ...      mask
absolute pos      P+1      P+2      P+3       ...      P+16
sample row          -        0        1       ...       14
predicted pos        -      P+2      P+3       ...      P+16
```

The known target bonus token anchors the block. The 15 mask rows are evaluated in
one draft forward and each row predicts the token at its own position. Because
this checkpoint sets `causal=true`, a query at position `Q` sees keys through
`Q`, never later query rows, and only positions `Q-511` through `Q` are in its
512-token sliding window.

Each draft layer follows standard pre-norm residual ordering:

```text
x = x + o_proj(softplus(head_gate) * attention(q_norm(Q), k_norm(K), V))
x = x + down_proj(silu(gate_proj(norm(x))) * up_proj(norm(x)))
```

The executable reference preserves vLLM's fused-add rounding detail: RMSNorm
consumes the float32 residual sum, while the residual carried to the next block
is rounded to the activation dtype.

The final draft state receives `norm` and sampled rows are projected by the
target-owned LM head.

## Executable reference

[`tt/dflash_reference.py`](../tt/dflash_reference.py) provides:

- strict config and safetensors layout validation;
- selective real-weight loading (one layer for fast diagnostics or all five);
- auxiliary-state combination and per-layer context K/V projection;
- causal grouped-query sliding attention, gating, dense SwiGLU, and final norm;
- target-owned embedding/LM-head interfaces; and
- construction of the `1 + N` proposal block.

Checkpoint-backed tests use both a fast layer-0 fingerprint and a complete
five-layer anchor+15 proposal with deterministic BF16 context/query tensors:

```bash
models/autoports/poolside_laguna_xs_2_1/.venv/bin/python -m pytest -q \
  models/autoports/poolside_laguna_xs_2_1/tests/test_dflash_reference.py
```

Set `LAGUNA_DFLASH_SNAPSHOT` to test another local snapshot. Otherwise the loader
uses the published revision
`5c36361aab23c8ed3afbd079c10c426b677bc607` in the standard Hugging Face cache.

## TT core qualification

[`tt/dflash_tt.py`](../tt/dflash_tt.py) is an explicit, default-off one-round core.
Construction requires `enable_experimental=True` and a bounded `max_seq_len`.
It strictly validates and maps the published BF16 weights, splits fused QKV rows,
loads the five auxiliary norms plus `fc`, `hidden_norm`, and final `norm`, and
constructs dense TP draft layers through `MultichipDecoder`. Draft weight-cache
keys use `dflash`-namespaced identities (`Lshared_dflash_*` and
`Ldflash_L0_*`), so draft layer numbers cannot alias target layer caches.
Auxiliary fusion and the draft final norm explicitly reuse the same qualified
HiFi4/fp32-destination compute kernel as the draft layers; they do not inherit an
implicit TTNN operation default.

The normal target `prefill_layers` and `decode_layers` functions remain free of
DFlash conditions and capture work. Separate explicit APIs capture post-layer
states 1, 13, 25, 33, and 39 only after validating the exact full 40-layer
target. They flatten the five slices in checkpoint order and retain at most 511
committed rows. A request-scoped proposal cache appends adjacent decode captures,
drops older rows, owns five bounded local draft KV pairs, and rejects cross-
request, after-end, and after-close use.

One TT proposal round builds the target-owned bonus/mask embeddings, resets the
context prefix to the same fused target state at every draft layer, carries only
the query suffix through all five layers, applies the draft final norm, selects
the 15 mask rows, and calls the target's raw column-sharded LM head. The raw
projection API deliberately cannot apply the target final norm.

The P150x2 gate on physical chips 2,3 qualifies shared auxiliary fusion and one
BF16 draft-layer prefill (16 context rows plus 16 query rows) against the official
CPU reference:

| Check | PCC | Required |
|---|---:|---:|
| Auxiliary normalize/concat/FC/hidden norm | 0.99986392 | 0.995 |
| Layer-0 query tail, including final norm | 0.99925524 | 0.995 |

The first successful run populated the namespaced BF16 tensor caches and took
10.41 s in pytest (12.43 s tool wall time). A warm replay took 1.49 s in pytest
(1.36 s test call, 3.53 s tool wall time) with 159/159 JIT cache hits. These are
bringup wall times, not steady-state layer latency measurements.

The complete default-off one-round gate then loaded all five draft layers plus
the target's real embedding and LM head. HiFi4/fp32-destination projection is
required: the initial HiFi2/LoFi qualification produced only 12/15 matching
proposal argmaxes, so that configuration was rejected rather than weakening the
gate. The qualified BF16 path produced:

| Full-five check | Result | Required |
|---|---:|---:|
| Auxiliary transfer PCC | 1.00000000 | 1.0 |
| Fused target-context PCC | 0.99987245 | 0.999 |
| Fifteen sampled hidden rows PCC | 0.99585289 | 0.995 |
| Target-owned raw LM-head logits PCC | 0.99654967 | 0.995 |
| Proposal argmax equality | 15 / 15 | 15 / 15 |

The exact TT and reference proposal IDs were
`[268, 88, 16351, 2750, 152, 920, 341, 341, 120, 120, 120, 268, 341, 389, 72]`.
Three stable-shape warm rounds measured 9.830, 8.168, and 7.987 ms (median
8.168 ms) with the program cache fixed at 63 entries. The DRAM view reported
1,477.3 MiB used of 31,069.5 MiB on each P150. This isolated one-round latency
does not include target verification, acceptance, or scheduler overhead.

Run the CPU/static contracts normally; the hardware case skips unless explicitly
enabled:

```bash
PYTHONPATH="$REPO_ROOT" pytest -q \
  "$REPO_ROOT/models/autoports/poolside_laguna_xs_2_1/tests/test_dflash_tt.py" \
  "$REPO_ROOT/models/autoports/poolside_laguna_xs_2_1/tests/test_dflash_serving.py" \
  "$REPO_ROOT/models/autoports/poolside_laguna_xs_2_1/tests/test_dflash_reference.py"
```

Run the isolated layer-0 or full-five hardware gate from `/tmp`, leaving
`TT_METAL_HOME` unset so the installed TTNN runtime cannot mix with
checkout-local dispatch kernels:

```bash
cd /tmp
env -u TT_METAL_HOME -u TT_MESH_GRAPH_DESC_PATH -u MESH_DEVICE \
  TT_VISIBLE_DEVICES=2,3 \
  TT_LAGUNA_RUN_DFLASH_TT_HW=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$REPO_ROOT" \
  "$REPO_ROOT/models/autoports/poolside_laguna_xs_2_1/.venv/bin/python" \
  -m pytest -sv --timeout=1800 \
  "$REPO_ROOT/models/autoports/poolside_laguna_xs_2_1/tests/test_dflash_tt.py::test_full_five_layer_one_round_pcc_and_warm_latency_chips_2_3"
```

Running that hardware command from the repository working directory is invalid
with the current installed environment: mesh open attempts to compile the newer
checkout `cq_dispatch.cpp` against installed headers and fails because
`init_telemetry` is undefined. No DFlash weights are loaded in that failure mode.

## Default-off served controller

[`tt/dflash_serving.py`](../tt/dflash_serving.py) now owns the single-request
verify/accept lifecycle. `TT_LAGUNA_DFLASH=1` selects it in the vLLM bridge; the
launcher additionally requires an explicit experimental acknowledgement,
`p150x2`, `LAGUNA_MAX_NUM_SEQS=1`, `TT_LAGUNA_PREFIX_CACHE=0`,
`TT_LAGUNA_HYBRID_KV=0`, and no ngram speculative mode. Non-greedy requests fail
closed at runtime. With the flag absent, the normal target prefill/decode methods
and decode trace are unchanged.

For a round beginning with known target token `b` at position `P`, the target
verifies the contiguous input `[b, d0, ..., d14]`. If exactly `m` draft tokens
match the target-greedy rows, the controller emits
`[d0, ..., d(m-1), target_greedy[m]]` one token per vLLM call. It appends only the
executed authoritative auxiliary rows `[b, d0, ..., d(m-1)]` to the rolling
511-row capture. The final target correction/bonus has not executed yet and is
the next round's known input. Rejected look-ahead target KV is overwritten at
the next contiguous verification start.

vLLM owns only the 64-token KV block containing the current input. A fixed
16-row target verify is therefore safe only when `P % 64 <= 48`. At residues
49–63, an empty-buffer controller executes one normal target row while capturing
its auxiliary state, then retries DFlash after the next block is allocated.
Already-verified buffered tokens continue to drain through the boundary. A full
round is allowed when `P + 16 == max_model_len` and rejected when it exceeds that
bound.

Device-free tests cover default-off policy, prefix/hybrid/non-greedy/B>1
failures, streamed prefill tail retention, request reset/close behavior, full and
partial acceptance, rejected-row rollback, residues 48/49/63, context exact-fit,
and a 700-token target-oracle equivalence stream. The complete five-layer CPU
contract continues to load the published checkpoint.

### Served P150x2 result and disposition

The bounded chips-2,3 gate used a real 40-layer target prefill, target-owned
embedding and raw LM head, one 16-row target verification, an independent
sequential-B1 verification, the served controller, and the residue-49 one-row
fallback. Its draft-accuracy contract is deliberately exact:

- every official raw-BF16 row with one maximum must select that exact token;
- when multiple official raw-BF16 logits are exactly equal to the maximum, the
  TT token must be a member of that complete exact maximum set; and
- target verification IDs and committed output tokens remain literally exact.

The earlier deterministic, non-tied hardware vector remains a separate literal
15/15 gate. For the real target context, 14 rows matched literal CPU `argmax`.
The remaining row was not a numerical preference in the official result: tokens
366 and 378 were exactly tied at BF16 logit 8.0. CPU `argmax` chose the lower
index 366, while TT chose 378 with logits 8.0625 and 7.96875. Token 378 is in the
official exact maximum set, so the draft contract passes without weakening any
non-tied row. The full diagnostic reported:

| Served correctness check | Result | Required |
|---|---:|---:|
| Fused real target-context PCC | 0.99999636 | 0.999 |
| Fifteen sampled hidden rows PCC | 0.99966037 | 0.995 |
| Draft raw LM-head logits PCC | 0.99972272 | 0.995 |
| Non-tied draft IDs | exact | exact |
| Tied draft selection | 378 in {366, 378} | member of exact maximum set |
| Batched-vs-sequential target auxiliary PCC | 0.99999988 | 0.995 |
| Batched-vs-sequential target-logit PCC | 1.00000000 | 0.995 |
| Target verification IDs | 16 / 16 | 16 / 16 |
| Controller committed output | `[1158, 672]` | literal equality |
| Residue-49 fallback auxiliary PCC / token | 1.00000000 / 1787 exact | 0.995 / exact |

Layer-output diagnostics showed the first TT/CPU numerical divergence in draft
layer 0 and gradual accumulation across the five layers. All-row PCC was
0.99996883, 0.99990618, 0.99978733, 0.99971479, and 0.99966586. The tied row's
PCC was 0.99997777, 0.99993199, 0.99987149, 0.99969751, and 0.99953252. This is
why the gate represents exact BF16 ties as a set rather than depending on one
backend's arbitrary ordering of equal maxima.

Correctness did not establish a performance win. This artificial context
accepted one draft and therefore committed only two output tokens. Three warm
full rounds were 189.654, 186.403, and 188.880 ms (median 188.880 ms), or at
least **94.440 ms per committed token before accounting for separate draft
cost**, versus the approximately 50.5 ms qualified baseline TPOT. This is a
clear no-regression failure. Residue-49 fallback rounds were 91.129, 89.366, and
93.508 ms (median 91.129 ms). Program-cache counts stayed fixed at 411 and 475;
DRAM was 11,607.1 MiB of 29,543.6 MiB. The controller remains experimental,
fail-closed, and default-off.

Remaining qualification work:

- qualify longer absolute positions and the 511-row window boundary in hardware;
- reduce the eager 16-row target-verification cost below baseline TPOT and then
  re-run a representative coding-workload acceptance/TPOT/TTFT gate;
- measure acceptance length and target calls per output token across a real
  workload; one artificial round is not served throughput qualification;
- optimize/capture the served target verify only after correctness is locked;
- qualify prefix-cache interaction separately before allowing the two modes to
  coexist.

# Functional Decoder — `Qwen/Qwen3.6-35B-A3B`

Stage 1/11 of the repo-local TTNN autoport pipeline. Deliverable:
`models/autoports/qwen_qwen3_6_35b_a3b/tt/functional_decoder.py` — a functionally complete
TTNN implementation of the HF `Qwen3_5MoeDecoderLayer`, covering **both** decoder layer
kinds, validated against HF on a single 1x1 Blackhole mesh.

* Hardware: 1 x Blackhole `p300c` chip (grid 11x10, 31.5 GiB usable DRAM), 1x1 mesh.
* HF reference: `transformers` 5.10.2 `Qwen3_5MoeDecoderLayer`, **float32**.
* Acceptance bar: **PCC >= 0.995** (skill default; no model-specific exception needed).
* Worst PCC in the main suite: **0.9999450** (§3.1). Worst anywhere including the 262144-token
  advertised-context cases: **0.9997685** — the long-context decode-SDPA anomaly was root-caused
  to one `SDPAProgramConfig` field and **fixed**, not waived (§3.8).
* Context: full HF-advertised **262144**, no capability reduction (§4).
* Warmed prefill and **traced** warmed decode measured with `tt-perf-report` (§5).
* Watcher-clean, determinism and runtime-fallback evidence: §3.6, §3.7, §3.9.

---

## 1. What the layer is

`config.json -> text_config` gives 40 decoder layers whose `layer_types` repeat
`[linear, linear, linear, full]` (`full_attention_interval = 4`):

| kind | count | layer indices | token mixer | per-sequence state |
|---|---|---|---|---|
| `linear_attention` | 30 | `(i+1) % 4 != 0` | `Qwen3_5MoeGatedDeltaNet` | conv state (3 taps x 16384) + recurrent state `[32,128,128]` |
| `full_attention` | 10 | 3, 7, ... 39 | `Qwen3_5MoeAttention` | paged K/V cache |

Everything after the mixer is shared and identical in both kinds: RMSNorm (`1 + w`) around a
256-expert / top-8 MoE with a sigmoid-gated shared expert. `tests/test_functional_decoder.py::test_layer_kinds_cover_the_whole_model`
pins that layer 0 and layer 3 are the only two distinct kinds, so testing those two indices
is complete coverage.

Full architecture notes (projection layouts, the head-interleaved `q_proj`/gate split, partial
RoPE, gate/decay math, MoE weight layout) are in `work_log.md` §2.

## 2. Public contract

```python
FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, **cfg)
    # state_dict keys = HF layer names with "model.language_model.layers.<i>." stripped.
    # All torch work (transposes, q/gate de-interleave, q/k head duplication, conv-tap split,
    # `1 + w` norm folding, RoPE tables, cache allocation) happens here.

prefill_forward(x, *, user_id=0, page_table=None, start_pos=0) -> ttnn.Tensor
    x           [1, 1, seq_len, 2048] TILE/DRAM. ANY seq_len >= 1 with
                start_pos + seq_len <= supported_context. Non-aligned lengths are padded,
                masked and sliced back internally.
    user_id     sequence slot: page_table row (full attention) / state slot (linear).
    page_table  int32 [max_batch, supported_context // 64] ROW_MAJOR. Required for
                full_attention, ignored for linear_attention.
    start_pos   absolute position of x[..., 0, :]; multiple of PREFILL_ALIGN (128) — an op
                contract, derived in §9. Lets a caller stream a long prompt across several
                calls (KV cache, conv state and recurrent state all carry over). Any single
                prefill of any seq_len needs start_pos = 0, so this never limits prompt length.
    returns     [1, 1, seq_len, 2048]

decode_forward(x, *, current_pos=None, page_table=None) -> ttnn.Tensor
    x            [1, 1, max_batch, 2048] TILE/DRAM, one token per slot.
    current_pos  int32 [max_batch] DEVICE tensor. Paged-cache write index, SDPA cur_pos and
                 (via an on-device typecast) the RoPE table lookup. -1 marks a slot inactive:
                 its attention is skipped and its paged K/V is left untouched. Required for
                 full_attention; linear_attention ignores it (its recurrence carries no
                 position), so it may be omitted for those layers.
                 Valid range is -1 or 0 .. supported_context-1. The upper bound is the
                 caller's responsibility: the value lives in device memory, so checking it
                 here would need a host read, which the runtime path forbids (§3.7). A value
                 >= supported_context indexes past the page-table row and the RoPE table.
    page_table   same tensor as prefill.
    returns      [1, 1, max_batch, 2048]
```

**One sequence per prefill call.** That matches the per-request prefill vLLM and the
downstream full-model stage issue. Batch >1 prefill = one call per `user_id`; each lands in
its own cache slot / page-table row and stays independent
(`test_prefill_per_user_slots`, 32 slots). Decode is fully batched.

**Per-slot state lifecycle.** A prefill with `start_pos = 0` starts a new sequence in that
slot, and for `linear_attention` it **zeroes that slot's conv and recurrent state first**. This
matters because the two kinds differ: full attention self-heals (the prefill rewrites every paged
block it will later read), while the DeltaNet state is a running summary that would otherwise
continue the previous occupant's sequence. Two consequences for a serving caller:

* Reusing a slot for a new request needs no explicit reset — just prefill it from `start_pos = 0`
  (`test_prefill_resets_linear_state_for_new_sequence`).
* `linear_attention` decode ignores `current_pos`, including the `-1` inactive marker, so an
  inactive slot's state still advances from whatever token sits in its row. That junk stays inside
  that slot (the recurrence is per-slot), its output row is discarded by the caller, and the next
  `start_pos = 0` prefill clears it — so it is self-healing rather than corrupting, but a caller
  must not read an inactive slot's output.

**Decode batch is fixed at `max_batch_size`.** The linear-attention conv and recurrent state
buffers are updated whole-tensor and in place, which is what lets one captured trace be
replayed at any position. Partial batches use `current_pos = -1` on unused full-attention
slots.

### Runtime knobs (`DecoderConfig`)

| field | default | note |
|---|---|---|
| `supported_context` | 262144 | HF-advertised; see `doc/context_contract.json` |
| `max_batch_size` | 1 | decode batch, and number of cache/state slots |
| `block_size` | 64 | paged KV block |
| `prefill_chunk_size` | 2048 | internal prefill chunking (multiple of 128) |
| `delta_chunk_size` | 64 | gated-delta-rule chunk (matches HF's default) |
| `moe_prefill_chunk_tokens` | 512 | bounds the dense-over-experts MoE intermediates |
| `sdpa_chunk` | 128 | prefill SDPA q/k chunk. Also **is** `PREFILL_ALIGN`: it is the modulus `start_pos` must satisfy, because chunked SDPA divides the offset by `q_chunk_size` (§9) |
| `activation_dtype` / `weight_dtype` / `kv_cache_dtype` | `bfloat16` | |
| `delta_dtype` | `float32` | HF pins the SSM state to fp32 (`mamba_ssm_dtype`) |
| `decode_sdpa_max_cores_per_head` | 1 | cores the decode SDPA splits each KV head's keys across. **The variable behind long-context decode accuracy** (§3.8); 1 is the only value correct at every `cur_pos`. `None` = pass no program config, which lets the op use all 55/head and costs accuracy at long context |
| `decode_sdpa_program_config` | `None` | escape hatch: a full `ttnn.SDPAProgramConfig` used verbatim, overriding the row above. For sweeps in later stages |

## 3. Correctness evidence

Commands (all from `$TT_METAL_HOME`):

```bash
# what logs/test_suite_main.log is: both files in one invocation (97 items)
python -m pytest \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_reference_math.py \
  models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
  -q --no-header -p no:cacheprovider

# either half alone also works: test_reference_math.py is CPU-only (no device, no checkpoint),
# test_functional_decoder.py is the device suite (synthetic weights from real-checkpoint stats)

# advertised-context evidence (slow; one process per case, see tests/run_long_context.sh)
pytest models/autoports/qwen_qwen3_6_35b_a3b/tests/test_long_context.py -m slow
```

Logs: `logs/test_suite_main.log` (the CPU algebra tests + the whole device suite, including the
real-weight cases — **99 passed, 0 failed**: 26 CPU algebra + 73 device cases),
`logs/long_*.log` (one per advertised-context case, **5 passed**), `logs/diag_*.txt` (the §3.8
diagnostics), `watcher/pytest.log` (**8 passed**). The suite is fast on a warm kernel cache
because `conftest.py` caches layer pairs per (kind, batch, context, weights) for the session:
building one costs ~1.5 GiB of weight conversion, and the 71 device cases share about a dozen.
Machine-readable PCC rows: `pcc.jsonl` (276), `pcc_real_weights.jsonl` (6),
`long_context.jsonl` (7). Every number quoted below is re-derivable from those files.

### 3.1 HF-vs-TTNN PCC, synthetic weights (real shapes)

Worst case per test family, re-derived from `pcc.jsonl` (the file holds exactly one whole-file
run: `harness.reset_log` clears it once per session, and a *filtered* session writes
`pcc_partial.jsonl` instead so a subset run can neither mix rows in nor delete them):

| family | n | worst PCC | worst case |
|---|---|---|---|
| `prefill[linear]` | 13 | 0.9999737 | seq=4096 |
| `prefill[full]` | 15 | 0.9999792 | seq=1 |
| `prefill-cont[linear]` | 2 | 0.9999812 | start=512 |
| `prefill-cont[full]` | 2 | 0.9999903 | start=0 |
| `prefill-fresh-slot[linear]` (reused slot, no reset) | 1 | 0.9999824 | seq=640 |
| `prefill-fresh-slot[full]` (reused slot, no reset) | 1 | 0.9999910 | seq=640 |
| `prefill-slot[linear]` (32 slots) | 32 | 0.9999901 | user=26 |
| `prefill-slot[full]` (32 slots) | 32 | 0.9999907 | user=10 |
| `decode[linear]` | 81 | 0.9999857 | pos=129, batch=1 |
| `decode[full]` | 82 | 0.9999849 | pos=149, batch=1 |
| `decode-ragged` (per-slot positions) | 4 | 0.9999933 | user=0, pos=128 |
| `decode-active-slot` (with `current_pos=-1` peers) | 2 | 0.9999930 | user=2 |
| `decode-seeded-state` (random DeltaNet state) | 1 | 0.9999720 | |
| `traced-decode[linear]` | 2 | 0.9999891 | pos=257, batch=8 |
| `traced-decode[full]` | 2 | 0.9999940 | pos=257, batch=8 |
| `paged-kv` cache contents | 2 | 0.9999835 | keys |
| `linear` conv/recurrent state | 2 | 0.9999450 | recurrent_state |

**Overall minimum: 0.9999450** (`linear recurrent_state`) vs a 0.995 bar — a margin of about
3 orders of magnitude in `1 - PCC`.

### 3.2 Real-weight PCC (`pcc_real_weights.jsonl`)

| case | PCC | rel-RMS |
|---|---|---|
| `prefill[linear] seq=1024` | 0.9999435 | 1.06% |
| `decode[linear] pos=512/513, batch=2` | 0.9999834 / 0.9999946 | 0.58% / 0.34% |
| `prefill[full] seq=1024` | 0.9999796 | 0.64% |
| `decode[full] pos=512/513, batch=2` | 0.9999856 / 0.9999908 | 0.54% / 0.44% |

Context for the rel-RMS: **HF's own bf16-vs-fp32 divergence on the same layers and input is
0.32% rel-RMS (PCC 0.999995)**, measured with real weights for both layer kinds
(`work_log.md` §5.2). The TTNN bf16 path lands at 0.34-1.06% rel-RMS, i.e. 1.1-3.4x that, which is
the expected cost of bf16 activations plus bf16 sparse matmuls; PCC stays ~2 orders of magnitude
inside the bar. Both
layer kinds load the **real** HF layer state dict through `from_state_dict`, so these runs also
pin the state-dict key and shape contract.

### 3.3 Sequence-length coverage

`test_prefill_pcc_sequence_lengths` covers, for **both** kinds:
1, 32 (one tile), 33 (past a tile), 64 (one paged block / one delta chunk), 65 (past a
block), 128 (`PREFILL_ALIGN`), 129 (past it), 1024, 1025, 2048 (internal prefill chunk),
2049 (forces chunk continuation), 3000 (divides nothing in play), 4096.

`test_decode_after_non_aligned_prefill` covers decode continuing from prefill lengths
65 / 129 / 320 / 1000 — the case that caught a real bug (see §7, item 2).

`test_long_context.py` runs the largest feasible length, `262143` — deliberately
non-aligned, so the longest test also exercises the pad/mask/slice path.

### 3.4 Paged cache and state behaviour

* Page tables are always a **random permutation** of the physical block pool
  (`make_page_table`), so an implementation that assumed identity mapping fails.
* `test_paged_kv_cache_contents_match_hf` compares the K/V actually written through the page
  table against HF's cache, and asserts the other three slots are untouched.
* `test_page_table_permutation_is_respected` re-runs decode with a permuted page table and
  asserts the output changes — a guard against silently treating the cache as contiguous.
* `test_linear_state_matches_hf` compares the conv and recurrent state against HF's, in HF's
  own layout (via the documented `tt_conv_state_to_hf` translation).
* `test_decode_from_seeded_random_linear_state` decodes from a random, non-prefill-derived
  state so every term of the recurrence contributes.
* `test_decode_ragged_current_positions` prefills four slots to 128/256/384/640 and decodes
  all four in one batched call, proving per-slot `current_pos` (KV write index, SDPA context
  length and RoPE lookup) is honoured.
* `test_decode_skips_inactive_slots_with_negative_position` decodes a batch of four with
  `current_pos = -1` on two slots: the active slots still match HF, the inactive slots' paged K/V
  is bit-unchanged (so the cache update does not scribble through a negative index) and the
  inactive rows are finite (so the clamped RoPE lookup stays in bounds).
* `test_prefill_resets_linear_state_for_new_sequence` dirties a slot with one sequence, then
  prefills a different sequence into the same slot at `start_pos = 0` **without** calling
  `reset_state()`, and compares against a fresh HF layer. Deliberately bypasses
  `prefill_and_compare`, which resets for `start_pos == 0` and hid this.
* `test_prefill_covering_whole_context_does_not_free_weights` prefills exactly
  `supported_context` tokens in one chunk at `supported_context <= prefill_chunk_size` and then
  runs two more forwards on the same layer — the regression test for §7 item 5.
* `test_decode_forward_rejects_out_of_contract_inputs` covers the documented raise paths:
  missing `page_table` / `current_pos`, `batch != max_batch_size`, misaligned `start_pos`,
  `start_pos + seq_len` past the context, and `user_id >= max_batch_size`.

### 3.5 Traced decode

`test_traced_decode_pcc` measures PCC **from trace replay output**, not from an uncaptured
forward: prefill → snapshot state → capture (which runs a warmup forward and perturbs the
state) → rewind → replay → compare. Two positions are replayed through the **same** captured
trace, proving `current_pos` is read from device memory rather than baked in.

`test_traced_decode_matches_eager` asserts trace replay is **bit-identical** to eager decode
from the same state (max abs diff exactly 0.0) for both layer kinds.

Two trace-safety properties this establishes that are not obvious:

* The MoE runs **inside** the trace with `nnz=None`, i.e. `sparse_matmul` resolves the
  non-zero count from the sparsity tensor at *replay* time. The two replayed steps use
  different tokens (different seeds), so they route to different experts and therefore
  produce different sparsity patterns — and both match HF. A capture-time-baked count would
  have deadlocked or mis-computed on the second replay.
* The linear-attention conv taps are shifted with in-place `ttnn.copy` and the recurrent state
  is updated with `output_tensor=state`, so all persistent buffers keep their addresses across
  replays. That is why `decode_forward` requires `batch == max_batch_size`.

### 3.6 Determinism

`test_prefill_determinism` / `test_decode_determinism`: three repeats of the same input from
the same state produce **bit-identical** output (max abs diff exactly 0.0), both kinds.

### 3.7 Runtime fallback audit

`test_no_runtime_host_fallback` is a static audit of the **whole runtime call graph** (26
methods reachable from `prefill_forward` / `decode_forward`, plus the module-level helpers)
for `torch`, `ttnn.from_torch`, `ttnn.to_torch`, `.cpu()`, `.item()`, `.tolist()`,
`ttnn.to_device`, `ttnn.from_device`, `copy_host_to_device_tensor`. Docstrings and comments
are stripped first so prose about torch does not trip it.

Two self-checks keep the audit from silently shrinking, because a hand-maintained list of what to
audit is the obvious failure mode: the test asserts the method list covers every `self._*` helper
the entry points call (this caught two newly added helpers during development), and the
module-level helper set is **derived from the module** rather than listed, with an explicit
setup-only exemption set — the round-2 review found `_view` missing from the old hand-written
list.

`test_no_host_ops_during_forward` is the dynamic counterpart: it monkeypatches
`ttnn.from_torch` / `to_torch` / `copy_host_to_device_tensor` / `to_device` / `from_device`
to raise, then runs a real prefill and a real decode for both kinds.

Setup-time conversion (`from_state_dict`, `_prepare_weights`, `_build_rope_tables`,
`_init_state`, `_to_device`) is exempt: that *is* the weight-loading boundary. Test-harness
input construction and PCC comparison are the other explicit boundaries.

### 3.8 Investigated anomaly: full-attention decode at 262144-token context

`test_longest_decode_context[full]` was the one number in the stage materially below the rest.
Three diagnostics, all kept in `tests/` with their output in `logs/`. The short version: the
variable is **how many cores the decode SDPA splits each KV head's keys across**, an
`SDPAProgramConfig` field, and selecting it explicitly fixes most of the gap.

**Step 1 — localise it (`diag_long_decode.py` -> `logs/diag_long_decode.txt`).** Sweep the decode
position over one cache and isolate the **attention branch** (the only position-dependent part of
the layer) by driving the TTNN and HF mixers directly. This is what said the loss is inside
attention rather than in the MoE, the residual or the cache read, and — via the control column —
that it is not operand precision:

| decode position (context) | layer PCC | TTNN attn vs fp32 HF | TTNN attn vs **bf16-operand HF control** | **control vs fp32 HF** | attn RMS |
|---|---|---|---|---|---|
| 1023 (1024) | 0.9999957 | 0.9997080 | 0.9997110 | 0.9999980 | 0.03327 |
| 8191 (8192) | 0.9999957 | 0.9996131 | 0.9996123 | 0.9999977 | 0.01292 |
| 32767 (32768) | 0.9999956 | 0.9986712 | 0.9986665 | 0.9999973 | 0.00639 |
| 131071 (131072) | 0.9999880 | 0.9864572 | 0.9864611 | 0.9999973 | 0.00306 |
| 262143 (262144) | 0.9999551 | 0.9702506 | 0.9702557 | 0.9999976 | 0.00221 |

The last column is a control: HF's own attention math with q/k/v rounded to bf16 and exact
accumulation matches fp32 to 2.4e-6 at **every** context. So **operand precision is not the
cause** — the device diverges from an exact bf16 reference (column 4) by exactly as much as from
fp32 (column 3), to 5 decimal places, at every context.

These are the numbers *after* the fix in step 3. The attention branch still loses accuracy as the
key count grows — 0.9997 at 1024 keys to 0.9703 at 262144 — but that residual is now a plain
accumulation floor rather than a decomposition bug: at 1 core per head there is no cross-core
reduction left, so what remains is bf16 accumulation over 262144 keys in a single pass, and the
control shows an exact-bf16 reference does not lose it. It dilutes to **0.9999551** at the layer
level here because the attention branch is one of two summed contributions and the residual
dominates. For reference, the same columns before the fix read 0.9990 / 0.9924 / 0.9188 / **0.7687**
at 8192 / 32768 / 131072 / 262144 keys.

The `layer PCC` column is not directly comparable to the headline number: this diagnostic seeds
its own cache and token (so it can sweep five positions over one cache in one run) while
`test_longest_decode_context[full]` decodes off a cache built by a real 262143-token prefill.
Different inputs, so different values. What the sweep is for is the *shape* of the curve and the
control column, both of which are input-independent.

(The file's final `token*8` row is a residual-dilution sensitivity check only, and it is the
cleanest demonstration that the layer number is dilution rather than accuracy: `input_layernorm` is
an RMS norm, so scaling the input token leaves the attention branch **bit-identical** — all three
attention columns repeat to the last digit — while the layer PCC moves 0.9999551 -> 0.9999970
purely because the residual got 8x larger. It is *not* a softmax-peaking control.)

**Step 2 — find the variable (`diag_sdpa_decode.py` -> `logs/diag_sdpa_decode.txt`).** Drive
`paged_scaled_dot_product_attention_decode` alone (no projections, RoPE, gate, o_proj or MoE) with
random K/V over a paged cache. Passing no program config changes **four** things at once, so all
four are swept independently: the core grid, `k_chunk_size`, `exp_approx_mode`, and — the one that
is easy to miss — `max_cores_per_head_batch`.

That last field is a *struct default* (`ttnn/cpp/ttnn/operations/transformer/sdpa_config.hpp:18`,
value 16) and the factory reads
`program_config.has_value() ? program_config->max_cores_per_head_batch : num_cores_available`
(`.../sdpa_decode/device/sdpa_decode_program_factory.cpp:192-193`). On this 11x10 part with batch 1
and 2 KV heads that means **no config gives 55 cores per head, while any explicit config defaults
to 16** — so "I passed a program config" and "the op runs on a different number of cores" are the
same event unless the field is set deliberately.

Sweeping it (device grid, `k_chunk_size=0`, exact `exp`) gives a clean ordering — the numbers are
op-level PCC against fp32 `scaled_dot_product_attention`:

| cores/head | 257 | 1024 | 4096 | 32768 | 131072 | 262143 |
|---|---|---|---|---|---|---|
| **1** | **0.9998** | **0.9998** | **0.9998** | **0.9980** | **0.9839** | **0.9704** |
| 2 | 0.1528 | 0.9998 | 0.9998 | 0.9995 | 0.9942 | 0.9891 |
| 4 | 0.1621 | 0.9998 | 0.9998 | 0.9997 | 0.9988 | 0.9973 |
| 8 | 0.1621 | 0.3721 | 0.9998 | 0.9998 | 0.9996 | 0.9991 |
| 16 (explicit default) | 0.1621 | 0.3721 | 0.9998 | 0.9998 | 0.9997 | 0.9996 |
| 32 | 0.1621 | 0.3721 | 0.1637 | 0.9998 | 0.9998 | 0.9997 |
| 55 (explicit) | 0.6959 | 0.5607 | 0.2024 | 0.0997 | 0.0638 | 0.0534 |
| 55 (**no config**) | 0.9998 | 0.9998 | 0.9995 | 0.9875 | 0.9170 | 0.7664 |

Three things fall out, and one open question:

* **More cores per head is more accurate at long context and *breaks* at short context.** Every
  setting from 2 upwards is structurally wrong below some context, and the threshold rises with the
  core count (2-4 break at 257; 8-16 at 1024; 32 at 4096). **1 core per head is the only setting
  correct at every context.**
* **`exp_approx_mode` is bit-identically irrelevant** — the approx and exact rows match to the last
  digit at 16 and at 110 cores. (The in-source comment at
  `.../sdpa_decode/device/kernels/rt_args_common.hpp:104` warns the dynamic chunk path is
  PCC-sensitive; that is not what this is.)
* **Neither the grid nor the chunk policy is the variable.** `8x8` at 16 cores/head equals `11x10`
  at 16 cores/head exactly; `k_chunk_size` 0/128 at 16 cores/head are identical. `k_chunk_size=512`
  merely *moves* the breakage (fine at 257, broken at 1024 and 4096). The cleanest cross-check is
  that the small-grid rows kept from the earlier sweep agree: a `1x1` or `2x1` grid *derives*
  1 core/head, and those rows match the explicit `max_cores_per_head_batch=1` row to the last
  digit at every context — the grid only ever mattered through the cores/head it implies.
* **Open, and reported rather than explained:** an explicit config with `max_cores=110` derives the
  same 55 cores/head as no config at all — same grid, same chunk policy, same `exp` — yet scores
  0.0534 against 0.7664. Everything the factory reads from the config is identical between those
  two rows (grid, `max_cores_per_head`, `exp_approx_mode`; `sub_core_grids` is unset in both), so
  this stage cannot account for the difference from the source. Both settings are rejected anyway,
  so it does not affect the decision, but it is a concrete two-config reproducer worth an upstream
  issue — as is a device op silently returning PCC 0.16 at context 257.

**Step 3 — the on-model decision (`diag_decode_sdpa_onmodel.py` ->
`logs/diag_decode_sdpa_onmodel.txt`).** The op sweep uses random K/V, so the three candidates are
re-measured on the **whole decoder layer** against HF, off a real prefilled cache, at five
contexts. One layer is built per context and prefilled once, so all three settings decode from the
same cache and the comparison is same-input:

| context | A: no config (55/head) | B: explicit, default 16/head | C: explicit, **1 core/head** |
|---|---|---|---|
| 258 | 0.9999950 | **0.4791771** | 0.9999949 |
| 1024 | 0.9999954 | **0.0304429** | 0.9999953 |
| 4096 | 0.9999954 | 0.9999958 | 0.9999957 |
| 32768 | 0.9999714 | 0.9999960 | 0.9999958 |
| 262144 | 0.9985674 | 0.9999958 | **0.9997685** |
| **worst over contexts** | 0.9985674 | 0.0304429 | **0.9997685** |

**Step 4 — the cost, measured on the same op** (`logs/diag_sdpa_decode.txt`, 20 warmed calls at
262144 keys):

| setting | cores/head | op time | op PCC @262143 |
|---|---|---|---|
| A no config | 55 | **28.166 ms** | 0.7664 |
| B explicit default | 16 | **1.403 ms** | 0.9996 |
| — | 4 | 2.898 ms | 0.9973 |
| **C shipped** | **1** | **11.519 ms** | 0.9704 |

There is **no accuracy/latency trade-off to make**: 1 core/head is 2.4x *faster* than the
no-config path as well as more accurate. Using 55 of 55 cores per head is both the slowest and the
least accurate setting measured — consistent with a decomposition that spends everything on
cross-core reduction. (16 cores/head is 8x faster again, and unusable below ~4096 keys.)

**C ships** (`DecoderConfig.decode_sdpa_max_cores_per_head = 1`): correct at every context, 6.5x closer
to HF than A at the advertised context (`1 - PCC` 2.3e-4 vs 1.4e-3) while matching it to the 7th
decimal everywhere else, and 2.4x faster there.

**Where it actually matters: small batch.** The field is `max_cores_per_head_*batch*`, and the
factory divides by the batch, so the *old* default's cores/head depends on how many slots decode
together — `min(110, 110 * B * 2) / B / 2` on this part:

| decode batch | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| cores/head with **no** config | 55 | 27 | 13 | 6 | 3 | **1** |
| cores/head with the shipped config | 1 | 1 | 1 | 1 | 1 | 1 |

So at batch 32 the two settings are *identical*, which is why the batch-32 decode tests never caught
this and why the §5 perf table barely moved (50.15 -> 50.08 ms). The bug lived in **small-batch,
long-context decode** — one or a few long requests, which is exactly the serving case and exactly
what `test_longest_decode_context` (batch 1) measures. The shipped config makes the decomposition
batch-independent: 1 core per (slot, KV head), so 2 cores at batch 1 and 64 at batch 32, and no
cross-core reduction inside a head at any batch. Batch 32 accuracy is unaffected and separately
covered — the `decode[full]` family's 82 rows include batch 32 at worst 0.9999849. B is the most accurate at long context but destroys a 1024-token decode,
so it is unshippable at any single setting — which is what made A look forced before this sweep
existed. An earlier version of this section claimed "whether a program config is passed at all" was
the discriminator and handed `optimize` a per-position config selection; both were wrong, and the
`max_cores_per_head_batch` axis is why.

### 3.9 Watcher-clean run

Command (`tests/run_watcher.sh`, exact env in `work_log.md` §7):

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=.../doc/functional_decoder/watcher \
  pytest .../test_functional_decoder.py -k "<selector>"
```

Selection deliberately spans both layer kinds and every device-facing mechanism the stage owns:
traced decode capture+replay, chunked prefill continuation, ragged per-slot `current_pos`,
`current_pos = -1` slot skipping, the paged-cache write path and the linear-attention state.

Result: **8 passed, 0 failed** (`watcher/pytest.log`), watcher log 5404 lines / 365927 bytes raw
(`watcher/generated/watcher/watcher.log.gz`, 15.7 KB gzipped — gzipped so the artifact stays under
this repo's 500 KB committed-file limit regardless of run length; inspect with `zless`/`zgrep`),
and **no fatal, sanitize, out-of-bounds, stack/L1 overflow, invalid-NOC, CB-overrun or
watcher-assert findings** — the automated grep in `run_watcher.sh` writes any hit to
`watcher/watcher_hits.txt`, which is empty.

The log's content is the expected benign mix: 10 periodic core-status dumps (`Dump #` blocks)
holding 2640 `Device ...` core-state rows and 2641 `k_ids:` kernel-id lines, stack-usage
summaries, a legend and attach/detach lines — no diagnostic classes. The line count is a function
of how many 10-second polls the run spans, not of correctness, so it varies between runs of the
same selector depending on how much kernel compilation is cached.

Watcher was **not** combined with the device profiler — `tests/run_perf.sh` is the separate
profiling run and `test_perf.py` fails fast if `TT_METAL_WATCHER` is set.

## 4. Capability contract

Measured at the full advertised context (`long_context.jsonl`, one pytest process per case so
each of these — the largest allocations in the stage — starts from empty device DRAM):

| case | value | PCC vs HF | device wall |
|---|---|---|---|
| `longest-prefill[linear]` | seq_len **262143** (non-aligned), tail-128 compared | 0.9999742 | 43.9 s |
| `longest-prefill[full]` | seq_len **262143** (non-aligned), tail-128 compared | 0.9999891 | 48.8 s |
| `longest-prefill[linear]` carried conv state | after 262143 tokens | 0.9999698 | |
| `longest-prefill[linear]` carried recurrent state | after 262143 tokens | 0.9998960 | |
| `longest-decode[linear]` | position **262143** after a 262143-token prefill | 0.9999860 | 0.04 s |
| `longest-decode[full]` | position **262143** after a 262143-token prefill | 0.9997685 | 0.05 s |
| batch-32 full-context paged KV | 131072 blocks, 2 x 8 GiB, **allocated on device** | n/a | |

`longest-decode[full]` is still the lowest number in the table, and §3.8 explains why in full: it
is the decode SDPA op's parallel decomposition, reproduced with **no model code** by
`diag_sdpa_decode.py`, and it is *not* operand precision, *not* `exp_approx_mode`, *not* the grid
and *not* the chunk policy — it is `max_cores_per_head_batch`, i.e. how many cores the op splits
each KV head's keys across. This stage **fixed** it rather than waiving it: shipping 1 core/head
took the advertised-context decode from 0.9985674 to the value in the table above and made every
context correct, while also being 2.4x faster at that context. What remains available to
`optimize` is 0.9999958 from 16 cores/head, which is unusable below ~4096 keys and therefore needs
per-context trace bucketing.

See `../context_contract.json` for the machine-readable version (regenerate with
`python models/autoports/qwen_qwen3_6_35b_a3b/tests/write_context_contract.py`; the contract is
derived from `long_context.jsonl`, and `test_context_contract_file_is_consistent` re-checks the
relationship so it cannot go stale). Summary:

| claim | evidence | remaining risk |
|---|---|---|
| context 262144 (= HF `max_position_embeddings`), **no reduction** | `test_longest_prefill` prefills 262143 tokens in one call; `test_longest_decode_context` decodes position 262143 | none for a single layer; the 40-layer model is stage 5's problem |
| batch 32 at full context fits DRAM | `test_max_batch_full_context_capacity` allocates the real 2 x 8 GiB (16 GiB) paged cache on device | leaves ~15 GiB for weights + activations on this 31.5 GiB part (expert weights are 1.61 GB/layer in bf16: 256 x (1024x2048 + 2048x512) x 2) |
| both layer kinds | `test_layer_kinds_cover_the_whole_model` + every test parameterised over `["linear", "full"]` | none |
| any logical seq_len | 13 lengths per kind incl. 1/33/65/129/1025/2049/3000/262143 | none |
| continuation prefill (`start_pos > 0`) | `test_prefill_chunk_continuation` streams a 1024-token prompt as two 512-token calls and matches HF piece-by-piece off the carried KV cache / conv+recurrent state | `start_pos` must be a multiple of `sdpa_chunk` (128); §9 records the chunked-SDPA integer-division op contract behind it and the only lever. Never limits prompt length — a fresh request prefills any `seq_len` in one call |
| paged prefill + paged decode, permuted page tables | §3.4 | none |
| per-slot `current_pos` | `test_decode_ragged_current_positions` | none |
| carried state across prefill calls | `test_prefill_chunk_continuation` | none |
| traced decode | §3.5 | none |

There is **no mode switch** in this model (no sliding-window / full-attention split within a
kind, no windowed masking) — `layer_types` selects the mixer and nothing else, so there is no
mode-switch axis to test beyond the two kinds.

## 5. Performance

Command: `./models/autoports/qwen_qwen3_6_35b_a3b/tests/run_perf.sh` (one Tracy run per
mode/kind — see `work_log.md` §7). Artifacts per case in `tracy/<kind>_<mode>/`:

| file | what it is |
|---|---|
| `<mode>_perf_report.txt` | **human-readable `tt-perf-report` table** (per-op device time, cores, DRAM %, FLOPs %, math fidelity) |
| `<mode>_perf_report.csv` | the same rows as CSV, signpost-filtered (`--csv`) |
| `<mode>_perf_report.console.log` | `--csv`-run stdout, kept for provenance |
| `<mode>_perf_report_stacked.csv` / `.png` | tt-perf-report's stacked breakdown |
| `<mode>_ops.csv.gz` | the raw post-processed Tracy ops CSV the report was built from (1.2-21 MB raw; `run_perf.sh` gzips it after the reports, so `gunzip -k` before re-running `tt-perf-report`). The two **decode** CSVs exceed this repo's 500 KB committed-file limit even gzipped (1.71 MB / 730 KB) and are excluded by `tracy/.gitignore`; both prefill CSVs (431 KB / 151 KB) are committed — see `tracy/README.md` |
| `tracy_run.log.gz` | full Tracy + pytest transcript (gzipped) |

Plus `perf_host_summary.jsonl` (host wall-clock rows) and `perf_summary.json` (reduced table,
produced by `tests/summarize_perf.py`).

All evidence is regenerated in one serialized pass, one hardware command at a time. The order
is whatever the runner scripts and the dependency between them require, and the artifact mtimes
are the record of it — not the order narrated in prose, which is easy to get wrong. Two things
about freshness are worth stating precisely, because "everything postdates the code" is a claim
that is easy to assert and easy to break:

* Three groups of files under `doc/` are not measurements of this layer and do not track code
  changes: `weight_stats/layer_{00,03}.json` (derived from the HF **checkpoint**;
  `test_weight_stats_match_real_checkpoint` re-validates it every run), the `.gitignore` and
  `README.md` files that describe the artifact policy, and `triage/` (a record of the §6 incident
  at the time it happened).
* Everything else under `doc/` postdates the final change to `tt/functional_decoder.py`. That is
  checkable rather than asserted:

  ```bash
  find models/autoports/qwen_qwen3_6_35b_a3b/doc -type f \
    ! -newer models/autoports/qwen_qwen3_6_35b_a3b/tt/functional_decoder.py
  ```

  should list only the three groups above. If a later source edit breaks that, prefer re-running
  the affected artifact over arguing about it; `git diff` tells you whether the edit could change
  behaviour at all (a docstring or comment cannot, and `test_no_runtime_host_fallback` strips both
  before scanning).

Note on committing evidence: the repository root `.gitignore` excludes `*.log`, `*.csv` and the
directory name `generated`, which silently kept the test logs, the watcher log and the filtered
perf CSVs out of the first commit of this stage. `doc/.gitignore` re-includes them, so the
evidence is reproducible from the commit and not just from a live worktree.

Measured windows are bounded by signposts (`PERF_PREFILL`/`PERF_PREFILL_END`,
`PERF_DECODE`/`PERF_DECODE_END`) after a compile+warmup pass and a device synchronize. **Decode
is measured from trace replay only** (`ttnn.execute_trace` in a loop). Watcher is never enabled
in a profiling run — `test_perf.py` fails fast if `TT_METAL_WATCHER` is set.

**Units:** this `tt-perf-report` (1.2.8) emits `Device Time` and `Op-to-Op Gap` in
**microseconds** (not the raw Tracy `DEVICE KERNEL DURATION [ns]`); `summarize_perf.py` sums
those columns and divides by the iteration count.

| case | shape | ops in window | device kernel / iter | op-to-op gap / iter | host wall / iter |
|---|---|---|---|---|---|
| prefill `linear` | seq 2048, batch 1 | 1750 (2 iters) | **341.85 ms** | 0.860 ms | 343.14 ms |
| prefill `full` | seq 2048, batch 1 | 392 (2 iters) | **281.52 ms** | 0.136 ms | 282.01 ms |
| decode `linear` (traced) | batch 32, `cur_pos` 4095 | 968 (8 iters) | **57.96 ms** | 0.089 ms | 58.08 ms |
| decode `full` (traced) | batch 32, `cur_pos` 4095 | 872 (8 iters) | **50.08 ms** | 0.238 ms | 50.35 ms |

As single-layer host throughput (`perf_host_summary.jsonl`, `tokens_per_s_host`): prefill
**7270 tok/s** (`full`) / **5963 tok/s** (`linear`) at seq 2048 batch 1, and decode **635 tok/s**
(`full`) / **551 tok/s** (`linear`) aggregated across the 32-slot batch (i.e. 32 tokens per
~50/58 ms traced iteration; `iters * batch / elapsed`, `test_perf.py:163`). These are *per layer*; a 40-layer model at this
per-layer cost would be far too slow to serve, which is what the `optimize` stage exists for and
what the two observations below point at.

Two things worth carrying into the `optimize` stage. Both come from the same three-way split,
which `summarize_perf.py` derives into `perf_summary.json` (`blocks`) by finding the mixer/MoE
boundary structurally — the last `LayerNormDeviceOperation` before the first sparse matmul in an
iteration, i.e. `post_attention_layernorm` — rather than by hand:

| case | token mixer | expert matmuls | MoE dense-intermediate elementwise | total |
|---|---|---|---|---|
| prefill `linear` | 63.60 ms (18.6%) | 227.20 ms (66.5%) | 51.05 ms (14.9%) | 341.85 ms |
| prefill `full` | 3.81 ms (1.4%) | 226.63 ms (80.5%) | 51.08 ms (18.1%) | 281.52 ms |
| decode `linear` | 8.83 ms (15.2%) | 18.58 ms (32.1%) | 30.55 ms (52.7%) | 57.96 ms |
| decode `full` | 1.05 ms (2.1%) | 18.57 ms (37.1%) | 30.46 ms (60.8%) | 50.08 ms |

* **The MoE is the whole cost; attention is rounding error.** For `full` layers the token mixer is
  1.4% of prefill and
  2.1% of decode. The optimisation
  budget belongs to expert routing, and — less obviously — to the **elementwise work over the
  dense-over-256-expert intermediates**, which is
  18.1% of prefill and
  60.8% of decode, i.e. larger than
  the expert matmuls themselves in decode. `linear` layers add the gated delta rule on top: mixer
  18.6% of prefill and
  15.2% of decode.
* **Device-bound, not dispatch-bound.** Op-to-op gap is 0.05-0.48% of device
  time (`gap/device` = 0.252% / 0.048% / 0.154% / 0.475% for the four rows in table order), and
  host wall-clock exceeds device kernel time by 0.17-0.54%. The traced decode
  has essentially no dispatch overhead left to remove.

**These rows are measured at `supported_context = 8192`, not the advertised 262144**
(`perf_summary.json` records it per row). Decode cost grows with `cur_pos`: the decode SDPA alone is
11.5 ms/call at 262144 keys versus ~1 ms here (§3.8), so an advertised-context decode step is
roughly 10 ms slower than the table shows. `test_perf.py` explains why the profiled shape is what it
is — batch 32 at the full context needs 16 GiB of paged K/V, leaving no room for a profiler buffer.

`linear` prefill issues 4.5x more ops than `full` (1750 vs 392 in the same window) for the same token count: the gated delta rule contributes a 32-step Python-driven chunk scan plus the UT transform per
2048-token chunk. It is still device-bound, so the op count is a latency risk only at much
shorter sequences.

## 6. Known limitations / optimisation opportunities

These are *functional-stage* compromises, all deliberate and all correctness-neutral. They
belong to the `optimize` stage, not here.

1. **MoE prefill is dense-per-tile-group.** `sparse_matmul` works per 32-token tile group, so
   every expert any token in the group selects runs for the whole group. Measured
   (`tests/measure_expert_union.py`, CPU only): the union of the group's top-8 selections is
   **162.3 of 256 experts** on average (min 152, max 174, over the 64 groups of the measured
   2048-token prefill), i.e. **20.3x** the ideal gather-by-expert work. The closed form for
   uniform routing, `256 * (1 - (1 - 8/256)^32) = 163.3`, agrees.

   At that density the fused gate/up runs at **~9.5 TFLOP/s with 32 of 64 cores busy**, because
   `sparse_matmul` parallelises over N tiles only (N=1024 -> 32 tiles). That rate is hand-derived,
   because `tt-perf-report` omits DRAM/FLOP utilization for these rows (`Warning:
   SparseMatmulDeviceOperation rows without numeric nnz were found`, in every
   `*_perf_report.console.log`) — `nnz` is resolved at runtime, see limitation 2. The derivation is
   checkable against `tracy/full_prefill/prefill_perf_report.csv`, where the row
   `SparseMatmulDeviceOperation active=?/4096 x 32 x 2048 x 1024` appears **8 times** with
   `Cores` 32.0 — issued 4x per prefill iteration (512 tokens = 16 tile-groups per call,
   `4096 = 16 groups x 256 experts`) over a 2-iteration window. Those rows sum to 293.95 ms,
   i.e. 36.74 ms per call and 146.97 ms per iteration, which is the §5 figure. Then
   `16 groups x 162.3 experts x (32 x 2048 x 1024 x 2) FLOP / 36.74 ms ~= 9.5 TFLOP/s`.
   Passing `tt-perf-report --active-experts 162` would make the tool report the utilization
   directly instead; the hand derivation is kept because it is the number quoted here.
2. **`nnz` is left inferred** for every `sparse_matmul`. A static count that disagrees with
   the actual non-zeros deadlocks the mcast receivers (tt-metal #45943 / #45052), and the
   count is data-dependent, so robustness wins here.
3. **q/k head duplication is folded into the weights**, growing the DeltaNet q|k|v block from
   8192 to `delta_qkv_width` = 12288 columns (`+4096x2048` MACs/token) to avoid a runtime
   `repeat_interleave`. With the `z` block riding the same conv the depthwise width is
   `conv_dim` = 16384, and the fused input projection is 16448 wide (`+ 2x32` for the `b|a`
   gates) — the `32 x 2048 x 16448` matmul row in the prefill perf report. A
   16-head-with-256-wide-value state layout would remove the duplication cost.
4. **Everything is DRAM-interleaved bf16** with no L1 sharding or program-config tuning
   outside the sparse matmuls and SDPA.
   One layout choice is load-bearing rather than default, and is documented at the site: the RoPE
   tables are **ROW_MAJOR**, because `ttnn.embedding` converts a TILE weight to row-major on every
   call, which at the advertised context would untilize 2 x 32 MiB per decode step to read
   `max_batch_size` rows. Prefill tilizes only the chunk it slices. Keeping both layouts instead
   costs 64 MiB per layer and ran the device out of DRAM in the real-weight tests.
5. **`delta_dtype = float32`** for the whole chunked delta rule, not just the state.
6. `chunked_scaled_dot_product_attention` is called with an **integer** `chunk_start_idx`, so one
   program is compiled per distinct prefill-chunk offset — 128 of them for a full-context prefill
   (262144 / 2048). The op's device-tensor form (`chunk_start_idx_tensor`) would make the offset a
   runtime argument and collapse that to one program; `probe_ttnn_ops.py` confirms it works on this
   build (PCC 0.999790). It is not used here only because feeding it without a host write needs a
   setup-time offsets table plus a per-chunk device slice, which is program-cache tuning rather
   than correctness — exactly the `optimize` stage's job. Nothing about the current path is wrong,
   it just compiles more programs than it needs to.
7. **Decode SDPA runs 1 core per KV head** (`decode_sdpa_max_cores_per_head = 1`). That is the
   only value correct at every `cur_pos` (§3.8), but it is a *serial* pass over the keys: at the
   advertised context the op alone is the dominant decode cost. More cores per head are more
   accurate at long context and structurally wrong below a context that grows with the core count,
   so `optimize` can only beat this by bucketing traces per context range — with the accuracy
   ceiling measured (16 cores/head reaches 0.9999958 at 262144 but 0.03 at 1024). §5 records what
   the serial pass costs.

## 7. Bugs found and fixed during bringup

1. **`ttnn.slice`, `ttnn.reshape` and `ttnn.to_memory_config` can return aliasing views.**
   `ttnn.slice` hands back the *input* (same buffer address) when the slice covers the whole
   tensor, `ttnn.reshape` is always a view, and `to_memory_config` returns its input when
   nothing has to move. Deallocating those results freed the caller's page table and the
   persistent conv/recurrent state buffers, and in the MoE it freed the buffer an earlier
   chunk's output still pointed at. Fixed with three explicit ownership helpers — `_subview`
   (returns an `owned` flag from a buffer-address comparison), `_owned_slice` (clones when the
   slice aliases) and `_move` — plus a `_view` helper that documents "the input stays the
   owner". **Measured effect of the MoE part of this fix:** `prefill[full] seq=2049` went from
   PCC 0.9998672 to 0.9999925, i.e. the aliasing was silently corrupting multi-chunk MoE
   prefill output, not just leaking.
2. **Padding tokens contaminated the linear-attention state.** Prefill pads up to
   `PREFILL_ALIGN`; the padded rows were advancing the conv and recurrent state past the end
   of the real sequence, so the next decode step continued from the wrong state. Output PCC
   could not see it (padded outputs are sliced off). Fixed by taking the conv context at the
   *logical* end and by zeroing `beta` and `g` on padded positions — which makes the
   recurrence step exactly the identity there. Regression test:
   `test_decode_after_non_aligned_prefill`.
3. **`conv_dim` was 3 blocks wide, not 4.** The z block rides the depthwise conv with an
   identity tap (so one conv+silu emits post-conv q/k/v *and* `silu(z)`), which the derived
   width did not account for.
4. **Repo-level:** the root `conftest.py` still imported the deleted
   `models/tt_transformers`, breaking pytest collection for the whole repository. Guarded the
   import with a no-op fallback.
5. **Persistent RoPE tables could be freed by their own slice.** `_full_attention_prefill`
   sliced `self.w["rope_cos"]/["rope_sin"]` with a raw `ttnn.slice` and then deallocated the
   result. For a chunk at `abs_pos == 0` whose padded length equals `supported_context` — reachable
   whenever `supported_context <= prefill_chunk_size`, e.g. `supported_context=1024` with
   `seq_len=1000` — that slice covers the whole table, aliases it, and the deallocate frees the
   layer's weights; the *next* forward then reads freed buffers. Every other slice in the file
   already went through `_subview`; these two did not. Fixed, plus the same treatment for
   `_valid_mask`'s slice of `ones_column`. Regression test:
   `test_prefill_covering_whole_context_does_not_free_weights`.
6. **`current_pos = -1` reached `ttnn.embedding` as an unsigned index.** The documented
   inactive-slot value was typecast `int32 -> uint32` and used to index the RoPE table, i.e. an
   out-of-table read. Now clamped on device (`ttnn.maximum(idx, 0)`) before the lookup; the
   clamped row is discarded because SDPA skips that slot. `test_decode_skips_inactive_slots_with_negative_position`
   now also asserts the inactive rows are finite. Cost: exactly one extra device op per traced
   `full` decode iteration — the scalar `maximum` lowers to a `UnaryDeviceOperation`, so the report
   shows 12 unary ops/iteration instead of 11 with the group total unchanged at 10.42 ms/iter, i.e.
   below run-to-run noise at this resolution.
7. **The RoPE tables were in the wrong layout, and it cost an O(context) op per decode step.**
   `ttnn.embedding` converts a TILE-layout weight to ROW_MAJOR **on every call**
   (`ttnn/cpp/ttnn/operations/embedding/embedding.cpp:30-32`), so the decode RoPE gather was
   untilizing both whole tables to read `max_batch_size` rows — 2 x 32 MiB per step per layer at the
   advertised context, and visible in the pre-fix Tracy window as
   `UntilizeDeviceOperation in0=[1,1,8192,64]` twice per iteration. The tables are now stored
   ROW_MAJOR and prefill tilizes only the chunk it slices. Effect on the traced `full` decode:
   **109 ops/iteration, down from 111** — the two untilizes are gone and the `Embeddings` ops are
   now preceded only by the `current_pos` typecast. Found by the round-2 review, not by a test; the
   cost is invisible at the profiled `supported_context = 8192` (16 us/step) and would have grown
   32x at the advertised context.
8. **A slot reused for a new sequence inherited the previous sequence's DeltaNet state.**
   `prefill_forward` loaded the slot's conv/recurrent carry unconditionally, so `start_pos = 0` — the
   caller's way of saying "new sequence here" — continued the previous occupant's recurrence.
   Full attention self-heals (the prefill rewrites every block it will read); linear attention does
   not. Invisible to every existing test because the comparison helper called `reset_state()` first.
   Now zeroed at `start_pos == 0`, with `test_prefill_resets_linear_state_for_new_sequence`
   deliberately bypassing that helper. Found by the round-2 review.
9. **The decode SDPA's parallel decomposition, not a program-config presence.** The op is far less
   accurate at long context when it splits a KV head's keys across many cores, and *structurally
   wrong* at short context above 1 core/head. This was misdiagnosed twice (first as the dynamic
   `k_chunk_size` path, then as "whether a program config is passed at all") before the
   `max_cores_per_head_batch` axis was swept. Now fixed and shipped — §3.8 has the sweep, the
   on-model comparison and the timing. The in-source comment at
   `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp:104` flags the
   dynamic chunk path as PCC-sensitive; that is not what this was.

## 8. Repository files this stage owns

| path | role |
|---|---|
| `tt/functional_decoder.py` | **the deliverable** — `FunctionalDecoder`, `DecoderConfig`, ownership helpers |
| `tt/reference.py` | HF reference / weight-loading boundary (torch); single-layer state-dict extraction, weight stats, synthetic weights, prefill/decode goldens, O(seq) tail references |
| `tests/harness.py` | matched (HF, TTNN) pair construction, page tables, state snapshot/restore, `TracedDecode`, PCC logging |
| `tests/conftest.py` | session device + layer-pair cache (each distinct key holds ~1.5 GiB of expert weights for the session and nothing evicts, so it logs every new key with the live count) |
| `tests/test_reference_math.py` | 26 CPU-only tests pinning the algebraic rewrites (no device, no checkpoint) |
| `tests/test_functional_decoder.py` | the correctness suite |
| `tests/test_long_context.py` | 262144-token advertised-context evidence |
| `tests/test_perf.py` | signposted warmed prefill / traced warmed decode |
| `tests/probe_ttnn_ops.py` | 21 device op-behaviour probes the design was built on |
| `tests/probe_dram_capacity.py` | allocates until the bank manager refuses, so `context_contract.json`'s `usable_dram_bytes` is a recorded measurement |
| `tests/diag_long_decode.py`, `tests/diag_sdpa_decode.py`, `tests/diag_decode_sdpa_onmodel.py` | the three diagnostics behind §3.8 (localise / op-only sweep / on-model control) |
| `tests/run_perf.sh`, `tests/run_watcher.sh`, `tests/run_long_context.sh` | reproducible runners |
| `tests/measure_expert_union.py` | CPU-only: the two MoE routing facts the analyses divide by — experts activated per 32-token sparse-matmul group (§6 limitation 1) and bf16 top-k selection agreement (`work_log.md` §5.1) |
| `tests/write_context_contract.py`, `tests/summarize_perf.py` | derive `context_contract.json` / `perf_summary.json` from evidence |
| `tests/dev_smoke.py` | single-command dev driver (build a pair, prefill, decode, print PCC) |
| `doc/functional_decoder/triage/` | the `tt-triage` attempt from the `work_log.md` §6 profiler-abort incident, and why its report is empty |

One repo file outside the autoport directory was touched: `conftest.py` (§7 item 4).

## 9. Notes to carry into later stages

* **`start_pos` must be a multiple of `sdpa_chunk` (128), which is an op contract.** Chunked
  SDPA turns the absolute offset into a chunk index by *integer division* — identically in both
  entry points this checkout offers: `chunk_start_idx / q_chunk_size` at
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp:133` for the scalar
  offset, and the same expression at
  `.../sdpa/device/kernels/dataflow/reader_interleaved.cpp:260` for the device-tensor
  (`chunk_start_idx_tensor`) offset. A misaligned offset places the causal-mask diagonal in the
  wrong tile and returns *silently* wrong values — the op validates only
  `chunk_start_idx >= 0` (`sdpa_device_operation.cpp:187`) — so `prefill_forward` rejects it
  rather than rounding. Consequences for later stages: a chunked-prefill or prefix-cache boundary
  that is page-aligned (64) but not 128-aligned is not usable as a `start_pos`; the only lever is
  lowering `DecoderConfig.sdpa_chunk` (floor `TILE` = 32, at a perf cost), not the flexible
  device-tensor path. This does not restrict prompt length: a prefill of **any** `seq_len` up to
  the full context works in one call (`start_pos = 0`), which is the path the full-model and vLLM
  stages use for a fresh request.
* **Batched decode PCC is computed over the concatenated batch** (`harness.decode_and_compare`),
  so at batch 32 a single bad slot is diluted ~32x. Per-slot coverage exists separately
  (`test_decode_ragged_current_positions`, `test_decode_skips_inactive_slots_with_negative_position`,
  `test_prefill_per_user_slots`), but all three are `full_attention`: there is **no per-slot
  linear-attention decode comparison above batch 2**. The DeltaNet recurrence is per-slot and the
  state is updated whole-tensor in place, so a slot-indexing error there would show as one bad row
  diluted into a batch average. A full-model stage adding batched paths should add that comparison.
* **The §3.8 fix is batch-dependent, and later stages decode at many batch sizes.** The old
  default was only wrong at small batch (55 cores/head at batch 1 down to 1 at batch 32 — §3.8 has
  the table); the shipped config pins 1 core per (slot, KV head) at every batch. A stage that
  changes the decode batch, or adds continuous batching where the active-slot count varies per step,
  inherits a *fixed* decomposition rather than one that silently changes with occupancy. That is the
  property to preserve.
* **The §3.8 handoff is a *latency* question now, not an accuracy one.** Shipping 1 core/head
  took the advertised-context decode from 0.9986 to 0.9997685 and made every context correct; what
  remains on the table is (a) 0.9999958 at 262144 from 16 cores/head, which is unusable below
  ~4096, and (b) the latency of the serial pass. Both are per-context-range choices, so they need
  trace bucketing; `decode_sdpa_max_cores_per_head` / `decode_sdpa_program_config` are the knobs.
* **Two upstream reproducers came out of §3.8** and are worth filing: an explicit
  `max_cores_per_head_batch=110` config scores 0.0534 where no config at all (same derived 55
  cores/head, same grid, same chunk policy, same `exp`) scores 0.7664; and every setting above 1
  core/head returns PCC ~0.16-0.37 at some short context instead of failing loudly. Both are one
  command to reproduce: `python tests/diag_sdpa_decode.py`.
* **A slot reused for a new sequence is safe, but only via `start_pos = 0`.** §2 has the contract;
  the mechanism is that `prefill_forward` zeroes the slot's DeltaNet carry there. There is no
  per-slot reset API, and `reset_state()` is all-slots, so a serving stage that wants to evict one
  request's state without prefilling must add one.
* **`ttnn.embedding` untilizes a TILE weight on every call** — the reason the RoPE tables are
  ROW_MAJOR (§6 limitation 4). Anything later that gathers rows from a big table on the runtime
  path should check the layout the same way; the cost scales with the *table*, not the gather.

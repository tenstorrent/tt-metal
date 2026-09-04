<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prefill spec feedback — notes from the Llama 3.1 8B bring-up

What the spec template (`prefill_spec_template_v0.json`, as filled in `spec_llama31_8b_v0.json`)
needed to carry and did not, discovered by actually running the bring-up recipe end to end on an 8x4
Blackhole Galaxy. Written for whoever revises the template next.

Findings are ordered by **cost of getting them wrong**, not by where they sit in the JSON. The ones
at the top fail *silently* — degraded PCC, wrong bytes migrated, a hang — which is exactly the class
of thing a spec is for.

Every claim below is backed by a test in this package; the test is named so it can be re-run.

---

## 1. `kv_cache` has no field for **KV heads per chip**, and the recipe states a special case as a constant

**Severity: high — silent wrong data.**

The recipe's §5.1 "Fixed — copy verbatim" table gives the per-chip cache shape as

```
[num_users * num_layers, 1, seq_local, head_dim]
```

with `1` written as a literal. That is only correct when `TP == n_kv_heads`, which is true for both
existing GQA packages by coincidence:

| model | n_kv_heads | TP | heads/chip |
|---|---|---|---|
| gpt_oss_d_p | 8 | 8 | 1 |
| minimax_m3 | 4 | 4 | 1 |
| **llama3_1_8b** | **8** | **4** | **2** |

Llama 3.1 8B is the first model here where the number is not 1. Nothing errors if you copy the
donor: the allocation succeeds, `update_padded_kv_cache` accepts it (it constrains the head dim only
in its TP-dedup mode, which this cache does not use), and the ring SDPA runs. You simply get one
KV head's worth of cache where you needed two.

**Proposed field** — derived, but stated and asserted rather than assumed:

```jsonc
"kv_cache": {
  "num_cache_tensors": 2,
  "n_kv_heads_per_chip": 2,          // n_kv_heads / tp_degree; MUST be >= 1 and integral
  "per_chip_shape": "[num_users * num_layers, n_kv_heads_per_chip, seq_local, head_dim]",
  "_constraint": "tp_degree must divide n_kv_heads; KV-head replication is not implemented in any package"
}
```

**Recipe fix:** change the §5.1 row from `1` to `n_kv_heads // tp`, and move "which heads a chip
holds" from the "not a decision" list into the *four* human decisions in §5.2 — because at
`n_kv_heads_per_chip > 1` the head index within a chip becomes part of the address math (see §2).

*Verified by* `tests/unit/test_kv_cache_write_vs_ref.py` (K/V 0.99987 at 2 heads/chip) and
`tests/unit/test_kv_cache_gqa_sp_vs_ref.py`.

---

## 2. The migration address walk's **ND-shard flattening order** is undocumented

**Severity: high — migration moves the wrong bytes, with valid-looking addresses.**

`gpt_oss_d_p/tt/runners/kv_chunk_table.py` advances a running `(bank_id, bank_offset)` counter
through nested `slot -> layer -> seq_chunk` loops. That is correct *only because* its cache has a
head dim of 1, so the loop nesting happens to match the ND-shard's flattening order. Add a real head
dim and the coincidence breaks.

The actual rule — the shard is `[1, 1, 32, head_dim]`, so shards enumerate row-major over
`(batch, head, seq_block)` — is written down nowhere. In closed form:

```
shard_index = ((slot * num_layers + layer) * n_kv_heads_per_chip + h_local) * seq_blocks + seq_block
bank_id     = shard_index % num_dram_banks
bank_offset = (shard_index // num_dram_banks) * chunk_size_bytes
```

which reduces to the donor's counter at `n_kv_heads_per_chip == 1`.

**Proposed:** state the flattening order in the recipe's KV section next to
`NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK`, and give the closed form as the reference implementation. A
stateful counter should be treated as an optimisation of it, not the definition.

*Verified by* `tests/test_kv_cache_table.py`, which reads raw bytes back from device DRAM at the
table's own addresses and compares bit-exactly, across 2 users x 3 layers x 16 configs.

---

## 3. `interfaces.decode.config_names_and_order` is under-specified for GQA

**Severity: high — a wrong config id maps a head to the wrong destination.**

The spec says:

```json
"decode": { "config_names_and_order": ["k", "v"] }
```

with the note "Two configs, and 'k' sorts before 'v' so config_id 0 = K, 1 = V."

That is not what the table has. The migration table carries **one config per (cache kind, KV head)** —
for Llama, 16 configs: `k_h0..k_h7` then `v_h0..v_h7`. The field as written describes cache *kinds*;
the src<->dst contract is over *configs*.

It also omits the rule that makes the ordering survive serialization: protobuf rebuilds configs
through a `std::map`, so ordering is **lexicographic over config names**. With 16 configs the natural
names `"0".."15"` put `"10"` before `"2"` and silently renumber every config id across an
export/import. The donors zero-pad; the spec never says so.

**Proposed:**

```jsonc
"decode": {
  "cache_kinds_and_order": ["k", "v"],
  "configs_per_kind": "n_kv_heads",     // total configs = len(cache_kinds) * n_kv_heads
  "config_id_order": "kind-major, then head: k_h0..k_hN-1, v_h0..v_hN-1",
  "config_name_encoding": "zero-padded decimal, width = digits(num_configs - 1), min 2",
  "_why": "protobuf import rebuilds configs through a std::map, so lexicographic name order IS config_id order"
}
```

*Verified by* `tests/test_kv_cache_table.py::test_kv_cache_table_protobuf_round_trip`.

---

## 4. No field for the **golden-trace head layout** (RoPE convention)

**Severity: high — a plausible-but-wrong PCC, and the most likely way to waste a day.**

`acceptance.golden_trace` is a path, and `architecture.rope.convention` is `"half_split"`. Neither
says what convention the **KV in the golden trace** is in.

They differ. The device applies RoPE with *interleaved* (Meta) cos/sin, which only reproduces HF's
half-split rotation because the q/k projection **rows are permuted at load** (`reverse_permute` /
`convert_hf_qkv_to_meta_format`). So the K sitting in the device cache is the interleave of the two
halves of the HF K. Comparing it against an HF-convention golden without undoing that permutation
does not error and does not give 0 — it gives a mid-range PCC that looks like a numerics problem and
sends you hunting through dtypes.

V is unaffected (`v_proj` is not permuted, V is not rotated), which makes the symptom even more
confusing: **V passes and K does not.**

**Proposed:**

```jsonc
"acceptance": {
  "golden_trace": null,
  "kv_golden_head_layout": "hf_half_split",   // or "meta_interleaved"
  "device_kv_head_layout": "meta_interleaved",
  "_note": "if these differ, the comparison must apply the head permutation. Affects K only; V is never rotated."
}
```

*Verified by* `tests/torch/test_llama_reference.py::test_meta_and_hf_rope_conventions_agree_under_head_permutation`,
which asserts the two are the same rotation up to that permutation.

---

## 5. The `rope` block cannot express **llama3** scaling (spec already flagged this — confirming, with a measurement)

**Severity: high. The spec's own `_TEMPLATE_GAP` called this out; it is real.**

v0's rope block is YaRN-shaped (`factor` / `beta_fast` / `beta_slow` / `mscale`). Llama 3.1 uses
llama3 smooth-ramp scaling, parameterised by `low_freq_factor` and `high_freq_factor`, with an
attention factor of exactly 1.0 (no mscale). The donor (`gpt_oss_d_p/tt/rope.py`) implements YaRN, so
the rope entry in the donor map does not transfer at all — only its plumbing does.

Worth adding to the gap note: **the error is position-dependent**. At position 0 every scaling rule
agrees, so a short-sequence unit test passes with the wrong rope. This bring-up therefore tests rope
at position 65536 on purpose.

**Proposed** — make the rope block a tagged union rather than a flat YaRN record:

```jsonc
"rope": {
  "type": "llama3",                  // "none" | "linear" | "yarn" | "llama3"
  "theta": 500000.0,
  "applied_width": 128,
  "convention": "half_split",
  "params": { "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
              "original_max_position": 8192 },
  "attention_factor": 1.0,           // llama3: exactly 1.0. YaRN: 0.1*ln(factor)+1
  "_test_requirement": "validate at a position >= original_max_position; every scaling rule agrees at position 0"
}
```

*Verified by* `tests/unit/test_rope_vs_ref.py` (0.99999 at offsets 0 and 65536, one-shot and indexed)
and `tests/torch/test_llama_reference.py::test_llama3_rope_frequencies_match_hf`.

---

## 6. `numerics` and `acceptance.pcc` are **inconsistent**, and the spec has no way to say so

**Severity: medium — a gate that cannot be met.**

The spec asks for `numerics.dense_mlp_weights = "bfp4"` and `acceptance.pcc.module = 0.999`.
Measured at real dims (4096 -> 14336 -> 4096), same random weights both sides:

| dense MLP weight dtype | PCC vs torch |
|---|---|
| bfloat16 | 0.9995 – 0.9998 |
| **bfloat4_b** | **0.9894 – 0.9897** |

bfp4 cannot reach 0.999, or even 0.99. The two fields are individually reasonable and jointly
unsatisfiable, and nothing in the template surfaces the conflict.

**Proposed:** make acceptance thresholds a function of the numerics choice.

```jsonc
"acceptance": {
  "pcc": {
    "module": { "bf16": 0.999, "bfp8": 0.99, "bfp4": 0.98 },
    "_note": "threshold is per weight dtype; a single number silently over- or under-gates"
  }
}
```

Same shape applies to the still-open `attn_weights` contest the spec records in `known_risks`
(llm_perf says bfp4, tt_transformers' accuracy path says bfp8). This package makes the dtype a
constructor argument on both the attention and the MLP so the question can be **measured** rather
than argued — but the spec should carry both a `target` and a `validated` value rather than one
number:

```jsonc
"numerics": { "attn_weights": { "target": "bfp4", "validated": "bfp8", "_owner": "unresolved, see known_risks" } }
```

*Verified by* `tests/unit/test_dense_mlp_vs_ref.py`, parametrized over both dtypes.

---

## 7. `topology.fabric_mode` does not capture what the fabric actually needs

**Severity: medium — the failure is a 10-second timeout and a dead run, not wrong data.**

The spec says `"fabric_mode": "2d"`. What a run actually needs is the **fabric config** and the
matching **CCL topology**, and they are a pair:

| machine | fabric config | `CCLManager` topology |
|---|---|---|
| plain-MESH Galaxy (this one) | `FABRIC_1D` | `ttnn.Topology.Linear` |
| torus pod (wrap links) | `FABRIC_1D_RING` | `ttnn.Topology.Ring` |

A plain-MESH single-Galaxy descriptor has no wrap-around links, so `FABRIC_1D_RING` cannot be opened
on it at all. Mismatching the pair does not degrade anything — it hangs or dies in fabric bring-up.

**Proposed:**

```jsonc
"topology": {
  "fabric_config": "FABRIC_1D",       // FABRIC_1D | FABRIC_1D_RING | FABRIC_2D
  "ccl_topology": "linear",           // MUST pair with fabric_config: 1D->linear, 1D_RING->ring
  "_note": "a plain-MESH galaxy has no wrap links; FABRIC_1D_RING is not openable there"
}
```

---

## 8. The spec says nothing about **how the bring-up is run**, and one rule is load-bearing

**Severity: medium — hours lost to a misleading error.**

Not a model fact, so arguably not the spec's job — but it belongs *somewhere*, and the recipe is
silent on it:

> **One fabric configuration per pytest process, and one mesh shape per process.**

Once a process has brought fabric up (or deliberately not, for a fabric-less `(1,1)` mesh), opening a
mesh with a different fabric config in that same process dies with

```
Fabric Router Sync: Timeout after 10000 ms on Device 1: expected status 0xa2b2c2d2 ...
Ethernet handshake likely failed -- the link may not be healthy.
```

The message points at cable integrity and link training. The links are fine. Observed in both
directions: `(1,1)` then `(1,4)`, and `(8,4)` then `(1,4)`.

Related, and worth recording as a machine capability rather than a model one: **a 4-chip submesh of
this Galaxy cannot bring fabric up at all** — the routers on the ethernet channels leaving the
submesh never complete the remote handshake. So `(1,4)` (TP without SP) is not a testable
configuration here, and a package targeting `(8,4)` should not write tests that assume it is.

**Proposed:** a `test_topology` block, or a line in the recipe's §4 mesh prerequisite. This
package's `run_bringup.sh` encodes the grouping.

---

## 8b. The recipe's **fixed references** are mutually inconsistent

**Severity: medium — an AttributeError, so at least it is loud.**

Recipe §3 lists pointers that "do NOT vary from model to model", to be taken as-is. Two of them do
not fit together:

| What | Pointer | Mode |
|---|---|---|
| CCL wrappers | `gpt_oss_d_p/tt/config.py` | copy |
| Embedding | `minimax_m3/tt/parallel_embedding.py` | copy |

`gpt_oss_d_p`'s `MeshConfig` exposes `allreduce` and `allgather`. The MiniMax embedding's 2D
vocab-parallel path calls `mesh_config.reduce_scatter`, which only MiniMax's `MeshConfig` has. Copy
both fixed references as instructed and the 2D embedding raises
`AttributeError: 'MeshConfig' object has no attribute 'reduce_scatter'` the first time it runs.

The fix is small (port `reduce_scatter` from the MiniMax `MeshConfig` — it is ten lines and the same
`reduce_scatter_minimal_async` call `allreduce` already makes internally). The point is that a table
of "fixed" pointers spanning two packages needs to be checked as a **set**, not entry by entry.
Worth noting the recipe's own §"Duplication the canonical entries are hiding" already flags
`parallelism.mesh_config` as copy-pasted across five packages — this is that duplication biting.

**Proposed:** either hoist `MeshConfig`/`CCLManager` into `common/prefill` (the recipe's stated P7
intent), or make the fixed-reference table name ONE package per coupled group the way the KV cluster
already does.

---

## 9. Smaller gaps, with proposed fields

| Gap | Proposal |
|---|---|
| No `mlp_mats` / `mlp_bias` (spec's own `_TEMPLATE_GAP`) — confirmed needed; Llama is 3-matrix gated, unbiased. | `"ffn": { "mlp_mats": 3, "mlp_bias": false, "gated": true }` |
| No `attention_bias` at the top of the attention block — it is in `dims` and easy to miss, and it changes whether the fused QKV carries a bias term at all. | promote to a first-class `attention.bias` |
| `shapes.chunk_size` is a single value, but a runtime can serve several (one indexed rope per size, block-cyclic period keyed on `chunk_size // sp`). | `"chunk_sizes": [4096], "default_chunk_size": 4096` |
| No field for the **embedding sharding mode** (1D emb-dim-only vs 2D vocab-parallel). It changes per-device memory and adds two SP collectives per chunk. Irrelevant for an 8B model, load-bearing for a 671B one. | `"embedding": { "shard_mode": "1d" \| "2d_vocab_parallel" }` |
| `performance_targets` is entirely null and the spec says so honestly — but there is no field distinguishing "no target yet" from "target is zero/NA". | add `"_status": "absent" \| "measured" \| "derived"` per block |
| `memory_budget.residency` is static; the spec's own note says it is a function of context length. | make it `"residency_by_context": [{"max_len": 16384, "moe_mlp": "SRAM"}, ...]` |

---

## 10. What the spec got right, and should keep

Worth recording, since the point of this exercise is to improve the template rather than only to
complain about it:

- **`_provenance` with MEASURED / EVIDENCED / DERIVED / ABSENT tiers.** This was the single most
  useful field. Knowing that the topology was *derived* rather than measured is what made it obvious
  that the 2-KV-heads-per-chip consequence had never been exercised by anyone.
- **`known_risks` as prose.** Five of the eight entries turned out to be real and were hit during
  this bring-up. The attn-weight-precision entry in particular saved a wrong default.
- **`shapes.divisibility` as explicit assertions.** They were copied straight into a test
  (`test_spec_topology_divisibility`) and cost nothing to check.
- **`_filled_note` per block.** Distinguishing "null because the model has no MoE" from "null because
  nobody filled it in" is what tier-2 completeness needs, and it worked.
- **Being explicit that `performance_targets` is deliberately empty**, with the non-targets that must
  not be mistaken for targets listed alongside. That is a genuinely hard thing to say well in a
  schema, and the prose does it.

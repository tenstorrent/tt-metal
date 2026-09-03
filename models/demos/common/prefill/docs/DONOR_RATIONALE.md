<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Donor rationale

Why each entry in `donor_template.json` exists, and which entries are weak enough to argue about.

## The test an entry has to pass

An entry earns a donor slot only if copying from an existing package prevents a failure that is
**silent** — one that shows up as degraded PCC, wrong addresses, or nondeterministic garbage rather
than an exception. Code that is merely tedious to retype does not qualify; the agent can write that
from the fact sheet.

By that test the entries sort into three groups.

## Modes

| Mode | Meaning |
|---|---|
| `copy` | Copy the file, adapt the deltas. The structure is the reusable part. |
| `import` | Use the existing module directly. Copying it is the bug. |
| `pattern` | Copy the shape only; content is model-specific. |
| `contract` | No donor file — the named doc is the authority. |

---

## Group 1 — Strong. Silent failure if written fresh.

**`kv_cache.*` (all seven entries).** See the dedicated section below.

**`parallelism.ccl_manager`.** Collectives signal completion through a 32-bit counter in L1 that
both chips read and write. Two back-to-back collectives sharing one semaphore address race: the
receiver either spins forever, or its wait is satisfied by a leftover count and it reads a buffer the
peer has not filled. The ping-pong sets in the donor exist precisely to make that impossible. The
failure is nondeterministic, scale-dependent, and non-local — it surfaces as bad output three ops
downstream.

**`parallelism.mesh_config` / `ccl_wrappers`.** "Allreduce" is not an op — it is reduce-scatter then
all-gather, with a specific deallocate ordering between them that bounds peak DRAM under long
context. Write it fresh and you get a correct-looking function that OOMs at 55k.

**`weights.weight_cache`.** The cache-populate run and the serving run must derive identical cache
key names. They do not share a code path, only a convention. Diverge and the cache silently misses —
the model loads, runs, and is slow, with no error.

**`testing.unit_pcc_scaffold`.** This is the verification mechanism for every other entry. The
donor's pattern — identical random weights on both sides, shared rope, single card, no checkpoint or
network — is what makes stage-level testing possible at all. Get it wrong and every downstream PCC
number is uninterpretable.

---

## Group 2 — Solid. Real non-obvious code, but failures are loud.

**`weights.loading`, `weights.dequant_and_permute`.** safetensors iteration, prefix filtering, qkv
fusion/swizzle, tile-alignment padding. Genuinely fiddly, and the donor has already discovered the
edge cases. But a mistake here throws or produces obviously garbage output. Worth a donor; not
critical.

**`compute.rope`.** Variant plumbing is easy; the *whole-cache indexed* rope is not. It is built
once, block-cyclic aware, and shares its alignment constraint with the KV cache. Arguably belongs in
the KV cluster.

**`compute.attention`.** Honest label: **partial**. What transfers is head split, projection
sharding, program configs, and the output-projection CCL tail. The attention math does not — that is
the model. Do not let the agent treat this as copy-with-edits.

**`compute.moe` (`import`).** The DeepSeek EP substrate is a real shared library, not a template.
The mode matters more than the path: copying it into a new package is the failure mode to prevent.

**`serving.runtime`.** The contract lives in a doc, but the implementation shape — `_resolve_kv`,
the chunk-range assertions, the pipeline-rank branches — is worth having in front of you.

---

## Dropped for the POC

Eight entries were cut as weak donors — either trivial to write from the fact sheet, or already
pointed at from elsewhere:

| Dropped | Why |
|---|---|
| `reference.torch_golden` | A golden depends on *this* model's HF reference, not another model's golden. Only justified when the attention family matches (Kimi-K3's MLA really was trimmed from DeepSeek-V3's `modeling_deepseek.py`) — too rare to keep as a field. |
| `compute.decoder_layer`, `compute.model` | The recipe's D2/M2 stage text already names `gpt_oss_d_p/tt/layer.py` and `tt/model.py` inline. No information lost. |
| `package.layout`, `package.config_constants` | Cosmetic, and the reusable half of config_constants is already `testing.config_diff_test`. |
| `serving.adapter`, `serving.registry_and_manifest` | `ADDING_A_PREFILL_MODEL.md` is a better reference than the file. |
| `weights.substate` | Twelve lines of dict filtering. |

Re-add any of these if a real bring-up shows it was load-bearing.

What survives the cut: 16 canonical entries the donor-finder never touches, and 9 it must fill.

---

## Duplication the canonical entries are hiding

Five canonical entries point at code already copy-pasted across packages. Their `notes` carry a
`SHOULD BE SHARED` prefix, so `grep "SHOULD BE SHARED" donor_template.json` is the hoist backlog.

| Entry | Copies | Evidence |
|---|---|---|
| `weights.weight_cache` | per-package | same convention re-encoded each time |
| `parallelism.ccl_manager` | 4 | `gpt_oss_d_p/tt/ccl.py` and `minimax_m3/tt/ccl.py` are both exactly 139 lines; 64 differ, nearly all docstrings |
| `parallelism.mesh_config` | 5 | `gpt_oss/`, `gpt_oss_d_p/`, `minimax_m3/`, `gemma4/`, `deepseek_v3_d_p/tt/moe/init_helpers.py` |
| `compute.norm` | 4+ | `models/common/rmsnorm.py` (258 lines) exists and no prefill package uses it |
| (`weights.substate`, now dropped) | 6 | all six md5s differ; 34–82 lines for the same 12 lines of logic |

For these, the donor pointer says "copy this from another model" when the honest answer is "this
should be one library". That is P7 on the GPT-OSS roadmap. It changes nothing today — you still
copy — but it marks where shared-code investment pays off.

---

## The KV cache cluster

This is the part that has to be right, and it is the one place where a **file-level** donor is the
wrong shape.

### Why it is different

KV is not a module. It is a contract spanning seven roles that never import each other:

| Role | What that file does |
|---|---|
| `layout_and_allocation` | Defines the cache tensors: per-chip shape, DRAM NdShard spec, slot packing |
| `write_op` | The `update_padded_kv_cache` call: slot/layer indexing, dtype match, tile alignment |
| `attention_read_path` | The SDPA that reads the block-cyclic cache back — why the layout is canonical |
| `sdpa_path_selection` | One-shot vs cache-backed per chunk; fails loud on unimplemented cases |
| `kvcaches_handle` | The `KvCaches` subclass the engine holds as an opaque handle |
| `runtime_offsets` | Maps `actual_start`/`actual_end` onto cache writes, plus range assertions |
| `address_table` | DRAM bank walk: turns (slot, layer, token range) into an address |

**Locate these files; do not assume their paths.** Package layouts differ. In `gpt_oss_d_p` and
`minimax_m3` the first four live under `tt/attention/` (`kv_cache.py`, `dense_sp.py`, `prefill.py`).
In `deepseek_v3_d_p` allocation is in `utils/kv_cache_utils.py` and the middle three collapse into
`tt/mla/mla.py`. Three of seven paths differ between those two layouts, which is why
`sub_entry_paths` in the donor template is filled per bring-up rather than defaulted.

All seven encode the same constants independently:

| Constant | Value |
|---|---|
| Per-chip shape | `[num_users * num_layers, 1, seq_local, head_dim]` |
| Slot formula | `slot = user_id * num_layers + layer_idx` |
| Contiguous tokens per DRAM bank | 32 |
| Shard distribution | `ROUND_ROBIN_1D` |
| Sequence sharding | SP block-cyclic on the SP axis |
| Alignment | `max_seq_len % (TILE_SIZE * sp) == 0` |

Nothing checks that they agree. Change the bank size in the allocator and the ring SDPA still runs —
it just reads the wrong tokens, and you get a PCC of 0.7 with no error. Change the slot formula and
migration pulls another user's cache. Change the sequence sharding and the address table produces
valid-looking addresses pointing at the wrong chip.

### Consequence for the donor map

`kv_cache` carries a **`donor_package`**, not seven independent paths, plus an explicit
`must_match_across_all_entries` block. Taking the allocator from GPT-OSS and the ring SDPA from M3 is
a defensible-sounding decision that produces a silently broken model.

The `must_match` block is in the JSON so an agent can verify it mechanically — grep the six files for
each constant and assert they agree — rather than trusting that a copy went cleanly.

### Why the layout is canonical, not arbitrary

`gpt_oss_d_p/tt/attention/kv_cache.py` and `minimax_m3/tt/attention/kv_cache.py` are near-identical,
down to the same `"Deliberately NOT init_kvpe_cache"` comment. That is not laziness. Two consumers
pin the layout:

1. **The chunked ring SDPA** reads the block-cyclic sequence directly out of the cache. It is a
   shared C++ op; the layout is its input format.
2. **The migration address walk** computes a DRAM address for any (slot, layer, token range). It
   needs the bank geometry to be exactly what the allocator used.

So the instruction is: **copy verbatim, change only these four things.**

| Varies | Examples |
|---|---|
| Number of cache tensors | MLA: 1 latent (`kvpe`) · GQA: 2 (`k`, `v`) · M3: 3 (`k`, `v`, `index_k`) |
| `head_dim` | 64, 128 |
| `cache_dtype` | `bfloat8_b` typical |
| Auxiliary caches | M3's `index_k` is TP-replicated, not head-sharded |

Which head a chip holds is **not** an allocation parameter — it is decided at write time by how the
input chunk is mesh-mapped.

### Where it can still go wrong

- `init_kvpe_cache` is MLA-specific (one latent cache). GQA packages deliberately avoid it but reuse
  its NdShard spec so `update_padded_kv_cache` works unchanged. An agent that "helpfully" calls it
  for a GQA model gets one cache where it needs two.
- Live divergence in the donors: M3 hardcodes `BH_NUM_DRAM_BANKS = 8`; GPT-OSS calls
  `get_num_dram_banks(mesh_device)`. Prefer the GPT-OSS form; the M3 constant is wrong on any other
  bank count.
- `kv_chunk_table.py` config ids are the src-to-dst migration contract, and protobuf rebuilds configs
  through a `std::map`, so `"10"` sorts before `"2"`. The donor zero-pads config names to survive
  this. Copy that, do not re-derive it.

### Verifying the cluster without hardware

`tests/test_kv_cache_table.py` is host-only address math. It is the cheapest possible check that the
allocator and the address table still agree. Run it host-side, long before anything touches a mesh.

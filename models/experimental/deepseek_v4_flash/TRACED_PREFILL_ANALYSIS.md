# DeepSeek V4 Flash: analysis of the open issues in `TRACED_PREFILL_ISSUES.md`

This is a read-only analysis: no code was run or changed. Line numbers refer to the working tree at the time of
writing. Suggested experiments are listed at the end, for someone with the device to run.

## Summary

**Most likely root cause of Issue 2 and Issue 3:** `reset_static_caches()` overwrites the lightning indexer's
page table with zeros.

- `DeepSeekV4Model.reset_static_caches()` fills every tensor listed in `_StaticLayerCache.__slots__`. That list
  includes `idx_page_table`, the constant identity page table (`[0, 1, 2, ...]`) through which the indexer reads
  and writes `idx_key_cache` and `comp_kv`.
- After the fill, every logical block maps to physical block 0. Every index-key write, new-entry write, score read
  and gather of the CSA indexer then goes to the same 64 rows.
- Decode only uses that table on the indexer path. With `index_dense_max_positions: 10240`, decode switches from
  dense CSA to the indexer at position 10239. That point lies inside the reported failure window (8192 is fine,
  10752 is not).
- `test_prefill_decode_demo.py` calls `reset_static_caches()` before the decode-only reference and again before
  the prefill commit. Both the Issue 2 reference and the Issue 3 generation therefore run the indexer against the
  zeroed table.
- The decode demos never call `reset_static_caches()`, so the bug stays hidden outside this test.

Each CSA layer then attends to the sliding ring plus a dozen or so of the 64 entries in block 0, each repeated
about 40 times, instead of the top 512 out of about 2700. That is enough to explain the PCC drop to 0.60 in Issue 2 and the
fluent but context-free, looping answer in Issue 3.

The proposed fix is one line (section 1.6). Two runs that need no code changes confirm or rule this out (section 5).

**Status:** experiments A and B behaved as predicted. With decode kept dense past the prompt (A), the 10752
failure went away; with the switch moved to 2048 (B), the failure appeared at 4096 tokens. The failure follows the
indexer path. The fix from section 1.6 is now applied in `tt/model.py` (`reset_static_caches`); experiment C is
pending.

---

## 1. Primary finding: `reset_static_caches()` zeroes `idx_page_table`

### 1.1 The code

`_StaticLayerCache` lists the indexer's page table and output buffer as ordinary slots:

```python
# tt/decode/attention.py:201-215
__slots__ = (
    "kv", "win_kv", "win_gate", "prev_kv", "prev_gate",
    "idx_win_kv", "idx_win_gate", "idx_prev_kv", "idx_prev_gate",
    "idx_key_cache", "comp_kv", "idx_page_table", "sel_kv",
)
```

The page table is created once as the identity mapping and nothing ever rewrites it:

```python
# tt/decode/attention.py:364-370
idx_page_table = ttnn.from_torch(
    torch.arange(n_blocks, dtype=torch.int32).reshape(1, n_blocks), dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

`reset_static_caches()` fills every slot that is not `None`:

```python
# tt/model.py:1784-1789
for sm in self.submeshes_io:
    for scache in sm["scaches"].values():
        for name in _StaticLayerCache.__slots__:
            buf = getattr(scache, name)
            if buf is not None:
                ttnn.fill(buf, _MASK_NEG if name == "prev_gate" else 0.0, output_tensor=buf)
```

`ttnn.fill` accepts this tensor. Its binding lists INT32 as a supported dtype
(`ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp:1853-1858`). The unary device op has a row-major path
and no layout or dtype check that would reject it (`unary_device_operation.cpp:28-116`, `:189-199`). So the call
runs quietly and `idx_page_table` becomes `[0, 0, 0, ...]`.

A search of `tt/` finds only these uses of `idx_page_table`: its construction, the two readers below, and a skip in
`_compressor_slots` (`tt/model.py:1489`). No code writes the identity mapping back.

Another part of the model already treats the table as constant. `_compressor_slots` skips `idx_key_cache` when
`idx_page_table` is set (`tt/model.py:1486-1490`), and `_commit_index_keys` documents that it writes
"identity paging: entry `w` at block `w // block`" (`tt/model.py:3118-3120`). Only `reset_static_caches` breaks
that assumption.

### 1.2 How a zeroed table breaks the indexer

All four indexer accesses go through the table:

| Access | Where | With the table zeroed |
|---|---|---|
| Index-key write, every step (dense and indexer traces) | `attention_csa.py:612-615`, `paged_update_cache(..., page_table=idx_page_table)` | Window `w` is written to block 0, row `w % 64` |
| New compressed entry write, indexer trace only | `writer_fused_lightning_select_kv.cpp:112-121`, `physical_row = page_table[block] * 64 + row % 64` | Entry `w` is written to `comp_kv` block 0, row `w % 64` |
| Key reads for scoring | `reader_fused_lightning_select_kv.cpp:108-115`, `first_tile_id = page_table[block] * tiles_per_block` | Every logical block scores the same 64 keys in block 0 |
| Gather into `sel_kv` | `writer_fused_lightning_select_kv.cpp:504-521`, `physical_blocks[i / 64] * 64 + i % 64` | Every selected row is read from `comp_kv` block 0 |

At 10752 tokens there are 2688 closed windows, or 42 logical blocks. Every block produces the same 64 scores, so
the top 512 turns out to be the best 12 or 13 distinct rows of block 0, each repeated about 40 times. The softmax
in `_attend` then runs over the ring (128 rows) plus those duplicates. The CSA layers lose almost all of their
long-range context.

Dense decode never reads through this table. Dense CSA attends `scache.kv` directly, and the index-key writes into
block 0 are harmless there because nothing reads them. That is why the bug only appears after the switch.

### 1.3 Why the failure starts between 8192 and 10752 tokens

`configs/system_configs.yaml:140` and `:261` set `index_dense_max_positions: 10240`.
`_index_dense_limit()` returns `max(10240, 4 * 512) = 10240` (`tt/model.py:1098-1110`), and
`_index_sparse_step(pos)` is `(pos + 1) >= 10240` (`tt/model.py:1112-1117`). So:

- Prompts of 1024, 4096 or 8192 tokens are decoded entirely on the dense trace and never read the zeroed table.
  PCC 0.93 to 0.98.
- A 10752-token prompt switches at position 10239. The last 512 prompt tokens run on the indexer trace. PCC 0.60,
  wrong argmax.

The switch point is the only length-dependent behaviour change in that window. The capacity limits the Issue 2
hypotheses suggest (`CSA_MAX_COMPRESSED_ENTRIES`, `max_blocks_per_core`) are not needed to explain the data.

### 1.4 Effect on Issue 2 (decode-only reference)

Sequence in `_decode_one` (`tests/prefill/test_prefill_decode_demo.py:748-764`):

1. `reset_session(sid)`, then `reset_static_caches()`: the table is zeroed.
2. `decode_prompt_traced(prompt.ids[:10752], 0)`:
   - Positions 0 to 10238 use the dense trace and are correct. The index keys all land in block 0, which is
     harmless for now.
   - At position 10239, `_copy_dense_csa_entries()` (`tt/model.py:1136-1158`, called at `:2710-2711`) writes
     entries 0 to 2559 into `comp_kv` blocks 0 to 39 with `slice_write`. That write bypasses the page table.
     `comp_kv` block 0 now holds entries 0 to 63, while `idx_key_cache` block 0 holds the keys of windows 2496 to
     2559. The keys no longer match the values.
   - Over the remaining 512 steps, 128 new windows close. Each new entry and its key go to block 0, row `w % 64`,
     so block 0 ends up holding the last 64 entries, about the last 256 tokens.
3. At the last prompt token, every CSA layer sees only about the last 256 tokens: the last 64 entries plus the
   128-token ring, which overlaps them. The prediction (`' to'`) is consistent with a model that has lost the
   passage and the chat template.

The conclusion in `TRACED_PREFILL_ISSUES.md` is right that the decode side is wrong at 10752. The cause is in how
the test resets decode state, not in the capacity of the indexer op.

### 1.5 Effect on Issue 3 (generation after prefill)

The same `_decode_one` calls `reset_static_caches()` again (line 764) before `commit_prefill_state`.
`_commit_index_keys` writes all 2688 keys and entries in identity layout, but decode reads them through the zeroed
table. Generation starts at position 10752, which is already on the indexer trace, so:

- Each CSA layer sees block 0 only: entries 0 to 63 (the first 256 prompt tokens, mostly the chat template and the
  start of the passage) plus the 128-token ring.
- Keys and values agree this time, because both were committed in the same layout. Attention is therefore coherent
  but has almost no context.
- New windows overwrite block 0 row by row as generation proceeds.
- HCA and sliding layers are unaffected (HCA uses its own paged pools and page tables, written by
  `_write_page_tables`).

A model that sees the question and options but has lost most of the passage is likely to produce "The correct
answer is (A)." and loop on it. The eager-prefill run went through the same `_decode_one` and also answered A, so
it was affected the same way. The difference between the two runs (an explanation and EOS versus a loop) is
therefore not evidence that the traced prefill state is worse.

### 1.6 Fix (applied after experiments A and B)

Skip the page table and use the same fill values as the rest of the model:

```python
# tt/model.py, reset_static_caches
for name in _StaticLayerCache.__slots__:
    if name == "idx_page_table":
        continue  # constant identity mapping the indexer traces read through, not per-sequence state
    buf = getattr(scache, name)
    if buf is not None:
        ttnn.fill(buf, self._empty_compressor_fill(name), output_tensor=buf)
```

`_empty_compressor_fill` (`tt/model.py:1494-1503`) already returns `_MASK_NEG` for both `prev_gate` and
`idx_prev_gate`, which also fixes the issue in section 2. Filling `sel_kv` is unnecessary but harmless, because
each indexer step rewrites all of its rows.

To make this harder to break again, the commit could restore the identity table itself: `commit_prefill_state`
could write `torch.arange(n_blocks)` with `copy_host_to_device_tensor`, which does not allocate. Alternatively,
`_StaticLayerCache` could keep a separate tuple of the names that are per-sequence state and loop over that.

---

## 2. Secondary: `reset_static_caches()` fills `idx_prev_gate` with 0 instead of `_MASK_NEG`

The same loop fills only `prev_gate` with `_MASK_NEG`. `idx_prev_gate` also gates window 0's absent Ca half, and
`_empty_compressor_fill` and `build_static_layer_cache` (`tt/decode/attention.py:349`) both initialise it to
`_MASK_NEG`. After the reset, the indexer compressor's first window gives softmax weight to four zero rows.

The practical effect is small:

- It only affects index key 0.
- `idx_prev_kv` is zero, so the zeros only rescale the pooled key, and the RMSNorm after pooling mostly removes the
  rescaling.
- The commit overwrites `idx_prev_gate` from the prefill state (`tt/model.py:3007-3010`), so only the decode-only
  reference is affected.

The fix in section 1.6 covers this as well.

---

## 3. Other things to know when reading the numbers

### 3.1 Decode-only is not an exact reference between 2048 and 10240 tokens

The reference and prefill (with the indexer on) attend the top `min(512, n_entries)` causally visible entries.
Decode below 10240 is dense: it attends all entries. The two agree up to 2048 tokens (512 entries). Between 2048
and 10240 tokens, decode-only is a different algorithm.

The table suggests the effect on last-token logits is small (0.95 at 4096 and 0.98 at 8192, against 0.93 at 1024
where both paths are dense). So the 0.93 to 0.98 range is mostly the ordinary numeric gap between the two
pipelines, not dense versus sparse. Still, a PCC around 0.95 against decode-only above 2048 tokens does not show
that prefill is right.

### 3.2 Even with the fix, prefill and decode will not match exactly past the switch

The two sides compute the indexer differently:

| | Decode (`fused_lightning_select_kv`) | Prefill (`indexer_score_dsa` + `ttnn.topk`) |
|---|---|---|
| Scores | custom_mm at `MathFidelity::LoFi` (the only fidelity it supports), fp32 buffer | bf16 logits |
| Selection | exact top-k by radix select, ties broken by core and index | `theta = min(topk values)`, then `scores >= theta`, so ties can select more than k |
| Head weights | read as bf16 (`uint16 << 16`) | as computed |

The selected sets can differ by a few entries near the threshold. Expect PCC somewhat below the dense-range values
after the switch, not a return to 1.0. If PCC is still around 0.6 after the fix, look at the op next (section 3.5).

### 3.3 "Traced matches eager" does not validate prefill against the reference

Issue 1's verification shows that traced prefill equals eager prefill: 4 layers at 10752 tokens, every state tensor
at PCC 1.0. Both share the same indexer implementation (`indexer_score_dsa`, `ttnn.topk`, the threshold mask), so
an indexer bug common to both would pass that check. `tests/prefill/test_full_model_prefill_verify.py` and
`tests/prefill/full_model_reference.py` compare against the torch reference and are the right tools for that
question. Only 4 layers were compared traced against eager. A full 43-layer `DEEPSEEK_V4_E2E_TRACE_CHECK=1` run
would cover the deeper stages, including the D2D socket hops on later stages.

### 3.4 `_copy_dense_csa_entries` at the switch

The one-off copy (`tt/model.py:1136-1158`) is correct once the page table is fixed, with two caveats:

- **Allocation under live traces.** It allocates temporaries (`ttnn.slice`, `to_layout`, `reshape`) on the replay
  thread while decode traces exist. The command queue runs in order, so the copy lands between two replays, and
  temporaries that reuse freed trace intermediates are overwritten by the next replay before anything reads them.
  The risk is low, but it is the same class of hazard as Issue 4.
- **Idempotent overwrite after the commit.** In a flow without the compare run, the throw-away capture step leaves
  `_csa_entries_copied = False`. The first generation step after the commit therefore copies `scache.kv` entries
  over the `comp_kv` rows the commit just wrote. This is harmless only if `_commit_dense_kv` put the same bf16
  values in `scache.kv`, which it should. Skipping the copy when the session was committed would be cleaner.

### 3.5 If the fix does not restore the 10752 result

Then the indexer path itself is suspect. In order:

1. `fused_lightning_select_kv` at about 2700 entries, compared against a torch top-k on the same keys and query.
   `tests/decode/test_csa_layer_indexer.py` is the natural place to add that case.
2. The switch-step copy (section 3.4).
3. The hard-coded `#define COMPRESS_RATE 4` in the op's kernels. It is correct for this model's CSA, but it is not
   validated against the tensor shapes.

### 3.6 Issue 4 ("allocating under active trace" after traced prefill) is most likely benign

`_export_states` (`tt/model.py:4253-4291`) slices the persistent FIFOs after `model.synchronize("traced prefill
done")`, so no replay is in flight when the slices are allocated. The test parks them to host and frees them.
`free_states` deallocates everything that is not persistent. `prefill.release_traced_prefill()` and `del prefill`
run before the decode model is built (`test_prefill_decode_demo.py:611-619`), so prefill and decode traces never
coexist. The warning is accurate but, in this flow, it does not point to corruption. Preallocated export buffers
would remove it if a clean log matters.

---

## 4. Small issues found along the way

- **The uncommitted `tt/model.py` diff deletes the blank line before `def _send_logits`** (around line 4053).
  Black and the pre-commit hook will flag it.
- **Broadcast workaround (Issue 1).** `_pkt_bcast_width` is always at least 16, because `_pkt_w` is a whole
  number of 64-byte pages, so the split never degrades to single-int rows. The decode stage-0 packet is 3·B int32
  padded to 64 B, well under the 4.3 KB threshold, so decode does not need the same workaround. The underlying
  `ttnn.broadcast` bug still needs a standalone repro and an issue, as the issues doc says.
- **Stale docstring.** `test_prefill_decode_demo.py:46` says "`test_prefill_decode_demo` leaves the lightning
  indexer off". Both tests pass `lightning_indexer=True` (lines 194 and 220).
- **Unnecessary fill.** `reset_static_caches` also fills `sel_kv`, and each `ttnn.fill` logs the
  allocation-under-trace warning for its scratch space. Skipping `sel_kv` removes some of that log noise.

---

## 5. Suggested experiments (device runs, for the user)

All three use the Issue 2 setup (full model, question 112, `DEEPSEEK_V4_E2E_COMPARE=1`). The first two need no
code change and move the dense-to-indexer switch point, which is the strongest test of section 1: if the failure
moves with the switch, the indexer path is the cause.

**A. Keep decode dense past the prompt: the 10752 failure should disappear.**

```bash
DEEPSEEK_V4_CACHE_DIR=/path/to/cache \
DEEPSEEK_V4_E2E_COMPARE=1 DEEPSEEK_V4_LONGBENCH_INDICES=112 \
DEEPSEEK_V4_INDEX_DENSE_MAX_POSITIONS=16384 \
pytest -s models/experimental/deepseek_v4_flash/tests/prefill/test_prefill_decode_demo.py::test_prefill_decode_demo
```

Expected if section 1 is right: PCC back in the 0.95 to 0.98 range, argmax `'</think>'`, and a generated answer
that no longer loops (it may still pick the wrong letter). This also tests Issue 3, because generation stays on the
dense trace. With the limit above `max_seq`, `_reachable_variants` captures no indexer traces at all
(`tt/model.py:1214-1217`), so this run never reads `idx_page_table`. `scache.kv` grows to about `max_seq / 4`
entries, which should fit.

**B. Move the switch early: the failure should appear at 4096 tokens.**

```bash
DEEPSEEK_V4_CACHE_DIR=/path/to/cache \
DEEPSEEK_V4_E2E_COMPARE=1 DEEPSEEK_V4_LONGBENCH_INDICES=112 \
DEEPSEEK_V4_E2E_MAX_INPUT=4096 DEEPSEEK_V4_INDEX_DENSE_MAX_POSITIONS=2048 \
pytest -s models/experimental/deepseek_v4_flash/tests/prefill/test_prefill_decode_demo.py::test_prefill_decode_demo
```

Expected with the bug: PCC far below the 0.95 recorded at 4096. This is a faster reproducer than the full 10752
prompt.

**C. With the fix from section 1.6 applied**, repeat B, then the original 10752 compare, then the plain generation
run (no `DEEPSEEK_V4_E2E_COMPARE`). Expected: B and the 10752 compare close to the dense-range PCC, with a little
loss from section 3.2, and Issue 3 no longer loops.

Optionally, as direct proof of the mechanism, read one layer's table back right after `reset_static_caches()` in
`_decode_one`, for example `ttnn.to_torch(ttnn.get_device_tensors(scache.idx_page_table)[0])`. It should print all
zeros.

# Matched HF control for the frozen B1/B2 gate

Prepared CPU-only on 2026-11-14; the model control has **not been run**. No production file, existing probe, TT precision policy, or frozen threshold was changed. This is a numerical discriminator, not permission to waive `shared_pool_generator_v5`.

The repository's existing `hf_reduced_reference.json` uses real original HF layers 0/1/3, but only B1: prompt `[785,13458,13]`, then forced inputs `785,13458,13`. It therefore does not establish whether original HF reproduces the failed shared-pool comparison. Searches of the reduced HF helpers, existing batch HF probe and full-model diagnostic reports found no completed original-HF control for the exact two prompts and forced 42/43/44 history.

The new `tests/probe_shared_pool_hf_control.py` uses:

- Pinned checkpoint `Qwen/Qwen3.8-Flash-Next`, revision `812d96e10a0dd82b69d749e68693698cf6c36daa`.
- Exact shared-pool prompts: `[785,13458,13]` and `[785,13,13458,13,785]*6 + [785,13,13458]`, lengths 3 and 33.
- Three forced decode inputs 42, 43, 44, at positions 3/4/5 and 33/34/35 respectively. No sampling or text-quality judgment.
- Original HF embedding, real layers 0/1/3 retaining original cache indices, terminal hyper-connection mixer, LM head, fresh original `DynamicCache` per run, and eager CPU attention.
- The existing reduced loader's checkpoint mmap assignment with original stored dtypes. PLE alone substitutes an exact checkpoint-row lookup with `SelectedRows.quantized=False`; it performs no NF4 quantization or dequantization. No TT weight policy is emulated. The original HF rotary constructor materializes its buffers outside the meta device, as in the existing helper.

Each of two repeats runs natural B1 for each prompt, B1 left-padded to width 33 for each prompt, native B2 mixed prompts, B2 initially reversed rows, and B2 using original HF `cache.reorder_cache([1,0])` after input 42. The latter mirrors the history/position ownership transition before 43 without emulating a TT page table. Every run uses seed zero. QSA's recorded position history is asserted against the visible logical positions at every step.

Native HF left padding is explicit EOS padding plus an attention mask. Its PLE n-gram path replaces masked IDs with EOS and excludes earlier EOS segments; its indexer groups only visible indices. Padding can nevertheless change numerical execution shape. Therefore the report separates:

1. Mixed B2 versus natural B1: exact logical histories, including any HF padding/chunk-shape effect.
2. Mixed B2 versus width-33 padded B1: the same physical prefill length/mask per request, isolating batch geometry more narrowly.
3. Padded versus natural B1: padding/physical-prefill-length effects alone.
4. Repeated B1/B2, reversed rows and reordered-cache B2: repeatability and order effects at the same batch width.

The original HF model receives one prefill call, not the TT dyadic chunk schedule. Decode calls have one token per row. Full-vocabulary logits and last-valid-token hidden outputs after each selected layer and terminal mixer are retained in a separate `.pt` artifact. JSON reports actual parameter/cache dtypes, cache shapes, backend versions/build, CPU threads, kernel function provenance, source/config/index hashes, and per-phase metrics. It asserts CPU execution, eager attention and no enabled kernel substitutions; no compilation or autocast is requested. Results apply only to this recorded HF CPU backend.

The unchanged `decode_geometry_comparison` from the frozen shared-pool probe scores both combined request rows and individual rows: whole PCC >=0.995, minimum row PCC >=0.99, top1 agreement =1, top5 >=0.98 and top100 =1, with finite logits. Same-geometry exactness and per-layer differences are also recorded. `completed` means the control executed; it is separate from `all_recorded_frozen_comparisons_pass`. Failed numerical comparisons are saved rather than suppressing later discriminators. Both requested output paths must be fresh.

## Scheduled command

Run from the current checkout, only after the parent schedules CPU work:

```bash
timeout --signal=INT --kill-after=30 1200 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_shared_pool_hf_control --threads 4 --repeats 2 --snapshot /home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/shared_pool_hf_control_v1.json --tensors /home/ttuser/qwen38-flash/reference/shared_pool_hf_control_v1.pt > models/autoports/qwen_qwen3_8_flash_next/doc/full_model/shared_pool_hf_control_v1.log 2>&1
```

There are 14 independent cache histories and 56 model forwards at the default two repeats. The previous four-forward reduced fixture recorded 18,172,454,552 mapped checkpoint bytes, peak RSS 3,229,520 KiB and 1.063 seconds; these historical numbers are not a prediction for this workload. Expect roughly tens of seconds to several minutes with a warm page cache, potentially longer under cold storage/CPU contention. The 20-minute external bound is a diagnostic budget, not a measured runtime. Reserve about 24 GiB of CPU memory headroom conservatively for the 16.92 GiB mapped weights plus touched expert pages and temporaries; actual resident RSS is likely lower but is recorded. PLE reads selected rows, not its full roughly 95-GiB table. Saved logits/hidden tensors are expected to be well below 0.2 GiB. No TT devices are opened or imported.

## Interpretation limits

An observed HF failure would establish that this original-HF CPU backend also fails the frozen comparison for these exact histories, under the recorded preparation geometry. It would not establish that its arithmetic failure has the same cause as TT, nor permit a precision change or gate waiver. A padded-only failure needs to be distinguished from a batch-only failure. Repeats and permutations must be inspected before treating any difference as deterministic geometry dependence.

An HF pass would reject “HF also does this” for this tested backend and history. It would not prove another HF hardware backend invariant, or alter the already observed TT native reduction boundary. This control cannot validate TT page ownership, migration, trace feedback, state layout, quantization, or full48 quality. The existing real TT capture already isolates its first difference to routed reduction; this HF control supplies the missing matched original-model observation, not a replacement causal analysis.

Preparation checks actually run (no model weights loaded): `py_compile` and `--self-test`; both passed. The self-test checks exact prompt/position packing, padding widths, and positive/negative controls for the unchanged frozen scorer.

Source SHA256 at handoff:

- New control: `9626977966535f18442411481ff1de66ca259043f1b94d8339cecc43d2220989`.
- Existing reduced loader: `7beea49521f9d7210a12c6f15a71528b2fdc5e15c3af6f1c606abb84af7945e8`.
- Existing shared-pool probe: `83a9079df74a105790adb0913f6aeb321b0e3bc384ac364f26fc850add77b6d9`.

## V2 preparation: exact survivor-width history

V1 has since been executed by the parent; its actual results are independently interpreted in `shared_pool_hf_control_interpretation.md`. V1 JSON/PT/log are preserved. The exact V1 source is archived at `sources/shared_pool_hf_control_v1.py.txt`, with the original SHA256 `9626977966535f18442411481ff1de66ca259043f1b94d8339cecc43d2220989`.

V2 adds one original-HF history per repeat to match the TT44 survivor control:

1. Prefill requests `[0,1]` and consume forced 42 at B2.
2. Before 43, use native `cache.reorder_cache([1,0])`; consume forced 43 in order `[1,0]` at B2.
3. Before 44, use native `cache.reorder_cache([1])` to retain original request0 alone. Consume 44 at logical position5 and B1. This corresponds to TT releasing physical slot0 (original request1) and continuing physical slot1 (original request0).

The native reorder API accepts an index subset and selects convolution, recurrent, KV, indexer-key and indexer-position state. The installed generic `batch_select_indices` API does not implement all linear-attention layer types, so V2 deliberately uses the complete native `reorder_cache` path for both permutation and selection. It does not reconstruct cache values, drop the survivor's old history, or emulate TT page tables. CPU self-test uses tiny real HF cache-layer instances to verify reversal followed by subset selection preserves original row0 across all three conv/recurrent state indices and KV/indexer/position tensors.

Each survivor phase is compared to uninterrupted natural B1 request0, padded B1 request0, uninterrupted mixed B2 request0 and migrated B2 request0. At 44, the latter two references share the same logical history and prior B2 width but continue at B2; their comparison with the survivor isolates the final B2→B1 width change more closely. Repeats compare every surviving request at all available phases. All existing V1 experiments remain. The frozen scorer is imported unchanged, including its documented tie-ID inconsistency; no historical metrics are rewritten.

Default output paths are now V2 and still reject existing files. Source SHA256 at V2 handoff is `559358ecd97438c17d24fe9adf6650f258209011ac123a2d2c8bd8d38b47f646`. The parent can run:

```bash
timeout --signal=INT --kill-after=30 1200 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_shared_pool_hf_control --threads 4 --repeats 2 --snapshot /home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/shared_pool_hf_control_v2.json --tensors /home/ttuser/qwen38-flash/reference/shared_pool_hf_control_v2.pt > models/autoports/qwen_qwen3_8_flash_next/doc/full_model/shared_pool_hf_control_v2.log 2>&1
```

V2 has 16 histories/64 forwards, versus V1's 14/56 and measured5.52 seconds/4.84GiB RSS. Similar warm-cache resource use is expected, not yet measured. No model execution was performed while preparing V2.

Checks run: lightweight `--self-test` passed; cached repository-pinned Black23.10.1, autoflake2.3.1 and isort5.13.2 checks passed. Black was also invoked directly with `--include '\.py$'` to ensure this autoport file is checked despite the repository's narrower configured include pattern. The ordinary matching pre-commit hooks also passed.

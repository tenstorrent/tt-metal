# SUMMARY (transformer family)
Note: finish_impl (dataflow_buffer.inl:274-319) drains both sides: TRISC unpack/pack wait tile_counters.posted==0; DM waits read_acked==read_posted. So any pushed-but-never-popped entry (persistent scaler/mask tiles, borrowed outputs) hangs at kernel end.
EXERCISED by tests (CONFIRMED):
- F-SP1/2/3 quasar SDPA prefill/chunked: identity_scale_in (1/0), col_identity (1/0), lightweight causal mask_in (2/0) never popped by compute.
- F-SD1/2 quasar SDPA decode/paged decode: identity_scale_in (2/0), causal mask_in (PNHt*Sk_chunk_t_dynamic / 0).
- F-RN1 rotary_embedding_llama decode-sharded compute: borrowed out pushed Ht*Wt, never popped (llama e2e decode + qwen3_vl).
- F-RN2 rotary_embedding_llama_fused_qk compute (TILE + RM): out pushed, never popped (fused_qk op tests).
NOT exercised (CONFIRMED by counts): F-KV1 row-major paged_fused_update_cache writer (index, page_table); F-RN3 sharded nlp_create_qkv_heads kv_out DM self-loop; F-RN4 sharded create_qkv_heads_decode batch_offset; F-SD3..7 decode attn-mask/sliding/block-pad/sink/sharded-out-nkv1/sharded page table.
SUSPECT: F-SP4 sliding-window writer mask-chunk count; F-SP5 subblocked Q not pushed on empty K range.
Balanced: paged_update_cache, paged_fused_update_cache tiled, paged_fill_cache, fill_cache, rotary MultiCore/PrefillSharded, nlp_create_qkv_heads(_decode) interleaved, nlp_concat_heads(_decode).

# DFB imbalance audit — family: transformer
(collect phase; analysis appended below)

## Ops called by tests/model (Quasar)
- ttnn.experimental.quasar.transformer.{scaled_dot_product_attention, chunked_scaled_dot_product_attention} -> experimental/quasar/transformer/sdpa (sdpa_program_factory.cpp)
- ttnn.experimental.quasar.transformer.{scaled_dot_product_attention_decode, paged_scaled_dot_product_attention_decode} -> experimental/quasar/transformer/sdpa_decode
- ttnn.transformer.* SDPA: only appears as captured "op" name strings; tests bind _OP = ttnn.experimental.quasar.transformer.* -> mainline transformer/sdpa NOT reached.
- ttnn.experimental.rotary_embedding_llama (3 factories) / rotary_embedding_llama_fused_qk (mainline experimental/transformer; rotary_embedding_llama in llama graph_ops is in _QUASAR_UNSUPPORTED_OPS skip list)
- ttnn.experimental.nlp_create_qkv_heads / _decode, nlp_concat_heads / _decode (mainline experimental/transformer)
- ttnn.experimental.paged_update_cache / paged_fused_update_cache / paged_fill_cache (experimental/paged_cache)
- ttnn.fill_cache (kv_cache/fill_cache_multi_core_program_factory)

---
## Sub-scope: KV cache (paged_update_cache, paged_fused_update_cache, paged_fill_cache, fill_cache)

### F-KV1 (CONFIRMED, NOT exercised by tests) — row-major paged_fused_update_cache writer never pops index / page_table
- writer_paged_row_major_fused_update_cache_interleaved_start_id.cpp:86 `dfb::index` wait_front(1), never popped; reader (reader_paged_row_major_fused_update_cache_interleaved_start_id.cpp:88/98) pushes 1. pushes=1 vs pops=0. Path: USE_INDEX_TENSOR.
- same writer :98 `dfb::page_table` wait_front(num_pages_to_read) (1 DRAM / batch_size sharded), never popped; reader pushes same at :109/:127 when update_idx != -1. pushes=N vs pops=0. Path: USE_INDEX_TENSOR && IS_PAGED_CACHE && update_idx != -1.
- Under #57646 reader finish() waits for acks forever (applies even when DFB is borrowed from sharded tensor).
- Exercised? No: selector paged_fused_update_cache_device_operation.cpp:28 picks row-major only when both inputs ROW_MAJOR; tests/ops/test_paged_fused_update_cache.py and attention_1d.py:1019 use TILE -> tiled factory.
- Fix: mirror tiled writer: `dfb_page_table.pop_front(num_pages_to_read)` after cache_id computed (paged branch); `dfb_index.pop_front(1)` after skip/else block (both paths).

### Notes (not balance bugs)
- F-KV2: tiled fused reader (reader_paged_fused_update_cache_interleaved_start_id.cpp:94,:117,:169-172) and tiled writer still use get_write_ptr/get_read_ptr as NoC targets and NoC self-loopback read; non-fused paged_update_cache has ARCH_QUASAR endpoint+offset / RISC-copy paths. Reachable on Quasar via test_paged_fused_update_cache.py.
- F-KV3: reader and writer independently read update_idx to decide skip; page_table push/pop gated on both agreeing. Both read via uncached alias -> low risk.

### Balanced (checked)
- paged_update_cache reader_update_cache_interleaved_start_id.cpp: input Wt; index 1; page_table 1 (non-skip); cache Wt/head (incl. skip).
- paged_update_cache writer_update_cache_interleaved_start_id.cpp: index 1 both paths; page_table 1 non-skip; untilized_input Wt; untilized_cache Wt/head; untilized_cache2 Wt/head; out Wt/head.
- paged_cache compute/update_cache.cpp (+ kernel_lib untilize_helpers.inl:229-269, tilize_helpers.inl:176-191): in->untilized_in Wt; per head cache->untilized_cache Wt, untilized_cache2->out Wt. ARCH_QUASAR pack_init no count change.
- paged_fused_update_cache tiled reader/writer/compute (has_work=1 always): balanced; page_table num_pages_to_read both sides.
- paged_fused_update_cache row-major compute + input/cache/untilized*/out DFBs: balanced (only index/page_table bad, F-KV1).
- paged_fill_cache reader_fill_cache_interleaved.cpp / writer_fill_cache_interleaved.cpp: Wt*num_rows both sides on all paths (skip range :161, SKIP_PAGE_TABLE_ENTRY :197); same noop early return; factory gives same num_rows/noop (:482-497, cache-hit :537-538). Quasar metadata (page_table/batch_idx/valid_seq_len) are scratchpads under ARCH_QUASAR. (Non-Quasar path has dangling reserve_back(1) at :86,:113,:140 — WH/BH only.)
- ttnn.fill_cache: kv_cache reader_fill_cache_interleaved_start_id.cpp pushes num_tiles; eltwise/unary writer_unary_interleaved_start_id_metal2.cpp pops num_pages; num_tiles==num_pages==num_blocks_per_core*Wt.
- kv_cache update_cache: not reachable from in-scope tests (not analyzed).
Coverage: 13 kernel files + kernel_lib tilize/untilize helpers.

---
## Sub-scope: Quasar SDPA prefill / chunked (experimental/quasar/transformer/sdpa)
Path on Quasar: factory forces non-streaming compute (sdpa_program_factory.cpp:420-421); kernels = reader_interleaved.cpp, writer_interleaved.cpp, compute/sdpa.cpp (+compute_common.hpp, dataflow_common.hpp, windowed_mask_gen.hpp). Causal & no mask/window -> lightweight pre-built causal mask (factory :428).
Test configs: llama SDPA causal [1,32,1024|128,64] chunks 64/128; llama chunked chunk_start_idx=0 causal 128/128; qwen3_vl case1 non-causal [1,16,12288,64] (no mask); case2 causal [1,32,4096,128] 256/256.

### F-SP1 (CONFIRMED, EXERCISED by every SDPA test) — identity_scale_in never popped
- writer_interleaved.cpp:105-109 pushes 1 (prepare_reduce_scaler, reduce_helpers_dataflow.inl:163,203); compute reduce_c compute_common.hpp:281 wait_front(1) per K chunk, never pops. pushes=1 pops=0.
### F-SP2 (CONFIRMED, EXERCISED by every SDPA test) — col_identity never popped
- writer_interleaved.cpp:110-111 pushes 1 (generate_bcast_col_scalar, generate_bcast_scalar_metal2.hpp:15,23); compute matmul_reduce compute_common.hpp:1379 waits per Q iter, never pops. pushes=1 pops=0.
### F-SP3 (CONFIRMED, EXERCISED by causal tests: llama SDPA, llama chunked, qwen3_vl case2) — lightweight causal mask_in never popped
- writer_interleaved.cpp:116-125 pushes 2 (-inf + diagonal; generate_lightweight_mask_tiles dataflow_common.hpp:612,637); compute sdpa.cpp:259 wait_front(2) "permanently fronted", never pops. pushes=2 pops=0.
Fix (all three): at end of standard branch in sdpa.cpp (after phase loop ~:329): wait_front(n) (needed for cores with no Q work) -> dummy_unpack(id) (TEN-4746 wait->pop trap) -> pop_front(n); n=1 identity_scale_in, n=1 col_identity, n=2 mask_in under `if constexpr (use_lightweight_causal_mask)`. (Streaming path sdpa.cpp:168/199 has the same leak but isn't built on Quasar.)

### F-SP4 (SUSPECT, not exercised) — sliding-window mask chunk count mismatch
- dataflow_common.hpp:1810-1823 generate_mask loops while k*Sk_chunk_t < q_high_idx, unclamped to Skt (non-causal: (q_chunk+1)*Sq_chunk_t); compute does ceil(Skt/Sk_chunk_t) (non-causal) or clamp to Skt (causal) and pops one mask chunk per K chunk. Non-causal Skt > bound -> writer under-produces (hang); causal Sq>Sk -> writer over-produces (unpopped at finish). Only with sliding_window_size. Fix: clamp writer loop to compute's K end.
### F-SP5 (SUSPECT, not exercised) — Q never pushed if K loop empty
- reader_interleaved.cpp:716-731 subblocked Q push only at first K chunk; compute pops Q every Q iter (compute_common.hpp:2402). Empty K range (windowed SDPA only) -> compute blocks. Independent of finish().

### Balanced (checked)
- q_in: Sq_chunk_t*DHt per Q iter both sides.
- k_in / v_in: same K-loop bounds (causal clamp to Skt); KV_CHAIN forward 1 reserve/push per chunk.
- qk_im (self-loop): matmul push / mask re-reserve / in-place exp / pop :2205 match.
- out_im_A/out_im_B (self-loop): net zero per Q iter.
- max_A/max_B (self-loop): Sq per chunk pushed; (n_k-1)*Sq popped in loop + Sq at :2396.
- sum_A (self-loop): per-chunk push, merge (push Sq, pop 2Sq), in-place reduce/recip, final pop net zero.
- exp_max_diff (self-loop): sub_exp_block push Sq, mul_block_bcast_cols pop Sq.
- out: compute push out_chunk_tiles/Q iter; write_block dataflow_common.hpp:1606/1622 pops same.
- mask_in other variants: padded non-causal (1 chunk/Q iter, consumed last K chunk); user mask (1 chunk/K chunk); windowed — balanced.
- windowed_k_range: 1/Q iter. chunk_start_idx_compute/_writer: 1 each (tensor chunk_start mode only). attention_sink: Sq_chunk_t/Q iter.
- Placeholder aliases (mask_in->q_in, sum_B->sum_A, recip_scratch, chunk_start): compiled-out paths only.
Not reachable on Quasar: compute_streaming.hpp, joint_sdpa.cpp, joint_reader.cpp, joint_writer.cpp.
Coverage: 3 kernels + 4 headers; 15 DFBs, 12 balanced.

---
## Sub-scope: rotary_embedding_llama(+fused_qk), nlp_create_qkv_heads(+decode), nlp_concat_heads(+decode)
Paths relative to ttnn/cpp/ttnn/operations/experimental/transformer.
Factory selection on Quasar:
- rotary_embedding_llama (device_operation.cpp:17-29): decode -> MultiCoreSharded (qwen3_vl test_rotary_embedding_llama.py HEIGHT_SHARDED [32,128] 1 core; llama e2e decode non-fused attention_1d.py:997/1000; debug_ops/test_quasar_rope_device.py; llama graph_ops case skipped). Prefill interleaved -> MultiCore (tests/ops/test_rotary_embedding_llama.py, attention_1d.py:491/499). PrefillSharded only if cos/trans_mat HEIGHT_SHARDED (no test found).
- fused_qk: TILE -> compute/rotary_embedding_llama_sharded.cpp; RM -> _sharded_row_major.cpp. llama ops/ & prototype_ops/ fused_qk tests (TILE); not e2e.
- nlp_create_qkv_heads: all tests INTERLEAVED DRAM, transpose_k_heads=False -> Interleaved factory.
- nlp_create_qkv_heads_decode: INTERLEAVED L1 -> Interleaved factory.
- nlp_concat_heads: interleaved reader + eltwise/unary writer_unary_interleaved_start_id_metal2.cpp.
- nlp_concat_heads_decode: default factory.

### F-RN1 (CONFIRMED, EXERCISED: llama e2e decode + qwen3_vl rotary test) — decode-sharded RoPE `out` pushed, never popped
- rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama_sharded.cpp: out reserve :93, pushed via ckl::add (bulk_output, PushPolicy::AtEnd) :165; no wait/pop. Factory self-loops all DFBs on compute (rotary_embedding_llama_sharded_program_factory.cpp:148-162), OUT_DFB borrowed from output (:123), no writer.
- Counts: pushes=Ht*Wt, pops=0 (qwen3_vl: Ht=1,Wt=4 -> 4 vs 0). TRISC pack finish() drain spins on posted!=0.
- Fix: after ht loop `out_dfb_obj.wait_front(Ht*Wt); out_dfb_obj.pop_front(Ht*Wt);` (single end pop safer than per-ht due to ring wrap; data stays in borrowed shard).
- Others in kernel balanced: trans_mat 1/1, sin/cos Wt/Wt (popped :170-171), input Wt/Wt per ht, rotated/sin_interm/cos_interm Wt/Wt per ht.

### F-RN2 (CONFIRMED, EXERCISED by fused_qk op tests; not e2e) — fused_qk `out` (q_out/k_out) pushed, never popped
- rotary_embedding_llama_fused_qk/device/kernels/compute/rotary_embedding_llama_sharded.cpp reserve :79, push Wt/ht :142; row-major variant rotary_embedding_llama_sharded_row_major.cpp :75/:121. Self-looped & borrowed (fused_qk_program_factory.cpp:112, 174-183).
- Counts: pushes=Ht*Wt (q_Ht or k_Ht), pops=0. Fix: same as F-RN1.
- Balanced: in Wt/Wt; rotated/sin_interm/cos_interm Wt/Wt; cos/sin/trans_mat no FIFO calls. (Separately lacks dummy_pack/pack_init Quasar handling of decode-sharded kernel — not a count issue.)

### F-RN3 (CONFIRMED by counts, NOT exercised — tests use interleaved) — sharded nlp_create_qkv_heads `kv_out` DM self-loop push-only
- nlp_create_qkv_heads/device/kernels/dataflow/reader_tm_tile_layout_nlp_create_qkv_heads_sharded.cpp reserve :107, push :135, no pop; K_OUT/V_OUT bound PRODUCER+CONSUMER at nlp_create_qkv_heads_program_factory.cpp:785-794. posted=num_kv_tiles, acked=0 -> DM finish spins. (Gen2 also rejects DM self-loops.)
- Fix: drop reserve/push and use get_write_ptr like q_out, or add wait_front/pop_front(num_kv_tiles).

### F-RN4 (CONFIRMED by counts, NOT exercised — needs batch_offset + sharded input) — create_qkv_heads_decode `batch_offset` push-only
- nlp_create_qkv_heads_decode/device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:47/52 and ..._on_subcoregrids.cpp:46/51 under USE_BATCH_OFFSET reserve/push 1, no pop; bound PRODUCER+CONSUMER on DM (nlp_create_qkv_heads_decode_sharded_program_factory.cpp:192-200). 1 vs 0.
- Fix: wait_front(1); pop_front(1) after reading index_ptr[0], or make it a Scratchpad like the interleaved Quasar path.

### Balanced (checked)
- rotary MultiCore (reader_rotary_embedding_llama_interleaved_start_id.cpp / compute rotary_embedding_llama.cpp / writer_rotary_embedding_llama_interleaved_start_id.cpp): trans_mat 1/1; input Wt/row; sin/cos per row (RELOAD=1) or reserve(my_cos_sin_tiles)/batch pushed in Wt chunks + bulk pop/batch (RELOAD=0); out writer waits only when seq_tile < rotary_Ht matching compute; intermediates Wt/Wt; idle cores trans_mat 1/1; zero region Scratchpad.
- rotary PrefillSharded (reader_rotary_embedding_llama_prefill_sharded.cpp + shared compute/writer): sharded cos/sin forces RELOAD=1, Wt/row; borrowed trans_mat 1/1.
- nlp_create_qkv_heads Interleaved reader/writer: head_parallel num_blocks*head_tiles vs num_blocks*q_out_w_tiles; else (Hq+2Hkv)*w per block both sides; KV_TIED rewinds tile id only. TRANSPOSE_K_HEADS transpose_wh_metal2.cpp balanced (not exercised).
- nlp_create_qkv_heads_decode Interleaved reader: no FIFO calls (write ptr only; Quasar aligned scratch = Scratchpad).
- nlp_concat_heads interleaved reader (ARCH_QUASAR 1 tile at a time) vs writer_unary metal2: num_blocks*in0_c*in0_w_tiles == num_pages.
- nlp_concat_heads sharded reader: no FIFO calls.
- nlp_concat_heads_decode (default + subcoregrid): no FIFO calls (dfb_q_out get_write_ptr only).
Coverage: 14 kernels, 4 imbalanced, 10 balanced.

---
## Sub-scope: Quasar SDPA decode / paged decode (experimental/quasar/transformer/sdpa_decode)
Files: sdpa_decode_program_factory.cpp (DFB specs ~594-836), kernels/dataflow/{reader_decode_all.cpp, writer_decode_all.cpp, dataflow_common.hpp}, kernels/compute/{sdpa_flash_decode.cpp, compute_common.hpp}; ttnn/kernel_lib/reduce_helpers_dataflow.inl, l1_helpers.hpp.
Test config (llama graph_ops paged decode + small_grid, ops/test_scaled_dot_product_attention_decode.py, qwen3_vl paged decode, attention_1d.py:836/852): causal (DRAM cur_pos_tensor), k_chunk_size=0 -> DYNAMIC_CHUNK_SIZE, Q TILE HEIGHT_SHARDED 1 core (no Q_LOCALLY_AVAILABLE / TILIZE_Q), K/V DRAM interleaved block 32, page table DRAM (scratchpad), DRAM out, no sink/window/mask, nkv=8 on 8x8/8x4 -> tree reduce (HAS_INTERMED_OUT); split small-grid max_cores_per_head_batch=1.

### F-SD1 (CONFIRMED, EXERCISED by every test) — identity_scale_in pushed 2, popped 0
- writer_decode_all.cpp:228 (reduce scaler) + :234 (zero tile) push 1 each; compute only waits: reduce_c compute_common.hpp:209, :286; matmul_blocks mask-fusion :1320. Writer finish() waits for 2 acks forever.
- Fix: end of compute next to q_in pop (sdpa_flash_decode.cpp:800): wait_front(2); dummy_unpack(identity_scale_in); pop_front(2) (reached only past has_local_data return = same cores the writer produced on).

### F-SD2 (CONFIRMED, EXERCISED by every test, causal) — causal mask_in pushed once, never popped
- writer generate_mask writer_decode_all.cpp:254 -> dataflow_common.hpp:264/309 pushes PNHt*Sk_chunk_t_dynamic once per launch on every core with local data. Compute: fused path (DYNAMIC_CHUNK_SIZE) matmul_blocks waits only (compute_common.hpp:1317); non-fused add_block_inplace<false> sdpa_flash_decode.cpp:476; only root waits at last chunk, non-root workers never touch it.
- Fix: end of compute under #ifdef IS_CAUSAL: wait_front(Sq_chunk_t*Sk_chunk_t_dynamic); dummy_unpack(dfb_mask_in); pop_front(same). Once per launch (writer generates once even if num_heads_per_core>1).

### F-SD3 (CONFIRMED, not exercised) — USE_ATTENTION_MASK + DYNAMIC_CHUNK_SIZE: mask_in never popped
- reader pushes mask_chunk_tiles per chunk per head (dataflow_common.hpp:193/211); add_mask_fusion true (sdpa_flash_decode.cpp:425) -> matmul_blocks waits only, popping add_block_inplace<true> :481 skipped. Also correctness bug (stale mask after chunk 0). Fix: after QK matmul_blocks when add_mask_fusion && use_attention_mask pop_front(qk_chunk_tiles_dynamic).
### F-SD4 (CONFIRMED, not exercised) — sliding_window_mask_in / block_pad_mask pushed once, never popped
- writer :240 / :248; compute waits only (sliding: sdpa_flash_decode.cpp:489 or compute_common.hpp:1317; block pad: :318, :468). Fix: end-of-compute wait/dummy_unpack/pop same counts; sliding guarded like writer (k_chunk_start == window_start_chunk && window_start_unaligned > 0).
### F-SD5 (CONFIRMED, not exercised) — attention_sink pushed PNHt once, never popped
- reader_decode_all.cpp:285/301; compute only reads via max_block (:729), sub_exp_block (:741). Fix: end-of-compute pop_front(Sq_chunk_t).
### F-SD6 (CONFIRMED, not exercised) — sharded output with num_kv_heads==1: out never popped
- compute pushes out_chunk_tiles on root (:780); writer wait/pop gated on num_kv_heads>1 || !is_out_sharded (writer_decode_all.cpp:429-431, 516-519). Fix: make writer wait_front+pop_front(out_chunk_tiles) unconditional (handshake only).
### F-SD7 (CONFIRMED, not exercised) — sharded page table: reader self-loop push-only
- reader_decode_all.cpp:328/332 reserve_back(B)+push_back(B); reader bound PRODUCER+CONSUMER (factory:731-738); no wait/pop. Fix: wait_front(B)+pop_front(B) after head loop, or scratchpad like non-sharded path.

### prev_max "double-pop" suspicion (memory note, ~:758) — NOT A BUG
- Now at sdpa_flash_decode.cpp:766. prev_max holds Sq after loop; move_block :610 pushes Sq per chunk, :576 pops previous; tree round pops Sq :695 / pushes Sq :697; root :766 drains last Sq; sink branch :743 pops cur_max (filled by max_block); non-root drains via move to out_m :793. Comment at :742 is stale (says prev/front, pops cur_max).

### Balanced (checked)
- q_in: read_q push 1x; compute wait :312 / pop :800; same early returns (idle, cur_pos==UINT32_MAX, no local data) before Q traffic; Q_LOCALLY_AVAILABLE borrow balanced.
- q_rm (TILIZE_Q): reader push / compute tilize helper consume.
- k_in / v_in: per chunk per head on all reader paths (read_k sender/receiver/no-mcast, read_v, read_kv_mask_chunks, reuse_k); compute pops in matmul_blocks compute_common.hpp:1348.
- writer_cur_pos / compute_cur_pos: 1/1 each, before skip return.
- m_in, l_in, out_o: writer pushes per active child round; compute pops l_in :657, m_in :696, out_o :683; same get_workload_for_core.
- out_worker, out_m, out_l: compute push on non-root (:791-795); writer send path pops (:370-414).
- out (DRAM, or sharded nkv>1): balanced.
- compute-only qk_im, out_im, out_accumulate_im, max_1, max_2, sum, exp_max_diff (+Quasar aliases): net 0 on root/non-root/single-core.
- mask_in non-causal no mask: untouched. intermed_out & non-sharded page table: scratchpads.

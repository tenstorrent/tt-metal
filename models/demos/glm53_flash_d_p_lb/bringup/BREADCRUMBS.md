# zai-org/GLM-5.3-Flash bring-up on mesh 2x4: breadcrumbs

Prior bring-up: glm53_flash_d_p (layers 0-4, mesh 2x2); CPU reference and model code shared. This one runs ALL 45
text layers on the 8x Blackhole p150b LoudBox. Append-only log, one section per task attempt: what was done,
decisions and why, gotchas, the re-run command, the verdict.

## Whole model on the LoudBox (2026-10-08)

What was done:
- Spec written by `new --prior glm53_flash_d_p --mesh 2,4`, then edited: layers 0-44, box 8x p150b, own golden dir
  (`paths.golden`: the prior's s4096 golden only has layers 0-4), `device.experts_dtype: bfloat4_b`,
  test_timeout_s 14400 (the first load converts the bfp4 expert cache, ~1 h). Hooks re-export the prior's hooks.
- glm53_flash_d_p model code made mesh-generic (2x2 unchanged, bit for bit):
  - tt/experts.py: the owner-rule assert against bfp4 removed; dtype passed in (build_experts weights_dtype, from the
    spec via hooks.experts_dtype; default bfp8); expert tensorbin cache keyed by mesh shape for meshes other than 2x2
    (experts/2x4/...: each tensorbin stacks one local slot over the whole mesh, so a 2x2 cache is wrong on 2x4).
  - tt/mla_attention.py, tt/indexer.py: the 2x2 asserts removed; the per-chip query-row tables reshape to
    (rows, cols, S / n, W) from mesh.shape instead of (2, 2, ...).
  - tt/model.py: experts_dtype threaded through TtGlmModel -> TtGlmBlock -> build_experts.
  Everything else already took its sizes from mesh.shape: KDA SP = rows (axis 0), TP = cols (axis 1, 16 heads per
  chip on 2x4); dense MLP / shared expert TP = 8; experts EP = 8 (36 per chip, dispatch on axis 0, group size 2,
  4 dispatch groups); the split residual (chip r C + c holds rows (r C + c) S/8 ..).

Decisions:
- Mesh 2x4 (the p150_x8 descriptor's native shape), not 4x2: KDA TP = 4 quarters the KDA weights (34 layers x
  ~270 MB bf16); experts keep the 2x2 dispatch geometry (axis 0, group size 2).
- bfp4 experts (owner): 4.08 GB per MoE layer / 8 chips = 0.51 GB per chip x 42 = 21.4 GB. bfp8 would be ~40 GB per
  chip. Everything else is unchanged from 2x2 (bf16, replicated DSA weights and caches, replicated embedding).

Results:
- Layers 0-4, 2x4, bfp8 experts, against the prior's s4096 golden: worst layer 0.999963, worst state 0.999158 (2x2:
  0.99995 / 0.99916). The generalised code is correct on 2x4.
- Layers 0-4, 2x4, bfp4 experts: worst layer 0.997897 (layer 4), state 0.999158 (experts do not touch the state).
- All 45 layers, 2x4, bfp4 experts, ladder s4096 (2 x 2048, golden from this spec, CPU top1 0.964 / 0.970): it fits
  and runs. Chunks 4.18 s (first) / 2.92 s (warm), host transfers per layer 0. top1_match 0.9684, top5_overlap 1.0,
  text_top5_acc_last_chunk 0.9985. Per-layer PCC: L0-2 1.0000, L3 0.9983, L4 0.9979, then a smooth drift
  (~0.0005 per MoE layer, partial recoveries at 24 and 34) to L40 0.9742, L41 0.9695, L42 0.9651, L43 0.9600,
  L44 0.9520; final hidden 0.9475; state min 0.9654 (kda_conv L44; kv_latent L43 0.9694). The gate FAILS on the
  thresholds tuned for the bfp8 subset (layer / state 0.97, final_hidden 0.97): L41-44, final hidden, state.
  The drift is gradual quantisation noise from bfp4 (L3 / L4 equal the 5-layer bfp4 run), not a mapping bug.
- Load: 72 min the first time (bfp4 cache conversion ~70 s per MoE layer, 3.8 GB per layer on disk, 160 GB total
  under generated/glm53_flash_d_p/tt_cache/experts/2x4).

- Attribution (same code / mesh, only the expert dtype differs): layers 0-14 on 2x4 with bfp8 experts
  (GLM_EXPERTS_CACHE=0: converted straight to the device, no 92 GB bfp8 cache) against this spec's full golden:
  PASS, L3 0.99997, L7 0.99991, L11 0.99985, L14 0.99979, state min 0.99916 (bfp4 at the same layers: 0.9983,
  0.9955, 0.9939, 0.9930). 12 MoE layers (3 of them DSA) lose 0.0002 at bfp8 vs 0.007 at bfp4 (~35x): the 2x4
  mapping is correct, and the whole-model gate failure is bfp4 precision. Extrapolated, bfp8 would end near 0.999 at
  L44, but bfp8 does not fit on 8 chips.
- Device smoke (tests/test_smoke.py, all 45 layers, bfp4, greedy): 'What is the capital of France? Answer in one
  word.' -> '</think>Paris' (the CPU reference's answer): PASS. With the expert cache built, load + smoke = 2 min 24 s.

Gotchas:
- KMD 2.9 on this box: uploading mmap-backed tensors (the ttnn tensorbin cache) spins forever in to_device. Always run
  with TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0.
- The up-front collect pass (default in run_safe_pytest.sh) loads the model twice and prints a fake "worst layer pcc
  0.000000"; use --no-precompile.

Re-run:
  TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONPATH=$PWD BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \
    BRINGUP_RUNG=s4096 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py
  (golden: python -m models.demos.common.bringup.reference.generate_golden --spec <spec> --rung s4096, 31 min, 30 GB)

## Prefill performance, whole model (2026-10-08, tests/test_perf.py)

Canonical prompt, 56320 tokens in 5120-token chunks, 2x4, bfp4 experts, eager (no trace), wall time with one sync
per chunk (host dispatch included). Load from the cache 51 s.
- Cold full prefill 44.3 s (first chunk 2.4 s, later chunks ~4 s each: new programs per chunk position).
- Warm full prefill 17.77 s = 3170 tok/s; chunk 1521 ms at 0 rising to 1688 ms at 51200 (indexer scores grow
  with the context).
- Per-layer, synced, chunk at 51200 (1755 ms): dsa_moe 60.9 ms/layer (11 layers, 670 ms, 38%), kda_moe
  32.4 ms/layer (31 layers, 1004 ms, 57%), kda_dense 19.1 ms/layer (3 layers, 57 ms).
- Last chunk next-token vs the text: top1 0.659, top5 0.876 (the 4k golden's CPU top1 is 0.964; no 56k CPU
  baseline yet, so not attributable).

## Accuracy in depth (2026-10-08)

Long context vs the text (tests/test_accuracy.py: LM head on the device, vocab-sharded, top-5 per chip merged on the
host; agrees 100% with the host fp32 head on sampled rows). 56320 tokens in 5120 chunks, bfp4: top1 per chunk
0.951, 0.929, 0.924, 0.920, 0.915, 0.903, 0.886, 0.863, 0.793, 0.751, 0.705 (all rows 0.867); top5 0.993 -> 0.898.
CPU reference (s56320 golden, in progress) chunk 0: top1 0.967 / top5 0.994.
Chunk-boundary check, 49152 tokens: 8192-token chunks top1 0.896, 2048-token chunks 0.891 (4x the boundaries,
-0.5 pt): the decay with position is not state hand-off between chunks.
DRAM after load: 29.04 GiB allocated, 2.79 GiB free per chip.

Magnitude through the layers (tests/test_inflation.py, s4096 golden, every captured step vs the CPU boundary):
no inflation. Residual norm ratio stays within 0.991-1.017 of the reference over all 45 layers; per-stream gains
track. Gain <dev, ref>/<ref, ref> 1.00 +- 0.006 through L33, 0.944 at L44 with norm 0.991 and rel 0.31
(gain^2 + rel^2 ~= norm^2: noise displacing signal under the norm-preserving RMSNorm / Sinkhorn, not a scale bias).
Error source: at L3 ffn_in rel 0.008 -> experts_out rel 0.098 (gain 1.008), shared expert (bf16) 0.005: bfp4
experts add ~10% unbiased relative error per MoE layer; residual rel 0.06 (L3) -> 0.10 (L10) -> 0.18 (L20) ->
0.31 (L44), fastest in the last layers where the reference residual RMS grows 0.04 -> 1.04.

## Fidelity and links (2026-10-08), now the spec defaults (device.experts_fidelity / attn_fidelity / moe_links)

Profile (Tracy + tt-perf-report, layers 2-4, chunk at 51200): device-bound (per chip 18.3 / 59.4 / 29.4 ms per
kda_dense / dsa_moe / kda_moe layer, ~1.62 s per chunk vs 1.69 s wall). Largest: MoE dispatch+combine 20% (1 link,
3-5 cores), matmuls 18%, AG/RS 18%, routed experts 16% (~30% of the HiFi4 roofline), indexer fp32 eltwise ~10%.
Knobs (env, read at construction; hooks.apply_device_settings sets them from the spec unless already set):
GLM_EXPERTS_FIDELITY, GLM_ATTN_FIDELITY (MLA, sparse SDPA, indexer, q_a, KDA incl. ttKDA's internal configs),
GLM_MOE_LINKS (TtExperts num_links: was 1, inherited from the MiMo 2x2 port; the LoudBox has 2 per axis).
Warm 56k prefill / 56k top1 vs text (all rows; last chunk top1 / top5):
  baseline HiFi4, 1 link        17.77 s  0.867  0.705 / 0.898
  experts LoFi                  15.93 s
  + 2 MoE links                 14.05 s  0.874  0.725 / 0.914
  + attention HiFi2             13.79 s  0.881  0.767 / 0.938   <- defaults
Lower fidelity is *more* accurate here (LoFi should equal HiFi4 on bfp4 weights, yet outputs differ): a HiFi4-path
precision issue to find (split KDA vs MLA/indexer HiFi2 next). CPU reference per chunk (s56320 golden): 0.967,
0.953, 0.955, 0.949, 0.948, 0.937, 0.933 at 0..30k: the device's position decay is device error, not the model.

## MiMo all-gather MoE ops on GLM (2026-10-08, GLM_EXPERTS_MODE=ag, tt/experts_ag.py)

What was done:
- New experts mode `ag` (glm53_flash_d_p/tt/experts_ag.py, port of MiMo's MoeAgBlock): fabric_all_gather of x /
  top-k over axis 0 (split layout; replicated input needs no gather) -> moe_ag_route_plan -> flat_routed_expert in
  indexed mode (clamped_silu, bfp4, row-major y) -> moe_ag_local_reduce with fused send-back (fabric_all_gather axis 0)
  -> fabric_reduce_scatter axis 1 (split) / reduce_scatter + all_gather (replicated). Own laid-out weight cache under
  generated/glm53_flash_d_p/tt_cache/flat/2x4 (~6 GB per MoE layer, ~250 GB, ~75 min to build from fp8).
- flat_routed_expert: new opt-in `down_fp32` (fp32 DEST, full-sync DST for the down projection; the Python builder's
  MIMO_FL_DN_ACC=fp32full, missing from the C++ op, so MiMo itself runs bf16 down). On by default in `ag`
  (GLM_AG_DOWN_FP32). fabric_all_gather semaphores must be L1_SMALL.
- Tests: tests/test_experts_ag.py (ag vs unified on the layer-4 golden, both layouts, timing), test_flat_scale_probe.py
  (one chip, random weights), test_flat_diag.py (flat y vs CPU math on the same bfp4 weights; reductions).

Results:
- Layer 4 (chunk 2048, bfp4): split call 2.34 ms (unified 5.56 ms). vs golden PCC 0.9883 (unified 0.9884), rel L2
  0.160 (0.154), scale coefficient 1.024 (unified 1.003 at LoFi, 1.012 at HiFi4). bf16 down: 1.048.
- Gain breakdown (layer 4, chip 0, 4 experts): bfp4 weights themselves +1.05% (CPU fp32 math, quantized vs exact);
  flat y vs CPU math on the quantized weights +1.0% (not reproduced by bfp8 x / h rounded on the host);
  bf16 reductions +0.1%. MiMo's own sweep (expert_precision_results.tsv, flatpy_dnfp32full) shows the same ~+1-1.5%.
- Ladder s4096 (all 45 layers): ag top1_match 0.9747, L44 0.9492, final hidden 0.9442, state 0.9666; unified (same
  day, spec defaults) 0.9684, 0.9520, 0.9472, 0.9676. Both fail the same gates (L41-44, final hidden, state: bfp4).
- Warm 56k prefill (test_perf): ag 10.99 s (5124 tok/s; kda_moe 18.6, dsa_moe 46.7 ms/layer) vs unified 13.77 s.
- 56k all-row top-1 vs text (test_accuracy): unified 0.8814, ag 0.8701, ag with routing weights x 0.979
  (GLM_AG_SCALE, experimental, default off) 0.8810.

Open: the remaining ~2% gain (flat +1% and the bfp4 weights' +1%, which unified at LoFi appears to cancel) is not
explained. Next steps proposed: weight bits held by each path vs bfp4(fp32) / bfp4(bf16(fp32)) (test_weight_bits.py,
not run yet), a plain-matmul LoFi / HiFi probe on real weights, per-stage error on the exact device bits.

### Follow-up (2026-10-08): root cause of the ag gain, KV-cache PCC, MoE-FFN dedupe
- Precision analysis (tests/test_weight_bits.py, test_flat_diag.py, test_flat_stage_probe.py, test_bfp8_rounding.py):
  unified holds bfp4(fp32 W) bit for bit, the flat op bfp4(bf16(W)) (0.03-0.23% of elements one step smaller: -0.24%
  on the output, not the gain). The flat expert is +1.25% vs fp32 math on its own bits; the cause is the packer
  rounding bf16 -> bfp8 ties away from zero (+0.28% per pack on bf16-valued data, ~10% of elements; fp32 sources
  are unbiased) at the x tilize, the h pack and the y pack. LoFi = HiFi4 for bfp4 x bfp8. Unified at LoFi sits
  -0.7% vs its own bits (bf16 operands truncated), cancelling the bfp4 weights' own +1.05%. Packer stochastic
  rounding (flat op option pack_stochastic_rounding, GLM_AG_PACK_SRND) removes it (coef 1.0005) but is NOT used
  (owner); the 0.979 routing scale was removed. Default ag: fp32 full-sync down only.
- KV cache PCC, s4096 ladder (owner's metric), ag vs unified: kv_latent min 0.9666 / mean 0.9840 vs 0.9693 / 0.9853;
  index_key 0.9881 / 0.9951 vs 0.9887 / 0.9953; layers and top1 as before (L44 0.9492 vs 0.9520).
- MoE FFN (ag only): the router hands idx / weights straight to the experts (no dense scatter + top-k); the experts
  gather x as tiles and the shared expert reuses that gather (no second axis-0 all_gather). Warm 56k prefill
  10.99 -> 10.74 s (5246 tok/s; unified 13.77 s). Tried and reverted: one fp32 reduce_scatter for shared + routed
  partials (no gain, 10.76 s).
- Per-op device time, chunk 5120 at 51200 (test_profile_ops GLM_PROF_LAYERS=2-4): kda_moe 18.7 ms (attention 8.5,
  experts 3.9, shared 3.8), dsa_moe 45.7 ms (indexer 24.8). ag block 4.05 ms (flat 1.96, two 5120x4096 gathers
  0.95, local reduce 0.49, fabric RS 0.35) vs unified 9.57 ms (dispatch 2.50, expert 3.22, combine 2.14).
- Tried and reverted: fabric_all_gather for the model's other row gathers (gather_half, gather_rows, KDA input,
  shared expert): 10.73 s vs 10.74 s, no gain (ttnn.all_gather is as fast at these shapes). Needs the tensors'
  topology re-declared 2D (split-layout tensors carry a 1D one), as MiMo's gather_full does.

## Indexer score: one fused op (2026-10-09), GLM_INDEXER_SCORE=bringup (default; "heads" kept)

- Before: score = sum_h w_h relu(q_h k^T) as 32 per-head ttnn.linear (fp32 [640, 14080] out, 0.415 ms each) + 32 fp32
  addcmul (0.277 ms each): 22 of the indexer's 24.8 ms at 51200, all DRAM traffic. ttnn.experimental.indexer_score_dsa
  ("op") does it in one op but only with bf16 DEST (head sum truncates), hence "heads".
- Now: ttnn.bringup.indexer_score_dsa (the indexer_score fork, hy4's) with fp32 DEST, k chunk 32 (per-column gate
  multiply, fidelity honoured), q chunk 64, bf16 gates: 0.434 ms; indexer 24.75 -> 2.79 ms per dsa layer.
- Component test L3 (chunk 1): overlap 0.99830 / worst row 0.9902 (heads 0.99832 / 0.9922). Both modes fail the
  same pooled-key row-norm check (ratio 0.9911 < 0.995, before the score step: pre-existing, unrelated).
- Warm 56k prefill 10.74 -> 9.34 s (6029 tok/s). s4096 KV PCC min/mean kv_latent 0.96674 / 0.98401 (was 0.96656 /
  0.98396), index_key 0.98806 / 0.99516 (0.98811 / 0.99510), kda_conv min 0.9653 (0.9599). 56k top1 0.8767 (ag
  before 0.8701, unified 0.8814).
- Indexer projections at HiFi4 (GLM_INDEXER_FIDELITY=HiFi4, opt-in): fixes the L3 component test's pooled-key gate
  (rel L2 0.0050 -> 0.0025, row-norm ratio min 0.9911 -> 0.9976; the HiFi2 failure predates the fused score). Full
  model: perf same (9.34 s), KV PCC same within 1e-4 (kv_latent 0.96660 / 0.98397, index_key 0.98760 / 0.99511),
  56k top1 0.8696 (HiFi2 0.8767; last chunk 0.7136 vs 0.7416). Left at the attention fidelity (HiFi2) by default.

## MLP / shared-expert reduce-scatter: MiMo fabric_reduce_scatter in bf16 (2026-10-09), GLM_SCATTER_OP=fabric_bf16 default

- scatter_rows (tt/common.py; dense MLP and shared expert outputs) was two fp32 ttnn.reduce_scatter (axis 0 + 1,
  ~1.8 ms per MoE layer). Now: typecast to bf16 + ttnn.bringup.fabric_reduce_scatter on axis 0 then 1.
- Warm 56k prefill 9.34 -> 8.95 s (6290 tok/s). s4096 KV PCC min/mean kv_latent 0.96653 / 0.98409 (fp32 0.96674 /
  0.98401), index_key 0.98813 / 0.99491 (0.98806 / 0.99516), kda_conv min 0.9620 (0.9653); final hidden 0.9445 both.
  56k top1 0.8804 (fp32 0.8767, unified 0.8814). GLM_SCATTER_OP=ttnn restores the fp32 path.

## KDA output: one MiMo fabric_reduce_scatter over rows (2026-10-09), GLM_KDA_OUT_RS=fabric default

- Split layout: ttKDA's o_proj + fp32 reduce_scatter_minimal_async (hidden dim) + fp32 all_gather (hidden) +
  mesh_partition (rows) is an all-reduce cut to a quarter of the rows. Now _GlmKDA returns o_proj's bf16 partial and
  TtKdaAttention reduces it with ttnn.bringup.fabric_reduce_scatter over rows on axis 1 (straight to the quarter).
- Warm 56k prefill 8.96 -> 8.47 s (6648 tok/s). s4096 KV PCC min/mean kv_latent 0.96713 / 0.98405, index_key 0.98798 /
  0.99491 (before 0.96653 / 0.98409, 0.98813 / 0.99491); final hidden 0.9445 (same).
- Pre-existing, found on the way: KDA at HiFi2 (attention fidelity) gives a ~3.4% low attention output on every row
  (L0 component test rel L2 0.0345, norm ratio 0.960..0.979, fails 0.02); HiFi4 0.0071, 0.990..0.998 (passes).
  GLM_KDA_FIDELITY overrides the KDA fidelity alone (default: GLM_ATTN_FIDELITY).
- KDA at HiFi4 (GLM_KDA_FIDELITY default HiFi4; the rest of attention stays HiFi2): warm 56k prefill 8.47 -> 8.62 s.
  s4096 KV PCC min/mean kv_latent 0.96868 / 0.98509 (HiFi2 0.96713 / 0.98405; unified 0.96928 / 0.98530),
  index_key 0.98867 / 0.99546 (0.98798 / 0.99491), kda_recurrent 0.97493 / 0.99403 (0.97242 / 0.99320); final hidden
  0.9465 (0.9445), L44 0.9508 (0.9492).

## Fabric packet payload 14400 B (2026-10-09), spec box.device_params.fabric_payload_bytes

- The spec had no router config: the fabric ran at the router default, 4352 B per packet (2 bf16 tiles). MiMo opens
  its mesh at 14400 B (14 KiB + 64). The ag path has no dispatch / combine (the reason the DeepSeek-family configs size
  the payload to a token row): every collective moves tile pages, floor(payload / page) per packet, so 14400 B = 7 bf16
  / 3 fp32 tiles. harness.device_params builds the FabricRouterConfig; BRINGUP_FABRIC_PAYLOAD overrides it.
- Warm 56k prefill 8.62 -> 8.26 s (6817 tok/s, KDA HiFi4). s4096 ladder identical (data movement only).
- Payload sweep (warm 56k prefill, KDA HiFi4): 4352 8.62 s, 6144 8.27, 8192 8.07, 10240 8.07, 12288 8.06, 14400 8.26,
  15232 8.28. Plateau 8192..12288; the larger packets lose (fewer packets in flight per router channel). Spec: 8192 =
  one bf16 token row (4096 x 2 B), as the DeepSeek-family configs size it to the dispatched row; 4 bf16 / 2 fp32 tiles.

## Matmuls: packer L1 accumulation (2026-10-09), tt/common.py mm_config (GLM_MM_L1ACC=0 restores)

- No GLM matmul had a tuned config; hifi4_config set packer_l1_acc=False for everything. tests/test_matmul_tune.py
  (one chip, the model's shapes / dtypes / fidelity): packer_l1_acc on gives identical results vs fp32 torch (same rel
  and scale on every shape) and ttnn.linear 1.1..2.4x faster: MLA o_proj 640x16384x4096 1.86 -> 0.76 ms, q_b 0.53 ->
  0.27, kv_a 0.43 -> 0.34, shared expert gate 0.34 -> 0.30. Applied to every ttnn.linear / matmul in mla_attention,
  indexer, mlp, q_a, router (norms, SDPA, score op unchanged).
- Warm 56k prefill 8.07 -> 7.66 s (7355 tok/s). s4096 KV PCC min/mean kv_latent 0.96876 / 0.98514, index_key 0.98899 /
  0.99550 (before 0.96868 / 0.98509, 0.98867 / 0.99546).
- Also from the sweep: HiFi2 matmuls are 0.28% low (scale 0.99722 vs fp32; HiFi4 1.00004) - MLA q_b / kv_a / o_proj
  run at the attention fidelity (HiFi2). minimal_matmul: another 1.2..1.7x at bf16 out, but fp32 out is 0.03% low
  (0.99969) and fp32 input fails; not used yet.
- All attention at HiFi4 (GLM_ATTN_FIDELITY=HiFi4; KDA already HiFi4): warm 56k prefill 7.66 -> 7.76 s, s4096 KV PCC
  within noise (kv_latent 0.96883 / 0.98505 vs 0.96876 / 0.98514, index_key 0.98929 / 0.99562 vs 0.98899 / 0.99550,
  final hidden 0.9465 vs 0.9460). The MLA / indexer HiFi2 bias (0.28%) does not reach the caches; kept at HiFi2.

## Explicit matmul configs (2026-10-09), tt/mm_configs.py (GLM_MM_CONFIGS=0: ttnn auto)

- tests/test_matmul_tune.py (one chip, every model matmul shape, math utilization vs the fidelity peak, fp32-truth
  error): explicit configs beat the auto picks on every shape, identical numerics. 2D multicast with MiMo's rule (M over
  the 10 grid rows, ceil(Nt / 11) N tiles per core, widest in0 block within L1) for every ttnn.linear (MLA kv_a / o_proj,
  indexer, shared expert / MLP, q_a, router; KDA o_proj via _GlmKDA); minimal_matmul blockings for KDA in-proj (M4 K8 N4:
  1.511 -> 1.260 ms, 91% of HiFi4) and MLA q_b. Shared expert gate + up fused into one N=512 matmul (2 x 0.297 ->
  0.261 ms); its down projection writes bf16 when the bf16 reduce-scatter follows (output-write bound, K=256).
- L1 budget 1.2 MB per core: 1.4 MB clashed with the model's other L1 buffers (the single-chip sweep has all of L1).
- Warm 56k prefill 7.66 -> 7.42 s (7595 tok/s). Component tests: KDA attention / dense MLP / shared expert / q_a /
  router pass. MLA attention component fails its norm gate (rel 0.017 > 0.012, norm ratio 0.98..0.99) with configs on
  and off alike: HiFi2 (HiFi4 passes, rel 0.0044); attention HiFi4 does not move KV PCC (above), kept HiFi2.

## DSA layers on the chip's own rows (2026-10-09), GLM_DSA_LOCAL=1 default

- Was: attn_norm all-gathered to all S rows on every chip, then every chip ran the indexer key / gate projections +
  pooling and the MLA kv_a + norm on all 5120 rows (8x duplicated) into its replicated caches. Now attn_norm, q_a, the
  pooled keys (640 rows = 160 whole pools) and the latent run on the chip's rows; pooled keys ([160, 128]) and latent
  ([640, 512]) are all-gathered (axis 1, then 0: natural row order) into the replicated caches.
- Warm 56k prefill 7.42 -> 7.03 s (8007 tok/s); last-chunk top1 vs text 0.652 (0.646 before).
- Not done: GLM non-flash's full pattern (striped caches + ring_indexer_score_dsa + sparse_sdpa on a striped latent)
  would also drop the replicated caches (latent ~57 MB per DSA layer per chip at 56k); no further time saving expected.

## Ring indexer with a striped pooled-key cache (2026-10-09), GLM_INDEXER_RING=1 default (split + GLM_DSA_LOCAL)

- GLM non-flash's pattern: the index-key cache striped over the mesh, ring_indexer_score_dsa gathering the other
  stripes while it scores. Flash's keys are pools of 4 tokens, so the op got key_stride R (indexer_score fork,
  CHANGELOG): key j visible to token p iff R j + R-1 <= p, block-cyclic key stripe chunk_local / R, validation and
  gather extent in key units, R staircase mask tiles + the -inf tile (tests/unit/test_key_stride*.py: single device
  exact -inf placement, rel 0.0019; 2x4 full-mesh ring PCC 0.9999985; R=1 regressions unchanged).
- GLM: each chip fill_caches the pools of its own rows into its local stripe (chunk c at local rows c S/32; local
  rows 2560, a multiple of the 2048/5120/8192 stripes) - no gather, no replicated copy (0.65 MB vs 3.6 MB per chip
  per layer); the ring score (full-mesh snake, Topology.Ring) applies the pool-causal mask, top-k on [0, kv).
  load_state / state_torch stripe / un-stripe at the harness boundary. Serving slots (bind_cache) need RING=0.
- Warm 56k prefill 7.09 s (replicated + gather 7.03 s); last-chunk top1 0.6519, identical to the replicated path.
- Batched check after the matmul configs, DSA own-row keys and the ring indexer (s4096 ladder + 56k): KV PCC min/mean
  kv_latent 0.96918 / 0.98528, index_key 0.98931 / 0.99575, kda_recurrent 0.97490 / 0.99403, kda_conv 0.96516 /
  0.99398 (unified 0.96928 / 0.98530, 0.98870 / 0.99533, 0.97420 / 0.99374, 0.96762 / 0.99431); final hidden 0.9463;
  56k top1 0.8673. Warm 56k prefill 7.09 s (from 10.74 s at the start of the day; unified 13.77 s).
- MLA per-head absorb matmuls ([64, 640, 256] x [64, 256, 512] and [64, 640, 512] x [64, 512, 256], 5% math):
  tests/test_matmul_tune.py::test_bmm_tune finds MatmulMultiCoreReuseProgramConfig 2.3x / 3.9x faster on one chip
  (0.620 -> 0.268, 0.905 -> 0.231 ms; correct, rel 4.5e-3). In the model: the best blocks (per_core_M 10 / 20,
  ~1.5 MB CBs) clash with L1 buffers at 1.46 MB; the 1.2 MB-budget blocks (5 / 10) run but give WRONG results
  (last-chunk top1 0.41 vs 0.65; GLM_MLA_BMM_L1=0 restores 0.6519). Not used - suspect the batched multi-core-reuse
  path with more blocks (256) than cores (110); to investigate before retrying.
- Warm-prefill noise: the same code measured 7.09 and 7.40 s on different runs today; compare changes back to back.
- Root cause of the batched-matmul breakage (supersedes the note above), localized with minimal repros in 16-22 s
  runs: ttnn's batched MatmulMultiCoreReuseProgramConfig (per-batch weights) is wrong whenever a core gets more than
  one output block (tests/test_mla_bmm_repro.py::test_mla_bmm_blocks: rel 5..25 vs fp32 for B*Mt/per_core_M > 110
  cores, 4.6e-3 at 64 blocks). The earlier single-chip sweep looked correct only because a same-shape auto matmul ran
  first and left the right data in L1. mm_configs.bmm now only emits one-block-per-core configs: o w_uv per_core_M 20
  (0.899 -> 0.276 ms per DSA layer, rel 1.4e-3 vs auto in the model); q w_uk has none that fits L1 (auto).
- Tools: tests/test_ab_layers.py (in-model A/B over a few layers, runtime module switches, base-vs-base determinism
  check; start 0 so KDA restarts from its zero state), tests/test_mla_bmm_repro.py (single chip / mesh repro, live-input
  replay via GLM_MLA_CHECK_DUMP, predecessor-op and block-count sweeps), mla_attention.CHECK_BMM (per-op config vs auto).

## Head-parallel MLA projections (2026-10-09), GLM_MLA_TP=1 default (split + own-row keys)

- GLM non-flash's MLA layout (deepseek_v3_d_p/tt/mla/mla.py _sparse_mla): projections head-parallel over TP, sparse
  SDPA fully sequence-parallel through a heads <-> sequence all_to_all around it (non-flash needs it because
  sparse_sdpa wants >= 32 heads per chip; GLM's 64 heads at tp=4 leave 16). Here: q_b / w_uk / w_uv / o_proj sharded
  by heads over mesh axis 1 (16 heads per chip), computed on the mesh row's S/2 rows (q latent gathered over axis 1);
  all_to_all_async_generic(in_dim=1, out_dim=2) to [64 heads, own S/8 rows] for sparse_sdpa (unchanged, ring indexer
  ids unchanged) and back (in_dim=2, out_dim=1); o_proj head partials -> fabric_reduce_scatter (bf16) to own rows.
  The all-to-all's in_dim is the GATHERED dim, out_dim the SPLIT one (tests/test_mla_tp_a2a.py, exact); 0.82 ms each
  way at [16, 2560, 512] <-> [64, 640, 512] bf16, 2 links.
- Layer-3 A/B (tests/test_ab_layers.py GLM_AB_CAPTURE_MLA=1, two builds via GLM_AB_SAVE / GLM_AB_REF): MLA output vs the
  replicated path rel 4.1e-3, PCC 0.999993 (bf16 head partial sums); layer output 4.8e-2 (MoE routing amplification).
- Full model, back to back: DRAM allocated per chip after load 29.90 -> 28.22 GB (of 31.83); last-chunk top1 0.630 ->
  0.637. Runtime (device profile, layer 3; the wall-time 7.39 -> 7.28 s was noise): MLA 7.31 -> 8.35 ms per DSA layer
  (+1.73 ms collectives: all-to-alls 0.68 + 0.67, reduce-scatter 0.25, q gather 0.13; -0.7 ms from quarter-size
  q_b / o_proj reads, w_uk / w_uv shapes and the 16-head concat) = ~+11 ms per chunk, ~+0.12 s at 56k.
- Perf measurement: test_perf's warm number is Python wall time over eager dispatch (host dispatch ~60% of device time),
  so host load moves it: the same code measured 7.09 and 7.40 s today. Decide on device time (profiler) or interleaved
  A/B runs, not single wall-time runs hours apart.

## Row gathers on fabric_all_gather (2026-10-09), common.gather_axis (GLM_FABRIC_GATHER=0: ttnn.all_gather)

- tests/test_all_gather_bench.py (common, generic: every all-gather shape in the op report; outputs checked
  bit-identical to ttnn.all_gather; device time from the profiler's per-call records): fabric_all_gather wins at every
  model shape; high_bw_all_gather is within a few % of ttnn.all_gather (faster on 2560x4096 axis 0 and 16x128x128 fp32,
  slower on 640x4096 axis 1 by 16% and on the 512-wide / 16x128x256 shapes) - fabric_all_gather is its fork with MiMo's
  changes. 640x4096 bf16 axis 1 (79 calls / chunk) 281 -> 208 us,
  640x1536 132 -> 88, 2560x4096 axis 0 340 -> 289, 640x512 / 2560x512 56 / 65 -> 42 / 53; ~7 ms per chunk. Non-tile
  rows (KDA's 3x6144 conv state) fail in fabric / high_bw: they keep ttnn.all_gather (as do ttKDA's internal gathers).
- gather_half / gather_rows / the KDA input gather go through gather_axis: a fresh output per call (ttnn.empty), the
  op's own cached semaphores, and a 2D topology re-declared on split-layout tensors (rows over both axes, as MiMo's
  gather_full) - the op rejects their 1D one. Layers 2-4 A/B vs ttnn.all_gather: bit-identical (0 differing elements).
- ttnn.all_gather / reduce_scatter / all_reduce ignore num_links (all_gather) or pick it when unset: they use every
  link on the axis (2 per axis on the LoudBox); the op report's per-link numbers now use the real axis links.

## MoE routing and flat_routed_expert utilization, final chunk, all 42 MoE layers (2026-10-09)

tests/test_moe_routing.py (full prefill for real context; route-plan counts per chip at the last chunk; flat op fenced by
profiler signposts; generated/glm53_flash_d_p_lb/moe_routing.json):
- active experts per chip 31..36 of 36 (median 36; 244 of 12096 expert slots empty: the op skips them, weights
  included); routed rows per chip 2165..9496 (mean 5120); tile padding +11.4% rows.
- flat op per chip 1.30..2.38 ms; slowest chip varies by layer; slowest-per-layer sum 84.1 ms vs median chip 69.7 ms
  (balancing worth ~14 ms per chunk). Time-weighted DRAM 67.5% of peak, math 28% of LoFi peak.
- Fit: ms = 0.90 + 0.138 us x padded rows (rms 0.04 ms; corr with rows 0.969, with active experts 0.14). 0.9 ms ~ reading
  all ~510 MB of bfp4 weights at peak DRAM (1.0 ms); the per-row part ~364 TFLOP/s (60% of LoFi). Weight streaming and
  compute look additive, not overlapped: overlapping them would be worth ~0.6 ms per layer (~25 ms per chunk).

## TODO: real-time-profiler-based per-op breakdown

- Needed: a per-op breakdown (testing/op_report.py) driven by the program real-time profiler instead of the device
  profiler's per-op syncs (23 min for 3 layers today; the whole model times out). Verified (tests/test_realtime_probe.py):
  active on the 2x4 LoudBox, no TT_METAL_DEVICE_PROFILER needed, records (chip, runtime_id, start, end, freq GHz, cores)
  via ttnn.device.RegisterProgramRealtimeProfilerCallback; a ttnn call's programs carry the device-op ids
  ttnn._ttnn.get_device_operation_id() returns before the call (i0 <= id < i1), so attribution needs no sync.
- To do: profiler mode that records the id range per call (no sync) and joins the callback's records; op_report fed
  from it for all 45 layers in one normal-speed pass. Check first against the device profiler on one layer; unsynced
  collectives include wait time on early chips (use the slowest chip's span or a synced CCL-only pass).

## MoE input: router on own rows + full-mesh gathers (2026-10-10), GLM_MOE_FULL_MESH=1 (default)

From MiMo-V2 d_p (b45a923d73c, gather_full: 300.2 -> 286.8 ms traced there). Before: ffn_norm all-gathered the
mesh row's half on axis 1 (0.21 ms), the router ran on those 2560 rows on all 4 chips of the row (0.54 ms), then x /
idx / w were gathered on axis 0 (0.30 ms). Now ffn_norm and the router run on the chip's own 640 rows (router 0.23
ms) and the experts gather x / idx / w once each over the whole mesh (fabric_all_gather cluster_axis=None, row-major
chip order = the same token order; the inputs' topology relabelled dim-2-over-all-chips as MiMo). The shared expert
reuses the gathered x as before.
- Block outputs bit-identical to the old path (layers 3 and 4, real weights, test_layer_perf GLM_LP_SAVE A/B).
- Per-layer device time (fake weights, chunk 5120 at 51200, busiest chip): dsa_moe 15.45 -> 14.95, kda_moe 11.45 ->
  10.91 ms; whole-model estimate 555.8 -> 533.6 ms per chunk (-4.0%).

## KDA chunk preparation: exact gate exponents in a fork (2026-10-10), GLM_KDA_DECAY=fork (default)

The +1.44 ms per KDA layer of `_PreciseDecayRecurrence` (14 ttnn ops recomputing k_dec_t after
prepare_chunk_recurrence, 9% of a chunk) is gone. The bias it corrected comes from prepare_chunk_recurrence reading
FP32 values at TF32 precision in the source registers: the subtraction G - G_last/2 and the copies of G_last/2 and of
the centered exponent into the SFPU (|G| ~ 150 at the -5 gate bound: a 0.125 step). The fork
`ttnn.bringup.prepare_chunk_recurrence(precise_gate_factors=True)` forms the three exponents as exact matmuls of the
BF16 gate with constant masks (+-scale/2) and exponentiates them straight from DST (CHANGELOG.md there). The source
program sits ~1.7 KB under the kernel config buffer; the option builds the compute kernel at -O2 and shares the q / k
norm reduce to fit.
- Layer 0 KDA component test: output rel 0.0072, state rel 0.0142, worst head 0.0227 (precise ops 0.0070 / 0.0140 /
  0.0222; source kernel 0.0070 / 0.0149 / 0.0537 FAIL).
- KDA attention per layer (fake weights, chunk 5120 at 51200): 5.75 (precise ops) -> 4.20 ms; the op itself 0.471 ms
  (source 0.586). Layer device time kda_moe 10.91 -> 9.37, kda_dense 10.26 -> 8.69 ms.
- GLM_KDA_DECAY=precise / kernel keep the old paths.
- Whole model, real weights (tests/test_perf.py, 56320 tokens in 5120 chunks, with the full-mesh MoE input above):
  warm prefill 6.01 s (9367 tok/s; was 7.03-7.09 s), chunks 541-560 ms; cold 12.59 s; last-chunk top1 vs text
  0.6744 / top5 0.8804 (was 0.652 top1).

## Full-mesh reduce-scatter as one ring (2026-10-11), GLM_SCATTER_OP=fabric_ring (default)

common.scatter_rows (shared expert, dense MLP) reduces over the whole mesh with one fabric_reduce_scatter
(cluster_axis=None, Ring: the 2x4 snake closes) instead of axis 0 then axis 1; meshes whose snake does not close keep
the two calls. The MLP still emits its down output in bf16 for it (mlp.py keyed that on the op name: without it the
ring path added a 0.28 ms typecast and a slower fp32 linear).
- Op, model shape (per chip [5120, 4096] bf16, 2 links, 8192 B payload): 427.7 vs 548.6 us
  (tests/unit/test_fabric_rs_model_shapes.py in fabric_reduce_scatter_ttnn).
- Per layer, real weights, device (busiest chip): kda_dense 8.72 -> 8.59, dsa_moe 14.79 -> 14.69, kda_moe 9.35 ->
  9.20 ms; block outputs vs the two calls rel 2e-4 .. 1.2e-3 (bf16 summation order).
- Whole model (test_perf): warm 56k prefill 6.07 s (6.01 before: the eager run is partly host-bound, the device
  saving of ~6 ms per chunk does not show in wall time); last-chunk top1 0.6726 / top5 0.8801 (0.6744 / 0.8804).

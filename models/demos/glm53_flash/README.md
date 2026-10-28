# GLM-5.3-Flash on the Blackhole Galaxy

[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash): 45 decoder layers (34 Kimi-delta linear attention
+ 11 DeepSeek sparse attention), 288 routed experts (top-8) + 1 shared, 4-stream manifold hyper-connection residual, 1 M
context, image input. Weights fp8 block-scaled on the hub, run as bfp8 experts / bf16 dense weights here.

## Design

| | |
|---|---|
| Mesh | 8x4 Blackhole Galaxy (32 x p150): tensor parallel 4 over the columns, 8 data-parallel rows (`GLM53_MESH_SHAPE=8,4`); every (2,4) block of the Galaxy runs the same code for bring-up |
| Experts | hybrid layout: the 288 experts split over the 4 columns (72 per chip), each expert's intermediate split over the 8 rows; routed rows streamed through a model-local expert op at <= 4 users per row, one dense pass at 32 per row; optional EP32 all-to-all dispatch / combine (`GLM53_MOE_LAYOUT=ep`) |
| Attention | KDA layers on the ttKDA kernels (chunked prefill, fused decode recurrence); DSA layers with a paged latent cache (512 bf16 per token) and a pooled-key index cache per user, top-2048 sparse attention; cache either replicated over a row's columns or user-partitioned (one slot per user per chip) |
| Prefill | sequence-parallel over the rows (each row attends its own slab against the full prefix), 16 K chunks, the MoE "regions" path (the d_p routed-expert kernels on a fabric-free region buffer), per-length traced prefill for short prompts |
| Decode | one traced step for all users of the mesh; fast exact mHC kernels, fused exact-fp32 router top-k, stream expert op; all custom kernels are model-local `ttnn.generic_op` programs |
| Residual | 4 mHC streams packed `[1,1,T,4*4096]` fp32, exact fp32 mixing |
| Vision | image features through the GLM vision tower on device, scattered at the image token id into the text prefill |

Details, measurements and the evidence trail: `doc/mesh/README.md` (Galaxy bring-up log), `doc/mesh/sparse_1m.md`
(1 M decode path and the user-partitioned cache), `doc/mesh/prefill_profile.md` (prefill cost model), `doc/reuse_map.md`
(which sibling-model layers were reused), `doc/vision/multimodal.md`, `tt/memory_plan.py` (users x context from the DRAM
headroom; `tests/test_memory_plan.py` prints the envelope).

## Performance

Measured 2026-10-21/22 on an 8x4 Blackhole Galaxy with the default configuration (hybrid expert layout, 45 layers resident,
bfp8 experts, fast exact mHC kernels, fused router on decode, the blocks expert prefill path up to 4 K positions and
the regions path above with the stacked expert copy kept for decode when it fits), `tests/test_mesh_model.py` (short prompts) and
`tests/test_mesh_long_context.py` (long prompts, one prompt per slot). Decode is the traced steady-state step with every
slot active at the stated decode ISL. ISL is per user: every decode slot attends over that many cached tokens, so the
mesh holds users x ISL tokens of cache (32 x 1 M = 33.5 M tokens). TTFT and decode of a row come from one run.
TTFT is per prompt (one prompt in flight) with the whole mesh on that
prompt: for prompts up to 2 K tokens the per-length traced prefill (replay time); above that the chunked prefill
(16 K chunks, sequence-parallel over the 8 rows and tensor-parallel over the 4 columns, compiled kernels, not
traced). The sequence-parallel prefill writes a prompt into the same slot of every row, so one prefill fills 8 slots
with the same prompt; rows with several users per row prefilled them one after another. Filling every slot with a
distinct prompt costs (distinct prompts) x TTFT: the aggregate prefill throughput (~2.2 K tok/s at 1 M, ~2 K at
64 K) is the capacity limit, e.g. 32 distinct 1 M prompts take ~4.2 h to prefill. The users x ISL envelope of the
memory planner (`tt/memory_plan.py`) is measured at every point: 32 x 1 M, 64 x 512 K, 128 x 256 K and 256 x 128 K
all fit with 1.1-2.0 GiB per chip to spare.

| Batch | Hardware | ISL (tokens) | TTFT | T/S/U | Target T/S/U | T/S | Notes |
|---|---|---|---|---|---|---|---|
| 1   | Galaxy 8x4 (Blackhole) | 16 (decode) / 128 (TTFT) | 118 ms | 17.5 | 60 | 17.5 | one run (2026-10-28, today's defaults: fused mHC pre-block + fused Sinkhorn + "extras" splits, the eager dispatch guard off, shape-static eager paths; `tests/test_mesh_model.py` with `GLM53_MODEL_B=1 GLM53_MODEL_TTFT=1 GLM53_TRACE_PREFILL=1`, arg/glm53_mesh 25806fdef6e): one user per row (8 slots, 140 tok/s over the mesh); decode step 57.0 ms; first token matches the HF head; chain pcc vs HF rows 0.947. Previous row (2026-10-26): 134 ms, 15.5. |
| 32  | Galaxy 8x4 (Blackhole) | 16 (decode) / 128 (TTFT) | 120 ms | 16.3 | 33 | 520 | one run (2026-10-28, today's defaults: fused mHC pre-block + fused Sinkhorn + "extras" splits, the eager dispatch guard off, shape-static eager paths; `tests/test_mesh_model.py` with `GLM53_MODEL_TTFT=1 GLM53_TRACE_PREFILL=1`, arg/glm53_mesh 25806fdef6e): one random token per user slot (diverse expert routing); decode step 61.5 ms; first token matches the HF head; chain pcc vs HF rows 0.947; 45-layer logits gate pcc 0.9952, top-1 93.0 %, top-5 95.5 % (bars 0.98 / 85 %). Previous row (2026-10-26): 135 ms, 14.5 (463) at 69.1 ms. |
| 32  | Galaxy 8x4 (Blackhole) | 32 (decode, speculative K=3) | 994 ms | 23.8 | 33 | 762 | one run (2026-10-27, `tests/test_mtp_spec.py`: MTP draft head, fused multi-step KDA commit, V/C/D graphs traced): lossless against plain greedy (64/64 tokens per user), 3.20 tokens per user per iteration, 42.0 ms per token; TTFT = the eager prefill with the MTP hidden capture (the traced prefill with the capture is a follow-up); harness only until the served speculative path lands |
| 256 | Galaxy 8x4 (Blackhole) | 16 (decode) / 128 (TTFT) | 145 ms | 8.9 | 19 | 2280 | one run (2026-10-26, today's defaults: sharded mHC collapse norm; dense expert pass `GLM53_DENSE_EXPERTS=1` at 32 users per row): one random token per user slot; decode step 112.3 ms; first token == HF |
| 8   | Galaxy 8x4 (Blackhole) | 65 514 | 32.4 s | 10.9 | 60 | 87 | one run (needle depth 0.56, retrieved): one prompt, replicated to the 8 rows; 2023 tok/s prefill on the regions path with the stacked copy kept; decode step 91.6 ms |
| 1   | Galaxy 8x4 (Blackhole) | 65 514 | 32.4 s | 10.9 | 60 | 10.9 | the same run as the 8-user row above: a single user's figures (one user per row) |
| 32  | Galaxy 8x4 (Blackhole) | 65 493 | 25.9 s | 12.5 | 33 | 399 | one run (needle depth 0.56, 2026-10-26, today's defaults): 4 prompts per row prefilled one after another at 2530 tok/s (regions prefill path, the streaming decode copy kept beside it); decode step 80.3 ms on the streaming expert op; needle retrieved by all 32 slots, rows and users agree |
| 8   | Galaxy 8x4 (Blackhole) | 1 047 526 | 485 s | 15.1 | 33 | 121 | one run (2026-10-28, today's defaults: fused mHC pre-block + fused Sinkhorn + "extras" splits, the eager dispatch guard off, shape-static eager paths; `tests/test_mesh_long_context.py` with `GLM53_LONG_N=1048576 GLM53_LONG_B=1 GLM53_PARTITION_USERS=0`, arg/glm53_mesh 25806fdef6e): one prompt, replicated to the 8 rows, prefilled at 2159 tok/s (16384-token chunks); decode step 66.2 ms at position 1 047 526; needle retrieved on 8 of 8 rows, rows agree. Previous row (2026-10-27): 492 s, 13.5 (108), 7 of 8 rows. |
| 1   | Galaxy 8x4 (Blackhole) | 1 047 526 | 485 s | 15.1 | 60 | 15.1 | the same run as the 8-user row above: a single user's figures (one user per row) |
| 32  | Galaxy 8x4 (Blackhole) | 1 047 526 (user-partitioned cache) | 470-520 s | 14.4 | 33 | 461 | one run (2026-10-28, today's defaults: fused mHC pre-block + fused Sinkhorn + "extras" splits, the eager dispatch guard off, shape-static eager paths; `tests/test_mesh_long_context.py` with `GLM53_LONG_N=1048576 GLM53_LONG_B=4 GLM53_LONG_USERS=all GLM53_PARTITION_USERS=1 GLM53_PREFILL_PARTITION=heads2users`, arg/glm53_mesh 25806fdef6e): 4 prompts per row prefilled one after another at 2016-2227 tok/s (16384-token chunks; per-user TTFT 519.5 / 474.7 / 475.1 / 470.4 s), decode step 69.4 ms at position 1 047 526; needle retrieved by 32 of 32 users, rows and users agree. Previous row (2026-10-27): 472-485 s, 12.7 (406), 30 of 32 needles. |
| 64  | Galaxy 8x4 (Blackhole) | 525 183 (user-partitioned cache, 2 slots per chip) | 223-240 s | 5.4 | 33 | 347 | one run (needle depth 0.56, gen 160): 8 prompts per row prefilled one after another at 2187-2357 tok/s; decode step 184.6 ms; needle retrieved by all 64 slots (the model reasons for ~100 tokens before answering at this length); 1.95 GiB free after the prefills |
| 128 | Galaxy 8x4 (Blackhole) | 261 769 (user-partitioned cache, 4 slots per chip) | 109-123 s | 5.3 | 33 | 681 | one run (needle depth 0.56): 16 prompts per row prefilled one after another at 2126-2400 tok/s; decode step 188.1 ms; needle retrieved by all 128 slots, users of a row bit-identical; 1.63 GiB free after the prefills |
| 256 | Galaxy 8x4 (Blackhole) | 130 844 (user-partitioned cache, 8 slots per chip) | 55-63 s | 5.0 | 19 | 1287 | one run (needle depth 0.56): 32 prompts per row prefilled one after another at 2078-2395 tok/s; decode step 199.0 ms; needle retrieved by all 256 slots, all 256 chains bit-identical; 1.12 GiB free after the prefills |
| 2   | 2x4 block (Blackhole), image + text | 286 (256 image tokens) | 1.1 s | 13.6 | - | 27 | one run (multimodal demo, today's defaults): eager vision tower 60 ms + prefill; decode step 73.7 ms; 45 layers bfp4 experts; first token == HF |

Experimental EP32 expert layout (`GLM53_MOE_LAYOUT=ep`, all-to-all dispatch / combine over the 32 chips; no
long-context prefill path and a prefill accuracy gap at 128 tokens, so not the default): with one random token
per user, traced decode step 71.2 ms at 32 users (14.1 tok/s per user, 450 aggregate; hybrid 83.1 ms), 139.2 ms at
256 users with the combine op (7.2 tok/s per user) and 126.3 ms with the dense row-exchange combine (7.9 tok/s per
user, 2028 aggregate; hybrid dense 115.5 ms); a reduce-scatter combine is under validation.

Accuracy on the same stack: on a 128-token natural prompt the next-token logits reach PCC 0.994 vs HF with top-1
agreement 90 % and top-5 overlap 94 % over the 128 positions (`tests/test_mesh_model_logits.py`; the residual-stream
PCC after 45 layers is ~0.90 and is depth-chaotic, not a correctness metric); the 16 K and 64 K needle prompts
generate 8/8 greedy tokens identical to the HF CPU
reference (`reference/compare_to_tt.py`); the needle is retrieved at 1 M for distances up to ~460 K tokens (32 of 32 slots in the 32-user run; a needle placed 523 K tokens before the question is missed, and on that prompt one slot per row also flipped its first token, under investigation); the
image demo's 16 greedy tokens match HF. Targets are the design goals of `~/ashai/GLM_5.3_DESIGN.md` (19 / 33 / 60
tok/s per user at 256 / 32 / 8 users); the optimization phase is in progress (fused mHC pre-block, the EP32 exchange,
the sparse decode at long context).

### Speculative decode (MTP draft head)

The checkpoint's MTP layer drafts K tokens per step and the model verifies them in one batched pass (`tt/spec_decode.py`:
three traced graphs, greedy only, K = min(K, 32 / users-per-row - 1)). Measured 2026-10-27 on the 8x4 (45 layers, 32 users = 4
per row, K = 3, fused multi-step KDA commit `GLM53_SPEC_COMMIT=fused`, V/C/D traced; `tests/test_mtp_spec.py`): lossless
against plain greedy (64/64 tokens per user), 2.45 accepted drafts = 3.20 tokens per user per iteration, 133 ms per iteration =
41.7 ms per token = **24.0 tok/s per user, 768 tok/s aggregate** against the plain traced step's 70 ms (14.5): x1.68; the table
row above is the rerun that also reports its TTFT (23.8 tok/s per user). Not served yet: the vLLM adapter's speculative path
(`GLM53_VLLM_SPEC=1`, the plugin's device-propose contract) is written and under device validation; it covers replicated-cache
profiles and the dense range (anchors <= 2050) first, so the 1 M profile (partitioned caches, long anchors) needs the
partitioned spec state and the sparse-range verify. Off by default (`GLM53_SPEC_K` unset).

## Serving (vLLM + tt-inference-server)

Stack: stock vLLM 0.30.0 + `vllm-tt-plugin` main (the model is registered in the plugin's built-in map on branch
`arg/glm53_registration`; until it merges the catalog passes
`TT_MODEL_CLASS_OVERRIDES=Glm5NextForConditionalGeneration=models.demos.glm53_flash.tt.generator_vllm:GLM53FlashForCausalLM`),
transformers 5.20.2 (`glm5_next` exists from 5.16: `requirements.txt`). tt-inference-server catalog (branch `arg/glm53_flash_galaxy`):
BLACKHOLE_GALAXY only; profiles `glm53_flash` = 1 M context x 4 distinct requests (users per row: the 32 slots are 8 replicas of 4
prompts until per-row prefill ownership lands) and `glm53_flash_128k` = 128 K x 28 (7 slots per chip: 32 ran out of DRAM in the
prefill warm-up; `tt/memory_plan.py` reserves a 16 K chunk's dense working set); `TT_METAL_OPERATION_TIMEOUT_SECONDS=120` (the
launcher's 5 s watchdog trips on 16 K prefill chunks at 64 K+ positions).

Validated 2026-10-27 on the 8x4 (serving smoke, warm cache, boot 15 min): exact first tokens on the 1 M profile (both reference
prompts), greedy determinism across rounds through the chat template (thinking on), 4 concurrent 65 514-token needles retrieved 4/4
(TTFT 26.5 / 52.6 / 78.6 / 104.8 s: the sequence-parallel prefill serves prompts one after another at ~2.5 K tok/s), `vllm bench`
128 x 128 tokens at 32 requests (4 at a time) 32/32.

Served decode (2026-10-27, today's defaults, `vllm bench` 128 x 128 tokens): 1 M profile 94.2 ms per token = 10.6 tok/s per user at
concurrency 1 (TTFT 272 ms) and 103.4 ms = 9.7 tok/s per user at 4 (37 tok/s output); 128 K profile (28 users per row, 224 traced
slots) 150.9 ms = 6.6 tok/s per user at 1 and 172.8 ms = 5.8 tok/s per user at 28 (142 tok/s output); boot ~10 min on a warm cache.
A served 1 M-context request (1 047 612 tokens, 160 out, 1 M profile) completes with TTFT 504.6 s (2076 tok/s prefill) and 95.4 ms per
token = 10.5 tok/s per user at the 1 M position (the sparse prefill's score block is 256 rows at 1 M: `tt/memory_plan.py`
`default_prefill_qblock`; 2048 rows ran out of DRAM at ~196 K, 512 at chunk 62 of 64). Long prompts under concurrency are
scheduler-bound today (the engine interleaves 2 prefill steps per decode step: 64 K x 28 on the 128 K profile completes at 0.34 tok/s
per user with 389 s TTFT); the tt-inference-server targets are these served points per profile. That is the traced replay
of the 1 M-capacity model on the regions-sourced streaming decode (~76 ms) plus ~26 ms of host work per step (one packed upload and
one read per row: `GLM53_DECODE_PACKED_INPUTS=1`, default since smoke r12). History: 372 ms with the Python page-table check,
192 after its removal, 174.6 with packed inputs, 103.2 once the streaming decode replaced the regions fallback (next paragraph). The replay costs 148 ms against 65 ms in the harness at a short cache,
and the harness reproduces it: the same 45-layer traced step at a 1 M cache with the served (user-partitioned) layout is 152 ms
for a 128-token prompt (`GLM53_MODEL_MAX_SEQ_LEN=1048576 GLM53_PARTITION_USERS=1`), so serving adds nothing to the device step.
The gap is a threshold, not capacity scaling: 256 K per user replays in 70.9 ms, 512 K and 1 M in ~152 ms. Above ~256 K x 4 users
the memory planner drops the second, DRAM-bank-sharded expert copy that the decode streaming op reads (it does not fit beside the
caches: `GLM53_REGIONS_KEEP_STACKED`), and decode falls back to the prefill regions lists (per-expert matmuls): ~+80 ms per step
over the 42 MoE layers (confirmed: 256 K with the copy forced off replays in 148.9 ms with the 1 M numerics). The 128 K profile ran the same fallback. Fixed by the regions-sourced streaming decode (`GLM53_REGIONS_DECODE=stream`,
the default since 2026-10-27; `custom_ops/expert_stream/regions.py`): the per-expert regions tensors already sit in the bank-local
column layout the stream kernels read, so one expert copy serves prefill and decode and the stacked copy is never built. Bit-identical
to the stacked path (one chip, the (2,4) block, and the 45-layer chain pcc 0.861457 = the stacked value); 1 M x 4 users per row
152 -> 75.7 ms per step (13.2 tok/s per user, 423 aggregate; TTFT 143 ms for 128 tokens, was 264), and at 256 K the default path
matches the stacked copy it replaces (70.3 vs 70.9 ms, with the sparse-decode bound on). The served 1 M profile follows in smoke r14.
bfp4 experts stay a separate lever (accuracy gate pending). The sparse-decode scorer and top-k also looped over the pool capacity (+0.94 ms per DSA layer per step); a per-step
live-pool bound (`GLM53_SPARSE_DECODE_BOUND=1`) is validated bit-identical on the block and on the 8x4 (152.2 -> 148.6 ms).

Open: (1) the KDA chronology scalars were allocated after the trace captures and clobbered by decode replays (the served 64 K random
prompt hung the ttnn KDA `reduce_affine_transforms` op twice, wedging the mesh): fixed with persistent per-mesh scalars written in
place plus an allocation guard after captures (`tt/kda.py`), validated; a second hang right after that prompt's last chunk is open
(a fresh-slot reproducer on the block passes at 2175, 2176, 65514 and 65536 tokens; slot reuse is next), so the 64 K random
benchmark point stays excluded; (2) eager prefill chunks pad to 256/512/1024/2048/16384 (`GLM53_PREFILL_BUCKETS`): a 1024-row slab
per chip (6.4-9.6 K-token single chunks on the 8x4) overflows L1 with the KDA scan's static buffers, and a 4096 bucket costs ~1 GB of
DRAM the 1 M x 32-slot model does not have; (3) 32 distinct 1 M users need the per-row prefill ownership
(`doc/mesh/row_ownership.md`).

## How to run

```
export W=<tt-metal build>  PYTHONPATH=$W:$W/ttnn  TT_METAL_HOME=$W ARCH_NAME=blackhole
# e2e chain + traced decode + TTFT on the 8x4, 4 users per row
GLM53_MESH_SHAPE=8,4 GLM53_MODEL_LAYERS=0-44 GLM53_MODEL_B=4 GLM53_MODEL_S=16 pytest models/demos/glm53_flash/tests/test_mesh_model.py -k chain -s
# 1 M needle, one slot per row (replicated cache), regions prefill
GLM53_MESH_SHAPE=8,4 GLM53_LONG_N=1048576 GLM53_LONG_B=1 GLM53_PARTITION_USERS=0 GLM53_PREFILL_EXPERTS=regions \
  GLM53_REGIONS_T=2048 GLM53_PREFILL_QBLOCK=512 GLM53_LONG_CHUNK=16384 GLM53_LONG_GEN=160 \
  pytest models/demos/glm53_flash/tests/test_mesh_long_context.py -s
# 1 M, 4 users per row with the user-partitioned cache (heads -> users prefill)
GLM53_MESH_SHAPE=8,4 GLM53_LONG_N=1048576 GLM53_LONG_B=4 GLM53_LONG_USERS=all GLM53_PARTITION_USERS=1 \
  GLM53_PREFILL_PARTITION=heads2users GLM53_PREFILL_EXPERTS=regions GLM53_REGIONS_T=2048 GLM53_PREFILL_QBLOCK=256 \
  pytest models/demos/glm53_flash/tests/test_mesh_long_context.py -s
# image + text demo on a (2,4) block
GLM53_MESH_SHAPE=2,4 GLM53_MODEL_EXPERT_DTYPE=bfp4 GLM53_MODEL_EXPERT_DOWN_DTYPE=bfp4 pytest models/demos/glm53_flash/demo/multimodal_demo.py -s
```

A (2,4) block of the Galaxy: `TT_VISIBLE_DEVICES=<8 chips> MESH_DEVICE=P150x8` with the `p150_x8` mesh graph descriptor
(see `doc/mesh/README.md`); 45 layers fit a block only with bfp4 experts.

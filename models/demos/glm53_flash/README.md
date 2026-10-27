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
| 1   | Galaxy 8x4 (Blackhole) | 16 (decode) / 128 (TTFT) | 134 ms | 15.5 | 60 | 15.5 | one run (2026-10-26, today's defaults: streaming routed-expert op + sharded mHC collapse norm; reproduced on the defaults-only snapshot: 64.5 ms / 136 ms): one user per row (the 8 rows decode the same prompt in parallel; a single user's figures); decode step 64.6 ms; first token == HF |
| 32  | Galaxy 8x4 (Blackhole) | 16 (decode) / 128 (TTFT) | 135 ms | 14.5 | 33 | 463 | one run (2026-10-26, today's defaults: the streaming routed-expert op for the decode-shaped rows, the 16-row prefill slabs included, and the sharded mHC collapse norm; reproduced on the defaults-only snapshot: 69.2 ms / 136 ms): one random token per user slot (diverse expert routing); decode step 69.1 ms; first token == HF; the logits gate with this op is queued (the block-op path: 0.9948 / top-1 95.3 %) |
| 256 | Galaxy 8x4 (Blackhole) | 16 (decode) / 128 (TTFT) | 145 ms | 8.9 | 19 | 2280 | one run (2026-10-26, today's defaults: sharded mHC collapse norm; dense expert pass `GLM53_DENSE_EXPERTS=1` at 32 users per row): one random token per user slot; decode step 112.3 ms; first token == HF |
| 8   | Galaxy 8x4 (Blackhole) | 65 514 | 32.4 s | 10.9 | 60 | 87 | one run (needle depth 0.56, retrieved): one prompt, replicated to the 8 rows; 2023 tok/s prefill on the regions path with the stacked copy kept; decode step 91.6 ms |
| 1   | Galaxy 8x4 (Blackhole) | 65 514 | 32.4 s | 10.9 | 60 | 10.9 | the same run as the 8-user row above: a single user's figures (one user per row) |
| 32  | Galaxy 8x4 (Blackhole) | 65 493 | 25.9 s | 12.5 | 33 | 399 | one run (needle depth 0.56, 2026-10-26, today's defaults): 4 prompts per row prefilled one after another at 2530 tok/s (regions prefill path, the streaming decode copy kept beside it); decode step 80.3 ms on the streaming expert op; needle retrieved by all 32 slots, rows and users agree |
| 8   | Galaxy 8x4 (Blackhole) | 1 047 612 | 459 s | 6.7 | 33 | 54 | one run (needle depth 0.56, retrieved within 64 tokens): one prompt, replicated to the 8 rows; 2284 tok/s prefill; decode step 148.9 ms through the regions expert path (the stacked expert copy does not fit beside the 1 M caches) |
| 1   | Galaxy 8x4 (Blackhole) | 1 047 612 | 459 s | 6.7 | 60 | 6.7 | the same run as the 8-user row above: a single user's figures (one user per row) |
| 32  | Galaxy 8x4 (Blackhole) | 1 047 612 (user-partitioned cache) | 456-489 s | 6.6 | 33 | 210 | one run (needle depth 0.56, gen 160): 4 prompts per row prefilled one after another (each replicated to the 8 rows, 31 min to fill the 32 slots) at 2142-2296 tok/s; needle retrieved by all 32 slots, the 32 chains bit-identical over 160 tokens; decode step 152.1 ms through the regions expert path |
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

## Serving (vLLM + tt-inference-server)

Stack: stock vLLM 0.30.0 + `vllm-tt-plugin` main (the model is registered in the plugin's built-in map on branch
`arg/glm53_registration`; until it merges the catalog passes
`TT_MODEL_CLASS_OVERRIDES=Glm5NextForConditionalGeneration=models.demos.glm53_flash.tt.generator_vllm:GLM53FlashForCausalLM`),
transformers 5.20.2 (`glm5_next` exists from 5.16: `requirements.txt`). tt-inference-server catalog (branch `arg/glm53_flash_galaxy`):
BLACKHOLE_GALAXY only; profiles `glm53_flash` = 1 M context x 4 distinct requests (users per row: the 32 slots are 8 replicas of 4
prompts until per-row prefill ownership lands) and `glm53_flash_128k` = 128 K x 32; `TT_METAL_OPERATION_TIMEOUT_SECONDS=120` (the
launcher's 5 s watchdog trips on 16 K prefill chunks at 64 K+ positions).

Validated 2026-10-27 on the 8x4 (serving smoke, warm cache, boot 15 min): exact first tokens on the 1 M profile (both reference
prompts), greedy determinism across rounds through the chat template (thinking on), 4 concurrent 65 514-token needles retrieved 4/4
(TTFT 26.5 / 52.6 / 78.6 / 104.8 s: the sequence-parallel prefill serves prompts one after another at ~2.5 K tok/s), `vllm bench`
128 x 128 tokens at 32 requests (4 at a time) 32/32.

Served decode (2026-10-27, `vllm bench` 128 x 128 tokens, 32 requests 4 at a time): 192 ms per token = 5.2 tok/s per user after
the Python page-table check was removed (372 ms before). The step is the traced replay at the 1 M-capacity model (148 ms) plus ~41 ms
of host work (32-chip id readback and 11 per-step position uploads to every chip: one packed upload and one read per row under
`GLM53_DECODE_PACKED_INPUTS=1`, device validation pending). The replay costs 148 ms against 65 ms in the harness at a short cache,
and the harness reproduces it: the same 45-layer traced step at a 1 M cache with the served (user-partitioned) layout is 152 ms
for a 128-token prompt (`GLM53_MODEL_MAX_SEQ_LEN=1048576 GLM53_PARTITION_USERS=1`), so serving adds nothing to the device step.
The gap is a threshold, not capacity scaling: 256 K per user replays in 70.9 ms, 512 K and 1 M in ~152 ms. Above ~256 K x 4 users
the memory planner drops the second, DRAM-bank-sharded expert copy that the decode streaming op reads (it does not fit beside the
caches: `GLM53_REGIONS_KEEP_STACKED`), and decode falls back to the prefill regions lists (per-expert matmuls): ~+80 ms per step
over the 42 MoE layers. The 128 K x 32-users profile runs the same fallback. Levers under test: a streaming decode that sources
the one resident expert copy directly (`GLM53_REGIONS_DECODE`, in progress), and bfp4 experts so both copies fit at 1 M (accuracy
gate pending). The sparse-decode scorer and top-k also looped over the pool capacity (+0.94 ms per DSA layer per step); a per-step
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

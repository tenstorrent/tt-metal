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

Measured 2026-10-21 on an 8x4 Blackhole Galaxy with the default configuration (hybrid expert layout, 45 layers resident,
bfp8 experts, fast exact mHC kernels, fused router), `tests/test_mesh_model.py` (short prompts) and
`tests/test_mesh_long_context.py` (long prompts, one prompt per slot). Decode is the traced steady-state step with every
slot active at the stated decode ISL. ISL is per user: every decode slot attends over that many cached tokens, so the
mesh holds users x ISL tokens of cache (32 x 1 M = 33.5 M tokens). TTFT and decode of a row come from one run.
TTFT is per prompt (one prompt in flight) with the whole mesh on that
prompt: for prompts up to 2 K tokens the per-length traced prefill (replay time); above that the chunked prefill
(16 K chunks, sequence-parallel over the 8 rows and tensor-parallel over the 4 columns, compiled kernels, not
traced). The sequence-parallel prefill writes a prompt into the same slot of every row, so one prefill fills 8 slots
with the same prompt; rows with several users per row prefilled them one after another. Filling every slot with a
distinct prompt costs (distinct prompts) x TTFT: the aggregate prefill throughput (~2.2 K tok/s at 1 M, ~2 K at
64 K) is the capacity limit, e.g. 32 distinct 1 M prompts take ~4.2 h to prefill.

| Batch | Hardware | ISL (tokens) | TTFT | T/S/U | Target T/S/U | T/S | Notes |
|---|---|---|---|---|---|---|---|
| 32  | Galaxy 8x4 (Blackhole) | 16 (decode) / 128 (TTFT) | 353 ms | 14.3 | 33 | 458 | decode step 69.8 ms; TTFT and decode from separate runs of the same config (one-run re-measure queued) |
| 256 | Galaxy 8x4 (Blackhole) | 16 (decode) / 128 (TTFT) | 139 ms | 8.7 | 19 | 2217 | one run: dense expert pass, one random token per user slot; decode step 115.5 ms; greedy id == host head for every user |
| 8   | Galaxy 8x4 (Blackhole) | 65 493 | 31.8 s | 9.8 | 60 | 79 | one prompt, replicated to the 8 rows; 2061 tok/s prefill; decode step 101.8 ms on the stream expert op |
| 32  | Galaxy 8x4 (Blackhole) | 65 493 | 27.3 s | 6.7 | 33 | 216 | one run: every slot populated (4 prompts per row prefilled one after another at 2400 tok/s on the MoE regions prefill path), needle retrieved by all 32 slots, users of a row bit-identical; decode step 148.2 ms through the regions expert kernel (the stacked block op decodes at 90 ms when its copy fits: one-run re-measure queued) |
| 8   | Galaxy 8x4 (Blackhole) | 1 047 612 | 460 s | 6.4 | 33 | 51 | one run (needle depth 0.56, retrieved by all 8 slots): one prompt, replicated to the 8 rows; 2276 tok/s prefill; decode step 156.0 ms through the regions expert path (the stacked expert copy does not fit beside the 1 M caches) |
| 32  | Galaxy 8x4 (Blackhole) | 1 047 612 (user-partitioned cache) | 456-489 s | 6.6 | 33 | 210 | one run (needle depth 0.56, gen 160): 4 prompts per row prefilled one after another (each replicated to the 8 rows, 31 min to fill the 32 slots) at 2142-2296 tok/s; needle retrieved by all 32 slots, the 32 chains bit-identical over 160 tokens; decode step 152.1 ms through the regions expert path |
| 2   | 2x4 block (Blackhole), image + text | 286 (256 image tokens) | 1.2 s | 6.9 | - | 14 | one run (multimodal demo): eager vision tower 90 ms + prefill; decode step 145 ms; 45 layers bfp4 experts; answer identical to HF (one-run re-measure with today's defaults queued) |

Experimental EP32 expert layout (`GLM53_MOE_LAYOUT=ep`, all-to-all dispatch / combine over the 32 chips; no
long-context prefill path yet, so not the default): traced decode step 58.1 ms at 32 users (17.2 tok/s per user,
551 aggregate) and 135.8 ms at 256 users (7.4 tok/s per user, 1885), same chain PCC and greedy ids.

Accuracy on the same stack: the 16 K and 64 K needle prompts generate 8/8 greedy tokens identical to the HF CPU
reference (`reference/compare_to_tt.py`); the needle is retrieved at 1 M for distances up to ~460 K tokens (32 of 32 slots in the 32-user run; a needle placed 523 K tokens before the question is missed, and on that prompt one slot per row also flipped its first token, under investigation); the
image demo's 16 greedy tokens match HF. Targets are the design goals of `~/ashai/GLM_5.3_DESIGN.md` (19 / 33 / 60
tok/s per user at 256 / 32 / 8 users); the optimization phase is in progress (fused mHC pre-block, the EP32 exchange,
the sparse decode at long context).

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

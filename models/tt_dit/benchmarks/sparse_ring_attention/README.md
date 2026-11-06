# Sparse ring attention benchmark (Blackhole Galaxy, SGLang-compatible inputs)

Per-call attention timing and accuracy of `ttnn.transformer.sparse_ring_sdpa` (block-sparse ring attention) on a
4x8 Blackhole Galaxy, on the same Q / K / V tensors an SGLang attention backend receives. The workload is
MiniMax-H3's DiT joint attention (text, condition keyframes, audio and video packed into one non-causal sequence;
56 heads, head dim 128). Results go into a JSONL schema that any other platform can fill in for a side-by-side
table (`merge_results.py`).

## What is measured

One attention call: Q / K / V in, attention output out. For the sparse backends the call includes their scoring
(block mask, pooled K / V or coarse output), which runs on device in the same trace as the attention op. Projections,
norms, RoPE, VSA's gate projection and the rest of the layer are not included. The inputs are the packed 768p t2va
sequence of MiniMax-H3 with a 39-token prompt:

| length | real tokens T (text + audio + video) | video grid (t, h, w) | dense layout: padded tokens (rows per device) | VSA tile layout: rung (rows per device) |
|---|---|---|---|---|
| 5 s | 37,749 (39 + 414 + 37,296) | 37 x 24 x 42 | 37,888 (4,736) | 45,312 (5,664), 668 tiles |
| 10 s | 73,425 (39 + 810 + 72,576) | 72 x 24 x 42 | 73,472 (9,184) | 82,688 (10,336), 1,202 tiles |
| 15 s | 109,101 (39 + 1,206 + 107,856) | 107 x 24 x 42 | 109,312 (13,664) | 120,320 (15,040), 1,802 tiles |

56 heads (14 per device, TP=4), head dim 128. The sequence is sharded over the 8 devices of the ring axis, heads over
the 4 devices of the other axis. `fa` and `sol_attn` run on the packed order padded to the working point; the padding
is masked (`logical_n`), so it is not attended. `video_sparse_attn` and `cube_sparse_attn` run on VSA's tile layout
(`vsa_geometry.py`): every 4 x 4 x 4 video cube and every 64-token prefix chunk is one 64-row block (ragged cubes
padded), at the bucket-ladder rung the pipeline would pick. The pipeline reorders the sequence into this layout once,
on the host, at model input, so the reorder is not part of the timed call. The tokens and the accuracy reference are
the same for every backend.

Timing: one eager call (compiles and is used for the accuracy check), one trace capture, then `--warmup` + `--reps`
trace replays. Each replay is timed on the host from `execute_trace` to `synchronize_device`, so dispatch is
excluded and the time is device time plus one synchronize. We report the median, p10 and p90 over the replays. For
the sparse backends the headline `ms_*` is scoring + attention in one trace; `extra.scoring_ms_median` and
`extra.attention_only_ms_median` time each part alone in its own trace (they need not add up exactly).

Accuracy: every backend's output is compared with exact attention computed in fp64 on the host from the same bf16
inputs, in the packed order (the tile-layout outputs are unpacked first). The comparison uses a slice (default:
first and last head, first and last SP shard of the packed order; shard 0 holds the text prefix), over real tokens
only:

- `rel_l2_vs_dense` = ||out - ref|| / ||ref||
- `pcc_vs_dense` = Pearson correlation of out and ref
- `scale_vs_dense` = <out, ref> / <ref, ref> (fitted output scale; PCC does not catch a scale error)

For the sparse backends, `extra.pcc_vs_block_sparse_ref` also checks the op against an fp64 reference of the
block-sparse result it is meant to compute, using the same mask (for the tile layout, on the first and last shard of
that layout). This is a kernel-correctness check, separate from the sparsity error.
`extra.max_abs_diff_traced_vs_eager` should be 0.

At the pipeline rung, VSA and cube process more rows than the dense layout: 45,312 / 82,688 / 120,320 against
37,749 / 73,425 / 109,101 real tokens (+20 % / +13 % / +10 %; ragged 4x4x4 cubes and the ladder's rounding). Their
times include that padding. `merge_results.py` therefore compares each record with the `fa` record of the same
`extra.rows_per_device` (the `fa` run with `{"rung": "ladder"}` for VSA / cube), falling back to the default `fa`
record, and shows each record's padded tokens and rows per device.

## Backends

`--attention-backend` takes SGLang's backend names (a comma-separated list runs several on each input):

| backend | what runs on the Galaxy | timed call |
|---|---|---|
| `fa` | dense ring attention (`ring_joint_scaled_dot_product_attention`), bf16, HiFi2 | the op |
| `sol_attn` | Sol pre-pass (`sol_prepass.py`: block means, pooled K / V all-gathers, diag-threshold routing to a 64x64 block mask), then `sparse_ring_sdpa` pooled route: kept blocks exact, every other block one pooled entry (mean K, mean V, weighted by its token count) | pre-pass + op |
| `video_sparse_attn` | `VSAPrepass` (`vsa_prepass.py` / `block_topk_routing.py`: tile means, one K / V means all-gather, per-head top-k of the video tiles for each video Q tile, coarse branch), then `sparse_ring_sdpa` kept-only (the fine branch) with per-block token counts | pre-pass + op |
| `cube_sparse_attn` | the same routing without the coarse branch, keeping a `topk_ratio` share of the video cubes | pre-pass + op |

`--attention-backend-config` uses SGLang's syntax (JSON or `k=v,k=v`) and applies to the sparse backends:

- `sol_attn`: `tau` (1.0), `thresh_type` (`diag`, the only one supported), `sink_start` (0, the only value supported:
  the prefix is packed first), `sink_tokens` (0; an int, `"text_len"` for the input's text length, or `"prefix"` for
  the whole non-video prefix). These are exact key-sink columns. `kv_splits` is recorded but has no effect (GPU
  kernel tuning). Options specific to this implementation: `band` (1: the Q block's own and adjacent key blocks are
  always exact), `dense_prefix_queries` (false: when true, query blocks inside the sink prefix attend densely through
  the mask), `k_chunk` (256).
- `video_sparse_attn`: `VSA_sparsity` (0.9), `vsa_mode` (`exempt`, the only one the device routing implements),
  `k_chunk` (256), `shuffle`, `rung` (below).
- `cube_sparse_attn`: `topk_ratio` (0.5), `local_cube_size` ([4, 4, 4], the only one supported), `k_chunk` (256),
  `shuffle`, `rung`.
- `shuffle` (true): VSA's shard-balanced tile placement, as the pipeline uses it: the prefix tiles are dealt
  round-robin to the front of the shards and the video tiles cut into one equal run per shard, so the dense prefix
  rows and every shard's work are spread evenly over the ring. false: the tiles fill shard 0, then shard 1, ...
  (the layout before the balancing), for comparison.
- `rung` (`"ladder"`): the pipeline's 4x8 bucket ladder rung; `"min"`: the fewest whole blocks that hold the tiles.
- `fa` takes only `rung` from the config (none by default): `"ladder"` / `"min"` runs the dense op on the same tile
  layout and rung as VSA / cube, with `logical_n` = the rung as they get it, so it does the same padded work. It
  then attends the pad rows too, so its accuracy fields are null (a timing baseline only).
- `dense_steps`, `dense_layers`, `dense_backend` are pipeline settings: they choose which calls go to the dense
  backend. They are recorded and listed in `extra.pipeline_keys_not_applied`. To estimate a full generation, weight
  the per-call times by the share of sparse calls.

Flags: `--lofi` (LoFi matmuls in the sparse op; the dense baseline and the scoring stay as they are), `--kv-dtype bf8`
(the sparse op reads bfloat8_b K / V; Q stays bf16). Sol's pre-pass reads the same bf16 Q and bf8 K / V natively and
returns bf8 pooled K / V. The VSA / cube routing reads bf16 only (its pool program), so with bf8 it scores the bf16
Q / K / V and the op gets bf8 copies of K / V uploaded beside them; no cast is timed.

Dense rows and columns: VSA and cube make every prefix (text, condition, audio) Q tile a dense row and keep every
prefix K tile, all through the block mask. Sol's `"sink_tokens": "prefix", "dense_prefix_queries": true` does the
same through its mask. Sol's normal cases have no dense rows and run on the packed order (no shuffle); Sol does not
run on the tile layout (its pre-pass has no per-block token counts for ragged tiles).

Differences from the GPU `sol_attn` kernel that may affect accuracy or kept fraction: 64-token blocks within each
SP shard, so the last block of a shard can be partial (partial blocks are always pooled, never exact); block 0 of
every shard is always exact; bf16 proxy scores.

### VSA and cube sparse attention

Both use the block top-k routing of the VSA model code, on 4 x 4 x 4 video cubes (64 tokens, one block each) and
64-token prefix chunks. A video Q cube scores its mean Q against the mean K of every video cube (fp32, per head)
and keeps the top `k = ceil((1 - sparsity) * video cubes)` (ties at the k-th score all kept) plus every prefix
block; prefix Q blocks keep everything. Dropped blocks are dropped (kept-only softmax, no pooled entry).

- `video_sparse_attn`: `sparsity = VSA_sparsity`. The routing also computes the coarse branch (softmax over the tile
  means, per tile), which is timed; the gated sum `fine + gate * coarse` and the gate projection are not (the base
  H3 checkpoint has a zero gate, so the accuracy fields compare the fine branch).
- `cube_sparse_attn`: `sparsity = 1 - topk_ratio`, no coarse branch. This is SGLang's training-free cube sparse
  attention (mean-pooled Q and K per cube, top-k ratio of cubes per query cube, text / audio / reference images
  dense) on this routing. Differences from SGLang's FlexAttention implementation: the attention kernel is the
  block-sparse ring op, not FlexAttention; one `topk_ratio` per run instead of a per-step `topk_ratio_list` (run
  each ratio of the list and weight by steps); the k-th-score rounding and tie handling follow this routing (keeps
  `ceil(ratio * n)`, ties kept); condition keyframes count as prefix (dense) here.

## Input format

One safetensors file per attention call (`--inputs` takes files and/or directories of `*.safetensors`):

| key | shape | dtype |
|---|---|---|
| `q`, `k`, `v` | `[B, T, H, D]` (SGLang NHD / BTHD), H = 56, D = 128, the full sequence and all heads | bf16 |

Optional string metadata: `layer`, `step`, `seconds`, `text_len`, `segments` (JSON object of segment lengths in
packed order, video last, e.g. `{"text": 39, "condition": 0, "audio": 414, "video": 37296}`), `video_grid` (JSON
list, the video token grid (t, h, w), e.g. `[37, 24, 42]`), `dense_tokens`. `video_sparse_attn` and
`cube_sparse_attn` need `segments` and `video_grid` (without `video_grid`, a video segment that is a whole number of
24 x 42 frames is taken as 768p 16:9). `--batch-index` picks the batch entry (CFG-batched calls have B = 2).

`--synthetic --seconds N --seed S` generates seeded `randn` bf16 tensors of the table above (T = 37,749 / 73,425 /
109,101, with their `segments`, `video_grid` and `text_len` = 39), derived from the pipeline's packing helpers.
`--save-inputs file.safetensors` writes them in this format, so another platform can run on exactly the same
tensors. Random data routes differently from real activations, so use captured tensors for accuracy numbers.

### Capturing inputs from SGLang

`capture_qkv_sglang.py` wraps any `sglang` command. It hooks the `forward(query, key, value, attn_metadata)` of
the attention backend impls under `sglang.multimodal_gen.runtime.layers.attention.backends` through a generated
`sitecustomize.py`, so worker processes are hooked too, and writes the selected calls in the format above:

```bash
python models/tt_dit/benchmarks/sparse_ring_attention/capture_qkv_sglang.py \
    --out /data/h3_qkv --layers 2,20,40 --steps 12,30 --seconds 5 \
    --segments text=<n>,condition=<n>,audio=<n>,video=<n> --video-grid <t>,<h>,<w> -- \
    sglang generate --model-path <MiniMax-H3 checkpoint> --prompt "<prompt>" --attention-backend fa <5 s T2VA args>
```

Assumptions are listed in the script's docstring. The main ones: run on one GPU (no sequence or tensor
parallelism), so a call sees the full sequence; layer and step are counted per backend-impl instance; calls are
filtered by 56 heads x 128 and at least 4,096 tokens.

## Result schema (JSONL, one record per input x backend)

| field | meaning |
|---|---|
| `platform` | `tenstorrent`, or e.g. `nvidia` |
| `device` | e.g. `Blackhole Galaxy 4x8 (TP=4 heads, SP=8 sequence)`, `H100 SXM 80GB` |
| `backend` | SGLang backend name: `fa`, `sol_attn`, `video_sparse_attn`, ... |
| `config` | the backend config dict as applied (defaults filled in) |
| `seconds` | 5 / 10 / 15, or null |
| `tokens` | real tokens T |
| `heads`, `head_dim` | 56, 128 |
| `dtype` | e.g. `bf16`, `q bf16, kv bf8 lofi` |
| `ms_median`, `ms_p10`, `ms_p90` | per-call attention time over the timed reps, ms (sparse backends: scoring + attention) |
| `kept_fraction` | kept share of the keepable key blocks, averaged over query blocks and heads (1.0 for dense) |
| `rel_l2_vs_dense`, `pcc_vs_dense`, `scale_vs_dense` | vs exact dense attention on the same inputs (see above) |
| `commit` | source commit (`+dirty` if modified) |
| `timestamp` | UTC ISO 8601 |

Optional fields: `input`, `layer`, `step`, `reps`, `extra` (platform-specific details; here for the sparse backends
`scoring_ms_median` and `attention_only_ms_median`, each part timed alone, and for VSA / cube the layout: `rung`,
`shuffle`, `video_grid`, `topk`, ...). A GPU record from FA /
`sol_attn` on the same safetensors files, timed with CUDA events over warm calls, fits this schema directly. If
the GPU reference is the FA bf16 output instead of exact attention, say so in `extra`; FA's own error vs exact is
at the 1e-3 relative-L2 level.

## Reproducing on a Galaxy

tt-metal: this branch (`sosborne/sparse-ring-attn-bench`). It holds the sparse ring op, the Sol pre-pass (from
`main`) and the VSA routing; nothing else to merge.

```bash
git clone https://github.com/tenstorrent/tt-metal.git && cd tt-metal
git checkout sosborne/sparse-ring-attn-bench
git submodule update --init --recursive
./build_metal.sh && ./create_venv.sh && source python_env/bin/activate
uv pip install safetensors
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_METAL_TDP_LIMIT_WATTS=160
```

Run (the bench opens the 4x8 mesh with a 1D ring fabric itself):

```bash
B=models/tt_dit/benchmarks/sparse_ring_attention
# smoke: one length, every backend at its defaults
python $B/bench_tt.py --synthetic --seconds 5 --reps 3 --warmup 1 \
    --attention-backend fa,sol_attn,video_sparse_attn,cube_sparse_attn --output smoke_tt.jsonl
# synthetic, all three lengths; also save the inputs for other platforms
for s in 5 10 15; do
  python $B/bench_tt.py --synthetic --seconds $s --save-inputs /data/syn_${s}s.safetensors \
      --attention-backend fa,sol_attn,video_sparse_attn,cube_sparse_attn --output results_tt.jsonl
done
# captured tensors; sink = text length, as SGLang's sol_attn with sink_tokens=<text length>
python $B/bench_tt.py --inputs /data/h3_qkv --attention-backend fa,sol_attn --output results_tt.jsonl
python $B/bench_tt.py --inputs /data/h3_qkv --attention-backend sol_attn \
    --attention-backend-config '{"tau": 1.0, "sink_tokens": "text_len"}' --output results_tt.jsonl
# LoFi with bf8 K / V
python $B/bench_tt.py --inputs /data/h3_qkv --attention-backend sol_attn --lofi --kv-dtype bf8 --output results_tt.jsonl
# VSA without the shard balancing, for comparison
python $B/bench_tt.py --inputs /data/h3_qkv --attention-backend video_sparse_attn \
    --attention-backend-config '{"VSA_sparsity": 0.9, "shuffle": false}' --output results_tt.jsonl

python $B/merge_results.py results_tt.jsonl results_gpu.jsonl --latest
```

Expected output: one log line per input and backend, for example

```
syn_5s... sol_attn: <ms> ms (p10 <ms>, p90 <ms>), kept 0.xxx, rel L2 0.0xxx, pcc 0.99xxx, scale 0.99xx; {...
  'pcc_vs_block_sparse_ref': 0.9999x, 'scoring_ms_median': ..., 'attention_only_ms_median': ...,
  'max_abs_diff_traced_vs_eager': 0.0}
```

`merge_results.py` then prints one markdown table per input, with a `vs fa` speedup column against the dense row
of the same platform. Each record takes up to a few minutes at 15 s, mostly for the host fp64 reference;
`--ref-heads` / `--ref-shards` shrink the slice. Compare timings only from the same machine and run: Galaxy
power limits differ between systems.

### Full matrix

`run_matrix.sh` runs every case worth comparing, at 5, 10 and 15 s (17 runs per length, appended to
`results_tt.jsonl`, then one table):

| Backend | Settings | dtypes |
|---|---|---|
| `fa` | dense baseline, at the working point and at the VSA / cube rung (`{"rung": "ladder"}`) | bf16 |
| `sol_attn` | `tau` 1.0, 1.5, 2.0 (higher keeps fewer blocks) | bf16, and LoFi with bf8 K/V |
| `sol_attn` | `tau` 1.0 with `dense_prefix_queries` and `sink_tokens` = `"prefix"` (prefix rows and keys dense, as VSA) | bf16 |
| `video_sparse_attn` | `VSA_sparsity` 0.9 (base H3) and 0.8 (distilled), exempt prefix, balanced layout | bf16; 0.9 also LoFi with bf8 K/V |
| `cube_sparse_attn` | `topk_ratio` 0.5, 0.6, 0.7, 0.8 (the ratios of SGLang's example schedule [1.0, 1.0, 0.8, 0.7, 0.6, 0.5, 0.5]; 1.0 steps are dense, i.e. `fa`), balanced layout | bf16; 0.5 also LoFi with bf8 K/V |

```bash
./$B/run_matrix.sh                       # all lengths, synthetic inputs (over an hour; 15 s dominates)
LENGTHS=5 ./$B/run_matrix.sh             # one length
INPUTS=/data/h3_qkv ./$B/run_matrix.sh   # captured tensors
```

On synthetic inputs Sol's kept fraction at a given `tau` does not reflect real activations, so the `tau` sweep there
gives time against kept fraction; accuracy needs captured inputs. VSA and cube keep a fixed share of the video
blocks, so their time does not depend on the data.

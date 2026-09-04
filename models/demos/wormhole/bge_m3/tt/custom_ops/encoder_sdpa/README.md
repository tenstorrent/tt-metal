# BGE-M3 encoder-only SDPA

Status: **in use by the data-parallel serving path**.

`BgeM3AttentionJit._attend` calls `bge_encoder_sdpa_experimental` for every
layer, and `ModelArgs` selects that class for the B12/S8192 data-parallel
shape. Treat this directory as production code.

This directory holds a model-local SDPA. It builds the program from Python
with `ttnn.generic_op`, so a change to the descriptor or to the kernels needs
no `_ttnn.so` rebuild.

## Shapes

The serving path folds 2 query chunks into the head dimension before the call,
so SDPA reads 4096 queries per head. The GQA head-broadcast keeps every query
head over the full sequence, so the result stays exact.

- Q `[6, 32, 4096, 64]` BF4
- K `[6, 16, 8192, 64]` BF4
- V `[6, 16, 8192, 64]` BF4
- non-causal, scale 1 (the build folds the scale into the Q weight)
- fixed 8x8 worker grid

`attention.py` picks the chunk plan:

| Request | q_chunk | k_chunk | fp32_dest_acc_en |
| --- | --- | --- | --- |
| serving (BF4) | 256 | 2048 | False |
| `quality_mode` masked | 128 | 512 | True |

The serving plan also sets `direct_concat_heads`, so the writer emits the
concat-head order and the separate concat-heads program is not needed.

## Compact valid lengths

`use_runtime_lengths` takes a compact `[B, 1]` uint32 tensor of per-request
valid lengths. The kernels build only the boundary mask tiles, and they skip a
key chunk that holds only padding. A dense `[B, 1, S, S]` mask costs about
1.5 GiB at B12/S8192, so the serving path never builds one.

## Why the forwarding semaphores are omitted

There are 6144 Q work units: `B6 * HQ32 * 32 Q chunks`. Dividing across 64
cores gives exactly 96 units per core, or three complete heads per core. No
head crosses a core boundary, so the KV forwarding chains have no participants.
The descriptor sends fourteen zero chain fields per core and allocates no
forwarding semaphore.

## Kernel provenance

`compute.cpp`, `compute_common.hpp`, `reader.cpp`, and `writer.cpp` started as
copies of `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/`.
They now carry the BF4 paths, the compact masking, the padded-chunk skip, and
the direct concat-head writer.

The copies drift when upstream changes the shared headers. A drift shows up as
a compile error on a changed SFPU signature. Re-sync the affected call sites
against the upstream kernels.

# Gemma3 Performance Benchmark Commands (Blackhole)

Repro commands for benchmarking Gemma3-12B / 27B decode throughput on Blackhole.
All numbers below are **ISL 128 / OSL 128, batch-1**, traced, with on-device sampling (ODS)
on multi-chip. Decode throughput is reported the apples-to-apples way we align with server CI:
`decode tps = 1000 / mean_tpot`.

## Common preamble

```bash
cd "$TT_METAL_HOME"        # tt-metal repo root
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole
MGD=$PWD/tt_metal/fabric/mesh_graph_descriptors      # mesh graph descriptor dir

# Reset before each run. NOTE: reset is per-BOARD. A 2x P300 host has 2 boards
# (indices 0 and 1), each with 2 chips -> `tt-smi -r 0 1` resets all 4 chips.
tt-smi -r 0 1
```

## ISL / OSL

- All `performance-batch-1` runs use `input_data_questions_prefill_128.json`.
- Prefill is bucketed to the **128** sequence-length bucket, so **ISL = 128**.
- **OSL** is `max_generated_tokens` in the `performance-batch-1` parametrize entry of
  `text_demo.py` (stock default **200**). For an exact **ISL 128 / OSL 128** measurement, set
  that entry to `max_generated_tokens=128` and `stop_at_eos=False` (fixed 128 decode steps).
- Decode t/s/u is steady-state (per-token) and effectively independent of OSL; TTFT reflects
  the ISL-128 prefill.

---

## TEXT-ONLY  (`text_demo.py`, `-k performance-batch-1`)

### Gemma3-12B — 1 chip (P150)
```bash
TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 HF_MODEL=google/gemma-3-12b-it \
TT_MESH_GRAPH_DESC_PATH=$MGD/p150_mesh_graph_descriptor.textproto \
pytest models/demos/multimodal/gemma3/demo/text_demo.py -k "performance-batch-1" -s -q
```

### Gemma3-12B — 2 chips (1× P300) + ODS
```bash
TT_VISIBLE_DEVICES=0,1 MESH_DEVICE=P300 HF_MODEL=google/gemma-3-12b-it TT_GEMMA3_ODS=1 \
TT_MESH_GRAPH_DESC_PATH=$MGD/p300_mesh_graph_descriptor.textproto \
pytest models/demos/multimodal/gemma3/demo/text_demo.py -k "performance-batch-1" -s -q
```

### Gemma3-27B — 2 chips (1× P300) + ODS
```bash
TT_VISIBLE_DEVICES=0,1 MESH_DEVICE=P300 HF_MODEL=google/gemma-3-27b-it TT_GEMMA3_ODS=1 \
TT_MESH_GRAPH_DESC_PATH=$MGD/p300_mesh_graph_descriptor.textproto \
pytest models/demos/multimodal/gemma3/demo/text_demo.py -k "performance-batch-1" -s -q
```

### Gemma3-27B — 4 chips (2× P300, full box) + ODS
```bash
TT_VISIBLE_DEVICES=0,1,2,3 MESH_DEVICE=P150x4 HF_MODEL=google/gemma-3-27b-it TT_GEMMA3_ODS=1 \
TT_MESH_GRAPH_DESC_PATH=$MGD/p300_x2_mesh_graph_descriptor.textproto \
pytest models/demos/multimodal/gemma3/demo/text_demo.py -k "performance-batch-1" -s -q
```
- 4 Blackhole chips auto-detect as device name `P150x4` (count-based); the logical `(1,4)`
  mesh maps onto the physical 2× P300 (`p300_x2` descriptor).
- For 12B on 4 chips, swap `HF_MODEL=google/gemma-3-12b-it`.

### Measured (ISL 128 / OSL 128, batch-1, traced, +ODS on multi-chip)

| Model | HW | TTFT | Decode tps |
|-------|--------------------------|----------|------------|
| Gemma3-12B | 1× P150 (1 chip)         | 68.1 ms  | 31.42 |
| Gemma3-12B | 1× P300 (2 chips)        | 113.3 ms | 44.47 |
| Gemma3-27B | 1× P300 (2 chips)        | 121.7 ms | 18.28 |
| Gemma3-27B | 2× P300 (4 chips)        | 140.0 ms | 27.49 |

---

## TEXT + IMAGE  (`vision_demo.py`, `-k "performance and batch1-trace"`)

Same env as the matching text-only config; only the test node changes. Example — 27B on 2× P300:

```bash
TT_VISIBLE_DEVICES=0,1 MESH_DEVICE=P300 HF_MODEL=google/gemma-3-27b-it TT_GEMMA3_ODS=1 \
GEMMA3_CONCAT_HEADS_MODE=off \
TT_MESH_GRAPH_DESC_PATH=$MGD/p300_mesh_graph_descriptor.textproto \
pytest models/demos/multimodal/gemma3/demo/vision_demo.py -k "performance and batch1-trace" -s -q
```

### Image runs currently FAIL with an L1 OOM
The SigLIP vision encoder overflows L1 in `ttnn.experimental.nlp_concat_heads`:

```
Out of Memory: Not enough space to allocate 6291456 B L1 buffer across 8 banks,
where each bank needs to store 786432 B, but bank size is 1436800 B
(allocated: 792832 B, free: 643968 B, largest free block: 641920 B)
```

- Reproduces on both 27B and 12B. The vision path was never validated on Blackhole (also not
  exercised on main), so this is a bring-up gap, not a regression from these optimizations.
- Attempted, did **not** fully resolve: forcing SDPA output to DRAM (OOM is on the
  `nlp_concat_heads` *output*, not its input). A real fix (DRAM output + chunking the
  concat-heads/vision attention to stay under the 1.5 MB/bank L1 cap) is tracked separately.
- So there is currently **no image (text+image) 128/128 number** on Blackhole. Text-only
  numbers above are unaffected.

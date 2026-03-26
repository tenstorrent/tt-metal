# Circular Buffer Model

The fundamental coordination primitive between reader, compute, and writer kernels.
This model is hardware-stable — the CB API rarely changes.

## Concept

A circular buffer (CB) is a typed, tile-oriented FIFO in L1 SRAM shared between
producer and consumer kernels:

```
Reader (DM0)         Compute (T0/T1/T2)       Writer (DM1)
     |                      |                       |
  DRAM → [CB_IN_0]  →  process  →  [CB_OUT_0]  →  DRAM
```

## Producer API (reader/writer kernel)

```cpp
cb_reserve_back(cb_id, num_tiles);   // wait until space is available
// ... fill tiles at cb_write_pointer(cb_id) ...
cb_push_back(cb_id, num_tiles);      // signal tiles are ready
```

## Consumer API (compute kernel)

```cpp
cb_wait_front(cb_id, num_tiles);     // wait until tiles are available
// ... read tiles at cb_read_pointer(cb_id) ...
cb_pop_front(cb_id, num_tiles);      // signal tiles are consumed
```

## Host-side Configuration

```cpp
CircularBufferConfig cb_config = CircularBufferConfig(
    total_size_bytes,
    {{cb_index, dataformat}}
).set_page_size(cb_index, tile_size_bytes);

auto cb = CreateCircularBuffer(program, core_range, cb_config);
```

See `tt_metal/api/tt-metalium/circular_buffer_config.hpp` for full API.

## Design Rules

- **CB index**: integer 0–31. Convention: 0/1 for inputs, 16 for output.
- **Size**: must fit in L1. Size all CBs at once and verify total ≤ 1.5 MB.
- **Double-buffering**: use 2× tile size to overlap data movement with compute.
- **Format**: must match the data format of the tensor being streamed.
- **Multiple CBs**: use separate CBs for each input tensor and for output.

## Approximate Tile Sizes by Format

| Format | Bytes/tile |
|--------|-----------|
| BFLOAT16 | ~2048 |
| BFLOAT8_B | ~1088 |
| BFLOAT4_B | ~576 |
| FLOAT32 | ~4096 |

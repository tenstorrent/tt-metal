# Serializing tensorbins with 16-byte-aligned payloads

The requirement discussed here is **16 bytes (128 bits), not 16 bits**.
Blackhole's pinned-upload dispatch check uses 16-byte L1 read alignment; its
current log misleadingly prints the 64-byte HOST read alignment instead.

## Why file layout matters

TTNN memory-maps tensorbins and uses pointers into the mapped file as upload
sources. The mapping base is page-aligned, so each payload's file offset
determines its alignment in memory. Pinning does not change that offset.

The current format is:

```text
[8-byte header_size][FlatBuffer metadata][tensor data region]
```

Each shard's `InlineFileStorage.offset` is relative to the tensor data region.
For a full-shard upload, the alignment requirement is:

```python
(8 + header_size + shard.offset) % 16 == 0
```

A partial upload must also include its source-region offset in this calculation.

## Writer changes

1. Align each unique shard's relative offset to 16 bytes while building the
   metadata. Record that padded offset in `InlineFileStorage.offset`, retain
   the original payload length in `size`, and reuse offsets for deduplicated shards.
2. Finish the FlatBuffer metadata, then pad its trailing end so that
   `8 + header_size` is divisible by 16. Include this padding in the stored
   `header_size`; do not include the 8-byte size field itself in that value.
3. Write each payload at its recorded offset, inserting padding between payloads
   where necessary. Tensor values, shapes, dtypes and shard sizes stay unchanged.

```python
def align16(value):
    return (value + 15) & ~15

# During metadata construction, for each unique payload:
shard_offset = align16(next_offset)
next_offset = shard_offset + len(payload)
# Record shard_offset and len(payload) in the metadata.

# After finalizing the metadata:
header_size = align16(8 + len(metadata)) - 8
output.write(struct.pack("<Q", header_size))
output.write(metadata)
output.write(bytes(header_size - len(metadata)))
# Write payloads with padding to their recorded relative offsets.
```

Aligning only the FlatBuffer itself is insufficient: the leading 8-byte size
field must be included. Aligning only the first payload is insufficient if later
shards start at unaligned offsets.

## Existing files and validation

If every existing shard offset is already divisible by 16, a local copy can be
fixed by extending the metadata region with padding and updating `header_size`.
Otherwise, repack the payload region and update the shard offsets as well.
The existing reader accepts trailing metadata padding; no new field is needed.

Our local MLP down-projection pair uses payload offsets **3000** (unaligned)
and **3008** (aligned). Adding 8 bytes of metadata padding preserved all payload
bytes, and both files matched exact device readback on all 32 shards.

Validate all shard offsets, compare payload bytes and loaded tensor metadata,
then compare device readback. Use `demo/compare_cache_alignment.py` for the local
pair. Absence of alignment logs proves neither alignment nor fast-path use: small
uploads and unsuccessful pin attempts bypass that check. Measure transfer time
separately from file loading and readback.

Implementation locations:

- `ttnn/core/tensor/serialization.cpp`: file writer, header size and mmap reader.
- `ttnn/core/tensor/flatbuffer/tensor_flatbuffer.cpp`: shard offsets and deduplication.
- `ttnn/core/tensor/flatbuffer/tensor.fbs`: `InlineFileStorage` schema.
- `tt_metal/impl/buffers/dispatch.cpp`: pinned-source alignment check.

Production serialization has not been changed. The experiment only created
local files under `/tmp/gemma_tensorbin_alignment`; `/mnt` files were not modified.

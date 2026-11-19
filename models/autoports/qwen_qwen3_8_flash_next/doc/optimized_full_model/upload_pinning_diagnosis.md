# Real-fixture upload pinning hypothesis

Status: AutoFix discriminating hypothesis; the driver root cause is **not proven**.
This investigation used CPU-only Torch inspection and source reads. It opened no
TT device, issued no pin ioctl, changed no implementation, and measured no device
performance. The stage owner reported a stall after `INPUT_UPLOAD_BEGIN` and
before `KEY_UPLOAD_DONE`, following reset, a passing mesh arithmetic check, and
passing synthetic raw scorer probes. Those observations locate the reported
failure before either scorer executes; they do not identify which upload
subsystem is blocked.

## Observed host-storage evidence

Fixture: `/home/ttuser/qwen38-flash/qsa_score_windows_205504.pt`.

| Tensor | Shape | dtype | Storage bytes | Address in observed CPU run | Mapping |
|---|---|---|---:|---|---|
| `source['keys']` | `[1,1,65536,128]` | `torch.bfloat16` | 16777216 | `0x786361c0a140` | File-backed `.pt`, `rw-p` |
| `.bfloat16()` | same | same | 16777216 | `0x786361c0a140` | Same file-backed mapping |
| `.bfloat16().clone()` | same | same | 16777216 | `0x786360bff040` | Anonymous `rw-p`, device/inode `00:00 0` |

The source is contiguous with strides `(8388608,8388608,128,1)`. Conversion to
its existing BF16 dtype does not allocate. Calling `.contiguous()` also does not
force a copy of this already-contiguous tensor. Thus the upload helper in the
failing probe preserves the file-backed storage, while the query path explicitly
clones. Pointer values are process-specific; pointer equality and mapping type
are the reproducible observations.

Reproduce without importing TTNN or opening devices:

```bash
python - <<'PY'
import torch

path = '/home/ttuser/qwen38-flash/qsa_score_windows_205504.pt'
source = torch.load(path, weights_only=True, mmap=True)
original = source['keys']
converted = original.bfloat16()
contiguous = converted.contiguous()
cloned = converted.clone()
for name, tensor in [
    ('original', original), ('converted', converted),
    ('contiguous', contiguous), ('cloned', cloned),
]:
    address = tensor.data_ptr()
    mapping = None
    with open('/proc/self/maps') as maps:
        for line in maps:
            low, high = [int(value, 16) for value in line.split()[0].split('-')]
            if low <= address < high:
                mapping = line.strip()
                break
    print(name, 'shape=', tuple(tensor.shape), 'dtype=', tensor.dtype,
          'strides=', tensor.stride(), 'contiguous=', tensor.is_contiguous(),
          'storage_bytes=', tensor.untyped_storage().nbytes(),
          'pointer=', hex(address), 'mapping=', mapping)
print('converted aliases source:', converted.data_ptr() == original.data_ptr())
print('contiguous aliases source:', contiguous.data_ptr() == original.data_ptr())
print('clone aliases source:', cloned.data_ptr() == original.data_ptr())
print('clone values equal:', torch.equal(cloned, original))
PY
```

Observed conversion/source alias: `True`; observed clone/source alias: `False`.
The clone copies the same BF16 values; no dtype/fidelity/model policy changes are
proposed.

## Inspected source chain

Paths below are relative to this tt-metal checkout unless absolute. Line numbers
refer to the inspected live source on 2026-11-14.

1. `models/autoports/qwen_qwen3_8_flash_next/tests/probe_causal_indexer_scores.py:31-32`
   uses `torch.load(..., mmap=True)` and `.bfloat16()` without cloning keys.
   Lines 60-67 call `ttnn.from_torch(x.contiguous(), ..., mesh_mapper=ReplicateTensorToMesh(mesh))`.
   Lines 78-80 bracket the stalled key upload. Line 83 already clones queries.
2. `ttnn/ttnn/operations/core.py:283-429` forwards `from_torch` to the Tensor
   constructor; no unconditional source clone is inserted.
3. `ttnn/core/tensor/py_to_tt_tensor.cpp:264-299` chooses the source layout when
   on-device conversion is legal and memory is sufficient, then forwards the
   host span to `create_distributed_tensor`. Lines 496-510 perform device
   placement and final layout conversion. BF16-to-BF16 tiled construction can
   therefore upload borrowed row-major data before device tilization.
4. `ttnn/core/distributed/distributed_tensor.cpp:148-159` returns a borrowed
   `HostBuffer` when the row-major physical and logical shapes agree and the
   dtype already matches. Lines 278-290 reuse one host buffer across every
   coordinate in the fully replicated distribution.
5. `tt_metal/impl/tensor/tensor_impl.cpp:19-31` sums each local shard's byte
   count without deduplicating shared backing storage. For four local replicas
   of these 16 MiB keys, the dispatch threshold sees **64 MiB**.
6. `tt_metal/impl/tensor/pinned_upload.hpp:18-20` sets the threshold to
   `32 * 1024 * 1024` bytes. `distributed_tensor_apis.cpp:100-105` selects
   pinning above that threshold when hardware/IOMMU support it. A 16 MiB
   single replica would be below the threshold, but the four-shard count is
   above it.
7. `tt_metal/impl/tensor/distributed_tensor_apis.cpp:313-331` uses that summed
   size and invokes `PinnedMemoryCache::try_pin` for each coordinate with
   `PinnedMemoryDeviceAccess::ReadOnly`. This is the kernel page-pinning path;
   the earlier `HostBuffer::pin()` accessor is only a lifetime owner, not a
   kernel pin operation.
8. `tt_metal/third_party/umd/tt-kmd-lib/src/tt_kmd_lib.c:491-521` marshals
   `TENSTORRENT_IOCTL_PIN_PAGES`, including the read-only flag.
9. Installed KMD `/usr/src/tenstorrent-2.10.0/memory.c:641-644` chooses
   `gup_flags = read_only ? 0 : FOLL_WRITE` and locks the per-file mutex;
   line 671 calls `pin_user_pages_fast_longterm`. Lines 203-206 implement
   that helper as `pin_user_pages_fast(..., gup_flags | FOLL_LONGTERM, ...)`.
   Source shows the mechanism but does not prove where the live call stalled.

The source chain supports a distinction between synthetic anonymous keys and
file-backed real-fixture keys. It does not establish that file-backed pinning
must fail generally, nor that this exact live execution reached the pin ioctl.
A kernel stack or a controlled upload comparison is required for that claim.

## Minimal discriminating fix and next observation

Change only fixture materialization before the upload:

```python
keys = source['keys'].bfloat16().clone()
```

Apply the same treatment to fixture-loaded keys in the raw probe when testing
capacity 65536. Small capacity 128 is a 32 KiB key tensor, so four replicas do
not cross the pinning threshold and cannot discriminate this specific path.

The stage owner should rerun the same bounded upload/probe on the same healthy
mesh, retaining the `INPUT_UPLOAD_BEGIN` / `KEY_UPLOAD_DONE` markers and exact
command/log. A passing cloned upload with unchanged values would support the
file-backed-storage/pinning hypothesis and allow scorer validation to proceed.
If it still stalls, the clone has refuted this proposed workaround; inspect the
live upload/KMD stack rather than blaming scorer CB or math logic. Even a
passing clone does not alone prove a particular kernel page-fault or spin-loop
root cause.

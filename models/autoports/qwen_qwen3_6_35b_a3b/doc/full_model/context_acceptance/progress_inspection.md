# Read-only host progress inspection

The parent requested a brief inspection of its running acceptance process, PID4007576, CPython3.12.12. The existing `../logs/context_progress_stack.txt`, `_2.txt`, and `_3.txt` expose decoder object pointers but not `layer_idx`. Read-only `/proc/4007576/mem` access was available to the owning user, so small bounded `os.pread` calls recovered the immutable Python attributes. No debugger attachment, process stop, signal, injected target code, dependency installation, TTNN import, device operation, or raw memory dump was used.

| Existing snapshot | Decoder address | `layer_idx` | `kind` |
| --- | --- | ---: | --- |
| `context_progress_stack.txt` | `0x724a980ca9f0` | 1 | linear_attention |
| `context_progress_stack_2.txt` | `0x724a980e6f30` | 4 | linear_attention |
| `context_progress_stack_3.txt` | `0x724a980fedb0` | 6 | linear_attention |

Snapshot3 reports offset163264 and public length262143: approximately62.3% through that layer's attention pass, not through the complete40-layer model. This inspection supplies neither throughput/ETA nor acceptance evidence. A stack snapshot identifies where the host was executing at one instant; device completion can lag queued host work.

The complete stable address map is `progress_layer_addresses.json`, recorded2026-11-14T15:06:52.495699Z. It contains40 entries, each checked against its actual integer `layer_idx` and attention kind. Object chain: `BoundedPrefill 0x724a90115250` → `.model 0x724ab5bc48f0` → `.layers 0x724a980bb3c0`. These addresses apply only to PID4007576 with `/proc/PID/stat` starttime8454562; PID reuse or process restart invalidates them. Models/layers are constructor-owned and remain referenced for this run.

## CPython layout references

The reader uses installed x86-64 CPython3.12 headers, not guessed scanning of arbitrary memory:

- `/usr/include/python3.12/internal/pycore_object.h:373`: managed dictionary/value pointer is three pointer widths before the object. Low bit1 denotes an unmaterialized values pointer; the reader rejects that case rather than interpreting it as a dict.
- `/usr/include/python3.12/cpython/dictobject.h:11`: dictionary keys and optional split-values pointers; `/usr/include/python3.12/internal/pycore_dict.h:64` onward defines entry counts, index-byte size, and Unicode/general entries.
- `/usr/include/python3.12/cpython/unicodeobject.h:52-148`: compact ASCII header length/hash/state; this build's ASCII payload starts at byte40. Only bounded ASCII attribute names/kinds are read.
- `/usr/include/python3.12/cpython/listobject.h:5`: list size and item-vector pointer.
- `/usr/include/python3.12/cpython/longintrepr.h:83-89,122`: compact integer sign/tag and first digit. Layer indices are0..39; the reader verifies this exact bounded case.

Live reads are not a stop-the-world snapshot. Here the requested model list and layer attributes are immutable after construction; bounds/type-layout assertions and index equality provide sanity checks. This is a process-specific diagnostic, not a general-purpose Python memory parser.

## Reproduction command

The following is the read-only procedure used, with an added PID-starttime guard for accidental later reuse. It prints only the small pointer/index map. It executes in the inspector process and never invokes Python APIs inside the target process. Do not reuse against another Python build without validating layouts and obtaining its own known object pointer.

```bash
python - <<'PY'
import os, struct, json
from pathlib import Path
pid = 4007576
start = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]
assert start == '8454562', 'Process identity changed; stop'
fd = os.open(f'/proc/{pid}/mem', os.O_RDONLY)
def read(p, n):
    return os.pread(fd, n, p)
def u64(p):
    return struct.unpack('<Q', read(p, 8))[0]
def string(p):
    h = read(p, 40)
    n = struct.unpack_from('<q', h, 16)[0]
    state = struct.unpack_from('<I', h, 32)[0]
    assert 0 <= n <= 200 and state & 64
    return read(p + 40, n).decode('ascii')
def attrs(obj):
    d = u64(obj - 24)
    assert d and not d & 1, 'Unmaterialized managed values; stop'
    keys, values = u64(d + 32), u64(d + 40)
    kh = read(keys, 32)
    ibytes, kind = kh[9], kh[10]
    n = struct.unpack_from('<q', kh, 24)[0]
    assert 0 <= n < 400 and ibytes < 20 and kind in (0, 1, 2)
    out = {}
    for i in range(n):
        e = keys + 32 + (1 << ibytes) + i * (24 if kind == 0 else 16)
        kp = u64(e + (8 if kind == 0 else 0))
        vp = u64(values + i * 8) if values else u64(e + (16 if kind == 0 else 8))
        if kp and vp:
            out[string(kp)] = vp
    return out
try:
    model = attrs(0x724a90115250)['model']
    layers = attrs(model)['layers']
    n, items = u64(layers + 16), u64(layers + 24)
    assert n == 40
    rows = []
    for i in range(n):
        obj = u64(items + i * 8)
        a = attrs(obj)
        tag = u64(a['layer_idx'] + 16)
        assert tag >> 3 <= 1
        digit = struct.unpack('<I', read(a['layer_idx'] + 24, 4))[0]
        index = 0 if tag & 3 == 1 else digit if tag & 3 == 0 else -digit
        assert index == i
        kind = string(a['kind'])
        assert kind in ('linear_attention', 'full_attention')
        rows.append({'index': index, 'address': hex(obj), 'kind': kind})
    print(json.dumps({'pid': pid, 'layers': rows}, indent=2))
finally:
    os.close(fd)
PY
```

For later parent-collected stack snapshots, simply look up the reported `layer` pointer in the saved JSON. No further target-memory access is needed to interpret those addresses during this process lifetime. Preserve normal device-usage ownership and avoid treating this inspection as a reason to interrupt or repeat the acceptance run.

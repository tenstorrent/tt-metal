# Minimal device-11 corruption reproducer

This directory is independently portable. It contains the exact hardware-qualified
`device_read_repro_v86.py`, its optional NoC helper, and a 5,119,861-byte fixture.
`run.py` adds path setup, file-integrity verification and unambiguous exit handling;
it does not change the device operation or its checking loop.

The default test is one stock `ttnn.experimental.fast_reduce_nc` operation on one
physical device, repeated against fixed packed reference bytes. It needs an already
built matching tt-metal checkout and its Python environment, with Torch and TTNN.
No model weights, serving process, KV cache, fabric, custom C++ kernel or full mesh
is required. This is an operation-level reproducer; the full handoff also contains
arithmetic-free copy evidence establishing the transfer-corruption mechanism.

## 1. Verify the files — no hardware

```bash
python3 verify.py
```

All file hashes must pass before loading the fixture. The fixture is a Torch archive
of activation/reference tensors, not model weights. Its SHA-256 is
`8c644b2ee1e87d8b46a9d3b866f4dcbd497ff822de36544c062991fb7e7dbe34`.

## 2. Run on the known host

Use a fresh output path. This opens physical UMD device 11, recorded as KMD 27 /
PCI `0000:44:00.0` on `wh-glx6u-04:49752`. Verify that mapping and availability
before running. Request slots, UMD IDs and KMD indices are different namespaces.
The command below is for the receiving team; no new hardware run was performed
when assembling this handoff.

```bash
unset TT_METAL_SLOW_DISPATCH_MODE TT_VISIBLE_DEVICES
/localdev/ctr-ifabijanic/python_env/bin/python run.py \
  --tt-metal /tmp/tt-metal-55041 \
  --library-path /opt/openmpi-v5.0.7-ulfm/lib \
  --device 11 --iterations 100000 \
  --output /tmp/device11-minimal-repro-unique
```

The wrapper works from any current directory. It uses the specified checkout as
its working directory and prepends that checkout's Python and native-library paths.
Use `--dry-run` to check files and inspect the command without importing TTNN.
On another machine, replace the checkout, interpreter/library paths and verified
device ID. Start with the recorded build: HEAD
`609714c3ce2769165ee94334bf652277725dbb00`. The fixture's expected output is bitwise;
changing the implementation or architecture needs a separately qualified reference.
The recorded checkout is an existing edited build, not a claim of a pristine release.

The test stops after preserving its first fault. Add `--keep-going` to finish the
iteration budget and retain all failures; this increases artifact storage.
There is no automatic reset, retry-until-pass, firmware change or assertion relaxation.

| Exit from `run.py` | Meaning |
|---|---|
| 0 | All requested iterations completed with no observed mismatch |
| 1 | A mismatch was captured and the completed report qualifies the result |
| 2 | Setup/runtime/integrity failure; inspect the log and `report.json` |

The wrapper distinguishes a Python import failure (which may also exit 1) from an
actual captured mismatch. A bounded clean run is not proof that the issue is fixed.
Use device 10 as a matching control only after verifying its current identity.

## 3. Inspect and audit the failure — CPU only

The output directory contains `report.json`, `launcher.json`, and
`fault-NNNNNN.pt.gz`. The capture contains unchanged source bytes, expected bytes,
bad output bytes, three rereads, and the mismatch description. Iterations in fault
filenames and example records are zero based.

```bash
python audit_capture.py /tmp/device11-minimal-repro-unique/fault-000559.pt.gz
```

Replace the example filename with the actual saved capture. This audit uses only the Python standard library and never imports Torch or TTNN.
Its restricted archive reader supports the recorded contiguous CPU Byte/BFloat16/Long
storages and rejects other formats explicitly. A stock reduction's output mutation need not
be XOR `0x40`: computation and BF8 packing transform an input transfer error.
The recorded first post-reset stock fault changed output byte 78,455 from
`0xDA` to `0xCA` (XOR `0x10`), with intact source and three matching rereads.

The unchanged test captured 6 / 30,000 stock failures before the recorded reset and
116 / 100,000 after it. Matched device-10 stock controls were clean in 30,000 and
100,000 copies of the operation. The full report documents the separate copy and
reader-only evidence; it does not diagnose a particular defective component.

## Files

- `device_read_repro_v86.py`: unchanged tested implementation.
- `55041-read-fixture-v86.pt.gz`: exact compact source/reference fixture.
- `noc_reduce_v63.py`: unchanged helper for optional explicit NoC controls; not
  imported by the default stock path.
- `run.py`: portable launcher; `verify.py`: standard-library file checker.
- `audit_capture.py`: CPU-only raw-capture audit.
- `raw_archive.py`: restricted byte reader for the recorded Torch archive format.
- `SHA256SUMS.json`: hashes of every file above and this README.

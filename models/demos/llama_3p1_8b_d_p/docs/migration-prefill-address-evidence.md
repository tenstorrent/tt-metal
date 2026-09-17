# Prefill cache-address validation

## Result

The independent 2K cache-address test passed on a 32-chip Blackhole Galaxy on
17 September 2026. It checked every physical page in both cache slots and all
32 layers.

| Check | Result |
| --- | ---: |
| Sequence capacity per slot | 2,048 tokens |
| Slots | 2 |
| Layers | 32 |
| K/V configurations | 16 (K and V for each of 8 KV heads) |
| Physical pages checked | 65,536 |
| Packed bytes checked | 285,212,672 |
| DRAM banks covered | 8 |
| Configuration/boundary combinations | 96 |
| Device test cases | 1 passed |
| Native environment check | Passed |
| Source files pinned before and after the run | 61, unchanged |
| Device close | Clean |

The test used BFP8_B cache tensors, SP=4 and TP=8. It completed on
`bh-glx-120-c04u08` under Slurm job `109038`. The device cluster closed at
16:11:40 UTC. The allocation was later released at the user's request.

## What the test proves

A migration manager uses an address table to find cache data. A copy can appear
correct when the same incorrect table is used for both its source and its
destination. This test gives the table an independent reference.

1. Make a unique tag from the slot, layer, KV head and token position.
2. Encode K with positive values and V with negative values. Use values that
   BF16 and BFP8_B can store exactly.
3. Write the tags with the production cache allocation and write functions.
4. Read each live TTNN tensor by its SP row, TP column and logical position.
   Do not use the address table for this read.
5. Compare the tensor values with the CPU tags.
6. Read each packed page through the exported address table. Decode its bytes
   and compare the result with the independently read tensor values.

All comparisons require exact values. This is an address and layout test, so
it does not use PCC or a numerical error tolerance. The host tests also check
that wrong slot, head and page mappings are detected.

The boundary checks cover token positions 0, 224, 256, 992, 1,024 and 2,016.
They cross tile, SP-stripe and 1K-chunk boundaries, and include the final page.

## Run the device test

Use the project's configured Blackhole runtime on one allocated Galaxy:

```bash
pytest -q -s \
  models/demos/llama_3p1_8b_d_p/tests/migration/test_kv_cache_table_readback.py \
  -k test_llama_kv_table_readback_matches_live_tensor
```

The recorded case is
`test_llama_kv_table_readback_matches_live_tensor[blackhole-line-galaxy-4x8]`.
Its device fixture selects the 4×8 mesh and the 1D fabric.

## Evidence

The tested source commit was
`70f1d4c7773823d773e07451f2d1e786a07ba5eb`.
The two published test files are byte-identical to that tested commit.

Shared evidence directory:
`/data/divanovic/llama31-8b-disagg/evidence/m03-prefill-table-address/device-preparation-004/`

| Artifact | SHA-256 |
| --- | --- |
| `result.json` | `6cb1620b7a1cf26102fc48546f487ab705bde84969ed90dd874c68e4c9814e75` |
| `device.log` | `a05269ea763edd356006cd208c69401f9c0ac4376e2207f8c5beeb944fba2c87` |
| `pytest.xml` | `3bbc0804bc580f193b57ce7355fa27b01485c47867630622f38cecb89a821d0d` |
| `source-pins.json` | `481e5c88f9bde15e4b04fe0515510ead0cce7ecf7b41b72b57ca1c177297deb8` |

`root-verification.json` records the independent review of these artifacts.
Earlier setup failures remain in separate attempt directories.

## Remaining migration checks

This result proves the 2K prefill address table resolves the intended data.
It does not prove native manager startup, TTNN/manager coexistence,
source registration, network transfer, transfer completion, or cache lifetime
during native transfer. Those are separate gates in the
[prefill migration test guide](migration-prefill-tests.md).

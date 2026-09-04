# ResNet50_Forge_Fe_bf16 — Quasar support matrix for the **bf16-only** tt-forge ResNet-50 ops

Per-op tests that replay **every op the tt-forge ResNet-50 graph issues under the BF16-ONLY
compiler config**, one at a time, on Quasar — using the configurations the **Forge compiler**
chose, not hand-picked ones.

Source of the configs: **`resnet50_forge_bf16_vs_quasar.xlsx`, sheet 1 — "Forge ops (bf16 only)"**,
141 ops in `@forward`. **Sheet 5 of the same workbook is the ledger of these tests** — one row per
test, with the config it replays and the result it got.

## Which compile this is, and why it is a separate directory

| | `../ResNet50_Forge_Fe/` | **this directory** |
|---|---|---|
| compiler config | `opt_level=2`, consteval, HiFi2, `remove_dead_values`, `max_legal_layouts=8`, bf16 | `CompilerConfig()` with **only** `enable_optimization_passes=True` and `default_df_override=Float16_b` |
| ops in `@forward` | 96 | **141** |
| memory | L1, HEIGHT/BLOCK **sharded**, pinned 56/49/25/16-core ranges | **DRAM, INTERLEAVED everywhere** — no shard spec, no core ranges |
| math | `MathFidelity.HiFi2` | **`MathFidelity.HiFi4`, `fp32_dest_acc_en=true`** |
| weights | pre-prepared by `prepare_conv2d_weights` in const-eval functions | **raw OIHW handed straight to the op** from host memory — no `prepare_conv2d_*` anywhere |
| layout ops | 4 `to_memory_config` (resharding) | a layout change around every conv |

The two are different compiles of the same model, so they are different tables and different
tests. Neither supersedes the other: the sharded one asks "does the optimised Forge output run on
Quasar?", this one asks "does the plain bf16 Forge output run on Quasar?".

**The practical difference when running:** because nothing here pins a core range, **no test is
ever skipped for device grid size** and every case runs unchanged on the 8×4 = 32-worker Quasar
part. The sharded suite skips its 56-core cases on anything smaller than a full part.

Same layout and conventions as `../ops/` and `../ResNet50_Forge_Fe/`: a flat directory of
standalone `test_*.py` files. No `conftest.py`, no shared helper module, no generated config
module. Each file carries its own config table and its own operand builder.

---

## What the graph contains

| Forge op | count | Quasar op | route | files |
|---|---|---|---|---|
| `ttnn.conv2d` | 53 | `quasar.conv2d` | direct | `test_op*_conv2d_*.py` |
| `ttnn.add` | 16 | `quasar.add` | direct | `test_op*_add_*.py` |
| `ttnn.relu` | 16 | — **none** — | fuse into the preceding `add` | `test_op*_relu_*.py` |
| `ttnn.reshape` | 2 | `quasar.reshape` | direct | `test_op002_…`, `test_op139_…` |
| `ttnn.permute` | 1 | — **none** — | `quasar.transpose` ×2 | `test_op001_permute_nchw2nhwc.py` |
| `ttnn.max_pool2d` | 1 | `quasar.max_pool2d` | direct | `test_op006_max_pool2d_stem.py` |
| `ttnn.mean` | 1 | — **none** — | `quasar.avg_pool2d` 7×7 | `test_op138_mean_global_avgpool.py` |
| `ttnn.linear` | 1 | `quasar.linear` | direct | `test_op140_linear_fc.py` |
| | **91** | | | **one file each** |

(`ttnn.deallocate` and `ttnn.get_device` produce no tensor and are not rows on sheet 1.)

**These are the 91 compute ops of the graph.** Sheet 1 also carries layout-plumbing rows that move a
tensor between TILE and ROW_MAJOR without computing anything; the workbook's comparison sheets 3 and
4 leave those out, and so does this directory.

**Three gaps**, the same three the optimised compile hits. `ttnn.experimental.quasar` binds
data-movement ops, conv2d, the pools, the matmul family and a **binary** front-end. It binds **no
plain unary activation** (hence no `relu` — `prelu`/`pow`/`polyval` are the only unary-with-param
ops there), **no general `permute`** (only `transpose`, a 2-axis swap), and **no reduction** at
all.

**There is no `xfail` anywhere in this suite.** A gap op has nothing to call, so instead of a probe
that xfails, each of those files runs the route that *does* exist — a fused `add`+RELU, `transpose`
×2, `avg_pool2d` — as a full device test with a real PCC or exact-equality check. Every file has
exactly one test, and it either passes or fails on its merits.

The gaps themselves are watched in **one** place rather than eighteen:
`test_op_inventory_bf16.py::test_forge_ops_map_onto_the_live_quasar_build` fails the day Quasar
binds one of them, which is the signal to add a direct test in the matching file.

The relu gap is the one that matters most here: this compile does **not** fuse relu into the
residual add, so 16 relus are left stranded on an op Quasar has no binding for. The hand-written
metal model fuses them (`quasar.add_(out, ds_out, activations=[UnaryWithParam(RELU)])`, see
`../ops/test_add.py`), and each `test_op*_relu_*.py` file runs that fusion as its workaround test.

---

## Layout — one file per op call-site

The whole point of this directory: **91 standalone test files, one per compute op of the graph**,
laid out and named the way `../ops/` is — a flat directory of self-contained files, each with its
own docstring, its own config constants and its own operand builder. No `conftest.py`, no shared helper
module, no config module. Hand any single file to the LLK team and it stands alone.

```
ResNet50_Forge_Fe_bf16/
  README.md
  test_op_inventory_bf16.py                    the census + topology + build/device checks (below)

  test_op001_permute_nchw2nhwc.py              no quasar permute -> transpose x2
  test_op002_reshape_stem_flatten.py
  test_op004_conv2d_conv1.py                   the 7x7 stem
  test_op006_max_pool2d_stem.py
  test_op007_conv2d_layer1_0_conv1.py
  test_op009_conv2d_layer1_0_conv2.py
  ...                                          (one per compute row, in @forward order)
  test_op136_add_layer4_2.py
  test_op137_relu_layer4_2.py                  no quasar relu   -> fused add+RELU
  test_op138_mean_global_avgpool.py            no quasar mean   -> avg_pool2d 7x7
  test_op139_reshape_classifier_squeeze.py
  test_op140_linear_fc.py
```

The file name carries the sheet-1 row and the module it replays, so the directory listing **is** the
graph in @forward order; the numbering skips the rows this suite does not cover. `ls
test_op*conv2d*` or `pytest ... -k layer3` selects what you want without a table to consult.

**100 tests — one per op file, plus the inventory**: 91 op tests + 9 in
`test_op_inventory_bf16.py`, of which **8 need no device**. No xfail, no skips.

### What keeps 91 loose files honest

Each op file declares five constants:

```python
SHEET_ROW = 4
FORGE_OP = "ttnn.conv2d"
QUASAR_OP = "quasar.conv2d"
OPERAND_SHAPES = ((1, 1, 50176, 3), (64, 3, 7, 7), (1, 1, 1, 64))
OUTPUT_SHAPE = (1, 1, 12544, 64)
```

`test_op_inventory_bf16.py` **parses those back off disk with `ast`** — no import, no ttnn, no
device — and checks the 91 files against ResNet-50 itself: there are exactly 91 of them with unique
sheet rows in range, the op census matches sheet 1, the 53 conv files' activation / weight / bias /
output
shapes match a topology re-derived from first principles (layers `[3,4,6,3]`, widths
`[64,128,256,512]`, expansion 4, stride on the 3×3), each conv file is named after the module it
replays, and the 16 adds and 16 relus follow the bottleneck widths with each relu sitting directly
after its add. A file that is renamed, deleted, duplicated or edited into something that is no
longer ResNet-50 fails **there**, loudly, instead of quietly testing the wrong numbers 90 files
away.

Each file's docstring also carries the **verbatim TTNN IR line** and the full operand table from
sheet 1, plus its observed status from the last run — so the provenance travels with the test.

The files are generated from sheet 1 by `quasar_analysis/gen_forge_bf16_op_tests.py`; re-run it
after a recompile and the whole directory is rebuilt, IR lines and all. `SKIP_OPS` there controls
which rows get a file.

---

## How to confirm the ops actually ran on a Quasar device

"The test passed" is not by itself proof. Three separate things have to hold, and each is checked
in a different way — none of them relies on reading a run header and trusting it.

### 1. The device is a Quasar part — asserted in every single test

Every generated op file carries `_assert_quasar(device)`, called at the top of each test before it
touches the device:

```python
assert device.arch() == ttnn.Arch.QUASAR, "this test ran on %s, not Arch.QUASAR…"
```

So a green tick in any file already means *green on Quasar*; there is no run in which the suite can
pass on the wrong arch. `test_op_inventory_bf16.py::test_device_under_test_is_quasar` additionally
prints the arch and the compute grid once per run.

### 2. The op is one only Quasar can run — structural

Every op called here is `ttnn.experimental.quasar.*`. Those build **Gen2 kernels**, and the Gen1
`DataMovementKernel` / `ComputeKernel` classes every Wormhole op uses `TT_FATAL` on Quasar and vice
versa. There is no arch auto-routing: `ttnn.conv2d` fails on Quasar and
`ttnn.experimental.quasar.conv2d` with identical arguments succeeds. So these calls cannot silently
execute somewhere else.

### 3. A device program was actually built and enqueued — the attestation run

This is the one that catches a host fallback or a no-op. Run the suite under the attestation
plugin, which captures the ttnn graph around every test and records what happened underneath it:

```bash
pytest -p quasar_analysis.pytest_quasar_attest \
       models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/ \
       --attest-out quasar_analysis/forge_fe_bf16_runs/dispatch_attestation.json
```

For each test it records the ttnn entry point that was called, the **device operation** it lowered
to, that operation's **duration in nanoseconds**, and the device buffers allocated for it — e.g.

```
ttnn.experimental.quasar.add  ->  BinaryNgDeviceOperation   242 511 806 ns   device_id 1
```

A device-operation node only exists if a program was created and enqueued on the device, so its
presence with a nonzero duration *is* the proof. (On real silicon `TT_METAL_DEVICE_PROFILER=1` adds
hardware-timer per-op timings on top of this; on craq-sim the cycle counter is only an activity
measure, so the graph capture is the meaningful signal here.) The plugin sorts every test into three buckets and
exits non-zero if anything ran on a non-Quasar arch:

| bucket | meaning |
|---|---|
| **DISPATCHED** | a device program ran on Quasar — attested |
| **VIEW** | the op was called and returned but created no device program |
| **XFAILED** | none — this suite has no xfail; kept in the plugin for suites that do |

**The VIEW bucket is why this check is worth running.** `quasar.reshape` on a shape whose last
dimension does not change is a metadata-only view — 0 ns, no device program — so both reshape tests
pass *without executing anything on the device*. That is correct ttnn behaviour, not a bug, but it
is exactly the kind of thing "the test passed" hides. Pass `--attest-strict` to make VIEW a failure
too.

Records land in `quasar_analysis/forge_fe_bf16_runs/dispatch_attestation.json`, one per test, so the
claim is auditable after the fact rather than a line of prose.

### What the attestation run actually reported — 2026-09-04

```
device tests captured : 92
arch reported         : Arch.QUASAR          <- all 92, no exceptions
DISPATCHED            : 35 passing tests ran a device program on Quasar
VIEW                  :  2 (both reshapes — metadata-only, 0 ns)
not an op test        :  1 (the arch/grid check calls no ttnn op)
failed                : 54, of which 54 reached the device first
total device-operation time: 9726.4 ms across 147 dispatches (4542.0 ms on passing tests)
```

Device operations that **completed** on passing tests: `BinaryNgDeviceOperation` ×32 (the adds and
fused add+relu), `HaloDeviceOperation` + `InterleavedToShardedDeviceOperation` +
`ShardedToInterleavedDeviceOperation` (max_pool2d), `TransposeDeviceOperation` ×2 (the decomposed
permute), `ReduceDeviceOperation` (avg_pool2d), `UntilizeWithUnpaddingDeviceOperation`.

**All 54 failing tests reached the device too** — they are not "nothing ran". Real Quasar programs
executed first and the recorded last operation is where each threw:

| failing test | device programs recorded | threw at |
|---|---|---|
| `op004_conv2d` (7×7 stem) | `InterleavedToSharded`, `Halo` | the halo — cause B |
| `op123_conv2d` (1×1) | `TilizeWithValPadding`, `Matmul` | the matmul — cause A |
| `op140_linear` | `Matmul` | the matmul — cause A |

So the two failure causes are reached *after* Quasar has done real work, which is what makes them
config gaps rather than "the op never ran".

---

## Running

Against the craq-sim functional Quasar simulator:

```bash
export TT_METAL_SIMULATOR=/proj_sw/user_dev/ctr-lelanchelian/sim/qsr/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1 ARCH_NAME=quasar
export TT_METAL_CACHE=<a directory you own>          # a root-owned cache breaks every kernel build

pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/
```

`libttsim.so` must sit beside its `soc_descriptor.yaml` — UMD reads the descriptor from the same
directory, not from a separate env var.

Useful selections:

```bash
pytest ... test_op_inventory_bf16.py -k "not device"   # the 8 host-only checks, no device at all
pytest ... -k conv2d                                   # all 53 convs
pytest ... -k layer3                                   # everything in layer3
pytest ... -k "fused or via_"                          # the three gap routes
pytest ... test_op004_conv2d_conv1.py                  # exactly one op
```

---

## What the tests check

Every device test asserts, before it looks at any number:

* the **output shape** the IR records, and for conv the returned `(out_h, out_w)`;
* the **page layout** (TILE vs ROW_MAJOR) the IR records;
* `INTERLEAVED` and `DRAM` on the result, because this compile pins those everywhere;
* for conv, that the op's internally-prepared weight has the shape
  `prepare_conv2d_weights` would have produced, so the two weight-prep paths are checked to agree.

Then the numeric check. Compute ops get a PCC bound (`0.98` for the convs and the fc, `0.99` for
add/relu/mean, `0.999` for max_pool2d — max selects, it does not accumulate). **Data-movement ops
— `reshape` and `permute` — are checked for EXACT equality**, with a PCC assert alongside so a
partial corruption reports a number instead of just "not equal". Moving data must not change one
bit of it, and a loose PCC would have let the `untilize` corruption through.

The whole-graph checks live in `test_op_inventory_bf16.py`, not in the op files — see *What keeps
141 loose files honest* above. It also checks the Forge→Quasar op map against the **live ttnn
build** in both directions, so it fails when a mapped quasar op vanishes **and** when one of the
three gaps closes.

---

## Status on 2026-09-04 (craq-sim, `Arch.QUASAR`, 8×4)

**100 tests: 46 passed, 54 failed, 0 xfailed.** One test per op file, plus 9 in the inventory.

* **Per-test ledger — `resnet50_forge_bf16_vs_quasar.xlsx`, sheet 5 "Unit tests (Quasar run)".**
  100 rows: the sheet-1 row each test replays, the file and function, the quasar op it calls, every
  operand and attribute **verbatim from sheet 1**, the torch golden, the assertion, the result, the
  root cause, a copy-pasteable pytest command, and — last two columns — **SHA-pinned GitHub
  permalinks to the test file and to the run log**, so every row is auditable from the sheet alone.
  Built by `quasar_analysis/build_sheet5_unit_tests.py`, which re-reads the written workbook and
  checks it against sheet 1 and the files on disk (**assertions re-derived from the written file,
  0 mismatches, 91/91 op files with a row**).
* **Everything is on the remote**, branch
  [`ctr-lelanchelian/resnet50-forge-fe-op-tests`](https://github.com/tenstorrent/tt-metal/tree/ctr-lelanchelian/resnet50-forge-fe-op-tests) — the 91 op files, the inventory, the
  generators, the attestation plugin, and the logs:
  [`all_ops.log`](https://github.com/tenstorrent/tt-metal/blob/ctr-lelanchelian/resnet50-forge-fe-op-tests/quasar_analysis/forge_fe_bf16_runs/all_ops.log) · [`dispatch_attestation.json`](https://github.com/tenstorrent/tt-metal/blob/ctr-lelanchelian/resnet50-forge-fe-op-tests/quasar_analysis/forge_fe_bf16_runs/dispatch_attestation.json) · [`SUMMARY.txt`](https://github.com/tenstorrent/tt-metal/blob/ctr-lelanchelian/resnet50-forge-fe-op-tests/quasar_analysis/forge_fe_bf16_runs/SUMMARY.txt)

| outcome | count | where |
|---|---|---|
| PASS | 46 | 16 adds, 16 fused add+RELU (the relu route), `max_pool2d`, `avg_pool2d` (the mean route), both reshapes, the transpose-decomposed permute, and the 9 inventory checks |
| FAIL | 54 | 53 convs, the fc |
| XFAIL | 0 | by design — see *There is no `xfail` anywhere in this suite* above |

All 54 failures are device-side and fall into **two** causes — neither is a config or test defect,
and the attestation shows **all 54 reached the device before throwing**:

* **A — 34 (33 convs + the fc): `fp32_dest_acc_en=true` is rejected.**
  `program_spec.cpp:1076` — *"Compute kernel 'compute' consumes FP32 DFB 'cb_intermed0' with
  enable_32_bit_dest=true, but provides no unpack_modes entry."* Forge sets this on every conv, the
  fc and the mean, and it is passed through verbatim. Confirmed to be this flag alone: with
  `fp32_dest_acc_en=False` the same conv gets past this check and then hits a *third* pre-existing
  bug — the quasar matmul in0 receiver kernel does not compile
  (`reader_bmm_tile_layout_in0_receiver_metal2.cpp:59`, `Semaphore::get_l1_addr()` is private). So
  conv numerics cannot be reached on this build from either direction yet. Note `avg_pool2d`
  accepts the same flag and passes.
* **B — 20 convs: the conv halo scratch DFB is self-looped**, which Gen2 forbids
  (`program_spec.cpp:1439`, `gather_scratch0` ×19 / `act_sharded` ×1 for the 7×7 stem). Exactly the
  20 convs that need a halo: the stem, all 16 3×3 convs, the 3 stride-2 1×1 downsamples.
  **`max_pool2d` needs a halo too and it passes** — so this is the conv halo path specifically, not
  the halo as such.

### A third finding, not covered by this suite

**`quasar.untilize` silently corrupts TILE → ROW_MAJOR** at some tile-grid shapes: no error, just
wrong data — PCC 0.755 on `[1,1,3136,256]`, 0.756 on `[1,1,3136,128]`, 0.865 on `[1,1,196,1024]`,
0.980 on `[1,1,50176,3]`, while other shapes are exact. A probe narrows it: the upload/download
round-trip is exact everywhere, `to_memory_config`-style dispatch is not involved, and
`untilize_with_unpadding` has an extra failure mode of its own — so the fault is in `untilize`.

It is the only Quasar issue found here that produces **wrong numbers rather than an error**, which
is why the data-movement tests in this directory assert exact equality rather than a PCC tolerance.
Nothing in this suite exercises it; reproduce it with:

```bash
python quasar_analysis/probe_quasar_untilize.py
```

Every op file's docstring ends with its own status line, so a single file tells you what it did last
time without opening the log.

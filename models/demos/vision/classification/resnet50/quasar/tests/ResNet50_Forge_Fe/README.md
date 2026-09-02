# ResNet50_Forge_Fe — Quasar support matrix for the tt-forge ResNet-50 ops

Per-op tests that replay **every op the tt-forge ResNet-50 graph issues**, one at a time, on Quasar
— using the configurations the **Forge compiler** chose, not hand-picked ones.

The sibling `../ops/` folder tests the ops the *hand-written metal* quasar resnet50 issues, with
device-derived sharding. This folder answers a different question: **if we point tt-forge at Quasar,
which of the ops it actually emits are supported, and with which of its configs?**

Same layout and conventions as `../ops/`: a flat directory of standalone `test_*.py` files. No
`conftest.py`, no shared helper module, no generated config module, no `tools/`. Each file carries
its own config table, its own memory-config builder, and its own grid guard, exactly as each
`../ops/` test carries its own sharding setup.

---

## What the Forge graph contains

96 compute ops in `@forward`, from the optimised compile
(`opt_level=2, consteval, HiFi2, remove_dead_values, max_legal_layouts=8, bf16`):

| Forge op | count | Quasar op | route | file |
|---|---|---|---|---|
| `ttnn.conv2d` | 53 | `quasar.conv2d` | direct | `test_conv2d_forge.py` |
| `ttnn.add` | 16 | `quasar.add` | direct | `test_add_forge.py` |
| `ttnn.relu` | 16 | — **none** — | fuse into the preceding `add` | `test_relu_forge.py` |
| `ttnn.to_memory_config` | 4 | `quasar.to_memory_config` | direct | `test_to_memory_config_forge.py` |
| `ttnn.reshape` | 2 | `quasar.reshape` | direct | `test_reshape_forge.py` |
| `ttnn.to_layout` | 1 | `quasar.to_layout` | direct | `test_to_layout_forge.py` |
| `ttnn.permute` | 1 | — **none** — | `quasar.transpose` ×2 | `test_permute_forge.py` |
| `ttnn.max_pool2d` | 1 | `quasar.max_pool2d` | direct | `test_max_pool2d_forge.py` |
| `ttnn.mean` | 1 | — **none** — | generic `ttnn.mean` | `test_mean_forge.py` |
| `ttnn.linear` | 1 | `quasar.linear` | direct | `test_linear_forge.py` |

Plus 276 `ttnn.deallocate` and, in the 106 const-eval functions, 53 `prepare_conv2d_weights` +
53 `prepare_conv2d_bias`. Quasar exposes neither prepare entry point — `quasar.conv2d` prepares
internally, and `test_conv2d_forge.py` asserts the two agree on the prepared shape.

**Three gaps.** `ttnn.experimental.quasar` binds data-movement ops, conv2d, the pools, the matmul
family and a **binary** front-end. It binds **no plain unary activation** (hence no `relu` —
`prelu`/`pow`/`polyval` are the only unary-with-param ops there), **no general `permute`** (only
`transpose`, a 2-axis swap), and **no reduction** (the quasar reduce has a device backend but no
python binding). Each gap has a probe test that resolves the op at runtime — it **xfails** with the
gap named while the op is absent, and starts exercising it for real the moment a binding lands — and
a workaround test that runs the route which does exist.

---

## Layout

```
ResNet50_Forge_Fe/
  README.md
  test_op_inventory.py            host-only triage, no device, ~1s
  test_conv2d_forge.py            1 table check (host-only) + 53 call-sites
  test_add_forge.py               16 call-sites
  test_relu_forge.py              16 gap probes + 16 fused workarounds
  test_to_memory_config_forge.py   4 call-sites
  test_reshape_forge.py            2 call-sites
  test_to_layout_forge.py          1 call-site
  test_permute_forge.py            1 gap probe + 1 workaround
  test_max_pool2d_forge.py         1 call-site
  test_mean_forge.py               1 gap probe + 1 workaround
  test_linear_forge.py             1 call-site
```

120 tests total, of which 6 need no device (5 in `test_op_inventory.py`, plus the conv table check).

---

## Where the configs come from, and how to re-check them

Every table is transcribed from the tt-forge TTNN MLIR that is checked in under `proof/`, and
carries, per call-site: shapes, kernel/stride/padding/dilation/groups, the fused activation, math
fidelity, every `Conv2dConfig` flag, the slice config, the matmul program config, and the
**complete memory config** — buffer, page layout, memory layout, shard shape in elements, core
ranges, and orientation.

Two sources in the repo, which cannot share a bug because they come from different printers:

- **the TTNN MLIR** — `proof/inputs/ttnn_onnx_resnet_resnet50_cv_image_cls_hf.mlir`
  (`module @onnx_resnet_resnet50_opt`, body in `@forward`). The only source for **page layout** and
  **core ranges**, which live in the `#ttnn_layout` attributes rather than in the op arguments.
- **the EmitPy render of that same MLIR** — `proof/emitpy/B_ttnn_route.py`, produced by
  `ttmlir-opt --ttnn-to-emitpy-pipeline` + `ttmlir-translate --mlir-to-python`. Real Python, so it
  can be read with `ast` instead of regexes, and it prints the `ShardOrientation` explicitly (all
  195 shard specs are `ROW_MAJOR`). See `proof/README.md` for how it was generated.

The two agree on all 53 convs and every other op; the tables here were built from that agreement.

Two of the checks are restated **as tests**, so they run on the Quasar machine where the tt-forge
tree is not present:

- `test_conv2d_forge.py::test_forge_conv_table_matches_resnet50_topology` rebuilds the 53-conv
  topology from first principles — `layers=[3,4,6,3]`, widths `[64,128,256,512]`, expansion 4,
  stride on the 3×3 — and checks the table against it, including the 33-fused-relu / 20-bare split
  and the duplicate bookkeeping.
- `test_op_inventory.py` checks the Forge→Quasar mapping against the **live** ttnn build, so it
  fails both when a mapped quasar op vanishes and when one of the three gaps closes.

---

## Running

Start with the host-only triage — no device, about a second, and it tells you which ops exist
before anything touches hardware:

```bash
pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/test_op_inventory.py
```

Then the real thing:

```bash
# everything
pytest -s models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe/

# one test per DISTINCT config -- the 53 convs collapse to 24, the 16 adds to 4
pytest -s .../ResNet50_Forge_Fe/ -k "not dup"

# only the gap workarounds (the routes that can pass today)
pytest -s .../ResNet50_Forge_Fe/ -k "fused or transpose or generic"

# one op / one layer / just the host-only checks
pytest -s .../ResNet50_Forge_Fe/test_conv2d_forge.py -k "L3"
pytest    .../ResNet50_Forge_Fe/test_conv2d_forge.py -k topology
```

On craq-sim / the emulator:

```bash
TT_METAL_SIMULATOR=~/sim/libttsim.so \
TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
pytest -s .../ResNet50_Forge_Fe/ -k "not dup"
```

### Grid requirements

Forge picked its layouts for a full part — up to 8×7 = 56 cores. Each test skips with a message
naming the required and available grid rather than failing obscurely, so on a small emulator grid
the run reports skips, not noise.

There is deliberately **no auto-shard escape hatch**: pinning Forge's memory config *is* the point
of this folder. The "does the kernel work with ttnn-chosen sharding" question is what `../ops/`
already answers, on the same ops.

---

## Known Quasar issues these will walk into

Pre-existing, documented in `../ops/`, and **not** defects in this suite:

- **conv2d, any `kernel_size > 1`** (the 7×7 stem and every 3×3): the fused
  `conv_bmm_tilize_metal2` path has deadlocked on Quasar — see `../test_conv_hang.py`. Expect a
  **hang**, not a failure; the modules carry `pytest.mark.timeout`.
- **max_pool2d**: has hung in the pool-reduce dest handshake in `compute_pool_2d.cpp`
  (`../ops/test_max_pool2d.py`).
- **mean / any scaled reduce**: the Quasar GAPOOL reduce has applied a fixed **~1.1504×** gain that
  WH/BH do not (`../ops/test_reduce_sum_mean.py`). `test_mean_forge.py` reports the observed gain in
  its assertion message so this is identifiable at a glance.
- **layer3/layer4 BLOCK_SHARDED convs** (512→1024, 1024→2048): have overflowed the `uint16_t`
  weights-DFB ring. Note Forge **pins** its own block-sharded layout here rather than letting the op
  reshard — a different setup from `../ops/test_conv2d.py`.
- **2D-mcast matmul**: hangs on Quasar. The Forge fc uses the **1D**-mcast config
  (`mcast_in0=True`), a different kernel path — a failure there is a distinct bug.

---

## Notes on the graph worth knowing before reading a failure

- The **stem conv is the only conv fed an INTERLEAVED activation**; every other one is fed a
  sharded tensor.
- **`max_pool2d` outputs ROW_MAJOR**, not TILE. That makes `layer1.0.conv1` and
  `layer1.0.downsample` the only two convs in the graph with a row-major activation.
- **`layer3.0.conv1` is still HEIGHT_SHARDED.** The height→block reshard happens in the two
  `to_memory_config` ops after it, which is why `layer3.0.conv2` and `layer3.0.downsample` are fed
  `[128, x]` block shards while the rest of layer3 is `[32, x]`.
- Model is **torchvision** ResNet-50 (`IMAGENET1K_V1`), not HuggingFace — `fc.weight` / `fc.bias` in
  the IR signature confirm it. The `_hf` in the dump filenames is a hardcoded test label, not the
  source.
- Batch size 1 throughout, bf16 activations and weights, MathFidelity **HiFi2**, all conv outputs in
  **L1**, `act_block_h_override=0`, slice config `l1_full/0`, symmetric padding = `kernel // 2`.
- The Forge fc weight is already **K×N** (`2048×1000`), not torch's N×K, and its bias is **rank 1**
  (`tensor<1000xbf16>`). If `quasar.linear` rejects the rank-1 bias, that is a genuine
  Forge-graph/Quasar mismatch worth reporting, not a test defect.

<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Meta tensors vs Fake tensors for up-front collect

Context: `up_front_collect.py` collects ttnn ops from existing pytest bodies by running
each body under `NO_DISPATCH` and stashing the ops, while neutralizing the expensive
**host-side torch work** (RNG, golden references, weight prep, PCC) that is irrelevant to
*what* gets collected. A test body is `build inputs -> call ttnn op -> verify`, the ops are
chained (op N+1's spec depends on op N's output shape), so any method **must propagate
shapes through the body**. Both meta and fake tensors do exactly that — carry shape/dtype,
allocate no storage, run no compute — they only differ in robustness around the edges.

**Relationship:** a meta tensor lives on a fake *device*; a `FakeTensor` is a dispatch-mode
*wrapper* that computes its metadata using meta internally, then adds the machinery to
survive real-world code. `FakeTensor.__mro__` is `FakeTensor -> Tensor`. Fake is **built on
top of** meta, not an alternative to it.

Verified on `torch 2.11.0+cpu`.

---

## Side-by-side

| Dimension | Meta tensor (`device="meta"`) | Fake tensor (`FakeTensorMode`) |
|---|---|---|
| What it is | a device / C++ dispatch key | a `__torch_dispatch__` tensor subclass + context-manager mode |
| `type(t)` | plain `torch.Tensor` | `FakeTensor` |
| Remembers real device | No — always reports `meta` | **Yes** — reports `cpu`/`cuda:0` while allocating nothing |
| `.item()` / `.tolist()` | **Raises** (`cannot be called on meta tensors`) | With a `ShapeEnv`, returns an **unbacked SymInt** (e.g. `zuf0`) — no raise |
| Mixing with real tensors | **Raises** (`device cpu is not on the expected device meta`) | `allow_non_fake_inputs=True` -> works |
| Op with no metadata rule | Raises `NotImplementedError` | `allow_fallback_kernels=True` (default) -> runs the real kernel for that one op |
| Factory functions (`randn`, `empty`) | Must pass `device="meta"` on every call | Auto-intercepted inside the mode — no `device=` injection |
| Dynamic / symbolic shapes | No (concrete ints only) | Yes, via `ShapeEnv` (the dynamic-shapes infra `torch.compile` rests on) |
| API stability | Public, stable (`device="meta"`) | Underscore namespace (`torch._subclasses.fake_tensor`) — pin to torch version |
| Per-op cost | Cheap (C++) | Python `__torch_dispatch__` overhead per op |

---

## Meta tensors

**Pros**
- Public, stable API (`device="meta"`); no namespace pinning concern.
- Cheapest per-op (pure C++ dispatch, no Python wrapper overhead).
- Simple mental model: a tensor with no storage.
- Already the basis of `_meta_host_ops` today — no migration cost.

**Cons**
- `.item()` / `.tolist()` **raise** -> requires the `_meta_item` / `_meta_tolist` stand-ins.
- Cannot mix with real tensors -> a real fixture/param tensor meeting a meta tensor in one op
  **raises before any ttnn op is stashed**. Suspected root cause of the
  **0-programs-collected collapse on model module tests**.
- Ops lacking a meta kernel **raise** with no graceful degradation.
- Factory functions need `device="meta"` injected on every call (the `_mk()` wrapper).
- Reports `device == "meta"`, so code branching on `.device` / `.is_cuda` sees the wrong device.
- No symbolic shapes — no path for data-dependent values.

## Fake tensors

**Pros**
- **Generic**: `__torch_dispatch__` propagates shapes through *any* torch op — no per-op
  denylist (subsumes the `randn`/`conv2d`/`layer_norm` swaps in `_cheap_host_ops`).
- `allow_non_fake_inputs=True` lets fake and real tensors coexist — directly targets the
  module-test collapse.
- `allow_fallback_kernels=True` degrades gracefully (runs the real kernel) instead of raising
  on an unsupported op.
- With a `ShapeEnv`, `.item()` returns a SymInt instead of raising — removes the
  `_meta_item` / `_meta_tolist` hacks for the common case.
- Factory functions auto-faked inside the mode — no `device=` injection.
- Remembers the pretend device, so `.device` / `.is_cuda` branches behave correctly.
- Maintained and battle-tested: it is the mechanism `torch.compile` / Dynamo / export use to
  push the whole model ecosystem through metadata-only tensors.
- Collapses the plugin's **two** mechanisms (`_cheap_host_ops` + `_meta_host_ops`) into **one**
  principled mode.

**Cons**
- Underscore-namespaced (`torch._subclasses.fake_tensor`) — not a public contract; pin to the
  torch version (relevant given this project's version-coupling sensitivity).
- Per-op Python dispatch overhead (slower than bare meta per op; usually irrelevant since
  collect cost is dominated by the `ttnn.from_torch` boundary, not torch ops).
- Data-dependent **control flow** still breaks: `if t.item() > 3:` raises
  `GuardOnDataDependentSymNode`. Catchable -> fall back, but not free.
- A `.item()`-derived value that later becomes a tensor *dimension* turns into a symbolic
  shape the collector can't concretize (tension: `ShapeEnv` keeps `.item()` alive but allows
  symbolic dims; `static_shapes=True` keeps dims concrete but re-raises on `.item()`).

---

## What neither one changes (irreducible)

- **The ttnn<->torch boundary.** `ttnn.from_torch` needs a real/shape tensor; meta and fake
  both have no storage, so you still intercept it and call `allocate_tensor_on_device`
  shape-only. This single adapter survives in both approaches — and per the collect-cost
  profiling it is **~70% of collect cost**. So fake's win is *cleanliness and generality*
  (one mode, likely fixes the module-test collapse), **not** a speed breakthrough; the speed
  lever is the shape-only `from_torch`, which already exists.
- **Shape propagation is mandatory.** Pure "record API calls without executing" can't work for
  chained multi-op bodies — both meta and fake exist precisely to provide it cheaply.

## Empirical result — op-level test (layer_norm) AND module test: fake loses in BOTH

`fake` is the worst mode in every regime measured. Repro: `scripts/precompile_bench/collect_mode_ab.sh`
(general) and `scripts/precompile_bench/sdxl_unet_collect_ab.sh` (module), collect-only, n300.

| target | fast | meta | fake |
|--------|-----:|-----:|-----:|
| op-level `test_layer_norm` (163 bodies) | **224** progs, 0 swallowed | **224** progs, 162 benign swallows | **170** progs, **36 real misses** |
| SDXL UNet module (1 body) | **216** progs | 3 progs | **0** progs |

**Op-level — why fake undercollects (170 vs 224):** all 36 misses are `test_layer_norm_with_padding`,
dying *before any op* with `GuardOnDataDependentSymNode: Could not guard on ... Eq(uN, uN)`. The
`ShapeEnv` (added so `.item()` wouldn't raise — fake's supposed edge over meta) **backfires**: it
turns `.item()` into an unbacked SymInt that poisons a downstream `==` in the padding logic. meta's
cruder `_meta_item -> return 1` gives a concrete value, so the body proceeds, stashes the op, and
only swallows later (benign, post-stash) -> full 224 coverage. So the ShapeEnv is a net negative for
collection coverage; a concrete `.item()` stand-in (meta's approach) is more robust.

**Verdict:** `fast` wins/ties everywhere (real tensors + surgical swaps). `meta` ties `fast` on op
tests (benign swallows) but collapses on modules. `fake` is worst in both. The "fake is the clean
generic apex" hypothesis is empirically dead — in BOTH regimes.

## RESULT — `null` (NullCompute) is the general winner (measured)

4-way A/B, collect-only, n300 (`scripts/precompile_bench/collect_mode_ab.sh`, runs under `flock`
on `/tmp/tt-device.lock` so it cooperates with run_safe_pytest / other agents):

| target | fast | **null** | fake | meta |
|--------|-----:|---------:|-----:|-----:|
| op `test_layer_norm` (163 bodies) | 224 / 6.6s | **224 / 8.6s** | 170 / 3.5s | 224 / 2.6s |
| op `test_conv_features` HEIGHT_SHARDED (64) | 65 / 7.2s | **65 / 5.9s** | 65 / 6.2s | 65 / 5.9s |
| op `test_conv_features` all (192) | 174 / 15.6s | **174 / 14.4s** | 174 / 14.9s | 174 / 14.4s |
| op `test_matmul` (845 bodies, after fixes below) | 533 / 69s | **533 / 70s** | — | — |
| SDXL UNet module | 216 / **20.4s** | **216 / 11.9s** | 0 / 0.3s | 3 / 37.4s |
(unique programs / wall; module times same-run = directly comparable)

### matmul: the regression that hardened null (fixed, full parity)

First run of `test_matmul.py` exposed null's one real flaw: skipped ops returned `torch.empty`
(garbage), and the matmul suite verifies with `assert_numeric_metrics` (allclose+frobenius+ULP+PCC;
only `comp_pcc` was stubbed) → 632 bodies aborted mid-body on garbage-vs-garbage comparisons:
888 ops/527 progs vs fast's 1037/533, swallowed 659 vs 176, 85s vs 69s. Three fixes → exact parity
(1037→533, swallowed 178, 70s vs 69s):

1. **zeros, not empty** for skipped-op outputs + `to_torch`. Deterministic zeros make
   golden(zeros) == readback(zeros), so ANY equality/allclose verifier passes without stubbing —
   the general safety net. A/B'd zeros vs empty: wall identical (memset invisible), empty adds
   garbage-flavored swallows. Settled.
2. **Stub `assert_numeric_metrics`** like comp_pcc (shared `_stub_verifiers()` across all modes;
   patches `from ... import` bindings via sys.modules scan). It wasn't just swallow noise —
   most of null's dispatch traffic was the verifier itself (skipped 18,960→6,469).
3. **Memoize meta-kernel-less overloads** (`_null_no_meta`) — fallback exception per call → 3
   unique overloads memoized (4408→3 exception unwinds). **rng→zeros**: factories run real by
   design, but real `randn`/`rand` were the entire residual 20s wall gap vs fast; values are
   never load-bearing (masks/indices come from randint/arange, kept real).

**`null` is FASTER than `fast` wherever there's real compute to skip** (conv2d −18%, UNet −42%). The
layernorm "slower" was NOT the per-op interception — it was the **fallback/exception path**: layernorm
hit `fallback=1008` (ops whose meta kernel raises -> `try meta -> except -> re-dispatch real`, an
exception per op), conv2d hit `fallback=0`. Fix: memoize the set of meta-kernel-less ops and skip the
meta attempt for them. The dispatch tax on the happy path is negligible.

`null` = `UP_FRONT_NULL_COMPUTE`: a `TorchDispatchMode` over EVERY aten op — factories/loads (no
tensor input) run real, tensor-consuming ops have compute skipped (shape via meta kernel -> real
empty). No op list, no threshold; tensors stay real so `from_pretrained`/`Parameter`/`.item()` work.

- **Only general mode that matches `fast`'s coverage everywhere** (224 op layer_norm, 65 op conv2d, 216 module).
- **FASTER than `fast` wherever there's real compute to skip**: SDXL UNet −42% (11.9 vs 20.4s),
  conv2d −18% (5.9 vs 7.2s). Ran the UNet body to completion (0 swallowed), skipped 24,383 aten ops.
- **Only slow on meta-kernel-less ops** (layernorm +30%, from the `fallback=1008` exception path, NOT
  the interception) — fixed by the `_null_no_meta` memoization. The happy-path dispatch tax is negligible.

**Recommendation:** `null` is the default for ALL collect — general (no list), robust (real tensors),
and faster than `fast` on real workloads. With the matmul fixes (zeros outputs, verifier stub, no-meta
memoization, rng→zeros) it matches fast everywhere measured; fast's only remaining edge is none — both
miss the custom-`tile=` from_torch path. `fake`/`meta` remain documented dead-ends for modules (and
`fake` for `.item`-padding op tests).

## (superseded) The general way — `SkipHeavyCompute` (real tensors, aten-level skip)

> `null`/NullCompute above is the realized, list-free version of this idea (intercept ALL ops, not a
> heavy-op list) and is what shipped as `UP_FRONT_NULL_COMPUTE`. Original sketch kept for context:

The failure mode of meta/fake is **unreal tensors**, not genericity. Keep tensors REAL (fast's
robustness) but move interception from by-NAME monkeypatching to the **aten dispatch level**, skipping
only heavy compute — shaping the skipped output via the op's meta kernel:

```python
class SkipHeavyCompute(TorchDispatchMode):
    HEAVY = ("convolution","addmm","mm","bmm","scaled_dot_product_attention")
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if any(h in func.name() for h in self.HEAVY):
            meta_out = func(*tree_map(to_meta, args), ...)      # shape via meta kernel
            return tree_map(lambda m: torch.zeros(m.shape, m.dtype), meta_out)  # REAL tensor, no FLOPs
        return func(*args, **(kwargs or {}))                    # cheap ops run real on real tensors
```

Host-validated: `nn.Parameter(real)` wrap works (the op that killed fake), `conv2d` skipped -> real
correct-shape tensor, `.item()` returns a real float (no SymInt poisoning -> would fix the 36 misses).

- **vs fake/meta:** real tensors -> survives `from_pretrained` + real `.item()`; the measured failures vanish.
- **vs fast:** aten-level catches `aten.convolution` on every call path (not just `F.conv2d` by name);
  smaller principled heavy-op list; cheap ops run real (more faithful than fast, which zeros `randn`).
- **Open risk:** per-op dispatch tax (every aten op routes through Python) — measure on a UNet-scale
  forward vs fast's near-zero overhead on unpatched ops. Composes with shape-only `from_torch`.
- **Longer term:** C++ graph walking (collect at the op adapter, never run the body's torch) sidesteps
  golden refs + weight prep + `from_pretrained`-in-body entirely.

## Empirical result — SDXL UNet *module* test (refutes the hypothesis below)

A/B/C on `tests/nightly/single_card/stable_diffusion_xl_base/test_module_tt_unet.py`
(512x512, 4-ch base UNet), collect-only, n300. Repro: `scripts/precompile_bench/sdxl_unet_collect_ab.sh`.

| mode | call time | ops stashed | unique programs | outcome |
|------|----------:|------------:|----------------:|---------|
| **fast** | 14.6 s | 2252 | **216** | body completed, full coverage, 0 swallowed |
| meta | 35.5 s | 6 | 3 | dies: `Tensor on device cpu is not on the expected device meta` |
| fake | 0.4 s | 0 | **0** | dies in `from_pretrained`: `nn.Parameter(FakeTensor)` -> `fake_device` clash |

**`fake` collapses HARDER than `meta` on module tests, not softer.** The `allow_non_fake_inputs`
hypothesis was wrong here: the body calls `from_pretrained` *inside* the collect window, and the
HF/accelerate/diffusers model-loading path is structurally incompatible with a global fake regime:
- `low_cpu_mem_usage=True` (default): accelerate `init_empty_weights` does
  `nn.Parameter(fake_tensor.to(device))` -> `FakeTensor.__new__() got an unexpected keyword
  argument 'fake_device'` (dies at model *construction*, before any op).
- `low_cpu_mem_usage=False`: safetensors load does `aten.set_.source_Storage` of a real **cpu**
  checkpoint into a meta-backed fake param -> "set the storage of a tensor on device meta to a
  storage on different device cpu". **A real checkpoint loaded in-body cannot be absorbed by a
  fake (or meta) tensor.**

`fast` wins **because it keeps real torch tensors** — model construction, checkpoint load, and
weight prep all run for real (cheaply: `from_torch` shape-only, 1605 shape-onlied), reaching every
ttnn op. The genericity that makes `meta`/`fake` "clean" — a global non-real-tensor regime over the
WHOLE body — is exactly what collides with real model-loading code. No free lunch via tensor-faking:
surgical+real (`fast`) is robust but hacky; global-fake (`meta`/`fake`) is clean but breaks on
`from_pretrained`. `fake`'s elegance only pays off on op-level golden tests (inputs synthesized
in-body via `torch.randn`, no model loading) — the same regime where `meta` ever worked.

## Recommendation (revised)

- **Module tests that load a model in-body (`from_pretrained`): use `fast`.** It is the only mode
  that survives, and it gets full coverage. `fake`/`meta` are structurally unable to ingest a real
  in-body checkpoint.
- **Op-level golden tests (inputs via `torch.randn`, no model loading): `fake` is the clean win**
  over both `meta` (closes the `.item()`/mixed-tensor edges) and the `fast` per-op denylist.
- Keep `fake` as the opt-in `UP_FRONT_FAKE_COLLECT` mode for the op-level case; do **not** make it
  the default or a replacement for `fast`. The earlier "one principled mode replaces two" goal does
  not hold across the module-test boundary.
- If a generic mode is wanted for module tests, the regime must be **scoped to exclude model
  loading/construction** (only fake the ttnn-op region, let `from_pretrained` run real) — which is
  hard for arbitrary bodies and is exactly what `fast`'s keep-real-tensors approach sidesteps.

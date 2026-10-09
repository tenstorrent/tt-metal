# Porting a DeepSeek-V4-Flash module to V4.1-Flash

This guide is the recipe for bringing a V4 module (`models/experimental/deepseek_v4_flash`)
over to the V4.1 checkpoint. It was worked out on decode attention
(`tt/decode/attention.py`), which is used as the running example throughout.

The goal of every port is the same: **a small V4.1 file that holds only what V4.1 changed,
and imports everything else from V4.** Don't copy V4 code. If V4 can't be reused as it is,
add a narrow hook to V4 that keeps V4's behavior by default, and override that hook in V4.1.

## 1. Layout

The V4.1 tree follows the V4 tree file for file, so the matching V4 module is always easy to find:

```
deepseek_v41_flash/
  tt/config.py                 config + checkpoint access (shared by every module)
  tt/decode/<module>.py        decode implementation
  tt/prefill/<module>.py       prefill implementation
  tests/reference.py           the checkpoint's own inference/model.py as a CPU reference
  tests/decode/test_<module>.py
```

These pieces are shared and should not be re-implemented:

| Need | Use |
| --- | --- |
| Base class, grids, sharded configs, `_MASK_NEG` | `deepseek_v4_flash/tt/common.py` |
| `Linear`, `LinearDecode`, `BatchedLinearDecode`, `DeepSeekV4RMSNorm` | `deepseek_v4_flash/tt/layers.py` |
| Snapshot lookup, safetensors access, HF-to-checkpoint names | `deepseek_v4_flash/tt/weight_loader.py` |
| fp8 / mxfp4 dequant | `deepseek_v4_flash/tt/quant.py` (pass `block=FP8_BLOCK`) |
| ttnn weight cache | `deepseek_v4_flash/tt/weight_cache.py` |
| V4.1 config, fp8 32x32 dequant thunks, per-module weight dicts | `deepseek_v41_flash/tt/config.py` |
| Module-specific helpers (RoPE, cache updates, packing, ...) | the matching V4 module, imported by name (private `_helpers` included) |

## 2. The recipe

### Step 1: Diff the math, not the code

Put the two reference implementations side by side: V4.1's `<snapshot>/inference/model.py`
and V4's `modular_deepseek_v4.py`, or the V4 checkpoint's own `inference/` code. Sort every
difference in the module into one of these buckets:

| Bucket | Example (attention) | What the port does |
| --- | --- | --- |
| Unchanged | q_a/q_b/o path, attention sink, inverse RoPE on the output | inherit |
| Shape-only | q_lora 1024 to 1280, new projection widths | override the layout table |
| Op removed | per-head RMSNorm on q after `wq_b` | add a V4 class-attribute switch, flip it in V4.1 |
| Op added or redesigned | ratio-1/2 compressor, cross-layer KV / index sharing | new methods in the V4.1 subclass, built from V4 helpers |
| Numerics-only, not reproduced | activation fp8/fp4 quant simulation, Hadamard | stub out in the reference (see Step 6) |

Also write down what V4.1 *deleted*. Deleted weights (`compressor.ape`, `indexer.compressor.*`)
are a sign that a whole V4 sub-path no longer applies, and you should not try to inherit it.

### Step 2: Config

`tt/config.py:load_config` returns the checkpoint's `text_config` as a `SimpleNamespace`.
V4 code expects some fields to exist with V4 meanings. Derive those fields in `load_config`
rather than branching inside V4. For example, V4 picks its compressor from `layer_types`, so
V4.1 sets every compressed layer to `"compressed_attention"`. That's a type V4 doesn't know,
so the V4 base class builds no compressor and the V4.1 subclass adds its own.

Don't patch V4's `configuration_deepseek_v4.py`. Its defaults, such as `num_hash_layers=3`,
are wrong for V4.1.

### Step 3: Weights

Each module gets a `<module>_weights(loader, layer_idx) -> dict` function in `tt/config.py`.
`attention_weights` is the template:

- Keys the V4 module reads by HF name (`q_a_proj.weight`, ...) are loaded with
  `translate=True`, so the V4 loader maps them onto checkpoint names.
- Tensors that are new in V4.1 keep their checkpoint names relative to the module
  (`compressor.wkv.weight`, `indexer.wq_b.weight`). Don't invent HF aliases for them.
- Values are **thunks** (`dequantized(...)`), so nothing is read until a weight-cache miss.
- `dequantized` handles fp8 E4M3 with **32x32** E8M0 block scales (V4 used 128x128) and
  returns unquantized tensors as fp32.
- Keep tensors the checkpoint stores in bf16 at bf16 on device (compressor, index `wk`,
  `weights_proj`, norms). Only the fp8 weights get `weight_dtype` (bf4/bf8).

Check that every key resolves, with the right shape, on the host before writing any device code:

```bash
PYTHONPATH=$PWD python_env/bin/python -c "
from models.experimental.deepseek_v41_flash.tt.config import *
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir
l = DeepseekV4WeightLoader(resolve_snapshot_dir(DEFAULT_MODEL_DIR))
print({k: tuple(v().shape) for k, v in attention_weights(l, 2).items()})"
```

### Step 4: The module

Subclass the V4 class and keep `__init__` additive: call `super().__init__`, then build only the
V4.1-specific pieces.

**Shape changes.** V4's decode layouts (`K`, `N`, `n_blocks` per projection) are a module-level
dict. Expose it as a class attribute (`decode_layouts = DECODE_LAYOUTS`) and override it in
V4.1. The base class reads `self.decode_layouts[name]`.

**Removed ops.** Add a class attribute to V4 that defaults to V4's behavior and gates that op
(attention: `q_head_norm = True`, read where `fuse_q_b_norm` is decided). Set it to `False` in
V4.1. Relax any asserts that assumed the op always runs, and leave the V4 code path untouched.
The V4 tests have to pass unchanged.

**Added ops.** Write them as methods on the subclass (`_latent`, `_write_index_key`), composed
from V4's helpers (`_apply_rope`, `_update_cache_at`, `_one_row_per_user`, ...). When a whole
V4 sub-module carries over in spirit, subclass it as well (`DeepSeekV41Indexer` inherits
`DeepSeekV4Indexer.select_kv` and only replaces its projections).

**Variants that are pure V4.** Delegate directly. Window-only V4.1 layers just call
`decode_static`.

**The entry point.** Keep V4's call contract (input shape and memory config, output shape) so
the decoder layer can call either version. Anything that depends on the position goes into a
host-side `decode_inputs(pos, ...) -> dict` that builds the device tensors (RoPE rows, cache
slots, masks, `cur_pos`) and the Python flags. The test and, later, the decoder layer then
just call `decode(hidden, scache=..., **decode_inputs(pos, ...))`.

**State shared across layers.** V4.1 shares compressed KV and top-k selections across layers.
Put the allocation in one `build_decode_caches(device, config, layers, max_seq)` function that
aliases buffers by source layer, and assert that each consumer's source layer is in the set
being built. Layers that share buffers have to run in layer order within a step.

### Step 5: Decode layout constraints

Most device failures in a port come from `matmul_decode` / `all_gather_for_matmul`
preconditions rather than from the math. Check these before asking for a device run:

- **The activation must be sharded.** `all_gather_for_matmul` FATALs with "Input must be
  sharded" on interleaved input. The decode row is row-major and width-sharded in L1:
  `width_sharded_l1_config(1, hidden_size, device, tile_height=1)`.
- **The A grid must contain the B grid**, and the output core grid must be a filled rectangle.
- **The K block must be an even number of tiles in [2, 256].** `n_blocks` decides this, so
  recheck it whenever K changes.
- **The output-mcast writer FATALs on bounding-box cores that have no reader.** For a fused
  norm feeding a downstream matmul, the producer's B grid has to sit inside the consumer's
  rectangle (attention: q_a uses `n_blocks=8`, an 8x1 strip inside q_b's 8x8).
- **Reuse replicas.** Projections that read the same activation should use the same cut
  (`K`, `n_blocks`, rectangle), so they all read the one all-gather replica. In attention,
  `kv_proj`, `compressor.wkv` and `compressor.wgate` share one, and `q_b_proj` and
  `indexer.wq_b` share another.
- **Watch L1 headroom around whole-grid ops.** Move tensors you'll need later out to DRAM
  before an op that uses the full grid (attention spills q around `fused_lightning_select_kv`).
- **Watch tile padding.** Reductions over a short axis stored in TILE layout run into the
  padding: a softmax over 2 rows sees 32. Use a closed form instead; for a pair,
  `softmax = sigmoid(g0 - g1)`.
- **Hardcoded constants in fused ops.** For example, `fused_lightning_select_kv` hardcodes a
  compress rate of 4, so V4.1 passes `cur_pos = 4n - 1` to make it score n keys. Also avoid
  calling ops on empty inputs (attention skips the select when n == 0).

### Step 6: Reference

`tests/reference.py` imports the checkpoint's own `inference/model.py`, so the reference can't
drift from what DeepSeek ships:

- A stub `kernel` module replaces tilelang. Activation-quant simulations become no-ops,
  `sparse_attn` becomes torch gather + softmax, and anything not needed raises
  `NotImplementedError`.
- `model.default_dtype = torch.float32`, and weights come from the same `dequantized` thunks
  as the device path, so every `Linear` takes plain `F.linear`.
- Build `ModelArgs` from `inference/config.json`, keeping only the dataclass fields.

To port a new module, add a `reference_<module>s(...)` next to `reference_attentions`. If the
module calls a kernel that is still stubbed (`fp4_gemm`, `fp8_gemm`, `hc_split_sinkhorn`), give
it a torch implementation in the stub; don't patch `model.py`.

### Step 7: Test

Use `tests/decode/test_attention.py` as the template:

- **Real weights, reference vs device, step by step** from position 0, so caches build up the
  way they do in the real model.
- **One parametrized case per structural variant**, choosing layer groups that exercise the
  sharing (attention: window-only `(0,)`, ratio-2 source plus consumer `(2, 3)`, ratio-1
  source plus consumer plus index-only source `(20, 21, 24)`).
- **Pick `seq_len` to cross every boundary**: ring wrap (`window`), first compressed entry,
  top-k actually dropping rows (`> index_topk` entries). Compare at the boundary positions and
  the last few, not at every step.
- **Use the production weight dtype** (`bfloat4_b`) and a PCC threshold (0.95). Log the PCC
  for every compared position.
- **Build the input exactly as the caller will** (memory config included). The first failure
  of this port was a TILE/DRAM test input going into an entry point that needs a width-sharded
  L1 row.
- `DEEPSEEK_V41_CACHE_DIR` keeps converted weights across runs, and `skipif` covers a missing
  snapshot.

Before any device run, check on the host: imports resolve, weight keys and shapes look right,
the reference runs on CPU, and lints/pre-commit pass. The card is shared, so ask before
running device tests.

## 3. What's left, per module

These come from the V4 vs V4.1 diff. The strategy column says which Step 4 pattern applies.

| V4 module | V4.1 change | Strategy |
| --- | --- | --- |
| `decode/attention_csa.py`, `attention_hca.py` | replaced by the ratio-1/2 path | done in `decode/attention.py`; not ported |
| `decode/moe.py` | hash routing removed (no `tid2eid`), new expert counts, MTP uses 128 experts top-3, `gate_temp` | done in `decode/moe.py` for the main layers; MTP (128 experts, top-3) not ported. Notes below |
| `decode/hyperconnection.py` | the pre-mix is computed one sublayer early and carried as `pre_mix`; no `hc_head` | new variant of the collapse that takes a `pre_mix` argument; the fused op returns `(post, comb, collapsed)` from a single stream, so it needs the carried-in variant |
| `decode/decoder_layer.py` | threads `pre_mix` through attention then FFN; engram before layers 1 and 14; layer-ordered shared caches | subclass; `pre_mix` becomes part of the residual state passed between layers |
| Engram (new) | n-gram hash lookup into ~98 GB tables per layer, fp8 `wkv`, sigmoid gate per copy | new module from `layers.py` primitives; row gather on the host, matmul and gate on the device |
| `embedding.py`, LM head | final collapse uses the last `f_pre`; no `hc_head` | override the collapse only |
| `decode/dspark.py` | taps are the inputs of layers 37–39; markov weights renamed (`embed` / `head`); no `hc_head` | config + weight-name mapping; inherit the rest |
| `prefill/*` | same math changes as decode | same recipe; share `config.py` and the reference |
| `weight_loader.py`, `quant.py` | 32x32 fp8 blocks; renamed and removed tensors | keep using V4's loader; put the V4.1 mapping in `config.py` |

MoE notes:

- `fused_experts` derived its geometry from a hard-coded 64 DRAM shards (D = 4096). It now
  derives it from D (64-column shards: 80 at D = 5120, with a 10x8 serial grid), so V4.1 runs the
  same op. V4's shapes give the same program as before.
- The 6-expert path splits `I` over 16 cores per expert in whole tiles. `I = 2304` is 72 tiles,
  so `DeepSeekV41PreloadedExperts` zero-pads it to 2560. This is exact, at ~11% more expert bytes.
- The op ranks experts on a bf16 `scores + bias` row. V4.1's bias reaches ~57, where the bf16 step
  swamps the score gaps (layer 39: 3/16 tokens pick the fp32 expert set). The router subtracts
  `bias.max()` on the host, which leaves the ranking unchanged (16/16).
- The shared expert is an `Expert` in V4.1 and takes the `swiglu_limit` clamp
  (two `ttnn.clamp`s ahead of V4's `silu` / `multiply`; `ttnn.clamped_silu_glu` gives a wrong
  result on these ROW_MAJOR width-sharded operands, PCC 0.53). It defaults to bfloat8_b; the routed
  experts are bfloat4_b.
- No layer is hash-routed. The block builds the learned router whenever no `gate` is injected,
  so V4.1 doesn't inject one.
- `tests/decode/test_moe.py` (16 random tokens, PCC bar 0.95): layer 2 at 0.971, layer 39 at
  0.995. Most tokens reach ~0.99, the bfloat4_b ceiling. A few drop to ~0.90, consistent with the
  bf16 ranking (the op takes no fp32 ranking row) swapping a near-tied 6th expert. With host-chosen
  ids the same tokens reach 0.986.

## 4. Checklist

- [ ] Every difference sorted into a bucket (Step 1); deleted weights listed.
- [ ] Config fields V4 keys on are derived in `load_config`.
- [ ] `<module>_weights` resolves every key on the host with the expected shapes.
- [ ] The V4.1 file only contains what changed; V4 hooks are class attributes with V4 defaults.
- [ ] The V4 tests for the touched V4 module still pass.
- [ ] Decode layouts satisfy the Step 5 constraints; shared activations share one cut.
- [ ] The reference runs on CPU through the checkpoint's own `model.py`.
- [ ] One test case per structural variant, crossing every boundary; input built as the
      caller builds it.
- [ ] Lints and pre-commit clean before asking for a device run.
